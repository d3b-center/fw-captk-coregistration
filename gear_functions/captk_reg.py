#!/usr/bin/env python3
# Ported verbatim from captk-greedy-reg/captk_reg.py (a local package, not yet in a repository), sha256 fda07b304dfe.
"""CaPTk BraTS-pipeline registration on top of upstream Greedy (picsl_greedy) and SimpleITK.

N4 bias correction (CaPTk recipe) -> anchor modality registered to the atlas -> every other modality registered to the
anchor -> originals resliced onto the atlas grid. All registration and reslicing is done by Greedy itself, with the same
options CaPTk's BraTSPipeline passes to the `greedy` binary.

  captk_reg.py --atlas atlas.nii.gz --out OUT [--anchor T1CE] T1=t1.nii.gz T1CE=t1ce.nii.gz T2=t2.nii.gz FL=fl.nii.gz

Writes OUT/NAME_to_SRI.nii.gz, OUT/<anchor>_to_SRI.mat and OUT/NAME_to_<anchor>.mat (Greedy .mat format).
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk
from picsl_greedy import Greedy3D

METRICS = {"NMI": "NMI", "MI": "MI", "SSD": "SSD", "NCC": "NCC {radius}"}


def n4_captk(img: sitk.Image) -> sitk.Image:
    """CaPTk's BiasCorrection::Run("n4"): Otsu mask (10 bins), shrink 4, N4 (order 3, 4 levels x 25 it, Wiener .01,
    FWHM .15, convergence 1e-7), then the log-bias field re-evaluated at full resolution and divided out."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetNumberOfHistogramBins(10)
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    otsu.Execute(img)
    lo_hi = sitk.MinimumMaximum(img)
    # CaPTk is built on ITK 4, which thresholds at the centre of the selected histogram bin; ITK 5 uses its upper edge.
    thr = otsu.GetThreshold() - (lo_hi[1] - lo_hi[0]) / 10 / 2
    mask = sitk.Cast(img > thr, sitk.sitkUInt8)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetSplineOrder(3)
    n4.SetWienerFilterNoise(0.01)
    n4.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    n4.SetMaximumNumberOfIterations([25] * 4)
    n4.SetConvergenceThreshold(1e-7)
    n4.Execute(sitk.Shrink(img, [4] * 3), sitk.Shrink(mask, [4] * 3))
    return img / sitk.Exp(n4.GetLogBiasFieldAsImage(img))


def _prepare(name, path, tmp, do_n4):
    """Worker: read the original (as float32), write the registration copy (N4-corrected unless disabled)."""
    img = sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32)
    out = os.path.join(tmp, f"{name}_reg.nii")
    sitk.WriteImage(n4_captk(img) if do_n4 else img, out)
    return out


def _register(fixed, moving, o):
    g = Greedy3D()
    metric = METRICS[o.metric].format(radius=o.ncc_radius)
    g.execute(
        f"-a -m {metric} -i fixed moving -o aff -ia-image-centers -n {o.iters} -dof {o.dof} -threads {o.threads}",
        fixed=sitk.ReadImage(fixed), moving=sitk.ReadImage(moving), aff=None,
    )
    return g["aff"]


def _reslice(atlas, original, mat_files, tmp, out_path):
    """Greedy `-r A B` applies A first, then B, so `mat_files` is given in that order. picsl_greedy 1.4.0.3 only accepts
    transforms for reslicing as files, and its command string cannot hold spaces, hence bare names inside `tmp`."""
    os.chdir(tmp)
    g = Greedy3D()
    g.execute(
        f"-rf my_atlas -ri LINEAR -rm my_moving my_out -r {' '.join(mat_files)}",
        my_atlas=sitk.ReadImage(atlas), my_moving=sitk.Cast(sitk.ReadImage(original), sitk.sitkFloat32), my_out=None,
    )
    # Greedy returns float64 here; CaPTk's outputs are float32.
    sitk.WriteImage(sitk.Cast(g["my_out"], sitk.sitkFloat32), out_path, useCompression=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modalities", nargs="+", metavar="NAME=IMAGE")
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor", help="modality registered to the atlas (default: T1CE if given, else the first)")
    ap.add_argument("--metric", default="NMI", choices=METRICS)
    ap.add_argument("--ncc-radius", default="2x2x2")
    ap.add_argument("--dof", type=int, default=6, choices=(6, 7, 12), help="6 = rigid (default), 7 = rigid + isotropic scaling, 12 = affine")
    ap.add_argument("--iters", default="100x50x10", help="Greedy -n schedule, coarse to fine")
    ap.add_argument("--threads", type=int, help="Greedy threads per registration (default: CPUs / registrations)")
    ap.add_argument("--no-n4", action="store_true", help="skip N4 bias correction before registration")
    o = ap.parse_args(argv)

    mods = dict(m.split("=", 1) for m in o.modalities)
    anchor = o.anchor or ("T1CE" if "T1CE" in mods else next(iter(mods)))
    if anchor not in mods:
        ap.error(f"anchor '{anchor}' is not among the given modalities")
    others = [m for m in mods if m != anchor]
    o.threads = o.threads or max(1, (os.cpu_count() or 1) // len(mods))
    os.makedirs(o.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp, ProcessPoolExecutor(len(mods), mp_context=mp.get_context("spawn")) as pool:
        reg = dict(zip(mods, pool.map(_prepare, mods, mods.values(), [tmp] * len(mods), [not o.no_n4] * len(mods))))
        # The anchor->atlas and every other->anchor registration only need the registration copies: run them together.
        jobs = {anchor: pool.submit(_register, o.atlas, reg[anchor], o)}
        jobs.update({m: pool.submit(_register, reg[anchor], reg[m], o) for m in others})
        mats = {m: j.result() for m, j in jobs.items()}
        mat_file = {m: (f"{anchor}_to_SRI" if m == anchor else f"{m}_to_{anchor}") + ".mat" for m in mats}
        for m, name in mat_file.items():
            np.savetxt(os.path.join(tmp, name), mats[m], fmt="%.9g")
            shutil.copyfile(os.path.join(tmp, name), os.path.join(o.out, name))
        for f in [
            pool.submit(_reslice, o.atlas, mods[m], [mat_file[anchor]] + ([] if m == anchor else [mat_file[m]]), tmp, os.path.join(o.out, f"{m}_to_SRI.nii.gz"))
            for m in mods
        ]:
            f.result()


if __name__ == "__main__":
    sys.exit(main())
