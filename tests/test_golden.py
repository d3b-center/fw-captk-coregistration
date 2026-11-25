"""Hold the ported registration to what the original gear (CaPTk Preprocessing -reg) computed on the same input.

Greedy draws its sample jitter at random, so two runs differ (over 40 runs on this 2 mm phantom the worst mean
displacement from the truth was 0.87 mm, the worst single point 1.9 mm, the worst image-statistic error 1.3 %).
The tolerances are about twice that: still far below what a wrong direction, a swapped transform order or a geometry
error produces (20 mm for an inverted matrix, the size of the shift for a translation error).
"""
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from gear_functions import captk_reg
from tests import phantom

GOLDEN = json.loads((Path(__file__).parent / "golden" / "original_coregistration.json").read_text())


def register_like_the_original(files, out_dir):
    """The original's defaults: Affine (12 DOF), NMI, 100x50x5 iterations, linear reslice, no bias correction."""
    captk_reg.main([
        "--atlas", files["fixed"], "--out", str(out_dir), "--no-n4", "--dof", "12", "--iters", "100x50x5", "--threads", "4",
        f"moving={files['moving']}",
    ])
    return np.loadtxt(out_dir / "moving_to_SRI.mat"), sitk.ReadImage(str(out_dir / "moving_to_SRI.nii.gz"))


def test_phantom_is_the_one_the_original_was_run_on(phantom_files):
    assert np.allclose(phantom_files["truth"], GOLDEN["truth_affine_ras"], atol=1e-9)


def test_transform_agrees_with_original_and_truth(phantom_files, tmp_path):
    matrix, _ = register_like_the_original(phantom_files, tmp_path)
    vs_original = phantom.displacement(phantom_files["points"], matrix, np.array(GOLDEN["original_affine_ras"]))
    vs_truth = phantom.displacement(phantom_files["points"], matrix, phantom_files["truth"])
    assert vs_original.mean() < 1.5 and vs_original.max() < 3.5
    assert vs_truth.mean() < 1.5 and vs_truth.max() < 3.5


def test_registered_image_agrees_with_original(phantom_files, tmp_path):
    _, registered = register_like_the_original(phantom_files, tmp_path)
    assert registered.GetPixelID() == sitk.sitkFloat32
    assert registered.GetSize() == phantom_files["fixed_image"].GetSize()  # on the fixed image's grid
    out = sitk.GetArrayFromImage(registered).astype(np.float64)
    fixed = sitk.GetArrayFromImage(phantom_files["fixed_image"]).astype(np.float64)
    assert abs(out.mean() / GOLDEN["registered_mean"] - 1) < 0.02
    assert abs(out.std() / GOLDEN["registered_std"] - 1) < 0.02
    assert np.corrcoef(out[fixed > 0], fixed[fixed > 0])[0, 1] > 0.96
