"""Regenerate original_coregistration.json from the ORIGINAL gear.

The original is commit 1df1c3e of this repository: `run` calls CaPTk's `Preprocessing -reg Affine -rIS 1` (Greedy
`-a -m NMI -ia-image-centers -dof 12 -n 100x50x5`, then a linear reslice) inside cbica/captk:2021.03.29. This
script gives it the synthetic phantom of tests/phantom.py and records what it computed.

  python tests/golden/make_golden.py            # needs docker and that image
  python tests/golden/make_golden.py --reuse D  # skip docker: read an earlier run's D/input and D/output

The committed JSON was produced with --reuse from a run of exactly these docker commands (2026-11-25).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tests import phantom  # noqa: E402

COMMIT = "1df1c3e"
IMAGE = "cbica/captk:2021.03.29"
IMAGE_DIGEST = "sha256:b81073b8375b7f9a8eed8a963d1423bb165b2dbdb1e104e3a32fde532d3fc09d"
CONFIG = {"registration_type": "Affine", "intermediate_files": 1, "output_file_name": "output.nii.gz"}


def run_original(work: Path) -> None:
    fixed, moving, _ = phantom.make_pair()
    for name, img in (("fixed_image", fixed), ("moving_image", moving)):
        (work / "input" / name).mkdir(parents=True)
        sitk.WriteImage(img, str(work / "input" / name / f"{name}.nii.gz"))
    (work / "output").mkdir()
    show = lambda path: subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=REPO)
    (work / "run").write_bytes(show("run"))
    os.chmod(work / "run", 0o755)
    (work / "manifest.json").write_bytes(show("manifest.json"))
    inputs = {n: {"base": "file", "location": {"path": f"/flywheel/v0/input/{n}/{n}.nii.gz"}} for n in ("moving_image", "fixed_image")}
    (work / "config.json").write_text(json.dumps({"config": CONFIG, "inputs": inputs}, indent=2))
    mounts = [("run", "run:ro"), ("manifest.json", "manifest.json:ro"), ("config.json", "config.json:ro"), ("input", "input:ro"), ("output", "output")]
    cmd = ["docker", "run", "--rm", "--platform", "linux/amd64", "-w", "/flywheel/v0", "--entrypoint", "/flywheel/v0/run"]
    for src, dst in mounts:
        cmd += ["-v", f"{work / src}:/flywheel/v0/{dst}"]
    subprocess.run(cmd + [IMAGE], check=True)


def summarize(work: Path) -> dict:
    fixed, _, truth = phantom.make_pair()
    out = sitk.GetArrayFromImage(sitk.ReadImage(str(work / "output" / "output.nii.gz"))).astype(np.float64)
    fx = sitk.GetArrayFromImage(fixed).astype(np.float64)
    return {
        "_source": {
            "repository": "fw-captk-coregistration", "commit": COMMIT, "image": IMAGE, "image_digest": IMAGE_DIGEST,
            "command": "run with config " + json.dumps(CONFIG), "input": "tests/phantom.py make_pair()",
        },
        "original_affine_ras": np.loadtxt(work / "output" / "affine_movingTOfixed_NMI.mat").tolist(),
        "truth_affine_ras": phantom.lps_to_ras(truth).tolist(),
        "registered_mean": float(out.mean()),
        "registered_std": float(out.std()),
        "registered_vs_fixed_corr": float(np.corrcoef(out[fx > 0], fx[fx > 0])[0, 1]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", type=Path)
    args = ap.parse_args()
    work = args.reuse
    if work is None:
        work = Path(tempfile.mkdtemp())
        run_original(work)
    (Path(__file__).parent / "original_coregistration.json").write_text(json.dumps(summarize(work), indent=1) + "\n")
