"""Synthetic head-like phantom, and a copy of it moved rigidly onto another grid with a contrast change.

No patient data: the test inputs are generated, so nothing imaging-related is committed.
"""
import numpy as np
import SimpleITK as sitk

# Motion applied to the phantom to make the moving image (radians, mm)
ROTATION = (0.12, -0.08, 0.15)
TRANSLATION = (5.0, -4.0, 3.0)


def make_phantom() -> sitk.Image:
    rng = np.random.default_rng(0)
    size, spacing = (80, 96, 72), (2.0, 2.0, 2.0)
    z, y, x = np.meshgrid(*(np.arange(n) * s for n, s in zip(size[::-1], spacing[::-1])), indexing="ij")
    c = [(n - 1) * s / 2 for n, s in zip(size, spacing)]
    img = np.zeros(size[::-1], np.float32)
    r2 = ((x - c[0]) / 62) ** 2 + ((y - c[1]) / 80) ** 2 + ((z - c[2]) / 56) ** 2
    img[r2 < 1] = 300  # head
    img[r2 < 0.75] = 600  # brain
    for _ in range(25):  # asymmetric structure the registration can lock onto
        p = rng.uniform(0.25, 0.75, 3) * np.array(size) * np.array(spacing)
        amplitude = rng.uniform(-250, 400)  # drawn before the width: the order the baseline phantom was made in
        width = rng.uniform(6, 14)
        blob = np.exp(-((x - p[0]) ** 2 + (y - p[1]) ** 2 + (z - p[2]) ** 2) / (2 * width**2))
        img += amplitude * blob * (r2 < 0.8)
    out = sitk.GetImageFromArray(np.clip(img, 0, None))
    out.SetSpacing(spacing)
    return out


def make_pair():
    """(fixed, moving, truth): moving sees the phantom through ROTATION/TRANSLATION on its own grid and with a gamma
    contrast change; `truth` maps fixed physical points (LPS) to moving physical points (what Greedy estimates)."""
    fixed = make_phantom()
    centre = np.array(fixed.GetSize()) * np.array(fixed.GetSpacing()) / 2
    tx = sitk.Euler3DTransform(tuple(centre), *ROTATION, TRANSLATION)
    ref = sitk.Image(72, 88, 64, sitk.sitkFloat32)
    ref.SetSpacing((2.2, 2.2, 2.2))
    ref.SetOrigin((-9.0, -12.0, -6.0))
    contrast = sitk.Cast(sitk.Pow(sitk.Cast(fixed, sitk.sitkFloat32) / 600.0, 1.5) * 600.0, sitk.sitkFloat32)
    moving = sitk.Resample(contrast, ref, tx.GetInverse(), sitk.sitkLinear, 0.0)
    return fixed, moving, tx


def lps_to_ras(tx: sitk.Euler3DTransform) -> np.ndarray:
    """The 4x4 matrix of a SimpleITK (LPS) rigid transform in the RAS convention of Greedy's .mat files."""
    m = np.eye(4)
    m[:3, :3] = np.array(tx.GetMatrix()).reshape(3, 3)
    c, t = np.array(tx.GetCenter()), np.array(tx.GetTranslation())
    m[:3, 3] = c + t - m[:3, :3] @ c
    flip = np.diag([-1, -1, 1, 1.0])
    return flip @ m @ flip


def brain_points(fixed: sitk.Image) -> np.ndarray:
    """Homogeneous RAS coordinates of a regular sample of brain voxels, to measure how far two transforms disagree."""
    idx = np.argwhere(sitk.GetArrayViewFromImage(fixed) > 400)[::37]
    pts = np.array([fixed.TransformIndexToPhysicalPoint((int(x), int(y), int(z))) for z, y, x in idx])
    pts[:, :2] *= -1
    return np.c_[pts, np.ones(len(pts))]


def displacement(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-point distance (mm) between two transforms."""
    return np.linalg.norm((points @ a.T)[:, :3] - (points @ b.T)[:, :3], axis=1)
