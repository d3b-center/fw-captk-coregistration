"""Util module."""
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


class InputError(Exception):
    """An input or option the gear cannot work with; the message says what to fix."""


def available_cpus() -> int:
    """The CPUs this job may run on (its affinity mask), not the host's.

    shortcut: honours cpusets, not a CFS quota (docker --cpus); Greedy then runs more threads than the quota allows,
    which slows it down but is harmless. Upgrade by reading /sys/fs/cgroup/cpu.max if jobs are quota-limited.
    """
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return os.cpu_count() or 1  # macOS, local development


def check_image(path: str, label: str) -> None:
    """Refuse an image Greedy cannot register: not readable, not 3D, or without any intensity variation."""
    name = Path(path).name
    try:
        image = sitk.ReadImage(path)
    except RuntimeError as error:
        raise InputError(f"{label}: {name} cannot be read as an image ({str(error).strip().splitlines()[-1]})") from error
    if image.GetDimension() != 3:
        raise InputError(f"{label}: expected a 3D image, but {name} has {image.GetDimension()} dimensions; pass a single 3D volume")
    values = sitk.GetArrayViewFromImage(image)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or finite.min() == finite.max():
        raise InputError(f"{label}: {name} has no intensity variation (empty or constant), so nothing can be registered")
