"""Main module."""
import logging
import shutil
from pathlib import Path

from . import captk_reg
from .parser import parse_config
from .util import InputError, available_cpus, check_image

log = logging.getLogger(__name__)

MODALITY = "moving"  # the name captk_reg.py gives the one image it registers; it names its output files
# What CaPTk's `Preprocessing -reg` used and this gear's manifest never exposed: NMI (captk_reg's default), 100x50x5
# iterations, and no bias correction.
ITERATIONS = "100x50x5"


def _base_name(path: str) -> str:
    return Path(path).name.removesuffix(".gz").removesuffix(".nii")


def run(gtk_context) -> int:
    """Main entrypoint.

    Args:
        gtk_context (GearToolkitContext)

    Returns:
        int: the exit code, 1 where an input or option cannot be used.
    """
    # ====================== SETUP ==============================================
    try:
        config = parse_config(gtk_context)
        check_image(config.moving_path, "moving_image")
        check_image(config.fixed_path, "fixed_image")
    except InputError as error:
        log.error("%s", error)
        return 1

    # ====================== Registration ======================================
    # captk_reg.py registers an "anchor" image to an "atlas" and writes <name>_to_SRI.*; here the fixed image is the atlas.
    work = Path(gtk_context.work_dir) / "coregistration"
    captk_reg.main([
        "--atlas", config.fixed_path, "--out", str(work), "--no-n4", "--dof", str(config.dof), "--iters", ITERATIONS,
        "--threads", str(available_cpus()), f"{MODALITY}={config.moving_path}",
    ])

    # ====================== Outputs ============================================
    output_dir = Path(gtk_context.output_dir)
    shutil.copyfile(work / f"{MODALITY}_to_SRI.nii.gz", output_dir / config.output_file_name)
    if config.intermediate_files:
        # the name CaPTk gave it (affine_<moving>TO<fixed>_<metric>.mat)
        matrix_name = f"affine_{_base_name(config.moving_path)}TO{_base_name(config.fixed_path)}_NMI.mat"
        shutil.copyfile(work / f"{MODALITY}_to_SRI.mat", output_dir / matrix_name)
    return 0
