"""Parser module to parse gear config.json."""
from dataclasses import dataclass

from .util import InputError

DEFAULT_REGISTRATION_TYPE = "Affine"
DEFAULT_OUTPUT_FILE_NAME = "output.nii.gz"


@dataclass(frozen=True)
class GearConfig:
    """What one run reads: its two input files and its options."""

    moving_path: str
    fixed_path: str
    dof: int
    intermediate_files: bool
    output_file_name: str


def parse_registration_type(value: str) -> int:
    """Greedy's degrees of freedom for the value CaPTk's `Preprocessing -reg` took.

    'Rigid' is 6; 'Affine' is 12, or 'Affine-6', 'Affine-7' (rigid plus isotropic scaling) and 'Affine-12'. Case does not
    matter and ':' may replace '-'. CaPTk ran a deformable registration for any other text; that is refused here.
    """
    text = value.strip().upper()
    if "RIGID" in text:
        return 6
    if "AFFINE" in text:
        parts = text.split("-") if "-" in text else text.split(":")
        if len(parts) == 1:
            return 12
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) in (6, 7, 12):
            return int(parts[1])
        raise InputError(f"registration_type {value!r}: Greedy only accepts 6, 7 or 12 as the affine DOF, e.g. 'Affine-12'")
    raise InputError(
        f"registration_type {value!r} is not supported: this gear runs Rigid or Affine[-6|-7|-12] registration with Greedy. "
        "Deformable registration, which CaPTk's Preprocessing -reg ran for any other value, is not available"
    )


def parse_config(gear_context) -> GearConfig:
    """Get the input paths and config options, refusing what the manifest cannot check."""
    paths = {name: gear_context.get_input_path(name) for name in ("moving_image", "fixed_image")}
    absent = [name for name, path in paths.items() if not path]
    if absent:
        raise InputError(f"missing required input(s): {absent}")

    dof = parse_registration_type(gear_context.config.get("registration_type") or DEFAULT_REGISTRATION_TYPE)

    # `flyw gear config` writes integer options as strings ("1"); the engine writes numbers. Accept both, as bash's $(( )) did.
    intermediate_files = gear_context.config.get("intermediate_files", 0)
    if str(intermediate_files) not in ("0", "1"):
        raise InputError(f"intermediate_files must be 0 or 1, not {intermediate_files!r}")

    output_file_name = gear_context.config.get("output_file_name") or DEFAULT_OUTPUT_FILE_NAME
    if output_file_name != output_file_name.replace("\\", "/").rsplit("/", 1)[-1] or output_file_name in {".", ".."}:
        raise InputError(f"output_file_name must be a file name, not a path: {output_file_name!r}")
    if not output_file_name.endswith(".nii.gz"):
        raise InputError(f"output_file_name must end in .nii.gz, the format the registered image is written in: {output_file_name!r}")

    return GearConfig(
        moving_path=paths["moving_image"],
        fixed_path=paths["fixed_image"],
        dof=dof,
        intermediate_files=str(intermediate_files) == "1",
        output_file_name=output_file_name,
    )
