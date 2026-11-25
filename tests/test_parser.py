import pytest

from gear_functions.parser import GearConfig, parse_config, parse_registration_type
from gear_functions.util import InputError

INPUTS = {"moving_image": "/in/moving.nii.gz", "fixed_image": "/in/fixed.nii.gz"}


@pytest.mark.parametrize(
    "text, dof",
    [("Affine", 12), ("affine", 12), ("AFFINE-12", 12), ("Affine-6", 6), ("Affine:7", 7), ("Rigid", 6), (" rigid ", 6)],
)
def test_registration_types_map_to_greedys_dof(text, dof):
    assert parse_registration_type(text) == dof


@pytest.mark.parametrize("text", ["Deformable", "deform", "", "Afine", "Affine-9", "Affine-x", "Affine-6-12"])
def test_anything_else_is_refused(text):
    with pytest.raises(InputError):
        parse_registration_type(text)


def test_deformable_refusal_says_what_changed():
    with pytest.raises(InputError, match="Deformable registration, which CaPTk's Preprocessing -reg ran for any other value"):
        parse_registration_type("Deformable")


def test_defaults_are_the_original_gears(make_context):
    assert parse_config(make_context(inputs=INPUTS)) == GearConfig("/in/moving.nii.gz", "/in/fixed.nii.gz", 12, False, "output.nii.gz")


def test_options_are_read(make_context):
    config = {"registration_type": "Rigid", "intermediate_files": 1, "output_file_name": "reg.nii.gz"}
    assert parse_config(make_context(config, INPUTS)) == GearConfig("/in/moving.nii.gz", "/in/fixed.nii.gz", 6, True, "reg.nii.gz")


def test_a_missing_input_is_refused(make_context):
    with pytest.raises(InputError, match=r"missing required input\(s\): \['fixed_image'\]"):
        parse_config(make_context(inputs={"moving_image": "/in/moving.nii.gz"}))


@pytest.mark.parametrize("bad", ["../escape.nii.gz", "sub/dir.nii.gz", "..", "win\\style.nii.gz"])
def test_an_output_name_that_is_a_path_is_refused(make_context, bad):
    with pytest.raises(InputError, match="must be a file name"):
        parse_config(make_context({"output_file_name": bad}, INPUTS))


def test_an_output_name_without_nii_gz_is_refused(make_context):
    with pytest.raises(InputError, match="must end in .nii.gz"):
        parse_config(make_context({"output_file_name": "registered.nii"}, INPUTS))


def test_intermediate_files_must_be_0_or_1(make_context):
    with pytest.raises(InputError, match="intermediate_files must be 0 or 1"):
        parse_config(make_context({"intermediate_files": 2}, INPUTS))
