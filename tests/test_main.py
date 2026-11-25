import numpy as np
import pytest
import SimpleITK as sitk

from gear_functions import captk_reg, main


def inputs(files):
    return {"moving_image": files["moving"], "fixed_image": files["fixed"]}


def test_a_run_writes_the_registered_image_on_the_fixed_grid(make_context, phantom_files):
    context = make_context(inputs=inputs(phantom_files))
    assert main.run(context) == 0
    assert [p.name for p in context.output_dir.iterdir()] == ["output.nii.gz"]  # no matrix unless asked for
    registered = sitk.ReadImage(str(context.output_dir / "output.nii.gz"))
    assert registered.GetSize() == phantom_files["fixed_image"].GetSize()
    assert registered.GetPixelID() == sitk.sitkFloat32


def test_intermediate_files_adds_the_matrix_under_capTks_name(make_context, phantom_files):
    context = make_context({"intermediate_files": 1, "output_file_name": "reg.nii.gz", "registration_type": "Rigid"}, inputs(phantom_files))
    assert main.run(context) == 0
    assert sorted(p.name for p in context.output_dir.iterdir()) == ["affine_movingTOfixed_NMI.mat", "reg.nii.gz"]
    matrix = np.loadtxt(context.output_dir / "affine_movingTOfixed_NMI.mat")
    assert abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-6  # 6 DOF: a rotation, no scaling or shear


def test_the_registration_is_close_to_the_truth(make_context, phantom_files):
    from tests import phantom

    context = make_context({"intermediate_files": 1}, inputs(phantom_files))
    assert main.run(context) == 0
    matrix = np.loadtxt(context.output_dir / "affine_movingTOfixed_NMI.mat")
    assert phantom.displacement(phantom_files["points"], matrix, phantom_files["truth"]).mean() < 1.5


def test_an_image_without_variation_is_refused_before_any_work(make_context, phantom_files, tmp_path):
    flat = str(tmp_path / "flat.nii.gz")
    sitk.WriteImage(sitk.Image(8, 8, 8, sitk.sitkFloat32), flat)
    context = make_context(inputs={"moving_image": flat, "fixed_image": phantom_files["fixed"]})
    assert main.run(context) == 1
    assert list(context.output_dir.iterdir()) == [] and list(context.work_dir.iterdir()) == []


def test_deformable_is_refused_with_exit_1(make_context, phantom_files):
    context = make_context({"registration_type": "Deformable"}, inputs(phantom_files))
    assert main.run(context) == 1
    assert list(context.output_dir.iterdir()) == []


def test_a_failure_inside_greedy_is_not_swallowed(make_context, phantom_files, monkeypatch):
    def broken(argv):
        raise RuntimeError("greedy blew up")

    monkeypatch.setattr(captk_reg, "main", broken)
    with pytest.raises(RuntimeError, match="greedy blew up"):  # the job must fail with its traceback
        main.run(make_context(inputs=inputs(phantom_files)))
