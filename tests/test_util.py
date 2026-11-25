import os

import numpy as np
import pytest
import SimpleITK as sitk

from gear_functions.util import InputError, available_cpus, check_image


def write(tmp_path, array, name="image.nii.gz"):
    path = str(tmp_path / name)
    sitk.WriteImage(sitk.GetImageFromArray(np.asarray(array, np.float32)), path)
    return path


def test_available_cpus_follows_the_affinity_mask(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert available_cpus() == 3


def test_an_ordinary_image_is_accepted(tmp_path):
    check_image(write(tmp_path, np.random.default_rng(0).random((6, 7, 8))), "moving_image")


def test_a_file_that_is_not_an_image_is_refused(tmp_path):
    path = tmp_path / "notes.nii.gz"
    path.write_text("not an image")
    with pytest.raises(InputError, match="moving_image: notes.nii.gz cannot be read"):
        check_image(str(path), "moving_image")


def test_a_4d_image_is_refused(tmp_path):
    path = str(tmp_path / "four_d.nii.gz")
    volume = sitk.GetImageFromArray(np.random.default_rng(0).random((6, 7, 8)).astype(np.float32))
    sitk.WriteImage(sitk.JoinSeries([volume, volume, volume]), path)
    with pytest.raises(InputError, match="expected a 3D image, but four_d.nii.gz has 4 dimensions"):
        check_image(path, "fixed_image")


@pytest.mark.parametrize("array", [np.zeros((6, 7, 8)), np.full((6, 7, 8), 5.0), np.full((6, 7, 8), np.nan)])
def test_an_image_without_intensity_variation_is_refused(tmp_path, array):
    with pytest.raises(InputError, match="no intensity variation"):
        check_image(write(tmp_path, array), "fixed_image")
