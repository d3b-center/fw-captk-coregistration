import SimpleITK as sitk
import pytest

from tests import phantom


@pytest.fixture(scope="session")
def phantom_files(tmp_path_factory):
    """The synthetic fixed/moving pair written as NIfTI, plus the points and truth to judge a registration by."""
    folder = tmp_path_factory.mktemp("phantom")
    fixed, moving, truth = phantom.make_pair()
    paths = {"fixed": str(folder / "fixed.nii.gz"), "moving": str(folder / "moving.nii.gz")}
    sitk.WriteImage(fixed, paths["fixed"])
    sitk.WriteImage(moving, paths["moving"])
    return {**paths, "fixed_image": fixed, "truth": phantom.lps_to_ras(truth), "points": phantom.brain_points(fixed)}
