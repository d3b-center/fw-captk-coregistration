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


class FakeContext:
    """The parts of GearToolkitContext the gear reads."""

    def __init__(self, tmp_path, config=None, inputs=None):
        self.config = config or {}
        self._inputs = inputs or {}
        self.output_dir = tmp_path / "output"
        self.work_dir = tmp_path / "work"
        self.output_dir.mkdir()
        self.work_dir.mkdir()

    def get_input_path(self, name):
        return self._inputs.get(name)


@pytest.fixture
def make_context(tmp_path):
    return lambda config=None, inputs=None: FakeContext(tmp_path, config, inputs)
