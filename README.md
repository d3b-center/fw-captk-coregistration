# CaPTk Preprocessing coregistration

Flywheel gear that registers a moving 3D NIfTI image to a fixed 3D NIfTI image and writes the moving image resliced
onto the fixed image's grid. It does what CaPTk's `Preprocessing -reg` did, with upstream
[Greedy](https://github.com/pyushkevich/greedy) (the `picsl_greedy` wheel) instead of CaPTk's bundled binary.

### Classification

_Category:_ Analysis

_Gear Level:_

- [ ] Project
- [ ] Subject
- [x] Session
- [x] Acquisition
- [ ] Analysis

The gear reads only its two input files and writes only files, so it does not check where it was launched.

## Dependencies

`picsl_greedy` 1.4.0.3 (Greedy, Apache-2.0), `SimpleITK` 2.5.8, `numpy`, `flywheel-gear-toolkit`. Pinned in
`requirements_greedy.txt` and `requirements_main.txt`; the image is `python:3.12-slim-bookworm` plus those (743 MB; the
CaPTk image the gear used before is 6.08 GB).

## Inputs

- **moving_image** (nifti, required): a single 3D volume to register to the fixed image.
- **fixed_image** (nifti, required): a single 3D volume. The output is on its grid.

The two images need to overlap roughly: registration starts by aligning their centres. No other preprocessing is
required; no bias correction is applied.

## Configuration

- **registration_type** (default `Affine`): `Rigid` (6 degrees of freedom), `Affine` (12), or `Affine-6`, `Affine-7`
  (rigid plus isotropic scaling), `Affine-12`. Not case-sensitive. Anything else is refused (see Limitations).
- **intermediate_files** (default `0`): `1` also saves the transformation matrix.
- **output_file_name** (default `output.nii.gz`): must end in `.nii.gz`.

## Usage

`gear_functions/main.py`:

1. Reads the options and refuses what cannot be used (`gear_functions/parser.py`), then refuses an input that is not a 3D
   image or has no intensity variation (`gear_functions/util.py`). A refusal logs one line and exits with code 1.
2. Runs `gear_functions/captk_reg.py` with one modality: Greedy `-a -m NMI -ia-image-centers -n 100x50x5 -dof <6|7|12>`
   (the values `Preprocessing -reg` used), then a linear reslice onto the fixed image.
3. Copies the result to `output/<output_file_name>` and, with `intermediate_files=1`, the matrix to
   `output/affine_<moving>TO<fixed>_NMI.mat` (Greedy `.mat` format: fixed to moving, RAS).

`captk_reg.py` is the registration tool of `captk-greedy-reg`, ported unchanged. It also has a multi-modality mode and N4
bias correction (the CaPTk BraTS recipe) that this gear does not use.

## Outputs

- `output/<output_file_name>`: the registered image, float32, on the fixed image's grid.
- `output/affine_<moving>TO<fixed>_NMI.mat`: only with `intermediate_files=1`.

## Limitations

- **Deformable registration is not available.** `Preprocessing -reg` ran a deformable registration for any
  `registration_type` that was not Rigid or Affine, including typos. The gear now refuses those values with a message. This
  removes a capability; whether to bring it back is for the maintainers to decide.
- **Runs are not reproducible.** Greedy 1.4 randomizes its sample jitter on every run and has no seed option. Repeated runs
  of one input differ by 0.07-0.15 mm on real T1/T1CE data and up to ~0.9 mm on the synthetic 2 mm phantom of the tests.
  The Greedy bundled with CaPTk gave identical matrices when repeated at the same thread count (checked with its macOS build).
- The registered image is float32 whatever the input type; Greedy reslicing wrote the input's own type.
- Only `.nii.gz` output names, 3D inputs, and the metric and iterations above (as before, they were never options).
- The thread count is the job's CPU affinity; a CPU quota (`docker --cpus`) is not seen, which can slow the run.
- Greedy prints several hundred log lines per registration.

## Development

```
python -m venv .venv && .venv/bin/pip install -r requirements_greedy.txt -r requirements_main.txt pytest
.venv/bin/python -m pytest
```

`tests/test_golden.py` holds the registration to values the original gear (commit `1df1c3e`, CaPTk 1.8.1 image) computed on
the synthetic phantom of `tests/phantom.py`, and to the known motion applied to it; `tests/golden/make_golden.py` records
how. Tolerances come from 40 repeated runs (see the test's docstring).

Running the container locally with `flyw` (CLI 23.0.3), two things the docs do not say:

- `flyw gear config` does not handle integer options and writes them as strings (`"1"`); the parser accepts that.
- The toolkit reads `destination["type"]` when the gear exits, and a generated `config.json` has an empty destination, so the
  run ends in `KeyError: 'type'` after the outputs are written. Add `"destination": {"id": "000000000000000000000000", "type": "analysis"}` to the
  prepared `config.json` (it is git-ignored). On Flywheel the engine writes it.

```
flyw gear config --new .
flyw gear config -i moving_image=moving.nii.gz -i fixed_image=fixed.nii.gz .
flyw gear run --prepare .        # prints the prepared folder; add the destination to its config.json
flyw gear run <prepared folder>  # inspect <prepared folder>/output
```

## Major gear version info

**0.1.0**: the registration engine changed from CaPTk's `Preprocessing -reg` (CaPTk 1.8.1 image) to upstream Greedy 1.4.0.3.
Names, inputs, options and defaults are unchanged. Different: deformable registration is refused, output is float32, runs are
not reproducible, the image is 743 MB instead of 6 GB, and the manifest no longer carries the old container's environment.
