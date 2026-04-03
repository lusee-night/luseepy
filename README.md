# luseepy
![workflow](https://github.com/lusee-night/luseepy/actions/workflows/luseepy-test.yml/badge.svg)

## About
A set of python utilities for performing various LuSEE-Night related calculations.

## Documentation

This software is documented on the ["Read the Docs" pages](https://luseepy.readthedocs.io/en/latest/)

## Cached Data

There are datasets stored in the LuSEE-Night [Google Drive](https://drive.google.com/drive/folders/0AM52i9DVjqkAUk9PVA).

## Docker (deprecated)

The Docker-based workflow is **deprecated**. Prefer a local Python environment (see [Developing](#developing)). Legacy build notes remain under [`docker/README.md`](docker/README.md) for reference only.

## Developing

Use a virtual environment and an editable install from the `luseepy` repository root:

```bash
git clone git@github.com:lusee-night/luseepy.git
cd luseepy
python -m venv .venv
source .venv/bin/activate   # or appropriate activate script on your OS
pip install -e ".[dev]"
```

Set the environment variables in [Environment variables](#environment-variables) (at minimum `LUSEE_DRIVE_DIR` when running simulations that need Drive data). Run tests or scripts with `python` directly, for example:

```bash
python tests/LunarCalendarTest.py
```

If you still use the legacy `setup_env.sh` helpers (`lpython`, `ljupyter`, etc.), they assume Docker and `LUSEE_IMAGE`; that path is unmaintained.

## Environment variables

User is expected to set up the following environment variables:

 * `LUSEEPY_PATH` -- path to the luseepy checkout
 * `LUSEEOPSIM_PATH` -- path to the lusee opsim package (if used).
 * `LUSEE_DRIVE_DIR` -- path to the checkout of the LuSEE-Night Google Drive

The legacy `setup_env.sh` script may also define `LUSEE_IMAGE` (Docker image name); it is only relevant if you use the deprecated container workflow.



## Singularity

__NB. The example below corresponds to an early version of software, and reference to the image below is deprecated.__

The `tests` folder contains CI-related and other testing scripts. Here's an example
of a simple test run with Singularity, on a SDCC/BNL node, from the `luseepy` folder:

```bash
singularity exec -B /direct/phenix+u/mxmp/projects/luseepy --env PYTHONPATH=/direct/phenix+u/mxmp/projects/luseepy docker://lusee/lusee-night-foundation:0.1 ./tests/LunarCalendarTest.py
```


## Cutting a new version

Cutting a new version entails:
 * having a clean (non dev) version in `__init__.py`
 * updating `setup_env.sh` (if still in use)
 * tagging the github
 * bumping version again in `__init__.py` to a +0.1 and a dev
 
Any small fixes after the fact should be bumping version by 0.01.
Large changes that break API should bump version into next integer.


## Starting with simulations

Go to `simulation` sub-directory. Make sure the `$LUSEE_DRIVE_DIR` points to the stuff from the LUSEE Drive that the simulations needs (ULSA maps, beam). Run a short simulation as

```
python driver/run_sim.py config/realistic_example.yaml
```

### Simulation engine (`simulation.engine`)

The driver selects the back end from the YAML **engine** keyword. You may set either top-level `engine` or `simulation.engine` (top-level wins if both are present).

| Config value | Back end |
| --- | --- |
| `croissant` | [CROISSANT](https://github.com/christianhbye/croissant) (spherical-harmonics / JAX) via `lusee.CroSimulator` |
| `luseepy` | Built-in `lusee.DefaultSimulator` |
| `default`, `lusee`, `numpy` | Same as `luseepy` (aliases) |

Croissant requires compatible installs of `croissant-sim` and `s2fft` (see `pyproject.toml`). Example:

```yaml
simulation:
  engine: croissant
  output: sim_example.fits
```

See `simulation/config/example.yaml` and `simulation/config/sim_choice_realistic.yaml` for full examples.

In the same directory, open a jupyter notebook and plot the results for the NE cross-correlation, imaginary part as:
```
import lusee
D=lusee.Data('output/sim_output.fits')
plt.imshow(D[:,'01I',:],aspect='auto',extent=(D.freq[0],D.freq[-1],len(D.times),0))
plt.colorbar()
plt.xlabel('frequency (MHz)')
plt.ylabel('time number')
```
