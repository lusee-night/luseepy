# luseepy
![workflow](https://github.com/lusee-night/luseepy/actions/workflows/luseepy-test.yml/badge.svg)

## About
A set of python utilities for performing various LuSEE-Night related calculations.

## Documentation

This software is documented on the ["Read the Docs" pages](https://luseepy.readthedocs.io/en/latest/)

## Cached Data

There are datasets stored in the LuSEE-Night [Google Drive](https://drive.google.com/drive/folders/0AM52i9DVjqkAUk9PVA).

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

## Environment variables

User is expected to set up the following environment variables:

 * `LUSEEPY_PATH` -- path to the luseepy checkout
 * `LUSEEOPSIM_PATH` -- path to the lusee opsim package (if used).
 * `LUSEE_DRIVE_DIR` -- path to the checkout of the LuSEE-Night Google Drive
## Cutting a new version

Cutting a new version entails:
 * having a clean (non dev) version in `__init__.py`
 * tagging the github
 * bumping version again in `__init__.py` to a +0.1 and a dev
 
Any small fixes after the fact should be bumping version by 0.01.
Large changes that break API should bump version into next integer.


## Starting with simulations

Go to the `simulation` sub-directory. Make sure `$LUSEE_DRIVE_DIR` points to a checkout of the LuSEE Drive containing the four-port instrument response (`Simulations/BeamModels/BGL_v16/`). Run a short simulation as

```
python driver/run_sim.py config/four_port_example.yaml
```

This forward-models a full-Stokes sky through the coupled four-port instrument response (`lusee.InstrumentResponse`, FITS v3: bare effective lengths, dense antenna impedance, `Rsky`/`Rmoon`/`Rloss`) and a receiver model, and writes the physical 4x4 cross-correlation covariance in V^2/Hz per time and frequency. For a full lunar day of the ULSA sky through the measured JFET receiver chain, run `config/realistic_example.yaml` (takes a few minutes).

### Simulation engine (`simulation.engine`)

A config with a `response:` section selects the four-port pipeline; the **engine** keyword picks the time kernel:

| Config value | Back end |
| --- | --- |
| `croissant` | MEPA frame, diagonal-in-m z-phase time evolution (`lusee.FullStokesCroSimulator`) |
| `topo` | Per-time-step Wigner rotations in the topocentric frame (`lusee.FullStokesTopoJaxSimulator`) |

Both are built on [CROISSANT](https://github.com/christianhbye/croissant) and `s2fft` (pinned in `pyproject.toml`); `luseepy` furnishes the Moon-frame kernels bundled with `lunarsky` automatically.

Read and plot the output covariance, e.g. the 0x1 cross-correlation, imaginary part:

```python
import fitsio
import matplotlib.pyplot as plt

fname = "output/four_port_covariance.fits"
data = fitsio.read(fname, ext="data")      # (time, product, freq) in V^2/Hz
freq = fitsio.read(fname, ext="freq")      # MHz
labels = [row["label"].strip() for row in fitsio.read(fname, ext="products")]

plt.imshow(data[:, labels.index("01I"), :], aspect="auto",
           extent=(freq[0], freq[-1], data.shape[0], 0))
plt.colorbar()
plt.xlabel("frequency (MHz)")
plt.ylabel("time number")
```

The output also carries the response/receiver matrices (`ZA`, `ZL`, `M`, `Rsky`, `Rmoon`, `Rloss`), covariance diagnostics, and full provenance (response content hash, package versions) in additional HDUs.

The legacy scalar-beam pipeline (per-antenna `lusee.Beam` with the `luseepy`/`default`/`croissant` scalar engines) is deprecated; its configs remain under `simulation/config/` marked as such and need the deprecated beams under `Simulations/OldBeamModels/`.
