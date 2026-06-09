# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`luseepy` is a Python package for simulating LuSEE-Night radio telescope observations on the lunar far side. It simulates instrument beams, sky models, and produces mock observation data (waterfall arrays) stored as FITS files.

## Environment Setup

Required environment variables:
- `LUSEE_DRIVE_DIR` — path to the LuSEE-Night Google Drive checkout (beam files, sky maps)
- `LUSEE_OUTPUT_DIR` — output directory for simulation results
- `LUSEEPY_PATH` — path to the luseepy checkout (optional)

Python 3.12 is required (`pyproject.toml` pins `requires-python = "==3.12.*"`). Build backend is flit. Local editable install:
```bash
pip install -e ".[dev]"        # editable + pytest
pip install -e ".[cuda12]"     # GPU jax on Linux (or [cuda13])
```
The `[croissant]` extra is legacy — `croissant-sim` and `s2fft` are already core dependencies. Docker (`docker/README.md`) is deprecated.

## Running Tests

Pytest is the primary runner. `pyproject.toml` sets `testpaths = ["tests"]`, excludes `tests/attic`, and registers an `integration` marker for tests that need real Drive data. `tests/conftest.py` exposes a `drive_dir` fixture that auto-skips when `LUSEE_DRIVE_DIR` is unset.

```bash
pytest                                           # full suite
pytest tests/test_observation.py                 # single file
pytest tests/test_sim.py::test_name              # single test
pytest -m "not integration"                      # skip Drive-dependent tests
```

Legacy script-style tests (`tests/SimTest.py`, `tests/SimReadTest.py`, `tests/LunarCalendarTest.py`, `tests/CoordTest.py`) still exist and can be run directly with `python`, but new tests use the pytest `test_*.py` convention.

CI runs tests via `.github/workflows/luseepy-test.yml` with `LUSEE_DRIVE_DIR=Drive` (a local tarball extracted during CI).

## Running a Simulation

```bash
cd simulation
python driver/run_sim.py config/realistic_example.yaml
```

The driver reads a YAML config and writes a FITS file. See `simulation/config/realistic_example.yaml` for the config schema.

## Architecture

### Core Data Flow

```
Observation (time/location) + Beam(s) + SkyModel → Simulator → FITS output → Data
```

### Key Classes

**`lusee.Observation`** (`lusee/Observation.py`)
Defines the lunar observatory location and time range. Handles coordinate transforms between lunar topocentric (Alt/Az), galactic, and ICRS frames using `lunarsky` and `astropy`. The `time_range` parameter accepts lunar day integer, calendar year strings (`"CY2025"`), fiscal year (`"FY25"`), or UTC range strings (`"2025-02-01 to 2025-03-01"`).

**`lusee.Beam`** (`lusee/Beam.py`)
Loads beam E-field data from FITS files (complex Etheta, Ephi arrays over freq × theta × phi grid). Key methods:
- `rotate(deg)` — rotate beam around zenith
- `taper_and_smooth(taper, beam_smooth)` — apply ground/sky taper and Gaussian frequency smoothing
- `get_healpix_alm(lmax, ...)` — compute spherical harmonic alm coefficients; this is the main beam→simulator interface
- `power_stokes(cross)` — compute Stokes [I,Q,U,V] power, optionally cross-beam

**`lusee.BeamGauss`** (`lusee/BeamGauss.py`)
Analytical Gaussian beam for testing without FITS data.

**`lusee.BeamInterpolator`** (`lusee/BeamInterpolator.py`)
Smooth interpolation of beam alm products across a parameter space (e.g., rotation angle). Uses JAX for differentiability. Implements the same `get_healpix_alm` interface as `Beam`, enabling drop-in use in simulators.

**`lusee.BeamCouplings`** (`lusee/BeamCouplings.py`)
Models cross-coupling between antennas (two-port impedance). Used by simulators for off-diagonal beam combinations.

**`lusee.CachedBeam`** (`lusee/CachedBeam.py`)
JAX pytree base class for parameterized beam caches. Subclasses define free parameters and a `transform_beam` method; the base exposes a `.efbeams` property that simulators consume. Used by autodiff calibration / map-making flows where the beam is differentiable.

### Sky Models (`lusee/SkyModels.py`, `lusee/MonoSkyModels.py`)

All sky models expose `get_alm(freq_ndx, freq)` returning a list of healpy-format alm arrays, and a `frame` attribute (`"galactic"`, `"MCMF"`, or `"equatorial"`).

- `FitsSky` — loads ULSA maps from FITS (galactic frame)
- `ConstSky` / `ConstSkyCane1979` / `DarkAgesMonopole` — monopole models (MCMF frame)
- `GalCenter` — point-like galactic center model
- `HealpixSky` / `SingleSourceHealpixSky` — custom healpix maps

### Simulators

**`lusee.SimulatorBase`** (`lusee/SimulatorBase.py`)
Abstract base. `prepare_beams()` pre-computes beam alm products for all antenna combinations, storing them in `self.efbeams`. Output of `simulate()` is stored in `self.result` as a numpy array of shape `(Ntimes, Ncombinations, Nfreq)`.

**`lusee.TopoNumpySimulator`** (`lusee/DefaultSimulator.py`)
Topocentric numpy simulator — per-timestep rotation of galactic sky alms into the observer frame using healpy rotators. Uses `mean_alm()` for the beam–sky integral. This is the historical "default" simulator (the YAML driver accepts `engine: luseepy`/`default`/`lusee`/`numpy` as aliases for it).

**`lusee.TopoJaxSimulator`** (`lusee/TopoJaxSimulator.py`)
JAX port of the topocentric simulator. Differentiable through beam parameters; consumes `CachedBeam`/`BeamInterpolator` outputs.

**`lusee.NumpySimulator`** (`lusee/NumpySimulator.py`)
Reference numpy implementation; primarily used for cross-checking.

**`lusee.CroSimulator`** (`lusee/CroSimulator.py`)
Engine using the `croissant` library and JAX. Works in MEPA (Moon-centred Ephemeris Pole Axis) with `rot_alm_z` phase rotations rather than per-time full sky rotation. `CroSimulator` is `None` if croissant fails to import (the import is guarded in `lusee/__init__.py`).

**`lusee.CalibratorSimulator`** (`lusee/CalibratorSimulator.py`)
Simulates calibrator satellite passes. Takes an `Observation` with `calibrator_tracks` populated (see `lusee.CalibratorTrack`, `lusee.Satellite`, `lusee.ObservedSatellite`) and a list of beams; produces complex E-field traces of shape `(NTime, NBeam, NFreq)` per pass. Separate code path from the sky simulators.

### Map-Making (`lusee/MapMaker.py`)

Wiener filter sky reconstruction via CG with autodiff adjoints (Camacho et al. 2026). Key functions:

- `build_instrument(beam_file, obs_range, freq, lmax)` — set up CroSimulator with rotated/tapered beams and Tground=0
- `solve(sim, data, sky_template, sigma, signal_prior, method='cg')` — Wiener filter solve in a real parameterization θ = [Re(alm); Im(alm, m>0)]. Supports `method='cg'` (default, with diagonal C_l preconditioner) and `method='direct'` (dense Cholesky, same as the paper). The sky is real but beams are complex; JAX traces through the complex beam math and returns real gradients. No Wirtinger conjugation needed (θ is real).
- `compute_cl_prior(sky_model, lmax)` — S^{-1} = 1/C_l from a sky model
- `compute_radiometric_noise(data, combinations, delta_f_hz, delta_t_sec)` — per-sample σ from the radiometer equation (Eq. 9 of the paper): σ²_ij = (T_ii T_jj + |V_ij|²)/(2ΔfΔt)

See `docs/wirtinger_cg.md` for the math (real vs complex parameterization, null-space analysis).

### Output: `lusee.Data` (`lusee/Data.py`)

Reads simulator FITS output. Extends `Observation`. Indexed as `D[:, '01I', :]` (time slice, combination label, freq slice).

### Simulation Driver (`simulation/driver/run_sim.py`)

`SimDriver` class reads YAML config, instantiates the appropriate objects, calls `simulator.simulate()`, and writes FITS. The `engine` keyword (top-level or under `simulation:`) selects the back end:

| Value | Back end |
| --- | --- |
| `croissant` | `lusee.CroSimulator` (JAX / s2fft) |
| `luseepy`, `default`, `lusee`, `numpy` | `lusee.TopoNumpySimulator` |

Config fields: `paths`, `sky`, `beam_config`, `beams`, `observation`, `simulation`.

## Coordinate Conventions

- Beam files use theta (0=zenith) × phi (0–360°) grids with wraparound at last phi bin (phi[0] == phi[-1] for most operations; the `-1` index is dropped in alm computation)
- `lmax` is used consistently in healpy convention; `grid2healpix_alm_fast` uses `lmax+1` internally (different convention from `pyshtools.legendre`)
- The Euler rotation used in `TopoNumpySimulator` follows `XYZ` convention via `rot2eul`/`eul2rot`

## Version Conventions

- Version in `lusee/__init__.py` as `__version__` and `__comment__` (dev suffix for unreleased)
- New release: clean version → tag → bump to `x.y dev`
- API-breaking changes → increment major integer; small fixes → increment by 0.01
