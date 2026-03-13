# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`luseepy` is a Python package for simulating LuSEE-Night radio telescope observations on the lunar far side. It simulates instrument beams, sky models, and produces mock observation data (waterfall arrays) stored as FITS files.

## Environment Setup

Required environment variables:
- `LUSEE_DRIVE_DIR` — path to the LuSEE-Night Google Drive checkout (beam files, sky maps)
- `LUSEE_OUTPUT_DIR` — output directory for simulation results
- `LUSEEPY_PATH` — path to the luseepy checkout (optional)

Docker-based development (see `docker/README.md`) or local install:
```bash
pip install .                          # core install
pip install ".[croissant]"             # with optional CroSimulator support
```

## Running Tests

```bash
# Unit tests (can run without LUSEE_DRIVE_DIR for most)
python tests/LunarCalendarTest.py -v
python tests/CoordTest.py

# Integration tests (require LUSEE_DRIVE_DIR with beam/sky data)
python tests/SimTest.py
python tests/SimReadTest.py <path-to-fits>

# Compare CroSimulator vs DefaultSimulator (requires croissant + LUSEE_DRIVE_DIR)
cd simulation && python driver/test_cro_vs_default.py config/sim_choice_realistic.yaml
```

CI runs tests via `.github/workflows/luseepy-test.yml`. It uses `LUSEE_DRIVE_DIR=Drive` (a local tarball extracted during CI).

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

### Sky Models (`lusee/SkyModels.py`, `lusee/MonoSkyModels.py`)

All sky models expose `get_alm(freq_ndx, freq)` returning a list of healpy-format alm arrays, and a `frame` attribute (`"galactic"`, `"MCMF"`, or `"equatorial"`).

- `FitsSky` — loads ULSA maps from FITS (galactic frame)
- `ConstSky` / `ConstSkyCane1979` / `DarkAgesMonopole` — monopole models (MCMF frame)
- `GalCenter` — point-like galactic center model
- `HealpixSky` / `SingleSourceHealpixSky` — custom healpix maps

### Simulators

**`lusee.SimulatorBase`** (`lusee/SimulatorBase.py`)
Abstract base. `prepare_beams()` pre-computes beam alm products for all antenna combinations, storing them in `self.efbeams`. Output of `simulate()` is stored in `self.result` as a numpy array of shape `(Ntimes, Ncombinations, Nfreq)`.

**`lusee.DefaultSimulator`** (`lusee/DefaultSimulator.py`)
Per-timestep rotation of galactic sky alms into the observer frame using healpy rotators. Uses `mean_alm()` for the beam–sky integral.

**`lusee.CroSimulator`** (`lusee/CroSimulator.py`)
Alternative engine using the `croissant` library and JAX. Works in MCMF frame with `rot_alm_z` phase rotations rather than per-time full sky rotation. Optional install: `pip install ".[croissant]"`. `CroSimulator` is `None` if croissant is not installed.

### Output: `lusee.Data` (`lusee/Data.py`)

Reads simulator FITS output. Extends `Observation`. Indexed as `D[:, '01I', :]` (time slice, combination label, freq slice).

### Simulation Driver (`simulation/driver/run_sim.py`)

`SimDriver` class reads YAML config, instantiates the appropriate objects, calls `simulator.simulate()`, and writes FITS. Supports `engine: default` (and `engine: croissant` in development). Config fields: `paths`, `sky`, `beam_config`, `beams`, `observation`, `simulation`.

## Coordinate Conventions

- Beam files use theta (0=zenith) × phi (0–360°) grids with wraparound at last phi bin (phi[0] == phi[-1] for most operations; the `-1` index is dropped in alm computation)
- `lmax` is used consistently in healpy convention; `grid2healpix_alm_fast` uses `lmax+1` internally (different convention from `pyshtools.legendre`)
- The Euler rotation used in `DefaultSimulator` follows `XYZ` convention via `rot2eul`/`eul2rot`

## Version Conventions

- Version in `lusee/__init__.py` as `__version__` and `__comment__` (dev suffix for unreleased)
- New release: clean version → tag → new docker image → bump to `x.y dev`
- API-breaking changes → increment major integer; small fixes → increment by 0.01
