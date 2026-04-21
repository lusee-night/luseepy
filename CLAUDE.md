# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`luseepy` is a Python package for simulating LuSEE-Night radio telescope observations on the lunar far side. It simulates instrument beams, sky models, and produces mock observation data (waterfall arrays) stored as FITS files.

For the end-to-end math trace of the simulator (sky harmonics, beam harmonics, the `mean_alm` inner product, ground coupling, T→V conversion), see `mathematical_description.tex` at the repo root (also mirrored in `docs/`). Consult it before answering questions about the physics or algebraic structure of the pipeline.

## Environment Setup

Required environment variables:
- `LUSEE_DRIVE_DIR` — path to the LuSEE-Night Google Drive checkout (beam files, sky maps)
- `LUSEE_OUTPUT_DIR` — output directory for simulation results
- `LUSEEPY_PATH` — path to the luseepy checkout (optional)

Dependencies are managed with `uv`. Python 3.12 is pinned in `pyproject.toml`.

```bash
uv sync                                # core install (includes croissant, hydra, jax, s2fft)
uv sync --extra cuda12                 # add GPU JAX (Linux)
uv sync --extra cuda13
```

`croissant-sim` is a core dependency, so `CroSimulator` is always available.

## Running Tests

Tests are pytest-based; `[tool.pytest.ini_options]` in `pyproject.toml` sets `testpaths = ["tests"]` and defines an `integration` marker for tests that require `LUSEE_DRIVE_DIR`.

```bash
uv run pytest                          # all tests
uv run pytest -m "not integration"     # skip tests needing LUSEE_DRIVE_DIR
uv run pytest tests/test_sim.py -v
```

CI runs tests via `.github/workflows/luseepy-test.yml`. It uses `LUSEE_DRIVE_DIR=Drive` (a local tarball extracted during CI).

## Running a Simulation

The driver uses Hydra. `simulation/driver/run_sim.py` is a thin wrapper around `simulation/driver/sim_driver.SimDriver` (via `run_sim_universal.run`).

```bash
cd simulation
python driver/run_sim.py --config-name=example                      # Hydra style
python driver/run_sim.py config/example.yaml                        # legacy path (auto-shimmed)
python driver/run_sim.py --config-name=example observation.lmax=64  # Hydra overrides
```

See `simulation/config/example.yaml` (starter) and `simulation/config/sim_choice_realistic.yaml` (full Croissant + plotting).

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
Alternative engine using the `croissant` library and JAX. Works in MEPA (Moon-centred Ephemeris Pole Axis) with `rot_alm_z` phase rotations rather than per-time full sky rotation.

**`lusee.JaxSimulator`** (`lusee/JaxSimulator.py`)
JAX-backed simulator using `s2fft` Wigner recursions for the per-time sky rotation. Supports an `extra_opts["time_batch_size"]` to control the mini-batch size of the vmapped rotation kernel (helps keep GPU memory bounded at large `lmax`). Selected via `simulation.engine: jaxsim` in the YAML config.

**`lusee.NumpySimulator`** (`lusee/NumpySimulator.py`)
Numpy reference implementation used for cross-checking against the JAX path (see `tests/test_np_wrapper.py`, `lusee/NpWrapper.py`).

### Map-Making (`lusee/MapMaker.py`)

Wiener filter sky reconstruction via CG with autodiff adjoints (Camacho et al. 2026). Key functions:

- `build_instrument(beam_file, obs_range, freq, lmax)` — set up CroSimulator with rotated/tapered beams and Tground=0
- `solve(sim, data, sky_template, sigma, signal_prior, method='cg')` — Wiener filter solve in a real parameterization θ = [Re(alm); Im(alm, m>0)]. Supports `method='cg'` (default, with diagonal C_l preconditioner) and `method='direct'` (dense Cholesky, same as the paper). The sky is real but beams are complex; JAX traces through the complex beam math and returns real gradients. No Wirtinger conjugation needed (θ is real).
- `compute_cl_prior(sky_model, lmax)` — S^{-1} = 1/C_l from a sky model
- `compute_radiometric_noise(data, combinations, delta_f_hz, delta_t_sec)` — per-sample σ from the radiometer equation (Eq. 9 of the paper): σ²_ij = (T_ii T_jj + |V_ij|²)/(2ΔfΔt)

See `docs/wirtinger_cg.md` for the math (real vs complex parameterization, null-space analysis).

### Output: `lusee.Data` (`lusee/Data.py`)

Reads simulator FITS output. Extends `Observation`. Indexed as `D[:, '01I', :]` (time slice, combination label, freq slice).

### Simulation Driver (`simulation/driver/`)

`sim_driver.SimDriver` reads a Hydra-resolved config dict, instantiates the appropriate objects, calls `simulator.simulate()`, and writes FITS. `run_sim.py` / `run_sim_universal.py` are the Hydra entry points; `run_sim_universal._apply_legacy_config_shim` rewrites a bare-path config argument (e.g. `config/example.yaml`) into Hydra's `--config-path=... --config-name=...` form. Supported engines via `simulation.engine`: `default`, `croissant`, `jaxsim`. Config fields: `paths`, `sky`, `beam_config`, `beams`, `observation`, `simulation`.

Other drivers in `simulation/driver/`: `run_Cro_sim.py`, `run_calibrator_sim.py` (+ `calibrator_sim_driver.py`), `run_batch.py`.

## Coordinate Conventions

- Beam files use theta (0=zenith) × phi (0–360°) grids with wraparound at last phi bin (phi[0] == phi[-1] for most operations; the `-1` index is dropped in alm computation)
- `lmax` is used consistently in healpy convention; `grid2healpix_alm_fast` uses `lmax+1` internally (different convention from `pyshtools.legendre`)
- The Euler rotation used in `DefaultSimulator` follows `XYZ` convention via `rot2eul`/`eul2rot`

## Version Conventions

- Version in `lusee/__init__.py` as `__version__` and `__comment__` (dev suffix for unreleased)
- New release: clean version → tag → bump to `x.y dev`
- API-breaking changes → increment major integer; small fixes → increment by 0.01
