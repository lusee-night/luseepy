# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

`luseepy` is a Python package for simulating LuSEE-Night radio telescope observations on the lunar far side. It simulates instrument beams, sky models, and produces mock observation data (waterfall arrays) stored as FITS files. It also contains the downlink ingest pipeline (`lusee.ingest`) that converts raw CCSDS telemetry into HDF5/FITS science products.

## Environment Setup

Required environment variables:
- `LUSEE_DRIVE_DIR` — path to the LuSEE-Night Google Drive checkout (beam files, sky maps)
- `LUSEE_OUTPUT_DIR` — output directory for simulation results
- `LUSEEPY_PATH` — path to the luseepy checkout (optional)

Docker-based development (see `docker/README.md`) or local install:
```bash
pip install .                          # core install
pip install ".[croissant]"             # with optional CroSimulator support
pip install ".[ingest]"                # downlink ingest extras (h5py, ...)
```

The ingest pipeline additionally needs the external `uncrater` packet decoder (separate repository) and, optionally, the private `lusee_telemetry` decoder; both are soft dependencies — absent packages degrade output with warnings rather than failing imports.

## Running Tests

```bash
# What CI runs (pytest collects tests/ including tests/ingest/):
uv run pytest tests/ -v

# Most unit tests run without LUSEE_DRIVE_DIR; integration tests are
# marked and need beam/sky data. Ingest tests skip cleanly when their
# optional deps/data are unavailable (the Graham HDF5 regression needs
# LUSEE_GRAHAM_H5_DIR).

# Legacy standalone scripts (older entry points, still runnable):
python tests/SimTest.py
python tests/SimReadTest.py <path-to-fits>
```

CI runs tests via `.github/workflows/luseepy-test.yml`. It uses `LUSEE_DRIVE_DIR=Drive` (a local tarball extracted during CI).

## Running a Simulation

```bash
cd simulation
python driver/run_sim.py config/realistic_example.yaml
```

`run_sim.py` is a thin CLI over `simulation/driver/sim_driver.py`, whose `SimDriver` reads the YAML config, instantiates the objects, and writes a FITS file. Engines: `topo-numpy` (original NumPy path), `topo` (JAX), `croissant`. See `simulation/config/realistic_example.yaml` for the config schema.

## Architecture

### Core Data Flow

```
Observation (time/location) + Beam(s) + SkyModel → Simulator → FITS output → Data
Raw CCSDS downlink → lusee.ingest → HDF5/FITS products → IngestData
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

**`lusee.LabeledArray`** (`lusee/LabeledArray.py`)
Thin wrapper decorating one array with informational `units` and `frame` string labels (see the `FRAME_*` constants). Purely informational — no unit algebra, no checks; labels are dropped inside jit and by numpy/jnp functions. Attach labels at Python boundaries via `label()`/`relabel()`; hot jitted kernels stay bare-array.

**`lusee.SpectrometerGain`** (`lusee/GainModel.py`)
Predicts spectrometer gain spectra and converts raw counts (SDU) to physical units (nV/sqrt(Hz) signed ASD, or nV^2/Hz / V^2/Hz PSD). Per-PC model families selectable via `set_models`; batched conversion API for bulk HDF5 work. Gain artifacts are vendored in `lusee/data/gain/`.

**`lusee.frequencies`** (`lusee/frequencies.py`)
`FrequencyMap` — a JAX-pytree index/interpolation map from a target frequency grid onto a source grid (built on the host in NumPy, differentiable two-point blend inside jit). Helpers: `frequencies_from_config`, `canonical_frequencies`, `frequency_indices_from_values`, `canonicalize_frequencies`.

**`lusee.spice_utils`** (`lusee/spice_utils.py`)
`ensure_lunarsky_moon_frame()` furnishes lunarsky's MOON_ME kernels into spiceypy's global pool (needed by CroSimulator).

### Sky Models (`lusee/SkyModels.py`, `lusee/MonoSkyModels.py`)

All sky models expose `get_alm(freq_ndx, freq)` returning a list of healpy-format alm arrays, and a `frame` attribute (`"galactic"`, `"MCMF"`, or `"equatorial"`).

- `FitsSky` — loads ULSA maps from FITS (galactic frame)
- `ConstSky` / `ConstSkyCane1979` / `DarkAgesMonopole` — monopole models (MCMF frame)
- `GalCenter` — point-like galactic center model
- `HealpixSky` / `SingleSourceHealpixSky` — custom healpix maps

### Simulators

**`lusee.SimulatorBase`** (`lusee/SimulatorBase.py`)
Abstract base. `prepare_beams()` pre-computes beam alm products for all antenna combinations, storing them in `self.efbeams`. Frequency handling goes through `FrequencyMap` (`freq_map_beam` / `freq_map_sky`; off-grid target frequencies are interpolated). Output of `simulate()` is stored in `self.result` as a bare numpy array of shape `(Ntimes, Ncombinations, Nfreq)`; `result_labeled` returns a units/frame-labeled view.

**`lusee.TopoNumpySimulator`** (`lusee/NumpySimulator.py`; legacy import path `lusee/DefaultSimulator.py`)
The original NumPy engine: per-timestep rotation of galactic sky alms into the observer frame using healpy rotators.

**`lusee.TopoJaxSimulator`** (`lusee/TopoJaxSimulator.py`)
JAX version of the topo engine with jitted/vmapped alm rotation and contraction kernels.

**`lusee.CroSimulator`** (`lusee/CroSimulator.py`)
Alternative engine using the `croissant` library and JAX. Works in MEPA (Moon-centred Ephemeris Pole Axis) with `rot_alm_z` phase rotations rather than per-time full sky rotation. Optional install: `pip install ".[croissant]"`. `CroSimulator` is `None` if croissant is not installed.

**`lusee.CalibratorSimulator`** (`lusee/CalibratorSimulator.py`)
Simulates the orbiting calibrator observations.

### Map-Making (`lusee/MapMaker.py`)

Wiener filter sky reconstruction via CG with autodiff adjoints (Camacho et al. 2026). Key functions:

- `build_instrument(beam_file, obs_range, freq, lmax)` — set up CroSimulator with rotated/tapered beams and Tground=0
- `solve(sim, data, sky_template, sigma, signal_prior, method='cg')` — Wiener filter solve in a real parameterization θ = [Re(alm); Im(alm, m>0)]. Supports `method='cg'` (default, with diagonal C_l preconditioner) and `method='direct'` (dense Cholesky, same as the paper). The sky is real but beams are complex; JAX traces through the complex beam math and returns real gradients. No Wirtinger conjugation needed (θ is real).
- `compute_cl_prior(sky_model, lmax)` — S^{-1} = 1/C_l from a sky model
- `compute_radiometric_noise(data, combinations, delta_f_hz, delta_t_sec)` — per-sample σ from the radiometer equation (Eq. 9 of the paper): σ²_ij = (T_ii T_jj + |V_ij|²)/(2ΔfΔt)

See `docs/wirtinger_cg.md` for the math (real vs complex parameterization, null-space analysis).

### Output: `lusee.Data` (`lusee/Data.py`)

Reads simulator FITS output. Extends `Observation`. Indexed as `D[:, '01I', :]` (time slice, combination label, freq slice); returns `LabeledArray`s (K, or V^2/Hz for the `...V` suffix).

### Downlink Ingest (`lusee/ingest/`)

Converts raw CCSDS binary downlinks (or extracted uncrater session dirs) into HDF5 (layout v3) / FITS science products plus sanity plots. Stages: CCSDS frame recovery → logical-packet reassembly → identity assignment → session split → decode to `Products` → `write_hdf5` / `write_fits`. Entry points: `process_flash`, `process_session`, `parse_flash`. Spectra are stored in SDU (bit-slice restored exactly once); physical-unit conversion is deferred to `SpectrometerGain`. `IngestData(Observation)` / `lusee.ingest.load()` read the products back with `lusee.Data`-style indexing; the time axis requires a recorded `time_scale` constant or an explicit `assume_scale=` (the writers record `time_scale` / `clock_source` / `clock_epoch_isot` provenance).

## Four-port instrument response (migration path)

The four-port refactor is implemented on this branch behind internal
modules; the legacy per-antenna API above remains the exported default
until the public cutover. Key pieces:

- **`lusee.InstrumentResponse`** — one coupled 4-port response loaded from
  instrument FITS v3 (`H_theta/H_phi` bare effective lengths in meters,
  dense complex `ZA`, `Rsky`/`Rmoon`/`Rloss`). Pair-Stokes maps/alms via
  the co-developed croissant polarization layer (mwss L=180 grid).
- **`lusee.ReceiverImpedance`** — differentiable receiver models;
  `JFETReceiver` defaults are the measured fmpre0/2/5/7 Bode fits;
  `spare_preamp_average_zload()` reproduces the HFSS export's load matrix.
- **`lusee.FullStokesSimulator`** — `FullStokesCroSimulator` (MEPA
  z-phases) and `FullStokesTopoJaxSimulator` (per-time Wigner), both full
  IQUV; covariance assembly in `lusee.Covariance`
  (`C_v = M K M†`, `M = ZL(ZA+ZL)^-1`).
- **`beam_conversion/receive_csv.py`** — five explicit input contracts;
  the default (since 2026-08-06) is bare effective length in meters
  (`Bare_Effective_Length_Fields_*` exports: `--input-kind bare
  --field-kind effective-length --field-units m --field-amplitude ratio
  --phi-source-zero-deg 180`); `loaded` (solver-side `ZL(ZA+ZL)^-1 H`
  fields, unloaded with a persisted `ZLoad` payload) remains for the
  legacy `Receive_Matrix_Fields_*` exports — both routes agree to 1e-15.
  Drivers: `lusee_bgl_v16.py` (legacy loaded conversion),
  `symmetrize_response.py` (C4 group average).
- Validated real artifacts live outside the repo in
  `../receive_matrix/lusee_bgl_v16_response_v3{,_c4sym}.fits`.
- Complex128 response validation requires `JAX_ENABLE_X64=True`; the
  croissant development checkout installs with
  `uv pip install -e ../croissant -e .` (pins the slosar/s2fft fork).

Docs: `docs/instrument_response.md` (usage + converter contracts),
`docs/four_port_physics_review.md` (physics review + provenance record),
`docs/old_vs_new.md` (legacy-pipeline comparison and C4 symmetry study).

## Coordinate Conventions

- Beam files use theta (0=zenith) × phi (0–360°) grids with wraparound at last phi bin (phi[0] == phi[-1] for most operations; the `-1` index is dropped in alm computation)
- `lmax` is used consistently in healpy convention; `grid2healpix_alm_fast` uses `lmax+1` internally (different convention from `pyshtools.legendre`)
- The Euler rotation used in `TopoNumpySimulator` follows `XYZ` convention via `rot2eul`/`eul2rot`
- SPICE/ephemeris code expects TDB: convert explicitly (e.g. `times[0].tdb.jd`) — passing a raw `.jd` (UTC-scale) introduces a ~69 s offset

## Four-port response: conventions and pitfalls

Learned while building the monopole / point-source / polarimeter
diagnostics in `big_refactor/old_vs_new/` (2026-08-06..10). Each item
below cost real debugging time; do not re-derive them from the headers.

### Topocentric azimuth

Stored response `phi` is the ENU azimuth measured **from East towards
North**, i.e. for a source at altitude `alt` and astronomical azimuth
`az` (from North towards East),

    theta = pi/2 - alt        phi = pi/2 - az

This is consistent with `SRCXAXIS=west` / `SRCYAXIS=south` / `SRCAZ0=180`
and with `BeamGauss`'s documented `az=0 -> E`, but it was **pinned
empirically**, not read off the header: pushing an I-only point source
through the full harmonic path (`FullStokesCroSimulator`) and comparing
the four auto time series against the direct kernel gives per-port
correlation 1.0000 for `phi = pi/2 - az` and only 0.85-0.92 for
`phi = az`. The check lives in
`big_refactor/old_vs_new/check_az_convention.py`.

WARNING: the legacy `CalibratorSimulator` feeds `LunarTopo.az`
(astronomical, from North) straight into `Beam.interp_Etheta`, whose
docstring says `az=0 -> E`. That path is internally inconsistent and must
not be used as the convention reference.

### Angular interpolation interpolates the KERNEL, not the fields

`InstrumentResponse._sample_periodic_maps` (used by `pair_stokes_at`, and
hence by `FullStokesCalibratorSimulator`) bilinearly interpolates the
already-formed pair-Stokes kernel. That is **not** the same as
interpolating `H` and then forming the quadratic products: on the 1 deg
grid the two differ by ~6e-4 relative. Consequences:

- Reproducing the library requires forming the kernel on the four
  surrounding grid nodes and interpolating that (see
  `old_vs_new/point_source_sim.py:sample_pair_kernel`), not sampling `H`
  first.
- Off-grid directions break exact algebraic identities at the ~1e-4
  level. The rank-1 structure below, and the "total polarized fraction
  = 1" identity, are exact only at grid nodes. Evaluate on nodes when you
  need machine precision.

### Performance traps

- `InstrumentResponse.target_matrices()` re-derives `Rsky` with a full
  spherical-harmonic transform over **all** native channels (~840 s for
  150 channels). If you already have a validated artifact, read
  `resp.ZA`, `resp.Rsky_native`, `resp.Rmoon_native`, `resp.Rloss_native`
  directly.
- `pair_stokes_at()` calls `all_pair_stokes_maps()` with no `freq_ndx`,
  materializing 10 pairs x 4 Stokes x 150 freqs x 91 x 361 complex
  (~3 GB) on **every** call. `FullStokesCalibratorSimulator.simulate()`
  therefore costs ~2 min per call. Slice the response to a few native
  channels before using it as a cross-check reference.
- With `T_moon = T_ant = 0` the `Rmoon`/`Rloss` terms in
  `assemble_open_covariance` are identically zero, so a source-only
  calculation needs neither.

### Rank structure of the port covariance

For a source in one direction, `v = R e` with `R = M H` the loaded 4x2
response, so `C = R B R+` with `B` the 2x2 source coherency. Hence
`rank(C) <= 2` for **any** single point source and `rank(C) = 1` exactly
when it is fully polarized; `Z_L` is invertible so neither the receiver
nor the mutual coupling changes this. Verified to <6e-16 on an exact grid
node (`old_vs_new/make_rank_figure.py`). A diffuse sky superposes many
such terms and is full rank. Useful as a calibration-free null test
whenever one source dominates.

## Version Conventions

- Version in `lusee/__init__.py` as `__version__` and `__comment__` (dev suffix for unreleased)
- New release: clean version → tag → new docker image → bump to `x.y dev`
- API-breaking changes → increment major integer; small fixes → increment by 0.01
