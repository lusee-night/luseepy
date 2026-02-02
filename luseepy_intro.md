# luseepy intro

This repository simulates LuSEE-Night style lunar radio observations by combining:
- instrument beams (from FITS or analytic Gaussian models),
- sky brightness models (monopole or FITS maps),
- lunar site coordinates and time ranges,
- and optional cross-couplings between antenna ports,
then writing synthetic spectra to FITS.

It is a forward model: for each time, it rotates the sky into the local lunar frame and computes beam-weighted sky temperatures (auto and cross) across frequencies. No PDE/ODE is solved; the main computation is spherical-harmonic convolution plus a ground-temperature term.

---

## Repository structure (python only)

### Core simulation pipeline
- `lusee/Simulation.py`
  - `Simulator`: prepares beams, rotates sky (if needed), integrates beam * sky in harmonic space, adds ground term, and writes results.
  - Key helpers: `mean_alm` (beam-sky coupling), `rot2eul`/`eul2rot` (rotation handling).

- `lusee/Observation.py`
  - `Observation`: defines time grid and lunar site (lat/long/height) and converts between local Alt/Az and sky coordinates using `lunarsky`.

- `lusee/Beam.py`
  - `Beam`: loads E-field beam patterns and impedance from FITS, provides power, cross-power, and healpix conversions.
  - `grid2healpix_*`: convert theta/phi grid data into spherical-harmonic coefficients or healpix maps.

- `lusee/BeamCouplings.py`
  - `BeamCouplings`: optional E-field cross-coupling (from 2-port beam simulations) for cross-correlation products.

- `lusee/SkyModels.py`
  - `FitsSky`: sky maps from FITS -> alms.
  - `ConstSky`, `ConstSkyCane1979`, `DarkAgesMonopole`, `GalCenter`: built-in toy/analytic sky models.

- `simulation/driver/run_sim.py`
  - YAML-driven CLI to build beams, sky, observation, and run `Simulator`.
  - This is the main entry point for batch runs.

### Analysis and utilities
- `lusee/Data.py`
  - `Data`: reads simulation FITS output and exposes convenience indexing (auto/cross, real/imag, voltage conversion).

- `lusee/Throughput.py`
  - Front-end electronics model: gain, impedance, noise, and conversion factors for voltages.

- `lusee/PCAanalyzer.py`
  - PCA tools for sky spectra and template SNR/chi2 analyses.

- `lusee/BeamGauss.py`
  - Analytic Gaussian beam generator, used when config says `beam_config.type: Gaussian`.

- `lusee/Satellite.py`
  - Orbital relay/calibrator toy model and transformations into the lunar topo frame.

- `lusee/LunarCalendar.py`
  - Lunar-day calendar computations using `lunarsky`, with optional caching.

---

## How simulation is done (step by step)

The main flow is in `simulation/driver/run_sim.py` and `lusee/Simulation.py`:

1. **Parse YAML config** (`run_sim.py`)
   - Defines sky model, beam model(s), observation site and time range, output paths.
   - `observation.freq` is a `np.arange(start, end, step)` list. The number of frequency bins is not hard-coded.

2. **Build observation** (`Observation`)
   - Uses lunar day or explicit time range to generate `times` with spacing `deltaT_sec`.
   - Builds `MoonLocation` from `lun_lat_deg`, `lun_long_deg`, and `lun_height_m`.

3. **Load or generate beams**
   - FITS beams (`Beam`) or analytic Gaussian beams (`BeamGauss`).
   - Rotated on the turntable by the configured angle.

4. **Set up beam couplings (optional)**
   - If YAML `beam_config.couplings` present, `BeamCouplings` uses 2-port beam data to compute a cross-power correction.

5. **Prepare beams for simulation** (`Simulator.prepare_beams`)
   - Applies a zenith/horizon taper in theta.
   - Computes cross-powers between antenna pairs.
   - Converts beam patterns to spherical harmonic coefficients (alms).
   - Computes a ground-coupling term based on the beam monopole.

6. **Simulate per time step** (`Simulator.simulate`)
   - If sky is in galactic frame, rotate sky alms into the local lunar frame using the site and time.
   - For each beam pair (auto and cross), compute:
     - `T_sky = <B_alm * S_alm>` via `mean_alm`.
     - `T_ground = Tground * groundPower` (real and imag if cross).
   - Store spectra per time, per product, per frequency.

7. **Write FITS output** (`Simulator.write`)
   - Output includes `data` array, `freq`, and `combinations`.
   - Per-beam impedance (`ZRe`, `ZIm`) are saved for throughput calculations.

---

## Data products and the 4-antenna / 16-product mapping

With 4 antennas, `run_sim.py` can generate all unique i<=j combinations. The simulator emits:
- 4 auto-correlation products (i==j): 4 real-valued spectra.
- 6 cross-correlation pairs (i<j): each yields real and imaginary parts, giving 12 spectra.

Total = 4 + 12 = 16 spectral products, matching the expected 4-antenna cross-correlation set. The output data array is shaped roughly as:

```
(time, products, frequency)
```

The number of frequency bins is purely defined by the input `freq` array. If you want 2048 bins, set `observation.freq` so `np.arange(start, end, step)` yields 2048 elements, or adjust the driver to accept an explicit list.

---

## Coordinates: role of landing latitude, longitude, and height

These values are used in `Observation` to create `MoonLocation` and to transform between lunar topocentric coordinates and sky coordinates with `lunarsky`:

- In `Simulator.simulate`, when the sky model is in the **galactic frame**, the code computes, for each time:
  - the galactic coordinates of the local zenith and local horizon direction,
  - constructs a rotation matrix from those vectors,
  - rotates sky spherical harmonics into the local frame.

Thus, the landing site (lat/long/height) determines how the sky rotates relative to the antenna beams over time. Height is included in the `MoonLocation` and can affect the exact coordinate transforms, but there is no explicit topography or terrain model beyond that.

If the sky model frame is set to `MCMF`, the simulator skips the rotation step (assumes the sky is already in the lunar body-fixed frame).

---

## Temperature and device parts

There are two temperature-related aspects in the current code:

1. **`Tground` in `Simulator`**
   - This is the only explicit physical temperature used in the forward model.
   - It adds a ground-emission term to each spectrum based on beam ground coupling.

2. **Electronics noise/throughput** in `Throughput`
   - The front-end model loads SPICE-simulated gains and noise spectra and computes impedance coupling factors and conversion to voltage units.
   - This is not tied to a thermal model of device parts; it is an electronic response model (noise_e, capacitance, resistance, gain tables).

There is no explicit thermal model of instrument components beyond these simplified parameters.

---

## Physics and numerical method

The simulator is not a time-domain physical solver. It uses linear radiometry and spherical-harmonic convolution:

- **Beam-sky coupling** uses
  - spherical harmonics (`alm`) of the beam power and sky brightness.
  - `mean_alm` essentially computes a spherical integral of `B(l,m) * S(l,m)` (with the correct normalization).

- **Rotation** uses
  - local basis vectors from `lunarsky` coordinate transforms,
  - a rotation matrix converted to Euler angles,
  - `healpy`'s `Rotator` to rotate sky alms.

- **Ground contribution** is a scalar temperature (`Tground`) scaled by a precomputed ground-power term from the beam monopole and optional cross-coupling.

So the core process is: rotate sky -> compute harmonic-space dot product with beams -> add ground term. There is no PDE/ODE integration or ray-tracing in this codebase.

---

## Notebooks (text-only summary)

The notebooks show how to use and diagnose the simulation outputs and beam models:
- `GaussianLbeam.ipynb`: builds a Gaussian beam, plots theta/phi cuts, and checks ground fraction.
- `grid2healpix_demo.ipynb`: demonstrates `grid2healpix` conversions.
- `sim_result.ipynb`: compares simulated spectra, different antenna lengths/angles, and throughput-based voltage scaling.
- `SNR.ipynb` and `eigen_analysis.ipynb`: PCA/SNR style analyses of simulated products.
- `satellite.ipynb`: tests satellite orbital geometry and lunar-topocentric transforms.

---

## Background from arXiv:2508.16773 (Linear map-making with LuSEE-Night)

Key points from the paper that map to this repository and its assumptions:

- Instrument context: LuSEE-Night is described as four ~3 m monopole antennas arranged as two horizontal crossed pseudo-dipoles on a rotational stage, sensitive to ~1–50 MHz, and producing 16 correlation products (4 auto + 6 complex cross). 
- The correlator channelizes into 2048 spectral bins covering 0–51.2 MHz with 25 kHz spacing.
- The analysis emphasizes that LuSEE-Night is not a traditional interferometer: due to close spacing and non-identical beams, each correlation product has its own beam response, so each product is treated as an independent beam-weighted integral of the sky.
- Map-making in the paper is a linear inversion (Wiener filter) that deconvolves the set of beam-weighted measurements into a low-resolution sky map and marginalizes systematics via the noise covariance.

How this connects to the code:

- The repository focuses on **forward modeling** those beam-weighted integrals (and their time evolution) rather than the map-making inversion. The `Simulator` computes the same class of observables the paper defines (auto and cross correlations treated independently) for a given sky model and time range.
- The paper’s 2048-bin, 0–51.2 MHz / 25 kHz channelization is not hard-coded here; the config-driven `freq` array in `run_sim.py` controls the binning. To match the paper, you would set `start=0`, `end=51.2`, `step=0.025` (MHz) so that `np.arange` yields 2048 bins. This aligns the simulator’s output products with the expected correlator outputs.

---

## Practical entry points

- Run a sim: `simulation/driver/run_sim.py <config.yaml>`
- Example configs: `simulation/config/example.yaml`, `simulation/config/realistic_example.yaml`, etc.
- Load data: `lusee.Data` for FITS output and `lusee.Throughput` for voltage/noise calculations.
