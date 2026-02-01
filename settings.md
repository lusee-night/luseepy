# Instrument‑related settings in this repository

This file documents the *instrument* parameters you can adjust **in this codebase** (not sky models or landing site). Each section explains what the parameter does, the physics intuition, and the math used in the code.

---

## 1) Beam model selection and rotation

**Where:** `simulation/driver/run_sim.py` (`beam_config.type`, `default_file`, per‑beam `angle`, `common_beam_angle`), and `lusee/Beam.py` / `lusee/BeamGauss.py`.

**What it changes:** The angular response pattern of each antenna, i.e., how strongly the instrument responds to different sky directions. This is a core “device” property. You can choose:

- **FITS beams** (`beam_config.type: fits`): E‑field patterns and impedance from electromagnetic simulations.
- **Gaussian beams** (`beam_config.type: Gaussian`): analytic, simplified beams used for tests.

**Physics intuition:** The antenna’s radiation pattern tells you how it weights the sky. A strong beam in direction 
`n̂` means power from that direction contributes more to the measured spectrum. Rotating the beam (turntable) changes the direction of this weighting, which changes the time‑dependent signal as the sky rotates.

**Math in the code:**

- Beams store complex E‑field components `Eθ(ν,θ,φ)` and `Eφ(ν,θ,φ)`.
- Power (auto) is
  
  `P(ν,θ,φ) = |Eθ|^2 + |Eφ|^2`  

- Cross‑power between two beams `i` and `j` is

  `P_ij(ν,θ,φ) = Eθ_i * Eθ_j* + Eφ_i * Eφ_j*`

- Rotation in `Beam.rotate(deg)` is implemented as a **shift in φ** for the beam maps (assuming full 0–360° sampling).

---

## 2) Beam smoothing (`beam_config.beam_smooth`)

**Where:** `simulation/driver/run_sim.py` passes `beam_smooth` into `lusee/Simulation.py` → `Simulator.prepare_beams`.

**What it changes:** Applies a **Gaussian filter** to the beam’s cross‑power maps, effectively smoothing angular structure.

**Physics intuition:** This mimics limited angular resolution or removes small‑scale numerical artifacts in the simulated beam. It is not a physical “instrument averaging” knob; it is a model simplification.

**Math in the code:**

The code calls `scipy.ndimage.gaussian_filter` on the beam power map (in θ–φ grid). In effect,

`P_smooth = Gσ * P`,

where `Gσ` is a Gaussian kernel of standard deviation `σ = beam_smooth`, and `*` is convolution over the angular grid.

---

## 3) Beam taper and ground temperature (`taper`, `Tground`)

**Where:** `lusee/Simulation.py` (`Simulator.__init__` and `prepare_beams`). `Tground` is set in YAML under `observation.Tground`.

**What it changes:**

- **Taper** reduces beam response near the horizon to suppress discontinuities and ground pickup.
- **Tground** adds a thermal emission term from the lunar surface (treated as a uniform temperature).

**Physics intuition:** Real antennas near the ground receive radiation from the surface as well as the sky. The code models ground pickup as a simple additive thermal term weighted by how much of the beam sees the ground.

**Math in the code:**

- The taper is applied in θ as
  
  `tapr(θ) = 1 − ( arctan((θ − π/2)/taper)/π + 0.5 )^2`

  This transitions the beam down near the horizon (`θ = π/2`).

- The ground term added to each spectrum is
  
  `T_ground_effect = Tground * groundPower`,

  where `groundPower` is computed from the **beam monopole** (ℓ=0 mode). For auto‑correlations it uses the real beam map; for cross‑correlations it uses both real and imaginary parts.

---

## 4) Cross‑coupling between antennas (`beam_config.couplings`)

**Where:** `lusee/BeamCouplings.py` and YAML `beam_config.couplings`.

**What it changes:** An additive correction for **cross‑power** spectra between antenna pairs, derived from 2‑port EM simulations.

**Physics intuition:** Closely spaced antennas are electromagnetically coupled. This means cross‑correlations are not only sky‑signal integrals; there is also a coupling term. The code models that with a frequency‑dependent correction from 2‑port beam simulations.

**Math in the code:**

- For a coupling entry, the code loads a 2‑port beam and computes a frequency‑dependent cross‑power:
  
  `cross_power(ν) = −sign + sign * sqrt(gain_conv_i * gain_conv_j) / (2 * gain_conv_2port)`

- During simulation, this modifies the “ground power” term for cross‑products (real and imaginary parts).

This is a simplified coupling model; it does **not** change the sky rotation or the beam harmonic transforms themselves.

---

## 5) Frequency grid (spectral binning)

**Where:** `simulation/driver/run_sim.py`, YAML `observation.freq`.

**What it changes:** The list of frequencies at which the simulation evaluates the sky + beam integral. This is the **simulated channelization**; it is not a model of hardware FFT or accumulation, but it represents the channel centers you want to output.

**Math in the code:**

The driver builds

`freq = np.arange(start, end, step)`

and everything downstream uses this array. The number of bins is `len(freq)`. For example, to mimic 2048 bins at 25 kHz across 0–51.2 MHz, you would use:

- `start = 0.0`, `end = 51.2`, `step = 0.025` (MHz)

so that `np.arange` returns 2048 elements.

---

## 6) Front‑end throughput and noise (`Throughput`)

**Where:** `lusee/Throughput.py` (used by `lusee/Data.py` to convert simulated temperatures into voltage units and noise estimates).

**What it changes:** The electronic response and noise of the front end. This does **not** change the sky integrals themselves; it changes how you interpret outputs (e.g., converting to V²/Hz).

**Parameters in code:**

- `noise_e`: amplifier noise (nV/√Hz). Used to set noise floor.
- `Cfront`: front‑end input capacitance (pF).
- `R4`: front‑end resistance (Ohm).
- `gain_set`: selects which precomputed gain curve to use (`'L'`, `'M'`, `'H'`).

**Physics intuition:** The antenna has impedance `Z_ant(ν)`. The front end has an effective impedance from `Cfront` and `R4`. The mismatch between them determines how much sky signal power is delivered into the receiver and how it converts into voltage.

**Math in the code:**

- Front‑end impedance magnitude is computed from

  `Z_rec(ν) = 1 / ( i ω Cfront + 1/R4 )`

- A coupling factor (called `Gamma_VD`) is

  `Γ(ν) = |Z_rec| / |Z_ant + Z_rec|`

- Temperature to voltage‑squared conversion:

  `T2Vsq(ν) = 4 k_B Re(Z_ant) * Γ(ν)^2`

This factor is used by `Data` to convert simulated spectra into voltage units. The actual gain curves are loaded from SPICE simulation files under `LUSEE_DRIVE_DIR`.

---

## What is *not* modeled here (for clarity)

- The **two‑stage spectrometer accumulation** (e.g., `2**13–2**14` and `2**4` averages) is **not present** in this repository.
- There is no explicit **time‑domain FFT/channelizer** or accumulation simulator.
- The code outputs perfect, noiseless sky + ground integrals unless you add noise externally (e.g., in analysis notebooks).

If you want to introduce hardware‑style averaging, you would add it *after* `Simulator.simulate` (for example, by averaging consecutive time samples and/or adding thermal noise consistent with `Throughput`).
