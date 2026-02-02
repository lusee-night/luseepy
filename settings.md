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

---

## End‑to‑end math trace (from output spectra back to inputs)

This section walks *backwards* from the final output arrays to the inputs, with explicit formulas for every symbol that appears in the code path. I keep SkyModels abstract (as requested), and I do **not** expand their internal physics.

### A. Final outputs from `Simulator`

The simulator produces a 3‑D array:

```
result[time_index, product_index, freq_index]
```

Call this output \( D_{t,p,\nu} \). It is written to FITS by `Simulator.write` in `lusee/Simulation.py` as the `data` extension.

For each time \( t \), each product \( p \) (auto or cross), and each frequency index \( \nu \), the code computes:

**Auto‑correlations (i == j):**

\[
D_{t,p,\nu} = \langle B_{ij}(\nu), S_t(\nu) \rangle + T_\mathrm{ground}\,G^\mathrm{(R)}_{ij}(\nu)
\]

**Cross‑correlations (i != j):**

Two products are stored: real and imaginary parts. The simulator appends them in order:

\[
D^\mathrm{(R)}_{t,p,\nu} = \langle B^\mathrm{(R)}_{ij}(\nu), S_t(\nu) \rangle + T_\mathrm{ground}\,G^\mathrm{(R)}_{ij}(\nu)
\]

\[
D^\mathrm{(I)}_{t,p,\nu} = \langle B^\mathrm{(I)}_{ij}(\nu), S_t(\nu) \rangle + T_\mathrm{ground}\,G^\mathrm{(I)}_{ij}(\nu)
\]

Here:
- \( S_t(\nu) \) is the sky model spherical‑harmonic coefficients at time \( t \) and frequency \( \nu \) (produced by `SkyModels`, which I keep abstract).
- \( B_{ij} \) are beam cross‑power coefficients for antenna pair \( (i,j) \) (derived from beam E‑fields).
- \( G^\mathrm{(R/I)}_{ij} \) are ground‑coupling coefficients derived from the beam monopole and optional coupling terms.

The inner product \( \langle \cdot,\cdot \rangle \) is performed by `mean_alm` in `Simulation.py`:

Let \( a_{\ell m} \) be the beam coefficients and \( s_{\ell m} \) the sky coefficients. `mean_alm` implements:

\[
\langle a, s \rangle
= \frac{1}{4\pi}\left(
\sum_{\ell=0}^{\ell_\text{max}} \Re[a_{\ell 0} s_{\ell 0}^*] \;+\;
2 \sum_{\ell=0}^{\ell_\text{max}}\sum_{m=1}^{\ell} \Re[a_{\ell m} s_{\ell m}^*]
\right)
\]

which is the standard spherical‑harmonic inner product assuming real sky maps.

So **the simulator output is in temperature units** (Kelvin‑like), because sky models are constructed as brightness temperature maps.

---

### B. Where \( S_t(\nu) \) comes from (sky, kept abstract)

The sky is provided by a `SkyModels` class, which returns harmonic coefficients:

```
sky = sky_model.get_alm(freq_indices, freq_values)
```

Denote this as \( S(\nu) = \{ s_{\ell m}(\nu) \} \). If the sky model is in the **galactic frame**, it is rotated into the local lunar frame at time \( t \):

1. `Observation.get_l_b_from_alt_az` gives the galactic coordinates of the local zenith (alt=\(\pi/2\)) and a horizon direction (alt=\(0\), az=\(0\)).
2. These define a rotation matrix \( R_t \) from sky coordinates to the local frame.
3. `healpy.Rotator` applies this rotation to the sky alms:

\[
S_t(\nu) = \mathcal{R}(R_t)\,S(\nu)
\]

If the sky model frame is `MCMF`, the code **does not rotate** and uses \( S_t(\nu) = S(\nu) \).

I do not expand `SkyModels` formulas here by request.

---

### C. Where the beam coefficients \( B_{ij} \) come from

Each antenna beam \( i \) is described by complex E‑field components on a \((\theta,\phi)\) grid:

```
Eθ_i(ν,θ,φ), Eφ_i(ν,θ,φ)
```

These are loaded from FITS by `Beam` or created by `BeamGauss`.

For a pair of antennas \( (i,j) \), the **cross‑power** map is:

\[
X_{ij}(\nu,\theta,\phi)
= E_{\theta,i}(\nu,\theta,\phi)\,E_{\theta,j}^*(\nu,\theta,\phi)
+ E_{\phi,i}(\nu,\theta,\phi)\,E_{\phi,j}^*(\nu,\theta,\phi)
\]

Then `Simulator.prepare_beams` applies:

1. **Taper** in \(\theta\):
   \[
   X'_{ij} = X_{ij} \cdot \text{tapr}(\theta)
   \]

2. **Gain convention scaling**:
   \[
   X''_{ij} = X'_{ij} \cdot \sqrt{g_i(\nu)\,g_j(\nu)}
   \]
   where \( g_i(\nu) \) is `gain_conv` from the beam file.

3. Optional **Gaussian smoothing** over \((\theta,\phi)\).

Then the code converts the real and imaginary parts into spherical harmonics:

```
beamreal = get_healpix(lmax, real(X''_ij))
beamimag = get_healpix(lmax, imag(X''_ij))   # only if i != j
```

Thus:

\[
B^\mathrm{(R)}_{ij}(\nu) = \{ a_{\ell m}^\mathrm{(R)}(\nu) \}
\quad,\quad
B^\mathrm{(I)}_{ij}(\nu) = \{ a_{\ell m}^\mathrm{(I)}(\nu) \}
\]

These are exactly the \( a_{\ell m} \) used in `mean_alm`.

---

### D. Where ground‑coupling terms \( G_{ij} \) come from

The code models ground pickup using the beam monopole term. For each frequency:

Let \( a_{00} \) be the \(\ell=0, m=0\) coefficient in `beamreal` (or `beamimag`).

Auto‑correlation case:

\[
G^\mathrm{(R)}_{ii}(\nu) = 1 - \frac{\Re[a_{00}(\nu)]}{\sqrt{4\pi}}
\]

Cross‑correlation case:

\[
G^\mathrm{(R)}_{ij}(\nu) = C_{ij}(\nu) - \frac{\Re[a_{00}(\nu)]}{\sqrt{4\pi}}
\]

where \( C_{ij}(\nu) \) is an optional coupling correction from `BeamCouplings`.

For the imaginary part of a cross‑product:

\[
G^\mathrm{(I)}_{ij}(\nu) = 0 - \frac{\Re[a_{00}^{(I)}(\nu)]}{\sqrt{4\pi}}
\]

Finally, these are multiplied by the scalar `Tground` to give the additive ground term in the output spectra.

---

### E. Mapping to “voltage units” in `Data` (optional post‑processing)

The simulator itself produces **temperature‑like spectra** \( D_{t,p,\nu} \) as above. If you later use `lusee.Data` to request outputs with the suffix `V`, it applies a conversion to voltage‑squared units using `Throughput`.

In `Data.__getitem__`, the conversion factor is:

```
T2V = sqrt(T2Vsq_i * T2Vsq_j)
```

so the returned quantity is:

\[
D^\mathrm{(V)}_{t,p,\nu} = D_{t,p,\nu}\,\sqrt{T2V_i(\nu)\,T2V_j(\nu)}
\]

This is **not** a simulator output; it is a derived quantity.

Where does `T2Vsq` come from? In `Throughput`:

1. Receiver impedance:
   \[
   Z_\mathrm{rec}(\nu) = \frac{1}{i\omega C_\mathrm{front} + 1/R_4}
   \]

2. Antenna impedance \( Z_\mathrm{ant}(\nu) \) is interpolated from beam FITS data.

3. Coupling factor:
   \[
   \Gamma(\nu) = \frac{|Z_\mathrm{rec}|}{|Z_\mathrm{ant} + Z_\mathrm{rec}|}
   \]

4. Temperature → voltage‑squared factor:
   \[
   T2V(\nu) = 4 k_B \Re[Z_\mathrm{ant}(\nu)]\,\Gamma(\nu)^2
   \]

So:

- **\(\Gamma\) is *not* the simulator output.**
- **\(T2V\)** is a conversion factor used only when you ask for voltage units.
- The **native simulator output is in temperature units** as computed in sections A–D.

---

### F. Summary graph of dependencies (textual)

1. **Inputs (instrument):** beam files or Gaussian params; beam rotation; optional beam smoothing; taper; coupling model; `Tground`.
2. **Beams → cross‑power maps** \( X_{ij} \) → harmonic coefficients \( B_{ij} \).
3. **Sky model** → harmonic coefficients \( S(\nu) \) → rotated to time \( S_t(\nu) \).
4. **Spectrum per product:** \( D_{t,p,\nu} = \langle B_{ij}, S_t \rangle + T_\mathrm{ground}\,G_{ij} \).
5. **Optional voltage units:** \( D^\mathrm{(V)} = D \times \sqrt{T2V_i T2V_j} \).

Every symbol above is either defined by a formula in this section or is stated as coming from a class with internal physics (SkyModels). No variables “float” without a source.
