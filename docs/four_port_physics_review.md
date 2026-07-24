# Four-port physical and scientific review

Date: 2026-07-23

Reviewed revisions:

- luseepy `codex/four-port-polarization-refactor` at
  `4ce6f6e3da51b89f3f28b4aff81dc46094397a8c`
- companion Croissant `codex/full-stokes-pair-response` at
  `4677c7e7d7fd52183bd5c2b70a24ce391cd7cef3`

The runtime checks used the local CUDA GPU with JAX x64 enabled and placed
`../croissant/src` first on `PYTHONPATH`. They did not use Lawrencium or the
public PyPI Croissant package. An independent `@review` pass checked the
normalization, radiometer statistics, covariance physicality, reciprocity,
and metadata conclusions below.

## Verdict

The central electrodynamics and coupled-network formulation are sound, but
the reviewed revisions are not scientifically release-ready. Two issues are
release blockers: local celestial rotations contain an unhandled reflection,
and peak/RMS response conversion likely reduces sky power by a factor of two.
The map-making noise model, finite-bandlimit covariance physicality, and
response-validation gaps also need resolution before scientific use.

In particular, the current revisions should not be used for:

- polarized celestial predictions;
- absolute calibration from peak-labelled solver exports; or
- quantitative map-making uncertainties.

## Findings

### 1. Blocker: `LunarTopo` rotations include an unhandled reflection

Croissant's
[`get_rot_mat`](../../croissant/src/croissant/rotations.py#L114)
returns Astropy `LunarTopo` transformations unchanged. The result is passed
to
[`rotmat_to_eulerZYZ`](../../croissant/src/croissant/rotations.py#L232)
and then to an SO(3) Wigner-D rotation.

The physical problem is a local-axis mismatch:

- Astropy `LunarTopo` represents the local chart as North-East-Up;
- luseepy/Croissant beam maps use East-North-Up, with positive phi measured
  counter-clockwise from East toward North.

For the LuSEE-Night site and epoch used in the regression, the unmodified
matrices had

```text
det(galactic -> LunarTopo) = -1
det(LunarTopo -> MEPA)     = -1
```

An improper matrix cannot be represented by the existing Euler/Wigner
rotation. Both the
[`FullStokesCroSimulator`](../lusee/FullStokesSimulator.py#L596)
and
[`FullStokesTopoJaxSimulator`](../lusee/FullStokesSimulator.py#L666)
therefore mishandle celestial polarization, through different improper
transform paths.

A local-GPU regression used a physical mixed-IQUV sky, `lmax=2`, one target
frequency, 29 daily samples, and `T_moon=0` to isolate the sky contribution.
It found:

```text
Cro versus Topo global relative difference       2.5576e-2
maximum per-time relative difference             3.7819e-2
isolated V relative difference                   approximately 2
```

The isolated V result is a sign reversal. Isolated Q and U differed at order
unity. The mixed sky itself was physical: Stokes I was about 180 K and
strictly dominated the few-kelvin polarized components.

As a diagnostic only, swapping North and East on the appropriate
`LunarTopo` matrix axis before Euler conversion changed the determinants to
`+1`. Without changing any repository file, that reduced the same
cross-engine discrepancy to

```text
global relative difference                       2.8429e-5
relative difference at the reference epoch       5.8821e-12
maximum per-time difference over 29 days         4.0322e-5
```

This also shows that, after the parity correction, Croissant's constant
sidereal phase approximation is accurate for this low-order full-cycle
case.

The existing
[`test_topo_and_cro_agree_for_celestial_sky_away_from_lunar_pole`](../tests/test_full_stokes_simulator.py#L245)
does not expose the problem. It spans only six hours, includes a constant
210 K lunar term, and uses an absolute tolerance comparable to the
sky-only discrepancy.

Required resolution:

1. Convert every local frame to the package's right-handed ENU convention
   before Euler conversion, swapping the correct row or column depending on
   transform direction.
2. Assert that every matrix sent to a Wigner rotation is orthogonal with
   determinant `+1`.
3. Add independent celestial I, Q, U, and V tests at several epochs, including
   a transported polarized point source and direct right-handed-coordinate
   references.
4. Apply the same audit to Earth `AltAz`, which has an analogous local-chart
   convention.

### 2. Blocker: peak/RMS conversion likely halves sky covariance

[`convert_fields_to_effective_length`](../beam_conversion/common.py#L147)
divides every `amplitude_convention="peak"` input by `sqrt(2)`, including an
input already identified as an effective length.

An effective length is a transfer ratio,

```text
H = V_open / E_incident,
```

and is invariant when numerator and denominator are represented consistently
as peak or RMS phasors. The same applies to a reciprocal transmit-field ratio
`E/I`.

For embedded exports, fields are first divided by simulated excitation
currents in
[`embedded_fields_to_bare`](../beam_conversion/common.py#L98), called from
[`convert_receive_csvs`](../beam_conversion/receive_csv.py#L172). If field
and source/current use the same solver convention, the peak/RMS factor has
already cancelled before the additional division.

A representation-invariance diagnostic constructed RMS and peak forms of the
same embedded physical response. The reconstructed bare `E/I` values agreed
to `6.7e-16`, but the converter produced

```text
H_peak / H_rms                    0.70710678
pair-response power ratio         0.5
```

Consequently, a peak-labelled response gives one half of the correct
`Rsky`, calibrator covariance, and anisotropic-sky covariance unless the
solver uses an unusual mixed convention. The current single `AMP_CONV`
metadata field cannot describe separate field and source conventions.

[`test_peak_and_re_conversion_are_applied_exactly_once`](../tests/test_response_conversion.py#L75)
codifies the extra division rather than testing physical
representation-invariance.

The blackbody enclosure test cannot detect this absolute-scale error:
[`compute_sky_moon_resistance`](../lusee/ResponsePhysics.py#L258) defines

```text
Rmoon = HermitianPart(ZA) - Rsky.
```

A response with `Rsky` too small simply gets a correspondingly larger
`Rmoon`, so equal-temperature blackbody closure still passes.

Required resolution:

1. Define the exact amplitude convention of each supported solver export.
2. Remove peak/RMS conversion from already-formed effective lengths and from
   consistently normalized `E/I` ratios.
3. If field and excitation can use different conventions, record and convert
   them separately.
4. Add an RMS-versus-peak representation-invariance test.
5. Validate absolute scale against an independent solver accepted/radiated
   power or plane-wave terminal-voltage result.

### 3. High: map-making radiometric noise is underestimated

[`compute_radiometric_noise`](../lusee/MapMaker.py#L319) cites the
radiometer equation

```text
variance = (Cii Cjj + |Cij|^2) / (2 delta_f delta_t)
```

and then assigns half of that again to each real and imaginary channel,
using a denominator of `4 delta_f delta_t`. The cited paper constructs its
data vector from 16 real observables and presents Equation 9 as the variance
of each visibility measurement:
[Camacho et al. 2026](https://arxiv.org/html/2508.16773v3).

More generally, for a proper complex Gaussian voltage vector and
`N = delta_f delta_t` independent complex samples,

```text
Var(Re C_hat_ij) =
    [Cii Cjj + Re(Cij^2)] / (2 N)

Var(Im C_hat_ij) =
    [Cii Cjj - Re(Cij^2)] / (2 N)

Cov(Re C_hat_ij, Im C_hat_ij) =
    Im(Cij^2) / (2 N).
```

Thus the real and imaginary components are not generally equal or
independent. Even for `Cij=0`, the current code is low by a factor of two in
variance, or `sqrt(2)` in standard deviation. This overweights cross-products
in map-making.

Products sharing antenna ports also have nonzero sampling covariance. The
current
[`solve`](../lusee/MapMaker.py#L178) accepts only diagonal inverse-variance
weights, which is a potentially poor approximation for a strongly coupled
four-port instrument.

Required resolution:

1. At minimum, implement the correct separate real/imaginary component
   variances.
2. Clearly distinguish the paper's diagonal-noise approximation from the
   exact complex-Wishart result.
3. Preferably construct the full packed 16-by-16 real covariance per
   time/frequency sample and allow `solve` to apply its inverse or a justified
   block approximation.

### 4. High: finite `lmax` can produce a non-PSD covariance

The sky and response are separately projected to finite-bandlimit harmonic
representations in Croissant
[`polarization.py`](../../croissant/src/croissant/polarization.py#L221).
The resulting contraction is Hermitianized by
[`load_covariance`](../lusee/Covariance.py#L77), but it is not checked for
positive semidefiniteness.

Separate truncation removes the exact positive-integral representation. A
strictly physical, unpolarized MWSS sky was tested with:

```text
grid                         5 x 8
lmax                         2
Stokes I                     1 K everywhere
one Stokes-I pixel           101 K
Stokes Q, U, V               0
T_moon                       0
```

Its open covariance eigenvalues were

```text
[-9.3338e-26, 1.6178e-30, 8.8962e-26, 4.8709e-25] V^2/Hz,
```

so `lambda_min/lambda_max = -0.1916`. This is a predictable consequence of
using an inadequate bandlimit for a sharp sky, but it violates the public
meaning of the result as a covariance matrix and the plan's PSD release
gate.

Required resolution:

- document the necessary bandlimited-physical input condition;
- provide an `lmax` convergence and runtime PSD diagnostic for simulation
  use;
- test realistic anisotropic skies and near-horizon structure against direct
  angular quadrature at operational `lmax`; and
- provide a positivity-preserving integration path when that guarantee is
  required.

Silently clipping negative eigenvalues would conceal inadequate angular
resolution and is not an acceptable default physical fix.

### 5. High: response validation lacks an independent absolute oracle

[`validate_response_matrices`](../lusee/ResponsePhysics.py#L272) checks
finite values, Hermiticity of `Rsky`/`Rmoon`, the Moon-complement identity,
`Rmoon` PSD, and agreement between stored and field-derived `Rsky`. It does
not explicitly check:

- reciprocity, `ZA approximately ZA.T`, when transmit fields are converted to
  receive effective lengths;
- PSD of field-derived `Rsky`;
- passivity of the dissipative part of `ZA`; or
- a maximum acceptable condition number for embedded-current unmixing.

The converter records `MAX_ICOND` in
[`receive_csv.py`](../beam_conversion/receive_csv.py#L201), but an arbitrarily
ill-conditioned unembedding can still be marked validated.

The
[`synthetic_four_port_response`](../lusee/SyntheticResponse.py#L15)
computes an analytic short-dipole radiation resistance, then replaces its
dissipative impedance with `2 * Rsky` from the same harmonic operator. The
fixture is useful for internal closure, but it cannot detect a shared
normalization error. At its deliberately coarse default grid, the constructed
diagonal resistance was about 32 percent above the analytic short-dipole
value, yet the fixture remained `VALIDATED=True`.

Production release validation should include:

1. a converged Hertzian-dipole effective-length/radiation-resistance oracle;
2. a raw solver plane-wave or accepted-power terminal result;
3. comparison of integrated radiated/absorbed power with the independent
   solver impedance decomposition;
4. reciprocity and passivity gates; and
5. a conditioning threshold justified by retained numerical precision.

### 6. Medium: coordinate metadata can be contradictory

Croissant
[`PolarizedSky`](../../croissant/src/croissant/polarization.py#L323)
stores both `coord` and `frame`. Its constructor allows, for example,
`coord="mcmf"` with `frame="topo"`.

[`compute_alm_eq`](../../croissant/src/croissant/polarization.py#L412)
chooses coordinate transport from `coord`, while luseepy
[`simulate`](../lusee/FullStokesSimulator.py#L391) chooses it from `frame`.
The same object can therefore represent different physical skies through the
two public paths. One attribute should be authoritative, `topo` should be
represented explicitly where supported, and inconsistent pairs should be
rejected.

### 7. Medium: whole-instrument rotation has the opposite azimuth sign

[`InstrumentResponse.rotate`](../lusee/InstrumentResponse.py#L707) rolls its
maps by `-bins`. A direct harmonic diagnostic found that
`rotate(+alpha)` produces

```text
exp(+i m alpha).
```

Croissant's documented positive
[`beam_az_rot`](../../croissant/src/croissant/beam.py#L51) applies

```text
exp(-i m alpha),
```

corresponding to an active counter-clockwise rotation from local East toward
North. Given the response metadata's right-handed positive-phi convention,
these operations have opposite signs.

The luseepy operation must either be documented as a coordinate rotation with
the opposite sign, or changed to the same active whole-instrument convention.
An analytic directional beam/source transit test should freeze the turntable
semantics.

### 8. Medium: physical-input and projection diagnostics are incomplete

Croissant verifies that IQUV pixel arrays are real, but does not diagnose the
coherency condition

```text
I >= 0
I^2 >= Q^2 + U^2 + V^2.
```

Signed sky components and map-making perturbations can be legitimate, so a
hard rejection is not always appropriate. A physical-simulation validation
mode should nevertheless report violations.

Similarly,
[`load_covariance`](../lusee/Covariance.py#L77) unconditionally projects its
result onto the Hermitian subspace without first reporting the
anti-Hermitian residual. That can hide convention or numerical errors.

## Scientifically sound components

The architecture is not a conceptual dead end. The following pieces are
physically consistent:

1. The pair-response definitions in
   [`pair_stokes_maps`](../lusee/InstrumentResponse.py#L497) agree with the
   declared coherency matrix:

   ```text
   [[I + Q, U - i V],
    [U + i V, I - Q]].
   ```

   In particular,

   ```text
   B_V = i (H_a_phi H_b_theta* - H_a_theta H_b_phi*)
   ```

   has the correct sign for the documented positive-V fixture.

2. Scaling effective-length products by `eta0/lambda^2`, followed by

   ```text
   K_sky = k_B integral W S dOmega,
   ```

   is consistent with RMS one-sided voltage PSDs and the Rayleigh-Jeans
   brightness convention used here.

3. The visible-sky resistance

   ```text
   Rsky = eta0/(4 lambda^2) integral H H^dagger dOmega
   ```

   and lunar fluctuation-dissipation term

   ```text
   K_moon = 4 k_B T_moon Rmoon
   ```

   have the correct factor of four.

4. The noncommuting four-port loading equation

   ```text
   M = ZL (ZA + ZL)^-1
   ```

   and

   ```text
   C_v = M K M^dagger
   ```

   are correct. The implementation's right-side solve is preferable to an
   explicit inverse on numerical-stability grounds.

5. The fixed-frame Croissant full-Stokes harmonic dual agrees with direct
   MWSS quadrature. Its IAU/COSMO conversion and algebraic positive-V fixture
   are internally consistent.

6. The point-source covariance normalization and native-frequency direct
   loaded-response comparison are correct.

7. The Rayleigh-Jeans approximation is entirely adequate for LuSEE-Night's
   1--50 MHz band and expected sky/lunar temperatures.

## Scope assumptions

The Moon-complement construction

```text
Rmoon = HermitianPart(ZA) - Rsky
```

is physically valid if:

- the effective lengths and `ZA` refer to the same passive reciprocal
  electromagnetic model and terminal reference planes;
- the visible-sky response already includes lunar reflection/scattering; and
- every complementary dissipative channel is in equilibrium at the single
  supplied `T_moon`.

The synthesized plan explicitly assumes a perfectly conducting antenna and
puts receiver additive noise outside the forward simulator. Under those
assumptions, the construction is coherent. If a production `ZA` includes
antenna, cable, lander, or other losses at distinct temperatures, the thermal
term must instead be decomposed as

```text
sum_alpha 4 k_B T_alpha R_alpha.
```

The current result is therefore the externally driven sky/Moon contribution
at the JFET input. It is not the complete detector or system-noise covariance,
because receiver/load additive noise and post-JFET gain are outside scope.

## Recommended release order

1. Correct and independently test the ENU/local-frame parity.
2. Resolve the peak/RMS response contract and establish absolute solver
   normalization.
3. Correct the radiometric-noise statistics and decide whether map-making
   supports full or explicitly approximate noise covariance.
4. Add covariance PSD/`lmax` adequacy diagnostics.
5. Add reciprocity, passivity, conditioning, and independent power oracles.
6. Unify coordinate and turntable-rotation conventions.
7. Run operational-`lmax` horizon and polarized-sky convergence tests against
   direct angular quadrature.
