# Four-port physical and scientific review

Date: 2026-07-30

Status: implementation update, item 1 of the recommended plan complete

Reviewed revisions:

- luseepy `codex/four-port-polarization-refactor`, implementation baseline
  `9589d7b9f5f463db1c2778a6f6a9c1613d60b6ec`
- companion Croissant `codex/full-stokes-pair-response-topo` at
  `daf1545bc57cb7fdf3d28cc468789a139c6eeb68`
- TeX source of truth:
  [`04_AsymmetricTwoAntenna.tex`](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex)

This review covers only the new four-port `InstrumentResponse` path. In
particular, constructing
[`TopoJaxSimulator`](../lusee/TopoJaxSimulator.py#L20) with an
`InstrumentResponse` dispatches to
[`FullStokesTopoJaxSimulator`](../lusee/FullStokesSimulator.py#L666).
The legacy beam path is out of scope.

No production four-port antenna response is currently available. The beam
files under `LUSEE_DRIVE_DIR` belong to the legacy path and were not used as
surrogates. Consequently, this review can establish whether the equations
and software contracts are correct, but it cannot certify absolute
normalization, angular convergence, conditioning, or material-loss
decomposition for a real response artifact.

## Verdict

The central full-Stokes and coupled-network calculation in the new
`TopoJaxSimulator` path is physically sound. The coherency convention,
pair-response kernels, effective-length scaling, visible-sky resistance,
four-port loading matrix, and covariance contraction agree with the TeX
derivation.

The earlier ENU parity blocker is fixed in the reviewed Croissant revision.
The earlier claim that peak-labelled inputs universally halve sky covariance
was too broad. The real converter problem is that one amplitude flag cannot
describe the field units and the separate field/source phasor conventions
required by the TeX. Depending on preprocessing, the current converter can be
correct, can be wrong by a factor of two in covariance, or can be wrong by
six orders of magnitude from untreated HFSS mV units.

The response and covariance schema now represents antenna metal loss
separately and applies its own antenna temperature. This removes the
previous software error for unequal lunar and antenna temperatures.
Producing a realistic lossy response still requires new lossy and PEC
solver products, which are not currently available.

One further change remains required before producing a scientifically
validated, realistic four-port response:

1. make the response-conversion unit and normalization contract explicit.

The map-making radiometric-noise calculation also has a definite factor and
complex-covariance error. It does not alter simulator voltage covariances,
but it does make map-making uncertainties and weights wrong.

Finite-`lmax` positivity and the remaining validation gaps are important
release-readiness risks, not demonstrated operational bugs: there is no real
four-port response on which to quantify them yet.

## Wrong-result findings

### 1. Resolved in software: antenna metal loss is separate from lunar loss

The TeX explicitly decomposes the physical covariance into sky, lunar, and
antenna-loss terms
([lines 425--430](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L425))
and requires

```text
Herm(ZA) = Rsky + Rmoon + Rloss
Rmoon = Herm(ZA) - Rsky - Rloss.
```

It recommends deriving the dense metal-loss matrix from lossy and PEC solver
runs
([lines 578--590](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L578)):

```text
Rloss = Herm(ZA_lossy) - Herm(ZA_PEC)
K_ant = 4 k_B T_ant Rloss.
```

The response schema now stores `Rsky`, `Rmoon`, and `Rloss`, and binds all
three into the content hash. `LOSSMODEL` and `RLOSSSRC` are required
provenance. A PEC response must explicitly declare `LOSSMODEL=PEC` and has a
validated zero `Rloss`; a lossy response must supply the matrix rather than
silently treating a missing matrix as zero.

[`compute_sky_moon_resistance`](../lusee/ResponsePhysics.py) now defines

```text
Rmoon = Herm(ZA) - Rsky - Rloss.
```

The writer and loader validate Hermiticity, the three-term resistance
identity, and PSD of `Rmoon` and `Rloss`. The interpolated target-frequency
matrices are checked again before simulation.
[`assemble_open_covariance`](../lusee/Covariance.py) now assembles

```text
K_open =
    K_sky
    + 4 k_B T_moon Rmoon
    + 4 k_B T_ant Rloss.
```

`T_ant` is propagated by the driver, full-Stokes simulator, and calibrator
and, when specified, recorded with `T_moon` in output provenance. An omitted
temperature for a PEC response remains unspecified in the output rather than
being misreported as a physical 0 K; zero is used only as the internal
multiplier of its zero loss matrix. Output also carries `LOSSMODEL` and
`RLOSSSRC`, and loading it exposes the loss provenance, temperatures, and
`Rloss`. A nonzero lossy response without an explicit antenna temperature is
rejected. Unequal-temperature, off-grid tests now distinguish Moon and metal
heating through both the simulator and calibrator, while lossy
equal-temperature blackbody closure verifies the complete resistance budget.
MapMaker explicitly sets both thermal multipliers to zero because its
forward operator is intentionally sky-linear and offset-free.

This resolves the identified programming bug. It does not manufacture the
missing physical inputs: a production `Rloss` and antenna-temperature model
still require new solver products or instrument-expert input. The current
schema revision was updated in place because no production version-3
four-port artifact exists; any experimental pre-change version-3 file is
intentionally incompatible with the new required provenance.

### 2. Blocker for response production: converter normalization is underspecified

The TeX gives two separate HFSS requirements:

- exported `rE_peak` is in mV and must be multiplied by
  `1e-3/sqrt(2)` to obtain RMS volts
  ([line 217](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L217));
- a 1 V peak incident wave at a matched port corresponds to a 2 V peak,
  or `sqrt(2)` V RMS, Thevenin source
  ([lines 249--253](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L249)).

The converter has no raw-field unit parameter.
[`convert_fields_to_effective_length`](../beam_conversion/common.py#L147)
accepts one `amplitude_convention`, divides every peak-labelled field by
`sqrt(2)`, and assumes the resulting `rE` values are in volts.
[`convert_receive_csvs`](../beam_conversion/receive_csv.py#L129) uses that
same flag after reconstructing currents from a separately supplied
`Vsource`, whose amplitude convention is neither declared nor checked.

There is a second normalization hole for direct bare input. With
`input_kind="bare"` and `field_kind="rE"`, the converter skips current
reconstruction and applies the reciprocity factor immediately. This is
correct only when each input pattern is already normalized to the open-port
1 A basis, so its actual quantity is `rE/I` in V/A. The TeX can set
`I_norm = 1 A` after embedded-pattern unmixing because that matrix operation
forces the recovered current basis to the identity. An arbitrary direct
bare solver export does not inherit that guarantee. The API accepts no
per-port/frequency complex normalization current and records no explicit
already-per-ampere assertion.

There is no universal factor-of-two diagnosis:

- Raw HFSS mV numbers passed directly to the converter are interpreted as
  volts. With otherwise correct RMS normalization, effective length is too
  large by `1000` and covariance by `10^6`.
- Peak fields unembedded with an RMS `Vsource` do need the field
  `1/sqrt(2)` conversion; the current operation is correct in that specific
  mixed representation.
- If both field and `Vsource` describe the same peak representation, their
  ratio is already invariant. The extra field-only division makes effective
  length too small by `1/sqrt(2)` and covariance too small by `1/2`.
- A bare `rE/I` transmit ratio is likewise peak/RMS invariant when field and
  current use the same convention. Scaling only the field changes the
  physical ratio.
- A genuine effective length is already a voltage/field ratio. When its
  numerator and denominator use the same phasor convention, labelling it
  peak must not change its numerical value. The current function changes it.

The gross untreated-mV case should normally make the derived `Rmoon`
non-PSD and fail validated response writing. That safety check does not
repair the unit contract, and it does not catch an underscaled response whose
missing `Rsky` is absorbed into the Moon complement.

The existing
[`test_peak_and_re_conversion_are_applied_exactly_once`](../tests/test_response_conversion.py#L75)
tests the current scalar division, not representation invariance of the
complete embedded-field/current calculation.

Required software changes:

1. Represent the physical input quantity and units separately: embedded
   `rE` in mV or V, bare `rE/I` in V/A, or effective length in m.
2. Represent field amplitude convention and excitation normalization
   separately: `Vsource` for embedded input, or a per-port/frequency complex
   normalization current for a direct bare export. Permit an already-per-
   ampere input only with explicit provenance.
3. Convert raw embedded fields and source voltages to SI RMS before
   reconstructing currents and unembedding.
4. Convert a direct bare field and its normalization current consistently
   before forming `rE/I`. Do not apply a field-only peak/RMS scale to
   `rE/I` or effective length.
5. Define the physical meaning of `input_kind="embedded"` combined with
   `field_kind="effective-length"`, or reject that combination.
6. Persist the original and canonicalized conventions in response
   provenance.
7. Add representation-invariance tests and an explicit HFSS regression for
   `rE_peak` in mV with `sqrt(2)` V RMS Thevenin excitation.

These transformations follow directly from the TeX and can be implemented
without additional expert judgment. Certifying a real artifact still
requires one raw solver export and an independent accepted-power or
plane-wave terminal-voltage check.

### 3. High for map-making: radiometric cross-product covariance is wrong

[`compute_radiometric_noise`](../lusee/MapMaker.py#L319) assigns the same
variance to the real and imaginary parts of a cross-product:

```text
(Caa Cbb + |Cab|^2) / (4 delta_f delta_t).
```

For a proper complex Gaussian voltage vector and
`N = delta_f delta_t` independent complex samples, the component statistics
are

```text
Var(Re C_hat_ab) =
    [Caa Cbb + Re(Cab^2)] / (2 N)

Var(Im C_hat_ab) =
    [Caa Cbb - Re(Cab^2)] / (2 N)

Cov(Re C_hat_ab, Im C_hat_ab) =
    Im(Cab^2) / (2 N).
```

For nonzero `Cab`, the equal-component estimate can under- or overestimate a
particular real or imaginary variance, depending on `Re(Cab^2)`. In the
important `Cab = 0` limit, both implemented variances are low by a factor of
two, so both standard deviations are low by `sqrt(2)`. Real and imaginary
channels are not generally equal or independent. Products that share ports
also have nonzero sampling covariance, while
[`solve`](../lusee/MapMaker.py#L178) currently accepts diagonal weights.

At minimum, use the correct separate real/imaginary variances and label the
remaining diagonal-noise treatment as an approximation. A later extension
can apply the full packed real covariance per time/frequency sample.

## Resolved or corrected conclusions from the earlier review

### ENU parity is fixed

Croissant commit `1b0902e`, included in the reviewed `daf1545`, converts
Astropy/lunarsky's left-handed North-East-Up local chart to the package's
right-handed East-North-Up convention in
[`get_rot_mat`](../../croissant/src/croissant/rotations.py#L95). It swaps the
appropriate source columns or target rows before an Euler/Wigner rotation.

At the LuSEE-Night site and a representative epoch, an independent direct
check found:

```text
det(rotation)                  1.0
orthogonality residual        about 1e-15
East/North/Up mapping errors   below 1e-15
```

The earlier determinant `-1` and polarized-V sign reversal are therefore not
current findings. The reviewed `FullStokesTopoJaxSimulator` now passes proper
SO(3) matrices into the per-time Wigner rotation.

### Whole-instrument rotation is not currently sign-inconsistent

[`InstrumentResponse.rotate`](../lusee/InstrumentResponse.py#L707) rolls
maps by `-bins`, moving a directional response toward decreasing ENU phi.
On the reviewed Croissant branch, `beam_rot` applies the same
`exp(+i m alpha)` harmonic phase and documents positive rotation using
astronomical azimuth: North toward East. For example, positive 90 degrees
moves an initially East-pointing axis toward South.

The luseepy and Croissant operations therefore agree. The remaining issue is
documentation: `PHIDEF=right-handed-about-+z` describes the spatial
coordinate, while the sign of `rotation_deg` is not stated in the luseepy
API or example config. Document the positive astronomical-azimuth convention
and add one directional map/harmonic regression. Reversing the operation
would be a new convention change, not a bug fix justified by the current
code.

## Resolution and validation risks

The following items should be addressed before scientific release, but the
available evidence does not show that they corrupt realistic current
simulations.

### Finite `lmax` does not guarantee a PSD covariance

The sky and response are separately truncated in harmonic space. Their
finite-bandlimit contraction is Hermitianized by
[`load_covariance`](../lusee/Covariance.py#L77), but separate truncation no
longer preserves the exact positive angular-integral representation.

A deliberately under-resolved diagnostic used a physical unpolarized sky
with one sharp hot pixel at `lmax=2` and obtained

```text
lambda_min / lambda_max = -0.1916.
```

This demonstrates that the API cannot promise PSD for arbitrary maps and
bandlimits. It is expected spectral-truncation behavior, not by itself a
programming error or evidence that an operational `lmax` is inadequate.
Without a real four-port response, the realistic magnitude is unknown.

Add covariance-eigenvalue and `lmax`-convergence diagnostics, and compare
representative horizon structure against direct angular quadrature. Do not
silently clip negative eigenvalues: clipping would hide inadequate angular
resolution.

### Response validation has no independent absolute oracle

[`validate_response_matrices`](../lusee/ResponsePhysics.py) now checks
finiteness, `Rsky`/`Rmoon`/`Rloss` Hermiticity, the three-term complement,
`Rmoon` and `Rloss` PSD, the PEC-zero-loss constraint, and agreement between
stored and field-derived `Rsky`. It does not yet explicitly gate or report:

- reciprocity, `ZA approximately ZA.T`, for the reciprocal antenna model;
- PSD of field-derived `Rsky`;
- passivity of `Herm(ZA)`;
- an acceptable condition number for embedded-current unmixing.

The converter records `MAX_ICOND`, but it has no justified acceptance
threshold. The synthetic response closes its resistance budget using the
same harmonic operator under test, so it is useful for internal consistency
but not an absolute normalization oracle.

Add mathematical reciprocity/passivity/PSD checks and conditioning
diagnostics now. Add a grid-converged Hertzian-dipole analytic oracle. A hard
conditioning threshold and a solver accepted-power oracle should wait for a
representative export and numerical-precision requirements.

### Coordinate metadata can still disagree

Croissant `PolarizedSky` stores both `coord` and `frame` and still permits
contradictory values. Its own coordinate transport consults `coord`, while
[`FullStokesSimulatorBase.simulate`](../lusee/FullStokesSimulator.py#L391)
prefers `frame`. Correctly constructed objects use matching values, so this
is interface hardening rather than a realistic wrong-result finding under
the stated review scope.

Make one field authoritative or reject disagreement at construction and at
the luseepy boundary.

### Physical-input and projection diagnostics are incomplete

Physical Stokes maps satisfy

```text
I >= 0
I^2 >= Q^2 + U^2 + V^2.
```

Signed components and map-making perturbations can be intentional, so these
conditions should be an optional physical-simulation diagnostic rather than
an unconditional rejection.

Likewise, `load_covariance` projects onto the Hermitian subspace without
reporting the pre-projection anti-Hermitian residual. Record that residual so
convention errors can be distinguished from roundoff.

## Scientifically sound components

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

   has the correct sign for the documented positive-V convention.

2. Scaling effective-length products by `eta0/lambda^2`, followed by

   ```text
   K_sky = k_B integral W S dOmega,
   ```

   is consistent with RMS one-sided voltage PSDs and Rayleigh-Jeans
   brightness temperature.

3. The visible-sky resistance

   ```text
   Rsky = eta0/(4 lambda^2) integral H H^dagger dOmega
   ```

   and each fluctuation-dissipation term

   ```text
   K_alpha = 4 k_B T_alpha R_alpha
   ```

   have the correct factor of four.

4. The noncommuting loading equation

   ```text
   M = ZL (ZA + ZL)^-1
   C_v = M K_open M^dagger
   ```

   is correct. The implementation's right-side solve avoids an unnecessary
   explicit inverse.

5. The effective-length bilinear form, pair ordering `a <= b`, visibility
   definition `<v_a v_b*>`, and complex conjugations are mutually
   consistent.

6. The fixed-frame full-Stokes harmonic dual agrees with direct MWSS
   quadrature in the synthetic checks. The IAU/COSMO conversion and
   algebraic positive-V fixture are internally consistent.

7. The point-source covariance normalization and native-frequency direct
   loaded-response comparison are correct.

8. The Rayleigh-Jeans approximation is entirely adequate for the
   LuSEE-Night band and expected sky, lunar, and antenna temperatures.

## Verification record and limitations

The luseepy constructors now match the reviewed Croissant API. The
topocentric synthetic fixture also uses Croissant's current canonical
`coord="topo"` spelling instead of the removed `mcmf` alias.

The focused new-path suite, including the explicit-loss tests, completed
without a compatibility shim with

```text
57 passed, 2 warnings
```

The warnings are Astropy's pre-existing assumption that two numerical
`TimeDelta` inputs are days. The independent ENU basis check passed at
machine precision.

These tests use synthetic responses. They do not replace validation against
a new lossy/PEC antenna simulation, because no such artifact is available.

## Recommended implementation order

The following work is mechanically and mathematically specified without
further expert input:

1. **Complete:** extend the response and covariance model with explicit
   `Rloss` and `T_ant`, including unequal-temperature closure tests.
2. Replace the converter's single amplitude flag with explicit field-unit,
   field-amplitude, and source-amplitude conventions; add HFSS and
   representation-invariance tests.
3. Correct real/imaginary radiometric variances and document the diagonal
   map-making approximation.
4. Add reciprocity, passivity, matrix-PSD, anti-Hermitian-residual, and
   condition-number diagnostics.
5. Add the grid-converged Hertzian-dipole oracle and direct-quadrature
   `lmax` convergence tests.
6. Reject `coord`/`frame` disagreement and document/test the positive
   astronomical-azimuth turntable convention.

Items that still require expert input or new data are the actual
lossy-versus-PEC `Rloss` artifact, the antenna temperature model, acceptance
thresholds for solver-current conditioning and operational `lmax`, and an
independent solver accepted-power or plane-wave normalization reference.
