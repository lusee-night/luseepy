# Four-port physical and scientific review

Date: 2026-07-30

Status: implementation update, items 1--3 of the recommended plan complete

Reviewed revisions:

- luseepy `codex/four-port-polarization-refactor`, implementation baseline
  `6849c2594ece58b3aca438dd381eebba30b24645`
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
was too broad. The converter now describes field units and field/source
phasor conventions separately, normalizes raw phasors to SI RMS before
forming a ratio, and treats already-normalized `rE/I` and effective length as
ratio quantities. This removes both the conditional factor-of-two error and
the untreated-HFSS-mV error from the supported conversion paths.

The response and covariance schema now represents antenna metal loss
separately and applies its own antenna temperature. This removes the
previous software error for unequal lunar and antenna temperatures.
Producing a realistic lossy response still requires new lossy and PEC
solver products, which are not currently available.

The two previously identified blockers in the software model are resolved.
A scientifically validated realistic response still requires a new solver
export and an independent accepted-power or plane-wave terminal-voltage
check; no such artifact is currently available.

The map-making radiometric-noise calculation now uses the correct separate
real and imaginary cross-product variances. The map-maker still uses
diagonal weights, so it does not yet represent covariance between the two
components or between products that share a port; that approximation is now
explicit.

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

### 2. Resolved in software: converter units and normalization are explicit

The TeX gives two separate HFSS requirements:

- exported `rE_peak` is in mV and must be multiplied by
  `1e-3/sqrt(2)` to obtain RMS volts
  ([line 217](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L217));
- a 1 V peak incident wave at a matched port corresponds to a 2 V peak,
  or `sqrt(2)` V RMS, Thevenin source
  ([lines 249--253](../../new_four_port_paper/04_AsymmetricTwoAntenna.tex#L249)).

The converter now accepts four mutually explicit physical contracts:

1. embedded raw `rE` in V or mV, with a separately described Thevenin
   `Vsource`;
2. direct bare raw `rE` in V or mV, with a complex
   `(frequency, 4)` normalization-current array;
3. an explicit already-per-ampere `rE/I` in V/A or mV/A; or
4. effective length in m.

For the first two paths, field and voltage/current phasors each carry their
own RMS/peak convention. They are independently converted to SI RMS before
the current reconstruction or division. Consequently, a peak/peak ratio is
unchanged, while the TeX's mixed HFSS representation correctly applies
`1e-3/sqrt(2)` to `rE_peak` and uses the independently supplied
`sqrt(2) V` RMS Thevenin source. Already-formed `rE/I` and effective length
are declared as ratio quantities and receive no field-only peak/RMS scale.

Embedded effective length is rejected because current-basis unmixing is
defined on the raw embedded field, not on a receive ratio that has already
lost its excitation normalization. A direct raw bare field is no longer
silently assumed to represent a 1 A basis.

Validated provenance now records `FIELD_KIND`, `FIELD_UNIT`, `FIELD_AMP`,
`NORM_KIND`, `NORM_UNIT`, `NORM_AMP`, and the canonical
`H[m],SI-RMS` representation. Cross-field combinations are validated both
when writing and loading. The canonical SI RMS `Vsource` or `Inorm`
numerical payload is persisted and included in `CONTENT`, as is `Zref` for
embedded input.

Tests now cover the exact HFSS mV-peak/`sqrt(2) V`-RMS case, peak/RMS
representation invariance, complex direct-current normalization, rejection
of embedded effective length, frequency selection, and round trips of both
normalization payloads. These close the identified software hole. Certifying
a real artifact still requires one raw solver export and an independent
accepted-power or plane-wave terminal-voltage check.

### 3. Resolved at diagonal order: radiometric component variances

For a proper complex Gaussian voltage vector and
`N = delta_f delta_t` independent complex samples, the component statistics
are

```text
Var(Re C_hat_ab) =
    [Caa Cbb + Re(Cab^2)] / (2 N)

Var(Im C_hat_ab) =
    [Caa Cbb - Re(Cab^2)] / (2 N)

Cov(Re C_hat_ab, Im C_hat_ab) = Im(Cab^2) / (2 N).
```

[`compute_radiometric_noise`](../lusee/MapMaker.py) now evaluates those two
variances separately in both the response-v3 product-label path and the
legacy combination path. Autos retain `Var(Caa) = Caa^2/N`. A previous
absolute variance floor of `1e-30` was also removed: it was unit-dependent
and overwhelmed realistic `V^2/Hz` covariance scales. Only negative
values are clamped to zero before the square root; for a physical covariance
such negatives can arise only from roundoff.

Deterministic formula tests cover complex phase and very small physical
units, and a proper-complex Gaussian Monte Carlo independently reproduces
the real variance, imaginary variance, and their nonzero covariance.

This fixes every diagonal variance passed to the current solver. It does not
make the likelihood fully exact: real and imaginary channels are not
generally independent, and products that share ports also have sampling
covariance. [`solve`](../lusee/MapMaker.py) currently accepts diagonal
weights, so the omitted covariance is explicitly documented as an
approximation. A later extension can apply the full packed-real covariance
per time/frequency sample.

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

The focused new-path and converter suite, including explicit-loss and
normalization tests, completed without a compatibility shim with

```text
68 passed, 2 warnings
```

The warnings are Astropy's pre-existing assumption that two numerical
`TimeDelta` inputs are days. The independent ENU basis check passed at
machine precision.

The broader non-ingest suite reached 230 passing tests with no failures
before a remaining slow legacy Healpy rotation test was manually
interrupted after five minutes. Collecting the ingest suite is independently
blocked by the sibling `uncrater` checkout not exporting the `Packet` API
expected by this luseepy revision.

These tests use synthetic responses. They do not replace validation against
a new lossy/PEC antenna simulation, because no such artifact is available.

## Recommended implementation order

The following work is mechanically and mathematically specified without
further expert input:

1. **Complete:** extend the response and covariance model with explicit
   `Rloss` and `T_ant`, including unequal-temperature closure tests.
2. **Complete:** replace the converter's single amplitude flag with explicit
   field-unit, field-amplitude, and normalization conventions; add HFSS and
   representation-invariance tests.
3. **Complete:** correct real/imaginary radiometric variances and document
   the diagonal map-making approximation.
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
