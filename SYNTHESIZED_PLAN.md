# Synthesized plan: four-port, full-Stokes instrument response

Target repository: `luseepy`.

Status: **Phases 1--7 are implemented on
`codex/four-port-polarization-refactor`; the destructive Phase 8 legacy
cleanup is explicitly deferred pending next-major-release sign-off, and
release validation plus production response provenance review remain**.

The companion implementation is on Croissant branch
`codex/full-stokes-pair-response` as version 6.0.0. During co-development,
install both local checkouts explicitly with
`uv pip install -e ../croissant -e .`; release Croissant 6 before publishing
the luseepy package that declares `croissant-sim==6.0.0`.

This document supersedes `PLAN.md` and `POLARIZATION_PLAN.md` as the
consumer-side integration plan. The detailed polarization derivations and
Croissant implementation remain in `POLARIZATION_PLAN.md`, but the interface
and sequencing decisions below are normative for the luseepy integration.

The implementation baseline must contain the contracts already established by:

- `an/fix_freqs`: arbitrary target-frequency arrays, `FrequencyMap`, native-grid
  interpolation, exact snap-on-match, and differentiability with respect to
  interpolated values;
- `an/decorated_arrays`: bare arrays in numerical kernels, `LabeledArray`
  units/frame decoration at Python boundaries, canonical spectral-density
  labels, and explicit time-scale provenance at persisted-data boundaries.

The refactor must preserve both contracts. It must not restore a canonical-grid
restriction, move labels into hot JAX kernels, or add per-time pixel-space
polarization work.

---

## 1. Scope and fixed decisions

### 1.1 Instrument model

The instrument is one coupled four-port network. At each frequency:

- `Z_A(f)` is the complex `4 x 4` bare antenna impedance matrix.
- `H(f, Omega)` is the complex `4 x 2` bare open-circuit receive response,
  with columns corresponding to the local `(theta, phi)` polarization basis
  and units of meters.
- `Z_L(f; theta_rx)` is the complex `4 x 4` receiver load matrix.
- The loading matrix is

  ```text
  M(f) = Z_L(f) [Z_A(f) + Z_L(f)]^-1.
  ```

  It should be evaluated with a batched solve by default; for this 4 x 4 matrix, an explicit inverse is not obviously prohibitive, so we can benchmark both approaches on a Linux workstation with GPU and decide based on measured performance.
- The loaded voltage response is

  ```text
  v(f, Omega) = M(f) H(f, Omega) e(f, Omega).
  ```

The antenna is treated as perfectly conducting:

```text
R_loss = 0
C_ant = 0.
```

Receiver additive noise and post-JFET gain remain outside the forward
simulator. The `SpectrometerGain` analysis path is not part of this refactor.

### 1.2 Output

The primary simulator result is the Hermitian covariance at the four JFET
inputs:

```text
C_v(t, f) = <v(t, f) v(t, f)^dagger>,
```

with units `V^2/Hz`.

The stored channel representation contains:

- four real autos;
- real and imaginary parts of the six unique crosses;
- 16 real channels in total when `products="all"`.

A temperature-equivalent view is a derived convenience product. It is not the
primary simulation result.

### 1.3 Engines and polarization

- Keep `CroSimulator` and `TopoJaxSimulator`.
- Remove `TopoNumpySimulator` after the public cutover and validation gate.
- Support I, Q, U, and V from the first public release of the new response
  model.
- Implement spin transforms, spin-correct rotations, and component-aware
  convolution once in Croissant.
- `TopoJaxSimulator` consumes the same transformed pair-response inputs but
  retains an independent per-time rotation/contraction kernel.
- Croissant's scalar and existing multipair APIs remain numerically and
  shape-compatible.

### 1.4 Environmental approximation

The first implementation assumes that the instrument lands on a locally flat region whose ground normal is aligned with the instrument `z` axis, so the sky region is the upper hemisphere `theta <= pi/2`.

### 1.5 Differentiability

The following derivatives must remain supported:

- receiver parameters;
- sky pixels or sky harmonic coefficients;
- cached/native pair-response values supplied through the `beam=` override;
- antenna impedance values when supplied as differentiable leaves.

Target-frequency values are configuration, not differentiable variables.
`FrequencyMap` brackets are constructed on the host. Gradients flow through
the interpolated values and all downstream JAX operations.

---

## 2. Mathematical factorization

### 2.1 Sky coherency

For Stokes values in Rayleigh-Jeans kelvin,

```text
B_E = (k_B eta_0 / lambda^2)
      [[I + Q, U - i V],
       [U + i V, I - Q]].
```

For rows

```text
h_a = (H_a,theta, H_a,phi),
```

define the bare pair-Stokes responses:

```text
P_ab^I = H_a,theta H_b,theta* + H_a,phi H_b,phi*
P_ab^Q = H_a,theta H_b,theta* - H_a,phi H_b,phi*
P_ab^U = H_a,theta H_b,phi*   + H_a,phi H_b,theta*
P_ab^V = i (H_a,phi H_b,theta* - H_a,theta H_b,phi*).
```

Only the ten pairs `a <= b` are stored because

```text
P_ba^S = (P_ab^S)*.
```

The Stokes-V sign remains provisional until the joint luseepy/Croissant
positive-V fixtures freeze it.

### 2.2 Frequency-scaled response kernel

The quantity that is interpolated for diffuse simulation is the physical
pair-response kernel

```text
W_ab^S(f, Omega) = [eta_0 / lambda(f)^2] P_ab^S(f, Omega).
```

This choice is normative. It preserves the existing `an/fix_freqs` strategy
of interpolating an already normalized response product and ensures that the
sky and blackbody-normalization paths use exactly the same off-grid operator.

Because the scale is constant over angle at one frequency, luseepy may:

1. ask Croissant to transform the bare `P` maps at native frequencies;
2. multiply the resulting harmonic coefficients by
   `eta_0 / lambda_native^2`;
3. interpolate the scaled coefficients with `FrequencyMap`.

No additional spherical transform is needed at a target frequency.

### 2.3 Open-circuit covariance

At a target frequency:

```text
K_sky,ab(t, f) =
    k_B sum_S integral W_ab^S(f, Omega) S_S(f, Omega, t) dOmega.
```

The visible-sky dissipative matrix is

```text
R_sky,ab(f) = (1/4) integral W_ab^I(f, Omega) dOmega.
```

For consistency under beam overrides and off-grid interpolation,
`R_sky(f_target)` is derived cheaply from the target I-response monopole.
The stored native `R_sky` matrices are validation data and provide an
equivalent fast path only after equality with the harmonic monopole has been
tested.

The Moon complement is

```text
R_moon(f) = Re{Z_A(f)} - R_sky(f),
```

where

```text
Re{Z_A} = [Z_A + Z_A^dagger] / 2.
```

The complete open-circuit covariance is

```text
K(t, f) =
    K_sky(t, f) + 4 k_B T_moon R_moon(f).
```

The receiver-input covariance is

```text
C_v(t, f) = M(f) K(t, f) M(f)^dagger.
```

### 2.4 Blackbody identity

For an isotropic, unpolarized sky at temperature `T` and `T_moon = T`,

```text
C_v =
    4 k_B T M Re{Z_A} M^dagger.
```

This must hold at native and genuinely off-grid target frequencies. The
off-grid test is a release gate because it simultaneously checks:

- interpolation of the physical pair kernel;
- the target-frequency wavelength factor;
- the `R_sky` monopole;
- the Moon complement;
- impedance interpolation;
- receiver loading.

---

## 3. Frequency contract

### 3.1 Native and target grids

Every frequency-dependent gridded object has a native source grid:

```text
f_response
f_sky
f_measured_receiver, when applicable.
```

The user supplies a target array:

```text
f_target.
```

The target array:

- is one-dimensional, nonempty, finite, and expressed in MHz;
- may be irregular, unsorted, and contain duplicates;
- is preserved exactly in output order;
- need not coincide with any native sample;
- must lie inside the supported interval of every required gridded input;
- is never extrapolated.

Native grids must be finite, host-side NumPy `float64`, and strictly
increasing. The response `freq` HDU remains `float64` even when field arrays
are stored as float32/complex64.

### 3.2 Default target grid

If the user does not supply `freq`, the response file's native grid is the
default candidate target grid. When every other input is analytic or covers
the complete response interval:

```text
f_target = Beam.freq.
```

When a required gridded input has a narrower native interval, the default is
the ordered subset of `Beam.freq` inside the common supported interval. This
is an interval intersection, not a requirement that response and sky sample
locations coincide. The driver reports any default channels removed by this
range restriction.

`Beam.freq` is authoritative as the response's native grid and as the default;
it is not a restriction that target frequencies must snap to response bins.

The old fixed 1--50 MHz canonical grid is not part of the new public
simulation contract. Legacy canonical helpers may remain temporarily for
unmigrated callers, but they must not be used by the new simulator path.

### 3.3 Response interpolation

Construct one map:

```text
freq_map_response = FrequencyMap.build(f_target, Beam.freq).
```

Use it as follows:

1. Evaluate expensive pair-Stokes transforms only at
   `freq_map_response.source_indices`.
2. Scale native pair alms by `eta_0 / lambda_native^2`.
3. Apply `freq_map_response.from_unique(...)` to produce target pair alms.
4. Apply `freq_map_response.from_native(...)` to cheap native matrices such
   as `Z_A`.
5. Derive target `R_sky` from the target I-response monopole.
6. Derive target `R_moon` from the complement, not from an independently
   chosen interpolation policy.

The number of response SHTs is therefore bounded by the number of unique
native bracket endpoints:

```text
N_response_SHT <= min(N_response_native, 2 N_target).
```

Exact and near-exact native matches use `alpha = 0` and recover direct
indexing exactly.

### 3.4 Sky interpolation

For a gridded sky:

```text
freq_map_sky = FrequencyMap.build(f_target, sky.freq).
```

Compute I/V scalar alms and Q/U spin alms only at
`freq_map_sky.source_indices`, then apply `from_unique(...)` componentwise.
Frequency interpolation is linear and commutes with the linear harmonic
transform and coordinate rotation.

For an analytic sky model implementing `get_alm_at_freq`:

```text
sky_target = sky.get_alm_at_freq(f_target).
```

No interpolation map is built for that sky.

This dispatch must be performed on the effective `sky=` override passed to
`simulate`, not only on the sky used at simulator construction.

There is no special exception for polarized skies: I, Q, U, and V follow the
same target-grid contract. The Q/U representation may be spin `+/-2` or E/B
internally, but every component remains aligned with `f_target`.

### 3.5 Receiver interpolation

Analytic receiver models are evaluated directly:

```text
Z_L,target = receiver.Z(f_target).
```

A measured gridded receiver builds its own `FrequencyMap` and interpolates
`Z_L` onto `f_target`.

Always form

```text
M_target =
    Z_L,target [Z_A,target + Z_L,target]^-1
```

after interpolation. Do not interpolate a precomputed loading matrix because
matrix inversion is nonlinear.

### 3.6 Croissant ownership

Croissant transforms and convolution primitives are frequency-axis agnostic.
They do not choose a spectral interpolation policy.

Luseepy is responsible for:

- transforming only required native response/sky samples;
- applying the physical response scale;
- interpolating onto `f_target`;
- passing frequency-aligned component arrays to Croissant.

Croissant's standalone `PolarizedSimulator` may require already aligned sky
and beam frequency axes. This does not change luseepy's composable-primitive
integration.

### 3.7 Frequency configuration

Retain the `an/fix_freqs` value-based forms:

```yaml
observation:
  freq:
    values: [12.5, 17.3, 12.5]
```

```yaml
observation:
  freq:
    start: 1.0
    end: 50.0
    step: 0.5
```

```yaml
observation:
  freq:
    start: 1.0
    end: 75.0
    n: 149
```

Preserve the established parser semantics for `step` versus `n`. Do not
reintroduce index-based frequency configuration.

---

## 4. Units, frames, and decorated-array boundaries

### 4.1 Numerical kernels

Hot JAX/Croissant kernels operate on bare arrays. `LabeledArray` wrappers are
not carried through inner SHT, rotation, interpolation, or contraction loops.

This preserves:

- stable JIT signatures;
- existing MapMaker tracing;
- compact pytrees;
- current GPU performance.

Labels are validated and attached at Python and persisted-file boundaries.

### 4.2 Canonical labels

Use the existing canonical ASCII strings:

| Quantity | Units label | Frame label |
|---|---|---|
| Sky I/Q/U/V | `K` | sky frame |
| `H_theta`, `H_phi` | `m` | response/instrument frame |
| Bare pair response `P` | `m^2` | response/instrument frame |
| Scaled pair kernel `W` | `Ohm` | response/instrument frame |
| `Z_A`, `Z_L`, `R_sky`, `R_moon` | `Ohm` | none |
| `M` | `1` | none |
| `C_v` | `V^2/Hz` | `topo` |
| Blackbody normalization | `V^2/(Hz K)` | `topo` |
| Temperature-equivalent result | `K` | `topo` |
| Frequency | `MHz` | none |

The primary simulation label must use the existing `V2_PER_HZ` constant from
`GainModel`, not a new spelling.

### 4.3 Simulator return contract

Preserve the decorated-array branch behavior:

- `simulate()` returns a bare NumPy or JAX array;
- `self.result` remains bare;
- `self.result_units = V2_PER_HZ`;
- `self.result_frame = FRAME_TOPO`;
- `result_labeled` returns a `LabeledArray` view.

This keeps JIT/grad callers compatible while preventing the old `"K"` label
from surviving the physical-output change.

### 4.4 Data return contract

`Data.__getitem__` returns `LabeledArray` values:

- `Data[..., "01R", ...]`: `V^2/Hz`, `topo`;
- `Data[..., "01I", ...]`: `V^2/Hz`, `topo`;
- `Data[..., "01C", ...]`: `V^2/Hz`, `topo`;
- the `K` suffix: `K`, `topo`.

The old `V` suffix is removed or retained temporarily as a documented no-op;
it must never apply the old `Throughput` conversion to an already physical
PSD.

### 4.5 Cross-package metadata

Croissant's public polarized objects carry explicit metadata for:

- units;
- coordinate frame;
- IAU/COSMO convention;
- Stokes order;
- spatial sampling;
- tangent-basis convention;
- pair ordering and baseline direction;
- native frequency grid and frequency units.

The luseepy adapter validates these values, unwraps any luseepy
`LabeledArray`, calls bare Croissant primitives, and attaches the luseepy
boundary label after physical postprocessing.

---

## 5. Timestamp contract

### 5.1 Simulation input

`simulate(times=...)` accepts an Astropy/lunarsky `Time` array or input that is
coerced once to a `Time` array with an explicit scale.

The supplied timestamps are authoritative. The simulator must not replace
them with

```text
arange(N_time) * obs.deltaT_sec.
```

### 5.2 Croissant epoch and phases

Use:

```text
times[0].tdb.jd
```

for the absolute SPICE/MEPA epoch.

Compute elapsed phase times from the actual supplied timestamps, converted to
TDB seconds relative to `times[0]`. This supports gaps, subsets, and
nonuniform cadence without changing the diagonal-in-`m` time kernel.

### 5.3 Result ownership

After a successful simulation:

```text
self.result_times = exact supplied Time array.
```

The length of `result_times` must equal the first dimension of `result`.
`write_fits()` writes `result_times`, not a synthetic reconstruction from
`time_range` and `deltaT_sec`.

### 5.4 Persisted time metadata

The simulator FITS output contains an explicit absolute time axis, preferably
MJD values, with:

- `TIMESYS`;
- `TIMEUNIT`;
- clock/source provenance;
- an explicit indication of whether any scale was assumed.

`TIMECONV='e+jwt'` remains response-phasor metadata and must not be used as a
timestamp-scale field.

New-format `Data` reads the exact stored timestamps. Legacy files may retain
the current uniform-grid reconstruction fallback.

---

## 6. Instrument response FITS v3

### 6.1 Header

The primary response header records:

```text
VERSION = 3
PORTS
SOURCE
SOURCE_ROOT
INPUT_KIND
FIELD_KIND
AMP_CONV
TIMECONV
ZA_SOURCE
GIT_SHA
COORDSYS
THETADEF
PHIDEF
OMEGADEF
POLBASIS
PHASEREF
VALIDATED
```

It also records native grid sizes/ranges, solver coordinate handedness,
arrival-versus-propagation direction, and the far-field phase origin.

### 6.2 HDUs

| HDU | Shape | Persisted units |
|---|---:|---|
| `freq` | `(Nfreq,)` | `MHz` |
| `H_theta_real/imag` | `(4,Nfreq,Ntheta,Nphi)` | `m` |
| `H_phi_real/imag` | `(4,Nfreq,Ntheta,Nphi)` | `m` |
| `ZA_real/imag` | `(Nfreq,4,4)` | `Ohm` |
| `Rsky_real/imag` | `(Nfreq,4,4)` | `Ohm` |
| `Rmoon_real/imag` | `(Nfreq,4,4)` | `Ohm` |
| `Vsource_real/imag` | `(Nfreq,4,4)` | `V` |
| `Zref` | `(Nfreq,4)` when needed | `Ohm` |

Each image HDU carries a machine-readable `BUNIT`; table columns use `TUNIT`.
The loader validates units rather than relying only on comments or file
version.

`freq` is always float64. `--dtype {float32,float64}` controls large field
and derived-map storage, not the native frequency axis.

### 6.3 Conversion

The converter supports independently specified:

```text
--input-kind {embedded,bare}
--field-kind {rE,effective-length}
--amplitude-convention {rms,peak}
```

For embedded inputs, reconstruct the bare basis using batched right-side
solves. Never use a batched `.T` or explicit inverse.

Compute native:

```text
R_sky = eta_0/(4 lambda^2) integral H H^dagger dOmega
R_moon = Re{Z_A} - R_sky.
```

Use the same endpoint-aware solid-angle quadrature later used by validation.
Negative eigenvalues of `R_moon` are a hard validation failure for production
responses and advisory only for explicitly marked placeholder `Z_A`.

### 6.4 Provenance and validation

Unknown solver amplitude, basis, units, or termination provenance is a hard
error unless `--allow-unvalidated` is supplied. Such files carry
`VALIDATED=False` and are rejected by flight-like configurations.
The implemented command-line converter receives the reviewed fields through
`--provenance-json`; it never promotes placeholder defaults to
`VALIDATED=True`.

---

## 7. Public object model

### 7.1 `Beam`

`Beam(fname)` loads one four-port response.

Bare numerical attributes:

```text
freq
theta
phi
H_theta
H_phi
ZA
Rsky_native
Rmoon_native
```

Metadata properties expose their units, frame, coordinates, port order, and
provenance without wrapping JAX leaves.

Required methods:

- `rotate(deg)`: whole-instrument azimuthal rotation for the initial model;
- `pair_stokes_maps(a, b, freq_ndx=None)`;
- `pair_stokes_alms_native(lmax, source_indices)`;
- `pair_stokes_alms(lmax, target_freqs)` as a convenience wrapper using
  `FrequencyMap`, not as an independent interpolation implementation;
- `loaded_response_at(...)` for diagnostic voltage-amplitude use;
- `sky_coupling_check()`.

The low-level native method accepts unique native indices. It must not confuse
those indices with one-index-per-target compatibility shims.

### 7.2 Croissant `PairStokesBeam`

Input layout:

```text
(pair, native_frequency, stokes, theta, phi)
```

with:

```text
pair = 10 unique a <= b pairs
stokes = (I, Q, U, V).
```

The samples are complex and have units `m^2`. Croissant performs transforms
without physical normalization. Luseepy applies `eta_0/lambda^2` to the
native transformed values before frequency interpolation.

### 7.3 Polarized sky

The polarized sky public layout is:

```text
(native_frequency, stokes, spatial...)
```

with real pixel values in kelvin, explicit IQUV ordering, a coordinate frame,
and an IAU public convention. COSMO conversion is explicit.

Existing I-only sky models provide I and zero/absent QUV through an adapter
without changing their scalar public return shapes.

### 7.4 Receiver impedance

Interface:

```text
receiver.Z(freq_mhz) -> (Nfreq, 4, 4) complex JAX array.
```

Implement:

- `JFETReceiver`;
- `IdealCapacitorReceiver`;
- `MeasuredReceiver`.

Parameters and their units are explicit metadata. Receiver pytrees contain
only differentiable numerical parameters as children and immutable model
metadata as auxiliary data.

### 7.5 SimulatorBase

Constructor:

```text
SimulatorBase(
    obs,
    beam,
    sky_model,
    receiver,
    T_moon=250.0,
    products="all",
    freq=None,
    lmax=...,
)
```

Construction:

1. set `f_target`;
2. build response and effective-sky frequency maps;
3. identify unique native response/sky indices;
4. prepare or load cached native harmonic products;
5. scale and interpolate pair-response alms;
6. interpolate/evaluate small matrices at targets;
7. prepare target-aligned sky inputs.

Simulation:

1. rotate target-aligned sky/pair alms;
2. convolve all polarization components;
3. assemble the Hermitian `K_sky`;
4. derive `R_sky` from the same target I-response monopole;
5. form the Moon complement;
6. compute `M K M^dagger`;
7. pack requested channels;
8. retain the exact timestamp array.

### 7.6 Calibrator

For covariance products at arbitrary frequencies, the calibrator uses the same
interpolated pair-Stokes kernel as the diffuse path, evaluated at the source
direction. This prevents the diffuse and point-source covariance paths from
silently adopting different frequency-interpolation conventions.

Direct `v = M H e` comparisons remain required at native frequencies to
validate signs and pair-map construction. Interpolation of coherent voltage
amplitudes is a separate diagnostic convention and is not used to define the
primary covariance product.

### 7.7 `Data`

`Data` reads:

- exact stored target frequencies and units;
- exact stored timestamps and time scale;
- packed product labels;
- primary `V^2/Hz` data;
- target matrices and blackbody normalization;
- response/receiver provenance.

It does not construct `Throughput`.

---

## 8. Simulation output FITS

### 8.1 Required HDUs

```text
data
freq
time
products
ZA_real/imag
ZL_real/imag
M_real/imag
Rsky_real/imag
Rmoon_real/imag
blackbody_normalization_real/imag
receiver_params
```

The file header includes:

- response path and content hash;
- response validation state;
- receiver model and channel mapping;
- engine and luseepy/Croissant/s2fft versions;
- target frequency interpolation method;
- sky model provenance;
- observation location;
- timestamp provenance.

### 8.2 Target-grid guarantee

Every frequency-dependent output HDU is indexed by the exact stored `freq`
array. No output HDU remains silently indexed by the response or sky native
grid.

The output preserves unsorted targets and duplicates. Native-grid diagnostics,
if retained, use explicitly named HDUs and include their own frequency axis.

### 8.3 Units

Each numeric HDU carries `BUNIT` or per-column `TUNIT`. At minimum:

```text
data                              V^2/Hz
freq                              MHz
time                              d (when stored as MJD)
ZA, ZL, Rsky, Rmoon               Ohm
M                                 1
blackbody_normalization           V^2/(Hz K)
```

The reader rejects contradictory units. Legacy files without unit tags follow
the legacy version-specific interpretation.

---

## 9. Performance architecture

### 9.1 No target-frequency SHT loop

Transforms occur only at unique native bracket endpoints. Target-frequency
interpolation acts on transformed arrays and small matrices.

Do not:

- interpolate full `H(theta, phi)` maps to every target and then transform;
- construct a target-frequency pixel cube solely for interpolation;
- rerun a native transform for duplicate targets.

### 9.2 No per-time pixel tensor

The production time path remains:

```text
sum over component, ell, m of
    sky_alm* x phase(time,m) x pair_alm.
```

Time enters through diagonal `m` phases in CroSimulator and through the
independent Wigner rotation in TopoJaxSimulator. Neither path creates
`(time, frequency, pixel)` arrays.

### 9.3 Linear-operation ordering

Frequency interpolation, harmonic transforms, and fixed coordinate rotations
are linear. Order them to minimize transformed batch size and memory:

- transform only native bracket endpoints;
- interpolate before repeated target/time work;
- apply frequency-independent rotations after interpolation when that reduces
  the batch dimension;
- never duplicate large native arrays merely to attach labels.

### 9.4 Caching

Cache native pair transforms with keys containing:

- response content hash;
- one response native index or a stable native-index chunk;
- `lmax`;
- spatial sampling;
- Stokes/spin convention version;
- complex precision;
- Croissant and s2fft versions;
- coordinate rotation metadata where the cached object is already rotated.

Use per-index or stable-chunk entries so overlapping target arrays reuse their
common native endpoints. Do not key one indivisible cache artifact by the
entire requested endpoint set.

The target `FrequencyMap` and interpolated target arrays are cheap and do not
belong in a cross-target native-transform cache unless profiling demonstrates
a benefit.

### 9.5 Batching and memory

- Batch native frequencies during pair transforms.
- Batch pairs/components if required by accelerator memory.
- Store only ten unique port pairs.
- Permit complex64 native response storage, while accumulating and emitting
  float64 science outputs when x64 mode is enabled.
- Keep frequency grids and interpolation weights float64.
- Measure host/device duplication of Croissant dense matrices and pairalms.

### 9.6 Complexity targets

Preparation:

```text
O(N_unique_response x N_pair x N_component x SHT_cost)
```

Per-time convolution:

```text
O(N_target x N_pair x N_component x lmax^2)
```

Receiver postprocessing:

```text
O(N_time x N_target x 4^3),
```

which is small relative to the harmonic contraction.

No implementation phase is complete until profiling confirms that off-grid
support does not introduce an SHT per target or a pixel operation per time.

---

## 10. Implementation sequence

Each phase adds its own tests. The legacy public implementation remains
available internally until the single public cutover in Phase 7.

### Phase 0: freeze conventions and branch contracts

1. Record the electrical, polarization, coordinate, unit, frequency, and time
   conventions in one normative document.
2. Preserve the `FrequencyMap` and `LabeledArray` public tests unchanged.
3. Add contract tests that fail if:
   - the new simulator rejects in-range off-grid frequencies;
   - target order or duplicates are lost;
   - `simulate()` starts returning a wrapper;
   - `result_labeled` has the wrong units/frame;
   - exact timestamps are replaced by nominal cadence.
4. Freeze the native-versus-target terminology used throughout APIs and file
   schemas.
5. Freeze the frequency-scaled `W` interpolation rule and blackbody relation.

### Phase 1: Croissant polarization prerequisites

Proceed in the Croissant repository according to `POLARIZATION_PLAN.md`:

1. pin corrected s2fft commit
   `cefdf468ec2540818bafb37ed60d7b1fbba2f21f`;
2. generalize transforms to spin and complex inputs;
3. implement the complex pair Q/U harmonic dual;
4. implement dense spin transforms;
5. add `PolarizedSky` and `PairStokesBeam`;
6. implement component-aware convolution and spin-correct rotations;
7. preserve scalar APIs and benchmarks.

The luseepy integration blocks on Croissant's transform and composable
convolution phases, not on its optional orchestration class.

### Phase 2: response converter and FITS v3

1. Implement CSV/Touchstone readers and placeholder/full `Z_A`.
2. validate solver basis, units, port order, amplitude convention, and solves;
3. write the machine-tagged FITS v3 schema;
4. compute native `R_sky` and `R_moon`;
5. keep `freq` float64 under every storage dtype;
6. add small synthetic converter round trips;
7. add noncommuting embedded-current and peak/RMS tests.

### Phase 3: four-port `Beam` and native transforms

1. Implement the four-port JAX pytree under a migration-internal name.
2. load and validate FITS units/metadata;
3. implement pair-Stokes maps;
4. delegate native transforms to Croissant;
5. cache transforms by unique native indices;
6. apply native physical frequency scaling;
7. interpolate scaled alms with `FrequencyMap`;
8. derive target `R_sky` from the target I monopole;
9. add native/off-grid equivalence and gradient tests.

### Phase 4: receiver and matrix postprocessing

1. Implement JFET, capacitor, and measured receiver models.
2. evaluate analytic models directly on targets;
3. use a separate `FrequencyMap` for measured receiver grids;
4. interpolate `Z_A`, then form `M`;
5. assemble Hermitian matrices from ten unique pairs;
6. implement the Moon complement and physicality diagnostics;
7. test complex noncommuting matrices and parameter gradients.

### Phase 5: both diffuse simulators

1. Implement target-aligned polarized sky preparation.
2. Implement CroSimulator with Croissant's component convolution.
3. Implement TopoJaxSimulator's independent polarized contraction.
4. preserve effective `sky=` and `beam=` override dispatch.
5. use actual timestamp differences and TDB epoch conversion.
6. pack requested products into the bare result.
7. attach only the boundary `result_labeled` view.
8. profile preparation, memory, JIT compilation, and per-time runtime.

### Phase 6: calibrator and point-source covariance

1. Evaluate the same target pair-response kernel at source directions.
2. compare with direct `M H B_E H^dagger M^dagger` at native frequencies.
3. add linear/circular IQUV and V-sign fixtures.
4. add at least one genuinely off-grid point-source covariance test.

### Phase 7: driver, Data, timestamps, and public cutover

1. update the value-based config schema without restoring frequency indices;
2. write exact target frequencies, units, and timestamps;
3. update `Data` labels and suffix grammar;
4. update MapMaker construction, channel packing, and radiometric noise;
5. update examples and documentation;
6. run the complete suite with both engines;
7. switch `Beam`, simulators, driver, `Data`, calibrator, and MapMaker
   atomically to the new public implementation.

### Phase 8: remove obsolete paths

Implementation status: deferred. The current branch keeps version/type-gated
legacy facades so existing scalar-beam callers continue to work. Deleting the
NumPy/per-antenna/converter paths is a separate next-major-release decision
because it is destructive and is not required for correctness or performance
of the FITS-v3 path.

After the cutover passes:

- delete the NumPy engine;
- delete old per-antenna coupling/throughput machinery;
- delete old converter implementations;
- delete tests only for code that was intentionally removed;
- retain all scalar Croissant regression tests;
- retain all `FrequencyMap`, decorated-array, gain, and ingest provenance tests;
- rerun the full suite immediately after deletion.

### Phase 9: release validation and benchmarks

Run the complete matrix in Section 11. Record:

- native and off-grid numerical tolerances;
- SHT count versus unique source indices;
- peak host/GPU memory;
- first-call compilation time;
- cached-call preparation time;
- per-time/per-frequency throughput;
- float32/complex64 and float64/complex128 accuracy where supported.

No release is made while a performance regression is explained only by
"polarization is more expensive"; the measured component-axis cost must match
the intended constant-factor increase and preserve the harmonic hot path.

---

## 11. Validation matrix

### 11.1 Frequency behavior

- exact native target;
- near-native target snaps with `alpha=0`;
- genuinely off-grid target;
- irregular, unsorted targets;
- duplicate targets;
- target at both native boundaries;
- out-of-range target fails with the offending value;
- response and sky on different native grids;
- analytic sky evaluated directly;
- gridded IQUV sky interpolated componentwise;
- effective `sky=` override with a different native grid;
- gradients through interpolated sky/beam/impedance values;
- float64 frequency-grid regression near 75 MHz.

### 11.2 Blackbody and matrix physics

- native-frequency blackbody enclosure;
- off-grid blackbody enclosure;
- random complex `Z_L`;
- Hermitian `K` and `C_v`;
- real autos;
- positive-semidefinite `C_v` for physical inputs;
- `R_sky` target monopole agrees with the interpolated physical response;
- Moon complement physicality checks;
- interpolation preserves `Z_A = R_sky + R_moon` in the dissipative part.

### 11.3 Polarization

- pure I, Q, U, and V;
- mixed IQUV;
- IAU/COSMO conversion;
- 45-degree tangent-basis rotation;
- spin `+2` and `-2` analytic modes;
- positive-V linear and circular feeds;
- baseline reversal/conjugation;
- complex leakage;
- no scalar rotation of Q and U.

### 11.4 Independent numerical paths

- CroSimulator versus TopoJaxSimulator;
- each harmonic engine versus direct native-grid IQUV quadrature;
- native point source versus direct matrix calculation;
- synthetic converter to FITS to Beam to pair maps;
- no shared sign error accepted solely through a round trip.

### 11.5 Decorated-array boundaries

- `simulate()` returns a bare array;
- `result_labeled` is `V^2/Hz`, `topo`;
- raw `Data` products are `V^2/Hz`, `topo`;
- K-equivalent products are `K`, `topo`;
- FITS unit tags round trip;
- contradictory units fail;
- labels do not appear in inner JAX kernel pytrees.

### 11.6 Timestamps

- uniform UTC input;
- nonuniform input;
- a subset with gaps;
- an equivalent instant represented in UTC and TDB;
- Croissant uses `times[0].tdb.jd`;
- elapsed phases use actual timestamps;
- FITS `TIMESYS` round trip;
- `Data.times` equals the stored sample times exactly;
- legacy reconstruction remains isolated to legacy files.

### 11.7 Performance

- number of response transforms equals unique native bracket endpoints;
- duplicate targets add no transforms;
- no target-frequency pixel cube is constructed;
- no time-frequency-pixel tensor is constructed;
- cached native transforms are reused across target grids;
- JIT cache remains stable for repeated calls with the same static shapes;
- frequency and pair batching respect the configured memory budget.

---

## 12. Configuration schema

Proposed breaking schema:

```yaml
response:
  file: <instrument-fits-v3>
  rotation_deg: 0
  require_validated: true

receiver:
  model: jfet
  channel_map: [fmpre0, fmpre2, fmpre5, fmpre7]
  params: {}

observation:
  time_range: "2025-03-01 00:00:00 to 2025-03-02 00:00:00"
  dt: 900
  T_moon: 250.0
  products: all
  freq:
    values: [12.5, 17.3, 25.0]

sky:
  type: file
  file: <sky-file>

simulation:
  engine: croissant
  lmax: 128
  output: simulation.fits
```

When `observation.freq` is omitted, use the response native channels inside
the common supported interval of all required gridded inputs. When it is
supplied, preserve the values exactly and apply the frequency contract in
Section 3.

---

## 13. Caveats and deferred work

- Solver CSV termination basis, field quantity, amplitude convention, and
  phase provenance must be resolved before production validation.
- Real `Z_A` replaces the placeholder without changing downstream APIs.
- The Stokes-V sign is frozen jointly with Croissant using independent source
  fixtures.
- No spectral extrapolation is performed. A gridded sky or measured receiver
  limits the supported target interval even when the response extends farther.
- `BeamInterpolator` adaptation remains a follow-up after the main cutover.
- Dense measured `Z_L` file format remains to be specified.
- The K-equivalent cross-product view remains advisory near normalization
  zeros.
- The temporary s2fft Git pin is replaced only after an upstream release
  passes the independent spin regression.

---

## 14. Acceptance criteria

The refactor is complete only when:

1. all four Stokes components produce correctly signed complex port
   correlations;
2. the primary output is a labeled and persisted `V^2/Hz` covariance;
3. arbitrary in-range target-frequency arrays work without extrapolation,
   preserve order/duplicates, and require transforms only at unique native
   bracket endpoints;
4. native and off-grid blackbody identities pass;
5. receiver, sky, and beam gradients pass finite-difference checks;
6. exact timestamps and their scales survive simulation and FITS round trips;
7. CroSimulator, TopoJaxSimulator, and direct-grid oracles agree;
8. scalar Croissant APIs and existing tests remain unchanged;
9. decorated-array, gain, ingest-provenance, and `FrequencyMap` contracts remain
   intact;
10. profiling confirms no per-target SHT loop, no per-time pixel path, and no
    unexplained memory or runtime regression.
