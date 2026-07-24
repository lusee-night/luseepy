# Four-port instrument response

The physical simulator accepts one coupled four-port response instead of four
independent scalar beams. The public `lusee.Beam(path)` facade dispatches
FITS-v3 files to `InstrumentResponse`; legacy FITS-v1/v2 files continue to
use the old scalar implementation during the migration window.

The response contains the bare open-circuit effective lengths
`H_theta/H_phi`, the full antenna impedance `ZA`, and native validation
matrices. Every image HDU carries `BUNIT`. Frequency is stored as float64 MHz.
The first implementation assumes a locally flat landing region whose ground
normal is aligned with the instrument z axis, so the visible sky is
`theta <= pi/2`.

Pair response maps are formed for the ten unique port pairs:

```text
P_I = H_a_theta H_b_theta* + H_a_phi H_b_phi*
P_Q = H_a_theta H_b_theta* - H_a_phi H_b_phi*
P_U = H_a_theta H_b_phi*   + H_a_phi H_b_theta*
P_V = i (H_a_phi H_b_theta* - H_a_theta H_b_phi*).
```

Croissant transforms only the unique native bracket endpoints required by a
`FrequencyMap`. LuSEE then scales the native coefficients by
`eta0/lambda^2` and linearly interpolates those physical response
coefficients. Irregular, unsorted, duplicate in-range target arrays are
preserved exactly; extrapolation is rejected.

Response transforms are chunked over native frequency and pair axes under a
512 MiB default workspace budget. Set
`LUSEE_RESPONSE_TRANSFORM_MAX_BYTES` to a positive byte count to tune this
for a particular accelerator. Chunking changes compilation granularity, not
the cached native coefficients or target interpolation.

The receiver is evaluated on those same target frequencies. The loading
matrix is formed after interpolation with a batched right-side solve:

```text
M = ZL (ZA + ZL)^-1.
```

The primary result is the JFET-input covariance `M K M^dagger` in `V^2/Hz`.
`simulate()` and `result` stay bare arrays for JAX. `result_labeled` attaches
`V^2/Hz` and `topo` only at the Python boundary.

Simulation FITS files store exact target frequencies and supplied timestamps
as two-part `JD1+JD2` values with `TIMESYS`, `TIMEUNIT`, clock source, and
scale-assumption metadata. This preserves Astropy's double-double time
precision; `Data` also retains read compatibility with the development-only
single-MJD form. The legacy `V` selection suffix is a no-op for new
physical-PSD files; a `K` suffix selects the derived
blackbody-normalized view.

See `simulation/config/four_port_example.yaml` for the new value-based
configuration.

Until Croissant 6 is published, a fresh two-repository development checkout
is installed explicitly with:

```bash
uv pip install -e ../croissant -e .
```

The luseepy package metadata intentionally contains the release dependency
`croissant-sim==6.0.0`, not a developer-local source override. Croissant must
therefore be released before this luseepy branch is released; after that,
normal `uv sync`/`uv run` resolution uses the published immutable artifact.

## Converter validation

`beam_conversion.receive_csv` writes `VALIDATED=True` only when every
required solver and coordinate convention is present in an explicit
`--provenance-json` object. At minimum, that object must identify
`SOURCE`, `SOURCE_ROOT`, `ZA_SOURCE`, `GIT_SHA`, `TIMECONV`, `COORDSYS`,
`THETADEF`, `PHIDEF`, `OMEGADEF`, `POLBASIS`, `PHASEREF`, and `PORTS`;
the converter records the chosen input, field, and amplitude conventions
from their command-line options. Missing or `UNKNOWN` values are rejected.
Use `--allow-unvalidated` only for diagnostic artifacts.

Validation derives `Rsky` from `H_theta/H_phi` with the same Croissant MWSS
monopole operator used by simulation. A supplied `Rsky` must match that
result; `Rmoon` must be its complement in
`(ZA + ZA^dagger)/2`, Hermitian, and positive semidefinite. The
`InstrumentResponse` loader repeats the geometry, convention, field/matrix,
and Moon-physicality checks, so `VALIDATED=True` is not accepted as a
substitute for checking the payload.

`CONTENT` hashes the canonical persisted precision of every numerical
response array together with semantic convention/provenance metadata. The
loader recomputes and verifies it. This makes the value suitable for
transform-cache and simulation provenance keys; it is intentionally not a
hash of the converter's higher-precision pre-cast source arrays.

`Data.provenance`, `Data.response_provenance`,
`Data.receiver_provenance`, `Data.sky_provenance`, and
`Data.software_versions` expose the corresponding simulation FITS header
records without putting strings into numerical arrays.

## Performance check

`examples/benchmark_full_stokes.py` exercises response preparation, the
polarized harmonic contraction, and the receiver loading solve without
constructing a target-frequency pixel cube or a time-frequency-pixel tensor.
It deliberately uses 64 irregular targets (including duplicates) backed by
eight native response channels.

On the 2026-07-23 Linux CPU and RTX 2080 Ti runs at `lmax=17`, 10 pairs,
64 frequencies, and 128 times, preparation performed exactly eight native
response transforms. Repeating preparation performed no additional
transforms. The cached harmonic contraction took 18.3 ms on CPU and
0.614 ms on GPU; the cached batch of 64 loading solves took 0.313 ms and
0.262 ms respectively.

`examples/benchmark_loading_matrix.py` uses explicit JAX `vmap`, lowering,
compilation, and synchronized timing to compare the production solve with a
closed-form 4x4 adjugate/determinant inverse. The determinant occasionally
saves a few hundredths of a millisecond at individual-user frequency counts,
but loses on large GPU batches, compiles more slowly, and becomes much less
accurate as conditioning worsens. The production implementation therefore
retains the batched solve. Full commands, CPU/GPU tables, memory results, and
remaining release gates are recorded in
[`four_port_evaluation.md`](four_port_evaluation.md).

MapMaker accepts FITS v3 directly as one four-port instrument and uses the
same physical 16-channel product order for radiometric-noise estimates. The
complex Eq. 9 cross variance is split equally between the stored real and
imaginary components, giving the documented factor of four in each packed
channel; both components are required to evaluate `|V_ij|^2`.
Legacy beam files remain available through the compatibility branch while
existing callers migrate.
