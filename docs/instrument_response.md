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

The receiver is evaluated on those same target frequencies. The loading
matrix is formed after interpolation with a batched right-side solve:

```text
M = ZL (ZA + ZL)^-1.
```

The primary result is the JFET-input covariance `M K M^dagger` in `V^2/Hz`.
`simulate()` and `result` stay bare arrays for JAX. `result_labeled` attaches
`V^2/Hz` and `topo` only at the Python boundary.

Simulation FITS files store exact target frequencies and exact supplied MJD
timestamps with `TIMESYS`, `TIMEUNIT`, clock source, and scale-assumption
metadata. `Data` reads these axes directly. The legacy `V` selection suffix
is a no-op for new physical-PSD files; a `K` suffix selects the derived
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

Validation also requires Hermitian `Rsky` and `Rmoon`,
`Rsky + Rmoon = (ZA + ZA^dagger)/2`, and positive-semidefinite `Rmoon`.
The `InstrumentResponse` loader repeats these convention and physical checks;
`VALIDATED=True` is not accepted as a substitute for checking the payload.
When a response supplies `CONTENT`, the loader uses it directly and does not
rehash the large field arrays during normal loading.

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

On the 2026-07-23 CPU development run at `lmax=17`, 10 pairs, 64 frequencies,
and 128 times, preparation performed exactly eight native response
transforms. Repeating preparation performed no additional transforms. The
cached harmonic contraction took 1.50 ms and the cached batch of 64 loading
solves took 0.061 ms. These numbers are a smoke baseline, not a substitute
for the planned Linux GPU release benchmark.

MapMaker accepts FITS v3 directly as one four-port instrument and uses the
same physical 16-channel product order for radiometric-noise estimates.
Legacy beam files remain available through the compatibility branch while
existing callers migrate.
