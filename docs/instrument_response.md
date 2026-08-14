# Four-port instrument response

The physical simulator accepts one coupled four-port response instead of four
independent scalar beams. The public `lusee.Beam(path)` facade dispatches
FITS-v3 files to `InstrumentResponse`; legacy FITS-v1/v2 files continue to
use the old scalar implementation during the migration window.

The response contains the bare open-circuit effective lengths
`H_theta/H_phi`, the full antenna impedance `ZA`, and the native
`Rsky`/`Rmoon`/`Rloss` decomposition. Every image HDU carries `BUNIT`.
Frequency is stored as float64 MHz. The first implementation assumes a
locally flat landing region whose ground normal is aligned with the
instrument z axis, so the visible sky is `theta <= pi/2`.
When the stored upper-hemisphere maps are padded with zero below the
horizon, the shared `theta=90 degrees` ring receives half weight. This is the
spherical-harmonic midpoint value of the horizon step and avoids an
`O(delta theta)` endpoint bias in both `Rsky` and sky contraction.

`InstrumentResponse.rotate` and the configuration's `rotation_deg` use
positive astronomical azimuth, from North toward East. The response grid is
right-handed ENU with `phi=0` at East, so `+90 degrees` moves a directional
response initially at `phi=0` to `phi=270 degrees`, or South. In map space
this is a roll by negative phi bins; in harmonic space it is the equivalent
`exp(+i m alpha)` phase.

Polarized sky providers must use consistent `coord` and `frame` metadata.
Equivalent aliases such as `equatorial` and `fk5` are accepted after
canonicalization, but contradictory physical frames are rejected before any
harmonic contraction. MCMF is body-fixed and is not accepted as an alias for
MEPA; it requires an epoch-dependent transport. A native MEPA sky is
understood in the MEPA frame frozen at the simulation's first timestamp,
while a native topocentric sky stays fixed in the instrument frame.
Croissant's sky object does not carry the MEPA epoch itself, so this is an
explicit caller contract; covariance FITS provenance records that reference
as `SKYREFJD`/`SKYREFSY`.

The Croissant engine rotates the topocentric response and celestial sky into
that frozen MEPA frame once, then advances the contraction with
`exp(-i m phi(t))` phases about the MEPA z axis. This is the intended fast
lunar-rotation approximation. It neglects small nonuniform-rotation and
non-z-axis libration/nutation corrections; the independent topocentric
engine remains available for direct per-timestamp rotations and comparison.

The four-port driver constructs its analytic CMB, Cane1979, and Dark Ages
monopoles as full-sky Galactic maps. It does not reuse the legacy monopole
fixture's MCMF label or hard-coded lower-sky cone mask; the four-port
response already applies its own horizon. Full-sky monopoles use an exact
`l=0` coefficient rather than a pixelized map transform. Legacy simulator
defaults are unchanged.

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
The four-port response, polarized-sky, and measured-receiver paths select
the `linear` frequency policy explicitly. This preserves their intentional
interpolation while the legacy simulator API retains its safer `exact`
default.

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

Before the final Hermitian projection, the simulator records the absolute
and relative anti-Hermitian residual for every time/frequency matrix. It also
records all four covariance eigenvalues, the minimum eigenvalue relative to
the largest absolute eigenvalue, and the per-frequency condition number of
`ZA + ZL`. These are diagnostics only: finite-`lmax` sky/response
contractions can be non-PSD when under-resolved, so the simulator neither
clips nor unconditionally rejects negative covariance eigenvalues.

Simulation FITS files store exact target frequencies and supplied timestamps
as two-part `JD1+JD2` values with `TIMESYS`, `TIMEUNIT`, clock source, and
scale-assumption metadata. This preserves Astropy's double-double time
precision; `Data` also retains read compatibility with the development-only
single-MJD form. The legacy `V` selection suffix is a no-op for new
physical-PSD files; a `K` suffix selects the derived
blackbody-normalized view.
The same files persist the covariance and loading diagnostics, with maximum
or minimum summaries in the data header; `Data.covariance_diagnostics`
exposes them after loading.

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
the converter records the input quantity, field unit and phasor convention,
normalization kind, normalization unit and phasor convention, and canonical
SI representation from their command-line options. Missing or `UNKNOWN`
values are rejected. Use `--allow-unvalidated` only for diagnostic artifacts.

The converter accepts five explicit physical input contracts:

- embedded raw `rE` in V or mV, with an independently described Thevenin
  `Vsource`;
- direct bare raw `rE` in V or mV, with a complex `(frequency, 4)`
  normalization-current array;
- explicitly pre-normalized bare `rE/I` in V/A or mV/A;
- effective length in m; or
- **loaded** effective length in m (`INPUT_KIND='loaded'`,
  `NORM_KIND='unloaded-zl'`): solver-side receive fields
  `R = ZL (ZA + ZL)^-1 H` with a documented load matrix baked in. The
  converter recovers bare `H = (ZA + ZL) ZL^-1 R` by batched solve and
  persists the applied `ZLoad` as a hash-bound complex
  `(frequency, 4, 4)` HDU, validated for shape, finiteness, and
  invertibility on write and load. This is the contract of the actual
  `Receive_Matrix_Fields_{N,E,S,W}.csv` production exports, which carry
  the spare-preamp-averaged JFET `ZL` reproduced exactly by
  `lusee.ReceiverImpedance.spare_preamp_average_zload`.

Embedded effective length is rejected because the circuit unmixing operation
is defined on the raw embedded fields. Peak/RMS scaling is applied separately
to raw fields and to `Vsource` or the normalization current before forming a
ratio. No peak/RMS scaling is applied to an already-formed `rE/I` or effective
length. For the HFSS convention in the TeX, use
`--field-units mV --field-amplitude peak` together with a `sqrt(2) V` RMS
Thevenin array and `--vsource-units V --vsource-amplitude rms`.

`ZA` may come from a Touchstone file (`--touchstone`) or from a dense
complex Z-matrix CSV with `freq_Hz,freq_MHz,re/im(Zij)` columns
(`--zmatrix-csv`, the skrf-converted `.s4p` export format). The stored
response azimuth convention is ENU with `phi=0` at local East; solver
grids with a different azimuth zero are rolled with
`--phi-source-zero-deg` (the ENU azimuth of the solver's `phi=0` axis)
and the applied roll is recorded as `SRCAZ0` provenance. The LuSEE HFSS
frame has the west antenna along `+x` and south along `+y`, so its
exports use `--phi-source-zero-deg 180`.

## Production artifacts

`beam_conversion/lusee_bgl_v16.py` is the one-command driver for the
LuSEE_BGL_V16 export set (loaded contract, `--zmatrix-csv`, 180-degree
azimuth roll, full solver provenance, `LOSSMODEL=PEC`). It produced

- `receive_matrix/lusee_bgl_v16_response_v3.fits` — the real unloaded
  response, `VALIDATED=True` at all 150 frequencies (0.5-75 MHz);
- `receive_matrix/lusee_bgl_v16_response_v3_c4sym.fits` — its C4
  group average (`beam_conversion/symmetrize_response.py`: circulant
  `ZA`, rotation-averaged patterns), the exactly four-fold-symmetric
  reference instrument.

**Default input contract (since 2026-08-06):** *bare* effective-length
exports (`Bare_Effective_Length_Fields_{N,E,S,W}.csv` with
`re/im(h_Theta)`, `re/im(h_Phi)` columns in meters, ratio convention,
from the ReceiveMatrix notebook's `export_fields_to_csv`) are the
standard solver deliverable going forward. Convert them with the plain
bare/effective-length contract:

```bash
JAX_ENABLE_X64=1 python -m beam_conversion.receive_csv \
    Bare_Effective_Length_Fields_{N,E,S,W}.csv \
    --output <response>.fits --zmatrix-csv Complex_Z_Matrix.csv \
    --input-kind bare --field-kind effective-length --field-units m \
    --field-amplitude ratio --pec --phi-source-zero-deg 180 \
    --dtype float64 --provenance-json <provenance>.json
```

This was cross-validated on 2026-08-06 against the unloaded `H` of
`lusee_bgl_v16_response_v3.fits`: identical grids and `ZA`, and
`H_theta`/`H_phi` agreeing to ~1e-15 relative (float64 rounding). The
loaded contract and the `lusee_bgl_v16.py` unloading driver remain
supported for the legacy `Receive_Matrix_Fields_*` exports only.
See `docs/four_port_physics_review.md` (provenance review) and
`docs/old_vs_new.md` (legacy-pipeline comparison) for the physics record.

Validation derives `Rsky` from `H_theta/H_phi` with the same Croissant MWSS
monopole operator used by simulation. A supplied `Rsky` must match that
result; reciprocal antennas must satisfy `ZA approximately ZA.T`;
`Herm(ZA)` must be passive; `Rsky + Rmoon + Rloss` must equal
`(ZA + ZA^dagger)/2`; and all three resistance matrices must be Hermitian and
positive semidefinite. A PEC response must declare its explicit zero
`Rloss`. The `InstrumentResponse` loader always repeats the geometry,
convention, and normalization-payload checks and verifies the persisted
`CONTENT` hash, which binds the payload bytes to what the writer validated.
The field/matrix re-derivation itself (recomputing `Rsky` from the fields
and re-running the dissipative-matrix gates) costs minutes of CPU on the
production artifact and therefore runs only when the loader is called with
`verify_physics=True` (config key `response.verify_physics` in the sim
driver).

`InstrumentResponse.response_diagnostics()` reports the corresponding
per-frequency residuals, eigenvalues, and `ZA` condition number. For an
embedded-field artifact it also reconstructs the current-unmixing condition
number from the persisted `ZA`, `Vsource`, and `Zref`; its maximum is stored
as hash-bound `MAX_ICOND`. This is report-only until a representative solver
export and numerical-error requirement justify an acceptance threshold.

The response tests include an absolute Hertzian-dipole oracle,
`Rsky = eta0 pi leff^2 D D^T/(3 lambda^2)`, evaluated across multiple MWSS
grids. A separate smooth-sky regression compares the harmonic contraction
with high-order Gauss--Legendre and periodic-azimuth quadrature and verifies
convergence through `lmax=8`. Neither expected result is computed with the
production spherical transform.

`CONTENT` hashes the canonical persisted precision of every numerical
response array, including canonical SI `Vsource` or `Inorm` data when
applicable, together with semantic convention/provenance metadata. The
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
same physical 16-channel product order for radiometric-noise estimates. For
`Cab = x + i y` and `N = delta_f delta_t`, the packed diagonal variances are
`(Caa Cbb + x^2 - y^2)/(2N)` for the real channel and
`(Caa Cbb - x^2 + y^2)/(2N)` for the imaginary channel. The current solver
uses these correct component variances but still neglects their mutual
covariance and covariance between products that share a port.
Legacy beam files remain available through the compatibility branch while
existing callers migrate.
