# Four-port full-Stokes implementation evaluation

Evaluation date: 2026-07-23

Compatibility re-evaluation: 2026-07-27

Branches:

- luseepy: `codex/four-port-polarization-refactor`
- Croissant: `codex/full-stokes-pair-response` at `4677c7e`

The evaluation used Croissant directly from
`../croissant/src` through `PYTHONPATH`. It did not install or import the
public PyPI implementation. The local source paths and branch heads were
checked before the test runs. All accepted GPU results below are from the
workstation's NVIDIA GeForce RTX 2080 Ti; no cluster result is included.

The environment still contains old `croissant-sim==5.1.3` distribution
metadata, so `importlib.metadata` reports that version even while
`croissant.__file__` resolves to the companion checkout. This does not
affect the imported code used here, but a local editable install of both
checkouts is required before generating production FITS provenance.

For the 2026-07-27 compatibility re-evaluation,
`croissant.__file__` resolved to the sibling checkout at rebased commit
`6bdc17f` and distribution/package version `6.0.0`. luseepy now treats
`coord` as authoritative, supports native topocentric input through the
companion branch, rejects MCMF and contradictory frame metadata, and treats
bare Croissant frequency values as MHz at its public boundary without
inventing a `frequency_units` attribute.

## Review outcome

An independent `@review` found issues in validation, memory scaling,
provenance, and JAX execution. The following changes close the release
blockers it identified:

- `Rsky` and `Rmoon` are now derived and checked with the same Croissant MWSS
  harmonic operator used by simulation, binding the stored matrices to the
  persisted effective-length fields.
- The canonical response hash covers every persisted response array,
  including optional source/reference data, plus physical convention and
  provenance metadata at the precision actually written.
- Response and polarized-sky units, authoritative coordinates, tangent basis,
  IAU/IQUV order, pair order, baseline direction, and visibility definition
  are checked at public boundaries. Bare Croissant frequencies follow
  luseepy's documented MHz contract.
- Response transforms are chunked over native frequency and pair under a
  configurable 512 MiB workspace budget. Croissant only retains a dense
  low-bandlimit HEALPix operator when one matrix fits its configurable
  256 MiB limit.
- The independent topocentric engine now compiles and vmaps its dynamic
  rotations over time. Traced simulations no longer store traced results in
  the simulator's public state.
- FITS timestamps use two-part `JD1+JD2`, record scale/coercion provenance,
  validate product-column units, and reject future schema versions rather
  than guessing their layout.
- Radiometric-noise calculation requires both real and imaginary cross
  products, and its per-component factor-of-four convention is documented.
- Regression coverage now includes Q/U celestial rotation agreement,
  antenna-impedance gradients, response-field/matrix consistency, content
  tampering, metadata rejection, transform chunking, exact timestamps, and
  partial cross-product rejection.

The intentionally destructive legacy cleanup in Phase 8 remains deferred, as
specified by `SYNTHESIZED_PLAN.md`.

## Test results

All CPU runs used JAX 0.9.2 with `JAX_PLATFORMS=cpu`,
`JAX_ENABLE_X64=1`, and the local Croissant source checkout.

| Backend | Suite | Result | Elapsed |
|---|---|---:|---:|
| CPU | luseepy `tests/` | 211 passed, 8 skipped | 473.98 s |
| CPU | Croissant `tests/` | 431 passed | 524.71 s |
| GPU | luseepy four-port, response, covariance, driver, and gradient tests | 54 passed | 273.60 s |
| GPU | Croissant dense, polarization, rotation, and multipair tests | 42 passed | 133.24 s |

Warnings were limited to expected offline IERS fallback, Astropy
`TimeDelta` warnings already present in the suite, Healpy deprecation
warnings, and a read-only pytest cache warning for the companion checkout.

## Loading-matrix decision

`examples/benchmark_loading_matrix.py` compares three compiled JAX
implementations:

1. the production batched right-side `solve`;
2. an explicit `vmap` of one-matrix `solve`;
3. an explicit `vmap` of a closed-form 4x4 adjugate/determinant inverse.

Every shape is lowered and compiled before 100 synchronized steady-state
timings. Gradients with respect to both `ZA` and `ZL` and a condition-number
sweep are also evaluated.

Selected steady-state results:

| Device / dtype / batch | Batched solve | Explicit determinant | Result |
|---|---:|---:|---|
| CPU, complex64, 64 | 0.256 ms | 0.217 ms | determinant saves 0.039 ms |
| CPU, complex64, 16,384 | 44.63 ms | 28.40 ms | determinant faster |
| CPU, complex128, 64 | 0.258 ms | 0.249 ms | effectively tied |
| CPU, complex128, 16,384 | 44.57 ms | 35.25 ms | determinant faster |
| GPU, complex64, 64 | 0.143 ms | 0.119 ms | determinant saves 0.024 ms |
| GPU, complex64, 16,384 | 0.290 ms | 0.361 ms | solve is 1.24x faster |
| GPU, complex128, 64 | 0.156 ms | 0.119 ms | determinant saves 0.037 ms |
| GPU, complex128, 16,384 | 1.240 ms | 2.710 ms | solve is 2.18x faster |

At the large GPU shape, production `solve` reaches 56.4 million matrices/s
in complex64 and 13.2 million matrices/s in complex128. The determinant
kernel reaches 45.3 million and 6.05 million matrices/s respectively, and
has substantially longer compilation.

For the tens-of-frequencies batches expected for individual use, the
determinant's occasional steady-state advantage is only a few hundredths of
a millisecond and does not repay its compilation or numerical cost. In the
complex64 condition sweep, at condition number approximately `1e4`, the
solve residual was `1.66e-4` while the determinant residual was `1.23e-2`;
near `9.7e5`, they were `7.27e-3` and `27.6`. In complex128 near condition
number `1e12`, they were `1.72e-5` and `2.27e3`. Well-conditioned values and
`ZA`/`ZL` gradients agree at the expected dtype precision.

Decision: retain the production batched `jnp.linalg.solve`. The explicit
determinant is viable only as a benchmark/reference kernel, not as the
default numerical implementation.

## End-to-end and memory benchmarks

The default full-Stokes benchmark uses eight native response frequencies,
64 irregular/duplicated target frequencies, 128 times, 10 pairs, and
`lmax=17`.

| Stage | CPU | RTX 2080 Ti |
|---|---:|---:|
| Response construction and validation | 2.287 s | 4.766 s |
| First response preparation | 5.792 s | 6.105 s |
| Cached response preparation | 33.2 ms | 6.50 ms |
| Sky transform | 2.377 s | 4.150 s |
| First compiled contraction | 179 ms | 332 ms |
| Cached contraction | 18.3 ms | 0.614 ms |
| First loading solve | 335 ms | 210 ms |
| Cached loading solve | 0.313 ms | 0.262 ms |

The GPU run used 192 MiB peak device memory. It transformed exactly the
eight unique native endpoints and did no extra work on cached preparation.

A larger individual-user workload used 12 native frequencies, 16 targets,
64 times, 10 pairs, two-degree response sampling, and `lmax=64`. It selected
a `(10 pairs, 5 frequencies)` transform chunk, retained 85.9 MB of output
alms, used 0.85 GB peak device memory, prepared the response in 15.75 s, and
ran the cached contraction in 1.31 ms.

Chunking bounds transform workspaces, not the required output. For example,
50 target frequencies, 10 pairs, four components, `lmax=128`, and complex128
coefficients inherently require about 0.99 GiB for the pair alms. Individual
users still need to choose target count, precision, and `lmax` for available
memory.

Croissant's `nside=32`, `lmax=30`, complex128 dense-HEALPix benchmark
retained 539.1 MiB across the spin 0, -2, and +2 matrices. On the RTX 2080 Ti
the three builds took 49.82 s total and cached applications took
6.28--6.48 ms. An `nside=64`, `lmax=64` matrix is estimated at about
3.09 GiB, so it is rejected by the default 256 MiB per-matrix limit and uses
the supported full-transform/reduction fallback.

## Remaining release gates

- Validate a real production solver response and its provenance; the current
  automated and performance fixtures are synthetic.
- Release Croissant 6 before resolving or publishing luseepy's
  `croissant-sim==6.0.0` dependency. Until then, developers must use the
  companion checkout explicitly.
- Obtain next-major-release sign-off before the deferred Phase 8 removal of
  legacy NumPy/scalar-beam paths.
- Decide whether native-transform reuse must persist across response
  instances or processes. The current cache is correctly keyed by instance,
  native index, and `lmax`, but is intentionally in-memory only.
