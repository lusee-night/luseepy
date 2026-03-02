# Jaxify Notes

## Scope
This file summarizes the current differences in this repo vs the clean reference:
`/Users/anigmetov/code/lusee_night/luseepy_clean/luseepy`

Focus:
- default simulator jaxification
- preserving a stable NumPy baseline
- parity/timing test workflow

## Engine Split
- `engine=luseepy` / `engine=default` / `engine=lusee` / `engine=jax`:
  uses `lusee.DefaultSimulator` (JAX-oriented path).
- `engine=numpy`:
  uses `lusee.NumpySimulator` (legacy NumPy path; baseline for parity checks).
- `engine=croissant`:
  unchanged.

Files:
- `simulation/driver/sim_driver.py`
- `lusee/DefaultSimulator.py`
- `lusee/NumpySimulator.py`
- `lusee/__init__.py`

## Precision Config
Config key:
- `simulation.jax_enable_x64: true|false`

Behavior:
- Applied in `SimDriver` before importing `lusee` (`jax.config.update("jax_enable_x64", ...)`).
- Affects JAX dtypes in the default engine path.

Files:
- `simulation/config/realistic_example.yaml`
- `simulation/driver/sim_driver.py`

## DefaultSimulator (JAX) Architecture
`DefaultSimulator` is now batch-functional:
1. Build rotation kernels once.
2. Rotate one sky across all times via `vmap`.
3. Contract beams x rotated sky in one batched JIT kernel.

Notable points:
- Old non-JAX `healpy.rotate_alm` branch removed from `DefaultSimulator`.
- Output products are flattened to one output axis for vectorized contraction.

File:
- `lusee/DefaultSimulator.py`

## Rotation Path (Critical Convention)
This is the fragile part and must stay consistent.

Current bridge:
1. Build local frame vectors:
   - `zhat` from zenith `(lz, bz)`
   - `yhat` from horizon `(ly, by)`
   - `xhat = cross(yhat, zhat)`
2. Build matrix `R = [xhat, yhat, zhat]^T`.
3. Convert `R` -> `(a, b, g)` via legacy `rot2eul`.
4. Recreate healpy rotation:
   - `hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype='XYZ', inv=False)`
5. Convert `rot.mat` to ZYZ with SciPy:
   - `ScipyRotation.from_matrix(rot.mat).as_euler("ZYZ")`
6. Use those ZYZ angles in JAX Wigner-d rotation.

Why:
- `s2fft` rotation API is ZYZ-based.
- Legacy semantics are encoded through healpy XYZ construction.
- Matrix bridge via healpy keeps conventions aligned to legacy outputs.

SciPy version currently in env:
- `scipy==1.16.1`

If SciPy changes:
- Re-verify `as_euler("ZYZ")` ordering/sign semantics.
- Re-run numeric parity checks.

## Wigner-d Implementation / Differentiability
Rotation kernel uses:
- `s2fft.utils.rotation.compute_full` (recurrence-based Wigner-d blocks)
- plus JAX phase factors + contraction glue.

We do not use a hand-coded closed-form Wigner implementation.

Gradient check script:
- `simulation/driver/gradcheck_s2fft_compute_full.py`

It checks autodiff vs finite-diff through:
- direct `compute_full` loss
- toy rotation loss

Observed relative errors are very small (around `1e-10` to `1e-11`) in x64 mode.

## NumPy Baseline Isolation (Important)
To keep `engine=numpy` robust and independent of JAX state:

1. `NumpySimulator` enforces complex128 sky ALMs before `healpy.rotate_alm`
   (`rotate_alm` requires double-precision complex arrays).
2. `NumpySimulator` uses a pure NumPy `_mean_alm_numpy` helper.
3. Sky models expose NumPy-native ALM accessor `get_alm_numpy`.
   - `NumpySimulator` uses `get_alm_numpy` when available.

Reason:
- In-process warmup/timing runs exposed cross-talk if NumPy path consumed JAX-returned arrays.

Files:
- `lusee/NumpySimulator.py`
- `lusee/SkyModels.py`

## Regression / Timing Scripts
### Clean-vs-modified parity
- `simulation/driver/compare_default_vs_clean.sh`
- Uses clean checkout and modified repo, compares FITS HDUs with tolerance.

### Default-vs-NumPy in this repo
- Shell entrypoint: `simulation/driver/compare_default_vs_numpy.sh`
- In-process runner: `simulation/driver/compare_default_vs_numpy_inproc.py`

Design:
- For each engine, instantiate simulator once.
- Run warmup `simulate()` first.
- Time second `simulate()` call on the same object (same process, same JIT state).

Shell script behavior:
- Runs two precision cases:
  1. `jax_enable_x64=false` with `ABS_TOL32` (default `1e-2`)
  2. `jax_enable_x64=true` with `ABS_TOL64` (default `1e-9`)
- Prints per-case summary and combined PASS/FAIL.

CLI:
- `--lmax`
- `--freq-end`

Env vars:
- `ABS_TOL32`
- `ABS_TOL64`

## Typical Command
```bash
simulation/driver/compare_default_vs_numpy.sh --lmax 8 --freq-end 3
```

## Constraints to Preserve
- Keep `engine=numpy` as baseline compatibility path.
- Keep `DefaultSimulator` as jaxified path.
- Do not modify rotation convention plumbing without re-validating parity.
- For timing comparisons, avoid separate-process warmup assumptions; use in-process second-run timing.
