# Jaxify Notes

## Scope
This file summarizes the main differences in this repo vs the clean reference at
`/Users/anigmetov/code/lusee_night/luseepy_clean/luseepy`, focused on the DefaultSimulator jaxification work.

## Engine Split
- `engine=luseepy` / `engine=default` / `engine=lusee` / `engine=jax`:
  uses `lusee.DefaultSimulator` (JAX-oriented path).
- `engine=numpy`:
  uses `lusee.NumpySimulator` (legacy NumPy path, intended to match clean behavior).
- `engine=croissant`:
  unchanged.

Files:
- `simulation/driver/sim_driver.py`
- `lusee/NumpySimulator.py`
- `lusee/__init__.py`

## Precision Config
Config key:
- `simulation.jax_enable_x64: true|false`

Important:
- This is applied in `SimDriver` before importing `lusee`, so JAX dtype mode is set early.

Files:
- `simulation/config/realistic_example.yaml`
- `simulation/driver/sim_driver.py`

## DefaultSimulator Refactor (JAX way)
`DefaultSimulator` now computes independent timesteps in a batched functional style:
1. Build a single-time rotation+contraction pipeline.
2. Vectorize over time with `vmap`.
3. JIT compile the batched kernels.

Also:
- The old non-JAX `healpy.rotate_alm` fallback branch was removed from `DefaultSimulator`.
- Output products are flattened into a single output axis and contracted in one batched tensor expression.

File:
- `lusee/DefaultSimulator.py`

## Rotation Convention (Critical)
This is the most fragile part.

Current convention bridge:
1. Build local frame vectors from transformed coordinates:
   - `zhat` from zenith `(lz, bz)`
   - `yhat` from horizon `(ly, by)`
   - `xhat = cross(yhat, zhat)`
2. Build matrix `R = [xhat, yhat, zhat]^T`.
3. Convert `R` to `(a, b, g)` via `rot2eul` (legacy code path).
4. Recreate the exact healpy rotation object with:
   - `hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype='XYZ', inv=False)`
5. Convert `rot.mat` to ZYZ Euler angles with SciPy:
   - `ScipyRotation.from_matrix(rot.mat).as_euler("ZYZ")`
6. Feed these `(alpha_zyz, beta_zyz, gamma_zyz)` into the JAX Wigner-d rotation kernel.

Why this is done:
- `s2fft` Wigner rotation path is ZYZ-based.
- The legacy simulator semantics are encoded through the healpy XYZ construction.
- Using the healpy matrix as the intermediary keeps conventions aligned.

Environment version used for this behavior:
- `scipy==1.16.1`

If SciPy is upgraded/downgraded:
- Re-verify `Rotation.from_matrix(...).as_euler("ZYZ")` semantics and output ordering/sign conventions.
- Re-run numeric parity checks against clean/healpy baseline.

## Differentiability-Oriented Changes
- `mean_alm` in `SimulatorBase` is now pure `jax.numpy` (no `np.asarray` bridge).
- Sky model `get_alm` now returns JAX arrays.
- Default simulator removed explicit NumPy conversion of sky coefficients.

Files:
- `lusee/SimulatorBase.py`
- `lusee/SkyModels.py`
- `lusee/DefaultSimulator.py`

## Validation / Regression
Script:
- `simulation/driver/compare_default_vs_clean.sh`

Notable behavior:
- Compares all FITS HDUs (including `data`).
- Supports `ABS_TOL` (e.g. `ABS_TOL=1e-9`).
- Restores `LUSEE_OUTPUT_DIR` after clean-run comparison.

Recent parity level:
- `max_abs_diff(data)` is typically ~`1e-10` on quick runs, within `1e-9` tolerance.

## Constraint to Keep
- Keep `engine=numpy` as the baseline compatibility path.
- Keep `DefaultSimulator` as the jaxified path.
- Do not change rotation convention plumbing without re-validating against clean/healpy outputs.
