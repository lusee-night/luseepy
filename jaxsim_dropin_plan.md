# Design Plan: Make `JaxSimulator.simulate` a drop-in for `CroSimulator.simulate`

Target caller: `examples/optax_maxlike.py` where the only simulator touch point is
```python
def forward(sim, sky):
    return sim.simulate(sky=sky)
```
`CroSimulator.simulate(self, times=None, *, sky=None, beam=None)` already flows
`jax.grad` through `sky.mapalm`. `JaxSimulator.simulate(self, times=None)` does
not accept `sky=`/`beam=`. This document describes what is required to close
that gap without code changes in this pass.

---

## 1. Proposed signature for `JaxSimulator.simulate`

Change from
```python
def simulate(self, times=None):
```
to
```python
def simulate(self, times=None, *, sky=None, beam=None):
    """
    :param sky: optional pytree sky with .get_alm(ndx) and .frame attribute;
                when supplied, its leaves are used in the traced path instead of
                self.sky_model. Must have the same frame as self.sky_model
                (re-using cached rotation transforms relies on it).
    :param beam: optional pytree that exposes either:
                   - a JAX-native tuple list self.efbeams shape (Nout, nfreq, alm_size), or
                   - a precomputed `_output_beams_flm` and `_output_ground`.
                 When omitted, the cached self._output_beams_flm is used.
    """
```

### Awkwardness / caveats specific to JaxSimulator

- `self._output_beams_flm` and `self._output_ground` are pre-stacked and
  pre-`hp_to_full_flm` transformed at `__init__` time. Accepting a `beam=`
  kwarg therefore requires a second, on-the-fly transformation path:
    - Either the caller passes a pytree that already carries a
      `_output_beams_flm` / `_output_ground` pair, or
    - JaxSimulator provides a small helper `_beams_to_output_tensors(efbeams)`
      that reproduces `_prepare_output_tensors` on the supplied pytree.
- Cached rotation angles `(alpha, beta, gamma)` depend only on `times` and the
  lunar coordinates in `self.obs`; they do not depend on `sky` or `beam`, so
  swapping either leaves the disk cache valid (see §3).
- `self.freq_ndx_sky` / `self.freq_ndx_beam` were computed from the **original**
  sky/beam at construction. A caller-supplied `sky=` is expected to match on
  `.freq` and `.frame`; we should either document that or add a cheap equality
  check (not a hard trace-time comparison).

---

## 2. Sky differentiability trace

Current call path in `JaxSimulator.simulate` (lusee/JaxSimulator.py:386 ff.):

1. Line 439 — `sky_base = jnp.asarray(self.sky_model.get_alm(self.freq_ndx_sky))`
   Runs at Python level inside `simulate` (which is itself **not** `jax.jit`-ed).
   - `HealpixSky.get_alm(ndx)` returns `self.mapalm[ndx]`, i.e. a **jnp array
     leaf of the pytree** (see `lusee/SkyModels.py:226-239`).
   - If a `sky` pytree is passed in and `simulate` is called inside a
     `jax.grad` / `jax.jit` trace, `self.mapalm` is a tracer and the resulting
     `sky_base` is a tracer. No captured-constant problem — Python-level call
     is fine because the kernel below is a freshly-traced function of
     `sky_base`.
2. Line 440 — `sky_base_flm = self._hp_to_full_flm_batch_jax(sky_base)`
   Runs inside a `jax.jit` boundary. Substituting a different sky just re-traces
   (recompile once per input shape/dtype; since sky shape/dtype are stable for
   optax updates, this is a one-shot warm-up cost).
3. Lines 499 / 520 — `_rotate_and_contract_time_batch_jax(sky_base_flm, alpha,
   beta, gamma, self._output_beams_flm, self._output_ground)`.
   `sky_base_flm` enters as a traced positional argument. Beams and ground are
   captured from `self`; they are jnp arrays so they become traced constants
   on first compile — **fine for gradients w.r.t. sky**, but not w.r.t. beam
   pytrees (they would need to be positional args to enable `beam=` grad flow).
   They already are positional in the current JIT signature, so wiring is
   straightforward.
4. Line 538 (MCMF path) — `self.simulate_at_single_time(sky_base_flm)` in a
   `vmap`. Same story: sky tracer flows in.

### Key finding: grad path is already traceable

Because (i) `simulate` is not itself jitted and (ii) the kernels that consume
`sky_base_flm` are jitted and take it positionally, adding a `sky=` kwarg only
requires overriding the source of `sky_base` — everything downstream already
handles tracers correctly.

### Precomputed tensors keyed off `self.sky_model` — none

The only sky-derived precomputation is inside `simulate` itself (`sky_base`,
`sky_base_flm`). Nothing is cached across calls keyed on `self.sky_model`.
`self.freq_ndx_sky` is derived from `self.freq` vs `self.sky_model.freq` in
`SimulatorBase.__init__` — that is the single static coupling; a drop-in
`sky=` must respect it (same `freq` grid).

### `_hp_to_full_flm_batch_jax` boundary

Created by `jax.jit(hp_to_full_flm_batch)` at line 271 inside `_setup_jax_rotate_ops`.
Invoked at line 440 in `simulate` — i.e. strictly **inside** the jit boundary.
Tracers flow through unchanged.

### Tracer-leak audit vs the recent `HealpixSky.tree_unflatten` fix

`HealpixSky.tree_unflatten` was fixed to use `np.asarray(freq)` so `freq` in
`aux_data` remains a static tuple rather than being promoted to a tracer. In
`JaxSimulator.simulate` the analogous risk would be any call like
`jnp.asarray(sky.freq)` that then feeds a static-shape computation. Scanning
the hot path:
- `self.freq_ndx_sky` is a Python int list set at construction (safe).
- `sky_model.get_alm(self.freq_ndx_sky)` — `ndx` is a Python list → converted
  via `jnp.atleast_1d(jnp.asarray(ndx))` inside `HealpixSky.get_alm`. Fine.
- No code path does `np.asarray(sky.freq).tolist()` or similar inside the
  traced region.

Conclusion: JaxSimulator does not reintroduce the tracer-leak pattern.

---

## 3. Caching side-effects

The pickle cache `<prefix>__rot_transforms_<key>.pickle` is keyed by
`_transform_cache_key(times)` which hashes (see `SimulatorBase.py:218-230`):
- `time_jd`
- `lun_lat_deg`, `lun_long_deg`, `lun_height_m`
- `deltaT_sec`

No `sky`/`beam` dependency. Repeated `simulate(sky=sky_i)` calls therefore:
- Reuse the on-disk cache across optax iterations.
- Reuse the in-memory `lzl, bzl, lyl, byl` if `simulate` is hoisted to
  populate a Python-level closure. Right now the code re-reads the pickle every
  call (lines 413-429) — inexpensive for optax, but an obvious optimisation is
  to remember the computed `(alpha, beta, gamma)` jax arrays on `self` keyed
  by `id(times)` or the hash itself to avoid redoing `_compute_zyz_angles`
  per step. **Out of scope** for this plan but noted.

### Proposed memoisation (optional, safe)

Inside `simulate`, after line 453, store
```python
self._last_zyz = (times, alpha, beta, gamma)
```
and skip recomputation when `times is self._last_zyz[0]`. This removes the
per-step Python cost of looping over `Nt` timestamps in `_compute_zyz_angles`,
which for 28-day lunar cycles at `DT_SEC=7200` is 336 iterations — small, but
Python-side overhead adds up across 20k optax steps.

---

## 4. Frame support

- `CroSimulator`: raises `NotImplementedError` if `sky_model.frame != "galactic"`
  (CroSimulator.py:133).
- `JaxSimulator`: supports `"galactic"` (→ rotate path) and `"MCMF"` (→ no
  rotation path); everything else raises (JaxSimulator.py:435).

Adding `sky=` kwarg must keep both paths working. The cached rotation
transforms `(lzl, bzl, lyl, byl)` are frame-independent (they are alt-az →
galactic-l-b transforms of the observatory pole). Switching between
galactic and MCMF sky pytrees across calls is therefore allowed — but the
cache file would still be keyed the same way, so there is no invalidation
risk. The only requirement: caller must not silently change
`sky.frame` mid-optimisation if the Python-level `do_rot` branch was chosen
during the first call (the kernel structure differs). In `examples/optax_maxlike.py`
the sky is always galactic, so this is a non-issue in practice.

Recommend: if `sky is not None` and `sky.frame != self.sky_model.frame`, raise
early with a clear error rather than silently branching.

---

## 5. Minimal edit plan

### File: `lusee/JaxSimulator.py`

Function `simulate` (lines 386-580). Sketch only (NOT applied):

```python
def simulate(self, times=None, *, sky=None, beam=None):
    if times is None:
        times = self.obs.times

    sky_model = sky if sky is not None else self.sky_model
    if sky is not None and sky_model.frame != self.sky_model.frame:
        raise ValueError("sky.frame must match self.sky_model.frame")

    if beam is None:
        output_beams_flm = self._output_beams_flm
        output_ground   = self._output_ground
    else:
        # Either accept a pre-built pair, or stack+transform on the fly.
        output_beams_flm, output_ground = self._beams_to_output_tensors(beam.efbeams)

    # ── rotation transform (unchanged) ───────────────────────────────
    if sky_model.frame == "galactic":
        do_rot = True
        # ... existing pickle cache logic, unchanged ...
    elif sky_model.frame == "MCMF":
        do_rot = False
    else:
        raise NotImplementedError

    Nt = len(times)

    # ── changed: read alm from sky_model arg, not self.sky_model ─────
    sky_base = jnp.asarray(sky_model.get_alm(self.freq_ndx_sky))
    sky_base_flm = self._hp_to_full_flm_batch_jax(sky_base)

    if do_rot:
        alpha, beta, gamma = self._compute_zyz_angles(lzl, bzl, lyl, byl, Nt)
        alpha, beta, gamma = map(jnp.asarray, (alpha, beta, gamma))
        time_batch_size = self._time_batch_size(Nt)
        # pass output_beams_flm / output_ground instead of self._output_beams_flm / self._output_ground
        if time_batch_size >= Nt:
            self.result = self._rotate_and_contract_time_batch_jax(
                sky_base_flm, alpha, beta, gamma,
                output_beams_flm, output_ground,
            )
        else:
            # ... chunked path, same substitution ...
    else:
        self.result = jax.vmap(
            lambda _: self._contract_single_sky_jax(
                sky_base_flm, output_beams_flm, output_ground
            )
        )(jnp.arange(Nt))

    # plotting path: unchanged (uses self.efbeams[0][2] — which refers
    # to the cached beam, not the traced one; acceptable because
    # plot_sky_and_beam is a side-effect-only dev knob).
    return self.result
```

Add helper:
```python
def _beams_to_output_tensors(self, efbeams):
    """Stack a (user-supplied) efbeams list into (_output_beams_flm, _output_ground).
    Mirrors the body of _prepare_output_tensors but returns rather than assigns."""
    ...
```

### Re-JIT cost concerns

- `_rotate_and_contract_time_batch_jax` is already jitted with explicit
  positional args `(sky_base_flm, alpha_batch, beta_batch, gamma_batch,
  output_beams_flm, output_ground)`. Under optax, sky shape/dtype are constant
  → **no re-tracing** across iterations.
- `alpha_batch` length changes between the last-chunk path and full path; this
  already forces two compilations today. Passing a different `beam` tensor
  with the same shape does not trigger re-tracing.
- A `beam` pytree with a *different* shape (e.g. different number of
  combinations) would retrace — but the example uses a fixed 10-combination
  layout, so not a concern.
- The plotting branch (`extra_opts["plot_sky_and_beam"]`) calls
  `_rotate_sky_flm_batch_jax(sky_base_flm, alpha[0], beta[0], gamma[0])`
  which has a different argument-shape signature (scalar angles vs batched)
  and triggers its own (one-time) compile. Disable it during optax loops.

---

## 6. Verification strategy

Add to `tests/test_jaxsim.py` (or a new file `tests/test_jaxsim_dropin.py`),
marked `@pytest.mark.integration` because it needs `LUSEE_DRIVE_DIR`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import lusee

@pytest.mark.integration
def test_jaxsim_sky_kwarg_grad_flows():
    """Drop-in: simulate(sky=..) must accept a traced sky and produce
    nonzero, sky-dependent gradients."""
    # Minimal two-frequency obs + Gaussian beams for fast compile.
    obs = lusee.Observation("2025-02-01 13:00 to 2025-02-01 15:00",
                            deltaT_sec=3600.0,
                            lun_lat_deg=-10.0, lun_long_deg=180.0)
    beams = [lusee.BeamGauss(id=str(i)) for i in range(2)]
    freq = np.array([30.0, 40.0])
    lmax = 16

    nside = 8
    maps = [np.random.RandomState(k).randn(12*nside**2) for k in (1, 2)]
    sky1 = lusee.sky.HealpixSky(nside, lmax, maps=maps, freq=freq,
                                frame="galactic")
    sky2 = lusee.sky.HealpixSky(
        nside, lmax,
        maps=[m + 1.0 for m in maps],      # perturb
        freq=freq, frame="galactic",
    )

    sim = lusee.JaxSimulator(obs, beams, sky1, Tground=0.0,
                             combinations=[(0,0),(1,1),(0,1)],
                             freq=freq, lmax=lmax)

    def loss_fn(sky):
        d = sim.simulate(sky=sky)        # must accept kwarg
        return jnp.sum(jnp.abs(d) ** 2)

    g1 = jax.grad(loss_fn)(sky1)
    g2 = jax.grad(loss_fn)(sky2)

    # Gradients wrt the mapalm leaf must be finite and nonzero for both.
    assert jnp.all(jnp.isfinite(g1.mapalm))
    assert jnp.all(jnp.isfinite(g2.mapalm))
    assert float(jnp.linalg.norm(g1.mapalm)) > 0.0
    assert float(jnp.linalg.norm(g2.mapalm)) > 0.0

    # And they must differ — otherwise the sky tracer was captured.
    assert not jnp.allclose(g1.mapalm, g2.mapalm)

    # Round-trip: passing the same sky twice gives bit-identical output
    # (no hidden randomness in the forward model).
    d1a = sim.simulate(sky=sky1)
    d1b = sim.simulate(sky=sky1)
    assert jnp.allclose(d1a, d1b)
```

Optionally add a second test that checks cache invariance:
```python
assert_cache_fingerprint_unchanged_after(simulate(sky=sky1), simulate(sky=sky2))
```
by comparing the mtime / hash of `extra_opts["cache_transform"]` file.

---

## 7. Caller change in `examples/optax_maxlike.py`

With the `sky=`/`beam=` kwargs implemented, `forward` is already generic:
```python
def forward(sim, sky):
    return sim.simulate(sky=sky)
```
The only edit needed is in `make_simulator`:
```python
-    sim = lusee.CroSimulator(
+    sim = lusee.JaxSimulator(
         obs, beams, dummy_sky,
         Tground=0.0, combinations=combinations,
         freq=freq, lmax=lmax,
     )
```
If `time_batch_size` tuning is desired, pass `extra_opts={"time_batch_size": N}`.
No other caller changes.

### Caveats for the optax example

- `examples/optax_maxlike.py` uses a `dummy_sky` of all ones for construction,
  relying on the fact that `CroSimulator` does not read `sky_model` until
  `simulate` runs. Under `JaxSimulator`, the constructor does **not** read
  sky alm either — only `self.freq_ndx_sky` is derived from `dummy_sky.freq`,
  which matches `sky_truth.freq`. So dummy-sky construction remains valid.
- The paper's noise/data path (`radiometric_sigma`, `forward(sim, sky_truth)`)
  uses `sim.combinations` and the output shape `(Nt, Nchannels, Nf)`. Both
  simulators produce the same layout, so no downstream reshuffling is needed.
- Confirm `LMAX=32` compiles inside GPU memory; otherwise set
  `extra_opts={"time_batch_size": 60}` or similar when constructing the
  simulator (this is a JaxSimulator-specific optimisation knob that
  CroSimulator does not expose — not a correctness issue).

---

## Summary of required changes (no code written this pass)

1. `lusee/JaxSimulator.py::simulate` — accept `sky=None, beam=None`,
   delegate to `sky_model.get_alm` on the passed pytree, and thread
   `output_beams_flm` / `output_ground` through the kernel calls.
2. `lusee/JaxSimulator.py::_beams_to_output_tensors` — small new helper to
   stack user-supplied `efbeams` into the flm/ground pair (for `beam=`).
3. `tests/test_jaxsim_dropin.py` — new integration test asserting
   sky-dependent gradient flow and round-trip determinism.
4. `examples/optax_maxlike.py::make_simulator` — change
   `lusee.CroSimulator` → `lusee.JaxSimulator` (one-line diff).
