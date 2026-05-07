# Why `examples/optax_maxlike.py` diverges while `lusee/MapMaker.py` converges

Both scripts minimise the same Gaussian log-posterior from Camacho et al. 2026,

    L(m) = (d - A m)^T N^{-1} (d - A m)  +  m^T S^{-1} m

but they use different parameterisations of `m`. The optax script treats
`sky.mapalm` (a complex `(nfreq, nalm)` healpy array) as the optimisation
variable and runs Adam on it; `lusee.mapmaker.solve` first re-parameterises
the sky in real coordinates `θ = [Re(a_{lm}); Im(a_{l,m>0})]`. That single
choice explains most of the disaster. In order of severity:

---

## 1. Alm parameterisation: `jax.grad` on complex leaves ≠ real optimisation

The true sky is real, so in healpy packing

    a_{l,0} ∈ ℝ                  (lmax+1 real dof)
    a_{l,m>0} ∈ ℂ                (lmax(lmax+1)/2 complex ⇒ 2× real dof)
    Im(a_{l,0}) ≡ 0              (hard constraint)

The correct real dof count is `(lmax+1)² = n_m0 + 2 n_mpos`, which is what
`MapMaker._real_alm_indices` + `theta_to_alm` encode.

In the optax script, `sky.mapalm` is a `complex128` array of shape
`(nfreq, nalm)`. JAX treats every complex leaf as a single complex variable
using the Wirtinger convention; for a real-valued `L(z)`, `jax.grad` returns
`∂L/∂z̄` (i.e. a complex number whose real and imaginary parts are the
partials with respect to `Re(z)` and `Im(z)` respectively, up to a factor
of ½). Adam then applies independent moment estimates to the real and
imaginary parts and updates them both. Consequences:

- **`Im(a_{l,0})` is a free, unconstrained parameter.** Adam will happily
  move it around. A is invariant to it (it corresponds to an antisymmetric
  pattern on the sphere that is not a real temperature field), so its
  gradient from the `chi²` term is ~0 (see item 2). Only the prior
  `S^{-1}|a|²` keeps it small. Its equilibrium value is set by noise in
  the prior gradient, not by data.
- **The prior weight on the m>0 complex dof is wrong by a factor of 2.**
  See item 3 for the derivation.
- **The resulting recovered `mapalm` is not the alm of a real sky.** When
  you later call `hp.alm2map(rec_alm, Nside)`, healpy implicitly assumes
  `Im(a_{l,0}) = 0` and that `(Re,Im)(a_{l,m>0})` carry half the variance.
  If the Adam fit has spread power into `Im(a_{l,0})` and has the m>0
  variance mis-scaled, the resulting real map is wrong even before any
  Hessian / conditioning argument.

So yes, `Im(a_{l,0})` becomes a rogue dof exactly as warned about in
`docs/wirtinger_cg.md` ("complex alm ⇒ κ ~ 10¹²"). CG in MapMaker sidesteps
this by construction; Adam cannot.

---

## 2. Forward-model reality: does CroSimulator actually see `Im(a_{l,0})`?

`CroSimulator.simulate` passes `sky.mapalm` through
`s2fft.sampling.reindex.flm_hp_to_2d_fast`, which reconstructs the
negative-m half via `a_{l,-m} = (-1)^m conj(a_{l,m})`. In particular it
uses `a_{l,0}` for the m=0 slot directly — **including its imaginary
part**. The 2D alm is then rotated (gal→MEPA, time phases), and
`crojax.simulator.convolve` dots it into a complex beam alm. The final
real waterfall is `vis.real / (4π)`.

What does `Im(a_{l,0}) ≠ 0` do? It contributes an imaginary monopole slot
to the 2D alm. Under rotation and dotting with a (complex) real-sky beam
alm (which satisfies `b_{l,-m} = (-1)^m conj(b_{l,m})` only approximately
— beams here are E-field patterns, not sky temperatures), it produces a
generally non-zero imaginary part in `vis`, but that imaginary part is
**thrown away** by `vis.real`. So the contribution of `Im(a_{l,0})` to the
data is `Re(sum_{l} Im(a_{l,0}) · 1j · conj(b_{l,0}))` = `-sum_l Im(a_{l,0})
· Im(b_{l,0})`. This is *not* identically zero, so `Im(a_{l,0})` has a
small but non-zero effect on the data, and it has a non-zero gradient from
`chi²`. In other words the mode is not in the exact null space of
CroSimulator's concrete numerical implementation, but it is in the null
space of the *physical* forward model; the response is a numerical
artefact of packing a non-real healpy alm through the 2D round-trip.

Either way, its effective eigenvalue in `A^T N^{-1} A + S^{-1}` is
negligible compared to the prior (`1/C_l` is ~5e-11 at low l for T_sky ~
10⁴ K), so Adam sees a ~flat direction and drifts. With a 2D rotation
amplifying any imaginary piece over 325 timesteps, this drift can easily
dominate the residual. CG + the real parameterisation removes the mode
entirely.

---

## 3. Prior scaling: factor-of-2 mismatch on m>0

For an isotropic Gaussian field with angular power spectrum `C_l`, the
proper log-prior is

    -log p(m) = ½ m^T S^{-1} m

with `S^{-1}` acting on *real* coordinates. The statistics of the complex
alm are

    E[a_{l,0}²]       = C_l              (real)
    E[Re(a_{l,m>0})²] = E[Im(a_{l,m>0})²] = C_l / 2
    E[|a_{l,m>0}|²]   = C_l

So in **real** parameters the diagonal of `S^{-1}` is

    S^{-1}_{Re(a_{l,0})}   = 1/C_l
    S^{-1}_{Re(a_{l,m>0})} = S^{-1}_{Im(a_{l,m>0})} = 2/C_l    ← factor 2

This is exactly what MapMaker encodes:

```python
S_inv_re = S_inv.at[:, mpos_idx].multiply(2.0)  # 2/C_l on m>0 real parts
S_inv_im = 2.0 * S_inv[:, mpos_idx]             # 2/C_l on m>0 imag parts
```

The optax script writes the prior in complex form:

```python
prior = jnp.real(jnp.sum(jnp.conj(alm) * Kinv_a * S_inv_cl))
```

with `S_inv_cl = 1/C_l`. Expanding at the K_f = I limit:

    prior = Σ_{f,l,m≥0} (1/C_l) |a_{l,m}|²
          = Σ_{f,l}   (1/C_l) ( a_{l,0}²  +  Σ_{m>0} (Re² + Im²) )

Comparing with the log-prior of the real parameterisation

    -log p = ½ Σ [ (1/C_l) Re(a_{l,0})² + Σ_{m>0} (2/C_l)(Re² + Im²) ]
           = ½ Σ (1/C_l) a_{l,0}²  +  Σ_{l,m>0} (1/C_l)(Re² + Im²)

we see that the optax expression matches the m>0 part up to the overall
factor of ½ (which is the same factor the `chi²` term is missing, so it
cancels out in terms of the *ratio* between data and prior, except for the
m=0 modes). Specifically, **the optax prior over-weights `a_{l,0}` by a
factor of 2 relative to `a_{l,m>0}`, compared to the correct real
parameterisation.**

Correct complex-alm form (equivalent to the real expression):

    prior = Σ_{f,l}   (1/C_l) * ( a_{l,0}²   +  2 * Σ_{m>0} |a_{l,m}|² )

i.e. you need `S_inv_cl` that carries a factor of 2 on m>0 entries, or
equivalently weight the `|a_{l,m>0}|²` term by 2 in the sum.

This error is not catastrophic (the low-l monopole is strongly data-
constrained, so a mis-weighted prior there matters little), but it is
*wrong*, and in combination with items 1 and 6 it biases which modes Adam
prioritises.

---

## 4. Data / noise layout

`sigma` has shape `(ntimes, nchannels, nfreq)`. `N_inv = 1/sigma²` has the
same shape. `data` has the same shape. `forward(sim, sky)` returns
`sim.simulate(...)` which is also `(ntimes, nchannels, nfreq)`
(`_simulate_croissant_mepa` builds `wfall` with that ordering). So
`(data - forward).ravel()` and `N_inv.ravel()` use the same C-order
traversal. **No bug here.** (MapMaker does the same flatten; see
`N_inv.ravel()` + `data.ravel()`.)

---

## 5. Frequency-coupling prior (`K_f ⊗ diag(C_l)`)

With `FREQ_LOGKERNEL_SIGMA = 0.15` dex and frequencies `[20, 25, 30]` MHz,

    |Δlog f|    = log(25/20) ≈ 0.22 dex, log(30/20) ≈ 0.405 dex
    K_f         ≈ exp(-Δ² / 2σ²)
                  [[1.00, 0.34, 0.026],
                   [0.34, 1.00, 0.34 ],
                   [0.026,0.34, 1.00]]

which is benign — K_f is well-conditioned (cond ≈ 10-20), and `K_f_inv`
couples the three frequencies mildly. The paper does not include this
kernel (it works frequency-by-frequency); the optax script adds it as a
smoothness prior. It is not a correctness bug, but at 3 frequencies it
adds little and costs a lot of conditioning (the effective prior on the
difference-mode between 20 and 30 MHz is a factor ~(1 − 0.026)/0.026 ~ 37
stronger than the common mode), which further spreads the range of
eigenvalues Adam has to navigate. Recommendation: set `K_f = I` first to
get closer to MapMaker's regime, then reintroduce the log-freq kernel as a
refinement.

At `K_f = I`, `K_f_inv = I`, and the optax prior reduces to the naive
single-frequency `Σ (1/C_l) |a_{l,m}|²` form — still missing the m>0 factor
of 2 (item 3).

---

## 6. Optimiser scaling: Adam vs a 10⁵-condition-number quadratic

The quadratic `H = A^T N^{-1} A + S^{-1}` is dominated by `S^{-1} = 1/C_l`
on low-dof modes and by `A^T N^{-1} A` on data-constrained modes. `1/C_l`
spans ~5 decades from l=0 to l=lmax (C_0 ~ 10¹⁰, C_{32} ~ 10⁵). The
corresponding eigenvalues of `H` span the same range. The optimum of a
quadratic with condition κ requires on the order of √κ steps of gradient
descent with optimally tuned step size (Nesterov); Adam on raw variables
(no preconditioner) behaves similarly up to log factors. So √κ ≈ √(10⁵) ≈
300 steps would be the *lower bound* — but:

- Adam's per-coordinate step size is governed by the 2nd-moment estimate,
  which in a quadratic bowl settles to `~|grad|`, so effectively it
  renormalises each coordinate by its gradient scale. In a quadratic this
  is a rough Jacobi preconditioner, *if* the gradient is dominated by the
  diagonal Hessian. Here the data term is off-diagonal in the alm basis
  (beam mode coupling), so Jacobi-like preconditioning fails for
  data-constrained modes.
- `LR = 1e-1` on coordinates that are O(10⁵) K (the monopole) moves the
  monopole by ~10⁴ K per step but low-l dipoles by <1 K per step. That
  mismatch of dynamical ranges together with a loss of O(10¹³) means the
  gradient's dynamic range is enormous, and Adam's second-moment average
  cannot span it reliably within 20k iterations.

20 k Adam steps on a bare, unwhitened complex alm at LR=1e-1 is **not**
enough to reach MapMaker quality (`ρ ≈ 0.99`). You would need at minimum
10⁵–10⁶ steps, a much smaller LR on large-variance modes, or a
preconditioner.

---

## 7. Minimal fix: recommendation

Three options; I recommend **(c) whitening + real parameterisation**.

### (a) Port to the real-θ parameterisation but keep Adam

- Replace `params = zeros_like(sky_truth)` (complex mapalm) with a real
  array `theta` of shape `(nfreq, (lmax+1)²)` built from
  `MapMaker._real_alm_indices`.
- Inside `loss_fn`, call `theta_to_alm(theta)` and build a dummy
  `HealpixSky` from it via `tree_unflatten` (same trick as `solve`).
- Use `S_inv_real` as in MapMaker (`multiply(2.0)` on m>0).
- Drop the prior factor of 2 bug.

Pro: kills the null-space mode (item 1), fixes the factor of 2 (item 3),
keeps the optax/Adam loop. Con: Adam still faces κ ~ 10⁵, so it will still
need many iterations and careful LR tuning.

### (b) Project complex params and double prior weight on m>0

- Keep `params = mapalm` complex.
- After each `optimizer.update`, project: `mapalm[:, m0_idx] =
  Re(mapalm[:, m0_idx])` (zero imaginary monopole strip).
- Replace `S_inv_cl` with `S_inv_cl_weighted` that is `1/C_l` on m=0 and
  `2/C_l` on m>0 (and drop the `Im(a_{l,0})` term from the prior, or just
  accept that the projection enforces it).

This is the smallest code diff. Still leaves the Hessian conditioning
unchanged, so Adam convergence is still slow (item 6).

### (c) **Recommended: whiten + real parameterisation**

Use the real parameterisation from (a), *and* optimise in a whitened basis

    θ' = θ / sqrt(Σ_θ)      where  Σ_θ = diag(S) in real coords
                              i.e.  C_l   for Re(a_{l,0})
                                    C_l/2 for Re(a_{l,m>0}), Im(a_{l,m>0})

Then the prior term becomes `||θ'||²` (isotropic Gaussian in whitened
space), and the data term `chi²` has Hessian
`sqrt(Σ_θ)^T A^T N^{-1} A sqrt(Σ_θ)`, which has spectrum bounded by
`λ_max(A^T N^{-1} A) · max C_l` — i.e. the ~10⁵ dynamic range from C_l is
gone, and the problem looks ~isotropic except where data are
well-constrained (those modes have eigenvalue > 1). Adam on `θ'` with
LR ≈ 1e-2 should converge in ~10²–10³ iterations. After optimisation,
recover `θ = sqrt(Σ_θ) · θ'` and `alm = theta_to_alm(θ)`.

This is exactly the preconditioner used in `MapMaker.solve`'s CG path
(`precond_diag = 1 / S_inv_real`), expressed as a change of variable
instead of a preconditioner. For Adam, the change-of-variable is the
correct formulation (optax has no first-class preconditioner API).

**Concrete patch outline** (pseudo-diff against `examples/optax_maxlike.py`,
no code changes made on disk):

```python
from lusee.MapMaker import _real_alm_indices

m0_idx, mpos_idx = _real_alm_indices(LMAX)
nalm = sky_truth.mapalm.shape[-1]
n_mpos = len(mpos_idx)

# Real S^{-1} diagonal (nfreq, nalm + n_mpos)
S_inv_cl = cl_inv_from_sky(sky_truth, LMAX)                 # (nfreq, nalm)
S_inv_re = S_inv_cl.at[:, mpos_idx].multiply(2.0)
S_inv_im = 2.0 * S_inv_cl[:, mpos_idx]
S_inv_real = jnp.concatenate([S_inv_re, S_inv_im], axis=-1) # (nfreq, n_theta)

# Whitening scales: sqrt(C_l in real coords) = 1 / sqrt(S_inv_real)
sqrt_S = 1.0 / jnp.sqrt(jnp.maximum(S_inv_real, 1e-30))

def theta_to_alm(theta):                                    # θ = (nfreq, n_theta)
    re = theta[:, :nalm]
    im_mpos = theta[:, nalm:]
    im_full = jnp.zeros_like(re).at[:, mpos_idx].set(im_mpos)
    return re + 1j * im_full

sky_aux = sky_truth.tree_flatten()[1]
def make_sky(alm):
    return lusee.sky.HealpixSky.tree_unflatten(sky_aux, (alm,))

def loss_fn(theta_prime):
    theta = sqrt_S * theta_prime                            # un-whiten
    alm   = theta_to_alm(theta)
    r = (data - sim.simulate(sky=make_sky(alm))).ravel()
    chi2 = jnp.sum(N_inv.ravel() * r ** 2)
    prior = jnp.sum(theta_prime ** 2)                       # isotropic in whitened basis
    # (drop K_f for the first pass; add back once convergence works)
    return chi2 + prior

theta_prime0 = jnp.zeros((nfreq, nalm + n_mpos))
```

Optimiser: `optax.adam(1e-2)` (was `1e-1`), `N_ITERS = 2000` should already
get close; 10 k for safety. Monitor `loss` and `ρ(1..10)` — they should
track mapmaker_demo's number (~0.99) within a few percent.

Dropping K_f simplifies debugging: verify that at K_f = I the optax solver
reaches MapMaker's `ρ ~ 0.99`; only then reintroduce a freq prior (and if
you do, fold it into the whitening: `Σ_θ ← C_l · K_f`, which just means
using the Cholesky of `K_f` in the freq axis).

---

## TL;DR

The three things that must change in `examples/optax_maxlike.py`:

1. **Switch to a real parameterisation** θ = [Re(a_{l,m}); Im(a_{l,m>0})]
   (eliminates `Im(a_{l,0})` rogue mode).
2. **Fix the prior by a factor of 2 on m>0 entries** (correct Gaussian
   density in real coords).
3. **Whiten by √C_l** (precondition Adam; reduces κ by ~10⁵).

Items 1+2 move you from "wildly wrong" to "correct but slow to converge".
Item 3 is what lets Adam reach `ρ ~ 0.99` in a reasonable wall-clock.
Until all three are in place, CG remains the right tool and Adam on raw
complex mapalm is the wrong tool.
