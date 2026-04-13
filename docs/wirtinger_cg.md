# CG map-making with JAX autodiff adjoints

## The real-sky parameterization (recommended)

The sky temperature is real, so its spherical harmonic coefficients satisfy
`a_{l,-m} = (-1)^m conj(a_{l,m})`.  Healpy stores only m >= 0, with
`a_{l,0}` real and `a_{l,m>0}` complex.

The natural CG variable is therefore real:

    θ = [Re(a_{lm}) for all (l,m);  Im(a_{lm}) for m > 0]

This has `nalm + n_{m>0}` real degrees of freedom (81 for lmax=8,
vs 90 if we used full complex alm).  The `Im(a_{l,0})` entries are
excluded — they are always zero for a real sky and lie in the exact
null space of the forward model.

### Why this matters

With complex alm as the CG variable, the 9 null-space modes
`Im(a_{l,0})` create condition numbers of ~10^12 (the prior S^{-1}
barely regularizes them).  CG cannot converge in float64 at this
condition number.  The real parameterization eliminates the null
space entirely, reducing κ to ~10^6.

### The forward model

The forward model maps real θ through complex beam convolution to
real data:

    θ (real) → complex alm → beam convolution (complex) → data (real)

The beam alm are complex (cross-correlations, E-field patterns),
but the output is always real (auto-correlations are real,
cross-correlations are split into separate real/imaginary channels).

### The adjoint

Since θ is real and the data is real, the forward model
`A: R^n -> R^m` is a standard real-linear map.  Its transpose
`A^T` is obtained directly from `jax.vjp`:

```python
def cg_matvec(theta):
    fwd, vjp_fn = jax.vjp(A, theta)
    return vjp_fn(N_inv * fwd)[0] + S_inv * theta
```

No conjugation is needed — `vjp` gives `A^T` for real inputs.
JAX traces through all the complex beam math internally; the
gradients land on the real θ components automatically.

### CG inner product

With real θ, `jax.scipy.sparse.linalg.cg` uses the standard real
dot product `<x, y> = sum(x * y)`.  The normal operator
`M = A^T N^{-1} A + S^{-1}` is real symmetric positive-definite,
which is the textbook CG setting.

### Diagonal preconditioner

The normal matrix `H = A^T N^{-1} A + S^{-1}` is ill-conditioned
because `S^{-1} = 1/C_l` spans ~5 orders of magnitude (C_0 ~ 10^10
vs C_{lmax} ~ 10^5).  Without preconditioning, CG can appear to
converge (small residual) while the solution is far from the true
optimum — producing spurious artifacts (negative holes in the
unobserved region).

The preconditioner `M = diag(C_l)` (signal covariance diagonal
in the real parameterization) fixes this.  The preconditioned
operator `M H = S(A^T N^{-1} A + S^{-1}) = S A^T N^{-1} A + I`
has eigenvalues in `[1, 1 + λ_max]`, where λ_max comes from the
data term.  Critically, modes in the null space of A (unobserved
by the data) get eigenvalue exactly 1 and converge immediately,
rather than being the slowest-converging modes.

At lmax=16, preconditioned CG-500 achieves ρ(1..10) = 0.9923 vs
0.8548 without preconditioning.  The Cholesky ground truth is 0.9974.

At lmax=32 with radiometric noise (κ ≈ 3×10⁹, rank 800/1089):
- Cholesky (exact): ρ(1..10) = 0.9972
- CG-1000 + S-precond: ρ(1..10) = 0.9937
- CG-500 + S-precond: ρ(1..10) = 0.9906
CG is close at low l but degrades at l > 15 where modes are only
partially observed.  The Jacobian has rank 800 out of 1089 real
parameters, meaning 289 modes are unobserved by the data.

### Direct solve vs CG

Camacho et al. 2026 use direct matrix inversion (dense Cholesky),
not CG.  At lmax=47, the map vector has N_modes ~ lmax² ~ 2209
parameters, which is small enough for brute-force inversion.  They
explicitly form and invert (A^T N^{-1} A + S^{-1})^{-1}.

Our CG approach is motivated by the multi-frequency case, where
joint reconstruction across all frequency channels makes the map
vector too large for direct inversion (N_modes × N_freq parameters).
The autodiff adjoint (jax.vjp) avoids ever forming the Jacobian or
normal matrix, so the cost is O(maxiter × forward_eval) regardless
of matrix size.

For single-frequency problems (N_modes ~ 10³), the direct solve is
feasible and gives the exact answer with no CG convergence error.
The notebook `notebooks/mapmaker_demo.ipynb` demonstrates CG at lmax=32.

The Jacobian build in `_solve_direct` uses `jax.jit(A)` to
JIT-compile the forward model.  This traces through CroSimulator
once (compiling Python loops, beam algebra, and waterfall assembly
into a single XLA kernel), then dispatches each column as a fast
compiled call.  Timings on a GTX 1660 (6 GB):

    lmax=8  (n=81):   29s total  (compile 28s + build 1s,  14ms/col)
    lmax=32 (n=1089): 282s total (compile ~70s + build ~205s)
    lmax=47 (n=2304): 1029s total (compile ~900s + build ~130s, ~56ms/col)

At lmax=47 (paper resolution), XLA compilation dominates because
`s2fft.utils.rotation.rotate_flms` uses a Python `for el in range(L)`
loop that gets unrolled into 48 copies of the loop body.  The
resulting XLA graph is O(L³) in size, leading to ~15 min compile
at L=48.  If s2fft used `lax.fori_loop`, this would be much faster.

The lmax=47 result (rho(1..10) = 0.9972, 1029s) matches the paper's
exact method at full resolution.

### Wiener filter bias and ρ_l

The Wiener filter is optimally biased — it shrinks the estimate
toward zero (the prior mean) to minimize total MSE.  The
theoretical cross-correlation coefficient is:

    ρ_l = sqrt(λ_l C_l / (1 + λ_l C_l))

where λ_l is the effective eigenvalue of A^T N^{-1} A at multipole
l (averaged over m), and C_l is the signal prior power spectrum.
This equals sqrt(C_l / (C_l + C_l^{noise,eff})) where
C_l^{noise,eff} = 1/λ_l encodes beam and scanning strategy.

Key identities for the exact Wiener filter (from R = I - DS^{-1},
D = (A^T N^{-1} A + S^{-1})^{-1}):

- Resolution: s_hat = Rs + Wn
- Cross-covariance: E[s_hat s^T] = S - D
- Auto-covariance: E[s_hat s_hat^T] = S - D  (same!)
- Power spectrum bias: E[C_l^hat] / C_l = ρ_l²
- Posterior error: E[(s - s_hat)(s - s_hat)^T] = D

So ρ_l < 1 is expected even with a perfect solve.  Higher lmax
means more poorly-constrained high-l modes, which can contaminate
low-l estimates through beam-induced mode coupling — this is why
lmax=32 gives ρ(1..10) ~ 0.99 vs the paper's ρ > 0.9 at lmax=47.

Per-l ρ from the direct solve at lmax=32 (25 MHz, 28-day obs):

    l=0-9:   ρ > 0.996   (well-constrained by data)
    l=10-13: ρ ~ 0.98-0.99
    l=14-28: ρ ~ 0.87-0.94  (partial sky coverage plateau)
    l=29-32: ρ ~ 0.59-0.81  (near lmax cutoff)

The break at l~14 is set by the sky fraction observable over one
lunar day (~30-40% of the sky), not the beam angular resolution.
At each high l, roughly (1 - f_sky) of the (2l+1) m-modes are in
or near the null space of A, giving a plateau rather than a sharp
cutoff.  The Jacobian rank is 800/1089 (289 null-space modes).

### Mode coupling and the resolution matrix

The diagonal posterior variance approximation ρ²_l = 1 - D_l/C_l
(summing diag(D) per l) significantly underestimates reconstruction
quality at high l.  The resolution matrix R = I - D S^{-1} has
off-diagonal elements that couple different (l,m) modes through
partial sky coverage — the same physics as pseudo-C_l mode coupling
in CMB analysis (Hivon et al. 2002).

Four ρ_l measures at lmax=32:

1. **diag(D)**: ρ²_l = 1 - tr(D_l)/tr(S_l), ignoring off-diagonal D
2. **Resolution R**: ρ from E_n[θ_hat] = R θ_true (noiseless, full R)
3. **MC mean**: average ρ over 30 noise realizations
4. **Empirical**: ρ from a single noise realization

    l     diag(D)  resol R  MC mean  MC_std
    1-9:   0.997    0.999    0.999   0.001
    14:    0.919    0.893    0.889   0.008
    20:    0.797    0.910    0.891   0.009
    24:    0.748    0.938    0.899   0.015
    32:    0.512    0.810    0.746   0.037

The mode coupling effect is dramatic at high l.  At l=24, the
diagonal D predicts ρ=0.75 but the resolution matrix shows ρ=0.94
(25% improvement from mode coupling).  At l=32: diagonal 0.51 vs
resolution 0.81 (58% improvement).

The resolution ρ predicts the MC mean closely at low l (where noise
is negligible) and provides an upper bound at high l (where noise
degrades the reconstruction).  The MC mean is 0.998 ± 0.001 for
l=1-10, matching the resolution ρ and confirming the diagonal D
(0.997) is a good but slightly conservative predictor at low l.

Physically, the off-diagonal R elements allow information from
well-constrained low-m modes at each l to partially reconstruct
the poorly-constrained high-m modes through the shared data
contributions.  This is sky-realization-dependent: the cross-terms
in the power spectrum involve phases between a_{lm} and a_{l'm'}.

### Noise covariance

The paper (Eq. 9) uses per-sample radiometric noise:

    σ²_ij(t) = (T_ii(t) T_jj(t) + |V_ij(t)|²) / (2 Δf Δt)

with Δf = 1 MHz, Δt = 7200 s.  This gives diagonal N with different
weights per baseline and time step.  Auto-correlations get σ ≈ 0.5 K,
cross-correlations get σ ≈ 0.1–0.3 K at 25 MHz (T_sky ~ 40,000 K).

For each real data element:
- Auto V_ii (real): σ² = T_ii² / (Δf Δt)
- Cross Re/Im (each): σ² = (T_ii T_jj + |V_ij|²) / (4 Δf Δt)

The factor of 4 (not 2) for cross Re/Im arises because the paper's
formula gives the complex visibility variance; each real component
gets half (Thompson, Moran & Swenson 2017, Eq. 6.44).

`compute_radiometric_noise()` estimates T_ii(t) from the data itself,
which is accurate at SNR >> 1.

## The complex-variable case (Wirtinger CG)

If the CG variable is complex (e.g., for a complex sky model or
when working directly with healpy's complex alm storage), the
Wirtinger convention matters.

### The problem

We want to solve:

    (A^H N^{-1} A + S^{-1}) s = A^H N^{-1} d

where `A` maps complex alm to real data, and `A^H` is the Hermitian
adjoint.  JAX's VJP uses the Wirtinger convention: for `f: C^n -> R^m`,
`vjp(v)` returns `A^T v` (unconjugated transpose), not `A^H v`.

### The fix

One line:

```python
A_hermitian_v = jnp.conj(vjp_fn(v)[0])
```

### Which inner product?

The operator `M = conj(vjp) ∘ A` is symmetric under the **real inner
product** `Re(x^H y)`, not the full complex inner product.  This is
exactly the inner product JAX's CG uses internally (`_vdot_real_part`),
so `jax.scipy.sparse.linalg.cg` works directly.

### The null-space problem

With complex alm, the `Im(a_{l,0})` modes are in the null space of A
(they don't affect the real sky).  The prior `S^{-1} ~ 1/C_l` barely
regularizes them (`S^{-1}_0 ~ 5e-11` for a 38,000 K monopole), creating
condition numbers of ~10^12.  Options:

1. **Real parameterization** (recommended): eliminate the null space
   by construction — see above.
2. **Penalty**: add `λ * 1j * Im(x) * m0_mask` to the matvec, which
   puts eigenvalue λ on the null-space modes.
3. **Direct solve**: build the matrix via `jax.jacfwd` and use Cholesky.
   Feasible for nalm < ~10^4.

## Other frameworks

PyTorch and TensorFlow use the **conjugate** Wirtinger convention:
`grad(f)(z) = df/dz`.  Their VJP directly returns `A^H v`, so no
conjugation is needed.  The Wirtinger issue is JAX-specific.

## Summary

| Scenario | CG variable | Adjoint | Inner product | Null space |
|---|---|---|---|---|
| Real sky (recommended) | real θ | `vjp` (= A^T) | standard real | eliminated |
| Complex sky / alm | complex x | `conj(vjp)` (= A^H) | `Re(x^H y)` | needs penalty or direct solve |
