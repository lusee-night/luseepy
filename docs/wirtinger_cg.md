# CG map-making with JAX autodiff adjoints

## The real-sky parameterization

The sky temperature is real, so `a_{l,-m} = (-1)^m conj(a_{l,m})`.
The CG variable is therefore real:

    theta = [Re(a_{lm}) for all (l,m);  Im(a_{lm}) for m > 0]

This eliminates the `Im(a_{l,0})` null space (always zero for a real
sky), reducing the condition number from ~10^12 to ~10^6.  With complex
alm, the 9 null-space modes at lmax=8 make CG non-convergent in float64.

### Forward model and adjoint

    theta (real) -> complex alm -> beam convolution (complex) -> data (real)

The beams are complex (E-field patterns, cross-correlations) but the
output is always real.  Since `A: R^n -> R^m`, its transpose `A^T` comes
directly from `jax.vjp` with no conjugation needed:

```python
def cg_matvec(theta):
    fwd, vjp_fn = jax.vjp(A, theta)
    return vjp_fn(N_inv * fwd)[0] + S_inv * theta
```

The normal operator `H = A^T N^{-1} A + S^{-1}` is real symmetric
positive-definite -- textbook CG (Shewchuk 1994).

### Diagonal preconditioner

`S^{-1} = 1/C_l` spans ~5 orders of magnitude, making H ill-conditioned
(kappa ~ 3e9 at lmax=32).  The preconditioner `M = diag(C_l)` gives
`MH = S A^T N^{-1} A + I` with eigenvalues in [1, 1+lambda_max].
Unobserved modes get eigenvalue 1 and converge immediately.

At lmax=32: Cholesky rho(1..10) = 0.9972, CG-500 = 0.9906.
See `notebooks/mapmaker_demo.ipynb` for a worked example.

### Direct solve vs CG

Camacho et al. (2026) use dense Cholesky, feasible at their resolution
(n ~ lmax^2 ~ 2300).  `method='direct'` in `solve()` builds the
Jacobian via `jax.jit(A)` and inverts H explicitly.  CG is motivated
by the multi-frequency case where n_modes x n_freq makes direct
inversion impractical.

### Wiener filter bias

The Wiener filter is optimally biased toward the prior mean, giving
rho_l < 1 even with a perfect solve.  The posterior covariance
D = H^{-1} satisfies `E[(s - s_hat)(s - s_hat)^T] = D` and
`rho_l^2 = 1 - D_l/C_l` in the diagonal approximation.
See Tegmark (1997) and Wandelt et al. (2004) for the general
CMB Wiener filter formalism.

The diagonal D approximation underestimates rho at high l because it
ignores off-diagonal mode coupling through partial sky coverage -- the
same pseudo-C_l coupling as in Hivon et al. (2002).  At lmax=32 l=24,
diagonal D predicts rho=0.75 but the full resolution matrix
R = I - D S^{-1} gives 0.94.

### Noise model

Per-sample radiometric noise following Camacho et al. (2026) Eq. 9:

    sigma^2_ij(t) = (T_ii T_jj + |V_ij|^2) / (2 df dt)

This gives a diagonal N with different weights per baseline and timestep.
The factor of 2 becomes 4 for the real/imaginary parts of cross-
correlations (Thompson, Moran & Swenson 2017, Eq. 6.44).

## Complex-variable case (Wirtinger CG)

If the CG variable is complex, JAX's VJP returns the unconjugated
transpose `A^T v` (Wirtinger convention), not the Hermitian adjoint
`A^H v`.  The fix is one line: `A_hermitian_v = jnp.conj(vjp_fn(v)[0])`.
The resulting operator is symmetric under `Re(x^H y)`, which is the
inner product JAX's CG uses (`_vdot_real_part`).

The real parameterization above is preferred -- it eliminates the null
space and avoids the Wirtinger subtlety entirely.  See Kreutz-Delgado
(2009) for the CR calculus background.

## References

- Camacho et al. (2026), arXiv:2508.16773 -- LuSEE-Night Wiener filter map-making
- Hivon et al. (2002), ApJ 567, 2 -- pseudo-C_l mode coupling (MASTER)
- Kreutz-Delgado (2009), arXiv:0906.4835 -- Wirtinger / CR calculus
- Shewchuk (1994), "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
- Tegmark (1997), ApJ 480, L87 -- CMB Wiener filter / optimal map-making
- Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio Astronomy", 3rd ed.
- Wandelt et al. (2004), PRD 70, 083511 -- exact CMB signal inference
