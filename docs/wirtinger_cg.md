# Autodiff adjoints for CG map-making

The map-maker uses a real parameterization `theta = [Re(a_lm); Im(a_lm, m>0)]`,
which eliminates the `Im(a_{l,0})` null space and makes `H = A^T N^{-1} A + S^{-1}`
real symmetric positive-definite. JAX's `vjp` gives `A^T` directly -- no
conjugation needed. A diagonal preconditioner `M = diag(C_l)` is used to
equalize the ~5 orders of magnitude in `S^{-1}`. See `notebooks/mapmaker_demo.ipynb`.

## Complex-variable case (Wirtinger CG)

If the CG variable were complex (e.g. complex beam unknowns), the
Wirtinger convention in JAX matters. For `f: C^n -> R^m`, JAX's VJP
returns the unconjugated transpose `A^T v`, not the Hermitian adjoint
`A^H v`. To solve the normal equations `(A^H N^{-1} A + S^{-1}) s = A^H N^{-1} d`:

```python
A_hermitian_v = jnp.conj(vjp_fn(v)[0])
```

The operator `conj(vjp) o A` is symmetric under the real inner product
`Re(x^H y)`, which is what JAX's CG uses internally (`_vdot_real_part`).
The steepest descent direction for a real-valued function of complex
variables is the conjugate Wirtinger gradient df/dz* (Brandwood 1983),
and CG in complex variables uses the real inner product Re(x^H y)
(Sorber et al. 2012).  JAX returns df/dz (unconjugated), so an explicit
conjugation is needed.  PyTorch and TensorFlow return df/dz* directly --
this subtlety is JAX-specific.

With complex alm, the `Im(a_{l,0})` modes sit in the null space of A
with `S^{-1} ~ 5e-11` regularization, creating kappa ~ 10^12. Options:
add a penalty `lambda * Im(x) * m0_mask`, use direct Cholesky, or
(preferred) switch to the real parameterization.

## References

- Brandwood (1983), IEE Proc. F 130(1), 11-16 -- complex gradient operator (original steepest descent proof)
- Camacho et al. (2026), arXiv:2508.16773 -- LuSEE-Night Wiener filter map-making
- Hivon et al. (2002), ApJ 567, 2 -- pseudo-C_l mode coupling (MASTER)
- Kreutz-Delgado (2009), arXiv:0906.4835 -- CR calculus tutorial
- Shewchuk (1994), "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain"
- Sorber, van Barel & De Lathauwer (2012), SIAM J. Optim. 22(3), 879-898 -- complex CG via Wirtinger calculus
- Tegmark (1997), ApJ 480, L87 -- CMB Wiener filter / optimal map-making
- Thompson, Moran & Swenson (2017), "Interferometry and Synthesis in Radio Astronomy", 3rd ed.
- Wandelt et al. (2004), PRD 70, 083511 -- exact CMB signal inference
