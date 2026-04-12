# Complex CG with JAX autodiff: the conjugation rule

## The problem

We want to solve the normal equations via CG:

    (A^H N^{-1} A + mu I) s = A^H N^{-1} d

where `A` maps complex sky alm to a real waterfall, and `A^H` is its
Hermitian adjoint.  JAX gives us `jax.vjp` for free — but does it give
`A^H`?

**No.**  JAX's VJP uses the Wirtinger convention.  For a function
`f: C^n -> R^m`, `vjp(v)` returns the **unconjugated transpose** `A^T v`,
not the **Hermitian adjoint** `A^H v`.

## Why this happens

JAX defines derivatives of real-valued functions of complex variables via
Wirtinger calculus.  For `f: C -> R`:

    jax.grad(f)(z) = df/dz-bar = conj(df/dz)

The VJP generalises this: for `f: C^n -> R^m` with real cotangent `v`,

    vjp(v) = (df/dz-bar)^T v

In matrix notation, if the real Jacobian is `J_r = df/d(Re z)` and
`J_i = df/d(Im z)`, then:

    vjp(v) = (1/2)(J_r + i J_i)^T v    [Wirtinger]
    A^H v  = (J_r + i J_i)^T v         [Hermitian adjoint, for real v]

Wait — those differ by a factor of 2?  Not quite.  The actual relationship,
for **real** cotangent `v`, is:

    vjp(v) = A^T v     (unconjugated transpose)
    A^H v  = conj(A^T v) = conj(vjp(v))

This is because `A^T` and `A^H` differ by conjugation:
`(A^H)_ij = conj(A_ji) = conj((A^T)_ij)`, and for real `v`,
`conj(A^T v) = conj(A)^T v = A^H v`.

## The fix

One line:

```python
def A_hermitian(v, x):
    _, vjp_fn = jax.vjp(A, x)
    return jnp.conj(vjp_fn(v)[0])
```

## Which inner product?

Standard complex CG uses `<x, y> = x^H y` and requires the operator to be
Hermitian: `M^H = M`.  But `M = A^H N^{-1} A` built from `conj(vjp)` is
Hermitian under the **real inner product** `Re(x^H y)`, not the full complex
inner product.

Specifically:

    Re(x^H M y) = Re(y^H M x)     [symmetric under Re]
    x^H M x  is real and > 0       [positive-definite]

This is enough for CG — we just replace all dot products with
`Re(x^H y)`:

```python
def vdot_real(x, y):
    return jnp.vdot(x.real, y.real) + jnp.vdot(x.imag, y.imag)
```

This is exactly the inner product JAX's built-in
`jax.scipy.sparse.linalg.cg` uses internally.  In `_cg_solve`
(`jax/_src/scipy/sparse/linalg.py`), every dot product goes through
`_vdot_real_tree` → `_vdot_real_part`:

```python
def _vdot_real_part(x, y):
  """Vector dot-product guaranteed to have a real valued result despite
     possibly complex input."""
  result = _vdot(x.real, y.real)
  if jnp.iscomplexobj(x) or jnp.iscomplexobj(y):
    result += _vdot(x.imag, y.imag)
  return result
```

This means `jax.scipy.sparse.linalg.cg` already handles our case
correctly — pass it `conj(vjp)`-based matvec and complex vectors, and it
just works.  No manual CG needed.

## Why "real-linear, not complex-linear"

The forward model `A` maps complex alm to a real waterfall.  It is
**real-linear** (`A(alpha x) = alpha A(x)` for real `alpha`) but not
**complex-linear** (`A(i x) != i A(x)` because the output is always real).

This is not a limitation or a hack — it reflects the physics:

- The **sky** is real: `a_{l,-m} = (-1)^m conj(a_{l,m})`
- **Autocorrelations** are real
- **Cross-correlations** are split into real/imaginary waterfall channels

A complex-linear operator would have a Hermitian normal form under the
standard inner product.  A real-linear operator has a symmetric normal form
under the real inner product.  CG works in both cases — you just need the
matching inner product.

## Other frameworks

PyTorch and TensorFlow use the **conjugate** Wirtinger convention:
`grad(f)(z) = df/dz-bar-bar = df/dz`.  Their VJP directly returns `A^H v`,
so no conjugation is needed.  This is a JAX-specific issue.

## Summary

| What you want | What JAX gives | Fix |
|---|---|---|
| `A^H v` (Hermitian adjoint) | `A^T v` (unconjugated transpose) | `conj(vjp(v))` |
| Symmetric CG operator | Symmetric under `Re(x^H y)` | Use `vdot_real` inner product |

The recipe for CG with JAX autodiff adjoints:

```python
# Normal equations operator — the conj() is the only non-obvious part
def matvec(x):
    fwd, vjp_fn = jax.vjp(A, x)
    return jnp.conj(vjp_fn(N_inv * fwd)[0]) + mu * x

# RHS
_, vjp_rhs = jax.vjp(A, x0)
rhs = jnp.conj(vjp_rhs(N_inv * data)[0])

# Solve — JAX's CG already uses Re(x^H y) internally
sky_hat, info = jax.scipy.sparse.linalg.cg(matvec, rhs, x0=x0)
```
