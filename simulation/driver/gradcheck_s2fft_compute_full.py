#!/usr/bin/env python3
"""Toy differentiability check for s2fft.utils.rotation.compute_full.

This script checks gradients with respect to beta:
1) Directly through a loss built only from compute_full outputs.
2) Through a toy Wigner-d based rotation pipeline.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def configure_x64() -> None:
    # Default to x64 for cleaner finite-difference checks.
    enabled = _as_bool(os.environ.get("JAX_ENABLE_X64", "1"))
    os.environ["JAX_ENABLE_X64"] = "1" if enabled else "0"
    jax.config.update("jax_enable_x64", enabled)
    print(f"JAX x64 enabled: {enabled}")


def build_dls(beta: jnp.ndarray, L: int) -> jnp.ndarray:
    """Build stacked Wigner-d blocks using compute_full recurrence."""
    from s2fft.utils.rotation import compute_full

    beta = jnp.asarray(beta)
    M = 2 * L - 1
    dls = jnp.zeros((L, M, M), dtype=beta.dtype)
    dl_iter = jnp.zeros((M, M), dtype=beta.dtype)
    for ell in range(L):
        dl_iter = compute_full(dl_iter, beta, L, ell)
        dls = dls.at[ell].set(dl_iter)
    return dls


def fd_central(f, x: float, eps: float = 1e-6) -> float:
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def rel_err(a: float, b: float) -> float:
    denom = max(1.0, abs(a), abs(b))
    return abs(a - b) / denom


def run_direct_check(L: int, beta0: float) -> tuple[float, float, float]:
    rng = np.random.default_rng(1)
    M = 2 * L - 1
    weights = jnp.asarray(rng.normal(size=(L, M, M)))

    def loss(beta):
        dls = build_dls(beta, L)
        return jnp.sum(dls * weights)

    grad_ad = float(jax.grad(loss)(beta0))
    grad_fd = fd_central(lambda x: float(loss(x)), beta0)
    return grad_ad, grad_fd, rel_err(grad_ad, grad_fd)


def run_rotation_check(L: int, beta0: float) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    M = 2 * L - 1
    m_offset = L - 1
    m_all = jnp.arange(-m_offset, m_offset + 1)

    # Valid m-mask for each ell.
    valid = np.zeros((L, M), dtype=np.float64)
    for ell in range(L):
        valid[ell, m_offset - ell : m_offset + ell + 1] = 1.0
    valid = jnp.asarray(valid)

    flm_real = jnp.asarray(rng.normal(size=(L, M)))
    flm_imag = jnp.asarray(rng.normal(size=(L, M)))
    flm = (flm_real + 1j * flm_imag) * valid
    target_real = jnp.asarray(rng.normal(size=(L, M)))
    target_imag = jnp.asarray(rng.normal(size=(L, M)))
    target = (target_real + 1j * target_imag) * valid

    alpha = 0.37
    gamma = -0.22

    def loss(beta):
        dls = build_dls(beta, L)
        exp_alpha = jnp.exp(-1j * m_all * alpha)
        exp_gamma = jnp.exp(-1j * m_all * gamma)
        flm_weighted = flm * exp_gamma[None, :] * valid
        flm_rot = jnp.einsum("lmn,ln->lm", dls, flm_weighted)
        flm_rot = flm_rot * exp_alpha[None, :] * valid
        return jnp.real(jnp.vdot(target, flm_rot))

    grad_ad = float(jax.grad(loss)(beta0))
    grad_fd = fd_central(lambda x: float(loss(x)), beta0)
    return grad_ad, grad_fd, rel_err(grad_ad, grad_fd)


def main() -> int:
    configure_x64()
    L = int(os.environ.get("L", "8"))
    beta0 = float(os.environ.get("BETA0", "0.9"))
    tol = float(os.environ.get("REL_TOL", "1e-4"))

    print(f"Settings: L={L}, beta0={beta0}, rel_tol={tol}")

    d_ad, d_fd, d_re = run_direct_check(L, beta0)
    print("\nDirect compute_full check:")
    print(f"  grad_autodiff = {d_ad:.12e}")
    print(f"  grad_finitediff = {d_fd:.12e}")
    print(f"  rel_error = {d_re:.3e}")

    r_ad, r_fd, r_re = run_rotation_check(L, beta0)
    print("\nToy rotation pipeline check:")
    print(f"  grad_autodiff = {r_ad:.12e}")
    print(f"  grad_finitediff = {r_fd:.12e}")
    print(f"  rel_error = {r_re:.3e}")

    ok = np.isfinite([d_ad, d_fd, r_ad, r_fd]).all() and d_re < tol and r_re < tol
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
