"""Toy test of the gradient-based sampler against analytic Gaussian posterior.

We bypass CroSimulator by constructing ops dict directly: a random linear
forward map A with Gaussian data/prior gives a known mean (Wiener filter) and
covariance. The NUTS chain should recover both within MC error.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest


def _build_toy_ops(ndata, n_theta, nfreq=1, seed=0):
    rng = np.random.default_rng(seed)
    J = jnp.asarray(rng.standard_normal((ndata, n_theta)) / np.sqrt(n_theta))
    sigma = 0.1
    N_inv = 1.0 / sigma**2
    # Prior: variances geometrically spaced 1 .. 1e-3
    S_diag = jnp.asarray(np.logspace(0, -3, n_theta))
    S_inv_real = (1.0 / S_diag).reshape(nfreq, n_theta)
    true_theta = jnp.asarray(rng.standard_normal((nfreq, n_theta))) * jnp.sqrt(S_diag)
    data = (J @ true_theta[0]) + sigma * jnp.asarray(rng.standard_normal(ndata))

    def A(theta):
        return (J @ theta[0]).ravel()

    def theta_to_alm(theta):
        return theta.astype(jnp.complex128)

    zero = jnp.zeros((nfreq, n_theta))
    _, vjp = jax.vjp(A, zero)
    rhs = vjp(N_inv * data.ravel())[0]
    ops = dict(
        A=A, theta_to_alm=theta_to_alm, N_inv=N_inv, S_inv_real=S_inv_real,
        rhs=rhs, zero=zero, n_theta=n_theta, nfreq=nfreq, nalm=n_theta,
        data_flat=data.ravel(),
    )
    # Analytic mean/covariance
    H = J.T @ J * N_inv + jnp.diag(S_inv_real.ravel())
    cov = jnp.linalg.inv(H)
    mean = cov @ rhs.ravel()
    return ops, np.asarray(mean), np.asarray(cov)


def test_nuts_matches_analytic_gaussian():
    from lusee.Sampler import _sample_from_ops

    ops, true_mean, true_cov = _build_toy_ops(ndata=80, n_theta=12, seed=1)
    # Warm-start at analytic mean to avoid burn-in pollution
    init = jnp.asarray(true_mean).reshape(1, -1)
    samples = _sample_from_ops(
        ops, num_samples=2000, num_warmup=500,
        seed=42, init_theta=init, target_accept=0.8, return_info=False,
    )
    samples_flat = np.asarray(samples).reshape(samples.shape[0], -1)
    est_mean = samples_flat.mean(axis=0)
    est_std = samples_flat.std(axis=0)
    true_std = np.sqrt(np.diag(true_cov))

    mean_err = np.abs(est_mean - true_mean) / true_std
    assert mean_err.max() < 1.5, f"mean off by {mean_err.max():.2f} sigma"

    rel_err = np.abs(est_std - true_std) / true_std
    assert rel_err.max() < 0.35, f"std relerr {rel_err.max():.2f}"


def test_constrained_realization_matches_analytic_gaussian():
    """CR sampler: each draw is independent + exact, so convergence is O(1/sqrt(N))."""
    import jax.scipy.sparse.linalg as jsl

    ops, true_mean, true_cov = _build_toy_ops(ndata=80, n_theta=12, seed=2)
    # Hand-wired CR loop so we don't depend on a CroSimulator object
    A = ops["A"]; N_inv = ops["N_inv"]; S_inv = ops["S_inv_real"]
    zero = ops["zero"]; d = ops["data_flat"]
    sigma_flat = 1.0 / jnp.sqrt(N_inv) * jnp.ones_like(d)
    prior_sqrt = jnp.sqrt(S_inv)

    def matvec(x):
        fwd, vjp = jax.vjp(A, x)
        return vjp(N_inv * fwd)[0] + S_inv * x

    key = jax.random.PRNGKey(9)
    N = 400
    samples = []
    for _ in range(N):
        key, k1, k2 = jax.random.split(key, 3)
        eta = sigma_flat * jax.random.normal(k1, d.shape)
        xi = prior_sqrt * jax.random.normal(k2, zero.shape)
        _, vjp = jax.vjp(A, zero)
        rhs = vjp(N_inv * (d + eta))[0] + xi
        x, _ = jsl.cg(matvec, rhs, x0=zero, maxiter=1000, tol=1e-12)
        samples.append(x)
    samp = np.asarray(jnp.stack(samples)).reshape(N, -1)

    est_mean = samp.mean(axis=0)
    est_std = samp.std(axis=0)
    true_std = np.sqrt(np.diag(true_cov))

    # CR draws are independent -> MC error = sigma/sqrt(N)
    z_mean = np.abs(est_mean - true_mean) / (true_std / np.sqrt(N))
    assert z_mean.max() < 4.0, f"CR mean off by {z_mean.max():.2f} sigma"

    # Std: chi2 sampling -> relative error ~ sqrt(2/N)
    rel_err = np.abs(est_std - true_std) / true_std
    assert rel_err.max() < 0.15, f"CR std relerr {rel_err.max():.2f}"


if __name__ == "__main__":
    test_nuts_matches_analytic_gaussian()
    test_constrained_realization_matches_analytic_gaussian()
    print("OK")
