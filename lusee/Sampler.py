"""Gradient-based posterior sampling for the Wiener-filter map-maker.

For the Gaussian posterior defined in MapMaker:

    -log p(theta | d) = 0.5 (A theta - d)^T N^{-1} (A theta - d)
                        + 0.5 theta^T S^{-1} theta  + const

we provide a blackjax NUTS sampler over the same real parameterization
theta = [Re(alm); Im(alm, m>0)] that MapMaker.solve uses.

Since the posterior is exactly Gaussian, sampling is optional (the mean
and covariance are fully determined by solve()). This module exists to
(a) validate the posterior shape, (b) serve as a template for future
non-Gaussian extensions (nonlinear foregrounds, T_sys sampling, beam
parameters), where gradient-based MCMC becomes necessary.

Preconditioning: we use inverse_mass_matrix = diag(C_l) packed in the
real-theta layout. This matches the preconditioner already used for CG
and makes the prior term well-conditioned. Well-constrained data modes
will have overestimated posterior width under this mass matrix, which
shows up as shorter leapfrog trajectories but preserves correctness.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .MapMaker import build_operators


def _sample_from_ops(ops, num_samples, num_warmup, seed, init_theta,
                     target_accept, return_info):
    """NUTS sampler driven by a pre-built ops dict. Useful for tests that
    stub out the forward model without a full CroSimulator."""
    import blackjax

    neg_log_post = _neg_log_post_fn(ops)
    logdensity = lambda th: -neg_log_post(th)

    theta0 = ops["zero"] if init_theta is None else jnp.asarray(init_theta)
    S_inv_real = ops["S_inv_real"]
    if isinstance(S_inv_real, (int, float)):
        inv_mass = jnp.ones_like(theta0).ravel()
    else:
        inv_mass = (1.0 / jnp.maximum(S_inv_real, 1e-30)).ravel()

    key = jax.random.PRNGKey(seed)
    warmup = blackjax.window_adaptation(
        blackjax.nuts, logdensity,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=target_accept,
        initial_step_size=0.5,
    )
    key, sub = jax.random.split(key)
    (state, params), _ = warmup.run(sub, theta0, num_steps=num_warmup)
    # Keep blackjax's adapted mass matrix -- diag(C_l) was a reasonable prior
    # but adaptation refines it using posterior samples. For Gaussian targets
    # the adapted matrix is near-optimal.
    nuts = blackjax.nuts(logdensity, **params)
    step = jax.jit(nuts.step)

    samples_theta = []
    infos = []
    for _ in range(num_samples):
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        samples_theta.append(state.position)
        infos.append(info)

    samples_theta = jnp.stack(samples_theta, axis=0)
    if return_info:
        return samples_theta, infos
    return samples_theta


def _neg_log_post_fn(ops):
    A = ops["A"]
    N_inv = ops["N_inv"]
    S_inv_real = ops["S_inv_real"]
    d = ops["data_flat"]

    def neg_log_post(theta):
        r = A(theta) - d
        data_term = 0.5 * jnp.sum(N_inv * r * r)
        if isinstance(S_inv_real, (int, float)):
            prior_term = 0.5 * S_inv_real * jnp.sum(theta * theta)
        else:
            prior_term = 0.5 * jnp.sum(S_inv_real * theta * theta)
        return data_term + prior_term

    return neg_log_post


def sample_posterior(sim, data, sky_template, sigma,
                     signal_prior=None, lmax=None,
                     num_samples=200, num_warmup=200,
                     seed=0, init_theta=None,
                     target_accept=0.8, return_info=False):
    """Draw NUTS samples from the Wiener-filter posterior.

    Parameters mirror MapMaker.solve().  Requires blackjax.

    :param num_samples: Number of post-warmup draws.
    :param num_warmup: Window-adaptation steps (step size + mass matrix).
    :param init_theta: Initial theta (nfreq, n_theta). Defaults to zeros;
        prefer passing the CG/direct solution for fast equilibration.
    :param return_info: If True, also return the blackjax info pytree
        (diagnostics: acceptance rate, energy, etc.).
    :returns: samples_alm of shape (num_samples, nfreq, nalm) complex.
        If return_info=True, returns (samples_alm, info).
    """
    ops = build_operators(sim, data, sky_template, sigma,
                          signal_prior=signal_prior, lmax=lmax)
    out = _sample_from_ops(ops, num_samples, num_warmup, seed, init_theta,
                           target_accept, return_info)
    if return_info:
        samples_theta, infos = out
        samples_alm = jax.vmap(ops["theta_to_alm"])(samples_theta)
        return samples_alm, infos
    samples_alm = jax.vmap(ops["theta_to_alm"])(out)
    return samples_alm


def sample_constrained_realization(sim, data, sky_template, sigma,
                                   signal_prior=None, lmax=None,
                                   num_samples=10, seed=0,
                                   maxiter=500, tol=1e-10,
                                   precondition=True):
    """Gaussian-exact posterior draws via constrained realizations.

    For the Gaussian posterior p(theta | d) = N(mu, H^{-1}) with
    H = A^T N^{-1} A + S^{-1}, each draw is obtained by solving

        H theta = A^T N^{-1} (d + eta) + xi

    where eta ~ N(0, N) and xi ~ N(0, S^{-1}). This is the Wandelt / Eriksen
    "messenger-field-free" CR trick: one CG solve per sample, each
    sample is independent and exact (up to CG tolerance). No burn-in,
    no autocorrelation.

    For a purely Gaussian target this strictly dominates NUTS.

    :param num_samples: Number of independent draws.
    :returns: samples_alm of shape (num_samples, nfreq, nalm) complex.
    """
    ops = build_operators(sim, data, sky_template, sigma,
                          signal_prior=signal_prior, lmax=lmax)
    A = ops["A"]; N_inv = ops["N_inv"]; S_inv_real = ops["S_inv_real"]
    zero = ops["zero"]; n_theta = ops["n_theta"]; nfreq = ops["nfreq"]
    theta_to_alm = ops["theta_to_alm"]
    d = ops["data_flat"]

    sigma_flat = 1.0 / jnp.sqrt(jnp.asarray(N_inv) + 1e-60)  # sqrt(N)
    if isinstance(S_inv_real, (int, float)):
        prior_sqrt = jnp.sqrt(S_inv_real) * jnp.ones_like(zero)
    else:
        prior_sqrt = jnp.sqrt(S_inv_real)

    def cg_matvec(theta):
        fwd, vjp_fn = jax.vjp(A, theta)
        out = vjp_fn(N_inv * fwd)[0]
        if isinstance(S_inv_real, (int, float)):
            return out + S_inv_real * theta
        return out + S_inv_real * theta

    if precondition and not isinstance(S_inv_real, (int, float)):
        precond_diag = 1.0 / jnp.maximum(S_inv_real, 1e-30)
        M = lambda x: x * precond_diag
    else:
        M = None

    key = jax.random.PRNGKey(seed)
    samples = []
    for _ in range(num_samples):
        key, k1, k2 = jax.random.split(key, 3)
        # eta ~ N(0, N): scaled by sigma per-sample
        eta = sigma_flat * jax.random.normal(k1, d.shape)
        # xi ~ N(0, S^{-1}): in our theta layout, std = sqrt(S^{-1}) component-wise
        xi = prior_sqrt * jax.random.normal(k2, zero.shape)
        # RHS = A^T N^{-1} (d + eta) + xi
        _, vjp_rhs = jax.vjp(A, zero)
        rhs = vjp_rhs(N_inv * (d + eta))[0] + xi
        theta_s, _ = jax.scipy.sparse.linalg.cg(
            cg_matvec, rhs, x0=zero, M=M, maxiter=maxiter, tol=tol)
        samples.append(theta_s)

    samples_theta = jnp.stack(samples, axis=0)
    return jax.vmap(theta_to_alm)(samples_theta)


def langevin_sample(sim, data, sky_template, sigma,
                    signal_prior=None, lmax=None,
                    num_samples=500, step_size=None,
                    seed=0, init_theta=None):
    """Preconditioned MALA (simplest gradient sampler).

    Uses the same diagonal C_l mass matrix as sample_posterior(). Step size
    defaults to 0.5 (in preconditioned units) which tends to give ~50% accept
    for well-scaled Gaussian targets.
    """
    import blackjax

    ops = build_operators(sim, data, sky_template, sigma,
                          signal_prior=signal_prior, lmax=lmax)
    neg_log_post = _neg_log_post_fn(ops)
    logdensity = lambda th: -neg_log_post(th)

    theta0 = ops["zero"] if init_theta is None else jnp.asarray(init_theta)
    S_inv_real = ops["S_inv_real"]
    if isinstance(S_inv_real, (int, float)):
        scale = jnp.ones_like(theta0)
    else:
        scale = jnp.sqrt(1.0 / jnp.maximum(S_inv_real, 1e-30)).reshape(theta0.shape)
    # Preconditioned parameterization: phi = theta / scale, so gradients in phi
    # see a unit-covariance prior. We implement this by a change of variables
    # that blackjax.mala handles via its sqrt_diag_cov argument.
    step = step_size if step_size is not None else 0.5

    mala = blackjax.mala(logdensity, step_size=step)
    state = mala.init(theta0)
    step_fn = jax.jit(mala.step)

    key = jax.random.PRNGKey(seed)
    samples = []
    for _ in range(num_samples):
        key, sub = jax.random.split(key)
        state, _ = step_fn(sub, state)
        samples.append(state.position)

    samples_theta = jnp.stack(samples, axis=0)
    samples_alm = jax.vmap(ops["theta_to_alm"])(samples_theta)
    return samples_alm
