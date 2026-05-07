"""
Time-per-iteration scaling for CG and L-BFGS as a function of ``lmax``.

Runs the *inner-loop step only* (one CG matvec ≈ one forward + one VJP,
one L-BFGS update = one value-and-grad + optax bookkeeping) twice per
configuration: once to pay tracing/compile, once for steady-state timing.
No plots, no maps, no solvers — just wall-clock per iteration.

    python examples/timing_scaling.py
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

# Shared configuration and helpers with the main benchmarks.
from optax_maxlike import (
    FREQ, DT_SEC, DF_HZ,
    make_simulator, forward, load_truth_sky,
    radiometric_sigma, cl_inv_full,
)


LMAX_GRID = (16, 32, 48, 64, 80)
N_REPEATS = 4           # total per-config iterations; first is trace+compile


def build_problem(lusee, lmax):
    """Build sim/data/prior common to both solvers, once per lmax."""
    sim = make_simulator(lusee, FREQ, lmax)
    sky_truth = load_truth_sky(lusee, FREQ, lmax)
    data_clean = forward(sim, sky_truth)
    sigma = radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(0), data_clean.shape)
    data = data_clean + noise
    cl_inv = cl_inv_full(sky_truth, lmax)
    return sim, sky_truth, data, sigma, cl_inv


def time_cg_step(lusee, sim, sky_truth, data, sigma, cl_inv, lmax):
    """Time one CG matvec: (A^T N^{-1} A + S^{-1}_real) θ."""
    from lusee.SkyModels import _real_alm_indices

    N_inv_flat = (1.0 / sigma ** 2).ravel()
    _, mpos_idx = _real_alm_indices(lmax)
    nfreq, nalm = sky_truth.mapalm.shape
    n_mpos = len(mpos_idx)
    n_theta = nalm + n_mpos

    # Real-coord prior diagonal (same as RealAlmSky.prior_inv_diag, flattened).
    s_re = cl_inv.at[:, mpos_idx].multiply(2.0)
    s_im = 2.0 * cl_inv[:, mpos_idx]
    S_inv_real = jnp.concatenate([s_re, s_im], axis=-1)
    mpos_idx_j = jnp.asarray(mpos_idx)

    def theta_to_alm(theta):
        re = theta[:, :nalm]
        im_mpos = theta[:, nalm:]
        im_full = jnp.zeros_like(re).at[:, mpos_idx_j].set(im_mpos)
        return re + 1j * im_full

    _, sky_aux = jax.tree_util.tree_flatten(sky_truth)

    def make_sky(alm):
        return jax.tree_util.tree_unflatten(sky_aux, (alm,))

    def A(theta):
        return sim.simulate(sky=make_sky(theta_to_alm(theta))).ravel()

    @jax.jit
    def cg_matvec(theta):
        fwd, vjp_fn = jax.vjp(A, theta)
        return vjp_fn(N_inv_flat * fwd)[0] + S_inv_real * theta

    theta = jnp.zeros((nfreq, n_theta))
    times = np.empty(N_REPEATS)
    for k in range(N_REPEATS):
        t0 = time.perf_counter()
        theta = cg_matvec(theta)
        theta.block_until_ready()
        times[k] = time.perf_counter() - t0
    return times


def time_lbfgs_step(lusee, sim, sky_truth, data, sigma, cl_inv, lmax):
    """Time one optax.lbfgs update (value+grad + line search + apply)."""
    data_flat = data.ravel()
    N_inv_flat = (1.0 / sigma ** 2).ravel()

    params = lusee.sky.RealAlmSky.zeros_like(sky_truth, lmax)
    re_diag, im_diag = params.prior_inv_diag(cl_inv)

    def loss_fn(sky):
        r = data_flat - forward(sim, sky).ravel()
        chi2 = jnp.sum(N_inv_flat * r ** 2)
        prior = (jnp.sum(re_diag * sky.re ** 2)
                 + jnp.sum(im_diag * sky.im_mpos ** 2))
        return chi2 + prior

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.lbfgs(memory_size=20)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(
            grad, opt_state, params,
            value=value, grad=grad, value_fn=loss_fn,
        )
        return optax.apply_updates(params, updates), opt_state, value

    times = np.empty(N_REPEATS)
    for k in range(N_REPEATS):
        t0 = time.perf_counter()
        params, opt_state, _ = step(params, opt_state)
        jax.block_until_ready(params.re)
        times[k] = time.perf_counter() - t0
    return times


def main():
    import lusee

    rows = []
    for lmax in LMAX_GRID:
        print(f"\n=== lmax={lmax} ===", flush=True)
        sim, sky_truth, data, sigma, cl_inv = build_problem(lusee, lmax)

        cg_times = time_cg_step(lusee, sim, sky_truth, data, sigma, cl_inv, lmax)
        print(f"  CG    iter times (s): {cg_times}", flush=True)

        lbfgs_times = time_lbfgs_step(
            lusee, sim, sky_truth, data, sigma, cl_inv, lmax
        )
        print(f"  LBFGS iter times (s): {lbfgs_times}", flush=True)

        # First iter = trace + compile; steady = min of remaining (tightest
        # lower bound, minimises OS jitter).
        rows.append(dict(
            lmax=lmax,
            cg_first=cg_times[0], cg_steady=float(np.min(cg_times[1:])),
            lbfgs_first=lbfgs_times[0],
            lbfgs_steady=float(np.min(lbfgs_times[1:])),
        ))

    print("\n\n| lmax | CG first (s) | CG steady (s) | LBFGS first (s) | LBFGS steady (s) |")
    print(  "|------|--------------|---------------|-----------------|------------------|")
    for r in rows:
        print(f"| {r['lmax']:4d} | {r['cg_first']:12.3f} | "
              f"{r['cg_steady']:13.6f} | "
              f"{r['lbfgs_first']:15.3f} | "
              f"{r['lbfgs_steady']:16.6f} |")


if __name__ == "__main__":
    main()
