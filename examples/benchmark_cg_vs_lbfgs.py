"""
Three-way mapmaking benchmark: CG (mapmaker) vs optax LBFGS (vanilla / whitened).

All three solve the same Wiener-filter problem on the same data and prior.
Each runner is fully self-contained: it builds its own simulator, sky truth,
data, noise, and prior. The setup is identical (same seeds), so the optimisers
are directly comparable.

Procedure
---------
1. ``run_cg(n_iters=N)`` — run ``lusee.mapmaker.solve(method='cg')`` for
   ``N`` iterations (default 4000). Records its mean ρ(1..10) per frequency
   as the LBFGS target.
2. ``run_lbfgs(target_rho_per_freq, whiten=False)`` — optax LBFGS until the
   recovered sky's mean ρ(1..10) at every frequency is ≥ CG's value
   (or ``max_iters`` cap).
3. ``run_lbfgs(target_rho_per_freq, whiten=True)`` — same with the
   change-of-variable preconditioner θ' = θ·√(S^{-1}).

Why ρ, not loss
---------------
The loss has a Gaussian prior that shrinks the recovered alm. Two solutions
can share the same loss while differing in pattern fidelity (one with
smaller-magnitude but better-aligned alm; the other louder but noisier).
ρ_ℓ is amplitude-blind, so it isolates pattern fidelity. Mean ρ(1..10) is
the standard low-ℓ reconstruction-quality metric (cf. Camacho+ 2026 Fig 5).

Output structure
----------------
``examples/benchmark_<sim>_lmax<L>/``
  ``cg/``             meta.json + per-frequency maps and ρ_ℓ plots
  ``lbfgs/``          + ``loss_history.npy``, ``iter_times.npy``
  ``lbfgs_whiten/``   + ``loss_history.npy``, ``iter_times.npy``
  ``loss_curves.png`` LBFGS trajectories with horizontal CG-final marker
  ``summary.json``

CLI
---
    python examples/benchmark_cg_vs_lbfgs.py --sim jax  --lmax 32
    python examples/benchmark_cg_vs_lbfgs.py --sim cro  --lmax 32 --cg_iters 4000
    # explicit ρ targets, skipping CG
    python examples/benchmark_cg_vs_lbfgs.py --skip_cg --target_rho 0.95,0.93,0.91
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import json
import time
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: ``import lusee`` is deferred to inside each runner — see CLAUDE.md
# (macOS + multiprocessing-at-import-time).


# ── Fixed config (matches examples/optax_maxlike.py) ──────────────────

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/Users/anigmetov/Data/lusee")
BEAM_FILE = DRIVE + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

FREQ = np.array([20.0, 25.0, 30.0])
OBS_RANGE = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
DT_SEC = 7200.0
DF_HZ = 1e6
NOISE_SEED = 0

SIMULATORS = {"jax": "JaxSimulator", "cro": "CroSimulator"}

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


Result = namedtuple("Result", [
    "method",         # "cg" | "lbfgs" | "lbfgs_whiten"
    "out_dir",        # output subdirectory
    "setup_s",        # wall time to build sim, sky, data, sigma, prior
    "first_iter_s",   # first iteration time (includes JIT compile)
    "total_s",        # total optimisation wall time (excl. setup, incl. compile)
    "median_iter_s",  # median post-compile iteration time
    "iter_times",     # np.ndarray of per-iter times (None for CG)
    "n_iters",        # iterations actually run
    "loss_history",   # np.ndarray (single-element [final] for CG)
    "final_loss",     # float
    "rho_per_freq",   # np.ndarray (nfreq,) — mean ρ(1..10) per frequency at the end
    "rho_check_iters",  # np.ndarray of iter indices where ρ was computed (None for CG)
    "rho_history",    # np.ndarray (nchecks, nfreq) — ρ values at those iters (None for CG)
    "sky_alm",        # complex (nfreq, nalm) recovered alm
])


# Multipole range used for the mean-ρ stopping criterion (Camacho+ 2026 Fig 5).
RHO_LMIN = 1
RHO_LMAX = 10


# ── Setup helper (called independently inside each runner) ───────────

def _build_setup(lusee, lmax, engine):
    """Build sim + truth sky + noisy data + per-channel σ + 1/C_l prior.

    Reproducible: the same (sim, lmax, engine) inputs produce bitwise
    identical (data, sigma, cl_inv) across runs because the noise PRNG seed
    is fixed (NOISE_SEED) and the simulator is deterministic.
    """
    layout = [("N", 0), ("E", -90), ("S", -180), ("W", -270)]
    beams = []
    for name, angle in layout:
        b = lusee.Beam(BEAM_FILE, id=name).rotate(angle)
        b.taper_and_smooth(taper=0.03)
        beams.append(b)
    obs = lusee.Observation(OBS_RANGE, deltaT_sec=DT_SEC,
                            lun_lat_deg=-10.0, lun_long_deg=180.0)
    combinations = [(0, 0), (1, 1), (2, 2), (3, 3),
                    (0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3)]
    dummy_sky = lusee.sky.HealpixSky(
        8, lmax, maps=[np.ones(12 * 64) for _ in FREQ],
        freq=FREQ, frame="galactic",
    )
    sim_cls = getattr(lusee, SIMULATORS[engine])
    sim = sim_cls(obs, beams, dummy_sky, Tground=0.0,
                  combinations=combinations, freq=FREQ, lmax=lmax)

    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    idx = [int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQ]
    maps = [hp.alm2map(np.asarray(sky_full.mapalm[i]), sky_full.Nside,
                       verbose=False) for i in idx]
    sky_truth = lusee.sky.HealpixSky(sky_full.Nside, lmax,
                                     maps=maps, freq=FREQ, frame="galactic")

    data_clean = sim.simulate(sky=sky_truth)
    sigma = _radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(NOISE_SEED),
                                      data_clean.shape)
    data = data_clean + noise
    cl_inv = _cl_inv_full(sky_truth, lmax)

    return sim, sky_truth, data, sigma, cl_inv


def _radiometric_sigma(data, combinations, df_hz, dt_sec):
    data = np.asarray(data)
    sigma = np.zeros_like(data, dtype=np.float64)
    auto_ch, ch = {}, 0
    for i, j in combinations:
        if i == j:
            auto_ch[i] = ch; ch += 1
        else:
            ch += 2
    ch = 0
    for i, j in combinations:
        if i == j:
            T = np.abs(data[:, ch, :])
            sigma[:, ch, :] = T / np.sqrt(df_hz * dt_sec)
            ch += 1
        else:
            Tii = np.abs(data[:, auto_ch[i], :])
            Tjj = np.abs(data[:, auto_ch[j], :])
            V2 = data[:, ch, :] ** 2 + data[:, ch + 1, :] ** 2
            s = np.sqrt((Tii * Tjj + V2) / (4.0 * df_hz * dt_sec))
            sigma[:, ch, :] = s; sigma[:, ch + 1, :] = s
            ch += 2
    return jnp.asarray(sigma)


def _cl_inv_full(sky, lmax):
    nfreq, nalm = sky.mapalm.shape
    out = np.zeros((nfreq, nalm))
    for f in range(nfreq):
        cl = hp.alm2cl(np.asarray(sky.mapalm[f]))
        for l in range(min(len(cl), lmax + 1)):
            if cl[l] <= 0:
                continue
            for m in range(l + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                if idx < nalm:
                    out[f, idx] = 1.0 / cl[l]
    return jnp.asarray(out)


def _evaluate_loss(lusee, sim, data, sigma, cl_inv, sky_alm, sky_truth, lmax):
    """Optax-convention loss: χ²(d, Aθ) + θ^T S_real^{-1} θ.

    No ½ factor — matches the loss_fn used by run_lbfgs, so CG and LBFGS
    final losses are on the same scale."""
    aux = sky_truth.tree_flatten()[1]
    sky_hat = type(sky_truth).tree_unflatten(aux, (jnp.asarray(sky_alm),))
    n_inv = (1.0 / sigma ** 2).ravel()
    r = jnp.asarray(data).ravel() - sim.simulate(sky=sky_hat).ravel()
    chi2 = float(jnp.sum(n_inv * r ** 2))

    real_sky = lusee.sky.RealAlmSky.from_healpix(sky_hat, lmax)
    re_diag, im_diag = real_sky.prior_inv_diag(cl_inv)
    prior = float(jnp.sum(re_diag * real_sky.re ** 2)
                  + jnp.sum(im_diag * real_sky.im_mpos ** 2))
    return chi2 + prior


def _mean_rho_per_freq(sky_alm, sky_truth, lmin=RHO_LMIN, lmax=RHO_LMAX):
    """Mean ρ_ℓ over ℓ ∈ [lmin, lmax] for each frequency. Returns shape (nfreq,)."""
    nfreq = sky_alm.shape[0]
    rhos = np.empty(nfreq, dtype=np.float64)
    for fi in range(nfreq):
        true_alm = np.asarray(sky_truth.mapalm[fi])
        rec_alm = np.asarray(sky_alm[fi])
        cl_t = hp.alm2cl(true_alm)
        cl_r = hp.alm2cl(rec_alm)
        rho_l = hp.alm2cl(true_alm, rec_alm) / np.sqrt(cl_t * cl_r + 1e-30)
        rhos[fi] = float(np.nanmean(rho_l[lmin:lmax + 1]))
    return rhos


# ── Per-method reconstruction plots (drop into out_dir) ──────────────

def _save_recon_plots(sky_truth, sky_alm, out_dir, lmax):
    os.makedirs(out_dir, exist_ok=True)
    nside = sky_truth.Nside
    for fi, f in enumerate(FREQ):
        true_alm = np.asarray(sky_truth.mapalm[fi])
        rec_alm = np.asarray(sky_alm[fi])
        true_map = hp.alm2map(true_alm, nside, verbose=False)
        rec_map = hp.alm2map(rec_alm, nside, verbose=False)
        resid = rec_map - true_map
        cl_true = hp.alm2cl(true_alm)
        cl_rec = hp.alm2cl(rec_alm)
        rho = hp.alm2cl(true_alm, rec_alm) / np.sqrt(cl_true * cl_rec + 1e-30)

        print(f"  [{f:.0f} MHz] resid [{resid.min():.0f}, {resid.max():.0f}] K | "
              f"mean ρ(1..10)={np.nanmean(rho[1:11]):.4f}")

        fig = plt.figure(figsize=(15, 4))
        hp.mollview(true_map, title=f"ULSA input ({f:.0f} MHz)",
                    cmap="inferno", sub=(1, 3, 1), fig=fig)
        hp.mollview(rec_map, title="reconstruction",
                    cmap="inferno", sub=(1, 3, 2), fig=fig)
        hp.mollview(resid, title="residual (rec − true)",
                    cmap="RdBu_r", sub=(1, 3, 3), fig=fig)
        fig.savefig(os.path.join(out_dir, f"maps_{int(f):02d}MHz.png"), dpi=120)
        plt.close(fig)

        ell = np.arange(len(rho))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(ell[1:lmax + 1], rho[1:lmax + 1], "o-", ms=4)
        ax1.axhline(0, color="k", lw=0.5)
        ax1.axhline(1, color="k", lw=0.5, ls="--")
        ax1.set_xlabel(r"Multipole $\ell$"); ax1.set_ylabel(r"$\rho_\ell$")
        ax1.set_title(f"ρ_ℓ at {f:.0f} MHz "
                      f"(mean ρ(1..10)={np.nanmean(rho[1:11]):.2f})")
        ax1.set_ylim(-0.2, 1.1); ax1.set_xlim(0, lmax + 1)
        ax2.semilogy(ell[1:lmax + 1], cl_true[1:lmax + 1], "k-", lw=2,
                     label=r"Input $C_\ell$")
        ax2.semilogy(ell[1:lmax + 1], cl_rec[1:lmax + 1], "r--", lw=2,
                     label=r"Recovered $C_\ell$")
        ax2.set_xlabel(r"Multipole $\ell$"); ax2.set_ylabel(r"$C_\ell$")
        ax2.set_title("angular power spectrum"); ax2.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rho_cl_{int(f):02d}MHz.png"),
                    dpi=120)
        plt.close(fig)


# ── Runner 1: CG (mapmaker) ──────────────────────────────────────────

def run_cg(*, lmax, n_iters, engine, out_dir):
    """Run lusee.mapmaker.solve with method='cg' for a fixed n_iters.

    Two-pass timing: a maxiter=1 warmup absorbs the JIT compile cost
    (jax.scipy.sparse.linalg.cg compiles its while_loop body once and JAX
    caches it across calls with the same shapes), then the timed pass
    runs the full ``n_iters`` steps in steady state."""
    import lusee
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== CG (mapmaker) ===\n  engine={engine}  lmax={lmax}  "
          f"maxiter={n_iters}  out={out_dir}")

    t0 = time.time()
    sim, sky_truth, data, sigma, cl_inv = _build_setup(lusee, lmax, engine)
    setup_s = time.time() - t0

    # Warmup: compile + 1 CG step. Subsequent solve() calls hit JAX's compile cache.
    t0 = time.time()
    warm = lusee.mapmaker.solve(
        sim, data, sky_truth, sigma,
        signal_prior=cl_inv, lmax=lmax,
        maxiter=1, tol=0.0, precondition=True, method="cg",
    )
    jax.block_until_ready(warm)
    first_iter_s = time.time() - t0

    # Timed run.
    t0 = time.time()
    sky_alm = lusee.mapmaker.solve(
        sim, data, sky_truth, sigma,
        signal_prior=cl_inv, lmax=lmax,
        maxiter=n_iters, tol=0.0, precondition=True, method="cg",
    )
    jax.block_until_ready(sky_alm)
    main_s = time.time() - t0
    total_s = first_iter_s + main_s
    median_iter_s = main_s / max(n_iters, 1)

    sky_alm_np = np.asarray(sky_alm)
    final_loss = _evaluate_loss(lusee, sim, data, sigma, cl_inv,
                                sky_alm_np, sky_truth, lmax)
    rho_per_freq = _mean_rho_per_freq(sky_alm_np, sky_truth)
    rho_str = "  ".join(f"{f:.0f}MHz:{r:.4f}" for f, r in zip(FREQ, rho_per_freq))
    print(f"  setup={setup_s:.1f}s  first_iter={first_iter_s:.2f}s  "
          f"main={main_s:.2f}s  per_iter≈{median_iter_s*1e3:.1f}ms  "
          f"final_loss={final_loss:.6e}")
    print(f"  mean ρ({RHO_LMIN}..{RHO_LMAX})  →  {rho_str}")

    _save_recon_plots(sky_truth, sky_alm, out_dir, lmax)

    meta = dict(
        method="cg", engine=engine, lmax=lmax, n_iters=n_iters,
        setup_s=setup_s, first_iter_s=first_iter_s, total_s=total_s,
        median_iter_s=median_iter_s, final_loss=final_loss,
        rho_per_freq=rho_per_freq.tolist(),
        rho_lmin=RHO_LMIN, rho_lmax=RHO_LMAX,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return Result(
        method="cg", out_dir=out_dir,
        setup_s=setup_s, first_iter_s=first_iter_s, total_s=total_s,
        median_iter_s=median_iter_s, iter_times=None, n_iters=n_iters,
        loss_history=np.array([final_loss]),
        final_loss=final_loss,
        rho_per_freq=rho_per_freq,
        rho_check_iters=None, rho_history=None,
        sky_alm=sky_alm_np,
    )


# ── Runner 2/3: optax LBFGS (vanilla / whitened) ─────────────────────

def run_lbfgs(*, lmax, target_rho_per_freq, max_iters, memory, whiten,
              engine, out_dir, rho_check_every=10):
    """Run optax.lbfgs until mean ρ(RHO_LMIN..RHO_LMAX) ≥ target at every
    frequency (or ``max_iters`` reached).

    ρ is computed every ``rho_check_every`` iterations: assemble the
    physical (dewhitened) sky alm on GPU, transfer to CPU, run a few cheap
    ``hp.alm2cl`` calls. At lmax≈32 this overhead is negligible per check."""
    import lusee
    os.makedirs(out_dir, exist_ok=True)
    label = "lbfgs_whiten" if whiten else "lbfgs"
    target_str = "  ".join(f"{f:.0f}MHz:{r:.4f}"
                            for f, r in zip(FREQ, target_rho_per_freq))
    print(f"\n=== {label} ===\n  engine={engine}  lmax={lmax}  "
          f"max_iters={max_iters}  whiten={whiten}  "
          f"check_every={rho_check_every}  out={out_dir}\n"
          f"  target ρ({RHO_LMIN}..{RHO_LMAX}): {target_str}")

    t0 = time.time()
    sim, sky_truth, data, sigma, cl_inv = _build_setup(lusee, lmax, engine)

    n_inv_flat = (1.0 / sigma ** 2).ravel()
    data_flat = jnp.asarray(data).ravel()

    params = lusee.sky.RealAlmSky.zeros_like(sky_truth, lmax)
    re_diag, im_diag = params.prior_inv_diag(cl_inv)

    if whiten:
        # θ_phys = θ_white · 1/√(S⁻¹) = θ_white · √S
        sqrt_S_re = 1.0 / jnp.sqrt(re_diag)
        sqrt_S_im = 1.0 / jnp.sqrt(im_diag)
        prior_diag_re = jnp.ones_like(re_diag)
        prior_diag_im = jnp.ones_like(im_diag)
    else:
        sqrt_S_re = jnp.ones_like(re_diag)
        sqrt_S_im = jnp.ones_like(im_diag)
        prior_diag_re = re_diag
        prior_diag_im = im_diag

    def loss_fn(sky):
        sky_phys = lusee.sky.RealAlmSky(
            re=sky.re * sqrt_S_re, im_mpos=sky.im_mpos * sqrt_S_im,
            lmax=sky.lmax, Nside=sky.Nside, frame=sky.frame,
        )
        r = data_flat - sim.simulate(sky=sky_phys).ravel()
        chi2 = jnp.sum(n_inv_flat * r ** 2)
        prior = (jnp.sum(prior_diag_re * sky.re ** 2)
                 + jnp.sum(prior_diag_im * sky.im_mpos ** 2))
        return chi2 + prior

    value_and_grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.lbfgs(memory_size=memory)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad_fn(params)
        updates, opt_state = optimizer.update(
            grad, opt_state, params,
            value=value, grad=grad, value_fn=loss_fn,
        )
        return optax.apply_updates(params, updates), opt_state, value

    setup_s = time.time() - t0

    target_rho_arr = np.asarray(target_rho_per_freq, dtype=np.float64)

    def _current_sky_alm():
        params_phys = lusee.sky.RealAlmSky(
            re=params.re * sqrt_S_re, im_mpos=params.im_mpos * sqrt_S_im,
            lmax=params.lmax, Nside=params.Nside, frame=params.frame,
        )
        return np.asarray(params_phys.mapalm)

    losses = np.empty(max_iters, dtype=np.float64)
    iter_times = np.empty(max_iters, dtype=np.float64)
    rho_check_iters = []
    rho_history = []  # list of length-nfreq arrays
    t_loop = time.time()
    t_prev = t_loop
    n_iters = 0
    reached = False
    for it in range(max_iters):
        params, opt_state, val = step(params, opt_state)
        jax.block_until_ready(params.re)
        t_now = time.time()
        losses[it] = float(val)
        iter_times[it] = t_now - t_prev
        t_prev = t_now
        n_iters = it + 1

        is_check_iter = ((it + 1) % rho_check_every == 0) or (it == max_iters - 1)
        if is_check_iter:
            rhos = _mean_rho_per_freq(_current_sky_alm(), sky_truth)
            rho_check_iters.append(it + 1)
            rho_history.append(rhos)
            if np.all(rhos >= target_rho_arr):
                reached = True
                break
    total_s = time.time() - t_loop

    losses = losses[:n_iters]
    iter_times = iter_times[:n_iters]
    rho_check_iters = np.asarray(rho_check_iters, dtype=np.int64)
    rho_history = np.asarray(rho_history, dtype=np.float64)

    sky_alm = _current_sky_alm()

    first_iter_s = float(iter_times[0])
    median_iter_s = (float(np.median(iter_times[1:])) if n_iters > 1
                    else first_iter_s)
    final_loss = float(losses[-1])
    final_rho = rho_history[-1] if len(rho_history) else _mean_rho_per_freq(
        sky_alm, sky_truth)
    rho_str = "  ".join(f"{f:.0f}MHz:{r:.4f}" for f, r in zip(FREQ, final_rho))

    print(f"  setup={setup_s:.1f}s  first_iter={first_iter_s:.2f}s  "
          f"total={total_s:.2f}s  per_iter≈{median_iter_s*1e3:.1f}ms  "
          f"n_iters={n_iters}  final_loss={final_loss:.6e}  "
          f"{'(reached target)' if reached else '(target NOT reached)'}")
    print(f"  mean ρ({RHO_LMIN}..{RHO_LMAX})  →  {rho_str}")

    _save_recon_plots(sky_truth, sky_alm, out_dir, lmax)
    np.save(os.path.join(out_dir, "loss_history.npy"), losses)
    np.save(os.path.join(out_dir, "iter_times.npy"), iter_times)
    np.save(os.path.join(out_dir, "rho_check_iters.npy"), rho_check_iters)
    np.save(os.path.join(out_dir, "rho_history.npy"), rho_history)

    meta = dict(
        method=label, engine=engine, lmax=lmax,
        max_iters=max_iters, n_iters=n_iters, memory=memory, whiten=whiten,
        target_rho_per_freq=target_rho_arr.tolist(), target_reached=reached,
        rho_lmin=RHO_LMIN, rho_lmax=RHO_LMAX, rho_check_every=rho_check_every,
        setup_s=setup_s, first_iter_s=first_iter_s, total_s=total_s,
        median_iter_s=median_iter_s, final_loss=final_loss,
        final_rho_per_freq=final_rho.tolist(),
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    return Result(
        method=label, out_dir=out_dir,
        setup_s=setup_s, first_iter_s=first_iter_s, total_s=total_s,
        median_iter_s=median_iter_s, iter_times=iter_times, n_iters=n_iters,
        loss_history=losses, final_loss=final_loss,
        rho_per_freq=final_rho,
        rho_check_iters=rho_check_iters, rho_history=rho_history,
        sky_alm=sky_alm,
    )


# ── Reporting ────────────────────────────────────────────────────────

def print_summary(results):
    methods = [r.method for r in results]
    rows = [
        ("Total time (s)",            lambda r: f"{r.total_s:.2f}"),
        ("# iters",                   lambda r: f"{r.n_iters}"),
        ("First iter / compile (s)",  lambda r: f"{r.first_iter_s:.3f}"),
        ("Per-iter after jit (s)",    lambda r: f"{r.median_iter_s:.4f}"),
        ("Setup (s)",                 lambda r: f"{r.setup_s:.1f}"),
        ("Final loss",                lambda r: f"{r.final_loss:.4e}"),
    ]
    for fi, f in enumerate(FREQ):
        rows.append((
            f"mean ρ({RHO_LMIN}..{RHO_LMAX}) @ {f:.0f} MHz",
            (lambda fi=fi: lambda r: f"{r.rho_per_freq[fi]:.4f}")(),
        ))
    col = 18
    header = f"{'metric':32s} | " + " | ".join(f"{m:>{col}s}" for m in methods)
    print()
    print("=" * len(header))
    print("Benchmark summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for label, fn in rows:
        row = f"{label:32s} | " + " | ".join(f"{fn(r):>{col}s}" for r in results)
        print(row)
    print("=" * len(header))


_LBFGS_COLOR = {"lbfgs": "C1", "lbfgs_whiten": "C2"}
_LBFGS_LABEL = {"lbfgs": "LBFGS", "lbfgs_whiten": "LBFGS + whiten"}


def plot_loss_curves(results, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cg = next((r for r in results if r.method == "cg"), None)
    if cg is not None:
        ax.axhline(cg.final_loss, color="C0", ls="--", lw=1.5,
                   label=f"CG final ({cg.n_iters} iters, {cg.total_s:.0f}s)")
    for r in results:
        if r.method in _LBFGS_COLOR:
            ax.semilogy(np.arange(1, len(r.loss_history) + 1),
                        r.loss_history, color=_LBFGS_COLOR[r.method], lw=1.5,
                        label=f"{_LBFGS_LABEL[r.method]} "
                              f"({r.n_iters} iters, {r.total_s:.0f}s)")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"loss = $\chi^2 + \theta^{T} S_{\rm real}^{-1} \theta$")
    ax.set_title("CG vs LBFGS vs LBFGS+whitening (loss)")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_rho_curves(results, out_path):
    """One subplot per frequency, showing CG-final dashed lines + LBFGS rho trajectories."""
    nfreq = len(FREQ)
    fig, axes = plt.subplots(1, nfreq, figsize=(5.5 * nfreq, 4.5),
                              sharey=True)
    if nfreq == 1:
        axes = [axes]
    cg = next((r for r in results if r.method == "cg"), None)
    for fi, (ax, f) in enumerate(zip(axes, FREQ)):
        if cg is not None:
            ax.axhline(cg.rho_per_freq[fi], color="C0", ls="--", lw=1.5,
                       label=f"CG final ({cg.n_iters} iters, {cg.total_s:.0f}s)")
        for r in results:
            if r.method in _LBFGS_COLOR and r.rho_history is not None and len(r.rho_history):
                ax.plot(r.rho_check_iters, r.rho_history[:, fi],
                        color=_LBFGS_COLOR[r.method], marker="o", ms=3, lw=1.2,
                        label=f"{_LBFGS_LABEL[r.method]} "
                              f"({r.n_iters} iters, {r.total_s:.0f}s)")
        ax.set_xlabel("iteration")
        if fi == 0:
            ax.set_ylabel(rf"mean $\rho({RHO_LMIN}..{RHO_LMAX})$")
        ax.set_title(f"{f:.0f} MHz")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(r"Reconstruction fidelity $\rho_\ell$ vs iteration")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", choices=tuple(SIMULATORS), default="jax",
                   help="Simulator engine (shared across all 3 runs).")
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--cg_iters", type=int, default=4000,
                   help="Fixed number of CG iterations (defines target loss).")
    p.add_argument("--max_lbfgs_iters", type=int, default=80000,
                   help="Cap on LBFGS iters; loop exits early on target-ρ.")
    p.add_argument("--memory", type=int, default=20,
                   help="LBFGS history length.")
    p.add_argument("--rho_check_every", type=int, default=10,
                   help="Compute mean ρ every N LBFGS iterations.")
    p.add_argument("--out", default=None,
                   help="Output root; default: examples/benchmark_<sim>_lmax<L>/")
    p.add_argument("--skip_cg", action="store_true",
                   help="Skip CG; requires --target_rho.")
    p.add_argument("--skip_lbfgs", action="store_true",
                   help="Skip vanilla LBFGS.")
    p.add_argument("--skip_lbfgs_whiten", action="store_true",
                   help="Skip whitened LBFGS.")
    p.add_argument("--target_rho", default=None,
                   help="Override CG-derived target. Either a single float "
                        "(applied to all frequencies) or a comma-separated "
                        "list with one entry per frequency.")
    return p.parse_args()


def _parse_target_rho(s, nfreq):
    """Accept '0.95' or '0.95,0.93,0.91'."""
    parts = [float(x) for x in s.split(",")]
    if len(parts) == 1:
        parts = parts * nfreq
    if len(parts) != nfreq:
        raise SystemExit(
            f"--target_rho must have 1 or {nfreq} comma-separated values, got {len(parts)}")
    return np.asarray(parts, dtype=np.float64)


def main():
    args = parse_args()
    out_root = args.out or os.path.join(
        EXAMPLES_DIR, f"benchmark_{args.sim}_lmax{args.lmax}")
    os.makedirs(out_root, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")
    print(f"sim={args.sim}  lmax={args.lmax}  out_root={out_root}")

    results = []

    if not args.skip_cg:
        results.append(run_cg(
            lmax=args.lmax, n_iters=args.cg_iters,
            engine=args.sim, out_dir=os.path.join(out_root, "cg"),
        ))

    if args.target_rho is not None:
        target_rho = _parse_target_rho(args.target_rho, len(FREQ))
    elif results:
        target_rho = results[0].rho_per_freq
    else:
        raise SystemExit("Need either CG run or --target_rho to set the target.")

    if not args.skip_lbfgs:
        results.append(run_lbfgs(
            lmax=args.lmax, target_rho_per_freq=target_rho,
            max_iters=args.max_lbfgs_iters, memory=args.memory,
            whiten=False, engine=args.sim,
            rho_check_every=args.rho_check_every,
            out_dir=os.path.join(out_root, "lbfgs"),
        ))

    if not args.skip_lbfgs_whiten:
        results.append(run_lbfgs(
            lmax=args.lmax, target_rho_per_freq=target_rho,
            max_iters=args.max_lbfgs_iters, memory=args.memory,
            whiten=True, engine=args.sim,
            rho_check_every=args.rho_check_every,
            out_dir=os.path.join(out_root, "lbfgs_whiten"),
        ))

    print_summary(results)
    plot_loss_curves(results, os.path.join(out_root, "loss_curves.png"))
    plot_rho_curves(results, os.path.join(out_root, "rho_curves.png"))

    summary = {r.method: dict(
        out_dir=r.out_dir, setup_s=r.setup_s, first_iter_s=r.first_iter_s,
        total_s=r.total_s, median_iter_s=r.median_iter_s,
        n_iters=r.n_iters, final_loss=r.final_loss,
        rho_per_freq=r.rho_per_freq.tolist(),
    ) for r in results}
    with open(os.path.join(out_root, "summary.json"), "w") as fh:
        json.dump({
            "sim": args.sim, "lmax": args.lmax, "freq": FREQ.tolist(),
            "target_rho_per_freq": target_rho.tolist(),
            "rho_lmin": RHO_LMIN, "rho_lmax": RHO_LMAX,
            "results": summary,
        }, fh, indent=2)

    print(f"\nAll outputs in: {out_root}")


if __name__ == "__main__":
    main()
