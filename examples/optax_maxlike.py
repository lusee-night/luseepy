"""
Minimise the Camacho et al. 2026 log-posterior with JAX + optax:

    L(m) = (d - A m)^T N^{-1} (d - A m) + m^T S^{-1} m

This is 2 * the Bayesian -log P: the 1/2 factors of Eq. 336 in the
paper are dropped. Same minimum as the paper, but the printed
``final_loss`` is twice the canonical -log P value.

Sky is parameterised by real DOFs via ``lusee.sky.RealAlmSky``
(leaves: ``re`` over all m, ``im_mpos`` over m>0), so ``Im(a_{l,0})`` is
not a parameter and the Gaussian prior's factor-of-2 on m>0 entries is
baked into ``RealAlmSky.prior_inv_diag``.

Simulator portability: the only simulator-specific line is
``forward(sim, sky) = sim.simulate(sky=sky)``. Both CroSimulator and
JaxSimulator accept the kwarg and flow gradients through the sky pytree.

CLI
---
    python examples/optax_maxlike.py --sim jax --opt adam  --lmax 32 --n_iters 2000
    python examples/optax_maxlike.py --sim jax --opt lbfgs --lmax 32 --n_iters 500
    python examples/optax_maxlike.py --sim cro --opt lbfgs --lmax 32 --n_iters 500

Outputs go to ``examples/optax_maxlike_out/<sim>_<opt>_i_<N>_lmax<L>/``:
  - ``loss_history.npy``   — one entry per iteration
  - ``meta.json``          — optimiser, lmax, n_iters, wall-clock, final loss
  - ``maps_<FF>MHz.png``, ``rho_cl_<FF>MHz.png`` — per-frequency plots
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# NOTE: `import lusee` happens inside main() — see CLAUDE.md
# ("macOS + multiprocessing-at-import-time") for why.


# ── Fixed simulation config (shared with examples/mapmaker_benchmark.py) ──

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/Users/anigmetov/Data/lusee")
BEAM_FILE = DRIVE + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

FREQ = np.array([20.0, 25.0, 30.0])
OBS_RANGE = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
DT_SEC = 7200.0
DF_HZ = 1e6

EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(EXAMPLES_DIR, "optax_maxlike_out")


# ── Simulator setup ────────────────────────────────────────────────

SIMULATORS = {"jax": "JaxSimulator", "cro": "CroSimulator"}


def make_simulator(lusee, freq, lmax, engine="jax"):
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
        8, lmax, maps=[np.ones(12 * 64) for _ in freq],
        freq=freq, frame="galactic",
    )
    sim_cls = getattr(lusee, SIMULATORS[engine])
    return sim_cls(
        obs, beams, dummy_sky,
        Tground=0.0, combinations=combinations, freq=freq, lmax=lmax,
    )


def forward(sim, sky):
    return sim.simulate(sky=sky)


def load_truth_sky(lusee, freq, lmax):
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    idx = [int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in freq]
    maps = [hp.alm2map(np.asarray(sky_full.mapalm[i]), sky_full.Nside,
                       verbose=False) for i in idx]
    return lusee.sky.HealpixSky(sky_full.Nside, lmax,
                                maps=maps, freq=freq, frame="galactic")


# ── Noise and prior ────────────────────────────────────────────────

def radiometric_sigma(data, combinations, df_hz, dt_sec):
    """Per-channel σ from the radiometer equation (paper Eq. 9)."""
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


def cl_inv_full(sky, lmax):
    """Per-(nfreq, nalm) array of 1/C_l, broadcast over m for each l."""
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


# ── Plots ──────────────────────────────────────────────────────────

def save_plots(sky_truth, sky_hat, out_dir, freqs, lmax):
    os.makedirs(out_dir, exist_ok=True)
    nside = sky_truth.Nside
    for fi, f in enumerate(freqs):
        true_alm = np.asarray(sky_truth.mapalm[fi])
        rec_alm = np.asarray(sky_hat.mapalm[fi])
        true_map = hp.alm2map(true_alm, nside, verbose=False)
        rec_map = hp.alm2map(rec_alm, nside, verbose=False)
        resid = rec_map - true_map

        cl_true = hp.alm2cl(true_alm)
        cl_rec = hp.alm2cl(rec_alm)
        rho = hp.alm2cl(true_alm, rec_alm) / np.sqrt(cl_true * cl_rec + 1e-30)

        print(f"[{f:.0f} MHz] a00 true={true_alm[0].real:.0f} "
              f"rec={rec_alm[0].real:.0f} | "
              f"resid [{resid.min():.0f}, {resid.max():.0f}] K | "
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


# ── main ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--opt", choices=("adam", "lbfgs"), default="lbfgs")
    p.add_argument("--sim", choices=tuple(SIMULATORS), default="jax",
                   help="Simulator engine: 'jax' (JaxSimulator, s2fft rotations) "
                        "or 'cro' (CroSimulator, Croissant MEPA pipeline)")
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--n_iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2,
                   help="Adam learning rate; ignored by lbfgs (uses linesearch)")
    p.add_argument("--memory", type=int, default=20,
                   help="L-BFGS memory size (history length)")
    p.add_argument("--tag", default=None,
                   help="Optional suffix for output directory name")
    p.add_argument("--whiten", action="store_true",
                   help="Optimise in whitened coordinates θ'=θ·√(S^-1) so the "
                        "prior is ||θ'||² (preconditioned via change-of-var).")
    return p.parse_args()


def main():
    args = parse_args()
    import lusee

    tag = f"_{args.tag}" if args.tag else ""
    whiten_suffix = "_whiten" if args.whiten else ""
    out_dir = os.path.join(
        OUT_ROOT,
        f"{args.sim}_{args.opt}_i_{args.n_iters}_lmax{args.lmax}{whiten_suffix}{tag}",
    )
    os.makedirs(out_dir, exist_ok=True)

    sim = make_simulator(lusee, FREQ, args.lmax, engine=args.sim)
    sky_truth = load_truth_sky(lusee, FREQ, args.lmax)

    data_clean = forward(sim, sky_truth)
    sigma = radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(0), data_clean.shape)
    data = data_clean + noise
    N_inv = 1.0 / sigma ** 2
    N_inv_flat = N_inv.ravel()
    data_flat = data.ravel()

    cl_inv = cl_inv_full(sky_truth, args.lmax)
    params = lusee.sky.RealAlmSky.zeros_like(sky_truth, args.lmax)
    re_diag, im_diag = params.prior_inv_diag(cl_inv)

    # Change-of-variable preconditioner. In whitened coords θ' = θ·√(S^-1)
    # the quadratic prior is ||θ'||² (isotropic) and the Hessian dynamic
    # range collapses by ~κ(S^-1). Non-whitened path: multiplications by 1.
    if args.whiten:
        sqrt_S_re = 1.0 / jnp.sqrt(re_diag)  # dewhiten factor
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
        r = data_flat - forward(sim, sky_phys).ravel()
        chi2 = jnp.sum(N_inv_flat * r ** 2)
        prior = (jnp.sum(prior_diag_re * sky.re ** 2)
                 + jnp.sum(prior_diag_im * sky.im_mpos ** 2))
        return chi2 + prior

    value_and_grad_fn = jax.value_and_grad(loss_fn)

    if args.opt == "adam":
        optimizer = optax.adam(args.lr)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            value, grad = value_and_grad_fn(params)
            updates, opt_state = optimizer.update(grad, opt_state)
            return optax.apply_updates(params, updates), opt_state, value

    elif args.opt == "lbfgs":
        optimizer = optax.lbfgs(memory_size=args.memory)
        opt_state = optimizer.init(params)

        @jax.jit
        def step(params, opt_state):
            value, grad = value_and_grad_fn(params)
            updates, opt_state = optimizer.update(
                grad, opt_state, params,
                value=value, grad=grad, value_fn=loss_fn,
            )
            return optax.apply_updates(params, updates), opt_state, value

    else:
        raise ValueError(args.opt)

    loss_history = np.empty(args.n_iters, dtype=np.float64)
    iter_times = np.empty(args.n_iters, dtype=np.float64)
    t_start = time.time()
    t_prev = t_start
    for it in range(args.n_iters):
        params, opt_state, loss = step(params, opt_state)
        jax.block_until_ready(params.re)
        t_now = time.time()
        loss_history[it] = float(loss)
        iter_times[it] = t_now - t_prev
        t_prev = t_now
    wall_clock = time.time() - t_start

    np.save(os.path.join(out_dir, "loss_history.npy"), loss_history)
    np.save(os.path.join(out_dir, "iter_times.npy"), iter_times)
    final_loss = float(loss_history[-1])
    meta = dict(
        sim=args.sim,
        opt=args.opt, lmax=args.lmax, n_iters=args.n_iters,
        lr=args.lr, memory=args.memory, whiten=args.whiten,
        freq=FREQ.tolist(),
        wall_clock_s=wall_clock,
        first_iter_s=float(iter_times[0]),
        median_iter_s=float(np.median(iter_times[1:])) if args.n_iters > 1 else None,
        final_loss=final_loss,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[sim={args.sim} {args.opt} lmax={args.lmax} n={args.n_iters}"
          f"{' whiten' if args.whiten else ''}] "
          f"wall={wall_clock:.1f}s  final_loss={final_loss:.6e}  "
          f"out={out_dir}")

    # Dewhiten (identity in the non-whiten branch) before plotting.
    params_phys = lusee.sky.RealAlmSky(
        re=params.re * sqrt_S_re, im_mpos=params.im_mpos * sqrt_S_im,
        lmax=params.lmax, Nside=params.Nside, frame=params.frame,
    )
    save_plots(sky_truth, params_phys, out_dir, FREQ, args.lmax)


if __name__ == "__main__":
    main()
