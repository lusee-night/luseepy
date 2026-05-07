"""
Benchmark ``lusee.mapmaker.solve`` (Camacho et al. 2026 Wiener filter)
for side-by-side comparison with ``examples/optax_maxlike.py``.

Same instrument / observation / noise config as the optax script, so the
wall-clock + final-loss numbers are directly comparable.

CLI
---
    python examples/mapmaker_benchmark.py --method cg     --lmax 32
    python examples/mapmaker_benchmark.py --method direct --lmax 32

Outputs go to ``examples/mapmaker_benchmark_out/<method>_lmax<L>/``:
  - ``meta.json``          — method, lmax, wall-clock, final loss
  - ``maps_<FF>MHz.png``, ``rho_cl_<FF>MHz.png`` — per-frequency plots

``jax.scipy.sparse.linalg.cg`` is a black-box solve; there is no per-
iteration loss history to record (solver emits only a final ``info``
code). ``final_loss`` is computed from the returned alm using the same
formula as the optax script.
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Shared helpers and config.
from optax_maxlike import (
    FREQ, OBS_RANGE, DT_SEC, DF_HZ, BEAM_FILE, SKY_FILE,
    make_simulator, forward, load_truth_sky,
    radiometric_sigma, cl_inv_full, save_plots,
)


EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(EXAMPLES_DIR, "mapmaker_benchmark_out")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=("cg", "direct"), default="cg")
    p.add_argument("--lmax", type=int, default=32)
    p.add_argument("--maxiter", type=int, default=2000,
                   help="CG maxiter; ignored by direct solve")
    p.add_argument("--tol", type=float, default=1e-12, help="CG tolerance")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    import lusee

    tag = f"_{args.tag}" if args.tag else ""
    out_dir = os.path.join(OUT_ROOT, f"{args.method}_lmax{args.lmax}{tag}")
    os.makedirs(out_dir, exist_ok=True)

    sim = make_simulator(lusee, FREQ, args.lmax)
    sky_truth = load_truth_sky(lusee, FREQ, args.lmax)

    data_clean = forward(sim, sky_truth)
    sigma = radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(0), data_clean.shape)
    data = data_clean + noise

    cl_inv = cl_inv_full(sky_truth, args.lmax)

    t_start = time.time()
    sky_hat_alm = lusee.mapmaker.solve(
        sim, data, sky_truth, sigma,
        signal_prior=cl_inv, lmax=args.lmax,
        maxiter=args.maxiter, tol=args.tol,
        precondition=True, method=args.method,
    )
    jax.block_until_ready(sky_hat_alm)
    wall_clock = time.time() - t_start

    # Wrap recovered alm into a HealpixSky for plotting.
    sky_hat = lusee.sky.HealpixSky(
        sky_truth.Nside, args.lmax,
        maps=[np.zeros(12 * sky_truth.Nside ** 2)] * len(FREQ),
        freq=FREQ, frame="galactic",
    )
    sky_hat.mapalm = sky_hat_alm

    # Final loss (same definition as optax_maxlike.py):
    #   chi²(d, A m) + m^T S^{-1}_real m
    # Use RealAlmSky to get the same real-coord prior.
    params = lusee.sky.RealAlmSky.from_healpix(sky_hat, args.lmax)
    re_diag, im_diag = params.prior_inv_diag(cl_inv)
    N_inv = 1.0 / sigma ** 2
    r = data.ravel() - forward(sim, sky_hat).ravel()
    chi2 = float(jnp.sum(N_inv.ravel() * r ** 2))
    prior = float(jnp.sum(re_diag * params.re ** 2)
                  + jnp.sum(im_diag * params.im_mpos ** 2))
    final_loss = chi2 + prior

    meta = dict(
        method=args.method, lmax=args.lmax,
        maxiter=args.maxiter, tol=args.tol,
        freq=FREQ.tolist(),
        wall_clock_s=wall_clock,
        chi2=chi2, prior=prior, final_loss=final_loss,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[{args.method} lmax={args.lmax}] wall={wall_clock:.1f}s  "
          f"final_loss={final_loss:.6e}  out={out_dir}")

    save_plots(sky_truth, sky_hat, out_dir, FREQ, args.lmax)


if __name__ == "__main__":
    main()
