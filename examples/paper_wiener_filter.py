"""
Reproduce Camacho et al. 2026 (arXiv:2508.16773) Wiener filter map-making
using lusee.mapmaker (autodiff CG instead of explicit matrix construction).

Usage:
    LUSEE_DRIVE_DIR=/path/to/LuSEE-Night python examples/paper_wiener_filter.py
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import jax
import jax.numpy as jnp
import numpy as np
import lusee
import healpy as hp

# ── Config ───────────────────────────────────────────────────────────

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM_FILE = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

# Paper values: lmax=47, all 50 freqs, full cycle
# Dev values for speed:
LMAX = 32
FREQ = np.array([25.0])
OBS_RANGE = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"

# ── 1. Build instrument ─────────────────────────────────────────────

sim, beams, obs = lusee.mapmaker.build_instrument(
    beam_file=BEAM_FILE,
    obs_range=OBS_RANGE,
    freq=FREQ,
    lmax=LMAX,
)

# ── 2. Load sky and simulate data ───────────────────────────────────

sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=LMAX)
freq_indices = [int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQ]
sky = lusee.sky.HealpixSky(
    sky_full.Nside, LMAX,
    maps=[hp.alm2map(np.asarray(sky_full.mapalm[fi]), sky_full.Nside, verbose=False)
          for fi in freq_indices],
    freq=FREQ, frame="galactic",
)

t0 = time.time()
data_clean = sim.simulate(sky=sky)
print(f"Simulated in {time.time() - t0:.1f}s, data shape = {data_clean.shape}")

sigma = 1.0
noise = sigma * jax.random.normal(jax.random.PRNGKey(42), data_clean.shape)
data = data_clean + noise
print(f"Noise sigma = {sigma} K, SNR ~ {float(jnp.std(data_clean)) / sigma:.0f}")

# ── 3. Solve ─────────────────────────────────────────────────────────

S_inv = lusee.mapmaker.compute_cl_prior(sky, LMAX)

t0 = time.time()
sky_hat = lusee.mapmaker.solve(
    sim, data, sky, sigma,
    signal_prior=S_inv, maxiter=50, tol=1e-8,
)
print(f"Solved in {time.time() - t0:.1f}s")

# ── 4. Evaluate ─────────────────────────────────────────────────────

for fi, f in enumerate(FREQ):
    true_alm = np.asarray(sky.mapalm[fi])
    rec_alm = np.asarray(sky_hat[fi])
    rho_l = hp.alm2cl(true_alm, rec_alm) / np.sqrt(
        hp.alm2cl(true_alm) * hp.alm2cl(rec_alm) + 1e-30)

    print(f"\n{f:.0f} MHz:")
    for l in range(min(11, LMAX + 1)):
        print(f"  l={l:2d}: rho={rho_l[l]:.4f}")
    print(f"  mean rho(1..10) = {np.nanmean(rho_l[1:11]):.4f}")

# ── 5. Plot ──────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for fi, f in enumerate(FREQ):
        true_alm = np.asarray(sky.mapalm[fi])
        rec_alm = np.asarray(sky_hat[fi])
        true_map = hp.alm2map(true_alm, sky.Nside, verbose=False)
        rec_map = hp.alm2map(rec_alm, sky.Nside, verbose=False)
        resid = rec_map - true_map

        fig = plt.figure(figsize=(15, 4))
        for i, (m, title) in enumerate([(true_map, f"Input sky ({f:.0f} MHz)"),
                                         (rec_map, "Recovered"),
                                         (resid, "Residual")]):
            hp.mollview(m, title=title, hold=True, sub=(1, 3, i + 1),
                        cmap="inferno" if i < 2 else "RdBu_r")

    out = os.path.join(os.path.dirname(__file__), "paper_wiener_filter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
except Exception as e:
    print(f"\nPlot failed: {e}")
