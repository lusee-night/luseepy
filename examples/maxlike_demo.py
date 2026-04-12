"""
Maximum-likelihood sky recovery (mapmaking) via conjugate gradients.

Uses the lusee.mapmaker module which handles beam rotation, horizon
taper, and the conj(vjp) adjoint correctly.  See docs/wirtinger_cg.md.
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import jax.numpy as jnp
import numpy as np
import lusee
import healpy as hp

# ── Setup ────────────────────────────────────────────────────────────

nside = 8
lmax = 2
freq = np.array([10.0])

beams = [
    lusee.BeamGauss(alt_deg=90, az_deg=0,  sigma_deg=20, one_over_freq_scaling=False, id="b0"),
    lusee.BeamGauss(alt_deg=60, az_deg=0,  sigma_deg=20, one_over_freq_scaling=False, id="b1"),
    lusee.BeamGauss(alt_deg=60, az_deg=90, sigma_deg=20, one_over_freq_scaling=False, id="b2"),
]

npix = 12 * nside**2
theta, phi = hp.pix2ang(nside, np.arange(npix))
true_map = (1000.0
            + 200.0 * np.cos(theta)
            + 100.0 * (3 * np.cos(theta)**2 - 1) / 2)
sky_true = lusee.sky.HealpixSky(nside, lmax, [true_map], freq=freq, frame="galactic")

# ── Simulate ─────────────────────────────────────────────────────────

obs = lusee.Observation(
    "2025-03-01 00:00:00 to 2025-03-02 00:00:00",
    deltaT_sec=1800.0, lun_lat_deg=45.0, lun_long_deg=0.0,
)
sim = lusee.CroSimulator(
    obs, beams, sky_true, Tground=0.0,
    combinations=[(0, 0), (1, 1), (2, 2)], freq=freq, lmax=lmax,
)

data_clean = sim.simulate(sky=sky_true)
sigma = 0.01
import jax
data = data_clean + sigma * jax.random.normal(
    jax.random.PRNGKey(42), data_clean.shape)

print(f"Data: {data.size} measurements, SNR ~ {float(jnp.std(data_clean)) / sigma:.0f}")

# ── Solve ────────────────────────────────────────────────────────────

t0 = time.time()
sky_hat = lusee.mapmaker.solve(sim, data, sky_true, sigma, maxiter=50, tol=1e-10)
print(f"Solved in {time.time() - t0:.1f}s")

# ── Evaluate ─────────────────────────────────────────────────────────

true_alm = np.asarray(sky_true.mapalm[0])
rec_alm = np.asarray(sky_hat[0])
true_map_r = hp.alm2map(true_alm, nside, verbose=False)
rec_map = hp.alm2map(rec_alm, nside, verbose=False)
resid = rec_map - true_map_r

print(f"Relative map error: {np.std(resid) / np.std(true_map_r):.4f}")
print(f"Waterfall residual: {np.std(np.asarray(sim.simulate(sky=lusee.sky.HealpixSky.tree_unflatten(sky_true.tree_flatten()[1], (sky_hat,))).ravel()) - np.asarray(data.ravel())):.4f} K  (noise = {sigma} K)")
