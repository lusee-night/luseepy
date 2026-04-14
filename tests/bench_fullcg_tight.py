"""Rerun the 10-freq full CG with tight tolerance for a fair SVD vs full-CG story.

The headline bench runs full CG with tol=1e-10, maxiter=800. JAX CG uses
relative tol, so the 10-freq RHS (~sqrt(10)x larger) leaves per-block
accuracy looser than a single-freq solve at the same tol. Bumping to
tol=1e-14, maxiter=3000 closes that artifact.

Saves into bench_fullcg_tight.npz so the demo notebook can add a fair row.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
import numpy as np
import jax
import healpy as hp

import lusee
from lusee import mapmaker as mm

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"
LMAX = 32
FREQS = np.arange(5.0, 51.0, 5.0)
DT = 7200.0
OBS = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
OUT = "/home/zack/luseepy/tests/bench_fullcg_tight.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


def mr(r, a, b): return float(np.nanmean(r[a:b + 1]))


def main():
    log("building ...")
    sim, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=list(maps),
                                freq=FREQS, frame="galactic")
    data_c = sim.simulate(sky=sky)
    sigma = mm.compute_radiometric_noise(data_c, delta_f_hz=1e6, delta_t_sec=DT)
    data = data_c + sigma * jax.random.normal(jax.random.PRNGKey(42), data_c.shape)
    truth = np.asarray(sky.mapalm)
    S_inv = mm.compute_cl_prior(sky, LMAX)

    log("full CG tight (tol=1e-14, maxiter=3000) ...")
    t0 = time.time()
    alm = np.asarray(mm.solve(sim, data, sky, sigma, signal_prior=S_inv,
                               maxiter=3000, tol=1e-14))
    t_tight = time.time() - t0
    log(f"  {t_tight:.1f}s")

    log("per-freq mean rho(1..20):")
    per_freq = np.zeros(len(FREQS))
    for fi, f in enumerate(FREQS):
        r = rho(truth[fi], alm[fi])
        per_freq[fi] = mr(r, 1, 20)
        log(f"  {f:5.1f} MHz  {per_freq[fi]:.4f}")

    ref_fi = int(np.argmin(np.abs(FREQS - 25.0)))
    r_ref = rho(truth[ref_fi], alm[ref_fi])
    log(f"At 25 MHz: rho(1..10)={mr(r_ref,1,10):.4f}  "
        f"rho(1..20)={mr(r_ref,1,20):.4f}  rho(1..32)={mr(r_ref,1,LMAX):.4f}")

    np.savez(OUT, freqs=FREQS, lmax=LMAX, alm_full_tight=alm,
             per_freq=per_freq, r_ref=r_ref, t=t_tight)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
