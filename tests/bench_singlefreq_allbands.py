"""Run single-freq CG at all 10 frequencies (not just 25 MHz) so we can
directly compare SVD K=4 vs the standard Wiener filter at each band.

The main bench only ran single-freq at 25 MHz. To claim "SVD beats the
reference Wiener filter across the band", we need the same across the
band. Saves per-freq rho curves into bench_singlefreq_allbands.npz.
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
OUT = "/home/zack/luseepy/tests/bench_singlefreq_allbands.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


def mr(r, a, b): return float(np.nanmean(r[a:b + 1]))


def main():
    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]

    alm_sf = np.zeros((len(FREQS), hp.Alm.getsize(LMAX)), dtype=complex)
    per_freq_10 = np.zeros(len(FREQS))
    per_freq_20 = np.zeros(len(FREQS))
    per_freq_32 = np.zeros(len(FREQS))

    t_total = time.time()
    for fi, f in enumerate(FREQS):
        log(f"single-freq CG at {f:.1f} MHz ...")
        t0 = time.time()
        sim, _, _ = mm.build_instrument(
            beam_file=BEAM, obs_range=OBS, freq=np.array([f]),
            lmax=LMAX, dt_sec=DT)
        sky = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=[maps[fi]],
                                    freq=np.array([f]), frame="galactic")
        data_c = sim.simulate(sky=sky)
        sigma = mm.compute_radiometric_noise(data_c, delta_f_hz=1e6, delta_t_sec=DT)
        data = data_c + sigma * jax.random.normal(jax.random.PRNGKey(42), data_c.shape)
        S_inv = mm.compute_cl_prior(sky, LMAX)
        a = np.asarray(mm.solve(sim, data, sky, sigma, signal_prior=S_inv,
                                 maxiter=800, tol=1e-10))[0]
        alm_sf[fi] = a

        truth_fi = np.asarray(sky.mapalm)[0]
        r = rho(truth_fi, a)
        per_freq_10[fi] = mr(r, 1, 10)
        per_freq_20[fi] = mr(r, 1, 20)
        per_freq_32[fi] = mr(r, 1, LMAX)
        log(f"  {f:5.1f} MHz  {time.time()-t0:.1f}s  "
            f"rho(1..10)={per_freq_10[fi]:.4f}  "
            f"rho(1..20)={per_freq_20[fi]:.4f}  "
            f"rho(1..32)={per_freq_32[fi]:.4f}")

    log(f"total: {time.time()-t_total:.1f}s")
    log("=== single-freq CG summary ===")
    for fi, f in enumerate(FREQS):
        log(f"  {f:5.1f} MHz  rho(1..20)={per_freq_20[fi]:.4f}")
    log(f"  mean rho(1..20) = {per_freq_20.mean():.4f}")

    np.savez(OUT, freqs=FREQS, lmax=LMAX, alm_sf=alm_sf,
             per_freq_10=per_freq_10, per_freq_20=per_freq_20,
             per_freq_32=per_freq_32)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
