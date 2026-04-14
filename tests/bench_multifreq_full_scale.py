"""Full-scale multifrequency Wiener filter benchmark.

Goal: demonstrate that the SVD-subspace multifreq solve outperforms the
single-frequency Wiener filter baseline (reproduction of Camacho+ 2026)
at its own scale: lmax=32, 325-timestep full sidereal month at dt_sec=7200s,
10 frequencies spanning the LuSEE band.

Compares three solvers at 25 MHz:
  1. single-freq CG (reference, matches notebooks/mapmaker_demo.ipynb)
  2. 10-freq full CG (freq-by-freq stacked)
  3. 10-freq SVD-subspace CG with K=3 (ULSA-derived templates)

Writes metrics + cl arrays to bench_multifreq_full_scale.npz for plotting.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp

import lusee
from lusee import mapmaker as mm

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM_FILE = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

LMAX = 32
DT_SEC = 7200.0
OBS = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"   # full sidereal rotation
FREQS = np.arange(5.0, 51.0, 5.0)                     # 10 freqs on canonical grid (5..50 MHz)
K_SVD = 3
OUT = "/home/zack/luseepy/tests/bench_multifreq_full_scale.npz"


def rho_per_ell(true_alm, rec):
    cl_t = hp.alm2cl(np.asarray(true_alm))
    cl_r = hp.alm2cl(np.asarray(rec))
    cl_c = hp.alm2cl(np.asarray(true_alm), np.asarray(rec))
    return cl_c / np.sqrt(cl_t * cl_r + 1e-30)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # ---- shared: load ULSA cube at these frequencies ----
    log(f"Loading ULSA at lmax={LMAX} ...")
    t0 = time.time()
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=LMAX)
    sky_freq = np.asarray(sky_full.freq, dtype=float)
    idx = np.array([int(np.argmin(np.abs(sky_freq - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky_multi = lusee.sky.HealpixSky(
        sky_full.Nside, LMAX,
        maps=[m for m in maps], freq=FREQS, frame="galactic",
    )
    log(f"  loaded in {time.time()-t0:.1f}s (Nside={sky_full.Nside})")

    # ---- Build 10-freq instrument (beams cached) ----
    log(f"Building 10-freq instrument ({len(FREQS)} freqs, lmax={LMAX}) ...")
    t0 = time.time()
    sim_multi, _, obs = mm.build_instrument(
        beam_file=BEAM_FILE, obs_range=OBS, freq=FREQS,
        lmax=LMAX, dt_sec=DT_SEC,
    )
    log(f"  instrument built in {time.time()-t0:.1f}s, ntimes={len(obs.times)}")

    # ---- Simulate data + radiometric noise ----
    log("Simulating noiseless waterfall ...")
    t0 = time.time()
    data_clean = sim_multi.simulate(sky=sky_multi)
    sigma_multi = mm.compute_radiometric_noise(
        data_clean, delta_f_hz=1e6, delta_t_sec=DT_SEC)
    data_multi = data_clean + sigma_multi * jax.random.normal(
        jax.random.PRNGKey(42), data_clean.shape)
    log(f"  simulate+noise: {time.time()-t0:.1f}s, data shape={data_multi.shape}")

    # ---- Priors ----
    S_inv_full = mm.compute_cl_prior(sky_multi, LMAX)
    F, spatial, sv = mm.compute_ulsa_svd(SKY_FILE, FREQS, K=K_SVD, lmax=LMAX)
    log(f"  SVD ratios: {sv[1]/sv[0]:.3e}, {sv[2]/sv[0]:.3e}")
    S_inv_beta = np.zeros((K_SVD, hp.Alm.getsize(LMAX)))
    for k in range(K_SVD):
        alm_k = hp.map2alm(spatial[k], lmax=LMAX)
        cl = hp.alm2cl(alm_k)
        for l in range(LMAX + 1):
            for m in range(l + 1):
                if cl[l] > 0:
                    S_inv_beta[k, hp.Alm.getidx(LMAX, l, m)] = 1.0 / cl[l]
    S_inv_beta = jnp.asarray(S_inv_beta)

    # ---- Solver 1: full 10-freq freq-by-freq CG ----
    log("10-freq full CG solve ...")
    t0 = time.time()
    alm_full = mm.solve(sim_multi, data_multi, sky_multi, sigma_multi,
                        signal_prior=S_inv_full, maxiter=800, tol=1e-10)
    t_full = time.time() - t0
    log(f"  full CG: {t_full:.1f}s")
    alm_full_np = np.asarray(alm_full)

    # ---- Solver 2: 10-freq SVD-subspace ----
    log(f"10-freq SVD-subspace solve (K={K_SVD}) ...")
    t0 = time.time()
    beta_hat = mm.solve_svd_multifreq(
        sim_multi, data_multi, sky_multi, sigma_multi, F,
        signal_prior=S_inv_beta, lmax=LMAX, maxiter=800, tol=1e-10,
    )
    t_svd = time.time() - t0
    rec_alm_svd = np.asarray(jnp.asarray(F) @ beta_hat)
    log(f"  SVD: {t_svd:.1f}s (n_theta={beta_hat.size})")

    # ---- Solver 3: single-freq CG at 25 MHz (reference) ----
    # pick freq closest to 25 MHz
    ref_fi = int(np.argmin(np.abs(FREQS - 25.0)))
    f_ref = FREQS[ref_fi]
    log(f"Single-freq reference at {f_ref:.1f} MHz ...")
    t0 = time.time()
    sim_sf, _, _ = mm.build_instrument(
        beam_file=BEAM_FILE, obs_range=OBS,
        freq=np.array([f_ref]), lmax=LMAX, dt_sec=DT_SEC,
    )
    sky_sf = lusee.sky.HealpixSky(
        sky_full.Nside, LMAX,
        maps=[maps[ref_fi]], freq=np.array([f_ref]), frame="galactic",
    )
    data_sf_clean = sim_sf.simulate(sky=sky_sf)
    sigma_sf = mm.compute_radiometric_noise(
        data_sf_clean, delta_f_hz=1e6, delta_t_sec=DT_SEC)
    data_sf = data_sf_clean + sigma_sf * jax.random.normal(
        jax.random.PRNGKey(42), data_sf_clean.shape)
    S_inv_sf = mm.compute_cl_prior(sky_sf, LMAX)
    alm_sf = mm.solve(sim_sf, data_sf, sky_sf, sigma_sf,
                      signal_prior=S_inv_sf, maxiter=800, tol=1e-10)
    t_sf = time.time() - t0
    alm_sf_np = np.asarray(alm_sf)[0]
    log(f"  single-freq: {t_sf:.1f}s")

    # ---- Metrics ----
    log("=== Metrics ===")
    truths = np.asarray(sky_multi.mapalm)
    rho_sf = rho_per_ell(truths[ref_fi], alm_sf_np)
    rho_full_ref = rho_per_ell(truths[ref_fi], alm_full_np[ref_fi])
    rho_svd_ref  = rho_per_ell(truths[ref_fi], rec_alm_svd[ref_fi])

    def mr(r, lo, hi): return float(np.nanmean(r[lo:hi+1]))
    log(f"At {f_ref:.1f} MHz, mean rho(1..10) / rho(1..20) / rho(1..32):")
    log(f"  single-freq:  {mr(rho_sf, 1, 10):.4f} / {mr(rho_sf, 1, 20):.4f} / {mr(rho_sf, 1, LMAX):.4f}")
    log(f"  full CG:      {mr(rho_full_ref, 1, 10):.4f} / {mr(rho_full_ref, 1, 20):.4f} / {mr(rho_full_ref, 1, LMAX):.4f}")
    log(f"  SVD K={K_SVD}:      {mr(rho_svd_ref, 1, 10):.4f} / {mr(rho_svd_ref, 1, 20):.4f} / {mr(rho_svd_ref, 1, LMAX):.4f}")

    # per-frequency summary
    rho_all = np.zeros((3, len(FREQS), LMAX + 1))
    for fi in range(len(FREQS)):
        rho_all[0, fi] = rho_per_ell(truths[fi], alm_full_np[fi])
        rho_all[1, fi] = rho_per_ell(truths[fi], rec_alm_svd[fi])
    rho_all[2, ref_fi] = rho_sf

    log("Per-freq mean rho(1..20):")
    for fi, f in enumerate(FREQS):
        log(f"  {f:5.1f} MHz  full={mr(rho_all[0,fi],1,20):.4f}  SVD={mr(rho_all[1,fi],1,20):.4f}")

    np.savez(OUT,
             freqs=FREQS, lmax=LMAX,
             truth_alm=truths, alm_full=alm_full_np,
             alm_svd=rec_alm_svd, alm_sf=alm_sf_np, ref_fi=ref_fi,
             t_full=t_full, t_svd=t_svd, t_sf=t_sf,
             sv=sv, rho_all=rho_all)
    log(f"Saved {OUT}")


if __name__ == "__main__":
    main()
