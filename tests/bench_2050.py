"""SVD-K=4 multifreq vs per-band single-freq CG on the 20-50 MHz window.

Restricts to the smooth-synchrotron regime (above the self-absorption
turnover around 15-20 MHz). 10 equally-spaced bands in 20-50 MHz, lmax=32,
full sidereal month, seed=42.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp

import lusee
from lusee import mapmaker as mm

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"
LMAX = 32
FREQS = np.arange(21.0, 51.0, 3.0)  # 10 bands, 3 MHz spacing on canonical 1-MHz grid
DT = 7200.0
OBS = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
K_SVD = 4
OUT = "/home/zack/luseepy/tests/bench_2050.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


def mr(r, a, b): return float(np.nanmean(r[a:b + 1]))


def main():
    log(f"FREQS = {FREQS}")
    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky_mf = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=list(maps),
                                   freq=FREQS, frame="galactic")

    log("build multifreq instrument ...")
    sim_mf, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    data_mf_c = sim_mf.simulate(sky=sky_mf)
    sigma_mf = mm.compute_radiometric_noise(data_mf_c, delta_f_hz=1e6, delta_t_sec=DT)
    data_mf = data_mf_c + sigma_mf * jax.random.normal(jax.random.PRNGKey(42), data_mf_c.shape)
    truth = np.asarray(sky_mf.mapalm)

    ulsa_cube = np.asarray(sky_full.maps)[idx]
    U, S, Vt = np.linalg.svd(ulsa_cube, full_matrices=False)
    F = U[:, :K_SVD] * S[:K_SVD]
    spatial = Vt[:K_SVD]
    signs = np.sign(F.sum(axis=0)); signs[signs == 0] = 1.0
    F *= signs; spatial *= signs[:, None]
    sv = S[:K_SVD]
    log(f"singular value ratios sv[k]/sv[0]: " + ", ".join(
        f"{sv[k]/sv[0]:.2e}" for k in range(min(K_SVD, len(sv)))))
    S_inv_beta = np.zeros((K_SVD, hp.Alm.getsize(LMAX)))
    for k in range(K_SVD):
        alm_k = hp.map2alm(spatial[k], lmax=LMAX)
        cl = hp.alm2cl(alm_k)
        for l in range(LMAX + 1):
            for m in range(l + 1):
                if cl[l] > 0:
                    S_inv_beta[k, hp.Alm.getidx(LMAX, l, m)] = 1.0 / cl[l]
    S_inv_beta = jnp.asarray(S_inv_beta)

    log("SVD K=4 multifreq solve ...")
    t0 = time.time()
    beta = mm.solve_svd_multifreq(sim_mf, data_mf, sky_mf, sigma_mf, F,
                                   signal_prior=S_inv_beta, lmax=LMAX,
                                   maxiter=800, tol=1e-10)
    rec_svd = np.asarray(jnp.asarray(F) @ beta)
    t_svd = time.time() - t0
    log(f"  SVD: {t_svd:.1f}s")

    alm_sf = np.zeros_like(rec_svd)
    t_sf_total = time.time()
    for fi, f in enumerate(FREQS):
        log(f"single-freq CG at {f:.2f} MHz ...")
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
        log(f"  {f:.2f} MHz {time.time()-t0:.1f}s")
    t_sf = time.time() - t_sf_total
    log(f"SF total: {t_sf:.1f}s")

    log(f"{'MHz':>6}  {'SF.lo':>7}  {'SVD.lo':>7}  {'SF.mid':>7}  {'SVD.mid':>7}  "
        f"{'SF.hi':>7}  {'SVD.hi':>7}")
    rho_sf_all = np.zeros((len(FREQS), LMAX + 1))
    rho_svd_all = np.zeros_like(rho_sf_all)
    for fi, f in enumerate(FREQS):
        r_sf = rho(truth[fi], alm_sf[fi]); r_sv = rho(truth[fi], rec_svd[fi])
        rho_sf_all[fi] = r_sf; rho_svd_all[fi] = r_sv
        log(f"{f:>6.2f}  {mr(r_sf,1,5):>7.4f}  {mr(r_sv,1,5):>7.4f}  "
            f"{mr(r_sf,6,15):>7.4f}  {mr(r_sv,6,15):>7.4f}  "
            f"{mr(r_sf,16,LMAX):>7.4f}  {mr(r_sv,16,LMAX):>7.4f}")
    def bandmean(arr, a, b):
        return float(np.mean([mr(arr[fi], a, b) for fi in range(len(FREQS))]))
    log(f"  MEAN   lo  SF={bandmean(rho_sf_all,1,5):.4f}   SVD={bandmean(rho_svd_all,1,5):.4f}")
    log(f"  MEAN   mid SF={bandmean(rho_sf_all,6,15):.4f}   SVD={bandmean(rho_svd_all,6,15):.4f}")
    log(f"  MEAN   hi  SF={bandmean(rho_sf_all,16,LMAX):.4f}   SVD={bandmean(rho_svd_all,16,LMAX):.4f}")

    np.savez(OUT, freqs=FREQS, lmax=LMAX, K=K_SVD,
             truth=truth, rec_svd=rec_svd, alm_sf=alm_sf,
             rho_sf=rho_sf_all, rho_svd=rho_svd_all,
             sv=sv, t_svd=t_svd, t_sf=t_sf)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
