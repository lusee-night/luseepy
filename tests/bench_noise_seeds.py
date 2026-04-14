"""Noise-seed robustness check for the high-ell SVD K=4 advantage.

Builds the 10-freq instrument once, then for each of 3 noise seeds compares
single-freq CG vs SVD K=4 at the reference frequency (25 MHz), reporting
rho in low/mid/high ell bins. If SVD wins high-ell for every seed, the
claim is robust; if it's seed-dependent we got lucky.

~30 min total: ~5 min build, ~5 min per single-freq CG x 3, ~2 min SVD x 3.
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
FREQS = np.arange(5.0, 51.0, 5.0)
DT = 7200.0
OBS = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
K_SVD = 4
SEEDS = [42, 123, 777]
REF_F = 25.0
OUT = "/home/zack/luseepy/tests/bench_noise_seeds.npz"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


def main():
    log("build multifreq instrument ...")
    t0 = time.time()
    sim_mf, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    log(f"  {time.time()-t0:.1f}s")

    log("build single-freq instrument at 25 MHz ...")
    t0 = time.time()
    sim_sf, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=np.array([REF_F]),
        lmax=LMAX, dt_sec=DT)
    log(f"  {time.time()-t0:.1f}s")

    # skies + truth
    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky_mf = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=list(maps),
                                   freq=FREQS, frame="galactic")
    ref_fi = int(np.argmin(np.abs(FREQS - REF_F)))
    sky_sf = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=[maps[ref_fi]],
                                   freq=np.array([REF_F]), frame="galactic")
    truth = np.asarray(sky_mf.mapalm)[ref_fi]

    # priors (shared across seeds)
    S_inv_mf = mm.compute_cl_prior(sky_mf, LMAX)
    S_inv_sf = mm.compute_cl_prior(sky_sf, LMAX)
    F, spatial, sv = mm.compute_ulsa_svd(SKY, FREQS, K=K_SVD, lmax=LMAX)
    S_inv_beta = np.zeros((K_SVD, hp.Alm.getsize(LMAX)))
    for k in range(K_SVD):
        alm_k = hp.map2alm(spatial[k], lmax=LMAX)
        cl = hp.alm2cl(alm_k)
        for l in range(LMAX + 1):
            for m in range(l + 1):
                if cl[l] > 0:
                    S_inv_beta[k, hp.Alm.getidx(LMAX, l, m)] = 1.0 / cl[l]
    S_inv_beta = jnp.asarray(S_inv_beta)

    # cached clean forwards
    data_mf_clean = sim_mf.simulate(sky=sky_mf)
    sigma_mf = mm.compute_radiometric_noise(data_mf_clean, delta_f_hz=1e6, delta_t_sec=DT)
    data_sf_clean = sim_sf.simulate(sky=sky_sf)
    sigma_sf = mm.compute_radiometric_noise(data_sf_clean, delta_f_hz=1e6, delta_t_sec=DT)

    results = {}
    for seed in SEEDS:
        log(f"=== seed {seed} ===")
        key_mf, key_sf = jax.random.split(jax.random.PRNGKey(seed))
        data_mf = data_mf_clean + sigma_mf * jax.random.normal(key_mf, data_mf_clean.shape)
        data_sf = data_sf_clean + sigma_sf * jax.random.normal(key_sf, data_sf_clean.shape)

        log("  SVD K=4 ...")
        t0 = time.time()
        beta = mm.solve_svd_multifreq(sim_mf, data_mf, sky_mf, sigma_mf, F,
                                       signal_prior=S_inv_beta, lmax=LMAX,
                                       maxiter=800, tol=1e-10)
        rec_svd = np.asarray(jnp.asarray(F) @ beta)[ref_fi]
        log(f"    {time.time()-t0:.1f}s")

        log("  single-freq CG at 25 MHz ...")
        t0 = time.time()
        rec_sf = np.asarray(mm.solve(sim_sf, data_sf, sky_sf, sigma_sf,
                                      signal_prior=S_inv_sf,
                                      maxiter=800, tol=1e-10))[0]
        log(f"    {time.time()-t0:.1f}s")

        r_sf = rho(truth, rec_sf); r_sv = rho(truth, rec_svd)
        bins = [(1, 5), (6, 15), (16, LMAX)]
        sf_b = [float(np.nanmean(r_sf[a:b + 1])) for a, b in bins]
        sv_b = [float(np.nanmean(r_sv[a:b + 1])) for a, b in bins]
        results[seed] = dict(rho_sf=r_sf, rho_svd=r_sv, sf_b=sf_b, sv_b=sv_b)
        log(f"  single-freq: lo {sf_b[0]:.4f}  mid {sf_b[1]:.4f}  hi {sf_b[2]:.4f}")
        log(f"  SVD K=4   : lo {sv_b[0]:.4f}  mid {sv_b[1]:.4f}  hi {sv_b[2]:.4f}")
        log(f"  delta hi  : {sv_b[2] - sf_b[2]:+.4f}")

    log("\n=== summary: rho(16..32) at 25 MHz ===")
    log(f"{'seed':>6}  {'single-freq':>12}  {'SVD K=4':>10}  {'delta':>8}")
    for seed in SEEDS:
        r = results[seed]
        log(f"{seed:>6}  {r['sf_b'][2]:>12.4f}  {r['sv_b'][2]:>10.4f}  "
            f"{r['sv_b'][2] - r['sf_b'][2]:>+8.4f}")

    np.savez(OUT, seeds=np.array(SEEDS), ref_f=REF_F, lmax=LMAX,
             **{f"rho_sf_{s}": results[s]['rho_sf'] for s in SEEDS},
             **{f"rho_svd_{s}": results[s]['rho_svd'] for s in SEEDS})
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
