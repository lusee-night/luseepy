"""K-sweep of SVD-subspace multifreq solver at full scale.

Tests whether more templates (K=2,3,4,5,6) improve the SVD reconstruction,
or whether K=3 is already capturing everything that matters. Reuses the
same instrument/data as bench_multifreq_full_scale.py for apples-to-apples
comparison with the K=3 result already on disk.
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
KS = [2, 3, 4, 5, 6]
OUT = "/home/zack/luseepy/tests/bench_svd_ksweep.npz"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def rho(a, b):
    return hp.alm2cl(np.asarray(a), np.asarray(b)) / np.sqrt(
        hp.alm2cl(np.asarray(a)) * hp.alm2cl(np.asarray(b)) + 1e-30)


def mr(r, lo, hi): return float(np.nanmean(r[lo:hi + 1]))


def main():
    log(f"building instrument (lmax={LMAX}, nfreq={len(FREQS)}) ...")
    t0 = time.time()
    sim, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    log(f"  {time.time() - t0:.1f}s")

    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=list(maps),
                                freq=FREQS, frame="galactic")
    data_c = sim.simulate(sky=sky)
    sigma = mm.compute_radiometric_noise(data_c, delta_f_hz=1e6, delta_t_sec=DT)
    data = data_c + sigma * jax.random.normal(jax.random.PRNGKey(42), data_c.shape)
    truth = np.asarray(sky.mapalm)

    results = {}
    for K in KS:
        log(f"SVD K={K} ...")
        F, spatial, sv = mm.compute_ulsa_svd(SKY, FREQS, K=K, lmax=LMAX)
        S_inv_beta = np.zeros((K, hp.Alm.getsize(LMAX)))
        for k in range(K):
            alm_k = hp.map2alm(spatial[k], lmax=LMAX)
            cl = hp.alm2cl(alm_k)
            for l in range(LMAX + 1):
                for m in range(l + 1):
                    if cl[l] > 0:
                        S_inv_beta[k, hp.Alm.getidx(LMAX, l, m)] = 1.0 / cl[l]
        S_inv_beta = jnp.asarray(S_inv_beta)

        t0 = time.time()
        beta = mm.solve_svd_multifreq(sim, data, sky, sigma, F,
                                       signal_prior=S_inv_beta, lmax=LMAX,
                                       maxiter=800, tol=1e-10)
        t = time.time() - t0
        rec = np.asarray(jnp.asarray(F) @ beta)

        per_freq = np.array([mr(rho(truth[fi], rec[fi]), 1, 20) for fi in range(len(FREQS))])
        ref_fi = int(np.argmin(np.abs(FREQS - 25.0)))
        r_ref = rho(truth[ref_fi], rec[ref_fi])
        log(f"  K={K}: {t:.1f}s  mean rho(1..20) = {per_freq.mean():.4f}  "
            f"rho(1..32) @ 25 MHz = {mr(r_ref, 1, LMAX):.4f}")
        results[K] = dict(per_freq=per_freq, r_ref=r_ref, t=t, sv=sv, rec=rec)

    log("\n=== K sweep summary ===")
    log(f"{'K':>3}  {'time':>6}  {'<rho(1..20)>':>12}  {'rho(1..32)@25':>15}  {'sigma_K/sigma_0':>15}")
    for K in KS:
        r = results[K]
        sv_ratio = r['sv'][-1] / r['sv'][0]
        log(f"{K:>3}  {r['t']:>5.1f}s  {r['per_freq'].mean():>12.4f}  "
            f"{mr(r['r_ref'], 1, LMAX):>15.4f}  {sv_ratio:>15.3e}")

    np.savez(OUT, freqs=FREQS, lmax=LMAX, Ks=np.array(KS), truth_alm=truth,
             **{f"rec_K{K}": results[K]['rec'] for K in KS},
             **{f"per_freq_K{K}": results[K]['per_freq'] for K in KS},
             **{f"r_ref_K{K}": results[K]['r_ref'] for K in KS},
             **{f"t_K{K}": results[K]['t'] for K in KS})
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
