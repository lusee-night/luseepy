"""End-to-end benchmark of Sampler + SVD-multifreq on real CroSimulator + ULSA.

Runs at small lmax on a real forward model and reports recovery metrics.
Run with LUSEE_DRIVE_DIR set (or defaults to /fs/zack/LuSEE-Night/).
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
from lusee.Sampler import sample_posterior, sample_constrained_realization

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM_FILE = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"


def build_single_freq(lmax=6, freq_mhz=25.0, dt_sec=7200.0,
                      obs_range="2025-02-01 13:00:00 to 2025-02-14 13:00:00"):
    freq = np.array([freq_mhz])
    sim, beams, obs = mm.build_instrument(
        beam_file=BEAM_FILE, obs_range=obs_range, freq=freq, lmax=lmax,
        dt_sec=dt_sec,
    )
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    fi = int(np.argmin(np.abs(np.asarray(sky_full.freq) - freq_mhz)))
    sky = lusee.sky.HealpixSky(
        sky_full.Nside, lmax,
        maps=[hp.alm2map(np.asarray(sky_full.mapalm[fi]), sky_full.Nside, verbose=False)],
        freq=freq, frame="galactic",
    )
    data_clean = sim.simulate(sky=sky)
    sigma = mm.compute_radiometric_noise(
        data_clean, delta_f_hz=1e6, delta_t_sec=dt_sec)
    key = jax.random.PRNGKey(42)
    data = data_clean + sigma * jax.random.normal(key, data_clean.shape)
    return sim, sky, data, sigma


def rho_metrics(true_alm, rec_alm, lmax):
    cl_t = hp.alm2cl(np.asarray(true_alm))
    cl_r = hp.alm2cl(np.asarray(rec_alm))
    cl_c = hp.alm2cl(np.asarray(true_alm), np.asarray(rec_alm))
    rho = cl_c / np.sqrt(cl_t * cl_r + 1e-30)
    return rho


def bench_sampler(lmax=4):
    print(f"\n=== NUTS sampler benchmark (lmax={lmax}) ===")
    t0 = time.time()
    sim, sky, data, sigma = build_single_freq(lmax=lmax)
    print(f"  built instrument in {time.time()-t0:.1f}s; "
          f"ntimes={data.shape[0]}, nch={data.shape[1]}")
    S_inv = mm.compute_cl_prior(sky, lmax)

    # MAP reference via CG
    t0 = time.time()
    alm_map = mm.solve(sim, data, sky, sigma, signal_prior=S_inv,
                       maxiter=1000, tol=1e-10)
    t_map = time.time() - t0
    rho_map = rho_metrics(sky.mapalm[0], alm_map[0], lmax)
    print(f"  MAP solve: {t_map:.1f}s, mean rho(1..{lmax}) = {np.nanmean(rho_map[1:lmax+1]):.4f}")

    # NUTS posterior
    # Seed init at MAP to skip burn-in
    # Convert alm MAP -> theta via same packing
    m0 = np.array([hp.Alm.getidx(lmax, l, 0) for l in range(lmax + 1)])
    nalm = hp.Alm.getsize(lmax)
    mpos = np.array([i for i in range(nalm) if i not in m0])
    alm_map_np = np.asarray(alm_map[0])
    theta0 = np.concatenate([alm_map_np.real, alm_map_np.imag[mpos]])
    theta0 = jnp.asarray(theta0.reshape(1, -1))

    t0 = time.time()
    N_SAMPLES, N_WARMUP = 150, 150
    samples_alm, infos = sample_posterior(
        sim, data, sky, sigma, signal_prior=S_inv, lmax=lmax,
        num_samples=N_SAMPLES, num_warmup=N_WARMUP,
        seed=7, init_theta=theta0, return_info=True,
    )
    t_nuts = time.time() - t0
    accept = float(np.mean([float(i.acceptance_rate) for i in infos]))
    print(f"  NUTS: {t_nuts:.1f}s for {N_WARMUP}+{N_SAMPLES}, accept={accept:.2f}")

    # Posterior stats
    samp_np = np.asarray(samples_alm)[:, 0, :]       # (N, nalm) complex
    post_mean = samp_np.mean(axis=0)
    # Compare post_mean vs MAP (should match for Gaussian)
    err = np.linalg.norm(post_mean - alm_map_np) / np.linalg.norm(alm_map_np)
    print(f"  <sample> - MAP relerr = {err:.3e}")
    rho_pmean = rho_metrics(sky.mapalm[0], post_mean, lmax)
    print(f"  posterior-mean rho(1..{lmax}) = {np.nanmean(rho_pmean[1:lmax+1]):.4f}")
    # Constrained realization: Gaussian-exact
    t0 = time.time()
    cr_alm = sample_constrained_realization(
        sim, data, sky, sigma, signal_prior=S_inv, lmax=lmax,
        num_samples=20, seed=11, maxiter=1000, tol=1e-10,
    )
    t_cr = time.time() - t0
    cr_np = np.asarray(cr_alm)[:, 0, :]
    cr_mean = cr_np.mean(axis=0)
    err_cr = np.linalg.norm(cr_mean - alm_map_np) / np.linalg.norm(alm_map_np)
    print(f"  CR: {t_cr:.1f}s for 20 indep samples, <CR>-MAP relerr={err_cr:.2e}")

    return dict(t_map=t_map, t_nuts=t_nuts, accept=accept, t_cr=t_cr,
                mean_vs_map=err, mean_vs_map_cr=err_cr,
                rho_map=rho_map, rho_pmean=rho_pmean)


def bench_svd(lmax=6, K=3, freqs_mhz=(15.0, 20.0, 25.0, 30.0, 35.0, 40.0)):
    print(f"\n=== SVD multifreq benchmark (lmax={lmax}, K={K}, nfreq={len(freqs_mhz)}) ===")
    freq = np.asarray(freqs_mhz, dtype=float)
    t0 = time.time()
    sim, beams, obs = mm.build_instrument(
        beam_file=BEAM_FILE,
        obs_range="2025-02-01 13:00:00 to 2025-02-14 13:00:00",
        freq=freq, lmax=lmax, dt_sec=7200.0,
    )
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    sky_freq = np.asarray(sky_full.freq, dtype=float)
    idx = np.array([int(np.argmin(np.abs(sky_freq - f))) for f in freq])
    maps = np.asarray(sky_full.maps)[idx]
    sky = lusee.sky.HealpixSky(sky_full.Nside, lmax,
                                maps=[m for m in maps],
                                freq=freq, frame="galactic")
    data_clean = sim.simulate(sky=sky)
    sigma = mm.compute_radiometric_noise(
        data_clean, delta_f_hz=1e6, delta_t_sec=7200.0)
    data = data_clean + sigma * jax.random.normal(
        jax.random.PRNGKey(3), data_clean.shape)
    print(f"  built instrument+data in {time.time()-t0:.1f}s")

    # SVD templates from the ULSA cube at these freqs
    F, spatial, S_vals = mm.compute_ulsa_svd(SKY_FILE, freq, K=K, lmax=lmax)
    print(f"  singular values: {S_vals}")

    # Prior on beta_k: use C_l of projected ULSA-cube onto mode k
    # spatial[k] is a pixel map; take its alm and compute C_l
    Cl_beta = []
    for k in range(K):
        alm_k = hp.map2alm(spatial[k], lmax=lmax)
        Cl_beta.append(hp.alm2cl(alm_k))
    S_inv_beta = np.zeros((K, hp.Alm.getsize(lmax)))
    for k in range(K):
        cl = Cl_beta[k]
        for l in range(lmax + 1):
            for m in range(l + 1):
                idx_lm = hp.Alm.getidx(lmax, l, m)
                if cl[l] > 0:
                    S_inv_beta[k, idx_lm] = 1.0 / cl[l]
    S_inv_beta = jnp.asarray(S_inv_beta)

    t0 = time.time()
    beta_hat = mm.solve_svd_multifreq(
        sim, data, sky, sigma, F, signal_prior=S_inv_beta, lmax=lmax,
        maxiter=1000, tol=1e-10,
    )
    t_svd = time.time() - t0

    # Reconstruct full-freq alm cube from beta
    rec_alm = jnp.asarray(F) @ beta_hat
    # Fidelity per frequency
    rhos = []
    for fi in range(len(freq)):
        r = rho_metrics(sky.mapalm[fi], rec_alm[fi], lmax)
        rhos.append(np.nanmean(r[1:lmax+1]))
    print(f"  SVD solve: {t_svd:.1f}s; mean rho per freq = "
          + ", ".join(f"{r:.3f}" for r in rhos))

    # Compare against full freq-by-freq CG solve as baseline
    t0 = time.time()
    S_inv_full = mm.compute_cl_prior(sky, lmax)
    alm_full = mm.solve(sim, data, sky, sigma, signal_prior=S_inv_full,
                        maxiter=1000, tol=1e-10)
    t_full = time.time() - t0
    rhos_full = []
    for fi in range(len(freq)):
        r = rho_metrics(sky.mapalm[fi], alm_full[fi], lmax)
        rhos_full.append(np.nanmean(r[1:lmax+1]))
    print(f"  full CG:   {t_full:.1f}s; mean rho per freq = "
          + ", ".join(f"{r:.3f}" for r in rhos_full))
    return dict(t_svd=t_svd, t_full=t_full, rhos_svd=rhos, rhos_full=rhos_full)


if __name__ == "__main__":
    r1 = bench_sampler(lmax=4)
    r2 = bench_svd(lmax=6, K=3)
    print("\n=== Summary ===")
    print(f"NUTS: sampler reproduces MAP within relerr={r1['mean_vs_map']:.2e}, "
          f"accept={r1['accept']:.2f}, rho_pmean={np.nanmean(r1['rho_pmean'][1:5]):.3f}")
    print(f"SVD : mean rho {np.mean(r2['rhos_svd']):.3f} vs full CG {np.mean(r2['rhos_full']):.3f}; "
          f"speedup {r2['t_full']/r2['t_svd']:.1f}x")
