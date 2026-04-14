"""Diagnose why 10-freq full CG underperforms single-freq.

Hypotheses:
  A) CG didn't converge in 800 iters (bump to 5000, tighter tol)
  B) Joint CG has worse conditioning than separate per-freq CGs
     (loop over freqs and solve each independently — block-diag baseline)
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
import time, numpy as np, jax, jax.numpy as jnp, healpy as hp
import lusee
from lusee import mapmaker as mm

DRIVE = os.environ.get("LUSEE_DRIVE_DIR", "/fs/zack/LuSEE-Night/")
BEAM = DRIVE + "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY = DRIVE + "Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

LMAX = 32
FREQS = np.arange(5.0, 51.0, 5.0)
DT = 7200.0
OBS = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"


def rho(a, b):
    return hp.alm2cl(a, b) / np.sqrt(hp.alm2cl(a) * hp.alm2cl(b) + 1e-30)


def mr(r, a, b): return float(np.nanmean(r[a:b+1]))


def build():
    print("building 10-freq instrument ...", flush=True)
    t0 = time.time()
    sim, _, obs = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    print(f"  {time.time()-t0:.1f}s", flush=True)
    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    sky_freq = np.asarray(sky_full.freq, dtype=float)
    idx = np.array([int(np.argmin(np.abs(sky_freq - f))) for f in FREQS])
    maps = np.asarray(sky_full.maps)[idx]
    sky = lusee.sky.HealpixSky(sky_full.Nside, LMAX, maps=list(maps),
                                freq=FREQS, frame="galactic")
    dc = sim.simulate(sky=sky)
    sigma = mm.compute_radiometric_noise(dc, delta_f_hz=1e6, delta_t_sec=DT)
    data = dc + sigma * jax.random.normal(jax.random.PRNGKey(42), dc.shape)
    return sim, sky, data, sigma, maps, sky_full.Nside


def main():
    sim, sky, data, sigma, maps, Nside = build()
    S_inv = mm.compute_cl_prior(sky, LMAX)
    truth = np.asarray(sky.mapalm)

    # Experiment A: full joint CG with much bigger budget
    print("\n[A] full joint CG, maxiter=5000, tol=1e-14 ...", flush=True)
    t0 = time.time()
    alm_A = np.asarray(mm.solve(
        sim, data, sky, sigma, signal_prior=S_inv, maxiter=5000, tol=1e-14))
    print(f"   {time.time()-t0:.1f}s", flush=True)

    # Experiment B: per-freq independent solves
    print("\n[B] per-freq independent CG (block-diag baseline) ...", flush=True)
    t0 = time.time()
    alm_B = np.zeros_like(alm_A)
    for fi, f in enumerate(FREQS):
        sim_sf, _, _ = mm.build_instrument(
            beam_file=BEAM, obs_range=OBS, freq=np.array([f]), lmax=LMAX, dt_sec=DT)
        sky_sf = lusee.sky.HealpixSky(Nside, LMAX, maps=[maps[fi]],
                                       freq=np.array([f]), frame="galactic")
        data_sf = data[:, :, fi:fi+1]
        sig_sf = sigma[:, :, fi:fi+1]
        S_sf = mm.compute_cl_prior(sky_sf, LMAX)
        alm_sf = np.asarray(mm.solve(sim_sf, data_sf, sky_sf, sig_sf,
                                      signal_prior=S_sf, maxiter=800, tol=1e-10))
        alm_B[fi] = alm_sf[0]
        print(f"   {f:5.1f} MHz done", flush=True)
    print(f"   total {time.time()-t0:.1f}s", flush=True)

    # Compare
    print(f"\n{'freq':>6} {'joint CG':>10} {'per-freq':>10} {'delta':>8}")
    for fi, f in enumerate(FREQS):
        r_a = mr(rho(truth[fi], alm_A[fi]), 1, 20)
        r_b = mr(rho(truth[fi], alm_B[fi]), 1, 20)
        print(f"{f:>6.1f} {r_a:>10.4f} {r_b:>10.4f} {r_b-r_a:>+8.4f}")

    # save intermediate state at top of main too so re-runs are safe

    np.savez("/home/zack/luseepy/tests/debug_fullcg.npz",
             truth=truth, alm_joint=alm_A, alm_perfreq=alm_B, freqs=FREQS)


if __name__ == "__main__":
    main()
