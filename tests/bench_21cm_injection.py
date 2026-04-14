"""Inject a toy 21-cm-like signal and measure signal attenuation by SVD K=4.

The SVD templates are fit to a foreground-only cube (ULSA). A real signal
has a frequency shape that is not well-represented in the span of the
K leading foreground modes. So the SVD solve will attenuate the signal
by whatever fraction is orthogonal to the subspace.

Toy signal: Gaussian absorption trough, amplitude -25 mK at 17.5 MHz,
width sigma=4 MHz. Spatially a pure monopole (uniform across sphere).
Injection: data = forward(FG + signal) + noise.
Reconstruction: SVD K=4 (FG-only templates) and independent single-freq CG.
Metric: recovered monopole amplitude vs truth at each frequency.

Produces bench_21cm_injection.npz with the recovered monopoles.
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
OUT = "/home/zack/luseepy/tests/bench_21cm_injection.npz"

SIG_AMP = -25e-3   # K
SIG_NU0 = 17.5     # MHz
SIG_SIG = 4.0      # MHz


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def signal_freq(nu):
    return SIG_AMP * np.exp(-(nu - SIG_NU0) ** 2 / (2 * SIG_SIG ** 2))


def main():
    log("building instrument ...")
    t0 = time.time()
    sim, _, _ = mm.build_instrument(
        beam_file=BEAM, obs_range=OBS, freq=FREQS, lmax=LMAX, dt_sec=DT)
    log(f"  {time.time()-t0:.1f}s")

    sky_full = lusee.sky.FitsSky(SKY, lmax=LMAX)
    idx = np.array([int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in FREQS])
    fg_maps = np.asarray(sky_full.maps)[idx]                     # (nfreq, npix)
    Nside = sky_full.Nside
    npix = 12 * Nside * Nside

    T21 = signal_freq(FREQS)                                     # (nfreq,)
    log("injected signal T_21(nu) [mK]:")
    for f, t in zip(FREQS, T21):
        log(f"  {f:5.1f} MHz  {t*1e3:+7.3f}")

    # total = FG + uniform signal per freq
    total_maps = fg_maps + T21[:, None] * np.ones((len(FREQS), npix))
    sky_true = lusee.sky.HealpixSky(Nside, LMAX, maps=list(total_maps),
                                     freq=FREQS, frame="galactic")
    data_c = sim.simulate(sky=sky_true)
    sigma = mm.compute_radiometric_noise(data_c, delta_f_hz=1e6, delta_t_sec=DT)
    data = data_c + sigma * jax.random.normal(jax.random.PRNGKey(42), data_c.shape)
    truth_alm = np.asarray(sky_true.mapalm)

    # FG-only SVD templates (critical: NOT trained on the injected signal)
    F, spatial, sv = mm.compute_ulsa_svd(SKY, FREQS, K=K_SVD, lmax=LMAX)
    log(f"SVD sv ratios: {sv[1]/sv[0]:.2e}, {sv[2]/sv[0]:.2e}, {sv[3]/sv[0]:.2e}")

    # Projection of signal onto the span of F:
    # fit T_21(nu) = F @ c in least-squares
    c, *_ = np.linalg.lstsq(F, T21, rcond=None)
    T21_proj = F @ c
    T21_ortho = T21 - T21_proj
    log("fraction of T_21 outside span(F):")
    for f, t, to in zip(FREQS, T21, T21_ortho):
        log(f"  {f:5.1f} MHz  T21={t*1e3:+.3f} mK  ortho={to*1e3:+.3f} mK  "
            f"ratio={to/max(abs(t),1e-30):.2f}")

    # SVD solve (K=4)
    S_inv_beta = np.zeros((K_SVD, hp.Alm.getsize(LMAX)))
    for k in range(K_SVD):
        alm_k = hp.map2alm(spatial[k], lmax=LMAX)
        cl = hp.alm2cl(alm_k)
        for l in range(LMAX + 1):
            for m in range(l + 1):
                if cl[l] > 0:
                    S_inv_beta[k, hp.Alm.getidx(LMAX, l, m)] = 1.0 / cl[l]
    S_inv_beta = jnp.asarray(S_inv_beta)

    log("SVD K=4 solve ...")
    t0 = time.time()
    beta = mm.solve_svd_multifreq(sim, data, sky_true, sigma, F,
                                   signal_prior=S_inv_beta, lmax=LMAX,
                                   maxiter=800, tol=1e-10)
    rec_svd = np.asarray(jnp.asarray(F) @ beta)
    log(f"  {time.time()-t0:.1f}s")

    # single-freq independent baseline
    log("single-freq CG at every band ...")
    rec_sf = np.zeros_like(rec_svd)
    for fi, f in enumerate(FREQS):
        t0 = time.time()
        sim_sf, _, _ = mm.build_instrument(
            beam_file=BEAM, obs_range=OBS, freq=np.array([f]),
            lmax=LMAX, dt_sec=DT)
        sky_sf = lusee.sky.HealpixSky(Nside, LMAX, maps=[total_maps[fi]],
                                       freq=np.array([f]), frame="galactic")
        data_sf_c = sim_sf.simulate(sky=sky_sf)
        sigma_sf = mm.compute_radiometric_noise(data_sf_c, delta_f_hz=1e6, delta_t_sec=DT)
        data_sf = data_sf_c + sigma_sf * jax.random.normal(
            jax.random.PRNGKey(42 + fi), data_sf_c.shape)
        S_inv_sf = mm.compute_cl_prior(sky_sf, LMAX)
        rec_sf[fi] = np.asarray(mm.solve(sim_sf, data_sf, sky_sf, sigma_sf,
                                          signal_prior=S_inv_sf,
                                          maxiter=800, tol=1e-10))[0]
        log(f"  {f:5.1f} MHz  {time.time()-t0:.1f}s")

    # Extract monopoles
    def mono(alm):
        return float(np.real(alm[hp.Alm.getidx(LMAX, 0, 0)])) / np.sqrt(4 * np.pi)

    mono_truth = np.array([mono(truth_alm[fi]) for fi in range(len(FREQS))])
    mono_svd   = np.array([mono(rec_svd[fi])   for fi in range(len(FREQS))])
    mono_sf    = np.array([mono(rec_sf[fi])    for fi in range(len(FREQS))])
    mono_fg    = np.array([mono(hp.map2alm(fg_maps[fi], lmax=LMAX)) for fi in range(len(FREQS))])

    log("monopole (T_ant) in K at each freq:")
    log(f"{'MHz':>5}  {'FG':>10}  {'FG+T21':>10}  {'SVD K=4':>10}  {'single-f':>10}  "
        f"{'SVD-truth':>10}  {'SF-truth':>10}")
    for fi, f in enumerate(FREQS):
        log(f"{f:>5.1f}  {mono_fg[fi]:>10.4f}  {mono_truth[fi]:>10.4f}  "
            f"{mono_svd[fi]:>10.4f}  {mono_sf[fi]:>10.4f}  "
            f"{(mono_svd[fi]-mono_truth[fi])*1e3:>+10.4f}  "
            f"{(mono_sf[fi]-mono_truth[fi])*1e3:>+10.4f}  [mK]")

    log("\n21-cm signal recovery: (recovered - FG_only) = apparent signal")
    log(f"{'MHz':>5}  {'truth T21':>10}  {'SVD recov':>10}  {'SF recov':>10}  "
        f"{'SVD/truth':>10}  {'SF/truth':>10}")
    for fi, f in enumerate(FREQS):
        recov_svd = (mono_svd[fi] - mono_fg[fi]) * 1e3
        recov_sf  = (mono_sf[fi]  - mono_fg[fi]) * 1e3
        t21 = T21[fi] * 1e3
        log(f"{f:>5.1f}  {t21:>+10.3f}  {recov_svd:>+10.3f}  {recov_sf:>+10.3f}  "
            f"{recov_svd/t21 if abs(t21)>0.1 else 0:>10.2f}  "
            f"{recov_sf/t21 if abs(t21)>0.1 else 0:>10.2f}")

    np.savez(OUT, freqs=FREQS, lmax=LMAX, T21=T21, mono_truth=mono_truth,
             mono_svd=mono_svd, mono_sf=mono_sf, mono_fg=mono_fg,
             sv=sv, rec_svd=rec_svd, rec_sf=rec_sf, truth_alm=truth_alm,
             T21_ortho=T21_ortho)
    log(f"saved {OUT}")


if __name__ == "__main__":
    main()
