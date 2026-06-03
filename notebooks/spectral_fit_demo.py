"""Spectral-sky fitting demo.

Fits a two-component sky model

    T(theta, f) = flux(theta) * (f / f_fid) ** beta(theta)

to mock LuSEE-Night data over a band of frequencies.  ``flux`` (a healpix map
in alm space) is *linear* and solved exactly by a Wiener filter; ``beta`` (a
real-space spectral-index map) is *non-linear* and fit by an outer optimiser.
This is the variable-projection / profile-likelihood pattern in
``lusee.fitting``.

Truth:  flux = ULSA at f_fid;  beta = a smooth map inside the prior box.
Run:    python notebooks/spectral_fit_demo.py
Needs:  LUSEE_DRIVE_DIR (beam + ULSA fits).  Set JAX_ENABLE_X64=1 for accuracy.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
import fitsio

import lusee


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DRIVE = os.environ["LUSEE_DRIVE_DIR"]
BEAM_FILE = DRIVE + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"


def build_truth(Nside, lmax, beta_nside, freq, f_fid):
    """Synthesize a true (flux_alm, beta_pix) pair."""
    # Flux: ULSA map at f_fid, projected to flux_alm in the s2fft basis.
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    fi = int(np.argmin(np.abs(np.asarray(sky_full.freq) - f_fid)))
    flux_map = hp.alm2map(np.asarray(sky_full.mapalm[fi]), Nside)
    flux_map = np.maximum(flux_map, 1.0)  # keep positive (it is a flux)
    flux_alm = lusee.SpectralHealpixSky.flux_alm_from_map(flux_map, lmax, Nside)

    # Beta: smooth dipole-ish pattern centered at -2.5, kept inside (-4,-1.5).
    npix = 12 * beta_nside ** 2
    theta, phi = hp.pix2ang(beta_nside, np.arange(npix))
    beta_pix = -2.5 + 0.25 * np.cos(theta) + 0.15 * np.sin(theta) * np.cos(phi)
    beta_pix = np.clip(beta_pix, -3.9, -1.6)
    return jnp.asarray(flux_alm), jnp.asarray(beta_pix)


def build_truth_ulsa(Nside, lmax, beta_nside, freq, f_fid):
    """Use the *real* ULSA maps at the fitted frequencies as the data-generating
    sky (so the power-law model is mis-specified -- ULSA is not a perfect power
    law).  Comparison flux = ULSA at f_fid; comparison beta = per-pixel power-law
    index from the lowest and highest fitted frequencies.

    Returns ``(data_sky, flux_true_alm, beta_true_pix)``.
    """
    maps_all = np.maximum(fitsio.read(SKY_FILE), 1e-3)   # (50, Npix32), positive
    ulsa_nside = hp.npix2nside(maps_all.shape[1])
    fr = np.arange(1, maps_all.shape[0] + 1, dtype=float)  # ULSA 1..50 MHz
    idx = [int(np.argmin(np.abs(fr - f))) for f in freq]

    # Data-generating sky: the actual ULSA maps (band-limited to lmax by the sim).
    data_sky = lusee.HealpixSky(ulsa_nside, lmax, maps=list(maps_all[idx]),
                                freq=freq, frame="galactic")

    # Comparison flux: ULSA at f_fid, in the fit's s2fft basis at working Nside.
    fi = int(np.argmin(np.abs(fr - f_fid)))
    flux_map = hp.ud_grade(maps_all[fi], Nside)
    flux_true = lusee.SpectralHealpixSky.flux_alm_from_map(flux_map, lmax, Nside)

    # Comparison beta: log(ULSA_hi / ULSA_lo) / log(f_hi / f_lo), at beta_nside.
    flo, fhi = float(min(freq)), float(max(freq))
    ilo = int(np.argmin(np.abs(fr - flo))); ihi = int(np.argmin(np.abs(fr - fhi)))
    mlo = hp.ud_grade(maps_all[ilo], beta_nside)
    mhi = hp.ud_grade(maps_all[ihi], beta_nside)
    beta_true = np.log(mhi / mlo) / np.log(fhi / flo)
    return data_sky, jnp.asarray(flux_true), jnp.asarray(beta_true)


def run(lmax=31, Nside=16, beta_nside=8,
        freq=(15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0), f_fid=25.0,
        obs_range="2025-02-01 13:00:00 to 2025-02-28 13:00:00",
        dt_sec=2 * 3600.0, taper=0.03, maxiter=120, inner_maxiter=400,
        fit_gain=False, truth="powerlaw", target_snr=1e4, noise_seed=42):

    freq = np.asarray(freq, dtype=float)
    print(f"=== Spectral fit ({truth} truth): lmax={lmax} Nside={Nside} "
          f"beta_nside={beta_nside} freq={freq.tolist()} f_fid={f_fid} ===")

    # Instrument (Tground=0, purely linear-in-flux forward).
    sim, beams, obs = lusee.mapmaker.build_instrument(
        beam_file=BEAM_FILE, obs_range=obs_range, freq=freq, lmax=lmax,
        taper=taper, dt_sec=dt_sec)
    print(f"{len(obs.times)} timesteps, {len(sim.combinations)} combinations")

    # Truth + mock data.  'powerlaw' generates self-consistent data from the
    # spectral model; 'ulsa' generates data from the real ULSA maps (model is
    # then mis-specified) and derives the comparison truth from them.
    if truth == "ulsa":
        data_sky, flux_true, beta_true = build_truth_ulsa(
            Nside, lmax, beta_nside, freq, f_fid)
    else:
        flux_true, beta_true = build_truth(Nside, lmax, beta_nside, freq, f_fid)
        data_sky = lusee.SpectralHealpixSky(
            flux_true, beta_true, Nside=Nside, lmax=lmax, freq=freq,
            f_fid=f_fid, beta_nside=beta_nside)

    t0 = time.time()
    data_clean = sim.simulate(sky=data_sky)
    print(f"forward (truth) in {time.time()-t0:.1f}s, data shape {data_clean.shape}")

    sigma = lusee.mapmaker.compute_radiometric_noise(
        data_clean, combinations=sim.combinations,
        delta_f_hz=1e6, delta_t_sec=dt_sec)
    if target_snr is not None:
        # Scale the (otherwise insane ~2e5) radiometric SNR down to a realistic
        # floor; high SNR makes chi2 model-mismatch-dominated and the landscape
        # needle-steep, which hurts the outer optimiser.
        snr0 = float(jnp.std(data_clean)) / float(jnp.median(sigma))
        sigma = sigma * (snr0 / target_snr)
    key = jax.random.PRNGKey(noise_seed)
    data = data_clean + sigma * jax.random.normal(key, data_clean.shape)
    N_inv = 1.0 / jnp.asarray(sigma) ** 2
    print(f"median sigma = {float(jnp.median(sigma)):.3f} K, "
          f"SNR ~ {float(jnp.std(data_clean))/float(jnp.median(sigma)):.0f}")

    # Flux prior C_l from the true flux power spectrum (oracle prior).
    cl_flux = hp.alm2cl(np.asarray(flux_true))

    # ---- Assemble the model from modules (Layer 2) and wire them (Layer 3) ----
    from lusee.SpectralSky import SpectralSkyModule
    from lusee.Fitting import BeamModule, InstrumentModule, Experiment
    exp = Experiment(
        sim,
        sky=SpectralSkyModule(lmax=lmax, Nside=Nside, freq=freq, f_fid=f_fid,
                              beta_nside=beta_nside, cl_flux=cl_flux,
                              beta_bounds=(-4.0, -1.5), beta_init=-2.5),
        beam=BeamModule(),                      # no free params (baseline)
        instrument=InstrumentModule(gain=fit_gain),  # optional bilinear gain
        data=data, N_inv=N_inv)
    print("\nParameters in the model:")
    for p in exp.paramset.params:
        print(f"  {p.name:12s} {p.kind:9s} "
              f"{'n='+str(12*beta_nside**2) if p.name=='sky.beta' else ''}")

    # ---- Fit (shared driver) ----
    t0 = time.time()
    out = exp.optimize(maxiter=maxiter, inner_maxiter=inner_maxiter)
    fit = {"flux_alm": out["linear"]["sky.flux"],
           "beta_pix": out["nonlinear"]["sky.beta"],
           "chi2": out["chi2"], "nfev": out["nfev"]}
    print(f"Fit in {time.time()-t0:.1f}s  ({fit['nfev']} evals, "
          f"final chi2={fit['chi2']:.4e})")
    if fit_gain:
        print(f"  recovered inst.gain = {float(out['nonlinear']['inst.gain']):.4f} "
              f"(truth 1.0)")

    # ---- Report ----
    beta_hat = fit["beta_pix"]
    beta_t = np.asarray(beta_true)
    print("\nBeta recovery (per coarse pixel):")
    print(f"  truth range  [{beta_t.min():.3f}, {beta_t.max():.3f}]")
    print(f"  recovered    [{beta_hat.min():.3f}, {beta_hat.max():.3f}]")
    print(f"  RMS(beta_hat - beta_true) = {np.sqrt(np.mean((beta_hat-beta_t)**2)):.4f}")

    flux_hat = np.asarray(fit["flux_alm"])
    flux_t = np.asarray(flux_true)
    rho = hp.alm2cl(flux_t, flux_hat) / np.sqrt(
        hp.alm2cl(flux_t) * hp.alm2cl(flux_hat) + 1e-30)
    print("\nFlux recovery (cross-correlation rho_l):")
    print(f"  a00: true={flux_t[0].real:.0f}, rec={flux_hat[0].real:.0f}")
    print(f"  mean rho(1..{min(10,lmax)}) = {np.nanmean(rho[1:min(11,lmax+1)]):.4f}")

    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"spectral_fit_result_{truth}.npz")
    np.savez(outfile, flux_true=flux_t, flux_hat=flux_hat, beta_true=beta_t,
             beta_hat=beta_hat, rho=rho, freq=freq, lmax=lmax, Nside=Nside,
             beta_nside=beta_nside, f_fid=f_fid,
             chi2_history=out["chi2_history"], converged=out["converged"])
    print(f"\nsaved arrays -> {outfile}")
    return fit


if __name__ == "__main__":
    run()
