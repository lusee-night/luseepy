"""Spectral-sky fitting demo.

Fits a two-component sky model

    T(theta, f) = flux(theta) * (f / f_fid) ** beta(theta)

to mock LuSEE-Night data over a band of frequencies.  ``flux`` (a healpix map
in alm space) is *linear* and solved exactly by a Wiener filter; ``beta`` (a
real-space spectral-index map) is *non-linear* and fit by an outer optimiser.
This is the variable-projection / profile-likelihood pattern in
``lusee.fitting``.

Truth: ``--truth ulsa`` uses the real ULSA maps (model is then mis-specified);
``--truth powerlaw`` generates self-consistent data from the model itself.

Command line (``--help`` for the full list; writes a single ``.npz`` that the
recovery notebook reads)::

    # MAP only, default Nside=16/lmax=31 ULSA fit:
    python notebooks/spectral_fit_demo.py
    # add Fisher (Wiener-weighted) recovery and HMC posterior, custom output:
    python notebooks/spectral_fit_demo.py --fisher --hmc -o /tmp/spec.npz
    # quick low-res self-consistent run:
    python notebooks/spectral_fit_demo.py --truth powerlaw --lmax 15 --nside 8 \\
        --beta-nside 4 --dt-hours 8 --hmc --num-samples 300

Needs ``LUSEE_DRIVE_DIR`` (beam + ULSA fits) when run; set ``JAX_ENABLE_X64=1``.
The ``run()`` function takes the same options as keyword arguments.
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
DRIVE = os.environ.get("LUSEE_DRIVE_DIR")  # resolved lazily so --help works unset
BEAM_FILE = (DRIVE or "") + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = (DRIVE or "") + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"


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
        dt_sec=2 * 3600.0, taper=0.03, maxiter=200, inner_maxiter=1500,
        inner_method="auto", dense_threshold=512, fisher_method="auto",
        fit_gain=False, truth="powerlaw", target_snr=1e4, compute_fisher=False,
        sample=False, num_samples=300, num_warmup=300, hmc_engine="nuts",
        hmc_num_integration_steps=10, noise_seed=42, outfile=None,
        beta_smooth=False, beta_n_center=16, beta_n_far=2, beta_d0=20.0,
        beta_smooth_w=1.0, beta_anchor=1e-4,
        fix_beta=None, beta_const=-2.5):

    if not DRIVE:
        raise SystemExit("LUSEE_DRIVE_DIR must be set (beam + ULSA data files).")
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

    # ---- Optional purely-angular GMRF smoothness prior on beta ----------------
    # Lets beta be fine near the galactic centre and smooth far away.  It is NOT
    # coverage-aware: where the data are uninformative (the never-observed patch)
    # beta is simply tied to its neighbours and floats with them -- a graceful
    # "we don't know here" behaviour, no hole-specific term.
    beta_prior = None
    beta_bounds = (-4.5, 0.0) if beta_smooth else (-4.0, -1.5)
    if beta_smooth:
        from lusee.SpectralSky import build_beta_smoothness_prior
        beta_prior = build_beta_smoothness_prior(
            beta_nside, n_center=beta_n_center, n_far=beta_n_far,
            d0_deg=beta_d0, mu=beta_const, smooth=beta_smooth_w,
            anchor=beta_anchor)
        print(f"beta GMRF smoothness prior: n_eff {beta_n_center}->"
              f"{beta_n_far} over d0={beta_d0:g} deg (smooth={beta_smooth_w:g})")

    # ---- Optional fixed-beta mode: purely linear in flux (single CG/dense) -----
    beta_fixed = None
    if fix_beta is not None:
        n_beta = 12 * beta_nside ** 2
        if fix_beta == "const":
            beta_fixed = np.full(n_beta, float(beta_const))
        elif fix_beta == "truth":
            beta_fixed = np.asarray(beta_true)
        else:
            raise SystemExit("fix_beta must be None, 'const', or 'truth'")
        print(f"\n=== FIXED-beta mode ('{fix_beta}'): flux is the only free "
              f"parameter (linear) ===")

    # ---- Assemble the model from modules (Layer 2) and wire them (Layer 3) ----
    from lusee.SpectralSky import SpectralSkyModule
    from lusee.Fitting import BeamModule, InstrumentModule, Experiment
    exp = Experiment(
        sim,
        sky=SpectralSkyModule(lmax=lmax, Nside=Nside, freq=freq, f_fid=f_fid,
                              beta_nside=beta_nside, cl_flux=cl_flux,
                              beta_bounds=beta_bounds, beta_init=-2.5,
                              beta_prior=beta_prior, beta_fixed=beta_fixed),
        beam=BeamModule(),                      # no free params (baseline)
        instrument=InstrumentModule(gain=fit_gain),  # optional bilinear gain
        data=data, N_inv=N_inv)
    print("\nParameters in the model:")
    for p in exp.paramset.params:
        print(f"  {p.name:12s} {p.kind:9s} "
              f"{'n='+str(12*beta_nside**2) if p.name=='sky.beta' else ''}")

    # ---- Fit (shared driver, or a single linear solve when beta is fixed) -----
    t0 = time.time()
    if beta_fixed is not None:
        lin = exp.linear_solve(method=("dense" if inner_method == "dense"
                                       else "cg"),    # CG by default: fast linear
                               inner_maxiter=inner_maxiter)
        r = jnp.asarray(data).ravel() - exp.predict(lin, {})
        S_inv = exp.paramset.Sinv_linear()
        theta = exp.paramset.pack_linear(lin)
        chi2 = float(jnp.sum(N_inv.ravel() * r ** 2) + jnp.sum(S_inv * theta ** 2))
        out = {"linear": lin, "nonlinear": {"sky.beta": beta_fixed},
               "chi2": chi2, "nfev": 1,
               "chi2_history": np.array([chi2]), "converged": True}
    else:
        out = exp.optimize(maxiter=maxiter, inner_maxiter=inner_maxiter,
                           inner_method=inner_method,
                           dense_threshold=dense_threshold)
    fit = {"flux_alm": out["linear"]["sky.flux"],
           "beta_pix": out["nonlinear"]["sky.beta"],
           "chi2": out["chi2"], "nfev": out["nfev"]}
    print(f"Fit in {time.time()-t0:.1f}s  ({fit['nfev']} evals, "
          f"final chi2={fit['chi2']:.4e})")
    if fit_gain and "inst.gain" in out["nonlinear"]:
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

    # ---- Fisher / SNR-weighted recovery of the flux block ----
    save = dict(flux_true=flux_t, flux_hat=flux_hat, beta_true=beta_t,
                beta_hat=beta_hat, rho=rho, freq=freq, lmax=lmax, Nside=Nside,
                beta_nside=beta_nside, f_fid=f_fid,
                chi2_history=out["chi2_history"], converged=out["converged"])
    if compute_fisher:
        from lusee.Fitting import linear_fisher, snr_weighted_recovery
        block = exp.paramset.linear[0].reparam
        t0 = time.time()
        fish = linear_fisher(exp.predict, exp.paramset, data, N_inv,
                             out["nonlinear"], method=fisher_method)
        rec = snr_weighted_recovery(block.natural_to_theta(flux_hat),
                                    block.natural_to_theta(flux_t), fish)
        print(f"\nFisher / Wiener-weighted flux recovery ({time.time()-t0:.0f}s):")
        print(f"  Wiener-weighted rho = {rec['rho_w']:.4f}  "
              f"(1 = recoverable modes recovered)")
        print(f"  residual power frac = {rec['resid_frac']:.4f}  (0 = perfect)")
        print(f"  effective measured modes n_eff = {rec['n_eff']:.1f} / {rec['n']}")
        ww = rec["whitened"][np.isfinite(rec["whitened"])]
        print(f"  (diagnostic) whitened residual std = {ww.std():.1f}")
        save.update(post_std=fish["std"], wiener=fish["wiener"],
                    whitened=rec["whitened"], rho_w=rec["rho_w"],
                    resid_frac=rec["resid_frac"], n_eff=rec["n_eff"])

    # ---- HMC posterior (mean = estimate, std = per-mode SNR) ----
    if sample:
        from lusee.Fitting import sample_posterior
        t0 = time.time()
        post = sample_posterior(
            exp.predict, exp.paramset, data, N_inv,
            num_samples=num_samples, num_warmup=num_warmup, seed=noise_seed,
            init_linear=exp.paramset.pack_linear(out["linear"]),
            init_nonlinear=exp.paramset.pack_nonlinear(out["nonlinear"]),
            engine=hmc_engine,
            hmc_num_integration_steps=hmc_num_integration_steps)
        print(f"\nHMC: {num_samples} samples in {time.time()-t0:.0f}s, "
              f"accept={post['accept']:.2f}")
        fa = np.asarray(post["linear"]["sky.flux"])          # (ns, nalm) complex
        flux_smaps = np.stack([hp.alm2map(fa[i], Nside) for i in range(len(fa))])
        bb = np.asarray(post["nonlinear"]["sky.beta"])        # (ns, nbeta)
        save.update(
            accept=post["accept"],
            flux_map_map=hp.alm2map(flux_hat, Nside),
            flux_true_map=hp.alm2map(flux_t, Nside),
            flux_post_mean_map=flux_smaps.mean(0),
            flux_post_std_map=flux_smaps.std(0),
            beta_post_mean=bb.mean(0), beta_post_std=bb.std(0))
        # how many sigma is each beta pixel from truth?
        bz = (bb.mean(0) - beta_t) / np.maximum(bb.std(0), 1e-9)
        print(f"  beta: post mean {bb.mean():.3f}+/-{bb.std(0).mean():.3f} "
              f"(truth mean {beta_t.mean():.3f}); |pull| median = "
              f"{np.median(np.abs(bz)):.2f} sigma")

    if outfile is None:
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"spectral_fit_result_{truth}.npz")
    np.savez(outfile, **save)
    print(f"\nsaved arrays -> {outfile}")
    return fit


def main(argv=None):
    import argparse

    class DefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
        def _get_help_string(self, action):
            help_text = action.help or ""
            if (action.option_strings
                    and action.default is not argparse.SUPPRESS
                    and "%(default)" not in help_text):
                if help_text:
                    help_text += " "
                help_text += "(default: %(default)s)"
            return help_text

    p = argparse.ArgumentParser(
        description="Spectral sky fit: T(theta,f) = flux(theta)*(f/f_fid)^beta(theta). "
                    "MAP (Wiener flux + L-BFGS-B beta), optional Fisher and HMC.",
        formatter_class=DefaultsHelpFormatter)
    p.add_argument("--lmax", type=int, default=31)
    p.add_argument("--nside", type=int, default=16, dest="Nside")
    p.add_argument("--beta-nside", type=int, default=8, dest="beta_nside")
    p.add_argument("--freq", type=float, nargs="+",
                   default=[15., 20., 25., 30., 35., 40., 45.],
                   help="frequencies in MHz")
    p.add_argument("--f-fid", type=float, default=25.0, dest="f_fid")
    p.add_argument("--obs-range", default="2025-02-01 13:00:00 to 2025-02-28 13:00:00",
                   dest="obs_range")
    p.add_argument("--dt-hours", type=float, default=4.0,
                   help="time step in hours")
    p.add_argument("--taper", type=float, default=0.03)
    p.add_argument("--truth", choices=["powerlaw", "ulsa"], default="ulsa",
                   help="generate data from the spectral model ('powerlaw') "
                        "or from the real ULSA maps ('ulsa')")
    p.add_argument("--target-snr", type=float, default=1e4, dest="target_snr")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--inner-maxiter", type=int, default=1500, dest="inner_maxiter")
    p.add_argument("--inner-method", choices=["auto", "cg", "dense"],
                   default="auto", dest="inner_method",
                   help="linear solve method for the VarPro inner step")
    p.add_argument("--dense-threshold", type=int, default=512, dest="dense_threshold",
                   help="n_linear threshold where --inner-method auto selects dense")
    p.add_argument("--fit-gain", action="store_true", dest="fit_gain",
                   help="fit a broadband instrument gain (bilinear with flux)")
    p.add_argument("--beta-smooth", action="store_true", dest="beta_smooth",
                   help="purely-angular GMRF smoothness prior on beta (fine near "
                        "gal. centre, smooth far away; hole handled gracefully)")
    p.add_argument("--beta-n-center", type=int, default=16, dest="beta_n_center",
                   help="target effective beta Nside at the galactic centre")
    p.add_argument("--beta-n-far", type=int, default=2, dest="beta_n_far",
                   help="target effective beta Nside far from the centre")
    p.add_argument("--beta-d0", type=float, default=20.0, dest="beta_d0",
                   help="Gaussian scale (deg) of the centre->far resolution ramp")
    p.add_argument("--beta-smooth-w", type=float, default=1.0, dest="beta_smooth_w",
                   help="overall smoothness weight (absolute -2logL units)")
    p.add_argument("--beta-anchor", type=float, default=1e-4, dest="beta_anchor",
                   help="weak uniform ridge toward beta_const (properness)")
    p.add_argument("--fix-beta", choices=["const", "truth"], default=None,
                   dest="fix_beta",
                   help="hold beta fixed and fit only the (linear) flux: "
                        "'const' = beta_const everywhere, 'truth' = the true map")
    p.add_argument("--beta-const", type=float, default=-2.5, dest="beta_const",
                   help="beta value for --fix-beta const and the prior/anchor mu")
    p.add_argument("--fisher", action="store_true", dest="compute_fisher",
                   help="compute the Fisher / Wiener-weighted flux recovery")
    p.add_argument("--fisher-method", choices=["auto", "dense", "loop"],
                   default="auto", dest="fisher_method",
                   help="linear Fisher construction method")
    p.add_argument("--hmc", action="store_true", dest="sample",
                   help="run HMC (NUTS) for the posterior mean/std")
    p.add_argument("--hmc-engine", choices=["nuts", "hmc"],
                   default="nuts", dest="hmc_engine",
                   help="BlackJAX HMC-family engine")
    p.add_argument("--hmc-steps", type=int, default=10,
                   dest="hmc_num_integration_steps",
                   help="integration steps for --hmc-engine hmc")
    p.add_argument("--num-samples", type=int, default=300, dest="num_samples")
    p.add_argument("--num-warmup", type=int, default=300, dest="num_warmup")
    p.add_argument("--seed", type=int, default=42, dest="noise_seed")
    p.add_argument("-o", "--output", default=None, dest="outfile",
                   help="output .npz path (default: spectral_fit_result_<truth>.npz)")
    for action in p._actions:
        if action.option_strings and action.help is None:
            action.help = "(default: %(default)s)"
    a = p.parse_args(argv)
    run(lmax=a.lmax, Nside=a.Nside, beta_nside=a.beta_nside, freq=tuple(a.freq),
        f_fid=a.f_fid, obs_range=a.obs_range, dt_sec=a.dt_hours * 3600.0,
        taper=a.taper, maxiter=a.maxiter, inner_maxiter=a.inner_maxiter,
        inner_method=a.inner_method, dense_threshold=a.dense_threshold,
        fit_gain=a.fit_gain, truth=a.truth, target_snr=a.target_snr,
        compute_fisher=a.compute_fisher, fisher_method=a.fisher_method,
        sample=a.sample, hmc_engine=a.hmc_engine,
        hmc_num_integration_steps=a.hmc_num_integration_steps,
        num_samples=a.num_samples, num_warmup=a.num_warmup,
        noise_seed=a.noise_seed, outfile=a.outfile,
        beta_smooth=a.beta_smooth, beta_n_center=a.beta_n_center,
        beta_n_far=a.beta_n_far, beta_d0=a.beta_d0,
        beta_smooth_w=a.beta_smooth_w, beta_anchor=a.beta_anchor,
        fix_beta=a.fix_beta, beta_const=a.beta_const)


if __name__ == "__main__":
    main()
