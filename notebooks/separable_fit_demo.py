"""Separable (space-frequency template) sky fitting demo.

Fits the bilinear model

    T(theta, f) = sum_i  flux_i(theta) * shape_i(f)        (n_templates = 2)

to mock LuSEE-Night data generated from the *real ULSA maps*.  The template
maps `flux_i` (alm) are the **linear** block (Wiener solve); the spectral shapes
`shape_i(f)` are the **non-linear** block (outer optimiser).  This is the same
`lusee.fitting` machinery as the power-law demo — only a different sky module —
so it exercises the registry's support for *multiple* linear blocks and the
bilinear flux x shape coupling.

Truth: a rank-``n_templates`` PCA (over frequency) of the ULSA maps, re-gauged so
each shape equals 1 at the reference frequency.  The data are the full ULSA maps,
so the model is mildly mis-specified.  The shapes are initialised from a PCA of
the *data* (``--init-from data``), i.e. truth-free.

Command line (``--help`` for the full list; writes a single ``.npz`` that the
recovery notebook reads)::

    # MAP only, default Nside=16/lmax=31, 2 templates:
    python notebooks/separable_fit_demo.py
    # 3 templates, with HMC posterior, custom output:
    python notebooks/separable_fit_demo.py --n-templates 3 --hmc -o /tmp/sep.npz
    # quick low-res run:
    python notebooks/separable_fit_demo.py --lmax 15 --nside 8 --dt-hours 8 --hmc

Needs ``LUSEE_DRIVE_DIR``; set ``JAX_ENABLE_X64=1``.  The ``run()`` function takes
the same options as keyword arguments.
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
from lusee.SeparableSky import SeparableSkyModule
from lusee.Fitting import BeamModule, InstrumentModule, Experiment

DRIVE = os.environ.get("LUSEE_DRIVE_DIR")  # resolved lazily so --help works unset
BEAM_FILE = (DRIVE or "") + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = (DRIVE or "") + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"


def build_truth(Nside, lmax, freq, ref_freq, n_templates):
    """Rank-`n_templates` PCA of the ULSA maps over frequency, gauge-anchored so
    each spectral shape equals 1 at `ref_freq`.

    Returns ``(data_sky, flux_alms_true, shapes_true)``:
      * data_sky      — HealpixSky of the *full* ULSA maps (the data we fit),
      * flux_alms_true — (n_templates, nalm) template maps in alm,
      * shapes_true    — (n_templates, nfreq) spectral shapes.
    """
    maps_all = np.maximum(fitsio.read(SKY_FILE), 1e-3)     # (50, Npix32)
    ulsa_nside = hp.npix2nside(maps_all.shape[1])
    fr = np.arange(1, maps_all.shape[0] + 1, dtype=float)
    idx = [int(np.argmin(np.abs(fr - f))) for f in freq]
    ulsa = maps_all[idx]                                    # (nfreq, Npix32)

    # Data-generating sky: the full ULSA maps at the fitted frequencies.
    data_sky = lusee.HealpixSky(ulsa_nside, lmax, maps=list(ulsa),
                                freq=freq, frame="galactic")

    # PCA over frequency at the working resolution.
    X = hp.ud_grade(ulsa, Nside).T                          # (Npix, nfreq)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    flux_maps = (U[:, :n_templates] * S[:n_templates])      # (Npix, n_templ)
    shapes = Vt[:n_templates]                               # (n_templ, nfreq)

    # Re-gauge: anchor each shape at the reference frequency (shape_i(f_ref)=1).
    k_ref = int(np.argmin(np.abs(np.asarray(freq) - ref_freq)))
    c = shapes[:, k_ref].copy()
    shapes = shapes / c[:, None]
    flux_maps = flux_maps * c[None, :]

    # Templates in healpy-packed alm (the separable forward is pure alm -- no
    # s2fft -- so we stay in the healpy convention throughout).
    flux_alms = np.stack([hp.map2alm(flux_maps[:, i], lmax)
                          for i in range(n_templates)])
    return data_sky, jnp.asarray(flux_alms), jnp.asarray(shapes)


def run(lmax=31, Nside=16, n_templates=2,
        freq=(15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0), ref_freq=25.0,
        obs_range="2025-02-01 13:00:00 to 2025-02-28 13:00:00",
        dt_sec=4 * 3600.0, taper=0.03, maxiter=200, inner_maxiter=1500,
        inner_method="auto", dense_threshold=512,
        init_from="data", shape_perturb=0.25, target_snr=1e4,
        sample=False, num_samples=300, num_warmup=300, hmc_engine="nuts",
        hmc_num_integration_steps=10, noise_seed=42, outfile=None):

    if not DRIVE:
        raise SystemExit("LUSEE_DRIVE_DIR must be set (beam + ULSA data files).")
    freq = np.asarray(freq, dtype=float)
    print(f"=== Separable fit: lmax={lmax} Nside={Nside} "
          f"n_templates={n_templates} freq={freq.tolist()} ref={ref_freq} ===")

    sim, beams, obs = lusee.mapmaker.build_instrument(
        beam_file=BEAM_FILE, obs_range=obs_range, freq=freq, lmax=lmax,
        taper=taper, dt_sec=dt_sec)
    print(f"{len(obs.times)} timesteps, {len(sim.combinations)} combinations")

    data_sky, flux_true, shapes_true = build_truth(
        Nside, lmax, freq, ref_freq, n_templates)

    t0 = time.time()
    data_clean = sim.simulate(sky=data_sky)
    print(f"forward (ULSA) in {time.time()-t0:.1f}s, data shape {data_clean.shape}")

    sigma = lusee.mapmaker.compute_radiometric_noise(
        data_clean, combinations=sim.combinations,
        delta_f_hz=1e6, delta_t_sec=dt_sec)
    if target_snr is not None:
        snr0 = float(jnp.std(data_clean)) / float(jnp.median(sigma))
        sigma = sigma * (snr0 / target_snr)
    data = data_clean + sigma * jax.random.normal(
        jax.random.PRNGKey(noise_seed), data_clean.shape)
    N_inv = 1.0 / jnp.asarray(sigma) ** 2
    print(f"median sigma = {float(jnp.median(sigma)):.3f} K, "
          f"SNR ~ {float(jnp.std(data_clean))/float(jnp.median(sigma)):.0f}")

    cl_flux = [hp.alm2cl(np.asarray(flux_true[i])) for i in range(n_templates)]

    k_ref = int(np.argmin(np.abs(freq - ref_freq)))
    if init_from == "data":
        # Realistic, truth-free initialisation: PCA of the *data* over frequency.
        # Each (time, channel) sample is a row; the first n_templates right
        # singular vectors are the dominant frequency modes of what the
        # instrument actually measured.  Anchor each to 1 at the reference
        # frequency (guarding against a near-zero crossing there).
        D = np.asarray(data).reshape(-1, len(freq))          # (Ntime*Nchan, Nfreq)
        _, _, Vt = np.linalg.svd(D, full_matrices=False)
        modes = Vt[:n_templates]                             # (n_templ, Nfreq)
        ref_vals = modes[:, k_ref]
        ref_vals = np.where(np.abs(ref_vals) < 1e-2,
                            np.sign(ref_vals + 1e-12) * 1e-2, ref_vals)
        shape_init = modes / ref_vals[:, None]
        print("shape init from PCA of the DATA over frequency (truth not used)")
    else:
        # Truth-perturbation init (diagnostic only -- assumes knowing the truth).
        rng = np.random.default_rng(noise_seed)
        shape_init = np.asarray(shapes_true) * (
            1.0 + shape_perturb * rng.standard_normal(np.asarray(shapes_true).shape))
        print(f"shape init perturbed by {shape_perturb:.0%} from truth")

    # ---- Assemble (Layer 2 + 3): only the sky module differs from the
    #      power-law demo; everything else is reused unchanged. ----
    sky_mod = SeparableSkyModule(
        lmax=lmax, freq=freq, ref_freq=ref_freq, n_templates=n_templates,
        cl_flux=cl_flux, shape_init=shape_init)
    exp = Experiment(sim, sky=sky_mod, beam=BeamModule(),
                     instrument=InstrumentModule(), data=data, N_inv=N_inv)
    print("\nParameters in the model:")
    for p in exp.paramset.params:
        print(f"  {p.name:12s} {p.kind}")

    t0 = time.time()
    res = exp.optimize(maxiter=maxiter, inner_maxiter=inner_maxiter,
                       inner_method=inner_method,
                       dense_threshold=dense_threshold)
    print(f"Fit in {time.time()-t0:.1f}s  ({res['nfev']} evals, "
          f"final chi2={res['chi2']:.4e})")

    flux_hat = np.stack([np.asarray(res["linear"][f"sep.flux.{i}"])
                         for i in range(n_templates)])
    shapes_hat = np.asarray(sky_mod._full_shapes(
        jnp.asarray(res["nonlinear"]["sep.shape"])))

    # ---- Report: gauge-invariant total flux at low/mid/high frequency ----
    test_freq = [float(freq.min()), float(freq[len(freq) // 2]), float(freq.max())]
    maps_all = np.maximum(fitsio.read(SKY_FILE), 1e-3)
    fr = np.arange(1, maps_all.shape[0] + 1, dtype=float)
    ulsa_test, rec_test = [], []
    print("\nTotal-flux recovery (gauge-invariant) vs ULSA:")
    for f in test_freq:
        kf = int(np.argmin(np.abs(freq - f)))
        rec_alm = np.einsum("i,ia->a", shapes_hat[:, kf], flux_hat)
        rec = hp.alm2map(rec_alm.astype(complex), Nside)
        ulsa = hp.ud_grade(maps_all[int(np.argmin(np.abs(fr - f)))], Nside)
        rho = (hp.alm2cl(rec_alm, hp.map2alm(ulsa, lmax)) /
               np.sqrt(hp.alm2cl(rec_alm) * hp.alm2cl(hp.map2alm(ulsa, lmax)) + 1e-30))
        print(f"  f={f:4.0f} MHz: rec/ULSA mean ratio = {rec.mean()/ulsa.mean():.3f}, "
              f"mean rho(1..{min(10,lmax)}) = {np.nanmean(rho[1:min(11,lmax+1)]):.4f}")
        ulsa_test.append(ulsa); rec_test.append(rec)

    print("\nSpectral shapes (recovered vs truth, anchored at f_ref=1):")
    for i in range(n_templates):
        print(f"  template {i}: truth {np.round(shapes_true[i],3).tolist()}")
        print(f"              rec   {np.round(shapes_hat[i],3).tolist()}")

    save = dict(flux_true=np.asarray(flux_true), flux_hat=flux_hat,
                shapes_true=np.asarray(shapes_true), shapes_hat=shapes_hat,
                shape_init=shape_init, freq=freq, test_freq=np.asarray(test_freq),
                lmax=lmax, Nside=Nside, ref_freq=ref_freq,
                ulsa_test=np.asarray(ulsa_test), rec_test=np.asarray(rec_test),
                chi2_history=res["chi2_history"], converged=res["converged"])

    # ---- HMC posterior (gauge-invariant total flux mean/std + shape errors) ----
    if sample:
        from lusee.Fitting import sample_posterior
        t0 = time.time()
        post = sample_posterior(
            exp.predict, exp.paramset, data, N_inv,
            num_samples=num_samples, num_warmup=num_warmup, seed=noise_seed,
            init_linear=exp.paramset.pack_linear(res["linear"]),
            init_nonlinear=exp.paramset.pack_nonlinear(res["nonlinear"]),
            engine=hmc_engine,
            hmc_num_integration_steps=hmc_num_integration_steps)
        print(f"\nHMC: {num_samples} samples in {time.time()-t0:.0f}s, "
              f"accept={post['accept']:.2f}")
        fa = [np.asarray(post["linear"][f"sep.flux.{i}"])
              for i in range(n_templates)]              # each (ns, nalm)
        sh_free = np.asarray(post["nonlinear"]["sep.shape"])  # (ns, n_templ, nfree)
        ns = sh_free.shape[0]
        full_sh = np.ones((ns, n_templates, len(freq)))
        full_sh[:, :, sky_mod.nonref] = sh_free
        pmean, pstd = [], []
        print("  total-flux posterior recovery vs ULSA:")
        for k, f in enumerate(test_freq):
            kf = int(np.argmin(np.abs(freq - f)))
            tot = sum(full_sh[:, i, kf][:, None] * fa[i]
                      for i in range(n_templates))       # (ns, nalm) per-sample total alm
            tmaps = np.stack([hp.alm2map(tot[s], Nside) for s in range(ns)])
            pmean.append(tmaps.mean(0)); pstd.append(tmaps.std(0))
            r = np.std(pmean[k] - ulsa_test[k]) / np.std(ulsa_test[k])
            snr = np.median(np.abs(pmean[k]) / np.maximum(pstd[k], 1e-9))
            print(f"    f={f:4.0f} MHz: post-mean resid {100*r:.0f}% of ULSA std; "
                  f"median per-pixel SNR={snr:.1f}")
        save.update(accept=post["accept"],
                    flux_post_mean_test=np.asarray(pmean),
                    flux_post_std_test=np.asarray(pstd),
                    shapes_post_mean=full_sh.mean(0),
                    shapes_post_std=full_sh.std(0))

    if outfile is None:
        outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "separable_fit_result.npz")
    np.savez(outfile, **save)
    print(f"\nsaved arrays -> {outfile}")
    return res


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
        description="Separable sky fit: T(theta,f) = sum_i flux_i(theta)*shape_i(f). "
                    "Template maps linear (Wiener), spectral shapes non-linear; "
                    "MAP + optional HMC.  Data are the real ULSA maps.",
        formatter_class=DefaultsHelpFormatter)
    p.add_argument("--lmax", type=int, default=31)
    p.add_argument("--nside", type=int, default=16, dest="Nside")
    p.add_argument("--n-templates", type=int, default=2, dest="n_templates")
    p.add_argument("--freq", type=float, nargs="+",
                   default=[15., 20., 25., 30., 35., 40., 45.],
                   help="frequencies in MHz")
    p.add_argument("--ref-freq", type=float, default=25.0, dest="ref_freq",
                   help="reference frequency where each shape is anchored to 1")
    p.add_argument("--obs-range", default="2025-02-01 13:00:00 to 2025-02-28 13:00:00",
                   dest="obs_range")
    p.add_argument("--dt-hours", type=float, default=4.0, help="time step in hours")
    p.add_argument("--taper", type=float, default=0.03)
    p.add_argument("--target-snr", type=float, default=1e4, dest="target_snr")
    p.add_argument("--init-from", choices=["data", "truth"], default="data",
                   dest="init_from",
                   help="shape init: 'data' (PCA of the data) or 'truth' (perturbed)")
    p.add_argument("--shape-perturb", type=float, default=0.25, dest="shape_perturb",
                   help="perturbation for init-from=truth")
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--inner-maxiter", type=int, default=1500, dest="inner_maxiter")
    p.add_argument("--inner-method", choices=["auto", "cg", "dense"],
                   default="auto", dest="inner_method",
                   help="linear solve method for the VarPro inner step")
    p.add_argument("--dense-threshold", type=int, default=512, dest="dense_threshold",
                   help="n_linear threshold where --inner-method auto selects dense")
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
                   help="output .npz path (default: separable_fit_result.npz)")
    for action in p._actions:
        if action.option_strings and action.help is None:
            action.help = "(default: %(default)s)"
    a = p.parse_args(argv)
    run(lmax=a.lmax, Nside=a.Nside, n_templates=a.n_templates, freq=tuple(a.freq),
        ref_freq=a.ref_freq, obs_range=a.obs_range, dt_sec=a.dt_hours * 3600.0,
        taper=a.taper, maxiter=a.maxiter, inner_maxiter=a.inner_maxiter,
        inner_method=a.inner_method, dense_threshold=a.dense_threshold,
        init_from=a.init_from, shape_perturb=a.shape_perturb,
        target_snr=a.target_snr, sample=a.sample, hmc_engine=a.hmc_engine,
        hmc_num_integration_steps=a.hmc_num_integration_steps,
        num_samples=a.num_samples,
        num_warmup=a.num_warmup, noise_seed=a.noise_seed, outfile=a.outfile)


if __name__ == "__main__":
    main()
