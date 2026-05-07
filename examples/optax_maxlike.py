"""
Minimal Optax-based Wiener-filter sky reconstruction for LuSEE-Night.

Minimises the Camacho et al. 2026 Gaussian log-posterior

    L(m) = (d - A m)^T N^{-1} (d - A m) + m^T S^{-1} m

with optax.lbfgs, where m is the sky parameterised by real degrees of
freedom (lusee.sky.RealAlmSky: re for all m, im_mpos for m>0). The point
of this script is pedagogical: it shows the smallest loop that gets you
from a forward-model (sim.simulate(sky=...)) to a reconstructed sky.
Use it as a template; swap the optimiser, loss, prior, or simulator
as needed.

Set WHITEN=False at the top of the file to compare against optimisation
in physical coords (typically converges much more slowly).

Run:
    python examples/optax_maxlike.py
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import jax.numpy as jnp
import optax
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import lusee inside main() -- see CLAUDE.md (macOS multiprocessing).


# Configuration
DRIVE = os.environ["LUSEE_DRIVE_DIR"]
BEAM_FILE = DRIVE + "/Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits"
SKY_FILE = DRIVE + "/Simulations/SkyModels/ULSA_32_ddi_smooth.fits"

FREQ = np.array([20.0, 25.0, 30.0])
OBS_RANGE = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
DT_SEC = 7200.0
DF_HZ = 1e6
LMAX = 32
N_ITERS = 500
LBFGS_MEMORY = 20

# Whitening (change of variable). With WHITEN=True we optimise in
# theta' = theta * sqrt(S^-1) so the prior is isotropic (||theta'||^2),
# which compresses the Hessian dynamic range by ~kappa(S^-1) and
# typically converges faster. Set False to optimise in physical coords.
WHITEN = True

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "optax_maxlike_out")


def make_simulator(lusee, freq, lmax):
    layout = [("N", 0), ("E", -90), ("S", -180), ("W", -270)]
    beams = []
    for name, angle in layout:
        b = lusee.Beam(BEAM_FILE, id=name).rotate(angle)
        b.taper_and_smooth(taper=0.03)
        beams.append(b)
    obs = lusee.Observation(OBS_RANGE, deltaT_sec=DT_SEC,
                            lun_lat_deg=-10.0, lun_long_deg=180.0)
    combinations = [(0, 0), (1, 1), (2, 2), (3, 3),
                    (0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3)]
    dummy_sky = lusee.sky.HealpixSky(
        8, lmax, maps=[np.ones(12 * 64) for _ in freq],
        freq=freq, frame="galactic",
    )
    return lusee.JaxSimulator(
        obs, beams, dummy_sky, Tground=0.0,
        combinations=combinations, freq=freq, lmax=lmax,
    )


def load_truth_sky(lusee, freq, lmax):
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    idx = [int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in freq]
    maps = [hp.alm2map(np.asarray(sky_full.mapalm[i]), sky_full.Nside,
                       verbose=False) for i in idx]
    return lusee.sky.HealpixSky(sky_full.Nside, lmax, maps=maps,
                                freq=freq, frame="galactic")


def radiometric_sigma(data, combinations, df_hz, dt_sec):
    """Per-channel sigma from the radiometer equation (Camacho 2026 Eq. 9)."""
    data = np.asarray(data)
    sigma = np.zeros_like(data, dtype=np.float64)
    auto_ch, ch = {}, 0
    for i, j in combinations:
        if i == j:
            auto_ch[i] = ch; ch += 1
        else:
            ch += 2
    ch = 0
    for i, j in combinations:
        if i == j:
            sigma[:, ch, :] = np.abs(data[:, ch, :]) / np.sqrt(df_hz * dt_sec)
            ch += 1
        else:
            Tii = np.abs(data[:, auto_ch[i], :])
            Tjj = np.abs(data[:, auto_ch[j], :])
            V2 = data[:, ch, :] ** 2 + data[:, ch + 1, :] ** 2
            s = np.sqrt((Tii * Tjj + V2) / (4.0 * df_hz * dt_sec))
            sigma[:, ch, :] = s; sigma[:, ch + 1, :] = s
            ch += 2
    return jnp.asarray(sigma)


def cl_inv_full(sky, lmax):
    """Per-(nfreq, nalm) array of 1/C_l from the truth sky."""
    nfreq, nalm = sky.mapalm.shape
    out = np.zeros((nfreq, nalm))
    for f in range(nfreq):
        cl = hp.alm2cl(np.asarray(sky.mapalm[f]))
        for l in range(min(len(cl), lmax + 1)):
            if cl[l] <= 0:
                continue
            for m in range(l + 1):
                idx = hp.Alm.getidx(lmax, l, m)
                if idx < nalm:
                    out[f, idx] = 1.0 / cl[l]
    return jnp.asarray(out)


def save_plots(sky_truth, sky_hat, out_dir, freqs):
    os.makedirs(out_dir, exist_ok=True)
    nside = sky_truth.Nside
    for fi, f in enumerate(freqs):
        true_alm = np.asarray(sky_truth.mapalm[fi])
        rec_alm = np.asarray(sky_hat.mapalm[fi])
        true_map = hp.alm2map(true_alm, nside, verbose=False)
        rec_map = hp.alm2map(rec_alm, nside, verbose=False)
        resid = rec_map - true_map
        fig = plt.figure(figsize=(15, 4))
        hp.mollview(true_map, title=f"ULSA input ({f:.0f} MHz)",
                    cmap="inferno", sub=(1, 3, 1), fig=fig)
        hp.mollview(rec_map, title="reconstruction",
                    cmap="inferno", sub=(1, 3, 2), fig=fig)
        hp.mollview(resid, title="residual (rec - true)",
                    cmap="RdBu_r", sub=(1, 3, 3), fig=fig)
        fig.savefig(os.path.join(out_dir, f"maps_{int(f):02d}MHz.png"), dpi=120)
        plt.close(fig)


def main():
    import lusee

    sim = make_simulator(lusee, FREQ, LMAX)
    sky_truth = load_truth_sky(lusee, FREQ, LMAX)

    # Synthesise mock data with radiometric noise
    data_clean = sim.simulate(sky=sky_truth)
    sigma = radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
    noise = sigma * jax.random.normal(jax.random.PRNGKey(0), data_clean.shape)
    data = data_clean + noise

    N_inv_flat = (1.0 / sigma ** 2).ravel()
    data_flat = data.ravel()

    # Real-DOF sky parameterisation. Leaves: re (all m), im_mpos (m>0).
    cl_inv = cl_inv_full(sky_truth, LMAX)
    params = lusee.sky.RealAlmSky.zeros_like(sky_truth, LMAX)
    re_diag, im_diag = params.prior_inv_diag(cl_inv)

    if WHITEN:
        sqrt_S_re = 1.0 / jnp.sqrt(re_diag)
        sqrt_S_im = 1.0 / jnp.sqrt(im_diag)
        prior_re = jnp.ones_like(re_diag)
        prior_im = jnp.ones_like(im_diag)
    else:
        sqrt_S_re = jnp.ones_like(re_diag)
        sqrt_S_im = jnp.ones_like(im_diag)
        prior_re, prior_im = re_diag, im_diag

    def dewhiten(sky):
        """Whitened -> physical sky. Identity when WHITEN=False."""
        return lusee.sky.RealAlmSky(
            re=sky.re * sqrt_S_re, im_mpos=sky.im_mpos * sqrt_S_im,
            lmax=sky.lmax, Nside=sky.Nside, frame=sky.frame,
        )

    def loss_fn(sky):
        residual = data_flat - sim.simulate(sky=dewhiten(sky)).ravel()
        chi2 = jnp.sum(N_inv_flat * residual ** 2)
        prior = (jnp.sum(prior_re * sky.re ** 2)
                 + jnp.sum(prior_im * sky.im_mpos ** 2))
        return chi2 + prior

    value_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.lbfgs(memory_size=LBFGS_MEMORY)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        value, grad = value_and_grad(params)
        updates, opt_state = optimizer.update(
            grad, opt_state, params,
            value=value, grad=grad, value_fn=loss_fn,
        )
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    # Optimisation loop
    for i in range(N_ITERS):
        params, opt_state, loss = step(params, opt_state)
        if i % 50 == 0 or i == N_ITERS - 1:
            print(f"iter {i:4d}: loss={float(loss):.6e}")

    save_plots(sky_truth, dewhiten(params), OUT_DIR, FREQ)
    print(f"plots saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
