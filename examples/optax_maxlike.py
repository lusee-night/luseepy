"""
Draft: directly minimise the Camacho et al. 2026 log-posterior with
JAX + optax, instead of solving the Wiener-filter normal equations.

    L(m) = (d - A m)^T N^{-1} (d - A m) + m^T S^{-1} m

The sky m is a pytree (HealpixSky), A m is a simulator call, and
optax updates operate leaf-wise — so the script is agnostic to the
pytree layout.

Simulator portability: the only place that touches a specific
simulator is `forward(sim, sky)`. CroSimulator accepts a sky= kwarg
that flows gradients; JaxSimulator does not yet, so a small shim
would be added there when needed.

This is a structural draft — beam/sky/observation choices and
hyper-parameters are placeholders.
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import numpy as np
import jax
import jax.numpy as jnp
import optax
import healpy as hp

import lusee


# ── Config ──────────────────────────────────────────────────────────

DRIVE = os.environ["LUSEE_DRIVE_DIR"]
BEAM_FILE = os.path.join(
    DRIVE,
    "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits",
)
SKY_FILE = os.path.join(DRIVE, "Simulations/SkyModels/ULSA_32_ddi_smooth.fits")

LMAX = 32
FREQ = np.arange(10.0, 41.0, 5.0)        # MHz — joint multi-freq solve
OBS_RANGE = "2025-02-01 13:00:00 to 2025-02-28 13:00:00"
DT_SEC = 7200.0
DF_HZ = 1e6

FREQ_LOGKERNEL_SIGMA = 0.15              # dex; larger → smoother in freq
FREQ_KERNEL_JITTER = 1e-6

LR = 1e-2
N_ITERS = 2000
PRINT_EVERY = 50


# ── Simulator setup (CroSimulator; swap later for JaxSimulator) ─────

def make_simulator(freq, lmax):
    """Build a CroSimulator with rotated + tapered beams and Tground=0.

    The paper's forward model is purely linear, so we drop the constant
    ground contribution. A dummy all-ones sky is used here only to
    satisfy the constructor; the real sky is passed into simulate().
    """
    layout = [("N", 0), ("E", -90), ("S", -180), ("W", -270)]
    beams = []
    for name, angle in layout:
        b = lusee.Beam(BEAM_FILE, id=name).rotate(angle)
        b.taper_and_smooth(taper=0.03)
        beams.append(b)

    obs = lusee.Observation(OBS_RANGE, deltaT_sec=DT_SEC,
                            lun_lat_deg=-10.0, lun_long_deg=180.0)

    # 4 autos + 6 crosses = 10 combinations (16 real channels)
    combinations = [(0, 0), (1, 1), (2, 2), (3, 3),
                    (0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3)]

    dummy_sky = lusee.sky.HealpixSky(
        8, lmax,
        maps=[np.ones(12 * 64) for _ in freq],
        freq=freq, frame="galactic",
    )
    sim = lusee.CroSimulator(
        obs, beams, dummy_sky,
        Tground=0.0, combinations=combinations,
        freq=freq, lmax=lmax,
    )
    return sim


def forward(sim, sky):
    """A @ m. One place to touch when switching simulators."""
    return sim.simulate(sky=sky)


def load_truth_sky(freq, lmax):
    sky_full = lusee.sky.FitsSky(SKY_FILE, lmax=lmax)
    idx = [int(np.argmin(np.abs(np.asarray(sky_full.freq) - f))) for f in freq]
    maps = [hp.alm2map(np.asarray(sky_full.mapalm[i]), sky_full.Nside, verbose=False)
            for i in idx]
    return lusee.sky.HealpixSky(sky_full.Nside, lmax,
                                maps=maps, freq=freq, frame="galactic")


# ── Noise and prior ────────────────────────────────────────────────

def radiometric_sigma(data, combinations, df_hz, dt_sec):
    """σ for each real channel from the radiometer equation (paper Eq. 9).

    data layout: (ntimes, nchannels, nfreq) with auto pairs contributing
    1 channel and cross pairs contributing (Re, Im) = 2 channels.
    """
    data = np.asarray(data)
    sigma = np.zeros_like(data, dtype=np.float64)

    auto_ch, ch = {}, 0
    for i, j in combinations:
        if i == j:
            auto_ch[i] = ch
            ch += 1
        else:
            ch += 2

    ch = 0
    for i, j in combinations:
        if i == j:
            T = np.abs(data[:, ch, :])
            sigma[:, ch, :] = T / np.sqrt(df_hz * dt_sec)
            ch += 1
        else:
            Tii = np.abs(data[:, auto_ch[i], :])
            Tjj = np.abs(data[:, auto_ch[j], :])
            V2 = data[:, ch, :] ** 2 + data[:, ch + 1, :] ** 2
            s = np.sqrt((Tii * Tjj + V2) / (4.0 * df_hz * dt_sec))
            sigma[:, ch, :] = s
            sigma[:, ch + 1, :] = s
            ch += 2
    return jnp.asarray(sigma)


def cl_inv_from_sky(sky, lmax):
    """Per-alm 1/C_l vector, broadcast to sky.mapalm shape."""
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


def freq_kernel_inv(freq, log_sigma, jitter):
    """Inverse of K_f = exp(-(Δlog f)^2 / 2σ^2) + jitter·I."""
    lf = jnp.log(jnp.asarray(freq))
    K = jnp.exp(-0.5 * (lf[:, None] - lf[None, :]) ** 2 / log_sigma ** 2)
    K = K + jitter * jnp.eye(len(freq))
    return jnp.linalg.inv(K)


# ── Assemble and run ───────────────────────────────────────────────

sim = make_simulator(FREQ, LMAX)
sky_truth = load_truth_sky(FREQ, LMAX)

data_clean = forward(sim, sky_truth)
sigma = radiometric_sigma(data_clean, sim.combinations, DF_HZ, DT_SEC)
noise = sigma * jax.random.normal(jax.random.PRNGKey(0), data_clean.shape)
data = data_clean + noise
N_inv = 1.0 / sigma ** 2

S_inv_cl = cl_inv_from_sky(sky_truth, LMAX)
K_f_inv = freq_kernel_inv(FREQ, FREQ_LOGKERNEL_SIGMA, FREQ_KERNEL_JITTER)


# Parameters = the sky pytree itself. Zero-initialised → prior mean.
params = jax.tree.map(jnp.zeros_like, sky_truth)


def loss_fn(sky):
    # Data term
    r = (data - forward(sim, sky)).ravel()
    chi2 = jnp.sum(N_inv.ravel() * r ** 2)

    # Prior with S = K_f ⊗ diag(C_l):
    #   m^T S^{-1} m = sum_{lm} (1/C_l) a[:,lm]^H K_f_inv a[:,lm]
    alm = sky.mapalm                                 # (nfreq, nalm) complex
    Kinv_a = jnp.einsum('fg,ga->fa', K_f_inv, alm)
    prior = jnp.real(jnp.sum(jnp.conj(alm) * Kinv_a * S_inv_cl))

    return chi2 + prior


value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

optimizer = optax.adam(LR)
opt_state = optimizer.init(params)


@jax.jit
def step(params, opt_state):
    loss, grads = value_and_grad(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss


t0 = time.time()
for it in range(N_ITERS):
    params, opt_state, loss = step(params, opt_state)
    if it % PRINT_EVERY == 0:
        print(f"iter {it:5d}  loss {float(loss):.6e}")
print(f"optimised in {time.time() - t0:.1f}s")

sky_hat = params                  # full HealpixSky pytree at the optimum
