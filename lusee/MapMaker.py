"""
Linear map-making via conjugate gradients with autodiff adjoints.

Implements the Wiener filter from Camacho et al. 2026 (arXiv:2508.16773):

    s_hat = (A^H N^{-1} A + S^{-1})^{-1} A^H N^{-1} d

where A is the beam-convolution forward model (CroSimulator with Tground=0),
and the Hermitian adjoint A^H is obtained via conj(jax.vjp).

See docs/wirtinger_cg.md for why the conjugation is needed.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp

from .Beam import Beam
from .Observation import Observation
from .CroSimulator import CroSimulator
from .SkyModels import HealpixSky, FitsSky


# Default antenna layout: 4 monopoles rotated by 90 degrees
DEFAULT_LAYOUT = [("N", 0), ("E", -90), ("S", -180), ("W", -270)]

# All 10 antenna combinations (4 auto + 6 cross -> 16 real channels)
DEFAULT_COMBINATIONS = [
    (0, 0), (1, 1), (2, 2), (3, 3),
    (0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3),
]


def build_instrument(beam_file, obs_range, freq, lmax,
                     layout=DEFAULT_LAYOUT, combinations=DEFAULT_COMBINATIONS,
                     taper=0.03, dt_sec=3600.0,
                     lun_lat_deg=-10.0, lun_long_deg=180.0):
    """Build a CroSimulator with correctly rotated and tapered beams.

    Returns (simulator, beams, observation).

    :param beam_file: Path to beam FITS file (e.g. hfss_lbl_3m_75deg.fits)
    :param obs_range: Observation time range string (e.g. "2025-02-01 to 2025-02-28")
    :param freq: Frequency array in MHz
    :param lmax: Maximum multipole for beam alm computation
    :param layout: List of (name, rotation_deg) tuples
    :param combinations: List of (i, j) beam index pairs
    :param taper: Horizon taper width in radians (0.03 default)
    :param dt_sec: Time step in seconds
    :param lun_lat_deg: Lunar latitude in degrees
    :param lun_long_deg: Lunar longitude in degrees
    """
    beams = []
    for name, angle in layout:
        b = Beam(beam_file, id=name)
        b = b.rotate(angle)             # returns a copy!
        b.taper_and_smooth(taper=taper)  # modifies in place
        beams.append(b)

    obs = Observation(
        obs_range, deltaT_sec=dt_sec,
        lun_lat_deg=lun_lat_deg, lun_long_deg=lun_long_deg,
    )

    # Tground=0: the paper's forward model (Eq 8) is purely linear
    sky_dummy = HealpixSky(
        8, lmax,
        maps=[np.ones(12 * 64) for _ in freq],
        freq=freq, frame="galactic",
    )
    sim = CroSimulator(
        obs, beams, sky_dummy, Tground=0.0,
        combinations=combinations, freq=freq, lmax=lmax,
    )
    return sim, beams, obs


def _estimate_diagonal(matvec, shape, n_probes=3, key=None):
    """Estimate the diagonal of a linear operator via stochastic probing.

    Uses Hutchinson's estimator: diag(M) ≈ mean(z * M(z)) where z is
    a random ±1 vector.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    diag_est = jnp.zeros(shape)
    for i in range(n_probes):
        z = 2.0 * jax.random.bernoulli(jax.random.fold_in(key, i), shape=shape).astype(float) - 1.0
        Mz = matvec(z)
        diag_est = diag_est + jnp.real(z * Mz)
    return diag_est / n_probes


def solve(sim, data, sky_template, sigma,
          signal_prior=None, precondition=True, maxiter=50, tol=1e-6):
    """Solve the normal equations via CG.

    :param sim: CroSimulator (Tground=0) from build_instrument
    :param data: Waterfall data array (ntimes, nchannels, nfreq)
    :param sky_template: HealpixSky with the correct shape/freq/frame
        (used for pytree metadata; values are ignored)
    :param sigma: Noise standard deviation (scalar or per-sample array)
    :param signal_prior: S^{-1} array, same shape as sky_template.mapalm.
        If None, uses a small Tikhonov regularizer (1e-6).
    :param precondition: If True, estimate diagonal preconditioner from
        the normal operator to improve CG convergence.
    :param maxiter: Maximum CG iterations
    :param tol: CG convergence tolerance
    :returns: Recovered sky mapalm array (same shape as sky_template.mapalm)
    """
    _, aux = sky_template.tree_flatten()

    def make_sky(mapalm):
        return HealpixSky.tree_unflatten(aux, (mapalm,))

    def A(mapalm):
        return sim.simulate(sky=make_sky(mapalm)).ravel()

    N_inv = 1.0 / jnp.asarray(sigma)**2

    if signal_prior is None:
        S_inv = 1e-6
    else:
        S_inv = jnp.asarray(signal_prior)

    def cg_matvec(x):
        """(A^H N^{-1} A + S^{-1}) x."""
        fwd, vjp_fn = jax.vjp(A, x)
        return jnp.conj(vjp_fn(N_inv * fwd)[0]) + S_inv * x

    # Diagonal preconditioner: M^{-1} ≈ 1 / diag(A^H N^{-1} A + S^{-1})
    if precondition:
        diag = _estimate_diagonal(cg_matvec, sky_template.mapalm.shape)
        diag = jnp.maximum(jnp.abs(diag), 1e-30)  # avoid division by zero
        precond = lambda x: x / diag
    else:
        precond = None

    zero = jnp.zeros_like(sky_template.mapalm)
    _, vjp_rhs = jax.vjp(A, zero)
    rhs = jnp.conj(vjp_rhs(N_inv * data.ravel())[0])

    sky_hat, info = jax.scipy.sparse.linalg.cg(
        cg_matvec, rhs, x0=zero, M=precond, maxiter=maxiter, tol=tol,
    )
    return sky_hat


def compute_cl_prior(sky_model, lmax):
    """Compute S^{-1} = 1/C_l from a sky model's power spectrum.

    :param sky_model: HealpixSky or FitsSky with mapalm
    :param lmax: Maximum multipole
    :returns: S_inv array with shape matching sky_model.mapalm
    """
    nfreq, nalm = sky_model.mapalm.shape
    s_inv = np.zeros((nfreq, nalm))
    for fi in range(nfreq):
        alm_f = np.asarray(sky_model.mapalm[fi])
        cl = hp.alm2cl(alm_f)
        for l in range(min(len(cl), lmax + 1)):
            if cl[l] > 0:
                for m in range(l + 1):
                    idx = hp.Alm.getidx(lmax, l, m)
                    if idx < nalm:
                        s_inv[fi, idx] = 1.0 / cl[l]
    return jnp.asarray(s_inv)
