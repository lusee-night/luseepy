"""
Linear map-making via conjugate gradients with autodiff adjoints.

Implements the Wiener filter from Camacho et al. 2026 (arXiv:2508.16773):

    s_hat = (A^T N^{-1} A + S^{-1})^{-1} A^T N^{-1} d

The sky is real, so we parameterize it with real variables
θ = [Re(a_{lm}); Im(a_{l,m>0})].  The forward model A maps θ to
real data through complex beam convolution, and A^T is obtained
directly from jax.vjp (no Wirtinger conjugation needed because θ
is real).  This eliminates the Im(a_{l,0}) null space that would
create condition numbers of ~10^12 with a complex parameterization.

See docs/wirtinger_cg.md for the complex-variable case (needed if
the CG variable itself is complex, e.g. for complex beam unknowns).
"""

import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp

from .Beam import Beam
from .Covariance import default_product_labels, normalize_products
from .InstrumentResponse import InstrumentResponse
from .Observation import Observation
from .CroSimulator import CroSimulator
from .ReceiverImpedance import JFETReceiver
from .SkyModels import HealpixSky


# Default antenna layout: 4 monopoles rotated by 90 degrees
DEFAULT_LAYOUT = [("N", 0), ("E", -90), ("S", -180), ("W", -270)]

# All 10 antenna combinations (4 auto + 6 cross -> 16 real channels)
DEFAULT_COMBINATIONS = [
    (0, 0), (1, 1), (2, 2), (3, 3),
    (0, 2), (1, 3), (0, 1), (1, 2), (0, 3), (2, 3),
]


class TargetAlignedHarmonicSky:
    """Differentiable alms already aligned with the simulator target axis."""

    def __init__(self, mapalm, freq, frame):
        self.mapalm = mapalm
        self.freq = np.asarray(freq, dtype=np.float64)
        self.frame = str(frame)

    def get_alm_at_freq(self, target_freq):
        target_freq = np.asarray(target_freq, dtype=np.float64)
        if not np.array_equal(target_freq, self.freq):
            raise ValueError(
                "TargetAlignedHarmonicSky can only be evaluated on its "
                "authoritative target frequency array."
            )
        return self.mapalm


def build_instrument(beam_file, obs_range, freq, lmax,
                     layout=DEFAULT_LAYOUT, combinations=DEFAULT_COMBINATIONS,
                     taper=0.03, dt_sec=3600.0,
                     lun_lat_deg=-10.0, lun_long_deg=180.0,
                     receiver=None, response_rotation_deg=0.0):
    """Build a CroSimulator for a response-v3 or legacy beam file.

    Returns ``(simulator, response_or_beams, observation)``.

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
    :param receiver: Receiver model for an instrument-response FITS v3 file
    :param response_rotation_deg: Whole-instrument azimuth rotation for v3
    """
    obs = Observation(
        obs_range, deltaT_sec=dt_sec,
        lun_lat_deg=lun_lat_deg, lun_long_deg=lun_long_deg,
    )

    # Tground=0: the paper's forward model (Eq 8) is purely linear
    sky_native_freq = np.unique(np.asarray(freq, dtype=np.float64))
    sky_dummy = HealpixSky(
        8, lmax,
        maps=[np.ones(12 * 64) for _ in sky_native_freq],
        freq=sky_native_freq, frame="galactic",
    )
    response_or_beam = Beam(beam_file)
    if isinstance(response_or_beam, InstrumentResponse):
        response = response_or_beam
        if response_rotation_deg:
            response = response.rotate(response_rotation_deg)
        if receiver is None:
            receiver = JFETReceiver()
        sim = CroSimulator(
            obs,
            response,
            sky_dummy,
            receiver,
            T_moon=0.0,
            products="all",
            freq=freq,
            lmax=lmax,
        )
        return sim, response, obs

    beams = []
    for name, angle in layout:
        b = Beam(beam_file, id=name)
        b = b.rotate(angle)
        b.taper_and_smooth(taper=taper)
        beams.append(b)
    sim = CroSimulator(
        obs, beams, sky_dummy, Tground=0.0,
        combinations=combinations, freq=freq, lmax=lmax,
    )
    return sim, beams, obs



def _real_alm_indices(lmax):
    """Return (m0_indices, mpos_indices) for healpy alm packing.

    m0_indices: alm indices with m=0 (real-only for a real sky)
    mpos_indices: alm indices with m>0 (complex)
    """
    m0 = np.array(
        [hp.Alm.getidx(lmax, ell, 0) for ell in range(lmax + 1)]
    )
    nalm = (lmax + 1) * (lmax + 2) // 2
    mpos = np.array([i for i in range(nalm) if i not in m0])
    return m0, mpos


def solve(sim, data, sky_template, sigma,
          signal_prior=None, lmax=None, maxiter=500, tol=1e-12,
          precondition=True, method='cg'):
    """Solve the normal equations in a real parameterization.

    The sky is real, so a_{l,0} are real and a_{l,m>0} are complex.
    The solver operates on a real vector θ = [Re(a_{lm}); Im(a_{l,m>0})],
    eliminating the Im(a_{l,0}) null space that would otherwise
    create condition numbers of ~10^12.

    :param sim: CroSimulator (Tground=0) from build_instrument
    :param data: Waterfall data array (ntimes, nchannels, nfreq)
    :param sky_template: HealpixSky with the correct shape/freq/frame
    :param sigma: Noise standard deviation (scalar or per-sample array)
    :param signal_prior: S^{-1} array, same shape as sky_template.mapalm.
        If None, uses a small Tikhonov regularizer (1e-6).
    :param lmax: Maximum multipole. If None, inferred from shape.
    :param maxiter: Maximum CG iterations (CG method only)
    :param tol: CG convergence tolerance (CG method only)
    :param precondition: If True, use diagonal S preconditioner (M = C_l)
        to improve CG convergence for ill-conditioned systems.
    :param method: 'cg' for conjugate gradients (default), 'direct' for
        dense Cholesky (exact but O(n_theta^3), feasible for n_theta < ~5000).
        The paper (Camacho et al. 2026) uses direct inversion.
    :returns: Recovered sky mapalm array (same shape as sky_template.mapalm)
    """
    target_freq = np.asarray(sky_template.freq, dtype=np.float64)
    sky_frame = sky_template.frame

    def make_sky(mapalm):
        return TargetAlignedHarmonicSky(
            mapalm,
            target_freq,
            sky_frame,
        )

    sigma_arr = jnp.asarray(sigma)
    N_inv = 1.0 / sigma_arr**2
    if N_inv.ndim > 0:
        N_inv = N_inv.ravel()  # match data.ravel() layout

    if signal_prior is None:
        S_inv = 1e-6
    else:
        S_inv = jnp.asarray(signal_prior)

    if lmax is None:
        nalm = sky_template.mapalm.shape[-1]
        lmax = int((-3 + np.sqrt(9 + 8 * (nalm - 1))) / 2)

    nfreq = sky_template.mapalm.shape[0]
    nalm = sky_template.mapalm.shape[-1]
    m0_idx, mpos_idx = _real_alm_indices(lmax)
    m0_idx = jnp.asarray(m0_idx)
    mpos_idx = jnp.asarray(mpos_idx)
    n_mpos = len(mpos_idx)
    # θ has shape (nfreq, nalm + n_mpos):
    #   θ[:, :nalm]  = Re(alm) for all (l,m)
    #   θ[:, nalm:]  = Im(alm) for m>0 only
    n_theta = nalm + n_mpos

    # S^{-1} in real parameterization.  For an isotropic field:
    #   E[a_{l,0}^2] = C_l           → S^{-1} = 1/C_l  for Re(a_{l,0})
    #   E[|a_{l,m>0}|^2] = C_l       → E[Re^2] = E[Im^2] = C_l/2
    #                                 → S^{-1} = 2/C_l  for Re and Im of m>0
    if isinstance(S_inv, (int, float)):
        S_inv_real = S_inv
    else:
        S_inv_re = S_inv.at[:, mpos_idx].multiply(2.0)  # 2/C_l for m>0
        S_inv_im = 2.0 * S_inv[:, mpos_idx]
        S_inv_real = jnp.concatenate([S_inv_re, S_inv_im], axis=-1)

    def theta_to_alm(theta):
        """Map real parameter vector to complex alm."""
        re = theta[:, :nalm]
        im_mpos = theta[:, nalm:]
        im_full = jnp.zeros((nfreq, nalm)).at[:, mpos_idx].set(
            im_mpos, unique_indices=True)
        return re + 1j * im_full

    def A(theta):
        """Forward model on real parameters."""
        alm = theta_to_alm(theta)
        return sim.simulate(sky=make_sky(alm)).ravel()

    if method == 'direct':
        if nfreq != 1:
            raise NotImplementedError(
                "method='direct' currently only supports single-frequency solves "
                "(see _solve_direct_singlefreq). Use method='cg' for nfreq > 1."
            )
        return _solve_direct_singlefreq(A, theta_to_alm, N_inv, S_inv_real,
                                        data, nfreq, n_theta)

    def cg_matvec(theta):
        """(A^T A / σ² + S^{-1}) θ  — real symmetric operator."""
        fwd, vjp_fn = jax.vjp(A, theta)
        return vjp_fn(N_inv * fwd)[0] + S_inv_real * theta

    # Diagonal preconditioner M = diag(C_l) in the real parameterization.
    # S^{-1} spans ~5 orders of magnitude (low l to high l), so this
    # dramatically improves conditioning.  Without it, CG can appear to
    # converge (small residual) while the solution is still far from the
    # true optimum — producing spurious negative holes.
    if precondition and not isinstance(S_inv_real, (int, float)):
        precond_diag = 1.0 / jnp.maximum(S_inv_real, 1e-30)

        def precond_fn(values):
            return values * precond_diag
    else:
        precond_fn = None

    zero = jnp.zeros((nfreq, n_theta))
    _, vjp_rhs = jax.vjp(A, zero)
    rhs = vjp_rhs(N_inv * data.ravel())[0]

    theta_hat, info = jax.scipy.sparse.linalg.cg(
        cg_matvec, rhs, x0=zero, M=precond_fn,
        maxiter=maxiter, tol=tol,
    )
    return theta_to_alm(theta_hat)


def _solve_direct_singlefreq(A, theta_to_alm, N_inv, S_inv_real, data, nfreq, n_theta):
    """Exact solve via Jacobian + Cholesky (same as Camacho et al. 2026).

    Single-frequency only: the Jacobian loop below probes unit vectors at
    ``zero.at[0, i].set(1.0)``, so only the nfreq=0 slice of theta is varied.
    Multi-frequency would need ``nfreq * n_theta`` columns.

    Builds the Jacobian column-by-column using the linearity of A,
    forms the normal matrix H = J^T N^{-1} J + S^{-1}, and solves
    via Cholesky decomposition.  All computation stays on GPU to
    avoid per-column sync overhead.

    Cost: O(n_theta) forward evaluations + O(n_theta^3) Cholesky.
    Feasible for n_theta < ~5000 (lmax < ~50 single-frequency).
    """
    zero = jnp.zeros((nfreq, n_theta))

    # JIT-compile A so the Python overhead (loops in CroSimulator, sky
    # construction) is paid once at trace time.  Subsequent calls dispatch
    # a single compiled XLA kernel.
    A_jit = jax.jit(A)
    _ = A_jit(zero)  # trace + compile (slow)

    # Build Jacobian column-by-column on GPU: J[:,i] = A(e_i) since A is linear.
    # Everything stays GPU-resident — no per-column GPU→CPU sync.
    cols = []
    for i in range(n_theta):
        e_i = zero.at[0, i].set(1.0)
        cols.append(A_jit(e_i))
    J = jnp.stack(cols, axis=-1)  # (ndata, n_theta)

    # Normal equations on GPU: H θ = J^T N^{-1} d
    d_flat = jnp.asarray(data).ravel()
    if isinstance(N_inv, (int, float)):
        JtNJ = N_inv * (J.T @ J)
        rhs = N_inv * (J.T @ d_flat)
    else:
        N_inv_j = jnp.asarray(N_inv).ravel()
        JtNJ = (J.T * N_inv_j) @ J
        rhs = (J.T * N_inv_j) @ d_flat

    if isinstance(S_inv_real, (int, float)):
        H = JtNJ + S_inv_real * jnp.eye(n_theta)
    else:
        H = JtNJ + jnp.diag(jnp.asarray(S_inv_real).ravel())

    # Cholesky solve on GPU
    L = jnp.linalg.cholesky(H)
    theta_flat = jax.scipy.linalg.cho_solve((L, True), rhs)
    theta_hat = theta_flat.reshape(nfreq, n_theta)

    return theta_to_alm(theta_hat)


def compute_radiometric_noise(data, combinations=None,
                              delta_f_hz=1e6, delta_t_sec=7200.0,
                              products=None):
    """Compute per-sample radiometric noise sigma from packed covariance.

    Implements the radiometer equation (Camacho et al. 2026, Eq. 9):

        sigma^2_ij = (T_ii T_jj + |V_ij|^2) / (2 df dt)

    For auto-correlations (real): sigma^2 = T_ii^2 / (df dt).
    For each Re/Im component of a cross-correlation:
        sigma^2 = (T_ii T_jj + |V_ij|^2) / (4 df dt).

    The data itself is used to estimate the system temperatures T_ii(t).
    This is a good approximation when SNR >> 1 (always true for
    total-power radiometry at LuSEE frequencies).

    :param data: Waterfall array (ntimes, nchannels, nfreq).
        The default is the response-v3 16-channel order.
    :param combinations: Legacy list of (i, j) beam-pair tuples
    :param delta_f_hz: Effective channel bandwidth in Hz (default 1 MHz).
    :param delta_t_sec: Integration time per sample in seconds (default 7200).
    :param products: Response-v3 product labels or ``"all"``
    :returns: Sigma array with the same shape as data.
    """
    data_np = np.asarray(data)
    ntimes, nchannels, nfreq = data_np.shape
    sigma = np.zeros_like(data_np, dtype=np.float64)

    if products is not None or combinations is None:
        labels = (
            default_product_labels()
            if products is None
            else normalize_products(products)
        )
        if len(labels) != nchannels:
            raise ValueError(
                f"Product metadata has {len(labels)} channels but data has "
                f"{nchannels}."
            )
        channel_for = {label: index for index, label in enumerate(labels)}
        for label, channel in channel_for.items():
            a, b = int(label[0]), int(label[1])
            auto_a = channel_for.get(f"{a}{a}R")
            auto_b = channel_for.get(f"{b}{b}R")
            if auto_a is None or auto_b is None:
                raise ValueError(
                    f"Products must include autos {a}{a}R and {b}{b}R."
                )
            T_aa = np.abs(data_np[:, auto_a, :])
            if a == b:
                sigma[:, channel, :] = T_aa / np.sqrt(
                    delta_f_hz * delta_t_sec
                )
                continue
            T_bb = np.abs(data_np[:, auto_b, :])
            real_channel = channel_for.get(f"{a}{b}R")
            imag_channel = channel_for.get(f"{a}{b}I")
            real = (
                data_np[:, real_channel, :]
                if real_channel is not None
                else 0.0
            )
            imag = (
                data_np[:, imag_channel, :]
                if imag_channel is not None
                else 0.0
            )
            variance = (
                T_aa * T_bb + real**2 + imag**2
            ) / (4.0 * delta_f_hz * delta_t_sec)
            sigma[:, channel, :] = np.sqrt(np.maximum(variance, 1e-30))
        return jnp.asarray(sigma)

    combinations = tuple(combinations)

    # First pass: record which channel holds each auto-correlation
    auto_ch = {}
    ch = 0
    for i, j in combinations:
        if i == j:
            auto_ch[i] = ch
            ch += 1
        else:
            ch += 2

    # Second pass: compute noise per channel
    ch = 0
    for i, j in combinations:
        if i == j:
            # Auto-correlation: σ² = T_ii² / (Δf Δt)
            T_ii = np.abs(data_np[:, ch, :])
            sigma[:, ch, :] = T_ii / np.sqrt(delta_f_hz * delta_t_sec)
            ch += 1
        else:
            # Cross-correlation: each Re/Im component has
            # σ² = (T_ii T_jj + |V_ij|²) / (4 Δf Δt)
            T_ii = np.abs(data_np[:, auto_ch[i], :])
            T_jj = np.abs(data_np[:, auto_ch[j], :])
            V_re = data_np[:, ch, :]
            V_im = data_np[:, ch + 1, :]
            V_sq = V_re**2 + V_im**2
            var = (T_ii * T_jj + V_sq) / (4.0 * delta_f_hz * delta_t_sec)
            sig = np.sqrt(np.maximum(var, 1e-30))
            sigma[:, ch, :] = sig
            sigma[:, ch + 1, :] = sig
            ch += 2

    return jnp.asarray(sigma)


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
        for ell in range(min(len(cl), lmax + 1)):
            if cl[ell] > 0:
                for m in range(ell + 1):
                    idx = hp.Alm.getidx(lmax, ell, m)
                    if idx < nalm:
                        s_inv[fi, idx] = 1.0 / cl[ell]
    return jnp.asarray(s_inv)
