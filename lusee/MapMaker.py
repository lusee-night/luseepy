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



def _real_alm_indices(lmax):
    """Return (m0_indices, mpos_indices) for healpy alm packing.

    m0_indices: alm indices with m=0 (real-only for a real sky)
    mpos_indices: alm indices with m>0 (complex)
    """
    m0 = np.array([hp.Alm.getidx(lmax, l, 0) for l in range(lmax + 1)])
    nalm = (lmax + 1) * (lmax + 2) // 2
    mpos = np.array([i for i in range(nalm) if i not in m0])
    return m0, mpos


def build_operators(sim, data, sky_template, sigma, signal_prior=None, lmax=None):
    """Build the forward model and normal-equation pieces in the real parameterization.

    Returns a dict with keys: A, theta_to_alm, N_inv, S_inv_real, rhs, zero,
    n_theta, nfreq, nalm, m0_idx, mpos_idx, lmax. Shared by solve() and
    gradient-based samplers.
    """
    _, aux = sky_template.tree_flatten()

    def make_sky(mapalm):
        return HealpixSky.tree_unflatten(aux, (mapalm,))

    sigma_arr = jnp.asarray(sigma)
    N_inv = 1.0 / sigma_arr**2
    if N_inv.ndim > 0:
        N_inv = N_inv.ravel()

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
    n_m0 = len(m0_idx)
    n_mpos = len(mpos_idx)
    n_theta = nalm + n_mpos

    if isinstance(S_inv, (int, float)):
        S_inv_real = S_inv
    else:
        S_inv_re = S_inv.at[:, mpos_idx].multiply(2.0)
        S_inv_im = 2.0 * S_inv[:, mpos_idx]
        S_inv_real = jnp.concatenate([S_inv_re, S_inv_im], axis=-1)

    def theta_to_alm(theta):
        re = theta[:, :nalm]
        im_mpos = theta[:, nalm:]
        im_full = jnp.zeros((nfreq, nalm)).at[:, mpos_idx].set(
            im_mpos, unique_indices=True)
        return re + 1j * im_full

    def A(theta):
        alm = theta_to_alm(theta)
        return sim.simulate(sky=make_sky(alm)).ravel()

    zero = jnp.zeros((nfreq, n_theta))
    _, vjp_rhs = jax.vjp(A, zero)
    rhs = vjp_rhs(N_inv * jnp.asarray(data).ravel())[0]

    return dict(
        A=A, theta_to_alm=theta_to_alm, make_sky=make_sky,
        N_inv=N_inv, S_inv_real=S_inv_real, rhs=rhs, zero=zero,
        n_theta=n_theta, nfreq=nfreq, nalm=nalm,
        m0_idx=m0_idx, mpos_idx=mpos_idx, lmax=lmax,
        data_flat=jnp.asarray(data).ravel(),
    )


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
    _, aux = sky_template.tree_flatten()

    def make_sky(mapalm):
        return HealpixSky.tree_unflatten(aux, (mapalm,))

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
    n_m0 = len(m0_idx)
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
        precond_fn = lambda x: x * precond_diag
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
    ndata = int(np.prod(data.shape))
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


def compute_ulsa_svd(ulsa_path, freq, K=3, lmax=None):
    """Compute the first K SVD frequency templates of a ULSA cube.

    Loads the ULSA FitsSky at the requested lmax, stacks maps into an
    (nfreq, npix) matrix, and takes the leading K left-singular vectors
    as frequency templates f_k(nu).

    :returns: (freq_templates (nfreq, K), spatial_maps (K, npix),
               singular_values (K,))
    """
    from .SkyModels import FitsSky
    sky = FitsSky(ulsa_path, lmax=lmax if lmax is not None else 64)
    # Select the frequencies we want
    freq = np.asarray(freq, dtype=float)
    sky_freq = np.asarray(sky.freq, dtype=float)
    idx = np.array([int(np.argmin(np.abs(sky_freq - f))) for f in freq])
    maps = np.asarray(sky.maps)[idx]  # (nfreq, npix)
    U, S, Vt = np.linalg.svd(maps, full_matrices=False)
    freq_templates = U[:, :K] * S[:K]  # absorb scale into templates
    spatial_maps = Vt[:K]              # unit-norm spatial modes
    return freq_templates, spatial_maps, S[:K]


def solve_svd_multifreq(sim, data, sky_template, sigma, freq_templates,
                        signal_prior=None, lmax=None,
                        maxiter=500, tol=1e-12, precondition=True):
    """Multi-frequency Wiener solve restricted to a frequency subspace.

    The sky is parameterized as

        a_{lm}(nu) = sum_{k=0}^{K-1} f_k(nu) * beta_k(l, m)

    where f_k(nu) are the K ``freq_templates`` (e.g. from ULSA SVD) and
    beta_k(l,m) are K single-frequency alm maps solved for jointly.
    This enforces that the sky lives in the outer product
    span(freq_templates) ⊗ alm_space, dramatically reducing the number
    of unknowns when nfreq >> K.

    For the LuSEE 5-50 MHz foreground, benchmarks (see notebooks/
    mapmaker_svd_multifreq_demo.ipynb) found K=4 ULSA-SVD templates to
    be the practical sweet spot: K=2 collapses, K=3 is the first usable
    setting, K=4 adds +0.05 in mean rho(1..20) and a large +0.13 gain
    at high-ell vs K=3, and K=5/6 plateau. The high-ell (ell 16..32)
    advantage over independent per-band Wiener solves is +0.04 to +0.15
    in rho at every frequency, at ~1/10th the wallclock cost.

    :param freq_templates: (nfreq, K) array of frequency mode amplitudes.
    :param signal_prior: S^{-1} of shape (K, nalm) -- prior on beta_k.
        If None, uses 1e-6 Tikhonov.
    :returns: beta_alm of shape (K, nalm) complex.
    """
    _, aux = sky_template.tree_flatten()

    def make_sky(mapalm):
        return HealpixSky.tree_unflatten(aux, (mapalm,))

    sigma_arr = jnp.asarray(sigma)
    N_inv = 1.0 / sigma_arr**2
    if N_inv.ndim > 0:
        N_inv = N_inv.ravel()

    F = jnp.asarray(freq_templates)              # (nfreq, K)
    nfreq_f, K = F.shape
    nfreq = sky_template.mapalm.shape[0]
    nalm = sky_template.mapalm.shape[-1]
    assert nfreq_f == nfreq, "freq_templates nfreq must match sky_template"

    if lmax is None:
        lmax = int((-3 + np.sqrt(9 + 8 * (nalm - 1))) / 2)
    m0_idx, mpos_idx = _real_alm_indices(lmax)
    m0_idx = jnp.asarray(m0_idx)
    mpos_idx = jnp.asarray(mpos_idx)
    n_mpos = len(mpos_idx)
    n_theta = nalm + n_mpos  # per SVD component

    if signal_prior is None:
        S_inv_real = 1e-6
    else:
        S_inv = jnp.asarray(signal_prior)        # (K, nalm)
        S_inv_re = S_inv.at[:, mpos_idx].multiply(2.0)
        S_inv_im = 2.0 * S_inv[:, mpos_idx]
        S_inv_real = jnp.concatenate([S_inv_re, S_inv_im], axis=-1)  # (K, n_theta)

    def theta_to_beta(theta):
        """theta: (K, n_theta) -> beta_alm: (K, nalm) complex."""
        re = theta[:, :nalm]
        im_mpos = theta[:, nalm:]
        im_full = jnp.zeros((K, nalm)).at[:, mpos_idx].set(
            im_mpos, unique_indices=True)
        return re + 1j * im_full

    def A(theta):
        beta_alm = theta_to_beta(theta)           # (K, nalm)
        # broadcast: alm[f, :] = sum_k F[f,k] * beta[k, :]
        alm = F @ beta_alm                        # (nfreq, nalm)
        return sim.simulate(sky=make_sky(alm)).ravel()

    def cg_matvec(theta):
        fwd, vjp_fn = jax.vjp(A, theta)
        out = vjp_fn(N_inv * fwd)[0]
        if isinstance(S_inv_real, (int, float)):
            return out + S_inv_real * theta
        return out + S_inv_real * theta

    if precondition and not isinstance(S_inv_real, (int, float)):
        precond_diag = 1.0 / jnp.maximum(S_inv_real, 1e-30)
        precond_fn = lambda x: x * precond_diag
    else:
        precond_fn = None

    zero = jnp.zeros((K, n_theta))
    _, vjp_rhs = jax.vjp(A, zero)
    rhs = vjp_rhs(N_inv * jnp.asarray(data).ravel())[0]

    theta_hat, info = jax.scipy.sparse.linalg.cg(
        cg_matvec, rhs, x0=zero, M=precond_fn,
        maxiter=maxiter, tol=tol,
    )
    return theta_to_beta(theta_hat)


def compute_radiometric_noise(data, combinations=DEFAULT_COMBINATIONS,
                              delta_f_hz=1e6, delta_t_sec=7200.0):
    """Compute per-sample radiometric noise σ from the data.

    Implements the radiometer equation (Camacho et al. 2026, Eq. 9):

        σ²_ij = (T_ii T_jj + |V_ij|²) / (2 Δf Δt)

    For auto-correlations (real): σ² = T_ii² / (Δf Δt).
    For each Re/Im component of a cross-correlation:
        σ² = (T_ii T_jj + |V_ij|²) / (4 Δf Δt).

    The data itself is used to estimate the system temperatures T_ii(t).
    This is a good approximation when SNR >> 1 (always true for
    total-power radiometry at LuSEE frequencies).

    :param data: Waterfall array (ntimes, nchannels, nfreq).
        Channel order follows CroSimulator convention: auto-correlations
        contribute 1 channel, cross-correlations contribute 2 (Re, Im).
    :param combinations: List of (i, j) beam-pair tuples matching the
        order used to build the simulator.
    :param delta_f_hz: Effective channel bandwidth in Hz (default 1 MHz).
    :param delta_t_sec: Integration time per sample in seconds (default 7200).
    :returns: σ array with same shape as data.
    """
    data_np = np.asarray(data)
    ntimes, nchannels, nfreq = data_np.shape
    sigma = np.zeros_like(data_np, dtype=np.float64)

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
        for l in range(min(len(cl), lmax + 1)):
            if cl[l] > 0:
                for m in range(l + 1):
                    idx = hp.Alm.getidx(lmax, l, m)
                    if idx < nalm:
                        s_inv[fi, idx] = 1.0 / cl[l]
    return jnp.asarray(s_inv)
