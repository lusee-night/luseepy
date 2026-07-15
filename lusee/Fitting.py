"""Profile-likelihood (variable-projection) fitting for models with a mix of
linear and non-linear parameters.

The organising principle is *conditional linearity*.  Write the parameters as
``theta = (eta, lam)`` where ``lam`` are linear and ``eta`` non-linear.  The
forward model is linear in ``lam`` at fixed ``eta``:

    m(eta, lam) = A(eta) @ lam

so for any ``eta`` the optimal ``lam`` is obtained exactly by a single linear
(Wiener) solve, and the outer optimiser only ever sees the small non-linear
block ``eta``.  The gradient of the profiled objective w.r.t. ``eta`` is exact
by the envelope theorem: at the inner optimum ``dF/dlam = 0``, so we may treat
``lam`` as constant (``stop_gradient``) and differentiate only the explicit
``eta`` dependence — no need to differentiate through the linear solve.

Layers
------
* **Registry** — :class:`Param` / :class:`ParamSet` declare each parameter's
  kind (linear/non-linear), prior, bounds, and (for linear params) a real
  reparameterisation.  ``ParamSet`` assembles the linear block ``S^{-1}``,
  packs/unpacks the linear and non-linear vectors, and sums the non-linear
  log-priors.  This is the reusable part that any model plugs into.
* **Linear block** — :class:`RealAlmBlock` reparameterises a real band-limited
  sky map as ``theta = [Re(a_lm); Im(a_{l,m>0})]`` and builds ``S^{-1}=1/C_l``.
* **Solver/driver** — :func:`linear_solve_cg` (matrix-free preconditioned CG,
  reverse-mode only so it works through ``s2fft``'s ``custom_vjp``) and
  :func:`profile_optimize` (generic VarPro: outer L-BFGS-B over ``eta``, inner
  exact ``lam`` solve).
* **Model wrappers** — :class:`Experiment` wires sky/beam/instrument modules
  into the ``ParamSet`` and ``predict`` assembly for
  :class:`~lusee.SpectralSky.SpectralHealpixSky`, i.e. the first concrete
  instance of the general system (see ``notebooks/spectral_fit_demo.py``).

The same ``ParamSet`` / ``predict`` pair is what an HMC log-posterior would
consume later (the marginalised-flux evidence just adds a ``1/2 log det`` term).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp
import scipy.optimize

from .MapMaker import _real_alm_indices, compute_radiometric_noise  # noqa: F401
# NOTE: this module is the generic fitting framework and deliberately does NOT
# import any concrete sky model.  Model-specific modules (e.g. SpectralSkyModule,
# SeparableSkyModule) live with their models and import Param/RealAlmBlock/
# ClPrior/Module from here -- the dependency points models -> framework.


# ============================================================================
# Linear block: real parameterisation of a band-limited real sky map
# ============================================================================

class RealAlmBlock:
    """Real parameterisation of a band-limited real sky map.

    A real sky has real ``a_{l,0}`` and complex ``a_{l,m>0}``; the imaginary
    parts of ``a_{l,0}`` are an exact null space.  We therefore parameterise by
    the real vector ``theta = [Re(a_lm) for all (l,m); Im(a_{l,m>0})]`` of
    length ``n_theta = nalm + n_mpos``.  (Same packing as ``MapMaker.solve``.)
    """

    def __init__(self, lmax):
        self.lmax = int(lmax)
        self.nalm = (lmax + 1) * (lmax + 2) // 2
        _, mpos = _real_alm_indices(lmax)
        self.mpos = jnp.asarray(mpos)
        self.n_mpos = int(len(mpos))
        self.n_theta = self.nalm + self.n_mpos

    def theta_to_natural(self, theta):
        """Real vector -> complex healpy-packed alm."""
        re = theta[:self.nalm]
        im = jnp.zeros(self.nalm).at[self.mpos].set(theta[self.nalm:],
                                                    unique_indices=True)
        return re + 1j * im

    def natural_to_theta(self, alm):
        """Complex healpy-packed alm -> real vector."""
        alm = jnp.asarray(alm)
        return jnp.concatenate([jnp.real(alm), jnp.imag(alm)[self.mpos]])

    def Sinv_from_cl(self, cl):
        """``S^{-1}`` in the real parameterisation from an angular power
        spectrum ``C_l``.

        For an isotropic Gaussian field ``E[a_{l,0}^2] = C_l`` and
        ``E[Re^2] = E[Im^2] = C_l/2`` for ``m>0``, hence ``S^{-1} = 1/C_l`` for
        the real m=0 part and ``2/C_l`` for the real and imaginary m>0 parts.
        """
        cl = np.asarray(cl)
        s = np.zeros(self.nalm)
        for l in range(min(len(cl), self.lmax + 1)):
            if cl[l] > 0:
                for m in range(l + 1):
                    s[hp.Alm.getidx(self.lmax, l, m)] = 1.0 / cl[l]
        s = jnp.asarray(s)
        s_re = s.at[self.mpos].multiply(2.0)
        s_im = 2.0 * s[self.mpos]
        return jnp.concatenate([s_re, s_im])


# ============================================================================
# Registry: Param / ParamSet
# ============================================================================

@dataclass
class ClPrior:
    """Gaussian prior on a linear (alm) block specified by its power spectrum."""
    cl: Any


@dataclass
class Param:
    """One model parameter block.

    :param name: Identifier used in the ``predict`` assembly dicts.
    :param kind: ``"linear"`` (solved by the inner Wiener step) or
        ``"nonlinear"`` (varied by the outer optimiser/sampler).
    :param reparam: *Linear only* — a real reparameterisation object exposing
        ``n_theta``, ``theta_to_natural``, ``natural_to_theta``, ``Sinv_from_cl``
        (e.g. :class:`RealAlmBlock`).
    :param prior: *Linear* — a :class:`ClPrior`.  *Non-linear* — an optional
        callable ``value -> log p`` (``None`` means flat).
    :param bounds: *Non-linear only* — ``(lo, hi)`` box applied per element
        (handled directly by L-BFGS-B; ``None`` means unbounded).
    :param init: *Non-linear only* — initial value (scalar broadcast or array).
    :param shape: *Non-linear only* — element shape of the block.
    :param bijector: optional; reserved for unconstrained HMC sampling.
    """
    name: str
    kind: str
    reparam: Any = None
    prior: Any = None
    bounds: Optional[Tuple[float, float]] = None
    init: Any = None
    shape: Tuple[int, ...] = ()
    bijector: Any = None


class ParamSet:
    """An ordered collection of :class:`Param` blocks.

    Knows how to (a) assemble the linear block ``S^{-1}``, (b) pack/unpack the
    linear real vector and the non-linear vector, and (c) sum the non-linear
    log-priors.  Optimisers and samplers are written once against this API.
    """

    def __init__(self, params):
        self.params = list(params)
        self.linear = [p for p in self.params if p.kind == "linear"]
        self.nonlinear = [p for p in self.params if p.kind == "nonlinear"]
        for p in self.linear:
            if p.reparam is None:
                raise ValueError(f"linear param '{p.name}' needs a reparam")
        # static layout
        self._lin_sizes = [p.reparam.n_theta for p in self.linear]
        self.n_linear = int(sum(self._lin_sizes))
        self._nl_sizes = [int(np.prod(p.shape)) for p in self.nonlinear]
        self.n_nonlinear = int(sum(self._nl_sizes))

    # -- linear block --------------------------------------------------------

    def Sinv_linear(self):
        """Concatenated ``S^{-1}`` over all linear blocks (real parameterisation)."""
        parts = []
        for p in self.linear:
            if not isinstance(p.prior, ClPrior):
                raise ValueError(f"linear param '{p.name}' needs a ClPrior")
            parts.append(p.reparam.Sinv_from_cl(p.prior.cl))
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def unpack_linear(self, theta):
        """Real vector -> ``{name: natural alm}``."""
        out, i = {}, 0
        for p, n in zip(self.linear, self._lin_sizes):
            out[p.name] = p.reparam.theta_to_natural(theta[i:i + n])
            i += n
        return out

    def pack_linear(self, natural):
        """``{name: natural alm}`` -> real vector."""
        return jnp.concatenate([
            p.reparam.natural_to_theta(natural[p.name]) for p in self.linear])

    # -- non-linear block ----------------------------------------------------

    def nonlinear_init(self):
        parts = []
        for p, n in zip(self.nonlinear, self._nl_sizes):
            v = p.init
            v = np.full(n, float(v)) if np.ndim(v) == 0 else np.asarray(v, float).ravel()
            parts.append(v)
        return np.concatenate(parts) if parts else np.zeros(0)

    def nonlinear_bounds(self):
        bounds = []
        for p, n in zip(self.nonlinear, self._nl_sizes):
            b = p.bounds if p.bounds is not None else (None, None)
            bounds.extend([(b[0], b[1])] * n)
        return bounds

    def unpack_nonlinear(self, vec):
        """Flat vector -> ``{name: value}`` (reshaped per block)."""
        out, i = {}, 0
        for p, n in zip(self.nonlinear, self._nl_sizes):
            out[p.name] = vec[i:i + n].reshape(p.shape) if p.shape else vec[i]
            i += n
        return out

    def pack_nonlinear(self, values):
        return jnp.concatenate([jnp.ravel(values[p.name]) for p in self.nonlinear])

    def logprior_nonlinear(self, values):
        """Sum of ``log p`` over non-linear blocks (flat priors contribute 0)."""
        total = 0.0
        for p in self.nonlinear:
            if p.prior is not None:
                total = total + p.prior(values[p.name])
        return total


# ============================================================================
# Linear solver
# ============================================================================

def linear_solve_cg(A, n_theta, N_inv, S_inv, data, maxiter=400, tol=1e-10,
                    x0=None):
    """Solve ``(A^T N^{-1} A + S^{-1}) theta = A^T N^{-1} d`` by preconditioned CG.

    ``A`` must be a *linear* map from a length-``n_theta`` real vector to the
    raveled data.  The normal-equation operator is applied matrix-free via
    ``jax.vjp`` (reverse-mode), so this works through ``s2fft``'s ``custom_vjp``
    transforms (which forbid forward-mode).  A diagonal ``M = diag(C_l)``
    preconditioner is used when ``S^{-1}`` is non-scalar; it equalises the
    several orders of magnitude between low- and high-``l`` prior weights and is
    what makes CG converge quickly.
    """
    N_inv = jnp.asarray(N_inv).ravel()
    d = jnp.asarray(data).ravel()

    # A is linear so A(0) = 0 and the rhs A^T N^{-1} d is the vjp at zero.
    _, vjp0 = jax.vjp(A, jnp.zeros(n_theta))
    rhs = vjp0(N_inv * d)[0]

    def matvec(theta):
        f, vjp = jax.vjp(A, theta)
        return vjp(N_inv * f)[0] + S_inv * theta

    M = None
    if jnp.ndim(S_inv) > 0:
        precond = 1.0 / jnp.maximum(jnp.asarray(S_inv), 1e-30)
        M = lambda x: x * precond

    theta_hat, _ = jax.scipy.sparse.linalg.cg(
        matvec, rhs, x0=x0, M=M, maxiter=maxiter, tol=tol)
    return theta_hat


def _dense_linear_system(A, n_theta, N_inv, S_inv, data):
    """Build dense normal equations for a linear forward map.

    Returns ``(H, rhs, cols)`` where ``cols[i] = A(e_i)``.  The column matrix is
    transposed relative to the conventional Jacobian so the weighted normal
    matrix can be formed without an extra transpose.
    """
    N_inv = jnp.asarray(N_inv).ravel()
    d = jnp.asarray(data).ravel()
    dtype = jnp.result_type(d, jnp.float64)
    eye = jnp.eye(int(n_theta), dtype=dtype)
    cols = jax.vmap(A)(eye)                         # (n_theta, ndata)
    weighted_cols = cols * N_inv[None, :]
    H = weighted_cols @ cols.T
    if jnp.ndim(S_inv) == 0:
        H = H + S_inv * jnp.eye(int(n_theta), dtype=H.dtype)
    else:
        H = H + jnp.diag(jnp.asarray(S_inv, dtype=H.dtype))
    rhs = cols @ (N_inv * d)
    return H, rhs, cols


def linear_solve_dense(A, n_theta, N_inv, S_inv, data):
    """Solve the linear Wiener step by explicitly building ``A``.

    This is intended for small linear blocks where GPU batching and dense
    Cholesky beat many matrix-free VJP/CG iterations.
    """
    H, rhs, _ = _dense_linear_system(A, n_theta, N_inv, S_inv, data)
    L = jnp.linalg.cholesky(H)
    return jax.scipy.linalg.cho_solve((L, True), rhs)


# ============================================================================
# Generic VarPro driver
# ============================================================================

def profile_optimize(predict, paramset, data, N_inv, *, maxiter=100,
                     inner_maxiter=1500, inner_tol=1e-10, max_restarts=3,
                     ftol=2.22e-9, gtol=1e-5, tol=None, verbose=True,
                     inner_method="auto", dense_threshold=512):
    """Variable-projection optimisation of a conditionally-linear model.

    :param predict: Assembly ``predict(linear: dict, nonlinear: dict) ->
        raveled prediction``.  Must be linear in the ``linear`` arguments and
        JAX-differentiable in the ``nonlinear`` ones.
    :param paramset: :class:`ParamSet` describing the parameters.
    :param data: Data array (any shape; raveled internally).
    :param N_inv: Inverse noise variance, same shape as ``data``.
    :returns: dict with ``linear`` (``{name: alm}``), ``nonlinear``
        (``{name: value}``), ``chi2``, ``nfev``, ``result``, ``paramset``.
    """
    d_flat = jnp.asarray(data).ravel()
    N_inv_flat = jnp.asarray(N_inv).ravel()
    S_inv = paramset.Sinv_linear()
    n_lin = paramset.n_linear
    if inner_method == "auto":
        solve_method = "dense" if n_lin <= int(dense_threshold) else "cg"
    else:
        solve_method = inner_method
    if solve_method not in {"cg", "dense"}:
        raise ValueError("inner_method must be 'auto', 'cg', or 'dense'")
    if verbose:
        print(f"  inner solve: {solve_method} (n_linear={n_lin})", flush=True)

    def solve_linear(nl, x0):
        A = lambda theta: predict(paramset.unpack_linear(theta), nl)
        if solve_method == "dense":
            return linear_solve_dense(A, n_lin, N_inv_flat, S_inv, d_flat)
        return linear_solve_cg(A, n_lin, N_inv_flat, S_inv, d_flat,
                               maxiter=inner_maxiter, tol=inner_tol, x0=x0)

    def loss(eta, x0):
        nl = paramset.unpack_nonlinear(eta)
        theta = jax.lax.stop_gradient(solve_linear(nl, x0))   # envelope theorem
        r = d_flat - predict(paramset.unpack_linear(theta), nl)
        chi2 = jnp.sum(N_inv_flat * r ** 2) + jnp.sum(S_inv * theta ** 2)
        return chi2 - 2.0 * paramset.logprior_nonlinear(nl), theta

    value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))

    # Warm-start the inner CG from the previous solution: consecutive inner
    # solves then share a residual pattern, which is what keeps the (envelope)
    # gradient consistent enough for the outer line search.
    x0_holder = [jnp.zeros(n_lin)]
    nfev = [0]
    history = []
    best = {"chi2": np.inf, "x": None}
    def scipy_fun(x):
        (v, theta), g = value_and_grad(jnp.asarray(x), x0_holder[0])
        x0_holder[0] = theta
        nfev[0] += 1
        vf = float(v)
        history.append(vf)
        if vf < best["chi2"]:                 # L-BFGS-B may end at a worse point
            best["chi2"] = vf
            best["x"] = np.array(x, dtype=np.float64)
        if verbose:
            print(f"  [{nfev[0]:3d}] chi2 = {vf:.6e}", flush=True)
        return vf, np.asarray(g, dtype=np.float64)

    # L-BFGS-B with restart-on-ABNORMAL: a failed line search corrupts the
    # limited-memory Hessian, so restart from the current point with a fresh one.
    x_start = paramset.nonlinear_init()
    opts = {"maxiter": maxiter, "ftol": ftol, "gtol": gtol}
    for attempt in range(max_restarts + 1):
        res = scipy.optimize.minimize(
            scipy_fun, x_start, jac=True, method="L-BFGS-B",
            bounds=paramset.nonlinear_bounds(), tol=tol, options=opts)
        msg = res.message.decode() if isinstance(res.message, bytes) else res.message
        if res.success or "ABNORMAL" not in msg or attempt == max_restarts:
            break
        if verbose:
            print(f"  -> restart {attempt+1}/{max_restarts} after line-search "
                  f"failure (chi2={float(res.fun):.4e})", flush=True)
        x_start = res.x

    gnorm = float(np.max(np.abs(np.asarray(res.jac)))) if res.jac is not None else np.nan
    # Use the best point ever seen, not L-BFGS-B's final iterate (which can be
    # worse after a line-search excursion).
    use_best = best["x"] is not None and best["chi2"] < float(res.fun)
    x_final = best["x"] if use_best else np.asarray(res.x)
    chi2_final = best["chi2"] if use_best else float(res.fun)

    # How stable is chi2 over the last few evals? (a few-sig-fig convergence check)
    tail = np.asarray(history[-min(8, len(history)):])
    tail_spread = float(tail.max() - tail.min()) / max(abs(chi2_final), 1e-30)
    converged = bool(res.success) and res.nit < maxiter and tail_spread < 1e-4
    if verbose:
        msg = res.message.decode() if isinstance(res.message, bytes) else res.message
        print(f"  -> stop: {msg}  (nit={res.nit}, |grad|inf={gnorm:.2e}, "
              f"success={res.success})", flush=True)
        print(f"  -> best chi2={chi2_final:.6e} (returned best={use_best}); "
              f"last-8 spread={tail_spread:.1e}; converged={converged}", flush=True)
        if res.nit >= maxiter:
            print("  -> WARNING: hit maxiter; increase maxiter or loosen tol.", flush=True)
        elif not converged:
            print("  -> WARNING: chi2 not stable to ~4 sig figs (erratic tail).", flush=True)

    nl_hat = paramset.unpack_nonlinear(jnp.asarray(x_final))
    lin_hat = paramset.unpack_linear(solve_linear(nl_hat, x0_holder[0]))
    return {
        "linear": lin_hat,
        "nonlinear": {k: np.asarray(v) for k, v in nl_hat.items()},
        "chi2": chi2_final,
        "chi2_history": np.asarray(history),
        "tail_spread": tail_spread,
        "nfev": nfev[0],
        "nit": int(res.nit),
        "grad_inf": gnorm,
        "converged": converged,
        "message": res.message,
        "result": res,
        "paramset": paramset,
    }


# ============================================================================
# Posterior uncertainty: Fisher / Laplace for the linear block
# ============================================================================

def linear_fisher(predict, paramset, data, N_inv, nonlinear, *,
                  method="auto", dense_threshold=4096):
    """Gaussian posterior of the linear block at fixed non-linear parameters.

    Conditional on ``nonlinear``, the linear block has an exact Gaussian
    posterior with precision (the curvature of -logL, i.e. 1/sigma^2 per mode)

        H = A^T N^{-1} A + S^{-1}

    where ``A`` is the (linear) forward at the given non-linear parameters.  ``A``
    is linear, so column ``i`` of its matrix is just ``A(e_i)`` -- no autodiff
    needed (and reverse-mode safe through ``s2fft``).  Builds ``H`` densely;
    intended for linear blocks up to a few thousand modes.

    :returns: dict with ``H`` (precision), ``cov`` = ``H^{-1}``, and ``std`` =
        per-mode posterior standard deviation ``sqrt(diag(cov))``, all in the
        block's real parameterisation.
    """
    n = paramset.n_linear
    if method == "auto":
        method = "dense" if n <= int(dense_threshold) else "loop"
    if method not in {"dense", "loop"}:
        raise ValueError("linear_fisher method must be 'auto', 'dense', or 'loop'")

    S_inv = np.broadcast_to(np.asarray(paramset.Sinv_linear()), (n,))
    if method == "dense":
        A = lambda theta: predict(paramset.unpack_linear(theta), nonlinear)

        @jax.jit
        def build_cov():
            H, _, _ = _dense_linear_system(
                A, n, jnp.asarray(N_inv).ravel(), jnp.asarray(S_inv),
                jnp.asarray(data).ravel()
            )
            L = jnp.linalg.cholesky(H)
            cov = jax.scipy.linalg.cho_solve(
                (L, True), jnp.eye(n, dtype=H.dtype)
            )
            return H, cov

        H, cov = build_cov()
        H = np.asarray(H)
        cov = np.asarray(cov)
    else:
        N_inv_flat = np.asarray(N_inv).ravel()
        A = jax.jit(lambda theta: predict(paramset.unpack_linear(theta), nonlinear))
        eye = np.eye(n)
        J = np.stack([np.asarray(A(jnp.asarray(eye[i]))) for i in range(n)], axis=1)
        H = (J.T * N_inv_flat) @ J + np.diag(S_inv)
        cov = np.linalg.inv(H)

    var = np.clip(np.diag(cov), 0, None)
    # Wiener gain per mode: w = signal/(signal+noise) = diag(I - S^-1 H^-1).
    # -> 1 for signal-dominated modes, 0 for noise-dominated ones.
    wiener = np.clip(1.0 - S_inv * var, 0.0, 1.0)
    return {"H": H, "cov": cov, "std": np.sqrt(var), "wiener": wiener}


def sample_posterior(predict, paramset, data, N_inv, *, num_samples=500,
                     num_warmup=500, seed=0, init_linear=None,
                     init_nonlinear=None, engine="nuts",
                     hmc_num_integration_steps=10,
                     initial_step_size=1.0,
                     target_acceptance_rate=0.8):
    """NUTS over the *joint* (linear, non-linear) posterior.

    log p(lam, eta | d) = -1/2 [ ||d - predict||^2_{N^-1} + lam^T S^-1 lam ]
                          + logprior(eta)

    The linear block is sampled rather than marginalised (so we get per-mode
    flux samples directly); the conditional-linear structure still helps because
    the linear directions are Gaussian and well-conditioned.  Bounded non-linear
    params are sampled in an unconstrained sigmoid space (with the log-Jacobian).

    :param engine: BlackJAX HMC-like engine. Supported values are ``"nuts"``
        (default) and ``"hmc"``.
    :param hmc_num_integration_steps: Static trajectory length for
        ``engine="hmc"``.
    :returns: dict with ``linear`` (``{name: (num_samples, ...) alm samples}``),
        ``nonlinear`` (``{name: (num_samples, ...) value samples}``), and
        ``accept`` (mean acceptance).  Posterior mean = estimate; std = per-mode
        1-sigma, i.e. the direct SNR.
    """
    import blackjax

    d_flat = jnp.asarray(data).ravel()
    N_inv_flat = jnp.asarray(N_inv).ravel()
    S_inv = jnp.asarray(paramset.Sinv_linear())
    n_lin = paramset.n_linear

    # Box transforms for the non-linear block (sigmoid where bounded).
    lo, hi = [], []
    for p, sz in zip(paramset.nonlinear, paramset._nl_sizes):
        b = p.bounds if p.bounds is not None else (-np.inf, np.inf)
        lo += [b[0]] * sz
        hi += [b[1]] * sz
    lo = jnp.asarray(lo, float)
    hi = jnp.asarray(hi, float)
    bounded = jnp.isfinite(lo) & jnp.isfinite(hi)
    span = jnp.where(bounded, hi - lo, 1.0)

    def eta_from_z(zeta):
        s = jax.nn.sigmoid(zeta)
        eta = jnp.where(bounded, lo + span * s, zeta)
        logj = jnp.where(bounded, jnp.log(span) + jnp.log(s) + jnp.log1p(-s), 0.0)
        return eta, jnp.sum(logj)

    def z_from_eta(eta):
        s = jnp.clip((eta - lo) / span, 1e-6, 1 - 1e-6)
        return jnp.where(bounded, jnp.log(s) - jnp.log1p(-s), eta)

    def logdensity(z):
        lam, zeta = z[:n_lin], z[n_lin:]
        eta_vec, logj = eta_from_z(zeta)
        nl = paramset.unpack_nonlinear(eta_vec)
        r = d_flat - predict(paramset.unpack_linear(lam), nl)
        chi2 = jnp.sum(N_inv_flat * r ** 2) + jnp.sum(S_inv * lam ** 2)
        return -0.5 * chi2 + paramset.logprior_nonlinear(nl) + logj

    lam0 = (jnp.zeros(n_lin) if init_linear is None
            else jnp.asarray(init_linear))
    eta0 = (jnp.asarray(paramset.nonlinear_init()) if init_nonlinear is None
            else jnp.asarray(init_nonlinear))
    z0 = jnp.concatenate([lam0, z_from_eta(eta0)])

    engine = engine.lower()
    if engine == "nuts":
        algorithm = blackjax.nuts
        warmup_kwargs = {}
    elif engine == "hmc":
        algorithm = blackjax.hmc
        warmup_kwargs = {"num_integration_steps": int(hmc_num_integration_steps)}
    else:
        raise ValueError("engine must be 'nuts' or 'hmc'")

    rng = jax.random.PRNGKey(seed)
    rng, wk = jax.random.split(rng)
    warmup = blackjax.window_adaptation(
        algorithm,
        logdensity,
        initial_step_size=float(initial_step_size),
        target_acceptance_rate=float(target_acceptance_rate),
        **warmup_kwargs,
    )
    (state, params), _ = warmup.run(wk, z0, num_steps=num_warmup)
    kernel = algorithm(logdensity, **params)

    @jax.jit
    def one_step(state, k):
        state, info = kernel.step(k, state)
        accept = getattr(info, "acceptance_rate", getattr(info, "is_accepted", jnp.nan))
        return state, (state.position, accept)

    rng, sk = jax.random.split(rng)
    _, (positions, accept) = jax.lax.scan(
        one_step, state, jax.random.split(sk, num_samples))

    lam_s = positions[:, :n_lin]
    eta_s = jax.vmap(lambda z: eta_from_z(z)[0])(positions[:, n_lin:])
    # unpack to named samples
    lin_out, i = {}, 0
    for p, sz in zip(paramset.linear, paramset._lin_sizes):
        lin_out[p.name] = jax.vmap(p.reparam.theta_to_natural)(
            lam_s[:, i:i + sz])
        i += sz
    nl_out, j = {}, 0
    for p, sz in zip(paramset.nonlinear, paramset._nl_sizes):
        block = eta_s[:, j:j + sz]
        nl_out[p.name] = block.reshape((num_samples,) + p.shape) if p.shape \
            else block[:, 0]
        j += sz
    return {"linear": lin_out, "nonlinear": nl_out,
            "accept": float(jnp.mean(accept))}


def snr_weighted_recovery(theta_hat, theta_true, fisher):
    """Wiener-weighted comparison of a recovered linear block against truth.

    Weights each mode by its Wiener gain ``w = signal/(signal+noise)`` (1 for
    signal-dominated modes, 0 for noise-dominated ones), so the comparison only
    counts the modes the data could actually measure -- without the high-SNR
    modes blowing up the metric (as an unbounded 1/sigma^2 weight would).

    :returns: dict with
        * ``rho_w`` -- Wiener-weighted correlation of recovered vs truth
          (1 = recoverable modes perfectly recovered),
        * ``resid_frac`` -- Wiener-weighted residual power / signal power
          (0 = perfect),
        * ``n_eff`` -- effective number of measured modes (sum of w),
        * ``n`` -- total modes, ``w`` -- per-mode weights,
        * ``whitened`` -- per-mode residual in sigma units (diagnostic).
    """
    h = np.asarray(theta_hat, dtype=float)
    t = np.asarray(theta_true, dtype=float)
    w = np.asarray(fisher["wiener"], dtype=float)
    rho_w = float(np.sum(w * h * t) /
                  np.sqrt(np.sum(w * h * h) * np.sum(w * t * t) + 1e-300))
    resid_frac = float(np.sum(w * (h - t) ** 2) / (np.sum(w * t * t) + 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        whitened = (h - t) / fisher["std"]
    return {"rho_w": rho_w, "resid_frac": resid_frac, "n_eff": float(w.sum()),
            "n": len(w), "w": w, "whitened": whitened}


# ============================================================================
# Layer 2: module ports
# ============================================================================
#
# A *module* is one physical component of the model (sky, beam, instrument).
# It declares its free parameters via ``params()`` (names are namespaced, e.g.
# ``"sky.flux"``) and implements a *port* — a method the assembly chains.  Ports
# receive the full merged parameter dict and pick out their own keys.  Modules
# with no free parameters still implement their port (with ``params()`` empty),
# so the assembly is uniform whether or not a component is being fit.


@dataclass
class GaussianLogPrior:
    """``log p(x) = -0.5 ((x-mu)/sigma)^2`` (constant dropped).  JAX-traceable."""
    mu: float
    sigma: float

    def __call__(self, x):
        return -0.5 * jnp.sum(((jnp.asarray(x) - self.mu) / self.sigma) ** 2)


@dataclass
class GraphSmoothnessPrior:
    """Gaussian Markov random field (GMRF) prior on a pixel-space block.

    A spatially-varying smoothness + anchor prior for a real per-pixel field
    (e.g. a spectral-index map).  With an undirected neighbour graph it is

        -2 log p(x) = sum_<ij> w_ij (x_i - x_j)^2  +  sum_i a_i (x_i - mu)^2

    the first term (smoothness) couples neighbours -- large ``w`` forces them to
    agree (locally coarse), small ``w`` lets the field vary pixel-to-pixel
    (locally fine).  The second (anchor/ridge) pulls the field toward ``mu``;
    making ``a_i`` large where the data are uninformative (low instrument
    coverage) pins the field there and removes degeneracy fuel, while ``a_i``
    small elsewhere lets the data drive it.

    This is the discrete Matern/SPDE field of Lindgren-Rue-Lindqvist (2011),
    precision ``Q = diag(a) + W L_graph``; its smoothing length is
    ``xi_i = sqrt(w_i / a_i)`` pixels (effective resolution
    ``n_eff = Nside / xi``).  The same ``Q`` can feed an HMC log-posterior.

    Edges are stored as parallel index arrays ``(i, j)`` with per-edge weights
    ``we``; the model-specific builder
    (:func:`lusee.SpectralSky.build_beta_smoothness_prior`) assembles them from a
    healpix neighbour graph and a coverage map.  All arrays are captured as jit
    constants, so the prior is JAX-traceable as a plain callable.
    """
    i: Any           # (n_edges,) edge endpoint indices
    j: Any           # (n_edges,) other endpoint
    we: Any          # (n_edges,) symmetric edge weights w_ij
    a: Any           # (n_pix,) per-pixel anchor weights
    mu: float        # anchor target

    def __call__(self, x):
        x = jnp.asarray(x)
        smooth = jnp.sum(jnp.asarray(self.we) * (x[self.i] - x[self.j]) ** 2)
        ridge = jnp.sum(jnp.asarray(self.a) * (x - self.mu) ** 2)
        return -0.5 * (smooth + ridge)


class Module:
    """Base class for a model component (Layer 2)."""
    name = "module"

    def params(self):
        """Return this module's :class:`Param` list (default: no free params)."""
        return []


class BeamModule(Module):
    """Beam port: ``beam(params) -> beam pytree or None``.

    The baseline beam has no free parameters, so the port returns ``None`` and
    the simulator uses its precomputed ``efbeams``.  A parameterised beam (e.g.
    regolith / impedance) would subclass this and return a ``CachedBeam`` pytree
    built from the parameters.
    """
    name = "beam"

    def beam(self, p):
        return None


class InstrumentModule(Module):
    """Instrument port: ``apply(params, vis) -> vis`` (data-space map).

    Optionally fits a scalar broadband ``inst.gain`` that multiplies the
    visibilities.  Gain is *bilinear* with the flux (data ~ gain * flux), so it
    belongs in the non-linear block while flux stays linear — the framework
    resolves the product automatically by keeping the two factors in different
    blocks.  Because a broadband gain is degenerate with the overall flux
    amplitude, it carries an informative Gaussian prior to be identifiable.
    """
    name = "inst"

    def __init__(self, *, gain=False, gain_prior=(1.0, 0.02),
                 gain_bounds=(0.5, 1.5)):
        self.gain = gain
        self.gain_prior = gain_prior
        self.gain_bounds = gain_bounds

    def params(self):
        if not self.gain:
            return []
        mu, sigma = self.gain_prior
        return [Param("inst.gain", "nonlinear", shape=(), init=mu,
                      prior=GaussianLogPrior(mu, sigma), bounds=self.gain_bounds)]

    def apply(self, p, vis):
        if not self.gain:
            return vis
        return vis * p["inst.gain"]


# ============================================================================
# Layer 3: explicit assembly
# ============================================================================

class Experiment:
    """Wire modules + simulator into one conditionally-linear forward model.

    Collects every module's parameters into a single :class:`ParamSet`, and
    builds ``predict(linear, nonlinear)`` by chaining the ports
    ``sky -> simulate(beam) -> instrument``.  The assembly is explicit Python
    (~5 lines) by design — the registry generalises *parameters and solving*,
    but the sky->beam->instrument composition is real physics wiring, not a
    generic graph.  The shared driver :func:`profile_optimize` runs unchanged.

    :param sim: ``CroSimulator``.
    :param sky: a sky module exposing ``sky(p)`` (e.g. :class:`SpectralSkyModule`).
    :param beam: a beam module exposing ``beam(p)`` (e.g. :class:`BeamModule`).
    :param instrument: an instrument module exposing ``apply(p, vis)``.
    :param data: data array.
    :param N_inv: inverse noise variance, same shape as ``data``.
    """

    def __init__(self, sim, *, sky, beam, instrument, data, N_inv):
        self.sim = sim
        self.sky_module = sky
        self.beam_module = beam
        self.inst_module = instrument
        self.data = data
        self.N_inv = N_inv
        self.paramset = ParamSet(
            sky.params() + beam.params() + instrument.params())

    def predict(self, linear, nonlinear):
        p = {**linear, **nonlinear}
        sky = self.sky_module.sky(p)
        beam = self.beam_module.beam(p)
        vis = self.sim.simulate(sky=sky, beam=beam).ravel()
        return self.inst_module.apply(p, vis)

    def optimize(self, **kw):
        """Run the shared VarPro driver over this experiment's parameters."""
        return profile_optimize(self.predict, self.paramset,
                                self.data, self.N_inv, **kw)

    def linear_solve(self, *, method="cg", inner_maxiter=2000, inner_tol=1e-10):
        """Single Wiener/CG solve of the linear block (no non-linear params).

        Used when the model has been made purely linear -- e.g. a fixed-``beta``
        sky -- so the forward ``m = A @ flux`` has no outer loop.  Returns the
        linear block as ``{name: natural alm}`` (like ``optimize()['linear']``).
        """
        ps = self.paramset
        if ps.nonlinear:
            raise ValueError("linear_solve requires a model with no free "
                             "non-linear parameters (got "
                             f"{[p.name for p in ps.nonlinear]}).")
        d = jnp.asarray(self.data).ravel()
        N_inv = jnp.asarray(self.N_inv).ravel()
        S_inv = ps.Sinv_linear()
        n = ps.n_linear
        A = lambda theta: self.predict(ps.unpack_linear(theta), {})
        if method == "dense":
            theta = linear_solve_dense(A, n, N_inv, S_inv, d)
        elif method == "cg":
            theta = linear_solve_cg(A, n, N_inv, S_inv, d,
                                    maxiter=inner_maxiter, tol=inner_tol)
        else:
            raise ValueError("method must be 'cg' or 'dense'")
        return ps.unpack_linear(theta)
