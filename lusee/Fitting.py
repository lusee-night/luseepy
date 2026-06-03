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


# ============================================================================
# Generic VarPro driver
# ============================================================================

def profile_optimize(predict, paramset, data, N_inv, *, maxiter=100,
                     inner_maxiter=400, inner_tol=1e-10, max_restarts=3,
                     ftol=2.22e-9, gtol=1e-5, tol=None, verbose=True):
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

    def solve_linear(nl, x0):
        A = lambda theta: predict(paramset.unpack_linear(theta), nl)
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
    def scipy_fun(x):
        (v, theta), g = value_and_grad(jnp.asarray(x), x0_holder[0])
        x0_holder[0] = theta
        nfev[0] += 1
        history.append(float(v))
        if verbose:
            print(f"  [{nfev[0]:3d}] chi2 = {float(v):.6e}", flush=True)
        return float(v), np.asarray(g, dtype=np.float64)

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
    converged = bool(res.success) and res.nit < maxiter
    if verbose:
        msg = res.message.decode() if isinstance(res.message, bytes) else res.message
        print(f"  -> stop: {msg}  (nit={res.nit}, |grad|inf={gnorm:.2e}, "
              f"success={res.success})", flush=True)
        if res.nit >= maxiter:
            print("  -> WARNING: hit maxiter; increase maxiter or loosen tol.", flush=True)

    nl_hat = paramset.unpack_nonlinear(jnp.asarray(res.x))
    lin_hat = paramset.unpack_linear(solve_linear(nl_hat, x0_holder[0]))
    return {
        "linear": lin_hat,
        "nonlinear": {k: np.asarray(v) for k, v in nl_hat.items()},
        "chi2": float(res.fun),
        "chi2_history": np.asarray(history),
        "nfev": nfev[0],
        "nit": int(res.nit),
        "grad_inf": gnorm,
        "converged": converged,
        "message": res.message,
        "result": res,
        "paramset": paramset,
    }


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
