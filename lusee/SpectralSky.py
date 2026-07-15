"""Two-component spectral sky model: a flux map and a spectral-index map.

The sky temperature is

    T(theta, f) = flux(theta) * (f / f_fid) ** beta(theta)
                = flux(theta) * exp(beta(theta) * ln(f / f_fid))

This is the simplest non-trivial model that exercises both kinds of free
parameter the fitting framework cares about:

* ``flux`` is **linear** — for fixed ``beta`` the data depend linearly on the
  flux map, so it is solved optimally by the (Wiener) linear solver.  It lives
  in spherical-harmonic space (``flux_alm``), which is the natural home for a
  ``C_l`` power-spectrum prior.
* ``beta`` is **non-linear** — it enters through an exponential, so it is fit
  by the outer (gradient-based) optimizer/sampler.  It lives in real space
  (``beta_pix``), which is the natural home for a per-pixel box prior and the
  per-pixel power law.

Why real-space multiplication (and not a harmonic convolution)?
``exp(beta * ln f)`` has no closed harmonic form, so ``beta`` must be evaluated
in pixel space regardless.  Once we are in pixel space the product
``flux * exp(...)`` is a cheap elementwise multiply, costing ``2 + Nfreq``
spherical-harmonic transforms in total.  Doing the same product as a harmonic
convolution would cost an O(lmax^5) Gaunt contraction *and* still need the
transform for the exponential — strictly more expensive.

All transforms use ``s2fft`` (JAX backend) so the whole ``get_alm`` is
differentiable: ``jax.grad`` flows to ``flux_alm`` (linear) and ``beta_pix``
(non-linear).  The model is a registered pytree whose children are exactly the
two free-parameter arrays, so it drops into ``CroSimulator.simulate(sky=...)``.

Note on s2fft healpix sampling: it requires ``L = lmax + 1 >= 2 * Nside``.
The inverse->forward round trip on the healpix grid is not exact (healpix has
no sampling theorem), but that approximation lives *identically* inside the
forward operator used to both generate and fit the data, so it introduces no
bias in the recovery.
"""

from functools import lru_cache

import numpy as np
import healpy as hp
import jax
import jax.numpy as jnp
import s2fft
from s2fft.sampling.reindex import flm_hp_to_2d_fast, flm_2d_to_hp_fast

from .frequencies import canonicalize_frequencies


@lru_cache(maxsize=None)
def _udgrade_parent_ring(beta_nside, Nside):
    """RING-ordered parent index mapping working pixels -> coarse beta pixels.

    Returns an int array ``parent`` of length ``12*Nside**2`` such that
    ``beta_pix[parent]`` upgrades a coarse (``beta_nside``) map to the working
    resolution by pixel replication (healpy ``ud_grade`` semantics).  Returns
    ``None`` when no upsampling is needed.  Depends only on the two (static)
    resolutions, so it is safe to call inside a traced ``get_alm``.
    """
    if beta_nside == Nside:
        return None
    if Nside % beta_nside != 0:
        raise ValueError(
            f"Nside ({Nside}) must be an integer multiple of beta_nside "
            f"({beta_nside}) for pixel-replication upsampling."
        )
    npix_hi = 12 * Nside * Nside
    nest_hi = hp.ring2nest(Nside, np.arange(npix_hi))
    ratio = (Nside // beta_nside) ** 2
    nest_lo = nest_hi // ratio
    return hp.nest2ring(beta_nside, nest_lo).astype(np.int64)


@jax.tree_util.register_pytree_node_class
class SpectralHealpixSky:
    """Sky model ``T(theta,f) = flux(theta) * (f/f_fid)**beta(theta)``.

    :param flux_alm: Healpy-packed complex alm of the flux map at ``f_fid``
        (the linear parameter), shape ``(nalm,)`` with ``nalm = (lmax+1)(lmax+2)/2``.
    :param beta_pix: Real spectral-index map (the non-linear parameter),
        shape ``(12*beta_nside**2,)`` in RING ordering.
    :param Nside: Working healpix resolution for the real-space product.
        Must satisfy ``lmax + 1 >= 2*Nside`` (s2fft healpix requirement).
    :param lmax: Maximum multipole.
    :param freq: Frequencies in MHz (defines the index order used by the simulator).
    :param f_fid: Fiducial frequency in MHz at which ``flux`` is defined.
    :param beta_nside: Resolution of ``beta_pix``; defaults to ``Nside``.  Lower
        values keep the non-linear problem small and identifiable.
    :param frame: Sky frame; must be ``"galactic"`` for ``CroSimulator``.
    """

    def __init__(self, flux_alm, beta_pix, *, Nside, lmax, freq, f_fid,
                 beta_nside=None, frame="galactic"):
        self.flux_alm = jnp.asarray(flux_alm)
        self.beta_pix = jnp.asarray(beta_pix)
        self.Nside = int(Nside)
        self.lmax = int(lmax)
        self.freq = canonicalize_frequencies(freq, as_jax=True)
        self.f_fid = float(f_fid)
        self.beta_nside = int(beta_nside) if beta_nside is not None else int(Nside)
        self.frame = frame
        if self.lmax + 1 < 2 * self.Nside:
            raise ValueError(
                f"s2fft healpix sampling requires lmax+1 >= 2*Nside; "
                f"got lmax={self.lmax}, Nside={self.Nside}."
            )

    # -- pytree protocol: children are the two free-parameter arrays ----------

    def tree_flatten(self):
        children = (self.flux_alm, self.beta_pix)
        aux = (
            self.Nside,
            self.lmax,
            tuple(np.asarray(self.freq).tolist()),
            self.f_fid,
            self.beta_nside,
            self.frame,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        Nside, lmax, freq, f_fid, beta_nside, frame = aux
        flux_alm, beta_pix = children
        sky = cls.__new__(cls)
        sky.flux_alm = flux_alm
        sky.beta_pix = beta_pix
        sky.Nside = Nside
        sky.lmax = lmax
        sky.freq = jnp.asarray(freq)
        sky.f_fid = f_fid
        sky.beta_nside = beta_nside
        sky.frame = frame
        return sky

    # -- model ----------------------------------------------------------------

    def beta_map(self):
        """Spectral-index map at the working resolution (RING ordering)."""
        parent = _udgrade_parent_ring(self.beta_nside, self.Nside)
        if parent is None:
            return self.beta_pix
        return self.beta_pix[jnp.asarray(parent)]

    def flux_map(self):
        """Flux map at the working resolution (RING ordering)."""
        L = self.lmax + 1
        flm2d = flm_hp_to_2d_fast(jnp.asarray(self.flux_alm), L)
        return s2fft.inverse(flm2d, L, nside=self.Nside, sampling="healpix",
                             method="jax", reality=True)

    def get_alm(self, ndx, freq=None):
        """Healpy-packed alm of the sky at the requested frequency indices.

        :param ndx: Frequency index or list of indices into ``self.freq``.
        :returns: Array ``(len(ndx), nalm)`` of complex alm, one per frequency.
        """
        L = self.lmax + 1
        Nside = self.Nside
        flux_map = self.flux_map()
        beta_map = self.beta_map()

        ndx = np.atleast_1d(np.asarray(ndx)).astype(int)
        freqs = jnp.asarray(self.freq)[jnp.asarray(ndx)]
        ln_ratio = jnp.log(freqs / self.f_fid)

        def one(lf):
            prod = flux_map * jnp.exp(beta_map * lf)
            flm = s2fft.forward(prod, L, nside=Nside, sampling="healpix",
                                method="jax", reality=True)
            return flm_2d_to_hp_fast(flm, L)

        return jax.vmap(one)(ln_ratio)

    # -- construction helpers -------------------------------------------------

    @classmethod
    def flux_alm_from_map(cls, flux_map, lmax, Nside):
        """Project a flux map to ``flux_alm`` using the *same* s2fft healpix
        transform as the forward model (so a truth built this way round-trips
        consistently through ``get_alm``).
        """
        L = lmax + 1
        flm = s2fft.forward(jnp.asarray(flux_map), L, nside=Nside,
                            sampling="healpix", method="jax", reality=True)
        return flm_2d_to_hp_fast(flm, L)


# The fitting adapter for this model lives next to the model (it depends on the
# generic registry in lusee.Fitting, not the other way around).
from .Fitting import (Param, RealAlmBlock, ClPrior, Module,  # noqa: E402
                      GraphSmoothnessPrior)


def coverage_map(sim, *, lmax, Nside, freq, f_fid=25.0, beta=-2.5,
                 n_probe=32, seed=0):
    """Per-direction instrument sensitivity ``diag(A^T A)`` for the flux block.

    ``A`` is the (linear, at fixed ``beta``) forward map from a flux *pixel* map
    to the raveled data.  Its per-pixel sensitivity ``diag(A^T A)_p = sum_d
    A_{dp}^2`` is the coverage: large where the instrument responds to flux in
    direction ``p`` over the observation, ~0 in never-observed directions.

    Estimated by Hutchinson on the *adjoint* only -- ``E_v[(A^T v)_p^2] =
    (A^T A)_{pp}`` for ``v ~ N(0, I)`` -- so it uses reverse-mode ``jax.vjp``
    exclusively (forward-mode is forbidden through ``s2fft``'s ``custom_vjp``).
    ``A`` is linear, so the VJP closure is built once and reused for every probe.

    :returns: RING-ordered float coverage map of length ``12*Nside**2``.
    """
    npix = 12 * Nside ** 2
    beta_const = jnp.full(npix, float(beta))

    def Apix(flux_map):
        flux_alm = SpectralHealpixSky.flux_alm_from_map(flux_map, lmax, Nside)
        sky = SpectralHealpixSky(flux_alm, beta_const, Nside=Nside, lmax=lmax,
                                 freq=freq, f_fid=f_fid, beta_nside=Nside)
        return sim.simulate(sky=sky).ravel()

    x0 = jnp.zeros(npix)
    y0, vjp = jax.vjp(Apix, x0)            # A is linear -> vjp independent of x0
    ndata = int(y0.shape[0])
    key = jax.random.PRNGKey(seed)
    acc = jnp.zeros(npix)
    for s in range(int(n_probe)):
        v = jax.random.normal(jax.random.fold_in(key, s), (ndata,))
        acc = acc + vjp(v)[0] ** 2
    return np.asarray(acc / float(n_probe))


def build_beta_smoothness_prior(beta_nside, *, n_center=16, n_far=2,
                                d0_deg=20.0, center_lonlat=(0.0, 0.0),
                                mu=-2.5, smooth=1.0, anchor=1e-4):
    """Purely angular GMRF smoothness prior on the ``beta`` map.

    Builds a :class:`~lusee.Fitting.GraphSmoothnessPrior` on the ``beta_nside``
    healpix grid whose *effective resolution* ramps from ``n_center`` near the
    galactic centre to ``n_far`` far from it (Gaussian in angular distance, scale
    ``d0_deg``).  It is **not** coverage-aware: it knows nothing about which
    directions the instrument sees.  Where the data are uninformative (the
    never-observed patch), ``beta`` is simply tied to its neighbours by the
    smoothness term and floats with them -- a graceful "we don't know here, so
    keep it smooth" behaviour rather than any hole-specific pinning.

    The smoothing weight is ``w_i = smooth * xi_i^2`` with
    ``xi_i = beta_nside / n_eff(theta_i)`` the local smoothing length in pixels;
    a weak *uniform* ``anchor`` ridge toward ``mu`` keeps the field proper
    (fixes the otherwise-free DC) without targeting any region.  Both ``smooth``
    and ``anchor`` are absolute ``-2logL`` weights and should be tuned so the
    smoothness is sub-dominant to the per-pixel beta likelihood curvature where
    the data are informative (too strong freezes ``beta`` and forces the flux to
    absorb the spectral mismatch).
    """
    npix = 12 * beta_nside ** 2

    # target effective resolution vs angular distance from the galactic centre
    vec = np.asarray(hp.pix2vec(beta_nside, np.arange(npix)))         # (3,npix)
    cen = hp.ang2vec(np.radians(90.0 - center_lonlat[1]),
                     np.radians(center_lonlat[0]))
    d = np.degrees(np.arccos(np.clip(cen @ vec, -1.0, 1.0)))
    g = np.exp(-(d / d0_deg) ** 2)                                    # 1->0
    n_eff = n_far + (n_center - n_far) * g
    xi = beta_nside / np.maximum(n_eff, 1e-6)                         # pixels
    w = float(smooth) * xi ** 2                                       # smoothing
    a = np.full(npix, float(anchor))                                 # weak ridge

    nb = hp.get_all_neighbours(beta_nside, np.arange(npix))          # (8,npix)
    i = np.repeat(np.arange(npix), nb.shape[0])
    j = nb.T.ravel()
    keep = j >= 0
    i, j = i[keep], j[keep]
    we = 0.5 * (w[i] + w[j])
    return GraphSmoothnessPrior(jnp.asarray(i), jnp.asarray(j),
                                jnp.asarray(we), jnp.asarray(a), float(mu))


class SpectralSkyModule(Module):
    """Sky port: ``sky(params) -> SpectralHealpixSky``.

    Declares ``sky.flux`` (linear, ``C_l`` prior) and ``sky.beta`` (non-linear).
    ``beta`` carries a flat box prior by default, or a supplied ``beta_prior``
    callable (e.g. a :class:`~lusee.Fitting.GraphSmoothnessPrior` from
    :func:`build_beta_smoothness_prior`) for spatially-varying smoothness.
    """
    name = "sky"

    def __init__(self, *, lmax, Nside, freq, f_fid, beta_nside, cl_flux,
                 beta_bounds=(-4.0, -1.5), beta_init=-2.5, frame="galactic",
                 beta_prior=None, beta_fixed=None):
        self.lmax = lmax
        self.Nside = Nside
        self.freq = freq
        self.f_fid = f_fid
        self.beta_nside = beta_nside
        self.cl_flux = cl_flux
        self.beta_bounds = beta_bounds
        self.beta_init = beta_init
        self.frame = frame
        self.beta_prior = beta_prior
        # When ``beta_fixed`` is given, beta is held at this map (length
        # 12*beta_nside**2) and dropped from the free parameters -- the model is
        # then purely linear in flux (a single Wiener/CG solve, no outer loop).
        self.beta_fixed = (None if beta_fixed is None
                           else jnp.asarray(beta_fixed))
        self.n_beta = 12 * beta_nside ** 2

    def params(self):
        flux = Param("sky.flux", "linear", reparam=RealAlmBlock(self.lmax),
                     prior=ClPrior(self.cl_flux))
        if self.beta_fixed is not None:
            return [flux]
        return [
            flux,
            Param("sky.beta", "nonlinear", shape=(self.n_beta,),
                  init=self.beta_init, bounds=self.beta_bounds,
                  prior=self.beta_prior),
        ]

    def sky(self, p):
        beta = self.beta_fixed if self.beta_fixed is not None else p["sky.beta"]
        return SpectralHealpixSky(
            p["sky.flux"], beta, Nside=self.Nside, lmax=self.lmax,
            freq=self.freq, f_fid=self.f_fid, beta_nside=self.beta_nside,
            frame=self.frame)
