"""Space-frequency separable sky model: a sum of spatial templates each with
its own free spectral shape.

    T(theta, f) = sum_i  flux_i(theta) * shape_i(f)

This is the bilinear example. For fixed spectral shapes the data depend
*linearly* on the template maps `flux_i`; for fixed maps they depend linearly on
the shapes. We keep the maps (`flux_i`, many pixels) as the **linear** block and
the shapes (`shape_i`, one value per fitted frequency — far fewer numbers) as the
**non-linear** block, exactly as the conditional-linearity split prescribes.

Unlike the power-law `SpectralHealpixSky`, this model needs **no spherical
harmonic transform** in the forward: the templates already live in `a_lm`, and a
per-frequency sky is just a scalar-weighted sum of them,

    a_lm(f) = sum_i  shape_i(f) * flux_i_alm,

so `get_alm` is a single `einsum` — cheap and exactly linear in the `flux_i_alm`.

Gauge. The decomposition is invariant under `flux_i -> flux_i / c_i`,
`shape_i -> c_i shape_i` (per-template scaling) and, more generally, under an
invertible mixing of the templates. The scaling is fixed by *anchoring* each
shape at a reference frequency, `shape_i(f_ref) = 1`, handled by
`SeparableSkyModule` (so `flux_i` is literally the i-th template map at
`f_ref`). The residual mixing freedom is broken at the MAP by the per-template
`C_l` priors; gauge-invariant quantities (the total flux at any frequency) are
what one should compare.
"""

import numpy as np
import jax
import jax.numpy as jnp

from .frequencies import canonicalize_frequencies


@jax.tree_util.register_pytree_node_class
class SeparableHealpixSky:
    """Sky ``T(theta,f) = sum_i flux_i(theta) * shape_i(f)``.

    :param flux_alms: Template maps in healpy-packed alm, shape
        ``(n_templates, nalm)`` (complex) — the linear parameters.
    :param shapes: Spectral shapes, shape ``(n_templates, nfreq)`` (real) — the
        non-linear parameters; ``shapes[i, k]`` is template ``i`` at ``freq[k]``.
    :param lmax: Maximum multipole.
    :param freq: Frequencies in MHz (defines the index order for the simulator).
    :param frame: Sky frame; ``"galactic"`` for ``CroSimulator``.
    """

    def __init__(self, flux_alms, shapes, *, lmax, freq, frame="galactic"):
        self.flux_alms = jnp.asarray(flux_alms)
        self.shapes = jnp.asarray(shapes)
        self.lmax = int(lmax)
        self.freq = canonicalize_frequencies(freq, as_jax=True)
        self.frame = frame

    def tree_flatten(self):
        children = (self.flux_alms, self.shapes)
        aux = (self.lmax, tuple(np.asarray(self.freq).tolist()), self.frame)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        lmax, freq, frame = aux
        flux_alms, shapes = children
        sky = cls.__new__(cls)
        sky.flux_alms = flux_alms
        sky.shapes = shapes
        sky.lmax = lmax
        sky.freq = jnp.asarray(freq)
        sky.frame = frame
        return sky

    def get_alm(self, ndx, freq=None):
        """Healpy-packed alm at the requested frequency indices.

        ``a_lm(f) = sum_i shape_i(f) * flux_i_alm`` — a scalar-weighted sum of
        the template alm, with no spherical-harmonic transform.
        """
        ndx = np.atleast_1d(np.asarray(ndx)).astype(int)
        sh = self.shapes[:, jnp.asarray(ndx)]              # (n_templ, K)
        return jnp.einsum("ik,ia->ka", sh, self.flux_alms)  # (K, nalm)

    def flux_map_at(self, freq_value, Nside):
        """Total flux map at one frequency (host-side helper for plotting)."""
        import healpy as hp
        fr = np.asarray(self.freq)
        k = int(np.argmin(np.abs(fr - freq_value)))
        alm = np.einsum("i,ia->a", np.asarray(self.shapes)[:, k],
                        np.asarray(self.flux_alms))
        return hp.alm2map(alm.astype(complex), Nside)


# The fitting adapter for this model lives next to the model (it depends on the
# generic registry in lusee.Fitting, not the other way around).
from .Fitting import Param, RealAlmBlock, ClPrior, Module  # noqa: E402


class SeparableSkyModule(Module):
    """Sky port for the separable model ``T = sum_i flux_i(theta) shape_i(f)``.

    Declares ``n_templates`` *linear* flux blocks (``sep.flux.0``, ``sep.flux.1``,
    …), each a `RealAlmBlock` with its own ``C_l`` prior, and one *non-linear*
    ``sep.shape`` block holding the spectral shapes.  The shapes are anchored at
    ``ref_freq`` (``shape_i(f_ref) = 1``) to fix the per-template scaling gauge:
    the non-linear parameter is the shape values at the *non-reference*
    frequencies, and the port reinserts 1 at the reference column.
    """
    name = "sep"

    def __init__(self, *, lmax, freq, ref_freq, n_templates, cl_flux,
                 shape_init, shape_bounds=None, frame="galactic"):
        self.lmax = lmax
        self.freq = np.asarray(freq, dtype=float)
        self.n_templ = int(n_templates)
        self.cl_flux = list(cl_flux)            # one C_l per template
        self.frame = frame
        self.ref_idx = int(np.argmin(np.abs(self.freq - ref_freq)))
        self.nonref = [k for k in range(len(self.freq)) if k != self.ref_idx]
        self.nfree = len(self.nonref)
        self.shape_init_free = np.asarray(shape_init, dtype=float)[:, self.nonref]
        self.shape_bounds = shape_bounds

    def params(self):
        ps = [Param(f"sep.flux.{i}", "linear", reparam=RealAlmBlock(self.lmax),
                    prior=ClPrior(self.cl_flux[i])) for i in range(self.n_templ)]
        ps.append(Param("sep.shape", "nonlinear",
                        shape=(self.n_templ, self.nfree),
                        init=self.shape_init_free, bounds=self.shape_bounds))
        return ps

    def _full_shapes(self, free):
        full = jnp.ones((self.n_templ, len(self.freq)))
        return full.at[:, jnp.asarray(self.nonref)].set(free)

    def sky(self, p):
        flux_alms = jnp.stack([p[f"sep.flux.{i}"] for i in range(self.n_templ)])
        shapes = self._full_shapes(p["sep.shape"])
        return SeparableHealpixSky(flux_alms, shapes, lmax=self.lmax,
                                   freq=self.freq, frame=self.frame)
