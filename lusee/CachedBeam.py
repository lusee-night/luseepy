"""Base class for cached beam pytrees.

Subclasses define free parameters and a ``transform_beam`` method.
The base handles efbeams storage, pytree flatten/unflatten, and the
``.efbeams`` property that the simulator consumes.

Example — scalar gain::

    @jax.tree_util.register_pytree_node_class
    class ScaledBeam(CachedBeam):
        def __init__(self, amplitude, base_efbeams):
            super().__init__(base_efbeams)
            self.amplitude = jnp.asarray(float(amplitude))

        def transform_beam(self, beamreal, groundpower):
            a = self.amplitude
            return a * beamreal, 1.0 - a * (1.0 - groundpower)

        def _param_leaves(self):
            return (self.amplitude,)

        @classmethod
        def _from_param_leaves(cls, params, base_efbeams):
            obj = cls.__new__(cls)
            CachedBeam.__init__(obj, base_efbeams)
            (obj.amplitude,) = params
            return obj
"""

import jax
import jax.numpy as jnp


class CachedBeam:
    """Base for beam pytrees that modify precomputed efbeams.

    Subclasses must:

    1. Call ``super().__init__(base_efbeams)`` with the simulator's
       ``sim.efbeams`` list.
    2. Implement :meth:`transform_beam` — how free parameters modify
       a single ``(beamreal, groundpower)`` pair.
    3. Implement :meth:`_param_leaves` — return a tuple of the
       subclass's free-parameter JAX arrays.
    4. Implement :meth:`_from_param_leaves` — classmethod that
       reconstructs the subclass from ``(param_leaves, base_efbeams)``.
    5. Register with ``@jax.tree_util.register_pytree_node_class``.
    """

    def __init__(self, base_efbeams):
        self._base_efbeams = list(base_efbeams)

    # -- subclass interface --------------------------------------------------

    def transform_beam(self, beamreal, groundpower):
        """Apply free parameters to a fixed beam pattern.

        :param beamreal: ``(Nfreq, Nalm)`` — fixed beam ALMs (already
            detached via ``stop_gradient``).
        :param groundpower: ``(Nfreq,)`` — fixed ground power fractions.
        :returns: ``(transformed_beamreal, transformed_groundpower)``.
        """
        raise NotImplementedError

    def _param_leaves(self):
        """Return a tuple of this subclass's free-parameter arrays."""
        raise NotImplementedError

    @classmethod
    def _from_param_leaves(cls, params, base_efbeams):
        """Reconstruct the subclass from param leaves and efbeams data."""
        raise NotImplementedError

    # -- efbeams property (consumed by simulator) ----------------------------

    @property
    def efbeams(self):
        result = []
        for ci, cj, br, bi, gpr, gpi in self._base_efbeams:
            br_s, gpr_s = self.transform_beam(
                jax.lax.stop_gradient(jnp.asarray(br)),
                jax.lax.stop_gradient(jnp.asarray(gpr)))
            if bi is not None:
                bi_s, gpi_s = self.transform_beam(
                    jax.lax.stop_gradient(jnp.asarray(bi)),
                    jax.lax.stop_gradient(jnp.asarray(gpi)))
            else:
                bi_s, gpi_s = None, 0.0
            result.append((ci, cj, br_s, bi_s, gpr_s, gpi_s))
        return result

    # -- pytree protocol (handles efbeams boilerplate) -----------------------

    def tree_flatten(self):
        param_leaves = self._param_leaves()
        efbeam_arrays = []
        combo_meta = []
        for ci, cj, br, bi, gpr, gpi in self._base_efbeams:
            efbeam_arrays.extend([jnp.asarray(br), jnp.asarray(gpr)])
            has_imag = bi is not None
            if has_imag:
                efbeam_arrays.extend([jnp.asarray(bi), jnp.asarray(gpi)])
            combo_meta.append((ci, cj, has_imag))
        children = tuple(param_leaves) + tuple(efbeam_arrays)
        aux = (len(param_leaves), tuple(combo_meta))
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        n_params, combo_meta = aux
        params = children[:n_params]
        idx = n_params
        base_efbeams = []
        for ci, cj, has_imag in combo_meta:
            br, gpr = children[idx], children[idx + 1]; idx += 2
            if has_imag:
                bi, gpi = children[idx], children[idx + 1]; idx += 2
            else:
                bi, gpi = None, 0.0
            base_efbeams.append((ci, cj, br, bi, gpr, gpi))
        return cls._from_param_leaves(params, base_efbeams)
