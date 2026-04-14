"""Regression tests for JAX-tracing behaviour of SkyModels pytrees."""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import numpy as np

from lusee.SkyModels import ConstSky


def test_constsky_scalar_T_supports_grad_through_pytree():
    """ConstSky.T() must handle _T as a 0-d JAX array after tree_unflatten.

    Regression for a bug where ``type(self._T) == float`` was used to branch
    between ``jnp.full`` and ``self._T[ndx]``. After pytree unflatten, ``_T``
    is a 0-d JAX array, so the check fell through to ``self._T[ndx]`` and
    raised ``IndexError: Too many indices: array is 0-dimensional``.
    """
    sky = ConstSky(Nside=8, lmax=4, T=200.0)

    def loss(sky_):
        alm = sky_.get_alm(jnp.array([0]))
        return jnp.sum(jnp.abs(alm) ** 2)

    # Before the fix, this raised IndexError during tracing.
    grad = jax.grad(loss)(sky)

    assert jnp.all(jnp.isfinite(grad.mapalm))
    # _T gradient should also be finite (scalar)
    assert jnp.isfinite(grad._T)


def test_constsky_vector_T_still_works():
    """Sanity check: array-valued T still indexes correctly."""
    T = jnp.array([100.0, 200.0, 300.0])
    sky = ConstSky(Nside=8, lmax=4, T=T)
    Ts = sky.T(jnp.array([0, 2]))
    np.testing.assert_allclose(np.asarray(Ts), [100.0, 300.0])
