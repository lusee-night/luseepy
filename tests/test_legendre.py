"""Tests for the pure-JAX associated Legendre function implementation."""

import os
os.environ["JAX_ENABLE_X64"] = "True"

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import sph_harm_y

from lusee.Beam import getLegendre, _getLegendre_packed


@pytest.mark.parametrize("lmax", [4, 16, 64])
@pytest.mark.parametrize("theta", [0.3, 1.0, 2.5])
def test_legendre_vs_scipy(lmax, theta):
    """Check that P_l^m values match scipy's sph_harm_y(l, m, theta, 0)."""
    L = getLegendre(lmax, theta)
    for l in range(lmax + 1):
        for m in range(l + 1):
            ref = sph_harm_y(l, m, theta, 0.0).real
            assert abs(float(L[l, m]) - ref) < 1e-12, f"l={l}, m={m}"


@pytest.mark.parametrize("lmax", [192, 384])
def test_legendre_high_lmax_no_nan(lmax):
    """Ensure no NaNs at the lmax values relevant for LuSEE-Night."""
    vals = _getLegendre_packed(lmax, 1.0)
    assert not jnp.any(jnp.isnan(vals))
    assert vals.shape == ((lmax + 1) * (lmax + 2) // 2,)


def test_legendre_packed_shape():
    lmax = 10
    vals = _getLegendre_packed(lmax, 0.5)
    assert vals.shape == ((lmax + 1) * (lmax + 2) // 2,)


def test_legendre_theta_zero():
    """At theta=0 (pole), only m=0 terms should be nonzero."""
    lmax = 32
    L = getLegendre(lmax, 0.0)
    for l in range(lmax + 1):
        ref = sph_harm_y(l, 0, 0.0, 0.0).real
        assert abs(float(L[l, 0]) - ref) < 1e-13
        for m in range(1, l + 1):
            assert abs(float(L[l, m])) < 1e-15, f"l={l}, m={m} should be 0 at pole"
