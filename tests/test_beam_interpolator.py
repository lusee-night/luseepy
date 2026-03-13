"""BeamInterpolator tests."""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from lusee.BeamInterpolator import BeamInterpolator


# ---- Mock / helpers ----

class MockBeam:
    """Beam stub returning fixed ALM data."""

    def __init__(self, alm_data):
        self.alm_data = np.asarray(alm_data, dtype=float)

    def get_healpix_alm(self, lmax, *, freq_ndx=None, other=None,
                         return_I_stokes_only=True, return_complex_components=False):
        if return_I_stokes_only:
            return self.alm_data
        return [self.alm_data, np.zeros_like(self.alm_data),
                np.zeros_like(self.alm_data), np.zeros_like(self.alm_data)]


def _make_loss(interp):
    """L(q) = ∑ interp(q)² — scalar loss for grad tests."""
    def loss(q):
        return jnp.sum(interp.interpolate(q) ** 2)
    return loss


def _fd_grad(fn, x, eps=1e-5):
    """Central finite-difference gradient."""
    g = np.zeros(len(x))
    for i in range(len(x)):
        xp = x.at[i].add(eps)
        xm = x.at[i].add(-eps)
        g[i] = (fn(xp) - fn(xm)) / (2 * eps)
    return g


# ---- Fixtures ----

@pytest.fixture
def linear_1d():
    """5 beams, 1D params, ALMs = p * basis."""
    rng = np.random.default_rng(42)
    params = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    basis = rng.standard_normal((3, 4))
    beams = [MockBeam(p * basis) for p in params]
    return beams, params, basis


@pytest.fixture
def linear_2d():
    """6 beams, 2D params, ALMs = a·p₀ + b·p₁."""
    rng = np.random.default_rng(99)
    params = rng.standard_normal((6, 2))
    ba = rng.standard_normal((2, 3))
    bb = rng.standard_normal((2, 3))
    beams = [MockBeam(p[0] * ba + p[1] * bb) for p in params]
    return beams, params


# ---- Constructor validation ----

def test_empty_beams_raises():
    with pytest.raises(ValueError, match="at least one"):
        BeamInterpolator([], np.array([]), lmax=2)


def test_params_beam_count_mismatch_raises():
    rng = np.random.default_rng(0)
    beams = [MockBeam(rng.standard_normal((2, 3))) for _ in range(3)]
    with pytest.raises(ValueError, match="must match"):
        BeamInterpolator(beams, np.array([1.0, 2.0]), lmax=2)


# ---- Interpolation ----

def test_exact_at_training_points():
    """RBF interpolant passes through training data exactly."""
    rng = np.random.default_rng(55)
    alm_values = [rng.standard_normal((3, 4)) for _ in range(5)]
    beams = [MockBeam(v) for v in alm_values]
    params = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    interp = BeamInterpolator(beams, params, lmax=3)

    for i, p in enumerate(params):
        r = interp.interpolate(jnp.array([p]))
        np.testing.assert_allclose(np.asarray(r[0]), alm_values[i], atol=1e-10)


def test_reproduces_linear_exactly(linear_1d):
    """With linear polynomial augmentation, linear data is exact everywhere."""
    beams, params, basis = linear_1d
    interp = BeamInterpolator(beams, params, lmax=3)

    for q_val in [2.0, 4.5, 6.0, 8.0]:
        r = interp.interpolate(jnp.array([q_val]))
        expected = q_val * basis
        np.testing.assert_allclose(np.asarray(r[0]), expected, atol=1e-10)


def test_jit_matches_eager(linear_1d):
    beams, params, _ = linear_1d
    interp = BeamInterpolator(beams, params, lmax=3)

    q = jnp.array([4.0])
    eager = interp.interpolate(q)
    jitted = jax.jit(interp.interpolate)(q)
    np.testing.assert_allclose(eager, jitted, atol=1e-12)


def test_stokes_output_shape():
    """return_I_stokes_only=False → 4 Stokes components."""
    rng = np.random.default_rng(10)
    beams = [MockBeam(rng.standard_normal((2, 3))) for _ in range(3)]
    params = np.array([0.0, 1.0, 2.0])
    interp = BeamInterpolator(beams, params, lmax=2, return_I_stokes_only=False)
    r = interp.interpolate(jnp.array([0.5]))
    assert r.shape[0] == 4


# ---- Gradients ----

@pytest.mark.parametrize("q_val", [2.0, 4.5, 6.0, 8.0])
def test_grad_1d(linear_1d, q_val):
    """JAX grad matches finite differences (1D)."""
    beams, params, _ = linear_1d
    interp = BeamInterpolator(beams, params, lmax=3)
    loss = _make_loss(interp)

    q = jnp.array([q_val])
    jax_g = np.asarray(jax.grad(loss)(q))
    fd_g = _fd_grad(loss, q)
    np.testing.assert_allclose(jax_g, fd_g, rtol=1e-4)


def test_grad_2d(linear_2d):
    """JAX grad matches finite differences (2D)."""
    beams, params = linear_2d
    interp = BeamInterpolator(beams, params, lmax=2)
    loss = _make_loss(interp)

    q = jnp.array([0.5, -0.3])
    jax_g = np.asarray(jax.grad(loss)(q))
    fd_g = _fd_grad(loss, q)
    np.testing.assert_allclose(jax_g, fd_g, rtol=1e-4)


def test_grad_symmetry():
    """Identical beams → zero gradient at midpoint."""
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((2, 3))
    beams = [MockBeam(basis), MockBeam(basis)]
    params = np.array([0.0, 2.0])
    interp = BeamInterpolator(beams, params, lmax=2)

    q = jnp.array([1.0])
    g = np.asarray(jax.grad(_make_loss(interp))(q))
    np.testing.assert_allclose(g, 0.0, atol=1e-10)
