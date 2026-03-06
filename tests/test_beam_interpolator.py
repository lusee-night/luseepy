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
        data = self.alm_data
        if return_I_stokes_only:
            return data
        return [data, np.zeros_like(data), np.zeros_like(data), np.zeros_like(data)]


def _weights_np(query, parameters, kernel_width):
    """Numpy reference implementation of Gaussian softmax weights."""
    dpar = (parameters - query[None, :]) / kernel_width[None, :]
    dist2 = np.sum(dpar * dpar, axis=1)
    logits = -0.5 * dist2
    logits -= np.max(logits)
    w = np.exp(logits)
    return w / np.sum(w)


def _analytic_grad_sum_sq(query, parameters, kernel_width, data):
    """∇_q ∑(interp²) via Nadaraya-Watson weight derivatives."""
    w = _weights_np(query, parameters, kernel_width)
    f = np.tensordot(w, data, axes=([0], [1]))

    g = (parameters - query[None, :]) / (kernel_width[None, :] ** 2)
    g_bar = np.sum(w[:, None] * g, axis=0)
    dw_dq = w[:, None] * (g - g_bar[None, :])

    grad = np.zeros(len(query))
    for m in range(len(query)):
        df_dqm = np.tensordot(dw_dq[:, m], data, axes=([0], [1]))
        grad[m] = 2.0 * np.sum(f * df_dqm)
    return grad


def _make_loss(interp):
    """L(q) = ∑ interp(q)² — scalar loss for grad tests."""
    def loss(q):
        return jnp.sum(interp.interpolate(q) ** 2)
    return loss


# ---- Fixtures ----

@pytest.fixture
def linear_1d():
    """5 beams, 1D params, ALMs = p * basis."""
    rng = np.random.default_rng(42)
    n_beams, n_freq, n_alm = 5, 3, 4
    params = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    basis = rng.standard_normal((n_freq, n_alm))
    beams = [MockBeam(p * basis) for p in params]
    kw = 2.0
    return beams, params, kw


@pytest.fixture
def linear_2d():
    """6 beams, 2D params, ALMs = a·p₀ + b·p₁."""
    rng = np.random.default_rng(99)
    n_beams, n_freq, n_alm = 6, 2, 3
    params = rng.standard_normal((n_beams, 2))
    ba = rng.standard_normal((n_freq, n_alm))
    bb = rng.standard_normal((n_freq, n_alm))
    beams = [MockBeam(p[0] * ba + p[1] * bb) for p in params]
    kw = np.array([1.0, 1.5])
    return beams, params, kw


# ---- Constructor validation tests ----

def test_empty_beams_raises():
    with pytest.raises(ValueError, match="at least one"):
        BeamInterpolator([], np.array([]), kernel_width=1.0, lmax=2)


def test_params_beam_count_mismatch_raises():
    rng = np.random.default_rng(0)
    beams = [MockBeam(rng.standard_normal((2, 3))) for _ in range(3)]
    with pytest.raises(ValueError, match="must match"):
        BeamInterpolator(beams, np.array([1.0, 2.0]), kernel_width=1.0, lmax=2)


def test_negative_kernel_width_raises():
    rng = np.random.default_rng(0)
    beams = [MockBeam(rng.standard_normal((2, 3)))]
    with pytest.raises(ValueError, match="positive"):
        BeamInterpolator(beams, np.array([0.0]), kernel_width=-1.0, lmax=2)


def test_zero_kernel_width_raises():
    rng = np.random.default_rng(0)
    beams = [MockBeam(rng.standard_normal((2, 3)))]
    with pytest.raises(ValueError, match="positive"):
        BeamInterpolator(beams, np.array([0.0]), kernel_width=0.0, lmax=2)


def test_vector_kernel_width_wrong_shape_raises():
    rng = np.random.default_rng(0)
    beams = [MockBeam(rng.standard_normal((2, 3)))]
    with pytest.raises(ValueError, match="kernel_width"):
        BeamInterpolator(beams, np.array([[0.0, 1.0]]),
                         kernel_width=np.array([1.0, 2.0, 3.0]), lmax=2)


# ---- Interpolation tests ----

def test_weights_sum_to_one(linear_1d):
    beams, params, kw = linear_1d
    interp = BeamInterpolator(beams, params, kernel_width=kw, lmax=3)
    w = interp._weights(jnp.array([4.0]))
    np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-12)


def test_exact_recovery_tight_kernel():
    """Tight kernel → exact recovery at training points."""
    rng = np.random.default_rng(55)
    n_freq, n_alm = 3, 4
    alm_values = [rng.standard_normal((n_freq, n_alm)) for _ in range(4)]
    beams = [MockBeam(v) for v in alm_values]
    params = np.array([0.0, 1.0, 2.0, 3.0])

    interp = BeamInterpolator(beams, params, kernel_width=0.01, lmax=3)

    for i, p in enumerate(params):
        r = interp.interpolate(jnp.array([p]))
        np.testing.assert_allclose(np.asarray(r[0]), alm_values[i], atol=1e-10)


def test_jit_matches_eager(linear_1d):
    beams, params, kw = linear_1d
    interp = BeamInterpolator(beams, params, kernel_width=kw, lmax=3)

    q = jnp.array([4.0])
    eager = interp.interpolate(q)
    jitted = jax.jit(interp.interpolate)(q)
    np.testing.assert_allclose(eager, jitted, atol=1e-12)


def test_stokes_output_shape():
    """return_I_stokes_only=False → 4 Stokes components."""
    rng = np.random.default_rng(10)
    beams = [MockBeam(rng.standard_normal((2, 3))) for _ in range(3)]
    params = np.array([0.0, 1.0, 2.0])
    interp = BeamInterpolator(beams, params, kernel_width=1.0, lmax=2,
                               return_I_stokes_only=False)
    r = interp.interpolate(jnp.array([0.5]))
    assert r.shape[0] == 4


# ---- Gradient tests ----

def _extract_data(interp):
    """Pull interp._data as numpy for analytic grad reference."""
    return np.asarray(interp._data)


@pytest.mark.parametrize("q_val", [2.0, 4.5, 6.0, 8.0])
def test_grad_1d(linear_1d, q_val):
    """JAX grad matches analytic Nadaraya-Watson gradient (1D)."""
    beams, params, kw = linear_1d
    interp = BeamInterpolator(beams, params, kernel_width=kw, lmax=3)
    data = _extract_data(interp)

    query = np.array([q_val])
    analytic = _analytic_grad_sum_sq(query, params[:, None], np.array([kw]), data)
    jax_g = np.asarray(jax.grad(_make_loss(interp))(jnp.array(query)))
    np.testing.assert_allclose(jax_g, analytic, rtol=1e-5)


def test_grad_2d(linear_2d):
    """JAX grad matches analytic Nadaraya-Watson gradient (2D)."""
    beams, params, kw = linear_2d
    interp = BeamInterpolator(beams, params, kernel_width=kw, lmax=2)
    data = _extract_data(interp)

    query = np.array([0.5, -0.3])
    analytic = _analytic_grad_sum_sq(query, params, kw, data)
    jax_g = np.asarray(jax.grad(_make_loss(interp))(jnp.array(query)))
    np.testing.assert_allclose(jax_g, analytic, rtol=1e-5)


def test_grad_symmetry():
    """Identical beams → zero gradient at midpoint."""
    rng = np.random.default_rng(7)
    basis = rng.standard_normal((2, 3))
    beams = [MockBeam(basis), MockBeam(basis)]
    params = np.array([0.0, 2.0])
    interp = BeamInterpolator(beams, params, kernel_width=1.0, lmax=2)

    q = jnp.array([1.0])
    g = np.asarray(jax.grad(_make_loss(interp))(q))
    np.testing.assert_allclose(g, 0.0, atol=1e-10)


def test_grad_at_training_point_tight_kernel(linear_1d):
    """Tight kernel → near-zero gradient at training point."""
    beams, params, _ = linear_1d
    interp = BeamInterpolator(beams, params, kernel_width=0.01, lmax=3)

    q = jnp.array([5.0])  # exact training point
    g = np.asarray(jax.grad(_make_loss(interp))(q))
    np.testing.assert_allclose(g, 0.0, atol=1e-3)
