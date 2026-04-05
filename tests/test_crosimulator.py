import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import lusee


# ---- helpers ----

def _make_sim():
    """Shared small simulation fixture."""
    if lusee.CroSimulator is None:
        pytest.skip("croissant/s2fft not installed")

    nside = 8
    lmax = 3 * nside - 1
    freq = np.array([5.0, 10.0, 15.0], dtype=float)

    npix = 12 * nside * nside
    base_map = np.ones(npix, dtype=float)
    maps = [base_map * t for t in (1.0, 2.0, 3.0)]
    sky = lusee.sky.HealpixSky(nside, lmax, maps, freq=freq, frame="galactic")

    beam = lusee.BeamGauss(
        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=20.0,
        one_over_freq_scaling=False,
        id="beam",
    )

    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 06:00:00",
        deltaT_sec=7200.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )

    sim = lusee.CroSimulator(
        obs,
        [beam],
        sky,
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
    )
    return sim, obs, beam, freq, lmax


# ---- tests ----

def test_crosimulator_runs_and_returns_expected_shape():
    sim, obs, _, freq, _ = _make_sim()
    result = sim.simulate()

    assert result.shape == (len(obs.times), 1, len(freq))
    assert np.all(np.isfinite(result))
    assert np.any(np.abs(result) > 0.0)


def test_crosimulator_jit():
    """jax.jit works on simulate with explicit args."""
    sim, *_ = _make_sim()

    @jax.jit
    def run(beam_alms, sky_mepa):
        return sim.simulate(beam_alms, sky_mepa)

    result_jit = run(sim.beam_alms, sim.sky_mepa)
    result_eager = sim.simulate()
    np.testing.assert_allclose(
        np.asarray(result_jit), np.asarray(result_eager), atol=1e-12)


def test_crosimulator_grad_beam():
    """jax.grad(sum(waterfall²)) w.r.t. beam_alms matches finite differences."""
    sim, *_ = _make_sim()

    @jax.jit
    def loss(beam_alms):
        wf = sim.simulate(beam_alms)
        return jnp.sum(wf.real ** 2)

    grad = jax.grad(loss)(sim.beam_alms)
    assert grad.shape == sim.beam_alms.shape
    assert jnp.any(jnp.abs(grad) > 0)

    # Finite-difference check on the element with largest gradient
    grad_real = grad.real
    idx = np.unravel_index(int(jnp.argmax(jnp.abs(grad_real))), grad.shape)
    eps = 1e-5
    fd = float(
        (loss(sim.beam_alms.at[idx].add(eps))
         - loss(sim.beam_alms.at[idx].add(-eps))) / (2 * eps)
    )
    np.testing.assert_allclose(float(grad_real[idx]), fd, rtol=1e-4)


def test_crosimulator_grad_sky():
    """jax.grad w.r.t. sky_mepa works."""
    sim, *_ = _make_sim()

    @jax.jit
    def loss(sky_mepa):
        return jnp.sum(sim.simulate(sky_mepa=sky_mepa).real ** 2)

    grad = jax.grad(loss)(sim.sky_mepa)
    assert grad.shape == sim.sky_mepa.shape
    assert jnp.any(jnp.abs(grad) > 0)


def test_crosimulator_grad_Tground():
    """jax.grad w.r.t. Tground works (requires nonzero ground_power)."""
    sim, *_ = _make_sim()
    # Use nonzero ground power so Tground actually affects the output.
    gp = jnp.ones_like(sim.ground_power) * 0.1

    @jax.jit
    def loss(Tg):
        return jnp.sum(sim.simulate(Tground=Tg, ground_power=gp).real ** 2)

    grad = jax.grad(loss)(jnp.array(200.0))
    assert jnp.isfinite(grad)
    assert jnp.abs(grad) > 0


# ---- default / override patterns ----

def test_defaults_held_constant():
    """Gradient w.r.t. beam_alms does not change when sky is default vs explicit
    (same value) — verifies defaults are treated as constants correctly."""
    sim, *_ = _make_sim()

    def loss_default_sky(beam_alms):
        return jnp.sum(sim.simulate(beam_alms) ** 2)

    def loss_explicit_sky(beam_alms):
        return jnp.sum(sim.simulate(beam_alms, sky_mepa=sim.sky_mepa) ** 2)

    g1 = jax.grad(loss_default_sky)(sim.beam_alms)
    g2 = jax.grad(loss_explicit_sky)(sim.beam_alms)
    np.testing.assert_allclose(np.asarray(g1), np.asarray(g2), atol=1e-12)


def test_grad_beam_sky_independent():
    """Perturbing sky should not affect the beam gradient when sky is a default
    (closed-over constant), but SHOULD affect it when sky is explicit."""
    sim, *_ = _make_sim()

    # Grad w.r.t. beam with default sky
    def loss_beam_only(b):
        return jnp.sum(sim.simulate(b) ** 2)
    g_default = jax.grad(loss_beam_only)(sim.beam_alms)

    # Grad w.r.t. beam with a scaled sky passed explicitly
    sky_scaled = sim.sky_mepa * 2.0
    def loss_beam_with_sky(b):
        return jnp.sum(sim.simulate(b, sky_mepa=sky_scaled) ** 2)
    g_scaled = jax.grad(loss_beam_with_sky)(sim.beam_alms)

    # These should differ because the sky changed the forward pass
    assert not jnp.allclose(g_default, g_scaled)


def test_grad_sky_with_default_beam():
    """Grad w.r.t. sky while beam is held at default, with FD check."""
    sim, *_ = _make_sim()

    @jax.jit
    def loss(sky):
        return jnp.sum(sim.simulate(sky_mepa=sky).real ** 2)

    grad = jax.grad(loss)(sim.sky_mepa)
    grad_real = grad.real
    idx = np.unravel_index(int(jnp.argmax(jnp.abs(grad_real))), grad.shape)
    eps = 1e-5
    fd = float(
        (loss(sim.sky_mepa.at[idx].add(eps))
         - loss(sim.sky_mepa.at[idx].add(-eps))) / (2 * eps)
    )
    np.testing.assert_allclose(float(grad_real[idx]), fd, rtol=1e-4)


def test_grad_beam_and_sky_simultaneous():
    """Gradient w.r.t. beam and sky simultaneously via a tuple arg."""
    sim, *_ = _make_sim()

    @jax.jit
    def loss(beam_and_sky):
        b, s = beam_and_sky
        return jnp.sum(sim.simulate(b, s) ** 2)

    grad_b, grad_s = jax.grad(loss)((sim.beam_alms, sim.sky_mepa))
    assert grad_b.shape == sim.beam_alms.shape
    assert grad_s.shape == sim.sky_mepa.shape
    assert jnp.any(jnp.abs(grad_b) > 0)
    assert jnp.any(jnp.abs(grad_s) > 0)


def test_grad_ground_power_with_defaults():
    """Grad w.r.t. ground_power while beam/sky are defaults."""
    sim, *_ = _make_sim()
    Tg = jnp.array(200.0)

    @jax.jit
    def loss(gp):
        return jnp.sum(sim.simulate(Tground=Tg, ground_power=gp).real ** 2)

    grad = jax.grad(loss)(sim.ground_power)
    assert grad.shape == sim.ground_power.shape
    assert jnp.all(jnp.isfinite(grad))


def test_explicit_args_match_defaults():
    """Passing all defaults explicitly produces the same result as no args."""
    sim, *_ = _make_sim()
    wf_default = sim.simulate()
    wf_explicit = sim.simulate(sim.beam_alms, sim.sky_mepa,
                               sim.Tground, sim.ground_power)
    np.testing.assert_allclose(
        np.asarray(wf_default), np.asarray(wf_explicit), atol=1e-12)
