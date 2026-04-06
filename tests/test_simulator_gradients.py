#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import pytest


def _small_grad_setup(tmp_path):
    import lusee

    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    freq = jnp.array([10.0])
    lmax = 8
    sky = lusee.sky.HarmonicPointSourceSky(lmax=lmax, l_deg=0.0, b_deg=0.0, freq=freq)
    beam = lusee.BeamGauss(
        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=20.0,
        one_over_freq_scaling=False,
        id="grad",
    )
    cache_prefix = str(tmp_path / "grad_cache")
    return lusee, obs, times, freq, lmax, sky, beam, cache_prefix


def test_jaxsim_grad_wrt_beam(tmp_path):
    """Differentiate a small JaxSimulator loss with respect to a Gaussian beam."""
    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

    def loss_fn(beam):
        sim = lusee.JaxSimulator(
            obs,
            [beam],
            sky,
            Tground=0.0,
            combinations=[(0, 0)],
            freq=freq,
            lmax=lmax,
            extra_opts={"cache_transform": cache_prefix},
        )
        result = sim.simulate(times=times)
        return jnp.real(jnp.vdot(result, result))

    grad_beam = jax.grad(loss_fn)(beam)

    assert isinstance(grad_beam, lusee.BeamGauss)
    assert jnp.isfinite(grad_beam.Etheta).all()
    assert jnp.isfinite(grad_beam.gain_conv).all()
    assert jnp.linalg.norm(grad_beam.Etheta) > 0
    assert jnp.linalg.norm(grad_beam.gain_conv) > 0


def test_jaxsim_grad_wrt_sky(tmp_path):
    """Differentiate a small JaxSimulator loss with respect to a point-source sky."""
    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

    def loss_fn(sky):
        sim = lusee.JaxSimulator(
            obs,
            [beam],
            sky,
            Tground=0.0,
            combinations=[(0, 0)],
            freq=freq,
            lmax=lmax,
            extra_opts={"cache_transform": cache_prefix},
        )
        result = sim.simulate(times=times)
        return jnp.real(jnp.vdot(result, result))

    grad_sky = jax.grad(loss_fn)(sky)

    assert isinstance(grad_sky, lusee.sky.HarmonicPointSourceSky)
    assert jnp.isfinite(grad_sky._alm).all()
    assert jnp.isfinite(grad_sky._T).all()
    assert jnp.linalg.norm(grad_sky._alm) > 0
    assert jnp.linalg.norm(grad_sky._T) > 0


def test_crosim_grad_wrt_beam(tmp_path):
    """Differentiate a small CroSimulator loss with respect to a Gaussian beam."""
    import lusee

    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")

    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

    def loss_fn(beam):
        sim = lusee.CroSimulator(
            obs,
            [beam],
            sky,
            Tground=0.0,
            combinations=[(0, 0)],
            freq=freq,
            lmax=lmax,
            extra_opts={"cache_transform": cache_prefix},
        )
        result = sim.simulate(times=times)
        return jnp.real(jnp.vdot(result, result))

    grad_beam = jax.grad(loss_fn)(beam)

    assert isinstance(grad_beam, lusee.BeamGauss)
    assert jnp.isfinite(grad_beam.Etheta).all()
    assert jnp.isfinite(grad_beam.gain_conv).all()
    assert jnp.linalg.norm(grad_beam.Etheta) > 0
    assert jnp.linalg.norm(grad_beam.gain_conv) > 0


def test_crosim_grad_wrt_sky(tmp_path):
    """Differentiate a small CroSimulator loss with respect to a point-source sky."""
    import lusee

    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")

    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

    def loss_fn(sky):
        sim = lusee.CroSimulator(
            obs,
            [beam],
            sky,
            Tground=0.0,
            combinations=[(0, 0)],
            freq=freq,
            lmax=lmax,
            extra_opts={"cache_transform": cache_prefix},
        )
        result = sim.simulate(times=times)
        return jnp.real(jnp.vdot(result, result))

    grad_sky = jax.grad(loss_fn)(sky)

    assert isinstance(grad_sky, lusee.sky.HarmonicPointSourceSky)
    assert jnp.isfinite(grad_sky._alm).all()
    assert jnp.isfinite(grad_sky._T).all()
    assert jnp.linalg.norm(grad_sky._alm) > 0
    assert jnp.linalg.norm(grad_sky._T) > 0


def test_crosim_grad_wrt_sky_precomputed(tmp_path):
    """Differentiate via sky= kwarg (precomputed simulator, approach A).

    Builds the simulator once, then differentiates only through simulate().
    Faster per iteration than the through-constructor approach above.
    """
    import lusee

    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")

    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

    sim = lusee.CroSimulator(
        obs,
        [beam],
        sky,
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"cache_transform": cache_prefix},
    )

    def loss_fn(sky):
        result = sim.simulate(times=times, sky=sky)
        return jnp.real(jnp.vdot(result, result))

    grad_sky = jax.grad(loss_fn)(sky)

    assert isinstance(grad_sky, lusee.sky.HarmonicPointSourceSky)
    assert jnp.isfinite(grad_sky._T).all()
    assert jnp.linalg.norm(grad_sky._T) > 0
