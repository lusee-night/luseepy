#!/usr/bin/env python3

import os

os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import pytest
import lusee
from lusee.frequencies import canonical_frequencies, frequency_indices_from_values


def _small_grad_setup(tmp_path):
    import lusee

    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    freq = canonical_frequencies(frequency_indices_from_values([10.0]), as_jax=True)
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


def test_jaxsim_grad_wrt_sky_precomputed(tmp_path):
    """Differentiate via sky= kwarg (precomputed JaxSimulator).

    Builds the simulator once, then differentiates only through simulate().
    Mirrors test_crosim_grad_wrt_sky_precomputed to verify the JaxSimulator
    drop-in path used by examples/optax_maxlike.py.
    """
    lusee, obs, times, freq, lmax, sky, beam, cache_prefix = _small_grad_setup(tmp_path)

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

    # sky= must reproduce the cached-sky output bit-for-bit when the same
    # pytree is supplied.
    wf_default = sim.simulate(times=times)
    wf_sky_kwarg = sim.simulate(times=times, sky=sky)
    assert jnp.allclose(wf_default, wf_sky_kwarg, atol=1e-12)

    def loss_fn(sky):
        result = sim.simulate(times=times, sky=sky)
        return jnp.real(jnp.vdot(result, result))

    grad_sky = jax.grad(loss_fn)(sky)

    assert isinstance(grad_sky, lusee.sky.HarmonicPointSourceSky)
    assert jnp.isfinite(grad_sky._T).all()
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


# ---------------------------------------------------------------------------
# ScaledBeam: dummy CachedBeam subclass for testing beam= kwarg
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class ScaledBeam(lusee.CachedBeam):
    """Beam = amplitude * cached pattern.  Tutorial CachedBeam subclass."""

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
        lusee.CachedBeam.__init__(obj, base_efbeams)
        (obj.amplitude,) = params
        return obj


def test_crosim_grad_wrt_beam_precomputed(tmp_path):
    """Differentiate via beam= kwarg with ScaledBeam (approach A for beams)."""
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

    sb = ScaledBeam(1.0, sim.efbeams)

    # amplitude=1 should match default
    wf_default = sim.simulate(times=times)
    wf_scaled = sim.simulate(times=times, beam=sb)
    assert jnp.allclose(wf_default, wf_scaled, atol=1e-12)

    # gradient w.r.t. amplitude
    def loss_fn(b):
        return jnp.real(jnp.vdot(sim.simulate(times=times, beam=b),
                                  sim.simulate(times=times, beam=b)))

    grad_beam = jax.grad(loss_fn)(sb)

    assert isinstance(grad_beam, ScaledBeam)
    assert jnp.isfinite(grad_beam.amplitude)
    assert jnp.abs(grad_beam.amplitude) > 0
