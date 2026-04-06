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


# ---------------------------------------------------------------------------
# ScaledBeam: dummy pytree beam for testing beam= kwarg
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
class ScaledBeam:
    """Beam = amplitude * precomputed efbeams.  Tutorial pytree."""

    def __init__(self, amplitude, base_efbeams):
        self.amplitude = jnp.asarray(float(amplitude))
        self._base_efbeams = base_efbeams

    @property
    def efbeams(self):
        a = self.amplitude
        result = []
        for ci, cj, br, bi, gpr, gpi in self._base_efbeams:
            br_s = a * jax.lax.stop_gradient(jnp.asarray(br))
            gpr_s = 1.0 - a * (1.0 - jax.lax.stop_gradient(jnp.asarray(gpr)))
            if bi is not None:
                bi_s = a * jax.lax.stop_gradient(jnp.asarray(bi))
                gpi_s = -a * jax.lax.stop_gradient(jnp.asarray(-gpi))
            else:
                bi_s = None
                gpi_s = 0.0
            result.append((ci, cj, br_s, bi_s, gpr_s, gpi_s))
        return result

    def tree_flatten(self):
        arrays = [self.amplitude]
        combo_meta = []
        for ci, cj, br, bi, gpr, gpi in self._base_efbeams:
            arrays.extend([jnp.asarray(br), jnp.asarray(gpr)])
            has_imag = bi is not None
            if has_imag:
                arrays.extend([jnp.asarray(bi), jnp.asarray(gpi)])
            combo_meta.append((ci, cj, has_imag))
        return tuple(arrays), tuple(combo_meta)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        obj.amplitude = children[0]
        idx = 1
        obj._base_efbeams = []
        for ci, cj, has_imag in aux:
            br, gpr = children[idx], children[idx + 1]; idx += 2
            if has_imag:
                bi, gpi = children[idx], children[idx + 1]; idx += 2
            else:
                bi, gpi = None, 0.0
            obj._base_efbeams.append((ci, cj, br, bi, gpr, gpi))
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
