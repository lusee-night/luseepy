#!/usr/bin/env python3

import os
import tempfile
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import pytest


def test_all_simulators_smoke(tmp_path):
    """Small end-to-end simulator run with only three times."""
    import lusee

    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    assert len(times) == 3

    freq = np.array([10.0])
    lmax = 8
    sky = lusee.sky.HarmonicPointSourceSky(lmax=lmax, l_deg=0.0, b_deg=0.0, freq=freq)
    beam = lusee.BeamGauss(
        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=20.0,
        one_over_freq_scaling=False,
        id="smoke",
    )

    cache_prefix = str(tmp_path / "sim_smoke_cache")

    default_sim = lusee.DefaultSimulator(
        obs,
        [lusee.NpWrapper(beam)],
        lusee.NpWrapper(sky),
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"cache_transform": cache_prefix},
    )
    default_result = np.asarray(default_sim.simulate(times=times))
    assert default_result.shape == (3, 1, 1)
    assert np.isfinite(default_result).all()

    jax_sim = lusee.JaxSimulator(
        obs,
        [beam],
        sky,
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"cache_transform": cache_prefix},
    )
    jax_result = np.asarray(jax_sim.simulate(times=times))
    assert jax_result.shape == (3, 1, 1)
    assert np.isfinite(jax_result).all()

    assert np.allclose(jax_result, default_result, rtol=1e-9, atol=1e-9)

    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")

    cro_sim = lusee.CroSimulator(
        obs,
        [beam],
        sky,
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={},
    )
    cro_result = np.asarray(cro_sim.simulate(times=times))
    assert cro_result.shape == (3, 1, 1)
    assert np.isfinite(cro_result).all()

    assert np.allclose(cro_result, default_result, rtol=5e-3, atol=5e-3)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        test_all_simulators_smoke(Path(tmpdir))
    print("test_simulator_smoke: passed.")
