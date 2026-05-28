"""Off-grid frequency interpolation in the simulator pipeline.

Covers three properties of the post-canonicalization refactor:
1. Snap-on-match: a target frequency within atol of a native beam frequency
   is bit-identical to running with the native value (no float garbage from
   alpha = 1e-10 instead of 0.0).
2. Off-grid runs produce finite, correctly-shaped output.
3. Out-of-range targets raise ValueError naming the offending value.
"""

import os
import pytest

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np


def _build_obs():
    import lusee

    return lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )


def _run_numpy_sim(freq):
    import lusee

    obs = _build_obs()
    times = obs.times
    lmax = 8
    sky_freq = np.asarray([1.0, 25.0, 50.0])  # broad sky grid covering the beam
    sky = lusee.sky.HarmonicPointSourceSky(lmax=lmax, l_deg=0.0, b_deg=0.0,
                                           freq=sky_freq)
    beam = lusee.BeamGauss(alt_deg=90.0, az_deg=0.0, sigma_deg=20.0,
                           one_over_freq_scaling=False, id="t")
    sim = lusee.TopoNumpySimulator(
        obs,
        [lusee.NpWrapper(beam)],
        lusee.NpWrapper(sky),
        Tground=0.0,
        combinations=[(0, 0)],
        freq=np.asarray(freq, dtype=float),
        lmax=lmax,
    )
    return np.asarray(sim.simulate(times=times))


def test_snap_on_match_is_bit_identical():
    """freq = 12.0 and freq = 12.0 + 1e-10 must produce identical waterfalls."""
    res_native = _run_numpy_sim([12.0])
    res_eps = _run_numpy_sim([12.0 + 1e-10])
    assert res_native.shape == res_eps.shape
    # Within atol of the snap-on-match path the simulator should produce
    # bit-identical floating-point output (alpha is exactly 0.0).
    assert np.array_equal(res_native, res_eps), (
        "snap-on-match did not produce bit-identical output; "
        f"max abs diff = {np.max(np.abs(res_native - res_eps))}"
    )


def test_offgrid_run_produces_finite_output():
    """A target between two native beam frequencies must run cleanly."""
    res = _run_numpy_sim([12.5, 17.3])
    assert res.shape[-1] == 2  # two frequency bins in the waterfall
    assert np.isfinite(res).all()


def test_out_of_range_raises():
    """Requesting 55 MHz on a 1-50 MHz BeamGauss must raise ValueError."""
    with pytest.raises(ValueError, match=r"out of range"):
        _run_numpy_sim([55.0])


if __name__ == "__main__":
    test_snap_on_match_is_bit_identical()
    test_offgrid_run_produces_finite_output()
    test_out_of_range_raises()
    print("test_offgrid_interpolation: passed.")
