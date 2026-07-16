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
    # Sky grid includes 12.0 so the snap-on-match test exercises a frequency
    # that is on-grid for BOTH the beam and the sky; otherwise the sky path
    # genuinely interpolates and a 1e-10 target shift perturbs the output at
    # machine precision, defeating the bit-identical check.
    sky_freq = np.asarray([1.0, 12.0, 25.0, 50.0])
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


def test_cro_simulate_sky_override_uses_override_grid():
    """simulate(sky=...) must interpolate the OVERRIDE sky on its own grid.

    Regression: the sky dispatch checked get_alm_at_freq on the effective sky
    but interpolated with the constructor sky's FrequencyMap -- crashing with
    AttributeError when the constructor sky was closed-form (freq_map_sky is
    None) and misinterpolating when the override grid differed.
    """
    import lusee

    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")

    obs = _build_obs()
    lmax = 8
    target_freq = np.asarray([10.0, 20.0])
    beam = lusee.BeamGauss(alt_deg=90.0, az_deg=0.0, sigma_deg=20.0,
                           one_over_freq_scaling=False, id="ovr")
    # constructor sky is closed-form (has get_alm_at_freq, so freq_map_sky is
    # None); the override sky is gridded, galactic, and off-grid for the targets
    base_sky = lusee.sky.ConstSkyCane1979(16, lmax=lmax)
    override_sky = lusee.sky.HarmonicPointSourceSky(
        lmax=lmax, l_deg=0.0, b_deg=0.0,
        freq=np.asarray([5.0, 15.0, 25.0]), T=np.asarray([1.0, 3.0, 5.0]),
    )

    sim = lusee.CroSimulator(obs, [beam], base_sky, Tground=0.0,
                             combinations=[(0, 0)], freq=target_freq, lmax=lmax)
    res_override = np.asarray(sim.simulate(times=obs.times, sky=override_sky))
    assert np.isfinite(res_override).all()

    sim_direct = lusee.CroSimulator(obs, [beam], override_sky, Tground=0.0,
                                    combinations=[(0, 0)], freq=target_freq, lmax=lmax)
    res_direct = np.asarray(sim_direct.simulate(times=obs.times))
    np.testing.assert_allclose(res_override, res_direct, rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    test_snap_on_match_is_bit_identical()
    test_offgrid_run_produces_finite_output()
    test_out_of_range_raises()
    print("test_offgrid_interpolation: passed.")
