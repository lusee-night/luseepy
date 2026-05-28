"""Extended frequency range (up to 75 MHz) via BeamGauss freq_min/freq_max kwargs.

The synthetic 1-75 MHz BeamGauss exists so the simulator pipeline can be
exercised beyond the historic 1-50 MHz canonical band; this is the smoke test
that the new code path is wired end-to-end.
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np


def test_simulator_runs_at_70_mhz():
    """Build a 1-75 MHz BeamGauss and run the simulator at three off-historic frequencies."""
    import lusee

    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    lmax = 8
    target_freq = np.asarray([10.0, 50.0, 70.0])

    sky_freq = np.linspace(1.0, 75.0, 75)
    sky = lusee.sky.HarmonicPointSourceSky(lmax=lmax, l_deg=0.0, b_deg=0.0,
                                           freq=sky_freq)
    beam = lusee.BeamGauss(
        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=20.0,
        one_over_freq_scaling=False,
        id="ext",
        freq_min=1.0,
        freq_max=75.0,
        Nfreq=75,
    )

    sim = lusee.TopoNumpySimulator(
        obs,
        [lusee.NpWrapper(beam)],
        lusee.NpWrapper(sky),
        Tground=0.0,
        combinations=[(0, 0)],
        freq=target_freq,
        lmax=lmax,
    )
    result = np.asarray(sim.simulate(times=times))
    assert result.shape == (len(times), 1, 3)
    assert np.isfinite(result).all()


if __name__ == "__main__":
    test_simulator_runs_at_70_mhz()
    print("test_extended_freq_range: passed.")
