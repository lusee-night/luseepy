import numpy as np
import pytest

import lusee
from lusee.frequencies import canonical_frequencies, frequency_indices_from_values


def test_crosimulator_runs_and_returns_expected_shape():
    if lusee.CroSimulator is None:
        pytest.skip("croissant/s2fft not installed")

    nside = 8
    lmax = 3 * nside - 1
    freq = canonical_frequencies(frequency_indices_from_values([5.0, 10.0, 15.0]))

    # Deterministic full-sky galactic map at each frequency.
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
    result = sim.simulate(times=obs.times)

    assert result.shape == (len(obs.times), 1, len(freq))
    assert np.all(np.isfinite(result))
    assert np.any(np.abs(result) > 0.0)
