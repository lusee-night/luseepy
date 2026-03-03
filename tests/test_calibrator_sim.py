"""
Unit tests for CalibratorSimulator.

Uses BeamGauss (no FITS data required) and ObservedSatellite (default Satellite).
"""
import numpy as np
import pytest

import lusee
from lusee import (
    Observation, CalibratorTrack, CalibratorSimulator,
    BeamGauss, Satellite, ObservedSatellite,
)

TONE_FREQS = np.array([10.0, 12.0, 14.0])        # MHz, NFreq=3
TONE_AMP   = np.ones(len(TONE_FREQS))            # one amplitude per frequency

# Angles match N/E/S/W layout used in the calibrator example config
_BEAM_ANGLES = [0, 90, 180, 270]
_SIGMA       = 20.0


@pytest.fixture(scope="module")
def beams():
    return [
        BeamGauss(dec_deg=0, sigma_deg=_SIGMA, phi_deg=90 - angle, id=name)
        for name, angle in zip(["N", "E", "S", "W"], _BEAM_ANGLES)
    ]


@pytest.fixture(scope="module")
def obs_and_tracks():
    # 1-hour steps over lunar day 2501 — fast enough for unit tests while
    # still catching multiple satellite overpasses (~11-hour period satellite).
    obs = Observation(2501, deltaT_sec=3600)
    sat = Satellite()
    os_ = ObservedSatellite(obs, sat)
    passes = os_.get_transit_indices()

    tracks = []
    for si, ei in passes:
        n = ei - si
        if n == 0:
            continue
        tracks.append(CalibratorTrack(
            times         = obs.times[si:ei],
            alt           = os_.alt[si:ei],
            az            = os_.az[si:ei],
            polarization  = np.zeros(n),
            tone_freqs    = TONE_FREQS,
            tone_amplitude= TONE_AMP,
        ))
    obs.calibrator_tracks = tracks
    return obs, tracks


@pytest.fixture(scope="module")
def simulator(obs_and_tracks, beams):
    obs, _ = obs_and_tracks
    sim = CalibratorSimulator(obs, beams)
    sim.simulate()
    return sim


def test_n_passes(obs_and_tracks, simulator):
    _, tracks = obs_and_tracks
    assert len(simulator.result) == len(tracks)


def test_result_shapes(obs_and_tracks, simulator):
    _, tracks = obs_and_tracks
    NBeam = 4
    NFreq = len(TONE_FREQS)
    for p, (res, track) in enumerate(zip(simulator.result, tracks)):
        assert res.shape == (len(track), NBeam, NFreq), \
            f"Pass {p}: expected shape ({len(track)}, {NBeam}, {NFreq}), got {res.shape}"
        assert res.dtype == complex, f"Pass {p}: expected complex dtype, got {res.dtype}"


def test_result_is_list(simulator):
    assert isinstance(simulator.result, list)
    assert len(simulator.result) > 0


def test_signal_nonzero(simulator):
    # At least one element per pass must be non-zero
    for p, res in enumerate(simulator.result):
        assert np.any(np.abs(res) > 0), f"Pass {p}: all results are zero"


def test_polarization_zero_gives_real_signal(obs_and_tracks, beams):
    """
    With polarization=0 and BeamGauss (Ephi=0, Etheta real),
    result = amp * Etheta * cos(0) = amp * Etheta, which is real.
    """
    obs, tracks = obs_and_tracks
    if not tracks:
        pytest.skip("No satellite passes found")

    track0 = tracks[0]
    # Confirm polarization is already 0 in the fixture
    assert np.all(track0.polarization == 0)

    # Simulate just the first pass
    tmp_obs = Observation.__new__(Observation)
    tmp_obs.__dict__.update(obs.__dict__)
    tmp_obs.calibrator_tracks = [track0]

    sim = CalibratorSimulator(tmp_obs, beams)
    sim.simulate()
    res = sim.result[0]

    # BeamGauss Etheta is real-valued; with pol=0, result should be real
    np.testing.assert_allclose(
        res.imag, 0, atol=1e-10,
        err_msg="With pol=0 and BeamGauss (Ephi=0, Etheta real), imaginary part should be zero"
    )
