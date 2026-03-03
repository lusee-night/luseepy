import pytest
import numpy as np
from lusee.CalibratorTrack import CalibratorTrack


N = 5
TIMES = np.arange(N, dtype=float)
ALT = np.linspace(0.1, 0.5, N)
AZ = np.linspace(1.0, 2.0, N)
POL = np.zeros(N)
FREQS = np.full(N, 10.0)
AMP = np.ones(N) * 100.0


@pytest.fixture
def ct():
    return CalibratorTrack(TIMES, ALT, AZ, POL, FREQS, AMP)


def test_len(ct):
    assert len(ct) == N


def test_array_shapes(ct):
    for attr in ("times", "alt", "az", "polarization", "tone_freqs", "tone_amplitude"):
        assert getattr(ct, attr).shape == (N,), f"{attr} has wrong shape"


def test_values_stored(ct):
    assert np.allclose(ct.alt, ALT)
    assert np.allclose(ct.az, AZ)
    assert np.allclose(ct.polarization, POL)
    assert np.allclose(ct.tone_freqs, FREQS)
    assert np.allclose(ct.tone_amplitude, AMP)


@pytest.mark.parametrize("bad_field", ["alt", "az", "polarization", "tone_freqs", "tone_amplitude"])
def test_mismatch_raises(bad_field):
    kwargs = dict(
        times=TIMES,
        alt=ALT,
        az=AZ,
        polarization=POL,
        tone_freqs=FREQS,
        tone_amplitude=AMP,
    )
    # Replace the target field with an array of wrong length
    kwargs[bad_field] = np.zeros(N + 1)
    with pytest.raises(ValueError, match=bad_field):
        CalibratorTrack(**kwargs)
