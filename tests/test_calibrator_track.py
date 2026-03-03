import pytest
import numpy as np
from lusee.CalibratorTrack import CalibratorTrack


N     = 5       # NTime
NFREQ = 3       # NFreq — deliberately different from N

TIMES = np.arange(N, dtype=float)
ALT   = np.linspace(0.1, 0.5, N)
AZ    = np.linspace(1.0, 2.0, N)
POL   = np.zeros(N)
FREQS = np.array([10.0, 12.0, 14.0])   # length NFREQ
AMP   = np.ones(NFREQ) * 100.0         # length NFREQ


@pytest.fixture
def ct():
    return CalibratorTrack(TIMES, ALT, AZ, POL, FREQS, AMP)


def test_len(ct):
    assert len(ct) == N


def test_array_shapes(ct):
    for attr in ("times", "alt", "az", "polarization"):
        assert getattr(ct, attr).shape == (N,), f"{attr} has wrong shape"
    for attr in ("tone_freqs", "tone_amplitude"):
        assert getattr(ct, attr).shape == (NFREQ,), f"{attr} has wrong shape"


def test_values_stored(ct):
    assert np.allclose(ct.alt, ALT)
    assert np.allclose(ct.az, AZ)
    assert np.allclose(ct.polarization, POL)
    assert np.allclose(ct.tone_freqs, FREQS)
    assert np.allclose(ct.tone_amplitude, AMP)


@pytest.mark.parametrize("bad_field", ["alt", "az", "polarization"])
def test_ntime_mismatch_raises(bad_field):
    kwargs = dict(
        times=TIMES, alt=ALT, az=AZ, polarization=POL,
        tone_freqs=FREQS, tone_amplitude=AMP,
    )
    kwargs[bad_field] = np.zeros(N + 1)
    with pytest.raises(ValueError, match=bad_field):
        CalibratorTrack(**kwargs)


def test_tone_amplitude_mismatch_raises():
    # tone_amplitude must match len(tone_freqs), not len(times)
    with pytest.raises(ValueError, match="tone_amplitude"):
        CalibratorTrack(TIMES, ALT, AZ, POL, FREQS, np.zeros(NFREQ + 1))


def test_tone_freqs_independent_length():
    # tone_freqs (and matching tone_amplitude) can be any length, independent of NTime
    for nfreq in [1, 3, N - 1, N + 7]:
        freqs = np.linspace(10.0, 30.0, nfreq)
        amp   = np.ones(nfreq)
        ct = CalibratorTrack(TIMES, ALT, AZ, POL, freqs, amp)
        assert len(ct.tone_freqs)     == nfreq
        assert len(ct.tone_amplitude) == nfreq
