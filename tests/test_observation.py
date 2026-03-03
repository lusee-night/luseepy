import numpy as np
from lusee import Observation
from lusee.CalibratorTrack import CalibratorTrack


def _make_track(n=3):
    t = np.arange(n, dtype=float)
    return CalibratorTrack(t, t, t, t, t, t)


def test_default_no_calibrator_tracks():
    O = Observation("2025-02-01 to 2025-02-02")
    assert O.calibrator_tracks == []


def test_single_track_stored():
    ct = _make_track()
    O = Observation("2025-02-01 to 2025-02-02", calibrator_tracks=[ct])
    assert len(O.calibrator_tracks) == 1
    assert O.calibrator_tracks[0] is ct


def test_multiple_tracks():
    ct1 = _make_track(3)
    ct2 = _make_track(5)
    O = Observation("2025-02-01 to 2025-02-02", calibrator_tracks=[ct1, ct2])
    assert len(O.calibrator_tracks) == 2
    assert O.calibrator_tracks[0] is ct1
    assert O.calibrator_tracks[1] is ct2


def test_input_list_not_aliased():
    ct = _make_track()
    source = [ct]
    O = Observation("2025-02-01 to 2025-02-02", calibrator_tracks=source)
    source.append(_make_track())
    assert len(O.calibrator_tracks) == 1
