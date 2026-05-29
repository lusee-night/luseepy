"""Tests for the public telemetry proxy.

The proxy delegates to the private ``lusee_telemetry`` package when it
is importable. The DCB / encoder layout, channel names, and conversion
formulas are not part of the public surface and therefore not tested
here -- those tests live with the private package.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from lusee.ingest import telemetry as telem


def _decoder_available() -> bool:
    return telem.has_decoder()


def test_no_decoder_returns_empty(monkeypatch):
    """With no decoder available, every proxy returns an empty / None default."""
    # Force-clear the cached decoder so we exercise the no-decoder path.
    monkeypatch.setattr(telem, "_decoder", None, raising=False)
    monkeypatch.setattr(telem, "_resolved", True, raising=False)
    # Also make the sys-path import fail by prepending a junk path that
    # isn't lusee_telemetry, then preventing future re-resolves.
    assert telem.telemetry_apids() == ()
    fpga, encoder = telem.parse_b01_packets([])
    assert fpga == {} and encoder == {}
    assert telem.find_legacy_sidecar("/tmp/no/such/dir") is None
    assert telem.parse_legacy_sidecar("/tmp/no/such/file") == {}
    assert telem.field_groups() == {}


def test_slice_arrays_by_window_inclusive_lower_exclusive_upper():
    arrays = {
        "mission_seconds": np.array([0.0, 5.0, 10.0, 15.0]),
        "x": np.array([10, 20, 30, 40]),
    }
    sliced = telem.slice_arrays_by_window(
        arrays, window_lower_raw_seconds=5.0, window_upper_raw_seconds=15.0,
    )
    np.testing.assert_array_equal(sliced["mission_seconds"], [5.0, 10.0])
    np.testing.assert_array_equal(sliced["x"], [20, 30])


def test_slice_arrays_by_window_open_lower():
    arrays = {
        "mission_seconds": np.array([0.0, 5.0, 10.0]),
        "x": np.array([1, 2, 3]),
    }
    sliced = telem.slice_arrays_by_window(
        arrays, window_lower_raw_seconds=None, window_upper_raw_seconds=10.0,
    )
    np.testing.assert_array_equal(sliced["mission_seconds"], [0.0, 5.0])


def test_slice_arrays_by_window_full_pass_through():
    arrays = {"mission_seconds": np.array([1.0, 2.0])}
    out = telem.slice_arrays_by_window(arrays, window_lower_raw_seconds=None,
                                        window_upper_raw_seconds=None)
    # When both ends are None the input is returned unchanged (same dict).
    assert out is arrays


@pytest.mark.skipif(
    not _decoder_available(),
    reason="lusee_telemetry decoder not on sys.path",
)
def test_telemetry_apids_with_decoder():
    apids = telem.telemetry_apids()
    assert isinstance(apids, tuple)
    assert all(isinstance(a, int) for a in apids)
    assert len(apids) > 0


@pytest.mark.skipif(
    not _decoder_available(),
    reason="lusee_telemetry decoder not on sys.path",
)
def test_field_groups_with_decoder():
    groups = telem.field_groups()
    # Optional in the decoder API; if present, must be dict[str, tuple[str, ...]].
    if groups:
        assert all(isinstance(k, str) for k in groups)
        assert all(isinstance(v, (tuple, list)) for v in groups.values())
