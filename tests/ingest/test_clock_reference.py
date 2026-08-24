"""Focused tests for the version-1 external clock-reference contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from lusee.ingest.clock_reference import (
    ClockReferenceFormatError,
    ClockReferenceUnavailableError,
    ClockSource,
    UnsupportedClockSourceError,
    load_clock_reference_set,
)


def clock_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": "2024-03-04T12:34:56.125",
        "time_scale": "utc",
        "source": "synthetic timing fixture",
        "assumed": False,
        "clocks": {
            "spectrometer": {"clock_reference_raw_seconds": 1234.5},
            "dcb": {"clock_reference_raw_seconds": -8.25},
        },
    }


def write_payload(path: Path, payload: dict[str, object]) -> bytes:
    raw = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_load_and_convert_distinct_nonzero_clock_anchors(tmp_path: Path):
    path = tmp_path / "landing.json"
    raw = write_payload(path, clock_payload())

    references = load_clock_reference_set(path)

    assert references.source_sha256 == hashlib.sha256(raw).hexdigest()
    assert references.time_scale == "utc"
    assert references.assumed is False
    assert references.require_reference("spectrometer").clock_reference_raw_seconds == 1234.5
    assert references.require_reference("dcb").clock_reference_raw_seconds == -8.25

    spectrometer = references.to_time(
        np.array([1233.5, 1234.5, 1236.5]),
        clock_source=ClockSource.SPECTROMETER,
    )
    np.testing.assert_allclose(
        (spectrometer - spectrometer[1]).sec,
        [-1.0, 0.0, 2.0],
        rtol=0.0,
        atol=1e-9,
    )
    dcb = references.to_time(-7.25, clock_source="dcb")
    assert (dcb - spectrometer[1]).sec == pytest.approx(1.0, abs=1e-9)
    assert dcb.scale == "utc"
    assert references.to_mjd(-7.25, clock_source="dcb") == pytest.approx(dcb.mjd)


def test_unmapped_clocks_never_borrow_another_anchor(tmp_path: Path):
    payload = clock_payload()
    payload["clocks"] = {
        "spectrometer": {"clock_reference_raw_seconds": 1234.5},
    }
    path = tmp_path / "landing.json"
    write_payload(path, payload)
    references = load_clock_reference_set(path)

    assert references.reference_for("dcb") is None
    assert references.reference_for("adc") is None
    with pytest.raises(ClockReferenceUnavailableError, match="dcb"):
        references.to_time(0.0, clock_source="dcb")
    with pytest.raises(ClockReferenceUnavailableError, match="adc"):
        references.to_time(0.0, clock_source="adc")
    with pytest.raises(UnsupportedClockSourceError, match="unknown"):
        references.reference_for("unknown")


def test_semantic_canonicalization_is_independent_of_source_bytes(tmp_path: Path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_raw = write_payload(first_path, clock_payload())
    second_raw = json.dumps(
        clock_payload(), sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    second_path.write_bytes(second_raw)

    first = load_clock_reference_set(first_path)
    second = load_clock_reference_set(second_path)

    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert first.source_sha256 != second.source_sha256
    assert first.source_sha256 == hashlib.sha256(first_raw).hexdigest()
    assert hash(first) == hash(first)
    with pytest.raises(FrozenInstanceError):
        first.source = "changed"
    with pytest.raises(FrozenInstanceError):
        first.clocks[0].clock_reference_raw_seconds = 0.0


def test_negative_zero_anchor_is_canonicalized(tmp_path: Path):
    payload = clock_payload()
    payload["clocks"] = {
        "spectrometer": {"clock_reference_raw_seconds": -0.0},
    }
    path = tmp_path / "landing.json"
    write_payload(path, payload)

    reference = load_clock_reference_set(path).require_reference("spectrometer")

    assert reference.clock_reference_raw_seconds == 0.0
    assert np.signbit(reference.clock_reference_raw_seconds) == np.bool_(False)


@pytest.mark.parametrize("scale", ["utc", "tai", "tt", "tdb", "tcg", "tcb", "ut1"])
def test_supported_absolute_time_scales_are_independent_of_legacy_v3(
    tmp_path: Path,
    scale: str,
):
    payload = clock_payload()
    payload["time_scale"] = scale
    path = tmp_path / "landing.json"
    write_payload(path, payload)

    assert load_clock_reference_set(path).time_scale == scale


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.pop("source"), "missing required"),
        (lambda p: p.update(extra=True), "unknown field"),
        (lambda p: p.update(format_version=True), "format_version"),
        (lambda p: p.update(format_version=2), "format_version"),
        (lambda p: p.update(reference_event="launch"), "reference_event"),
        (lambda p: p.update(time_scale="UTC"), "time_scale"),
        (lambda p: p.update(time_scale="unknown"), "time_scale"),
        (lambda p: p.update(clock_reference_isot="2024-03-04 12:34:56"), "clock_reference_isot"),
        (lambda p: p.update(clock_reference_isot="2024-03-04T12:34:56Z"), "clock_reference_isot"),
        (lambda p: p.update(clock_reference_isot="2024-02-31T12:34:56"), "clock_reference_isot"),
        (lambda p: p.update(source="   "), "source"),
        (lambda p: p.update(assumed=1), "assumed"),
        (lambda p: p.update(clocks=[]), "clocks"),
        (lambda p: p.update(clocks={"other": {"clock_reference_raw_seconds": 0}}), "unsupported clock"),
        (lambda p: p.update(clocks={"adc": {"clock_reference_raw_seconds": 0}}), "cannot be anchored"),
        (lambda p: p.update(clocks={"spectrometer": {"clock_reference_raw_seconds": True}}), "numeric"),
        (lambda p: p.update(clocks={"spectrometer": {"clock_reference_raw_seconds": "0"}}), "numeric"),
        (lambda p: p.update(clocks={"spectrometer": {"clock_reference_raw_seconds": float("inf")}}), "invalid JSON number"),
        (lambda p: p.update(clocks={"spectrometer": {"clock_reference_raw_seconds": 10**400}}), "finite"),
        (lambda p: p.update(clocks={"spectrometer": {"clock_reference_raw_seconds": 0, "units": "s"}}), "unknown field"),
    ],
)
def test_rejects_malformed_contract(
    tmp_path: Path,
    mutate,
    message: str,
):
    payload = clock_payload()
    mutate(payload)
    path = tmp_path / "landing.json"
    write_payload(path, payload)

    with pytest.raises(ClockReferenceFormatError, match=message):
        load_clock_reference_set(path)


def test_rejects_duplicate_json_keys(tmp_path: Path):
    path = tmp_path / "landing.json"
    path.write_text(
        '{"format_version":1,"format_version":1}',
        encoding="ascii",
    )
    with pytest.raises(ClockReferenceFormatError, match="duplicate key"):
        load_clock_reference_set(path)


def test_missing_path_preserves_io_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_clock_reference_set(tmp_path / "missing.json")


def test_raw_conversion_rejects_boolean_and_nonfinite_values(tmp_path: Path):
    path = tmp_path / "landing.json"
    write_payload(path, clock_payload())
    references = load_clock_reference_set(path)

    with pytest.raises(TypeError, match="boolean"):
        references.to_time(True, clock_source="spectrometer")
    with pytest.raises(TypeError, match="numeric"):
        references.to_time("0", clock_source="spectrometer")
    with pytest.raises(ValueError, match="finite"):
        references.to_time([0.0, np.nan], clock_source="spectrometer")
    with pytest.raises(ValueError, match="unrepresentable absolute time"):
        references.to_time(1e308, clock_source="spectrometer")
