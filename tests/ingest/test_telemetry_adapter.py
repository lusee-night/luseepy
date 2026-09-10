from __future__ import annotations

import numpy as np
import pytest

from lusee.ingest import telemetry
from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.reassembly import LogicalPacket
from lusee.ingest.session import Session, assign_telemetry_to_sessions


FIELD_NAMES = tuple(f"field_{index}" for index in range(57))
UNITS = tuple("V" if index % 2 else "degC" for index in range(57))


def fixed_result(row_count: int = 2) -> dict[str, object]:
    raw_counts = np.arange(row_count * 57, dtype=np.uint16).reshape(
        row_count, 57
    )
    values = raw_counts.astype(np.float64)
    valid = np.ones((row_count, 57), dtype=np.bool_)
    if row_count:
        values[-1, -1] = np.nan
        valid[-1, -1] = False
    return {
        "field_names": FIELD_NAMES,
        "units": UNITS,
        "source_indices": np.arange(row_count, dtype=np.int64) * 2,
        "mission_seconds": np.arange(100, 100 + row_count, dtype=np.uint32),
        "lusee_subsecs": np.arange(row_count, dtype=np.uint16),
        "raw_counts": raw_counts,
        "values": values,
        "valid": valid,
    }


def make_data(
    mission_seconds=(1000,),
    lusee_subsecs=(0,),
    *,
    source_kind="b01_0x314",
) -> telemetry.TelemetryData:
    row_count = len(mission_seconds)
    result = fixed_result(row_count)
    result["mission_seconds"] = np.asarray(
        mission_seconds, dtype=np.uint32
    )
    result["lusee_subsecs"] = np.asarray(lusee_subsecs, dtype=np.uint16)
    return telemetry.telemetry_from_result(result, source_kind=source_kind)


def logical_packet(appid: int, *, single_packet: bool = True) -> LogicalPacket:
    return LogicalPacket(
        appid=appid,
        start_seq=1,
        seq=2,
        blob=b"payload",
        single_packet=single_packet,
    )


def clock_references(*sources: ClockSource) -> ClockReferenceSet:
    anchors = {
        ClockSource.SPECTROMETER: 100.0,
        ClockSource.DCB: 1000.0,
    }
    return ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2026-01-01T00:00:00",
        time_scale="utc",
        clocks=tuple(ClockReference(source, anchors[source]) for source in sources),
        source="synthetic test",
        assumed=False,
        source_sha256="a" * 64,
    )


def test_fixed_result_crosses_as_one_57_column_table():
    data = telemetry.telemetry_from_result(
        fixed_result(),
        source_kind="b01_0x314",
    )

    assert data.source_kind == "b01_0x314"
    assert data.field_names == FIELD_NAMES
    assert data.units == UNITS
    assert data.source_indices.dtype == np.dtype(np.int64)
    assert data.mission_seconds.dtype == np.dtype(np.uint32)
    assert data.lusee_subsecs.dtype == np.dtype(np.uint16)
    assert data.mjd_times.dtype == np.dtype(np.float64)
    assert data.raw_counts.dtype == np.dtype(np.uint16)
    assert data.values.dtype == np.dtype(np.float64)
    assert data.valid.dtype == np.dtype(np.bool_)
    assert data.raw_counts.shape == data.values.shape == data.valid.shape == (2, 57)
    assert data.raw_seconds.tolist() == [100.0, 101.0 + 1.0 / 65536.0]
    assert np.isnan(data.values[-1, -1])
    assert not data.valid[-1, -1]
    assert np.isnan(data.mjd_times).all()


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("source_indices", np.array([0, 1], dtype=np.int32), "dtype int64"),
        ("raw_counts", np.zeros((2, 56), dtype=np.uint16), "invalid shape"),
        ("field_names", FIELD_NAMES[:-1], "57 unique"),
        ("values", np.zeros((2, 57), dtype=np.float32), "dtype float64"),
    ],
)
def test_malformed_fixed_results_are_rejected(field, replacement, match):
    result = fixed_result()
    result[field] = replacement

    with pytest.raises((TypeError, ValueError), match=match):
        telemetry.telemetry_from_result(result, source_kind="b01_0x314")


def test_validity_is_exactly_finite_and_invalid_is_nan():
    result = fixed_result(1)
    result["values"][0, 0] = np.inf
    result["valid"][0, 0] = False

    with pytest.raises(ValueError, match="invalid telemetry values must be NaN"):
        telemetry.telemetry_from_result(result, source_kind="b01_0x314")


class Decoder:
    def __init__(self, result=None, error=None):
        self.result = fixed_result() if result is None else result
        self.error = error
        self.packet_calls = []
        self.sidecar_calls = []

    def decode_b01_packets(self, packets):
        self.packet_calls.append(tuple(packets))
        if self.error is not None:
            raise self.error
        return self.result

    def decode_legacy_sidecar(self, path):
        self.sidecar_calls.append(path)
        if self.error is not None:
            raise self.error
        return self.result


def test_b01_filters_before_decoder_and_accepts_multiframe_logical_packet():
    decoder = Decoder()
    selected = logical_packet(0x314, single_packet=False)

    data = telemetry.decode_b01_packets(
        [logical_packet(0x325), logical_packet(0x123), selected],
        decoder=decoder,
    )

    assert data is not None
    assert data.source_kind == "b01_0x314"
    assert decoder.packet_calls == [(selected,)]


@pytest.mark.parametrize("appids", [(), (0x325,), (0x123, 0x325)])
def test_unrelated_or_0x325_only_input_never_invokes_decoder(appids):
    decoder = Decoder(error=AssertionError("must not be called"))

    assert telemetry.decode_b01_packets(
        [logical_packet(appid) for appid in appids],
        decoder=decoder,
    ) is None
    assert decoder.packet_calls == []


@pytest.mark.parametrize("source", ["b01", "sidecar"])
@pytest.mark.parametrize(
    "decoder",
    [None, Decoder(error=RuntimeError("decoder failed"))],
)
def test_missing_or_raising_decoder_warns_once_and_returns_none(
    tmp_path,
    monkeypatch,
    source,
    decoder,
):
    if decoder is None:
        monkeypatch.setattr(telemetry, "private_decoder", None)
        monkeypatch.setattr(
            telemetry,
            "decoder_import_error",
            ModuleNotFoundError("lusee_telemetry"),
        )

    with pytest.warns(UserWarning, match="telemetry skipped") as caught:
        if source == "b01":
            result = telemetry.decode_b01_packets(
                [logical_packet(0x314)],
                decoder=decoder,
            )
        else:
            result = telemetry.decode_legacy_sidecar(
                tmp_path / "DCB_telemetry.json",
                decoder=decoder,
            )

    assert result is None
    assert len(caught) == 1


@pytest.mark.parametrize("source", ["b01", "sidecar"])
def test_malformed_decoder_result_warns_once_and_is_skipped(tmp_path, source):
    decoder = Decoder(result={"wrong": "shape"})

    with pytest.warns(UserWarning, match="wrong fields") as caught:
        if source == "b01":
            result = telemetry.decode_b01_packets(
                [logical_packet(0x314)], decoder=decoder
            )
        else:
            result = telemetry.decode_legacy_sidecar(
                tmp_path / "DCB_telemetry.json", decoder=decoder
            )

    assert result is None
    assert len(caught) == 1


@pytest.mark.parametrize("source", ["b01", "sidecar"])
@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("field_names", ("bad\x00name", *FIELD_NAMES[1:]), "NUL"),
        ("units", ("\ud800", *UNITS[1:]), "valid UTF-8"),
    ],
)
def test_invalid_decoder_strings_warn_once_and_are_skipped(
    tmp_path,
    source,
    field,
    replacement,
    match,
):
    malformed = fixed_result()
    malformed[field] = replacement
    decoder = Decoder(result=malformed)

    with pytest.warns(UserWarning, match=match) as caught:
        if source == "b01":
            result = telemetry.decode_b01_packets(
                [logical_packet(0x314)], decoder=decoder
            )
        else:
            result = telemetry.decode_legacy_sidecar(
                tmp_path / "DCB_telemetry.json", decoder=decoder
            )

    assert result is None
    assert len(caught) == 1


def test_legacy_sidecar_uses_same_fixed_table(tmp_path):
    path = tmp_path / "DCB_telemetry.json"
    path.write_bytes(b"")
    decoder = Decoder(result=fixed_result(0))

    data = telemetry.decode_legacy_sidecar(path, decoder=decoder)

    assert data is not None
    assert data.source_kind == "legacy_binary_sidecar"
    assert data.raw_counts.shape == (0, 57)
    assert decoder.sidecar_calls == [path]


def test_b01_rows_are_mapped_and_sliced_disjointly():
    sessions = [
        Session(ordinal=0, start_raw_seconds=100.0),
        Session(ordinal=1, start_raw_seconds=200.0),
    ]
    source = make_data(
        mission_seconds=(999, 1000, 1100),
        lusee_subsecs=(32768, 0, 32768),
    )

    with pytest.warns(UserWarning, match="before the first science session"):
        mapped = assign_telemetry_to_sessions(
            sessions,
            source,
            clock_reference_set=clock_references(
                ClockSource.SPECTROMETER,
                ClockSource.DCB,
            ),
        )

    assert mapped is not None
    assert mapped.source_indices.tolist() == [0, 2, 4]
    assert sessions[0].telemetry.source_indices.tolist() == [2]
    assert sessions[1].telemetry.source_indices.tolist() == [4]
    assert set(sessions[0].telemetry.source_indices).isdisjoint(
        sessions[1].telemetry.source_indices
    )
    assert np.isfinite(mapped.mjd_times).all()


@pytest.mark.parametrize(
    "sessions,references,warning",
    [
        (
            [Session(ordinal=0, start_raw_seconds=100.0)],
            (ClockSource.SPECTROMETER,),
            "clock reference",
        ),
        (
            [Session(ordinal=0, start_raw_seconds=None)],
            (ClockSource.SPECTROMETER, ClockSource.DCB),
            "session start is missing",
        ),
    ],
)
def test_unassignable_b01_rows_are_omitted(sessions, references, warning):
    with pytest.warns(UserWarning, match=warning):
        result = assign_telemetry_to_sessions(
            sessions,
            make_data(),
            clock_reference_set=clock_references(*references),
        )

    assert result is None
    assert all(session.telemetry is None for session in sessions)


def test_zero_row_b01_table_is_attached_to_every_session():
    sessions = [Session(ordinal=0), Session(ordinal=1)]

    source = assign_telemetry_to_sessions(
        sessions,
        make_data(mission_seconds=(), lusee_subsecs=()),
        clock_reference_set=clock_references(),
    )

    assert source is not None
    assert source.row_count == 0
    assert [session.telemetry.row_count for session in sessions] == [0, 0]


def test_dcb_time_mapping_never_borrows_another_clock():
    data = make_data()

    unmapped = telemetry.map_dcb_absolute_time(
        data,
        clock_reference_set=clock_references(ClockSource.SPECTROMETER),
    )
    assert np.isnan(unmapped.mjd_times).all()

    bad = data.with_mjd_times(np.array([60000.0], dtype=np.float64))
    with pytest.raises(ValueError, match="without a DCB reference"):
        telemetry.map_dcb_absolute_time(
            bad,
            clock_reference_set=clock_references(ClockSource.SPECTROMETER),
        )
