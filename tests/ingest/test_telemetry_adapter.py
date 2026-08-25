from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from lusee.ingest import telemetry
from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.issues import IssueCollector
from lusee.ingest.reassembly import LogicalPacket
from lusee.ingest.session import Session, assign_telemetry_to_sessions

FIELD_NAMES = ("temperature", "raw_adc")


def packet(appid=0x314, *, single_packet=True, file_index=None):
    return LogicalPacket(
        appid=appid,
        start_seq=7,
        seq=8,
        blob=b"payload",
        single_packet=single_packet,
        bank="b01",
        file_index=file_index,
    )


def block(source_kind, *, input_indices=(), invalid_second=False):
    input_indices = np.asarray(input_indices, dtype=np.int64)
    n_rows = input_indices.size
    mission_seconds = np.arange(100, 100 + n_rows, dtype=np.uint32)
    subsecs = np.arange(n_rows, dtype=np.uint16)
    raw_counts = np.zeros((n_rows, len(FIELD_NAMES)), dtype=np.uint16)
    values = np.zeros((n_rows, len(FIELD_NAMES)), dtype=np.float64)
    valid = np.ones((n_rows, len(FIELD_NAMES)), dtype=np.bool_)
    if invalid_second and n_rows:
        values[0, 1] = np.nan
        valid[0, 1] = False
    return {
        "source_kind": source_kind,
        "field_names": FIELD_NAMES,
        "input_indices": input_indices,
        "mission_seconds": mission_seconds,
        "lusee_subsecs": subsecs,
        "raw_counts": raw_counts,
        "values": values,
        "valid": valid,
    }


def empty_encoder_block():
    return {
        "source_kind": "b01_0x325",
        "field_names": (),
        "input_indices": np.empty(0, dtype=np.int64),
        "mission_seconds": np.empty(0, dtype=np.uint32),
        "lusee_subsecs": np.empty(0, dtype=np.uint16),
        "raw_counts": np.empty((0, 0), dtype=np.uint16),
        "values": np.empty((0, 0), dtype=np.float64),
        "valid": np.empty((0, 0), dtype=np.bool_),
    }


class FakeDetailedTelemetryV1:
    def __init__(self):
        self.b01_calls = []
        self.sidecar_calls = []
        self.mutate_result = None

    def decoder_info(self):
        return {
            "api_version": 1,
            "decoder_name": "generic-test-decoder",
            "decoder_version": "test-1",
            "claimed_appids": (0x314, 0x325),
        }

    def field_metadata(self):
        return {
            "temperature": {
                "unit": "K",
                "kind": "continuous",
                "interpolation": "linear",
                "display_group": "thermal",
            },
            "raw_adc": {
                "unit": "count",
                "kind": "uncalibrated_raw_count",
                "interpolation": "none",
                "display_group": None,
            },
        }

    def decode_b01_packets(self, packets):
        self.b01_calls.append(packets)
        fpga_indices = tuple(
            index for index, item in enumerate(packets) if item.appid == 0x314
        )
        encoder_indices = tuple(
            index for index, item in enumerate(packets) if item.appid == 0x325
        )
        unclaimed_indices = tuple(
            index
            for index, item in enumerate(packets)
            if item.appid not in (0x314, 0x325)
        )
        result = {
            "fpga": block(
                "b01_0x314",
                input_indices=fpga_indices,
                invalid_second=bool(fpga_indices),
            ),
            "encoder": empty_encoder_block(),
            "issues": (
                [
                    {
                        "code": "telemetry_decoder.encoder_layout_unvalidated",
                        "severity": "warning",
                        "action": "rejected",
                        "message": "encoder layout is not validated",
                        "input_index": encoder_indices[0],
                        "appid": 0x325,
                        "sequence_count": packets[encoder_indices[0]].seq,
                        "field": None,
                        "details": {},
                    }
                ]
                if encoder_indices
                else []
            ),
            "counts": {
                "input_packet_count": len(packets),
                "claimed_packet_count": len(fpga_indices) + len(encoder_indices),
                "unclaimed_packet_count": len(unclaimed_indices),
                "fpga_input_packet_count": len(fpga_indices),
                "fpga_output_record_count": len(fpga_indices),
                "fpga_dropped_packet_count": 0,
                "encoder_input_packet_count": len(encoder_indices),
                "encoder_output_record_count": 0,
                "encoder_rejected_packet_count": len(encoder_indices),
                "claimed_appid_counts": {
                    0x314: len(fpga_indices),
                    0x325: len(encoder_indices),
                },
                "unclaimed_appid_counts": {
                    appid: sum(item.appid == appid for item in packets)
                    for appid in sorted({packets[i].appid for i in unclaimed_indices})
                },
            },
        }
        if self.mutate_result is not None:
            self.mutate_result(result)
        return result

    def decode_legacy_sidecar(self, path):
        self.sidecar_calls.append(path)
        size = path.stat().st_size
        n_rows = int(size > 0)
        trailing = int(size > 0)
        result = {
            "fpga": block(
                "legacy_binary_sidecar",
                input_indices=range(n_rows),
                invalid_second=bool(n_rows),
            ),
            "issues": (
                [
                    {
                        "code": "telemetry_decoder.trailing_partial_record",
                        "severity": "warning",
                        "action": "dropped",
                        "message": "trailing bytes were dropped",
                        "input_index": None,
                        "appid": None,
                        "sequence_count": None,
                        "field": None,
                        "details": {"trailing_byte_count": trailing},
                    }
                ]
                if trailing
                else []
            ),
            "counts": {
                "input_byte_count": size,
                "complete_record_count": n_rows,
                "trailing_byte_count": trailing,
                "output_record_count": n_rows,
                "dropped_record_count": 0,
            },
        }
        if self.mutate_result is not None:
            self.mutate_result(result)
        return result


def test_absent_input_does_not_resolve_decoder(monkeypatch):
    calls = []

    def fail_import(name):
        calls.append(name)
        raise AssertionError("decoder import must not be attempted")

    monkeypatch.setattr(telemetry.importlib, "import_module", fail_import)
    result = telemetry.decode_b01_packets(None)

    assert result.input_state is telemetry.TelemetryInputState.ABSENT
    assert result.decoder_status is telemetry.TelemetryDecoderStatus.NOT_NEEDED
    assert result.coverage is telemetry.TelemetryCoverage.ABSENT
    assert calls == []


@pytest.mark.parametrize(
    ("missing_name", "expected"),
    [
        ("lusee_telemetry", telemetry.TelemetryDecoderStatus.UNAVAILABLE),
        ("decoder_internal_dependency", telemetry.TelemetryDecoderStatus.BROKEN),
    ],
)
def test_import_failure_states_are_distinct(monkeypatch, missing_name, expected):
    def fail_import(name):
        raise ModuleNotFoundError("synthetic import failure", name=missing_name)

    monkeypatch.setattr(telemetry.importlib, "import_module", fail_import)
    result = telemetry.decode_b01_packets([packet()])

    assert result.input_state is telemetry.TelemetryInputState.PRESENT
    assert result.decoder_status is expected
    assert result.coverage.value == expected.value
    assert result.issues[0].code == f"telemetry_adapter.decoder_{expected.value}"


def test_incompatible_decoder_is_not_reported_as_absent():
    class MissingDetailedMethod(FakeDetailedTelemetryV1):
        decode_b01_packets = None

    result = telemetry.decode_b01_packets(
        [packet()], decoder=MissingDetailedMethod()
    )

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert result.coverage is telemetry.TelemetryCoverage.INCOMPATIBLE
    assert result.decoder_info is not None
    assert result.field_metadata
    assert result.issues[0].code == "telemetry_adapter.decoder_incompatible"


def test_metadata_failure_preserves_valid_decoder_identity():
    class BrokenMetadata(FakeDetailedTelemetryV1):
        def field_metadata(self):
            raise RuntimeError("synthetic metadata failure")

    result = telemetry.decode_b01_packets([packet()], decoder=BrokenMetadata())

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.BROKEN
    assert result.decoder_info is not None
    assert result.field_metadata == ()


def test_valid_detailed_result_preserves_exact_arrays_and_multiframe_packet():
    decoder = FakeDetailedTelemetryV1()
    logical_packet = packet(single_packet=False)
    result = telemetry.decode_b01_packets(
        [logical_packet],
        decoder=decoder,
    )

    assert decoder.b01_calls == [(logical_packet,)]
    assert result.coverage is telemetry.TelemetryCoverage.DECODED
    assert result.fpga.field_names == FIELD_NAMES
    assert result.fpga.raw_counts.dtype == np.dtype(np.uint16)
    assert result.fpga.values.dtype == np.dtype(np.float64)
    assert result.fpga.valid.dtype == np.dtype(np.bool_)
    assert result.fpga.raw_counts[0, 1] == 0
    assert np.isnan(result.fpga.values[0, 1])
    assert not result.fpga.valid[0, 1]
    assert result.fpga.raw_counts[0, 0] == 0
    assert result.fpga.values[0, 0] == 0.0
    assert result.fpga.valid[0, 0]
    assert result.fpga.raw_seconds[0] == 100.0
    assert np.isnan(result.fpga.mjd_times[0])
    assert not result.fpga.mjd_time_valid[0]
    assert not result.fpga.values.flags.writeable


def test_encoder_is_counts_and_issue_only():
    decoder = FakeDetailedTelemetryV1()
    result = telemetry.decode_b01_packets(
        [packet(0x325, file_index=12)],
        decoder=decoder,
    )

    assert result.coverage is telemetry.TelemetryCoverage.PARTIAL
    assert result.encoder.row_count == 0
    assert result.encoder.field_names == ()
    assert result.counts.scalar("encoder_input_packet_count") == 1
    assert result.counts.scalar("encoder_rejected_packet_count") == 1
    assert result.issues[0].code.endswith("encoder_layout_unvalidated")
    assert result.issues[0].packet_index == 12


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["fpga"].__setitem__(
            "mission_seconds",
            result["fpga"]["mission_seconds"].astype(np.float64),
        ),
        lambda result: result["fpga"]["values"].__setitem__((0, 1), 0.0),
        lambda result: result["fpga"]["values"].__setitem__((0, 1), np.inf),
        lambda result: result["fpga"].__setitem__(
            "field_names", ("raw_adc", "temperature")
        ),
        lambda result: result["fpga"].__setitem__(
            "input_indices", np.asarray([1], dtype=np.int64)
        ),
        lambda result: result["counts"].__setitem__(
            "input_packet_count", 2
        ),
        lambda result: result["counts"].__setitem__(
            "claimed_appid_counts", {0x314: 0, 0x325: 1}
        ),
        lambda result: result["counts"].__setitem__(
            "fpga_dropped_packet_count", 1
        ),
    ],
)
def test_malformed_decoder_results_are_incompatible(mutation):
    decoder = FakeDetailedTelemetryV1()
    decoder.mutate_result = mutation
    result = telemetry.decode_b01_packets([packet()], decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert result.fpga is None
    assert result.issues[-1].code == "telemetry_adapter.decoder_incompatible"


def test_invalid_engineering_value_requires_field_scoped_issue():
    decoder = FakeDetailedTelemetryV1()

    def invalidate_without_issue(result):
        result["fpga"]["values"][0, 0] = np.nan
        result["fpga"]["valid"][0, 0] = False

    decoder.mutate_result = invalidate_without_issue
    result = telemetry.decode_b01_packets([packet()], decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert "field-scoped issue" in result.issues[-1].message


def test_private_issue_field_must_exist_in_metadata():
    decoder = FakeDetailedTelemetryV1()

    def add_unknown_field_issue(result):
        result["issues"].append({
            "code": "telemetry_decoder.invalid_engineering_value",
            "severity": "warning",
            "action": "kept",
            "message": "synthetic invalid field",
            "input_index": 0,
            "appid": 0x314,
            "sequence_count": 8,
            "field": "unknown_field",
            "details": {},
        })

    decoder.mutate_result = add_unknown_field_issue
    result = telemetry.decode_b01_packets([packet()], decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert "outside field_metadata" in result.issues[-1].message


def test_field_scoped_issue_authorizes_invalid_engineering_value():
    decoder = FakeDetailedTelemetryV1()

    def invalidate_with_issue(result):
        result["fpga"]["values"][0, 0] = np.nan
        result["fpga"]["valid"][0, 0] = False
        result["issues"].append({
            "code": "telemetry_decoder.invalid_engineering_value",
            "severity": "warning",
            "action": "kept",
            "message": "temperature conversion failed",
            "input_index": 0,
            "appid": 0x314,
            "sequence_count": 8,
            "field": "temperature",
            "details": {},
        })

    decoder.mutate_result = invalidate_with_issue
    result = telemetry.decode_b01_packets([packet()], decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.AVAILABLE
    assert result.coverage is telemetry.TelemetryCoverage.PARTIAL
    assert not result.fpga.valid[0, 0]
    assert result.issues[0].as_dict()["details"]["field"] == "temperature"


def test_decoder_runtime_failure_is_broken():
    class BrokenDecoder(FakeDetailedTelemetryV1):
        def decode_b01_packets(self, packets):
            raise RuntimeError("synthetic failure")

    result = telemetry.decode_b01_packets([packet()], decoder=BrokenDecoder())

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.BROKEN
    assert result.coverage is telemetry.TelemetryCoverage.BROKEN
    assert result.decoder_info.decoder_name == "generic-test-decoder"
    assert tuple(field.name for field in result.field_metadata) == FIELD_NAMES


def test_decoder_runtime_diagnostic_is_sanitized_for_storage():
    class BrokenDecoder(FakeDetailedTelemetryV1):
        def decode_b01_packets(self, packets):
            raise RuntimeError("bad\x00diagnostic\ud800")

    result = telemetry.decode_b01_packets([packet()], decoder=BrokenDecoder())

    assert "\x00" not in result.issues[0].message
    result.issues[0].message.encode("utf-8")


def test_telemetry_text_and_count_contracts_fail_before_storage():
    with pytest.raises(ValueError, match="null character"):
        telemetry.TelemetryDecoderInfo(
            api_version=1,
            decoder_name="bad\x00name",
            decoder_version="1",
            claimed_appids=(0x314,),
        )
    with pytest.raises(ValueError, match="valid UTF-8"):
        telemetry.TelemetryFieldMetadata(
            name="temperature",
            unit="\ud800",
            kind="continuous",
            interpolation="linear",
            display_group=None,
        )
    with pytest.raises(ValueError, match="uint64-range"):
        telemetry.TelemetryCounts(
            source="legacy_sidecar",
            scalar_counts=(
                ("input_byte_count", 1 << 64),
                ("complete_record_count", 0),
                ("trailing_byte_count", 1),
                ("output_record_count", 0),
                ("dropped_record_count", 0),
            ),
        )


def test_decoder_loss_without_private_issue_gets_public_issue():
    class LossWithoutIssue(FakeDetailedTelemetryV1):
        def decode_b01_packets(self, packets):
            result = super().decode_b01_packets(packets)
            result["fpga"] = block("b01_0x314")
            result["counts"]["fpga_output_record_count"] = 0
            result["counts"]["fpga_dropped_packet_count"] = 1
            result["issues"] = []
            return result

    result = telemetry.decode_b01_packets(
        [packet()],
        decoder=LossWithoutIssue(),
    )

    assert result.coverage is telemetry.TelemetryCoverage.PARTIAL
    assert result.issues[0].code == "telemetry_adapter.decoder_reported_loss"


def test_result_rejects_contradictory_status_and_coverage():
    result = telemetry.decode_b01_packets(
        [packet()],
        decoder=FakeDetailedTelemetryV1(),
    )

    with pytest.raises(ValueError, match="present-empty telemetry coverage"):
        replace(result, coverage=telemetry.TelemetryCoverage.PRESENT_EMPTY)
    with pytest.raises(ValueError, match="must resolve the decoder"):
        replace(result, decoder_status=telemetry.TelemetryDecoderStatus.NOT_NEEDED)


def test_noncontinuous_field_cannot_request_linear_interpolation():
    with pytest.raises(ValueError, match="requires a continuous field"):
        telemetry.TelemetryFieldMetadata(
            name="flags",
            unit="count",
            kind="bitmask",
            interpolation="linear",
            display_group=None,
        )


def test_public_array_names_are_reserved_from_engineering_fields():
    with pytest.raises(ValueError, match="reserved for public arrays"):
        telemetry.TelemetryFieldMetadata(
            name="raw_seconds",
            unit="s",
            kind="continuous",
            interpolation="linear",
            display_group=None,
        )

    class ReservedFieldDecoder(FakeDetailedTelemetryV1):
        def field_metadata(self):
            return {
                "raw_seconds": {
                    "unit": "s",
                    "kind": "continuous",
                    "interpolation": "linear",
                    "display_group": None,
                },
            }

    result = telemetry.decode_b01_packets(
        [packet()],
        decoder=ReservedFieldDecoder(),
    )

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert "reserved for public arrays" in result.issues[-1].message


def test_public_array_names_are_reserved_in_telemetry_blocks():
    with pytest.raises(ValueError, match="reserved for public arrays"):
        telemetry.TelemetryBlock(
            source_kind="b01_0x314",
            field_names=("raw_seconds",),
            input_indices=np.empty(0, dtype=np.int64),
            mission_seconds=np.empty(0, dtype=np.uint32),
            lusee_subsecs=np.empty(0, dtype=np.uint16),
            raw_seconds=np.empty(0, dtype=np.float64),
            mjd_times=np.empty(0, dtype=np.float64),
            mjd_time_valid=np.empty(0, dtype=np.bool_),
            raw_counts=np.empty((0, 1), dtype=np.uint16),
            values=np.empty((0, 1), dtype=np.float64),
            valid=np.empty((0, 1), dtype=np.bool_),
        )


def test_sidecar_states_and_public_discovery(tmp_path):
    decoder = FakeDetailedTelemetryV1()
    assert telemetry.find_legacy_sidecar(tmp_path) is None

    sidecar = tmp_path / telemetry.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"")
    assert telemetry.find_legacy_sidecar(tmp_path) == sidecar
    empty = telemetry.decode_legacy_sidecar(sidecar, decoder=decoder)
    assert empty.input_state is telemetry.TelemetryInputState.PRESENT_EMPTY
    assert empty.coverage is telemetry.TelemetryCoverage.PRESENT_EMPTY
    assert empty.fpga.row_count == 0

    sidecar.write_bytes(b"abc")
    partial = telemetry.decode_legacy_sidecar(sidecar, decoder=decoder)
    assert partial.input_state is telemetry.TelemetryInputState.PRESENT
    assert partial.coverage is telemetry.TelemetryCoverage.PARTIAL
    assert partial.fpga.row_count == 1
    assert partial.counts.scalar("trailing_byte_count") == 1
    assert partial.issues[0].code.endswith("trailing_partial_record")


def test_sidecar_issue_indices_are_tied_to_complete_input_records(tmp_path):
    decoder = FakeDetailedTelemetryV1()
    decoder.mutate_result = lambda result: result["issues"][0].__setitem__(
        "input_index",
        1,
    )
    sidecar = tmp_path / telemetry.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"abc")

    result = telemetry.decode_legacy_sidecar(sidecar, decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE
    assert result.decoder_info is not None
    assert result.field_metadata


def test_sidecar_decoder_must_account_for_nonempty_input(tmp_path):
    decoder = FakeDetailedTelemetryV1()

    def discard_accounting(result):
        result["fpga"] = block("legacy_binary_sidecar")
        result["issues"] = []
        result["counts"].update({
            "complete_record_count": 0,
            "trailing_byte_count": 0,
            "output_record_count": 0,
        })

    decoder.mutate_result = discard_accounting
    sidecar = tmp_path / telemetry.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"abc")

    result = telemetry.decode_legacy_sidecar(sidecar, decoder=decoder)

    assert result.decoder_status is telemetry.TelemetryDecoderStatus.INCOMPATIBLE


def test_adapter_owns_immutable_copies():
    decoder = FakeDetailedTelemetryV1()
    captured = {}

    def capture(result):
        captured["values"] = result["fpga"]["values"]

    decoder.mutate_result = capture
    result = telemetry.decode_b01_packets([packet()], decoder=decoder)
    before = deepcopy(result.fpga.values)
    captured["values"][0, 0] = 99.0

    assert np.array_equal(result.fpga.values, before, equal_nan=True)


def clock_reference_set(*, include_dcb=True):
    clocks = [
        ClockReference(
            clock_source=ClockSource.SPECTROMETER,
            clock_reference_raw_seconds=10.0,
        )
    ]
    if include_dcb:
        clocks.append(ClockReference(
            clock_source=ClockSource.DCB,
            clock_reference_raw_seconds=1000.0,
        ))
    return ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2027-05-01T00:00:00",
        time_scale="utc",
        clocks=tuple(clocks),
        source="synthetic telemetry assignment test",
        assumed=True,
        source_sha256="0" * 64,
    )


def result_with_times(mission_seconds, subsecs):
    decoder = FakeDetailedTelemetryV1()
    packets = [packet() for _ in mission_seconds]
    result = telemetry.decode_b01_packets(packets, decoder=decoder)
    n_rows = len(mission_seconds)
    seconds = np.asarray(mission_seconds, dtype=np.uint32)
    fractions = np.asarray(subsecs, dtype=np.uint16)
    ticks = seconds.astype(np.uint64) * np.uint64(65536) + fractions
    block_with_times = telemetry.TelemetryBlock(
        source_kind=result.fpga.source_kind,
        field_names=result.fpga.field_names,
        input_indices=np.arange(n_rows, dtype=np.int64),
        mission_seconds=seconds,
        lusee_subsecs=fractions,
        raw_seconds=ticks.astype(np.float64) / 65536.0,
        mjd_times=np.full(n_rows, np.nan, dtype=np.float64),
        mjd_time_valid=np.zeros(n_rows, dtype=np.bool_),
        raw_counts=result.fpga.raw_counts,
        values=result.fpga.values,
        valid=result.fpga.valid,
    )
    return result.with_blocks(fpga=block_with_times)


def test_session_assignment_uses_distinct_anchors_and_subseconds():
    result = result_with_times(
        [999, 1009, 1010],
        [65535, 65535, 0],
    )
    second = Session(ordinal=1, start_raw_seconds=20.0)
    first = Session(ordinal=0, start_raw_seconds=10.0)
    sessions = [second, first]
    collector = IssueCollector()

    unassigned = assign_telemetry_to_sessions(
        sessions,
        result,
        clock_reference_set=clock_reference_set(),
        issue_collector=collector,
    )

    assert first.telemetry.fpga.input_indices.tolist() == [1]
    assert second.telemetry.fpga.input_indices.tolist() == [2]
    assert first.telemetry.fpga.raw_seconds[0] == 1009 + 65535 / 65536
    assert second.telemetry.fpga.raw_seconds[0] == 1010.0
    assert first.telemetry.fpga.mjd_time_valid.tolist() == [True]
    assert second.telemetry.fpga.mjd_time_valid.tolist() == [True]
    assert first.telemetry.fpga.valid.tolist() == [result.fpga.valid[1].tolist()]
    assert unassigned.unassigned_fpga.input_indices.tolist() == [0]
    assert first.telemetry.unassigned_fpga.input_indices.tolist() == [0]
    assert second.telemetry.unassigned_fpga.input_indices.tolist() == [0]
    assert collector.issues[0].code == "telemetry_assignment.pre_session_rows"


def test_missing_dcb_anchor_retains_all_rows_unassigned():
    result = result_with_times([1000], [1])
    session = Session(ordinal=0, start_raw_seconds=10.0)
    collector = IssueCollector()

    unassigned = assign_telemetry_to_sessions(
        [session],
        result,
        clock_reference_set=clock_reference_set(include_dcb=False),
        issue_collector=collector,
    )

    assert session.telemetry.fpga.row_count == 0
    assert session.telemetry.unassigned_fpga.raw_seconds.tolist() == [
        1000 + 1 / 65536
    ]
    assert not session.telemetry.unassigned_fpga.mjd_time_valid[0]
    assert unassigned.unassigned_fpga.row_count == 1
    assert collector.issues[0].code == (
        "telemetry_assignment.missing_clock_reference"
    )


@pytest.mark.parametrize(
    "sessions",
    [
        [Session(ordinal=0, start_raw_seconds=None)],
        [
            Session(ordinal=0, start_raw_seconds=20.0),
            Session(ordinal=1, start_raw_seconds=10.0),
        ],
    ],
)
def test_invalid_session_boundaries_never_force_rows_into_a_session(sessions):
    result = result_with_times([1000], [0])
    collector = IssueCollector()

    unassigned = assign_telemetry_to_sessions(
        sessions,
        result,
        clock_reference_set=clock_reference_set(),
        issue_collector=collector,
    )

    assert unassigned.unassigned_fpga.row_count == 1
    assert all(session.telemetry.fpga.row_count == 0 for session in sessions)
    assert collector.issues[0].code == (
        "telemetry_assignment.invalid_session_boundaries"
    )
