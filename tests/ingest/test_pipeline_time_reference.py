from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from lusee.ingest import decode, fits_writer, hdf5_writer, pipeline
from lusee.ingest.clock_reference import load_clock_reference_set
from lusee.ingest.constants import BANK_FILENAME, TELEMETRY_BANK
from lusee.ingest.decode import Products
from lusee.ingest.issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    ValidatedCounts,
)
from lusee.ingest.telemetry import (
    TelemetryBlock,
    TelemetryCounts,
    TelemetryCoverage,
    TelemetryDecodeResult,
    TelemetryDecoderInfo,
    TelemetryDecoderStatus,
    TelemetryFieldMetadata,
    TelemetryInputState,
)
from lusee.ingest.write_request import FamilyCoverage, WriteRequest


def write_landing_reference(
    tmp_path,
    *,
    name="landing.json",
    reference_raw_seconds=100.0,
    reference_isot="2027-05-01T00:00:00",
    clocks=None,
):
    if clocks is None:
        clocks = {
            "spectrometer": {
                "clock_reference_raw_seconds": reference_raw_seconds,
            },
        }
    path = tmp_path / name
    path.write_text(json.dumps({
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": reference_isot,
        "time_scale": "utc",
        "clocks": clocks,
        "source": "synthetic pipeline test",
        "assumed": True,
    }), encoding="utf-8")
    return path


def valid_products(*, packet_map_verified=True):
    provenance = DecodeProvenance.from_report(
        distribution_version="1.0.0",
        decoder_source_commit="a" * 40,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant=None,
        schema_assumed=False,
        binding_source_release="test-release",
        binding_source_commit="b" * 40,
        abi_fingerprint="c" * 64,
        execution_mode="collect",
        input_packet_count=0,
        valid_packet_count=0,
        appid_counts=(),
        issue_counts=(),
        canonical_report={"fixture": "pipeline-time"},
    )
    return Products(
        sw_version=0x307,
        fw_version=1,
        fw_id=2,
        fw_date=3,
        fw_time=4,
        start_unique_packet_id=5,
        start_time_32=98304,
        start_time_16=0,
        start_raw_seconds=1.5,
        decode_provenance=provenance,
        packet_map_status=(
            "verified" if packet_map_verified else "unavailable"
        ),
        packet_map_format_version=1 if packet_map_verified else None,
        raw_flash_provenance_unavailable_reason=(
            None if packet_map_verified else "packet_map_missing_legacy_session"
        ),
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=0,
            valid_packets=0,
            product_rows=(),
        ),
    )


def fake_worker_result(kwargs):
    return pipeline.SessionResult(
        session_ordinal=kwargs["ordinal"],
        session_name=kwargs["name"] or "session_000",
        source_path=str(kwargs["source_path"]),
        source_kind=kwargs["source_kind"],
    )


def empty_telemetry_block(source_kind, field_names=()):
    field_names = tuple(field_names)
    return TelemetryBlock(
        source_kind=source_kind,
        field_names=field_names,
        input_indices=np.empty(0, dtype=np.int64),
        mission_seconds=np.empty(0, dtype=np.uint32),
        lusee_subsecs=np.empty(0, dtype=np.uint16),
        raw_seconds=np.empty(0, dtype=np.float64),
        mjd_times=np.empty(0, dtype=np.float64),
        mjd_time_valid=np.empty(0, dtype=np.bool_),
        raw_counts=np.empty((0, len(field_names)), dtype=np.uint16),
        values=np.empty((0, len(field_names)), dtype=np.float64),
        valid=np.empty((0, len(field_names)), dtype=np.bool_),
    )


def present_empty_telemetry(source):
    decoder_info = TelemetryDecoderInfo(
        api_version=1,
        decoder_name="test-decoder",
        decoder_version="test-1",
        claimed_appids=(0x314, 0x325),
    )
    if source == "b01":
        counts = TelemetryCounts(
            source="b01",
            scalar_counts=(
                ("input_packet_count", 0),
                ("claimed_packet_count", 0),
                ("unclaimed_packet_count", 0),
                ("fpga_input_packet_count", 0),
                ("fpga_output_record_count", 0),
                ("fpga_dropped_packet_count", 0),
                ("encoder_input_packet_count", 0),
                ("encoder_output_record_count", 0),
                ("encoder_rejected_packet_count", 0),
            ),
        )
        fpga = empty_telemetry_block("b01_0x314", ("temperature",))
        encoder = empty_telemetry_block("b01_0x325")
    else:
        counts = TelemetryCounts(
            source="legacy_sidecar",
            scalar_counts=(
                ("input_byte_count", 0),
                ("complete_record_count", 0),
                ("trailing_byte_count", 0),
                ("output_record_count", 0),
                ("dropped_record_count", 0),
            ),
        )
        fpga = empty_telemetry_block(
            "legacy_binary_sidecar",
            ("temperature",),
        )
        encoder = None
    return TelemetryDecodeResult(
        input_source=source,
        input_state=TelemetryInputState.PRESENT_EMPTY,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=TelemetryCoverage.PRESENT_EMPTY,
        decoder_info=decoder_info,
        field_metadata=(
            TelemetryFieldMetadata(
                name="temperature",
                unit="K",
                kind="continuous",
                interpolation="linear",
                display_group="thermal",
            ),
        ),
        fpga=fpga,
        encoder=encoder,
        counts=counts,
    )


def unavailable_telemetry(source, *, present_empty=False):
    collector = IssueCollector()
    collector.record(
        code="telemetry_adapter.decoder_unavailable",
        severity="warning",
        stage="telemetry_decode",
        message="test decoder is unavailable",
        action="kept",
    )
    return TelemetryDecodeResult(
        input_source=source,
        input_state=(
            TelemetryInputState.PRESENT_EMPTY
            if present_empty
            else TelemetryInputState.PRESENT
        ),
        decoder_status=TelemetryDecoderStatus.UNAVAILABLE,
        coverage=TelemetryCoverage.UNAVAILABLE,
        issues=collector.issues,
    )


def telemetry_with_raw_seconds(raw_seconds):
    mission_seconds = np.asarray(raw_seconds, dtype=np.uint32)
    n_rows = mission_seconds.size
    fpga = TelemetryBlock(
        source_kind="b01_0x314",
        field_names=("temperature",),
        input_indices=np.arange(n_rows, dtype=np.int64),
        mission_seconds=mission_seconds,
        lusee_subsecs=np.zeros(n_rows, dtype=np.uint16),
        raw_seconds=mission_seconds.astype(np.float64),
        mjd_times=np.full(n_rows, np.nan, dtype=np.float64),
        mjd_time_valid=np.zeros(n_rows, dtype=np.bool_),
        raw_counts=np.zeros((n_rows, 1), dtype=np.uint16),
        values=np.zeros((n_rows, 1), dtype=np.float64),
        valid=np.ones((n_rows, 1), dtype=np.bool_),
    )
    return TelemetryDecodeResult(
        input_source="b01",
        input_state=TelemetryInputState.PRESENT,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=TelemetryCoverage.DECODED,
        decoder_info=TelemetryDecoderInfo(
            api_version=1,
            decoder_name="test-decoder",
            decoder_version="test-1",
            claimed_appids=(0x314, 0x325),
        ),
        field_metadata=(TelemetryFieldMetadata(
            name="temperature",
            unit="K",
            kind="continuous",
            interpolation="linear",
            display_group="thermal",
        ),),
        fpga=fpga,
        encoder=empty_telemetry_block("b01_0x325"),
        counts=TelemetryCounts(
            source="b01",
            scalar_counts=(
                ("input_packet_count", n_rows),
                ("claimed_packet_count", n_rows),
                ("unclaimed_packet_count", 0),
                ("fpga_input_packet_count", n_rows),
                ("fpga_output_record_count", n_rows),
                ("fpga_dropped_packet_count", 0),
                ("encoder_input_packet_count", 0),
                ("encoder_output_record_count", 0),
                ("encoder_rejected_packet_count", 0),
            ),
            claimed_appid_counts=((0x314, n_rows),),
        ),
    )


@pytest.mark.parametrize("damage", ["missing", "malformed", "no_spectrometer"])
def test_flash_reference_preflight_precedes_reads_and_destinations(
    tmp_path,
    monkeypatch,
    damage,
):
    landing = tmp_path / "landing.json"
    if damage == "malformed":
        landing.write_text("{}", encoding="utf-8")
    elif damage == "no_spectrometer":
        landing = write_landing_reference(
            tmp_path,
            clocks={"dcb": {"clock_reference_raw_seconds": 0.0}},
        )
    monkeypatch.setattr(
        pipeline,
        "_parse_flash_loaded",
        lambda *args, **kwargs: pytest.fail("flash banks were read"),
    )
    sessions_root = tmp_path / "sessions"

    with pytest.raises((OSError, ValueError)):
        pipeline.process_flash(
            tmp_path / "flash",
            landing_time_file=landing,
            sessions_root=sessions_root,
        )

    assert not sessions_root.exists()


def test_session_names_and_manifest_time_use_spectrometer_anchor(tmp_path):
    reference = load_clock_reference_set(write_landing_reference(tmp_path))

    assert (
        pipeline.default_session_name(2, 100.0, reference)
        == "session_002_20270501_000000"
    )
    assert (
        pipeline.default_session_name(2, 0.0, reference)
        == "session_002_20270430_235820"
    )
    assert (
        pipeline.default_session_name(2, -1.0, reference)
        == "session_002_20270430_235819"
    )
    assert (
        pipeline._start_utc_iso(100.000244140625, reference)
        == "2027-05-01T00:00:00.000244141Z"
    )


@pytest.mark.parametrize("supply_file", [False, True])
def test_process_session_reuses_matching_embedded_reference(
    tmp_path,
    monkeypatch,
    supply_file,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    landing = write_landing_reference(tmp_path)
    expected = load_clock_reference_set(landing)
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_schema_version": 3,
        "clock_reference": expected.as_record(),
    }), encoding="ascii")
    seen = []
    monkeypatch.setattr(
        pipeline,
        "_process_one_session",
        lambda **kwargs: seen.append(kwargs) or fake_worker_result(kwargs),
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: None,
    )

    pipeline.process_session(
        session_dir,
        landing_time_file=landing if supply_file else None,
        h5_dir=tmp_path / "h5",
        rederive_telemetry=False,
    )

    assert len(seen) == 1
    assert seen[0]["clock_reference_set"] == expected


@pytest.mark.parametrize("damage", ["contradiction", "corrupt_embedded"])
def test_process_session_rejects_bad_embedded_reference_before_worker(
    tmp_path,
    monkeypatch,
    damage,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    landing = write_landing_reference(tmp_path)
    reference = load_clock_reference_set(landing)
    record = reference.as_record()
    supplied = landing
    if damage == "contradiction":
        supplied = write_landing_reference(
            tmp_path,
            name="other.json",
            reference_isot="2027-05-02T00:00:00",
        )
    else:
        record["source_sha256"] = "bad"
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_schema_version": 3,
        "clock_reference": record,
    }), encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "_process_one_session",
        lambda **kwargs: pytest.fail("session worker ran"),
    )

    with pytest.raises(ValueError):
        pipeline.process_session(
            session_dir,
            landing_time_file=supplied,
            h5_dir=tmp_path / "h5",
            rederive_telemetry=False,
        )

    assert not (tmp_path / "h5").exists()


def test_process_session_allows_raw_diagnostic_but_requires_reference_for_output(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    seen = []
    monkeypatch.setattr(
        pipeline,
        "_process_one_session",
        lambda **kwargs: seen.append(kwargs) or fake_worker_result(kwargs),
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: None,
    )

    pipeline.process_session(session_dir, rederive_telemetry=False)

    assert len(seen) == 1
    assert seen[0]["clock_reference_set"] is None
    with pytest.raises(ValueError, match="requires landing_time_file"):
        pipeline.process_session(
            session_dir,
            h5_dir=tmp_path / "h5",
            rederive_telemetry=False,
        )
    assert len(seen) == 1
    assert not (tmp_path / "h5").exists()


@pytest.mark.parametrize("decoder_available", [False, True])
def test_present_empty_sidecar_records_explicit_decoder_outcome(
    tmp_path,
    monkeypatch,
    decoder_available,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / pipeline.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"")
    landing = write_landing_reference(tmp_path)
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: sidecar,
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_legacy_sidecar",
        lambda path, *, issue_collector=None: (
            present_empty_telemetry("legacy_sidecar")
            if decoder_available
            else unavailable_telemetry(
                "legacy_sidecar",
                present_empty=True,
            )
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )

    result = pipeline.process_session(
        session_dir,
        landing_time_file=landing,
        h5_dir=tmp_path / "h5",
        rederive_telemetry=False,
    )

    assert result.telemetry_decoder_status == (
        "available" if decoder_available else "unavailable"
    )
    assert result.telemetry_coverage == (
        "present_empty" if decoder_available else "unavailable"
    )
    assert result.h5_path is not None


@pytest.mark.parametrize("window", ["before_find", "before_decode"])
def test_disappearing_sidecar_is_broken_without_losing_science(
    tmp_path,
    monkeypatch,
    window,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / pipeline.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"synthetic")
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )
    if window == "before_find":
        def disappear_before_find(path):
            sidecar.unlink()

        monkeypatch.setattr(
            pipeline.telemetry_mod,
            "find_legacy_sidecar",
            disappear_before_find,
        )
    else:
        original_decode = pipeline.telemetry_mod.decode_legacy_sidecar

        def disappear_before_decode(path, **kwargs):
            sidecar.unlink()
            return original_decode(path, **kwargs)

        monkeypatch.setattr(
            pipeline.telemetry_mod,
            "decode_legacy_sidecar",
            disappear_before_decode,
        )

    result = pipeline.process_session(
        session_dir,
        landing_time_file=write_landing_reference(tmp_path),
        h5_dir=tmp_path / "h5",
        rederive_telemetry=False,
    )

    assert result.telemetry_source == "sidecar"
    assert result.telemetry_decoder_status == "broken"
    assert result.telemetry_coverage == "broken"
    assert result.h5_path is not None


def test_present_empty_b01_is_persisted_with_typed_state(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    telemetry = present_empty_telemetry("b01")
    session = pipeline.Session(ordinal=0, telemetry=telemetry)
    monkeypatch.setattr(
        pipeline,
        "_parse_flash_loaded",
        lambda *args, **kwargs: ([session], telemetry, None),
    )
    monkeypatch.setattr(
        pipeline,
        "write_uncrater_session",
        lambda session, path: path.mkdir(parents=True) or path,
    )
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )
    results = pipeline.process_flash(
        flash_dir,
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=tmp_path / "sessions",
        h5_dir=tmp_path / "h5",
    )

    assert len(results) == 1
    assert results[0].telemetry_coverage == "present_empty"
    assert results[0].h5_path is not None


def test_flash_override_b01_disabled_is_persisted_as_broken(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )
    result = pipeline.process_session(
        session_dir,
        landing_time_file=write_landing_reference(tmp_path),
        flash_root=flash_dir,
        h5_dir=tmp_path / "h5",
        rederive_telemetry=False,
    )

    assert result.telemetry_decoder_status == "broken"
    assert result.telemetry_coverage == "broken"
    assert result.telemetry_source == "flash"
    assert result.h5_path is not None


def test_moved_session_keeps_science_when_recorded_b01_is_unreachable(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    landing = write_landing_reference(tmp_path)
    reference = load_clock_reference_set(landing)
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_schema_version": 3,
        "clock_reference": reference.as_record(),
        "flash_source_path": str(tmp_path / "missing-flash"),
        "telemetry_input_sources": ["b01"],
    }), encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )

    result = pipeline.process_session(
        session_dir,
        h5_dir=tmp_path / "h5",
    )

    assert result.telemetry_decoder_status == "broken"
    assert result.telemetry_coverage == "broken"
    assert result.h5_path is not None
    assert (tmp_path / "h5" / f"{result.session_name}.h5").is_file()


def test_unreachable_b01_issue_is_retained_when_sidecar_is_selected(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / pipeline.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"")
    landing = write_landing_reference(tmp_path)
    reference = load_clock_reference_set(landing)
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_schema_version": 3,
        "clock_reference": reference.as_record(),
        "flash_source_path": str(tmp_path / "missing-flash"),
        "telemetry_input_sources": ["b01"],
    }), encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )
    def decode_sidecar(*args, issue_collector=None, **kwargs):
        issue = issue_collector.record(
            code="telemetry_adapter.sidecar_warning",
            severity="warning",
            stage="telemetry_decode",
            message="synthetic sidecar warning",
            action="kept",
        )
        return present_empty_telemetry("legacy_sidecar").with_issues((issue,))

    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_legacy_sidecar",
        decode_sidecar,
    )

    result = pipeline.process_session(
        session_dir,
        h5_dir=tmp_path / "h5",
    )

    assert result.telemetry_source == "sidecar"
    assert result.telemetry_input_sources == ["b01", "legacy_sidecar"]
    assert result.telemetry_decoder_status == "available"
    assert result.telemetry_coverage == "partial"
    with h5py.File(result.h5_path, "r") as h5:
        assert h5["telemetry/issue_refs/issue_index"][:].tolist() == [0, 1, 2]


def test_reachable_b01_records_ignored_sidecar_selection(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / pipeline.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"")
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    landing = write_landing_reference(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )
    monkeypatch.setattr(
        pipeline,
        "_rederive_telemetry_from_flash",
        lambda *args, **kwargs: present_empty_telemetry("b01"),
    )

    result = pipeline.process_session(
        session_dir,
        landing_time_file=landing,
        flash_root=flash_dir,
        h5_dir=tmp_path / "h5",
    )

    assert result.telemetry_source == "flash"
    assert result.telemetry_input_sources == ["b01", "legacy_sidecar"]
    assert result.telemetry_decoder_status == "available"
    assert result.telemetry_coverage == "partial"


def test_broken_reachable_b01_falls_back_to_sidecar(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    sidecar = session_dir / pipeline.LEGACY_TELEMETRY_SIDECAR_NAME
    sidecar.write_bytes(b"")
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    landing = write_landing_reference(tmp_path)
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )

    def broken_b01(*args, issue_collector=None, **kwargs):
        marker = issue_collector.mark()
        return pipeline._broken_b01_telemetry(
            issue_collector,
            issue_marker=marker,
            error_type="synthetic_failure",
        )

    monkeypatch.setattr(
        pipeline,
        "_rederive_telemetry_from_flash",
        broken_b01,
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_legacy_sidecar",
        lambda *args, **kwargs: present_empty_telemetry("legacy_sidecar"),
    )

    result = pipeline.process_session(
        session_dir,
        landing_time_file=landing,
        flash_root=flash_dir,
        h5_dir=tmp_path / "h5",
    )

    assert result.telemetry_source == "sidecar"
    assert result.telemetry_input_sources == ["b01", "legacy_sidecar"]
    assert result.telemetry_decoder_status == "available"
    assert result.telemetry_coverage == "partial"


def test_rederive_preserves_unassigned_rows_and_maps_dcb_time(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    decoded = telemetry_with_raw_seconds((990, 1000, 1010))
    monkeypatch.setattr(pipeline, "parse_bank_file", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_b01_packets",
        lambda packets, *, issue_collector=None: decoded,
    )
    reference = load_clock_reference_set(write_landing_reference(
        tmp_path,
        clocks={
            "spectrometer": {"clock_reference_raw_seconds": 100.0},
            "dcb": {"clock_reference_raw_seconds": 1000.0},
        },
    ))
    assignment_collector = IssueCollector()
    assignment_issue = assignment_collector.record(
        code="telemetry_assignment.pre_session_rows",
        severity="warning",
        stage="telemetry_assignment",
        message="b01 telemetry rows before the first session were retained unassigned",
        action="kept",
        details={"row_count": 1},
    )

    result = pipeline._rederive_telemetry_from_flash(
        flash_dir,
        window_lower_elapsed_seconds=0.0,
        window_upper_elapsed_seconds=10.0,
        clock_reference_set=reference,
        assignment_mode="assigned_with_pre_session",
        unassigned_upper_elapsed_seconds=0.0,
        assignment_issues=[assignment_issue.as_dict()],
        issue_collector=IssueCollector(),
    )

    assert result.fpga.raw_seconds.tolist() == [1000.0]
    assert result.unassigned_fpga.raw_seconds.tolist() == [990.0]
    assert result.fpga.mjd_time_valid.tolist() == [True]
    assert result.unassigned_fpga.mjd_time_valid.tolist() == [True]
    assert result.issues[-1].code == "telemetry_assignment.pre_session_rows"


def test_rederive_all_unassigned_keeps_mapped_full_block(tmp_path, monkeypatch):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    decoded = telemetry_with_raw_seconds((1000, 1001))
    monkeypatch.setattr(pipeline, "parse_bank_file", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_b01_packets",
        lambda packets, *, issue_collector=None: decoded,
    )
    reference = load_clock_reference_set(write_landing_reference(
        tmp_path,
        clocks={
            "spectrometer": {"clock_reference_raw_seconds": 100.0},
            "dcb": {"clock_reference_raw_seconds": 1000.0},
        },
    ))

    result = pipeline._rederive_telemetry_from_flash(
        flash_dir,
        window_lower_elapsed_seconds=None,
        window_upper_elapsed_seconds=None,
        clock_reference_set=reference,
        assignment_mode="all_unassigned",
        issue_collector=IssueCollector(),
    )

    assert result.fpga.row_count == 0
    assert result.unassigned_fpga.raw_seconds.tolist() == [1000.0, 1001.0]
    assert result.unassigned_fpga.mjd_time_valid.tolist() == [True, True]
    assert result.coverage is TelemetryCoverage.PARTIAL


def test_rederive_records_missing_prerequisite_with_replayed_issue(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    decoded = telemetry_with_raw_seconds((1000,))
    monkeypatch.setattr(pipeline, "parse_bank_file", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_b01_packets",
        lambda packets, *, issue_collector=None: decoded,
    )
    reference = load_clock_reference_set(write_landing_reference(tmp_path))
    assignment_collector = IssueCollector()
    assignment_issue = assignment_collector.record(
        code="telemetry_assignment.legacy_window_converted",
        severity="warning",
        stage="telemetry_assignment",
        message="synthetic legacy conversion",
        action="kept",
    )

    result = pipeline._rederive_telemetry_from_flash(
        flash_dir,
        window_lower_elapsed_seconds=0.0,
        window_upper_elapsed_seconds=None,
        clock_reference_set=reference,
        assignment_mode="assigned_with_pre_session",
        unassigned_upper_elapsed_seconds=0.0,
        assignment_issues=[assignment_issue.as_dict()],
        issue_collector=IssueCollector(),
    )

    assert result.fpga.row_count == 0
    assert result.unassigned_fpga.row_count == 1
    assert [issue.code for issue in result.issues[-2:]] == [
        "telemetry_assignment.legacy_window_converted",
        "telemetry_assignment.rederive_unassigned",
    ]
    assert result.issues[-1].as_dict()["details"]["missing"] == ["dcb"]


def test_original_v3_raw_window_converts_to_elapsed_assignment(tmp_path):
    reference = load_clock_reference_set(write_landing_reference(
        tmp_path,
        clocks={
            "spectrometer": {"clock_reference_raw_seconds": 100.0},
            "dcb": {"clock_reference_raw_seconds": 1000.0},
        },
    ))
    manifest = {
        "manifest_schema_version": 3,
        "start_raw_seconds": 110.0,
        "telemetry_window_lower_raw_seconds": None,
        "telemetry_window_upper_raw_seconds": 120.0,
    }

    lower, upper, mode, unassigned_upper, issues = (
        pipeline._telemetry_assignment_from_manifest(
            manifest,
            clock_reference_set=reference,
        )
    )

    assert (lower, upper) == (10.0, 20.0)
    assert mode == "assigned_with_pre_session"
    assert unassigned_upper == 10.0
    assert [issue["code"] for issue in issues] == [
        "telemetry_assignment.legacy_window_converted"
    ]


def test_manifest_rejects_mixed_telemetry_assignment_contracts(tmp_path):
    reference = load_clock_reference_set(write_landing_reference(tmp_path))
    manifest = {
        "telemetry_window_lower_raw_seconds": 100.0,
        "telemetry_window_lower_elapsed_seconds": 0.0,
    }

    with pytest.raises(ValueError, match="mixes telemetry assignment contracts"):
        pipeline._telemetry_assignment_from_manifest(
            manifest,
            clock_reference_set=reference,
        )


def test_session_worker_passes_one_write_request_to_both_writers(
    tmp_path,
    monkeypatch,
):
    reference = load_clock_reference_set(write_landing_reference(tmp_path))
    calls = []
    monkeypatch.setattr(
        hdf5_writer,
        "write_hdf5",
        lambda request, path: calls.append(("hdf5", request, path)),
    )
    monkeypatch.setattr(
        fits_writer,
        "write_fits",
        lambda request, path: calls.append(("fits", request, path)),
    )

    pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name="session_000",
        ordinal=0,
        h5_dir=tmp_path / "h5",
        fits_dir=tmp_path / "fits",
        plots_dir=None,
        manifest_dir=None,
        issue_collector=IssueCollector(),
        clock_reference_set=reference,
        source_path=tmp_path / "session",
        products=valid_products(),
    )

    assert [call[0] for call in calls] == ["hdf5", "fits"]
    assert isinstance(calls[0][1], WriteRequest)
    assert calls[0][1] is calls[1][1]
    assert calls[0][1].clock_reference_set == reference


def test_dropped_family_issue_reaches_production_family_status(
    tmp_path,
    monkeypatch,
):
    issue = IngestIssue(
        issue_id="issue-00000001",
        code="decode_adapter.invalid_normal_product",
        severity=IssueSeverity.ERROR,
        stage="decode_adapter",
        message="normal spectrum was rejected",
        action=IssueAction.DROPPED,
    )
    products = valid_products()
    products.issues = (issue,)
    products.family_issue_ids = decode._dropped_family_issue_ids(products.issues)
    products.quality_status = DataQuality.PARTIAL
    requests = []
    monkeypatch.setattr(
        hdf5_writer,
        "write_hdf5",
        lambda request, path: requests.append(request),
    )

    pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name="session_000",
        ordinal=0,
        h5_dir=tmp_path / "h5",
        plots_dir=None,
        manifest_dir=None,
        issue_collector=IssueCollector(),
        clock_reference_set=load_clock_reference_set(
            write_landing_reference(tmp_path)
        ),
        products=products,
    )

    spectra = next(
        status
        for status in requests[0].family_statuses
        if status.family == "spectra"
    )
    assert spectra.coverage is FamilyCoverage.INVALID_OR_DROPPED
    assert spectra.quality is DataQuality.FAILED
    assert spectra.issue_ids == (issue.issue_id,)


def test_manifest_v3_records_clock_digest_and_packet_map_state(tmp_path):
    reference = load_clock_reference_set(write_landing_reference(tmp_path))
    telemetry = unavailable_telemetry("b01")
    result = pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name=None,
        ordinal=4,
        h5_dir=None,
        plots_dir=None,
        manifest_dir=tmp_path / "manifests",
        issue_collector=IssueCollector(),
        clock_reference_set=reference,
        telemetry=telemetry,
        telemetry_input_sources=("b01",),
        telemetry_decoder_status="unavailable",
        products=valid_products(),
    )

    manifest = json.loads((
        tmp_path / "manifests" / f"{result.session_name}.json"
    ).read_text("ascii"))
    assert manifest["manifest_schema_version"] == 3
    assert manifest["telemetry_assignment_contract_version"] == 2
    assert manifest["clock_reference"] == reference.as_record()
    assert manifest["clock_reference"]["source_sha256"] == reference.source_sha256
    assert manifest["clock_reference"]["assumed"] is True
    assert manifest["packet_map_status"] == "verified"
    assert manifest["packet_map_format_version"] == 1
    assert manifest["raw_flash_provenance_unavailable_reason"] is None
    assert manifest["telemetry_input_sources"] == ["b01"]
    assert manifest["telemetry_decoder_status"] == "unavailable"
    assert result.start_time_utc == "2027-04-30T23:58:21.500000000Z"
