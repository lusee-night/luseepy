from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lusee.ingest import pipeline
from lusee.ingest.decode import Products
from lusee.ingest.issues import IssueCollector
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ValidatedCounts,
)


def write_landing_reference(tmp_path: Path, *, event: str = "landing") -> Path:
    path = tmp_path / f"{event}.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": "2027-05-01T00:00:00",
        "time_scale": "utc",
        "clocks": {
            "spectrometer": {"clock_reference_raw_seconds": 0.0},
        },
        "source": f"synthetic {event}",
        "assumed": True,
    }), encoding="ascii")
    return path


def concrete_products() -> Products:
    return Products(
        decode_provenance=DecodeProvenance(
            decoder_name="uncrater",
            distribution_version="1.0.0",
            decoder_source_commit=None,
            reported_schema_ids=(0x307,),
            selected_schema_id=0x307,
            binding_key="307",
            schema_variant=None,
            schema_assumed=False,
            binding_source_release="test-release",
            binding_source_commit="a" * 40,
            abi_fingerprint="b" * 64,
            execution_mode=ExecutionMode.COLLECT,
            input_packet_count=0,
            valid_packet_count=0,
            appid_counts=(),
            issue_counts=(),
            canonical_report_json="{}",
        ),
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=0,
            valid_packets=0,
            product_rows=(),
        ),
    )


def install_one_session_flash(monkeypatch, telemetry=None):
    telemetry = telemetry or pipeline.telemetry_mod.TelemetryDecodeResult.absent()

    def parse_flash(path, *, clock_reference_set, issue_collector, capture):
        session = pipeline.Session(ordinal=0, telemetry=telemetry)
        capture.session_count = 1
        return [session], telemetry, None

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", parse_flash)
    monkeypatch.setattr(
        pipeline,
        "write_uncrater_session",
        lambda session, destination: destination.mkdir(parents=True),
    )
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: concrete_products(),
    )


def test_flash_source_identity_is_independent_of_run_provenance(tmp_path):
    payload = b"captured source"
    fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    first_clock = pipeline._load_landing_reference(
        write_landing_reference(tmp_path, event="first")
    )
    second_clock = pipeline._load_landing_reference(
        write_landing_reference(tmp_path, event="second")
    )

    first_run, first_input, first_reason = pipeline._flash_identity(
        fingerprint,
        first_clock,
        (0x307, "first"),
    )
    second_run, second_input, second_reason = pipeline._flash_identity(
        fingerprint,
        second_clock,
        (0x306, "second"),
    )

    assert first_input == second_input
    assert first_input is not None
    assert first_run != second_run
    assert first_reason is second_reason is None


@pytest.mark.parametrize(
    ("fingerprint", "complete", "reason"),
    [
        ({}, True, "no_captured_bank_sources"),
        (
            {
                "b02/FFFFFFFE": {
                    "size_bytes": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            },
            False,
            "source_bank_unreadable",
        ),
    ],
)
def test_incomplete_flash_source_has_no_input_identity(
    tmp_path,
    fingerprint,
    complete,
    reason,
):
    clock = pipeline._load_landing_reference(write_landing_reference(tmp_path))

    _run_id, input_identity, unavailable = pipeline._flash_identity(
        fingerprint,
        clock,
        source_identity_complete=complete,
    )

    assert input_identity is None
    assert unavailable == reason


def test_incomplete_source_reasons_are_part_of_the_run_identity(tmp_path):
    payload = b"captured source"
    fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    clock = pipeline._load_landing_reference(write_landing_reference(tmp_path))

    first, first_input, _ = pipeline._flash_identity(
        fingerprint,
        clock,
        source_identity_complete=False,
        source_identity_unavailable_reasons=("b03:missing",),
    )
    second, second_input, _ = pipeline._flash_identity(
        fingerprint,
        clock,
        source_identity_complete=False,
        source_identity_unavailable_reasons=("b04:missing",),
    )

    assert first != second
    assert first_input is second_input is None


def test_flash_manifest_records_source_identity_failure_reasons(
    tmp_path,
    monkeypatch,
):
    payload = b"captured source"

    def parse_flash(path, *, clock_reference_set, issue_collector, capture):
        capture.record_bank(
            bank="b02",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            recovered_frames=0,
        )
        capture.record_unreadable_bank(bank="b03", reason="not_regular_file")
        return (
            [],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", parse_flash)
    result = pipeline.process_flash(
        tmp_path / "flash",
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=tmp_path / "sessions",
    )
    manifest = json.loads(
        (tmp_path / "sessions" / "flash.json").read_text("ascii")
    )

    assert result.input_identity_sha256 is None
    assert result.input_identity_unavailable_reason == "source_bank_unreadable"
    assert result.source_identity_unavailable_reasons == [
        "b03:not_regular_file"
    ]
    assert result.status == "failed"
    assert result.status_issue_codes == ["pipeline.no_usable_sessions"]
    assert manifest["source_identity_unavailable_reasons"] == [
        "b03:not_regular_file"
    ]


def test_flash_is_failed_when_every_session_decode_is_failed(
    tmp_path,
    monkeypatch,
):
    install_one_session_flash(monkeypatch)
    products = concrete_products()
    products.quality_status = DataQuality.FAILED
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: products,
    )

    result = pipeline.process_flash(
        tmp_path / "flash",
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=tmp_path / "sessions",
    )

    assert result.status == "failed"
    assert result.status_issue_codes == ["decode.no_usable_products"]
    assert result[0].status == "failed"


@pytest.mark.parametrize("captured_present", [False, True])
def test_flash_uses_captured_b01_state(
    tmp_path,
    monkeypatch,
    captured_present,
):
    collector = IssueCollector()
    issue = collector.record(
        code="telemetry_adapter.synthetic_unavailable",
        severity="warning",
        stage="telemetry_decode",
        message="synthetic telemetry is unavailable",
        action="kept",
    )
    present = pipeline.telemetry_mod.TelemetryDecodeResult(
        input_source="b01",
        input_state=pipeline.telemetry_mod.TelemetryInputState.PRESENT_EMPTY,
        decoder_status=pipeline.telemetry_mod.TelemetryDecoderStatus.UNAVAILABLE,
        coverage=pipeline.telemetry_mod.TelemetryCoverage.UNAVAILABLE,
        issues=(issue,),
    )
    captured = (
        present
        if captured_present
        else pipeline.telemetry_mod.TelemetryDecodeResult.absent()
    )
    install_one_session_flash(monkeypatch, telemetry=captured)
    result = pipeline.process_flash(
        tmp_path / "flash",
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=tmp_path / "sessions",
        issue_collector=collector,
    )

    expected = ["b01"] if captured.input_source == "b01" else []
    assert result[0].telemetry_input_sources == expected


@pytest.mark.parametrize("bad_value", [1, 0, "yes", None])
def test_public_process_boundaries_require_boolean_overwrite(
    tmp_path,
    bad_value,
):
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        pipeline.process_flash(
            tmp_path / "missing-flash",
            landing_time_file=tmp_path / "missing-landing.json",
            sessions_root=tmp_path / "sessions",
            overwrite=bad_value,
        )
    with pytest.raises(TypeError, match="overwrite must be a boolean"):
        pipeline.process_session(
            tmp_path / "missing-session",
            overwrite=bad_value,
        )


def test_flash_output_alias_is_rejected_before_session_install(
    tmp_path,
    monkeypatch,
):
    install_one_session_flash(monkeypatch)

    with pytest.raises(ValueError, match="destinations alias"):
        pipeline.process_flash(
            tmp_path / "flash",
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
            session_name=lambda *args: "flash.json",
        )

    assert not (tmp_path / "sessions" / "flash.json").exists()


def test_flash_output_cannot_replace_a_source_bank(tmp_path, monkeypatch):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / pipeline.SCIENCE_BANKS[0]
    bank_dir.mkdir(parents=True)
    bank_path = bank_dir / pipeline.BANK_FILENAME
    bank_path.write_bytes(b"captured bank")
    install_one_session_flash(monkeypatch)
    installs = []
    monkeypatch.setattr(
        pipeline,
        "write_uncrater_session",
        lambda *args, **kwargs: installs.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="protected input"):
        pipeline.process_flash(
            flash_dir,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=flash_dir,
            session_name=lambda *args: pipeline.SCIENCE_BANKS[0],
            overwrite=True,
        )

    assert installs == []
    assert bank_path.read_bytes() == b"captured bank"
    assert not (flash_dir / pipeline.FLASH_MANIFEST_NAME).exists()


def test_flash_outputs_cannot_be_nested_under_source_root(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    flash_dir.mkdir()
    install_one_session_flash(monkeypatch)

    with pytest.raises(ValueError, match="protected input"):
        pipeline.process_flash(
            flash_dir,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=flash_dir / "outputs",
        )

    assert not (flash_dir / "outputs").exists()


@pytest.mark.parametrize(
    "name",
    ["session", "packet_map", "DCB_telemetry"],
)
def test_session_manifest_cannot_alias_an_ingestion_input(tmp_path, name):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    with pytest.raises(ValueError, match="aliases an ingestion input"):
        pipeline.process_session(
            session_dir,
            manifest_dir=session_dir,
            name=name,
            overwrite=True,
        )


@pytest.mark.parametrize("strict", [False, True])
def test_failure_manifest_inventories_extracted_session_directory(
    tmp_path,
    monkeypatch,
    strict,
):
    def parse_flash(path, *, clock_reference_set, issue_collector, capture):
        capture.session_count = 1
        return (
            [pipeline.Session(ordinal=0)],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

    def write_session(session, destination):
        destination.mkdir(parents=True)
        (destination / pipeline.PACKET_MAP_FILENAME).write_bytes(b"packet map")

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", parse_flash)
    monkeypatch.setattr(pipeline, "write_uncrater_session", write_session)
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic decoder failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        pipeline.process_flash(
            tmp_path / "flash",
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
            issue_collector=IssueCollector("strict") if strict else None,
        )

    manifest = json.loads(
        (tmp_path / "sessions" / "flash.json").read_text("ascii")
    )
    session = manifest["sessions"][0]
    assert session["session_dir"] == "session_000"
    assert "packet_map" in session["output_artifacts"]
    assert session["status"] == "failed"
    assert session["status_issue_codes"] == ["pipeline.flash_failed"]
    assert session["issue_counts"] == {"pipeline.flash_failed": 1}
    assert session["issues"][0]["code"] == "pipeline.flash_failed"
    assert session["packet_map_status"] == "verified"
    assert session["packet_map_format_version"] == 1
    assert session["contracts"]["output_layout_version"] == 4
    failed_families = [
        status
        for status in session["family_statuses"]
        if status["reason"] == "stage_failed_before_decode"
    ]
    assert failed_families
    assert all(
        status["coverage"] == "invalid_or_dropped"
        and status["quality"] == "failed"
        and status["issue_codes"] == ["pipeline.flash_failed"]
        for status in failed_families
    )
    assert (
        tmp_path / "sessions" / "session_000" / "session.json"
    ).is_file()


@pytest.mark.parametrize("strict", [False, True])
def test_session_decode_failure_manifest_has_complete_status_contracts(
    tmp_path,
    monkeypatch,
    strict,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic decoder failure")
        ),
    )

    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        pipeline.process_session(
            session_dir,
            landing_time_file=write_landing_reference(tmp_path),
            manifest_dir=tmp_path / "manifests",
            rederive_telemetry=False,
            issue_collector=IssueCollector("strict") if strict else None,
        )

    manifest = json.loads(
        (tmp_path / "manifests" / "session_000.json").read_text("ascii")
    )
    assert manifest["status"] == "failed"
    assert manifest["status_issue_codes"] == ["pipeline.session_failed"]
    assert manifest["issue_counts"] == {"pipeline.session_failed": 1}
    assert manifest["issues"][0]["code"] == "pipeline.session_failed"
    assert manifest["contracts"]["output_layout_version"] == 4
    assert manifest["stage_counts"]["decode"] == {
        "input_packets": 0,
        "valid_packets": 0,
        "invalid_packets": 0,
    }
    failed_families = [
        status
        for status in manifest["family_statuses"]
        if status["reason"] == "stage_failed_before_decode"
    ]
    assert failed_families
    assert all(
        status["issue_codes"] == ["pipeline.session_failed"]
        for status in failed_families
    )


def test_flash_explicit_overwrite_replaces_existing_session_coherently(
    tmp_path,
    monkeypatch,
):
    sessions_root = tmp_path / "sessions"
    existing = sessions_root / "session_000"
    existing.mkdir(parents=True)
    (sessions_root / "flash.json").write_bytes(b"old run\n")

    def parse_flash(path, *, clock_reference_set, issue_collector, capture):
        capture.session_count = 1
        return (
            [pipeline.Session(ordinal=0)],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

    calls = []

    def write_session(session, destination, *, overwrite=False):
        calls.append(overwrite)

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", parse_flash)
    monkeypatch.setattr(pipeline, "write_uncrater_session", write_session)
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: concrete_products(),
    )

    result = pipeline.process_flash(
        tmp_path / "flash",
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=sessions_root,
        overwrite=True,
    )

    assert calls == [True]
    assert result.overwrite is True
    assert json.loads((sessions_root / "flash.json").read_text("ascii"))[
        "overwrite"
    ] is True
