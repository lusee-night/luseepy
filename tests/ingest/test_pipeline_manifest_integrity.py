from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lusee.ingest import fits_writer, hdf5_writer, pipeline
from lusee.ingest.decode import Products
from lusee.ingest.issues import IssueAction, IssueCollector, IssueSeverity
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ValidatedCounts,
)


def write_landing_reference(tmp_path: Path) -> Path:
    path = tmp_path / "landing.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": "2027-05-01T00:00:00",
        "time_scale": "utc",
        "clocks": {
            "spectrometer": {"clock_reference_raw_seconds": 0.0},
        },
        "source": "synthetic manifest-integrity test",
        "assumed": True,
    }), encoding="ascii")
    return path


def concrete_products() -> Products:
    provenance = DecodeProvenance(
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
    )
    return Products(
        decode_provenance=provenance,
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=0,
            valid_packets=0,
            product_rows=(),
        ),
    )


def test_flash_capture_uses_diagnostic_bytes_for_source_identity(
    tmp_path,
    monkeypatch,
):
    payload = b"exact bank bytes\x00\xff"
    bank_path = tmp_path / "FFFFFFFE"
    bank_path.write_bytes(payload)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    frame = object()
    seen = []

    def diagnostic(path, *, bank, issue_collector):
        seen.append((path, bank, issue_collector))
        return SimpleNamespace(
            input_size_bytes=len(payload),
            input_sha256=expected_sha256,
            frames=(frame,),
        )

    monkeypatch.setattr(pipeline, "parse_bank_file_diagnostic", diagnostic)
    monkeypatch.setattr(
        pipeline,
        "parse_bank_file",
        lambda *args, **kwargs: pytest.fail("non-diagnostic parser was used"),
    )
    collector = IssueCollector()
    capture = pipeline._FlashParseCapture()

    frames = pipeline._frames_from_bank(
        bank_path,
        bank="b02",
        issue_collector=collector,
        capture=capture,
    )

    assert frames == (frame,)
    assert seen == [(bank_path, "b02", collector)]
    assert capture.source_fingerprint == {
        "b02/FFFFFFFE": {
            "size_bytes": len(payload),
            "sha256": expected_sha256,
        },
    }


@pytest.mark.parametrize("input_kind", ["missing", "file"])
def test_flash_root_preflight_does_not_create_a_manifest(tmp_path, input_kind):
    flash_path = tmp_path / "flash"
    if input_kind == "file":
        flash_path.write_bytes(b"not a directory")

    with pytest.raises(NotADirectoryError):
        pipeline.process_flash(
            flash_path,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
        )

    assert not (tmp_path / "sessions" / "flash.json").exists()


def test_present_nonregular_science_bank_is_partial(tmp_path, monkeypatch):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / pipeline.SCIENCE_BANKS[0]
    bank_dir.mkdir(parents=True)
    (bank_dir / pipeline.BANK_FILENAME).symlink_to("missing-bank")
    monkeypatch.setattr(
        pipeline,
        "detect_sw_version",
        lambda packets, *, issue_collector=None: None,
    )
    monkeypatch.setattr(
        pipeline,
        "assign_identities",
        lambda packets, **kwargs: packets,
    )
    monkeypatch.setattr(
        pipeline,
        "split_sessions",
        lambda packets, *, issue_collector=None: [],
    )
    monkeypatch.setattr(
        pipeline,
        "assign_telemetry_to_sessions",
        lambda *args, **kwargs: None,
    )
    collector = IssueCollector()

    sessions, _telemetry, _unassigned = pipeline.parse_flash(
        flash_dir,
        landing_time_file=write_landing_reference(tmp_path),
        issue_collector=collector,
    )

    assert sessions == []
    issue = next(
        issue
        for issue in collector.issues
        if issue.code == "source.science_bank_unreadable"
    )
    assert issue.bank == pipeline.SCIENCE_BANKS[0]
    assert issue.action is IssueAction.REJECTED


@pytest.mark.parametrize(
    ("allow_changed", "expected_status", "expected_action", "decode_calls"),
    [
        (False, "broken", IssueAction.REJECTED, 0),
        (True, "unavailable", IssueAction.OVERRIDDEN, 1),
    ],
)
def test_rederive_changed_b01_requires_recorded_override(
    tmp_path,
    monkeypatch,
    allow_changed,
    expected_status,
    expected_action,
    decode_calls,
):
    bank_dir = tmp_path / pipeline.TELEMETRY_BANK
    bank_dir.mkdir()
    payload = b"changed b01 bytes"
    (bank_dir / pipeline.BANK_FILENAME).write_bytes(payload)
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        pipeline,
        "parse_bank_file_diagnostic",
        lambda *args, **kwargs: SimpleNamespace(
            input_size_bytes=len(payload),
            input_sha256=observed_sha256,
            frames=(),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "reassemble_logical_packets",
        lambda *args, **kwargs: iter(()),
    )
    calls = []

    def decode(packets, *, issue_collector):
        calls.append((packets, issue_collector))
        issue = issue_collector.record(
            code="telemetry_adapter.synthetic_unavailable",
            severity="error",
            stage="telemetry_decode",
            message="synthetic decoder is unavailable",
            action="kept",
        )
        return pipeline.telemetry_mod.TelemetryDecodeResult(
            input_source="b01",
            input_state=pipeline.telemetry_mod.TelemetryInputState.PRESENT_EMPTY,
            decoder_status=(
                pipeline.telemetry_mod.TelemetryDecoderStatus.UNAVAILABLE
            ),
            coverage=pipeline.telemetry_mod.TelemetryCoverage.UNAVAILABLE,
            issues=(issue,),
        )

    monkeypatch.setattr(pipeline.telemetry_mod, "decode_b01_packets", decode)
    collector = IssueCollector()
    result = pipeline._rederive_telemetry_from_flash(
        tmp_path,
        window_lower_elapsed_seconds=None,
        window_upper_elapsed_seconds=None,
        clock_reference_set=None,
        issue_collector=collector,
        expected_source_fingerprint={
            f"{pipeline.TELEMETRY_BANK}/{pipeline.BANK_FILENAME}": {
                "size_bytes": len(payload),
                "sha256": "0" * 64,
            },
        },
        allow_changed_flash_source=allow_changed,
    )

    assert result.decoder_status.value == expected_status
    assert len(calls) == decode_calls
    changed = next(
        issue for issue in result.issues if issue.code == "source.flash_changed"
    )
    assert changed.action is expected_action
    assert changed.severity is (
        IssueSeverity.WARNING if allow_changed else IssueSeverity.ERROR
    )


def test_manifest_refuses_an_existing_destination(tmp_path):
    destination = tmp_path / "session.json"
    destination.write_bytes(b"existing manifest\n")
    result = pipeline.SessionResult(
        session_ordinal=0,
        session_name="session_000",
        source_path=str(tmp_path),
        source_kind="session",
    )

    with pytest.raises(FileExistsError):
        pipeline.write_manifest(result, destination)

    assert destination.read_bytes() == b"existing manifest\n"


def test_public_flash_manifest_writer_requires_the_canonical_destination(
    tmp_path,
):
    result = pipeline.FlashResult(
        flash_result_id=f"flash-{'a' * 16}",
        input_identity_sha256=None,
        input_identity_unavailable_reason="no_captured_bank_sources",
        source_path=str(tmp_path / "flash"),
        source_fingerprint={},
    )
    sessions_root = tmp_path / "sessions"

    with pytest.raises(ValueError, match="sessions_root/flash.json"):
        pipeline.write_flash_manifest(
            result,
            tmp_path / "external" / "flash.json",
            sessions_root=sessions_root,
        )

    destination = pipeline.write_flash_manifest(
        result,
        sessions_root / "flash.json",
        sessions_root=sessions_root,
    )
    manifest = json.loads(destination.read_text("ascii"))
    assert manifest["locator_contract"]["base"] == "sessions_root"


def test_legacy_size_mtime_fingerprint_is_read_only(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    legacy_fingerprint = {
        "b01/FFFFFFFE": {"size": 17, "mtime": 1234.5},
    }
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_schema_version": 3,
        "flash_source_fingerprint": legacy_fingerprint,
    }), encoding="ascii")

    manifest = pipeline._read_in_session_manifest(session_dir, strict=True)

    assert manifest is not None
    assert manifest["flash_source_fingerprint"] == legacy_fingerprint
    assert manifest["_legacy_flash_source_fingerprint"] is True

    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / pipeline.TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / pipeline.BANK_FILENAME).write_bytes(b"present source")
    telemetry = pipeline._rederive_telemetry_from_flash(
        flash_dir,
        window_lower_elapsed_seconds=None,
        window_upper_elapsed_seconds=None,
        clock_reference_set=None,
        expected_source_fingerprint=legacy_fingerprint,
    )
    assert (
        telemetry.decoder_status
        is pipeline.telemetry_mod.TelemetryDecoderStatus.BROKEN
    )
    assert telemetry.issues[0].code == "source.flash_fingerprint_invalid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flash_result_id", "flash-not-hex"),
        ("flash_manifest_sha256", "A" * 64),
    ],
)
def test_session_manifest_rejects_invalid_flash_link_identifiers(
    tmp_path,
    field,
    value,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_kind": "session",
        "manifest_schema_version": 3,
        field: value,
    }), encoding="ascii")

    with pytest.raises(ValueError, match=field):
        pipeline._read_in_session_manifest(session_dir, strict=True)


@pytest.mark.parametrize(
    "missing_field",
    ["flash_result_id", "flash_manifest_sha256"],
)
def test_session_manifest_requires_complete_flash_link(tmp_path, missing_field):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = {
        "manifest_kind": "session",
        "manifest_schema_version": 3,
        "flash_result_id": f"flash-{'a' * 16}",
        "flash_manifest_sha256": "b" * 64,
        "flash_input_identity_sha256": "c" * 64,
    }
    del manifest[missing_field]
    (session_dir / "session.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="must include both"):
        pipeline._read_in_session_manifest(session_dir, strict=True)


@pytest.mark.parametrize("damage", ["malformed_fingerprint", "wrong_identity"])
def test_session_manifest_binds_source_identity_without_sibling(
    tmp_path,
    damage,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    payload = b"captured source"
    fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    manifest = {
        "manifest_kind": "session",
        "manifest_schema_version": 3,
        "flash_result_id": f"flash-{'a' * 16}",
        "flash_manifest_sha256": "b" * 64,
        "flash_input_identity_sha256": (
            pipeline._flash_source_identity_sha256(fingerprint)
        ),
        "flash_source_fingerprint": fingerprint,
    }
    if damage == "malformed_fingerprint":
        manifest["flash_source_fingerprint"] = []
    else:
        manifest["flash_input_identity_sha256"] = "c" * 64
    (session_dir / "session.json").write_text(
        json.dumps(manifest),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="fingerprint|source identity"):
        pipeline._read_in_session_manifest(session_dir, strict=True)


@pytest.mark.parametrize(
    "mismatch",
    [
        "digest",
        "result_id",
        "source_identity",
        "locator",
        "clock_reference",
        "telemetry_window",
        "h5_locator",
        "fits_locator",
        "plot_locator",
        "start_time_utc",
        "telemetry_rows",
        "unassigned_telemetry_rows",
        "telemetry_provenance",
    ],
)
def test_session_manifest_verifies_sibling_flash_manifest(tmp_path, mismatch):
    sessions_root = tmp_path / "sessions"
    session_dir = sessions_root / "session_000"
    session_dir.mkdir(parents=True)
    flash_result_id = f"flash-{'a' * 16}"
    source_payload = b"captured source"
    source_fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
        },
    }
    input_identity = pipeline._flash_source_identity_sha256(
        source_fingerprint
    )
    clock_reference = {"source_sha256": "d" * 64}
    linked_session_dir = "other" if mismatch == "locator" else "session_000"
    flash_payload = json.dumps({
        "manifest_kind": "flash",
        "manifest_schema_version": 3,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": "flash.json",
            "external_copies_are_mirrors": True,
        },
        "flash_result_id": flash_result_id,
        "input_identity_sha256": input_identity,
        "input_identity_unavailable_reason": None,
        "source_fingerprint": source_fingerprint,
        "clock_reference": clock_reference,
        "sessions": [{
            "session_name": "session_000",
            "session_ordinal": 0,
            "session_dir": linked_session_dir,
            "h5_path": "../h5/session_000.h5",
            "fits_path": "../fits/session_000.fits",
            "plot_paths": ["../plots/session_000/spectra.png"],
            "start_time_utc": "2026-08-25T00:00:00.000",
            "n_telemetry_rows": 2,
            "n_unassigned_telemetry_rows": 1,
            "telemetry_provenance": {"source": "b01"},
            "telemetry_window_lower_elapsed_seconds": 1.0,
        }],
    }).encode("ascii")
    (sessions_root / "flash.json").write_bytes(flash_payload)
    session_manifest = {
        "manifest_kind": "session",
        "manifest_schema_version": 3,
        "flash_result_id": flash_result_id,
        "flash_manifest_sha256": hashlib.sha256(flash_payload).hexdigest(),
        "flash_input_identity_sha256": input_identity,
        "flash_source_fingerprint": source_fingerprint,
        "session_name": "session_000",
        "session_ordinal": 0,
        "clock_reference": clock_reference,
        "h5_path": "../../h5/session_000.h5",
        "fits_path": "../../fits/session_000.fits",
        "plot_paths": ["../../plots/session_000/spectra.png"],
        "start_time_utc": "2026-08-25T00:00:00.000",
        "n_telemetry_rows": 2,
        "n_unassigned_telemetry_rows": 1,
        "telemetry_provenance": {"source": "b01"},
        "telemetry_window_lower_elapsed_seconds": 1.0,
    }
    if mismatch == "digest":
        session_manifest["flash_manifest_sha256"] = "d" * 64
    elif mismatch == "result_id":
        session_manifest["flash_result_id"] = f"flash-{'e' * 16}"
    elif mismatch == "source_identity":
        session_manifest["flash_input_identity_sha256"] = "f" * 64
    elif mismatch == "clock_reference":
        session_manifest["clock_reference"] = {"source_sha256": "e" * 64}
    elif mismatch == "telemetry_window":
        session_manifest["telemetry_window_lower_elapsed_seconds"] = 2.0
    elif mismatch == "h5_locator":
        session_manifest["h5_path"] = "../../h5/other.h5"
    elif mismatch == "fits_locator":
        session_manifest["fits_path"] = "../../fits/other.fits"
    elif mismatch == "plot_locator":
        session_manifest["plot_paths"] = [
            "../../plots/session_000/other.png"
        ]
    elif mismatch == "start_time_utc":
        session_manifest["start_time_utc"] = "2026-08-25T00:00:01.000"
    elif mismatch == "telemetry_rows":
        session_manifest["n_telemetry_rows"] = 3
    elif mismatch == "unassigned_telemetry_rows":
        session_manifest["n_unassigned_telemetry_rows"] = 2
    else:
        session_manifest["telemetry_provenance"] = {"source": "sidecar"}
    (session_dir / "session.json").write_text(
        json.dumps(session_manifest),
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="does not match"):
        pipeline._read_in_session_manifest(session_dir, strict=True)


def test_process_session_uses_source_only_flash_identity(tmp_path, monkeypatch):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    landing_time_file = write_landing_reference(tmp_path)
    clock_reference = pipeline._load_landing_reference(
        landing_time_file
    ).as_record()
    flash_result_id = f"flash-{'a' * 16}"
    source_payload = b"captured source"
    source_fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": len(source_payload),
            "sha256": hashlib.sha256(source_payload).hexdigest(),
        },
    }
    input_identity = pipeline._flash_source_identity_sha256(
        source_fingerprint
    )
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_kind": "session",
        "manifest_schema_version": 3,
        "flash_result_id": flash_result_id,
        "flash_manifest_sha256": "b" * 64,
        "flash_input_identity_sha256": input_identity,
        "flash_source_fingerprint": source_fingerprint,
        "clock_reference": clock_reference,
    }), encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: concrete_products(),
    )
    requests = []

    def write_hdf5(request, destination):
        requests.append(request)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic hdf5")

    monkeypatch.setattr(hdf5_writer, "write_hdf5", write_hdf5)
    result = pipeline.process_session(
        session_dir,
        landing_time_file=landing_time_file,
        h5_dir=tmp_path / "h5",
        rederive_telemetry=False,
    )

    assert len(requests) == 1
    provenance = requests[0].run_provenance
    assert provenance.input_identity == input_identity
    assert provenance.input_identity_kind == "flash_source_sha256"
    assert provenance.input_identity != flash_result_id
    assert result.flash_result_id == flash_result_id
    assert result.flash_input_identity_sha256 == input_identity


def test_manifest_only_session_does_not_claim_persisted_rows(tmp_path):
    products = concrete_products()
    products.housekeeping = [
        SimpleNamespace(
            provenance=SimpleNamespace(decoder_issue_ids=()),
        ),
    ]

    result = pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name="session_000",
        ordinal=0,
        h5_dir=None,
        fits_dir=None,
        plots_dir=None,
        manifest_dir=None,
        issue_collector=IssueCollector(),
        products=products,
    )

    assert result.decoded_rows_by_family["housekeeping"] == 1
    assert result.persisted_rows_by_family["housekeeping"] == 0
    housekeeping = next(
        status
        for status in result.family_statuses
        if status["family"] == "housekeeping"
    )
    assert housekeeping["decoded_rows"] == 1
    assert housekeeping["persisted_rows"] == 0
    assert housekeeping["coverage"] == "decoded_not_persisted"


def test_failed_decoder_quality_is_output_choice_invariant(tmp_path):
    products = concrete_products()
    products.quality_status = DataQuality.FAILED

    result = pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name="session_000",
        ordinal=0,
        h5_dir=None,
        fits_dir=None,
        plots_dir=None,
        manifest_dir=None,
        issue_collector=IssueCollector(),
        products=products,
    )

    assert result.status == "failed"
    assert result.status_issue_codes == ["decode.no_usable_products"]
    assert result.issue_counts == {"decode.no_usable_products": 1}


def test_session_targeted_parse_issue_marks_only_its_session(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    flash_dir.mkdir()

    def parse_flash(
        path,
        *,
        clock_reference_set,
        issue_collector,
        capture,
    ):
        capture.session_count = 2
        issue_collector.record(
            code="session_start.synthetic_damage",
            severity="warning",
            stage="session_start",
            message="session zero has damaged startup metadata",
            action="kept",
            session="ordinal:0",
        )
        return (
            [pipeline.Session(ordinal=0), pipeline.Session(ordinal=1)],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

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

    result = pipeline.process_flash(
        flash_dir,
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=tmp_path / "sessions",
    )

    assert result.status == "partial"
    assert result[0].status == "partial"
    assert result[0].status_issue_codes == [
        "session_start.synthetic_damage"
    ]
    assert result[1].status == "clean"
    assert result[1].status_issue_codes == []


def install_second_writer_failure(monkeypatch):
    def write_hdf5(request, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"surviving HDF5 artifact")

    def write_fits(request, path):
        raise RuntimeError("synthetic FITS failure")

    monkeypatch.setattr(hdf5_writer, "write_hdf5", write_hdf5)
    monkeypatch.setattr(fits_writer, "write_fits", write_fits)


def test_session_failure_manifest_inventories_surviving_hdf5(
    tmp_path,
    monkeypatch,
):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: concrete_products(),
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: None,
    )
    install_second_writer_failure(monkeypatch)

    with pytest.raises(RuntimeError, match="synthetic FITS failure"):
        pipeline.process_session(
            session_dir,
            landing_time_file=write_landing_reference(tmp_path),
            h5_dir=tmp_path / "h5",
            fits_dir=tmp_path / "fits",
            manifest_dir=tmp_path / "manifests",
            rederive_telemetry=False,
        )

    manifest = json.loads(
        (tmp_path / "manifests" / "session_000.json").read_text("ascii")
    )
    artifact = manifest["output_artifacts"]["hdf5"]
    h5_path = tmp_path / "h5" / "session_000.h5"
    assert manifest["status"] == "failed"
    assert artifact["size_bytes"] == h5_path.stat().st_size
    assert artifact["sha256"] == hashlib.sha256(h5_path.read_bytes()).hexdigest()
    assert manifest["persisted_rows_by_family"] == {
        family: 0 for family in manifest["persisted_rows_by_family"]
    }


def test_flash_failure_manifest_inventories_surviving_hdf5(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    flash_dir.mkdir()
    source_sha256 = hashlib.sha256(b"source").hexdigest()

    def parse_flash(
        path,
        *,
        clock_reference_set,
        issue_collector,
        capture,
    ):
        capture.record_bank(
            bank="b02",
            size_bytes=6,
            sha256=source_sha256,
            recovered_frames=1,
        )
        capture.session_count = 1
        return (
            [pipeline.Session(ordinal=0)],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

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
    install_second_writer_failure(monkeypatch)

    with pytest.raises(RuntimeError, match="synthetic FITS failure"):
        pipeline.process_flash(
            flash_dir,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
            h5_dir=tmp_path / "h5",
            fits_dir=tmp_path / "fits",
        )

    manifest = json.loads(
        (tmp_path / "sessions" / "flash.json").read_text("ascii")
    )
    session = manifest["sessions"][0]
    h5_path = tmp_path / "h5" / "session_000.h5"
    assert manifest["status"] == "failed"
    assert session["status"] == "failed"
    assert session["committed_artifacts"] == ["hdf5"]
    assert session["output_artifacts"]["hdf5"]["size_bytes"] == (
        h5_path.stat().st_size
    )
    assert session["output_artifacts"]["hdf5"]["sha256"] == (
        hashlib.sha256(h5_path.read_bytes()).hexdigest()
    )


def test_flash_manifests_are_final_once_portable_and_digest_linked(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "private" / "flash"
    flash_dir.mkdir(parents=True)
    source_payload = b"source bytes"
    source_sha256 = hashlib.sha256(source_payload).hexdigest()
    def parse_flash(
        path,
        *,
        clock_reference_set,
        issue_collector,
        capture,
    ):
        capture.record_bank(
            bank="b02",
            size_bytes=len(source_payload),
            sha256=source_sha256,
            recovered_frames=2,
        )
        capture.science_logical_packets = 1
        capture.identity_input_packets = 1
        capture.identity_kept_packets = 1
        capture.session_count = 1
        issue_collector.record(
            code="framing.synthetic_damage",
            severity="warning",
            stage="framing",
            message="one synthetic frame was dropped",
            action="dropped",
            bank="b02",
        )
        return (
            [pipeline.Session(ordinal=0)],
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        )

    def write_session(session, destination):
        destination.mkdir(parents=True)
        return destination

    def read_session(
        path,
        *,
        strict=False,
        schema_variant=None,
        issue_collector,
    ):
        return concrete_products()

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", parse_flash)
    monkeypatch.setattr(pipeline, "write_uncrater_session", write_session)
    monkeypatch.setattr(pipeline, "read_uncrater_session", read_session)
    sessions_root = tmp_path / "sessions"
    manifests = tmp_path / "manifests"
    result = pipeline.process_flash(
        flash_dir,
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=sessions_root,
        manifest_dir=manifests,
    )

    assert isinstance(result, pipeline.FlashResult)
    assert len(result) == 1
    assert list(result) == result.session_results
    assert result[0] is result.session_results[0]
    assert result.status == "partial"
    assert result.status_issue_codes == ["framing.synthetic_damage"]
    assert result[0].status == "clean"
    assert result[0].status_issue_codes == []
    assert result[0].input_packet_count == 0
    assert result.stage_counts["input"] == {
        "bank_files": 1,
        "bytes": len(source_payload),
    }
    assert result.stage_counts["framing"]["recovered_frames"] == 2
    assert result.stage_counts["framing"]["dropped_frames"] == 1

    internal_run = (sessions_root / "flash.json").read_bytes()
    external_run = (manifests / "flash.json").read_bytes()
    assert internal_run == external_run
    assert hashlib.sha256(internal_run).hexdigest() == result.manifest_sha256
    run_manifest = json.loads(internal_run)
    assert "manifest_sha256" not in run_manifest
    assert "manifest_paths" not in run_manifest
    assert run_manifest["locator_contract"] == {
        "version": 1,
        "base": "sessions_root",
        "canonical_manifest": "flash.json",
        "external_copies_are_mirrors": True,
    }
    assert run_manifest["decoder_provenance"][0]["selected_schema_id"] == 0x307
    assert str(tmp_path) not in internal_run.decode("ascii")

    for manifest_path in (
        sessions_root / "session_000" / "session.json",
        manifests / "session_000.json",
    ):
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        assert str(tmp_path) not in manifest_bytes.decode("ascii")
        assert "n_packets" not in manifest
        assert manifest["flash_result_id"] == result.flash_result_id
        assert manifest["flash_manifest_sha256"] == result.manifest_sha256
