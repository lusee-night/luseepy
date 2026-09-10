from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lusee.ingest import fits_writer, hdf5_writer, pipeline, viz
from lusee.ingest.decode import Products
from lusee.ingest.issues import IssueAction, IssueCollector
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ValidatedCounts,
)


ABSENT_TELEMETRY_DIAGNOSTICS = {
    "telemetry_status": "absent",
    "telemetry_reason": None,
    "telemetry_source": None,
    "n_telemetry_rows": 0,
}


@pytest.mark.parametrize(
    "record",
    [
        ABSENT_TELEMETRY_DIAGNOSTICS,
        {
            "telemetry_status": "decoded",
            "telemetry_reason": None,
            "telemetry_source": "b01_0x314",
            "n_telemetry_rows": 0,
        },
        {
            "telemetry_status": "decoded",
            "telemetry_reason": None,
            "telemetry_source": "legacy_binary_sidecar",
            "n_telemetry_rows": 3,
        },
        {
            "telemetry_status": "skipped",
            "telemetry_reason": "decoder unavailable",
            "telemetry_source": "b01_0x314",
            "n_telemetry_rows": 0,
        },
    ],
)
def test_current_manifest_accepts_coherent_telemetry_diagnostics(record):
    pipeline._validate_manifest_telemetry_fields(record)


@pytest.mark.parametrize(
    "update",
    [
        {"telemetry_status": "unknown"},
        {"telemetry_source": "unknown"},
        {"n_telemetry_rows": True},
        {"n_telemetry_rows": -1},
        {"telemetry_source": "b01_0x314"},
        {
            "telemetry_status": "decoded",
            "telemetry_source": None,
        },
        {
            "telemetry_status": "skipped",
            "telemetry_source": "b01_0x314",
            "telemetry_reason": None,
        },
        {
            "telemetry_status": "skipped",
            "telemetry_source": "b01_0x314",
            "telemetry_reason": "decoder unavailable",
            "n_telemetry_rows": 1,
        },
    ],
)
def test_current_manifest_rejects_invalid_telemetry_diagnostics(update):
    record = dict(ABSENT_TELEMETRY_DIAGNOSTICS)
    record.update(update)

    with pytest.raises(ValueError, match="telemetry"):
        pipeline._validate_manifest_telemetry_fields(record)


def test_legacy_v3_telemetry_diagnostics_remain_readable():
    pipeline._validate_manifest_telemetry_fields({
        "telemetry_decoder_status": "not_needed",
        "telemetry_coverage": "absent",
        "telemetry_source": None,
        "n_telemetry_rows": 0,
    })


def test_current_session_manifest_rejects_missing_discriminators(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_kind": "session",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "telemetry_source": "b01_0x314",
        "n_telemetry_rows": 2,
    }), encoding="ascii")

    with pytest.raises(ValueError, match="telemetry diagnostics"):
        pipeline._read_in_session_manifest(session_dir, strict=True)


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
        lambda packets, *, issue_collector=None, schema_resolution=None: None,
    )
    monkeypatch.setattr(
        pipeline,
        "assign_identities",
        lambda packets, **kwargs: packets,
    )
    monkeypatch.setattr(
        pipeline,
        "split_sessions",
        lambda packets, *, issue_collector=None, schema_resolution=None: [],
    )
    collector = IssueCollector()

    sessions, _telemetry = pipeline.parse_flash(
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
        "telemetry_decoder_status": "not_needed",
        "telemetry_coverage": "absent",
        "telemetry_source": None,
        "n_telemetry_rows": 0,
    }), encoding="ascii")

    manifest = pipeline._read_in_session_manifest(session_dir, strict=True)

    assert manifest is not None
    assert manifest["flash_source_fingerprint"] == legacy_fingerprint
    assert manifest["_legacy_flash_source_fingerprint"] is True


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
        **ABSENT_TELEMETRY_DIAGNOSTICS,
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
        **ABSENT_TELEMETRY_DIAGNOSTICS,
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
        **ABSENT_TELEMETRY_DIAGNOSTICS,
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
        "h5_locator",
        "fits_locator",
        "plot_locator",
        "start_time_utc",
        "telemetry_status",
        "telemetry_reason",
        "telemetry_source",
        "telemetry_rows",
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
            "telemetry_status": "decoded",
            "telemetry_reason": None,
            "telemetry_source": "b01_0x314",
            "n_telemetry_rows": 2,
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
        "telemetry_status": "decoded",
        "telemetry_reason": None,
        "telemetry_source": "b01_0x314",
        "n_telemetry_rows": 2,
    }
    if mismatch == "digest":
        session_manifest["flash_manifest_sha256"] = "d" * 64
    elif mismatch == "result_id":
        session_manifest["flash_result_id"] = f"flash-{'e' * 16}"
    elif mismatch == "source_identity":
        session_manifest["flash_input_identity_sha256"] = "f" * 64
    elif mismatch == "clock_reference":
        session_manifest["clock_reference"] = {"source_sha256": "e" * 64}
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
    elif mismatch == "telemetry_status":
        session_manifest.update({
            "telemetry_status": "skipped",
            "telemetry_reason": "decoder unavailable",
            "n_telemetry_rows": 0,
        })
    elif mismatch == "telemetry_reason":
        session_manifest.update({
            "telemetry_status": "skipped",
            "telemetry_reason": "different decoder failure",
            "n_telemetry_rows": 0,
        })
    elif mismatch == "telemetry_source":
        session_manifest["telemetry_source"] = "legacy_binary_sidecar"
    elif mismatch == "telemetry_rows":
        session_manifest["n_telemetry_rows"] = 3
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
        **ABSENT_TELEMETRY_DIAGNOSTICS,
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


def test_existing_plot_directory_is_refused_before_product_write(
    tmp_path,
    monkeypatch,
):
    plot_dest = tmp_path / "plots" / "session_000"
    plot_dest.mkdir(parents=True)
    sentinel = plot_dest / "stale.png"
    sentinel.write_bytes(b"stale")

    monkeypatch.setattr(
        hdf5_writer,
        "write_hdf5",
        lambda *args, **kwargs: pytest.fail("HDF5 write preceded plot refusal"),
    )

    with pytest.raises(FileExistsError) as error:
        pipeline._process_one_session(
            session_dir=tmp_path / "session",
            name="session_000",
            ordinal=0,
            h5_dir=tmp_path / "h5",
            fits_dir=None,
            plots_dir=tmp_path / "plots",
            manifest_dir=None,
            issue_collector=IssueCollector(),
            clock_reference_set=pipeline._load_landing_reference(
                write_landing_reference(tmp_path)
            ),
            products=concrete_products(),
        )

    assert error.value.args == (plot_dest,)
    assert sentinel.read_bytes() == b"stale"


def test_plot_overwrite_replaces_directory(tmp_path, monkeypatch):
    plot_dest = tmp_path / "plots" / "session_000"
    plot_dest.mkdir(parents=True)
    stale = plot_dest / "stale.png"
    stale.write_bytes(b"stale")

    def write_hdf5(request, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"synthetic hdf5")

    def plot_session(h5_path, destination, *, plots):
        assert h5_path.is_file()
        assert destination == plot_dest
        assert not stale.exists()
        path = destination / "new.png"
        path.write_bytes(b"new")
        return [path]

    monkeypatch.setattr(hdf5_writer, "write_hdf5", write_hdf5)
    monkeypatch.setattr(viz, "plot_session", plot_session)

    result = pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name="session_000",
        ordinal=0,
        h5_dir=tmp_path / "h5",
        fits_dir=None,
        plots_dir=tmp_path / "plots",
        manifest_dir=None,
        issue_collector=IssueCollector(),
        clock_reference_set=pipeline._load_landing_reference(
            write_landing_reference(tmp_path)
        ),
        products=concrete_products(),
        overwrite=True,
    )

    assert result.plot_paths == [str((plot_dest / "new.png").resolve())]
    assert not stale.exists()


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


def test_session_writer_exception_leaves_output_for_manual_rerun(
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
        )

    h5_path = tmp_path / "h5" / "session_000.h5"
    assert h5_path.read_bytes() == b"surviving HDF5 artifact"
    assert not (tmp_path / "manifests" / "session_000.json").exists()


def test_flash_writer_exception_leaves_output_for_manual_rerun(
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

    h5_path = tmp_path / "h5" / "session_000.h5"
    assert h5_path.read_bytes() == b"surviving HDF5 artifact"
    assert not (tmp_path / "sessions" / "flash.json").exists()
    assert not (
        tmp_path / "sessions" / "session_000" / "session.json"
    ).exists()


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
