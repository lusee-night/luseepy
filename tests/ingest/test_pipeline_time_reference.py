from __future__ import annotations

import json

import pytest

import lusee.ingest.fits_writer as fits_writer
import lusee.ingest.hdf5_writer as hdf5_writer
from lusee.ingest import decode, pipeline
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
def test_present_empty_sidecar_cannot_be_silently_omitted(
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
        "has_decoder",
        lambda: decoder_available,
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: sidecar if decoder_available else None,
    )
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "parse_legacy_sidecar",
        lambda path: {},
    )
    monkeypatch.setattr(
        pipeline,
        "read_uncrater_session",
        lambda *args, **kwargs: valid_products(),
    )

    with pytest.raises(ValueError, match="cannot omit or persist"):
        pipeline.process_session(
            session_dir,
            landing_time_file=landing,
            h5_dir=tmp_path / "h5",
            rederive_telemetry=False,
        )

    assert not (tmp_path / "h5").exists()


def test_present_empty_b01_cannot_be_silently_omitted(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    bank_dir = flash_dir / TELEMETRY_BANK
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")
    monkeypatch.setattr(
        pipeline,
        "_parse_flash_loaded",
        lambda *args, **kwargs: ([pipeline.Session(ordinal=0)], {}, {}),
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
    monkeypatch.setattr(pipeline.telemetry_mod, "has_decoder", lambda: False)

    with pytest.raises(ValueError, match="cannot omit or persist"):
        pipeline.process_flash(
            flash_dir,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
            h5_dir=tmp_path / "h5",
        )

    assert not (tmp_path / "h5").exists()


def test_flash_override_b01_is_detected_without_rederiving(
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
    monkeypatch.setattr(pipeline.telemetry_mod, "has_decoder", lambda: False)

    with pytest.raises(ValueError, match="cannot omit or persist"):
        pipeline.process_session(
            session_dir,
            landing_time_file=write_landing_reference(tmp_path),
            flash_root=flash_dir,
            h5_dir=tmp_path / "h5",
            rederive_telemetry=False,
        )

    assert not (tmp_path / "h5").exists()


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
    result = pipeline._process_one_session(
        session_dir=tmp_path / "session",
        name=None,
        ordinal=4,
        h5_dir=None,
        plots_dir=None,
        manifest_dir=tmp_path / "manifests",
        issue_collector=IssueCollector(),
        clock_reference_set=reference,
        telemetry_input_sources=("b01",),
        telemetry_decoder_status="unavailable",
        products=valid_products(),
    )

    manifest = json.loads((
        tmp_path / "manifests" / f"{result.session_name}.json"
    ).read_text("ascii"))
    assert manifest["manifest_schema_version"] == 3
    assert manifest["clock_reference"] == reference.as_record()
    assert manifest["clock_reference"]["source_sha256"] == reference.source_sha256
    assert manifest["clock_reference"]["assumed"] is True
    assert manifest["packet_map_status"] == "verified"
    assert manifest["packet_map_format_version"] == 1
    assert manifest["raw_flash_provenance_unavailable_reason"] is None
    assert manifest["telemetry_input_sources"] == ["b01"]
    assert manifest["telemetry_decoder_status"] == "unavailable"
    assert result.start_time_utc == "2027-04-30T23:58:21.500000000Z"
