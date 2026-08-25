from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import pytest

from lusee.ingest import pipeline
from lusee.ingest.decode import Products
from lusee.ingest.products import DecodeProvenance, ExecutionMode


def write_landing_reference(tmp_path):
    path = tmp_path / "landing.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": "2027-05-01T00:00:00",
        "time_scale": "utc",
        "clocks": {
            "spectrometer": {"clock_reference_raw_seconds": 0.0},
        },
        "source": "synthetic pipeline test",
        "assumed": True,
    }), encoding="utf-8")
    return path


def make_products(
    *,
    reported_schema_ids=(0x307,),
    selected_schema_id=0x307,
    binding_key="307",
    schema_assumed=False,
    appid_counts=(),
    issue_counts=(),
):
    input_packet_count = sum(count for _, count in appid_counts)
    provenance = DecodeProvenance(
        decoder_name="uncrater",
        distribution_version="1.0.0",
        decoder_source_commit=None,
        reported_schema_ids=reported_schema_ids,
        selected_schema_id=selected_schema_id,
        binding_key=binding_key,
        schema_variant=None,
        schema_assumed=schema_assumed,
        binding_source_release="test-release",
        binding_source_commit="a" * 40,
        abi_fingerprint=f"{selected_schema_id:064x}",
        execution_mode=ExecutionMode.COLLECT,
        input_packet_count=input_packet_count,
        valid_packet_count=input_packet_count,
        appid_counts=appid_counts,
        issue_counts=issue_counts,
        canonical_report_json="{}",
    )
    return Products(decode_provenance=provenance)


def install_synthetic_flash(monkeypatch, products):
    sessions = [pipeline.Session(ordinal=index) for index in range(len(products))]
    monkeypatch.setattr(
        pipeline,
        "_parse_flash_loaded",
        lambda *args, **kwargs: (
            sessions,
            pipeline.telemetry_mod.TelemetryDecodeResult.absent(),
            None,
        ),
    )

    def materialize(session, session_dir):
        session_dir.mkdir(parents=True)
        return session_dir

    monkeypatch.setattr(pipeline, "write_uncrater_session", materialize)
    reads = []

    def read(session_dir, **kwargs):
        index = int(session_dir.name.removeprefix("session_"))
        reads.append((index, kwargs))
        warnings.warn(f"decode warning {index}", RuntimeWarning, stacklevel=2)
        return products[index]

    monkeypatch.setattr(pipeline, "read_uncrater_session", read)
    return reads


def test_flash_preflight_accepts_same_binding_with_different_evidence(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    flash_dir.mkdir()
    products = [
        make_products(appid_counts=((0x209, 1),)),
        make_products(
            reported_schema_ids=(),
            schema_assumed=True,
            appid_counts=((0x210, 2),),
            issue_counts=(("packet_damage", 1),),
        ),
    ]
    reads = install_synthetic_flash(monkeypatch, products)
    original_worker = pipeline._process_one_session
    worker_products = []

    def worker(**kwargs):
        worker_products.append(kwargs["products"])
        warnings.warn(
            f"writer warning {kwargs['ordinal']}", RuntimeWarning, stacklevel=2
        )
        return original_worker(**kwargs)

    monkeypatch.setattr(pipeline, "_process_one_session", worker)
    landing = write_landing_reference(tmp_path)

    with pytest.warns(RuntimeWarning) as captured:
        results = pipeline.process_flash(
            flash_dir,
            landing_time_file=landing,
            sessions_root=tmp_path / "sessions",
        )

    assert [index for index, _ in reads] == [0, 1]
    assert (
        products[0].decode_provenance.reported_schema_ids
        != products[1].decode_provenance.reported_schema_ids
    )
    assert pipeline._binding_identity(products[0]) == pipeline._binding_identity(
        products[1]
    )
    assert all(
        call["issue_collector"] is reads[0][1]["issue_collector"]
        for _, call in reads
    )
    assert worker_products == products
    assert len(captured) == 4
    assert results[0].n_warnings == 0
    assert results[0].warnings_summary == []
    assert results[1].n_warnings == 0
    manifest = json.loads((
        tmp_path / "sessions" / "session_000" / "session.json"
    ).read_text("ascii"))
    assert manifest["manifest_schema_version"] == 3
    assert manifest["clock_reference"]["assumed"] is True
    assert manifest["clock_reference"]["source_sha256"]


def test_flash_preflight_mismatch_refuses_all_product_writes(
    tmp_path,
    monkeypatch,
):
    flash_dir = tmp_path / "flash"
    flash_dir.mkdir()
    products = [
        make_products(
            reported_schema_ids=(0x305,),
            selected_schema_id=0x305,
            binding_key="305",
        ),
        make_products(),
    ]
    products[0].housekeeping = [
        SimpleNamespace(
            provenance=SimpleNamespace(decoder_issue_ids=()),
        ),
    ]
    reads = install_synthetic_flash(monkeypatch, products)
    worker_calls = []
    manifest_calls = []
    process_one_session = pipeline._process_one_session

    def track_product_writes(**kwargs):
        if any(
            kwargs[directory] is not None
            for directory in ("h5_dir", "fits_dir", "plots_dir")
        ):
            worker_calls.append(kwargs)
        return process_one_session(**kwargs)

    monkeypatch.setattr(pipeline, "_process_one_session", track_product_writes)
    monkeypatch.setattr(
        pipeline,
        "write_manifest",
        lambda *args, **kwargs: manifest_calls.append((args, kwargs)),
    )

    with pytest.warns(RuntimeWarning), pytest.raises(
        RuntimeError,
        match="conservative guard, not a forced input-wide binding",
    ) as error:
        pipeline.process_flash(
            flash_dir,
            landing_time_file=write_landing_reference(tmp_path),
            sessions_root=tmp_path / "sessions",
            h5_dir=tmp_path / "h5",
            fits_dir=tmp_path / "fits",
            plots_dir=tmp_path / "plots",
            manifest_dir=tmp_path / "manifests",
        )

    assert isinstance(error.value.ingest_result, pipeline.FlashResult)
    assert error.value.ingest_result.status == "failed"
    assert error.value.ingest_result.issue_counts == {
        "pipeline.flash_failed": 1
    }

    assert [index for index, _ in reads] == [0, 1]
    assert worker_calls == []
    assert manifest_calls == []
    assert not (tmp_path / "h5").exists()
    assert not (tmp_path / "fits").exists()
    assert not (tmp_path / "plots").exists()
    failure_manifest = json.loads(
        (tmp_path / "manifests" / "flash.json").read_text("ascii")
    )
    assert failure_manifest["status"] == "failed"
    assert failure_manifest["failure"]["stage"] == "decoder_preflight"
    assert failure_manifest["status_issue_codes"] == [
        "pipeline.flash_failed"
    ]
    assert len(failure_manifest["sessions"]) == 2
    for index, session in enumerate(failure_manifest["sessions"]):
        assert session["status"] == "failed"
        assert session["status_issue_codes"] == ["pipeline.flash_failed"]
        assert session["issue_counts"] == {"pipeline.flash_failed": 1}
        assert session["issues"][0]["code"] == "pipeline.flash_failed"
        assert set(session["stage_counts"]) == {
            "decode",
            "persistence",
            "products",
            "session_input",
        }
        assert session["family_statuses"]
        assert session["contracts"]["output_layout_version"] == 4
        if index == 0:
            housekeeping = next(
                status
                for status in session["family_statuses"]
                if status["family"] == "housekeeping"
            )
            assert housekeeping["coverage"] == "decoded_not_persisted"
            assert housekeeping["decoded_rows"] == 1
            assert housekeeping["persisted_rows"] == 0
