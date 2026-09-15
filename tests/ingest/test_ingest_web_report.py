"""Focused synthetic tests for the static ingest web report."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_layout_v4_hdf5 import make_all_family_request
from test_layout_v4_hdf5_products import make_metadata, make_provenance

from lusee.ingest import hdf5_writer
from lusee.ingest.constants import NPRODUCTS
from lusee.ingest.write_request import FAMILY_TYPES, family_statuses_for_products
from scripts import ingest_web_report as report

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REPORT_SECTIONS = {
    "spectra",
    "tr_spectra",
    "grimm_spectra",
    "telemetry",
    "housekeeping",
    "zoom_spectra",
    "waveform",
    "calibrator",
    "provenance",
    "inventory",
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def make_report_request():
    request = make_all_family_request()
    products = request.products
    template = products.spectra[0]

    additions = (
        (151, 110.0, 7, 250.0),
        (201, 120.0, 15, 1_000.0),
    )
    for uid, raw_seconds, product, offset in additions:
        data = np.full_like(template.data, np.nan)
        data[product] = offset + np.arange(template.nfreq, dtype=np.float32)
        present = np.zeros(NPRODUCTS, dtype=np.bool_)
        present[product] = True
        products.spectra.append(
            replace(
                template,
                data=data,
                product_present=present,
                unique_packet_id=uid,
                raw_seconds=raw_seconds,
                metadata=make_metadata(uid, raw_seconds),
                provenance=make_provenance(
                    uid,
                    raw_seconds,
                    ("metadata", f"normal_product_{product:02d}"),
                ),
            )
        )

    source_packet_count = sum(
        len(row.provenance.source_packets)
        for family, _ in FAMILY_TYPES
        for row in getattr(products, family)
    )
    product_rows = tuple(
        sorted(
            (family, len(getattr(products, family)))
            for family, _ in FAMILY_TYPES
            if getattr(products, family)
        )
    )
    products.validated_counts = replace(
        products.validated_counts,
        input_packets=source_packet_count,
        valid_packets=source_packet_count,
        product_rows=product_rows,
    )
    products.decode_provenance = replace(
        products.decode_provenance,
        input_packet_count=source_packet_count,
        valid_packet_count=source_packet_count,
        appid_counts=((0x270, source_packet_count),),
        canonical_report_json=canonical_json(
            {
                "fixture": "layout-v4-all-products-web-report",
                "packet_count": source_packet_count,
            }
        ),
    )
    return replace(
        request,
        family_statuses=family_statuses_for_products(
            products,
            family_issue_ids={},
        ),
    )


def test_generate_report_covers_all_families_endpoints_and_all_packets(
    tmp_path: Path,
):
    h5_path = tmp_path / "all-products.h5"
    hdf5_writer.write_hdf5(make_report_request(), h5_path)
    dangerous = '<script data-report-test="danger">alert(1) & more</script>'
    with h5py.File(h5_path, "a") as h5:
        h5.attrs["dangerous_report_attribute"] = dangerous

    with h5py.File(h5_path, "r") as h5:
        persisted_overview = {
            "input_packet_count": int(h5.attrs["input_packet_count"]),
            "valid_packet_count": int(h5.attrs["valid_packet_count"]),
            "product_row_count": len(h5["provenance/product_rows/family"]),
            "source_packet_count": len(h5["provenance/source_packets/role"]),
        }
    assert persisted_overview == {
        "input_packet_count": 36,
        "valid_packet_count": 36,
        "product_row_count": 14,
        "source_packet_count": 36,
    }

    output_dir = tmp_path / "report"
    result = report.generate_report(h5_path, output_dir)

    assert result.index_path == output_dir / "index.html"
    assert result.summary_path == output_dir / "summary.json"
    assert result.index_path.is_file()
    assert result.summary_path.is_file()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert {
        "report_format_version",
        "hdf5",
        "validation",
        "overview",
        "sections",
        "selections",
        "datasets",
        "group_attributes",
        "plots",
    } <= set(summary)
    assert summary["report_format_version"] == report.REPORT_FORMAT_VERSION
    assert {
        name: int(summary["overview"][name]) for name in persisted_overview
    } == persisted_overview
    assert REPORT_SECTIONS <= set(summary["sections"])

    sections = summary["sections"]
    assert sections["spectra"]["present"] is True
    assert sections["spectra"]["row_count"] == 3
    assert sections["tr_spectra"]["present"] is True
    assert sections["grimm_spectra"]["present"] is True
    assert sections["telemetry"]["present"] is False
    assert sections["housekeeping"]["row_count"] == 2
    assert sections["zoom_spectra"]["present"] is True
    assert sections["waveform"]["present"] is True
    assert sections["calibrator"]["present"] is True
    calibrator_families = sections["calibrator"]["families"]
    assert calibrator_families["data"]["page_timing"]["present"] is True
    assert calibrator_families["raw_pfb"]["page_timing"]["present"] is True
    assert calibrator_families["debug"]["page_timing"]["present"] is True

    expected_downloads = {
        "assets/source_packets.csv",
        "assets/product_rows.csv",
        "assets/product_schema_refs.csv",
        "assets/product_issue_refs.csv",
        "assets/family_issue_refs.csv",
    }
    assert expected_downloads <= set(summary["downloads"])
    for relative_path in summary["downloads"]:
        assert (output_dir / relative_path).is_file()
    relation_tables = sections["provenance"]["relation_tables"]
    assert relation_tables["/provenance/product_rows"]["row_count"] == 14
    assert relation_tables["/provenance/product_rows"]["csv"] == (
        "assets/product_rows.csv"
    )

    selections = summary["selections"]["spectra"]
    assert {
        key: {name: selections[key][name] for name in ("row", "unique_id")}
        for key in ("first", "last")
    } == {
        "first": {"row": 0, "unique_id": 101},
        "last": {"row": 2, "unique_id": 201},
    }
    product_statistics = sections["spectra"]["product_statistics"]
    assert product_statistics["unique_ids"] == [101, 151, 201]
    assert len(product_statistics["packet_median"]) == 3
    assert product_statistics["per_product"][7]["packets_present"] == 1

    relative_assets = {
        path.relative_to(output_dir).as_posix() for path in result.asset_paths
    }
    assert set(summary["plots"]) == relative_assets
    assert {
        "assets/spectra_first.png",
        "assets/spectra_last.png",
        "assets/spectra_all_packets.png",
    } <= relative_assets
    assert relative_assets
    for asset_path in result.asset_paths:
        assert asset_path.is_file()
        assert asset_path.read_bytes().startswith(PNG_SIGNATURE)

    attribute = next(
        item
        for item in summary["group_attributes"]
        if item["path"] == "/"
        and item["name"] == "dangerous_report_attribute"
    )
    assert attribute["value"] == dangerous
    index = result.index_path.read_text(encoding="utf-8")
    assert dangerous not in index
    assert html.escape(dangerous, quote=True) in index
    for label in (
        "normal spectra",
        "time-resolved spectra",
        "grimm",
        "telemetry",
        "housekeeping",
        "zoom",
        "waveform",
        "calibrator",
        "page timing and clock validity",
        "provenance",
        "dataset inventory",
    ):
        assert label in index.lower()


def expected_cdi_tree_id(cdi_dir: Path) -> str:
    digest = hashlib.sha256(b"lusee-cdi-tree-v1\0")
    for path in sorted(cdi_dir.glob("*.bin"), key=lambda item: item.name):
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(payload)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return f"sha256:{digest.hexdigest()}"


def test_issue_table_renders_every_source_locator():
    rendered = report.issues_html(({
        "issue_id": "issue-00000001",
        "severity": "warning",
        "stage": "decode",
        "code": "decode.fixture",
        "action": "dropped",
        "input_identity": "flash-identity",
        "session": "session-01",
        "bank": "b01",
        "byte_offset": 1234,
        "frame_index": 12,
        "packet_index": 34,
        "appid": 0x314,
        "sequence_count": 56,
        "uid": 78,
        "message": "fixture issue",
        "details": {"reason": "synthetic"},
    },))

    for header in (
        "issue ID",
        "input identity",
        "session",
        "byte offset",
        "frame index",
        "packet index",
        "sequence count",
    ):
        assert f"<th>{header}</th>" in rendered
    for value in (
        "issue-00000001",
        "flash-identity",
        "session-01",
        "1,234",
        "12",
        "34",
        "56",
        "78",
    ):
        assert value in rendered


def write_cdi_tree(root: Path, name: str, payloads: dict[str, bytes]):
    tree_dir = root / "trees" / name
    cdi_dir = tree_dir / "cdi_output"
    cdi_dir.mkdir(parents=True)
    for filename, payload in payloads.items():
        (cdi_dir / filename).write_bytes(payload)
    tree_id = expected_cdi_tree_id(cdi_dir)
    (tree_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest_schema_version": 1,
                "payload_path": "cdi_output",
                "tree_id": tree_id,
                "tree_id_algorithm": "lusee-cdi-tree-v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tree_id, tree_dir, cdi_dir


def test_cdi_tree_id_uses_the_centralized_inventory_contract(tmp_path: Path):
    cdi_dir = tmp_path / "cdi_output"
    cdi_dir.mkdir()
    (cdi_dir / "00002_0270.bin").write_bytes(b"second packet")
    (cdi_dir / "00001_0209.bin").write_bytes(b"first packet")

    expected = expected_cdi_tree_id(cdi_dir)
    assert report.cdi_tree_id(cdi_dir) == expected
    assert report.cdi_tree_id(cdi_dir).startswith("sha256:")
    assert len(report.cdi_tree_id(cdi_dir)) == len("sha256:") + 64

    (cdi_dir / "00002_0270.bin").write_bytes(b"changed packet")
    assert report.cdi_tree_id(cdi_dir) != expected


def test_read_corpus_targets_accepts_empty_and_rejects_changed_cdi_payload(
    tmp_path: Path,
):
    empty_id, empty_tree, empty_dir = write_cdi_tree(
        tmp_path,
        "empty-tree",
        {},
    )
    expected_empty_id = (
        "sha256:" + hashlib.sha256(b"lusee-cdi-tree-v1\0").hexdigest()
    )
    assert empty_id == expected_empty_id
    assert report.cdi_tree_id(empty_dir) == expected_empty_id

    def entry(tree_id: str, tree_dir: Path, cdi_dir: Path):
        return {
            "tree_id": tree_id,
            "cdi_output": cdi_dir.relative_to(tmp_path).as_posix(),
            "manifest": (tree_dir / "manifest.json")
            .relative_to(tmp_path)
            .as_posix(),
        }

    manifest_path = tmp_path / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "trees": [entry(empty_id, empty_tree, empty_dir)],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    targets = report.read_corpus_targets(tmp_path, "cdi")
    assert tuple(target.tree_id for target in targets) == (empty_id,)

    changed_id, changed_tree, changed_dir = write_cdi_tree(
        tmp_path,
        "changed-tree",
        {"00000_0209.bin": b"original packet"},
    )
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "trees": [
                    entry(empty_id, empty_tree, empty_dir),
                    entry(changed_id, changed_tree, changed_dir),
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (changed_dir / "00000_0209.bin").write_bytes(b"changed packet")

    with pytest.raises(ValueError, match="CDI payload identity disagrees"):
        report.read_corpus_targets(tmp_path, "cdi")


def test_corpus_targets_are_manifest_order_independent_and_skip_derived_cdi(
    tmp_path: Path,
):
    first_id, first_tree, first_input = write_cdi_tree(
        tmp_path,
        "tree-z",
        {"00000_0209.bin": b"first"},
    )
    second_id, second_tree, second_input = write_cdi_tree(
        tmp_path,
        "tree-a",
        {"00000_0209.bin": b"second"},
    )
    entries = [
        {
            "tree_id": second_id,
            "cdi_output": second_input.relative_to(tmp_path).as_posix(),
            "manifest": (second_tree / "manifest.json")
            .relative_to(tmp_path)
            .as_posix(),
        },
        {
            "tree_id": first_id,
            "cdi_output": first_input.relative_to(tmp_path).as_posix(),
            "manifest": (first_tree / "manifest.json")
            .relative_to(tmp_path)
            .as_posix(),
        },
    ]
    (tmp_path / "corpus_manifest.json").write_text(
        json.dumps({"manifest_version": 1, "trees": entries}) + "\n",
        encoding="utf-8",
    )

    sidecar_dir = (
        tmp_path / "telemetry_sidecars" / second_id.replace(":", "-")
    )
    sidecar_dir.mkdir(parents=True)
    sidecar = sidecar_dir / "DCB_telemetry.json"
    sidecar.write_bytes(b"synthetic sidecar")
    (tmp_path / "telemetry_sidecars" / "telemetry_sidecars_manifest.json").write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "entries": [
                    {
                        "tree_id": second_id,
                        "tree_directory": sidecar_dir.name,
                        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                        "size_bytes": sidecar.stat().st_size,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    targets = report.read_corpus_targets(tmp_path, "cdi")
    assert tuple(target.tree_id for target in targets) == tuple(
        sorted((first_id, second_id))
    )
    by_id = {target.tree_id: target for target in targets}
    assert by_id[first_id].tree_dir == first_tree
    assert by_id[first_id].input_dir == first_input
    assert by_id[first_id].telemetry_sidecar is None
    assert by_id[second_id].tree_dir == second_tree
    assert by_id[second_id].input_dir == second_input
    assert by_id[second_id].telemetry_sidecar == sidecar

    selected, skipped = report.select_cdi_targets(
        tuple(reversed(targets)),
        {second_id},
    )
    assert tuple(target.tree_id for target in selected) == (first_id,)
    assert tuple(target.tree_id for target in skipped) == (second_id,)


def test_generate_corpus_reports_runs_raw_first_and_suppresses_exact_cdi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    ccsds_root = tmp_path / "ccsds"
    cdi_root = tmp_path / "cdi"
    ccsds_root.mkdir()
    cdi_root.mkdir()
    landing = tmp_path / "landing.json"
    landing.write_text("{}\n", encoding="utf-8")

    raw_id = f"sha256:{'1' * 64}"
    derived_id = f"sha256:{'a' * 64}"
    derived_only_id = f"sha256:{'d' * 64}"
    standalone_id = f"sha256:{'a' * 63}b"

    def target(kind: str, tree_id: str, root: Path) -> report.CorpusTarget:
        tree_dir = root / "trees" / tree_id.replace(":", "-")
        input_dir = tree_dir / ("raw_flash" if kind == "ccsds" else "cdi_output")
        input_dir.mkdir(parents=True)
        return report.CorpusTarget(
            kind=kind,
            tree_id=tree_id,
            tree_dir=tree_dir,
            input_dir=input_dir,
        )

    raw_target = target("ccsds", raw_id, ccsds_root)
    skipped_target = target("cdi", derived_id, cdi_root)
    standalone_target = target("cdi", standalone_id, cdi_root)
    discovery_calls: list[str] = []
    processing_calls: list[tuple[str, str]] = []

    def fake_read_corpus_targets(root, kind):
        discovery_calls.append(kind)
        if kind == "ccsds":
            assert Path(root).resolve() == ccsds_root
            return (raw_target,)
        assert Path(root).resolve() == cdi_root
        return (skipped_target, standalone_target)

    def fake_report(index_path: Path) -> report.ReportResult:
        index_path.parent.mkdir(parents=True)
        index_path.write_text(
            "<!doctype html><title>synthetic</title>",
            encoding="utf-8",
        )
        summary_path = index_path.with_name("summary.json")
        summary_path.write_text("{}\n", encoding="utf-8")
        return report.ReportResult(
            index_path=index_path,
            summary_path=summary_path,
            asset_paths=(),
            summary={},
        )

    raw_reports: list[dict[str, object]] = []

    def fake_process_ccsds(target_value, landing_value, subdir, overwrite):
        processing_calls.append(("ccsds", target_value.tree_id))
        assert target_value is raw_target
        assert landing_value == landing.resolve()
        assert subdir == report.DEFAULT_REPORT_SUBDIR
        assert overwrite is False
        for session_name, hdf5_name, cdi_id in (
            ("session_000", "raw-000.h5", derived_id),
            ("session_001", "raw-001.h5", derived_only_id),
        ):
            generated = fake_report(
                raw_target.tree_dir
                / report.DEFAULT_REPORT_SUBDIR
                / "reports"
                / session_name
                / "index.html"
            )
            raw_reports.append(
                {
                    "session_name": session_name,
                    "hdf5_name": hdf5_name,
                    "derived_cdi_tree_id": cdi_id,
                    "report": generated,
                }
            )
        return raw_reports, [derived_id, derived_only_id], []

    def fake_process_cdi(target_value, landing_value, subdir, overwrite):
        processing_calls.append(("cdi", target_value.tree_id))
        assert target_value is standalone_target
        assert landing_value == landing.resolve()
        assert subdir == report.DEFAULT_REPORT_SUBDIR
        assert overwrite is False
        session_name = "session"
        generated = fake_report(
            target_value.tree_dir
            / report.DEFAULT_REPORT_SUBDIR
            / "reports"
            / session_name
            / "index.html"
        )
        return [
            {
                "session_name": session_name,
                "hdf5_name": "standalone.h5",
                "derived_cdi_tree_id": target_value.tree_id,
                "report": generated,
            }
        ], []

    monkeypatch.setattr(report, "read_corpus_targets", fake_read_corpus_targets)
    monkeypatch.setattr(report, "process_ccsds_target", fake_process_ccsds)
    monkeypatch.setattr(report, "process_cdi_target", fake_process_cdi)

    result = report.generate_corpus_reports(
        ccsds_corpus=ccsds_root,
        cdi_corpus=cdi_root,
        landing_time_file=landing,
    )

    assert discovery_calls == ["ccsds", "cdi"]
    assert processing_calls == [
        ("ccsds", raw_id),
        ("cdi", standalone_id),
    ]
    assert result.derived_cdi_tree_ids == tuple(
        sorted((derived_id, derived_only_id))
    )
    assert result.skipped_cdi_tree_ids == (derived_id,)
    assert result.failures == ()
    assert len(result.reports) == 3
    assert tuple(item.index_path for item in result.reports) == (
        raw_reports[0]["report"].index_path,
        raw_reports[1]["report"].index_path,
        standalone_target.tree_dir
        / report.DEFAULT_REPORT_SUBDIR
        / "reports"
        / "session"
        / "index.html",
    )

    assert not (skipped_target.tree_dir / report.DEFAULT_REPORT_SUBDIR).exists()
    raw_global = ccsds_root / report.DEFAULT_REPORT_SUBDIR
    cdi_global = cdi_root / report.DEFAULT_REPORT_SUBDIR
    for output_dir in (raw_global, cdi_global):
        assert (output_dir / "index.html").is_file()
        assert (output_dir / "summary.json").is_file()

    raw_summary = json.loads(
        (raw_global / "summary.json").read_text(encoding="utf-8")
    )
    cdi_summary = json.loads(
        (cdi_global / "summary.json").read_text(encoding="utf-8")
    )
    assert [item["hdf5_name"] for item in raw_summary["reports"]] == [
        "raw-000.h5",
        "raw-001.h5",
    ]
    assert [item["hdf5_name"] for item in cdi_summary["reports"]] == [
        "standalone.h5"
    ]
    assert cdi_summary["skipped_cdi_tree_ids"] == [derived_id]
    cdi_index = (cdi_global / "index.html").read_text(encoding="utf-8")
    assert standalone_id in cdi_index
    assert derived_id in cdi_index
    assert "suppressed" in cdi_index.lower()


def test_object_metadata_none_rows_are_counted_as_missing():
    values = np.empty(3, dtype=object)
    values[:] = (
        np.array([1, 2], dtype=np.int32),
        None,
        np.array([3, 4], dtype=np.int32),
    )

    records = report.field_columns_model(
        {"optional_tr_field": values},
        presence=None,
        row_count=3,
    )

    assert len(records) == 1
    assert records[0]["rows_present"] == 2
    assert records[0]["rows_missing"] == 1
    assert records[0]["statistics"]["count"] == 4


def test_numeric_field_matrix_preserves_present_zero_length_vectors():
    rows = (
        {"empty": np.empty(0, dtype=np.int16)},
        {"empty": None},
        {"empty": np.empty(0, dtype=np.int16)},
    )

    matrix = report.numeric_field_matrix(rows, "empty")

    assert matrix is not None
    assert matrix.shape == (3, 0)
    assert matrix.dtype == np.dtype(np.float64)


def test_waveform_sibling_rows_do_not_create_global_ordering_alarms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = SimpleNamespace(
        spectra_unique_ids=None,
        spectra_raw_times=None,
        spectra_raw_time_valid=None,
        spectra_mjd_times=None,
        spectra_mjd_time_valid=None,
        tr_unique_ids=None,
        tr_raw_times=None,
        tr_raw_time_valid=None,
        tr_mjd_times=None,
        tr_mjd_time_valid=None,
        zoom_unique_ids=None,
        zoom_raw_times=None,
        zoom_raw_time_valid=None,
        zoom_mjd_times=None,
        zoom_mjd_time_valid=None,
        grimm_unique_ids=None,
        grimm_raw_times=None,
        grimm_raw_time_valid=None,
        grimm_mjd_times=None,
        grimm_mjd_time_valid=None,
        housekeeping_unique_ids=None,
        housekeeping_raw_times=None,
        housekeeping_raw_time_valid=None,
        housekeeping_mjd_times=None,
        housekeeping_mjd_time_valid=None,
        waveform_data=np.array(
            [[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 5], [5, 4, 3, 2]],
            dtype=np.int16,
        ),
        waveform_channels=np.array([0, 1, 0, 1], dtype=np.uint8),
        waveform_unique_ids=np.array([7, 7, 8, 8], dtype=np.uint32),
        waveform_raw_times=np.array([10.0, 10.0, 11.0, 11.0]),
        waveform_raw_time_valid=np.ones(4, dtype=np.bool_),
        waveform_mjd_times=np.array([60_000.0, 60_000.0, 60_001.0, 60_001.0]),
        waveform_mjd_time_valid=np.ones(4, dtype=np.bool_),
        waveform_adc_timestamps=np.array([100, 101, 200, 200], dtype=np.uint64),
        waveform_adc_timestamp_valid=np.ones(4, dtype=np.bool_),
        calibrator={},
        telemetry=None,
    )

    timing = report.timing_and_identity_model(bundle)
    waveform_timing = next(
        item for item in timing if item["family"] == "waveform groups"
    )
    assert waveform_timing["row_count"] == 2
    assert waveform_timing["stored_channel_rows"] == 4
    assert waveform_timing["identifier_nonincreasing_steps"] == 0
    assert waveform_timing["raw_time_nonincreasing_steps"] == 0

    monkeypatch.setattr(
        report,
        "plot_waveform_endpoint",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "plot_waveform_summary",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "plot_waveform_heatmaps",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        report,
        "save_figure",
        lambda figure, assets_dir, name, assets: f"assets/{name}",
    )
    _, model, _ = report.waveform_section(bundle, tmp_path, [])
    assert model["adc_timestamp_within_uid_disagreement_count"] == 1


def test_report_overwrite_refuses_output_ancestor_of_input(tmp_path: Path):
    output_dir = tmp_path / "report-parent"
    output_dir.mkdir()
    h5_path = output_dir / "input.h5"
    original = b"synthetic input remains intact"
    h5_path.write_bytes(original)

    with pytest.raises(ValueError, match="must not contain"):
        report.generate_report(h5_path, output_dir, overwrite=True)

    assert h5_path.read_bytes() == original


def test_derived_cdi_is_suppressed_when_raw_report_generation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from lusee.ingest import pipeline

    ccsds_root = tmp_path / "ccsds"
    cdi_root = tmp_path / "cdi"
    raw_tree = ccsds_root / "trees" / "raw"
    raw_input = raw_tree / "raw_flash"
    raw_input.mkdir(parents=True)
    cdi_tree = cdi_root / "trees" / "derived"
    cdi_input = cdi_tree / "cdi_output"
    cdi_input.mkdir(parents=True)
    landing = tmp_path / "landing.json"
    landing.write_text("{}\n", encoding="utf-8")
    h5_path = tmp_path / "raw-session.h5"
    h5_path.write_bytes(b"synthetic HDF5 placeholder")
    session_dir = tmp_path / "raw-session"
    (session_dir / "cdi_output").mkdir(parents=True)

    raw_id = f"sha256:{'1' * 64}"
    derived_id = f"sha256:{'2' * 64}"
    raw_target = report.CorpusTarget(
        kind="ccsds",
        tree_id=raw_id,
        tree_dir=raw_tree,
        input_dir=raw_input,
    )
    cdi_target = report.CorpusTarget(
        kind="cdi",
        tree_id=derived_id,
        tree_dir=cdi_tree,
        input_dir=cdi_input,
    )
    session = SimpleNamespace(
        session_name="session_000",
        session_ordinal=0,
        session_dir=session_dir,
        h5_path=h5_path,
    )
    flash_result = SimpleNamespace(
        session_results=(session,),
        flash_result_id="flash-result",
        input_identity_sha256="3" * 64,
    )
    monkeypatch.setattr(pipeline, "process_flash", lambda *args, **kwargs: flash_result)
    monkeypatch.setattr(report, "cdi_tree_id", lambda path: derived_id)

    def fail_report(*args, **kwargs):
        raise RuntimeError("synthetic report failure")

    monkeypatch.setattr(report, "generate_report", fail_report)
    reports, derived, failures = report.process_ccsds_target(
        raw_target,
        landing,
        report.DEFAULT_REPORT_SUBDIR,
        False,
    )
    assert reports == []
    assert derived == [derived_id]
    assert failures[0]["error"] == "synthetic report failure"

    processing_calls = []
    orchestration_tree = ccsds_root / "trees" / "raw-orchestration"
    orchestration_input = orchestration_tree / "raw_flash"
    orchestration_input.mkdir(parents=True)
    orchestration_target = report.CorpusTarget(
        kind="ccsds",
        tree_id=raw_id,
        tree_dir=orchestration_tree,
        input_dir=orchestration_input,
    )

    def fake_read_targets(root, kind):
        return (orchestration_target,) if kind == "ccsds" else (cdi_target,)

    monkeypatch.setattr(report, "read_corpus_targets", fake_read_targets)
    monkeypatch.setattr(
        report,
        "process_ccsds_target",
        lambda *args, **kwargs: (reports, derived, failures),
    )

    def unexpected_cdi_processing(*args, **kwargs):
        processing_calls.append("cdi")
        return [], []

    monkeypatch.setattr(report, "process_cdi_target", unexpected_cdi_processing)
    result = report.generate_corpus_reports(
        ccsds_corpus=ccsds_root,
        cdi_corpus=cdi_root,
        landing_time_file=landing,
    )

    assert processing_calls == []
    assert result.derived_cdi_tree_ids == (derived_id,)
    assert result.skipped_cdi_tree_ids == (derived_id,)
    assert len(result.failures) == 1
