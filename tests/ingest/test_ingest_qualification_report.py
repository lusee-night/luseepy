"""Focused synthetic tests for the ingestion qualification report."""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lusee.ingest as ingest
from scripts import ingest_qualification_report as report

SYNTHETIC_REFERENCE_ISOT = "2024-03-04T05:06:07"
SYNTHETIC_SPECTROMETER_ANCHOR = 1234.5
SYNTHETIC_DCB_ANCHOR = 6789.25


def family_counts(**updates: int) -> dict[str, int]:
    counts = {family: 0 for family in report.FAMILIES}
    counts.update(updates)
    return counts


def family_metadata(counts: dict[str, int]) -> dict[str, dict[str, str]]:
    return {
        family: {
            "unit": "synthetic packet",
            "basis": "focused test inventory",
            "input_state": "present" if count > 0 else "absent",
        }
        for family, count in counts.items()
    }


def test_products_summary_counts_strict_calibrator_families():
    products = SimpleNamespace(
        calibrator_metadata=[object()],
        calibrator_data=[object(), object()],
        calibrator_raw_pfb=[object()],
        calibrator_debug=[object(), object(), object()],
    )

    assert report.products_summary(products)["calibrator"] == 7


def make_direct_config(
    tmp_path: Path,
    target_ids: tuple[str, ...],
) -> report.QualificationConfig:
    landing = tmp_path / "landing.json"
    landing.write_text(
        json.dumps(
            {
                "format_version": 1,
                "reference_event": "landing",
                "clock_reference_isot": SYNTHETIC_REFERENCE_ISOT,
                "time_scale": "utc",
                "source": "synthetic test reference",
                "assumed": True,
                "clocks": {
                    "spectrometer": {
                        "clock_reference_raw_seconds": SYNTHETIC_SPECTROMETER_ANCHOR
                    },
                    "dcb": {"clock_reference_raw_seconds": SYNTHETIC_DCB_ANCHOR},
                },
            }
        ),
        encoding="ascii",
    )
    manifest = tmp_path / "corpus-manifest.json"
    manifest.write_text("{}\n", encoding="ascii")
    targets = []
    for target_id in target_ids:
        source = tmp_path / f"{target_id}-source"
        source.mkdir()
        targets.append(
            report.TargetConfig(
                target_id=target_id,
                kind="cdi",
                source_path=source,
                telemetry_sidecar=None,
                observed_families=family_counts(normal=1),
            )
        )
    return report.QualificationConfig(
        format_version=1,
        run_id="synthetic-run",
        subject_commit="a" * 40,
        landing_time_file=landing,
        corpus_manifest_paths=(manifest,),
        targets=tuple(targets),
        spectrometer_clock_source="spectrometer",
        dcb_clock_source="dcb",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def empty_artifact(target_id: str) -> report.SessionArtifacts:
    return report.SessionArtifacts(
        target_id=target_id,
        session_id="session_000000",
        decoded_summary=family_counts(),
        h5_path=None,
        fits_path=None,
    )


def test_raw_qualification_threads_landing_file_to_parse_flash(
    tmp_path: Path,
    monkeypatch,
):
    config = make_direct_config(tmp_path, ("raw",))
    target = replace(config.targets[0], kind="raw")
    adapter = report.load_baseline_clock_adapter(config)
    seen = []

    def fake_parse_flash(path, *, landing_time_file):
        seen.append((path, landing_time_file))
        return [], {}, {}

    monkeypatch.setattr(ingest, "parse_flash", fake_parse_flash)

    artifacts = report.execute_target_default(
        target,
        tmp_path / "work",
        adapter,
    )

    assert artifacts == []
    assert seen == [(target.source_path, config.landing_time_file)]


def test_run_attempts_targets_after_failure_and_writes_failure_report(tmp_path: Path):
    config = make_direct_config(tmp_path, ("first", "broken", "last"))
    calls = []

    def executor(target, work_dir, clock_adapter):
        from astropy.time import Time

        calls.append(target.target_id)
        assert clock_adapter.spectrometer_raw_seconds == (
            SYNTHETIC_SPECTROMETER_ANCHOR
        )
        assert clock_adapter.dcb_raw_seconds == SYNTHETIC_DCB_ANCHOR
        assert clock_adapter.mjd_epoch_offset_days == pytest.approx(
            Time(SYNTHETIC_REFERENCE_ISOT, format="isot", scale="utc").mjd
        )
        if target.target_id == "broken":
            raise RuntimeError("synthetic target failure")
        return [empty_artifact(target.target_id)]

    output = tmp_path / "report"
    result = report.run_qualification(config, output, target_executor=executor)

    assert calls == ["broken", "first", "last"]
    assert result.attempted_target_ids == ("broken", "first", "last")
    assert result.status == "complete_with_issues"
    for target_id in result.attempted_target_ids:
        pdf = output / "trees" / target_id / "report.pdf"
        assert pdf.is_file() and pdf.stat().st_size > 0
    issues = read_jsonl(output / "issues.jsonl")
    assert any(
        row.get("target_id") == "broken"
        and row.get("stage") == "target_execute"
        and row.get("code") == "stage_failed.RuntimeError"
        for row in issues
    )
    metrics = read_jsonl(output / "metrics.jsonl")
    assert any(
        row.get("target_id") == "broken"
        and row.get("stage") == "decoded"
        and row.get("state") == "stage_failed"
        for row in metrics
    )


def test_canonical_jsonl_is_repeatable_and_sanitizes_private_paths(tmp_path: Path):
    config = make_direct_config(tmp_path, ("broken",))

    def executor(target, work_dir, clock_adapter):
        raise RuntimeError(
            f"failed under {target.source_path} while writing {work_dir / 'temporary.h5'}"
        )

    first = tmp_path / "first-report"
    second = tmp_path / "second-report"
    report.run_qualification(config, first, target_executor=executor)
    report.run_qualification(config, second, target_executor=executor)

    for name in ("metrics.jsonl", "issues.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert str(tmp_path).encode("ascii") not in (first / name).read_bytes()
    issue_bytes = (first / "issues.jsonl").read_bytes()
    assert b"<private-path>" in issue_bytes
    assert b"temporary.h5" in issue_bytes


def test_capture_call_retains_diagnostics_before_failure(tmp_path: Path):
    def fail_after_diagnostics():
        warnings.warn(
            f"runtime warning under {tmp_path}",
            RuntimeWarning,
        )
        print(f"Warning: stdout diagnostic under {tmp_path}")
        print(
            f"Warning: stderr diagnostic under {tmp_path}",
            file=sys.stderr,
        )
        raise RuntimeError(f"failure under {tmp_path}")

    ok, value, records = report.capture_call(
        fail_after_diagnostics,
        target_id="diagnostic-target",
        session_id="session_000",
        stage="synthetic",
        private_paths=[tmp_path],
    )

    assert not ok
    assert value is None
    assert {record["code"] for record in records} == {
        "warning.RuntimeWarning",
        "diagnostic.stdout",
        "diagnostic.stderr",
        "stage_failed.RuntimeError",
    }
    assert all(str(tmp_path) not in str(record) for record in records)
    assert all("<private-path>" in record["message"] for record in records)


def test_coverage_states_and_zero_statistics_are_explicit():
    states = {
        report.coverage_state(observed_count=1, count=1),
        report.coverage_state(observed_count=0, count=0),
        report.coverage_state(observed_count=1, count=0),
        report.coverage_state(observed_count=1, count=0, supported=False),
        report.coverage_state(observed_count=1, count=0, failed=True),
        report.coverage_state(
            observed_count=0,
            count=0,
            input_state="present_empty",
        ),
    }
    assert states == set(report.COVERAGE_STATES)
    assert report.coverage_state(
        observed_count=0,
        count=1,
        input_state="absent",
    ) == "present"

    metric = report.array_metric(
        "target",
        "session",
        "normal/data",
        np.zeros((2, 3), dtype=np.int16),
        "hdf5",
    )
    assert metric["valid_count"] == 6
    assert metric["missing_count"] == 0
    assert metric["minimum"] == metric["maximum"] == metric["mean"] == 0.0

    target = report.TargetConfig(
        target_id="calibrator-gap",
        kind="cdi",
        source_path=Path("."),
        observed_families=family_counts(calibrator=1),
    )
    rows = report.build_target_coverage(
        target,
        family_counts(calibrator=1),
        family_counts(calibrator=1),
        family_counts(),
        family_counts(),
        family_counts(),
        set(),
    )
    calibrator = {
        row["stage"]: row["state"]
        for row in rows
        if row["family"] == "calibrator"
    }
    assert calibrator == {
        "observed_input": "present",
        "decoded": "present",
        "hdf5": "present",
        "fits": "unsupported",
        "reader": "unsupported",
        "plotted": "unsupported",
    }

    writer_failure_rows = report.build_target_coverage(
        replace(target, target_id="writer-failure", observed_families=family_counts(normal=1)),
        family_counts(normal=1),
        family_counts(),
        family_counts(),
        family_counts(),
        family_counts(),
        {"hdf5", "fits"},
    )
    writer_failure = {
        row["stage"]: row["state"]
        for row in writer_failure_rows
        if row["family"] == "normal"
    }
    assert writer_failure == {
        "observed_input": "present",
        "decoded": "present",
        "hdf5": "stage_failed",
        "fits": "stage_failed",
        "reader": "stage_failed",
        "plotted": "stage_failed",
    }

    present_empty = report.TargetConfig(
        target_id="present-empty-sidecar",
        kind="cdi",
        source_path=Path("."),
        observed_families=family_counts(),
        observed_family_metadata={
            family: report.ObservedFamilyEvidence(
                unit="sidecar record" if family == "telemetry" else "packet file",
                basis="independent inventory",
                input_state="present_empty" if family == "telemetry" else "absent",
            )
            for family in report.FAMILIES
        },
    )
    present_empty_rows = report.build_target_coverage(
        present_empty,
        family_counts(),
        family_counts(),
        family_counts(),
        family_counts(),
        family_counts(),
        set(),
        telemetry_state="present_empty",
    )
    telemetry_rows = [
        row for row in present_empty_rows if row["family"] == "telemetry"
    ]
    assert {row["state"] for row in telemetry_rows} == {"present_empty"}
    assert {row["observed_unit"] for row in telemetry_rows} == {
        "sidecar record"
    }


@pytest.mark.parametrize(
    (
        "observed_count",
        "input_state",
        "decoded_count",
        "expected_state",
        "issue_code",
    ),
    (
        (1, "present", 0, "invalid_or_dropped", "coverage.invalid_or_dropped"),
        (0, "absent", 1, "present", "coverage.unexpected_presence"),
        (0, "present_empty", 1, "present", "coverage.unexpected_presence"),
    ),
)
def test_coverage_contradictions_force_run_issue(
    tmp_path: Path,
    observed_count: int,
    input_state: str,
    decoded_count: int,
    expected_state: str,
    issue_code: str,
):
    config = make_direct_config(tmp_path, ("coverage-target",))
    config = replace(
        config,
        targets=(replace(
            config.targets[0],
            observed_families=family_counts(normal=observed_count),
            observed_family_metadata={
                "normal": report.ObservedFamilyEvidence(
                    unit="synthetic packet",
                    basis="focused test inventory",
                    input_state=input_state,
                )
            },
        ),),
    )

    def executor(target, work_dir, clock_adapter):
        return [replace(
            empty_artifact(target.target_id),
            decoded_summary=family_counts(normal=decoded_count),
        )]

    output = tmp_path / "report"
    result = report.run_qualification(
        config,
        output,
        target_executor=executor,
    )

    assert result.status == "complete_with_issues"
    issues = read_jsonl(output / "issues.jsonl")
    assert any(row.get("code") == issue_code for row in issues)
    decoded = next(
        row for row in read_jsonl(output / "metrics.jsonl")
        if row.get("record_type") == "coverage"
        and row.get("family") == "normal"
        and row.get("stage") == "decoded"
    )
    assert decoded["state"] == expected_state


def test_representative_selection_honors_stable_identity_and_fails_closed():
    candidates = [
        {"session_id": "session_b", "reader_row": 0, "unique_packet_id": 7},
        {"session_id": "session_a", "reader_row": 1, "unique_packet_id": 8},
    ]
    selected = report.select_candidate(
        candidates,
        None,
        lambda item: (str(item["session_id"]), int(item["reader_row"])),
    )
    assert selected == candidates[1]
    selected = report.select_candidate(
        candidates,
        {"session_id": "session_b", "unique_packet_id": 7},
        lambda item: 0,
    )
    assert selected == candidates[0]
    assert report.select_candidate(
        candidates,
        {"session_id": "session_b", "unique_packet_id": 99},
        lambda item: 0,
    ) is None


def test_bundle_semantic_comparator_reports_every_mismatch_class():
    hdf5_bundle = SimpleNamespace(
        spectra=np.array([1.0, np.nan], dtype=np.float32),
        spectra_unique_ids=np.array([1], dtype=np.int16),
        tr_spectra=np.array([1, 2], dtype=np.int16),
        zoom_spectra=np.array([1], dtype=np.int16),
        grimm_spectra=np.array([1, 2], dtype=np.int16),
        constants={"hdf5_only": np.array([1], dtype=np.int16)},
    )
    fits_bundle = SimpleNamespace(
        spectra=np.array([1.0, np.nan], dtype=np.float32),
        tr_spectra=np.array([[1, 2]], dtype=np.int16),
        zoom_spectra=np.array([1], dtype=np.int32),
        grimm_spectra=np.array([1, 3], dtype=np.int16),
        constants={"fits_only": np.array([1], dtype=np.int16)},
    )
    records = report.compare_semantics(
        report.bundle_semantics(hdf5_bundle),
        report.bundle_semantics(fits_bundle),
    )
    by_field = {record["field_path"]: record for record in records}

    assert by_field["normal/data"]["status"] == "equal"
    assert by_field["normal/unique_ids"]["status"] == "missing_fits"
    assert by_field["constants/fits_only"]["status"] == "missing_hdf5"
    assert by_field["tr/data"]["status"] == "shape_mismatch"
    assert by_field["zoom/data"]["status"] == "dtype_mismatch"
    assert by_field["grimm/data"]["status"] == "value_mismatch"
    assert by_field["grimm/data"]["first_index"] == 1


def test_public_writer_reader_parity_for_supported_fields(tmp_path: Path):
    pytest.importorskip("h5py")
    from test_layout_v4_hdf5 import make_all_family_request

    from lusee.ingest.fits_writer import write_fits
    from lusee.ingest.hdf5_writer import write_hdf5

    request = make_all_family_request()
    h5_path = tmp_path / "session.h5"
    fits_path = tmp_path / "session.fits"
    write_hdf5(request, h5_path)
    write_fits(request, fits_path)

    hdf5 = report.read_public_bundle(h5_path, "h5")
    fits = report.read_public_bundle(fits_path, "fits")
    parity = {
        row["field_path"]: row
        for row in report.compare_semantics(
            report.bundle_semantics(hdf5.bundle),
            report.bundle_semantics(fits.bundle),
            target_id="writer-reader",
            session_id="session",
        )
    }
    assert parity["normal/data"]["status"] == "equal"
    assert parity["normal/unique_ids"]["status"] == "equal"
    assert parity["normal/raw_times"]["status"] == "equal"
    assert parity["normal/mjd_times"]["status"] == "equal"


def test_family_pages_are_unique_hdf5_based_and_label_open_contracts(tmp_path: Path):
    bundle = SimpleNamespace(
        layout_version=3,
        spectra=np.zeros((2, 16, 8), dtype=np.float32),
        spectra_units="SDU",
        spectra_representation="gain_model_input_sdu",
        spectra_metadata={"Navgf": np.array([1, 2])},
        spectra_raw_times=np.array([10.0, 12.0]),
        spectra_mjd_times=np.array([60000.0, 60000.5]),
        tr_spectra=np.full((2, 16, 2, 3), np.nan, dtype=np.float32),
        tr_unique_ids=np.array([10, 11], dtype=np.int64),
        tr_raw_times=np.array([20.0, 23.0]),
        tr_mjd_times=np.array([60001.0, 60001.25]),
        tr_navg2_per_sample=np.array([2, 2]),
        tr_length_per_sample=np.array([3, 3]),
        zoom_spectra=np.ones((1, 4, 64), dtype=np.float32),
        zoom_unique_ids=np.array([20], dtype=np.int64),
        zoom_pfb_indices=np.array([7]),
        zoom_raw_times=np.array([30.0]),
        zoom_mjd_times=np.array([60002.0]),
        waveforms={channel: np.zeros((3, 16), dtype=np.int16) for channel in range(4)},
        waveform_times={channel: np.array([100.0, 102.0, 105.0]) for channel in range(4)},
        grimm_spectra=np.ones((1, 16, 32), dtype=np.float32),
        grimm_unique_ids=np.array([30], dtype=np.int64),
        grimm_raw_times=np.array([40.0]),
        dcb_fpga={
            "mission_seconds": np.array([100.0, 101.0]),
            "lusee_subsecs": np.array([0.0, 1.0]),
            "THERM_DCB": np.array([1.0, 2.0]),
            "VMON_1V8": np.array([3.0, 4.0]),
            "SPE_P5_C": np.array([5.0, 6.0]),
            "ADC_PWR": np.array([0.0, 2.0]),
        },
        dcb_encoder={
            "mission_seconds": np.array([100.0, 101.0]),
            "lusee_subsecs": np.array([0.0, 1.0]),
            "enc_pos": np.array([10, 12], dtype=np.int64),
            "enc_status": np.array([1, 1], dtype=np.int64),
        },
        housekeeping={
            2: {
                "raw_seconds": np.array([200.0, 203.0]),
                "time": np.array([200.0, 203.0]),
                "ok": np.array([0, 1], dtype=np.uint8),
                "reading": np.array([0.0, 2.0]),
            }
        },
        session_invariants={"schema": "legacy"},
        constants={
            "integration_count": np.array(4, dtype=np.int16),
            "clock_epoch_isot": SYNTHETIC_REFERENCE_ISOT,
            "mjd_epoch_offset_days": np.array(60373.0),
            "raw_time_subtract_seconds": np.array(
                SYNTHETIC_SPECTROMETER_ANCHOR
            ),
            "clock_source": "spectrometer",
            "time_scale": "utc",
        },
    )
    bundle.tr_spectra[0, 4] = 0.0
    config = make_direct_config(tmp_path, ("plot-target",))
    adapter = report.load_baseline_clock_adapter(config)
    target = config.targets[0]
    sessions = [report.PlotSession("session_000", bundle, np.arange(8, dtype=float))]

    pages, selections, issues, failed = report.build_family_pages(
        target, sessions, {}, adapter
    )
    try:
        families = [page.family for page in pages]
        assert len(families) == len(set(families))
        assert families == [
            "normal",
            "tr",
            "zoom",
            "waveform",
            "grimm",
            "telemetry",
            "housekeeping",
            "configuration",
        ]
        assert not failed
        assert set(selections) == {
            f"plot-target/{family}"
            for family in families
            if family != "configuration"
        }
        normal = next(page for page in pages if page.family == "normal")
        assert all("stored bin index" in axis.get_xlabel() for axis in normal.figure.axes[5:10])
        assert any(issue["code"] == "plot.frequency_axis_unresolved" for issue in issues)
        assert any(issue["code"] == "plot.grimm_layout_unsupported" for issue in issues)
        telemetry = next(page for page in pages if page.family == "telemetry")
        assert any("ADC time unmapped" in text.get_text() for text in telemetry.figure.texts)
        telemetry_text = "\n".join(text.get_text() for text in telemetry.figure.texts)
        assert "assumed DCB FPGA seconds from landing" in telemetry_text
        assert "assumed DCB encoder seconds from landing" in telemetry_text
        assert telemetry_text.count("sampling_gap_median") == 2
        assert telemetry_text.count("sampling_gap_maximum") == 2
        assert "FPGA ADC_PWR" in telemetry_text
        assert "valid=2" in telemetry_text
        assert "missing=0" in telemetry_text
        assert "minimum=0" in telemetry_text
        assert "maximum=2" in telemetry_text
        assert "mean=1" in telemetry_text
        assert "unit unestablished" in telemetry_text
        assert "ENC enc_pos:" in telemetry_text
        assert "ENC enc_status:" in telemetry_text
        encoder_summary = report.categorical_summary_text(
            bundle.dcb_encoder["enc_status"],
            "unestablished",
        )
        assert "mode=1" in encoder_summary
        assert "counts={1:2}" in encoder_summary
        assert "mean=" not in encoder_summary
        housekeeping = next(page for page in pages if page.family == "housekeeping")
        housekeeping_text = "\n".join(
            text.get_text() for text in housekeeping.figure.texts
        )
        assert "valid=2" in housekeeping_text
        assert "missing=0" in housekeeping_text
        assert "ok:" in housekeeping_text
        assert "counts={0:1,1:1}" in housekeeping_text
        ok_detail = housekeeping_text.split("ok:", 1)[1].split(
            "raw_seconds:", 1
        )[0]
        assert "mean=" not in ok_detail
        assert "raw_seconds coverage" in housekeeping_text
        assert "time coverage" in housekeeping_text
        assert "sampling_gap_median=3" in housekeeping_text
        waveform = next(page for page in pages if page.family == "waveform")
        assert waveform.count == 1
        for family in ("normal", "tr", "zoom", "waveform", "grimm", "telemetry"):
            page = next(item for item in pages if item.family == family)
            page_text = "\n".join(text.get_text() for text in page.figure.texts)
            assert "sampling_gap_median" in page_text
            assert "sampling_gap_maximum" in page_text
        for family in ("normal", "tr", "zoom"):
            page = next(item for item in pages if item.family == family)
            page_text = "\n".join(text.get_text() for text in page.figure.texts)
            assert "qualification adapter; TIME-004" in page_text
            assert "MJD days (assumed; TIME-004)" in page_text
        configuration = next(
            page for page in pages if page.family == "configuration"
        )
        configuration_text = "\n".join(
            text.get_text() for text in configuration.figure.texts
        )
        assert "provenance=canonical public reader" in configuration_text
        assert "session/schema" in configuration_text
        assert "constants/integration_count" in configuration_text
        assert "Absolute-time constants are qualification assumptions (TIME-004)" in configuration_text
        assert "MJD day (qualification assumption; TIME-004)" in configuration_text
        assert "constants/raw_time_subtract_seconds" in configuration_text
        assert "raw clock second anchor" in configuration_text
    finally:
        import matplotlib.pyplot as plt

        for page in pages:
            plt.close(page.figure)


def test_encoder_status_metrics_jsonl_uses_counts_and_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert report.categorical_field_path("telemetry/encoder/enc_status")
    assert not report.categorical_field_path("telemetry/encoder/enc_pos")
    assert not report.categorical_field_path(
        "telemetry/encoder/mission_seconds"
    )
    assert report.categorical_field_path("housekeeping/type_2/ok")
    assert "TIME-004" in report.field_units("normal/mjd_times")
    assert "TIME-004" in report.field_units(
        "constants/mjd_epoch_offset_days"
    )
    config = make_direct_config(tmp_path, ("categorical",))
    observed = family_counts(telemetry=3, waveform=3)
    observed_metadata = {
        family: report.ObservedFamilyEvidence(**value)
        for family, value in family_metadata(observed).items()
    }
    config = replace(
        config,
        targets=(replace(
            config.targets[0],
            observed_families=observed,
            observed_family_metadata=observed_metadata,
        ),),
    )
    bundle = SimpleNamespace(
        layout_version=3,
        waveforms={0: np.zeros((3, 4), dtype=np.int16)},
        waveform_times={0: np.array([100.0, 102.0, 103.0])},
        dcb_encoder={
            "mission_seconds": np.array([100.0, 102.0, 103.0]),
            "lusee_subsecs": np.array([0.0, 0.0, 0.0]),
            "enc_pos": np.array([10, 20, 30], dtype=np.int64),
            "enc_status": np.array([1, 2, 2], dtype=np.int64),
        },
    )

    page = report.telemetry_page(
        config.targets[0],
        [report.PlotSession("session_000", bundle, None)],
        None,
        report.load_baseline_clock_adapter(config),
    )
    assert page is not None
    try:
        page_text = "\n".join(text.get_text() for text in page.figure.texts)
        assert "assumed DCB encoder seconds from landing" in page_text
        assert "ENC enc_pos:" in page_text
        assert "minimum=10" in page_text
        assert "maximum=30" in page_text
        assert "mean=20" in page_text
        assert "ENC enc_status:" in page_text
        assert "counts={2:2,1:1}" in page_text
    finally:
        import matplotlib.pyplot as plt

        plt.close(page.figure)

    def executor(target, work_dir, clock_adapter):
        work_dir.mkdir(parents=True, exist_ok=True)
        h5_path = work_dir / "synthetic.h5"
        h5_path.write_bytes(b"synthetic")
        return [report.SessionArtifacts(
            target_id=target.target_id,
            session_id="session_000",
            decoded_summary=observed,
            h5_path=h5_path,
            fits_path=None,
            telemetry_state="decoded",
        )]

    monkeypatch.setattr(report, "inspect_hdf5", lambda path: observed)
    monkeypatch.setattr(
        report,
        "read_public_bundle",
        lambda path, prefer_format: report.ReaderView(bundle, None),
    )
    output = tmp_path / "report"
    report.run_qualification(config, output, target_executor=executor)

    metrics = read_jsonl(output / "metrics.jsonl")
    metric = next(
        row
        for row in metrics
        if row.get("record_type") == "array"
        and row.get("field_path") == "telemetry/encoder/enc_status"
    )
    assert metric["counts"] == {"1": 1, "2": 2}
    assert metric["mode"] == "2"
    assert metric["mode_count"] == 2
    assert "minimum" not in metric
    assert "maximum" not in metric
    assert "mean" not in metric
    evidence = report.comparison_evidence(metric)
    assert evidence["counts"] == metric["counts"]
    assert evidence["mode"] == "2"
    assert evidence["mode_count"] == 2
    position = next(
        row for row in metrics
        if row.get("field_path") == "telemetry/encoder/enc_pos"
    )
    assert position["minimum"] == 10
    assert position["maximum"] == 30
    assert position["mean"] == 20.0
    assert "counts" not in position
    encoder_time = next(
        row for row in metrics
        if row.get("field_path") == "telemetry/encoder/mission_seconds"
    )
    assert encoder_time["time_minimum"] == 100.0
    assert encoder_time["time_maximum"] == 103.0
    assert encoder_time["sampling_gap_median"] == 1.5
    assumed_mjd = next(
        row for row in metrics
        if row.get("field_path") == "telemetry/display/assumed_mjd"
    )
    assert "TIME-004" in assumed_mjd["units"]
    time_source = next(
        row for row in metrics
        if row.get("field_path") == "telemetry/display/time_source"
    )
    assert time_source["counts"] == {"encoder": 1}
    assert time_source["mode"] == "encoder"
    waveform_time = next(
        row for row in metrics
        if row.get("field_path") == "waveform/channel_0/times"
    )
    assert waveform_time["units"] == (
        "raw ADC clock second (unmapped; not absolute time)"
    )
    assert "TIME-004" not in waveform_time["units"]
    assert waveform_time["time_minimum"] == 100.0
    assert waveform_time["sampling_gap_median"] == 1.5


def test_comparison_records_retain_before_after_evidence():
    previous = [{
        "record_type": "coverage",
        "target_id": "tree",
        "session_id": "all",
        "family": "normal",
        "stage": "decoded",
        "field_path": "",
        "count": 1,
    }]
    current = [{**previous[0], "count": 2}]
    comparison = report.compare_metric_sets(previous, current)
    assert comparison == [{
        "record_type": "comparison",
        "target_id": "tree",
        "session_id": "all",
        "family": "normal",
        "stage": "decoded",
        "field_path": "",
        "status": "changed",
        "previous": previous[0],
        "current": current[0],
    }]

    issue_comparison = report.compare_issue_sets(
        [{"target_id": "tree", "stage": "decode", "code": "x", "severity": "warning", "occurrence_count": 2}],
        [{"target_id": "tree", "stage": "decode", "code": "x", "severity": "warning", "occurrence_count": 1}],
    )
    assert issue_comparison[0]["previous_count"] == 2
    assert issue_comparison[0]["current_count"] == 1
    assert report.compare_selections(
        {"tree/tr": {"unique_packet_id": 1}},
        {"tree/tr": {"unique_packet_id": 2}},
    )[0]["status"] == "changed"


def write_config_file(
    tmp_path: Path,
    targets: list[dict[str, object]],
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    landing = tmp_path / "landing.json"
    landing.write_text("{}\n", encoding="ascii")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="ascii")
    source = tmp_path / "source"
    source.mkdir()
    for target in targets:
        target["source_path"] = source.name
        target.setdefault("observed_families", family_counts())
        target.setdefault(
            "observed_family_metadata",
            family_metadata(target["observed_families"]),
        )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "run_id": "validation-test",
                "subject_commit": "b" * 40,
                "landing_time_file": landing.name,
                "corpus_manifest_paths": [manifest.name],
                "spectrometer_clock_source": "spectrometer",
                "dcb_clock_source": "dcb",
                "expected_source_commits": {
                    "uncrater": "1" * 40,
                    "lusee_telemetry": "2" * 40,
                },
                "targets": targets,
            }
        ),
        encoding="ascii",
    )
    return config


def test_config_rejects_duplicate_targets_and_nonlegacy_profile(tmp_path: Path):
    duplicate = write_config_file(
        tmp_path / "duplicate",
        [
            {"target_id": "same", "kind": "cdi"},
            {"target_id": "same", "kind": "cdi"},
        ],
    )
    with pytest.raises(ValueError, match="target IDs must be unique"):
        report.load_config(duplicate)

    nonlegacy_root = tmp_path / "nonlegacy"
    nonlegacy_root.mkdir()
    nonlegacy = write_config_file(
        nonlegacy_root,
        [
            {
                "target_id": "raw-target",
                "kind": "raw",
                "reassembly_profile": "ccsds",
            }
        ],
    )
    with pytest.raises(ValueError, match="supports only the explicit legacy"):
        report.load_config(nonlegacy)

    symbolic_root = tmp_path / "symbolic"
    symbolic_root.mkdir()
    symbolic = write_config_file(
        symbolic_root,
        [{"target_id": "symbolic", "kind": "cdi"}],
    )
    value = json.loads(symbolic.read_text(encoding="ascii"))
    value["subject_commit"] = "main"
    symbolic.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="exact full Git SHA"):
        report.load_config(symbolic)


def test_python_source_digest_ignores_runtime_cache_files(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="ascii")
    before = report.hash_python_source_tree(source)
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"runtime-specific cache")
    assert report.hash_python_source_tree(source) == before
    (source / "module.py").write_text("VALUE = 2\n", encoding="ascii")
    assert report.hash_python_source_tree(source) != before


def test_run_removes_all_temporary_derivatives(tmp_path: Path):
    config = make_direct_config(tmp_path, ("clean",))
    created_paths = []

    def executor(target, work_dir, clock_adapter):
        derivative = work_dir / "nested" / "temporary.h5"
        derivative.parent.mkdir(parents=True)
        derivative.write_bytes(b"temporary science derivative")
        created_paths.extend((work_dir, derivative))
        return [empty_artifact(target.target_id)]

    output = tmp_path / "report"
    report.run_qualification(config, output, target_executor=executor)

    assert created_paths
    assert all(not path.exists() for path in created_paths)
    assert not list(output.rglob("*.h5"))
    assert (output / "trees" / "clean" / "report.pdf").is_file()


def test_run_refuses_nonempty_output_directory(tmp_path: Path):
    config = make_direct_config(tmp_path, ("fresh-output",))
    output = tmp_path / "report"
    output.mkdir()
    (output / "previous-run.txt").write_text("stale", encoding="ascii")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        report.run_qualification(config, output, target_executor=lambda *args: [])


@pytest.mark.parametrize("failure_phase", ("coverage", "pdf"))
def test_post_execute_failure_isolated_and_later_target_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
):
    config = make_direct_config(tmp_path, ("broken", "later"))

    def executor(target, work_dir, clock_adapter):
        work_dir.mkdir(parents=True, exist_ok=True)
        h5_path = work_dir / "synthetic.h5"
        h5_path.write_bytes(b"synthetic")
        return [report.SessionArtifacts(
            target_id=target.target_id,
            session_id="session_000",
            decoded_summary=family_counts(normal=1),
            h5_path=h5_path,
            fits_path=None,
        )]

    monkeypatch.setattr(
        report,
        "inspect_hdf5",
        lambda path: family_counts(normal=1),
    )

    def reader(path, output_format):
        bundle = SimpleNamespace(
            marker=path.parent.name,
            layout_version=3,
            spectra=np.zeros((1, 16, 4), dtype=np.float32),
            spectra_units="SDU",
            spectra_representation="gain_model_input_sdu",
            spectra_metadata={"Navgf": np.array([1])},
        )
        return report.ReaderView(bundle, np.arange(4, dtype=float))

    monkeypatch.setattr(report, "read_public_bundle", reader)
    if failure_phase == "coverage":
        original = report.build_target_coverage

        def fail_coverage(target, *args, **kwargs):
            if target.target_id == "broken":
                raise RuntimeError("synthetic coverage failure")
            return original(target, *args, **kwargs)

        monkeypatch.setattr(report, "build_target_coverage", fail_coverage)
    elif failure_phase == "pdf":
        original = report.write_target_pdf

        def fail_pdf(path, target, *args, **kwargs):
            if target.target_id == "broken":
                raise RuntimeError("synthetic PDF failure")
            return original(path, target, *args, **kwargs)

        monkeypatch.setattr(report, "write_target_pdf", fail_pdf)
    output = tmp_path / "report"
    result = report.run_qualification(config, output, target_executor=executor)

    assert result.attempted_target_ids == ("broken", "later")
    assert (output / "trees" / "broken" / "report.pdf").is_file()
    assert (output / "trees" / "later" / "report.pdf").is_file()
    issues = read_jsonl(output / "issues.jsonl")
    assert any(
        row.get("target_id") == "broken"
        and row.get("stage") == "target_report"
        and str(row.get("code", "")).startswith("stage_failed.")
        for row in issues
    )


def test_fits_reader_is_canonical_plot_fallback_after_hdf5_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = make_direct_config(tmp_path, ("fits-fallback",))
    bundle = SimpleNamespace(
        layout_version=3,
        spectra=np.zeros((1, 16, 4), dtype=np.float32),
        spectra_units="SDU",
        spectra_representation="gain_model_input_sdu",
        spectra_metadata={"Navgf": np.array([1])},
    )

    def executor(target, work_dir, clock_adapter):
        work_dir.mkdir(parents=True, exist_ok=True)
        fits_path = work_dir / "session.fits"
        fits_path.write_bytes(b"synthetic")
        writer_issue = report.issue_record(
            target.target_id,
            "session_000",
            "hdf5",
            "stage_failed.RuntimeError",
            "synthetic HDF5 failure",
            severity="error",
        )
        return [report.SessionArtifacts(
            target_id=target.target_id,
            session_id="session_000",
            decoded_summary=family_counts(normal=1),
            h5_path=None,
            fits_path=fits_path,
            issues=(writer_issue,),
        )]

    monkeypatch.setattr(
        report,
        "inspect_fits",
        lambda path: family_counts(normal=1),
    )
    monkeypatch.setattr(
        report,
        "read_public_bundle",
        lambda path, prefer_format: report.ReaderView(bundle, np.arange(4)),
    )
    output = tmp_path / "report"
    report.run_qualification(config, output, target_executor=executor)

    coverage = read_jsonl(output / "metrics.jsonl")
    plotted = next(
        row
        for row in coverage
        if row.get("record_type") == "coverage"
        and row.get("family") == "normal"
        and row.get("stage") == "plotted"
    )
    assert plotted["state"] == "present"
    assert plotted["count"] == 1
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="ascii"))
    assert manifest["selections"]["fits-fallback/normal"] == {
        "mode": "all_valid_rows"
    }
    assert manifest["semantic_options"]["plot_source_policy"] == (
        "hdf5_public_reader_preferred_fits_fallback"
    )
    assert manifest["semantic_options"]["actual_plot_source_formats"] == [
        "fits"
    ]


def test_semantic_failure_in_one_session_does_not_skip_later_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = make_direct_config(tmp_path, ("multi-session",))

    def executor(target, work_dir, clock_adapter):
        work_dir.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for session_id in ("session_bad", "session_good"):
            h5_path = work_dir / f"{session_id}.h5"
            h5_path.write_bytes(b"synthetic")
            artifacts.append(report.SessionArtifacts(
                target_id=target.target_id,
                session_id=session_id,
                decoded_summary=family_counts(normal=1),
                h5_path=h5_path,
                fits_path=None,
            ))
        return artifacts

    monkeypatch.setattr(
        report,
        "inspect_hdf5",
        lambda path: family_counts(normal=1),
    )

    def reader(path, prefer_format):
        return report.ReaderView(
            SimpleNamespace(
                marker=path.stem,
                layout_version=3,
                spectra=np.zeros((1, 16, 4), dtype=np.float32),
                spectra_units="SDU",
                spectra_representation="gain_model_input_sdu",
                spectra_metadata={"Navgf": np.array([1])},
            ),
            np.arange(4),
        )

    monkeypatch.setattr(report, "read_public_bundle", reader)
    original_bundle_semantics = report.bundle_semantics

    def semantics(bundle, frequency_mhz=None, clock_adapter=None):
        if bundle.marker == "session_bad":
            raise RuntimeError("synthetic semantic failure")
        return original_bundle_semantics(bundle, frequency_mhz, clock_adapter)

    monkeypatch.setattr(report, "bundle_semantics", semantics)
    output = tmp_path / "report"
    report.run_qualification(config, output, target_executor=executor)

    issues = read_jsonl(output / "issues.jsonl")
    assert any(
        row.get("session_id") == "session_bad"
        and row.get("stage") == "semantic"
        and row.get("code") == "stage_failed.RuntimeError"
        for row in issues
    )
    metrics = read_jsonl(output / "metrics.jsonl")
    assert any(
        row.get("session_id") == "session_good"
        and row.get("field_path") == "normal/data"
        for row in metrics
    )
    normal_coverage = {
        row["stage"]: row
        for row in metrics
        if row.get("record_type") == "coverage"
        and row.get("family") == "normal"
    }
    assert normal_coverage["reader"]["state"] == "present"
    assert normal_coverage["reader"]["count"] == 2
    assert normal_coverage["plotted"]["state"] == "present"
    assert normal_coverage["plotted"]["count"] == 2
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="ascii"))
    assert manifest["semantic_options"]["actual_plot_source_formats"] == ["hdf5"]
    assert (output / "trees" / "multi-session" / "report.pdf").is_file()


def test_end_to_end_comparison_writes_human_report_and_uses_bound_artifacts(
    tmp_path: Path,
):
    baseline_config = make_direct_config(tmp_path, ("compare",))
    baseline_issue = report.issue_record(
        "compare",
        "session_000",
        "decoded",
        "synthetic.baseline",
        "synthetic baseline issue",
    )

    def baseline_executor(target, work_dir, clock_adapter):
        return [replace(
            empty_artifact(target.target_id),
            issues=(baseline_issue,),
        )]

    baseline = tmp_path / "baseline"
    report.run_qualification(
        baseline_config,
        baseline,
        target_executor=baseline_executor,
    )
    baseline_manifest_path = baseline / "run_manifest.json"
    baseline_manifest = json.loads(
        baseline_manifest_path.read_text(encoding="ascii")
    )
    canonical_issues = baseline / "issues.jsonl"
    bound_issues = baseline / "manifest-bound-issues.jsonl"
    bound_issues.write_bytes(canonical_issues.read_bytes())
    canonical_issues.unlink()
    baseline_manifest["artifacts"]["issues"] = {
        "path": bound_issues.name,
        "sha256": report.hash_file(bound_issues),
    }
    baseline_manifest["selections"] = {
        "compare/tr": {"session_id": "session_000", "unique_packet_id": 7}
    }
    baseline_manifest_path.write_text(
        json.dumps(baseline_manifest, sort_keys=True),
        encoding="ascii",
    )

    final_target = replace(
        baseline_config.targets[0],
        observed_families=family_counts(normal=1),
    )
    final_config = replace(baseline_config, targets=(final_target,))

    def final_executor(target, work_dir, clock_adapter):
        return [replace(
            empty_artifact(target.target_id),
            decoded_summary=family_counts(normal=1),
        )]

    final = tmp_path / "final"
    report.run_qualification(
        final_config,
        final,
        compare_to=baseline / "metrics.jsonl",
        target_executor=final_executor,
    )

    metrics = read_jsonl(final / "metrics.jsonl")
    metric_comparisons = [
        row for row in metrics if row.get("record_type") == "comparison"
    ]
    issue_comparisons = [
        row for row in metrics if row.get("record_type") == "issue_comparison"
    ]
    selection_comparisons = [
        row for row in metrics if row.get("record_type") == "selection_comparison"
    ]
    assert any(row.get("status") == "changed" for row in metric_comparisons)
    assert any(row.get("status") == "changed" for row in issue_comparisons)
    synthetic_issue_comparison = next(
        row for row in issue_comparisons
        if row.get("code") == "synthetic.baseline"
    )
    assert synthetic_issue_comparison["previous_count"] == 1
    assert synthetic_issue_comparison["current_count"] == 0
    assert selection_comparisons == [{
        "record_type": "selection_comparison",
        "target_id": "compare",
        "family": "tr",
        "field_path": "compare/tr",
        "status": "removed",
        "previous": {
            "session_id": "session_000",
            "unique_packet_id": 7,
        },
    }]
    comparison_pdf = final / "baseline_final_comparison.pdf"
    assert comparison_pdf.is_file() and comparison_pdf.stat().st_size > 0
    final_manifest = json.loads((final / "run_manifest.json").read_text(encoding="ascii"))
    artifact = final_manifest["artifacts"]["comparison_pdf"]
    assert artifact["sha256"] == report.hash_file(comparison_pdf)
    lines = report.build_comparison_summary_lines(
        metric_comparisons,
        issue_comparisons,
        selection_comparisons,
        final_config.targets,
        final_manifest["comparison_baseline_reference"],
        report.comparison_run_reference(
            final / "run_manifest.json",
            final_manifest,
        ),
    )
    text = "\n".join(lines)
    assert "baseline=" in text and "final=" in text
    assert "Issue-code count changes" in text
    assert "Frozen-selection changes" in text
    assert "<compare-to>/trees/compare/report.pdf" in text
    assert "./trees/compare/report.pdf" in text


@pytest.mark.parametrize(
    ("artifact_name", "mutation", "message"),
    (
        ("issues.jsonl", "missing", "issues artifact is missing"),
        ("corpus_summary.pdf", "tampered", "summary PDF artifact digest mismatch"),
    ),
)
def test_comparison_rejects_missing_or_tampered_required_artifact(
    tmp_path: Path,
    artifact_name: str,
    mutation: str,
    message: str,
):
    config = make_direct_config(tmp_path, ("artifact-binding",))
    baseline = tmp_path / "baseline"
    report.run_qualification(
        config,
        baseline,
        target_executor=lambda target, work_dir, clock_adapter: [
            empty_artifact(target.target_id)
        ],
    )
    artifact = baseline / artifact_name
    if mutation == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(artifact.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=message):
        report.resolve_comparison_files(baseline / "metrics.jsonl")
