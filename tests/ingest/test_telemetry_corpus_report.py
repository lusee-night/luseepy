"""Focused tests for the standalone telemetry corpus PDF report."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import lusee.ingest as ingest
from lusee.ingest import TelemetryData
from scripts import telemetry_corpus_report as report


def field_names() -> tuple[str, ...]:
    names = [*report.GRAHAM_REQUIRED_FIELDS, report.LOW_ANALOG_FIELD]
    names.extend(
        f"EXTRA_{index:02d}"
        for index in range(57 - len(names))
    )
    return tuple(names)


def telemetry_table(
    row_count: int,
    *,
    invalid: tuple[int, str] | None = None,
    low_analog_row: int | None = None,
) -> TelemetryData:
    names = field_names()
    values = np.ones((row_count, 57), dtype=np.float64)
    valid = np.ones((row_count, 57), dtype=np.bool_)
    values[:, names.index("THERM_FPGA")] = 30.4
    values[:, names.index("SPE_ADC0_T")] = 29.8
    values[:, names.index("SPE_ADC1_T")] = 28.5
    values[:, names.index("SPE_1VA8_V")] = 1.8
    values[:, names.index("SPE_1VAD8_V")] = 1.799
    values[:, names.index("VMON_1V2D")] = 1.201
    values[:, names.index("SPE_1VAD8_C")] = 0.045
    values[:, names.index("PFPS_PA0_T")] = 25.0
    values[:, names.index("PFPS_PA1_T")] = 26.0
    values[:, names.index("PFPS_PA2_T")] = 27.0
    values[:, names.index("PFPS_PA3_T")] = 28.0
    if invalid is not None:
        row, name = invalid
        index = names.index(name)
        values[row, index] = np.nan
        valid[row, index] = False
    if low_analog_row is not None:
        values[low_analog_row, names.index("SPE_1VA8_V")] = 0.0
    return TelemetryData(
        source_kind="legacy_binary_sidecar",
        field_names=names,
        units=tuple("unit" for _ in names),
        source_indices=np.arange(row_count, dtype=np.int64),
        mission_seconds=np.arange(100, 100 + row_count, dtype=np.uint32),
        lusee_subsecs=np.zeros(row_count, dtype=np.uint16),
        mjd_times=np.full(row_count, np.nan, dtype=np.float64),
        raw_counts=np.ones((row_count, 57), dtype=np.uint16),
        values=values,
        valid=valid,
    )


def decoded_tree(
    telemetry: TelemetryData | None,
    *,
    input_packets: int | None = None,
) -> report.DecodedTree:
    row_count = 0 if telemetry is None else telemetry.row_count
    return report.DecodedTree(
        target_id="tree-001",
        source_path=Path("/private/corpus/tree-001"),
        telemetry_path=Path("/private/corpus/tree-001/DCB_telemetry.json"),
        source_kind="legacy_binary_sidecar",
        input_packets=row_count if input_packets is None else input_packets,
        telemetry=telemetry,
    )


def assess(telemetry: TelemetryData, *, input_packets: int | None = None):
    return report.assess_tree(
        decoded_tree(telemetry, input_packets=input_packets),
        low_analog_threshold=0.015,
        gain_validator=lambda _telemetry, _mask: None,
    )


def test_four_values_plot_and_three_values_table():
    enough = assess(telemetry_table(4))
    short = assess(telemetry_table(3))

    assert enough.quality == "good" and enough.enough_data
    assert report.split_field_indices(enough) == (list(range(57)), [])
    assert short.quality == "good" and not short.enough_data
    assert report.split_field_indices(short) == ([], list(range(57)))


def test_noncritical_invalid_and_low_analog_packets_are_dropped_and_counted():
    noncritical = assess(
        telemetry_table(5, invalid=(1, "EXTRA_00")),
        input_packets=6,
    )
    low_analog = assess(telemetry_table(5, low_analog_row=2))

    assert noncritical.quality == "good"
    assert noncritical.structural_bad_packets == 1
    assert noncritical.conversion_bad_packets == 1
    assert noncritical.bad_packets == 2
    assert noncritical.retained_packets == 4
    assert low_analog.quality == "good"
    assert low_analog.low_analog_bad_packets == 1
    assert low_analog.bad_packets == 1
    assert low_analog.retained_packets == 4


def test_invalid_analog_value_is_conversion_bad_but_not_low_analog():
    assessment = assess(
        telemetry_table(5, invalid=(1, "SPE_1VA8_V")),
    )

    assert assessment.quality == "good"
    assert assessment.conversion_bad_packets == 1
    assert assessment.low_analog_bad_packets == 0
    assert assessment.bad_packets == 1


def test_low_analog_and_conversion_categories_overlap_but_total_does_not():
    assessment = assess(telemetry_table(
        5,
        invalid=(1, "SPE_ADC0_T"),
        low_analog_row=1,
    ))

    assert assessment.quality == "bad"
    assert assessment.conversion_bad_packets == 1
    assert assessment.low_analog_bad_packets == 1
    assert assessment.bad_packets == 1


def test_invalid_graham_field_marks_whole_tree_bad():
    assessment = assess(
        telemetry_table(5, invalid=(2, "PFPS_PA2_T")),
    )

    assert assessment.quality == "bad"
    assert assessment.serious_fields == ("PFPS_PA2_T",)
    assert assessment.gain_check == "not_run"
    assert report.field_statistics(assessment) == []


def test_native_gain_failure_marks_tree_bad():
    assessment = report.assess_tree(
        decoded_tree(telemetry_table(5)),
        low_analog_threshold=0.015,
        gain_validator=lambda _telemetry, _mask: "synthetic nonpositive gain",
    )

    assert assessment.quality == "bad"
    assert assessment.reason == "synthetic nonpositive gain"
    assert assessment.gain_check == "failed"


def test_inventory_contradiction_does_not_invent_bad_packet_rows():
    assessment = assess(telemetry_table(5), input_packets=4)

    assert assessment.quality == "bad"
    assert assessment.bad_packets == 0
    assert assessment.retained_packets == 5
    assert assessment.reason == "decoded 5 packets from an inventory of 4"


def test_configured_raw_telemetry_without_surviving_0x314_is_bad(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "raw"
    bank = source / "b01" / "FFFFFFFE"
    bank.parent.mkdir(parents=True)
    bank.write_bytes(b"damaged")
    target = SimpleNamespace(
        target_id="raw-damaged",
        source_path=source,
        observed_family_metadata={
            "telemetry": SimpleNamespace(input_state="present")
        },
    )
    monkeypatch.setattr(ingest, "parse_bank_file", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        ingest,
        "reassemble_logical_packets",
        lambda *args, **kwargs: iter(()),
    )

    decoded = report.decode_raw_target(target)
    assessment = report.assess_tree(
        decoded,
        low_analog_threshold=0.015,
        gain_validator=lambda _telemetry, _mask: None,
    )

    assert decoded is not None
    assert decoded.failure == "no complete logical 0x314 packets recovered"
    assert assessment.quality == "bad"
    assert assessment.reason == decoded.failure


def test_empty_input_is_bad_and_status_only(tmp_path: Path):
    assessment = assess(telemetry_table(0))
    path = tmp_path / "empty.pdf"

    assert assessment.quality == "bad"
    assert report.write_tree_pdf(path, assessment) == 1
    assert path.read_bytes().startswith(b"%PDF")


def test_good_plot_and_short_table_pdf_page_counts(tmp_path: Path):
    enough = assess(telemetry_table(4))
    short = assess(telemetry_table(3))
    plot_path = tmp_path / "plots.pdf"
    table_path = tmp_path / "tables.pdf"

    assert report.write_tree_pdf(plot_path, enough) == 11
    assert report.write_tree_pdf(table_path, short) == 4
    assert plot_path.stat().st_size > table_path.stat().st_size > 1_000


def test_aggregate_keeps_good_short_separate_from_good_enough():
    records = [
        {
            "source_kind": "legacy_binary_sidecar",
            "source_path": "/z",
            "quality": "good",
            "enough_data": False,
            "input_packets": 3,
            "bad_packets": 0,
            "retained_packets": 3,
        },
        {
            "source_kind": "legacy_binary_sidecar",
            "source_path": "/b",
            "quality": "good",
            "enough_data": True,
            "input_packets": 4,
            "bad_packets": 0,
            "retained_packets": 4,
        },
        {
            "source_kind": "legacy_binary_sidecar",
            "source_path": "/a",
            "quality": "good",
            "enough_data": True,
            "input_packets": 5,
            "bad_packets": 1,
            "retained_packets": 4,
        },
        {
            "source_kind": "b01_0x314",
            "source_path": "/raw",
            "quality": "bad",
            "enough_data": False,
            "input_packets": 2,
            "bad_packets": 2,
            "retained_packets": 0,
        },
    ]

    counts = report.aggregate_records(records)

    legacy = counts["legacy_binary_sidecar"]
    assert legacy["good_trees"] == 3
    assert legacy["good_enough_trees"] == 2
    assert legacy["good_short_trees"] == 1
    assert legacy["example_good_enough_directory"] == "/a"
    assert counts["b01_0x314"]["bad_trees"] == 1
    assert counts["all"] == {
        "trees": 4,
        "good_trees": 3,
        "bad_trees": 1,
        "good_enough_trees": 2,
        "good_short_trees": 1,
    }


def test_csv_retains_structured_ccsds_diagnostics(tmp_path: Path):
    path = tmp_path / "summary.csv"
    record = {
        "target_id": "raw-1",
        "source_kind": "b01_0x314",
        "source_path": "/raw-1",
        "telemetry_path": "/raw-1/b01/FFFFFFFE",
        "quality": "bad",
        "enough_data": False,
        "input_packets": 0,
        "decoded_packets": 0,
        "structural_bad_packets": 0,
        "conversion_bad_packets": 0,
        "low_analog_bad_packets": 0,
        "bad_packets": 0,
        "retained_packets": 0,
        "serious_fields": [],
        "reason": "no complete logical 0x314 packets recovered",
        "gain_check": "not_run",
        "ccsds_warning_count_ignored_for_quality": 1,
        "ccsds_issue_counts_ignored_for_quality": {
            "reassembly.trailing_partial_packet": 2,
            "ccsds.crc_mismatch": 1,
        },
        "pdf_path": "/report/raw-1.pdf",
        "pdf_pages": 1,
    }

    report.write_csv(path, [record])

    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["ccsds_issue_counts_ignored_for_quality"] == (
        '{"ccsds.crc_mismatch":1,"reassembly.trailing_partial_packet":2}'
    )
