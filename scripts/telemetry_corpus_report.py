#!/usr/bin/env python3
"""Build one telemetry PDF per configured ingest corpus tree.

The private qualification config supplies corpus paths. This script keeps the
production ingest path unchanged: legacy sidecars go through
``decode_legacy_sidecar``; raw b01 banks go through the existing CCSDS parser,
logical-packet reassembler, and ``decode_b01_packets``. It requires the
optional private decoder and writes private paths only to the caller-selected
output directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import textwrap
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


REPORT_SCHEMA_VERSION = 1
PLOT_VALUES_MINIMUM = 4
LOW_ANALOG_FIELD = "SPE_1VA8_V"

# Graham's gain model uses the first six fields. His paired preamp-noise model
# additionally uses the four per-channel preamp temperatures. There are no
# reviewed numeric operating ranges in luseepy, so validity is checked here
# without inventing bounds.
GRAHAM_REQUIRED_FIELDS = (
    "THERM_FPGA",
    "SPE_ADC0_T",
    "SPE_ADC1_T",
    "SPE_1VAD8_V",
    "VMON_1V2D",
    "SPE_1VAD8_C",
    "PFPS_PA0_T",
    "PFPS_PA1_T",
    "PFPS_PA2_T",
    "PFPS_PA3_T",
)
GAIN_MODEL_FIELDS = GRAHAM_REQUIRED_FIELDS[:6]


@dataclass(frozen=True, slots=True)
class DecodedTree:
    """Telemetry decoder output plus counts known before decoding."""

    target_id: str
    source_path: Path
    telemetry_path: Path
    source_kind: Literal["legacy_binary_sidecar", "b01_0x314"]
    input_packets: int
    telemetry: Any | None
    framing_warnings: tuple[str, ...] = ()
    framing_issue_counts: tuple[tuple[str, int], ...] = ()
    decoder_warnings: tuple[str, ...] = ()
    failure: str | None = None


@dataclass(frozen=True, slots=True)
class TreeAssessment:
    """One tree's classification and the retained rows used for its PDF."""

    decoded: DecodedTree
    quality: Literal["good", "bad"]
    enough_data: bool
    decoded_packets: int
    structural_bad_packets: int
    conversion_bad_packets: int
    low_analog_bad_packets: int
    bad_packets: int
    retained_packets: int
    retained_mask: np.ndarray
    serious_fields: tuple[str, ...] = ()
    reason: str | None = None
    gain_check: str = "not_run"


def capture_warnings(function: Callable[[], Any]) -> tuple[Any, tuple[str, ...]]:
    """Run one stage and return its result and warning messages."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, tuple(str(item.message) for item in caught)


def load_qualification_config(path: Path | str):
    """Reuse the existing private-corpus selection without copying its schema."""

    try:
        from scripts.ingest_qualification_report import load_config
    except ModuleNotFoundError:
        # Direct ``python scripts/telemetry_corpus_report.py`` execution puts
        # scripts/, rather than the repository root, first on sys.path.
        from ingest_qualification_report import load_config
    return load_config(path)


def load_low_analog_threshold() -> float:
    """Read the established bad-packet threshold from the private decoder."""

    try:
        implementation = importlib.import_module("lusee_telemetry._impl")
        value = implementation.BAD_PACKET_SPE_1VA8_V_THRESHOLD
    except (AttributeError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "this report requires the installed private lusee_telemetry decoder"
        ) from exc
    threshold = float(value)
    if not np.isfinite(threshold):
        raise ValueError("private decoder bad-packet threshold is not finite")
    return threshold


def decode_legacy_target(target) -> DecodedTree:
    """Decode one configured legacy binary sidecar."""

    from lusee.ingest import decode_legacy_sidecar

    metadata = target.observed_family_metadata["telemetry"]
    if metadata.unit != "legacy_sidecar_record":
        raise ValueError(
            f"{target.target_id}: telemetry inventory is not sidecar records"
        )
    input_packets = int(target.observed_families["telemetry"])
    try:
        telemetry, decoder_warnings = capture_warnings(
            lambda: decode_legacy_sidecar(target.telemetry_sidecar)
        )
    except Exception as exc:  # noqa: BLE001
        return DecodedTree(
            target_id=target.target_id,
            source_path=target.source_path,
            telemetry_path=target.telemetry_sidecar,
            source_kind="legacy_binary_sidecar",
            input_packets=input_packets,
            telemetry=None,
            failure=f"{type(exc).__name__}: {exc}",
        )
    return DecodedTree(
        target_id=target.target_id,
        source_path=target.source_path,
        telemetry_path=target.telemetry_sidecar,
        source_kind="legacy_binary_sidecar",
        input_packets=input_packets,
        telemetry=telemetry,
        decoder_warnings=decoder_warnings,
    )


def decode_raw_target(target) -> DecodedTree | None:
    """Recover and decode complete logical 0x314 packets from one b01 bank."""

    from lusee.ingest import (
        IssueCollector,
        decode_b01_packets,
        parse_bank_file,
        reassemble_logical_packets,
    )
    from lusee.ingest.constants import BANK_FILENAME, TELEMETRY_BANK
    from lusee.ingest.telemetry import TELEMETRY_APPID

    bank_path = target.source_path / TELEMETRY_BANK / BANK_FILENAME
    configured_present = (
        target.observed_family_metadata["telemetry"].input_state != "absent"
    )
    if not bank_path.is_file():
        if not configured_present:
            return None
        return DecodedTree(
            target_id=target.target_id,
            source_path=target.source_path,
            telemetry_path=bank_path,
            source_kind="b01_0x314",
            input_packets=0,
            telemetry=None,
            failure="configured telemetry input has no readable b01 bank",
        )

    collector = IssueCollector()

    def recover_packets():
        frames = parse_bank_file(
            bank_path,
            bank=TELEMETRY_BANK,
            issue_collector=collector,
        )
        return tuple(reassemble_logical_packets(
            frames,
            byteswap_pairs=False,
            bank=TELEMETRY_BANK,
            issue_collector=collector,
        ))

    try:
        packets, framing_warnings = capture_warnings(recover_packets)
    except Exception as exc:  # noqa: BLE001
        return DecodedTree(
            target_id=target.target_id,
            source_path=target.source_path,
            telemetry_path=bank_path,
            source_kind="b01_0x314",
            input_packets=0,
            telemetry=None,
            framing_issue_counts=tuple(collector.counts().items()),
            failure=f"CCSDS recovery failed: {type(exc).__name__}: {exc}",
        )
    selected = tuple(packet for packet in packets if packet.appid == TELEMETRY_APPID)
    if not selected:
        if not configured_present:
            return None
        return DecodedTree(
            target_id=target.target_id,
            source_path=target.source_path,
            telemetry_path=bank_path,
            source_kind="b01_0x314",
            input_packets=0,
            telemetry=None,
            framing_warnings=framing_warnings,
            framing_issue_counts=tuple(collector.counts().items()),
            failure="no complete logical 0x314 packets recovered",
        )
    try:
        telemetry, decoder_warnings = capture_warnings(
            lambda: decode_b01_packets(selected)
        )
    except Exception as exc:  # noqa: BLE001
        return DecodedTree(
            target_id=target.target_id,
            source_path=target.source_path,
            telemetry_path=bank_path,
            source_kind="b01_0x314",
            input_packets=len(selected),
            telemetry=None,
            framing_warnings=framing_warnings,
            framing_issue_counts=tuple(collector.counts().items()),
            failure=f"{type(exc).__name__}: {exc}",
        )
    return DecodedTree(
        target_id=target.target_id,
        source_path=target.source_path,
        telemetry_path=bank_path,
        source_kind="b01_0x314",
        input_packets=len(selected),
        telemetry=telemetry,
        framing_warnings=framing_warnings,
        framing_issue_counts=tuple(collector.counts().items()),
        decoder_warnings=decoder_warnings,
    )


def validate_native_gains(telemetry: Any, row_mask: np.ndarray) -> str | None:
    """Require positive finite gain at all native anchors for L, M, and H."""

    from lusee.GainModel import SpectrometerGain

    model = SpectrometerGain()
    values = {
        name: telemetry.values[row_mask, telemetry.field_names.index(name)]
        for name in GAIN_MODEL_FIELDS
    }
    for level in ("L", "M", "H"):
        try:
            gains = model.predict_gain_batch(level, values)
        except Exception as exc:  # noqa: BLE001
            return f"{level}-gain prediction failed: {type(exc).__name__}: {exc}"
        invalid_rows = ~np.all(np.isfinite(gains) & (gains > 0.0), axis=(1, 2))
        count = int(np.count_nonzero(invalid_rows))
        if count:
            return (
                f"{count} retained packet(s) produce nonpositive or nonfinite "
                f"{level}-gain at native model anchors"
            )
    return None


def bad_assessment(
    decoded: DecodedTree,
    *,
    decoded_packets: int,
    structural_bad_packets: int,
    conversion_bad_packets: int,
    low_analog_bad_packets: int,
    retained_mask: np.ndarray,
    serious_fields: Sequence[str] = (),
    reason: str,
    gain_check: str = "not_run",
) -> TreeAssessment:
    row_bad_packets = int(np.count_nonzero(~retained_mask))
    return TreeAssessment(
        decoded=decoded,
        quality="bad",
        enough_data=False,
        decoded_packets=decoded_packets,
        structural_bad_packets=structural_bad_packets,
        conversion_bad_packets=conversion_bad_packets,
        low_analog_bad_packets=low_analog_bad_packets,
        bad_packets=structural_bad_packets + row_bad_packets,
        retained_packets=int(np.count_nonzero(retained_mask)),
        retained_mask=retained_mask,
        serious_fields=tuple(serious_fields),
        reason=reason,
        gain_check=gain_check,
    )


def assess_tree(
    decoded: DecodedTree,
    *,
    low_analog_threshold: float,
    gain_validator: Callable[[Any, np.ndarray], str | None] = validate_native_gains,
) -> TreeAssessment:
    """Classify a tree, dropping ordinary bad packets before field plots."""

    telemetry = decoded.telemetry
    if telemetry is None:
        return bad_assessment(
            decoded,
            decoded_packets=0,
            structural_bad_packets=decoded.input_packets,
            conversion_bad_packets=0,
            low_analog_bad_packets=0,
            retained_mask=np.zeros(0, dtype=np.bool_),
            reason=decoded.failure or "telemetry decoder returned no table",
        )

    row_count = int(telemetry.row_count)
    if decoded.input_packets < row_count:
        return bad_assessment(
            decoded,
            decoded_packets=row_count,
            structural_bad_packets=0,
            conversion_bad_packets=0,
            low_analog_bad_packets=0,
            retained_mask=np.ones(row_count, dtype=np.bool_),
            reason=(
                f"decoded {row_count} packets from an inventory of "
                f"{decoded.input_packets}"
            ),
        )
    structural_bad = decoded.input_packets - row_count
    if row_count == 0:
        return bad_assessment(
            decoded,
            decoded_packets=0,
            structural_bad_packets=structural_bad,
            conversion_bad_packets=0,
            low_analog_bad_packets=0,
            retained_mask=np.zeros(0, dtype=np.bool_),
            reason="telemetry input contains no decoded packets",
        )

    field_indices = {
        name: index for index, name in enumerate(telemetry.field_names)
    }
    missing = [
        name
        for name in (*GRAHAM_REQUIRED_FIELDS, LOW_ANALOG_FIELD)
        if name not in field_indices
    ]
    if missing:
        return bad_assessment(
            decoded,
            decoded_packets=row_count,
            structural_bad_packets=structural_bad,
            conversion_bad_packets=0,
            low_analog_bad_packets=0,
            retained_mask=np.zeros(row_count, dtype=np.bool_),
            serious_fields=missing,
            reason=f"required telemetry fields are missing: {missing}",
        )

    conversion_bad_mask = ~np.all(telemetry.valid, axis=1)
    analog_index = field_indices[LOW_ANALOG_FIELD]
    analog_values = telemetry.values[:, analog_index]
    low_analog_mask = (
        telemetry.valid[:, analog_index]
        & np.isfinite(analog_values)
        & (analog_values < low_analog_threshold)
    )
    row_bad_mask = conversion_bad_mask | low_analog_mask
    retained_mask = ~row_bad_mask

    serious_fields = []
    for name in GRAHAM_REQUIRED_FIELDS:
        index = field_indices[name]
        usable = telemetry.valid[:, index] & np.isfinite(telemetry.values[:, index])
        if not np.all(usable):
            serious_fields.append(name)
    conversion_bad = int(np.count_nonzero(conversion_bad_mask))
    low_analog_bad = int(np.count_nonzero(low_analog_mask))
    if serious_fields:
        return bad_assessment(
            decoded,
            decoded_packets=row_count,
            structural_bad_packets=structural_bad,
            conversion_bad_packets=conversion_bad,
            low_analog_bad_packets=low_analog_bad,
            retained_mask=retained_mask,
            serious_fields=serious_fields,
            reason=(
                "invalid engineering values in Graham gain/noise field(s): "
                + ", ".join(serious_fields)
            ),
        )
    if not np.any(retained_mask):
        return bad_assessment(
            decoded,
            decoded_packets=row_count,
            structural_bad_packets=structural_bad,
            conversion_bad_packets=conversion_bad,
            low_analog_bad_packets=low_analog_bad,
            retained_mask=retained_mask,
            reason="no telemetry packets remain after bad-packet filtering",
        )

    gain_failure = gain_validator(telemetry, retained_mask)
    if gain_failure is not None:
        return bad_assessment(
            decoded,
            decoded_packets=row_count,
            structural_bad_packets=structural_bad,
            conversion_bad_packets=conversion_bad,
            low_analog_bad_packets=low_analog_bad,
            retained_mask=retained_mask,
            reason=gain_failure,
            gain_check="failed",
        )

    retained = int(np.count_nonzero(retained_mask))
    return TreeAssessment(
        decoded=decoded,
        quality="good",
        enough_data=retained >= PLOT_VALUES_MINIMUM,
        decoded_packets=row_count,
        structural_bad_packets=structural_bad,
        conversion_bad_packets=conversion_bad,
        low_analog_bad_packets=low_analog_bad,
        bad_packets=structural_bad + int(np.count_nonzero(row_bad_mask)),
        retained_packets=retained,
        retained_mask=retained_mask,
        gain_check="passed",
    )


def field_statistics(assessment: TreeAssessment) -> list[dict[str, object]]:
    """Return plot/table statistics in the decoder's fixed field order."""

    if assessment.quality != "good":
        return []
    telemetry = assessment.decoded.telemetry
    result = []
    for index, (name, unit) in enumerate(zip(
        telemetry.field_names,
        telemetry.units,
    )):
        values = telemetry.values[assessment.retained_mask, index]
        values = values[np.isfinite(values)]
        record: dict[str, object] = {
            "index": index,
            "name": name,
            "unit": unit,
            "count": int(values.size),
            "minimum": None,
            "median": None,
            "maximum": None,
        }
        if values.size:
            record.update({
                "minimum": float(np.min(values)),
                "median": float(np.median(values)),
                "maximum": float(np.max(values)),
            })
        result.append(record)
    return result


def split_field_indices(assessment: TreeAssessment) -> tuple[list[int], list[int]]:
    """Split fields at the requested more-than-three-values boundary."""

    plot_indices = []
    table_indices = []
    for record in field_statistics(assessment):
        destination = (
            plot_indices
            if int(record["count"]) >= PLOT_VALUES_MINIMUM
            else table_indices
        )
        destination.append(int(record["index"]))
    return plot_indices, table_indices


def add_page_footer(figure, page_number: int) -> None:
    figure.text(
        0.97,
        0.018,
        f"page {page_number}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.4",
    )


def cover_figure(assessment: TreeAssessment):
    import matplotlib.pyplot as plt

    decoded = assessment.decoded
    figure = plt.figure(figsize=(11, 8.5))
    color = "#147d3f" if assessment.quality == "good" else "#a61b1b"
    status = "GOOD" if assessment.quality == "good" else "BAD TREE"
    if assessment.quality == "good" and not assessment.enough_data:
        status = "GOOD - SHORT TABLE"
    figure.text(0.055, 0.94, "LuSEE telemetry corpus report", fontsize=20, weight="bold")
    figure.text(0.055, 0.875, status, fontsize=18, weight="bold", color=color)

    lines = [
        f"target: {decoded.target_id}",
        f"source kind: {decoded.source_kind}",
        f"tree directory: {decoded.source_path}",
        f"telemetry input: {decoded.telemetry_path}",
        "",
        f"input packets: {decoded.input_packets}",
        f"decoded packets: {assessment.decoded_packets}",
        f"bad packets dropped: {assessment.bad_packets}",
        f"  structural/malformed: {assessment.structural_bad_packets}",
        f"  conversion-invalid rows*: {assessment.conversion_bad_packets}",
        f"  low analog-rail rows*: {assessment.low_analog_bad_packets}",
        "  * row categories may overlap; the bad-packet total counts their union",
        f"retained packets: {assessment.retained_packets}",
        f"native-anchor gain check: {assessment.gain_check}",
    ]
    if assessment.serious_fields:
        lines.append("serious fields: " + ", ".join(assessment.serious_fields))
    if assessment.reason:
        lines.extend(("", "reason: " + assessment.reason))
    if decoded.decoder_warnings:
        lines.extend(("", "decoder warnings:"))
        lines.extend(f"- {message}" for message in decoded.decoder_warnings)
    if decoded.framing_warnings:
        lines.extend((
            "",
            f"CCSDS diagnostics ignored for quality: {len(decoded.framing_warnings)} warning(s)",
        ))
    if decoded.framing_issue_counts:
        issue_count = sum(count for _code, count in decoded.framing_issue_counts)
        lines.append(
            f"CCSDS structured issues ignored for quality: {issue_count}"
        )

    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(
            line,
            width=112,
            subsequent_indent="  ",
        ) or [""])
    figure.text(
        0.058,
        0.82,
        "\n".join(wrapped[:40]),
        family="monospace",
        fontsize=8.2,
        va="top",
        linespacing=1.35,
    )
    if assessment.quality == "bad":
        figure.text(
            0.055,
            0.075,
            "Field plots intentionally skipped for this tree.",
            fontsize=11,
            weight="bold",
            color=color,
        )
    return figure


def plot_page_figure(
    assessment: TreeAssessment,
    field_indices: Sequence[int],
    page_number: int,
):
    import matplotlib.pyplot as plt

    telemetry = assessment.decoded.telemetry
    figure, axes = plt.subplots(3, 2, figsize=(11, 8.5))
    figure.suptitle(
        f"{assessment.decoded.target_id} - telemetry fields",
        fontsize=13,
        weight="bold",
    )
    retained_times = telemetry.raw_seconds[assessment.retained_mask]
    x = retained_times - retained_times[0]
    for axis, field_index in zip(axes.flat, field_indices):
        name = telemetry.field_names[field_index]
        unit = telemetry.units[field_index]
        values = telemetry.values[assessment.retained_mask, field_index]
        finite = np.isfinite(values)
        axis.plot(
            x[finite],
            values[finite],
            marker=".",
            markersize=2.5,
            linewidth=0.65,
            color="#235789",
        )
        axis.set_title(f"{field_index:02d}  {name} [{unit or 'unitless'}]", fontsize=9)
        axis.set_xlabel("seconds since first retained packet", fontsize=7)
        axis.tick_params(labelsize=7)
        axis.grid(True, linewidth=0.35, alpha=0.45)
        finite_values = values[finite]
        stats = (
            f"n={finite_values.size}  min={np.min(finite_values):.6g}\n"
            f"median={np.median(finite_values):.6g}  max={np.max(finite_values):.6g}"
        )
        axis.text(
            0.01,
            0.98,
            stats,
            transform=axis.transAxes,
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )
    for axis in axes.flat[len(field_indices):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0.025, 0.035, 0.975, 0.94))
    add_page_footer(figure, page_number)
    return figure


def table_page_figure(
    assessment: TreeAssessment,
    field_indices: Sequence[int],
    page_number: int,
):
    import matplotlib.pyplot as plt

    telemetry = assessment.decoded.telemetry
    figure, axis = plt.subplots(figsize=(11, 8.5))
    axis.axis("off")
    figure.suptitle(
        f"{assessment.decoded.target_id} - fields with at most three values",
        fontsize=13,
        weight="bold",
        y=0.96,
    )
    rows = []
    for field_index in field_indices:
        values = telemetry.values[assessment.retained_mask, field_index]
        values = values[np.isfinite(values)]
        rows.append([
            f"{field_index:02d}",
            telemetry.field_names[field_index],
            telemetry.units[field_index] or "unitless",
            str(values.size),
            ", ".join(f"{value:.8g}" for value in values) or "-",
        ])
    table = axis.table(
        cellText=rows,
        colLabels=("#", "field", "unit", "N", "values"),
        cellLoc="left",
        colLoc="left",
        bbox=(0.025, 0.06, 0.95, 0.84),
        colWidths=(0.05, 0.27, 0.11, 0.06, 0.51),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#dce8f2")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7f9")
    add_page_footer(figure, page_number)
    return figure


def write_tree_pdf(path: Path, assessment: TreeAssessment) -> int:
    """Write one tree PDF and return its page count."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    page_number = 1
    with PdfPages(
        path,
        metadata={
            "Title": f"LuSEE telemetry - {assessment.decoded.target_id}",
            "CreationDate": None,
            "ModDate": None,
        },
    ) as pdf:
        cover = cover_figure(assessment)
        add_page_footer(cover, page_number)
        pdf.savefig(cover)
        plt.close(cover)
        if assessment.quality == "bad":
            return page_number

        plot_indices, table_indices = split_field_indices(assessment)
        for start in range(0, len(plot_indices), 6):
            page_number += 1
            figure = plot_page_figure(
                assessment,
                plot_indices[start:start + 6],
                page_number,
            )
            pdf.savefig(figure)
            plt.close(figure)
        for start in range(0, len(table_indices), 22):
            page_number += 1
            figure = table_page_figure(
                assessment,
                table_indices[start:start + 22],
                page_number,
            )
            pdf.savefig(figure)
            plt.close(figure)
    return page_number


def assessment_record(
    assessment: TreeAssessment,
    *,
    pdf_path: Path,
    page_count: int,
) -> dict[str, object]:
    decoded = assessment.decoded
    return {
        "target_id": decoded.target_id,
        "source_kind": decoded.source_kind,
        "source_path": str(decoded.source_path),
        "telemetry_path": str(decoded.telemetry_path),
        "quality": assessment.quality,
        "enough_data": assessment.enough_data,
        "input_packets": decoded.input_packets,
        "decoded_packets": assessment.decoded_packets,
        "structural_bad_packets": assessment.structural_bad_packets,
        "conversion_bad_packets": assessment.conversion_bad_packets,
        "low_analog_bad_packets": assessment.low_analog_bad_packets,
        "bad_packets": assessment.bad_packets,
        "retained_packets": assessment.retained_packets,
        "serious_fields": list(assessment.serious_fields),
        "reason": assessment.reason,
        "gain_check": assessment.gain_check,
        "decoder_warnings": list(decoded.decoder_warnings),
        "ccsds_warning_count_ignored_for_quality": len(decoded.framing_warnings),
        "ccsds_issue_counts_ignored_for_quality": dict(
            decoded.framing_issue_counts
        ),
        "pdf_path": str(pdf_path.resolve()),
        "pdf_pages": page_count,
        "field_statistics": field_statistics(assessment),
    }


def aggregate_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute good/bad counts and example paths for each source category."""

    result: dict[str, object] = {}
    for source_kind in ("legacy_binary_sidecar", "b01_0x314"):
        selected = [row for row in records if row["source_kind"] == source_kind]
        good = [row for row in selected if row["quality"] == "good"]
        bad = [row for row in selected if row["quality"] == "bad"]
        good_enough = [row for row in good if bool(row["enough_data"])]
        short = [row for row in good if not bool(row["enough_data"])]
        examples = sorted(str(row["source_path"]) for row in good_enough)
        result[source_kind] = {
            "trees": len(selected),
            "good_trees": len(good),
            "bad_trees": len(bad),
            "good_enough_trees": len(good_enough),
            "good_short_trees": len(short),
            "input_packets": sum(int(row["input_packets"]) for row in selected),
            "bad_packets": sum(int(row["bad_packets"]) for row in selected),
            "retained_packets": sum(int(row["retained_packets"]) for row in selected),
            "example_good_enough_directory": examples[0] if examples else None,
        }
    all_rows = list(records)
    result["all"] = {
        "trees": len(all_rows),
        "good_trees": sum(row["quality"] == "good" for row in all_rows),
        "bad_trees": sum(row["quality"] == "bad" for row in all_rows),
        "good_enough_trees": sum(
            row["quality"] == "good" and bool(row["enough_data"])
            for row in all_rows
        ),
        "good_short_trees": sum(
            row["quality"] == "good" and not bool(row["enough_data"])
            for row in all_rows
        ),
    }
    return result


def write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    field_names = (
        "target_id",
        "source_kind",
        "source_path",
        "telemetry_path",
        "quality",
        "enough_data",
        "input_packets",
        "decoded_packets",
        "structural_bad_packets",
        "conversion_bad_packets",
        "low_analog_bad_packets",
        "bad_packets",
        "retained_packets",
        "serious_fields",
        "reason",
        "gain_check",
        "ccsds_warning_count_ignored_for_quality",
        "ccsds_issue_counts_ignored_for_quality",
        "pdf_path",
        "pdf_pages",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = {name: record.get(name, "") for name in field_names}
            row["serious_fields"] = ",".join(record["serious_fields"])
            row["ccsds_issue_counts_ignored_for_quality"] = json.dumps(
                record["ccsds_issue_counts_ignored_for_quality"],
                sort_keys=True,
                separators=(",", ":"),
            )
            writer.writerow(row)


def run_report(config_path: Path, output_dir: Path) -> dict[str, object]:
    """Decode every recognized telemetry tree and write PDFs plus summaries."""

    from lusee.GainModel import get_models

    config = load_qualification_config(config_path)
    threshold = load_low_analog_threshold()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for target in config.targets:
        decoded = None
        if target.kind == "cdi" and target.telemetry_sidecar is not None:
            decoded = decode_legacy_target(target)
        elif target.kind == "raw":
            decoded = decode_raw_target(target)
        if decoded is None:
            continue
        assessment = assess_tree(
            decoded,
            low_analog_threshold=threshold,
        )
        pdf_path = output_dir / decoded.source_kind / f"{decoded.target_id}.pdf"
        page_count = write_tree_pdf(pdf_path, assessment)
        records.append(assessment_record(
            assessment,
            pdf_path=pdf_path,
            page_count=page_count,
        ))

    records.sort(key=lambda row: str(row["target_id"]))
    summary = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "config_path": str(config_path.resolve()),
        "config_digest": config.config_digest,
        "criteria": {
            "bad_packet_rows": (
                "any invalid engineering field or the private decoder's "
                "established low SPE_1VA8_V diagnostic"
            ),
            "serious_tree_fields": list(GRAHAM_REQUIRED_FIELDS),
            "gain_check": (
                "positive finite L/M/H gain at all 16 native anchors for all "
                "four channels"
            ),
            "gain_models": get_models(),
            "numeric_operating_ranges": None,
            "numeric_operating_ranges_reason": (
                "no reviewed telemetry operating bounds exist in luseepy"
            ),
            "enough_data": f"at least {PLOT_VALUES_MINIMUM} retained packets",
            "ccsds_diagnostics_affect_quality": False,
        },
        "counts": aggregate_records(records),
        "trees": records,
    }
    json_path = output_dir / "telemetry_summary.json"
    csv_path = output_dir / "telemetry_summary.csv"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(csv_path, records)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="private ingest qualification JSON config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/pdf/telemetry-corpus"),
        help="private report directory (default: output/pdf/telemetry-corpus)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_report(args.config, args.output_dir)
    for source_kind in ("legacy_binary_sidecar", "b01_0x314", "all"):
        counts = summary["counts"][source_kind]
        print(
            f"{source_kind}: trees={counts['trees']} good={counts['good_trees']} "
            f"bad={counts['bad_trees']} good_enough={counts['good_enough_trees']} "
            f"good_short={counts['good_short_trees']}"
        )
        example = counts.get("example_good_enough_directory")
        if example is not None:
            print(f"  example good directory: {example}")
    print(f"report: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
