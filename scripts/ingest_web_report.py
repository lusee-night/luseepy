#!/usr/bin/env python3
"""Build static human-QA web reports for layout-v4 ingest HDF5 files.

The report is intentionally static: one ``index.html``, a machine-readable
``summary.json``, and local PNG assets.  It can be opened from disk or copied
unchanged to a plain web server.

Two modes are provided::

    python scripts/ingest_web_report.py hdf5 session.h5
    python scripts/ingest_web_report.py corpus \
        --ccsds-corpus /path/to/ccsds_corpus \
        --cdi-corpus /path/to/cdi_output_corpus \
        --landing-time-file /path/to/landing.json

Corpus mode preserves the operational lineage CCSDS -> extracted CDI -> HDF5.
After raw CCSDS trees are processed, CDI tree identities derived from those
trees are excluded from the standalone CDI pass, so each final HDF5 has one
report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import shutil
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

REPORT_FORMAT_VERSION = 1
DEFAULT_REPORT_SUBDIR = "web_report"
MAX_INVENTORY_SAMPLE_VALUES = 250_000
PRODUCT_NAMES = (
    "Ch0 auto",
    "Ch1 auto",
    "Ch2 auto",
    "Ch3 auto",
    "Ch0 x Ch1 real",
    "Ch0 x Ch1 imag",
    "Ch0 x Ch2 real",
    "Ch0 x Ch2 imag",
    "Ch0 x Ch3 real",
    "Ch0 x Ch3 imag",
    "Ch1 x Ch2 real",
    "Ch1 x Ch2 imag",
    "Ch1 x Ch3 real",
    "Ch1 x Ch3 imag",
    "Ch2 x Ch3 real",
    "Ch2 x Ch3 imag",
)
ZOOM_COMPONENTS = ("AA", "BB", "ABR", "ABI")


@dataclass(frozen=True)
class ReportResult:
    """Paths and compact model produced for one HDF5 report."""

    index_path: Path
    summary_path: Path
    asset_paths: tuple[Path, ...]
    summary: Mapping[str, object]


@dataclass(frozen=True)
class CorpusTarget:
    """One manifest-selected raw CCSDS or CDI corpus tree."""

    kind: Literal["ccsds", "cdi"]
    tree_id: str
    tree_dir: Path
    input_dir: Path
    telemetry_sidecar: Path | None = None


@dataclass(frozen=True)
class CorpusRunResult:
    """Outcome summary for one corpus invocation."""

    reports: tuple[ReportResult, ...]
    derived_cdi_tree_ids: tuple[str, ...]
    skipped_cdi_tree_ids: tuple[str, ...]
    failures: tuple[Mapping[str, str], ...]


def import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required; install luseepy with the ingest extra"
        ) from exc
    return h5py


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for ingest web reports") from exc
    plt.rcParams.update({
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.2,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
    })
    return plt


def json_value(value: object) -> object:
    """Convert NumPy and HDF5 values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def decode_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def format_number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "NaN" if math.isnan(number) else ("+inf" if number > 0 else "-inf")
        magnitude = abs(number)
        if magnitude != 0 and (magnitude >= 1e6 or magnitude < 1e-4):
            return f"{number:.5e}"
        return f"{number:.6g}"
    return decode_text(value)


def format_shape(shape: Sequence[int]) -> str:
    return "(" + ", ".join(f"{int(value):,}" for value in shape) + ")"


def format_array(value: object, *, limit: int = 12) -> str:
    if value is None:
        return "-"
    array = np.asarray(value)
    if array.ndim == 0:
        return format_number(array.item())
    flat = array.reshape(-1)
    shown = ", ".join(format_number(item) for item in flat[:limit])
    suffix = "" if flat.size <= limit else f", ... ({flat.size:,} values)"
    return f"[{shown}{suffix}] shape={format_shape(array.shape)}"


def escaped(value: object) -> str:
    return html.escape(format_number(value), quote=True)


def html_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    css_class: str = "",
    row_attributes: Sequence[Mapping[str, object]] | None = None,
) -> str:
    body = []
    row_attributes = tuple(row_attributes or ())
    for row_index, row in enumerate(rows):
        attrs = ""
        if row_index < len(row_attributes):
            attrs = "".join(
                f' data-{html.escape(str(name), quote=True)}="{html.escape(str(value), quote=True)}"'
                for name, value in row_attributes[row_index].items()
            )
        cells = "".join(f"<td>{escaped(value)}</td>" for value in row)
        body.append(f"<tr{attrs}>{cells}</tr>")
    heading = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    classes = "data-table" + (f" {css_class}" if css_class else "")
    return (
        '<div class="table-wrap">'
        f'<table class="{classes}"><thead><tr>{heading}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_statistics(value: object) -> dict[str, object]:
    """Return exact finite-value statistics for one in-memory numeric array."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        array = np.abs(array)
    try:
        numeric = np.asarray(array, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return {"count": int(array.size), "numeric": False}
    finite = numeric[np.isfinite(numeric)]
    result: dict[str, object] = {
        "count": int(numeric.size),
        "finite_count": int(finite.size),
        "nonfinite_count": int(numeric.size - finite.size),
        "finite_fraction": float(finite.size / numeric.size) if numeric.size else None,
        "numeric": True,
        "minimum": None,
        "q01": None,
        "q05": None,
        "median": None,
        "mean": None,
        "standard_deviation": None,
        "rms": None,
        "q95": None,
        "q99": None,
        "maximum": None,
        "negative_fraction": None,
        "zero_fraction": None,
    }
    if finite.size:
        q01, q05, median, q95, q99 = np.percentile(
            finite, (1.0, 5.0, 50.0, 95.0, 99.0)
        )
        result.update({
            "minimum": float(np.min(finite)),
            "q01": float(q01),
            "q05": float(q05),
            "median": float(median),
            "mean": float(np.mean(finite)),
            "standard_deviation": float(np.std(finite)),
            "rms": float(np.sqrt(np.mean(np.square(finite)))),
            "q95": float(q95),
            "q99": float(q99),
            "maximum": float(np.max(finite)),
            "negative_fraction": float(np.count_nonzero(finite < 0) / finite.size),
            "zero_fraction": float(np.count_nonzero(finite == 0) / finite.size),
        })
    return result


def sampled_dataset_array(dataset) -> tuple[np.ndarray, bool]:
    """Read a deterministic row sample for the inventory statistics."""
    if dataset.size == 0:
        return np.asarray(dataset[...]), False
    if dataset.size <= MAX_INVENTORY_SAMPLE_VALUES or dataset.ndim == 0:
        return np.asarray(dataset[...]), False
    trailing = int(np.prod(dataset.shape[1:], dtype=np.int64)) if dataset.ndim > 1 else 1
    row_limit = max(1, MAX_INVENTORY_SAMPLE_VALUES // max(trailing, 1))
    if dataset.shape[0] <= row_limit:
        return np.asarray(dataset[...]), False
    indices = np.linspace(0, dataset.shape[0] - 1, row_limit, dtype=np.int64)
    indices = np.unique(indices)
    return np.asarray(dataset[indices]), True


def dataset_inventory(h5) -> list[dict[str, object]]:
    """Inventory every dataset, including local attributes and sampled stats."""
    h5py = import_h5py()
    records: list[dict[str, object]] = []

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        array, sampled = sampled_dataset_array(obj)
        record: dict[str, object] = {
            "path": f"/{name}",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "size": int(obj.size),
            "storage_bytes": int(obj.id.get_storage_size()),
            "compression": obj.compression,
            "sampled": sampled,
            "sample_count": int(array.size),
            "attributes": {
                key: json_value(value) for key, value in sorted(obj.attrs.items())
            },
        }
        if obj.dtype.kind in "biufc":
            record["statistics"] = finite_statistics(array)
        elif obj.dtype.kind in "OSU":
            values = [decode_text(item) for item in array.reshape(-1)]
            counts = Counter(values)
            record["statistics"] = {
                "count": len(values),
                "unique_count": len(counts),
                "most_common": counts.most_common(8),
            }
        records.append(record)

    h5.visititems(visit)
    return records


def group_attribute_inventory(h5) -> list[dict[str, object]]:
    h5py = import_h5py()
    records: list[dict[str, object]] = []

    def add(path: str, group) -> None:
        if not isinstance(group, h5py.Group):
            return
        for name, value in sorted(group.attrs.items()):
            records.append({
                "path": path or "/",
                "name": name,
                "value": json_value(value),
                "dtype": str(np.asarray(value).dtype),
                "shape": list(np.asarray(value).shape),
            })

    add("/", h5)
    h5.visititems(lambda name, obj: add(f"/{name}", obj))
    return records


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"report directory already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    (output_dir / "assets").mkdir()
    return output_dir


def save_figure(figure, assets_dir: Path, name: str, assets: list[Path]) -> str:
    path = assets_dir / name
    figure.savefig(path, dpi=135, bbox_inches="tight", facecolor="white")
    import_pyplot().close(figure)
    assets.append(path)
    return f"assets/{path.name}"


def figure_html(path: str, title: str, caption: str) -> str:
    return (
        '<figure class="report-figure">'
        f'<a href="{html.escape(path, quote=True)}">'
        f'<img src="{html.escape(path, quote=True)}" alt="{html.escape(title, quote=True)}" loading="lazy">'
        "</a>"
        f"<figcaption><strong>{html.escape(title)}</strong><br>{html.escape(caption)}</figcaption>"
        "</figure>"
    )


def empty_section(message: str) -> str:
    return f'<div class="empty-state">{html.escape(message)}</div>'


def time_axis(raw_times: object, valid: object | None) -> tuple[np.ndarray, str]:
    raw = np.asarray(raw_times, dtype=np.float64)
    if raw.size == 0:
        return np.arange(0, dtype=np.float64), "row"
    mask = (
        np.isfinite(raw)
        if valid is None
        else np.asarray(valid, dtype=np.bool_) & np.isfinite(raw)
    )
    if np.all(mask):
        return raw - raw[0], "seconds since first row"
    return np.arange(raw.size, dtype=np.float64), "row index (time unavailable)"


def row_caption(
    label: str,
    row: int,
    unique_ids: object | None,
    raw_times: object | None,
    raw_valid: object | None,
) -> str:
    parts = [label, f"row {row}"]
    if unique_ids is not None:
        parts.append(f"UID {int(np.asarray(unique_ids)[row])}")
    if raw_times is not None:
        valid = True if raw_valid is None else bool(np.asarray(raw_valid)[row])
        if valid and np.isfinite(np.asarray(raw_times, dtype=np.float64)[row]):
            parts.append(f"raw time {float(np.asarray(raw_times)[row]):.6f} s")
        else:
            parts.append("raw time unavailable")
    return "; ".join(parts)


PACKET_STATISTIC_NAMES = (
    "minimum",
    "q01",
    "q05",
    "median",
    "mean",
    "standard_deviation",
    "rms",
    "q95",
    "q99",
    "maximum",
    "finite_fraction",
    "negative_fraction",
    "zero_fraction",
)


def product_packet_statistics(
    row_count: int,
    product_count: int,
    values_for: Callable[[int, int], np.ndarray],
) -> dict[str, np.ndarray]:
    """Reduce every row/product over its native sample axes."""
    result = {
        name: np.full((row_count, product_count), np.nan, dtype=np.float64)
        for name in PACKET_STATISTIC_NAMES
    }
    for row in range(row_count):
        for product in range(product_count):
            stats = finite_statistics(values_for(row, product))
            if not stats.get("numeric"):
                continue
            for name in PACKET_STATISTIC_NAMES:
                value = stats.get(name)
                if value is not None:
                    result[name][row, product] = float(value)
    return result


def product_statistics_model(
    stats: Mapping[str, np.ndarray],
    unique_ids: object | None,
    labels: Sequence[str] = PRODUCT_NAMES,
) -> dict[str, object]:
    medians = np.asarray(stats["median"])
    row_count, product_count = medians.shape
    records = []
    for product in range(product_count):
        present = np.isfinite(medians[:, product])
        packet_medians = medians[present, product]
        minimum = np.asarray(stats["minimum"])[present, product]
        maximum = np.asarray(stats["maximum"])[present, product]
        records.append({
            "product": product,
            "label": labels[product] if product < len(labels) else f"Product {product}",
            "packets_present": int(np.count_nonzero(present)),
            "packets_missing": int(row_count - np.count_nonzero(present)),
            "minimum": float(np.min(minimum)) if minimum.size else None,
            "median_of_packet_medians": (
                float(np.median(packet_medians)) if packet_medians.size else None
            ),
            "maximum": float(np.max(maximum)) if maximum.size else None,
            "median_finite_fraction": (
                float(np.nanmedian(np.asarray(stats["finite_fraction"])[:, product]))
                if np.any(np.isfinite(np.asarray(stats["finite_fraction"])[:, product]))
                else None
            ),
            "median_negative_fraction": (
                float(np.nanmedian(np.asarray(stats["negative_fraction"])[:, product]))
                if np.any(np.isfinite(np.asarray(stats["negative_fraction"])[:, product]))
                else None
            ),
            "median_zero_fraction": (
                float(np.nanmedian(np.asarray(stats["zero_fraction"])[:, product]))
                if np.any(np.isfinite(np.asarray(stats["zero_fraction"])[:, product]))
                else None
            ),
        })
    return {
        "row_count": row_count,
        "product_count": product_count,
        "unique_ids": (
            json_value(np.asarray(unique_ids)) if unique_ids is not None else None
        ),
        "per_product": records,
        "packet_median": json_value(stats["median"]),
        "packet_q01": json_value(stats["q01"]),
        "packet_q99": json_value(stats["q99"]),
        "packet_rms": json_value(stats["rms"]),
        "packet_finite_fraction": json_value(stats["finite_fraction"]),
    }


def product_statistics_table(model: Mapping[str, object]) -> str:
    rows = []
    for record in model["per_product"]:
        rows.append((
            f"{record['product']:02d}",
            record["label"],
            record["packets_present"],
            record["packets_missing"],
            record["minimum"],
            record["median_of_packet_medians"],
            record["maximum"],
            record["median_finite_fraction"],
            record["median_negative_fraction"],
            record["median_zero_fraction"],
        ))
    return html_table(
        (
            "#",
            "product",
            "packets present",
            "missing",
            "global min",
            "median packet median",
            "global max",
            "median finite frac.",
            "median negative frac.",
            "median zero frac.",
        ),
        rows,
    )


def use_log_axis(axis, values: np.ndarray, product: int) -> None:
    finite = np.asarray(values)[np.isfinite(values)]
    if product < 4 and finite.size and np.all(finite > 0):
        axis.set_yscale("log")
    elif finite.size and np.nanmax(np.abs(finite)) > 0:
        threshold = max(float(np.nanpercentile(np.abs(finite), 5)), 1e-12)
        axis.set_yscale("symlog", linthresh=threshold)


def plot_spectrum_endpoint(data: np.ndarray, frequency_count: int, title: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    x = np.arange(frequency_count)
    for product, axis in enumerate(axes.flat):
        values = np.asarray(data[product, :frequency_count], dtype=np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            axis.plot(x[finite], values[finite], linewidth=0.7, color="#1565c0")
            use_log_axis(axis, values, product)
            axis.text(
                0.02,
                0.03,
                f"finite {np.count_nonzero(finite):,}/{frequency_count:,}",
                transform=axis.transAxes,
                fontsize=6.5,
                color="#455a64",
            )
        else:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", color="#b71c1c")
        axis.set_title(f"{product:02d} {PRODUCT_NAMES[product]}")
        if product >= 12:
            axis.set_xlabel("output bin index")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def plot_packet_product_summary(
    stats: Mapping[str, np.ndarray],
    title: str,
    x: np.ndarray,
    x_label: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    median = np.asarray(stats["median"])
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    finite_fraction = np.asarray(stats["finite_fraction"])
    for product, axis in enumerate(axes.flat):
        valid = np.isfinite(median[:, product])
        if np.any(valid):
            axis.fill_between(
                x[valid],
                low[valid, product],
                high[valid, product],
                color="#90caf9",
                alpha=0.45,
                linewidth=0,
                label="q01-q99",
            )
            axis.plot(
                x[valid],
                median[valid, product],
                color="#0d47a1",
                linewidth=0.75,
                marker=".",
                markersize=2,
                label="median",
            )
            use_log_axis(axis, median[:, product], product)
        incomplete = np.isfinite(finite_fraction[:, product]) & (
            finite_fraction[:, product] < 1.0
        )
        if np.any(incomplete):
            ymin, _ = axis.get_ylim()
            axis.scatter(
                x[incomplete],
                np.full(np.count_nonzero(incomplete), ymin),
                marker="x",
                s=10,
                color="#d32f2f",
                label="incomplete",
            )
        missing = int(np.count_nonzero(~valid))
        axis.set_title(f"{product:02d} {PRODUCT_NAMES[product]} | missing {missing}")
        if product >= 12:
            axis.set_xlabel(x_label)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def robust_image_limits(values: np.ndarray, diverging: bool) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, (1.0, 99.0))
    if low == high:
        scale = abs(float(low)) or 1.0
        low, high = float(low) - scale * 0.05, float(high) + scale * 0.05
    if diverging:
        bound = max(abs(float(low)), abs(float(high)))
        return -bound, bound
    return float(low), float(high)


def downsample_rows(values: np.ndarray, limit: int = 512) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    indices = np.linspace(0, values.shape[0] - 1, limit, dtype=np.int64)
    return values[np.unique(indices)]


def plot_spectra_waterfalls(
    data: np.ndarray,
    frequency_count: int,
    title: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True, sharey=True)
    for product, axis in enumerate(axes.flat):
        values = downsample_rows(
            np.asarray(data[:, product, :frequency_count], dtype=np.float64)
        )
        vmin, vmax = robust_image_limits(values, product >= 4)
        axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r" if product >= 4 else "viridis",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(f"{product:02d} {PRODUCT_NAMES[product]}")
        if product >= 12:
            axis.set_xlabel("output bin index")
        if product % 4 == 0:
            axis.set_ylabel("sampled packet row")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def normal_spectra_section(bundle, assets_dir: Path, assets: list[Path]):
    data_value = bundle.spectra
    if data_value is None:
        return empty_section("Normal spectra are absent."), {"present": False}, {}
    data = np.asarray(data_value, dtype=np.float64)
    row_count = int(data.shape[0])
    if row_count == 0:
        return empty_section("The normal-spectra table has zero rows."), {
            "present": True,
            "row_count": 0,
        }, {}
    counts = (
        np.asarray(bundle.spectra_frequency_counts, dtype=np.int64)
        if bundle.spectra_frequency_counts is not None
        else np.full(row_count, data.shape[2], dtype=np.int64)
    )
    counts = np.clip(counts, 0, data.shape[2])
    provider = lambda row, product: data[row, product, : counts[row]]
    stats = product_packet_statistics(row_count, 16, provider)
    stats_model = product_statistics_model(stats, bundle.spectra_unique_ids)
    x, x_label = time_axis(bundle.spectra_raw_times, bundle.spectra_raw_time_valid)
    html_parts = []
    selections: dict[str, object] = {}
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(
            label,
            row,
            bundle.spectra_unique_ids,
            bundle.spectra_raw_times,
            bundle.spectra_raw_time_valid,
        )
        figure = plot_spectrum_endpoint(
            data[row], int(counts[row]), f"Normal spectra: {caption}"
        )
        path = save_figure(figure, assets_dir, f"spectra_{label}.png", assets)
        html_parts.append(figure_html(path, f"Normal spectra - {label} packet", caption))
        selections[label] = {
            "row": row,
            "unique_id": (
                int(np.asarray(bundle.spectra_unique_ids)[row])
                if bundle.spectra_unique_ids is not None
                else None
            ),
            "frequency_count": int(counts[row]),
        }
    summary_path = save_figure(
        plot_packet_product_summary(
            stats,
            "Normal spectra: robust statistics over every packet",
            x,
            x_label,
        ),
        assets_dir,
        "spectra_all_packets.png",
        assets,
    )
    html_parts.append(figure_html(
        summary_path,
        "Normal spectra - all-packet statistics",
        "Each panel shows the packet median and q01-q99 range over meaningful output bins. Red marks identify partially finite packets.",
    ))
    grid_rows = (
        np.asarray(bundle.spectra_frequency_window_index, dtype=np.int64)
        if bundle.spectra_frequency_window_index is not None
        else (
            np.asarray(bundle.spectra_navgf, dtype=np.int64)
            if bundle.spectra_navgf is not None
            else np.zeros(row_count, dtype=np.int64)
        )
    )
    waterfall_groups = []
    for group_number, (grid_value, frequency_count) in enumerate(sorted({
        (int(grid_rows[row]), int(counts[row])) for row in range(row_count)
    })):
        selected = np.flatnonzero(
            (grid_rows == grid_value) & (counts == frequency_count)
        )
        waterfall_path = save_figure(
            plot_spectra_waterfalls(
                data[selected],
                frequency_count,
                "Normal spectra waterfall: "
                f"grid row {grid_value}, {frequency_count} output bins",
            ),
            assets_dir,
            f"spectra_waterfalls_{group_number:02d}.png",
            assets,
        )
        html_parts.append(figure_html(
            waterfall_path,
            f"Normal spectra - grid {grid_value} waterfall",
            f"{selected.size} packet row(s), {frequency_count} meaningful output bins. Different firmware frequency windows are never stacked together; x is output-bin index, not MHz.",
        ))
        waterfall_groups.append({
            "grid_row": grid_value,
            "frequency_count": frequency_count,
            "row_count": int(selected.size),
        })
    navgf_counts = (
        Counter(
            int(value)
            for value in np.asarray(bundle.spectra_navgf, dtype=np.int64)
        )
        if bundle.spectra_navgf is not None
        else Counter()
    )
    window_indices = (
        np.asarray(bundle.spectra_frequency_window_index, dtype=np.int64)
        if bundle.spectra_frequency_window_index is not None
        else None
    )
    window_histogram = (
        dict(Counter(int(value) for value in window_indices))
        if window_indices is not None
        else {}
    )
    window_records = [
        contract.as_record() for contract in bundle.spectra_frequency_windows
    ]
    html_parts.append('<h3>All-packet product statistics</h3>')
    html_parts.append(product_statistics_table(stats_model))
    html_parts.append('<h3>Frequency-window use</h3>')
    html_parts.append(html_table(
        ("Navgf", "rows"),
        sorted(navgf_counts.items()),
    ) if navgf_counts else empty_section("Navgf metadata is absent."))
    if window_histogram:
        html_parts.append(html_table(
            ("frequency-window row", "spectra rows"),
            sorted(window_histogram.items()),
        ))
    if window_records:
        html_parts.append(html_table(
            (
                "window row",
                "Navgf",
                "native bins",
                "output bins",
                "stride",
                "included offsets",
                "firmware divisor",
                "frequency coordinate",
                "source commit",
            ),
            (
                (
                    index,
                    record["navgf"],
                    record["native_count"],
                    record["output_count"],
                    record["stride"],
                    format_array(record["included_offsets"]),
                    record["firmware_divisor"],
                    record["frequency_coordinate_status"],
                    record["source_commit"],
                )
                for index, record in enumerate(window_records)
            ),
        ))
    return "".join(html_parts), {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "frequency_count_histogram": dict(Counter(int(value) for value in counts)),
        "navgf_histogram": dict(navgf_counts),
        "frequency_window_index_histogram": window_histogram,
        "frequency_windows": window_records,
        "waterfall_groups": waterfall_groups,
        "product_statistics": stats_model,
    }, selections


def plot_heatmap_endpoint(data: np.ndarray, title: str, x_label: str, y_label: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(15, 11))
    for product, axis in enumerate(axes.flat):
        values = np.asarray(data[product], dtype=np.float64)
        finite = np.isfinite(values)
        if np.any(finite):
            vmin, vmax = robust_image_limits(values, product >= 4)
            axis.imshow(
                values,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r" if product >= 4 else "viridis",
                vmin=vmin,
                vmax=vmax,
            )
            axis.text(
                0.02,
                0.03,
                f"range {format_number(np.nanmin(values))} .. {format_number(np.nanmax(values))}",
                transform=axis.transAxes,
                fontsize=6,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"},
            )
        else:
            axis.text(0.5, 0.5, "missing", ha="center", va="center", color="#b71c1c")
        axis.set_title(f"{product:02d} {PRODUCT_NAMES[product]}")
        if product >= 12:
            axis.set_xlabel(x_label)
        if product % 4 == 0:
            axis.set_ylabel(y_label)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def tr_spectra_section(bundle, assets_dir: Path, assets: list[Path]):
    data_value = bundle.tr_spectra
    if data_value is None:
        return empty_section("Time-resolved spectra are absent."), {"present": False}, {}
    data = np.asarray(data_value, dtype=np.float64)
    row_count = int(data.shape[0])
    if row_count == 0:
        return empty_section("The time-resolved table has zero rows."), {
            "present": True,
            "row_count": 0,
        }, {}
    provider = lambda row, product: data[row, product]
    stats = product_packet_statistics(row_count, 16, provider)
    stats_model = product_statistics_model(stats, bundle.tr_unique_ids)
    x, x_label = time_axis(bundle.tr_raw_times, bundle.tr_raw_time_valid)
    html_parts = []
    selections = {}
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(
            label,
            row,
            bundle.tr_unique_ids,
            bundle.tr_raw_times,
            bundle.tr_raw_time_valid,
        )
        if bundle.tr_navg2_per_sample is not None:
            caption += f"; Navg2 {int(np.asarray(bundle.tr_navg2_per_sample)[row])}"
        if bundle.tr_length_per_sample is not None:
            caption += f"; Ntr {int(np.asarray(bundle.tr_length_per_sample)[row])}"
        path = save_figure(
            plot_heatmap_endpoint(
                data[row],
                f"Time-resolved spectra: {caption}",
                "time-resolved sample index",
                "average index",
            ),
            assets_dir,
            f"tr_spectra_{label}.png",
            assets,
        )
        html_parts.append(figure_html(path, f"Time-resolved - {label} packet", caption))
        selections[label] = {
            "row": row,
            "unique_id": (
                int(np.asarray(bundle.tr_unique_ids)[row])
                if bundle.tr_unique_ids is not None
                else None
            ),
            "navg2": (
                int(np.asarray(bundle.tr_navg2_per_sample)[row])
                if bundle.tr_navg2_per_sample is not None
                else None
            ),
            "tr_length": (
                int(np.asarray(bundle.tr_length_per_sample)[row])
                if bundle.tr_length_per_sample is not None
                else None
            ),
        }
    path = save_figure(
        plot_packet_product_summary(
            stats,
            "Time-resolved spectra: robust statistics over every packet",
            x,
            x_label,
        ),
        assets_dir,
        "tr_spectra_all_packets.png",
        assets,
    )
    html_parts.append(figure_html(
        path,
        "Time-resolved - all-packet statistics",
        "Per packet and product, the median and q01-q99 envelope reduce the native Navg2 x Ntr plane.",
    ))
    html_parts.append(product_statistics_table(stats_model))
    navg2_histogram = (
        dict(Counter(
            int(value) for value in np.asarray(bundle.tr_navg2_per_sample)
        ))
        if bundle.tr_navg2_per_sample is not None
        else {}
    )
    tr_length_histogram = (
        dict(Counter(
            int(value) for value in np.asarray(bundle.tr_length_per_sample)
        ))
        if bundle.tr_length_per_sample is not None
        else {}
    )
    if navg2_histogram or tr_length_histogram:
        html_parts.extend((
            "<h3>Time-resolved settings</h3>",
            html_table(
                ("setting", "value", "rows"),
                (
                    [("Navg2", value, count) for value, count in sorted(navg2_histogram.items())]
                    + [("Ntr", value, count) for value, count in sorted(tr_length_histogram.items())]
                ),
            ),
        ))
    return "".join(html_parts), {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "navg2": int(data.shape[2]),
        "tr_length": int(data.shape[3]),
        "navg2_histogram": navg2_histogram,
        "tr_length_histogram": tr_length_histogram,
        "product_statistics": stats_model,
    }, selections


def plot_grimm_endpoint(
    data: np.ndarray,
    average_valid: np.ndarray,
    title: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    averages = np.flatnonzero(average_valid)
    labels = ("value 0", "value 1", "value 2", "value 3")
    colors = ("#0d47a1", "#2e7d32", "#ef6c00", "#6a1b9a")
    for product, axis in enumerate(axes.flat):
        if averages.size:
            for value_index, (label, color) in enumerate(zip(labels, colors)):
                axis.plot(
                    averages,
                    data[averages, product, value_index],
                    linewidth=0.75,
                    color=color,
                    label=label,
                )
        else:
            axis.text(0.5, 0.5, "no valid averages", ha="center", va="center")
        axis.set_title(f"{product:02d} {PRODUCT_NAMES[product]}")
        if product >= 12:
            axis.set_xlabel("average index")
        if product == 0:
            axis.legend(ncol=2)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def grimm_section(bundle, assets_dir: Path, assets: list[Path]):
    data_value = bundle.grimm_spectra
    if data_value is None:
        return empty_section("Grimm spectra are absent."), {"present": False}, {}
    data = np.asarray(data_value)
    row_count = int(data.shape[0])
    if row_count == 0:
        return empty_section("The Grimm table has zero rows."), {
            "present": True,
            "row_count": 0,
        }, {}
    average_valid = (
        np.asarray(bundle.grimm_average_valid, dtype=np.bool_)
        if bundle.grimm_average_valid is not None
        else np.ones(data.shape[:2], dtype=np.bool_)
    )
    x, x_label = time_axis(bundle.grimm_raw_times, bundle.grimm_raw_time_valid)
    html_parts = []
    selections = {}
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(
            label,
            row,
            bundle.grimm_unique_ids,
            bundle.grimm_raw_times,
            bundle.grimm_raw_time_valid,
        )
        path = save_figure(
            plot_grimm_endpoint(
                data[row], average_valid[row], f"Grimm spectra: {caption}"
            ),
            assets_dir,
            f"grimm_{label}.png",
            assets,
        )
        html_parts.append(figure_html(path, f"Grimm - {label} packet", caption))
        selections[label] = {
            "row": row,
            "unique_id": (
                int(np.asarray(bundle.grimm_unique_ids)[row])
                if bundle.grimm_unique_ids is not None
                else None
            ),
            "valid_averages": int(np.count_nonzero(average_valid[row])),
        }
    component_models = []
    for component in range(data.shape[3]):
        stats = product_packet_statistics(
            row_count,
            16,
            lambda row, product, component=component: data[
                row, average_valid[row], product, component
            ],
        )
        stats_model = product_statistics_model(stats, bundle.grimm_unique_ids)
        path = save_figure(
            plot_packet_product_summary(
                stats,
                f"Grimm value {component}: robust statistics over every packet",
                x,
                x_label,
            ),
            assets_dir,
            f"grimm_all_packets_value_{component}.png",
            assets,
        )
        html_parts.append(figure_html(
            path,
            f"Grimm value {component} - all-packet statistics",
            "Statistics use only averages selected by average_valid. Each of the four stored value components is kept separate; none is treated as frequency.",
        ))
        html_parts.append(product_statistics_table(stats_model))
        component_models.append({
            "value_index": component,
            "product_statistics": stats_model,
        })
    return "".join(html_parts), {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "navg2_histogram": dict(Counter(
            int(value)
            for value in np.asarray(bundle.grimm_navg2_per_sample, dtype=np.int64)
        )) if bundle.grimm_navg2_per_sample is not None else {},
        "value_component_statistics": component_models,
    }, selections


def field_columns_model(
    columns: Mapping[str, object],
    presence: Mapping[str, object] | None,
    row_count: int,
) -> list[dict[str, object]]:
    """Summarize every row-aligned metadata column without flattening fields."""
    records = []
    presence = presence or {}
    for name in sorted(columns):
        array = np.asarray(columns[name])
        if array.ndim == 0 or array.shape[0] != row_count:
            records.append({
                "name": name,
                "status": "not_row_aligned",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            })
            continue
        mask_value = presence.get(name)
        if mask_value is not None:
            mask = np.asarray(mask_value, dtype=np.bool_)
        elif array.dtype.kind == "O" and array.ndim == 1:
            mask = np.asarray(
                [array[row] is not None for row in range(row_count)],
                dtype=np.bool_,
            )
        else:
            mask = np.ones(row_count, dtype=np.bool_)
        if mask.shape != (row_count,):
            mask = np.all(mask.reshape(row_count, -1), axis=1)
        selected = array[mask]
        record: dict[str, object] = {
            "name": name,
            "status": "present",
            "shape": list(array.shape),
            "value_shape": list(array.shape[1:]),
            "dtype": str(array.dtype),
            "rows_present": int(np.count_nonzero(mask)),
            "rows_missing": int(row_count - np.count_nonzero(mask)),
            "first_row": (
                format_array(array[0]) if row_count and mask[0] else "-"
            ),
            "last_row": (
                format_array(array[-1]) if row_count and mask[-1] else "-"
            ),
        }
        if array.dtype.kind == "O":
            native_values = [array[row] for row in np.flatnonzero(mask)]
            native_arrays = [np.asarray(value) for value in native_values]
            record["dtype"] = ", ".join(sorted({str(value.dtype) for value in native_arrays}))
            record["value_shapes"] = sorted({
                format_shape(value.shape) for value in native_arrays
            })
            if native_arrays and all(value.dtype.kind in "biufc" for value in native_arrays):
                record["statistics"] = finite_statistics(
                    np.concatenate([value.reshape(-1) for value in native_arrays])
                )
            else:
                values = [format_array(value) for value in native_values]
                record["statistics"] = {
                    "count": len(values),
                    "unique_count": len(set(values)),
                    "most_common": Counter(values).most_common(6),
                }
        elif array.dtype.kind in "biufc":
            record["statistics"] = finite_statistics(selected)
        else:
            values = [decode_text(item) for item in selected.reshape(-1)]
            record["statistics"] = {
                "count": len(values),
                "unique_count": len(set(values)),
                "most_common": Counter(values).most_common(6),
            }
        records.append(record)
    return records


def field_columns_table(records: Sequence[Mapping[str, object]]) -> str:
    rows = []
    for record in records:
        statistics = record.get("statistics", {})
        if statistics.get("numeric"):
            summary = (
                f"min {format_number(statistics.get('minimum'))}; "
                f"median {format_number(statistics.get('median'))}; "
                f"max {format_number(statistics.get('maximum'))}; "
                f"finite {format_number(statistics.get('finite_fraction'))}"
            )
        elif statistics:
            summary = (
                f"{statistics.get('unique_count', 0)} unique; "
                + ", ".join(
                    f"{value} x{count}"
                    for value, count in statistics.get("most_common", ())
                )
            )
        else:
            summary = record.get("status", "-")
        rows.append((
            record["name"],
            (
                ", ".join(record["value_shapes"])
                if record.get("value_shapes")
                else format_shape(record.get("value_shape", record.get("shape", ())))
            ),
            record.get("dtype"),
            record.get("rows_present"),
            record.get("rows_missing"),
            record.get("first_row"),
            record.get("last_row"),
            summary,
        ))
    return html_table(
        (
            "field",
            "value shape",
            "dtype",
            "rows present",
            "missing",
            "first row",
            "last row",
            "all-row summary",
        ),
        rows,
    )


def plot_metadata_matrix_fields(
    columns: Mapping[str, object],
    names: Sequence[str],
    title: str,
):
    present = [name for name in names if name in columns]
    if not present:
        return None
    plt = import_pyplot()
    figure, axes = plt.subplots(len(present), 1, figsize=(13, 2.4 * len(present)))
    if len(present) == 1:
        axes = [axes]
    for axis, name in zip(axes, present):
        values = np.asarray(columns[name])
        if values.ndim == 1:
            axis.plot(np.arange(values.shape[0]), values, marker=".", linewidth=0.7)
            axis.set_ylabel(name)
        else:
            matrix = values.reshape(values.shape[0], -1).T
            axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
            )
            axis.set_ylabel(f"{name} component")
        axis.set_xlabel("spectrum row")
        axis.set_title(name)
    figure.suptitle(title, fontsize=13, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def spectrum_metadata_html(bundle, assets_dir: Path, assets: list[Path]) -> tuple[str, object]:
    row_count = 0 if bundle.spectra is None else int(np.asarray(bundle.spectra).shape[0])
    columns = dict(bundle.spectra_metadata)
    adc_valid = columns.get("adc_statistics_valid")
    if adc_valid is not None:
        adc_valid = np.asarray(adc_valid, dtype=np.bool_)
        for name in ("adc_min", "adc_max", "adc_mean", "adc_rms"):
            if name not in columns:
                continue
            values = np.asarray(columns[name], dtype=np.float64).copy()
            field_present = bundle.spectra_metadata_present.get(name)
            if field_present is not None:
                values[~np.asarray(field_present, dtype=np.bool_)] = np.nan
            values[~adc_valid] = np.nan
            columns[name] = values
    records = field_columns_model(
        columns,
        bundle.spectra_metadata_present,
        row_count,
    )
    if not records:
        return empty_section("Spectrum metadata are absent."), []
    parts = ["<h3>Complete spectrum-metadata field table</h3>", field_columns_table(records)]
    figures = (
        (
            "spectra_metadata_adc.png",
            "ADC statistics and validity",
            ("adc_min", "adc_max", "adc_mean", "adc_rms", "adc_statistics_valid"),
        ),
        (
            "spectra_metadata_gain_bitslice.png",
            "Requested and actual gain / bitslice",
            ("requested_gain", "actual_gain", "requested_bitslice", "actual_bitslice"),
        ),
        (
            "spectra_metadata_state.png",
            "Spectrometer state, errors, and overflows",
            (
                "navgf",
                "correlation_products_mask",
                "errors",
                "spectrum_overflow",
                "notch_overflow",
                "spectrometer_enable",
                "calibrator_enable",
            ),
        ),
        (
            "spectra_metadata_rails.png",
            "Metadata telemetry rails and FPGA temperature",
            ("telemetry_v1_0", "telemetry_v1_8", "telemetry_v2_5", "telemetry_t_fpga"),
        ),
    )
    for filename, title, names in figures:
        figure = plot_metadata_matrix_fields(columns, names, title)
        if figure is not None:
            path = save_figure(figure, assets_dir, filename, assets)
            parts.append(figure_html(path, title, "All rows; component axes retain their native stored ordering."))
    return "".join(parts), records


def longest_false_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=np.bool_):
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def telemetry_field_model(telemetry) -> list[dict[str, object]]:
    values = np.asarray(telemetry.values, dtype=np.float64)
    raw_counts = np.asarray(telemetry.raw_counts)
    valid = np.asarray(telemetry.valid, dtype=np.bool_)
    records = []
    for index, (name, unit) in enumerate(zip(telemetry.field_names, telemetry.units)):
        mask = valid[:, index] & np.isfinite(values[:, index])
        engineering = values[mask, index]
        counts = raw_counts[:, index]
        invalid_counts = raw_counts[~mask, index]
        stats = finite_statistics(engineering)
        records.append({
            "index": index,
            "name": name,
            "unit": unit,
            "rows": int(values.shape[0]),
            "valid_count": int(np.count_nonzero(mask)),
            "invalid_count": int(values.shape[0] - np.count_nonzero(mask)),
            "valid_fraction": float(np.mean(mask)) if mask.size else None,
            "longest_invalid_run": longest_false_run(mask),
            "first_value": float(values[0, index]) if values.shape[0] and mask[0] else None,
            "last_value": float(values[-1, index]) if values.shape[0] and mask[-1] else None,
            "first_raw_count": int(raw_counts[0, index]) if values.shape[0] else None,
            "last_raw_count": int(raw_counts[-1, index]) if values.shape[0] else None,
            "raw_count_minimum": int(np.min(counts)) if counts.size else None,
            "raw_count_maximum": int(np.max(counts)) if counts.size else None,
            "invalid_raw_count_minimum": (
                int(np.min(invalid_counts)) if invalid_counts.size else None
            ),
            "invalid_raw_count_maximum": (
                int(np.max(invalid_counts)) if invalid_counts.size else None
            ),
            "statistics": stats,
        })
    return records


def telemetry_statistics_table(records: Sequence[Mapping[str, object]]) -> str:
    return html_table(
        (
            "#",
            "field",
            "unit",
            "valid",
            "invalid",
            "longest invalid run",
            "min",
            "q01",
            "median",
            "mean",
            "q99",
            "max",
            "raw count range",
            "invalid-row raw range",
        ),
        (
            (
                f"{record['index']:02d}",
                record["name"],
                record["unit"] or "unitless",
                f"{record['valid_count']}/{record['rows']} ({format_number(record['valid_fraction'])})",
                record["invalid_count"],
                record["longest_invalid_run"],
                record["statistics"].get("minimum"),
                record["statistics"].get("q01"),
                record["statistics"].get("median"),
                record["statistics"].get("mean"),
                record["statistics"].get("q99"),
                record["statistics"].get("maximum"),
                f"{format_number(record['raw_count_minimum'])} .. {format_number(record['raw_count_maximum'])}",
                f"{format_number(record['invalid_raw_count_minimum'])} .. {format_number(record['invalid_raw_count_maximum'])}",
            )
            for record in records
        ),
    )


def plot_telemetry_fields(telemetry, indices: Sequence[int], title: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(13, 9))
    values = np.asarray(telemetry.values, dtype=np.float64)
    valid = np.asarray(telemetry.valid, dtype=np.bool_)
    x = np.asarray(telemetry.raw_seconds, dtype=np.float64)
    x = x - x[0] if x.size else x
    for axis, field_index in zip(axes.flat, indices):
        mask = valid[:, field_index] & np.isfinite(values[:, field_index])
        if np.count_nonzero(mask) > 3:
            axis.plot(
                x[mask],
                values[mask, field_index],
                marker=".",
                markersize=2.5,
                linewidth=0.7,
                color="#0d47a1",
            )
        else:
            shown = ", ".join(format_number(item) for item in values[mask, field_index])
            axis.text(
                0.5,
                0.5,
                shown or "no valid values",
                ha="center",
                va="center",
                wrap=True,
            )
        axis.set_title(
            f"{field_index:02d} {telemetry.field_names[field_index]} "
            f"[{telemetry.units[field_index] or 'unitless'}] | "
            f"valid {np.count_nonzero(mask)}/{mask.size}"
        )
        axis.set_xlabel("seconds since first telemetry packet")
    for axis in axes.flat[len(indices):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=13, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def plot_telemetry_heatmaps(telemetry):
    plt = import_pyplot()
    values = np.asarray(telemetry.values, dtype=np.float64)
    valid = np.asarray(telemetry.valid, dtype=np.bool_)
    standardized = np.full(values.shape, np.nan, dtype=np.float64)
    for field in range(values.shape[1]):
        mask = valid[:, field] & np.isfinite(values[:, field])
        if not np.any(mask):
            continue
        center = float(np.median(values[mask, field]))
        q25, q75 = np.percentile(values[mask, field], (25.0, 75.0))
        scale = float(q75 - q25)
        if not scale:
            scale = float(np.std(values[mask, field])) or 1.0
        standardized[mask, field] = (values[mask, field] - center) / scale
    standardized = np.clip(standardized, -6.0, 6.0)
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    axes[0].imshow(
        downsample_rows(valid).T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
    )
    axes[0].set_ylabel("field index")
    axes[0].set_title("Validity (red invalid, green valid)")
    axes[1].imshow(
        downsample_rows(standardized).T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-6,
        vmax=6,
    )
    axes[1].set_ylabel("field index")
    axes[1].set_xlabel("sampled telemetry packet row")
    axes[1].set_title("Engineering values standardized per field (median / IQR)")
    figure.suptitle("Telemetry validity and across-packet behavior", fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def telemetry_section(bundle, assets_dir: Path, assets: list[Path]):
    telemetry = bundle.telemetry
    if telemetry is None:
        return empty_section("Telemetry is absent or was skipped."), {"present": False}, {}
    row_count = int(telemetry.row_count)
    records = telemetry_field_model(telemetry)
    if row_count == 0:
        return (
            empty_section("The fixed 57-field telemetry table has zero packets.")
            + telemetry_statistics_table(records),
            {
                "present": True,
                "row_count": 0,
                "source_kind": telemetry.source_kind,
                "field_statistics": records,
            },
            {},
        )
    first, last = 0, row_count - 1
    endpoint_rows = []
    for record in records:
        endpoint_rows.append((
            f"{record['index']:02d}",
            record["name"],
            record["unit"] or "unitless",
            record["first_value"],
            record["first_raw_count"],
            "valid" if record["first_value"] is not None else "invalid",
            record["last_value"],
            record["last_raw_count"],
            "valid" if record["last_value"] is not None else "invalid",
        ))
    parts = [
        "<h3>First and last telemetry packets</h3>",
        html_table(
            (
                "#",
                "field",
                "unit",
                "first value",
                "first raw",
                "first status",
                "last value",
                "last raw",
                "last status",
            ),
            endpoint_rows,
        ),
    ]
    heatmap_path = save_figure(
        plot_telemetry_heatmaps(telemetry),
        assets_dir,
        "telemetry_overview.png",
        assets,
    )
    parts.append(figure_html(
        heatmap_path,
        "Telemetry validity and standardized-value overview",
        "Every stored field is included. Display rows are deterministically sampled only when necessary.",
    ))
    for page, start in enumerate(range(0, len(records), 6), start=1):
        indices = tuple(range(start, min(start + 6, len(records))))
        path = save_figure(
            plot_telemetry_fields(
                telemetry,
                indices,
                f"Telemetry fields {indices[0]:02d}-{indices[-1]:02d}",
            ),
            assets_dir,
            f"telemetry_fields_{page:02d}.png",
            assets,
        )
        parts.append(figure_html(
            path,
            f"Telemetry fields {indices[0]:02d}-{indices[-1]:02d}",
            "Fields with more than three valid packets are plotted over elapsed DCB time; shorter fields show their values directly.",
        ))
    parts.extend(("<h3>All-field statistics</h3>", telemetry_statistics_table(records)))
    raw_seconds = np.asarray(telemetry.raw_seconds, dtype=np.float64)
    cadence = np.diff(raw_seconds)
    source_indices = np.asarray(telemetry.source_indices, dtype=np.int64)
    source_gap_count = int(np.count_nonzero(np.diff(source_indices) != 1))
    mjd_valid_count = int(np.count_nonzero(np.isfinite(telemetry.mjd_times)))
    parts.extend((
        "<h3>Telemetry packet timing and source coverage</h3>",
        html_table(
            ("check", "value"),
            (
                ("source kind", telemetry.source_kind),
                ("packet rows", row_count),
                ("mission seconds first", int(telemetry.mission_seconds[first])),
                ("mission seconds last", int(telemetry.mission_seconds[last])),
                ("source-index non-unit steps", source_gap_count),
                ("MJD-valid rows", f"{mjd_valid_count}/{row_count}"),
                ("median cadence (s)", finite_statistics(cadence).get("median")),
            ),
        ),
    ))
    selections = {
        "first": {"row": first, "source_index": int(source_indices[first])},
        "last": {"row": last, "source_index": int(source_indices[last])},
    }
    return "".join(parts), {
        "present": True,
        "row_count": row_count,
        "source_kind": telemetry.source_kind,
        "mission_seconds_first": int(telemetry.mission_seconds[first]),
        "mission_seconds_last": int(telemetry.mission_seconds[last]),
        "cadence_seconds": finite_statistics(cadence),
        "source_index_gap_count": source_gap_count,
        "mjd_valid_count": mjd_valid_count,
        "field_statistics": records,
    }, selections


def plot_zoom_endpoint(data: np.ndarray, title: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    x = np.arange(data.shape[1])
    colors = ("#0d47a1", "#2e7d32", "#ef6c00", "#6a1b9a")
    for component, axis in enumerate(axes.flat):
        values = np.asarray(data[component], dtype=np.float64)
        axis.plot(x, values, color=colors[component], linewidth=0.9)
        axis.set_title(ZOOM_COMPONENTS[component])
        axis.set_xlabel("zoom bin index")
        axis.set_ylabel("native value")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_zoom_summary(stats: Mapping[str, np.ndarray], x: np.ndarray, x_label: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    median = np.asarray(stats["median"])
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    for component, axis in enumerate(axes.flat):
        valid = np.isfinite(median[:, component])
        axis.fill_between(
            x[valid], low[valid, component], high[valid, component],
            color="#bbdefb", alpha=0.5, linewidth=0,
        )
        axis.plot(x[valid], median[valid, component], color="#0d47a1", linewidth=0.8)
        axis.set_title(ZOOM_COMPONENTS[component])
        axis.set_xlabel(x_label)
        axis.set_ylabel("median and q01-q99")
    figure.suptitle("Zoom spectra: statistics over every packet", fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def zoom_section(bundle, assets_dir: Path, assets: list[Path]):
    value = bundle.zoom_spectra
    if value is None:
        return empty_section("Calibrator zoom spectra are absent."), {"present": False}, {}
    data = np.asarray(value, dtype=np.float64)
    row_count = int(data.shape[0])
    if row_count == 0:
        return empty_section("The zoom table has zero rows."), {
            "present": True,
            "row_count": 0,
        }, {}
    stats = product_packet_statistics(
        row_count, 4, lambda row, component: data[row, component]
    )
    stats_model = product_statistics_model(
        stats, bundle.zoom_unique_ids, labels=ZOOM_COMPONENTS
    )
    x, x_label = time_axis(bundle.zoom_raw_times, bundle.zoom_raw_time_valid)
    parts = []
    selections = {}
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(
            label,
            row,
            bundle.zoom_unique_ids,
            bundle.zoom_raw_times,
            bundle.zoom_raw_time_valid,
        )
        pfb_bins = bundle.zoom_pfb_bins
        pfb_indices = bundle.zoom_pfb_indices
        if pfb_bins is not None:
            caption += f"; PFB bin {int(np.asarray(pfb_bins)[row])}"
        if pfb_indices is not None:
            caption += f"; PFB index {int(np.asarray(pfb_indices)[row])}"
        path = save_figure(
            plot_zoom_endpoint(data[row], f"Zoom spectra: {caption}"),
            assets_dir,
            f"zoom_{label}.png",
            assets,
        )
        parts.append(figure_html(path, f"Zoom - {label} packet", caption))
        selections[label] = {
            "row": row,
            "unique_id": (
                int(np.asarray(bundle.zoom_unique_ids)[row])
                if bundle.zoom_unique_ids is not None
                else None
            ),
            "pfb_bin": (
                int(np.asarray(pfb_bins)[row]) if pfb_bins is not None else None
            ),
            "pfb_index": (
                int(np.asarray(pfb_indices)[row])
                if pfb_indices is not None
                else None
            ),
        }
    path = save_figure(
        plot_zoom_summary(stats, x, x_label),
        assets_dir,
        "zoom_all_packets.png",
        assets,
    )
    parts.append(figure_html(
        path,
        "Zoom - all-packet statistics",
        "Median and q01-q99 are computed over all 64 native zoom bins for AA, BB, ABR, and ABI.",
    ))
    parts.append(product_statistics_table(stats_model))
    pfb_histogram = (
        dict(Counter(int(value) for value in np.asarray(bundle.zoom_pfb_bins)))
        if bundle.zoom_pfb_bins is not None
        else {}
    )
    pfb_index_histogram = (
        dict(Counter(int(value) for value in np.asarray(bundle.zoom_pfb_indices)))
        if bundle.zoom_pfb_indices is not None
        else {}
    )
    parts.append("<h3>PFB-bin and index distribution</h3>")
    parts.append(html_table(
        ("coordinate", "value", "packets"),
        (
            [("PFB bin", value, count) for value, count in sorted(pfb_histogram.items())]
            + [("PFB index", value, count) for value, count in sorted(pfb_index_histogram.items())]
        ),
    ))
    return "".join(parts), {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "pfb_bin_histogram": pfb_histogram,
        "pfb_index_histogram": pfb_index_histogram,
        "component_statistics": stats_model,
    }, selections


def waveform_metrics(data: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(data, dtype=np.float64)
    return {
        "minimum": np.min(values, axis=1),
        "maximum": np.max(values, axis=1),
        "mean": np.mean(values, axis=1),
        "rms": np.sqrt(np.mean(np.square(values), axis=1)),
        "peak_to_peak": np.ptp(values, axis=1),
        "clipped_fraction": np.mean(
            (values == np.iinfo(np.int16).min)
            | (values == np.iinfo(np.int16).max),
            axis=1,
        ),
    }


def plot_waveform_endpoint(values: np.ndarray, title: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 1, figsize=(14, 7))
    waveform = np.asarray(values, dtype=np.float64)
    axes[0].plot(np.arange(waveform.size), waveform, linewidth=0.55, color="#0d47a1")
    axes[0].set_xlabel("ADC sample index")
    axes[0].set_ylabel("raw ADC count")
    axes[0].set_title("Waveform")
    centered = waveform - np.mean(waveform)
    power = np.square(np.abs(np.fft.rfft(centered)))
    axes[1].semilogy(
        np.arange(power.size), np.maximum(power, np.finfo(np.float64).tiny),
        linewidth=0.65, color="#6a1b9a",
    )
    axes[1].set_xlabel("FFT bin index (sample rate not established)")
    axes[1].set_ylabel("derived power")
    axes[1].set_title("Derived FFT-power diagnostic")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_waveform_summary(
    metrics: Mapping[str, np.ndarray],
    channels: np.ndarray,
    x: np.ndarray,
    x_label: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    specs = (
        ("mean", "mean count"),
        ("rms", "RMS count"),
        ("peak_to_peak", "peak-to-peak count"),
        ("clipped_fraction", "int16 clipping fraction"),
    )
    colors = ("#0d47a1", "#2e7d32", "#ef6c00", "#6a1b9a")
    for axis, (name, label) in zip(axes.flat, specs):
        values = np.asarray(metrics[name])
        for channel in sorted({int(value) for value in channels}):
            selected = channels == channel
            axis.plot(
                x[selected], values[selected], marker=".", markersize=3,
                linewidth=0.7, color=colors[channel % len(colors)], label=f"ch{channel}",
            )
        axis.set_title(label)
        axis.set_xlabel(x_label)
        axis.legend(ncol=4)
    figure.suptitle("Waveforms: statistics over every packet", fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def plot_waveform_heatmaps(data: np.ndarray, channels: np.ndarray):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    sample_indices = np.linspace(
        0, data.shape[1] - 1, min(data.shape[1], 2048), dtype=np.int64
    )
    for channel, axis in enumerate(axes):
        selected = np.flatnonzero(channels == channel)
        if selected.size:
            selected = selected[
                np.linspace(0, selected.size - 1, min(selected.size, 256), dtype=np.int64)
            ]
            image = data[np.ix_(selected, sample_indices)]
            vmin, vmax = robust_image_limits(image, True)
            axis.imshow(
                image,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_ylabel(f"ch{channel} row")
        else:
            axis.text(0.5, 0.5, f"channel {channel} absent", ha="center", va="center")
        axis.set_title(f"Channel {channel}")
    axes[-1].set_xlabel("sampled ADC sample index")
    figure.suptitle("Waveform overview across all packets", fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def waveform_section(bundle, assets_dir: Path, assets: list[Path]):
    value = bundle.waveform_data
    if value is None:
        return empty_section("Raw ADC waveforms are absent."), {"present": False}, {}
    data = np.asarray(value)
    row_count = int(data.shape[0])
    if row_count == 0:
        return empty_section("The waveform table has zero rows."), {
            "present": True,
            "row_count": 0,
        }, {}
    channels = np.asarray(bundle.waveform_channels, dtype=np.int64)
    adc_timestamps = (
        np.asarray(bundle.waveform_adc_timestamps, dtype=np.uint64)
        if bundle.waveform_adc_timestamps is not None
        else None
    )
    adc_valid = (
        np.asarray(bundle.waveform_adc_timestamp_valid, dtype=np.bool_)
        if bundle.waveform_adc_timestamp_valid is not None
        else (
            np.ones(row_count, dtype=np.bool_)
            if adc_timestamps is not None
            else np.zeros(row_count, dtype=np.bool_)
        )
    )
    metrics = waveform_metrics(data)
    x, x_label = time_axis(bundle.waveform_raw_times, bundle.waveform_raw_time_valid)
    parts = []
    selections = {}
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(
            label,
            row,
            bundle.waveform_unique_ids,
            bundle.waveform_raw_times,
            bundle.waveform_raw_time_valid,
        )
        caption += f"; channel {int(channels[row])}"
        if adc_timestamps is not None and adc_valid[row]:
            caption += f"; ADC timestamp {int(adc_timestamps[row])}"
        elif adc_timestamps is not None:
            caption += "; ADC timestamp unavailable"
        path = save_figure(
            plot_waveform_endpoint(data[row], f"Waveform: {caption}"),
            assets_dir,
            f"waveform_{label}.png",
            assets,
        )
        parts.append(figure_html(path, f"Waveform - {label} packet", caption))
        selections[label] = {
            "row": row,
            "unique_id": (
                int(np.asarray(bundle.waveform_unique_ids)[row])
                if bundle.waveform_unique_ids is not None
                else None
            ),
            "channel": int(channels[row]),
            "adc_timestamp": (
                int(adc_timestamps[row])
                if adc_timestamps is not None and adc_valid[row]
                else None
            ),
        }
    path = save_figure(
        plot_waveform_summary(metrics, channels, x, x_label),
        assets_dir,
        "waveform_all_packets.png",
        assets,
    )
    parts.append(figure_html(
        path,
        "Waveform - all-packet statistics",
        "Mission time drives the horizontal axis; the independent ADC counter is reported separately.",
    ))
    heatmap_path = save_figure(
        plot_waveform_heatmaps(data, channels),
        assets_dir,
        "waveform_heatmaps.png",
        assets,
    )
    parts.append(figure_html(
        heatmap_path,
        "Waveform - all-packet heatmaps",
        "Rows and ADC samples are deterministically sampled only for display.",
    ))
    table_rows = []
    for row in range(row_count):
        table_rows.append((
            row,
            int(channels[row]),
            int(np.asarray(bundle.waveform_unique_ids)[row]) if bundle.waveform_unique_ids is not None else None,
            int(adc_timestamps[row]) if adc_timestamps is not None and adc_valid[row] else None,
            bool(adc_valid[row]),
            metrics["minimum"][row],
            metrics["maximum"][row],
            metrics["mean"][row],
            metrics["rms"][row],
            metrics["peak_to_peak"][row],
            metrics["clipped_fraction"][row],
        ))
    parts.append(html_table(
        ("row", "channel", "UID", "ADC timestamp", "ADC time valid", "min", "max", "mean", "RMS", "peak-to-peak", "clipped fraction"),
        table_rows,
    ))
    nonmonotonic_by_channel = {}
    if adc_timestamps is not None:
        for channel in sorted({int(value) for value in channels}):
            selected = (channels == channel) & adc_valid
            channel_adc = adc_timestamps[selected]
            nonmonotonic_by_channel[str(channel)] = (
                int(np.count_nonzero(channel_adc[1:] <= channel_adc[:-1]))
                if channel_adc.size > 1
                else 0
            )
    unique_ids = (
        np.asarray(bundle.waveform_unique_ids, dtype=np.int64)
        if bundle.waveform_unique_ids is not None
        else None
    )
    within_uid_disagreement = 0
    if adc_timestamps is not None and unique_ids is not None:
        for uid in np.unique(unique_ids):
            selected = (unique_ids == uid) & adc_valid
            if np.unique(adc_timestamps[selected]).size > 1:
                within_uid_disagreement += 1
    parts.extend((
        "<h3>ADC timestamp health</h3>",
        html_table(
            ("check", "value"),
            (
                ("valid rows", f"{np.count_nonzero(adc_valid)}/{row_count}"),
                ("within-UID disagreements", within_uid_disagreement),
                *(
                    (f"channel {channel} non-increasing steps", count)
                    for channel, count in sorted(nonmonotonic_by_channel.items())
                ),
            ),
        ),
    ))
    return "".join(parts), {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "channel_histogram": dict(Counter(int(value) for value in channels)),
        "adc_timestamp_valid_rows": int(np.count_nonzero(adc_valid)),
        "adc_timestamp_invalid_rows": int(row_count - np.count_nonzero(adc_valid)),
        "adc_timestamp_nonmonotonic_steps": int(sum(nonmonotonic_by_channel.values())),
        "adc_timestamp_nonmonotonic_steps_by_channel": nonmonotonic_by_channel,
        "adc_timestamp_within_uid_disagreement_count": within_uid_disagreement,
        "packet_metrics": {name: json_value(value) for name, value in metrics.items()},
    }, selections


def flatten_mapping(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for name, item in value.items():
            path = f"{prefix}.{name}" if prefix else str(name)
            if isinstance(item, Mapping):
                result.update(flatten_mapping(item, path))
            else:
                result[path] = item
        return result
    return {prefix or "value": value}


def field_rows_model(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    flattened = [flatten_mapping(row) for row in rows]
    names = sorted({name for row in flattened for name in row})
    records = []
    for name in names:
        values = [row.get(name) for row in flattened]
        present = [value for value in values if value is not None]
        dtypes = sorted({str(np.asarray(value).dtype) for value in present})
        shapes = sorted({format_shape(np.asarray(value).shape) for value in present})
        numeric_arrays = []
        numeric = True
        for value in present:
            array = np.asarray(value)
            if array.dtype.kind not in "biufc":
                numeric = False
                break
            numeric_arrays.append(array.reshape(-1))
        statistics: Mapping[str, object]
        if numeric and numeric_arrays:
            statistics = finite_statistics(np.concatenate(numeric_arrays))
        else:
            strings = [format_array(value) for value in present]
            statistics = {
                "count": len(strings),
                "unique_count": len(set(strings)),
                "most_common": Counter(strings).most_common(6),
            }
        records.append({
            "name": name,
            "rows_present": len(present),
            "rows_missing": len(values) - len(present),
            "dtypes": dtypes,
            "shapes": shapes,
            "first_row": format_array(values[0]) if values and values[0] is not None else "-",
            "last_row": format_array(values[-1]) if values and values[-1] is not None else "-",
            "statistics": statistics,
        })
    return records


def field_rows_table(records: Sequence[Mapping[str, object]]) -> str:
    rows = []
    for record in records:
        stats = record["statistics"]
        if stats.get("numeric"):
            summary = (
                f"min {format_number(stats.get('minimum'))}; "
                f"median {format_number(stats.get('median'))}; "
                f"max {format_number(stats.get('maximum'))}"
            )
        else:
            summary = (
                f"{stats.get('unique_count', 0)} unique; "
                + ", ".join(
                    f"{value} x{count}" for value, count in stats.get("most_common", ())
                )
            )
        rows.append((
            record["name"],
            record["rows_present"],
            record["rows_missing"],
            ", ".join(record["dtypes"]),
            ", ".join(record["shapes"]),
            record["first_row"],
            record["last_row"],
            summary,
        ))
    return html_table(
        ("field", "rows present", "missing", "dtype variants", "shape variants", "first row", "last row", "all-row summary"),
        rows,
    )


def numeric_field_matrix(
    rows: Sequence[Mapping[str, object]],
    name: str,
) -> np.ndarray | None:
    flattened = [flatten_mapping(row) for row in rows]
    present = [
        np.asarray(row[name])
        for row in flattened
        if row.get(name) is not None
    ]
    if not present or any(value.dtype.kind not in "biufc" for value in present):
        return None
    shape = present[0].shape
    if any(value.shape != shape for value in present):
        return None
    width = int(np.prod(shape))
    if width == 0:
        return np.empty((len(rows), 0), dtype=np.float64)
    matrix = np.full((len(rows), width), np.nan)
    for row_index, row in enumerate(flattened):
        if row.get(name) is not None:
            matrix[row_index] = np.asarray(row[name], dtype=np.float64).reshape(-1)
    return matrix


def plot_numeric_field_page(
    rows: Sequence[Mapping[str, object]],
    names: Sequence[str],
    title: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(3, 2, figsize=(13, 9))
    for axis, name in zip(axes.flat, names):
        matrix = numeric_field_matrix(rows, name)
        if matrix is None:
            axis.text(0.5, 0.5, "variant shapes", ha="center", va="center")
        elif matrix.shape[1] == 0:
            axis.text(0.5, 0.5, "empty vector", ha="center", va="center")
        elif matrix.shape[1] <= 8:
            for component in range(matrix.shape[1]):
                axis.plot(
                    np.arange(matrix.shape[0]),
                    matrix[:, component],
                    marker=".",
                    markersize=2.5,
                    linewidth=0.65,
                    label=(f"component {component}" if matrix.shape[1] > 1 else None),
                )
            if matrix.shape[1] > 1:
                axis.legend(ncol=min(4, matrix.shape[1]))
            axis.set_xlabel("packet row")
        else:
            vmin, vmax = robust_image_limits(matrix, True)
            axis.imshow(
                matrix.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_xlabel("packet row")
            axis.set_ylabel("component")
        axis.set_title(name)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=13, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def numeric_field_figures(
    rows: Sequence[Mapping[str, object]],
    records: Sequence[Mapping[str, object]],
    title: str,
    filename_prefix: str,
    assets_dir: Path,
    assets: list[Path],
) -> str:
    if len(rows) <= 3:
        return '<p class="note">Three or fewer rows: values are clearer in the complete table than in a timeline.</p>'
    names = [
        str(record["name"])
        for record in records
        if record.get("statistics", {}).get("numeric")
    ]
    parts = []
    for page, start in enumerate(range(0, len(names), 6), start=1):
        page_names = names[start:start + 6]
        path = save_figure(
            plot_numeric_field_page(
                rows,
                page_names,
                f"{title}: fields {start + 1}-{start + len(page_names)}",
            ),
            assets_dir,
            f"{filename_prefix}_fields_{page:02d}.png",
            assets,
        )
        parts.append(figure_html(
            path,
            f"{title} - numeric fields {start + 1}-{start + len(page_names)}",
            "Every numeric field with a stable shape is plotted. Larger vectors use a row-by-component heatmap; variant-shape fields remain in the complete table.",
        ))
    return "".join(parts)


def housekeeping_section(bundle, assets_dir: Path, assets: list[Path]):
    rows = bundle.housekeeping_fields
    if not rows:
        return empty_section("Housekeeping packets are absent."), {"present": False}, {}
    row_count = len(rows)
    types = np.asarray(bundle.housekeeping_types, dtype=np.int64)
    versions = np.asarray(bundle.housekeeping_versions, dtype=np.int64)
    errors = np.asarray(bundle.housekeeping_firmware_errors, dtype=np.uint64)
    enriched_rows_list = []
    for row in range(row_count):
        payload = dict(rows[row])
        adc_valid_value = payload.get("adc_statistics_valid")
        if adc_valid_value is not None:
            adc_valid = np.asarray(adc_valid_value, dtype=np.bool_)
            for name in ("adc_min", "adc_max", "adc_mean", "adc_rms"):
                if payload.get(name) is None:
                    continue
                values = np.asarray(payload[name], dtype=np.float64).copy()
                values[~adc_valid] = np.nan
                payload[name] = values
        enriched_rows_list.append({
            "packet": {
                "type": int(types[row]),
                "version": int(versions[row]),
                "firmware_errors": int(errors[row]),
            },
            "fields": payload,
        })
    enriched_rows = tuple(enriched_rows_list)
    records = field_rows_model(enriched_rows)
    endpoint = []
    for row in (0, row_count - 1):
        endpoint.append((
            row,
            int(np.asarray(bundle.housekeeping_unique_ids)[row]) if bundle.housekeeping_unique_ids is not None else None,
            int(types[row]),
            f"0x{int(versions[row]):X}",
            f"0x{int(errors[row]):08X}",
            format_array(flatten_mapping(enriched_rows[row]["fields"])),
        ))
    parts = [
        "<h3>First and last housekeeping packets</h3>",
        html_table(("row", "UID", "type", "version", "firmware errors", "fields"), endpoint),
        "<h3>Complete housekeeping field table</h3>",
        field_rows_table(records),
        numeric_field_figures(
            enriched_rows,
            records,
            "Housekeeping",
            "housekeeping",
            assets_dir,
            assets,
        ),
        "<h3>Packet-type and error summary</h3>",
        html_table(
            ("HK type", "rows"),
            sorted(Counter(int(value) for value in types).items()),
        ),
        html_table(
            ("firmware error word", "rows"),
            (
                (f"0x{value:08X}", count)
                for value, count in sorted(
                    Counter(int(item) for item in errors).items()
                )
            ),
        ),
    ]
    selections = {
        "first": {"row": 0, "unique_id": int(np.asarray(bundle.housekeeping_unique_ids)[0]) if bundle.housekeeping_unique_ids is not None else None},
        "last": {"row": row_count - 1, "unique_id": int(np.asarray(bundle.housekeeping_unique_ids)[-1]) if bundle.housekeeping_unique_ids is not None else None},
    }
    return "".join(parts), {
        "present": True,
        "row_count": row_count,
        "type_histogram": dict(Counter(int(value) for value in types)),
        "version_histogram": {f"0x{key:X}": value for key, value in Counter(int(item) for item in versions).items()},
        "firmware_error_nonzero_rows": int(np.count_nonzero(errors)),
        "field_statistics": records,
    }, selections


def plot_complex_endpoint(data: np.ndarray, title: str, x_label: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
    x = np.arange(data.shape[1])
    for channel in range(4):
        values = np.asarray(data[channel], dtype=np.complex128)
        axes[channel, 0].plot(x, np.abs(values), linewidth=0.75, color="#0d47a1")
        axes[channel, 0].set_ylabel(f"ch{channel}")
        axes[channel, 0].set_title(f"Channel {channel} magnitude")
        axes[channel, 1].plot(x, np.angle(values), linewidth=0.75, color="#6a1b9a")
        axes[channel, 1].set_title(f"Channel {channel} phase")
        axes[channel, 1].set_ylim(-math.pi, math.pi)
    axes[-1, 0].set_xlabel(x_label)
    axes[-1, 1].set_xlabel(x_label)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def plot_complex_summary(
    stats: Mapping[str, np.ndarray],
    title: str,
    x: np.ndarray,
    x_label: str,
):
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    median = np.asarray(stats["median"])
    low = np.asarray(stats["q01"])
    high = np.asarray(stats["q99"])
    for channel, axis in enumerate(axes.flat):
        valid = np.isfinite(median[:, channel])
        axis.fill_between(
            x[valid], low[valid, channel], high[valid, channel],
            color="#bbdefb", alpha=0.5, linewidth=0,
        )
        axis.plot(x[valid], median[valid, channel], color="#0d47a1", linewidth=0.8)
        axis.set_title(f"Channel {channel} magnitude")
        axis.set_xlabel(x_label)
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def calibrator_page_timing(
    family: Mapping[str, object],
) -> tuple[str, Mapping[str, object]]:
    raw = np.asarray(family.get("page_raw_seconds", []), dtype=np.float64)
    mjd = np.asarray(family.get("page_mjd_times", []), dtype=np.float64)
    valid = np.asarray(family.get("page_mjd_time_valid", []), dtype=np.bool_)
    if raw.ndim != 2 or raw.size == 0:
        return empty_section("Per-page calibrator timing is unavailable."), {
            "present": False,
        }
    if mjd.shape != raw.shape or valid.shape != raw.shape:
        raise ValueError("calibrator page timing arrays do not have matching shapes")

    unique_ids = np.asarray(family["unique_ids"], dtype=np.uint32)
    if unique_ids.shape != (raw.shape[0],):
        raise ValueError("calibrator page timing rows do not match unique IDs")
    usable_mjd = np.where(valid & np.isfinite(mjd), mjd, np.nan)
    page_records = []
    for page in range(raw.shape[1]):
        raw_stats = finite_statistics(raw[:, page])
        mjd_stats = finite_statistics(usable_mjd[:, page])
        page_records.append({
            "page": page,
            "raw_finite_rows": raw_stats["finite_count"],
            "raw_minimum_seconds": raw_stats["minimum"],
            "raw_median_seconds": raw_stats["median"],
            "raw_maximum_seconds": raw_stats["maximum"],
            "valid_mjd_rows": mjd_stats["finite_count"],
            "mjd_minimum": mjd_stats["minimum"],
            "mjd_maximum": mjd_stats["maximum"],
        })

    endpoints = []
    for label, row in (("first", 0), ("last", raw.shape[0] - 1)):
        endpoints.append({
            "selection": label,
            "row": row,
            "unique_id": int(unique_ids[row]),
            "raw_seconds": raw[row].tolist(),
            "mjd_times": usable_mjd[row].tolist(),
            "mjd_valid": valid[row].tolist(),
        })

    deltas = np.diff(raw, axis=1)
    finite_deltas = np.isfinite(deltas)
    decreasing = np.any(finite_deltas & (deltas < 0), axis=1)
    repeated = np.any(finite_deltas & (deltas == 0), axis=1)
    model = {
        "present": True,
        "row_count": int(raw.shape[0]),
        "page_count": int(raw.shape[1]),
        "page_statistics": page_records,
        "endpoint_rows": endpoints,
        "valid_mjd_values": int(np.count_nonzero(valid & np.isfinite(mjd))),
        "invalid_mjd_values": int(valid.size - np.count_nonzero(valid & np.isfinite(mjd))),
        "raw_page_delta_statistics": finite_statistics(deltas),
        "rows_with_decreasing_page_times": int(np.count_nonzero(decreasing)),
        "rows_with_repeated_page_times": int(np.count_nonzero(repeated)),
    }
    body = "".join((
        (
            '<p class="note">Each record is assembled from several source pages. '
            "Invalid converted clock values are shown as NaN; raw page times remain "
            "visible.</p>"
        ),
        html_table(
            (
                "selection",
                "row",
                "UID",
                "raw page seconds",
                "page MJD (invalid = NaN)",
                "MJD validity",
            ),
            (
                (
                    record["selection"],
                    record["row"],
                    record["unique_id"],
                    format_array(record["raw_seconds"]),
                    format_array(record["mjd_times"]),
                    format_array(record["mjd_valid"]),
                )
                for record in endpoints
            ),
        ),
        html_table(
            (
                "page",
                "raw finite rows",
                "raw minimum s",
                "raw median s",
                "raw maximum s",
                "valid MJD rows",
                "MJD minimum",
                "MJD maximum",
            ),
            (
                (
                    record["page"],
                    record["raw_finite_rows"],
                    record["raw_minimum_seconds"],
                    record["raw_median_seconds"],
                    record["raw_maximum_seconds"],
                    record["valid_mjd_rows"],
                    record["mjd_minimum"],
                    record["mjd_maximum"],
                )
                for record in page_records
            ),
        ),
        html_table(
            ("check", "value"),
            (
                ("valid converted MJD page values", model["valid_mjd_values"]),
                ("invalid converted MJD page values", model["invalid_mjd_values"]),
                ("rows with decreasing raw page times", model["rows_with_decreasing_page_times"]),
                ("rows with repeated raw page times", model["rows_with_repeated_page_times"]),
                (
                    "median raw step between pages (s)",
                    model["raw_page_delta_statistics"]["median"],
                ),
            ),
        ),
    ))
    return body, model


def complex_family_html(
    name: str,
    family: Mapping[str, object],
    assets_dir: Path,
    assets: list[Path],
    *,
    x_label: str,
) -> tuple[str, Mapping[str, object], Mapping[str, object]]:
    data = np.asarray(family["data"], dtype=np.complex128)
    row_count = int(data.shape[0])
    unique_ids = np.asarray(family["unique_ids"], dtype=np.uint32)
    raw_times = np.asarray(family["raw_seconds"], dtype=np.float64)
    raw_valid = np.asarray(family["raw_time_valid"], dtype=np.bool_)
    stats = product_packet_statistics(
        row_count, 4, lambda row, channel: np.abs(data[row, channel])
    )
    stats_model = product_statistics_model(
        stats, unique_ids, labels=tuple(f"Channel {channel}" for channel in range(4))
    )
    x, axis_label = time_axis(raw_times, raw_valid)
    parts = []
    selections = {}
    slug = name.replace(" ", "_").lower()
    for label, row in (("first", 0), ("last", row_count - 1)):
        caption = row_caption(label, row, unique_ids, raw_times, raw_valid)
        path = save_figure(
            plot_complex_endpoint(data[row], f"{name}: {caption}", x_label),
            assets_dir,
            f"{slug}_{label}.png",
            assets,
        )
        parts.append(figure_html(path, f"{name} - {label} packet", caption))
        selections[label] = {"row": row, "unique_id": int(unique_ids[row])}
    path = save_figure(
        plot_complex_summary(
            stats, f"{name}: magnitude statistics over every packet", x, axis_label
        ),
        assets_dir,
        f"{slug}_all_packets.png",
        assets,
    )
    parts.append(figure_html(
        path,
        f"{name} - all-packet statistics",
        "Magnitude is derived from the stored real and imaginary arrays; no physical unit is inferred.",
    ))
    parts.append(product_statistics_table(stats_model))
    timing_html, timing_model = calibrator_page_timing(family)
    parts.extend(("<h4>Page timing and clock validity</h4>", timing_html))
    model = {
        "present": True,
        "row_count": row_count,
        "shape": list(data.shape),
        "magnitude_statistics": stats_model,
        "page_timing": timing_model,
    }
    return "".join(parts), model, selections


def plot_gphase(gphase: np.ndarray, title: str):
    plt = import_pyplot()
    figure, axes = plt.subplots(3, 1, figsize=(13, 10))
    axes[0].plot(np.arange(gphase.shape[1]), gphase[0], linewidth=0.65)
    axes[0].set_title("First row")
    axes[0].set_xlabel("gphase index")
    axes[1].plot(np.arange(gphase.shape[1]), gphase[-1], linewidth=0.65)
    axes[1].set_title("Last row")
    axes[1].set_xlabel("gphase index")
    shown = downsample_rows(np.asarray(gphase))
    vmin, vmax = robust_image_limits(shown, True)
    axes[2].imshow(
        shown.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_title("All rows")
    axes[2].set_xlabel("sampled calibrator row")
    axes[2].set_ylabel("gphase index")
    figure.suptitle(title, fontsize=14, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def calibrator_section(bundle, assets_dir: Path, assets: list[Path]):
    calibrator = bundle.calibrator
    if not calibrator:
        return empty_section("Calibrator packets are absent."), {"present": False}, {}
    parts: list[str] = []
    model: dict[str, object] = {"present": True, "families": {}}
    selections: dict[str, object] = {}

    metadata = calibrator.get("metadata")
    if metadata is not None and len(metadata["fields"]) == 0:
        parts.extend((
            "<h3>Calibrator metadata</h3>",
            empty_section("The calibrator metadata group has zero rows."),
        ))
        model["families"]["metadata"] = {"row_count": 0}
        metadata = None
    if metadata is not None:
        rows = tuple(metadata["fields"])
        from_debug = np.asarray(metadata["from_debug"], dtype=np.bool_)
        enriched_rows = tuple(
            {"from_debug": bool(from_debug[row]), "fields": rows[row]}
            for row in range(len(rows))
        )
        records = field_rows_model(enriched_rows)
        parts.extend((
            "<h3>Calibrator metadata</h3>",
            html_table(
                ("row", "UID", "from debug", "first/last fields"),
                (
                    (
                        row,
                        int(np.asarray(metadata["unique_ids"])[row]),
                        bool(from_debug[row]),
                        format_array(flatten_mapping(rows[row])),
                    )
                    for row in sorted({0, len(rows) - 1})
                ),
            ),
            field_rows_table(records),
            numeric_field_figures(
                enriched_rows,
                records,
                "Calibrator metadata",
                "calibrator_metadata",
                assets_dir,
                assets,
            ),
        ))
        model["families"]["metadata"] = {
            "row_count": len(rows),
            "from_debug_rows": int(np.count_nonzero(from_debug)),
            "field_statistics": records,
        }
        selections["metadata"] = {
            "first": {"row": 0, "unique_id": int(np.asarray(metadata["unique_ids"])[0])},
            "last": {"row": len(rows) - 1, "unique_id": int(np.asarray(metadata["unique_ids"])[-1])},
        }

    data_family = calibrator.get("data")
    if data_family is not None and len(data_family["unique_ids"]) == 0:
        parts.extend((
            "<h3>Calibrator complex data</h3>",
            empty_section("The calibrator complex-data group has zero rows."),
        ))
        model["families"]["data"] = {"row_count": 0}
        data_family = None
    if data_family is not None:
        parts.append("<h3>Calibrator complex data</h3>")
        family_html, family_model, family_selection = complex_family_html(
            "Calibrator data",
            data_family,
            assets_dir,
            assets,
            x_label="calibrator data index",
        )
        parts.append(family_html)
        gphase = np.asarray(data_family["gphase"], dtype=np.int32)
        path = save_figure(
            plot_gphase(gphase, "Calibrator gphase: first and last rows"),
            assets_dir,
            "calibrator_gphase.png",
            assets,
        )
        parts.append(figure_html(
            path,
            "Calibrator gphase",
            "Native integer gphase arrays; horizontal coordinates are stored indices.",
        ))
        g_nacc = np.asarray(data_family["g_nacc"], dtype=np.int32)
        parts.extend((
            "<h3>Calibrator accumulation settings</h3>",
            html_table(
                ("field", "minimum", "median", "maximum"),
                ((
                    "g_nacc",
                    finite_statistics(g_nacc).get("minimum"),
                    finite_statistics(g_nacc).get("median"),
                    finite_statistics(g_nacc).get("maximum"),
                ),),
            ),
        ))
        family_model = dict(family_model)
        family_model.update({
            "g_nacc_statistics": finite_statistics(g_nacc),
            "gphase_statistics": finite_statistics(gphase),
        })
        model["families"]["data"] = family_model
        selections["data"] = family_selection

    raw_pfb = calibrator.get("raw_pfb")
    if raw_pfb is not None and len(raw_pfb["unique_ids"]) == 0:
        parts.extend((
            "<h3>Calibrator raw PFB</h3>",
            empty_section("The calibrator raw-PFB group has zero rows."),
        ))
        model["families"]["raw_pfb"] = {"row_count": 0}
        raw_pfb = None
    if raw_pfb is not None:
        parts.append("<h3>Calibrator raw PFB</h3>")
        family_html, family_model, family_selection = complex_family_html(
            "Calibrator raw PFB", raw_pfb, assets_dir, assets, x_label="PFB bin index"
        )
        parts.append(family_html)
        model["families"]["raw_pfb"] = family_model
        selections["raw_pfb"] = family_selection

    debug = calibrator.get("debug")
    if debug is not None and len(debug["unique_ids"]) == 0:
        parts.extend((
            "<h3>Calibrator debug pages</h3>",
            empty_section("The calibrator debug group has zero rows."),
        ))
        model["families"]["debug"] = {
            "row_count": 0,
            "page_count": len(debug.get("pages", ())),
        }
        debug = None
    if debug is not None:
        page_models = tuple(debug["pages"])
        row_count = len(debug["unique_ids"])
        rows: list[dict[str, object]] = [{} for _ in range(row_count)]
        for page_index, page in enumerate(page_models):
            for row in range(row_count):
                rows[row][f"page_{page_index}"] = page["fields"][row]
        records = field_rows_model(rows)
        timing_html, timing_model = calibrator_page_timing(debug)
        parts.extend((
            "<h3>Calibrator debug pages</h3>",
            html_table(
                ("row", "UID", "page raw times", "first/last fields"),
                (
                    (
                        row,
                        int(np.asarray(debug["unique_ids"])[row]),
                        format_array(np.asarray(debug["page_raw_seconds"])[row]),
                        format_array(flatten_mapping(rows[row])),
                    )
                    for row in sorted({0, row_count - 1})
                ),
            ),
            field_rows_table(records),
            numeric_field_figures(
                rows,
                records,
                "Calibrator debug pages",
                "calibrator_debug",
                assets_dir,
                assets,
            ),
            "<h4>Page timing and clock validity</h4>",
            timing_html,
        ))
        model["families"]["debug"] = {
            "row_count": row_count,
            "page_count": len(page_models),
            "page_timing": timing_model,
            "field_statistics": records,
        }
        selections["debug"] = {
            "first": {"row": 0, "unique_id": int(np.asarray(debug["unique_ids"])[0])},
            "last": {"row": row_count - 1, "unique_id": int(np.asarray(debug["unique_ids"])[-1])},
        }

    return "".join(parts), model, selections


def direct_dataset_values(group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][...])


def direct_value(array: np.ndarray | None, row: int) -> object:
    if array is None:
        return None
    value = array[row]
    if np.asarray(value).ndim == 0:
        item = np.asarray(value).item()
        return decode_text(item) if isinstance(item, (bytes, np.bytes_)) else item
    return format_array(value)


def write_csv_asset(
    path: Path,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    assets: list[Path],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])
    assets.append(path)
    return path


def family_status_model(h5) -> list[dict[str, object]]:
    path = "/status/families"
    if path not in h5:
        return []
    group = h5[path]
    families = direct_dataset_values(group, "family")
    count = 0 if families is None else len(families)
    columns = {
        name: direct_dataset_values(group, name)
        for name in (
            "family",
            "supported",
            "coverage",
            "quality",
            "decoded_rows",
            "persisted_rows",
            "reason",
            "reason_valid",
        )
    }
    records = []
    for row in range(count):
        reason_valid = bool(columns["reason_valid"][row])
        records.append({
            "family": decode_text(columns["family"][row]),
            "supported": bool(columns["supported"][row]),
            "coverage": decode_text(columns["coverage"][row]),
            "quality": decode_text(columns["quality"][row]),
            "decoded_rows": int(columns["decoded_rows"][row]),
            "persisted_rows": int(columns["persisted_rows"][row]),
            "reason": decode_text(columns["reason"][row]) if reason_valid else None,
        })
    return records


def family_status_html(records: Sequence[Mapping[str, object]]) -> str:
    if not records:
        return empty_section("Family status table is unavailable.")
    return html_table(
        ("family", "supported", "coverage", "quality", "decoded rows", "persisted rows", "reason"),
        (
            (
                record["family"],
                record["supported"],
                record["coverage"],
                record["quality"],
                record["decoded_rows"],
                record["persisted_rows"],
                record["reason"],
            )
            for record in records
        ),
        css_class="status-table",
    )


def issue_model(h5) -> list[dict[str, object]]:
    if "/issues" not in h5:
        return []
    group = h5["/issues"]
    count = int(group.attrs.get("count", 0))
    names = tuple(group.keys())
    columns = {name: np.asarray(group[name][...]) for name in names}
    records = []
    for row in range(count):
        record: dict[str, object] = {}
        for name, values in columns.items():
            if name.endswith("_valid"):
                continue
            valid_name = f"{name}_valid"
            if valid_name in columns and not bool(columns[valid_name][row]):
                record[name] = None
            else:
                record[name] = json_value(values[row])
        details = record.get("details_json")
        if isinstance(details, str):
            try:
                record["details"] = json.loads(details)
            except json.JSONDecodeError:
                record["details"] = details
        records.append(record)
    return records


def issues_html(records: Sequence[Mapping[str, object]]) -> str:
    if not records:
        return '<div class="good-state">No stored ingest issues.</div>'
    severity = Counter(str(record.get("severity")) for record in records)
    stage = Counter(str(record.get("stage")) for record in records)
    code = Counter(str(record.get("code")) for record in records)
    parts = [
        '<div class="metric-grid">',
        *(f'<div class="metric"><span>{html.escape(name)}</span><strong>{count:,}</strong></div>' for name, count in sorted(severity.items())),
        "</div>",
        "<h3>Issue codes</h3>",
        html_table(("code", "count"), sorted(code.items(), key=lambda item: (-item[1], item[0]))),
        "<h3>Issue stages</h3>",
        html_table(("stage", "count"), sorted(stage.items(), key=lambda item: (-item[1], item[0]))),
        "<h3>Complete issue list</h3>",
        html_table(
            (
                "#",
                "issue ID",
                "severity",
                "stage",
                "code",
                "action",
                "input identity",
                "session",
                "bank",
                "byte offset",
                "frame index",
                "packet index",
                "APID",
                "sequence count",
                "UID",
                "message",
                "details",
            ),
            (
                (
                    index,
                    record.get("issue_id"),
                    record.get("severity"),
                    record.get("stage"),
                    record.get("code"),
                    record.get("action"),
                    record.get("input_identity"),
                    record.get("session"),
                    record.get("bank"),
                    record.get("byte_offset"),
                    record.get("frame_index"),
                    record.get("packet_index"),
                    record.get("appid"),
                    record.get("sequence_count"),
                    record.get("uid"),
                    record.get("message"),
                    json.dumps(record.get("details"), sort_keys=True, default=str),
                )
                for index, record in enumerate(records)
            ),
        ),
    ]
    return "".join(parts)


def decoder_provenance_model(h5) -> dict[str, object]:
    path = "/provenance/decoder"
    if path not in h5:
        return {}
    group = h5[path]
    attrs = {name: json_value(value) for name, value in sorted(group.attrs.items())}
    datasets = {name: json_value(group[name][...]) for name in sorted(group.keys())}
    return {"attributes": attrs, "datasets": datasets}


def provenance_group_model(h5, path: str) -> dict[str, object]:
    if path not in h5:
        return {}
    group = h5[path]
    return {
        "attributes": {
            name: json_value(value) for name, value in sorted(group.attrs.items())
        },
        "datasets": {
            name: json_value(group[name][...]) for name in sorted(group.keys())
        },
    }


def provenance_group_html(model: Mapping[str, object], absent: str) -> str:
    if not model:
        return empty_section(absent)
    rows = []
    rows.extend(
        ("attribute", name, format_array(value))
        for name, value in model.get("attributes", {}).items()
    )
    rows.extend(
        ("dataset", name, format_array(value))
        for name, value in model.get("datasets", {}).items()
    )
    return html_table(("storage", "name", "value"), rows)


def sequence_health_model(
    name: str,
    identifiers: object | None,
    raw_times: object | None,
    raw_valid: object | None,
    mjd_times: object | None,
    mjd_valid: object | None,
) -> dict[str, object] | None:
    candidates = (
        identifiers,
        raw_times,
        raw_valid,
        mjd_times,
        mjd_valid,
    )
    row_count = next(
        (int(np.asarray(value).shape[0]) for value in candidates if value is not None),
        0,
    )
    if row_count == 0:
        return None
    ids = (
        np.asarray(identifiers, dtype=np.int64)
        if identifiers is not None
        else None
    )
    raw = (
        np.asarray(raw_times, dtype=np.float64)
        if raw_times is not None
        else np.full(row_count, np.nan, dtype=np.float64)
    )
    raw_mask = (
        np.asarray(raw_valid, dtype=np.bool_)
        if raw_valid is not None
        else np.isfinite(raw)
    ) & np.isfinite(raw)
    mjd = (
        np.asarray(mjd_times, dtype=np.float64)
        if mjd_times is not None
        else np.full(row_count, np.nan, dtype=np.float64)
    )
    mjd_mask = (
        np.asarray(mjd_valid, dtype=np.bool_)
        if mjd_valid is not None
        else np.isfinite(mjd)
    ) & np.isfinite(mjd)
    valid_raw = raw[raw_mask]
    cadence = np.diff(valid_raw)
    return {
        "family": name,
        "row_count": row_count,
        "identifier_first": int(ids[0]) if ids is not None else None,
        "identifier_last": int(ids[-1]) if ids is not None else None,
        "identifier_nonincreasing_steps": (
            int(np.count_nonzero(np.diff(ids) <= 0))
            if ids is not None and ids.size > 1
            else 0
        ),
        "identifier_nonunit_steps": (
            int(np.count_nonzero(np.diff(ids) != 1))
            if ids is not None and ids.size > 1
            else 0
        ),
        "raw_time_valid_rows": int(np.count_nonzero(raw_mask)),
        "raw_time_first": float(valid_raw[0]) if valid_raw.size else None,
        "raw_time_last": float(valid_raw[-1]) if valid_raw.size else None,
        "raw_time_nonincreasing_steps": int(np.count_nonzero(cadence <= 0)),
        "cadence_seconds": finite_statistics(cadence),
        "mjd_time_valid_rows": int(np.count_nonzero(mjd_mask)),
    }


def timing_and_identity_model(bundle) -> list[dict[str, object]]:
    if bundle is None:
        return []
    rows = []
    families = (
        (
            "normal spectra",
            bundle.spectra_unique_ids,
            bundle.spectra_raw_times,
            bundle.spectra_raw_time_valid,
            bundle.spectra_mjd_times,
            bundle.spectra_mjd_time_valid,
        ),
        (
            "time-resolved spectra",
            bundle.tr_unique_ids,
            bundle.tr_raw_times,
            bundle.tr_raw_time_valid,
            bundle.tr_mjd_times,
            bundle.tr_mjd_time_valid,
        ),
        (
            "zoom spectra",
            bundle.zoom_unique_ids,
            bundle.zoom_raw_times,
            bundle.zoom_raw_time_valid,
            bundle.zoom_mjd_times,
            bundle.zoom_mjd_time_valid,
        ),
        (
            "Grimm spectra",
            bundle.grimm_unique_ids,
            bundle.grimm_raw_times,
            bundle.grimm_raw_time_valid,
            bundle.grimm_mjd_times,
            bundle.grimm_mjd_time_valid,
        ),
        (
            "housekeeping",
            bundle.housekeeping_unique_ids,
            bundle.housekeeping_raw_times,
            bundle.housekeeping_raw_time_valid,
            bundle.housekeeping_mjd_times,
            bundle.housekeeping_mjd_time_valid,
        ),
    )
    for family in families:
        record = sequence_health_model(*family)
        if record is not None:
            rows.append(record)
    if (
        bundle.waveform_unique_ids is not None
        and np.asarray(bundle.waveform_unique_ids).size
    ):
        waveform_ids = np.asarray(bundle.waveform_unique_ids)
        first_rows = np.flatnonzero(np.concatenate((
            np.asarray([True]),
            waveform_ids[1:] != waveform_ids[:-1],
        )))

        def waveform_subset(value):
            return None if value is None else np.asarray(value)[first_rows]

        record = sequence_health_model(
            "waveform groups",
            waveform_subset(bundle.waveform_unique_ids),
            waveform_subset(bundle.waveform_raw_times),
            waveform_subset(bundle.waveform_raw_time_valid),
            waveform_subset(bundle.waveform_mjd_times),
            waveform_subset(bundle.waveform_mjd_time_valid),
        )
        if record is not None:
            record["stored_channel_rows"] = int(waveform_ids.size)
            rows.append(record)
    for name, family in sorted(bundle.calibrator.items()):
        if not isinstance(family, Mapping) or "unique_ids" not in family:
            continue
        record = sequence_health_model(
            f"calibrator {name}",
            family.get("unique_ids"),
            family.get("raw_seconds"),
            family.get("raw_time_valid"),
            family.get("mjd_times"),
            family.get("mjd_time_valid"),
        )
        if record is not None:
            rows.append(record)
    if bundle.telemetry is not None:
        telemetry = bundle.telemetry
        record = sequence_health_model(
            "telemetry",
            telemetry.source_indices,
            telemetry.raw_seconds,
            None,
            telemetry.mjd_times,
            None,
        )
        if record is not None:
            rows.append(record)
    return rows


def timing_and_identity_html(records: Sequence[Mapping[str, object]]) -> str:
    if not records:
        return empty_section("No row-aligned product timing is available.")
    return html_table(
        (
            "family",
            "rows",
            "identifier first..last",
            "non-increasing IDs",
            "non-unit ID steps",
            "raw time valid",
            "raw time first..last",
            "non-increasing raw time",
            "median cadence (s)",
            "MJD valid",
        ),
        (
            (
                record["family"],
                record["row_count"],
                f"{format_number(record['identifier_first'])} .. {format_number(record['identifier_last'])}",
                record["identifier_nonincreasing_steps"],
                record["identifier_nonunit_steps"],
                f"{record['raw_time_valid_rows']}/{record['row_count']}",
                f"{format_number(record['raw_time_first'])} .. {format_number(record['raw_time_last'])}",
                record["raw_time_nonincreasing_steps"],
                record["cadence_seconds"].get("median"),
                f"{record['mjd_time_valid_rows']}/{record['row_count']}",
            )
            for record in records
        ),
    )


def source_packet_model(h5) -> dict[str, object]:
    path = "/provenance/source_packets"
    if path not in h5:
        return {"row_count": 0}
    group = h5[path]
    row_count = int(group["provenance_index"].shape[0])
    original = np.asarray(group["original_appid"], dtype=np.int64)
    roles = [decode_text(value) for value in group["role"][...]]
    normalized = np.asarray(group["normalized_appid"], dtype=np.int64)
    normalized_valid = np.asarray(group["normalized_appid_valid"], dtype=np.bool_)
    optional_validity = {}
    for name in (
        "normalized_appid",
        "packet_index",
        "frame_start",
        "frame_stop",
        "byte_offset_start",
        "byte_offset_stop",
        "filename",
        "bank",
    ):
        valid_name = f"{name}_valid"
        if valid_name in group:
            optional_validity[name] = int(np.count_nonzero(~np.asarray(group[valid_name], dtype=np.bool_)))
    return {
        "row_count": row_count,
        "role_histogram": dict(Counter(roles)),
        "original_appid_histogram": {
            f"0x{key:03X}": value for key, value in Counter(int(item) for item in original).items()
        },
        "normalized_appid_histogram": {
            f"0x{key:03X}": value
            for key, value in Counter(int(item) for item in normalized[normalized_valid]).items()
        },
        "unavailable_optional_fields": optional_validity,
    }


def write_source_packet_csv(h5, assets_dir: Path, assets: list[Path]) -> Path | None:
    path = "/provenance/source_packets"
    if path not in h5:
        return None
    group = h5[path]
    names = sorted(group.keys())
    row_count = int(group["provenance_index"].shape[0])

    def rows():
        for row in range(row_count):
            values = []
            for name in names:
                value = group[name][row]
                if np.asarray(value).ndim == 0:
                    value = np.asarray(value).item()
                values.append(decode_text(value) if isinstance(value, (bytes, np.bytes_)) else value)
            yield values

    return write_csv_asset(assets_dir / "source_packets.csv", names, rows(), assets)


def provenance_table(
    h5,
    path: str,
    label: str,
    filename: str,
    assets_dir: Path,
    assets: list[Path],
) -> tuple[str, Mapping[str, object]]:
    if path not in h5:
        return empty_section(f"{label} is unavailable."), {"present": False}
    group = h5[path]
    names = sorted(group.keys())
    if not names:
        return empty_section(f"{label} has no columns."), {
            "present": True,
            "row_count": 0,
            "columns": [],
        }
    row_count = int(group[names[0]].shape[0])
    if any(group[name].ndim == 0 or group[name].shape[0] != row_count for name in names):
        raise ValueError(f"{path} is not a row-aligned provenance table")
    columns = {name: np.asarray(group[name][...]) for name in names}

    def row_values(row: int) -> list[object]:
        return [direct_value(columns[name], row) for name in names]

    def csv_rows():
        for row in range(row_count):
            yield row_values(row)

    csv_path = write_csv_asset(
        assets_dir / filename,
        names,
        csv_rows(),
        assets,
    )
    if row_count <= 50:
        preview_indices = tuple(range(row_count))
    else:
        preview_indices = tuple(range(25)) + tuple(range(row_count - 25, row_count))
    preview = [
        {name: json_value(value) for name, value in zip(names, row_values(row))}
        for row in preview_indices
    ]
    preview_html = html_table(
        ("row", *names),
        ((row, *row_values(row)) for row in preview_indices),
    )
    if row_count > len(preview_indices):
        preview_note = (
            f'<p class="note">Showing the first and last 25 of {row_count:,} rows. '
            "The CSV contains every row and column.</p>"
        )
    else:
        preview_note = f'<p class="note">Showing all {row_count:,} rows.</p>'
    body = "".join((
        f'<p><a class="download" href="assets/{html.escape(csv_path.name, quote=True)}">Download complete {html.escape(label)} CSV</a></p>',
        preview_note,
        preview_html,
    ))
    return body, {
        "present": True,
        "row_count": row_count,
        "columns": names,
        "preview_row_indices": list(preview_indices),
        "preview": preview,
        "csv": f"assets/{csv_path.name}",
    }


def provenance_section(h5, assets_dir: Path, assets: list[Path], bundle=None):
    family_records = family_status_model(h5)
    issue_records = issue_model(h5)
    decoder = decoder_provenance_model(h5)
    run = provenance_group_model(h5, "/run_provenance")
    clock = provenance_group_model(h5, "/clock_reference")
    invariants = provenance_group_model(h5, "/session_invariants")
    constants = provenance_group_model(h5, "/constants")
    timing = timing_and_identity_model(bundle)
    root_attributes = {
        name: json_value(value) for name, value in sorted(h5.attrs.items())
    }
    source_packets = source_packet_model(h5)
    product_rows = 0
    product_histogram: dict[str, int] = {}
    if "/provenance/product_rows" in h5:
        families = [decode_text(value) for value in h5["/provenance/product_rows/family"][...]]
        product_rows = len(families)
        product_histogram = dict(Counter(families))
    csv_path = write_source_packet_csv(h5, assets_dir, assets)
    relation_tables = {}
    relation_html = []
    for path, label, filename in (
        (
            "/provenance/product_rows",
            "product-row provenance",
            "product_rows.csv",
        ),
        (
            "/provenance/product_schema_refs",
            "product-to-schema references",
            "product_schema_refs.csv",
        ),
        (
            "/provenance/product_issue_refs",
            "product-to-issue references",
            "product_issue_refs.csv",
        ),
        (
            "/status/family_issue_refs",
            "family-to-issue references",
            "family_issue_refs.csv",
        ),
    ):
        table_html, table_model = provenance_table(
            h5, path, label, filename, assets_dir, assets
        )
        relation_tables[path] = table_model
        relation_html.extend((f"<h3>{html.escape(label.title())}</h3>", table_html))
    parts = [
        "<h3>Root status and row counts</h3>",
        html_table(
            ("attribute", "value"),
            ((name, format_array(value)) for name, value in root_attributes.items()),
        ),
        "<h3>Run provenance</h3>",
        provenance_group_html(run, "Run provenance is unavailable."),
        "<h3>Clock reference</h3>",
        provenance_group_html(clock, "Clock reference is unavailable."),
        "<h3>Session invariants</h3>",
        provenance_group_html(invariants, "Session invariants are unavailable."),
        "<h3>Location and constants</h3>",
        provenance_group_html(constants, "Session constants are unavailable."),
        "<h3>Timing and identity health</h3>",
        '<p class="note">Non-unit identifier steps are descriptive: product families may legitimately skip packet IDs. Non-increasing steps are the ordering alarm.</p>',
        timing_and_identity_html(timing),
        "<h3>Family coverage and quality</h3>",
        family_status_html(family_records),
        "<h3>Decoder</h3>",
        provenance_group_html(decoder, "Decoder provenance is unavailable."),
        "<h3>Product rows by family</h3>",
        html_table(("family", "rows"), sorted(product_histogram.items())),
        "<h3>Source-packet roles and APIDs</h3>",
        html_table(
            ("role", "rows"), sorted(source_packets.get("role_histogram", {}).items())
        ),
        html_table(
            ("original APID", "rows"), sorted(source_packets.get("original_appid_histogram", {}).items())
        ),
    ]
    if csv_path is not None:
        parts.append(
            f'<p><a class="download" href="assets/{html.escape(csv_path.name, quote=True)}">Download complete source-packet table as CSV</a></p>'
        )
    parts.extend(relation_html)
    parts.extend(("<h3>Stored issues</h3>", issues_html(issue_records)))
    model = {
        "root_attributes": root_attributes,
        "run_provenance": run,
        "clock_reference": clock,
        "session_invariants": invariants,
        "constants": constants,
        "timing_and_identity": timing,
        "family_status": family_records,
        "issues": issue_records,
        "decoder": decoder,
        "product_row_count": product_rows,
        "product_row_histogram": product_histogram,
        "source_packets": source_packets,
        "relation_tables": relation_tables,
    }
    return "".join(parts), model


def inventory_html(
    datasets: Sequence[Mapping[str, object]],
    attributes: Sequence[Mapping[str, object]],
) -> str:
    def display_value(value: object) -> str:
        return value if isinstance(value, str) else json.dumps(
            value, sort_keys=True, default=str
        )

    dataset_rows = []
    for record in datasets:
        stats = record.get("statistics", {})
        if stats.get("numeric"):
            summary = (
                f"min {format_number(stats.get('minimum'))}; "
                f"median {format_number(stats.get('median'))}; "
                f"max {format_number(stats.get('maximum'))}; "
                f"finite {format_number(stats.get('finite_fraction'))}"
            )
        elif stats:
            summary = f"{stats.get('unique_count', 0)} unique"
        else:
            summary = "-"
        if record.get("sampled"):
            summary += f"; sampled {record.get('sample_count', 0):,}/{record.get('size', 0):,}"
        dataset_rows.append((
            record["path"],
            format_shape(record["shape"]),
            record["dtype"],
            record["size"],
            record["storage_bytes"],
            record.get("compression") or "none",
            summary,
            display_value(record.get("attributes", {})),
        ))
    attribute_rows = (
        (
            record["path"],
            record["name"],
            record["dtype"],
            format_shape(record["shape"]),
            display_value(record["value"]),
        )
        for record in attributes
    )
    return "".join((
        '<p class="note">Inventory statistics are exact for small datasets. Large datasets use a deterministic row sample and are labeled accordingly; family health plots still reduce every packet.</p>',
        "<h3>Every dataset</h3>",
        html_table(("path", "shape", "dtype", "values", "storage bytes", "compression", "statistics", "dataset attributes"), dataset_rows),
        "<h3>Every group/root attribute</h3>",
        html_table(("group", "attribute", "dtype", "shape", "value"), attribute_rows),
    ))


REPORT_SECTIONS = (
    ("spectra", "Normal spectra"),
    ("tr_spectra", "Time-resolved spectra"),
    ("grimm_spectra", "Grimm spectra"),
    ("telemetry", "Telemetry"),
    ("housekeeping", "Housekeeping"),
    ("zoom_spectra", "Zoom spectra"),
    ("waveform", "Waveform"),
    ("calibrator", "Calibrator"),
    ("provenance", "Provenance and issues"),
    ("inventory", "Dataset inventory"),
)


REPORT_CSS = """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #617086;
  --line: #dce3eb;
  --paper: #ffffff;
  --canvas: #f4f7fa;
  --nav: #10233f;
  --nav-ink: #e9f2ff;
  --accent: #0b6bcb;
  --good: #157347;
  --warn: #9a5b00;
  --bad: #b42318;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
a { color: var(--accent); }
.layout { display: grid; grid-template-columns: 245px minmax(0, 1fr); min-height: 100vh; }
.sidebar {
  position: sticky; top: 0; height: 100vh; overflow-y: auto;
  padding: 24px 18px; background: var(--nav); color: var(--nav-ink);
}
.sidebar .eyebrow { color: #8fc6ff; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; }
.sidebar h1 { margin: 6px 0 20px; font-size: 18px; overflow-wrap: anywhere; }
.sidebar nav a {
  display: block; padding: 7px 10px; border-radius: 6px; color: #d5e7fa;
  text-decoration: none; font-size: 13px;
}
.sidebar nav a:hover { background: #1d3e66; color: white; }
.sidebar .downloads { margin-top: 18px; padding-top: 14px; border-top: 1px solid #315173; }
.sidebar .downloads a { display: block; color: #a9d4ff; font-size: 12px; margin: 5px 0; }
main { min-width: 0; padding: 34px clamp(20px, 4vw, 60px) 80px; }
.hero {
  padding: 28px; border-radius: 14px; color: white;
  background: linear-gradient(125deg, #12345b, #0b6bcb);
  box-shadow: 0 8px 26px rgba(31, 55, 81, .15);
}
.hero h2 { margin: 0 0 7px; font-size: clamp(25px, 4vw, 38px); line-height: 1.15; }
.hero p { margin: 4px 0; color: #deefff; overflow-wrap: anywhere; }
.metric-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
  gap: 12px; margin: 22px 0;
}
.metric {
  min-width: 0; padding: 15px 16px; border: 1px solid var(--line);
  border-radius: 9px; background: var(--paper);
}
.metric span { display: block; color: var(--muted); font-size: 11px; letter-spacing: .07em; text-transform: uppercase; }
.metric strong { display: block; margin-top: 4px; font-size: 21px; overflow-wrap: anywhere; }
.report-section {
  scroll-margin-top: 20px; margin-top: 28px; padding: 24px;
  border: 1px solid var(--line); border-radius: 12px; background: var(--paper);
  box-shadow: 0 2px 9px rgba(28, 45, 66, .05);
}
.report-section > h2 { margin: 0 0 7px; font-size: 24px; }
.report-section > h3, .report-section h3 { margin: 28px 0 10px; font-size: 17px; }
.report-figure { margin: 22px 0 30px; }
.report-figure img {
  display: block; width: 100%; height: auto; border: 1px solid var(--line);
  border-radius: 8px; background: white;
}
.report-figure figcaption { padding-top: 8px; color: var(--muted); font-size: 13px; }
.table-wrap { width: 100%; overflow-x: auto; margin: 12px 0 24px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th {
  position: sticky; top: 0; z-index: 1; padding: 8px 10px; background: #eaf1f8;
  color: #23364d; text-align: left; white-space: nowrap;
}
.data-table td { padding: 7px 10px; border-bottom: 1px solid #e7ecf1; vertical-align: top; overflow-wrap: anywhere; }
.data-table tbody tr:nth-child(even) { background: #f8fafc; }
.note, .empty-state, .good-state, .error-state { padding: 13px 15px; border-radius: 7px; }
.note { background: #eef6ff; color: #31506f; }
.empty-state { background: #f1f3f5; color: #657180; }
.good-state { background: #e8f7ef; color: var(--good); }
.error-state { background: #fff0ef; color: var(--bad); white-space: pre-wrap; overflow-wrap: anywhere; }
.download { display: inline-block; padding: 8px 11px; border-radius: 6px; background: #e8f3ff; text-decoration: none; }
.status-pill { display: inline-block; padding: 3px 8px; border-radius: 99px; font-size: 12px; font-weight: 650; }
.status-good { color: #0b5733; background: #ccebdc; }
.status-bad { color: #8b1b13; background: #ffd8d4; }
details { margin: 12px 0; }
summary { cursor: pointer; color: var(--accent); font-weight: 600; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9em; }
@media (max-width: 850px) {
  .layout { display: block; }
  .sidebar { position: relative; height: auto; }
  .sidebar nav { columns: 2; }
  main { padding: 20px 12px 50px; }
  .hero, .report-section { padding: 18px; }
}
@media print {
  .layout { display: block; }
  .sidebar { display: none; }
  main { padding: 0; }
  .report-section { break-before: page; box-shadow: none; }
}
"""


def root_overview(h5, h5_path: Path) -> dict[str, object]:
    product_rows = (
        int(h5["/provenance/product_rows/family"].shape[0])
        if "/provenance/product_rows/family" in h5
        else 0
    )
    source_packets = (
        int(h5["/provenance/source_packets/role"].shape[0])
        if "/provenance/source_packets/role" in h5
        else 0
    )
    issues = int(h5["/issues"].attrs.get("count", 0)) if "/issues" in h5 else 0
    result = {
        "layout_version": int(h5.attrs.get("layout_version", -1)),
        "quality_status": decode_text(h5.attrs.get("quality_status", "unknown")),
        "execution_mode": decode_text(h5.attrs.get("execution_mode", "unknown")),
        "input_packet_count": int(h5.attrs.get("input_packet_count", 0)),
        "valid_packet_count": int(h5.attrs.get("valid_packet_count", 0)),
        "invalid_packet_count": max(
            0,
            int(h5.attrs.get("input_packet_count", 0))
            - int(h5.attrs.get("valid_packet_count", 0)),
        ),
        "product_row_count": product_rows,
        "source_packet_count": source_packets,
        "issue_count": issues,
        "file_bytes": int(h5_path.stat().st_size),
    }
    if "/status/families" in h5:
        group = h5["/status/families"]
        result["family_count"] = int(group["family"].shape[0])
        if "persisted_rows" in group:
            result["persisted_family_rows"] = int(
                np.sum(np.asarray(group["persisted_rows"], dtype=np.int64))
            )
    return result


def validation_model(h5_path: Path) -> tuple[object | None, dict[str, object]]:
    caught = []
    try:
        from lusee.ingest.constants import INGEST_LAYOUT_VERSION
        from lusee.ingest.obs_factory import load_bundle

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            bundle = load_bundle(h5_path)
        caught = [str(record.message) for record in records]
        if bundle.layout_version != INGEST_LAYOUT_VERSION:
            raise ValueError(
                f"expected layout v{INGEST_LAYOUT_VERSION}, got "
                f"{bundle.layout_version}"
            )
        return bundle, {
            "status": "valid",
            "layout_version": int(bundle.layout_version),
            "warnings": caught,
        }
    except Exception as exc:  # noqa: BLE001 - validation failures belong in the report
        return None, {
            "status": "invalid",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "warnings": caught,
        }


def metric_cards(overview: Mapping[str, object], validation: Mapping[str, object]) -> str:
    cards = (
        ("layout", f"v{overview.get('layout_version', '?')}"),
        ("reader validation", validation.get("status", "unknown")),
        ("quality", overview.get("quality_status", "unknown")),
        ("input packets", overview.get("input_packet_count", 0)),
        ("valid packets", overview.get("valid_packet_count", 0)),
        ("product rows", overview.get("product_row_count", 0)),
        ("source provenance links", overview.get("source_packet_count", 0)),
        ("stored issues", overview.get("issue_count", 0)),
        ("HDF5 size", f"{int(overview.get('file_bytes', 0)) / (1024 * 1024):.2f} MiB"),
    )
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric"><span>{html.escape(str(label))}</span>'
        f'<strong>{escaped(value)}</strong></div>'
        for label, value in cards
    ) + "</div>"


def document_html(
    *,
    h5_name: str,
    digest: str,
    overview: Mapping[str, object],
    validation: Mapping[str, object],
    section_html: Mapping[str, str],
    source_csv_present: bool,
) -> str:
    nav = "".join(
        f'<a href="#{name}">{html.escape(title)}</a>'
        for name, title in REPORT_SECTIONS
    )
    validation_class = (
        "status-good" if validation.get("status") == "valid" else "status-bad"
    )
    validation_detail = ""
    if validation.get("status") != "valid":
        validation_detail = (
            '<div class="error-state"><strong>Reader validation failed.</strong> '
            + html.escape(str(validation.get("error", "unknown failure")))
            + "</div>"
        )
    elif validation.get("warnings"):
        validation_detail = (
            '<details><summary>Reader warnings</summary>'
            + html_table(
                ("#", "warning"),
                enumerate(validation.get("warnings", ()), start=1),
            )
            + "</details>"
        )
    sections = "".join(
        f'<section class="report-section" id="{name}" data-section="{name}">'
        f"<h2>{html.escape(title)}</h2>{section_html.get(name, '')}</section>"
        for name, title in REPORT_SECTIONS
    )
    source_csv_link = (
        '<a href="assets/source_packets.csv">Source-packet CSV</a>'
        if source_csv_present
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>LuSEE ingest report - {html.escape(h5_name)}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <div class="eyebrow">LuSEE ingest QA</div>
  <h1>{html.escape(h5_name)}</h1>
  <nav><a href="#overview">Overview</a>{nav}</nav>
  <div class="downloads">
    <a href="summary.json">Machine-readable summary</a>
    {source_csv_link}
  </div>
</aside>
<main>
  <header class="hero" id="overview">
    <h2>Ingest science-product report</h2>
    <p>{html.escape(h5_name)}</p>
    <p><code>SHA-256 {html.escape(digest)}</code></p>
    <p><span class="status-pill {validation_class}">reader {html.escape(str(validation.get('status', 'unknown')))}</span></p>
  </header>
  {metric_cards(overview, validation)}
  {validation_detail}
  {sections}
</main>
</div>
</body>
</html>
"""


def section_failure(name: str, exc: Exception) -> tuple[str, dict[str, object], dict[str, object]]:
    message = f"{name} report generation failed: {type(exc).__name__}: {exc}"
    return f'<div class="error-state">{html.escape(message)}</div>', {
        "present": None,
        "report_error_type": type(exc).__name__,
        "report_error": str(exc),
    }, {}


def generate_report(
    h5_path: Path | str,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> ReportResult:
    """Generate one portable static report from one layout-v4 HDF5 file."""
    h5_path = Path(h5_path).resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    requested_output = Path(output_dir).resolve()
    if requested_output == h5_path or requested_output in h5_path.parents:
        raise ValueError("report output must not contain or replace the input HDF5")
    output_dir = prepare_output_dir(requested_output, overwrite)
    assets_dir = output_dir / "assets"
    assets: list[Path] = []
    digest = sha256_file(h5_path)
    bundle, validation = validation_model(h5_path)

    h5py = import_h5py()
    with h5py.File(h5_path, "r") as h5:
        overview = root_overview(h5, h5_path)
        datasets = dataset_inventory(h5)
        group_attributes = group_attribute_inventory(h5)

        section_html: dict[str, str] = {}
        section_models: dict[str, object] = {}
        selections: dict[str, object] = {}
        builders = (
            ("spectra", "normal spectra", normal_spectra_section),
            ("tr_spectra", "time-resolved spectra", tr_spectra_section),
            ("grimm_spectra", "Grimm spectra", grimm_section),
            ("telemetry", "telemetry", telemetry_section),
            ("housekeeping", "housekeeping", housekeeping_section),
            ("zoom_spectra", "zoom spectra", zoom_section),
            ("waveform", "waveform", waveform_section),
            ("calibrator", "calibrator", calibrator_section),
        )
        for key, label, builder in builders:
            if bundle is None:
                body, model, selected = (
                    empty_section("Unavailable because layout-v4 reader validation failed."),
                    {"present": None, "reason": "reader_validation_failed"},
                    {},
                )
            else:
                try:
                    body, model, selected = builder(bundle, assets_dir, assets)
                    if key == "spectra":
                        metadata_html, metadata_model = spectrum_metadata_html(
                            bundle, assets_dir, assets
                        )
                        body += metadata_html
                        model = dict(model)
                        model["metadata_fields"] = metadata_model
                    elif key == "tr_spectra":
                        row_count = int(model.get("row_count", 0))
                        tr_metadata = field_columns_model(
                            bundle.tr_metadata, None, row_count
                        )
                        body += "<h3>Complete time-resolved metadata</h3>"
                        body += (
                            field_columns_table(tr_metadata)
                            if tr_metadata
                            else empty_section("Time-resolved metadata are absent.")
                        )
                        model = dict(model)
                        model["metadata_fields"] = tr_metadata
                except Exception as exc:  # noqa: BLE001 - keep other family panels
                    body, model, selected = section_failure(label, exc)
            section_html[key] = body
            section_models[key] = model
            selections[key] = selected

        try:
            body, model = provenance_section(h5, assets_dir, assets, bundle)
        except Exception as exc:  # noqa: BLE001 - inventory remains useful
            body, model, _ = section_failure("provenance", exc)
        section_html["provenance"] = body
        section_models["provenance"] = model
        section_html["inventory"] = inventory_html(datasets, group_attributes)
        section_models["inventory"] = {
            "dataset_count": len(datasets),
            "group_attribute_count": len(group_attributes),
            "sampled_dataset_count": sum(bool(item.get("sampled")) for item in datasets),
        }

    plot_assets = tuple(path for path in assets if path.suffix.lower() == ".png")
    download_assets = tuple(path for path in assets if path.suffix.lower() == ".csv")
    source_csv_present = (assets_dir / "source_packets.csv").is_file()
    summary = json_value({
        "report_format_version": REPORT_FORMAT_VERSION,
        "hdf5": {
            "name": h5_path.name,
            "sha256": digest,
            "size_bytes": h5_path.stat().st_size,
        },
        "validation": validation,
        "overview": overview,
        "sections": section_models,
        "selections": selections,
        "datasets": datasets,
        "group_attributes": group_attributes,
        "plots": [path.relative_to(output_dir).as_posix() for path in plot_assets],
        "downloads": [
            path.relative_to(output_dir).as_posix() for path in download_assets
        ],
    })
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    index_path = output_dir / "index.html"
    index_path.write_text(
        document_html(
            h5_name=h5_path.name,
            digest=digest,
            overview=overview,
            validation=validation,
            section_html=section_html,
            source_csv_present=source_csv_present,
        ),
        encoding="utf-8",
    )
    return ReportResult(
        index_path=index_path,
        summary_path=summary_path,
        asset_paths=plot_assets,
        summary=summary,
    )


def cdi_tree_id(cdi_dir: Path | str) -> str:
    """Return the centralized-corpus identity for one extracted CDI tree."""
    cdi_dir = Path(cdi_dir).resolve()
    if not cdi_dir.is_dir():
        raise NotADirectoryError(cdi_dir)
    entries = tuple(cdi_dir.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".bin"
        for path in entries
    ):
        raise ValueError(f"CDI payload has a non-packet entry: {cdi_dir}")
    paths = sorted(entries, key=lambda path: path.name)
    digest = hashlib.sha256(b"lusee-cdi-tree-v1\0")
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        packet_digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                packet_digest.update(block)
        digest.update(packet_digest.digest())
    return f"sha256:{digest.hexdigest()}"


def raw_tree_id(raw_dir: Path | str) -> str:
    """Return the centralized-corpus identity for one raw FLASH tree."""
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    digest = hashlib.sha256(b"lusee-ccsds-raw-tree-v1\0")
    entries = sorted(
        raw_dir.rglob("*"),
        key=lambda path: path.relative_to(raw_dir).as_posix(),
    )
    for path in entries:
        relative = path.relative_to(raw_dir).as_posix().encode("utf-8")
        if path.is_symlink():
            raise ValueError(f"raw payload has a symbolic link: {path}")
        if path.is_dir():
            digest.update(b"D\0")
            digest.update(relative)
            digest.update(b"\0")
        elif path.is_file():
            digest.update(b"F\0")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            file_digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    file_digest.update(block)
            digest.update(file_digest.digest())
        else:
            raise ValueError(f"raw payload has an unsupported entry: {path}")
    return f"sha256:{digest.hexdigest()}"


def read_json_object(path: Path, label: str) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def corpus_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the corpus")
    resolved = (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} resolves outside the corpus")
    return resolved


def telemetry_sidecar_map(root: Path) -> dict[str, Path]:
    manifest_path = (
        root / "telemetry_sidecars" / "telemetry_sidecars_manifest.json"
    )
    if not manifest_path.is_file():
        return {}
    manifest = read_json_object(manifest_path, "telemetry sidecar manifest")
    if int(manifest.get("manifest_version", -1)) != 1:
        raise ValueError("unsupported telemetry sidecar manifest version")
    result: dict[str, Path] = {}
    entries = manifest.get("entries", ())
    if not isinstance(entries, list):
        raise TypeError("telemetry sidecar manifest entries must be a list")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"telemetry sidecar entry {index} must be an object")
        tree_id = entry.get("tree_id")
        directory = entry.get("tree_directory")
        if not isinstance(tree_id, str) or not isinstance(directory, str):
            raise TypeError(f"telemetry sidecar entry {index} is incomplete")
        if Path(directory).name != directory or directory in (".", ".."):
            raise ValueError(f"telemetry sidecar entry {index} has an unsafe directory")
        if tree_id in result:
            raise ValueError(f"duplicate telemetry sidecar tree_id: {tree_id}")
        path = (
            root / "telemetry_sidecars" / directory / "DCB_telemetry.json"
        ).resolve()
        sidecar_root = (root / "telemetry_sidecars").resolve()
        if sidecar_root not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        if "size_bytes" not in entry or "sha256" not in entry:
            raise ValueError(f"telemetry sidecar checksum is missing for {tree_id}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"telemetry sidecar size disagrees for {tree_id}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"telemetry sidecar digest disagrees for {tree_id}")
        result[tree_id] = path
    return result


def read_corpus_targets(
    corpus_root: Path | str,
    kind: Literal["ccsds", "cdi"],
) -> tuple[CorpusTarget, ...]:
    """Read manifest-selected corpus trees without rescanning private sources."""
    if kind not in ("ccsds", "cdi"):
        raise ValueError("corpus kind must be 'ccsds' or 'cdi'")
    root = Path(corpus_root).resolve()
    manifest_path = root / "corpus_manifest.json"
    manifest = read_json_object(manifest_path, f"{kind} corpus manifest")
    if int(manifest.get("manifest_version", -1)) != 1:
        raise ValueError(f"unsupported corpus manifest version: {manifest_path}")
    entries = manifest.get("trees")
    if not isinstance(entries, list):
        raise TypeError(f"corpus manifest trees must be a list: {manifest_path}")
    sidecars = telemetry_sidecar_map(root) if kind == "cdi" else {}
    input_key = "raw_flash" if kind == "ccsds" else "cdi_output"
    targets = []
    seen = set()
    seen_inputs = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"corpus tree {index} must be an object")
        tree_id = entry.get("tree_id")
        digest_text = (
            tree_id.removeprefix("sha256:") if isinstance(tree_id, str) else ""
        )
        try:
            digest_is_hex = len(digest_text) == 64 and int(digest_text, 16) >= 0
        except ValueError:
            digest_is_hex = False
        if not digest_is_hex:
            raise ValueError(f"corpus tree {index} has no valid tree_id")
        if tree_id in seen:
            raise ValueError(f"duplicate corpus tree_id: {tree_id}")
        seen.add(tree_id)
        tree_manifest = corpus_path(
            root, entry.get("manifest"), f"trees[{index}].manifest"
        )
        input_dir = corpus_path(
            root, entry.get(input_key), f"trees[{index}].{input_key}"
        )
        if not tree_manifest.is_file():
            raise FileNotFoundError(tree_manifest)
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        expected_input = (tree_manifest.parent / input_key).resolve()
        if input_dir != expected_input:
            raise ValueError(f"corpus payload path disagrees for {tree_id}")
        if input_dir in seen_inputs:
            raise ValueError(f"duplicate corpus payload path: {input_dir}")
        seen_inputs.add(input_dir)
        tree_record = read_json_object(tree_manifest, f"corpus tree {tree_id}")
        expected_algorithm = (
            "lusee-ccsds-raw-tree-v1" if kind == "ccsds" else "lusee-cdi-tree-v1"
        )
        if (
            int(tree_record.get("manifest_schema_version", -1)) != 1
            or tree_record.get("tree_id") != tree_id
            or tree_record.get("tree_id_algorithm") != expected_algorithm
            or tree_record.get("payload_path") != input_key
        ):
            raise ValueError(f"per-tree manifest disagrees for {tree_id}")
        computed_tree_id = (
            raw_tree_id(input_dir) if kind == "ccsds" else cdi_tree_id(input_dir)
        )
        if computed_tree_id != tree_id:
            raise ValueError(f"{kind.upper()} payload identity disagrees for {tree_id}")
        targets.append(CorpusTarget(
            kind=kind,
            tree_id=tree_id,
            tree_dir=tree_manifest.parent,
            input_dir=input_dir,
            telemetry_sidecar=sidecars.get(tree_id),
        ))
    return tuple(sorted(targets, key=lambda target: target.tree_id))


def select_cdi_targets(
    targets: Sequence[CorpusTarget],
    derived_tree_ids: Iterable[str],
) -> tuple[tuple[CorpusTarget, ...], tuple[CorpusTarget, ...]]:
    """Split standalone CDI trees by exact identity after the raw pass."""
    derived = set(derived_tree_ids)
    ordered = sorted(targets, key=lambda target: target.tree_id)
    selected = tuple(target for target in ordered if target.tree_id not in derived)
    skipped = tuple(target for target in ordered if target.tree_id in derived)
    return selected, skipped


def prepare_tree_output(target: CorpusTarget, report_subdir: str, overwrite: bool) -> Path:
    output_dir = target.tree_dir / report_subdir
    input_dir = target.input_dir.resolve()
    resolved_output = output_dir.resolve()
    if (
        resolved_output == input_dir
        or resolved_output in input_dir.parents
        or input_dir in resolved_output.parents
    ):
        raise ValueError("tree report output overlaps the ingestion input")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"tree report already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    return output_dir


def relative_link(page: Path, target: Path) -> str:
    return Path(os.path.relpath(target, page.parent)).as_posix()


def tree_index_html(
    *,
    target: CorpusTarget,
    reports: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, str]],
    index_path: Path,
) -> str:
    report_rows = []
    for record in reports:
        report = record["report"]
        assert isinstance(report, ReportResult)
        report_rows.append((
            record.get("session_name"),
            record.get("hdf5_name"),
            record.get("derived_cdi_tree_id"),
            f'<a href="{html.escape(relative_link(index_path, report.index_path), quote=True)}">open report</a>',
        ))
    if report_rows:
        rows_html = []
        for session_name, hdf5_name, derived_id, link in report_rows:
            rows_html.append(
                "<tr>"
                f"<td>{escaped(session_name)}</td>"
                f"<td>{escaped(hdf5_name)}</td>"
                f"<td>{escaped(derived_id)}</td>"
                f"<td>{link}</td>"
                "</tr>"
            )
        report_table = (
            '<div class="table-wrap"><table class="data-table"><thead><tr>'
            "<th>session</th><th>HDF5</th><th>derived CDI identity</th><th>report</th>"
            f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
        )
    else:
        report_table = empty_section("No final HDF5 report was produced.")
    failures_html = ""
    if failures:
        failures_html = (
            "<h2>Failures</h2>"
            + html_table(
                ("stage", "error type", "message"),
                (
                    (
                        failure.get("stage"),
                        failure.get("error_type"),
                        failure.get("error"),
                    )
                    for failure in failures
                ),
            )
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LuSEE corpus tree - {html.escape(target.tree_id)}</title><style>{REPORT_CSS}</style></head>
<body><main><header class="hero"><h2>{html.escape(target.kind.upper())} corpus tree</h2>
<p><code>{html.escape(target.tree_id)}</code></p></header>
<section class="report-section"><h2>Final HDF5 reports</h2>{report_table}{failures_html}</section>
</main></body></html>
"""


def write_tree_index(
    output_dir: Path,
    target: CorpusTarget,
    reports: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, str]],
) -> Path:
    index_path = output_dir / "index.html"
    index_path.write_text(
        tree_index_html(
            target=target,
            reports=reports,
            failures=failures,
            index_path=index_path,
        ),
        encoding="utf-8",
    )
    summary_records = []
    for record in reports:
        report = record["report"]
        assert isinstance(report, ReportResult)
        summary_records.append({
            "session_name": record.get("session_name"),
            "session_ordinal": record.get("session_ordinal"),
            "hdf5_name": record.get("hdf5_name"),
            "hdf5_sha256": report.summary.get("hdf5", {}).get("sha256"),
            "derived_cdi_tree_id": record.get("derived_cdi_tree_id"),
            "flash_result_id": record.get("flash_result_id"),
            "flash_input_identity_sha256": record.get(
                "flash_input_identity_sha256"
            ),
            "report": relative_link(output_dir / "tree_summary.json", report.index_path),
        })
    (output_dir / "tree_summary.json").write_text(
        json.dumps(
            json_value({
                "tree_id": target.tree_id,
                "kind": target.kind,
                "reports": summary_records,
                "failures": failures,
            }),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def process_ccsds_target(
    target: CorpusTarget,
    landing_time_file: Path,
    report_subdir: str,
    overwrite: bool,
) -> tuple[
    list[Mapping[str, object]],
    list[str],
    list[Mapping[str, str]],
]:
    output_dir = prepare_tree_output(target, report_subdir, overwrite)
    reports: list[Mapping[str, object]] = []
    derived_ids: list[str] = []
    failures: list[Mapping[str, str]] = []
    try:
        from lusee.ingest.pipeline import process_flash

        result = process_flash(
            target.input_dir,
            landing_time_file=landing_time_file,
            sessions_root=output_dir / "sessions",
            h5_dir=output_dir / "hdf5",
            manifest_dir=output_dir / "manifests",
        )
        for session in result.session_results:
            if session.h5_path is None or session.session_dir is None:
                failures.append({
                    "stage": f"session_output:{session.session_name}",
                    "error_type": "ValueError",
                    "error": "ingestion did not produce both HDF5 and session paths",
                })
                continue
            try:
                derived_id = cdi_tree_id(Path(session.session_dir) / "cdi_output")
            except Exception as exc:  # noqa: BLE001 - continue with later sessions
                failures.append({
                    "stage": f"cdi_identity:{session.session_name}",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            derived_ids.append(derived_id)
            try:
                generated = generate_report(
                    session.h5_path,
                    output_dir / "reports" / session.session_name,
                )
                reports.append({
                    "session_name": session.session_name,
                    "session_ordinal": session.session_ordinal,
                    "hdf5_name": Path(session.h5_path).name,
                    "derived_cdi_tree_id": derived_id,
                    "flash_result_id": result.flash_result_id,
                    "flash_input_identity_sha256": result.input_identity_sha256,
                    "report": generated,
                })
            except Exception as exc:  # noqa: BLE001 - continue with later sessions
                artifacts = getattr(session, "output_artifacts", {})
                hdf5_artifact = (
                    artifacts.get("hdf5", {})
                    if isinstance(artifacts, Mapping)
                    else {}
                )
                failures.append({
                    "stage": f"report:{session.session_name}",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "session_name": session.session_name,
                    "session_ordinal": session.session_ordinal,
                    "hdf5_name": Path(session.h5_path).name,
                    "hdf5_sha256": hdf5_artifact.get("sha256"),
                    "derived_cdi_tree_id": derived_id,
                    "flash_result_id": result.flash_result_id,
                    "flash_input_identity_sha256": result.input_identity_sha256,
                })
    except Exception as exc:  # noqa: BLE001 - preserve a per-tree failure page
        failures.append({
            "stage": "process_flash",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
    write_tree_index(output_dir, target, reports, failures)
    return reports, derived_ids, failures


def process_cdi_target(
    target: CorpusTarget,
    landing_time_file: Path,
    report_subdir: str,
    overwrite: bool,
) -> tuple[list[Mapping[str, object]], list[Mapping[str, str]]]:
    output_dir = prepare_tree_output(target, report_subdir, overwrite)
    reports: list[Mapping[str, object]] = []
    failures: list[Mapping[str, str]] = []
    try:
        from lusee.ingest.pipeline import process_session

        session_dir = output_dir / "session_input"
        session_dir.mkdir()
        (session_dir / "cdi_output").symlink_to(target.input_dir, target_is_directory=True)
        if target.telemetry_sidecar is not None:
            (session_dir / "DCB_telemetry.json").symlink_to(target.telemetry_sidecar)
        result = process_session(
            session_dir,
            landing_time_file=landing_time_file,
            h5_dir=output_dir / "hdf5",
            manifest_dir=output_dir / "manifests",
            name="session",
        )
        if result.h5_path is None:
            raise ValueError("ingestion did not produce an HDF5 file")
    except Exception as exc:  # noqa: BLE001 - preserve a per-tree failure page
        failures.append({
            "stage": "process_session",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
    else:
        try:
            generated = generate_report(
                result.h5_path,
                output_dir / "reports" / result.session_name,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the ingested HDF5
            failures.append({
                "stage": f"report:{result.session_name}",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
        else:
            reports.append({
                "session_name": result.session_name,
                "session_ordinal": result.session_ordinal,
                "hdf5_name": Path(result.h5_path).name,
                "derived_cdi_tree_id": target.tree_id,
                "report": generated,
            })
    write_tree_index(output_dir, target, reports, failures)
    return reports, failures


def corpus_index_html(
    *,
    kind: str,
    records: Sequence[Mapping[str, object]],
    skipped: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, str]],
    index_path: Path,
) -> str:
    rows = []
    for record in sorted(records, key=lambda item: (str(item["tree_id"]), str(item.get("session_name")))):
        report = record["report"]
        assert isinstance(report, ReportResult)
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(record['tree_id']))}</code></td>"
            f"<td>{escaped(record.get('session_name'))}</td>"
            f"<td>{escaped(record.get('hdf5_name'))}</td>"
            f'<td><a href="{html.escape(relative_link(index_path, report.index_path), quote=True)}">open report</a></td>'
            "</tr>"
        )
    table = (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        "<th>tree identity</th><th>session</th><th>HDF5</th><th>report</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        if rows else empty_section("No HDF5 reports were produced.")
    )
    skipped_html = ""
    if skipped:
        skipped_html = (
            '<section class="report-section"><h2>Standalone CDI trees suppressed</h2>'
            '<p class="note">These exact CDI identities were generated during the CCSDS pass. Their standalone copies were not ingested again.</p>'
            + html_table(
                ("CDI tree identity", "standalone sidecar", "covered by raw session(s)"),
                (
                    (
                        record["tree_id"],
                        record["telemetry_sidecar_present"],
                        ", ".join(
                            f"{item.get('ccsds_tree_id')} / {item.get('session_name')}"
                            for item in record.get("covered_by", ())
                        ) or "raw HDF5 report failed; rerun corpus mode",
                    )
                    for record in skipped
                ),
            )
            + "</section>"
        )
    failure_html = ""
    if failures:
        failure_html = (
            '<section class="report-section"><h2>Failed targets or sessions</h2>'
            + html_table(
                ("kind", "tree identity", "stage", "error type", "message"),
                (
                    (
                        record.get("kind"),
                        record.get("tree_id"),
                        record.get("stage"),
                        record.get("error_type"),
                        record.get("error"),
                    )
                    for record in failures
                ),
            )
            + "</section>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LuSEE {html.escape(kind)} corpus reports</title><style>{REPORT_CSS}</style></head>
<body><main><header class="hero"><h2>LuSEE {html.escape(kind.upper())} corpus reports</h2>
<p>{len(records):,} final HDF5 report(s); {len(failures):,} failure(s)</p></header>
<section class="report-section"><h2>Final HDF5 reports</h2>{table}</section>{skipped_html}{failure_html}
</main></body></html>
"""


def write_corpus_index(
    corpus_root: Path,
    kind: str,
    report_subdir: str,
    records: Sequence[Mapping[str, object]],
    skipped: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, str]],
    overwrite: bool,
) -> Path:
    output_dir = corpus_root / report_subdir
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"corpus index already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    index_path = output_dir / "index.html"
    index_path.write_text(
        corpus_index_html(
            kind=kind,
            records=records,
            skipped=skipped,
            failures=failures,
            index_path=index_path,
        ),
        encoding="utf-8",
    )
    serializable_records = []
    for record in records:
        report = record["report"]
        assert isinstance(report, ReportResult)
        serializable_records.append({
            "tree_id": record.get("tree_id"),
            "session_name": record.get("session_name"),
            "session_ordinal": record.get("session_ordinal"),
            "hdf5_name": record.get("hdf5_name"),
            "hdf5_sha256": report.summary.get("hdf5", {}).get("sha256"),
            "derived_cdi_tree_id": record.get("derived_cdi_tree_id"),
            "flash_result_id": record.get("flash_result_id"),
            "flash_input_identity_sha256": record.get(
                "flash_input_identity_sha256"
            ),
            "report": relative_link(output_dir / "summary.json", report.index_path),
        })
    skipped_ids = sorted(str(record["tree_id"]) for record in skipped)
    (output_dir / "summary.json").write_text(
        json.dumps(
            json_value({
                "kind": kind,
                "reports": serializable_records,
                "skipped_cdi_tree_ids": skipped_ids,
                "suppressed_cdi_trees": skipped,
                "failures": failures,
            }),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def generate_corpus_reports(
    *,
    ccsds_corpus: Path | str,
    cdi_corpus: Path | str,
    landing_time_file: Path | str,
    report_subdir: str = DEFAULT_REPORT_SUBDIR,
    overwrite: bool = False,
) -> CorpusRunResult:
    """Ingest raw trees first, suppress derived CDI copies, and report HDF5s."""
    if report_subdir != DEFAULT_REPORT_SUBDIR:
        raise ValueError(f"report_subdir is fixed to {DEFAULT_REPORT_SUBDIR!r}")
    ccsds_root = Path(ccsds_corpus).resolve()
    cdi_root = Path(cdi_corpus).resolve()
    landing = Path(landing_time_file).resolve()
    if not landing.is_file():
        raise FileNotFoundError(landing)
    ccsds_targets = read_corpus_targets(ccsds_root, "ccsds")
    cdi_targets = read_corpus_targets(cdi_root, "cdi")
    if not overwrite:
        existing = [
            path
            for path in (
                ccsds_root / report_subdir,
                cdi_root / report_subdir,
                *(target.tree_dir / report_subdir for target in ccsds_targets),
                *(target.tree_dir / report_subdir for target in cdi_targets),
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "corpus report output already exists; use --overwrite: "
                + ", ".join(str(path) for path in existing)
            )

    raw_records: list[Mapping[str, object]] = []
    cdi_records: list[Mapping[str, object]] = []
    failures: list[Mapping[str, str]] = []
    derived: list[str] = []
    report_results: list[ReportResult] = []
    for target in ccsds_targets:
        try:
            target_reports, target_derived, target_failures = process_ccsds_target(
                target, landing, report_subdir, overwrite
            )
        except Exception as exc:  # noqa: BLE001 - continue with later trees
            target_reports = []
            target_derived = []
            target_failures = [{
                "stage": "prepare_target",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }]
        for record in target_reports:
            enriched = {"tree_id": target.tree_id, **record}
            raw_records.append(enriched)
            report_results.append(record["report"])
        derived.extend(target_derived)
        failures.extend(
            {"kind": "ccsds", "tree_id": target.tree_id, **failure}
            for failure in target_failures
        )

    selected, skipped = select_cdi_targets(cdi_targets, derived)
    if overwrite:
        for target in skipped:
            stale_report = target.tree_dir / report_subdir
            if stale_report.exists():
                shutil.rmtree(stale_report)
    for target in selected:
        try:
            target_reports, target_failures = process_cdi_target(
                target, landing, report_subdir, overwrite
            )
        except Exception as exc:  # noqa: BLE001 - continue with later trees
            target_reports = []
            target_failures = [{
                "stage": "prepare_target",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }]
        for record in target_reports:
            enriched = {"tree_id": target.tree_id, **record}
            cdi_records.append(enriched)
            report_results.append(record["report"])
        failures.extend(
            {"kind": "cdi", "tree_id": target.tree_id, **failure}
            for failure in target_failures
        )

    skipped_ids = tuple(target.tree_id for target in skipped)
    skipped_records = []
    for target in skipped:
        covered_by = [
            {
                "ccsds_tree_id": record["tree_id"],
                "session_name": record.get("session_name"),
                "session_ordinal": record.get("session_ordinal"),
                "hdf5_name": record.get("hdf5_name"),
            }
            for record in raw_records
            if record.get("derived_cdi_tree_id") == target.tree_id
        ]
        covered_by.extend(
            {
                "ccsds_tree_id": failure.get("tree_id"),
                "session_name": failure.get("session_name"),
                "session_ordinal": failure.get("session_ordinal"),
                "hdf5_name": failure.get("hdf5_name"),
                "hdf5_sha256": failure.get("hdf5_sha256"),
                "report_status": "failed",
            }
            for failure in failures
            if (
                failure.get("kind") == "ccsds"
                and failure.get("derived_cdi_tree_id") == target.tree_id
            )
        )
        skipped_records.append({
            "tree_id": target.tree_id,
            "telemetry_sidecar_present": target.telemetry_sidecar is not None,
            "precedence": "raw_ccsds",
            "covered_by": covered_by,
        })
    write_corpus_index(
        ccsds_root,
        "ccsds",
        report_subdir,
        raw_records,
        (),
        tuple(failure for failure in failures if failure.get("kind") == "ccsds"),
        overwrite,
    )
    write_corpus_index(
        cdi_root,
        "cdi",
        report_subdir,
        cdi_records,
        skipped_records,
        tuple(failure for failure in failures if failure.get("kind") == "cdi"),
        overwrite,
    )
    return CorpusRunResult(
        reports=tuple(report_results),
        derived_cdi_tree_ids=tuple(sorted(set(derived))),
        skipped_cdi_tree_ids=skipped_ids,
        failures=tuple(failures),
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate static human-QA pages for ingest HDF5 products"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    hdf5_parser = subparsers.add_parser(
        "hdf5", help="report one or more existing HDF5 files"
    )
    hdf5_parser.add_argument("paths", type=Path, nargs="+")
    hdf5_parser.add_argument("--output-dir", type=Path)
    hdf5_parser.add_argument("--overwrite", action="store_true")

    corpus_parser = subparsers.add_parser(
        "corpus", help="ingest and report the centralized CCSDS/CDI corpora"
    )
    corpus_parser.add_argument("--ccsds-corpus", type=Path, required=True)
    corpus_parser.add_argument("--cdi-corpus", type=Path, required=True)
    corpus_parser.add_argument("--landing-time-file", type=Path, required=True)
    corpus_parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.command == "hdf5":
        if args.output_dir is not None and len(args.paths) != 1:
            raise SystemExit("--output-dir requires exactly one HDF5 path")
        for path in args.paths:
            output_dir = (
                args.output_dir
                if args.output_dir is not None
                else path.resolve().parent / DEFAULT_REPORT_SUBDIR / path.stem
            )
            result = generate_report(path, output_dir, overwrite=args.overwrite)
            print(result.index_path)
        return 0
    result = generate_corpus_reports(
        ccsds_corpus=args.ccsds_corpus,
        cdi_corpus=args.cdi_corpus,
        landing_time_file=args.landing_time_file,
        overwrite=args.overwrite,
    )
    print(f"reports: {len(result.reports)}")
    print(f"standalone CDI trees skipped: {len(result.skipped_cdi_tree_ids)}")
    print(f"failures: {len(result.failures)}")
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
