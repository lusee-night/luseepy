"""Per-session sanity-check plots.

Each plotting function takes a validated ingest bundle or HDF5/FITS path,
plus an output PNG path. They render quickly and are designed for
end-of-pipeline visual sanity checks; they are not publication graphics.

Standalone use::

    from lusee.ingest import viz
    viz.plot_session("session_001.h5", "out/plots/")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .constants import INGEST_LAYOUT_VERSION
from .obs_factory import LegacyIngestWarning, SessionBundle, load_bundle

log = logging.getLogger(__name__)


_AVAILABLE_PLOTS = (
    "spectra_waterfall",
    "spectra_mean",
    "adc_stats",
    "dcb_telemetry",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_bundle(source) -> SessionBundle:
    if isinstance(source, SessionBundle):
        return source
    filename = getattr(source, "filename", None)
    return load_bundle(filename if filename is not None else source)


def _normal_groups(source) -> tuple[SessionBundle, ...]:
    bundle = _as_bundle(source)
    if bundle.spectra is None:
        raise FileNotFoundError("normal spectra are absent")
    if bundle.layout_version in (2, 3):
        warnings.warn(
            "layout-v2/v3 plots use legacy_unverified frequency-bin metadata",
            LegacyIngestWarning,
            stacklevel=3,
        )
    groups = bundle.split_by_frequency_grid()
    return groups or (bundle,)


def _normal_cube(bundle: SessionBundle) -> np.ndarray:
    spectra = np.asarray(bundle.spectra)
    if spectra.shape[0] == 0:
        return spectra
    if bundle.spectra_frequency_counts is None:
        if bundle.layout_version not in (2, 3):
            raise ValueError("normal-spectrum frequency counts are absent")
        meaningful_count = bundle.frequency_window_for_row(0).output_count
        if spectra.shape[2] < meaningful_count:
            raise ValueError(
                "legacy spectrum backing width is smaller than its Navgf window"
            )
        counts = np.full(
            spectra.shape[0],
            meaningful_count,
            dtype=np.int64,
        )
    else:
        counts = np.asarray(bundle.spectra_frequency_counts)
    if counts.size == 0 or np.unique(counts).size != 1:
        raise ValueError("plotting requires one homogeneous frequency window")
    return spectra[:, :, : int(counts[0])]


def _grid_tag(bundle: SessionBundle, include: bool) -> str:
    if not include:
        return ""
    navgf = bundle.frequency_window_for_row(0).navgf
    return f"_navgf{navgf}"


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for lusee.ingest.viz; install matplotlib"
        ) from exc


_PRODUCT_NAMES = {
    0: "Ch0 Auto", 1: "Ch1 Auto", 2: "Ch2 Auto", 3: "Ch3 Auto",
    4: "Ch0xCh1 Re", 5: "Ch0xCh1 Im", 6: "Ch0xCh2 Re", 7: "Ch0xCh2 Im",
    8: "Ch0xCh3 Re", 9: "Ch0xCh3 Im", 10: "Ch1xCh2 Re", 11: "Ch1xCh2 Im",
    12: "Ch1xCh3 Re", 13: "Ch1xCh3 Im", 14: "Ch2xCh3 Re", 15: "Ch2xCh3 Im",
}


# ---------------------------------------------------------------------------
# Plot: per-product waterfall + mean (one figure per product)
# ---------------------------------------------------------------------------

def _plot_one_waterfall(plt, spectra: np.ndarray, product: int, out_path: Path,
                        title_suffix: str = "") -> Path:
    """Single product waterfall + mean spectrum panel.

    Mirrors the legacy ``receive/plot_spectra.py:plot_waterfall`` layout:
    autocorrelation products (0..3) are rendered on a viridis colormap as
    log10(power) with non-positive values masked to NaN; cross-correlation
    products (4..15) on a diverging RdBu_r colormap on the linear scale.
    Color range is autoscaled per-product, not shared.
    """
    n_time, n_products, n_freq = spectra.shape
    if product >= n_products:
        return out_path
    data = spectra[:, product, :]
    if product < 4:
        data_plot = np.copy(data).astype(np.float64)
        data_plot[(data_plot <= 0) | ~np.isfinite(data_plot)] = np.nan
        data_plot = np.log10(data_plot + 1e-10)
        cmap = "viridis"
        cbar_label = "log10(Power)"
    else:
        data_plot = data.astype(np.float64)
        cmap = "RdBu_r"
        cbar_label = "Value"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])
    ax1 = axes[0]
    im = ax1.imshow(data_plot, aspect="auto", cmap=cmap,
                    origin="lower", interpolation="nearest")
    ax1.set_xlabel("Frequency bin")
    ax1.set_ylabel("Time index")
    ax1.set_title(
        f"{_PRODUCT_NAMES.get(product, f'Product {product}')} - Waterfall{title_suffix}"
    )
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(cbar_label)

    ax2 = axes[1]
    with np.errstate(invalid="ignore"):
        mean_spectrum = np.nanmean(data, axis=0)
        std_spectrum = np.nanstd(data, axis=0)
    freq_bins = np.arange(n_freq)
    if product < 4:
        valid = (mean_spectrum > 0) & np.isfinite(mean_spectrum)
        if np.any(valid):
            ax2.semilogy(freq_bins[valid], mean_spectrum[valid], "b-", label="Mean")
            ax2.fill_between(
                freq_bins[valid],
                np.maximum(mean_spectrum[valid] - std_spectrum[valid], 1e-10),
                mean_spectrum[valid] + std_spectrum[valid],
                alpha=0.3, color="blue",
            )
    else:
        ax2.plot(freq_bins, mean_spectrum, "b-", label="Mean")
        ax2.fill_between(freq_bins,
                         mean_spectrum - std_spectrum,
                         mean_spectrum + std_spectrum,
                         alpha=0.3, color="blue")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Frequency bin")
    ax2.set_ylabel("Mean value")
    ax2.set_title("Time-averaged spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_freq)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectra_waterfall(
    source,
    out_path: Path | str,
    *,
    products: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Render one waterfall PNG per correlation product.

    ``out_path`` is interpreted as a directory; one image per product is
    written into it as ``spectra_waterfall_p<NN>.png``. For backward
    compatibility, if ``out_path`` looks like a single PNG path, only the
    first selected product is written there.

    By default all 16 products are rendered. Pass ``products=[0, 1, ...]``
    to render a subset.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    groups = _normal_groups(source)
    written: List[Path] = []
    for group in groups:
        spectra = _normal_cube(group)
        n_time, n_products, _ = spectra.shape
        if n_time == 0:
            continue
        targets = (
            list(products) if products is not None else list(range(n_products))
        )
        if (
            out_path.suffix.lower() == ".png"
            and len(targets) == 1
            and len(groups) == 1
        ):
            written.append(
                _plot_one_waterfall(plt, spectra, targets[0], out_path)
            )
            continue
        out_path.mkdir(parents=True, exist_ok=True)
        tag = _grid_tag(group, len(groups) > 1)
        for product in targets:
            png = out_path / f"spectra_waterfall{tag}_p{product:02d}.png"
            written.append(
                _plot_one_waterfall(
                    plt,
                    spectra,
                    product,
                    png,
                    title_suffix=tag.replace("_", " "),
                )
            )
    return written


# ---------------------------------------------------------------------------
# Plot: per-product mean spectrum (one figure per product)
# ---------------------------------------------------------------------------

def _plot_one_mean(plt, spectra: np.ndarray, product: int, out_path: Path,
                   title_suffix: str = "") -> Path:
    n_time, n_products, n_freq = spectra.shape
    if product >= n_products:
        return out_path
    data = spectra[:, product, :].astype(np.float64)
    with np.errstate(invalid="ignore"):
        mean_spectrum = np.nanmean(data, axis=0)
        std_spectrum = np.nanstd(data, axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    freq_bins = np.arange(n_freq)
    if product < 4:
        valid = (mean_spectrum > 0) & np.isfinite(mean_spectrum)
        if np.any(valid):
            ax.semilogy(freq_bins[valid], mean_spectrum[valid], "b-", label="Mean")
            ax.fill_between(
                freq_bins[valid],
                np.maximum(mean_spectrum[valid] - std_spectrum[valid], 1e-10),
                mean_spectrum[valid] + std_spectrum[valid],
                alpha=0.3, color="blue",
            )
        ax.set_ylabel("Power (log)")
    else:
        ax.plot(freq_bins, mean_spectrum, "b-", label="Mean")
        ax.fill_between(freq_bins,
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.3, color="blue")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_ylabel("Value")
    ax.set_xlabel("Frequency bin")
    ax.set_title(
        f"{_PRODUCT_NAMES.get(product, f'Product {product}')} - "
        f"time-averaged{title_suffix}"
    )
    ax.set_xlim(0, n_freq)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_spectra_mean(
    source,
    out_path: Path | str,
    *,
    products: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Render one mean-spectrum PNG per correlation product.

    Same interface as :func:`plot_spectra_waterfall`.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    groups = _normal_groups(source)
    written: List[Path] = []
    for group in groups:
        spectra = _normal_cube(group)
        n_time, n_products, _ = spectra.shape
        if n_time == 0:
            continue
        targets = (
            list(products) if products is not None else list(range(n_products))
        )
        if (
            out_path.suffix.lower() == ".png"
            and len(targets) == 1
            and len(groups) == 1
        ):
            written.append(_plot_one_mean(plt, spectra, targets[0], out_path))
            continue
        out_path.mkdir(parents=True, exist_ok=True)
        tag = _grid_tag(group, len(groups) > 1)
        for product in targets:
            png = out_path / f"spectra_mean{tag}_p{product:02d}.png"
            written.append(
                _plot_one_mean(
                    plt,
                    spectra,
                    product,
                    png,
                    title_suffix=tag.replace("_", " "),
                )
            )
    return written


# ---------------------------------------------------------------------------
# Plot: ADC stats over time
# ---------------------------------------------------------------------------

def plot_adc_stats(source, out_path: Path | str) -> Path:
    """Min / max / mean / rms ADC stats over time, per channel (4 panels)."""
    plt = _require_matplotlib()
    out_path = Path(out_path)
    bundle = _as_bundle(source)
    names = ("adc_min", "adc_max", "adc_mean", "adc_rms")
    if not all(name in bundle.spectra_metadata for name in names):
        raise FileNotFoundError("ADC statistic fields are absent")
    arrays = [
        np.asarray(bundle.spectra_metadata[name], dtype=np.float64).copy()
        for name in names
    ]
    statistic_valid = bundle.spectra_metadata.get("adc_statistics_valid")
    if statistic_valid is None:
        statistic_valid = np.ones(arrays[0].shape, dtype=np.bool_)
    else:
        statistic_valid = np.asarray(statistic_valid, dtype=np.bool_)
    for name, array in zip(names, arrays):
        field_present = bundle.spectra_metadata_present.get(name)
        if field_present is not None:
            array[~np.asarray(field_present, dtype=np.bool_)] = np.nan
        array[~statistic_valid] = np.nan
    adc_min, adc_max, adc_mean, adc_rms = arrays
    raw_times = np.asarray(bundle.spectra_raw_times)
    raw_valid = bundle.spectra_raw_time_valid
    n = raw_times.size
    if n == 0:
        return out_path
    if raw_valid is not None and not np.all(raw_valid):
        x = np.arange(n)
        x_label = "spectrum row"
    else:
        x = raw_times - raw_times[0]
        x_label = "seconds since session start"

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    titles = ("ADC min", "ADC max", "ADC mean", "ADC rms")
    for ax, arr, title in zip(axes.flat, (adc_min, adc_max, adc_mean, adc_rms), titles):
        for ch in range(arr.shape[1]):
            ax.plot(x, arr[:, ch], lw=0.8, label=f"ch{ch}")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: DCB telemetry summary
# ---------------------------------------------------------------------------

def _telemetry_field_groups(bundle: SessionBundle) -> dict[str, tuple[str, ...]]:
    if bundle.layout_version == INGEST_LAYOUT_VERSION:
        collection = bundle.telemetry_fpga
        if collection is None:
            return {}
        grouped: dict[str, list[str]] = {}
        for item in collection.field_metadata:
            grouped.setdefault(item.display_group or "other", []).append(item.name)
        return {name: tuple(fields) for name, fields in grouped.items()}
    from . import telemetry as telemetry_mod

    return telemetry_mod.field_groups()


def _unassigned_telemetry_values(bundle: SessionBundle) -> dict[str, np.ndarray]:
    collection = bundle.telemetry_fpga
    if bundle.layout_version != INGEST_LAYOUT_VERSION or collection is None:
        return {}
    return collection.unassigned_engineering_values()


def plot_dcb_telemetry(source, out_path: Path | str) -> Path:
    """Summary plot of representative DCB telemetry channels.

    Layout-v4 panel grouping comes from persisted field metadata. Legacy
    layouts use the optional decoder's compatibility grouping when available.

    Raises ``FileNotFoundError`` if ``/DCB_telemetry`` is missing.
    """
    out_path = Path(out_path)
    bundle = _as_bundle(source)
    panel_groups = _telemetry_field_groups(bundle)
    telemetry = bundle.dcb_fpga
    unassigned_only = False
    if not telemetry or not any(np.asarray(value).size for value in telemetry.values()):
        unassigned = _unassigned_telemetry_values(bundle)
        if unassigned:
            telemetry = unassigned
            unassigned_only = True
    if not telemetry:
        raise FileNotFoundError("DCB telemetry is absent")
    if "raw_seconds" in telemetry:
        t = np.asarray(telemetry["raw_seconds"], dtype=np.float64)
    else:
        if (
            "mission_seconds" not in telemetry
            or "lusee_subsecs" not in telemetry
        ):
            raise FileNotFoundError("FPGA telemetry time axis is absent")
        ms = np.asarray(telemetry["mission_seconds"], dtype=np.float64)
        ss = np.asarray(telemetry["lusee_subsecs"], dtype=np.float64)
        t = ms + ss / 65536.0
    if t.size == 0:
        raise FileNotFoundError("FPGA telemetry has zero samples")
    t = t - t[0]
    if panel_groups:
        groups = []
        for title, fields in panel_groups.items():
            present: List[tuple] = []
            for name in fields:
                if name in telemetry:
                    present.append((name, np.asarray(telemetry[name])))
            if present:
                groups.append(
                    (f"{title} (unassigned)" if unassigned_only else title, present)
                )
    else:
        present = [
            (name, np.asarray(telemetry[name]))
            for name in sorted(telemetry)
            if name not in ("mission_seconds", "lusee_subsecs", "raw_seconds")
        ]
        title = "FPGA telemetry (unassigned)" if unassigned_only else "FPGA telemetry"
        groups = [(title, present)] if present else []

    if not groups:
        raise FileNotFoundError("no DCB telemetry channels found")

    plt = _require_matplotlib()
    n_panels = len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (title, present) in zip(axes, groups):
        for fname, arr in present:
            ax.plot(t, arr, lw=0.8, label=fname)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    sample_kind = "unassigned telemetry" if unassigned_only else "telemetry"
    axes[-1].set_xlabel(f"seconds since first {sample_kind} sample")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

# Plots that emit a single PNG vs. one PNG per correlation product.
_PER_PRODUCT_PLOTS = {"spectra_waterfall", "spectra_mean"}
_DISPATCH = {
    "spectra_waterfall": plot_spectra_waterfall,
    "spectra_mean": plot_spectra_mean,
    "adc_stats": plot_adc_stats,
    "dcb_telemetry": plot_dcb_telemetry,
}


def plot_session(
    source,
    plots_out_dir: Path | str,
    *,
    plots: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Render the requested plot set into ``plots_out_dir``.

    Per-product spectra plots produce one PNG per correlation product
    (``spectra_waterfall_pNN.png`` / ``spectra_mean_pNN.png``); the
    others produce a single ``<name>.png``. Plots whose input data is
    absent are skipped quietly with a log message. Returns the full list
    of written paths.
    """
    plots = tuple(plots) if plots else _AVAILABLE_PLOTS
    plots_out_dir = Path(plots_out_dir)
    plots_out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    bundle = _as_bundle(source)
    for name in plots:
        fn = _DISPATCH.get(name)
        if fn is None:
            log.warning("unknown plot '%s'; skipping", name)
            continue
        try:
            if name in _PER_PRODUCT_PLOTS:
                paths = fn(bundle, plots_out_dir)
                written.extend(paths)
            else:
                out_path = plots_out_dir / f"{name}.png"
                fn(bundle, out_path)
                written.append(out_path)
        except FileNotFoundError as exc:
            log.info("skipping plot '%s': %s", name, exc)
        except Exception as exc:    # noqa: BLE001
            log.warning("plot '%s' failed: %s", name, exc)
    return written
