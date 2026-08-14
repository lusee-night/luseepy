"""Per-session sanity-check plots.

Each plotting function takes either an open ``h5py.File`` or a path,
plus an output PNG path. They render quickly and are designed for
end-of-pipeline visual sanity checks; they are not publication graphics.

Standalone use::

    from lusee.ingest import viz
    viz.plot_session("session_001.h5", "out/plots/")
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import h5py
import numpy as np

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

@contextmanager
def _open_h5(h5: Union[h5py.File, Path, str]):
    if isinstance(h5, h5py.File):
        yield h5
        return
    p = Path(h5)
    f = h5py.File(p, "r")
    try:
        yield f
    finally:
        f.close()


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
    h5: Union[h5py.File, Path, str],
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
    with _open_h5(h5) as f:
        if "spectra" not in f or "data" not in f["spectra"]:
            raise FileNotFoundError("/spectra/data not present in HDF5")
        spectra = f["spectra"]["data"][...]
    n_time, n_products, _ = spectra.shape
    if n_time == 0:
        log.warning("/spectra/data has zero rows; not plotting waterfall")
        return []
    targets = list(products) if products is not None else list(range(n_products))

    written: List[Path] = []
    if out_path.suffix.lower() == ".png" and len(targets) == 1:
        written.append(_plot_one_waterfall(plt, spectra, targets[0], out_path))
        return written

    out_path.mkdir(parents=True, exist_ok=True)
    for p in targets:
        png = out_path / f"spectra_waterfall_p{p:02d}.png"
        written.append(_plot_one_waterfall(plt, spectra, p, png))
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
    h5: Union[h5py.File, Path, str],
    out_path: Path | str,
    *,
    products: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Render one mean-spectrum PNG per correlation product.

    Same interface as :func:`plot_spectra_waterfall`.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    with _open_h5(h5) as f:
        if "spectra" not in f or "data" not in f["spectra"]:
            raise FileNotFoundError("/spectra/data not present in HDF5")
        spectra = f["spectra"]["data"][...]
    n_time, n_products, _ = spectra.shape
    if n_time == 0:
        log.warning("/spectra/data has zero rows; not plotting mean")
        return []
    targets = list(products) if products is not None else list(range(n_products))

    written: List[Path] = []
    if out_path.suffix.lower() == ".png" and len(targets) == 1:
        written.append(_plot_one_mean(plt, spectra, targets[0], out_path))
        return written

    out_path.mkdir(parents=True, exist_ok=True)
    for p in targets:
        png = out_path / f"spectra_mean_p{p:02d}.png"
        written.append(_plot_one_mean(plt, spectra, p, png))
    return written


# ---------------------------------------------------------------------------
# Plot: ADC stats over time
# ---------------------------------------------------------------------------

def plot_adc_stats(h5: Union[h5py.File, Path, str], out_path: Path | str) -> Path:
    """Min / max / mean / rms ADC stats over time, per channel (4 panels)."""
    plt = _require_matplotlib()
    out_path = Path(out_path)
    with _open_h5(h5) as f:
        if "spectra" not in f or "metadata" not in f["spectra"]:
            raise FileNotFoundError("/spectra/metadata not present in HDF5")
        md = f["spectra"]["metadata"]
        if not all(k in md for k in ("adc_min", "adc_max", "adc_mean", "adc_rms")):
            raise FileNotFoundError("ADC stat fields missing in /spectra/metadata")
        adc_min = md["adc_min"][...]
        adc_max = md["adc_max"][...]
        adc_mean = md["adc_mean"][...]
        adc_rms = md["adc_rms"][...]
        raw_times = f["spectra"]["raw_times"][...]
    n = raw_times.size
    if n == 0:
        return out_path

    x = raw_times - raw_times[0] if raw_times[0] != 0 else raw_times

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    titles = ("ADC min", "ADC max", "ADC mean", "ADC rms")
    for ax, arr, title in zip(axes.flat, (adc_min, adc_max, adc_mean, adc_rms), titles):
        for ch in range(arr.shape[1]):
            ax.plot(x, arr[:, ch], lw=0.8, label=f"ch{ch}")
        ax.set_title(title)
        ax.set_xlabel("seconds since session start")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Plot: DCB telemetry summary
# ---------------------------------------------------------------------------

def plot_dcb_telemetry(h5: Union[h5py.File, Path, str], out_path: Path | str) -> Path:
    """Summary plot of representative DCB telemetry channels.

    Channel names and panel grouping come from the private
    ``lusee_telemetry`` decoder (via :func:`lusee.ingest.telemetry.field_groups`).
    Without a decoder, every ``fpga_*`` channel found in the file is
    plotted on a single panel.

    Raises ``FileNotFoundError`` if ``/DCB_telemetry`` is missing.
    """
    plt = _require_matplotlib()
    out_path = Path(out_path)
    from . import telemetry as _telemetry_mod
    panel_groups = _telemetry_mod.field_groups()    # may be empty

    with _open_h5(h5) as f:
        if "DCB_telemetry" not in f:
            raise FileNotFoundError("/DCB_telemetry not present in HDF5")
        g = f["DCB_telemetry"]
        if "fpga_mission_seconds" not in g or "fpga_lusee_subsecs" not in g:
            raise FileNotFoundError("FPGA telemetry time axis missing")
        ms = g["fpga_mission_seconds"][...]
        ss = g["fpga_lusee_subsecs"][...]
        t = ms + ss * (1.0 / 65536.0)
        if t.size == 0:
            log.warning("FPGA telemetry has zero samples; not plotting")
            return out_path
        t = t - t[0]
        if panel_groups:
            groups = []
            for title, fields in panel_groups.items():
                present: List[tuple] = []
                for fname in fields:
                    key = f"fpga_{fname}"
                    if key in g:
                        present.append((fname, g[key][...]))
                if present:
                    groups.append((title, present))
        else:
            present = [
                (k[len("fpga_"):], g[k][...])
                for k in sorted(g)
                if k.startswith("fpga_") and k not in ("fpga_mission_seconds",
                                                       "fpga_lusee_subsecs")
            ]
            groups = [("FPGA telemetry", present)] if present else []

    if not groups:
        log.warning("no DCB telemetry channels found; not plotting")
        return out_path

    n_panels = len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]
    for ax, (title, present) in zip(axes, groups):
        for fname, arr in present:
            ax.plot(t, arr, lw=0.8, label=fname)
        ax.set_title(title)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("seconds since first telemetry sample")
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
    h5_path: Path | str,
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
    with _open_h5(h5_path) as f:
        for name in plots:
            fn = _DISPATCH.get(name)
            if fn is None:
                log.warning("unknown plot '%s'; skipping", name)
                continue
            try:
                if name in _PER_PRODUCT_PLOTS:
                    paths = fn(f, plots_out_dir)
                    written.extend(paths)
                else:
                    out_path = plots_out_dir / f"{name}.png"
                    fn(f, out_path)
                    written.append(out_path)
            except FileNotFoundError as exc:
                log.info("skipping plot '%s': %s", name, exc)
            except Exception as exc:    # noqa: BLE001
                log.warning("plot '%s' failed: %s", name, exc)
    return written
