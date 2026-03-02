#!/usr/bin/env python3

import argparse
from typing import List, Tuple

import h5py
import numpy as np
from matplotlib import pyplot as plt
from icecream import ic

from data import Spectra, TELEMETRY_FIELD_ALIASES
from telemetry_utils import TELEMETRY_FIELD_NAMES


def _pick_fields(data: Spectra) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    voltage: List[Tuple[str, str]] = []
    temps: List[Tuple[str, str]] = []

    for key in TELEMETRY_FIELD_NAMES:
        attr = TELEMETRY_FIELD_ALIASES.get(key, key)
        if not hasattr(data, attr):
            continue
        if ("THERM" in key) or key.endswith("_T"):
            temps.append((attr, key))
        elif "V" in key:
            voltage.append((attr, key))

    return voltage[:2], temps[:2]


def _load_interpolated_times_from_h5(source: str) -> np.ndarray:
    with h5py.File(source, "r") as f:
        group = f.get("spectra_interpolated_telemetry")
        if group is None or "time" not in group:
            return np.array([], dtype=np.float64)
        return np.asarray(group["time"][()], dtype=np.float64).reshape(-1)


def _ticks_from_times(*arrays: np.ndarray) -> np.ndarray:
    valid = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays if np.asarray(a).size > 0]
    if not valid:
        return np.array([], dtype=np.float64)
    merged = np.concatenate(valid, axis=0)
    return np.unique(merged)


def plot_session(source: str, freq_bin: int) -> None:
    data = Spectra(source=source)
    voltage_fields, temp_fields = _pick_fields(data)

    spectra = np.asarray(data.data)
    if spectra.ndim != 3 or spectra.shape[0] == 0:
        raise RuntimeError(f"Unexpected spectra shape: {spectra.shape}")
    if freq_bin < 0 or freq_bin >= spectra.shape[2]:
        raise ValueError(f"freq_bin {freq_bin} out of range [0, {spectra.shape[2] - 1}]")

    spectra_times = np.asarray(data.time, dtype=np.float64).reshape(-1)
    interp_times = _load_interpolated_times_from_h5(source)
    if spectra_times.size != spectra.shape[0]:
        spectra_times = np.arange(spectra.shape[0], dtype=np.float64)

    if interp_times.size > 0 and interp_times.size == spectra_times.size:
        max_abs_diff = float(np.max(np.abs(spectra_times - interp_times)))
        ic(max_abs_diff, spectra_times.size)
    else:
        print(
            spectra_times.size,
            interp_times.size,
            "could not compare full time arrays",
        )

    xticks = _ticks_from_times(spectra_times, interp_times)
    x = spectra_times

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    y_bin = np.asarray(spectra[:, 0, freq_bin], dtype=np.float64).reshape(-1)
    axes[0].plot(x[:len(y_bin)], y_bin, "-o", markersize=3.5, linewidth=1.0, label=f"prod0 bin{freq_bin}")
    axes[0].set_ylabel("Spectra power")
    axes[0].set_title(f"Spectra single-bin sanity (product 0, freq bin {freq_bin})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)

    for attr, label in temp_fields:
        y = np.asarray(getattr(data, attr)).reshape(-1)
        axes[1].plot(x[:len(y)], y, "-o", label=label, linewidth=1.0, markersize=3.0)
    axes[1].set_ylabel("Temperature [C]")
    axes[1].set_title("Telemetry temperatures")
    axes[1].grid(True, alpha=0.3)
    if temp_fields:
        axes[1].legend(loc="best", fontsize=8)

    for attr, label in voltage_fields:
        y = np.asarray(getattr(data, attr)).reshape(-1)
        axes[2].plot(x[:len(y)], y, "-o", label=label, linewidth=1.0, markersize=3.0)
    axes[2].set_ylabel("Voltage")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Telemetry voltages")
    axes[2].grid(True, alpha=0.3)
    if voltage_fields:
        axes[2].legend(loc="best", fontsize=8)

    if xticks.size > 0:
        axes[-1].set_xticks(xticks)
        axes[-1].tick_params(axis="x", rotation=90)

    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spectra and selected interpolated telemetry fields from an HDF5 file."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to HDF5 file (or source accepted by data.Spectra).",
    )
    parser.add_argument(
        "--freq-bin",
        type=int,
        default=0,
        help="Frequency bin index [0..2047] for single-bin spectra panel.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    plot_session(args.source, args.freq_bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
