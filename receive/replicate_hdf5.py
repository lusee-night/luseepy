#!/usr/bin/env python3

import os
import re
import shutil
from datetime import datetime, timedelta

import h5py
import numpy as np


def parse_session_filename(path: str):
    base = os.path.basename(path)
    match = re.match(r"session_(\d+)_(\d{8})_(\d{6})\.h5$", base)
    if not match:
        raise ValueError(f"Unrecognized session filename: {base}")
    idx = int(match.group(1))
    date_str = match.group(2)
    time_str = match.group(3)
    dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return idx, dt, base


def build_session_filename(idx: int, dt: datetime) -> str:
    return f"session_{idx:03d}_{dt:%Y%m%d_%H%M%S}.h5"


def compute_time_delta(h5_path: str) -> float:
    times = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if not key.startswith("item_"):
                continue
            meta = f[key].get("meta")
            if meta is None:
                continue
            if "time" not in meta.attrs:
                continue
            t = np.asarray(meta.attrs["time"])
            if t.ndim == 0:
                times.append(float(t))
            else:
                times.append(float(np.min(t)))
                times.append(float(np.max(t)))

    if not times:
        raise RuntimeError("No time attributes found in any item meta groups")

    tmin = min(times)
    tmax = max(times)
    return (tmax - tmin) * 3.0


def shift_times_in_place(h5_path: str, delta: float):
    with h5py.File(h5_path, "r+") as f:
        for key in f.keys():
            if not key.startswith("item_"):
                continue
            meta = f[key].get("meta")
            if meta is None or "time" not in meta.attrs:
                continue
            t = np.asarray(meta.attrs["time"])
            meta.attrs["time"] = t + delta


def main():
    src = "session_001_20251105_120504.h5"
    idx, dt, _ = parse_session_filename(src)
    new_idx = idx + 1
    new_dt = dt + timedelta(days=1)
    dst = build_session_filename(new_idx, new_dt)

    if not os.path.exists(src):
        raise FileNotFoundError(src)

    shutil.copy2(src, dst)

    delta = compute_time_delta(src)
    shift_times_in_place(dst, delta)

    print(f"Copied {src} -> {dst}")
    print(f"Applied time delta: {delta}")


if __name__ == "__main__":
    main()
