#!/usr/bin/env python3

import os
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np
from icecream import ic


class Data:
    """Base class for HDF5-backed data products."""

    def __init__(
        self,
        source: Optional[Iterable[str]] = None,
        *,
        data_root_env: str = "LUSEE_SESSIONS_DIR",
        id_pattern: str = "*{id}*.h5",
        simulated: bool = False,
    ):
        self.sources = self._resolve_sources(source, data_root_env, id_pattern)
        self.simulated = simulated

        self.times = np.array([], dtype=np.float64)
        self.meta: Dict[str, np.ndarray] = {}

        self._load_all()

    def _resolve_sources(
        self,
        source: Optional[Iterable[str]],
        data_root_env: str,
        id_pattern: str,
    ) -> List[str]:
        if source is None:
            return []

        if isinstance(source, (list, tuple)):
            return [str(s) for s in source]

        source_str = str(source)
        if os.path.isdir(source_str):
            files = [
                os.path.join(source_str, f)
                for f in sorted(os.listdir(source_str))
                if f.endswith(".h5")
            ]
            return files

        if os.path.exists(source_str):
            return [source_str]

        data_root = os.environ.get(data_root_env)
        if not data_root:
            raise FileNotFoundError(
                f"Source '{source_str}' not found and {data_root_env} is not set"
            )
        pattern = id_pattern.format(id=source_str)
        matches = sorted(
            [
                os.path.join(data_root, name)
                for name in os.listdir(data_root)
                if name.endswith(".h5") and self._match_pattern(name, pattern)
            ]
        )
        if not matches:
            raise FileNotFoundError( f"No HDF5 files matching '{pattern}' under {data_root_env}={data_root}")
        return matches

    def _match_pattern(self, name: str, pattern: str) -> bool:
        if pattern == "*":
            return True
        if pattern.startswith("*") and pattern.endswith("*"):
            return pattern.strip("*") in name
        if pattern.startswith("*"):
            return name.endswith(pattern.strip("*"))
        if pattern.endswith("*"):
            return name.startswith(pattern.strip("*"))
        return name == pattern

    def _load_all(self):
        data_chunks = []
        time_chunks = []
        meta_chunks: Dict[str, List[np.ndarray]] = {}

        for path in self.sources:
            with h5py.File(path, "r") as f:
                for item_name in self._iter_items(f):
                    item_group = f[item_name]
                    data, times, meta = self._load_item(item_group)
                    if data is None:
                        continue

                    data_chunks.append(data)
                    time_chunks.append(times)

                    meta_arrays = self._expand_meta(meta, len(times))
                    for key, arr in meta_arrays.items():
                        meta_chunks.setdefault(key, []).append(arr)

        if data_chunks:
            self.data = np.concatenate(data_chunks, axis=0)
        else:
            self.data = np.array([])

        if time_chunks:
            self.times = np.concatenate(time_chunks, axis=0)
        else:
            self.times = np.array([], dtype=np.float64)

        for key, arrays in meta_chunks.items():
            self.meta[key] = np.concatenate(arrays, axis=0)
            setattr(self, key, self.meta[key])

    def _iter_items(self, h5file: h5py.File) -> List[str]:
        return sorted([key for key in h5file.keys() if key.startswith("item_")])

    def _load_item(self, item_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def _load_meta(self, meta_group: h5py.Group) -> Dict[str, np.ndarray]:
        return self._read_group_recursive(meta_group)

    def _load_times_from_meta(self, meta_group: Optional[h5py.Group], n_rows: int) -> Optional[np.ndarray]:
        if meta_group is None:
            return None
        if "time" not in meta_group.attrs:
            return None
        time_value = meta_group.attrs["time"]
        time_arr = np.asarray(time_value)
        if time_arr.ndim == 0:
            return np.full((n_rows,), time_arr, dtype=np.float64)
        if time_arr.shape[0] == n_rows:
            return time_arr.astype(np.float64)
        return None

    def _read_group_recursive(self, group: h5py.Group, prefix: str = "") -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        for key, value in group.attrs.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}__{key}"
            name = self._alias_name(name)
            out[name] = np.asarray(value)

        for key, item in group.items():
            name = f"{prefix}{key}" if not prefix else f"{prefix}__{key}"
            name = self._alias_name(name)
            if isinstance(item, h5py.Dataset):
                out[name] = item[()]
            elif isinstance(item, h5py.Group):
                out.update(self._read_group_recursive(item, name))

        return out

    def _alias_name(self, name: str) -> str:
        alias_map = {
            "telemetry_T_FPGA": "FPGA_temperature",
        }
        return alias_map.get(name, name)

    def _expand_meta(self, meta: Dict[str, np.ndarray], n_rows: int) -> Dict[str, np.ndarray]:
        expanded: Dict[str, np.ndarray] = {}
        for key, value in meta.items():
            arr = np.asarray(value)

            if arr.ndim == 0:
                expanded[key] = np.full((n_rows,), arr, dtype=arr.dtype)
                continue

            expanded[key] = np.broadcast_to(arr, (n_rows,) + arr.shape).copy()

        return expanded

    def __add__(self, other: "Data") -> "Data":
        if type(self) is not type(other):
            raise TypeError("Can only add Data objects of the same type")
        combined = self.__class__(source=[])
        combined.data = np.concatenate([self.data, other.data], axis=0)
        combined.times = np.concatenate([self.times, other.times], axis=0)
        combined.meta = {}
        for key in set(self.meta.keys()).union(other.meta.keys()):
            if key in self.meta and key in other.meta:
                combined.meta[key] = np.concatenate([self.meta[key], other.meta[key]], axis=0)
            elif key in self.meta:
                combined.meta[key] = self.meta[key]
            else:
                combined.meta[key] = other.meta[key]
            setattr(combined, key, combined.meta[key])

        return combined


class Spectra(Data):
    """Spectra data: (n_time, 16, 2048) + timestamps."""

    def _load_item(self, item_group: h5py.Group):
        if "spectra/data" not in item_group:
            return None, None, None

        data = item_group["spectra/data"][()]
        meta_group = item_group.get("meta")
        times = self._load_times_from_meta(meta_group, data.shape[0])
        if times is None:
            times = np.full((data.shape[0],), np.nan, dtype=np.float64)

        meta = self._load_meta(meta_group) if meta_group else {}
        return data, times, meta


class TRSpectra(Data):
    """Time-resolved spectra data: (n_time, 16, Navg2, tr_length)."""

    def _load_item(self, item_group: h5py.Group):
        if "tr_spectra/data" not in item_group:
            return None, None, None

        data = item_group["tr_spectra/data"][()]
        meta_group = item_group.get("meta")
        times = self._load_times_from_meta(meta_group, data.shape[0])
        if times is None:
            times = np.full((data.shape[0],), np.nan, dtype=np.float64)

        meta = self._load_meta(meta_group) if meta_group else {}
        return data, times, meta


class ZoomSpectra(Data):
    """Zoom spectra data from calibrator zoom packets."""

    def _load_all(self):
        time_chunks = []
        meta_chunks: Dict[str, List[np.ndarray]] = {}
        ch1_autocorr_chunks = []
        ch2_autocorr_chunks = []
        ch1_2_corr_real_chunks = []
        ch1_2_corr_imag_chunks = []
        unique_id_chunks = []
        pfb_index_chunks = []

        for path in self.sources:
            with h5py.File(path, "r") as f:
                for item_name in self._iter_items(f):
                    item_group = f[item_name]
                    if "calibrator/zoom_spectra" not in item_group:
                        continue

                    zoom_group = item_group["calibrator/zoom_spectra"]
                    times = zoom_group.get("timestamps")
                    if times is None:
                        times = np.array([], dtype=np.float64)
                    else:
                        times = times[()]

                    time_chunks.append(times)
                    ch1_autocorr_chunks.append(zoom_group["ch1_autocorr"][()])
                    ch2_autocorr_chunks.append(zoom_group["ch2_autocorr"][()])
                    ch1_2_corr_real_chunks.append(zoom_group["ch1_2_corr_real"][()])
                    ch1_2_corr_imag_chunks.append(zoom_group["ch1_2_corr_imag"][()])
                    unique_id_chunks.append(zoom_group["unique_ids"][()])
                    pfb_index_chunks.append(zoom_group["pfb_indices"][()])

                    meta_group = item_group.get("meta")
                    meta = self._load_meta(meta_group) if meta_group else {}
                    meta_arrays = self._expand_meta(meta, len(times))
                    for key, arr in meta_arrays.items():
                        meta_chunks.setdefault(key, []).append(arr)

        self.times = np.concatenate(time_chunks, axis=0) if time_chunks else np.array([], dtype=np.float64)
        self.ch1_autocorr = np.concatenate(ch1_autocorr_chunks, axis=0) if ch1_autocorr_chunks else np.array([])
        self.ch2_autocorr = np.concatenate(ch2_autocorr_chunks, axis=0) if ch2_autocorr_chunks else np.array([])
        self.ch1_2_corr_real = np.concatenate(ch1_2_corr_real_chunks, axis=0) if ch1_2_corr_real_chunks else np.array([])
        self.ch1_2_corr_imag = np.concatenate(ch1_2_corr_imag_chunks, axis=0) if ch1_2_corr_imag_chunks else np.array([])
        self.unique_ids = np.concatenate(unique_id_chunks, axis=0) if unique_id_chunks else np.array([])
        self.pfb_indices = np.concatenate(pfb_index_chunks, axis=0) if pfb_index_chunks else np.array([])
        self.data = self.ch1_autocorr

        self.meta = {}
        for key, arrays in meta_chunks.items():
            self.meta[key] = np.concatenate(arrays, axis=0)
            setattr(self, key, self.meta[key])

    def __add__(self, other: "ZoomSpectra") -> "ZoomSpectra":
        if type(self) is not type(other):
            raise TypeError("Can only add Data objects of the same type")
        combined = ZoomSpectra(source=[])
        combined.times = np.concatenate([self.times, other.times], axis=0)
        combined.ch1_autocorr = np.concatenate([self.ch1_autocorr, other.ch1_autocorr], axis=0)
        combined.ch2_autocorr = np.concatenate([self.ch2_autocorr, other.ch2_autocorr], axis=0)
        combined.ch1_2_corr_real = np.concatenate([self.ch1_2_corr_real, other.ch1_2_corr_real], axis=0)
        combined.ch1_2_corr_imag = np.concatenate([self.ch1_2_corr_imag, other.ch1_2_corr_imag], axis=0)
        combined.unique_ids = np.concatenate([self.unique_ids, other.unique_ids], axis=0)
        combined.pfb_indices = np.concatenate([self.pfb_indices, other.pfb_indices], axis=0)
        combined.data = combined.ch1_autocorr

        combined.meta = {}
        for key in set(self.meta.keys()).union(other.meta.keys()):
            if key in self.meta and key in other.meta:
                combined.meta[key] = np.concatenate([self.meta[key], other.meta[key]], axis=0)
            elif key in self.meta:
                combined.meta[key] = self.meta[key]
            else:
                combined.meta[key] = other.meta[key]
            setattr(combined, key, combined.meta[key])

        return combined


class CalibratorData(Data):
    """Calibrator data packets (variable sized)."""

    def _load_all(self):
        packet_chunks = []
        time_chunks = []
        meta_chunks: Dict[str, List[np.ndarray]] = {}

        for path in self.sources:
            with h5py.File(path, "r") as f:
                for item_name in self._iter_items(f):
                    item_group = f[item_name]
                    if "calibrator/data" not in item_group:
                        continue

                    cal_group = item_group["calibrator/data"]
                    packets = [cal_group[name][()] for name in sorted(cal_group.keys())]
                    packet_chunks.extend(packets)
                    time_chunks.append(np.zeros((len(packets),), dtype=np.float64))

                    meta_group = item_group.get("meta")
                    meta = self._load_meta(meta_group) if meta_group else {}
                    meta_arrays = self._expand_meta(meta, len(packets))
                    for key, arr in meta_arrays.items():
                        meta_chunks.setdefault(key, []).append(arr)

        self.packets = packet_chunks
        self.times = np.concatenate(time_chunks, axis=0) if time_chunks else np.array([], dtype=np.float64)
        self.data = np.array([], dtype=np.float32)

        self.meta = {}
        for key, arrays in meta_chunks.items():
            self.meta[key] = np.concatenate(arrays, axis=0)
            setattr(self, key, self.meta[key])

    def __add__(self, other: "CalibratorData") -> "CalibratorData":
        if type(self) is not type(other):
            raise TypeError("Can only add Data objects of the same type")
        combined = CalibratorData(source=[])
        combined.packets = self.packets + other.packets
        combined.times = np.concatenate([self.times, other.times], axis=0)
        combined.data = np.array([], dtype=np.float32)

        combined.meta = {}
        for key in set(self.meta.keys()).union(other.meta.keys()):
            if key in self.meta and key in other.meta:
                combined.meta[key] = np.concatenate([self.meta[key], other.meta[key]], axis=0)
            elif key in self.meta:
                combined.meta[key] = self.meta[key]
            else:
                combined.meta[key] = other.meta[key]
            setattr(combined, key, combined.meta[key])

        return combined


if __name__ == "__main__":
    data = Spectra(source = "session_001_20251105_120504.h5")
    print(dir(data))
    ic(data.data.shape)
    ic(np.sum(np.sum(data.data, axis=2), axis=1))
