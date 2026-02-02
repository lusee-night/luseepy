#!/usr/bin/env python3

import os
import glob
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

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

        self.time = np.array([], dtype=np.float64)
        self.meta: Dict[str, np.ndarray] = {}
        self.lun_lat_deg = None
        self.lun_long_deg = None
        self.lun_height_m = None

        self._load_all()

    def _resolve_sources(
        self,
        source: Optional[Iterable[str]],
        data_root_env: str,
        id_pattern: str,
    ) -> List[str]:
        if isinstance(source, (list, tuple)):
            return [str(s) for s in source]

        source_str = "" if source is None else str(source)
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
        pattern = os.path.join(data_root, id_pattern.format(id=source_str))
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError( f"No HDF5 files matching '{pattern}' under {data_root_env}={data_root}")
        return matches

    def _load_all(self):
        data_chunks = []
        meta_chunks: Dict[str, List[np.ndarray]] = {}

        constants_seen = False
        for path in self.sources:
            with h5py.File(path, "r") as f:
                if "session_invariants" in f:
                    invariants = f["session_invariants"].attrs
                    if "software_version" in invariants:
                        self.software_version = int(invariants["software_version"])
                    if "firmware_version" in invariants:
                        self.firmware_version = int(invariants["firmware_version"])
                    if "firmware_id" in invariants:
                        self.firmware_id = int(invariants["firmware_id"])
                    if "firmware_date" in invariants:
                        self.firmware_date = int(invariants["firmware_date"])
                    if "firmware_time" in invariants:
                        self.firmware_time = int(invariants["firmware_time"])
                    if "start_unique_packet_id" in invariants:
                        self.start_unique_packet_id = int(invariants["start_unique_packet_id"])
                    if "start_time_32" in invariants:
                        self.start_time_32 = int(invariants["start_time_32"])
                    if "start_time_16" in invariants:
                        self.start_time_16 = int(invariants["start_time_16"])
                if "constants" in f:
                    consts = f["constants"].attrs
                    self.lun_lat_deg = float(consts["lun_lat_deg"])
                    self.lun_long_deg = float(consts["lun_long_deg"])
                    self.lun_height_m = float(consts["lun_height_m"])
                    constants_seen = True
                for item_name in self._iter_items(f):
                    item_group = f[item_name]
                    data, meta = self._load_item(item_group)
                    if data is None:
                        continue

                    data_chunks.append(data)

                    meta_arrays = self._expand_meta(meta, len(data))
                    for key, arr in meta_arrays.items():
                        meta_chunks.setdefault(key, []).append(arr)

        if data_chunks:
            self.data = np.concatenate(data_chunks, axis=0)
        else:
            self.data = np.array([])

        for key, arrays in meta_chunks.items():
            self.meta[key] = np.concatenate(arrays, axis=0)
            setattr(self, key, self.meta[key])

        if "time" in self.meta:
            self.time = self.meta["time"]
        if not constants_seen:
            warnings.warn("Missing constants group in HDF5 file; landing coordinates set to None")

    def _iter_items(self, h5file: h5py.File) -> List[str]:
        return sorted([key for key in h5file.keys() if key.startswith("item_")])

    def _load_item(self, item_group: h5py.Group) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        raise NotImplementedError

    def _load_meta(self, meta_group: h5py.Group) -> Dict[str, np.ndarray]:
        return self._read_group_recursive(meta_group)

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
        combined.time = np.concatenate([self.time, other.time], axis=0)
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
            return None, None

        data = item_group["spectra/data"][()]
        meta_group = item_group.get("meta")
        meta = self._load_meta(meta_group) if meta_group else {}
        return data, meta


class TRSpectra(Data):
    """Time-resolved spectra data: (n_time, 16, Navg2, tr_length)."""

    def _load_item(self, item_group: h5py.Group):
        if "tr_spectra/data" not in item_group:
            return None, None

        data = item_group["tr_spectra/data"][()]
        meta_group = item_group.get("meta")
        meta = self._load_meta(meta_group) if meta_group else {}
        return data, meta


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

        self.time = np.concatenate(time_chunks, axis=0) if time_chunks else np.array([], dtype=np.float64)
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
        combined.time = np.concatenate([self.time, other.time], axis=0)
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
        self.time = np.concatenate(time_chunks, axis=0) if time_chunks else np.array([], dtype=np.float64)
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
        combined.time = np.concatenate([self.time, other.time], axis=0)
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
    from icecream import ic
    #
    # data = Spectra(source="session_001_20251105_120504.h5")
    # print(dir(data))
    # ic(data.data.shape)
    # ic(data.FPGA_temperature.shape)
    # # ic(data.FPGA_temperature)
    #
    # ic(data.time.shape)
    # # ic(np.sum(np.sum(data.data, axis=2), axis=1))
    # # ic(data.time[1:] - data.time[:-1])
    # ic(data.software_version)

    data = Spectra(source=["session_001_20251105_120504.h5", "session_002_20251106_120504.h5"])
    # data = Spectra(source=["session_001_20251105_120504.h5"])
    print(dir(data))
    ic(data.data.shape)
    ic(data.FPGA_temperature.shape)
    # ic(data.FPGA_temperature)

    ic(data.time.shape)
    # ic(np.sum(np.sum(data.data, axis=2), axis=1))
    # ic(data.time[1:] - data.time[:-1])
    ic(data.software_version)
