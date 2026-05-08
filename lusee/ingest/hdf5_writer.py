"""Stage 7: write a session's products to an HDF5 file (layout v2).

The schema follows spec section 8. Group / dataset coordinates:

  /                           file root, top-level attributes
  /session_invariants         attributes only, sourced from Hello
  /constants                  attributes only, lunar location + MJD calibration
  /spectra/                   dense spectra cube + per-row metadata
  /tr_spectra/                dense TR cube
  /calibrator/zoom_spectra/   zoom calibrator cube (optional)
  /calibrator/data/           variable-length cal data (optional)
  /grimm_spectra/             optional
  /waveform/                  per-channel ADC waveforms (optional)
  /housekeeping/              per-packet records (optional)
  /DCB_telemetry/             FPGA + encoder time series (optional)
  /spectra_interpolated_telemetry/  resampled FPGA telemetry on the spectra
                                    time axis (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import h5py
import numpy as np

from .constants import (
    DEFAULT_LUN_HEIGHT_M,
    DEFAULT_LUN_LAT_DEG,
    DEFAULT_LUN_LONG_DEG,
    DEFAULT_MJD_EPOCH_OFFSET_DAYS,
    DEFAULT_RAW_TIME_SUBTRACT_SECONDS,
    HDF5_DEFAULT_COMPRESSION,
    HDF5_DEFAULT_COMPRESSION_OPTS,
    HDF5_LAYOUT_VERSION,
    NCHANNELS,
    NPRODUCTS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
    ZOOM_COMPONENTS,
)
from .decode import Products

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compression defaults
# ---------------------------------------------------------------------------

def _gzip_kwargs(min_elems: int = 32) -> dict:
    return {
        "compression": HDF5_DEFAULT_COMPRESSION,
        "compression_opts": HDF5_DEFAULT_COMPRESSION_OPTS,
    }


def _create_dataset(group: h5py.Group, name: str, data: np.ndarray) -> h5py.Dataset:
    if data.size == 0:
        # h5py refuses gzip on empty datasets; create uncompressed.
        return group.create_dataset(name, data=data)
    return group.create_dataset(name, data=data, **_gzip_kwargs())


# ---------------------------------------------------------------------------
# Session-invariants & constants groups
# ---------------------------------------------------------------------------

def _write_session_invariants(h5: h5py.File, products: Products) -> None:
    g = h5.create_group("session_invariants")
    fields = {
        "software_version": products.sw_version,
        "firmware_version": products.fw_version,
        "firmware_id": products.fw_id,
        "firmware_date": products.fw_date,
        "firmware_time": products.fw_time,
        "start_unique_packet_id": products.start_unique_packet_id,
        "start_time_32": products.start_time_32,
        "start_time_16": products.start_time_16,
    }
    for k, v in fields.items():
        if v is None:
            continue
        g.attrs[k] = np.int64(v)


def _write_constants(
    h5: h5py.File,
    *,
    lun_lat_deg: float,
    lun_long_deg: float,
    lun_height_m: float,
    raw_time_subtract_seconds: float,
    mjd_epoch_offset_days: float,
) -> None:
    g = h5.create_group("constants")
    g.attrs["lun_lat_deg"] = np.float64(lun_lat_deg)
    g.attrs["lun_long_deg"] = np.float64(lun_long_deg)
    g.attrs["lun_height_m"] = np.float64(lun_height_m)
    g.attrs["raw_time_subtract_seconds"] = np.float64(raw_time_subtract_seconds)
    g.attrs["mjd_epoch_offset_days"] = np.float64(mjd_epoch_offset_days)


def _to_mjd(raw_seconds: np.ndarray, raw_subtract: float, mjd_offset: float) -> np.ndarray:
    return (raw_seconds - raw_subtract) / 86400.0 + mjd_offset


# ---------------------------------------------------------------------------
# /spectra
# ---------------------------------------------------------------------------

def _write_spectra(
    h5: h5py.File,
    products: Products,
    *,
    raw_subtract: float,
    mjd_offset: float,
) -> None:
    if not products.spectra:
        return
    n = len(products.spectra)
    cube = np.full((n, NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    metadata_arrays: Dict[str, list] = {}
    for i, s in enumerate(products.spectra):
        cube[i] = s.data
        unique_ids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
        for k, v in s.metadata.items():
            metadata_arrays.setdefault(k, []).append(v)

    # Sample-retention: drop rows that are entirely NaN (no products at all).
    finite_mask = np.any(np.isfinite(cube.reshape(n, -1)), axis=1)
    keep = np.flatnonzero(finite_mask)
    n_kept = keep.size
    cube = cube[keep]
    unique_ids = unique_ids[keep]
    raw_times = raw_times[keep]
    mjd_times = _to_mjd(raw_times, raw_subtract, mjd_offset)
    original_indices = keep.astype(np.int64)

    g = h5.create_group("spectra")
    g.attrs["count"] = np.int64(n_kept)
    _create_dataset(g, "data", cube)
    _create_dataset(g, "unique_ids", unique_ids)
    _create_dataset(g, "raw_times", raw_times)
    _create_dataset(g, "mjd_times", mjd_times)
    _create_dataset(g, "original_indices", original_indices)

    md = g.create_group("metadata")
    for name, items in metadata_arrays.items():
        if len(items) != n:
            continue
        try:
            arr = np.asarray(items)
        except Exception as exc:    # noqa: BLE001
            log.debug("skipping metadata field %s: %s", name, exc)
            continue
        if arr.size == 0:
            continue
        arr = arr[keep]
        if arr.dtype.kind in ("U", "O", "S"):
            continue
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int64)
        elif arr.dtype.kind == "i":
            arr = arr.astype(np.int64)
        elif arr.dtype.kind in ("f", "u"):
            arr = arr.astype(np.float64) if arr.dtype.kind == "f" else arr.astype(np.int64)
        _create_dataset(md, name, arr)


# ---------------------------------------------------------------------------
# /tr_spectra
# ---------------------------------------------------------------------------

def _write_tr_spectra(
    h5: h5py.File,
    products: Products,
    *,
    raw_subtract: float,
    mjd_offset: float,
) -> None:
    if not products.tr_spectra:
        return
    n = len(products.tr_spectra)
    navg2_max = max(s.navg2 for s in products.tr_spectra)
    tr_len_max = max(s.tr_length for s in products.tr_spectra)
    cube = np.full((n, NPRODUCTS, navg2_max, tr_len_max), np.nan, dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    navg2_per = np.zeros(n, dtype=np.int64)
    tr_len_per = np.zeros(n, dtype=np.int64)
    metadata_arrays: Dict[str, list] = {}
    for i, s in enumerate(products.tr_spectra):
        cube[i, :, :s.navg2, :s.tr_length] = s.data
        unique_ids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
        navg2_per[i] = s.navg2
        tr_len_per[i] = s.tr_length
        for k, v in s.metadata.items():
            metadata_arrays.setdefault(k, []).append(v)

    finite_mask = np.any(np.isfinite(cube.reshape(n, -1)), axis=1)
    keep = np.flatnonzero(finite_mask)
    n_kept = keep.size
    cube = cube[keep]
    unique_ids = unique_ids[keep]
    raw_times = raw_times[keep]
    navg2_per = navg2_per[keep]
    tr_len_per = tr_len_per[keep]
    mjd_times = _to_mjd(raw_times, raw_subtract, mjd_offset)
    original_indices = keep.astype(np.int64)

    g = h5.create_group("tr_spectra")
    g.attrs["count"] = np.int64(n_kept)
    g.attrs["tr_spectra_Navg2"] = np.int64(navg2_max)
    g.attrs["tr_spectra_tr_length"] = np.int64(tr_len_max)
    _create_dataset(g, "data", cube)
    _create_dataset(g, "unique_ids", unique_ids)
    _create_dataset(g, "raw_times", raw_times)
    _create_dataset(g, "mjd_times", mjd_times)
    _create_dataset(g, "navg2_per_sample", navg2_per)
    _create_dataset(g, "tr_length_per_sample", tr_len_per)
    _create_dataset(g, "original_indices", original_indices)

    md = g.create_group("metadata")
    for name, items in metadata_arrays.items():
        if len(items) != n:
            continue
        try:
            arr = np.asarray(items)
        except Exception:    # noqa: BLE001
            continue
        if arr.size == 0 or arr.dtype.kind in ("U", "O", "S"):
            continue
        arr = arr[keep]
        if arr.dtype == np.bool_:
            arr = arr.astype(np.int64)
        _create_dataset(md, name, arr)


# ---------------------------------------------------------------------------
# /calibrator
# ---------------------------------------------------------------------------

def _write_zoom_spectra(
    h5: h5py.File,
    products: Products,
    *,
    raw_subtract: float,
    mjd_offset: float,
) -> None:
    if not products.zoom_spectra:
        return
    n = len(products.zoom_spectra)
    cube = np.zeros((n, ZOOM_COMPONENTS, ZOOM_BINS), dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    pfb_indices = np.zeros(n, dtype=np.int32)
    raw_times = np.zeros(n, dtype=np.float64)
    for i, z in enumerate(products.zoom_spectra):
        cube[i] = z.data
        unique_ids[i] = z.unique_packet_id
        pfb_indices[i] = z.pfb_index
        raw_times[i] = z.raw_seconds
    mjd_times = _to_mjd(raw_times, raw_subtract, mjd_offset)
    original_indices = np.arange(n, dtype=np.int64)

    cal = h5.require_group("calibrator")
    g = cal.create_group("zoom_spectra")
    g.attrs["count"] = np.int64(n)
    _create_dataset(g, "data", cube)
    _create_dataset(g, "unique_ids", unique_ids)
    _create_dataset(g, "pfb_indices", pfb_indices)
    _create_dataset(g, "raw_times", raw_times)
    _create_dataset(g, "mjd_times", mjd_times)
    _create_dataset(g, "original_indices", original_indices)


def _write_cal_data(h5: h5py.File, products: Products) -> None:
    if not products.cal_data:
        return
    cal = h5.require_group("calibrator")
    g = cal.create_group("data")
    g.attrs["count"] = np.int64(len(products.cal_data))
    for sample in products.cal_data:
        name = f"packet_{sample.packet_idx}_ch_{sample.channel_idx}"
        _create_dataset(g, name, sample.data)


# ---------------------------------------------------------------------------
# /grimm_spectra
# ---------------------------------------------------------------------------

def _write_grimm_spectra(
    h5: h5py.File,
    products: Products,
    *,
    raw_subtract: float,
    mjd_offset: float,
) -> None:
    if not products.grimm_spectra:
        return
    n = len(products.grimm_spectra)
    cube = np.full((n, NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(products.grimm_spectra):
        cube[i] = s.data
        unique_ids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
    mjd_times = _to_mjd(raw_times, raw_subtract, mjd_offset)
    g = h5.create_group("grimm_spectra")
    g.attrs["count"] = np.int64(n)
    _create_dataset(g, "data", cube)
    _create_dataset(g, "unique_ids", unique_ids)
    _create_dataset(g, "raw_times", raw_times)
    _create_dataset(g, "mjd_times", mjd_times)


# ---------------------------------------------------------------------------
# /waveform
# ---------------------------------------------------------------------------

def _write_waveform(h5: h5py.File, products: Products) -> None:
    if not products.waveforms:
        return
    by_channel: Dict[int, list] = {}
    for w in products.waveforms:
        by_channel.setdefault(w.channel, []).append(w)
    g = h5.create_group("waveform")
    g.attrs["total_count"] = np.int64(len(products.waveforms))
    g.attrs["channels"] = np.array(sorted(by_channel.keys()), dtype=np.int64)
    for ch in sorted(by_channel.keys()):
        items = by_channel[ch]
        gch = g.create_group(f"channel_{ch}")
        gch.attrs["count"] = np.int64(len(items))
        gch.attrs["channel"] = np.int64(ch)
        wf = np.zeros((len(items), WAVEFORM_SAMPLES), dtype=np.int16)
        ts = np.zeros(len(items), dtype=np.float64)
        for i, w in enumerate(items):
            wf[i] = w.data
            ts[i] = w.raw_seconds
        _create_dataset(gch, "waveforms", wf)
        _create_dataset(gch, "timestamps", ts)


# ---------------------------------------------------------------------------
# /housekeeping
# ---------------------------------------------------------------------------

def _stack_vec(values, n: int, dtype) -> np.ndarray:
    """Stack a list of (n,) array-like rows into a (len, n) array."""
    out = np.zeros((len(values), n), dtype=dtype)
    for i, v in enumerate(values):
        if v is None:
            continue
        arr = np.asarray(v).reshape(-1)
        m = min(arr.size, n)
        out[i, :m] = arr[:m]
    return out


def _stack_bytes(values, n: int) -> np.ndarray:
    """Stack a list of (n,)-shape S1 byte rows into a (len, n) S1 array."""
    out = np.full((len(values), n), b"\x00", dtype="S1")
    for i, v in enumerate(values):
        if v is None:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind == "S":
            arr = v.reshape(-1)
        else:
            arr = np.asarray(v, dtype="S1").reshape(-1)
        m = min(arr.size, n)
        out[i, :m] = arr[:m]
    return out


_HK_TYPE_UNITS = {
    "raw_seconds": "s",
    "time": "s",
    "adc_mean": "ADC counts",
    "adc_rms": "ADC counts",
}


def _write_one_hk_type(
    parent: h5py.Group,
    type_id: int,
    rows,
) -> None:
    """Write one /housekeeping/type_<N> subgroup as a per-field table."""
    if not rows:
        return
    # Sort rows by raw_seconds for monotonic time axis.
    rows = sorted(rows, key=lambda r: r.raw_seconds)
    n = len(rows)
    g = parent.create_group(f"type_{type_id}")
    g.attrs["count"] = np.int64(n)

    # Common base columns (always present).
    upid = np.array([r.unique_packet_id for r in rows], dtype=np.int64)
    version = np.array([r.version for r in rows], dtype=np.int16)
    errors = np.array([r.errors for r in rows], dtype=np.int32)
    raw_seconds = np.array([r.raw_seconds for r in rows], dtype=np.float64)
    _create_dataset(g, "unique_packet_id", upid)
    _create_dataset(g, "version", version)
    _create_dataset(g, "errors", errors)
    ds = _create_dataset(g, "raw_seconds", raw_seconds)
    ds.attrs["units"] = "s"
    ds.attrs["long_name"] = "mission time of HK packet"

    # Per-type fields. Read each row's `fields` dict; missing fields default
    # to 0 / NaN as appropriate for the dtype below.
    if type_id == 0:
        time = np.array(
            [float(r.fields.get("time", 0.0) or 0.0) for r in rows],
            dtype=np.float64,
        )
        ds = _create_dataset(g, "time", time)
        ds.attrs["units"] = "s"
    elif type_id == 1:
        adc_min = _stack_vec([r.fields.get("adc_min") for r in rows], 4, np.int16)
        adc_max = _stack_vec([r.fields.get("adc_max") for r in rows], 4, np.int16)
        adc_mean = _stack_vec([r.fields.get("adc_mean") for r in rows], 4, np.float32)
        adc_rms = _stack_vec([r.fields.get("adc_rms") for r in rows], 4, np.float32)
        actual_gain = _stack_bytes([r.fields.get("actual_gain") for r in rows], 4)
        for name, arr, units in (
            ("adc_min", adc_min, "ADC counts"),
            ("adc_max", adc_max, "ADC counts"),
            ("adc_mean", adc_mean, "ADC counts"),
            ("adc_rms", adc_rms, "ADC counts"),
        ):
            ds = _create_dataset(g, name, arr)
            ds.attrs["units"] = units
        ds = _create_dataset(g, "actual_gain", actual_gain)
        ds.attrs["long_name"] = "per-channel gain code (L/M/H/A)"
    elif type_id == 2:
        time = np.array(
            [float(r.fields.get("time", 0.0) or 0.0) for r in rows],
            dtype=np.float64,
        )
        ds = _create_dataset(g, "time", time)
        ds.attrs["units"] = "s"
        ok = np.array(
            [int(bool(r.fields.get("ok", False))) for r in rows],
            dtype=np.uint8,
        )
        ds = _create_dataset(g, "ok", ok)
        ds.attrs["long_name"] = "request acknowledged flag"
        # Auto-discover the union of telemetry_* channels across all rows
        # in this session, then write one (n,) float32 dataset per channel,
        # NaN-filling rows that didn't carry that channel.
        telem_keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.fields:
                if k.startswith("telemetry") and k not in seen:
                    telem_keys.append(k)
                    seen.add(k)
        for k in telem_keys:
            arr = np.full(n, np.nan, dtype=np.float32)
            for i, r in enumerate(rows):
                v = r.fields.get(k)
                if v is None:
                    continue
                try:
                    arr[i] = float(v)
                except (TypeError, ValueError):
                    pass
            _create_dataset(g, k, arr)
    elif type_id == 3:
        checksum = np.array(
            [int(r.fields.get("checksum", 0) or 0) for r in rows],
            dtype=np.int64,
        )
        weight_ndx = np.array(
            [int(r.fields.get("weight_ndx", 0) or 0) for r in rows],
            dtype=np.int32,
        )
        _create_dataset(g, "checksum", checksum)
        _create_dataset(g, "weight_ndx", weight_ndx)
    else:
        # Unknown / deferred subtypes (e.g. 100, 101). Record only base
        # columns; do not try to interpret fields.
        log.info("housekeeping type %d not modeled; wrote base columns only", type_id)


def _write_housekeeping(h5: h5py.File, products: Products) -> None:
    if not products.housekeeping:
        return
    g = h5.create_group("housekeeping")
    g.attrs["count"] = np.int64(len(products.housekeeping))

    by_type: Dict[int, list] = {}
    for hk in products.housekeeping:
        by_type.setdefault(hk.hk_type, []).append(hk)
    for type_id in sorted(by_type.keys()):
        _write_one_hk_type(g, type_id, by_type[type_id])


# ---------------------------------------------------------------------------
# /DCB_telemetry and /spectra_interpolated_telemetry
# ---------------------------------------------------------------------------

def _write_dcb_telemetry(
    h5: h5py.File,
    fpga: Optional[Mapping[str, np.ndarray]],
    encoder: Optional[Mapping[str, np.ndarray]],
) -> None:
    if not fpga and not encoder:
        return
    g = h5.create_group("DCB_telemetry")
    if fpga:
        ms = fpga.get("mission_seconds")
        ss = fpga.get("lusee_subsecs")
        if ms is not None:
            _create_dataset(g, "fpga_mission_seconds", np.asarray(ms, dtype=np.float64))
        if ss is not None:
            _create_dataset(g, "fpga_lusee_subsecs", np.asarray(ss, dtype=np.float64))
        for fname, arr in fpga.items():
            if fname in ("mission_seconds", "lusee_subsecs"):
                continue
            _create_dataset(g, f"fpga_{fname}", np.asarray(arr, dtype=np.float64))
    if encoder:
        for k in ("mission_seconds", "lusee_subsecs"):
            arr = encoder.get(k)
            if arr is not None:
                _create_dataset(g, f"encoder_{k}", np.asarray(arr, dtype=np.float64))
        if encoder.get("enc_pos") is not None:
            _create_dataset(g, "enc_pos", np.asarray(encoder["enc_pos"], dtype=np.int64))
        if encoder.get("enc_status") is not None:
            _create_dataset(g, "enc_status", np.asarray(encoder["enc_status"], dtype=np.int64))


def _interpolate_telemetry(
    fpga: Mapping[str, np.ndarray],
    spectra_raw_times: np.ndarray,
    *,
    mode: str = "normalized",
) -> Dict[str, np.ndarray]:
    """Resample FPGA telemetry onto ``spectra_raw_times``. See spec section 8.10."""
    out: Dict[str, np.ndarray] = {}
    n_target = spectra_raw_times.size
    if n_target == 0:
        return out

    ms = np.asarray(fpga.get("mission_seconds", np.empty(0)), dtype=np.float64)
    ss = np.asarray(fpga.get("lusee_subsecs", np.empty(0)), dtype=np.float64)
    if ms.size == 0:
        return out
    tele_t = ms + ss * (1.0 / 65536.0)

    # Average duplicates.
    order = np.argsort(tele_t, kind="stable")
    tele_t = tele_t[order]
    uniq, inv = np.unique(tele_t, return_inverse=True)

    if mode == "normalized":
        if tele_t.size > 1:
            tt_min, tt_max = tele_t.min(), tele_t.max()
            sp_min, sp_max = spectra_raw_times.min(), spectra_raw_times.max()
        else:
            tt_min = tt_max = tele_t[0]
            sp_min = sp_max = spectra_raw_times[0]
        if tt_max > tt_min:
            xt = (uniq - tt_min) / (tt_max - tt_min)
        else:
            xt = np.zeros_like(uniq)
        if sp_max > sp_min:
            xs = (spectra_raw_times - sp_min) / (sp_max - sp_min)
        else:
            xs = np.zeros_like(spectra_raw_times)
    else:
        xt = uniq
        xs = spectra_raw_times

    for fname, vals in fpga.items():
        if fname in ("mission_seconds", "lusee_subsecs"):
            continue
        v = np.asarray(vals, dtype=np.float64)[order]
        avg = np.zeros_like(uniq)
        cnt = np.zeros_like(uniq)
        for i, j in enumerate(inv):
            avg[j] += v[i]
            cnt[j] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            avg = np.where(cnt > 0, avg / np.maximum(cnt, 1), 0.0)
        if uniq.size == 1:
            out[fname] = np.full(n_target, avg[0], dtype=np.float64)
        else:
            out[fname] = np.interp(xs, xt, avg)
    return out


def _write_interpolated_telemetry(
    h5: h5py.File,
    fpga: Optional[Mapping[str, np.ndarray]],
    products: Products,
    *,
    raw_subtract: float,
    mjd_offset: float,
    mode: str = "normalized",
) -> None:
    if not fpga or not products.spectra:
        return
    spec_raw = np.array([s.raw_seconds for s in products.spectra], dtype=np.float64)
    finite_mask = np.array([
        np.any(np.isfinite(s.data)) for s in products.spectra
    ])
    spec_raw = spec_raw[finite_mask]
    if spec_raw.size == 0:
        return
    interp = _interpolate_telemetry(fpga, spec_raw, mode=mode)
    if not interp:
        return
    g = h5.create_group("spectra_interpolated_telemetry")
    _create_dataset(g, "time", spec_raw)
    if raw_subtract or mjd_offset:
        _create_dataset(g, "mjd_time",
                        _to_mjd(spec_raw, raw_subtract, mjd_offset))
    for fname, arr in interp.items():
        _create_dataset(g, fname, arr)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_hdf5(
    products: Products,
    dest: Path | str,
    *,
    cdi_directory: Optional[Path | str] = None,
    fpga_telemetry: Optional[Mapping[str, np.ndarray]] = None,
    encoder_telemetry: Optional[Mapping[str, np.ndarray]] = None,
    interpolate_telemetry: bool = False,
    interpolation_mode: str = "normalized",
    lun_lat_deg: float = DEFAULT_LUN_LAT_DEG,
    lun_long_deg: float = DEFAULT_LUN_LONG_DEG,
    lun_height_m: float = DEFAULT_LUN_HEIGHT_M,
    raw_time_subtract_seconds: float = DEFAULT_RAW_TIME_SUBTRACT_SECONDS,
    mjd_epoch_offset_days: float = DEFAULT_MJD_EPOCH_OFFSET_DAYS,
) -> Path:
    """Write a ``Products`` instance + optional telemetry to ``dest`` (HDF5)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dest, "w") as h5:
        h5.attrs["cdi_directory"] = str(cdi_directory) if cdi_directory else ""
        h5.attrs["layout_version"] = np.int64(HDF5_LAYOUT_VERSION)
        h5.attrs["n_items"] = np.int64(0)

        _write_session_invariants(h5, products)
        _write_constants(
            h5,
            lun_lat_deg=lun_lat_deg,
            lun_long_deg=lun_long_deg,
            lun_height_m=lun_height_m,
            raw_time_subtract_seconds=raw_time_subtract_seconds,
            mjd_epoch_offset_days=mjd_epoch_offset_days,
        )
        _write_spectra(
            h5, products,
            raw_subtract=raw_time_subtract_seconds,
            mjd_offset=mjd_epoch_offset_days,
        )
        _write_tr_spectra(
            h5, products,
            raw_subtract=raw_time_subtract_seconds,
            mjd_offset=mjd_epoch_offset_days,
        )
        _write_zoom_spectra(
            h5, products,
            raw_subtract=raw_time_subtract_seconds,
            mjd_offset=mjd_epoch_offset_days,
        )
        _write_cal_data(h5, products)
        _write_grimm_spectra(
            h5, products,
            raw_subtract=raw_time_subtract_seconds,
            mjd_offset=mjd_epoch_offset_days,
        )
        _write_waveform(h5, products)
        _write_housekeeping(h5, products)
        _write_dcb_telemetry(h5, fpga_telemetry, encoder_telemetry)
        if interpolate_telemetry and fpga_telemetry:
            _write_interpolated_telemetry(
                h5, fpga_telemetry, products,
                raw_subtract=raw_time_subtract_seconds,
                mjd_offset=mjd_epoch_offset_days,
                mode=interpolation_mode,
            )
    log.info("wrote HDF5 %s", dest)
    return dest
