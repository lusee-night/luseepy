"""FITS writer that mirrors the HDF5 layout.

The HDU naming and column structure are designed to be a 1-to-1
translation of the HDF5 schema produced by :mod:`lusee.ingest.hdf5_writer`.
The mapping rules are:

  HDF5                                            FITS
  ----                                            ----
  group with attributes only                  ->  IMAGE HDU, NAXIS=0,
                                                  header keywords
  dense cube dataset (spectra cube etc.)      ->  IMAGE HDU
  group of (N,)-aligned datasets              ->  BINTABLE HDU, one
                                                  column per field
  per-channel subgroup (waveforms)            ->  one BINTABLE per channel

Same retention rule as HDF5: rows of `/spectra` and `/tr_spectra` whose
entire cube slice is NaN are dropped, so the FITS row counts match
exactly. All time-series tables use TUNITn for units where known.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from astropy.io import fits

from .constants import (
    BITSLICE_REFERENCE,
    DEFAULT_CLOCK_SOURCE,
    DEFAULT_LUN_HEIGHT_M,
    DEFAULT_LUN_LAT_DEG,
    DEFAULT_LUN_LONG_DEG,
    DEFAULT_MJD_EPOCH_OFFSET_DAYS,
    DEFAULT_RAW_TIME_SUBTRACT_SECONDS,
    DEFAULT_TIME_SCALE,
    KNOWN_TIME_SCALES,
    HDF5_LAYOUT_VERSION,
    NCHANNELS,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    SPECTRA_REPRESENTATION,
    SPECTRA_UNITS,
)
from .decode import Products
from .hdf5_writer import _interpolate_telemetry, _to_mjd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column construction helpers
# ---------------------------------------------------------------------------

def _format_for_dtype(dt: np.dtype) -> Optional[str]:
    """Map a numpy dtype to its FITS BINTABLE TFORM base character."""
    k = dt.kind
    if k == "f":
        return "E" if dt.itemsize == 4 else "D"
    if k in ("i", "u", "b"):
        if dt.itemsize <= 1:
            return "I"   # FITS has no signed int8; promote to int16
        if dt.itemsize == 2:
            return "I"
        if dt.itemsize == 4:
            return "J"
        return "K"
    return None


def _column(name: str, arr: np.ndarray, *,
            unit: Optional[str] = None,
            disp: Optional[str] = None) -> Optional[fits.Column]:
    """Build a fits.Column from a numpy array (1-D or N-D-per-row).

    Returns None if the dtype is not representable as a numeric / byte
    BINTABLE column.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind == "S":
        # Per-row bytes. Joining (N, k) S1 -> (N,) S<k> is the natural form.
        if arr.ndim == 2 and arr.dtype.itemsize == 1:
            joined = np.array([b"".join(row) for row in arr], dtype=f"S{arr.shape[1]}")
            tform = f"{arr.shape[1]}A"
            return fits.Column(name=name, format=tform, array=joined, unit=unit, disp=disp)
        if arr.ndim == 1:
            width = arr.dtype.itemsize
            tform = f"{width}A" if width > 1 else "1A"
            return fits.Column(name=name, format=tform, array=arr, unit=unit, disp=disp)
        return None

    base = _format_for_dtype(arr.dtype)
    if base is None:
        return None

    if arr.ndim == 1:
        return fits.Column(name=name, format=base, array=arr, unit=unit, disp=disp)

    per_row = int(np.prod(arr.shape[1:]))
    tform = f"{per_row}{base}"
    if arr.ndim == 2:
        return fits.Column(name=name, format=tform, array=arr, unit=unit, disp=disp)
    # Higher-dim per-row: use TDIM. FITS convention: fastest axis first.
    inner_dims = list(reversed(arr.shape[1:]))
    dim = "(" + ",".join(str(d) for d in inner_dims) + ")"
    return fits.Column(name=name, format=tform, array=arr, dim=dim,
                       unit=unit, disp=disp)


def _bintable_from_columns(cols: List[fits.Column], *,
                           name: str, extdesc: str = "") -> fits.BinTableHDU:
    hdu = fits.BinTableHDU.from_columns(cols, name=name)
    if extdesc:
        hdu.header["EXTDESC"] = extdesc
    return hdu


def _empty_image_hdu(name: str, *, kvs: Mapping[str, object],
                     extdesc: str = "") -> fits.ImageHDU:
    """An IMAGE HDU with NAXIS=0 carrying only header keywords."""
    hdu = fits.ImageHDU(data=None, name=name)
    if extdesc:
        hdu.header["EXTDESC"] = extdesc
    for k, v in kvs.items():
        if v is None:
            continue
        # FITS keyword cap is 8 chars; allow longer via HIERARCH.
        key = k if len(k) <= 8 else f"HIERARCH {k}"
        hdu.header[key] = v
    return hdu


# ---------------------------------------------------------------------------
# Per-section writers
# ---------------------------------------------------------------------------

def _spectra_retention(products: Products) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror of the HDF5 retention rule. Returns (cube[keep], unique_ids[keep],
    raw_times[keep], keep_indices)."""
    n = len(products.spectra)
    cube = np.full((n, NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(products.spectra):
        cube[i] = s.data
        unique_ids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
    finite_mask = np.any(np.isfinite(cube.reshape(n, -1)), axis=1)
    keep = np.flatnonzero(finite_mask)
    return cube[keep], unique_ids[keep], raw_times[keep], keep


def _tr_retention(products: Products):
    n = len(products.tr_spectra)
    if n == 0:
        return None
    navg2_max = max(s.navg2 for s in products.tr_spectra)
    tr_len_max = max(s.tr_length for s in products.tr_spectra)
    cube = np.full((n, NPRODUCTS, navg2_max, tr_len_max), np.nan, dtype=np.float32)
    unique_ids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    navg2_per = np.zeros(n, dtype=np.int64)
    tr_len_per = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(products.tr_spectra):
        cube[i, :, :s.navg2, :s.tr_length] = s.data
        unique_ids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
        navg2_per[i] = s.navg2
        tr_len_per[i] = s.tr_length
    finite_mask = np.any(np.isfinite(cube.reshape(n, -1)), axis=1)
    keep = np.flatnonzero(finite_mask)
    return (cube[keep], unique_ids[keep], raw_times[keep],
            navg2_per[keep], tr_len_per[keep], keep, navg2_max, tr_len_max)


def _stack_metadata(products_list, keep: np.ndarray) -> Dict[str, np.ndarray]:
    """Aggregate per-row metadata dicts into (N, ...) arrays then trim by keep."""
    bag: Dict[str, list] = {}
    n = len(products_list)
    for s in products_list:
        for k, v in s.metadata.items():
            bag.setdefault(k, []).append(v)
    out: Dict[str, np.ndarray] = {}
    for k, items in bag.items():
        if len(items) != n:
            continue
        try:
            arr = np.asarray(items)
        except Exception:    # noqa: BLE001
            continue
        if arr.size == 0 or arr.dtype.kind in ("U", "O"):
            continue
        out[k] = arr[keep]
    return out


def _times_columns(unique_ids, raw_times, raw_subtract, mjd_offset,
                   original_indices) -> List[fits.Column]:
    cols = [
        _column("UNIQUE_ID", unique_ids),
        _column("RAW_TIME", raw_times, unit="s"),
        _column("MJD_TIME", _to_mjd(raw_times, raw_subtract, mjd_offset), unit="d"),
        _column("ORIG_IDX", original_indices.astype(np.int64)),
    ]
    return [c for c in cols if c is not None]


def _build_spectra_hdus(products, raw_subtract, mjd_offset) -> List[fits.HDUList]:
    if not products.spectra:
        return []
    cube, uids, raw_times, keep = _spectra_retention(products)
    if cube.shape[0] == 0:
        return []
    hdus: List = []
    img = fits.ImageHDU(data=cube, name="SPECTRA")
    img.header["EXTDESC"] = "Spectra cube (n_time, n_product, n_channel)"
    img.header["NTIME"] = cube.shape[0]
    img.header["NPROD"] = cube.shape[1]
    img.header["NCHAN"] = cube.shape[2]
    img.header["BUNIT"] = SPECTRA_UNITS
    img.header["BITSREST"] = (True, "actual bit-slice restored at ingestion")
    img.header["BITSREF"] = BITSLICE_REFERENCE
    img.header["NORMVER"] = SPECTRA_NORMALIZATION_VERSION
    img.header["HIERARCH REPRESENTATION"] = SPECTRA_REPRESENTATION
    hdus.append(img)
    hdus.append(_bintable_from_columns(
        _times_columns(uids, raw_times, raw_subtract, mjd_offset, keep),
        name="SPECTRA_TIMES",
        extdesc="Per-row time/identity for /SPECTRA",
    ))
    md = _stack_metadata(products.spectra, keep)
    if md:
        cols = []
        for name, arr in md.items():
            col = _column(name[:68], arr)
            if col is not None:
                cols.append(col)
        if cols:
            hdus.append(_bintable_from_columns(
                cols, name="SPECTRA_META",
                extdesc="Per-row metadata for /SPECTRA",
            ))
    return hdus


def _build_tr_hdus(products, raw_subtract, mjd_offset) -> List:
    out = _tr_retention(products)
    if out is None:
        return []
    cube, uids, raw_times, navg2_per, tr_len_per, keep, navg2_max, tr_len_max = out
    if cube.shape[0] == 0:
        return []
    hdus: List = []
    img = fits.ImageHDU(data=cube, name="TR_SPECTRA")
    img.header["EXTDESC"] = "TR spectra (n_time, n_product, navg2, tr_length)"
    img.header["NTIME"] = cube.shape[0]
    img.header["NPROD"] = cube.shape[1]
    img.header["NAVG2"] = navg2_max
    img.header["TR_LEN"] = tr_len_max
    hdus.append(img)
    cols = _times_columns(uids, raw_times, raw_subtract, mjd_offset, keep)
    cols.append(_column("NAVG2", navg2_per.astype(np.int64)))
    cols.append(_column("TR_LEN", tr_len_per.astype(np.int64)))
    hdus.append(_bintable_from_columns(
        [c for c in cols if c is not None],
        name="TR_TIMES",
        extdesc="Per-row time/identity for /TR_SPECTRA",
    ))
    md = _stack_metadata(products.tr_spectra, keep)
    if md:
        cols2 = [c for c in (_column(name[:68], arr) for name, arr in md.items()) if c is not None]
        if cols2:
            hdus.append(_bintable_from_columns(
                cols2, name="TR_META",
                extdesc="Per-row metadata for /TR_SPECTRA",
            ))
    return hdus


def _build_zoom_hdus(products, raw_subtract, mjd_offset) -> List:
    if not products.zoom_spectra:
        return []
    n = len(products.zoom_spectra)
    cube = np.zeros((n, 4, 64), dtype=np.float32)
    uids = np.zeros(n, dtype=np.int64)
    pfb = np.zeros(n, dtype=np.int32)
    raw_times = np.zeros(n, dtype=np.float64)
    for i, z in enumerate(products.zoom_spectra):
        cube[i] = z.data
        uids[i] = z.unique_packet_id
        pfb[i] = z.pfb_index
        raw_times[i] = z.raw_seconds
    hdus = []
    img = fits.ImageHDU(data=cube, name="ZOOM_DATA")
    img.header["EXTDESC"] = "Calibrator zoom spectra (n_time, 4, 64)"
    img.header["NTIME"] = cube.shape[0]
    hdus.append(img)
    cols = _times_columns(uids, raw_times, raw_subtract, mjd_offset, np.arange(n, dtype=np.int64))
    cols.append(_column("PFB_IDX", pfb.astype(np.int32)))
    hdus.append(_bintable_from_columns(
        [c for c in cols if c is not None],
        name="ZOOM_TIMES",
        extdesc="Per-row time/identity for /ZOOM_DATA",
    ))
    return hdus


def _build_grimm_hdus(products, raw_subtract, mjd_offset) -> List:
    if not products.grimm_spectra:
        return []
    n = len(products.grimm_spectra)
    cube = np.full((n, NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    uids = np.zeros(n, dtype=np.int64)
    raw_times = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(products.grimm_spectra):
        cube[i] = s.data
        uids[i] = s.unique_packet_id
        raw_times[i] = s.raw_seconds
    hdus = []
    img = fits.ImageHDU(data=cube, name="GRIMM")
    img.header["EXTDESC"] = "Grimm spectra (n_time, n_product, n_channel)"
    hdus.append(img)
    hdus.append(_bintable_from_columns(
        _times_columns(uids, raw_times, raw_subtract, mjd_offset,
                       np.arange(n, dtype=np.int64)),
        name="GRIMM_TIMES",
        extdesc="Per-row time/identity for /GRIMM",
    ))
    return hdus


def _build_waveform_hdus(products) -> List:
    if not products.waveforms:
        return []
    by_channel: Dict[int, list] = {}
    for w in products.waveforms:
        by_channel.setdefault(w.channel, []).append(w)
    hdus = []
    for ch in sorted(by_channel.keys()):
        items = by_channel[ch]
        wf = np.stack([w.data for w in items]).astype(np.int16)
        ts = np.array([w.raw_seconds for w in items], dtype=np.float64)
        cols = [
            _column("WAVEFORM", wf),
            _column("TIMESTAMP", ts, unit="s"),
        ]
        hdu = _bintable_from_columns(
            [c for c in cols if c is not None],
            name=f"WF_CH{ch}",
            extdesc=f"Raw ADC waveforms, channel {ch}",
        )
        hdu.header["CHANNEL"] = int(ch)
        hdu.header["WFLEN"] = int(wf.shape[1])
        hdus.append(hdu)
    return hdus


def _build_housekeeping_hdus(products) -> List:
    if not products.housekeeping:
        return []
    by_type: Dict[int, list] = {}
    for hk in products.housekeeping:
        by_type.setdefault(hk.hk_type, []).append(hk)
    hdus = []
    for type_id in sorted(by_type.keys()):
        rows = sorted(by_type[type_id], key=lambda r: r.raw_seconds)
        n = len(rows)
        cols: List[fits.Column] = []
        cols.append(_column("UPID", np.array([r.unique_packet_id for r in rows], dtype=np.int64)))
        cols.append(_column("VERSION", np.array([r.version for r in rows], dtype=np.int16)))
        cols.append(_column("ERRORS", np.array([r.errors for r in rows], dtype=np.int32)))
        cols.append(_column("RAW_TIME", np.array([r.raw_seconds for r in rows], dtype=np.float64), unit="s"))

        if type_id == 0:
            cols.append(_column("TIME", np.array([float(r.fields.get("time", 0.0) or 0.0) for r in rows], dtype=np.float64), unit="s"))
        elif type_id == 1:
            for name in ("adc_min", "adc_max"):
                arr = np.zeros((n, 4), dtype=np.int16)
                for i, r in enumerate(rows):
                    v = r.fields.get(name)
                    if v is not None:
                        m = min(np.asarray(v).size, 4)
                        arr[i, :m] = np.asarray(v).reshape(-1)[:m]
                cols.append(_column(name.upper(), arr, unit="ADC counts"))
            for name in ("adc_mean", "adc_rms"):
                arr = np.zeros((n, 4), dtype=np.float32)
                for i, r in enumerate(rows):
                    v = r.fields.get(name)
                    if v is not None:
                        m = min(np.asarray(v).size, 4)
                        arr[i, :m] = np.asarray(v).reshape(-1)[:m]
                cols.append(_column(name.upper(), arr, unit="ADC counts"))
            ag = np.full((n, 4), b"\x00", dtype="S1")
            for i, r in enumerate(rows):
                v = r.fields.get("actual_gain")
                if v is None:
                    continue
                arr = np.asarray(v).reshape(-1) if not isinstance(v, np.ndarray) else v.reshape(-1)
                m = min(arr.size, 4)
                ag[i, :m] = arr[:m]
            cols.append(_column("ACT_GAIN", ag, disp="A4"))
        elif type_id == 2:
            cols.append(_column("TIME", np.array([float(r.fields.get("time", 0.0) or 0.0) for r in rows], dtype=np.float64), unit="s"))
            cols.append(_column("OK", np.array([int(bool(r.fields.get("ok", False))) for r in rows], dtype=np.int16)))
            telem_keys: List[str] = []
            seen: set = set()
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
                # FITS column names cap at 68 chars; the channel name is
                # already short enough.
                cols.append(_column(k[:68], arr))
        elif type_id == 3:
            cols.append(_column("CHECKSUM", np.array([int(r.fields.get("checksum", 0) or 0) for r in rows], dtype=np.int64)))
            cols.append(_column("WGT_NDX", np.array([int(r.fields.get("weight_ndx", 0) or 0) for r in rows], dtype=np.int32)))

        cols = [c for c in cols if c is not None]
        hdu = _bintable_from_columns(
            cols, name=f"HK_T{type_id}",
            extdesc=f"Housekeeping records, type {type_id}",
        )
        hdu.header["HKTYPE"] = type_id
        hdus.append(hdu)
    return hdus


def _build_dcb_telemetry_hdus(fpga, encoder) -> List:
    hdus: List = []
    if fpga:
        ms = np.asarray(fpga.get("mission_seconds", np.empty(0)), dtype=np.float64)
        ss = np.asarray(fpga.get("lusee_subsecs", np.empty(0)), dtype=np.float64)
        if ms.size:
            cols = [_column("MS", ms, unit="s"), _column("SUBSEC", ss, unit="ticks")]
            for fname, arr in fpga.items():
                if fname in ("mission_seconds", "lusee_subsecs"):
                    continue
                col = _column(fname[:68], np.asarray(arr, dtype=np.float64))
                if col is not None:
                    cols.append(col)
            cols = [c for c in cols if c is not None]
            hdu = _bintable_from_columns(
                cols, name="DCB_FPGA",
                extdesc="DCB FPGA telemetry time series",
            )
            hdus.append(hdu)
    if encoder and encoder.get("mission_seconds") is not None:
        ms = np.asarray(encoder["mission_seconds"], dtype=np.float64)
        ss = np.asarray(encoder.get("lusee_subsecs", np.zeros_like(ms)), dtype=np.float64)
        cols = [
            _column("MS", ms, unit="s"),
            _column("SUBSEC", ss, unit="ticks"),
            _column("ENC_POS", np.asarray(encoder.get("enc_pos", np.zeros_like(ms)), dtype=np.int64)),
            _column("ENC_STAT", np.asarray(encoder.get("enc_status", np.zeros_like(ms)), dtype=np.int64)),
        ]
        cols = [c for c in cols if c is not None]
        hdus.append(_bintable_from_columns(
            cols, name="DCB_ENC",
            extdesc="DCB encoder telemetry time series",
        ))
    return hdus


def _build_interp_telemetry_hdu(fpga, products, raw_subtract, mjd_offset, mode) -> List:
    if not fpga or not products.spectra:
        return []
    spec_raw = np.array([s.raw_seconds for s in products.spectra], dtype=np.float64)
    finite_mask = np.array([np.any(np.isfinite(s.data)) for s in products.spectra])
    spec_raw = spec_raw[finite_mask]
    if spec_raw.size == 0:
        return []
    interp = _interpolate_telemetry(fpga, spec_raw, mode=mode)
    if not interp:
        return []
    cols = [
        _column("RAW_TIME", spec_raw, unit="s"),
        _column("MJD_TIME", _to_mjd(spec_raw, raw_subtract, mjd_offset), unit="d"),
    ]
    for k, arr in interp.items():
        col = _column(k[:68], np.asarray(arr, dtype=np.float64))
        if col is not None:
            cols.append(col)
    cols = [c for c in cols if c is not None]
    return [_bintable_from_columns(
        cols, name="SPEC_INTERP",
        extdesc="FPGA telemetry interpolated onto /SPECTRA time axis",
    )]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_fits(
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
    time_scale: str = DEFAULT_TIME_SCALE,
    clock_source: str = DEFAULT_CLOCK_SOURCE,
    clock_epoch_isot: Optional[str] = None,
) -> Path:
    """Write a ``Products`` instance + optional telemetry to ``dest`` (FITS).

    The HDU layout mirrors the HDF5 schema: the same product cubes, the
    same per-row identity tables, and the same per-type housekeeping
    tables, just expressed as IMAGE / BINTABLE HDUs.  Time provenance
    (``time_scale``/``clock_source``/``clock_epoch_isot``) lands in the
    CONSTANTS HDU as TIMESYS/CLKSRC/CLKEPOCH.
    """
    if time_scale not in KNOWN_TIME_SCALES:
        raise ValueError(
            f"time_scale must be one of {KNOWN_TIME_SCALES}; got {time_scale!r}"
        )
    # Keep FITS and HDF5 views of the same Products object in the same SDU
    # state.  TR and Grimm products are deliberately not normalized here.
    products.restore_spectra_bitslices()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Primary HDU: file-level provenance + the HDF5-equivalent root attrs.
    primary = fits.PrimaryHDU()
    primary.header["LAYOUTV"] = HDF5_LAYOUT_VERSION
    primary.header["ORIGIN"] = "lusee.ingest.fits_writer"
    if cdi_directory is not None:
        primary.header["CDI_DIR"] = str(cdi_directory)[:68]

    hdus: List = [primary]

    hdus.append(_empty_image_hdu(
        "SESSION_INV",
        kvs={
            "SW_VERS":  products.sw_version,
            "FW_VERS":  products.fw_version,
            "FW_ID":    products.fw_id,
            "FW_DATE":  products.fw_date,
            "FW_TIME":  products.fw_time,
            "ST_UPID":  products.start_unique_packet_id,
            "ST_T32":   products.start_time_32,
            "ST_T16":   products.start_time_16,
        },
        extdesc="Session-invariant identity from Hello packet",
    ))
    hdus.append(_empty_image_hdu(
        "CONSTANTS",
        kvs={
            "LUN_LAT":  lun_lat_deg,
            "LUN_LON":  lun_long_deg,
            "LUN_HGT":  lun_height_m,
            "RAWSHFT":  raw_time_subtract_seconds,
            "MJDOFF":   mjd_epoch_offset_days,
            # FITS convention spells time-scale values uppercase; the
            # reader lowercases on the way back in
            "TIMESYS":  time_scale.upper(),
            "CLKSRC":   clock_source,
            "CLKEPOCH": clock_epoch_isot,
        },
        extdesc="Lunar landing coordinates and MJD calibration",
    ))

    hdus.extend(_build_spectra_hdus(products, raw_time_subtract_seconds, mjd_epoch_offset_days))
    hdus.extend(_build_tr_hdus(products, raw_time_subtract_seconds, mjd_epoch_offset_days))
    hdus.extend(_build_zoom_hdus(products, raw_time_subtract_seconds, mjd_epoch_offset_days))
    hdus.extend(_build_grimm_hdus(products, raw_time_subtract_seconds, mjd_epoch_offset_days))
    hdus.extend(_build_waveform_hdus(products))
    hdus.extend(_build_housekeeping_hdus(products))
    hdus.extend(_build_dcb_telemetry_hdus(fpga_telemetry, encoder_telemetry))
    if interpolate_telemetry and fpga_telemetry:
        hdus.extend(_build_interp_telemetry_hdu(
            fpga_telemetry, products,
            raw_time_subtract_seconds, mjd_epoch_offset_days,
            interpolation_mode,
        ))

    fits.HDUList(hdus).writeto(dest, overwrite=True)
    log.info("wrote FITS %s", dest)
    return dest
