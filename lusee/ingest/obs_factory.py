"""Factory that builds a :class:`lusee.Observation` (specifically an
:class:`IngestData`) from one or more HDF5 / FITS files written by
:mod:`lusee.ingest`.

Inputs may be:

* a single file path (HDF5 or FITS);
* a single directory; the directory is walked recursively and HDF5 files
  win over FITS when both are present (override with ``prefer_format``);
* an iterable of any of the above.

Sessions are loaded, sorted by start time, and concatenated along the
spectra time axis. The result is a single ``IngestData`` whose
``self.times`` carries the actual irregular sample times (not a uniform
synthetic grid), so :class:`Observation`'s ``get_track_*`` methods work
out of the box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.time import TimeDelta
from lunarsky.time import Time as LunarTime

from lusee.Observation import Observation

from .constants import NCHANNELS, NPRODUCTS

log = logging.getLogger(__name__)

# Pixel→MHz conversion at full resolution (Navgf=1) -- mirrors uncrater.
_FREQ_STEP_MHZ_NAVGF1 = 0.025


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

@dataclass
class SessionBundle:
    """In-memory mirror of one HDF5/FITS session, format-agnostic."""

    spectra: Optional[np.ndarray] = None
    spectra_unique_ids: Optional[np.ndarray] = None
    spectra_raw_times: Optional[np.ndarray] = None
    spectra_mjd_times: Optional[np.ndarray] = None
    spectra_metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    tr_spectra: Optional[np.ndarray] = None
    tr_unique_ids: Optional[np.ndarray] = None
    tr_raw_times: Optional[np.ndarray] = None
    tr_mjd_times: Optional[np.ndarray] = None
    tr_navg2_per_sample: Optional[np.ndarray] = None
    tr_length_per_sample: Optional[np.ndarray] = None
    tr_metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    zoom_spectra: Optional[np.ndarray] = None
    zoom_unique_ids: Optional[np.ndarray] = None
    zoom_pfb_indices: Optional[np.ndarray] = None
    zoom_raw_times: Optional[np.ndarray] = None
    zoom_mjd_times: Optional[np.ndarray] = None

    grimm_spectra: Optional[np.ndarray] = None
    grimm_unique_ids: Optional[np.ndarray] = None
    grimm_raw_times: Optional[np.ndarray] = None

    waveforms: Dict[int, np.ndarray] = field(default_factory=dict)
    waveform_times: Dict[int, np.ndarray] = field(default_factory=dict)

    housekeeping: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)

    dcb_fpga: Dict[str, np.ndarray] = field(default_factory=dict)
    dcb_encoder: Dict[str, np.ndarray] = field(default_factory=dict)
    interp_telemetry: Dict[str, np.ndarray] = field(default_factory=dict)

    session_invariants: Dict[str, Any] = field(default_factory=dict)
    constants: Dict[str, float] = field(default_factory=dict)

    source_path: Optional[Path] = None
    layout_version: Optional[int] = None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_H5_EXTS = (".h5", ".hdf5")
_FITS_EXTS = (".fits", ".fit")


def _is_h5(p: Path) -> bool:
    return p.suffix.lower() in _H5_EXTS


def _is_fits(p: Path) -> bool:
    return p.suffix.lower() in _FITS_EXTS


def _resolve_one(target: Path, *, prefer_format: str) -> List[Path]:
    """Resolve a single path entry to a list of files."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)

    primary_exts = _H5_EXTS if prefer_format == "h5" else _FITS_EXTS
    fallback_exts = _FITS_EXTS if prefer_format == "h5" else _H5_EXTS

    primary = sorted(p for ext in primary_exts for p in target.rglob(f"*{ext}"))
    if primary:
        return primary
    fallback = sorted(p for ext in fallback_exts for p in target.rglob(f"*{ext}"))
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"no .h5 / .hdf5 / .fits files found under {target}"
    )


def _resolve_paths(target, *, prefer_format: str = "h5") -> List[Path]:
    if isinstance(target, (str, Path)):
        items: List[Path] = [Path(target)]
    else:
        items = [Path(p) for p in target]

    out: List[Path] = []
    for t in items:
        out.extend(_resolve_one(t, prefer_format=prefer_format))
    if not out:
        raise FileNotFoundError("no input files resolved")
    return out


# ---------------------------------------------------------------------------
# HDF5 reader
# ---------------------------------------------------------------------------

def _load_h5(path: Path) -> SessionBundle:
    import h5py

    bundle = SessionBundle(source_path=path)
    with h5py.File(path, "r") as f:
        if "layout_version" in f.attrs:
            bundle.layout_version = int(f.attrs["layout_version"])

        # session_invariants
        if "session_invariants" in f:
            for k, v in f["session_invariants"].attrs.items():
                bundle.session_invariants[k] = _scalarize(v)

        # constants
        if "constants" in f:
            for k, v in f["constants"].attrs.items():
                bundle.constants[k] = float(_scalarize(v))

        # spectra
        if "spectra" in f and "data" in f["spectra"]:
            sp = f["spectra"]
            bundle.spectra = sp["data"][...]
            bundle.spectra_unique_ids = _read_or_none(sp, "unique_ids")
            bundle.spectra_raw_times = _read_or_none(sp, "raw_times")
            bundle.spectra_mjd_times = _read_or_none(sp, "mjd_times")
            if "metadata" in sp:
                for k, ds in sp["metadata"].items():
                    bundle.spectra_metadata[k] = ds[...]

        # tr_spectra
        if "tr_spectra" in f and "data" in f["tr_spectra"]:
            tr = f["tr_spectra"]
            bundle.tr_spectra = tr["data"][...]
            bundle.tr_unique_ids = _read_or_none(tr, "unique_ids")
            bundle.tr_raw_times = _read_or_none(tr, "raw_times")
            bundle.tr_mjd_times = _read_or_none(tr, "mjd_times")
            bundle.tr_navg2_per_sample = _read_or_none(tr, "navg2_per_sample")
            bundle.tr_length_per_sample = _read_or_none(tr, "tr_length_per_sample")
            if "metadata" in tr:
                for k, ds in tr["metadata"].items():
                    bundle.tr_metadata[k] = ds[...]

        # zoom_spectra
        zs_path = "calibrator/zoom_spectra"
        if zs_path in f and "data" in f[zs_path]:
            zs = f[zs_path]
            bundle.zoom_spectra = zs["data"][...]
            bundle.zoom_unique_ids = _read_or_none(zs, "unique_ids")
            bundle.zoom_pfb_indices = _read_or_none(zs, "pfb_indices")
            bundle.zoom_raw_times = _read_or_none(zs, "raw_times")
            bundle.zoom_mjd_times = _read_or_none(zs, "mjd_times")

        # grimm
        if "grimm_spectra" in f and "data" in f["grimm_spectra"]:
            gr = f["grimm_spectra"]
            bundle.grimm_spectra = gr["data"][...]
            bundle.grimm_unique_ids = _read_or_none(gr, "unique_ids")
            bundle.grimm_raw_times = _read_or_none(gr, "raw_times")

        # waveform/channel_<N>/{waveforms, timestamps}
        if "waveform" in f:
            for ch_name, gch in f["waveform"].items():
                if not ch_name.startswith("channel_"):
                    continue
                ch = int(ch_name.split("_", 1)[1])
                if "waveforms" in gch:
                    bundle.waveforms[ch] = gch["waveforms"][...]
                if "timestamps" in gch:
                    bundle.waveform_times[ch] = gch["timestamps"][...]

        # housekeeping/type_<N>/<field>
        if "housekeeping" in f:
            for tname, gtype in f["housekeeping"].items():
                if not tname.startswith("type_"):
                    continue
                type_id = int(tname.split("_", 1)[1])
                bundle.housekeeping[type_id] = {
                    k: ds[...] for k, ds in gtype.items()
                }

        # DCB telemetry
        if "DCB_telemetry" in f:
            g = f["DCB_telemetry"]
            for k, ds in g.items():
                arr = ds[...]
                if k.startswith("fpga_"):
                    bundle.dcb_fpga[k[len("fpga_"):]] = arr
                elif k.startswith("encoder_"):
                    bundle.dcb_encoder[k[len("encoder_"):]] = arr
                elif k in ("enc_pos", "enc_status"):
                    bundle.dcb_encoder[k] = arr

        # spectra_interpolated_telemetry
        if "spectra_interpolated_telemetry" in f:
            for k, ds in f["spectra_interpolated_telemetry"].items():
                bundle.interp_telemetry[k] = ds[...]

    return bundle


def _read_or_none(g, name: str):
    return g[name][...] if name in g else None


def _scalarize(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, np.ndarray):
        return v
    return v


# ---------------------------------------------------------------------------
# FITS reader
# ---------------------------------------------------------------------------

def _load_fits(path: Path) -> SessionBundle:
    from astropy.io import fits

    bundle = SessionBundle(source_path=path)
    with fits.open(path) as hdul:
        primary = hdul[0]
        if "LAYOUTV" in primary.header:
            bundle.layout_version = int(primary.header["LAYOUTV"])

        # SESSION_INV / CONSTANTS: header keywords
        if "SESSION_INV" in [h.name for h in hdul]:
            inv = hdul["SESSION_INV"].header
            for k in ("SW_VERS", "FW_VERS", "FW_ID", "FW_DATE", "FW_TIME",
                     "ST_UPID", "ST_T32", "ST_T16"):
                if k in inv:
                    bundle.session_invariants[_inv_key_to_h5(k)] = int(inv[k])
        if "CONSTANTS" in [h.name for h in hdul]:
            cst = hdul["CONSTANTS"].header
            for k_fits, k_h5 in (
                ("LUN_LAT", "lun_lat_deg"),
                ("LUN_LON", "lun_long_deg"),
                ("LUN_HGT", "lun_height_m"),
                ("RAWSHFT", "raw_time_subtract_seconds"),
                ("MJDOFF", "mjd_epoch_offset_days"),
            ):
                if k_fits in cst:
                    bundle.constants[k_h5] = float(cst[k_fits])

        names = [h.name for h in hdul]

        if "SPECTRA" in names:
            bundle.spectra = np.asarray(hdul["SPECTRA"].data, dtype=np.float32)
        if "SPECTRA_TIMES" in names:
            t = hdul["SPECTRA_TIMES"].data
            bundle.spectra_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.spectra_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.spectra_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
        if "SPECTRA_META" in names:
            t = hdul["SPECTRA_META"].data
            for n in t.dtype.names:
                bundle.spectra_metadata[n] = np.asarray(t[n])

        if "TR_SPECTRA" in names:
            bundle.tr_spectra = np.asarray(hdul["TR_SPECTRA"].data, dtype=np.float32)
        if "TR_TIMES" in names:
            t = hdul["TR_TIMES"].data
            bundle.tr_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.tr_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.tr_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
            if "NAVG2" in t.dtype.names:
                bundle.tr_navg2_per_sample = np.asarray(t["NAVG2"])
            if "TR_LEN" in t.dtype.names:
                bundle.tr_length_per_sample = np.asarray(t["TR_LEN"])
        if "TR_META" in names:
            t = hdul["TR_META"].data
            for n in t.dtype.names:
                bundle.tr_metadata[n] = np.asarray(t[n])

        if "ZOOM_DATA" in names:
            bundle.zoom_spectra = np.asarray(hdul["ZOOM_DATA"].data, dtype=np.float32)
        if "ZOOM_TIMES" in names:
            t = hdul["ZOOM_TIMES"].data
            bundle.zoom_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.zoom_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.zoom_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
            if "PFB_IDX" in t.dtype.names:
                bundle.zoom_pfb_indices = np.asarray(t["PFB_IDX"])

        if "GRIMM" in names:
            bundle.grimm_spectra = np.asarray(hdul["GRIMM"].data, dtype=np.float32)
        if "GRIMM_TIMES" in names:
            t = hdul["GRIMM_TIMES"].data
            bundle.grimm_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.grimm_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)

        # Waveforms: WF_CH<N>
        for h in hdul:
            if h.name.startswith("WF_CH"):
                ch = int(h.name[len("WF_CH"):])
                t = h.data
                if t is None:
                    continue
                bundle.waveforms[ch] = np.asarray(t["WAVEFORM"], dtype=np.int16)
                bundle.waveform_times[ch] = np.asarray(t["TIMESTAMP"], dtype=np.float64)

        # Housekeeping HK_T<N>
        for h in hdul:
            if h.name.startswith("HK_T"):
                try:
                    type_id = int(h.name[len("HK_T"):])
                except ValueError:
                    continue
                t = h.data
                if t is None:
                    continue
                bundle.housekeeping[type_id] = {
                    n: np.asarray(t[n]) for n in t.dtype.names
                }

        if "DCB_FPGA" in names:
            t = hdul["DCB_FPGA"].data
            for n in t.dtype.names:
                if n == "MS":
                    bundle.dcb_fpga["mission_seconds"] = np.asarray(t[n], dtype=np.float64)
                elif n == "SUBSEC":
                    bundle.dcb_fpga["lusee_subsecs"] = np.asarray(t[n], dtype=np.float64)
                else:
                    bundle.dcb_fpga[n] = np.asarray(t[n])

        if "DCB_ENC" in names:
            t = hdul["DCB_ENC"].data
            mapping = {"MS": "mission_seconds", "SUBSEC": "lusee_subsecs",
                       "ENC_POS": "enc_pos", "ENC_STAT": "enc_status"}
            for n in t.dtype.names:
                bundle.dcb_encoder[mapping.get(n, n)] = np.asarray(t[n])

        if "SPEC_INTERP" in names:
            t = hdul["SPEC_INTERP"].data
            for n in t.dtype.names:
                bundle.interp_telemetry[n] = np.asarray(t[n])

    return bundle


def _inv_key_to_h5(fits_key: str) -> str:
    """Map a SESSION_INV FITS keyword back to the HDF5 attr name."""
    return {
        "SW_VERS":  "software_version",
        "FW_VERS":  "firmware_version",
        "FW_ID":    "firmware_id",
        "FW_DATE":  "firmware_date",
        "FW_TIME":  "firmware_time",
        "ST_UPID":  "start_unique_packet_id",
        "ST_T32":   "start_time_32",
        "ST_T16":   "start_time_16",
    }.get(fits_key, fits_key)


# ---------------------------------------------------------------------------
# Bundle loading dispatch
# ---------------------------------------------------------------------------

def _load_one(path: Path) -> SessionBundle:
    if _is_h5(path):
        return _load_h5(path)
    if _is_fits(path):
        return _load_fits(path)
    raise ValueError(f"unrecognized file extension: {path}")


# ---------------------------------------------------------------------------
# Concatenation helpers
# ---------------------------------------------------------------------------

def _concat_optional(arrs: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [a for a in arrs if a is not None and a.size > 0]
    if not valid:
        return None
    return np.concatenate(valid, axis=0)


def _pad_to(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Pad ``arr`` along axes 1..ndim to match ``target_shape`` with NaN."""
    if arr.shape[1:] == target_shape:
        return arr
    pad_widths = [(0, 0)] + [
        (0, t - s) for s, t in zip(arr.shape[1:], target_shape)
    ]
    return np.pad(arr, pad_widths, mode="constant", constant_values=np.nan)


def _concat_dict_arrays(
    dicts: Sequence[Dict[str, np.ndarray]],
    *,
    n_per_source: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Concatenate per-field dicts. Missing fields in a source are NaN-filled."""
    keys = sorted({k for d in dicts for k in d})
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        chunks = []
        for d, n in zip(dicts, n_per_source):
            if k in d:
                chunks.append(d[k])
            else:
                ref = next(iter(d.values()), None)
                if ref is None:
                    continue
                tail = ref.shape[1:]
                fill = np.full((n,) + tail, np.nan,
                               dtype=np.float64 if ref.dtype.kind == "f" else ref.dtype)
                chunks.append(fill)
        if chunks:
            out[k] = np.concatenate(chunks, axis=0)
    return out


def _merge_housekeeping(
    bundles: Sequence[SessionBundle],
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    type_ids = sorted({tid for b in bundles for tid in b.housekeeping})
    for tid in type_ids:
        sub_dicts = [b.housekeeping.get(tid, {}) for b in bundles]
        ns = [
            (next(iter(d.values())).shape[0] if d else 0)
            for d in sub_dicts
        ]
        out[tid] = _concat_dict_arrays(sub_dicts, n_per_source=ns)
    return out


def _concat_bundles(bundles: Sequence[SessionBundle]) -> SessionBundle:
    """Combine multiple SessionBundles into one, sorted by start time."""
    if len(bundles) == 1:
        return bundles[0]

    # Sort by first raw_time (or mjd_time if available)
    def _key(b: SessionBundle) -> float:
        if b.spectra_raw_times is not None and b.spectra_raw_times.size > 0:
            return float(b.spectra_raw_times[0])
        return 0.0
    bundles = sorted(bundles, key=_key)

    out = SessionBundle()
    out.source_path = None  # multi-source

    # Spectra
    out.spectra = _concat_optional([b.spectra for b in bundles])
    out.spectra_unique_ids = _concat_optional([b.spectra_unique_ids for b in bundles])
    out.spectra_raw_times = _concat_optional([b.spectra_raw_times for b in bundles])
    if all(b.spectra_mjd_times is not None for b in bundles):
        out.spectra_mjd_times = _concat_optional([b.spectra_mjd_times for b in bundles])

    n_per = [
        b.spectra.shape[0] if b.spectra is not None else 0 for b in bundles
    ]
    out.spectra_metadata = _concat_dict_arrays(
        [b.spectra_metadata for b in bundles], n_per_source=n_per
    )

    # TR spectra: pad each to common (max_navg2, max_tr_length)
    if any(b.tr_spectra is not None for b in bundles):
        present = [b for b in bundles if b.tr_spectra is not None]
        max_p = max(b.tr_spectra.shape[1] for b in present)
        max_n2 = max(b.tr_spectra.shape[2] for b in present)
        max_tl = max(b.tr_spectra.shape[3] for b in present)
        padded = [_pad_to(b.tr_spectra, (max_p, max_n2, max_tl)) for b in present]
        out.tr_spectra = np.concatenate(padded, axis=0)
        out.tr_unique_ids = _concat_optional([b.tr_unique_ids for b in present])
        out.tr_raw_times = _concat_optional([b.tr_raw_times for b in present])
        if all(b.tr_mjd_times is not None for b in present):
            out.tr_mjd_times = _concat_optional([b.tr_mjd_times for b in present])
        out.tr_navg2_per_sample = _concat_optional([b.tr_navg2_per_sample for b in present])
        out.tr_length_per_sample = _concat_optional([b.tr_length_per_sample for b in present])
        n_tr = [b.tr_spectra.shape[0] for b in present]
        out.tr_metadata = _concat_dict_arrays(
            [b.tr_metadata for b in present], n_per_source=n_tr
        )

    # Zoom
    out.zoom_spectra = _concat_optional([b.zoom_spectra for b in bundles])
    out.zoom_unique_ids = _concat_optional([b.zoom_unique_ids for b in bundles])
    out.zoom_pfb_indices = _concat_optional([b.zoom_pfb_indices for b in bundles])
    out.zoom_raw_times = _concat_optional([b.zoom_raw_times for b in bundles])
    if all(b.zoom_mjd_times is not None for b in bundles):
        out.zoom_mjd_times = _concat_optional([b.zoom_mjd_times for b in bundles])

    # Grimm
    out.grimm_spectra = _concat_optional([b.grimm_spectra for b in bundles])
    out.grimm_unique_ids = _concat_optional([b.grimm_unique_ids for b in bundles])
    out.grimm_raw_times = _concat_optional([b.grimm_raw_times for b in bundles])

    # Waveforms by channel
    chans = sorted({c for b in bundles for c in b.waveforms})
    for c in chans:
        chunks = [b.waveforms[c] for b in bundles if c in b.waveforms]
        ts_chunks = [b.waveform_times[c] for b in bundles if c in b.waveform_times]
        if chunks:
            out.waveforms[c] = np.concatenate(chunks, axis=0)
        if ts_chunks:
            out.waveform_times[c] = np.concatenate(ts_chunks, axis=0)

    # Housekeeping
    out.housekeeping = _merge_housekeeping(bundles)

    # DCB telemetry
    n_fpga = [
        (b.dcb_fpga["mission_seconds"].size
         if "mission_seconds" in b.dcb_fpga else 0)
        for b in bundles
    ]
    out.dcb_fpga = _concat_dict_arrays(
        [b.dcb_fpga for b in bundles], n_per_source=n_fpga,
    )
    n_enc = [
        (b.dcb_encoder["mission_seconds"].size
         if "mission_seconds" in b.dcb_encoder else 0)
        for b in bundles
    ]
    out.dcb_encoder = _concat_dict_arrays(
        [b.dcb_encoder for b in bundles], n_per_source=n_enc,
    )

    # Interpolated telemetry: keep only if all sessions had it
    if all(b.interp_telemetry for b in bundles):
        out.interp_telemetry = _concat_dict_arrays(
            [b.interp_telemetry for b in bundles], n_per_source=n_per,
        )

    # Constants and session-invariants: take from first; warn on disagreement
    out.constants = dict(bundles[0].constants)
    for b in bundles[1:]:
        for k, v in b.constants.items():
            if k in out.constants and out.constants[k] != v:
                log.warning("constant %s disagrees across sessions: %s vs %s; "
                            "keeping first", k, out.constants[k], v)
    out.session_invariants = dict(bundles[0].session_invariants)
    out.layout_version = bundles[0].layout_version

    return out


# ---------------------------------------------------------------------------
# IngestData
# ---------------------------------------------------------------------------

# (i, j) -> (real_index, imag_index_or_None) inside the (16,) product axis.
_PRODUCT_INDEX = {
    (0, 0): (0, None),
    (1, 1): (1, None),
    (2, 2): (2, None),
    (3, 3): (3, None),
    (0, 1): (4, 5),
    (0, 2): (6, 7),
    (0, 3): (8, 9),
    (1, 2): (10, 11),
    (1, 3): (12, 13),
    (2, 3): (14, 15),
}


def _resolve_combination(comb) -> Tuple[int, int, str, float, bool]:
    """Translate a 'NNX' string or (i, j[, x]) tuple to a normalized record.

    Returns ``(i, j, mode, sign, conjugate)`` with ``i <= j`` (symmetric
    access ``i > j`` is mapped to its mirror plus sign/conjugate
    adjustments: real channels are unchanged, imaginary channels are
    negated, complex channels are conjugated).
    """
    sign = 1.0
    if isinstance(comb, str):
        s = comb
        if s.startswith("-"):
            sign = -1.0
            s = s[1:]
        if len(s) < 2:
            raise ValueError(f"combination must be at least 'NN': {comb!r}")
        i = int(s[0]); j = int(s[1])
        mode = s[2:] if len(s) >= 3 else ("R" if i == j else "C")
    else:
        if len(comb) == 2:
            i, j = comb
            mode = "R" if i == j else "C"
        elif len(comb) == 3:
            i, j, mode = comb
        else:
            raise ValueError(f"combination tuple must have 2 or 3 elements: {comb!r}")
        i, j = int(i), int(j)
    conjugate = False
    if i > j:
        i, j = j, i
        if mode.startswith("I"):
            sign = -sign
        elif mode.startswith("C"):
            conjugate = True
    return i, j, mode, sign, conjugate


class IngestData(Observation):
    """An :class:`Observation` populated from one or more lusee.ingest files.

    Indexing (mirrors :class:`lusee.Data`)::

        data[time_idx, "00",  freq_idx]   # autocorrelation 0 (real)
        data[time_idx, "01R", freq_idx]   # real part of cross 0x1
        data[time_idx, "01I", freq_idx]   # imag part of cross 0x1
        data[time_idx, "01C", freq_idx]   # complex (R + 1j*I)
        data[time_idx, "-01R", freq_idx]  # negated real

    Plotting helpers: :meth:`plot_waterfall`, :meth:`plot_mean_spectrum`,
    :meth:`plot_dcb`, :meth:`plot_adc_stats`.

    The list of underlying per-file bundles is exposed at
    ``data.bundles`` for power users that need lower-level access.
    """

    def __init__(
        self,
        paths,
        *,
        prefer_format: str = "h5",
        time_source: str = "spectra",
        mission_epoch=None,
    ):
        files = _resolve_paths(paths, prefer_format=prefer_format)
        log.info("loading %d file(s): %s", len(files),
                 [str(p) for p in files])

        self.bundles: List[SessionBundle] = [_load_one(p) for p in files]
        bundle = _concat_bundles(self.bundles)

        # Time axis with fallback ladder
        time_axis_mjd, time_axis_raw, source_used = _pick_time_axis(
            bundle, preferred=time_source,
        )
        if time_axis_mjd is None and time_axis_raw is None:
            raise ValueError("no time axis found in any input file")

        # Build LunarTime; prefer MJD when calibrated.
        if time_axis_mjd is not None and _mjd_is_calibrated(bundle):
            times_obj = LunarTime(time_axis_mjd, format="mjd", scale="utc")
        elif mission_epoch is not None:
            t0 = LunarTime(mission_epoch)
            times_obj = t0 + TimeDelta(time_axis_raw * u.s)
        else:
            # uncrater's Packet_Metadata.time is Unix-epoch seconds; use
            # the Unix epoch as the default placeholder so absolute dates
            # come out roughly right even without an explicit MJD
            # calibration. Override via ``mission_epoch=...`` if a
            # different mission convention applies.
            t0 = LunarTime("1970-01-01T00:00:00", scale="utc")
            times_obj = t0 + TimeDelta(time_axis_raw * u.s)

        # Compute median cadence from raw_times for Observation.deltaT.
        if time_axis_raw is not None and time_axis_raw.size > 1:
            dt_med = float(np.nanmedian(np.diff(time_axis_raw)))
            if not np.isfinite(dt_med) or dt_med <= 0:
                dt_med = 1.0
        else:
            dt_med = 1.0

        # Constants -> Observation.__init__
        c = bundle.constants
        # Observation does np.arange(t0, t1, deltaT).astype(Time) internally,
        # which leaks an astropy "missing unit" warning out of the parent's
        # TimeDelta arithmetic. Silence it: we're going to overwrite
        # self.times immediately afterwards anyway.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            super().__init__(
                time_range=(times_obj[0].isot, times_obj[-1].isot),
                lun_lat_deg=c.get("lun_lat_deg", self.default_lun_lat_deg),
                lun_long_deg=c.get("lun_long_deg", self.default_lun_long_deg),
                lun_height_m=c.get("lun_height_m", self.default_lun_height_m),
                deltaT_sec=dt_med,
            )
        # Override the synthetic uniform grid with the actual sample times.
        self.times = times_obj
        self.time_source = source_used
        self._mission_epoch_override = mission_epoch is not None

        # Public attributes
        self.spectra = bundle.spectra
        self.unique_ids = bundle.spectra_unique_ids
        self.raw_times = bundle.spectra_raw_times
        self.mjd_times = bundle.spectra_mjd_times
        self.metadata = bundle.spectra_metadata

        self.tr_spectra = bundle.tr_spectra
        self.tr_unique_ids = bundle.tr_unique_ids
        self.tr_raw_times = bundle.tr_raw_times
        self.tr_mjd_times = bundle.tr_mjd_times
        self.tr_navg2_per_sample = bundle.tr_navg2_per_sample
        self.tr_length_per_sample = bundle.tr_length_per_sample
        self.tr_metadata = bundle.tr_metadata

        self.zoom_spectra = bundle.zoom_spectra
        self.zoom_unique_ids = bundle.zoom_unique_ids
        self.zoom_pfb_indices = bundle.zoom_pfb_indices
        self.zoom_raw_times = bundle.zoom_raw_times

        self.grimm_spectra = bundle.grimm_spectra
        self.waveforms = bundle.waveforms
        self.waveform_times = bundle.waveform_times
        self.housekeeping = bundle.housekeeping
        self.dcb_telemetry = bundle.dcb_fpga
        self.encoder_telemetry = bundle.dcb_encoder
        self.interp_telemetry = bundle.interp_telemetry
        self.session_invariants = bundle.session_invariants

        self.source_paths: List[Path] = [b.source_path for b in self.bundles
                                          if b.source_path is not None]
        self.layout_version = bundle.layout_version

        # Frequency axis: assume the dominant Navgf within the session.
        self.freq = self._derive_freq()

        # Convenience: count properties
        self.Nspectra = self.spectra.shape[0] if self.spectra is not None else 0
        self.Nfreq = self.spectra.shape[2] if self.spectra is not None else 0
        self.Nproducts = NPRODUCTS

    # -------------------- Indexing --------------------

    def __getitem__(self, req):
        """``data[time_idx, comb, freq_idx]`` -- see class docstring."""
        if self.spectra is None:
            raise ValueError("no /spectra in this IngestData")
        time_idx, comb, freq_idx = req
        i, j, mode, sign, conjugate = _resolve_combination(comb)
        try:
            re_idx, im_idx = _PRODUCT_INDEX[(i, j)]
        except KeyError as exc:
            raise KeyError(f"unknown channel pair ({i}, {j})") from exc

        head = mode[0] if mode else "R"
        if head == "R":
            data = self.spectra[time_idx, re_idx, freq_idx]
        elif head == "I":
            if im_idx is None:
                raise ValueError(f"no imaginary channel for autocorrelation ({i},{j})")
            data = self.spectra[time_idx, im_idx, freq_idx]
        elif head == "C":
            if im_idx is None:
                raise ValueError(f"no complex form for autocorrelation ({i},{j})")
            data = (self.spectra[time_idx, re_idx, freq_idx]
                    + 1j * self.spectra[time_idx, im_idx, freq_idx])
            if conjugate:
                data = np.conj(data)
        else:
            raise ValueError(f"unknown combination mode {mode!r}")
        return sign * data

    def cross(self, i: int, j: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        """Convenience: complex cross-correlation as one ndarray."""
        return self[time_idx, (i, j, "C"), freq_idx]

    def auto(self, i: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        return self[time_idx, (i, i, "R"), freq_idx]

    # -------------------- Misc helpers --------------------

    def session_boundaries(self) -> List[Tuple[int, int, Optional[Path]]]:
        """For multi-file loads: list of ``(start, end, source)`` tuples."""
        out = []
        cursor = 0
        for b in self.bundles:
            n = b.spectra.shape[0] if b.spectra is not None else 0
            out.append((cursor, cursor + n, b.source_path))
            cursor += n
        return out

    def _derive_freq(self) -> np.ndarray:
        """Best-effort frequency axis derivation from per-row Navgf."""
        navgf_arr = self.metadata.get("Navgf")
        if navgf_arr is None or navgf_arr.size == 0:
            navgf = 1
        else:
            vals, counts = np.unique(navgf_arr, return_counts=True)
            navgf = int(vals[np.argmax(counts)])
            if vals.size > 1:
                log.info("Navgf varies across rows %s; using mode=%d",
                         vals.tolist(), navgf)
        if navgf <= 1:
            return np.arange(NCHANNELS) * _FREQ_STEP_MHZ_NAVGF1
        if navgf == 2:
            return np.arange(NCHANNELS // 2) * (_FREQ_STEP_MHZ_NAVGF1 * 2)
        return np.arange(NCHANNELS // 4) * (_FREQ_STEP_MHZ_NAVGF1 * 4)

    def __repr__(self) -> str:
        n_sp = self.spectra.shape[0] if self.spectra is not None else 0
        n_tr = self.tr_spectra.shape[0] if self.tr_spectra is not None else 0
        n_zm = self.zoom_spectra.shape[0] if self.zoom_spectra is not None else 0
        n_src = len(self.source_paths)
        has_tlm = bool(self.dcb_telemetry)
        return (f"IngestData(N_spectra={n_sp}, N_tr={n_tr}, "
                f"N_zoom={n_zm}, sources={n_src}, telemetry={has_tlm})")

    # -------------------- Plotting --------------------

    def plot_waterfall(self, comb, *, ax=None, log=True, **imshow_kw):
        """Render an (n_time, n_freq) waterfall for one combination string."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        data = np.asarray(self[:, comb, :], dtype=np.float64)
        if np.iscomplexobj(data):
            data = np.abs(data)
        plot = data
        cbar_label = "value"
        if log:
            with np.errstate(invalid="ignore"):
                plot = np.where(data > 0, np.log10(np.abs(data)), np.nan)
            cbar_label = "log10|value|"
        kwargs = dict(aspect="auto", origin="lower", cmap="viridis")
        kwargs.update(imshow_kw)
        kwargs.setdefault("extent", [self.freq[0], self.freq[-1] if self.freq.size else self.Nfreq,
                                      0, plot.shape[0]])
        im = ax.imshow(plot, **kwargs)
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("sample index")
        ax.set_title(f"{comb} -- waterfall")
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
        return ax

    def plot_mean_spectrum(self, comb, *, ax=None, log=True):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        data = np.asarray(self[:, comb, :], dtype=np.float64)
        if np.iscomplexobj(data):
            data = np.abs(data)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
        x = self.freq if self.freq.size == mean.size else np.arange(mean.size)
        if log:
            valid = (mean > 0) & np.isfinite(mean)
            ax.semilogy(x[valid], mean[valid], "b-", label="mean")
            ax.fill_between(x[valid],
                            np.maximum(mean[valid] - std[valid], 1e-12),
                            mean[valid] + std[valid],
                            alpha=0.3, color="blue")
        else:
            ax.plot(x, mean, "b-", label="mean")
            ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="blue")
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("|value|")
        ax.set_title(f"{comb} -- time-averaged spectrum (N={data.shape[0]})")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_dcb(self, channels=None, *, ax=None):
        """Plot one or more DCB telemetry channels vs mission time."""
        import matplotlib.pyplot as plt
        if not self.dcb_telemetry:
            raise ValueError("no DCB telemetry available")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        ms = self.dcb_telemetry.get("mission_seconds")
        if ms is None or ms.size == 0:
            raise ValueError("DCB telemetry has no time axis")
        ss = self.dcb_telemetry.get("lusee_subsecs")
        if ss is None:
            ss = np.zeros_like(ms)
        t = ms + ss * (1.0 / 65536.0)
        t = t - t[0]
        if channels is None:
            channels = ["THERM_FPGA", "THERM_DCB", "VMON_6V"]
        for ch in channels:
            if ch in self.dcb_telemetry:
                ax.plot(t, self.dcb_telemetry[ch], label=ch, lw=0.9)
            else:
                log.info("dcb channel %s not present", ch)
        ax.set_xlabel("seconds since first telemetry sample")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_adc_stats(self, *, fig=None):
        """Per-channel ADC min/max/mean/rms over time, 4 panels."""
        import matplotlib.pyplot as plt
        adc_min = self.metadata.get("adc_min")
        adc_max = self.metadata.get("adc_max")
        adc_mean = self.metadata.get("adc_mean")
        adc_rms = self.metadata.get("adc_rms")
        if any(a is None for a in (adc_min, adc_max, adc_mean, adc_rms)):
            raise ValueError("ADC stat metadata not present in this IngestData")
        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        else:
            axes = np.array(fig.subplots(2, 2)).reshape(2, 2)
        x = self.raw_times - self.raw_times[0] if self.raw_times.size else np.arange(adc_min.shape[0])
        for ax, arr, title in zip(
            axes.flat,
            (adc_min, adc_max, adc_mean, adc_rms),
            ("ADC min", "ADC max", "ADC mean", "ADC rms"),
        ):
            for ch in range(arr.shape[1]):
                ax.plot(x, arr[:, ch], lw=0.8, label=f"ch{ch}")
            ax.set_title(title)
            ax.set_xlabel("seconds since session start")
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------

def _pick_time_axis(
    bundle: SessionBundle,
    *,
    preferred: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return (mjd_times, raw_times, source_name) by preference, else fallback ladder."""
    candidates = [
        ("spectra", bundle.spectra_mjd_times, bundle.spectra_raw_times),
        ("tr_spectra", bundle.tr_mjd_times, bundle.tr_raw_times),
        ("zoom_spectra", bundle.zoom_mjd_times, bundle.zoom_raw_times),
        ("grimm_spectra", None, bundle.grimm_raw_times),
    ]
    order = [preferred] + [name for name, _, _ in candidates if name != preferred]
    by_name = {name: (mjd, raw) for name, mjd, raw in candidates}

    for name in order:
        if name not in by_name:
            continue
        mjd, raw = by_name[name]
        if (raw is not None and raw.size > 0) or (mjd is not None and mjd.size > 0):
            return mjd, raw, name
    return None, None, preferred


def _mjd_is_calibrated(bundle: SessionBundle) -> bool:
    """MJD is meaningful only when the calibration constants are non-default."""
    raw_subtract = bundle.constants.get("raw_time_subtract_seconds", 0.0)
    mjd_offset = bundle.constants.get("mjd_epoch_offset_days", 0.0)
    return bool(raw_subtract or mjd_offset)


# ---------------------------------------------------------------------------
# Convenience free function
# ---------------------------------------------------------------------------

def load(
    paths,
    *,
    prefer_format: str = "h5",
    time_source: str = "spectra",
    mission_epoch=None,
) -> IngestData:
    """Build an :class:`IngestData` from one or more files / directories.

    See :class:`IngestData` for the full kwarg description.
    """
    return IngestData(
        paths,
        prefer_format=prefer_format,
        time_source=time_source,
        mission_epoch=mission_epoch,
    )


PathsLike = Union[str, Path, Iterable[Union[str, Path]]]
