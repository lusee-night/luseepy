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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from astropy import units as u
from astropy.time import TimeDelta
from lunarsky.time import Time as LunarTime

from lusee.Observation import Observation

from .constants import (
    BITSLICE_REFERENCE,
    DEFAULT_CLOCK_SOURCE,
    DEFAULT_TIME_SCALE,
    NCHANNELS,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    SPECTRA_REPRESENTATION,
    SPECTRA_UNITS,
)
from .decode import canonical_actual_bitslice, restore_bitsliced_spectra

log = logging.getLogger(__name__)

# Pixel→MHz conversion at full resolution (Navgf=1) -- mirrors uncrater.
_FREQ_STEP_MHZ_NAVGF1 = 0.025

# Antenna / ADC channels (auto products 0-3). Distinct from NCHANNELS, which
# is the number of frequency bins (2048); the gain model is per ADC channel.
_N_ADC_CHANNELS = 4

# Spectrometer gain_state enum (coreloop core_loop.h) -> gain-model level.
# GAIN_DISABLE (3) / GAIN_AUTO (4) have no L/M/H model and map to None
# (those channels come out NaN).
_GAIN_CODE_TO_LEVEL = {0: "L", 1: "M", 2: "H"}


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
    spectra_units: Optional[str] = None
    spectra_representation: Optional[str] = None
    spectra_normalization_version: Optional[int] = None

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
    # numeric calibration values plus string time-provenance attrs
    # (time_scale, clock_source, clock_epoch_isot)
    constants: Dict[str, Union[float, str]] = field(default_factory=dict)

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

    primary = {
        p.relative_to(target).with_suffix(""): p
        for ext in primary_exts
        for p in target.rglob(f"*{ext}")
    }
    fallback = {
        p.relative_to(target).with_suffix(""): p
        for ext in fallback_exts
        for p in target.rglob(f"*{ext}")
    }

    resolved = dict(fallback)
    resolved.update(primary)
    if resolved:
        return sorted(resolved.values())
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

        # constants: numeric values coerced to float; the provenance attrs
        # are strings by schema (key-based, so a numeric-looking value like
        # clock_source="2" stays a string)
        if "constants" in f:
            for k, v in f["constants"].attrs.items():
                sv = _scalarize(v)
                if isinstance(sv, bytes):
                    sv = sv.decode()
                if k in ("time_scale", "clock_source", "clock_epoch_isot"):
                    bundle.constants[k] = str(sv)
                    continue
                try:
                    bundle.constants[k] = float(sv)
                except (TypeError, ValueError):
                    bundle.constants[k] = str(sv)

        # spectra
        if "spectra" in f and "data" in f["spectra"]:
            sp = f["spectra"]
            data_ds = sp["data"]
            bundle.spectra = data_ds[...]
            if (bundle.spectra.ndim != 3
                    or bundle.spectra.shape[1] != NPRODUCTS):
                raise ValueError(
                    f"{path}: normal spectra must have shape "
                    f"(N, {NPRODUCTS}, Nfreq); got {bundle.spectra.shape}"
                )
            bundle.spectra_units = _attr_text(data_ds.attrs.get("units"))
            bundle.spectra_representation = _attr_text(
                data_ds.attrs.get("representation")
            )
            if "normalization_version" in data_ds.attrs:
                bundle.spectra_normalization_version = int(
                    data_ds.attrs["normalization_version"]
                )
            bundle.spectra_unique_ids = _read_or_none(sp, "unique_ids")
            bundle.spectra_raw_times = _read_or_none(sp, "raw_times")
            bundle.spectra_mjd_times = _read_or_none(sp, "mjd_times")
            if "metadata" in sp:
                for k, ds in sp["metadata"].items():
                    bundle.spectra_metadata[k] = ds[...]

            n_spectra = bundle.spectra.shape[0]
            actual_bitslice = bundle.spectra_metadata.get("actual_bitslice")
            if actual_bitslice is None:
                raise ValueError(
                    f"{path}: normal spectra have no actual_bitslice metadata"
                )
            if (bundle.layout_version == 3
                    and np.asarray(actual_bitslice).shape
                    != (n_spectra, NPRODUCTS)):
                raise ValueError(
                    f"{path}: layout-v3 actual_bitslice must have exact shape "
                    f"({n_spectra}, {NPRODUCTS}); got "
                    f"{np.asarray(actual_bitslice).shape}"
                )
            # Keep critical metadata in one stable shape for conversion and
            # auditing, even when older writers used (N,1,16).
            bundle.spectra_metadata["actual_bitslice"] = (
                canonical_actual_bitslice(actual_bitslice, n_spectra)
            )
            actual_gain = bundle.spectra_metadata.get("actual_gain")
            if bundle.layout_version == 3 and actual_gain is None:
                raise ValueError(
                    f"{path}: layout-v3 normal spectra have no actual_gain metadata"
                )
            if actual_gain is not None:
                actual_gain = np.asarray(actual_gain)
                if (bundle.layout_version == 3
                        and actual_gain.shape
                        != (n_spectra, _N_ADC_CHANNELS)):
                    raise ValueError(
                        f"{path}: layout-v3 actual_gain must have exact shape "
                        f"({n_spectra}, {_N_ADC_CHANNELS}); got "
                        f"{actual_gain.shape}"
                    )
                if actual_gain.size != n_spectra * _N_ADC_CHANNELS:
                    raise ValueError(
                        f"{path}: actual_gain must have shape "
                        f"({n_spectra}, {_N_ADC_CHANNELS}); got {actual_gain.shape}"
                    )
                bundle.spectra_metadata["actual_gain"] = actual_gain.reshape(
                    n_spectra, _N_ADC_CHANNELS
                )

            if bundle.layout_version == 2:
                bundle.spectra = restore_bitsliced_spectra(
                    bundle.spectra,
                    bundle.spectra_metadata["actual_bitslice"],
                )
                bundle.spectra_units = SPECTRA_UNITS
                bundle.spectra_representation = SPECTRA_REPRESENTATION
                bundle.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION
                warnings.warn(
                    f"{path}: layout-v2 spectra were bit-slice restored in memory; "
                    "re-ingest to produce an unambiguous layout-v3 file",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif bundle.layout_version == 3:
                if int(data_ds.attrs.get("bitslice_restored", 0)) != 1:
                    raise ValueError(
                        f"{path}: layout-v3 /spectra/data does not declare "
                        "bitslice_restored=1"
                    )
                if int(data_ds.attrs.get("bitslice_reference", -1)) != BITSLICE_REFERENCE:
                    raise ValueError(
                        f"{path}: unsupported layout-v3 bit-slice reference"
                    )
                if bundle.spectra_units != SPECTRA_UNITS:
                    raise ValueError(
                        f"{path}: layout-v3 spectra units must be {SPECTRA_UNITS!r}; "
                        f"got {bundle.spectra_units!r}"
                    )
                if bundle.spectra_representation != SPECTRA_REPRESENTATION:
                    raise ValueError(
                        f"{path}: layout-v3 spectra representation must be "
                        f"{SPECTRA_REPRESENTATION!r}; got "
                        f"{bundle.spectra_representation!r}"
                    )
                if (bundle.spectra_normalization_version
                        != SPECTRA_NORMALIZATION_VERSION):
                    raise ValueError(
                        f"{path}: layout-v3 normalization_version must be "
                        f"{SPECTRA_NORMALIZATION_VERSION}; got "
                        f"{bundle.spectra_normalization_version!r}"
                    )
            else:
                raise ValueError(
                    f"{path}: normal spectra require explicit layout_version 2 or 3; "
                    f"got {bundle.layout_version!r}"
                )

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


def _attr_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii")
    return str(value)


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
            for k_fits, k_h5 in (
                ("TIMESYS", "time_scale"),
                ("CLKSRC", "clock_source"),
                ("CLKEPOCH", "clock_epoch_isot"),
            ):
                if k_fits in cst:
                    v = str(cst[k_fits])
                    # TIMESYS values are uppercase by FITS convention;
                    # astropy scales are lowercase
                    bundle.constants[k_h5] = (
                        v.lower() if k_h5 == "time_scale" else v
                    )

        names = [h.name for h in hdul]

        if "SPECTRA" in names:
            spectra_hdu = hdul["SPECTRA"]
            bundle.spectra = np.asarray(spectra_hdu.data, dtype=np.float32)
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

        if bundle.spectra is not None:
            spectra_hdu = hdul["SPECTRA"]
            if (bundle.spectra.ndim != 3
                    or bundle.spectra.shape[1] != NPRODUCTS):
                raise ValueError(
                    f"{path}: normal spectra must have shape "
                    f"(N, {NPRODUCTS}, Nfreq); got {bundle.spectra.shape}"
                )
            actual_key, actual_bitslice = _metadata_get_ci(
                bundle.spectra_metadata, "actual_bitslice"
            )
            if actual_bitslice is None:
                raise ValueError(
                    f"{path}: normal spectra have no actual_bitslice metadata"
                )
            canonical_bs = canonical_actual_bitslice(
                actual_bitslice, bundle.spectra.shape[0]
            )
            if actual_key != "actual_bitslice":
                bundle.spectra_metadata.pop(actual_key, None)
            bundle.spectra_metadata["actual_bitslice"] = canonical_bs

            gain_key, actual_gain = _metadata_get_ci(
                bundle.spectra_metadata, "actual_gain"
            )
            if bundle.layout_version == 3 and actual_gain is None:
                raise ValueError(
                    f"{path}: layout-v3 normal spectra have no actual_gain metadata"
                )
            if actual_gain is not None:
                actual_gain = np.asarray(actual_gain)
                expected = bundle.spectra.shape[0] * _N_ADC_CHANNELS
                if actual_gain.size != expected:
                    raise ValueError(
                        f"{path}: actual_gain has shape {actual_gain.shape}; "
                        f"expected ({bundle.spectra.shape[0]}, {_N_ADC_CHANNELS})"
                    )
                if gain_key != "actual_gain":
                    bundle.spectra_metadata.pop(gain_key, None)
                bundle.spectra_metadata["actual_gain"] = actual_gain.reshape(
                    bundle.spectra.shape[0], _N_ADC_CHANNELS
                )

            if bundle.layout_version == 2:
                bundle.spectra = restore_bitsliced_spectra(
                    bundle.spectra, canonical_bs
                )
                warnings.warn(
                    f"{path}: layout-v2 FITS spectra were bit-slice restored "
                    "in memory; re-ingest to layout v3",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif bundle.layout_version == 3:
                if not bool(spectra_hdu.header.get("BITSREST", False)):
                    raise ValueError(
                        f"{path}: layout-v3 SPECTRA does not declare BITSREST"
                    )
                if int(spectra_hdu.header.get("BITSREF", -1)) != BITSLICE_REFERENCE:
                    raise ValueError(f"{path}: unsupported SPECTRA bit reference")
                if int(spectra_hdu.header.get("NORMVER", -1)) \
                        != SPECTRA_NORMALIZATION_VERSION:
                    raise ValueError(f"{path}: unsupported SPECTRA normalization version")
                if str(spectra_hdu.header.get("BUNIT", "")) != SPECTRA_UNITS:
                    raise ValueError(f"{path}: SPECTRA units must be {SPECTRA_UNITS!r}")
                if str(spectra_hdu.header.get("REPRESENTATION", "")) \
                        != SPECTRA_REPRESENTATION:
                    raise ValueError(
                        f"{path}: SPECTRA representation must be "
                        f"{SPECTRA_REPRESENTATION!r}"
                    )
            else:
                raise ValueError(
                    f"{path}: normal spectra require explicit layout_version 2 or 3"
                )
            bundle.spectra_units = SPECTRA_UNITS
            bundle.spectra_representation = SPECTRA_REPRESENTATION
            bundle.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION

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


def _metadata_get_ci(metadata: Dict[str, np.ndarray], wanted: str):
    """Return ``(real_key, value)`` from a case-insensitive FITS mapping."""
    for key, value in metadata.items():
        if key.lower() == wanted.lower():
            return key, value
    return wanted, None


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
        prototype = next(d[k] for d in dicts if k in d)
        tail = prototype.shape[1:]
        fill_dtype = (
            prototype.dtype
            if prototype.dtype.kind in {"f", "c"}
            else np.float64
        )
        chunks = []
        for d, n in zip(dicts, n_per_source):
            if k in d:
                chunks.append(d[k])
            else:
                fill = np.full((n,) + tail, np.nan, dtype=fill_dtype)
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
    units = {b.spectra_units for b in bundles if b.spectra is not None}
    representations = {
        b.spectra_representation for b in bundles if b.spectra is not None
    }
    if len(units) > 1 or len(representations) > 1:
        raise ValueError(
            "cannot concatenate normal spectra with different normalization "
            f"contracts: units={units}, representations={representations}"
        )
    out.spectra_units = next(iter(units), None)
    out.spectra_representation = next(iter(representations), None)
    out.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION
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
            if k == "time_scale":
                continue  # merged specially below, keep-first does not apply
            if k in out.constants and out.constants[k] != v:
                log.warning("constant %s disagrees across sessions: %s vs %s; "
                            "keeping first", k, out.constants[k], v)
    # time_scale is load-bearing (it decides the astropy scale of the merged
    # time axis), so it does not go through keep-first: disagreement is an
    # error, and a bundle without it degrades the merge to "unknown" so the
    # reader demands an explicit assume_scale instead of silently adopting
    # another file's scale.
    scales = {
        str(b.constants.get("time_scale", DEFAULT_TIME_SCALE))
        for b in bundles
    }
    if len(scales) > 1:
        if scales - {DEFAULT_TIME_SCALE}:
            known = sorted(scales - {DEFAULT_TIME_SCALE})
            if len(known) > 1:
                raise ValueError(
                    f"input files record contradicting time scales: {known}"
                )
        out.constants["time_scale"] = DEFAULT_TIME_SCALE
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

    Time scale: the astropy scale of ``self.times`` comes from the file's
    recorded ``time_scale`` constant when present; files without it (or
    with "unknown") require an explicit ``assume_scale=`` argument (e.g.
    ``assume_scale="utc"``). ``self.time_provenance`` records the resolved
    scale, the clock source, and whether the scale was assumed.
    """

    def __init__(
        self,
        paths,
        *,
        prefer_format: str = "h5",
        time_source: str = "spectra",
        mission_epoch=None,
        assume_scale=None,
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

        # Time-scale provenance: never guess. Files written by the current
        # writers record a /constants attr ``time_scale`` (TIMESYS in FITS);
        # for older files, or files written with the honest default
        # "unknown", the caller must state the scale via ``assume_scale``.
        if assume_scale is not None:
            assume_scale = str(assume_scale).lower()
            if assume_scale == DEFAULT_TIME_SCALE:
                raise ValueError(
                    "assume_scale='unknown' is not a usable scale; pass the "
                    "astropy scale the mission clock actually runs on"
                )
        if assume_scale is not None:
            # check against every input file, not just the merged constant:
            # a scale-less file in the mix degrades the merged value to
            # "unknown", which must not let assume_scale contradict a scale
            # another file explicitly recorded
            for b in self.bundles:
                bscale = str(b.constants.get("time_scale", DEFAULT_TIME_SCALE))
                if bscale != DEFAULT_TIME_SCALE and bscale != assume_scale:
                    raise ValueError(
                        f"assume_scale={assume_scale!r} contradicts the "
                        f"recorded time_scale={bscale!r} in {b.source_path}"
                    )
        file_scale = str(bundle.constants.get("time_scale", DEFAULT_TIME_SCALE))
        if file_scale != DEFAULT_TIME_SCALE:
            scale = file_scale
            scale_from_assume = False
        else:
            scale = assume_scale
            scale_from_assume = assume_scale is not None

        def require_scale():
            if scale is None:
                raise ValueError(
                    "the input file(s) do not record a time scale (constants "
                    "attr 'time_scale' missing or 'unknown'). Pass e.g. "
                    "assume_scale='utc' (the previous implicit behavior) or "
                    "the astropy scale the mission clock actually runs on."
                )
            return scale

        # Build LunarTime; prefer MJD when calibrated. ``scale_assumed`` is
        # per-branch: only branches that consume the resolved scale set it.
        scale_assumed = False
        if time_axis_mjd is not None and _mjd_is_calibrated(bundle):
            times_obj = LunarTime(time_axis_mjd, format="mjd",
                                  scale=require_scale())
            scale_assumed = scale_from_assume
        elif mission_epoch is not None:
            # the caller-provided epoch defines the time arithmetic and
            # carries its own scale; a resolved scale that disagrees with
            # it is a contradiction, not something to silently ignore
            t0 = LunarTime(mission_epoch)
            if scale is not None and scale != str(t0.scale):
                raise ValueError(
                    f"mission_epoch has scale {str(t0.scale)!r} but the "
                    f"{'file records' if not scale_from_assume else 'caller assumed'} "
                    f"time scale {scale!r}"
                )
            times_obj = t0 + TimeDelta(time_axis_raw * u.s)
        else:
            # uncrater's Packet_Metadata.time is Unix-epoch seconds; use
            # the Unix epoch as the default placeholder so absolute dates
            # come out roughly right even without an explicit MJD
            # calibration. Override via ``mission_epoch=...`` if a
            # different mission convention applies.
            t0 = LunarTime("1970-01-01T00:00:00", scale=require_scale())
            times_obj = t0 + TimeDelta(time_axis_raw * u.s)
            scale_assumed = scale_from_assume

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
        self.time_provenance = {
            "scale": str(times_obj.scale),
            "source": str(bundle.constants.get("clock_source",
                                               DEFAULT_CLOCK_SOURCE)),
            "assumed": scale_assumed,
        }
        self._mission_epoch_override = mission_epoch is not None

        # Public attributes
        if bundle.spectra is None:
            self.spectra = None
        else:
            from lusee.LabeledArray import FRAME_TOPO, label
            self.spectra = label(
                bundle.spectra,
                units=bundle.spectra_units or SPECTRA_UNITS,
                frame=FRAME_TOPO,
            )
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
                # LabeledArray.conj() preserves the raw SDU decoration;
                # np.conj() intentionally returns a bare array.
                data = data.conj()
        else:
            raise ValueError(f"unknown combination mode {mode!r}")
        return sign * data

    def cross(self, i: int, j: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        """Convenience: complex cross-correlation as one ndarray."""
        return self[time_idx, (i, j, "C"), freq_idx]

    def auto(self, i: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        return self[time_idx, (i, i, "R"), freq_idx]

    # -------------------- Physical-unit conversion --------------------

    # Telemetry channels the PCA gain model regresses on (see
    # lusee.GainModel.SpectrometerGain). All six live in
    # ``spectra_interpolated_telemetry`` once a session is ingested with
    # ``interpolate_telemetry=True``.
    _GAIN_TELEMETRY_KEYS = (
        "THERM_FPGA", "SPE_ADC0_T", "SPE_ADC1_T",
        "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C",
    )

    @staticmethod
    def _resolve_levels(levels):
        """Normalize a gain-level spec to a length-4 per-channel sequence.

        Accepts 'H' (one level for all four channels), 'HHHH' / 'LMHH'
        (one char per channel), a length-4 sequence, or a
        ``{channel: level}`` dict (passed through unchanged).
        """
        if isinstance(levels, str):
            s = levels.strip().upper()
            if len(s) == 1:
                return [s] * _N_ADC_CHANNELS
            if len(s) == _N_ADC_CHANNELS:
                return list(s)
            raise ValueError(
                f"string gain level must be 1 or {_N_ADC_CHANNELS} chars; got {levels!r}"
            )
        if isinstance(levels, dict):
            return levels
        levels = list(levels)
        if len(levels) != _N_ADC_CHANNELS:
            raise ValueError(f"need {_N_ADC_CHANNELS} per-channel levels; got {levels!r}")
        return levels

    def to_physical(self, levels=None, *, freqs_mhz=None, gain=None,
                    chunk_size=None):
        """Convert stored SDU spectra to nV/sqrt(Hz) on demand.

        The conversion is applied here, after loading; it is never stored
        in the HDF5. Bit-slice restoration has already happened during
        ingestion and is not an optional calibration switch. This method
        uses :class:`lusee.GainModel.SpectrometerGain` with the
        per-sample interpolated telemetry written by the ingest pipeline.

        Per spectra time sample the model predicts a gain spectrum for each
        channel from that sample's telemetry, then maps counts to
        nV/sqrt(Hz): autos as ``sqrt(X / G)`` and crosses as
        ``sign(X) * sqrt(|X| / sqrt(Ga*Gb))``. Bins outside the model's
        anchor-frequency range (or with invalid gain/power) come out NaN.
        This cross convention is retained for compatibility and ASD sanity
        plots; use :meth:`to_physical_psd` for linear cross components.

        :param levels: Per-channel ADC gain setting. The default (None) reads
            the realized per-sample, per-channel gain from
            ``/spectra/metadata/actual_gain`` (GAIN_LOW/MED/HIGH -> 'L'/'M'/'H';
            disable/auto codes have no model and yield NaN for that channel).
            Pass an explicit level to override every sample: a single letter
            for all four channels, a length-4 string / sequence, or a
            ``{channel: level}`` dict.
        :param freqs_mhz: Frequency grid for the conversion; defaults to
            ``self.freq`` (the per-bin spectrometer frequencies in MHz).
        :param gain: An existing :class:`SpectrometerGain` to reuse; a
            cached one is created on first use otherwise.
        :param chunk_size: Optional positive number of time rows per gain-model
            batch. This bounds intermediate gain memory without changing the
            returned array or the model snapshot used for the call.
        :returns: :class:`lusee.LabeledArray` shaped like ``self.spectra``
            (Nspectra, NPRODUCTS, Nfreq), units "nV/sqrt(Hz)".

        Model-family selection is process-wide. ``lusee.GainModel.set_models``
        changes the selection used by the next call; ``convert_batch``
        snapshots that selection once so a concurrent update cannot split a
        conversion across two model configurations.
        """
        from lusee.LabeledArray import FRAME_TOPO, label

        gain, tel, level_rows, freqs = self._gain_conversion_inputs(
            levels, freqs_mhz, gain
        )
        out = gain.convert_batch(
            np.asarray(self.spectra),
            tel,
            level_rows,
            freqs_mhz=freqs,
            chunk_size=chunk_size,
        )
        return label(out, frame=FRAME_TOPO)

    def to_physical_psd(self, levels=None, *, freqs_mhz=None, gain=None,
                        units="V^2/Hz", chunk_size=None):
        """Return the physically linear input-referred spectral-density cube.

        Autos are ``X/G`` and cross real/imaginary components are
        ``X/sqrt(Ga*Gb)``.  Unlike :meth:`to_physical`, this does not apply a
        signed square root to cross components.  ``units`` may be ``"V^2/Hz"``
        (default) or the gain model's native ``"nV^2/Hz"``.  The result is
        computed on demand and is never persisted in the ingest file.
        """
        from lusee.LabeledArray import FRAME_TOPO, label

        gain, tel, level_rows, freqs = self._gain_conversion_inputs(
            levels, freqs_mhz, gain
        )
        out = gain.convert_batch_psd(
            np.asarray(self.spectra),
            tel,
            level_rows,
            freqs_mhz=freqs,
            units=units,
            chunk_size=chunk_size,
        )
        return label(out, frame=FRAME_TOPO)

    # Short, discoverable alias; the longer name makes the relationship with
    # the legacy ASD-returning to_physical() explicit in user-facing code.
    def to_psd(self, levels=None, *, freqs_mhz=None, gain=None,
               units="V^2/Hz", chunk_size=None):
        return self.to_physical_psd(
            levels,
            freqs_mhz=freqs_mhz,
            gain=gain,
            units=units,
            chunk_size=chunk_size,
        )

    def _gain_conversion_inputs(self, levels, freqs_mhz, gain):
        """Validate and prepare shared inputs for vectorized conversion."""
        from lusee.GainModel import SpectrometerGain

        if self.spectra is None:
            raise ValueError("no /spectra in this IngestData to convert")
        if not self.interp_telemetry:
            raise ValueError(
                "no interpolated telemetry available; re-ingest the session "
                "with interpolate_telemetry=True so the gain model has the "
                "per-sample telemetry it regresses on"
            )
        missing = [k for k in self._GAIN_TELEMETRY_KEYS
                   if k not in self.interp_telemetry]
        if missing:
            raise ValueError(
                f"interpolated telemetry is missing gain-model channels {missing}"
            )
        if gain is None:
            gain = getattr(self, "_gain_model", None)
            if gain is None:
                gain = SpectrometerGain()
                self._gain_model = gain
        freqs = self.freq if freqs_mhz is None else np.asarray(freqs_mhz, dtype=float)
        level_rows = np.asarray(self._level_rows(levels), dtype=object)
        telemetry = {
            k: np.asarray(self.interp_telemetry[k], dtype=float)
            for k in self._GAIN_TELEMETRY_KEYS
        }
        return gain, telemetry, level_rows, freqs

    def _level_rows(self, levels):
        """Per-sample, per-channel gain levels as a list of NCH-long lists.

        ``levels=None`` reads the realized gain codes from
        ``/spectra/metadata/actual_gain`` (one row per spectra sample); any
        explicit ``levels`` is resolved once and broadcast to every sample.
        Entries are 'L'/'M'/'H', or None where the recorded code has no
        L/M/H model (disable/auto).
        """
        n = self.spectra.shape[0]
        if levels is not None:
            resolved = self._resolve_levels(levels)
            if isinstance(resolved, dict):
                row = [resolved[ch] for ch in range(_N_ADC_CHANNELS)]
            else:
                row = list(resolved)
            return [row] * n
        ag = self.metadata.get("actual_gain")
        if ag is None:
            raise ValueError(
                "levels=None auto-detects the gain from "
                "/spectra/metadata/actual_gain, which is absent; re-ingest with "
                "the current pipeline (it propagates the meta 'base' fields) or "
                "pass levels= explicitly"
            )
        ag = np.asarray(ag).reshape(n, -1)[:, -_N_ADC_CHANNELS:]
        return [[self._level_from_code(c) for c in ag[t]] for t in range(n)]

    @staticmethod
    def _level_from_code(code):
        """Map one actual_gain entry to 'L'/'M'/'H', or None if unmodeled.

        Accepts the integer gain_state enum (GAIN_LOW/MED/HIGH = 0/1/2) or a
        char code ('L'/'M'/'H'); disable/auto/NaN/unknown -> None.
        """
        if isinstance(code, (bytes, np.bytes_)):
            code = code.decode("ascii", "ignore")
        if isinstance(code, str):
            code = code.strip().upper()
            return code if code in ("L", "M", "H") else None
        try:
            numeric = float(code)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric != np.rint(numeric):
            return None
        return _GAIN_CODE_TO_LEVEL.get(int(numeric))

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
    assume_scale=None,
) -> IngestData:
    """Build an :class:`IngestData` from one or more files / directories.

    See :class:`IngestData` for the full kwarg description.
    """
    return IngestData(
        paths,
        prefer_format=prefer_format,
        time_source=time_source,
        mission_epoch=mission_epoch,
        assume_scale=assume_scale,
    )


PathsLike = Union[str, Path, Iterable[Union[str, Path]]]
