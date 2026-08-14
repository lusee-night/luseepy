"""Stage 6: decode an uncrater session directory into in-memory products.

Wraps ``uncrater.Collection`` and converts its typed packet objects into
plain numpy arrays / dicts in a ``Products`` dataclass. ``hdf5_writer``
consumes the result; it never sees uncrater types.

Lazy-imports uncrater so that ``lusee.ingest`` is importable without it.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .constants import (
    BITSLICE_REFERENCE,
    NCHANNELS,
    NPRODUCTS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
    ZOOM_COMPONENTS,
)
from .session import raw_seconds_from_split_time

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpectrumSample:
    data: np.ndarray              # shape (NPRODUCTS, NCHANNELS), float32, NaN where missing
    unique_packet_id: int
    raw_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    # True means ``data`` has already been restored with actual_bitslice to
    # the bit-31 accumulator convention used by the gain-model artifacts.
    bitslice_restored: bool = False

    def restore_bitslice(self) -> None:
        """Restore normal-spectrum data to gain-model input SDU exactly once."""
        if "actual_bitslice" not in self.metadata:
            raise ValueError(
                f"normal spectrum {self.unique_packet_id} has no actual_bitslice; "
                "ingestion cannot safely normalize it"
            )
        # Validate the mandatory normalization input first, so malformed
        # bit-slice metadata is reported directly even if another required
        # provenance field is also absent.
        canonical_actual_bitslice(self.metadata["actual_bitslice"], 1)
        if "actual_gain" not in self.metadata:
            raise ValueError(
                f"normal spectrum {self.unique_packet_id} has no actual_gain; "
                "layout-v3 ingestion requires the realized four-channel gain"
            )
        # Validate even an explicitly pre-restored sample: layout v3 requires
        # the realized metadata for provenance and later auditing.
        actual_gain = np.asarray(self.metadata["actual_gain"])
        if actual_gain.size != 4:
            raise ValueError(
                f"normal spectrum {self.unique_packet_id} actual_gain has "
                f"{actual_gain.size} values; expected 4"
            )
        actual_gain = actual_gain.reshape(4)
        if actual_gain.dtype.kind in ("U", "S", "O"):
            try:
                actual_gain = actual_gain.astype("S1")
            except (TypeError, ValueError) as exc:
                raise ValueError("actual_gain character codes are malformed") from exc
        self.metadata["actual_gain"] = actual_gain
        if self.bitslice_restored:
            return
        self.data = restore_bitsliced_spectra(
            np.asarray(self.data)[None, ...],
            self.metadata["actual_bitslice"],
        )[0]
        self.bitslice_restored = True


@dataclass
class TRSpectrumSample:
    data: np.ndarray              # shape (NPRODUCTS, navg2, tr_length), float32
    unique_packet_id: int
    raw_seconds: float
    navg2: int
    tr_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoomSample:
    data: np.ndarray              # shape (4, ZOOM_BINS), float32
    unique_packet_id: int
    pfb_index: int
    raw_seconds: float            # inherited from preceding spectrum metadata


@dataclass
class WaveformSample:
    data: np.ndarray              # shape (WAVEFORM_SAMPLES,) int16
    channel: int
    unique_packet_id: int
    raw_seconds: float


@dataclass
class HKSample:
    hk_type: int
    version: int
    unique_packet_id: int
    errors: int
    raw_seconds: float = 0.0
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CalDataSample:
    packet_idx: int
    channel_idx: int
    data: np.ndarray              # variable-length float32


@dataclass
class Products:
    spectra: List[SpectrumSample] = field(default_factory=list)
    tr_spectra: List[TRSpectrumSample] = field(default_factory=list)
    zoom_spectra: List[ZoomSample] = field(default_factory=list)
    grimm_spectra: List[SpectrumSample] = field(default_factory=list)
    waveforms: List[WaveformSample] = field(default_factory=list)
    housekeeping: List[HKSample] = field(default_factory=list)
    cal_data: List[CalDataSample] = field(default_factory=list)

    # Session-invariant fields decoded from the Hello packet.
    sw_version: Optional[int] = None
    fw_version: Optional[int] = None
    fw_id: Optional[int] = None
    fw_date: Optional[int] = None
    fw_time: Optional[int] = None
    start_unique_packet_id: Optional[int] = None
    start_time_32: Optional[int] = None
    start_time_16: Optional[int] = None
    start_raw_seconds: Optional[float] = None

    def restore_spectra_bitslices(self) -> None:
        """Enforce the normal-spectrum SDU invariant for every sample.

        This intentionally does not touch TR or Grimm spectra: their firmware
        bit-slice semantics have not been established.
        """
        for sample in self.spectra:
            sample.restore_bitslice()


# ---------------------------------------------------------------------------
# Helper extractors
# ---------------------------------------------------------------------------

_META_FIELDS_FLAT = (
    "Navg1_shift", "Navg2_shift", "Navgf",
    "tr_start", "tr_stop", "tr_avg_shift",
    "_time_32", "_time_16", "_uC_time",
    "errors", "format",
)
_META_FIELDS_VEC = {
    "actual_bitslice": 16,
    "bitslice": 16,
    "actual_gain": 4,
    "gain": 4,
    "gain_auto_min": 4,
    "gain_auto_mult": 4,
    "adc_min": 4,
    "adc_max": 4,
    "adc_mean": 4,
    "adc_rms": 4,
    "adc_invalid_count_min": 4,
    "adc_invalid_count_max": 4,
}


def _as_vec(v, n: int) -> np.ndarray:
    arr = np.asarray(list(v) if hasattr(v, "__iter__") else [v])
    if arr.size < n:
        out = np.zeros(n, dtype=arr.dtype if arr.size else np.int64)
        out[:arr.size] = arr
        return out
    return arr[:n]


def _as_exact_vec(v, n: int, name: str) -> np.ndarray:
    """Return one metadata vector without silently padding or truncating it."""
    arr = np.asarray(list(v) if hasattr(v, "__iter__") else [v])
    if arr.size != n:
        raise ValueError(f"metadata {name} has {arr.size} values; expected {n}")
    return arr.reshape(n)


def canonical_actual_bitslice(values, n_rows: int) -> np.ndarray:
    """Canonicalize actual bit-slice metadata to integral ``(N, 16)``.

    A single row accepts one 16-element vector.  Multiple rows must contain
    exactly ``N*16`` row-major values with first dimension ``N``; a lone
    vector is never broadcast across persisted rows.  No fallback to the
    requested ``bitslice`` field is permitted.
    """
    arr = np.asarray(values)
    if n_rows == 1 and arr.size == NPRODUCTS:
        arr = arr.reshape(1, NPRODUCTS)
    elif arr.size == n_rows * NPRODUCTS and arr.shape[0] == n_rows:
        arr = arr.reshape(n_rows, NPRODUCTS)
    else:
        raise ValueError(
            "actual_bitslice must contain 16 values per normal-spectrum row; "
            f"got shape {arr.shape} for {n_rows} rows"
        )
    try:
        numeric = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual_bitslice must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("actual_bitslice contains non-finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise ValueError("actual_bitslice values must be integral")
    if np.any((rounded < 0) | (rounded > BITSLICE_REFERENCE)):
        raise ValueError(
            f"actual_bitslice must lie in [0, {BITSLICE_REFERENCE}]"
        )
    return rounded.astype(np.int16, copy=False)


def restore_bitsliced_spectra(data, actual_bitslice) -> np.ndarray:
    """Restore a normal-spectra cube to gain-model input SDU.

    ``data`` must have shape ``(N, 16, Nfreq)``.  Scaling is performed with
    ``ldexp`` so the exact power-of-two operation is explicit and efficient::

        SDU = decoded * 2**(actual_bitslice - 31)
    """
    cube = np.asarray(data)
    if cube.ndim != 3 or cube.shape[1] != NPRODUCTS:
        raise ValueError(
            f"normal spectra must have shape (N, {NPRODUCTS}, Nfreq); "
            f"got {cube.shape}"
        )
    bit_slice = canonical_actual_bitslice(actual_bitslice, cube.shape[0])
    exponent = bit_slice.astype(np.int16) - BITSLICE_REFERENCE
    return np.ldexp(cube, exponent[:, :, None])


def _meta_get(meta_pkt, name):
    """Read a metadata field from the packet, falling back to its ``base`` struct.

    Packet_Metadata sets a few fields directly (format, adc_*, telemetry_*) but
    leaves most in the C ``base`` sub-struct (actual_bitslice, bitslice,
    actual_gain, Navgf, Navg*_shift, ...). Look in both so they all survive.
    """
    v = getattr(meta_pkt, name, None)
    if v is None:
        base = getattr(meta_pkt, "base", None)
        if base is not None:
            v = getattr(base, name, None)
    return v


def extract_metadata(meta_pkt) -> Dict[str, Any]:
    """Pull out the documented metadata fields. Missing fields default to 0/empty."""
    out: Dict[str, Any] = {}
    for name in _META_FIELDS_FLAT:
        v = _meta_get(meta_pkt, name)
        if v is not None:
            out[name] = int(v) if isinstance(v, (int, np.integer)) else float(v)
    for name, n in _META_FIELDS_VEC.items():
        v = _meta_get(meta_pkt, name)
        if v is not None:
            if name in ("actual_bitslice", "bitslice", "actual_gain", "gain"):
                out[name] = _as_exact_vec(v, n, name)
            else:
                out[name] = _as_vec(v, n)
    # route is nested as a struct with .plus[4] and .minus[4]
    route = getattr(meta_pkt, "route", None)
    if route is not None:
        out["route_plus"] = _as_vec(getattr(route, "plus", []), 4)
        out["route_minus"] = _as_vec(getattr(route, "minus", []), 4)
    # Embedded low-rate spectrometer telemetry (when present) shows up as
    # attribute starting with "telemetry" (or "telemetry_*"). Copy any.
    for attr in dir(meta_pkt):
        if attr.startswith("telemetry") and not attr.startswith("_"):
            v = getattr(meta_pkt, attr, None)
            if v is None or callable(v):
                continue
            try:
                out[attr] = float(v)
            except (TypeError, ValueError):
                try:
                    out[attr] = np.asarray(v)
                except Exception:    # noqa: BLE001
                    pass
    return out


def _meta_raw_seconds(meta_pkt) -> float:
    """Mission-time seconds for a metadata packet.

    Modern uncrater (SW 0x307+) exposes a pre-decoded ``time`` attribute
    on Packet_Metadata; older schemas exposed split ``time_32`` /
    ``time_16`` (or the underscored variants). Try the decoded form
    first, then fall back to recomputing from the split fields.
    """
    t = getattr(meta_pkt, "time", None)
    if t is not None:
        try:
            return float(t)
        except (TypeError, ValueError):
            pass
    t32 = getattr(meta_pkt, "_time_32", None) or getattr(meta_pkt, "time_32", None)
    t16 = getattr(meta_pkt, "_time_16", None) or getattr(meta_pkt, "time_16", None)
    if t32 is None or t16 is None:
        return 0.0
    return raw_seconds_from_split_time(int(t32), int(t16))


def _spectrum_dict_to_array(spec_dict: dict) -> np.ndarray:
    """Build a (NPRODUCTS, NCHANNELS) float32 array from one uncrater spectrum dict.

    The dict has keys 0..15 for present products (each has .data with up to
    NCHANNELS values) and "meta" for the metadata packet. Missing products
    and trailing channels in reduced-resolution spectra stay NaN.
    """
    out = np.full((NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
    for k in range(NPRODUCTS):
        pkt = spec_dict.get(k)
        if pkt is None:
            continue
        data = getattr(pkt, "data", None)
        if data is None:
            continue
        arr = np.asarray(data, dtype=np.float32)
        n = min(arr.size, NCHANNELS)
        out[k, :n] = arr[:n]
    return out


def _tr_spectrum_dict_to_array(spec_dict: dict) -> Optional[np.ndarray]:
    """Build a (NPRODUCTS, navg2, tr_length) array. None if no products."""
    present = []
    for k in range(NPRODUCTS):
        pkt = spec_dict.get(k)
        if pkt is None:
            continue
        data = getattr(pkt, "data", None)
        if data is None:
            continue
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1) if arr.ndim == 1 else arr
        present.append((k, arr))
    if not present:
        return None
    navg2 = max(arr.shape[0] for _, arr in present)
    tr_length = max(arr.shape[1] for _, arr in present)
    out = np.full((NPRODUCTS, navg2, tr_length), np.nan, dtype=np.float32)
    for k, arr in present:
        out[k, :arr.shape[0], :arr.shape[1]] = arr
    return out


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------

def read_uncrater_session(session_dir: Path | str) -> Products:
    """Decode an uncrater session directory into a ``Products`` instance."""
    try:
        from uncrater import Collection  # type: ignore[import-not-found]
        from uncrater.Packet_Spectrum import Packet_Grimm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "uncrater package is required to decode session directories; "
            "install it (see lusee.ingest README) to use read_uncrater_session"
        ) from exc

    session_dir = Path(session_dir)
    cdi = session_dir / "cdi_output"
    if not cdi.is_dir():
        cdi = session_dir   # Layout A: bare directory of *.bin files

    coll = Collection(str(cdi))
    products = Products()

    # ---- Hello / session-invariants ----
    for pkt in coll.cont:
        if getattr(pkt, "appid", None) == 0x209:    # AppID_uC_Start
            try:
                pkt._read()
            except Exception as exc:    # noqa: BLE001
                warnings.warn(f"Hello decode failed: {exc}", RuntimeWarning, stacklevel=2)
                break
            products.sw_version = int(getattr(pkt, "SW_version", 0)) or None
            products.fw_version = int(getattr(pkt, "FW_Version", 0)) or None
            products.fw_id = int(getattr(pkt, "FW_ID", 0)) or None
            products.fw_date = int(getattr(pkt, "FW_Date", 0)) or None
            products.fw_time = int(getattr(pkt, "FW_Time", 0)) or None
            products.start_unique_packet_id = int(getattr(pkt, "unique_packet_id", 0)) or None
            t32 = getattr(pkt, "time_32", None)
            t16 = getattr(pkt, "time_16", None)
            if t32 is not None:
                products.start_time_32 = int(t32)
            if t16 is not None:
                products.start_time_16 = int(t16)
            if t32 is not None and t16 is not None:
                products.start_raw_seconds = raw_seconds_from_split_time(int(t32), int(t16))
            break

    # ---- Normal spectra ----
    for sd in getattr(coll, "spectra", []):
        meta = sd.get("meta")
        if meta is None:
            continue
        try:
            meta._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"metadata decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        data = _spectrum_dict_to_array(sd)
        sample = SpectrumSample(
            data=data,
            unique_packet_id=int(getattr(meta, "unique_packet_id", 0)),
            raw_seconds=_meta_raw_seconds(meta),
            metadata=extract_metadata(meta),
        )
        # Bit-slice restoration is part of ingestion, not physical-unit
        # calibration.  Abort rather than write ambiguous counts.
        sample.restore_bitslice()
        products.spectra.append(sample)

    # ---- TR spectra ----
    for sd in getattr(coll, "tr_spectra", []):
        meta = sd.get("meta")
        if meta is None:
            continue
        try:
            meta._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"TR metadata decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        data = _tr_spectrum_dict_to_array(sd)
        if data is None:
            continue
        navg2 = data.shape[1]
        tr_length = data.shape[2]
        products.tr_spectra.append(TRSpectrumSample(
            data=data,
            unique_packet_id=int(getattr(meta, "unique_packet_id", 0)),
            raw_seconds=_meta_raw_seconds(meta),
            navg2=navg2,
            tr_length=tr_length,
            metadata=extract_metadata(meta),
        ))

    # ---- Zoom spectra ----
    last_meta_seconds = 0.0
    spec_iter = iter(products.spectra)
    next_spec = next(spec_iter, None)
    for zpkt in getattr(coll, "zoom_spectra_packets", []):
        try:
            zpkt._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"zoom decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        zdata = np.asarray(getattr(zpkt, "data", []), dtype=np.float32)
        if zdata.size != ZOOM_COMPONENTS * ZOOM_BINS:
            zdata = zdata.reshape(-1)
            arr = np.zeros((ZOOM_COMPONENTS, ZOOM_BINS), dtype=np.float32)
            n = min(zdata.size, arr.size)
            arr.flat[:n] = zdata[:n]
        else:
            arr = zdata.reshape(ZOOM_COMPONENTS, ZOOM_BINS)
        # Zoom inherits the timestamp of the nearest preceding spectrum metadata.
        upid = int(getattr(zpkt, "unique_packet_id", 0))
        while next_spec is not None and next_spec.unique_packet_id <= upid:
            last_meta_seconds = next_spec.raw_seconds
            next_spec = next(spec_iter, None)
        products.zoom_spectra.append(ZoomSample(
            data=arr,
            unique_packet_id=upid,
            pfb_index=int(getattr(zpkt, "pfb_index", getattr(zpkt, "pfb", 0))),
            raw_seconds=last_meta_seconds,
        ))

    # ---- Waveforms ----
    for wpkt in getattr(coll, "waveform_packets", []):
        try:
            wpkt._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"waveform decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        data = np.asarray(getattr(wpkt, "data", []), dtype=np.int16)
        if data.size != WAVEFORM_SAMPLES:
            tmp = np.zeros(WAVEFORM_SAMPLES, dtype=np.int16)
            n = min(data.size, WAVEFORM_SAMPLES)
            tmp[:n] = data[:n]
            data = tmp
        ts = float(getattr(wpkt, "timestamp", 0.0) or 0.0)
        products.waveforms.append(WaveformSample(
            data=data,
            channel=int(getattr(wpkt, "channel", 0)),
            unique_packet_id=int(getattr(wpkt, "unique_packet_id", 0)),
            raw_seconds=ts,
        ))

    # ---- Housekeeping ----
    for hk in getattr(coll, "housekeeping_packets", []):
        try:
            hk._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"housekeeping decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        hk_type = int(getattr(hk, "hk_type", 0))
        # Surface raw_seconds for HK rows. uncrater exposes a "time" field
        # for hk_type 0/2 derived from the heartbeat / core_state mission
        # time; fall back to 0.0 for types that do not carry a time field.
        raw_seconds = 0.0
        t = getattr(hk, "time", None)
        if t is not None:
            try:
                raw_seconds = float(t)
            except (TypeError, ValueError):
                raw_seconds = 0.0
        sample = HKSample(
            hk_type=hk_type,
            version=int(getattr(hk, "version", 0)),
            unique_packet_id=int(getattr(hk, "unique_packet_id", 0)),
            errors=int(getattr(hk, "errors", 0)),
            raw_seconds=raw_seconds,
            fields={},
        )
        for attr in ("time", "ok", "checksum", "weight_ndx"):
            v = getattr(hk, attr, None)
            if v is not None:
                try:
                    sample.fields[attr] = type(v)(v) if isinstance(v, (int, bool, float)) else float(v)
                except (TypeError, ValueError):
                    pass
        for attr in ("adc_min", "adc_max", "adc_mean", "adc_rms"):
            v = getattr(hk, attr, None)
            if v is not None:
                sample.fields[attr] = _as_vec(v, 4)
        # actual_gain in hk_type=1 is delivered as a list of single-char
        # ASCII gain codes ('L'/'M'/'H'/'A'), not int64s as the spec
        # section 8.10 documents. Preserve as fixed-width ASCII bytes
        # so it round-trips faithfully through HDF5.
        ag = getattr(hk, "actual_gain", None)
        if ag is not None:
            try:
                if isinstance(ag, (list, tuple)) and ag and isinstance(ag[0], str):
                    sample.fields["actual_gain"] = np.array(
                        [s.encode("ascii", errors="replace")[:1] or b"?" for s in ag],
                        dtype="S1",
                    )
                else:
                    sample.fields["actual_gain"] = _as_vec(ag, 4)
            except Exception as exc:    # noqa: BLE001
                warnings.warn(
                    f"housekeeping actual_gain coercion failed ({exc}); skipping",
                    RuntimeWarning,
                    stacklevel=2,
                )
        for attr in dir(hk):
            if attr.startswith("telemetry") and not attr.startswith("_"):
                v = getattr(hk, attr, None)
                if v is None or callable(v):
                    continue
                try:
                    sample.fields[attr] = float(v)
                except (TypeError, ValueError):
                    pass
        products.housekeeping.append(sample)

    # ---- Calibrator data (variable-length per-channel arrays) ----
    for i, packet_data in enumerate(getattr(coll, "calib_data", [])):
        if not isinstance(packet_data, (list, tuple)) and not hasattr(packet_data, "__len__"):
            continue
        try:
            iter_ch = enumerate(packet_data)
        except TypeError:
            continue
        for j, ch_arr in iter_ch:
            if ch_arr is None:
                continue
            arr = np.asarray(ch_arr, dtype=np.float32)
            products.cal_data.append(CalDataSample(
                packet_idx=i,
                channel_idx=j,
                data=arr,
            ))

    # ---- Grimm spectra ----
    for pkt in getattr(coll, "cont", []):
        if getattr(pkt, "appid", None) != 0x2A0:
            continue
        try:
            pkt._read()
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"grimm decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        data = np.asarray(getattr(pkt, "data", []), dtype=np.float32)
        if data.ndim == 1:
            arr = np.full((NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
            n = min(data.size, NCHANNELS)
            arr[0, :n] = data[:n]
        else:
            arr = np.full((NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32)
            for k in range(min(data.shape[0], NPRODUCTS)):
                n = min(data.shape[1], NCHANNELS)
                arr[k, :n] = data[k, :n]
        products.grimm_spectra.append(SpectrumSample(
            data=arr,
            unique_packet_id=int(getattr(pkt, "unique_packet_id", 0)),
            raw_seconds=0.0,
            metadata={},
        ))

    log.info(
        "session %s: %d spectra, %d tr_spectra, %d zoom, %d waveforms, %d hk",
        session_dir,
        len(products.spectra),
        len(products.tr_spectra),
        len(products.zoom_spectra),
        len(products.waveforms),
        len(products.housekeeping),
    )
    return products
