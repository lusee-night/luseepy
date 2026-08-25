"""Stage 6: decode an uncrater session directory into in-memory products.

Wraps ``uncrater.Collection`` and converts its typed packet objects into
plain numpy arrays / dicts in a ``Products`` dataclass. ``hdf5_writer``
consumes the result; it never sees uncrater types.

Lazy-imports uncrater so that ``lusee.ingest`` is importable without it.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .constants import (
    BITSLICE_REFERENCE,
    NCHANNELS,
    NPRODUCTS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
)
from .frequency_contract import spectrometer_frequency_window
from .issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from .packet_map import (
    PACKET_MAP_FORMAT_VERSION,
    PacketMap,
    PacketMapError,
    read_packet_map,
)
from .products import (
    CalibratorDataSample,
    CalibratorDebugPage,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ProductProvenance,
    SpectrumMetadata,
    ValidatedCounts,
)
from .products import GrimmSample as ValidatedGrimmSample
from .products import HKSample as ValidatedHKSample
from .products import SpectrumSample as ValidatedSpectrumSample
from .products import TRSpectrumSample as ValidatedTRSpectrumSample
from .products import WaveformSample as ValidatedWaveformSample
from .products import ZoomSample as ValidatedZoomSample
from .session import raw_seconds_from_split_time
from .uncrater_adapter import (
    IncompatibleUncraterError,
    UncraterBindingInfo,
    binding_info,
    collection_provenance,
    import_decode_issues,
    load_uncrater,
    make_collection,
    read_packet,
    source_packet_provenance,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def _legacy_product_provenance() -> ProductProvenance:
    """Mark rows whose exact packet provenance awaits the repaired adapter."""
    return ProductProvenance.unavailable("legacy_adapter_provenance_pending")


def _decode_quality(
    provenance: DecodeProvenance,
    *,
    usable_product_count: int,
    adaptation_issue_count: int = 0,
) -> DataQuality:
    """Classify aggregate quality after product adaptation is complete."""
    if type(usable_product_count) is not int or usable_product_count < 0:
        raise ValueError("usable_product_count must be a nonnegative integer")
    if type(adaptation_issue_count) is not int or adaptation_issue_count < 0:
        raise ValueError("adaptation_issue_count must be a nonnegative integer")
    if usable_product_count == 0:
        return DataQuality.FAILED
    if (
        provenance.input_packet_count > 0
        and provenance.valid_packet_count == 0
    ):
        return DataQuality.FAILED
    if provenance.issue_counts or adaptation_issue_count:
        return DataQuality.PARTIAL
    if (
        provenance.valid_packet_count is not None
        and provenance.valid_packet_count < provenance.input_packet_count
    ):
        return DataQuality.PARTIAL
    return DataQuality.CLEAN


def _usable_product_count(products: "Products") -> int:
    """Count rows with a real payload after compatibility adaptation."""
    array_rows = (
        *products.spectra,
        *products.tr_spectra,
        *products.zoom_spectra,
        *products.waveforms,
        *products.cal_data,
        *products.grimm_spectra,
        *products.calibrator_data,
        *products.calibrator_raw_pfb,
    )
    return (
        len(products.housekeeping)
        + len(products.calibrator_metadata)
        + len(products.calibrator_debug)
        + sum(
        np.asarray(row.data).size > 0
        and bool(np.any(np.isfinite(row.data)))
        for row in array_rows
        )
    )


@dataclass
class SpectrumSample:
    """Mutable layout-v2/v3 compatibility row, not a validated record."""

    data: np.ndarray              # shape (NPRODUCTS, NCHANNELS), float32, NaN where missing
    unique_packet_id: int
    raw_seconds: float | None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # True means ``data`` has already been restored with actual_bitslice to
    # the bit-31 accumulator convention used by the gain-model artifacts.
    bitslice_restored: bool = False
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )

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
    """Mutable layout-v2/v3 compatibility row, not a validated record."""

    data: np.ndarray              # shape (NPRODUCTS, navg2, tr_length), float32
    unique_packet_id: int
    raw_seconds: float | None
    navg2: int
    tr_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )


@dataclass
class ZoomSample:
    data: np.ndarray              # shape (4, ZOOM_BINS), float32
    unique_packet_id: int
    pfb_index: int
    raw_seconds: float | None     # inherited from preceding spectrum metadata
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )


@dataclass
class WaveformSample:
    data: np.ndarray              # shape (WAVEFORM_SAMPLES,) int16
    channel: int
    unique_packet_id: int
    raw_seconds: float | None
    adc_timestamp: np.uint64 | None = None
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )


@dataclass
class HKSample:
    hk_type: int
    version: int
    unique_packet_id: int
    errors: int
    raw_seconds: float | None = None
    fields: Dict[str, Any] = field(default_factory=dict)
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )


@dataclass
class CalDataSample:
    packet_idx: int
    channel_idx: int
    data: np.ndarray              # variable-length float32
    provenance: ProductProvenance = field(
        default_factory=_legacy_product_provenance
    )


@dataclass
class Products:
    spectra: List[SpectrumSample | ValidatedSpectrumSample] = field(
        default_factory=list
    )
    tr_spectra: List[TRSpectrumSample | ValidatedTRSpectrumSample] = field(
        default_factory=list
    )
    zoom_spectra: List[ZoomSample | ValidatedZoomSample] = field(
        default_factory=list
    )
    grimm_spectra: List[SpectrumSample | ValidatedGrimmSample] = field(
        default_factory=list
    )
    waveforms: List[WaveformSample | ValidatedWaveformSample] = field(
        default_factory=list
    )
    housekeeping: List[HKSample | ValidatedHKSample] = field(
        default_factory=list
    )
    cal_data: List[CalDataSample] = field(default_factory=list)
    calibrator_metadata: List[CalibratorMetadataSample] = field(
        default_factory=list
    )
    calibrator_data: List[CalibratorDataSample] = field(default_factory=list)
    calibrator_raw_pfb: List[CalibratorRawPFBSample] = field(
        default_factory=list
    )
    calibrator_debug: List[CalibratorDebugSample] = field(default_factory=list)

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
    decode_provenance: DecodeProvenance = field(
        default_factory=DecodeProvenance.unavailable
    )
    packet_map_status: str = "unavailable"
    packet_map_format_version: int | None = None
    raw_flash_provenance_unavailable_reason: str | None = "packet_map_not_loaded"
    family_issue_ids: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    quality_status: DataQuality | None = None
    validated_counts: ValidatedCounts = field(default_factory=ValidatedCounts)
    issues: tuple[IngestIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decode_provenance, DecodeProvenance):
            raise TypeError("decode_provenance must be a DecodeProvenance record")
        if self.packet_map_status not in ("verified", "unavailable"):
            raise ValueError("packet_map_status must be verified or unavailable")
        if self.packet_map_status == "verified":
            if self.packet_map_format_version != PACKET_MAP_FORMAT_VERSION:
                raise ValueError("verified packet map has the wrong format version")
            if self.raw_flash_provenance_unavailable_reason is not None:
                raise ValueError(
                    "verified packet map cannot have an unavailable reason"
                )
        else:
            if self.packet_map_format_version is not None:
                raise ValueError("unavailable packet map cannot have a format version")
            if not isinstance(
                self.raw_flash_provenance_unavailable_reason,
                str,
            ) or not (
                self.raw_flash_provenance_unavailable_reason
            ):
                raise ValueError("unavailable packet map requires a reason")
        known_families = {
            "spectra",
            "tr_spectra",
            "zoom_spectra",
            "waveforms",
            "housekeeping",
            "grimm_spectra",
            "calibrator_metadata",
            "calibrator_data",
            "calibrator_raw_pfb",
            "calibrator_debug",
        }
        normalized_family_issues = {}
        for family, issue_ids in self.family_issue_ids.items():
            if family not in known_families:
                raise ValueError(f"unknown product family {family!r}")
            issue_ids = tuple(issue_ids)
            if any(
                not isinstance(issue_id, str) or not issue_id
                for issue_id in issue_ids
            ):
                raise ValueError("family issue IDs must be nonempty strings")
            if len(set(issue_ids)) != len(issue_ids):
                raise ValueError("family issue IDs must not contain duplicates")
            normalized_family_issues[family] = issue_ids
        self.family_issue_ids = normalized_family_issues
        if self.quality_status is not None:
            self.quality_status = DataQuality(self.quality_status)
        if not isinstance(self.validated_counts, ValidatedCounts):
            raise TypeError("validated_counts must be a ValidatedCounts record")
        self.issues = tuple(self.issues)
        if any(not isinstance(issue, IngestIssue) for issue in self.issues):
            raise TypeError("issues must contain IngestIssue records")

    @property
    def execution_mode(self) -> ExecutionMode | None:
        """Decoder strictness, intentionally separate from data quality."""
        return self.decode_provenance.execution_mode

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


def _meta_raw_seconds(meta_pkt) -> float | None:
    """Mission-time seconds for a metadata packet.

    Modern uncrater (SW 0x307+) exposes a pre-decoded ``time`` attribute
    on Packet_Metadata; older schemas exposed split ``time_32`` /
    ``time_16`` (or the underscored variants). Try the decoded form
    first, then fall back to recomputing from the split fields.
    """
    t = getattr(meta_pkt, "time", None)
    if t is not None:
        result = _finite_seconds_or_none(t)
        if result is not None:
            return result
    t32 = getattr(meta_pkt, "_time_32", None)
    if t32 is None:
        t32 = getattr(meta_pkt, "time_32", None)
    t16 = getattr(meta_pkt, "_time_16", None)
    if t16 is None:
        t16 = getattr(meta_pkt, "time_16", None)
    if t32 is None or t16 is None:
        return None
    return raw_seconds_from_split_time(int(t32), int(t16))


def _finite_seconds_or_none(value: object) -> float | None:
    """Return a finite seconds value, or None for missing/invalid input."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _waveform_adc_timestamp(packet: object) -> np.uint64 | None:
    """Return an ADC timestamp only when waveform metadata was attached."""
    if getattr(packet, "meta", None) is None:
        return None
    value = getattr(packet, "timestamp", None)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        return None
    integer = int(value)
    if not 0 <= integer <= np.iinfo(np.uint64).max:
        return None
    return np.uint64(integer)


_CURRENT_METADATA_BINDINGS = frozenset(
    {"305", "306-early", "306-final", "307"}
)
_SUPPORTED_METADATA_BINDINGS = _CURRENT_METADATA_BINDINGS | {"203"}


def _required_public_attribute(value: object, name: str) -> object:
    if not hasattr(value, name):
        raise IncompatibleUncraterError(
            f"uncrater {type(value).__name__} has no public {name} field"
        )
    return getattr(value, name)


def _required_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must lie in [{minimum}, {maximum}]; got {result}"
        )
    return result


def _required_uint(value: object, *, name: str, bits: int) -> int:
    return _required_integer(
        value,
        name=name,
        minimum=0,
        maximum=(1 << bits) - 1,
    )


def _exact_integer_vector(
    value: object,
    *,
    name: str,
    length: int,
    dtype: np.dtype | type,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of {length} integers") from exc
    if len(items) != length:
        raise ValueError(f"{name} has {len(items)} values; expected {length}")
    normalized = [
        _required_integer(
            item,
            name=f"{name}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )
        for index, item in enumerate(items)
    ]
    return np.asarray(normalized, dtype=dtype)


def _exact_uint_vector(
    value: object,
    *,
    name: str,
    length: int,
    bits: int,
    dtype: np.dtype | type,
) -> np.ndarray:
    return _exact_integer_vector(
        value,
        name=name,
        length=length,
        dtype=dtype,
        minimum=0,
        maximum=(1 << bits) - 1,
    )


def _exact_float_vector(
    value: object,
    *,
    name: str,
    length: int,
) -> np.ndarray:
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of {length} values") from exc
    if len(items) != length:
        raise ValueError(f"{name} has {len(items)} values; expected {length}")
    normalized = []
    for index, item in enumerate(items):
        number = _finite_seconds_or_none(item)
        if number is None:
            raise ValueError(f"{name}[{index}] must be finite")
        normalized.append(number)
    return np.asarray(normalized, dtype=np.float64)


def _required_boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _required_finite_float(value: object, *, name: str) -> float:
    result = _finite_seconds_or_none(value)
    if result is None:
        raise ValueError(f"{name} must be finite")
    return result


def _packet_binding_key(packet: object) -> str:
    schema = _required_public_attribute(packet, "schema")
    key = _required_public_attribute(schema, "binding_key")
    if not isinstance(key, str) or not key:
        raise IncompatibleUncraterError(
            "uncrater packet schema.binding_key must be a nonempty string"
        )
    return key


def _validate_science_packet_identity(
    packet: object,
    *,
    family: str,
) -> None:
    max_priority = 4 if family == "normal" else 3
    _required_integer(
        _required_public_attribute(packet, "priority"),
        name=f"{family} packet priority",
        minimum=1,
        maximum=max_priority,
    )
    _required_uint(
        _required_public_attribute(packet, "unique_packet_id"),
        name=f"{family} packet unique_packet_id",
        bits=32,
    )
    _required_uint(
        _required_public_attribute(packet, "crc"),
        name=f"{family} packet CRC",
        bits=32,
    )


def _extract_route(base: object) -> tuple[np.ndarray, np.ndarray]:
    route = _required_public_attribute(base, "route")
    try:
        entries = list(route)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("metadata route must be an iterable of four entries") from exc
    if len(entries) != 4:
        raise ValueError(f"metadata route has {len(entries)} entries; expected 4")
    plus = np.asarray(
        [
            _required_uint(
                _required_public_attribute(entry, "plus"),
                name=f"route[{index}].plus",
                bits=8,
            )
            for index, entry in enumerate(entries)
        ],
        dtype=np.uint8,
    )
    minus = np.asarray(
        [
            _required_uint(
                _required_public_attribute(entry, "minus"),
                name=f"route[{index}].minus",
                bits=8,
            )
            for index, entry in enumerate(entries)
        ],
        dtype=np.uint8,
    )
    return plus, minus


def _extract_spectrum_metadata(
    meta: object,
    *,
    binding_key: str,
) -> SpectrumMetadata:
    """Normalize one reviewed public Packet_Metadata without loose fallbacks."""
    read_packet(meta)
    if _packet_binding_key(meta) != binding_key:
        raise ValueError("metadata packet binding disagrees with Collection")
    if binding_key not in _SUPPORTED_METADATA_BINDINGS:
        raise IncompatibleUncraterError(
            f"metadata binding {binding_key!r} has no luseepy field map"
        )

    base = _required_public_attribute(meta, "base")
    version = _required_uint(
        _required_public_attribute(meta, "version"),
        name="metadata version",
        bits=16,
    )
    reported_version = _required_public_attribute(meta, "reported_version")
    if reported_version is not None:
        reported_version = _required_uint(
            reported_version,
            name="metadata reported_version",
            bits=16,
        )
        if version != reported_version:
            raise ValueError(
                "metadata version disagrees with packet reported_version"
            )
    unique_packet_id = _required_uint(
        _required_public_attribute(meta, "unique_packet_id"),
        name="metadata unique_packet_id",
        bits=32,
    )
    uc_time = _required_uint(
        _required_public_attribute(base, "uC_time"),
        name="metadata uC_time",
        bits=64,
    )
    time_32 = _required_uint(
        _required_public_attribute(base, "time_32"),
        name="metadata time_32",
        bits=32,
    )
    time_16 = _required_uint(
        _required_public_attribute(base, "time_16"),
        name="metadata time_16",
        bits=16,
    )
    route_plus, route_minus = _extract_route(base)

    navgf = _required_uint(
        _required_public_attribute(base, "Navgf"),
        name="metadata Navgf",
        bits=8,
    )
    frequency_contract = spectrometer_frequency_window(navgf)
    expected_frequency_count = _required_uint(
        _required_public_attribute(meta, "expected_frequency_count"),
        name="metadata expected_frequency_count",
        bits=16,
    )
    if expected_frequency_count != frequency_contract.output_count:
        raise ValueError(
            "metadata expected_frequency_count disagrees with the firmware contract"
        )

    output_format = _required_uint(
        _required_public_attribute(base, "format"),
        name="metadata format",
        bits=8,
    )
    public_format = _required_uint(
        _required_public_attribute(meta, "format"),
        name="metadata public format",
        bits=8,
    )
    if output_format != public_format:
        raise ValueError("metadata public format disagrees with base.format")
    errors = _required_uint(
        _required_public_attribute(base, "errors"),
        name="metadata errors",
        bits=32,
    )
    public_errors = _required_uint(
        _required_public_attribute(meta, "errormask"),
        name="metadata errormask",
        bits=32,
    )
    if errors != public_errors:
        raise ValueError("metadata errormask disagrees with base.errors")

    weight = _required_uint(
        _required_public_attribute(meta, "weight"),
        name="metadata weight",
        bits=16,
    )
    weight_field = "weight_previous" if binding_key == "203" else "weight"
    base_weight = _required_uint(
        _required_public_attribute(base, weight_field),
        name=f"metadata {weight_field}",
        bits=16,
    )
    if weight != base_weight:
        raise ValueError(f"metadata public weight disagrees with base.{weight_field}")

    public_time = _required_finite_float(
        _required_public_attribute(meta, "time"),
        name="metadata time",
    )
    split_time = raw_seconds_from_split_time(time_32, time_16)
    if public_time != split_time:
        raise ValueError("metadata public time disagrees with split mission time")

    current = binding_key in _CURRENT_METADATA_BINDINGS

    def current_uint(name: str, bits: int) -> int | None:
        if not current:
            return None
        return _required_uint(
            _required_public_attribute(base, name),
            name=f"metadata {name}",
            bits=bits,
        )

    return SpectrumMetadata(
        version=version,
        unique_packet_id=unique_packet_id,
        uc_time=uc_time,
        time_32=time_32,
        time_16=time_16,
        tvs_sensors=_exact_uint_vector(
            _required_public_attribute(base, "TVS_sensors"),
            name="metadata TVS_sensors",
            length=4,
            bits=16,
            dtype=np.uint16,
        ),
        requested_gain=_exact_uint_vector(
            _required_public_attribute(base, "gain"),
            name="metadata gain",
            length=4,
            bits=8,
            dtype=np.uint8,
        ),
        gain_auto_min=_exact_uint_vector(
            _required_public_attribute(base, "gain_auto_min"),
            name="metadata gain_auto_min",
            length=4,
            bits=16,
            dtype=np.uint16,
        ),
        gain_auto_mult=_exact_uint_vector(
            _required_public_attribute(base, "gain_auto_mult"),
            name="metadata gain_auto_mult",
            length=4,
            bits=16,
            dtype=np.uint16,
        ),
        route_plus=route_plus,
        route_minus=route_minus,
        navg1_shift=_required_uint(
            _required_public_attribute(base, "Navg1_shift"),
            name="metadata Navg1_shift",
            bits=8,
        ),
        navg2_shift=_required_uint(
            _required_public_attribute(base, "Navg2_shift"),
            name="metadata Navg2_shift",
            bits=8,
        ),
        notch=_required_uint(
            _required_public_attribute(base, "notch"),
            name="metadata notch",
            bits=8,
        ),
        navgf=navgf,
        high_fraction=_required_uint(
            _required_public_attribute(base, "hi_frac"),
            name="metadata hi_frac",
            bits=8,
        ),
        medium_fraction=_required_uint(
            _required_public_attribute(base, "med_frac"),
            name="metadata med_frac",
            bits=8,
        ),
        requested_bitslice=_exact_uint_vector(
            _required_public_attribute(base, "bitslice"),
            name="metadata bitslice",
            length=NPRODUCTS,
            bits=8,
            dtype=np.uint8,
        ),
        bitslice_keep_bits=_required_uint(
            _required_public_attribute(base, "bitslice_keep_bits"),
            name="metadata bitslice_keep_bits",
            bits=8,
        ),
        output_format=output_format,
        reject_ratio=_required_uint(
            _required_public_attribute(base, "reject_ratio"),
            name="metadata reject_ratio",
            bits=8,
        ),
        reject_max_bad=_required_uint(
            _required_public_attribute(base, "reject_maxbad"),
            name="metadata reject_maxbad",
            bits=8,
        ),
        tr_start=_required_uint(
            _required_public_attribute(base, "tr_start"),
            name="metadata tr_start",
            bits=16,
        ),
        tr_stop=_required_uint(
            _required_public_attribute(base, "tr_stop"),
            name="metadata tr_stop",
            bits=16,
        ),
        tr_average_shift=_required_uint(
            _required_public_attribute(base, "tr_avg_shift"),
            name="metadata tr_avg_shift",
            bits=16,
        ),
        errors=errors,
        correlation_products_mask=_required_uint(
            _required_public_attribute(base, "corr_products_mask"),
            name="metadata corr_products_mask",
            bits=16,
        ),
        actual_gain=_exact_uint_vector(
            _required_public_attribute(base, "actual_gain"),
            name="metadata actual_gain",
            length=4,
            bits=8,
            dtype=np.uint8,
        ),
        actual_bitslice=_exact_uint_vector(
            _required_public_attribute(base, "actual_bitslice"),
            name="metadata actual_bitslice",
            length=NPRODUCTS,
            bits=8,
            dtype=np.uint8,
        ),
        spectrum_overflow=_required_uint(
            _required_public_attribute(base, "spec_overflow"),
            name="metadata spec_overflow",
            bits=16,
        ),
        notch_overflow=_required_uint(
            _required_public_attribute(base, "notch_overflow"),
            name="metadata notch_overflow",
            bits=16,
        ),
        adc_min=_exact_integer_vector(
            _required_public_attribute(meta, "adc_min"),
            name="metadata adc_min",
            length=4,
            dtype=np.int64,
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        ),
        adc_max=_exact_integer_vector(
            _required_public_attribute(meta, "adc_max"),
            name="metadata adc_max",
            length=4,
            dtype=np.int64,
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        ),
        adc_valid_count=_exact_integer_vector(
            _required_public_attribute(meta, "adc_valid_count"),
            name="metadata adc_valid_count",
            length=4,
            dtype=np.int64,
            minimum=0,
            maximum=(1 << 63) - 1,
        ),
        adc_invalid_count_max=_exact_integer_vector(
            _required_public_attribute(meta, "adc_invalid_count_max"),
            name="metadata adc_invalid_count_max",
            length=4,
            dtype=np.int64,
            minimum=0,
            maximum=(1 << 63) - 1,
        ),
        adc_invalid_count_min=_exact_integer_vector(
            _required_public_attribute(meta, "adc_invalid_count_min"),
            name="metadata adc_invalid_count_min",
            length=4,
            dtype=np.int64,
            minimum=0,
            maximum=(1 << 63) - 1,
        ),
        adc_total_count=_exact_integer_vector(
            _required_public_attribute(meta, "adc_total_count"),
            name="metadata adc_total_count",
            length=4,
            dtype=np.int64,
            minimum=0,
            maximum=(1 << 63) - 1,
        ),
        adc_mean=_exact_float_vector(
            _required_public_attribute(meta, "adc_mean"),
            name="metadata adc_mean",
            length=4,
        ),
        adc_rms=_exact_float_vector(
            _required_public_attribute(meta, "adc_rms"),
            name="metadata adc_rms",
            length=4,
        ),
        spectrometer_enable=_required_boolean(
            _required_public_attribute(base, "spectrometer_enable"),
            name="metadata spectrometer_enable",
        ),
        calibrator_enable=_required_boolean(
            _required_public_attribute(base, "calibrator_enable"),
            name="metadata calibrator_enable",
        ),
        random_state=_required_uint(
            _required_public_attribute(base, "rand_state"),
            name="metadata rand_state",
            bits=32,
        ),
        weight=weight,
        weight_current=_required_uint(
            _required_public_attribute(base, "weight_current"),
            name="metadata weight_current",
            bits=16,
        ),
        telemetry_v1_0=_required_finite_float(
            _required_public_attribute(meta, "telemetry_V1_0"),
            name="metadata telemetry_V1_0",
        ),
        telemetry_v1_8=_required_finite_float(
            _required_public_attribute(meta, "telemetry_V1_8"),
            name="metadata telemetry_V1_8",
        ),
        telemetry_v2_5=_required_finite_float(
            _required_public_attribute(meta, "telemetry_V2_5"),
            name="metadata telemetry_V2_5",
        ),
        telemetry_t_fpga=_required_finite_float(
            _required_public_attribute(meta, "telemetry_T_FPGA"),
            name="metadata telemetry_T_FPGA",
        ),
        loop_count_min=current_uint("loop_count_min", 16),
        loop_count_max=current_uint("loop_count_max", 16),
        grimm_enable=current_uint("grimm_enable", 8),
        averaging_mode=current_uint("averaging_mode", 8),
        num_bad_min_current=current_uint("num_bad_min_current", 16),
        num_bad_max_current=current_uint("num_bad_max_current", 16),
        num_bad_min=current_uint("num_bad_min", 16),
        num_bad_max=current_uint("num_bad_max", 16),
    )


def _packet_has_fatal_issue(packet: object) -> bool:
    status = _required_public_attribute(packet, "decode_status")
    issues = _required_public_attribute(status, "issues")
    if not isinstance(issues, tuple):
        raise IncompatibleUncraterError(
            "uncrater DecodeStatus.issues must be an immutable tuple"
        )
    return any(bool(_required_public_attribute(issue, "fatal")) for issue in issues)


def _exact_ndarray(
    value: object,
    *,
    name: str,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    finite: bool = False,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(
            f"{name} must have dtype {expected_dtype}; got {value.dtype}"
        )
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    if finite and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _record_kept_adapter_issue(
    issue_collector: IssueCollector,
    *,
    code: str,
    message: str,
    packet: object,
    uid: int | None,
    details: Mapping[str, object],
) -> str:
    packet_index = _required_public_attribute(packet, "packet_index")
    appid = _required_public_attribute(packet, "appid")
    issue = issue_collector.record(
        code=code,
        severity=IssueSeverity.WARNING,
        stage="decode_adapter",
        message=message,
        action=IssueAction.KEPT,
        packet_index=(
            packet_index if type(packet_index) is int and packet_index >= 0 else None
        ),
        appid=appid if type(appid) is int and 0 <= appid <= 0x7FF else None,
        uid=uid,
        details=details,
    )
    return issue.issue_id


def _auxiliary_provenance(
    *,
    packets: Sequence[object],
    roles: Sequence[str],
    unique_packet_id: int,
    uid_source: str,
    uid_source_role: str,
    raw_seconds: float | None,
    time_source: str | None,
    time_source_role: str | None,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    adapter_issue_ids: Sequence[str] = (),
) -> ProductProvenance:
    if len(packets) != len(roles) or not packets:
        raise ValueError("auxiliary provenance packets and roles must align")
    source_packets = tuple(
        source_packet_provenance(
            packet,
            role=role,
            binding=selected_binding,
        )
        for packet, role in zip(packets, roles)
    )
    issue_ids = _issue_ids_for_packets(packets, issue_ids_by_packet_index)
    for issue_id in adapter_issue_ids:
        if issue_id not in issue_ids:
            issue_ids.append(issue_id)
    return ProductProvenance(
        source_packets=source_packets,
        uid=unique_packet_id,
        uid_source=uid_source,
        uid_source_role=uid_source_role,
        reported_schema_ids=selected_binding.reported_schema_ids,
        selected_schema_id=selected_binding.selected_schema_id,
        decoder_issue_ids=tuple(issue_ids),
        time_source=time_source if raw_seconds is not None else None,
        time_source_role=time_source_role if raw_seconds is not None else None,
        clock_source="spectrometer" if raw_seconds is not None else None,
        time_valid=raw_seconds is not None,
    )


def _metadata_uid_lookup(
    metadata_packets: Sequence[object],
    normalized_metadata: Mapping[int, SpectrumMetadata | None],
) -> dict[int, tuple[tuple[object, SpectrumMetadata], ...]]:
    lookup: dict[int, list[tuple[object, SpectrumMetadata]]] = defaultdict(list)
    for packet in metadata_packets:
        metadata = normalized_metadata.get(id(packet))
        if metadata is not None:
            lookup[metadata.unique_packet_id].append((packet, metadata))
    return {
        uid: tuple(items)
        for uid, items in lookup.items()
    }


def _associate_metadata_by_uid(
    *,
    packet: object,
    unique_packet_id: int,
    family: str,
    lookup: Mapping[int, tuple[tuple[object, SpectrumMetadata], ...]],
    issue_collector: IssueCollector,
) -> tuple[object | None, SpectrumMetadata | None, tuple[str, ...]]:
    matches = lookup.get(unique_packet_id, ())
    if len(matches) == 1:
        metadata_packet, metadata = matches[0]
        return metadata_packet, metadata, ()
    if not matches:
        return None, None, ()
    packet_indices = [
        _required_public_attribute(metadata_packet, "packet_index")
        for metadata_packet, _ in matches
    ]
    issue_id = _record_kept_adapter_issue(
        issue_collector,
        code=f"decode_adapter.ambiguous_{family}_time_metadata",
        message=(
            f"{family} UID {unique_packet_id} matches multiple spectrum "
            "metadata packets; mission time remains unavailable"
        ),
        packet=packet,
        uid=unique_packet_id,
        details={"metadata_packet_indices": packet_indices},
    )
    return None, None, (issue_id,)


def _required_group_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise IncompatibleUncraterError(
            f"uncrater {name} entries must be dictionaries"
        )
    if frozenset(value) != keys:
        raise IncompatibleUncraterError(
            f"uncrater {name} entry keys disagree with the reviewed contract"
        )
    return value


def _validated_group_pages(
    group: Mapping[str, object],
    *,
    name: str,
    packet_class: type,
    page_count: int,
    unique_packet_id: int,
    selected_binding: UncraterBindingInfo,
) -> tuple[tuple[object, ...], tuple[float, ...]]:
    pages = group["pages"]
    if type(pages) is not tuple or len(pages) != page_count:
        raise IncompatibleUncraterError(
            f"uncrater {name} pages must be an exact {page_count}-packet tuple"
        )
    times = []
    for page_index, packet in enumerate(pages):
        if not isinstance(packet, packet_class):
            raise IncompatibleUncraterError(
                f"uncrater {name} page {page_index} has an invalid class"
            )
        if _packet_has_fatal_issue(packet):
            raise IncompatibleUncraterError(
                f"uncrater {name} published a fatal page"
            )
        if _packet_binding_key(packet) != selected_binding.binding_key:
            raise IncompatibleUncraterError(
                f"uncrater {name} page binding disagrees with Collection"
            )
        page_uid = _required_uint(
            _required_public_attribute(packet, "unique_packet_id"),
            name=f"{name} page {page_index} unique_packet_id",
            bits=32,
        )
        if page_uid != unique_packet_id:
            raise ValueError(f"{name} page UID disagrees with its group UID")
        times.append(_required_finite_float(
            _required_public_attribute(packet, "time"),
            name=f"{name} page {page_index} time",
        ))
    return pages, tuple(times)


def _validate_spectrum_groups(
    collection: object,
    *,
    name: str,
    packet_class: type,
    metadata_class: type,
) -> tuple[object, ...]:
    groups = _required_public_attribute(collection, name)
    if not isinstance(groups, (list, tuple)):
        raise IncompatibleUncraterError(
            f"uncrater Collection.{name} must be a group sequence"
        )
    metadata_packets = []
    seen_metadata: set[int] = set()
    for group in groups:
        if type(group) is not dict:
            raise IncompatibleUncraterError(
                f"uncrater Collection.{name} groups must be dictionaries"
            )
        if "meta" not in group or not isinstance(group["meta"], metadata_class):
            raise IncompatibleUncraterError(
                f"uncrater Collection.{name} group has no Packet_Metadata"
            )
        meta = group["meta"]
        if _packet_has_fatal_issue(meta):
            raise IncompatibleUncraterError(
                f"uncrater Collection.{name} published fatal metadata"
            )
        if id(meta) in seen_metadata:
            raise IncompatibleUncraterError(
                f"uncrater Collection.{name} repeats one metadata packet"
            )
        seen_metadata.add(id(meta))
        metadata_packets.append(meta)
        for product, packet in group.items():
            if product == "meta":
                continue
            if type(product) is not int or not 0 <= product < NPRODUCTS:
                raise IncompatibleUncraterError(
                    f"uncrater Collection.{name} has an invalid product key"
                )
            if not isinstance(packet, packet_class):
                raise IncompatibleUncraterError(
                    f"uncrater Collection.{name} has an invalid packet class"
                )
            if _required_public_attribute(packet, "meta") is not meta:
                raise IncompatibleUncraterError(
                    f"uncrater Collection.{name} packet metadata disagrees"
                )
            if _required_public_attribute(packet, "product") != product:
                raise IncompatibleUncraterError(
                    f"uncrater Collection.{name} packet product disagrees"
                )
            if _packet_has_fatal_issue(packet):
                raise IncompatibleUncraterError(
                    f"uncrater Collection.{name} published a fatal packet"
                )
    return tuple(metadata_packets)


def _candidate_packets(
    collection: object,
    *,
    packet_class: type,
) -> dict[int, dict[int, list[object]]]:
    candidates: dict[int, dict[int, list[object]]] = defaultdict(
        lambda: defaultdict(list)
    )
    packets = _required_public_attribute(collection, "cont")
    if not isinstance(packets, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.cont must be a packet sequence"
        )
    for packet in packets:
        if not isinstance(packet, packet_class):
            continue
        meta = getattr(packet, "meta", None)
        if meta is None:
            continue
        product = _required_public_attribute(packet, "product")
        if type(product) is not int or not 0 <= product < NPRODUCTS:
            raise IncompatibleUncraterError(
                "uncrater spectrum packet has an invalid product field"
            )
        candidates[id(meta)][product].append(packet)
    return {
        meta_id: {product: list(items) for product, items in per_product.items()}
        for meta_id, per_product in candidates.items()
    }


def _record_adapter_issue(
    issue_collector: IssueCollector,
    *,
    code: str,
    message: str,
    packet: object,
    uid: int | None,
    details: dict[str, object],
) -> str:
    packet_index = _required_public_attribute(packet, "packet_index")
    appid = _required_public_attribute(packet, "appid")
    issue = issue_collector.record(
        code=code,
        severity=IssueSeverity.ERROR,
        stage="decode_adapter",
        message=message,
        action=IssueAction.DROPPED,
        packet_index=(
            packet_index if type(packet_index) is int and packet_index >= 0 else None
        ),
        appid=appid if type(appid) is int and 0 <= appid <= 0x7FF else None,
        uid=uid,
        details=details,
    )
    return issue.issue_id


def _issue_ids_for_packets(
    packets: Sequence[object],
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
) -> list[str]:
    issue_ids = []
    seen = set()
    for packet in packets:
        packet_index = _required_public_attribute(packet, "packet_index")
        if type(packet_index) is not int:
            raise IncompatibleUncraterError(
                "uncrater packet_index must be an integer"
            )
        for issue_id in issue_ids_by_packet_index.get(packet_index, ()):
            if issue_id not in seen:
                issue_ids.append(issue_id)
                seen.add(issue_id)
    return issue_ids


def _product_provenance(
    *,
    meta: object,
    accepted_packets: Sequence[object],
    all_candidates: Sequence[object],
    family: str,
    metadata: SpectrumMetadata,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    adapter_issue_ids: Sequence[str],
) -> ProductProvenance:
    source_packets = [
        source_packet_provenance(
            meta,
            role="metadata",
            binding=selected_binding,
        )
    ]
    source_packets.extend(
        source_packet_provenance(
            packet,
            role=f"{family}_product_{product:02d}",
            binding=selected_binding,
        )
        for packet in accepted_packets
        for product in (_required_public_attribute(packet, "product"),)
    )
    issue_ids = _issue_ids_for_packets(
        [meta, *all_candidates], issue_ids_by_packet_index
    )
    for issue_id in adapter_issue_ids:
        if issue_id not in issue_ids:
            issue_ids.append(issue_id)
    raw_seconds = metadata.raw_seconds
    return ProductProvenance(
        source_packets=tuple(source_packets),
        uid=metadata.unique_packet_id,
        uid_source="Packet_Metadata.unique_packet_id",
        uid_source_role="metadata",
        reported_schema_ids=selected_binding.reported_schema_ids,
        selected_schema_id=selected_binding.selected_schema_id,
        decoder_issue_ids=tuple(issue_ids),
        time_source=(
            "Packet_Metadata.base.time_32_time_16"
            if raw_seconds is not None
            else None
        ),
        time_source_role="metadata" if raw_seconds is not None else None,
        clock_source="spectrometer",
        time_valid=raw_seconds is not None,
    )


def _adapt_normal_spectrum(
    *,
    meta: object,
    metadata: SpectrumMetadata,
    candidates: dict[int, list[object]],
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> ValidatedSpectrumSample | None:
    contract = spectrometer_frequency_window(metadata.navgf)
    data = np.full(
        (NPRODUCTS, contract.output_count), np.nan, dtype=np.float32
    )
    product_present = np.zeros(NPRODUCTS, dtype=np.bool_)
    accepted_packets = []
    all_candidates = []
    adapter_issue_ids = []
    for product in range(NPRODUCTS):
        packets = candidates.get(product, [])
        all_candidates.extend(packets)
        valid_packets = []
        restored_packets = []
        for packet in packets:
            if _packet_has_fatal_issue(packet):
                continue
            try:
                _validate_science_packet_identity(packet, family="normal")
                if _packet_binding_key(packet) != selected_binding.binding_key:
                    raise ValueError("packet binding disagrees with Collection")
                if _required_public_attribute(packet, "meta") is not meta:
                    raise ValueError("packet metadata association disagrees")
                packet_data = _required_public_attribute(packet, "data")
                if type(packet_data) is not np.ndarray:
                    raise TypeError("normal packet data must be a numpy.ndarray")
                if packet_data.dtype != np.dtype(np.float64):
                    raise TypeError(
                        "normal packet data must retain uncrater float64 dtype"
                    )
                if packet_data.shape != (contract.output_count,):
                    raise ValueError(
                        "normal packet data shape disagrees with metadata Navgf"
                    )
                restored = np.ldexp(
                    packet_data.astype(np.float32),
                    int(metadata.actual_bitslice[product]) - BITSLICE_REFERENCE,
                )
                if not np.all(np.isfinite(restored)):
                    raise ValueError("restored normal packet data is not finite")
            except (TypeError, ValueError) as exc:
                adapter_issue_ids.append(_record_adapter_issue(
                    issue_collector,
                    code="decode_adapter.invalid_normal_product",
                    message=(
                        f"normal spectrum UID {metadata.unique_packet_id} product "
                        f"{product} was dropped: {exc}"
                    ),
                    packet=packet,
                    uid=metadata.unique_packet_id,
                    details={"error": str(exc), "product": product},
                ))
                continue
            valid_packets.append(packet)
            restored_packets.append(restored)
        if len(valid_packets) > 1:
            indices = [
                _required_public_attribute(packet, "packet_index")
                for packet in valid_packets
            ]
            adapter_issue_ids.append(_record_adapter_issue(
                issue_collector,
                code="decode_adapter.duplicate_normal_product",
                message=(
                    f"normal spectrum UID {metadata.unique_packet_id} has "
                    f"{len(valid_packets)} valid packets for product {product}"
                ),
                packet=valid_packets[0],
                uid=metadata.unique_packet_id,
                details={"packet_indices": indices, "product": product},
            ))
            continue
        if not valid_packets:
            continue
        packet = valid_packets[0]
        restored = restored_packets[0]
        data[product] = restored
        product_present[product] = True
        accepted_packets.append(packet)
    if not accepted_packets:
        return None
    provenance = _product_provenance(
        meta=meta,
        accepted_packets=accepted_packets,
        all_candidates=all_candidates,
        family="normal",
        metadata=metadata,
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
        adapter_issue_ids=adapter_issue_ids,
    )
    return ValidatedSpectrumSample(
        data=data,
        product_present=product_present,
        navgf=metadata.navgf,
        frequency_contract=contract,
        unique_packet_id=metadata.unique_packet_id,
        raw_seconds=metadata.raw_seconds,
        metadata=metadata,
        provenance=provenance,
    )


def _adapt_tr_spectrum(
    *,
    meta: object,
    metadata: SpectrumMetadata,
    candidates: dict[int, list[object]],
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> ValidatedTRSpectrumSample | None:
    navg2 = 1 << metadata.navg2_shift
    span = metadata.tr_stop - metadata.tr_start
    average = 1 << metadata.tr_average_shift
    if span <= 0 or span % average:
        issue_id = _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_tr_geometry",
            message=(
                f"TR spectrum UID {metadata.unique_packet_id} has invalid "
                "metadata geometry"
            ),
            packet=meta,
            uid=metadata.unique_packet_id,
            details={
                "tr_average_shift": metadata.tr_average_shift,
                "tr_start": metadata.tr_start,
                "tr_stop": metadata.tr_stop,
            },
        )
        _ = issue_id
        return None
    tr_length = span // average
    data = np.zeros(
        (NPRODUCTS, navg2, tr_length), dtype=np.int32
    )
    product_present = np.zeros(NPRODUCTS, dtype=np.bool_)
    accepted_packets = []
    all_candidates = []
    adapter_issue_ids = []
    for product in range(NPRODUCTS):
        packets = candidates.get(product, [])
        all_candidates.extend(packets)
        valid_packets = []
        packet_data_arrays = []
        for packet in packets:
            if _packet_has_fatal_issue(packet):
                continue
            try:
                _validate_science_packet_identity(packet, family="tr")
                if _packet_binding_key(packet) != selected_binding.binding_key:
                    raise ValueError("packet binding disagrees with Collection")
                if _required_public_attribute(packet, "meta") is not meta:
                    raise ValueError("packet metadata association disagrees")
                packet_data = _required_public_attribute(packet, "data")
                if type(packet_data) is not np.ndarray:
                    raise TypeError("TR packet data must be a numpy.ndarray")
                if packet_data.dtype != np.dtype(np.int32):
                    raise TypeError("TR packet data must retain native int32 dtype")
                if packet_data.shape != (navg2, tr_length):
                    raise ValueError(
                        "TR packet data shape disagrees with metadata geometry"
                    )
            except (TypeError, ValueError) as exc:
                adapter_issue_ids.append(_record_adapter_issue(
                    issue_collector,
                    code="decode_adapter.invalid_tr_product",
                    message=(
                        f"TR spectrum UID {metadata.unique_packet_id} product "
                        f"{product} was dropped: {exc}"
                    ),
                    packet=packet,
                    uid=metadata.unique_packet_id,
                    details={"error": str(exc), "product": product},
                ))
                continue
            valid_packets.append(packet)
            packet_data_arrays.append(packet_data)
        if len(valid_packets) > 1:
            indices = [
                _required_public_attribute(packet, "packet_index")
                for packet in valid_packets
            ]
            adapter_issue_ids.append(_record_adapter_issue(
                issue_collector,
                code="decode_adapter.duplicate_tr_product",
                message=(
                    f"TR spectrum UID {metadata.unique_packet_id} has "
                    f"{len(valid_packets)} valid packets for product {product}"
                ),
                packet=valid_packets[0],
                uid=metadata.unique_packet_id,
                details={"packet_indices": indices, "product": product},
            ))
            continue
        if not valid_packets:
            continue
        packet = valid_packets[0]
        packet_data = packet_data_arrays[0]
        data[product] = packet_data
        product_present[product] = True
        accepted_packets.append(packet)
    if not accepted_packets:
        return None
    provenance = _product_provenance(
        meta=meta,
        accepted_packets=accepted_packets,
        all_candidates=all_candidates,
        family="tr",
        metadata=metadata,
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
        adapter_issue_ids=adapter_issue_ids,
    )
    return ValidatedTRSpectrumSample(
        data=data,
        product_present=product_present,
        unique_packet_id=metadata.unique_packet_id,
        raw_seconds=metadata.raw_seconds,
        navg2=navg2,
        tr_length=tr_length,
        metadata=metadata,
        provenance=provenance,
    )


def _adapt_zoom_packet(
    packet: object,
    *,
    metadata_lookup: Mapping[
        int, tuple[tuple[object, SpectrumMetadata], ...]
    ],
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> ValidatedZoomSample | None:
    try:
        if _packet_binding_key(packet) != selected_binding.binding_key:
            raise ValueError("zoom packet binding disagrees with Collection")
        unique_packet_id = _required_uint(
            _required_public_attribute(packet, "unique_packet_id"),
            name="zoom unique_packet_id",
            bits=32,
        )
        pfb_bin = _required_uint(
            _required_public_attribute(packet, "pfb_bin"),
            name="zoom pfb_bin",
            bits=16,
        )
        components = []
        for name in ("AA", "BB", "ABR", "ABI"):
            components.append(_exact_ndarray(
                _required_public_attribute(packet, name),
                name=f"zoom {name}",
                dtype=np.float32,
                shape=(ZOOM_BINS,),
                finite=True,
            ))
        data = np.stack(components)
    except (TypeError, ValueError) as exc:
        uid_value = getattr(packet, "unique_packet_id", None)
        uid = (
            int(uid_value)
            if isinstance(uid_value, (int, np.integer))
            and not isinstance(uid_value, (bool, np.bool_))
            and 0 <= int(uid_value) <= 0xFFFFFFFF
            else None
        )
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_zoom",
            message=f"zoom packet was dropped: {exc}",
            packet=packet,
            uid=uid,
            details={"error": str(exc)},
        )
        return None

    metadata_packet, metadata, adapter_issue_ids = _associate_metadata_by_uid(
        packet=packet,
        unique_packet_id=unique_packet_id,
        family="zoom",
        lookup=metadata_lookup,
        issue_collector=issue_collector,
    )
    source_packets = [packet]
    roles = ["zoom"]
    if metadata_packet is not None:
        source_packets.append(metadata_packet)
        roles.append("time_metadata")
    raw_seconds = None if metadata is None else metadata.raw_seconds
    provenance = _auxiliary_provenance(
        packets=source_packets,
        roles=roles,
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Cal_ZoomSpectra.unique_packet_id",
        uid_source_role="zoom",
        raw_seconds=raw_seconds,
        time_source=(
            "Packet_Metadata.base.time_32_time_16"
            if raw_seconds is not None
            else None
        ),
        time_source_role="time_metadata" if raw_seconds is not None else None,
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
        adapter_issue_ids=adapter_issue_ids,
    )
    return ValidatedZoomSample(
        data=data,
        unique_packet_id=unique_packet_id,
        pfb_bin=pfb_bin,
        raw_seconds=raw_seconds,
        provenance=provenance,
    )


def _adapt_waveform_group(
    group_value: object,
    *,
    decoder: object,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> list[ValidatedWaveformSample]:
    group = _required_group_mapping(
        group_value,
        name="Collection.waveform_groups",
        keys=frozenset({"packets", "meta", "schema_binding"}),
    )
    if group["schema_binding"] != selected_binding.binding_key:
        raise IncompatibleUncraterError(
            "uncrater waveform group binding disagrees with Collection"
        )
    meta = group["meta"]
    if not isinstance(meta, decoder.Packet_Waveform_Meta):
        raise IncompatibleUncraterError(
            "uncrater waveform group has no Packet_Waveform_Meta"
        )
    if _packet_has_fatal_issue(meta):
        raise IncompatibleUncraterError(
            "uncrater waveform group published fatal metadata"
        )
    try:
        if _packet_binding_key(meta) != selected_binding.binding_key:
            raise ValueError("waveform metadata binding disagrees with Collection")
        unique_packet_id = _required_uint(
            _required_public_attribute(meta, "unique_packet_id"),
            name="waveform unique_packet_id",
            bits=32,
        )
        time_32 = _required_uint(
            _required_public_attribute(meta, "time_32"),
            name="waveform time_32",
            bits=32,
        )
        time_16 = _required_uint(
            _required_public_attribute(meta, "time_16"),
            name="waveform time_16",
            bits=16,
        )
        raw_seconds = raw_seconds_from_split_time(time_32, time_16)
        decoded_time = _required_finite_float(
            _required_public_attribute(meta, "time"),
            name="waveform metadata time",
        )
        if decoded_time != raw_seconds:
            raise ValueError("waveform metadata split and decoded times disagree")
        adc_timestamp = np.uint64(_required_uint(
            _required_public_attribute(meta, "timestamp"),
            name="waveform ADC timestamp",
            bits=64,
        ))
    except (TypeError, ValueError) as exc:
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_waveform_metadata",
            message=f"waveform group was dropped: {exc}",
            packet=meta,
            uid=None,
            details={"error": str(exc)},
        )
        return []

    packets = group["packets"]
    if type(packets) is not dict or not 1 <= len(packets) <= 4:
        raise IncompatibleUncraterError(
            "uncrater waveform group packets must be a nonempty channel dictionary"
        )
    if list(packets) != sorted(packets):
        raise IncompatibleUncraterError(
            "uncrater waveform group channels must use sorted insertion order"
        )
    samples = []
    for channel, packet in packets.items():
        if type(channel) is not int or not 0 <= channel < 4:
            raise IncompatibleUncraterError(
                "uncrater waveform group has an invalid channel key"
            )
        if not isinstance(packet, decoder.Packet_Waveform):
            raise IncompatibleUncraterError(
                "uncrater waveform group has an invalid packet class"
            )
        if _packet_has_fatal_issue(packet):
            raise IncompatibleUncraterError(
                "uncrater waveform group published a fatal waveform packet"
            )
        try:
            if _packet_binding_key(packet) != selected_binding.binding_key:
                raise ValueError("waveform packet binding disagrees with Collection")
            packet_channel = _required_uint(
                _required_public_attribute(packet, "ch"),
                name="waveform channel",
                bits=2,
            )
            if packet_channel != channel:
                raise ValueError("waveform packet channel disagrees with group key")
            if _required_public_attribute(packet, "meta") is not meta:
                raise ValueError("waveform packet metadata association disagrees")
            packet_timestamp = _required_uint(
                _required_public_attribute(packet, "timestamp"),
                name="waveform packet ADC timestamp",
                bits=64,
            )
            if packet_timestamp != int(adc_timestamp):
                raise ValueError("waveform packet and metadata timestamps disagree")
            data = _exact_ndarray(
                _required_public_attribute(packet, "waveform"),
                name="waveform data",
                dtype=np.int16,
                shape=(WAVEFORM_SAMPLES,),
            )
        except (TypeError, ValueError) as exc:
            _record_adapter_issue(
                issue_collector,
                code="decode_adapter.invalid_waveform",
                message=(
                    f"waveform UID {unique_packet_id} channel {channel} "
                    f"was dropped: {exc}"
                ),
                packet=packet,
                uid=unique_packet_id,
                details={"channel": channel, "error": str(exc)},
            )
            continue
        waveform_role = f"waveform_channel_{channel}"
        provenance = _auxiliary_provenance(
            packets=(packet, meta),
            roles=(waveform_role, "waveform_metadata"),
            unique_packet_id=unique_packet_id,
            uid_source="Packet_Waveform_Meta.unique_packet_id",
            uid_source_role="waveform_metadata",
            raw_seconds=raw_seconds,
            time_source="Packet_Waveform_Meta.time_32_time_16",
            time_source_role="waveform_metadata",
            selected_binding=selected_binding,
            issue_ids_by_packet_index=issue_ids_by_packet_index,
        )
        samples.append(ValidatedWaveformSample(
            data=data,
            channel=channel,
            unique_packet_id=unique_packet_id,
            raw_seconds=raw_seconds,
            adc_timestamp=adc_timestamp,
            provenance=provenance,
        ))
    return samples


def _adapt_grimm_packet(
    packet: object,
    *,
    metadata_lookup: Mapping[
        int, tuple[tuple[object, SpectrumMetadata], ...]
    ],
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> ValidatedGrimmSample | None:
    try:
        if _packet_binding_key(packet) != selected_binding.binding_key:
            raise ValueError("Grimm packet binding disagrees with Collection")
        unique_packet_id = _required_uint(
            _required_public_attribute(packet, "unique_packet_id"),
            name="Grimm unique_packet_id",
            bits=32,
        )
        data = _required_public_attribute(packet, "data")
        if type(data) is not np.ndarray:
            raise TypeError("Grimm data must be a numpy.ndarray")
        if data.dtype != np.dtype(np.int32):
            raise TypeError("Grimm data must retain native int32 dtype")
        if data.ndim != 3 or data.shape[0] == 0 or data.shape[1:] != (16, 4):
            raise ValueError(
                "Grimm data must have native shape (Navg2, 16, 4)"
            )
        navg2 = data.shape[0]
    except (TypeError, ValueError) as exc:
        uid_value = getattr(packet, "unique_packet_id", None)
        uid = (
            int(uid_value)
            if isinstance(uid_value, (int, np.integer))
            and not isinstance(uid_value, (bool, np.bool_))
            and 0 <= int(uid_value) <= 0xFFFFFFFF
            else None
        )
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_grimm",
            message=f"Grimm packet was dropped: {exc}",
            packet=packet,
            uid=uid,
            details={"error": str(exc)},
        )
        return None

    metadata_packet, metadata, adapter_issue_ids = _associate_metadata_by_uid(
        packet=packet,
        unique_packet_id=unique_packet_id,
        family="grimm",
        lookup=metadata_lookup,
        issue_collector=issue_collector,
    )
    source_packets = [packet]
    roles = ["grimm"]
    if metadata_packet is not None:
        source_packets.append(metadata_packet)
        roles.append("time_metadata")
    raw_seconds = None if metadata is None else metadata.raw_seconds
    provenance = _auxiliary_provenance(
        packets=source_packets,
        roles=roles,
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Grimm.unique_packet_id",
        uid_source_role="grimm",
        raw_seconds=raw_seconds,
        time_source=(
            "Packet_Metadata.base.time_32_time_16"
            if raw_seconds is not None
            else None
        ),
        time_source_role="time_metadata" if raw_seconds is not None else None,
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
        adapter_issue_ids=adapter_issue_ids,
    )
    return ValidatedGrimmSample(
        data=data,
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
        navg2=navg2,
        provenance=provenance,
    )


_HK_FIELD_NAMES = (
    "time_32",
    "time_16",
    "adc_min",
    "adc_max",
    "adc_valid_count",
    "adc_invalid_count_max",
    "adc_invalid_count_min",
    "adc_total_count",
    "adc_mean",
    "adc_rms",
    "adc_statistics_valid",
    "actual_gain",
    "telemetry_v1_0",
    "telemetry_v1_8",
    "telemetry_v2_5",
    "telemetry_t_fpga",
    "ok",
    "checksum",
    "weight_ndx",
    "meta_valid",
    "size",
    "checksum_meta",
    "checksum_data",
    "region_1",
    "region_2",
    "size_1",
    "size_2",
    "checksum_1_meta",
    "checksum_1_data",
    "checksum_2_meta",
    "checksum_2_data",
    "status",
)

_HK_TYPES_BY_BINDING = {
    "203": frozenset({0, 1}),
    "305": frozenset({0, 1, 2, 3}),
    "306-early": frozenset({0, 1, 2, 3}),
    "306-final": frozenset({0, 1, 2, 3, 100, 101}),
    "307": frozenset({0, 1, 2, 3, 100, 101}),
}


def _new_field_union(names: Sequence[str]) -> tuple[dict[str, object | None], dict[str, bool]]:
    return ({name: None for name in names}, {name: False for name in names})


def _put_field(
    fields: dict[str, object | None],
    field_present: dict[str, bool],
    name: str,
    value: object,
) -> None:
    fields[name] = value
    field_present[name] = True


def _housekeeping_raw_time(
    packet: object,
    *,
    hk_type: int,
) -> tuple[float | None, int | None, int | None]:
    if hk_type == 0:
        core_state = _required_public_attribute(packet, "core_state")
        source = _required_public_attribute(core_state, "base")
    elif hk_type == 2:
        source = _required_public_attribute(packet, "heartbeat")
    else:
        return None, None, None
    time_32 = _required_uint(
        _required_public_attribute(source, "time_32"),
        name=f"housekeeping type {hk_type} time_32",
        bits=32,
    )
    time_16 = _required_uint(
        _required_public_attribute(source, "time_16"),
        name=f"housekeeping type {hk_type} time_16",
        bits=16,
    )
    raw_seconds = raw_seconds_from_split_time(time_32, time_16)
    public_time = _required_finite_float(
        _required_public_attribute(packet, "time"),
        name=f"housekeeping type {hk_type} time",
    )
    if public_time != raw_seconds:
        raise ValueError("housekeeping split and decoded times disagree")
    return raw_seconds, time_32, time_16


def _extract_housekeeping_adc_fields(
    packet: object,
    fields: dict[str, object | None],
    field_present: dict[str, bool],
) -> None:
    specifications = (
        ("min", "adc_min", np.int64),
        ("max", "adc_max", np.int64),
        ("valid_count", "adc_valid_count", np.int64),
        ("invalid_count_max", "adc_invalid_count_max", np.int64),
        ("invalid_count_min", "adc_invalid_count_min", np.int64),
        ("total_count", "adc_total_count", np.int64),
        ("mean", "adc_mean", np.float64),
        ("rms", "adc_rms", np.float64),
    )
    arrays: dict[str, np.ndarray] = {}
    for source_name, public_name, dtype in specifications:
        value = _exact_ndarray(
            _required_public_attribute(packet, source_name),
            name=f"housekeeping {source_name}",
            dtype=dtype,
            shape=(4,),
            finite=np.issubdtype(np.dtype(dtype), np.floating),
        )
        alias = _exact_ndarray(
            _required_public_attribute(packet, public_name),
            name=f"housekeeping {public_name}",
            dtype=dtype,
            shape=(4,),
            finite=np.issubdtype(np.dtype(dtype), np.floating),
        )
        if not np.array_equal(value, alias):
            raise ValueError(
                f"housekeeping {source_name} and {public_name} disagree"
            )
        arrays[public_name] = value
        _put_field(fields, field_present, public_name, value)
    valid_count = arrays["adc_valid_count"]
    if np.any(valid_count < 0):
        raise ValueError("housekeeping ADC valid counts must be nonnegative")
    for name in (
        "adc_invalid_count_max",
        "adc_invalid_count_min",
        "adc_total_count",
    ):
        if np.any(arrays[name] < 0):
            raise ValueError(f"housekeeping {name} must be nonnegative")
    expected_total = (
        valid_count
        + arrays["adc_invalid_count_max"]
        + arrays["adc_invalid_count_min"]
    )
    if not np.array_equal(arrays["adc_total_count"], expected_total):
        raise ValueError("housekeeping ADC total counts are inconsistent")
    if np.any(arrays["adc_rms"] < 0):
        raise ValueError("housekeeping ADC RMS must be nonnegative")
    _put_field(
        fields,
        field_present,
        "adc_statistics_valid",
        np.asarray(valid_count > 0, dtype=np.bool_),
    )


def _extract_housekeeping_telemetry(
    packet: object,
    fields: dict[str, object | None],
    field_present: dict[str, bool],
) -> None:
    telemetry = _required_public_attribute(packet, "telemetry")
    if not isinstance(telemetry, Mapping) or set(telemetry) != {
        "V1_0",
        "V1_8",
        "V2_5",
        "T_FPGA",
    }:
        raise TypeError("housekeeping telemetry must use the reviewed mapping")
    mapping = (
        ("V1_0", "telemetry_V1_0", "telemetry_v1_0"),
        ("V1_8", "telemetry_V1_8", "telemetry_v1_8"),
        ("V2_5", "telemetry_V2_5", "telemetry_v2_5"),
        ("T_FPGA", "telemetry_T_FPGA", "telemetry_t_fpga"),
    )
    for source_name, attribute_name, public_name in mapping:
        value = _required_finite_float(
            telemetry[source_name],
            name=f"housekeeping telemetry {source_name}",
        )
        direct = _required_finite_float(
            _required_public_attribute(packet, attribute_name),
            name=f"housekeeping {attribute_name}",
        )
        if value != direct:
            raise ValueError(
                f"housekeeping telemetry {source_name} aliases disagree"
            )
        _put_field(fields, field_present, public_name, value)


def _adapt_housekeeping_packet(
    packet: object,
    *,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> ValidatedHKSample | None:
    try:
        if _packet_binding_key(packet) != selected_binding.binding_key:
            raise ValueError("housekeeping packet binding disagrees with Collection")
        hk_type = _required_uint(
            _required_public_attribute(packet, "hk_type"),
            name="housekeeping type",
            bits=16,
        )
        supported = _HK_TYPES_BY_BINDING.get(selected_binding.binding_key)
        if supported is None or hk_type not in supported:
            raise ValueError(
                f"housekeeping type {hk_type} is unsupported by binding "
                f"{selected_binding.binding_key}"
            )
        version = _required_uint(
            _required_public_attribute(packet, "version"),
            name="housekeeping version",
            bits=16,
        )
        unique_packet_id = _required_uint(
            _required_public_attribute(packet, "unique_packet_id"),
            name="housekeeping unique_packet_id",
            bits=32,
        )
        errors = _required_uint(
            _required_public_attribute(packet, "errors"),
            name="housekeeping firmware errors",
            bits=32,
        )
        raw_seconds, time_32, time_16 = _housekeeping_raw_time(
            packet,
            hk_type=hk_type,
        )
        fields, field_present = _new_field_union(_HK_FIELD_NAMES)
        if time_32 is not None and time_16 is not None:
            _put_field(fields, field_present, "time_32", time_32)
            _put_field(fields, field_present, "time_16", time_16)

        if hk_type in (0, 1):
            _extract_housekeeping_adc_fields(packet, fields, field_present)
        if hk_type == 1:
            gains = _required_public_attribute(packet, "actual_gain")
            if not isinstance(gains, list) or len(gains) != 4 or any(
                type(value) is not str or value not in {"L", "M", "H"}
                for value in gains
            ):
                raise ValueError(
                    "housekeeping actual_gain must contain four L/M/H codes"
                )
            _put_field(fields, field_present, "actual_gain", tuple(gains))
        if hk_type in (0, 2):
            _extract_housekeeping_telemetry(packet, fields, field_present)
        if hk_type == 2:
            _put_field(
                fields,
                field_present,
                "ok",
                _required_boolean(
                    _required_public_attribute(packet, "ok"),
                    name="housekeeping heartbeat ok",
                ),
            )
        if hk_type == 3:
            checksum_name = (
                "crc"
                if selected_binding.binding_key in {"305", "306-early"}
                else "checksum"
            )
            _put_field(
                fields,
                field_present,
                "checksum",
                _required_uint(
                    _required_public_attribute(packet, checksum_name),
                    name="housekeeping checksum",
                    bits=32,
                ),
            )
            _put_field(
                fields,
                field_present,
                "weight_ndx",
                _required_uint(
                    _required_public_attribute(packet, "weight_ndx"),
                    name="housekeeping weight_ndx",
                    bits=16,
                ),
            )
        if hk_type == 100:
            _put_field(
                fields,
                field_present,
                "meta_valid",
                _exact_uint_vector(
                    _required_public_attribute(packet, "meta_valid"),
                    name="housekeeping meta_valid",
                    length=6,
                    bits=8,
                    dtype=np.uint8,
                ),
            )
            for name in ("size", "checksum_meta", "checksum_data"):
                _put_field(
                    fields,
                    field_present,
                    name,
                    _exact_uint_vector(
                        _required_public_attribute(packet, name),
                        name=f"housekeeping {name}",
                        length=6,
                        bits=32,
                        dtype=np.uint32,
                    ),
                )
        if hk_type == 101:
            report = _required_public_attribute(packet, "report")
            for name in ("region_1", "region_2", "status"):
                _put_field(
                    fields,
                    field_present,
                    name,
                    _required_integer(
                        _required_public_attribute(report, name),
                        name=f"housekeeping report {name}",
                        minimum=-(1 << 31),
                        maximum=(1 << 31) - 1,
                    ),
                )
            for name in (
                "size_1",
                "size_2",
                "checksum_1_meta",
                "checksum_1_data",
                "checksum_2_meta",
                "checksum_2_data",
            ):
                _put_field(
                    fields,
                    field_present,
                    name,
                    _required_uint(
                        _required_public_attribute(report, name),
                        name=f"housekeeping report {name}",
                        bits=32,
                    ),
                )
    except (TypeError, ValueError) as exc:
        uid_value = getattr(packet, "unique_packet_id", None)
        uid = (
            int(uid_value)
            if isinstance(uid_value, (int, np.integer))
            and not isinstance(uid_value, (bool, np.bool_))
            and 0 <= int(uid_value) <= 0xFFFFFFFF
            else None
        )
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_housekeeping",
            message=f"housekeeping packet was dropped: {exc}",
            packet=packet,
            uid=uid,
            details={"error": str(exc)},
        )
        return None

    provenance = _auxiliary_provenance(
        packets=(packet,),
        roles=("housekeeping",),
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Housekeep.unique_packet_id",
        uid_source_role="housekeeping",
        raw_seconds=raw_seconds,
        time_source="Packet_Housekeep.time",
        time_source_role="housekeeping",
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
    )
    return ValidatedHKSample(
        hk_type=hk_type,
        version=version,
        unique_packet_id=unique_packet_id,
        errors=errors,
        raw_seconds=raw_seconds,
        fields=fields,
        field_present=field_present,
        provenance=provenance,
    )


_CAL_METADATA_FIELD_NAMES = (
    "version",
    "time_32",
    "time_16",
    "have_lock",
    "snr_on",
    "snr_off",
    "mode",
    "powertop_slice",
    "sum1_slice",
    "sum2_slice",
    "fd_slice",
    "sd2_slice",
    "prod1_slice",
    "prod2_slice",
    "errors",
    "bitslicer_errors",
    "drift_shift",
    "drift_wire",
    "drift_raw",
    "drift",
    "error_regs",
    "stats_snr_max",
    "stats_snr_min",
    "stats_ptop_max",
    "stats_ptop_min",
    "stats_pbot_max",
    "stats_pbot_min",
    "stats_fd_max",
    "stats_fd_min",
    "stats_sd_max",
    "stats_sd_min",
    "stats_sd_positive_count",
    "stats_lock_count",
    "error_cal_phaser",
    "error_averager",
    "error_process",
    "error_stage3",
    "error_check",
    "snr_max",
    "snr_min",
    "state_mode",
    "state_readout_mode",
    "state_navg2",
    "state_navg3",
    "state_drift_guard",
    "state_drift_step",
    "state_antenna_mask",
    "state_notch_index",
    "state_snr_on",
    "state_snr_off",
    "state_nsettle",
    "state_delta_drift_cor_a",
    "state_delta_drift_cor_b",
    "state_pfb_index",
    "state_weight_ndx",
    "state_powertop_slice",
    "state_sum1_slice",
    "state_sum2_slice",
    "state_prod1_slice",
    "state_prod2_slice",
)


def _extract_calibrator_stats(
    packet: object,
    fields: dict[str, object | None],
    field_present: dict[str, bool],
) -> None:
    stats = _required_public_attribute(packet, "stats")
    unsigned_arrays = (
        ("SNR_max", "stats_snr_max"),
        ("SNR_min", "stats_snr_min"),
        ("ptop_max", "stats_ptop_max"),
        ("ptop_min", "stats_ptop_min"),
        ("pbot_max", "stats_pbot_max"),
        ("pbot_min", "stats_pbot_min"),
    )
    for source_name, public_name in unsigned_arrays:
        _put_field(
            fields,
            field_present,
            public_name,
            _exact_uint_vector(
                _required_public_attribute(stats, source_name),
                name=f"calibrator metadata stats {source_name}",
                length=4,
                bits=32,
                dtype=np.uint32,
            ),
        )
    for source_name, public_name in (
        ("FD_max", "stats_fd_max"),
        ("FD_min", "stats_fd_min"),
        ("SD_max", "stats_sd_max"),
        ("SD_min", "stats_sd_min"),
    ):
        _put_field(
            fields,
            field_present,
            public_name,
            _exact_integer_vector(
                _required_public_attribute(stats, source_name),
                name=f"calibrator metadata stats {source_name}",
                length=4,
                dtype=np.int32,
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            ),
        )
    positive_count = _exact_integer_vector(
        _required_public_attribute(stats, "SD_positive_count"),
        name="calibrator metadata stats SD_positive_count",
        length=4,
        dtype=np.int64,
        minimum=-(1 << 15),
        maximum=(1 << 16) - 1,
    )
    _put_field(
        fields,
        field_present,
        "stats_sd_positive_count",
        positive_count,
    )
    _put_field(
        fields,
        field_present,
        "stats_lock_count",
        _required_integer(
            _required_public_attribute(stats, "lock_count"),
            name="calibrator metadata stats lock_count",
            minimum=-(1 << 31),
            maximum=(1 << 31) - 1,
        ),
    )


def _extract_calibrator_errors(
    packet: object,
    *,
    binding_key: str,
    fields: dict[str, object | None],
    field_present: dict[str, bool],
) -> None:
    if binding_key in {"305", "306-early"}:
        _put_field(
            fields,
            field_present,
            "error_regs",
            _exact_uint_vector(
                _required_public_attribute(packet, "error_regs"),
                name="calibrator metadata error_regs",
                length=30,
                bits=32,
                dtype=np.uint32,
            ),
        )
        return
    error = _required_public_attribute(packet, "error_reg")
    for source_name, public_name, length in (
        ("cal_phaser_err", "error_cal_phaser", 2),
        ("averager_err", "error_averager", 16),
        ("process_err", "error_process", 8),
        ("stage3_err", "error_stage3", 4),
    ):
        _put_field(
            fields,
            field_present,
            public_name,
            _exact_uint_vector(
                _required_public_attribute(error, source_name),
                name=f"calibrator metadata error {source_name}",
                length=length,
                bits=32,
                dtype=np.uint32,
            ),
        )
    _put_field(
        fields,
        field_present,
        "error_check",
        _required_uint(
            _required_public_attribute(error, "check"),
            name="calibrator metadata error check",
            bits=32,
        ),
    )


def _extract_calibrator_203_state(
    packet: object,
    fields: dict[str, object | None],
    field_present: dict[str, bool],
) -> None:
    state = _required_public_attribute(packet, "state")
    specifications = (
        ("mode", "state_mode", 8),
        ("readout_mode", "state_readout_mode", 8),
        ("Navg2", "state_navg2", 8),
        ("Navg3", "state_navg3", 8),
        ("drift_guard", "state_drift_guard", 8),
        ("drift_step", "state_drift_step", 8),
        ("antenna_mask", "state_antenna_mask", 8),
        ("notch_index", "state_notch_index", 8),
        ("SNRon", "state_snr_on", 32),
        ("SNRoff", "state_snr_off", 32),
        ("Nsettle", "state_nsettle", 32),
        ("delta_drift_corA", "state_delta_drift_cor_a", 32),
        ("delta_drift_corB", "state_delta_drift_cor_b", 32),
        ("pfb_index", "state_pfb_index", 16),
        ("weight_ndx", "state_weight_ndx", 16),
        ("powertop_slice", "state_powertop_slice", 8),
        ("sum1_slice", "state_sum1_slice", 8),
        ("sum2_slice", "state_sum2_slice", 8),
        ("prod1_slice", "state_prod1_slice", 8),
        ("prod2_slice", "state_prod2_slice", 8),
    )
    for source_name, public_name, bits in specifications:
        _put_field(
            fields,
            field_present,
            public_name,
            _required_uint(
                _required_public_attribute(state, source_name),
                name=f"calibrator metadata state {source_name}",
                bits=bits,
            ),
        )


def _extract_calibrator_metadata_fields(
    packet: object,
    *,
    binding_key: str,
    include_derived_drift: bool,
) -> tuple[dict[str, object | None], dict[str, bool]]:
    fields, field_present = _new_field_union(_CAL_METADATA_FIELD_NAMES)
    _put_field(
        fields,
        field_present,
        "version",
        _required_uint(
            _required_public_attribute(packet, "version"),
            name="calibrator metadata version",
            bits=16,
        ),
    )
    for name, bits in (("time_32", 32), ("time_16", 16)):
        _put_field(
            fields,
            field_present,
            name,
            _required_uint(
                _required_public_attribute(packet, name),
                name=f"calibrator metadata {name}",
                bits=bits,
            ),
        )
    _put_field(
        fields,
        field_present,
        "have_lock",
        _exact_uint_vector(
            _required_public_attribute(packet, "have_lock"),
            name="calibrator metadata have_lock",
            length=4,
            bits=16,
            dtype=np.uint16,
        ),
    )
    if binding_key == "203":
        _put_field(
            fields,
            field_present,
            "snr_max",
            _required_integer(
                _required_public_attribute(packet, "SNR_max"),
                name="calibrator metadata SNR_max",
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            ),
        )
        _put_field(
            fields,
            field_present,
            "snr_min",
            _required_integer(
                _required_public_attribute(packet, "SNR_min"),
                name="calibrator metadata SNR_min",
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            ),
        )
        _put_field(
            fields,
            field_present,
            "error_regs",
            _exact_uint_vector(
                _required_public_attribute(packet, "error_regs"),
                name="calibrator metadata error_regs",
                length=30,
                bits=32,
                dtype=np.uint32,
            ),
        )
        _extract_calibrator_203_state(packet, fields, field_present)
    else:
        scalar_fields = (
            ("SNRon", "snr_on", 32),
            ("SNRoff", "snr_off", 32),
            ("powertop_slice", "powertop_slice", 8),
            ("sum1_slice", "sum1_slice", 8),
            ("sum2_slice", "sum2_slice", 8),
            ("fd_slice", "fd_slice", 8),
            ("sd2_slice", "sd2_slice", 8),
            ("prod1_slice", "prod1_slice", 8),
            ("prod2_slice", "prod2_slice", 8),
            ("errors", "errors", 32),
            ("bitslicer_errors", "bitslicer_errors", 32),
            ("drift_shift", "drift_shift", 8),
        )
        for source_name, public_name, bits in scalar_fields:
            _put_field(
                fields,
                field_present,
                public_name,
                _required_uint(
                    _required_public_attribute(packet, source_name),
                    name=f"calibrator metadata {source_name}",
                    bits=bits,
                ),
            )
        if fields["drift_shift"] > 16:
            raise ValueError("calibrator metadata drift_shift exceeds 16")
        if binding_key in {"306-final", "307"}:
            _put_field(
                fields,
                field_present,
                "mode",
                _required_uint(
                    _required_public_attribute(packet, "mode"),
                    name="calibrator metadata mode",
                    bits=8,
                ),
            )
        _extract_calibrator_stats(packet, fields, field_present)
        _extract_calibrator_errors(
            packet,
            binding_key=binding_key,
            fields=fields,
            field_present=field_present,
        )
    if include_derived_drift:
        _put_field(
            fields,
            field_present,
            "drift_raw",
            _exact_ndarray(
                _required_public_attribute(packet, "drift_raw"),
                name="calibrator metadata drift_raw",
                dtype=np.int64,
                shape=(1024,),
            ),
        )
        _put_field(
            fields,
            field_present,
            "drift",
            _exact_ndarray(
                _required_public_attribute(packet, "drift"),
                name="calibrator metadata drift",
                dtype=np.float64,
                shape=(1024,),
                finite=True,
            ),
        )
    else:
        if binding_key == "203":
            drift_wire = _exact_integer_vector(
                _required_public_attribute(packet, "drift"),
                name="embedded calibrator metadata drift",
                length=1024,
                dtype=np.int32,
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            )
        else:
            drift_wire = _exact_integer_vector(
                _required_public_attribute(packet, "drift"),
                name="embedded calibrator metadata drift",
                length=128,
                dtype=np.int16,
                minimum=-(1 << 15),
                maximum=(1 << 15) - 1,
            )
        _put_field(
            fields,
            field_present,
            "drift_wire",
            drift_wire,
        )
    return fields, field_present


def _adapt_calibrator_metadata_packet(
    packet: object,
    *,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> CalibratorMetadataSample | None:
    try:
        if _packet_binding_key(packet) != selected_binding.binding_key:
            raise ValueError(
                "calibrator metadata binding disagrees with Collection"
            )
        unique_packet_id = _required_uint(
            _required_public_attribute(packet, "unique_packet_id"),
            name="calibrator metadata unique_packet_id",
            bits=32,
        )
        if _required_public_attribute(packet, "from_debug") is not False:
            raise ValueError("direct calibrator metadata is marked from_debug")
        fields, field_present = _extract_calibrator_metadata_fields(
            packet,
            binding_key=selected_binding.binding_key,
            include_derived_drift=True,
        )
        raw_seconds = raw_seconds_from_split_time(
            fields["time_32"],
            fields["time_16"],
        )
        decoded_time = _required_finite_float(
            _required_public_attribute(packet, "time"),
            name="calibrator metadata time",
        )
        if decoded_time != raw_seconds:
            raise ValueError("calibrator metadata split and decoded times disagree")
    except (TypeError, ValueError) as exc:
        uid_value = getattr(packet, "unique_packet_id", None)
        uid = (
            int(uid_value)
            if isinstance(uid_value, (int, np.integer))
            and not isinstance(uid_value, (bool, np.bool_))
            and 0 <= int(uid_value) <= 0xFFFFFFFF
            else None
        )
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_calibrator_metadata",
            message=f"calibrator metadata was dropped: {exc}",
            packet=packet,
            uid=uid,
            details={"error": str(exc)},
        )
        return None
    provenance = _auxiliary_provenance(
        packets=(packet,),
        roles=("calibrator_metadata",),
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Cal_Metadata.unique_packet_id",
        uid_source_role="calibrator_metadata",
        raw_seconds=raw_seconds,
        time_source="Packet_Cal_Metadata.time_32_time_16",
        time_source_role="calibrator_metadata",
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
    )
    return CalibratorMetadataSample(
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
        from_debug=False,
        fields=fields,
        field_present=field_present,
        provenance=provenance,
    )


def _calibrator_group_header(
    group_value: object,
    *,
    name: str,
    keys: frozenset[str],
    selected_binding: UncraterBindingInfo,
) -> tuple[dict[str, object], int]:
    group = _required_group_mapping(
        group_value,
        name=name,
        keys=keys,
    )
    if group["schema_binding"] != selected_binding.binding_key:
        raise IncompatibleUncraterError(
            f"uncrater {name} binding disagrees with Collection"
        )
    unique_packet_id = _required_uint(
        group["unique_packet_id"],
        name=f"{name} unique_packet_id",
        bits=32,
    )
    return group, unique_packet_id


def _adapt_calibrator_data_group(
    group_value: object,
    *,
    decoder: object,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> CalibratorDataSample | None:
    group, unique_packet_id = _calibrator_group_header(
        group_value,
        name="Collection.calibrator_data_groups",
        keys=frozenset({
            "unique_packet_id",
            "pages",
            "data",
            "gNacc",
            "gphase",
            "schema_binding",
        }),
        selected_binding=selected_binding,
    )
    pages_value = group["pages"]
    if type(pages_value) is not tuple or not pages_value:
        raise IncompatibleUncraterError(
            "uncrater calibrator data group has no page tuple"
        )
    issue_packet = pages_value[0]
    try:
        pages, page_raw_seconds = _validated_group_pages(
            group,
            name="calibrator data",
            packet_class=decoder.Packet_Cal_Data,
            page_count=3,
            unique_packet_id=unique_packet_id,
            selected_binding=selected_binding,
        )
        page_arrays = []
        for page_index, packet in enumerate(pages):
            data_page = _required_uint(
                _required_public_attribute(packet, "data_page"),
                name=f"calibrator data page {page_index} index",
                bits=2,
            )
            if data_page != page_index:
                raise ValueError("calibrator data page index disagrees with order")
            if page_index < 2:
                page_arrays.append(_exact_ndarray(
                    _required_public_attribute(packet, "data"),
                    name=f"calibrator data page {page_index} payload",
                    dtype=np.int32,
                    shape=(4, 512),
                ))
        page_2 = pages[2]
        g_nacc = _required_integer(
            _required_public_attribute(page_2, "gNacc"),
            name="calibrator data gNacc",
            minimum=-(1 << 31),
            maximum=(1 << 31) - 1,
        )
        gphase = _exact_ndarray(
            _required_public_attribute(page_2, "gphase"),
            name="calibrator data gphase",
            dtype=np.int32,
            shape=(1024,),
        )
        group_g_nacc = _required_integer(
            group["gNacc"],
            name="calibrator data group gNacc",
            minimum=-(1 << 31),
            maximum=(1 << 31) - 1,
        )
        group_gphase = _exact_ndarray(
            group["gphase"],
            name="calibrator data group gphase",
            dtype=np.int32,
            shape=(1024,),
        )
        if group_g_nacc != g_nacc or not np.array_equal(group_gphase, gphase):
            raise ValueError("calibrator data group and page-2 fields disagree")
        data = _exact_ndarray(
            group["data"],
            name="calibrator data group payload",
            dtype=np.complex128,
            shape=(4, 512),
            finite=True,
        )
        expected = page_arrays[0].astype(np.complex128)
        expected += 1j * page_arrays[1]
        if not np.array_equal(data, expected):
            raise ValueError("calibrator data group payload disagrees with pages")
    except (TypeError, ValueError) as exc:
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_calibrator_data",
            message=f"calibrator data group was dropped: {exc}",
            packet=issue_packet,
            uid=unique_packet_id,
            details={"error": str(exc)},
        )
        return None
    raw_seconds = page_raw_seconds[0]
    roles = tuple(f"calibrator_data_page_{index}" for index in range(3))
    provenance = _auxiliary_provenance(
        packets=pages,
        roles=roles,
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Cal_Data.unique_packet_id",
        uid_source_role=roles[0],
        raw_seconds=raw_seconds,
        time_source="Packet_Cal_Data.time",
        time_source_role=roles[0],
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
    )
    return CalibratorDataSample(
        data=data,
        g_nacc=g_nacc,
        gphase=gphase,
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
        page_raw_seconds=np.asarray(page_raw_seconds, dtype=np.float64),
        provenance=provenance,
    )


def _adapt_calibrator_pfb_group(
    group_value: object,
    *,
    decoder: object,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> CalibratorRawPFBSample | None:
    group, unique_packet_id = _calibrator_group_header(
        group_value,
        name="Collection.calibrator_pfb_groups",
        keys=frozenset({
            "unique_packet_id",
            "pages",
            "data",
            "schema_binding",
        }),
        selected_binding=selected_binding,
    )
    pages_value = group["pages"]
    if type(pages_value) is not tuple or not pages_value:
        raise IncompatibleUncraterError(
            "uncrater calibrator raw-PFB group has no page tuple"
        )
    issue_packet = pages_value[0]
    try:
        pages, page_raw_seconds = _validated_group_pages(
            group,
            name="calibrator raw PFB",
            packet_class=decoder.Packet_Cal_RawPFB,
            page_count=8,
            unique_packet_id=unique_packet_id,
            selected_binding=selected_binding,
        )
        page_arrays = []
        for page_index, packet in enumerate(pages):
            channel = _required_uint(
                _required_public_attribute(packet, "channel"),
                name=f"calibrator raw-PFB page {page_index} channel",
                bits=2,
            )
            part = _required_uint(
                _required_public_attribute(packet, "part"),
                name=f"calibrator raw-PFB page {page_index} part",
                bits=1,
            )
            if channel != page_index // 2 or part != page_index % 2:
                raise ValueError(
                    "calibrator raw-PFB channel/part disagrees with page order"
                )
            page_arrays.append(_exact_ndarray(
                _required_public_attribute(packet, "data"),
                name=f"calibrator raw-PFB page {page_index} payload",
                dtype=np.int32,
                shape=(NCHANNELS,),
            ))
        data = _exact_ndarray(
            group["data"],
            name="calibrator raw-PFB group payload",
            dtype=np.complex128,
            shape=(4, NCHANNELS),
            finite=True,
        )
        expected = np.stack([
            page_arrays[2 * channel].astype(np.complex128)
            + 1j * page_arrays[2 * channel + 1]
            for channel in range(4)
        ])
        if not np.array_equal(data, expected):
            raise ValueError("calibrator raw-PFB group payload disagrees with pages")
    except (TypeError, ValueError) as exc:
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_calibrator_raw_pfb",
            message=f"calibrator raw-PFB group was dropped: {exc}",
            packet=issue_packet,
            uid=unique_packet_id,
            details={"error": str(exc)},
        )
        return None
    raw_seconds = page_raw_seconds[0]
    roles = tuple(f"calibrator_raw_pfb_page_{index}" for index in range(8))
    provenance = _auxiliary_provenance(
        packets=pages,
        roles=roles,
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Cal_RawPFB.unique_packet_id",
        uid_source_role=roles[0],
        raw_seconds=raw_seconds,
        time_source="Packet_Cal_RawPFB.time",
        time_source_role=roles[0],
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
    )
    return CalibratorRawPFBSample(
        data=data,
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
        page_raw_seconds=np.asarray(page_raw_seconds, dtype=np.float64),
        provenance=provenance,
    )


_CAL_DEBUG_FIELD_NAMES = (
    "raw_seconds",
    "embedded_metadata",
    "have_lock",
    "lock_ant",
    "drift",
    "powertop0",
    "powertop1",
    "powertop2",
    "powertop3",
    "powerbot0",
    "powerbot1",
    "powerbot2",
    "powerbot3",
    "fd0",
    "fd1",
    "fd2",
    "fd3",
    "sd0",
    "sd1",
    "sd2",
    "sd3",
    "fdx",
    "sdx",
    "snr0",
    "snr1",
    "snr2",
    "snr3",
)

_CAL_DEBUG_PAGE_ARRAYS = {
    0: (
        ("have_lock", np.int64),
        ("lock_ant", np.int64),
        ("drift", np.float64),
        ("powertop0", np.int64),
    ),
    1: (
        ("powertop1", np.int64),
        ("powertop2", np.int64),
        ("powertop3", np.int64),
    ),
    2: (
        ("powerbot0", np.int64),
        ("powerbot1", np.int64),
        ("powerbot2", np.int64),
    ),
    3: (
        ("powerbot3", np.int64),
        ("fd0", np.int64),
        ("fd1", np.int64),
    ),
    4: (
        ("fd2", np.int64),
        ("fd3", np.int64),
        ("sd0", np.int64),
    ),
    5: (
        ("sd1", np.int64),
        ("sd2", np.int64),
        ("sd3", np.int64),
    ),
    6: (
        ("fdx", np.int64),
        ("sdx", np.int64),
        ("snr0", np.float64),
    ),
    7: (
        ("snr1", np.float64),
        ("snr2", np.float64),
        ("snr3", np.float64),
    ),
}


def _embedded_calibrator_metadata(
    metadata: object,
    *,
    unique_packet_id: int,
    binding_key: str,
) -> dict[str, object]:
    embedded_uid = _required_uint(
        _required_public_attribute(metadata, "unique_packet_id"),
        name="embedded calibrator metadata unique_packet_id",
        bits=32,
    )
    if embedded_uid != unique_packet_id:
        raise ValueError("embedded calibrator metadata UID disagrees with group")
    if _required_public_attribute(metadata, "from_debug") is not True:
        raise ValueError("embedded calibrator metadata is not marked from_debug")
    fields, field_present = _extract_calibrator_metadata_fields(
        metadata,
        binding_key=binding_key,
        include_derived_drift=False,
    )
    raw_seconds = raw_seconds_from_split_time(
        fields["time_32"],
        fields["time_16"],
    )
    decoded_time = _required_finite_float(
        _required_public_attribute(metadata, "time"),
        name="embedded calibrator metadata time",
    )
    if decoded_time != raw_seconds:
        raise ValueError(
            "embedded calibrator metadata split and decoded times disagree"
        )
    return {
        "fields": fields,
        "field_present": field_present,
        "raw_seconds": raw_seconds,
    }


def _adapt_calibrator_debug_group(
    group_value: object,
    *,
    decoder: object,
    selected_binding: UncraterBindingInfo,
    issue_ids_by_packet_index: Mapping[int, tuple[str, ...]],
    issue_collector: IssueCollector,
) -> CalibratorDebugSample | None:
    group, unique_packet_id = _calibrator_group_header(
        group_value,
        name="Collection.calibrator_debug_groups",
        keys=frozenset({
            "unique_packet_id",
            "pages",
            "schema_binding",
        }),
        selected_binding=selected_binding,
    )
    pages_value = group["pages"]
    if type(pages_value) is not tuple or not pages_value:
        raise IncompatibleUncraterError(
            "uncrater calibrator debug group has no page tuple"
        )
    issue_packet = pages_value[0]
    try:
        pages, page_raw_seconds = _validated_group_pages(
            group,
            name="calibrator debug",
            packet_class=decoder.Packet_Cal_Debug,
            page_count=8,
            unique_packet_id=unique_packet_id,
            selected_binding=selected_binding,
        )
        normalized_pages = []
        for page_index, packet in enumerate(pages):
            debug_page = _required_uint(
                _required_public_attribute(packet, "debug_page"),
                name=f"calibrator debug page {page_index} index",
                bits=3,
            )
            if debug_page != page_index:
                raise ValueError(
                    "calibrator debug page index disagrees with order"
                )
            fields, field_present = _new_field_union(_CAL_DEBUG_FIELD_NAMES)
            _put_field(
                fields,
                field_present,
                "raw_seconds",
                page_raw_seconds[page_index],
            )
            for field_name, dtype in _CAL_DEBUG_PAGE_ARRAYS[page_index]:
                _put_field(
                    fields,
                    field_present,
                    field_name,
                    _exact_ndarray(
                        _required_public_attribute(packet, field_name),
                        name=(
                            f"calibrator debug page {page_index} "
                            f"{field_name}"
                        ),
                        dtype=dtype,
                        shape=(1024,),
                        finite=np.issubdtype(np.dtype(dtype), np.floating),
                    ),
                )
            if page_index == 0:
                _put_field(
                    fields,
                    field_present,
                    "embedded_metadata",
                    _embedded_calibrator_metadata(
                        _required_public_attribute(packet, "metadata"),
                        unique_packet_id=unique_packet_id,
                        binding_key=selected_binding.binding_key,
                    ),
                )
            normalized_pages.append(CalibratorDebugPage(
                page=page_index,
                fields=fields,
                field_present=field_present,
            ))
    except (TypeError, ValueError) as exc:
        _record_adapter_issue(
            issue_collector,
            code="decode_adapter.invalid_calibrator_debug",
            message=f"calibrator debug group was dropped: {exc}",
            packet=issue_packet,
            uid=unique_packet_id,
            details={"error": str(exc)},
        )
        return None
    raw_seconds = page_raw_seconds[0]
    roles = tuple(f"calibrator_debug_page_{index}" for index in range(8))
    provenance = _auxiliary_provenance(
        packets=pages,
        roles=roles,
        unique_packet_id=unique_packet_id,
        uid_source="Packet_Cal_Debug.unique_packet_id",
        uid_source_role=roles[0],
        raw_seconds=raw_seconds,
        time_source="Packet_Cal_Debug.time",
        time_source_role=roles[0],
        selected_binding=selected_binding,
        issue_ids_by_packet_index=issue_ids_by_packet_index,
    )
    return CalibratorDebugSample(
        pages=tuple(normalized_pages),
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
        page_raw_seconds=np.asarray(page_raw_seconds, dtype=np.float64),
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------

_PRODUCT_LISTS_WITH_PROVENANCE = (
    "spectra",
    "tr_spectra",
    "zoom_spectra",
    "grimm_spectra",
    "waveforms",
    "housekeeping",
    "cal_data",
    "calibrator_metadata",
    "calibrator_data",
    "calibrator_raw_pfb",
    "calibrator_debug",
)
_DROPPED_ADAPTER_ISSUE_FAMILY = {
    "decode_adapter.invalid_normal_product": "spectra",
    "decode_adapter.duplicate_normal_product": "spectra",
    "decode_adapter.invalid_tr_geometry": "tr_spectra",
    "decode_adapter.invalid_tr_product": "tr_spectra",
    "decode_adapter.duplicate_tr_product": "tr_spectra",
    "decode_adapter.invalid_zoom": "zoom_spectra",
    "decode_adapter.invalid_waveform_metadata": "waveforms",
    "decode_adapter.invalid_waveform": "waveforms",
    "decode_adapter.invalid_grimm": "grimm_spectra",
    "decode_adapter.invalid_housekeeping": "housekeeping",
    "decode_adapter.invalid_calibrator_metadata": "calibrator_metadata",
    "decode_adapter.invalid_calibrator_data": "calibrator_data",
    "decode_adapter.invalid_calibrator_raw_pfb": "calibrator_raw_pfb",
    "decode_adapter.invalid_calibrator_debug": "calibrator_debug",
}


def _dropped_family_issue_ids(
    issues: Sequence[IngestIssue],
    *,
    contextual_issue_ids: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, tuple[str, ...]]:
    by_family: dict[str, set[str]] = defaultdict(set)
    for issue in issues:
        family = _DROPPED_ADAPTER_ISSUE_FAMILY.get(issue.code)
        if family is not None:
            by_family[family].add(issue.issue_id)
    for family, issue_ids in (contextual_issue_ids or {}).items():
        by_family[family].update(issue_ids)
    known_issue_ids = {issue.issue_id for issue in issues}
    referenced_issue_ids = set().union(*by_family.values()) if by_family else set()
    if not referenced_issue_ids <= known_issue_ids:
        raise ValueError("family issue attribution references an unknown issue")
    return {
        family: tuple(
            issue.issue_id
            for issue in issues
            if issue.issue_id in issue_ids
        )
        for family, issue_ids in sorted(by_family.items())
    }


def _enrich_products_from_packet_map(
    products: Products,
    packet_map: PacketMap,
) -> None:
    """Attach verified extracted-file identity without reconstructing frames."""
    lookup = packet_map.by_output_index
    for list_name in _PRODUCT_LISTS_WITH_PROVENANCE:
        rows = getattr(products, list_name)
        for row_index, row in enumerate(rows):
            provenance = getattr(row, "provenance", None)
            if not isinstance(provenance, ProductProvenance):
                continue
            if not provenance.source_packets:
                continue
            enriched_packets = []
            for source in provenance.source_packets:
                if source.packet_index is None:
                    raise PacketMapError(
                        "decoded source packet lacks the packet-map output index"
                    )
                entry = lookup.get(source.packet_index)
                if entry is None:
                    raise PacketMapError(
                        "decoded source packet is absent from packet_map.json"
                    )
                if source.original_appid != entry.original_appid:
                    raise PacketMapError(
                        "decoded source packet original AppID disagrees with packet map"
                    )
                if entry.unique_packet_id != provenance.uid:
                    raise PacketMapError(
                        "decoded product UID disagrees with packet map"
                    )
                if (
                    source.normalized_appid is not None
                    and source.normalized_appid != entry.normalized_appid
                ):
                    raise PacketMapError(
                        "decoded source packet normalized AppID disagrees "
                        "with packet map"
                    )
                if source.filename not in (None, entry.output_filename):
                    raise PacketMapError(
                        "decoded source packet filename disagrees with packet map"
                    )
                if source.bank not in (None, entry.source_bank):
                    raise PacketMapError(
                        "decoded source packet bank disagrees with packet map"
                    )
                enriched_packets.append(replace(
                    source,
                    filename=entry.output_filename,
                    bank=entry.source_bank,
                    original_appid=entry.original_appid,
                    normalized_appid=entry.normalized_appid,
                ))
            rows[row_index] = replace(
                row,
                provenance=replace(
                    provenance,
                    source_packets=tuple(enriched_packets),
                ),
            )

def read_uncrater_session(
    session_dir: Path | str,
    *,
    issue_collector: IssueCollector | None = None,
    strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
) -> Products:
    """Decode an uncrater session directory into a ``Products`` instance."""
    if issue_collector is None:
        issue_collector = IssueCollector("strict" if strict else "collect")
    elif not isinstance(issue_collector, IssueCollector):
        raise TypeError("issue_collector must be an IssueCollector")
    issue_marker = issue_collector.mark()
    session_dir = Path(session_dir)
    cdi = session_dir / "cdi_output"
    if not cdi.is_dir():
        cdi = session_dir   # Layout A: bare directory of *.bin files

    decoder = load_uncrater()
    packet_map = read_packet_map(
        session_dir,
        cdi,
        normalize_appid=decoder.normalize_dcb_appid,
    )
    coll = make_collection(
        cdi,
        strict=strict,
        diagnostic_override=diagnostic_override,
        schema_variant=schema_variant,
    )
    selected_binding = binding_info(coll)
    log.info(
        "session %s uses uncrater binding %s (schema 0x%03x, assumed=%s)",
        session_dir,
        selected_binding.binding_key,
        selected_binding.selected_schema_id,
        selected_binding.schema_assumed,
    )
    decode_provenance = collection_provenance(coll, strict=strict)
    products = Products(
        decode_provenance=decode_provenance,
        packet_map_status="verified" if packet_map is not None else "unavailable",
        packet_map_format_version=(
            PACKET_MAP_FORMAT_VERSION if packet_map is not None else None
        ),
        raw_flash_provenance_unavailable_reason=(
            None if packet_map is not None else "packet_map_missing_legacy_session"
        ),
    )
    imported_issues = import_decode_issues(
        coll,
        issue_collector,
        selected_binding,
    )
    contextual_family_issue_ids: dict[str, list[str]] = defaultdict(list)

    def record_packet_issues(family: str, packets: Sequence[object]) -> None:
        issue_ids = _issue_ids_for_packets(
            packets,
            imported_issues.issue_ids_by_packet_index,
        )
        if issue_ids:
            contextual_family_issue_ids[family].extend(issue_ids)

    fatal_packet_families = (
        (decoder.Packet_Spectrum, "spectra"),
        (decoder.Packet_TR_Spectrum, "tr_spectra"),
        (decoder.Packet_Cal_ZoomSpectra, "zoom_spectra"),
        (decoder.Packet_Waveform_Meta, "waveforms"),
        (decoder.Packet_Waveform, "waveforms"),
        (decoder.Packet_Housekeep, "housekeeping"),
        (decoder.Packet_Grimm, "grimm_spectra"),
        (decoder.Packet_Cal_Metadata, "calibrator_metadata"),
        (decoder.Packet_Cal_Data, "calibrator_data"),
        (decoder.Packet_Cal_RawPFB, "calibrator_raw_pfb"),
        (decoder.Packet_Cal_Debug, "calibrator_debug"),
    )
    for packet in _required_public_attribute(coll, "cont"):
        if not _packet_has_fatal_issue(packet):
            continue
        for packet_class, family in fatal_packet_families:
            if isinstance(packet, packet_class):
                record_packet_issues(family, (packet,))

    # ---- Hello / session-invariants ----
    for pkt in coll.cont:
        if not isinstance(pkt, decoder.Packet_Hello):
            continue
        if _packet_has_fatal_issue(pkt):
            continue
        source_packet_provenance(
            pkt,
            role="hello",
            binding=selected_binding,
        )
        try:
            sw_version = _required_uint(
                _required_public_attribute(pkt, "SW_version"),
                name="Hello SW_version",
                bits=32,
            )
            fw_version = _required_uint(
                _required_public_attribute(pkt, "FW_Version"),
                name="Hello FW_Version",
                bits=32,
            )
            fw_id = _required_uint(
                _required_public_attribute(pkt, "FW_ID"),
                name="Hello FW_ID",
                bits=32,
            )
            fw_date = _required_uint(
                _required_public_attribute(pkt, "FW_Date"),
                name="Hello FW_Date",
                bits=32,
            )
            fw_time = _required_uint(
                _required_public_attribute(pkt, "FW_Time"),
                name="Hello FW_Time",
                bits=32,
            )
            uid = _required_uint(
                _required_public_attribute(pkt, "unique_packet_id"),
                name="Hello unique_packet_id",
                bits=32,
            )
            t32 = _required_uint(
                _required_public_attribute(pkt, "time_32"),
                name="Hello time_32",
                bits=32,
            )
            t16 = _required_uint(
                _required_public_attribute(pkt, "time_16"),
                name="Hello time_16",
                bits=16,
            )
        except (TypeError, ValueError) as exc:
            _record_adapter_issue(
                issue_collector,
                code="decode_adapter.invalid_hello",
                message=f"Hello packet was dropped: {exc}",
                packet=pkt,
                uid=None,
                details={"error": str(exc)},
            )
            continue
        products.sw_version = sw_version
        products.fw_version = fw_version
        products.fw_id = fw_id
        products.fw_date = fw_date
        products.fw_time = fw_time
        products.start_unique_packet_id = uid
        products.start_time_32 = t32
        products.start_time_16 = t16
        products.start_raw_seconds = raw_seconds_from_split_time(t32, t16)
        break

    # ---- Strict metadata, normal spectra, and TR spectra ----
    normal_metadata = _validate_spectrum_groups(
        coll,
        name="spectra",
        packet_class=decoder.Packet_Spectrum,
        metadata_class=decoder.Packet_Metadata,
    )
    tr_metadata = _validate_spectrum_groups(
        coll,
        name="tr_spectra",
        packet_class=decoder.Packet_TR_Spectrum,
        metadata_class=decoder.Packet_Metadata,
    )
    normal_candidates = _candidate_packets(
        coll, packet_class=decoder.Packet_Spectrum
    )
    tr_candidates = _candidate_packets(
        coll, packet_class=decoder.Packet_TR_Spectrum
    )

    normalized_metadata: dict[int, SpectrumMetadata | None] = {}
    for meta in (*normal_metadata, *tr_metadata):
        if id(meta) in normalized_metadata:
            continue
        # This validates the public packet/status/schema surface before field
        # adaptation; API drift is a preflight incompatibility, not row damage
        source_packet_provenance(
            meta,
            role="metadata",
            binding=selected_binding,
        )
        try:
            normalized_metadata[id(meta)] = _extract_spectrum_metadata(
                meta,
                binding_key=selected_binding.binding_key,
            )
        except (TypeError, ValueError) as exc:
            uid_value = getattr(meta, "unique_packet_id", None)
            uid = (
                int(uid_value)
                if isinstance(uid_value, (int, np.integer))
                and not isinstance(uid_value, (bool, np.bool_))
                and 0 <= int(uid_value) <= 0xFFFFFFFF
                else None
            )
            issue_id = _record_adapter_issue(
                issue_collector,
                code="decode_adapter.invalid_spectrum_metadata",
                message=f"spectrum metadata was dropped: {exc}",
                packet=meta,
                uid=uid,
                details={"error": str(exc)},
            )
            if normal_candidates.get(id(meta)):
                contextual_family_issue_ids["spectra"].append(issue_id)
            if tr_candidates.get(id(meta)):
                contextual_family_issue_ids["tr_spectra"].append(issue_id)
            normalized_metadata[id(meta)] = None

    for meta in normal_metadata:
        metadata = normalized_metadata[id(meta)]
        if metadata is None:
            continue
        sample = _adapt_normal_spectrum(
            meta=meta,
            metadata=metadata,
            candidates=normal_candidates.get(id(meta), {}),
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        # A metadata-only group is not a fabricated all-NaN science row
        if sample is not None:
            products.spectra.append(sample)

    for meta in tr_metadata:
        metadata = normalized_metadata[id(meta)]
        if metadata is None:
            continue
        sample = _adapt_tr_spectrum(
            meta=meta,
            metadata=metadata,
            candidates=tr_candidates.get(id(meta), {}),
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.tr_spectra.append(sample)

    metadata_lookup = _metadata_uid_lookup(
        normal_metadata,
        normalized_metadata,
    )

    # ---- Zoom spectra ----
    zoom_packets = _required_public_attribute(coll, "zoom_spectra_packets")
    if not isinstance(zoom_packets, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.zoom_spectra_packets must be a packet sequence"
        )
    for packet in zoom_packets:
        if not isinstance(packet, decoder.Packet_Cal_ZoomSpectra):
            raise IncompatibleUncraterError(
                "uncrater Collection.zoom_spectra_packets has an invalid class"
            )
        if _packet_has_fatal_issue(packet):
            raise IncompatibleUncraterError(
                "uncrater Collection.zoom_spectra_packets published a fatal packet"
            )
        sample = _adapt_zoom_packet(
            packet,
            metadata_lookup=metadata_lookup,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.zoom_spectra.append(sample)

    # ---- Waveforms ----
    waveform_groups = _required_public_attribute(coll, "waveform_groups")
    if not isinstance(waveform_groups, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.waveform_groups must be a group sequence"
        )
    for group in waveform_groups:
        products.waveforms.extend(_adapt_waveform_group(
            group,
            decoder=decoder,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        ))

    # ---- Housekeeping ----
    housekeeping_packets = _required_public_attribute(
        coll,
        "housekeeping_packets",
    )
    if not isinstance(housekeeping_packets, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.housekeeping_packets must be a packet sequence"
        )
    for packet in housekeeping_packets:
        if not isinstance(packet, decoder.Packet_Housekeep):
            raise IncompatibleUncraterError(
                "uncrater Collection.housekeeping_packets has an invalid class"
            )
        if _packet_has_fatal_issue(packet):
            raise IncompatibleUncraterError(
                "uncrater Collection.housekeeping_packets published a fatal packet"
            )
        sample = _adapt_housekeeping_packet(
            packet,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.housekeeping.append(sample)

    # ---- Calibrator metadata and complete multipart products ----
    for packet in _required_public_attribute(coll, "cont"):
        if not isinstance(packet, decoder.Packet_Cal_Metadata):
            continue
        if _packet_has_fatal_issue(packet):
            continue
        sample = _adapt_calibrator_metadata_packet(
            packet,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.calibrator_metadata.append(sample)

    calibrator_data_groups = _required_public_attribute(
        coll,
        "calibrator_data_groups",
    )
    if not isinstance(calibrator_data_groups, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.calibrator_data_groups must be a group sequence"
        )
    for group in calibrator_data_groups:
        sample = _adapt_calibrator_data_group(
            group,
            decoder=decoder,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.calibrator_data.append(sample)

    calibrator_pfb_groups = _required_public_attribute(
        coll,
        "calibrator_pfb_groups",
    )
    if not isinstance(calibrator_pfb_groups, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.calibrator_pfb_groups must be a group sequence"
        )
    for group in calibrator_pfb_groups:
        sample = _adapt_calibrator_pfb_group(
            group,
            decoder=decoder,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.calibrator_raw_pfb.append(sample)

    calibrator_debug_groups = _required_public_attribute(
        coll,
        "calibrator_debug_groups",
    )
    if not isinstance(calibrator_debug_groups, (list, tuple)):
        raise IncompatibleUncraterError(
            "uncrater Collection.calibrator_debug_groups must be a group sequence"
        )
    for group in calibrator_debug_groups:
        sample = _adapt_calibrator_debug_group(
            group,
            decoder=decoder,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.calibrator_debug.append(sample)

    # ---- Grimm spectra ----
    for packet in _required_public_attribute(coll, "cont"):
        if not isinstance(packet, decoder.Packet_Grimm):
            continue
        if _packet_has_fatal_issue(packet):
            continue
        sample = _adapt_grimm_packet(
            packet,
            metadata_lookup=metadata_lookup,
            selected_binding=selected_binding,
            issue_ids_by_packet_index=imported_issues.issue_ids_by_packet_index,
            issue_collector=issue_collector,
        )
        if sample is not None:
            products.grimm_spectra.append(sample)

    if packet_map is not None:
        _enrich_products_from_packet_map(products, packet_map)

    log.info(
        (
            "session %s: %d spectra, %d tr_spectra, %d zoom, %d waveforms, "
            "%d hk, %d Grimm, %d calibrator metadata, %d data, %d raw-PFB, "
            "%d debug"
        ),
        session_dir,
        len(products.spectra),
        len(products.tr_spectra),
        len(products.zoom_spectra),
        len(products.waveforms),
        len(products.housekeeping),
        len(products.grimm_spectra),
        len(products.calibrator_metadata),
        len(products.calibrator_data),
        len(products.calibrator_raw_pfb),
        len(products.calibrator_debug),
    )
    products.issues = issue_collector.since(issue_marker)
    products.family_issue_ids = _dropped_family_issue_ids(
        products.issues,
        contextual_issue_ids=contextual_family_issue_ids,
    )
    product_rows = [
        (name, len(rows))
        for name, rows in (
            ("spectra", products.spectra),
            ("tr_spectra", products.tr_spectra),
            ("zoom_spectra", products.zoom_spectra),
            ("waveforms", products.waveforms),
            ("housekeeping", products.housekeeping),
            ("grimm_spectra", products.grimm_spectra),
            ("calibrator_metadata", products.calibrator_metadata),
            ("calibrator_data", products.calibrator_data),
            ("calibrator_raw_pfb", products.calibrator_raw_pfb),
            ("calibrator_debug", products.calibrator_debug),
        )
        if rows
    ]
    products.validated_counts = ValidatedCounts(
        input_packets=products.decode_provenance.input_packet_count,
        valid_packets=products.decode_provenance.valid_packet_count,
        product_rows=tuple(product_rows),
    )
    products.quality_status = _decode_quality(
        products.decode_provenance,
        usable_product_count=_usable_product_count(products),
        adaptation_issue_count=sum(
            issue.stage == "decode_adapter" for issue in products.issues
        ),
    )
    return products
