"""Stage 6: decode an uncrater session directory into in-memory products.

Wraps ``uncrater.Collection`` and converts its typed packet objects into
plain numpy arrays / dicts in a ``Products`` dataclass. ``hdf5_writer``
consumes the result; it never sees uncrater types.

Lazy-imports uncrater so that ``lusee.ingest`` is importable without it.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .constants import (
    BITSLICE_REFERENCE,
    NCHANNELS,
    NPRODUCTS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
    ZOOM_COMPONENTS,
)
from .frequency_contract import spectrometer_frequency_window
from .issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from .products import (
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ProductProvenance,
    SpectrumMetadata,
    ValidatedCounts,
)
from .products import SpectrumSample as ValidatedSpectrumSample
from .products import TRSpectrumSample as ValidatedTRSpectrumSample
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
    )
    return len(products.housekeeping) + sum(
        np.asarray(row.data).size > 0
        and bool(np.any(np.isfinite(row.data)))
        for row in array_rows
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
    decode_provenance: DecodeProvenance = field(
        default_factory=DecodeProvenance.unavailable
    )
    quality_status: DataQuality | None = None
    validated_counts: ValidatedCounts = field(default_factory=ValidatedCounts)
    issues: tuple[IngestIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decode_provenance, DecodeProvenance):
            raise TypeError("decode_provenance must be a DecodeProvenance record")
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


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------

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
    )
    imported_issues = import_decode_issues(
        coll,
        issue_collector,
        selected_binding,
    )

    # ---- Hello / session-invariants ----
    for pkt in coll.cont:
        if getattr(pkt, "appid", None) == 0x209:    # AppID_uC_Start
            try:
                read_packet(pkt)
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

    # ---- Strict metadata, normal spectra, and TR spectra ----
    decoder = load_uncrater()
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
            _record_adapter_issue(
                issue_collector,
                code="decode_adapter.invalid_spectrum_metadata",
                message=f"spectrum metadata was dropped: {exc}",
                packet=meta,
                uid=uid,
                details={"error": str(exc)},
            )
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

    # ---- Zoom spectra ----
    last_meta_seconds = None
    spec_iter = iter(products.spectra)
    next_spec = next(spec_iter, None)
    for zpkt in getattr(coll, "zoom_spectra_packets", []):
        try:
            read_packet(zpkt)
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
            read_packet(wpkt)
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"waveform decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        data = np.asarray(getattr(wpkt, "data", []), dtype=np.int16)
        if data.size != WAVEFORM_SAMPLES:
            tmp = np.zeros(WAVEFORM_SAMPLES, dtype=np.int16)
            n = min(data.size, WAVEFORM_SAMPLES)
            tmp[:n] = data[:n]
            data = tmp
        products.waveforms.append(WaveformSample(
            data=data,
            channel=int(getattr(wpkt, "channel", 0)),
            unique_packet_id=int(getattr(wpkt, "unique_packet_id", 0)),
            raw_seconds=None,
            adc_timestamp=_waveform_adc_timestamp(wpkt),
        ))

    # ---- Housekeeping ----
    for hk in getattr(coll, "housekeeping_packets", []):
        try:
            read_packet(hk)
        except Exception as exc:    # noqa: BLE001
            warnings.warn(f"housekeeping decode failed: {exc}", RuntimeWarning, stacklevel=2)
            continue
        hk_type = int(getattr(hk, "hk_type", 0))
        # Surface raw_seconds for HK rows. uncrater exposes a "time" field
        # for hk_type 0/2 derived from the heartbeat / core_state mission
        # time; types without a time field remain explicitly missing.
        raw_seconds = _finite_seconds_or_none(getattr(hk, "time", None))
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
            read_packet(pkt)
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
            raw_seconds=None,
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
    products.issues = issue_collector.since(issue_marker)
    product_rows = []
    if products.spectra:
        product_rows.append(("spectra", len(products.spectra)))
    if products.tr_spectra:
        product_rows.append(("tr_spectra", len(products.tr_spectra)))
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
