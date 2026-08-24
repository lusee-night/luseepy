"""Validated, format-neutral input contract for layout-v4 writers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .clock_reference import ClockReferenceSet, ClockSource
from .constants import MISSION_TIME_FRACT_DIVISOR, MISSION_TIME_FRACT_SHIFT
from .decode import Products
from .issues import IngestIssue
from .products import (
    CalibratorDataSample,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    DataQuality,
    GrimmSample,
    HKSample,
    SpectrumSample,
    TRSpectrumSample,
    WaveformSample,
    ZoomSample,
)

FAMILY_TYPES = (
    ("spectra", SpectrumSample),
    ("tr_spectra", TRSpectrumSample),
    ("zoom_spectra", ZoomSample),
    ("waveforms", WaveformSample),
    ("housekeeping", HKSample),
    ("grimm_spectra", GrimmSample),
    ("calibrator_metadata", CalibratorMetadataSample),
    ("calibrator_data", CalibratorDataSample),
    ("calibrator_raw_pfb", CalibratorRawPFBSample),
    ("calibrator_debug", CalibratorDebugSample),
)
UNSUPPORTED_FAMILIES = (
    "fw_direct_spectra",
    "legacy_cal_data",
    "dcb_telemetry",
    "encoder_telemetry",
    "legacy_sidecar_telemetry",
    "interpolated_telemetry",
)
ALL_FAMILIES = tuple(name for name, _ in FAMILY_TYPES) + UNSUPPORTED_FAMILIES


class FamilyCoverage(StrEnum):
    """How one known product family is represented in this request."""

    PERSISTED = "persisted"
    ABSENT_IN_INPUT = "absent_in_input"
    INVALID_OR_DROPPED = "invalid_or_dropped"
    UNSUPPORTED = "unsupported"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_utf8(value: str, name: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc


def _optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string or None")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a null character")
    _validate_utf8(value, name)
    return value


def _optional_uint(value: int | None, bits: int, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer or None")
    return value


def _validate_issue_for_storage(issue: IngestIssue) -> None:
    for name in ("issue_id", "code", "stage", "message"):
        value = getattr(issue, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"issue {name} must be a nonempty string")
        if "\x00" in value:
            raise ValueError(f"issue {name} must not contain a null character")
        _validate_utf8(value, f"issue {name}")
    for name in ("input_identity", "bank", "session"):
        _optional_text(getattr(issue, name), f"issue {name}")
    for name, bits in (
        ("byte_offset", 64),
        ("frame_index", 64),
        ("packet_index", 64),
        ("appid", 11),
        ("sequence_count", 14),
        ("uid", 32),
    ):
        _optional_uint(getattr(issue, name), bits, f"issue {name}")


def _validate_session_invariants(products: Products) -> None:
    fields = (
        ("sw_version", 32),
        ("fw_version", 32),
        ("fw_id", 32),
        ("fw_date", 32),
        ("fw_time", 32),
        ("start_unique_packet_id", 32),
        ("start_time_32", 32),
        ("start_time_16", 16),
    )
    values = tuple(getattr(products, name) for name, _ in fields)
    present = all(value is not None for value in values)
    if any(value is not None for value in values) != present:
        raise ValueError("Hello session invariants must be all present or all absent")
    for name, bits in fields:
        _optional_uint(getattr(products, name), bits, f"Products.{name}")
    raw_seconds = products.start_raw_seconds
    if not present:
        if raw_seconds is not None:
            raise ValueError(
                "start_raw_seconds cannot be present without Hello invariants"
            )
        return
    raw_seconds = _finite_float(raw_seconds, "Products.start_raw_seconds")
    combined = (products.start_time_16 << 32) + products.start_time_32
    expected = (combined >> MISSION_TIME_FRACT_SHIFT) / MISSION_TIME_FRACT_DIVISOR
    if raw_seconds != expected:
        raise ValueError("start_raw_seconds disagrees with Hello split time")


def _validate_product_provenance_for_storage(row: object) -> None:
    provenance = row.provenance
    for name in (
        "uid_source",
        "uid_source_role",
        "time_source",
        "time_source_role",
        "clock_source",
    ):
        _optional_text(getattr(provenance, name), f"product provenance {name}")
    for packet in provenance.source_packets:
        for name in ("role", "filename", "bank"):
            _optional_text(getattr(packet, name), f"source packet {name}")
        for name in (
            "packet_index",
            "frame_start",
            "frame_stop",
            "byte_offset_start",
            "byte_offset_stop",
        ):
            _optional_uint(
                getattr(packet, name),
                64,
                f"source packet {name}",
            )


def _validate_field_names(
    values: Mapping[str, object],
    *,
    context: str,
) -> None:
    for name, value in values.items():
        if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None:
            raise ValueError(f"{context} field names must be lowercase identifiers")
        if isinstance(value, Mapping):
            _validate_field_names(value, context=f"{context}.{name}")
        else:
            _validate_normalized_field_value(
                value,
                context=f"{context}.{name}",
            )


def _validate_normalized_field_value(value: object, *, context: str) -> None:
    if value is None or type(value) in (bool, bytes):
        return
    if type(value) is str:
        if "\x00" in value:
            raise ValueError(f"{context} must not contain a null character")
        _validate_utf8(value, context)
        return
    if type(value) is int:
        if not -(1 << 63) <= value < 1 << 64:
            raise ValueError(f"{context} integer is not storage-representable")
        return
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{context} must not produce object dtype")
    if array.ndim > 31:
        raise ValueError(
            f"{context} must have at most 31 dimensions for layout v4"
        )
    if array.dtype.metadata is not None:
        raise TypeError(
            f"{context} dtype metadata is not portable for layout v4"
        )
    if not array.dtype.isnative:
        raise TypeError(
            f"{context} non-native dtype {array.dtype} is not portable "
            "for layout v4"
        )
    portable_dtype = (
        array.dtype.kind == "b"
        or (array.dtype.kind in "iu" and array.dtype.itemsize in (1, 2, 4, 8))
        or (array.dtype.kind == "f" and array.dtype.itemsize in (4, 8))
        or (array.dtype.kind == "c" and array.dtype.itemsize in (8, 16))
        or array.dtype.kind in "SU"
    )
    if not portable_dtype:
        raise TypeError(
            f"{context} dtype {array.dtype} is not portable for layout v4"
        )
    if array.dtype.kind == "U":
        for item in array.reshape(-1).tolist():
            if "\x00" in item:
                raise ValueError(f"{context} must not contain a null character")
            _validate_utf8(item, context)


def _validate_clock_conversions(
    products: Products,
    reference_set: ClockReferenceSet | None,
) -> None:
    if reference_set is None:
        return
    raw_by_source: dict[str, list[float]] = {}
    for family, _ in FAMILY_TYPES:
        for row in getattr(products, family):
            source = row.provenance.clock_source
            if (
                row.raw_seconds is not None
                and source is not None
                and reference_set.reference_for(source) is not None
            ):
                raw_by_source.setdefault(source, []).append(row.raw_seconds)
    spectrometer = ClockSource.SPECTROMETER.value
    if reference_set.reference_for(spectrometer) is not None:
        for family in (
            "calibrator_data",
            "calibrator_raw_pfb",
            "calibrator_debug",
        ):
            for row in getattr(products, family):
                raw_by_source.setdefault(spectrometer, []).extend(
                    row.page_raw_seconds.tolist()
                )
    for source, raw_seconds in raw_by_source.items():
        reference_set.to_mjd(
            np.asarray(raw_seconds, dtype=np.float64),
            clock_source=source,
        )


def _validate_field_union(
    rows: list[Mapping[str, object]],
    *,
    context: str,
) -> None:
    keys = sorted({key for row in rows for key in row})
    for name in keys:
        values = [row[name] for row in rows if row.get(name) is not None]
        mapping_values = [value for value in values if isinstance(value, Mapping)]
        if mapping_values and len(mapping_values) != len(values):
            raise TypeError(f"{context}.{name} mixes mappings and arrays")
        if mapping_values:
            _validate_field_union(
                mapping_values,
                context=f"{context}.{name}",
            )


@dataclass(frozen=True, slots=True)
class LunarLocation:
    """Validated lunar observing location recorded in every v4 file."""

    latitude_deg: float
    longitude_deg: float
    height_m: float

    def __post_init__(self) -> None:
        latitude = _finite_float(self.latitude_deg, "latitude_deg")
        longitude = _finite_float(self.longitude_deg, "longitude_deg")
        height = _finite_float(self.height_m, "height_m")
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude_deg must lie in [-90, 90]")
        if not -180.0 <= longitude <= 360.0:
            raise ValueError("longitude_deg must lie in [-180, 360]")
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "longitude_deg", longitude)
        object.__setattr__(self, "height_m", height)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Portable input identity plus optional local diagnostic provenance."""

    input_identity: str | None
    input_identity_kind: str | None
    input_identity_unavailable_reason: str | None
    source_kind: str
    source_path: str | None = None
    pipeline_version: str | None = None

    def __post_init__(self) -> None:
        identity = _optional_text(self.input_identity, "input_identity")
        identity_kind = _optional_text(self.input_identity_kind, "input_identity_kind")
        unavailable = _optional_text(
            self.input_identity_unavailable_reason,
            "input_identity_unavailable_reason",
        )
        if (identity is None) == (unavailable is None):
            raise ValueError(
                "record exactly one of input_identity or "
                "input_identity_unavailable_reason"
            )
        if (identity is None) != (identity_kind is None):
            raise ValueError(
                "input_identity and input_identity_kind must be recorded together"
            )
        if (
            identity_kind is not None
            and re.fullmatch(r"[a-z][a-z0-9_]*", identity_kind) is None
        ):
            raise ValueError("input_identity_kind must be a lowercase identifier")
        source_kind = _optional_text(self.source_kind, "source_kind")
        if source_kind is None:
            raise ValueError("source_kind must be present")
        source_path = _optional_text(self.source_path, "source_path")
        if source_path is not None and not Path(source_path).is_absolute():
            raise ValueError("source_path must be absolute when recorded")
        pipeline_version = _optional_text(self.pipeline_version, "pipeline_version")
        object.__setattr__(self, "input_identity", identity)
        object.__setattr__(self, "input_identity_kind", identity_kind)
        object.__setattr__(self, "input_identity_unavailable_reason", unavailable)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "pipeline_version", pipeline_version)

    def as_record(self) -> dict[str, str | None]:
        """Return one deterministic writer-facing record."""
        return {
            "input_identity": self.input_identity,
            "input_identity_kind": self.input_identity_kind,
            "input_identity_unavailable_reason": (
                self.input_identity_unavailable_reason
            ),
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "pipeline_version": self.pipeline_version,
        }


@dataclass(frozen=True, slots=True)
class InterpolationPolicy:
    """Named telemetry-alignment policy retained even when interpolation is off."""

    mode: str = "none"
    maximum_gap_seconds: float | None = None
    extrapolate: bool = False

    def __post_init__(self) -> None:
        if self.mode != "none":
            raise ValueError(
                "layout-v4 interpolation is unavailable until the reviewed "
                "field-aware telemetry policy is implemented"
            )
        if self.maximum_gap_seconds is not None:
            maximum_gap = _finite_float(self.maximum_gap_seconds, "maximum_gap_seconds")
            if maximum_gap <= 0.0:
                raise ValueError("maximum_gap_seconds must be positive")
            object.__setattr__(self, "maximum_gap_seconds", maximum_gap)
        if type(self.extrapolate) is not bool:
            raise TypeError("extrapolate must be a boolean")
        if self.extrapolate:
            raise ValueError("layout-v4 interpolation never extrapolates silently")


@dataclass(frozen=True, slots=True)
class FamilyStatus:
    """Explicit support, coverage, quality, and issue state for one family."""

    family: str
    supported: bool
    coverage: FamilyCoverage | str
    quality: DataQuality | str
    decoded_rows: int
    issue_ids: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.family not in ALL_FAMILIES:
            raise ValueError(f"unknown product family {self.family!r}")
        if type(self.supported) is not bool:
            raise TypeError("family supported must be a boolean")
        coverage = FamilyCoverage(self.coverage)
        quality = DataQuality(self.quality)
        if type(self.decoded_rows) is not int or self.decoded_rows < 0:
            raise ValueError("decoded_rows must be a nonnegative integer")
        issue_ids = tuple(self.issue_ids)
        if any(not isinstance(issue_id, str) or not issue_id for issue_id in issue_ids):
            raise ValueError("family issue_ids must contain nonempty strings")
        if len(set(issue_ids)) != len(issue_ids):
            raise ValueError("family issue_ids must not contain duplicates")
        reason = _optional_text(self.reason, "family reason")

        if quality is DataQuality.CLEAN and issue_ids:
            raise ValueError("a clean family cannot reference issues")
        if quality is DataQuality.PARTIAL and not issue_ids:
            raise ValueError("a partial family requires an issue reference")

        if self.supported:
            if coverage is FamilyCoverage.UNSUPPORTED:
                raise ValueError("a supported family cannot be marked unsupported")
            if self.decoded_rows:
                if coverage is not FamilyCoverage.PERSISTED:
                    raise ValueError("a supported family with rows must be persisted")
                if quality is DataQuality.FAILED:
                    raise ValueError("a persisted family cannot have failed quality")
            elif coverage is FamilyCoverage.PERSISTED:
                raise ValueError("an empty family cannot be marked persisted")
            if (
                coverage is FamilyCoverage.ABSENT_IN_INPUT
                and quality is not DataQuality.CLEAN
            ):
                raise ValueError("an absent input family must be clean")
            if (
                coverage is FamilyCoverage.INVALID_OR_DROPPED
                and quality is not DataQuality.FAILED
            ):
                raise ValueError("an invalid or dropped family must be failed")
            if coverage is FamilyCoverage.INVALID_OR_DROPPED and not issue_ids:
                raise ValueError(
                    "an invalid or dropped family requires an issue reference"
                )
        else:
            if coverage is not FamilyCoverage.UNSUPPORTED:
                raise ValueError("an unsupported family needs unsupported coverage")
            if quality is not DataQuality.FAILED or self.decoded_rows != 0:
                raise ValueError(
                    "an unsupported family must have failed quality and zero rows"
                )
            if reason is None:
                raise ValueError("an unsupported family requires a reason")

        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "issue_ids", issue_ids)
        object.__setattr__(self, "reason", reason)


def family_statuses_for_products(
    products: Products,
    *,
    family_issue_ids: Mapping[str, tuple[str, ...]],
) -> tuple[FamilyStatus, ...]:
    """Build complete statuses from explicit per-family issue attribution."""
    unknown = set(family_issue_ids) - {family for family, _ in FAMILY_TYPES}
    if unknown:
        raise ValueError(
            "family_issue_ids contains unknown supported families: "
            + ", ".join(sorted(unknown))
        )
    statuses = []
    for family, _ in FAMILY_TYPES:
        rows = len(getattr(products, family))
        issue_ids = tuple(family_issue_ids.get(family, ()))
        if rows:
            coverage = FamilyCoverage.PERSISTED
            quality = DataQuality.PARTIAL if issue_ids else DataQuality.CLEAN
        elif issue_ids:
            coverage = FamilyCoverage.INVALID_OR_DROPPED
            quality = DataQuality.FAILED
        else:
            coverage = FamilyCoverage.ABSENT_IN_INPUT
            quality = DataQuality.CLEAN
        statuses.append(
            FamilyStatus(
                family=family,
                supported=True,
                coverage=coverage,
                quality=quality,
                decoded_rows=rows,
                issue_ids=issue_ids,
            )
        )
    for family in UNSUPPORTED_FAMILIES:
        statuses.append(
            FamilyStatus(
                family=family,
                supported=False,
                coverage=FamilyCoverage.UNSUPPORTED,
                quality=DataQuality.FAILED,
                decoded_rows=0,
                reason="not_implemented_in_layout_v4",
            )
        )
    return tuple(sorted(statuses, key=lambda status: status.family))


def _empty_array_mapping() -> Mapping[str, np.ndarray]:
    return MappingProxyType({})


def _freeze_array_mapping(
    value: Mapping[str, np.ndarray] | None,
    name: str,
) -> Mapping[str, np.ndarray]:
    if value is None:
        return _empty_array_mapping()
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    normalized: dict[str, np.ndarray] = {}
    lengths = set()
    for key, raw in sorted(value.items()):
        if (
            not isinstance(key, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) is None
        ):
            raise ValueError(f"{name} keys must be identifiers")
        if type(raw) is not np.ndarray:
            raise TypeError(f"{name}.{key} must be a numpy.ndarray")
        if raw.ndim == 0:
            raise ValueError(f"{name}.{key} must have a row axis")
        if raw.dtype.hasobject:
            raise TypeError(f"{name}.{key} must not have object dtype")
        immutable = np.frombuffer(raw.tobytes(order="C"), dtype=raw.dtype).reshape(
            raw.shape
        )
        normalized[key] = immutable
        lengths.add(raw.shape[0])
    if len(lengths) > 1:
        raise ValueError(f"{name} arrays must have one row count")
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class WriteRequest:
    """Whole-request preflight contract shared by layout-v4 file writers."""

    products: Products
    clock_reference_set: ClockReferenceSet | None
    clock_reference_unavailable_reason: str | None
    location: LunarLocation
    run_provenance: RunProvenance
    issues: tuple[IngestIssue, ...]
    family_statuses: tuple[FamilyStatus, ...]
    fpga_telemetry: Mapping[str, np.ndarray] = field(
        default_factory=_empty_array_mapping
    )
    encoder_telemetry: Mapping[str, np.ndarray] = field(
        default_factory=_empty_array_mapping
    )
    interpolation_policy: InterpolationPolicy = field(
        default_factory=InterpolationPolicy
    )
    overwrite: bool = False
    hdf5_compression: str | None = "gzip"
    hdf5_compression_level: int | None = 1

    def __post_init__(self) -> None:
        if not isinstance(self.products, Products):
            raise TypeError("products must be a Products instance")
        if self.clock_reference_set is not None and not isinstance(
            self.clock_reference_set, ClockReferenceSet
        ):
            raise TypeError("clock_reference_set must be a ClockReferenceSet or None")
        if self.clock_reference_set is not None:
            _optional_text(
                self.clock_reference_set.source,
                "clock_reference_set.source",
            )
        clock_unavailable = _optional_text(
            self.clock_reference_unavailable_reason,
            "clock_reference_unavailable_reason",
        )
        if (self.clock_reference_set is None) == (clock_unavailable is None):
            raise ValueError(
                "record exactly one of clock_reference_set or "
                "clock_reference_unavailable_reason"
            )
        object.__setattr__(
            self,
            "clock_reference_unavailable_reason",
            clock_unavailable,
        )
        if not isinstance(self.location, LunarLocation):
            raise TypeError("location must be a LunarLocation")
        if not isinstance(self.run_provenance, RunProvenance):
            raise TypeError("run_provenance must be a RunProvenance")
        issues = tuple(self.issues)
        if any(not isinstance(issue, IngestIssue) for issue in issues):
            raise TypeError("issues must contain IngestIssue records")
        for issue in issues:
            _validate_issue_for_storage(issue)
        if len({issue.issue_id for issue in issues}) != len(issues):
            raise ValueError("issues must not contain duplicate issue IDs")
        object.__setattr__(self, "issues", issues)
        family_statuses = tuple(self.family_statuses)
        if any(not isinstance(status, FamilyStatus) for status in family_statuses):
            raise TypeError("family_statuses must contain FamilyStatus records")
        if tuple(status.family for status in family_statuses) != tuple(
            sorted(ALL_FAMILIES)
        ):
            raise ValueError(
                "family_statuses must contain every known family in sorted order"
            )
        object.__setattr__(self, "family_statuses", family_statuses)
        if self.fpga_telemetry or self.encoder_telemetry:
            raise ValueError(
                "layout-v4 telemetry persistence requires the reviewed typed "
                "telemetry boundary"
            )
        object.__setattr__(
            self,
            "fpga_telemetry",
            _freeze_array_mapping(self.fpga_telemetry, "fpga_telemetry"),
        )
        object.__setattr__(
            self,
            "encoder_telemetry",
            _freeze_array_mapping(self.encoder_telemetry, "encoder_telemetry"),
        )
        if not isinstance(self.interpolation_policy, InterpolationPolicy):
            raise TypeError("interpolation_policy must be an InterpolationPolicy")
        if type(self.overwrite) is not bool:
            raise TypeError("overwrite must be a boolean")
        if self.hdf5_compression not in (None, "gzip"):
            raise ValueError("hdf5_compression must be None or 'gzip'")
        if self.hdf5_compression is None:
            if self.hdf5_compression_level is not None:
                raise ValueError("hdf5_compression_level requires hdf5_compression")
        elif (
            type(self.hdf5_compression_level) is not int
            or not 0 <= self.hdf5_compression_level <= 9
        ):
            raise ValueError("gzip compression level must lie in [0, 9]")
        self.validate()

    def validate(self) -> None:
        """Revalidate mutable Products state before any output is opened."""
        products = self.products
        _validate_session_invariants(products)
        if (
            type(products.quality_status) is not DataQuality
            or products.quality_status
            not in (DataQuality.CLEAN, DataQuality.PARTIAL)
        ):
            raise ValueError(
                "layout-v4 output requires clean or partial product quality"
            )
        if products.quality_status is DataQuality.PARTIAL and not self.issues:
            raise ValueError("partial product quality requires a recorded issue")
        if products.decode_provenance.unavailable_reason is not None:
            raise ValueError("layout-v4 output requires concrete decoder provenance")
        for name in (
            "decoder_name",
            "distribution_version",
            "binding_key",
            "schema_variant",
            "binding_source_release",
        ):
            _optional_text(
                getattr(products.decode_provenance, name),
                f"decoder provenance {name}",
            )
        if products.issues != self.issues:
            raise ValueError("WriteRequest issues must exactly match Products.issues")
        if products.cal_data:
            raise ValueError(
                "legacy anonymous cal_data rows are unsupported in layout v4"
            )
        if self.fpga_telemetry or self.encoder_telemetry:
            raise ValueError(
                "layout-v4 telemetry persistence requires the reviewed typed "
                "telemetry boundary"
            )

        product_rows = []
        issue_ids = {issue.issue_id for issue in self.issues}
        for family, expected_type in FAMILY_TYPES:
            rows = getattr(products, family)
            if type(rows) is not list:
                raise TypeError(f"Products.{family} must be a list")
            for row in rows:
                if type(row) is not expected_type:
                    raise TypeError(
                        f"Products.{family} contains a legacy or invalid row"
                    )
                _validate_product_provenance_for_storage(row)
                if family in ("housekeeping", "calibrator_metadata"):
                    _validate_field_names(row.fields, context=family)
                elif family == "calibrator_debug":
                    for page in row.pages:
                        _validate_field_names(
                            page.fields,
                            context=f"calibrator_debug.page_{page.page}",
                        )
                missing_issue_ids = set(row.provenance.decoder_issue_ids) - issue_ids
                if missing_issue_ids:
                    raise ValueError(
                        f"Products.{family} provenance references unknown issues"
                    )
            if rows:
                product_rows.append((family, len(rows)))
            if family in ("housekeeping", "calibrator_metadata"):
                _validate_field_union(
                    [row.fields for row in rows],
                    context=family,
                )
            elif family == "calibrator_debug" and rows:
                for page_index in range(rows[0].page_count):
                    _validate_field_union(
                        [row.pages[page_index].fields for row in rows],
                        context=f"calibrator_debug.page_{page_index}",
                    )

        expected_rows = tuple(sorted(product_rows))
        if products.validated_counts.product_rows != expected_rows:
            raise ValueError(
                "validated product counts disagree with strict family rows"
            )
        if products.validated_counts.persisted_rows is not None:
            raise ValueError(
                "pre-write Products must not already record persisted rows"
            )
        if (
            products.validated_counts.input_packets
            != products.decode_provenance.input_packet_count
            or products.validated_counts.valid_packets
            != products.decode_provenance.valid_packet_count
        ):
            raise ValueError("validated packet counts disagree with decoder provenance")
        for name, count in (
            ("validated input packet count", products.validated_counts.input_packets),
            ("validated valid packet count", products.validated_counts.valid_packets),
            (
                "decoder input packet count",
                products.decode_provenance.input_packet_count,
            ),
            (
                "decoder valid packet count",
                products.decode_provenance.valid_packet_count,
            ),
        ):
            _optional_uint(count, 64, name)
        for code, _ in products.decode_provenance.issue_counts:
            if _optional_text(code, "decoder issue code") is None:
                raise ValueError("decoder issue code must be present")
        for _, count in (
            *products.decode_provenance.appid_counts,
            *products.decode_provenance.issue_counts,
        ):
            _optional_uint(count, 64, "decoder summary count")
        tr_geometry = {(row.navg2, row.tr_length) for row in products.tr_spectra}
        if len(tr_geometry) > 1:
            raise ValueError("one layout-v4 request cannot contain mixed TR geometries")
        _validate_clock_conversions(products, self.clock_reference_set)
        status_by_family = {status.family: status for status in self.family_statuses}
        for status in self.family_statuses:
            if set(status.issue_ids) - issue_ids:
                raise ValueError(
                    f"family status for {status.family} references unknown issues"
                )
        for family, _ in FAMILY_TYPES:
            status = status_by_family[family]
            family_rows = getattr(products, family)
            rows = len(family_rows)
            if not status.supported or status.decoded_rows != rows:
                raise ValueError(f"family status for {family} disagrees with Products")
            row_issue_ids = {
                issue_id
                for row in family_rows
                for issue_id in row.provenance.decoder_issue_ids
            }
            if row_issue_ids - set(status.issue_ids):
                raise ValueError(
                    f"family status for {family} omits row issue references"
                )
            if (
                products.quality_status is DataQuality.CLEAN
                and status.quality is not DataQuality.CLEAN
            ):
                raise ValueError(
                    "clean product quality cannot contain a degraded "
                    f"supported family ({family})"
                )
        for family in UNSUPPORTED_FAMILIES:
            if status_by_family[family].supported:
                raise ValueError(f"family {family} is not implemented in layout v4")


__all__ = [
    "ALL_FAMILIES",
    "FAMILY_TYPES",
    "FamilyCoverage",
    "FamilyStatus",
    "InterpolationPolicy",
    "LunarLocation",
    "RunProvenance",
    "WriteRequest",
    "family_statuses_for_products",
]
