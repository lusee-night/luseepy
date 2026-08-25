"""Public boundary for the optional private telemetry decoder."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from .clock_reference import ClockReferenceSet, ClockSource
from .issues import IngestIssue, IssueAction, IssueCollector, IssueSeverity

TELEMETRY_DECODER_API_VERSION = 1
LEGACY_TELEMETRY_SIDECAR_NAME = "DCB_telemetry.json"

BLOCK_KEYS = frozenset({
    "source_kind",
    "field_names",
    "input_indices",
    "mission_seconds",
    "lusee_subsecs",
    "raw_counts",
    "values",
    "valid",
})
DECODER_INFO_KEYS = frozenset({
    "api_version",
    "decoder_name",
    "decoder_version",
    "claimed_appids",
})
FIELD_METADATA_KEYS = frozenset({
    "unit",
    "kind",
    "interpolation",
    "display_group",
})
ISSUE_KEYS = frozenset({
    "code",
    "severity",
    "action",
    "message",
    "input_index",
    "appid",
    "sequence_count",
    "field",
    "details",
})
B01_RESULT_KEYS = frozenset({"fpga", "encoder", "issues", "counts"})
SIDECAR_RESULT_KEYS = frozenset({"fpga", "issues", "counts"})
B01_SCALAR_COUNT_KEYS = frozenset({
    "input_packet_count",
    "claimed_packet_count",
    "unclaimed_packet_count",
    "fpga_input_packet_count",
    "fpga_output_record_count",
    "fpga_dropped_packet_count",
    "encoder_input_packet_count",
    "encoder_output_record_count",
    "encoder_rejected_packet_count",
})
SIDECAR_SCALAR_COUNT_KEYS = frozenset({
    "input_byte_count",
    "complete_record_count",
    "trailing_byte_count",
    "output_record_count",
    "dropped_record_count",
})
FIELD_KINDS = frozenset({
    "continuous",
    "categorical",
    "bitmask",
    "uncalibrated_raw_count",
})
INTERPOLATION_KINDS = frozenset({"linear", "nearest", "hold", "none"})
IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
MAX_UINT64 = int(np.iinfo(np.uint64).max)
RESERVED_ENGINEERING_FIELD_NAMES = frozenset({
    "field_names",
    "input_indices",
    "lusee_subsecs",
    "mission_seconds",
    "mjd_time_valid",
    "mjd_times",
    "raw_counts",
    "raw_seconds",
    "session_index",
    "source_index",
    "source_kind",
    "valid",
    "values",
})


def validate_storage_text(
    value: object,
    name: str,
    *,
    allow_empty: bool,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a nonempty string"
        raise ValueError(f"{name} must be {qualifier}")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a null character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def safe_diagnostic_text(value: object) -> str:
    """Return storable text for an untrusted decoder exception."""
    if value is None:
        return "no diagnostic was supplied"
    try:
        text = str(value)
    except Exception:  # noqa: BLE001
        text = f"<{type(value).__name__} diagnostic unavailable>"
    text = text.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return text.replace("\x00", "\\x00") or "no diagnostic was supplied"


class TelemetryInputState(StrEnum):
    """Whether one recognized telemetry source exists."""

    ABSENT = "absent"
    PRESENT_EMPTY = "present_empty"
    PRESENT = "present"


class TelemetryDecoderStatus(StrEnum):
    """Availability and compatibility of the optional decoder."""

    NOT_NEEDED = "not_needed"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    BROKEN = "broken"
    INCOMPATIBLE = "incompatible"


class TelemetryCoverage(StrEnum):
    """Outcome for one telemetry input source."""

    ABSENT = "absent"
    PRESENT_EMPTY = "present_empty"
    DECODED = "decoded"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    BROKEN = "broken"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class TelemetryDecoderInfo:
    """Validated identity of one compatible private decoder."""

    api_version: int
    decoder_name: str
    decoder_version: str
    claimed_appids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.api_version) is not int:
            raise TypeError("telemetry decoder api_version must be an integer")
        if self.api_version != TELEMETRY_DECODER_API_VERSION:
            raise ValueError(
                "telemetry decoder API version must be "
                f"{TELEMETRY_DECODER_API_VERSION}"
            )
        for name, value in (
            ("decoder_name", self.decoder_name),
            ("decoder_version", self.decoder_version),
        ):
            validate_storage_text(value, f"telemetry {name}", allow_empty=False)
        appids = tuple(self.claimed_appids)
        if any(
            type(appid) is not int or not 0 <= appid <= 0x7FF
            for appid in appids
        ):
            raise ValueError("claimed telemetry AppIDs must be 11-bit integers")
        if len(set(appids)) != len(appids):
            raise ValueError("claimed telemetry AppIDs must be unique")
        object.__setattr__(self, "claimed_appids", appids)


@dataclass(frozen=True, slots=True)
class TelemetryFieldMetadata:
    """Public metadata for one decoder-supplied engineering field."""

    name: str
    unit: str
    kind: str
    interpolation: str
    display_group: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or IDENTIFIER_RE.fullmatch(self.name) is None:
            raise ValueError("telemetry field names must be identifiers")
        if self.name in RESERVED_ENGINEERING_FIELD_NAMES:
            raise ValueError(
                f"telemetry field name {self.name!r} is reserved for public arrays"
            )
        validate_storage_text(
            self.unit,
            "telemetry field unit",
            allow_empty=True,
        )
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"unknown telemetry field kind {self.kind!r}")
        if self.interpolation not in INTERPOLATION_KINDS:
            raise ValueError(
                f"unknown telemetry interpolation policy {self.interpolation!r}"
            )
        if self.kind != "continuous" and self.interpolation == "linear":
            raise ValueError(
                "linear telemetry interpolation requires a continuous field"
            )
        if self.display_group is not None and (
            not isinstance(self.display_group, str) or not self.display_group.strip()
        ):
            raise ValueError("telemetry display_group must be nonempty or None")
        if self.display_group is not None:
            validate_storage_text(
                self.display_group,
                "telemetry display_group",
                allow_empty=False,
            )


def immutable_exact_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    ndim: int,
    shape: tuple[int | None, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if len(shape) != ndim or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} has invalid shape {value.shape}")
    return np.frombuffer(value.tobytes(order="C"), dtype=expected_dtype).reshape(
        value.shape
    )


@dataclass(frozen=True, slots=True)
class TelemetryBlock:
    """Validated row-aligned telemetry values, raw counts, and masks."""

    source_kind: str
    field_names: tuple[str, ...]
    input_indices: np.ndarray
    mission_seconds: np.ndarray
    lusee_subsecs: np.ndarray
    raw_seconds: np.ndarray
    mjd_times: np.ndarray
    mjd_time_valid: np.ndarray
    raw_counts: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.source_kind not in (
            "b01_0x314",
            "b01_0x325",
            "legacy_binary_sidecar",
        ):
            raise ValueError(f"unknown telemetry source_kind {self.source_kind!r}")
        field_names = tuple(self.field_names)
        if any(
            not isinstance(name, str) or IDENTIFIER_RE.fullmatch(name) is None
            for name in field_names
        ):
            raise ValueError("telemetry field_names must contain identifiers")
        reserved_names = RESERVED_ENGINEERING_FIELD_NAMES.intersection(field_names)
        if reserved_names:
            raise ValueError(
                "telemetry field_names contain names reserved for public arrays: "
                + ", ".join(sorted(reserved_names))
            )
        if len(set(field_names)) != len(field_names):
            raise ValueError("telemetry field_names must be unique")
        if type(self.mission_seconds) is not np.ndarray or self.mission_seconds.ndim != 1:
            raise ValueError("mission_seconds must have one row axis")
        n_rows = self.mission_seconds.shape[0]
        n_fields = len(field_names)
        arrays = {
            "input_indices": immutable_exact_array(
                self.input_indices,
                name="telemetry input_indices",
                dtype=np.int64,
                ndim=1,
                shape=(n_rows,),
            ),
            "mission_seconds": immutable_exact_array(
                self.mission_seconds,
                name="telemetry mission_seconds",
                dtype=np.uint32,
                ndim=1,
                shape=(n_rows,),
            ),
            "lusee_subsecs": immutable_exact_array(
                self.lusee_subsecs,
                name="telemetry lusee_subsecs",
                dtype=np.uint16,
                ndim=1,
                shape=(n_rows,),
            ),
            "raw_seconds": immutable_exact_array(
                self.raw_seconds,
                name="telemetry raw_seconds",
                dtype=np.float64,
                ndim=1,
                shape=(n_rows,),
            ),
            "mjd_times": immutable_exact_array(
                self.mjd_times,
                name="telemetry mjd_times",
                dtype=np.float64,
                ndim=1,
                shape=(n_rows,),
            ),
            "mjd_time_valid": immutable_exact_array(
                self.mjd_time_valid,
                name="telemetry mjd_time_valid",
                dtype=np.bool_,
                ndim=1,
                shape=(n_rows,),
            ),
            "raw_counts": immutable_exact_array(
                self.raw_counts,
                name="telemetry raw_counts",
                dtype=np.uint16,
                ndim=2,
                shape=(n_rows, n_fields),
            ),
            "values": immutable_exact_array(
                self.values,
                name="telemetry values",
                dtype=np.float64,
                ndim=2,
                shape=(n_rows, n_fields),
            ),
            "valid": immutable_exact_array(
                self.valid,
                name="telemetry valid",
                dtype=np.bool_,
                ndim=2,
                shape=(n_rows, n_fields),
            ),
        }
        if np.any(arrays["input_indices"] < 0):
            raise ValueError("telemetry input_indices must be nonnegative")
        if n_rows > 1 and np.any(np.diff(arrays["input_indices"]) <= 0):
            raise ValueError("telemetry input_indices must be strictly increasing")
        if not np.array_equal(np.isfinite(arrays["values"]), arrays["valid"]):
            raise ValueError(
                "telemetry values must be finite exactly where valid is true"
            )
        if np.any(~arrays["valid"] & ~np.isnan(arrays["values"])):
            raise ValueError("invalid telemetry values must be NaN")
        if not np.array_equal(
            np.isfinite(arrays["mjd_times"]), arrays["mjd_time_valid"]
        ):
            raise ValueError(
                "telemetry MJD values must be finite exactly where its mask is true"
            )
        if np.any(
            ~arrays["mjd_time_valid"] & ~np.isnan(arrays["mjd_times"])
        ):
            raise ValueError("invalid telemetry MJD values must be NaN")
        ticks = (
            arrays["mission_seconds"].astype(np.uint64) * np.uint64(65536)
            + arrays["lusee_subsecs"].astype(np.uint64)
        )
        expected_raw_seconds = ticks.astype(np.float64) / 65536.0
        if not np.array_equal(arrays["raw_seconds"], expected_raw_seconds):
            raise ValueError(
                "telemetry raw_seconds disagrees with integer source columns"
            )
        object.__setattr__(self, "field_names", field_names)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @classmethod
    def from_decoder_record(
        cls,
        value: object,
        *,
        expected_source: str,
    ) -> TelemetryBlock:
        """Validate one exact detailed-API field block."""
        record = exact_mapping(value, "telemetry field block", BLOCK_KEYS)
        if record["source_kind"] != expected_source:
            raise ValueError(
                "telemetry field block has unexpected source_kind "
                f"{record['source_kind']!r}"
            )
        mission_seconds = record["mission_seconds"]
        lusee_subsecs = record["lusee_subsecs"]
        if type(mission_seconds) is not np.ndarray or type(lusee_subsecs) is not np.ndarray:
            raise TypeError("telemetry time columns must be numpy.ndarray values")
        if mission_seconds.dtype != np.dtype(np.uint32) or mission_seconds.ndim != 1:
            raise TypeError("telemetry mission_seconds must be a uint32 vector")
        if lusee_subsecs.dtype != np.dtype(np.uint16) or lusee_subsecs.ndim != 1:
            raise TypeError("telemetry lusee_subsecs must be a uint16 vector")
        if mission_seconds.shape != lusee_subsecs.shape:
            raise ValueError("telemetry time columns must have equal row counts")
        ticks = (
            mission_seconds.astype(np.uint64) * np.uint64(65536)
            + lusee_subsecs.astype(np.uint64)
        )
        raw_seconds = ticks.astype(np.float64) / 65536.0
        n_rows = mission_seconds.shape[0]
        return cls(
            source_kind=record["source_kind"],
            field_names=record["field_names"],
            input_indices=record["input_indices"],
            mission_seconds=mission_seconds,
            lusee_subsecs=lusee_subsecs,
            raw_seconds=raw_seconds,
            mjd_times=np.full(n_rows, np.nan, dtype=np.float64),
            mjd_time_valid=np.zeros(n_rows, dtype=np.bool_),
            raw_counts=record["raw_counts"],
            values=record["values"],
            valid=record["valid"],
        )

    @property
    def row_count(self) -> int:
        return self.mission_seconds.shape[0]

    def slice_rows(self, selector: np.ndarray) -> TelemetryBlock:
        """Return an immutable row slice using one validated selector."""
        if type(selector) is not np.ndarray or selector.dtype != np.dtype(np.bool_):
            raise TypeError("telemetry row selector must be a boolean ndarray")
        if selector.shape != (self.row_count,):
            raise ValueError("telemetry row selector has the wrong shape")
        return TelemetryBlock(
            source_kind=self.source_kind,
            field_names=self.field_names,
            input_indices=self.input_indices[selector],
            mission_seconds=self.mission_seconds[selector],
            lusee_subsecs=self.lusee_subsecs[selector],
            raw_seconds=self.raw_seconds[selector],
            mjd_times=self.mjd_times[selector],
            mjd_time_valid=self.mjd_time_valid[selector],
            raw_counts=self.raw_counts[selector],
            values=self.values[selector],
            valid=self.valid[selector],
        )

    def with_mjd_times(self, mjd_times: np.ndarray) -> TelemetryBlock:
        """Attach DCB-clock absolute times without changing raw time."""
        if type(mjd_times) is not np.ndarray or mjd_times.dtype != np.dtype(np.float64):
            raise TypeError("telemetry mjd_times must be a float64 ndarray")
        return TelemetryBlock(
            source_kind=self.source_kind,
            field_names=self.field_names,
            input_indices=self.input_indices,
            mission_seconds=self.mission_seconds,
            lusee_subsecs=self.lusee_subsecs,
            raw_seconds=self.raw_seconds,
            mjd_times=mjd_times,
            mjd_time_valid=np.isfinite(mjd_times),
            raw_counts=self.raw_counts,
            values=self.values,
            valid=self.valid,
        )


@dataclass(frozen=True, slots=True)
class TelemetryCounts:
    """Validated decoder counts with optional AppID breakdowns."""

    source: str
    scalar_counts: tuple[tuple[str, int], ...]
    claimed_appid_counts: tuple[tuple[int, int], ...] = ()
    unclaimed_appid_counts: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.source not in ("b01", "legacy_sidecar"):
            raise ValueError("unknown telemetry count source")
        scalar_counts = tuple(self.scalar_counts)
        expected = (
            B01_SCALAR_COUNT_KEYS
            if self.source == "b01"
            else SIDECAR_SCALAR_COUNT_KEYS
        )
        if frozenset(name for name, _ in scalar_counts) != expected:
            raise ValueError("telemetry scalar count keys violate the API contract")
        if len(scalar_counts) != len(expected):
            raise ValueError("telemetry scalar counts contain duplicate keys")
        if any(
            type(count) is not int or not 0 <= count <= MAX_UINT64
            for _, count in scalar_counts
        ):
            raise ValueError("telemetry counts must be uint64-range integers")
        claimed = validate_appid_count_pairs(
            self.claimed_appid_counts, "claimed_appid_counts"
        )
        unclaimed = validate_appid_count_pairs(
            self.unclaimed_appid_counts, "unclaimed_appid_counts"
        )
        if self.source == "legacy_sidecar" and (claimed or unclaimed):
            raise ValueError("legacy-sidecar counts cannot contain AppID maps")
        scalar = dict(scalar_counts)
        if self.source == "b01":
            if (
                scalar["claimed_packet_count"]
                + scalar["unclaimed_packet_count"]
                != scalar["input_packet_count"]
                or sum(count for _, count in claimed)
                != scalar["claimed_packet_count"]
                or sum(count for _, count in unclaimed)
                != scalar["unclaimed_packet_count"]
                or scalar["fpga_input_packet_count"]
                + scalar["encoder_input_packet_count"]
                != scalar["claimed_packet_count"]
                or scalar["fpga_output_record_count"]
                + scalar["fpga_dropped_packet_count"]
                != scalar["fpga_input_packet_count"]
                or scalar["encoder_output_record_count"]
                + scalar["encoder_rejected_packet_count"]
                != scalar["encoder_input_packet_count"]
            ):
                raise ValueError("b01 telemetry count equations disagree")
        else:
            input_bytes = scalar["input_byte_count"]
            complete = scalar["complete_record_count"]
            trailing = scalar["trailing_byte_count"]
            if complete == 0:
                bytes_accounted = trailing == input_bytes
            else:
                record_bytes, remainder = divmod(
                    input_bytes - trailing,
                    complete,
                ) if trailing < input_bytes else (0, 1)
                bytes_accounted = remainder == 0 and trailing < record_bytes
            if (
                scalar["output_record_count"]
                + scalar["dropped_record_count"]
                != complete
                or not bytes_accounted
            ):
                raise ValueError("legacy-sidecar telemetry count equations disagree")
        object.__setattr__(self, "scalar_counts", tuple(sorted(scalar_counts)))
        object.__setattr__(self, "claimed_appid_counts", claimed)
        object.__setattr__(self, "unclaimed_appid_counts", unclaimed)

    def scalar(self, name: str) -> int:
        for key, value in self.scalar_counts:
            if key == name:
                return value
        raise KeyError(name)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = dict(self.scalar_counts)
        if self.source == "b01":
            result["claimed_appid_counts"] = dict(self.claimed_appid_counts)
            result["unclaimed_appid_counts"] = dict(self.unclaimed_appid_counts)
        return result


@dataclass(frozen=True, slots=True)
class TelemetryDecodeResult:
    """Complete public result for one recognized telemetry source."""

    input_source: str | None
    input_state: TelemetryInputState | str
    decoder_status: TelemetryDecoderStatus | str
    coverage: TelemetryCoverage | str
    decoder_info: TelemetryDecoderInfo | None = None
    field_metadata: tuple[TelemetryFieldMetadata, ...] = ()
    fpga: TelemetryBlock | None = None
    unassigned_fpga: TelemetryBlock | None = None
    encoder: TelemetryBlock | None = None
    counts: TelemetryCounts | None = None
    counts_scope: str = "full_input"
    issues: tuple[IngestIssue, ...] = ()

    def __post_init__(self) -> None:
        input_state = TelemetryInputState(self.input_state)
        decoder_status = TelemetryDecoderStatus(self.decoder_status)
        coverage = TelemetryCoverage(self.coverage)
        if self.input_source not in (None, "b01", "legacy_sidecar"):
            raise ValueError("unknown telemetry input source")
        if self.counts_scope != "full_input":
            raise ValueError("telemetry counts_scope must be 'full_input'")
        metadata = tuple(self.field_metadata)
        issues = tuple(self.issues)
        if any(not isinstance(item, TelemetryFieldMetadata) for item in metadata):
            raise TypeError("field_metadata must contain TelemetryFieldMetadata")
        if any(not isinstance(item, IngestIssue) for item in issues):
            raise TypeError("telemetry issues must contain IngestIssue records")
        if len({issue.issue_id for issue in issues}) != len(issues):
            raise ValueError("telemetry issues must have unique IDs")
        if input_state is TelemetryInputState.ABSENT:
            if self.input_source is not None:
                raise ValueError("absent telemetry cannot name an input source")
            if decoder_status is not TelemetryDecoderStatus.NOT_NEEDED:
                raise ValueError("absent telemetry must not resolve the decoder")
            if coverage is not TelemetryCoverage.ABSENT:
                raise ValueError("absent telemetry must have absent coverage")
            if any((
                self.decoder_info,
                metadata,
                self.fpga,
                self.unassigned_fpga,
                self.encoder,
                self.counts,
                issues,
            )):
                raise ValueError("absent telemetry cannot contain decoded state")
        else:
            if self.input_source is None:
                raise ValueError("present telemetry must name an input source")
            if decoder_status is TelemetryDecoderStatus.NOT_NEEDED:
                raise ValueError("present telemetry must resolve the decoder")
        if decoder_status is TelemetryDecoderStatus.AVAILABLE:
            if self.decoder_info is None or self.counts is None or self.fpga is None:
                raise ValueError("available telemetry requires decoder data")
            if not metadata:
                raise ValueError("available telemetry requires field metadata")
            if tuple(item.name for item in metadata) != self.fpga.field_names:
                raise ValueError("telemetry metadata order must match FPGA fields")
            if self.unassigned_fpga is not None and (
                self.unassigned_fpga.field_names != self.fpga.field_names
                or self.unassigned_fpga.source_kind != self.fpga.source_kind
            ):
                raise ValueError(
                    "unassigned telemetry must use the FPGA field contract"
                )
            if self.input_source == "b01" and self.encoder is None:
                raise ValueError("b01 telemetry requires an encoder result block")
            if self.input_source == "legacy_sidecar" and self.encoder is not None:
                raise ValueError("legacy sidecar cannot contain an encoder block")
            if (
                self.input_source == "legacy_sidecar"
                and self.unassigned_fpga is not None
            ):
                raise ValueError(
                    "session-scoped legacy sidecar cannot contain unassigned rows"
                )
            expected_count_source = (
                "b01" if self.input_source == "b01" else "legacy_sidecar"
            )
            if self.counts.source != expected_count_source:
                raise ValueError("telemetry counts use the wrong input source")
            expected_fpga_source = (
                "b01_0x314"
                if self.input_source == "b01"
                else "legacy_binary_sidecar"
            )
            if self.fpga.source_kind != expected_fpga_source:
                raise ValueError("FPGA telemetry uses the wrong source kind")
            if (
                self.encoder is not None
                and self.encoder.source_kind != "b01_0x325"
            ):
                raise ValueError("encoder telemetry uses the wrong source kind")
            if self.encoder is not None and (
                self.encoder.field_names or self.encoder.row_count
            ):
                raise ValueError(
                    "encoder measurements must remain empty until validated"
                )
            input_count = self.counts.scalar(
                "input_packet_count"
                if self.input_source == "b01"
                else "input_byte_count"
            )
            if (input_count == 0) != (
                input_state is TelemetryInputState.PRESENT_EMPTY
            ):
                raise ValueError("telemetry input state disagrees with input count")
            output_count = self.counts.scalar(
                "fpga_output_record_count"
                if self.input_source == "b01"
                else "output_record_count"
            )
            stored_indices = self.fpga.input_indices
            if self.unassigned_fpga is not None:
                if np.intersect1d(
                    stored_indices,
                    self.unassigned_fpga.input_indices,
                    assume_unique=True,
                ).size:
                    raise ValueError(
                        "assigned and unassigned telemetry rows overlap"
                    )
                stored_indices = np.concatenate((
                    stored_indices,
                    self.unassigned_fpga.input_indices,
                ))
            if stored_indices.size > output_count:
                raise ValueError(
                    "stored telemetry rows exceed the full-input output count"
                )
            if (
                self.input_source == "legacy_sidecar"
                and stored_indices.size != output_count
            ):
                raise ValueError(
                    "stored sidecar rows must equal the decoder output count"
                )
            input_index_limit = (
                input_count
                if self.input_source == "b01"
                else self.counts.scalar("complete_record_count")
            )
            if stored_indices.size and np.any(
                stored_indices >= input_index_limit
            ):
                raise ValueError(
                    "stored telemetry input indices exceed the full input"
                )
            if self.input_source == "b01":
                assert self.encoder is not None
                encoder_output = self.counts.scalar(
                    "encoder_output_record_count"
                )
                if self.encoder.row_count != encoder_output:
                    raise ValueError(
                        "stored encoder rows disagree with the output count"
                    )
                if self.encoder.row_count and np.any(
                    self.encoder.input_indices >= input_count
                ):
                    raise ValueError(
                        "stored encoder input indices exceed the full input"
                    )
                claimed_counts = dict(self.counts.claimed_appid_counts)
                unclaimed_appids = {
                    appid for appid, _ in self.counts.unclaimed_appid_counts
                }
                if (
                    claimed_counts.get(0x314, 0)
                    != self.counts.scalar("fpga_input_packet_count")
                    or claimed_counts.get(0x325, 0)
                    != self.counts.scalar("encoder_input_packet_count")
                    or not set(claimed_counts).issubset(
                        self.decoder_info.claimed_appids
                    )
                    or unclaimed_appids.intersection(
                        self.decoder_info.claimed_appids
                    )
                ):
                    raise ValueError(
                        "b01 claimed AppID counts disagree with stored roles"
                    )
            if input_state is TelemetryInputState.PRESENT_EMPTY and (
                stored_indices.size
                or (self.encoder is not None and self.encoder.row_count)
            ):
                raise ValueError("present-empty telemetry cannot contain rows")
            loss_counts = (
                (
                    self.counts.scalar("fpga_dropped_packet_count"),
                    self.counts.scalar("encoder_rejected_packet_count"),
                )
                if self.input_source == "b01"
                else (
                    self.counts.scalar("trailing_byte_count"),
                    self.counts.scalar("dropped_record_count"),
                )
            )
            degraded = bool(issues) or any(loss_counts)
            if coverage not in (
                TelemetryCoverage.PRESENT_EMPTY,
                TelemetryCoverage.DECODED,
                TelemetryCoverage.PARTIAL,
            ):
                raise ValueError("available decoder has invalid telemetry coverage")
            if coverage is TelemetryCoverage.PRESENT_EMPTY and not (
                input_state is TelemetryInputState.PRESENT_EMPTY
                and not degraded
            ):
                raise ValueError("present-empty telemetry coverage is inconsistent")
            if coverage is TelemetryCoverage.DECODED and not (
                input_state is TelemetryInputState.PRESENT and not degraded
            ):
                raise ValueError("decoded telemetry coverage is inconsistent")
            if coverage is TelemetryCoverage.PARTIAL and not issues:
                raise ValueError("partial telemetry coverage requires an issue")
            if self.unassigned_fpga is not None and (
                self.unassigned_fpga.row_count
                and coverage is not TelemetryCoverage.PARTIAL
            ):
                raise ValueError("unassigned telemetry requires partial coverage")
        elif decoder_status is not TelemetryDecoderStatus.NOT_NEEDED:
            if any((
                self.fpga,
                self.unassigned_fpga,
                self.encoder,
                self.counts,
            )):
                raise ValueError("failed decoder state cannot contain decoded data")
            if metadata and self.decoder_info is None:
                raise ValueError(
                    "telemetry field metadata requires decoder identity"
                )
            if (
                decoder_status is TelemetryDecoderStatus.UNAVAILABLE
                and (self.decoder_info is not None or metadata)
            ):
                raise ValueError(
                    "unavailable telemetry cannot contain decoder identity"
                )
            if coverage is not TelemetryCoverage(decoder_status.value):
                raise ValueError("decoder failure and telemetry coverage disagree")
            if not issues:
                raise ValueError("decoder failure must record an issue")
        object.__setattr__(self, "input_state", input_state)
        object.__setattr__(self, "decoder_status", decoder_status)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "field_metadata", metadata)
        object.__setattr__(self, "issues", issues)

    @classmethod
    def absent(cls) -> TelemetryDecodeResult:
        return cls(
            input_source=None,
            input_state=TelemetryInputState.ABSENT,
            decoder_status=TelemetryDecoderStatus.NOT_NEEDED,
            coverage=TelemetryCoverage.ABSENT,
        )

    @property
    def issue_ids(self) -> tuple[str, ...]:
        return tuple(issue.issue_id for issue in self.issues)

    @property
    def has_rows(self) -> bool:
        return self.fpga is not None and self.fpga.row_count > 0

    def with_blocks(
        self,
        *,
        fpga: TelemetryBlock,
        unassigned_fpga: TelemetryBlock | None = None,
        encoder: TelemetryBlock | None = None,
        issues: tuple[IngestIssue, ...] | None = None,
        coverage: TelemetryCoverage | str | None = None,
    ) -> TelemetryDecodeResult:
        """Return this result with row-sliced or time-mapped blocks."""
        if self.decoder_status is not TelemetryDecoderStatus.AVAILABLE:
            raise ValueError("cannot replace unavailable telemetry blocks")
        return TelemetryDecodeResult(
            input_source=self.input_source,
            input_state=self.input_state,
            decoder_status=self.decoder_status,
            decoder_info=self.decoder_info,
            field_metadata=self.field_metadata,
            fpga=fpga,
            unassigned_fpga=unassigned_fpga,
            encoder=self.encoder if encoder is None else encoder,
            counts=self.counts,
            counts_scope=self.counts_scope,
            issues=self.issues if issues is None else issues,
            coverage=self.coverage if coverage is None else coverage,
        )

    def with_issues(
        self,
        issues: tuple[IngestIssue, ...],
    ) -> TelemetryDecodeResult:
        """Return this result with a complete source-scoped issue tuple."""
        issue_records = tuple(issues)
        coverage = self.coverage
        if (
            self.decoder_status is TelemetryDecoderStatus.AVAILABLE
            and issue_records
        ):
            coverage = TelemetryCoverage.PARTIAL
        return TelemetryDecodeResult(
            input_source=self.input_source,
            input_state=self.input_state,
            decoder_status=self.decoder_status,
            coverage=coverage,
            decoder_info=self.decoder_info,
            field_metadata=self.field_metadata,
            fpga=self.fpga,
            unassigned_fpga=self.unassigned_fpga,
            encoder=self.encoder,
            counts=self.counts,
            counts_scope=self.counts_scope,
            issues=issue_records,
        )


class TelemetryCollection:
    """Ordered multi-session view of assigned FPGA telemetry rows.

    Decoder counts, issues, encoder state, and unassigned rows remain on the
    individual ``sessions`` because those records are input-scoped. Source-local
    ``input_indices`` are disambiguated by the row-aligned ``session_index``.
    """

    __slots__ = (
        "decoder_info",
        "field_metadata",
        "field_names",
        "input_indices",
        "lusee_subsecs",
        "mission_seconds",
        "mjd_time_valid",
        "mjd_times",
        "raw_counts",
        "raw_seconds",
        "session_boundaries",
        "session_index",
        "session_input_identities",
        "session_row_counts",
        "session_sources",
        "session_unassigned_row_counts",
        "sessions",
        "source_kind",
        "unassigned_fpga_blocks",
        "valid",
        "values",
    )

    def __init__(
        self,
        sessions: Sequence[TelemetryDecodeResult],
        *,
        session_sources: Sequence[Path | None],
        session_input_identities: Sequence[tuple[str, str] | None] | None = None,
    ) -> None:
        session_results = tuple(sessions)
        sources = tuple(session_sources)
        if len(session_results) != len(sources):
            raise ValueError(
                "telemetry sessions and source paths must have equal length"
            )
        if any(
            not isinstance(result, TelemetryDecodeResult)
            for result in session_results
        ):
            raise TypeError(
                "telemetry sessions must contain TelemetryDecodeResult values"
            )
        if any(source is not None and not isinstance(source, Path) for source in sources):
            raise TypeError("telemetry session sources must be pathlib.Path or None")
        identities = (
            (None,) * len(session_results)
            if session_input_identities is None
            else tuple(session_input_identities)
        )
        if len(identities) != len(session_results):
            raise ValueError(
                "telemetry sessions and input identities must have equal length"
            )
        if any(
            identity is not None
            and (
                type(identity) is not tuple
                or len(identity) != 2
                or any(not isinstance(value, str) or not value for value in identity)
            )
            for identity in identities
        ):
            raise TypeError(
                "telemetry input identities must be nonempty string pairs or None"
            )

        available = tuple(
            result
            for result in session_results
            if result.decoder_status is TelemetryDecoderStatus.AVAILABLE
        )
        decoder_info = available[0].decoder_info if available else None
        metadata = available[0].field_metadata if available else ()
        if any(result.decoder_info != decoder_info for result in available[1:]):
            raise ValueError(
                "cannot concatenate telemetry decoded by different decoders"
            )
        if any(result.field_metadata != metadata for result in available[1:]):
            raise ValueError(
                "cannot concatenate incompatible telemetry field contracts"
            )

        blocks = tuple(
            result.fpga
            for result in session_results
            if result.decoder_status is TelemetryDecoderStatus.AVAILABLE
            and result.fpga is not None
        )
        field_names = tuple(item.name for item in metadata)
        n_fields = len(field_names)
        row_counts = tuple(
            result.fpga.row_count
            if result.decoder_status is TelemetryDecoderStatus.AVAILABLE
            and result.fpga is not None
            else 0
            for result in session_results
        )
        boundaries = []
        offset = 0
        for count in row_counts:
            boundaries.append((offset, offset + count))
            offset += count

        def concatenate(
            name: str,
            *,
            dtype: np.dtype[Any] | type[np.generic],
            tail_shape: tuple[int, ...] = (),
        ) -> np.ndarray:
            if blocks:
                value = np.concatenate(
                    [getattr(block, name) for block in blocks], axis=0
                )
            else:
                value = np.empty((0, *tail_shape), dtype=dtype)
            if value.dtype != np.dtype(dtype):
                raise TypeError(
                    f"concatenated telemetry {name} has unexpected dtype"
                )
            value.setflags(write=False)
            return value

        source_index = np.concatenate([
            np.full(count, index, dtype=np.int64)
            for index, count in enumerate(row_counts)
        ]) if any(row_counts) else np.empty(0, dtype=np.int64)
        source_index.setflags(write=False)

        self.sessions = session_results
        self.session_sources = sources
        self.session_input_identities = identities
        self.decoder_info = decoder_info
        self.field_metadata = metadata
        self.field_names = field_names
        self.session_row_counts = row_counts
        self.session_boundaries = tuple(boundaries)
        self.session_unassigned_row_counts = tuple(
            result.unassigned_fpga.row_count
            if result.unassigned_fpga is not None
            else 0
            for result in session_results
        )
        self.unassigned_fpga_blocks = tuple(
            result.unassigned_fpga for result in session_results
        )
        self.session_index = source_index
        self.source_kind = tuple(
            block.source_kind
            for block in blocks
            for _ in range(block.row_count)
        )
        self.input_indices = concatenate("input_indices", dtype=np.int64)
        self.mission_seconds = concatenate("mission_seconds", dtype=np.uint32)
        self.lusee_subsecs = concatenate("lusee_subsecs", dtype=np.uint16)
        self.raw_seconds = concatenate("raw_seconds", dtype=np.float64)
        self.mjd_times = concatenate("mjd_times", dtype=np.float64)
        self.mjd_time_valid = concatenate("mjd_time_valid", dtype=np.bool_)
        self.raw_counts = concatenate(
            "raw_counts", dtype=np.uint16, tail_shape=(n_fields,)
        )
        self.values = concatenate(
            "values", dtype=np.float64, tail_shape=(n_fields,)
        )
        self.valid = concatenate(
            "valid", dtype=np.bool_, tail_shape=(n_fields,)
        )

    @property
    def row_count(self) -> int:
        """Number of assigned FPGA rows in the concatenated view."""
        return self.mission_seconds.size

    def engineering_values(self) -> dict[str, np.ndarray]:
        """Return the legacy-style engineering-value view for plotting."""
        values = {
            "mission_seconds": self.mission_seconds,
            "lusee_subsecs": self.lusee_subsecs,
            "raw_seconds": self.raw_seconds,
        }
        values.update({
            name: self.values[:, index]
            for index, name in enumerate(self.field_names)
        })
        return values

    def unassigned_engineering_values(
        self,
        *,
        session_index: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return input-scoped rows, deduplicated only by stable identity."""
        if session_index is not None and (
            type(session_index) is not int
            or not 0 <= session_index < len(self.sessions)
        ):
            raise IndexError("telemetry session_index is out of range")
        entries = [
            (index, self.session_input_identities[index], block)
            for index, block in enumerate(self.unassigned_fpga_blocks)
            if block is not None
            and block.row_count
            and (session_index is None or index == session_index)
        ]
        if not entries:
            return {}

        def blocks_equal(left: TelemetryBlock, right: TelemetryBlock) -> bool:
            if (
                left.source_kind != right.source_kind
                or left.field_names != right.field_names
            ):
                return False
            for name in (
                "input_indices",
                "mission_seconds",
                "lusee_subsecs",
                "raw_seconds",
                "mjd_times",
                "mjd_time_valid",
                "raw_counts",
                "values",
                "valid",
            ):
                left_value = getattr(left, name)
                right_value = getattr(right, name)
                equal_nan = left_value.dtype.kind == "f"
                if not np.array_equal(
                    left_value,
                    right_value,
                    equal_nan=equal_nan,
                ):
                    return False
            return True

        if session_index is None and len(entries) > 1:
            if any(identity is None for _, identity, _ in entries):
                raise ValueError(
                    "multiple session-scoped unassigned telemetry blocks require "
                    "session_index because their input identity is unavailable"
                )
            unique_entries = []
            for entry in entries:
                _, identity, block = entry
                matches = [
                    previous
                    for previous in unique_entries
                    if previous[1] == identity
                ]
                if not matches:
                    unique_entries.append(entry)
                elif any(blocks_equal(block, previous[2]) for previous in matches):
                    continue
                else:
                    raise ValueError(
                        "one telemetry input identity has inconsistent unassigned "
                        "blocks; select a session_index"
                    )
            entries = unique_entries

        blocks = tuple(entry[2] for entry in entries)

        def concatenate(name: str) -> np.ndarray:
            return np.concatenate(
                [getattr(block, name) for block in blocks], axis=0
            )

        values = {
            "mission_seconds": concatenate("mission_seconds"),
            "lusee_subsecs": concatenate("lusee_subsecs"),
            "raw_seconds": concatenate("raw_seconds"),
            "session_index": np.concatenate([
                np.full(block.row_count, -1, dtype=np.int64)
                for _, _, block in entries
            ]),
            "source_index": np.concatenate([
                np.full(block.row_count, index, dtype=np.int64)
                for index, _, block in entries
            ]),
        }
        engineering = concatenate("values")
        values.update({
            name: engineering[:, index]
            for index, name in enumerate(self.field_names)
        })
        return values


class IncompatibleDecoder(ValueError):
    """A present decoder or result violates detailed API v1."""


def exact_mapping(
    value: object,
    name: str,
    keys: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IncompatibleDecoder(f"{name} must be a mapping")
    if frozenset(value) != keys or len(value) != len(keys):
        raise IncompatibleDecoder(f"{name} has incompatible keys")
    return value


def validate_appid_count_pairs(
    value: Iterable[tuple[int, int]],
    name: str,
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(value)
    try:
        valid = all(
            type(appid) is int
            and 0 <= appid <= 0x7FF
            and type(count) is int
            and 0 <= count <= MAX_UINT64
            for appid, count in pairs
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain AppID/count pairs") from exc
    if not valid:
        raise ValueError(f"{name} must map AppIDs to uint64-range integers")
    if len({appid for appid, _ in pairs}) != len(pairs):
        raise ValueError(f"{name} contains duplicate AppIDs")
    return tuple(sorted(pairs))


def parse_decoder_info(value: object) -> TelemetryDecoderInfo:
    record = exact_mapping(value, "decoder_info", DECODER_INFO_KEYS)
    claimed_appids = record["claimed_appids"]
    if type(claimed_appids) is not tuple:
        raise IncompatibleDecoder("decoder_info.claimed_appids must be a tuple")
    try:
        return TelemetryDecoderInfo(
            api_version=record["api_version"],
            decoder_name=record["decoder_name"],
            decoder_version=record["decoder_version"],
            claimed_appids=claimed_appids,
        )
    except (TypeError, ValueError) as exc:
        raise IncompatibleDecoder(safe_diagnostic_text(exc)) from exc


def parse_field_metadata(value: object) -> tuple[TelemetryFieldMetadata, ...]:
    if not isinstance(value, Mapping):
        raise IncompatibleDecoder("field_metadata must be a mapping")
    fields = []
    try:
        for name, raw_metadata in value.items():
            metadata = exact_mapping(
                raw_metadata,
                f"field_metadata.{name}",
                FIELD_METADATA_KEYS,
            )
            fields.append(TelemetryFieldMetadata(name=name, **metadata))
    except (TypeError, ValueError) as exc:
        raise IncompatibleDecoder(safe_diagnostic_text(exc)) from exc
    if not fields:
        raise IncompatibleDecoder("field_metadata must not be empty")
    return tuple(fields)


def load_decoder(
    decoder: object | None,
) -> tuple[object | None, TelemetryDecoderStatus, str | None]:
    if decoder is not None:
        return decoder, TelemetryDecoderStatus.AVAILABLE, None
    try:
        return (
            importlib.import_module("lusee_telemetry"),
            TelemetryDecoderStatus.AVAILABLE,
            None,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "lusee_telemetry":
            return None, TelemetryDecoderStatus.UNAVAILABLE, safe_diagnostic_text(exc)
        return None, TelemetryDecoderStatus.BROKEN, safe_diagnostic_text(exc)
    except Exception as exc:  # noqa: BLE001
        return None, TelemetryDecoderStatus.BROKEN, safe_diagnostic_text(exc)


def resolve_decoder(
    decoder: object | None,
) -> tuple[
    object | None,
    TelemetryDecoderStatus,
    str | None,
    TelemetryDecoderInfo | None,
    tuple[TelemetryFieldMetadata, ...],
]:
    loaded, status, error = load_decoder(decoder)
    if loaded is None:
        return None, status, error, None, ()
    if not callable(getattr(loaded, "decoder_info", None)):
        return (
            None,
            TelemetryDecoderStatus.INCOMPATIBLE,
            "decoder is missing callable decoder_info()",
            None,
            (),
        )
    try:
        raw_info = loaded.decoder_info()
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            TelemetryDecoderStatus.BROKEN,
            safe_diagnostic_text(exc),
            None,
            (),
        )
    try:
        info = parse_decoder_info(raw_info)
    except IncompatibleDecoder as exc:
        return (
            None,
            TelemetryDecoderStatus.INCOMPATIBLE,
            safe_diagnostic_text(exc),
            None,
            (),
        )
    if not callable(getattr(loaded, "field_metadata", None)):
        return (
            None,
            TelemetryDecoderStatus.INCOMPATIBLE,
            "decoder is missing callable field_metadata()",
            info,
            (),
        )
    try:
        raw_metadata = loaded.field_metadata()
    except Exception as exc:  # noqa: BLE001
        return (
            None,
            TelemetryDecoderStatus.BROKEN,
            safe_diagnostic_text(exc),
            info,
            (),
        )
    try:
        metadata = parse_field_metadata(raw_metadata)
    except IncompatibleDecoder as exc:
        return (
            None,
            TelemetryDecoderStatus.INCOMPATIBLE,
            safe_diagnostic_text(exc),
            info,
            (),
        )
    for name in ("decode_b01_packets", "decode_legacy_sidecar"):
        if not callable(getattr(loaded, name, None)):
            return (
                None,
                TelemetryDecoderStatus.INCOMPATIBLE,
                f"decoder is missing callable {name}()",
                info,
                metadata,
            )
    return loaded, TelemetryDecoderStatus.AVAILABLE, None, info, metadata


def record_boundary_failure(
    *,
    issue_collector: IssueCollector,
    status: TelemetryDecoderStatus,
    input_source: str,
    error: str | None,
) -> IngestIssue:
    diagnostic = safe_diagnostic_text(error)
    return issue_collector.record(
        code=f"telemetry_adapter.decoder_{status.value}",
        severity=IssueSeverity.ERROR,
        stage="telemetry_decode",
        message=(
            f"telemetry input {input_source!r} was retained but its decoder is "
            f"{status.value}: {diagnostic}"
        ),
        action=IssueAction.KEPT,
        details={"decoder_status": status.value, "error": diagnostic},
    )


def failed_result(
    *,
    input_source: str,
    input_state: TelemetryInputState,
    status: TelemetryDecoderStatus,
    error: str | None,
    issue_collector: IssueCollector,
    marker: int,
    decoder_info: TelemetryDecoderInfo | None = None,
    field_metadata: tuple[TelemetryFieldMetadata, ...] = (),
) -> TelemetryDecodeResult:
    record_boundary_failure(
        issue_collector=issue_collector,
        status=status,
        input_source=input_source,
        error=error,
    )
    return TelemetryDecodeResult(
        input_source=input_source,
        input_state=input_state,
        decoder_status=status,
        coverage=TelemetryCoverage(status.value),
        decoder_info=decoder_info,
        field_metadata=field_metadata,
        issues=issue_collector.since(marker),
    )


def failed_input_result(
    *,
    input_source: str,
    error: str,
    issue_collector: IssueCollector,
    marker: int,
) -> TelemetryDecodeResult:
    diagnostic = safe_diagnostic_text(error)
    issue_collector.record(
        code=f"telemetry_input.{input_source}_unreadable",
        severity=IssueSeverity.ERROR,
        stage="telemetry_input",
        message=(
            f"recognized telemetry input {input_source!r} could not be read; "
            "independent science data was retained"
        ),
        action=IssueAction.REJECTED,
        details={"error": diagnostic},
    )
    return TelemetryDecodeResult(
        input_source=input_source,
        input_state=TelemetryInputState.PRESENT,
        decoder_status=TelemetryDecoderStatus.BROKEN,
        coverage=TelemetryCoverage.BROKEN,
        issues=issue_collector.since(marker),
    )


def parse_counts(value: object, *, source: str) -> TelemetryCounts:
    if not isinstance(value, Mapping):
        raise IncompatibleDecoder("telemetry counts must be a mapping")
    if source == "b01":
        expected = B01_SCALAR_COUNT_KEYS | {
            "claimed_appid_counts",
            "unclaimed_appid_counts",
        }
    else:
        expected = SIDECAR_SCALAR_COUNT_KEYS
    record = exact_mapping(value, "telemetry counts", frozenset(expected))
    scalar_keys = (
        B01_SCALAR_COUNT_KEYS
        if source == "b01"
        else SIDECAR_SCALAR_COUNT_KEYS
    )
    scalars = []
    for key in scalar_keys:
        count = record[key]
        if type(count) is not int or count < 0:
            raise IncompatibleDecoder(
                f"telemetry count {key} must be a nonnegative integer"
            )
        scalars.append((key, count))
    claimed: tuple[tuple[int, int], ...] = ()
    unclaimed: tuple[tuple[int, int], ...] = ()
    if source == "b01":
        for key in ("claimed_appid_counts", "unclaimed_appid_counts"):
            raw_map = record[key]
            if not isinstance(raw_map, Mapping):
                raise IncompatibleDecoder(f"{key} must be a mapping")
            try:
                parsed = validate_appid_count_pairs(tuple(raw_map.items()), key)
            except (TypeError, ValueError) as exc:
                raise IncompatibleDecoder(safe_diagnostic_text(exc)) from exc
            if key == "claimed_appid_counts":
                claimed = parsed
            else:
                unclaimed = parsed
    try:
        return TelemetryCounts(
            source=source,
            scalar_counts=tuple(scalars),
            claimed_appid_counts=claimed,
            unclaimed_appid_counts=unclaimed,
        )
    except (TypeError, ValueError) as exc:
        raise IncompatibleDecoder(safe_diagnostic_text(exc)) from exc


def validate_b01_count_consistency(
    counts: TelemetryCounts,
    *,
    packets: Sequence[Any],
    fpga: TelemetryBlock,
    encoder: TelemetryBlock,
    decoder_info: TelemetryDecoderInfo,
) -> None:
    input_count = len(packets)
    if counts.scalar("input_packet_count") != input_count:
        raise IncompatibleDecoder("input_packet_count disagrees with adapter input")
    claimed = counts.scalar("claimed_packet_count")
    unclaimed = counts.scalar("unclaimed_packet_count")
    if claimed + unclaimed != input_count:
        raise IncompatibleDecoder(
            "claimed and unclaimed packet counts are inconsistent"
        )
    if sum(count for _, count in counts.claimed_appid_counts) != claimed:
        raise IncompatibleDecoder("claimed AppID counts are inconsistent")
    if sum(count for _, count in counts.unclaimed_appid_counts) != unclaimed:
        raise IncompatibleDecoder("unclaimed AppID counts are inconsistent")
    if not {appid for appid, _ in counts.claimed_appid_counts}.issubset(
        decoder_info.claimed_appids
    ):
        raise IncompatibleDecoder(
            "claimed AppID counts contain an unclaimed AppID"
        )
    actual_counts: dict[int, int] = {}
    for packet in packets:
        appid = getattr(packet, "appid", None)
        if type(appid) is not int or not 0 <= appid <= 0x7FF:
            raise IncompatibleDecoder("decoder input contains an invalid AppID")
        actual_counts[appid] = actual_counts.get(appid, 0) + 1
    actual_claimed = {
        appid: count
        for appid, count in actual_counts.items()
        if appid in decoder_info.claimed_appids
    }
    actual_unclaimed = {
        appid: count
        for appid, count in actual_counts.items()
        if appid not in decoder_info.claimed_appids
    }
    reported_claimed = {
        appid: count
        for appid, count in counts.claimed_appid_counts
        if count
    }
    reported_unclaimed = {
        appid: count
        for appid, count in counts.unclaimed_appid_counts
        if count
    }
    if reported_claimed != actual_claimed:
        raise IncompatibleDecoder("claimed AppID counts disagree with decoder input")
    if reported_unclaimed != actual_unclaimed:
        raise IncompatibleDecoder(
            "unclaimed AppID counts disagree with decoder input"
        )
    fpga_input = counts.scalar("fpga_input_packet_count")
    encoder_input = counts.scalar("encoder_input_packet_count")
    if fpga_input + encoder_input != claimed:
        raise IncompatibleDecoder(
            "FPGA and encoder input counts are inconsistent"
        )
    if fpga_input != actual_counts.get(0x314, 0):
        raise IncompatibleDecoder("FPGA input count disagrees with decoder input")
    if encoder_input != actual_counts.get(0x325, 0):
        raise IncompatibleDecoder("encoder input count disagrees with decoder input")
    fpga_output = counts.scalar("fpga_output_record_count")
    encoder_output = counts.scalar("encoder_output_record_count")
    fpga_dropped = counts.scalar("fpga_dropped_packet_count")
    encoder_rejected = counts.scalar("encoder_rejected_packet_count")
    if fpga_output != fpga.row_count:
        raise IncompatibleDecoder(
            "FPGA output count disagrees with decoded rows"
        )
    if encoder_output != encoder.row_count:
        raise IncompatibleDecoder(
            "encoder output count disagrees with decoded rows"
        )
    if fpga_input != fpga_output + fpga_dropped:
        raise IncompatibleDecoder("FPGA input/output/drop counts are inconsistent")
    if encoder_input != encoder_output + encoder_rejected:
        raise IncompatibleDecoder(
            "encoder input/output/rejected counts are inconsistent"
        )
    for block, expected_appid in ((fpga, 0x314), (encoder, 0x325)):
        if np.any(block.input_indices >= input_count):
            raise IncompatibleDecoder(
                "telemetry output input_index is outside decoder input"
            )
        if any(
            packets[int(index)].appid != expected_appid
            for index in block.input_indices
        ):
            raise IncompatibleDecoder(
                "telemetry output input_index refers to the wrong AppID"
            )


def validate_sidecar_count_consistency(
    counts: TelemetryCounts,
    *,
    input_byte_count: int,
    fpga: TelemetryBlock,
) -> None:
    if counts.scalar("input_byte_count") != input_byte_count:
        raise IncompatibleDecoder(
            "sidecar input_byte_count disagrees with the file"
        )
    complete = counts.scalar("complete_record_count")
    output = counts.scalar("output_record_count")
    dropped = counts.scalar("dropped_record_count")
    if output != fpga.row_count:
        raise IncompatibleDecoder(
            "sidecar output count disagrees with decoded rows"
        )
    if output + dropped != complete:
        raise IncompatibleDecoder("sidecar record counts are inconsistent")
    if np.any(fpga.input_indices >= complete):
        raise IncompatibleDecoder(
            "sidecar output input_index is outside the complete records"
        )
    if counts.scalar("trailing_byte_count") > input_byte_count:
        raise IncompatibleDecoder(
            "sidecar trailing byte count exceeds file size"
        )


def validate_private_issue(value: object) -> Mapping[str, Any]:
    record = exact_mapping(value, "telemetry decoder issue", ISSUE_KEYS)
    if not isinstance(record["code"], str) or not record["code"].startswith(
        "telemetry_decoder."
    ):
        raise IncompatibleDecoder("telemetry issue code is invalid")
    try:
        validate_storage_text(
            record["code"],
            "telemetry issue code",
            allow_empty=False,
        )
        validate_storage_text(
            record["message"],
            "telemetry issue message",
            allow_empty=False,
        )
    except ValueError as exc:
        raise IncompatibleDecoder(safe_diagnostic_text(exc)) from exc
    try:
        IssueSeverity(record["severity"])
        IssueAction(record["action"])
    except ValueError as exc:
        raise IncompatibleDecoder(
            "telemetry issue severity/action is invalid"
        ) from exc
    for key, maximum in (
        ("input_index", None),
        ("appid", 0x7FF),
        ("sequence_count", 0x3FFF),
    ):
        item = record[key]
        if item is not None and (
            type(item) is not int
            or item < 0
            or (maximum is not None and item > maximum)
        ):
            raise IncompatibleDecoder(f"telemetry issue {key} is invalid")
    if record["field"] is not None and not isinstance(record["field"], str):
        raise IncompatibleDecoder(
            "telemetry issue field must be a string or None"
        )
    if not isinstance(record["details"], Mapping):
        raise IncompatibleDecoder("telemetry issue details must be a mapping")
    return record


def import_private_issues(
    values: object,
    *,
    issue_collector: IssueCollector,
    packets: Sequence[Any] | None,
    field_names: tuple[str, ...],
    input_count: int | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if type(values) not in (list, tuple):
        raise IncompatibleDecoder("telemetry issues must be a list or tuple")
    parsed = tuple(validate_private_issue(value) for value in values)
    known_fields = frozenset(field_names)
    if any(
        issue["field"] is not None and issue["field"] not in known_fields
        for issue in parsed
    ):
        raise IncompatibleDecoder(
            "telemetry issue field is outside field_metadata"
        )
    for issue in parsed:
        details = dict(issue["details"])
        input_index = issue["input_index"]
        details["decoder_input_index"] = input_index
        if issue["field"] is not None:
            details["field"] = issue["field"]
        packet = None
        if packets is not None and input_index is not None:
            if input_index >= len(packets):
                raise IncompatibleDecoder(
                    "telemetry issue input_index is outside decoder input"
                )
            packet = packets[input_index]
            if issue["appid"] is not None and issue["appid"] != getattr(
                packet, "appid", None
            ):
                raise IncompatibleDecoder(
                    "telemetry issue AppID disagrees with decoder input"
                )
            if issue["sequence_count"] is not None and issue[
                "sequence_count"
            ] != getattr(packet, "seq", None):
                raise IncompatibleDecoder(
                    "telemetry issue sequence count disagrees with decoder input"
                )
        elif (
            input_index is not None
            and input_count is not None
            and input_index >= input_count
        ):
            raise IncompatibleDecoder(
                "telemetry issue input_index is outside decoder input"
            )
        packet_index = getattr(packet, "file_index", None)
        if type(packet_index) is not int:
            packet_index = None
        issue_collector.record(
            code=issue["code"],
            severity=issue["severity"],
            stage="telemetry_decode",
            message=issue["message"],
            action=issue["action"],
            bank=getattr(packet, "bank", None),
            packet_index=packet_index,
            appid=(
                getattr(packet, "appid", None)
                if packet is not None
                else issue["appid"]
            ),
            sequence_count=(
                getattr(packet, "seq", None)
                if packet is not None
                else issue["sequence_count"]
            ),
            details=details,
        )
    return parsed


def validate_engineering_issue_coverage(
    block: TelemetryBlock,
    metadata: tuple[TelemetryFieldMetadata, ...],
    issues: tuple[Mapping[str, Any], ...],
) -> None:
    """Require a source-scoped issue for each failed engineering conversion."""
    global_fields = {
        issue["field"]
        for issue in issues
        if issue["field"] is not None and issue["input_index"] is None
    }
    indexed_fields = {
        (issue["field"], issue["input_index"])
        for issue in issues
        if issue["field"] is not None and issue["input_index"] is not None
    }
    for field_index, field in enumerate(metadata):
        if field.kind == "uncalibrated_raw_count":
            continue
        for row_index in np.flatnonzero(~block.valid[:, field_index]):
            input_index = int(block.input_indices[row_index])
            if field.name not in global_fields and (
                field.name,
                input_index,
            ) not in indexed_fields:
                raise IncompatibleDecoder(
                    "invalid engineering value lacks a field-scoped issue"
                )


def telemetry_loss_counts(counts: TelemetryCounts) -> tuple[tuple[str, int], ...]:
    """Return nonzero decoder loss counters from the reviewed public schema."""
    names = (
        ("fpga_dropped_packet_count", "encoder_rejected_packet_count")
        if counts.source == "b01"
        else ("trailing_byte_count", "dropped_record_count")
    )
    return tuple(
        (name, counts.scalar(name))
        for name in names
        if counts.scalar(name)
    )


def ensure_loss_issue(
    counts: TelemetryCounts,
    *,
    issue_collector: IssueCollector,
    marker: int,
) -> None:
    """Ensure decoder-reported loss cannot produce an issue-free partial file."""
    loss_counts = telemetry_loss_counts(counts)
    if not loss_counts or issue_collector.since(marker):
        return
    issue_collector.record(
        code="telemetry_adapter.decoder_reported_loss",
        severity=IssueSeverity.WARNING,
        stage="telemetry_decode",
        message="the telemetry decoder reported dropped or rejected input",
        action=IssueAction.DROPPED,
        details={"loss_counts": dict(loss_counts)},
    )


def coverage_for_available(
    *,
    input_state: TelemetryInputState,
    counts: TelemetryCounts,
    issue_count: int,
) -> TelemetryCoverage:
    if input_state is TelemetryInputState.PRESENT_EMPTY and issue_count == 0:
        return TelemetryCoverage.PRESENT_EMPTY
    if issue_count or telemetry_loss_counts(counts):
        return TelemetryCoverage.PARTIAL
    return TelemetryCoverage.DECODED


def decode_b01_packets(
    logical_packets: Iterable[Any] | None,
    *,
    issue_collector: IssueCollector | None = None,
    decoder: object | None = None,
) -> TelemetryDecodeResult:
    """Decode one present b01 source through detailed telemetry API v1.

    ``None`` means that the b01 source is absent and does not import the
    optional decoder. An empty iterable means a present-empty source.
    """
    if logical_packets is None:
        return TelemetryDecodeResult.absent()
    packets = tuple(logical_packets)
    if issue_collector is None:
        issue_collector = IssueCollector()
    marker = issue_collector.mark()
    input_state = (
        TelemetryInputState.PRESENT_EMPTY
        if not packets
        else TelemetryInputState.PRESENT
    )
    loaded, status, error, info, metadata = resolve_decoder(decoder)
    if loaded is None:
        return failed_result(
            input_source="b01",
            input_state=input_state,
            status=status,
            error=error,
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    try:
        raw_result = loaded.decode_b01_packets(packets)
    except Exception as exc:  # noqa: BLE001
        return failed_result(
            input_source="b01",
            input_state=input_state,
            status=TelemetryDecoderStatus.BROKEN,
            error=safe_diagnostic_text(exc),
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    try:
        result = exact_mapping(
            raw_result,
            "decode_b01_packets result",
            B01_RESULT_KEYS,
        )
        fpga = TelemetryBlock.from_decoder_record(
            result["fpga"], expected_source="b01_0x314"
        )
        encoder = TelemetryBlock.from_decoder_record(
            result["encoder"], expected_source="b01_0x325"
        )
        if encoder.field_names or encoder.row_count:
            raise IncompatibleDecoder(
                "encoder measurements must remain empty until the layout is validated"
            )
        assert info is not None
        if tuple(item.name for item in metadata) != fpga.field_names:
            raise IncompatibleDecoder(
                "field_metadata order disagrees with FPGA field_names"
            )
        counts = parse_counts(result["counts"], source="b01")
        validate_b01_count_consistency(
            counts,
            packets=packets,
            fpga=fpga,
            encoder=encoder,
            decoder_info=info,
        )
        private_issues = import_private_issues(
            result["issues"],
            issue_collector=issue_collector,
            packets=packets,
            field_names=fpga.field_names,
        )
        validate_engineering_issue_coverage(fpga, metadata, private_issues)
        ensure_loss_issue(
            counts,
            issue_collector=issue_collector,
            marker=marker,
        )
    except (TypeError, ValueError) as exc:
        return failed_result(
            input_source="b01",
            input_state=input_state,
            status=TelemetryDecoderStatus.INCOMPATIBLE,
            error=safe_diagnostic_text(exc),
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    issues = issue_collector.since(marker)
    return TelemetryDecodeResult(
        input_source="b01",
        input_state=input_state,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=coverage_for_available(
            input_state=input_state,
            counts=counts,
            issue_count=len(issues),
        ),
        decoder_info=info,
        field_metadata=metadata,
        fpga=fpga,
        encoder=encoder,
        counts=counts,
        issues=issues,
    )


def decode_legacy_sidecar(
    path: Path | str | None,
    *,
    issue_collector: IssueCollector | None = None,
    decoder: object | None = None,
) -> TelemetryDecodeResult:
    """Decode one independently recognized historic binary sidecar."""
    if path is None:
        return TelemetryDecodeResult.absent()
    sidecar = Path(path)
    if issue_collector is None:
        issue_collector = IssueCollector()
    marker = issue_collector.mark()
    if not (sidecar.exists() or sidecar.is_symlink()):
        return failed_input_result(
            input_source="legacy_sidecar",
            error="recognized sidecar is no longer reachable",
            issue_collector=issue_collector,
            marker=marker,
        )
    try:
        input_byte_count = sidecar.stat().st_size
    except OSError as exc:
        return failed_input_result(
            input_source="legacy_sidecar",
            error=safe_diagnostic_text(exc),
            issue_collector=issue_collector,
            marker=marker,
        )
    input_state = (
        TelemetryInputState.PRESENT_EMPTY
        if input_byte_count == 0
        else TelemetryInputState.PRESENT
    )
    loaded, status, error, info, metadata = resolve_decoder(decoder)
    if loaded is None:
        return failed_result(
            input_source="legacy_sidecar",
            input_state=input_state,
            status=status,
            error=error,
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    try:
        raw_result = loaded.decode_legacy_sidecar(sidecar)
    except Exception as exc:  # noqa: BLE001
        return failed_result(
            input_source="legacy_sidecar",
            input_state=input_state,
            status=TelemetryDecoderStatus.BROKEN,
            error=safe_diagnostic_text(exc),
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    try:
        result = exact_mapping(
            raw_result,
            "decode_legacy_sidecar result",
            SIDECAR_RESULT_KEYS,
        )
        fpga = TelemetryBlock.from_decoder_record(
            result["fpga"], expected_source="legacy_binary_sidecar"
        )
        assert info is not None
        if tuple(item.name for item in metadata) != fpga.field_names:
            raise IncompatibleDecoder(
                "field_metadata order disagrees with sidecar field_names"
            )
        counts = parse_counts(result["counts"], source="legacy_sidecar")
        validate_sidecar_count_consistency(
            counts,
            input_byte_count=input_byte_count,
            fpga=fpga,
        )
        private_issues = import_private_issues(
            result["issues"],
            issue_collector=issue_collector,
            packets=None,
            field_names=fpga.field_names,
            input_count=counts.scalar("complete_record_count"),
        )
        validate_engineering_issue_coverage(fpga, metadata, private_issues)
        ensure_loss_issue(
            counts,
            issue_collector=issue_collector,
            marker=marker,
        )
    except (TypeError, ValueError) as exc:
        return failed_result(
            input_source="legacy_sidecar",
            input_state=input_state,
            status=TelemetryDecoderStatus.INCOMPATIBLE,
            error=safe_diagnostic_text(exc),
            issue_collector=issue_collector,
            marker=marker,
            decoder_info=info,
            field_metadata=metadata,
        )
    issues = issue_collector.since(marker)
    return TelemetryDecodeResult(
        input_source="legacy_sidecar",
        input_state=input_state,
        decoder_status=TelemetryDecoderStatus.AVAILABLE,
        coverage=coverage_for_available(
            input_state=input_state,
            counts=counts,
            issue_count=len(issues),
        ),
        decoder_info=info,
        field_metadata=metadata,
        fpga=fpga,
        counts=counts,
        issues=issues,
    )


def find_legacy_sidecar(session_dir: Path | str) -> Path | None:
    """Recognize the public historic filename without loading the decoder."""
    path = Path(session_dir) / LEGACY_TELEMETRY_SIDECAR_NAME
    return path if path.exists() or path.is_symlink() else None


def has_decoder(*, decoder: object | None = None) -> bool:
    """Return whether detailed telemetry API v1 is currently usable."""
    loaded, status, _error, _info, _metadata = resolve_decoder(decoder)
    return loaded is not None and status is TelemetryDecoderStatus.AVAILABLE


def telemetry_apids(*, decoder: object | None = None) -> tuple[int, ...]:
    """Return claimed AppIDs for a compatible installed decoder."""
    loaded, status, _error, info, _metadata = resolve_decoder(decoder)
    if loaded is None or status is not TelemetryDecoderStatus.AVAILABLE:
        return ()
    assert info is not None
    return info.claimed_appids


def block_engineering_values(
    block: TelemetryBlock | None,
) -> dict[str, np.ndarray]:
    """Return the established loose-array view for compatibility callers."""
    if block is None:
        return {}
    result = {
        "mission_seconds": block.mission_seconds,
        "lusee_subsecs": block.lusee_subsecs,
        "raw_seconds": block.raw_seconds,
    }
    result.update({
        name: block.values[:, index]
        for index, name in enumerate(block.field_names)
    })
    return result


def parse_b01_packets(
    logical_packets: Iterable[Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compatibility wrapper around the detailed typed decoder boundary."""
    result = decode_b01_packets(logical_packets)
    if result.decoder_status is not TelemetryDecoderStatus.AVAILABLE:
        return {}, {}
    return (
        block_engineering_values(result.fpga),
        block_engineering_values(result.encoder),
    )


def parse_legacy_sidecar(path: Path | str) -> dict[str, np.ndarray]:
    """Compatibility wrapper returning legacy sidecar engineering arrays."""
    result = decode_legacy_sidecar(path)
    if result.decoder_status is not TelemetryDecoderStatus.AVAILABLE:
        return {}
    return block_engineering_values(result.fpga)


def field_groups() -> dict[str, tuple[str, ...]]:
    """Group compatible decoder fields by optional display-group metadata."""
    loaded, status, _error, _info, metadata = resolve_decoder(None)
    if loaded is None or status is not TelemetryDecoderStatus.AVAILABLE:
        return {}
    groups: dict[str, list[str]] = {}
    for field in metadata:
        if field.display_group is not None:
            groups.setdefault(field.display_group, []).append(field.name)
    return {name: tuple(fields) for name, fields in groups.items()}


def slice_arrays_by_window(
    arrays: Mapping[str, np.ndarray],
    *,
    window_lower_raw_seconds: float | None = None,
    window_upper_raw_seconds: float | None = None,
    time_key: str = "mission_seconds",
) -> dict[str, np.ndarray]:
    """Filter every loose compatibility array to one half-open time window."""
    if not arrays or time_key not in arrays:
        return dict(arrays)
    times = arrays[time_key]
    if times.size == 0:
        return dict(arrays)
    mask = np.ones(times.size, dtype=np.bool_)
    if window_lower_raw_seconds is not None:
        mask &= times >= window_lower_raw_seconds
    if window_upper_raw_seconds is not None:
        mask &= times < window_upper_raw_seconds
    if mask.all():
        return dict(arrays)
    return {name: values[mask] for name, values in arrays.items()}


def map_dcb_absolute_time(
    result: TelemetryDecodeResult,
    *,
    clock_reference_set: ClockReferenceSet,
    issue_collector: IssueCollector,
) -> TelemetryDecodeResult:
    """Map telemetry through only an explicit DCB clock reference."""
    if not isinstance(result, TelemetryDecodeResult):
        raise TypeError("result must be a TelemetryDecodeResult")
    if not isinstance(clock_reference_set, ClockReferenceSet):
        raise TypeError("clock_reference_set must be a ClockReferenceSet")
    if not isinstance(issue_collector, IssueCollector):
        raise TypeError("issue_collector must be an IssueCollector")
    if result.decoder_status is not TelemetryDecoderStatus.AVAILABLE:
        return result
    if result.fpga is None:
        raise ValueError("available telemetry has no FPGA block")
    blocks = tuple(
        block
        for block in (result.fpga, result.unassigned_fpga)
        if block is not None and block.row_count
    )
    if not blocks:
        return result
    dcb_reference = clock_reference_set.reference_for(ClockSource.DCB)
    if dcb_reference is None:
        if any(block.mjd_time_valid.any() for block in blocks):
            raise ValueError("telemetry has absolute time without a DCB reference")
        issue = issue_collector.record(
            code="telemetry_time.missing_dcb_reference",
            severity=IssueSeverity.WARNING,
            stage="telemetry_time",
            message=(
                "telemetry raw time was retained without absolute time because "
                "the landing reference has no DCB clock"
            ),
            action=IssueAction.KEPT,
            details={"clock_source": ClockSource.DCB.value},
        )
        return result.with_blocks(
            fpga=result.fpga,
            unassigned_fpga=result.unassigned_fpga,
            issues=(*result.issues, issue),
            coverage=TelemetryCoverage.PARTIAL,
        )

    def map_block(block: TelemetryBlock | None) -> TelemetryBlock | None:
        if block is None or block.row_count == 0:
            return block
        return block.with_mjd_times(np.asarray(
            clock_reference_set.to_mjd(
                block.raw_seconds,
                clock_source=ClockSource.DCB,
            ),
            dtype=np.float64,
        ))

    mapped_fpga = map_block(result.fpga)
    assert mapped_fpga is not None
    return result.with_blocks(
        fpga=mapped_fpga,
        unassigned_fpga=map_block(result.unassigned_fpga),
    )
