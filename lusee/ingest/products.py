"""Immutable provenance and validation records for decoded ingest products."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .clock_reference import ClockSource
from .constants import (
    BITSLICE_REFERENCE,
    MISSION_TIME_FRACT_DIVISOR,
    MISSION_TIME_FRACT_SHIFT,
    NCHANNELS,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    SPECTRA_REPRESENTATION,
    SPECTRA_UNITS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
)
from .frequency_contract import FrequencyWindowContract


class DataQuality(StrEnum):
    """Usability of decoded or persisted scientific data."""

    CLEAN = "clean"
    PARTIAL = "partial"
    FAILED = "failed"


class ExecutionMode(StrEnum):
    """Decoder issue handling, separate from scientific data quality."""

    COLLECT = "collect"
    STRICT = "strict"


def _optional_nonnegative(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")
    return value


def _required_nonnegative(value: int, name: str) -> int:
    result = _optional_nonnegative(value, name)
    if result is None:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _optional_appid(value: int | None, name: str) -> int | None:
    value = _optional_nonnegative(value, name)
    if value is not None and value > 0x7FF:
        raise ValueError(f"{name} must lie in the 11-bit CCSDS AppID range")
    return value


def _required_appid(value: int, name: str) -> int:
    result = _optional_appid(value, name)
    if result is None:
        raise ValueError(f"{name} must be an 11-bit CCSDS AppID")
    return result


def _optional_uid(value: int | None, name: str = "uid") -> int | None:
    value = _optional_nonnegative(value, name)
    if value is not None and value > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in an unsigned 32-bit integer")
    return value


def _optional_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string or None")
    return value


def _required_text(value: str, name: str) -> str:
    result = _optional_text(value, name)
    if result is None:
        raise ValueError(f"{name} must be a nonempty string")
    return result


def _schema_ids(values: tuple[int, ...], name: str) -> tuple[int, ...]:
    if any(type(value) is not int or not 0 <= value <= 0xFFFF for value in values):
        raise ValueError(f"{name} must contain unsigned 16-bit schema IDs")
    normalized = tuple(sorted(set(values)))
    if len(normalized) != len(values):
        raise ValueError(f"{name} must not contain duplicate schema IDs")
    return normalized


def _count_pairs(
    values: tuple[tuple[int | str, int], ...],
    *,
    key_type: type,
    name: str,
) -> tuple[tuple[int | str, int], ...]:
    normalized = []
    seen = set()
    for key, count in values:
        if type(key) is not key_type:
            raise ValueError(f"{name} has an invalid key {key!r}")
        if key_type is str and not key:
            raise ValueError(f"{name} keys must not be empty")
        if key_type is int and not 0 <= key <= 0x7FF:
            raise ValueError(f"{name} AppID keys must lie in [0, 0x7ff]")
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}")
        if type(count) is not int or count < 0:
            raise ValueError(f"{name} counts must be nonnegative integers")
        seen.add(key)
        normalized.append((key, count))
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class SourcePacketProvenance:
    """Location and AppID identity of one packet contributing to a product."""

    role: str
    original_appid: int
    filename: str | None = None
    packet_index: int | None = None
    normalized_appid: int | None = None
    bank: str | None = None
    frame_start: int | None = None
    frame_stop: int | None = None
    byte_offset_start: int | None = None
    byte_offset_stop: int | None = None

    def __post_init__(self) -> None:
        role = _required_text(self.role, "source packet role")
        original_appid = _required_appid(self.original_appid, "original_appid")
        normalized_appid = _optional_appid(
            self.normalized_appid, "normalized_appid"
        )
        packet_index = _optional_nonnegative(self.packet_index, "packet_index")
        bank = _optional_text(self.bank, "bank")
        filename = _optional_text(self.filename, "filename")
        if filename is not None:
            if Path(filename).name != filename or "/" in filename or "\\" in filename:
                raise ValueError("filename must contain a basename only")
        frame_start, frame_stop = self._validate_span(
            self.frame_start, self.frame_stop, "frame"
        )
        offset_start, offset_stop = self._validate_span(
            self.byte_offset_start, self.byte_offset_stop, "byte_offset"
        )
        if all(
            locator is None
            for locator in (filename, packet_index, frame_start, offset_start)
        ):
            raise ValueError(
                "source packet provenance requires a concrete locator"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "original_appid", original_appid)
        object.__setattr__(self, "normalized_appid", normalized_appid)
        object.__setattr__(self, "packet_index", packet_index)
        object.__setattr__(self, "bank", bank)
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "frame_start", frame_start)
        object.__setattr__(self, "frame_stop", frame_stop)
        object.__setattr__(self, "byte_offset_start", offset_start)
        object.__setattr__(self, "byte_offset_stop", offset_stop)

    @staticmethod
    def _validate_span(
        start: int | None,
        stop: int | None,
        name: str,
    ) -> tuple[int | None, int | None]:
        if (start is None) != (stop is None):
            raise ValueError(f"{name} span requires both start and stop")
        start = _optional_nonnegative(start, f"{name}_start")
        stop = _optional_nonnegative(stop, f"{name}_stop")
        if start is not None and stop < start:
            raise ValueError(f"{name}_stop must not precede {name}_start")
        return start, stop


@dataclass(frozen=True, slots=True)
class ProductProvenance:
    """Immutable packet, UID, schema, issue, and clock identity for one row."""

    source_packets: tuple[SourcePacketProvenance, ...] = ()
    uid: int | None = None
    uid_source: str | None = None
    uid_source_role: str | None = None
    reported_schema_ids: tuple[int, ...] = ()
    selected_schema_id: int | None = None
    decoder_issue_ids: tuple[str, ...] = ()
    time_source: str | None = None
    time_source_role: str | None = None
    clock_source: str | None = None
    time_valid: bool = False
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        source_packets = tuple(self.source_packets)
        if any(not isinstance(item, SourcePacketProvenance) for item in source_packets):
            raise TypeError("source_packets must contain SourcePacketProvenance records")
        uid = _optional_uid(self.uid)
        uid_source = _optional_text(self.uid_source, "uid_source")
        uid_source_role = _optional_text(self.uid_source_role, "uid_source_role")
        if (uid is None) != (uid_source is None):
            raise ValueError("uid and uid_source must be recorded together")
        if uid_source_role is not None and uid_source is None:
            raise ValueError("uid_source_role requires uid_source")
        reported = _schema_ids(self.reported_schema_ids, "reported_schema_ids")
        selected = self.selected_schema_id
        if selected is not None:
            if type(selected) is not int or not 0 <= selected <= 0xFFFF:
                raise ValueError("selected_schema_id must be an unsigned 16-bit value")
        issues = tuple(self.decoder_issue_ids)
        if any(not isinstance(issue, str) or not issue for issue in issues):
            raise ValueError("decoder_issue_ids must contain nonempty strings")
        if len(set(issues)) != len(issues):
            raise ValueError("decoder_issue_ids must not contain duplicates")
        time_source = _optional_text(self.time_source, "time_source")
        time_source_role = _optional_text(self.time_source_role, "time_source_role")
        clock_source = _optional_text(self.clock_source, "clock_source")
        if clock_source is not None:
            try:
                clock_source = ClockSource(clock_source).value
            except ValueError as exc:
                raise ValueError(
                    "clock_source must be spectrometer, dcb, or adc"
                ) from exc
        if time_source_role is not None and time_source is None:
            raise ValueError("time_source_role requires time_source")
        unavailable_reason = _optional_text(
            self.unavailable_reason, "unavailable_reason"
        )
        if type(self.time_valid) is not bool:
            raise TypeError("time_valid must be a boolean")
        if self.time_valid and (time_source is None or clock_source is None):
            raise ValueError("valid product time requires time_source and clock_source")
        roles = {packet.role for packet in source_packets}
        for source_role, name in (
            (uid_source_role, "uid_source_role"),
            (time_source_role, "time_source_role"),
        ):
            if source_role is not None and source_role not in roles:
                raise ValueError(f"{name} does not identify a source packet role")
        if source_packets and unavailable_reason is not None:
            raise ValueError(
                "unavailable_reason cannot accompany concrete source packets"
            )
        if not source_packets and unavailable_reason is None:
            raise ValueError(
                "provenance without source packets requires unavailable_reason"
            )
        if source_packets and uid is None:
            raise ValueError(
                "concrete product provenance requires uid and uid_source"
            )
        object.__setattr__(self, "source_packets", source_packets)
        object.__setattr__(self, "uid", uid)
        object.__setattr__(self, "uid_source", uid_source)
        object.__setattr__(self, "uid_source_role", uid_source_role)
        object.__setattr__(self, "reported_schema_ids", reported)
        object.__setattr__(self, "decoder_issue_ids", issues)
        object.__setattr__(self, "time_source", time_source)
        object.__setattr__(self, "time_source_role", time_source_role)
        object.__setattr__(self, "clock_source", clock_source)
        object.__setattr__(self, "unavailable_reason", unavailable_reason)

    @classmethod
    def unavailable(cls, reason: str = "caller_constructed") -> "ProductProvenance":
        """Return explicit unknown provenance for compatibility constructors."""
        return cls(unavailable_reason=reason)

    def validate_product_identity(
        self,
        *,
        unique_packet_id: int,
        raw_seconds: float | None,
    ) -> None:
        """Validate one product row against this provenance record."""
        row_uid = _optional_uid(unique_packet_id, "unique_packet_id")
        if self.uid is None:
            raise ValueError("ProductProvenance does not record the product UID")
        if self.uid != row_uid:
            raise ValueError("product UID disagrees with ProductProvenance")
        if raw_seconds is not None:
            if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, (int, float)):
                raise TypeError("raw_seconds must be numeric or None")
            if not math.isfinite(float(raw_seconds)):
                raise ValueError("raw_seconds must be finite or None")
        if self.time_valid != (raw_seconds is not None):
            raise ValueError("raw_seconds presence disagrees with time_valid")


def _required_uint(value: int, bits: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit in an unsigned {bits}-bit integer")
    return value


def _optional_uint(value: int | None, bits: int, name: str) -> int | None:
    if value is None:
        return None
    return _required_uint(value, bits, name)


def _required_bool(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, name)


def _readonly_array(
    value: np.ndarray,
    *,
    dtype: np.dtype | type,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """Copy an exact ndarray into storage backed by immutable bytes."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(
            f"{name} must have dtype {expected_dtype}; got {value.dtype}"
        )
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}")
    immutable_bytes = value.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=expected_dtype).reshape(shape)


def _required_int32(value: int, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not -(1 << 31) <= value < 1 << 31:
        raise ValueError(f"{name} must fit in a signed 32-bit integer")
    return value


def _immutable_field_value(value: object, name: str) -> object:
    """Copy one normalized field value into a deeply immutable form."""
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) in (float, complex):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.inexact) and not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError(f"{name} must not have object dtype")
        if np.issubdtype(value.dtype, np.inexact) and not np.all(
            np.isfinite(value)
        ):
            raise ValueError(f"{name} must contain only finite values")
        immutable_bytes = value.tobytes(order="C")
        return np.frombuffer(immutable_bytes, dtype=value.dtype).reshape(value.shape)
    if type(value) is tuple:
        return tuple(
            _immutable_field_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError(f"{name} nested mapping keys must be nonempty strings")
        normalized = {}
        for key, item in sorted(value.items()):
            normalized[key] = _immutable_field_value(item, f"{name}.{key}")
        return MappingProxyType(normalized)
    raise TypeError(
        f"{name} must be a normalized scalar, ndarray, tuple, or mapping"
    )


def _normalized_fields(
    values: Mapping[str, object | None],
    presence: Mapping[str, bool],
    *,
    name: str,
) -> tuple[Mapping[str, object | None], Mapping[str, bool]]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} fields must be a mapping")
    if not isinstance(presence, Mapping):
        raise TypeError(f"{name} field_present must be a mapping")
    value_keys = set(values)
    presence_keys = set(presence)
    if value_keys != presence_keys:
        raise ValueError(f"{name} fields and field_present must have identical keys")
    if any(
        not isinstance(key, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*", key) is None
        for key in value_keys
    ):
        raise ValueError(f"{name} field names must be lowercase identifiers")

    normalized_values = {}
    normalized_presence = {}
    for key in sorted(value_keys):
        is_present = presence[key]
        if type(is_present) is not bool:
            raise TypeError(f"{name} field presence values must be booleans")
        value = values[key]
        if is_present == (value is None):
            raise ValueError(
                f"{name} field {key!r} presence disagrees with its value"
            )
        normalized_values[key] = _immutable_field_value(
            value, f"{name}.{key}"
        )
        normalized_presence[key] = is_present
    return MappingProxyType(normalized_values), MappingProxyType(normalized_presence)


@dataclass(frozen=True, slots=True)
class SpectrumMetadata(Mapping[str, object]):
    """Normalized science metadata shared by normal and TR spectrum rows."""

    version: int
    unique_packet_id: int
    uc_time: int
    time_32: int | None
    time_16: int | None
    tvs_sensors: np.ndarray
    requested_gain: np.ndarray
    gain_auto_min: np.ndarray
    gain_auto_mult: np.ndarray
    route_plus: np.ndarray
    route_minus: np.ndarray
    navg1_shift: int
    navg2_shift: int
    notch: int
    navgf: int
    high_fraction: int
    medium_fraction: int
    requested_bitslice: np.ndarray
    bitslice_keep_bits: int
    output_format: int
    reject_ratio: int
    reject_max_bad: int
    tr_start: int
    tr_stop: int
    tr_average_shift: int
    errors: int
    correlation_products_mask: int
    actual_gain: np.ndarray
    actual_bitslice: np.ndarray
    spectrum_overflow: int
    notch_overflow: int
    adc_min: np.ndarray
    adc_max: np.ndarray
    adc_valid_count: np.ndarray
    adc_invalid_count_max: np.ndarray
    adc_invalid_count_min: np.ndarray
    adc_total_count: np.ndarray
    adc_mean: np.ndarray
    adc_rms: np.ndarray
    spectrometer_enable: bool
    calibrator_enable: bool
    random_state: int
    weight: int
    weight_current: int
    telemetry_v1_0: float
    telemetry_v1_8: float
    telemetry_v2_5: float
    telemetry_t_fpga: float
    loop_count_min: int | None = None
    loop_count_max: int | None = None
    grimm_enable: int | None = None
    averaging_mode: int | None = None
    num_bad_min_current: int | None = None
    num_bad_max_current: int | None = None
    num_bad_min: int | None = None
    num_bad_max: int | None = None
    adc_statistics_valid: np.ndarray = field(init=False, repr=False)
    current_fields_present: bool = field(init=False)

    def __post_init__(self) -> None:
        uint_fields = (
            ("version", 16),
            ("unique_packet_id", 32),
            ("uc_time", 64),
            ("navg1_shift", 8),
            ("navg2_shift", 8),
            ("notch", 8),
            ("navgf", 8),
            ("high_fraction", 8),
            ("medium_fraction", 8),
            ("bitslice_keep_bits", 8),
            ("output_format", 8),
            ("reject_ratio", 8),
            ("reject_max_bad", 8),
            ("tr_start", 16),
            ("tr_stop", 16),
            ("tr_average_shift", 16),
            ("errors", 32),
            ("correlation_products_mask", 16),
            ("spectrum_overflow", 16),
            ("notch_overflow", 16),
            ("random_state", 32),
            ("weight", 16),
            ("weight_current", 16),
        )
        for name, bits in uint_fields:
            _required_uint(getattr(self, name), bits, name)

        if (self.time_32 is None) != (self.time_16 is None):
            raise ValueError("time_32 and time_16 must be present together")
        _optional_uint(self.time_32, 32, "time_32")
        _optional_uint(self.time_16, 16, "time_16")
        if self.navgf not in (1, 2, 3, 4):
            raise ValueError("navgf must be one of 1, 2, 3, or 4")
        if self.navg2_shift > 15:
            raise ValueError("navg2_shift must lie in [0, 15]")
        if self.tr_average_shift > 15:
            raise ValueError("tr_average_shift must lie in [0, 15]")
        if self.tr_start > NCHANNELS or self.tr_stop > NCHANNELS:
            raise ValueError(
                f"TR bounds must lie in [0, {NCHANNELS}]"
            )

        array_fields = (
            ("tvs_sensors", np.uint16, (4,)),
            ("requested_gain", np.uint8, (4,)),
            ("gain_auto_min", np.uint16, (4,)),
            ("gain_auto_mult", np.uint16, (4,)),
            ("route_plus", np.uint8, (4,)),
            ("route_minus", np.uint8, (4,)),
            ("requested_bitslice", np.uint8, (NPRODUCTS,)),
            ("actual_gain", np.uint8, (4,)),
            ("actual_bitslice", np.uint8, (NPRODUCTS,)),
            ("adc_min", np.int64, (4,)),
            ("adc_max", np.int64, (4,)),
            ("adc_valid_count", np.int64, (4,)),
            ("adc_invalid_count_max", np.int64, (4,)),
            ("adc_invalid_count_min", np.int64, (4,)),
            ("adc_total_count", np.int64, (4,)),
            ("adc_mean", np.float64, (4,)),
            ("adc_rms", np.float64, (4,)),
        )
        normalized_arrays: dict[str, np.ndarray] = {}
        for name, dtype, shape in array_fields:
            normalized = _readonly_array(
                getattr(self, name), dtype=dtype, shape=shape, name=name
            )
            normalized_arrays[name] = normalized
            object.__setattr__(self, name, normalized)

        requested_bitslice = normalized_arrays["requested_bitslice"]
        if np.any(
            (requested_bitslice > BITSLICE_REFERENCE)
            & (requested_bitslice != np.uint8(0xFF))
        ):
            raise ValueError(
                "requested_bitslice values must lie in "
                f"[0, {BITSLICE_REFERENCE}] or equal 255"
            )
        actual_bitslice = normalized_arrays["actual_bitslice"]
        if np.any(actual_bitslice > BITSLICE_REFERENCE):
            raise ValueError(
                "actual_bitslice values must lie in "
                f"[0, {BITSLICE_REFERENCE}]"
            )

        count_names = (
            "adc_valid_count",
            "adc_invalid_count_max",
            "adc_invalid_count_min",
            "adc_total_count",
        )
        for name in count_names:
            if np.any(normalized_arrays[name] < 0):
                raise ValueError(f"{name} values must be nonnegative")
        expected_total = np.asarray(
            [
                int(valid) + int(high) + int(low)
                for valid, high, low in zip(
                    normalized_arrays["adc_valid_count"],
                    normalized_arrays["adc_invalid_count_max"],
                    normalized_arrays["adc_invalid_count_min"],
                )
            ],
            dtype=np.int64,
        )
        if not np.array_equal(
            normalized_arrays["adc_total_count"], expected_total
        ):
            raise ValueError(
                "adc_total_count must equal valid plus both invalid counts"
            )
        if not np.all(np.isfinite(normalized_arrays["adc_mean"])):
            raise ValueError("adc_mean values must be finite")
        if not np.all(np.isfinite(normalized_arrays["adc_rms"])):
            raise ValueError("adc_rms values must be finite")
        if np.any(normalized_arrays["adc_rms"] < 0):
            raise ValueError("adc_rms values must be nonnegative")
        adc_statistics_valid = _readonly_array(
            np.asarray(
                normalized_arrays["adc_valid_count"] > 0,
                dtype=np.bool_,
            ),
            dtype=np.bool_,
            shape=(4,),
            name="adc_statistics_valid",
        )
        object.__setattr__(
            self, "adc_statistics_valid", adc_statistics_valid
        )

        _required_bool(self.spectrometer_enable, "spectrometer_enable")
        _required_bool(self.calibrator_enable, "calibrator_enable")
        for name in (
            "telemetry_v1_0",
            "telemetry_v1_8",
            "telemetry_v2_5",
            "telemetry_t_fpga",
        ):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), name)
            )

        current_fields = (
            ("loop_count_min", 16),
            ("loop_count_max", 16),
            ("grimm_enable", 8),
            ("averaging_mode", 8),
            ("num_bad_min_current", 16),
            ("num_bad_max_current", 16),
            ("num_bad_min", 16),
            ("num_bad_max", 16),
        )
        presence = tuple(getattr(self, name) is not None for name, _ in current_fields)
        if any(presence) and not all(presence):
            raise ValueError(
                "current-schema metadata fields must be all present or all absent"
            )
        for name, bits in current_fields:
            _optional_uint(getattr(self, name), bits, name)
        object.__setattr__(self, "current_fields_present", all(presence))

    @property
    def raw_seconds(self) -> float | None:
        """Return exact split mission time, or None when it is absent."""
        if self.time_32 is None or self.time_16 is None:
            return None
        combined = (self.time_16 << 32) + self.time_32
        return (
            (combined >> MISSION_TIME_FRACT_SHIFT)
            / MISSION_TIME_FRACT_DIVISOR
        )

    def as_mapping(self) -> Mapping[str, object]:
        """Return a read-only mapping for transitional writer compatibility."""
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.init and getattr(self, item.name) is not None
        }
        values["adc_statistics_valid"] = self.adc_statistics_valid
        values["current_fields_present"] = self.current_fields_present
        return MappingProxyType(values)

    def __getitem__(self, key: str) -> object:
        return self.as_mapping()[key]

    def __iter__(self):
        return iter(self.as_mapping())

    def __len__(self) -> int:
        return len(self.as_mapping())

    def items(self):
        """Return a read-only metadata items view."""
        return self.as_mapping().items()


def _validate_concrete_product_provenance(
    provenance: ProductProvenance,
    *,
    unique_packet_id: int,
    raw_seconds: float | None,
) -> None:
    if not isinstance(provenance, ProductProvenance):
        raise TypeError("provenance must be a ProductProvenance record")
    if provenance.unavailable_reason is not None:
        raise ValueError("strict spectrum records require concrete provenance")
    if provenance.selected_schema_id is None:
        raise ValueError("strict spectrum provenance requires a selected schema")
    if provenance.clock_source != ClockSource.SPECTROMETER.value:
        raise ValueError(
            "normal and TR spectrum provenance must use the spectrometer clock"
        )
    provenance.validate_product_identity(
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
    )


def _validate_auxiliary_product_provenance(
    provenance: ProductProvenance,
    *,
    unique_packet_id: int,
    raw_seconds: float | None,
    product_name: str,
) -> None:
    if not isinstance(provenance, ProductProvenance):
        raise TypeError("provenance must be a ProductProvenance record")
    if provenance.unavailable_reason is not None:
        raise ValueError(f"strict {product_name} records require concrete provenance")
    if provenance.selected_schema_id is None:
        raise ValueError(f"strict {product_name} provenance requires a selected schema")
    provenance.validate_product_identity(
        unique_packet_id=unique_packet_id,
        raw_seconds=raw_seconds,
    )
    if (
        raw_seconds is not None
        and provenance.clock_source != ClockSource.SPECTROMETER.value
    ):
        raise ValueError(
            f"{product_name} mission time must use the spectrometer clock"
        )


def _auxiliary_identity(
    unique_packet_id: int,
    raw_seconds: float | None,
    provenance: ProductProvenance,
    product_name: str,
) -> tuple[int, float | None]:
    uid = _optional_uid(unique_packet_id, "unique_packet_id")
    if uid is None:
        raise ValueError("unique_packet_id must be present")
    time = _optional_finite_float(raw_seconds, "raw_seconds")
    _validate_auxiliary_product_provenance(
        provenance,
        unique_packet_id=uid,
        raw_seconds=time,
        product_name=product_name,
    )
    return uid, time


@dataclass(frozen=True, slots=True)
class SpectrumSample:
    """One exact restored-SDU normal-spectrum row."""

    data: np.ndarray
    product_present: np.ndarray
    navgf: int
    frequency_contract: FrequencyWindowContract
    unique_packet_id: int
    raw_seconds: float | None
    metadata: SpectrumMetadata
    provenance: ProductProvenance
    units: str = field(init=False, default=SPECTRA_UNITS)
    representation: str = field(init=False, default=SPECTRA_REPRESENTATION)
    normalization_version: int = field(
        init=False, default=SPECTRA_NORMALIZATION_VERSION
    )
    bitslice_reference: int = field(init=False, default=BITSLICE_REFERENCE)

    def __post_init__(self) -> None:
        if type(self.navgf) is not int or self.navgf not in (1, 2, 3, 4):
            raise ValueError("navgf must be one of 1, 2, 3, or 4")
        if not isinstance(self.frequency_contract, FrequencyWindowContract):
            raise TypeError(
                "frequency_contract must be a FrequencyWindowContract"
            )
        if self.frequency_contract.navgf != self.navgf:
            raise ValueError("navgf disagrees with frequency_contract")
        nfreq = self.frequency_contract.output_count
        data = _readonly_array(
            self.data,
            dtype=np.float32,
            shape=(NPRODUCTS, nfreq),
            name="normal spectrum data",
        )
        product_present = _readonly_array(
            self.product_present,
            dtype=np.bool_,
            shape=(NPRODUCTS,),
            name="normal spectrum product_present",
        )
        if not np.any(product_present):
            raise ValueError("normal spectrum must contain at least one product")
        if not np.all(np.isfinite(data[product_present])):
            raise ValueError("present normal-spectrum products must be finite")
        if np.any(~product_present) and not np.all(
            np.isnan(data[~product_present])
        ):
            raise ValueError(
                "absent normal-spectrum product planes must contain only NaN"
            )

        unique_packet_id = _optional_uid(
            self.unique_packet_id, "unique_packet_id"
        )
        if unique_packet_id is None:
            raise ValueError("unique_packet_id must be present")
        raw_seconds = _optional_finite_float(self.raw_seconds, "raw_seconds")
        if not isinstance(self.metadata, SpectrumMetadata):
            raise TypeError("metadata must be a SpectrumMetadata record")
        if self.metadata.unique_packet_id != unique_packet_id:
            raise ValueError("metadata UID disagrees with spectrum UID")
        if self.metadata.navgf != self.navgf:
            raise ValueError("metadata navgf disagrees with spectrum navgf")
        if self.metadata.weight == 0:
            raise ValueError("normal spectrum metadata weight must be nonzero")
        if self.metadata.raw_seconds != raw_seconds:
            raise ValueError("metadata split time disagrees with raw_seconds")
        _validate_concrete_product_provenance(
            self.provenance,
            unique_packet_id=unique_packet_id,
            raw_seconds=raw_seconds,
        )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "product_present", product_present)
        object.__setattr__(self, "unique_packet_id", unique_packet_id)
        object.__setattr__(self, "raw_seconds", raw_seconds)

    @property
    def nfreq(self) -> int:
        """Number of native output channels in this row."""
        return self.data.shape[1]

    def restore_bitslice(self) -> None:
        """Assert the immutable row is already in restored gain-model SDU."""
        if (
            self.units != SPECTRA_UNITS
            or self.representation != SPECTRA_REPRESENTATION
            or self.normalization_version != SPECTRA_NORMALIZATION_VERSION
            or self.bitslice_reference != BITSLICE_REFERENCE
            or self.data.dtype != np.dtype(np.float32)
            or self.data.shape
            != (NPRODUCTS, self.frequency_contract.output_count)
        ):
            raise ValueError(
                "normal spectrum does not satisfy the restored-SDU invariant"
            )


@dataclass(frozen=True, slots=True)
class TRSpectrumSample:
    """One exact native-integer time-resolved spectrum row."""

    data: np.ndarray
    product_present: np.ndarray
    unique_packet_id: int
    raw_seconds: float | None
    navg2: int
    tr_length: int
    metadata: SpectrumMetadata
    provenance: ProductProvenance
    units: str = field(init=False, default="unit_unestablished")
    representation: str = field(init=False, default="native_int32")

    def __post_init__(self) -> None:
        navg2 = _required_nonnegative(self.navg2, "navg2")
        tr_length = _required_nonnegative(self.tr_length, "tr_length")
        if navg2 == 0 or tr_length == 0:
            raise ValueError("navg2 and tr_length must be positive")
        if not isinstance(self.metadata, SpectrumMetadata):
            raise TypeError("metadata must be a SpectrumMetadata record")

        start = self.metadata.tr_start
        stop = self.metadata.tr_stop
        if stop <= start:
            raise ValueError("TR stop must be greater than TR start")
        span = stop - start
        average = 1 << self.metadata.tr_average_shift
        if span % average:
            raise ValueError(
                "TR span must be divisible by its averaging factor"
            )
        expected_navg2 = 1 << self.metadata.navg2_shift
        expected_tr_length = span // average
        if navg2 != expected_navg2:
            raise ValueError("navg2 disagrees with metadata navg2_shift")
        if tr_length != expected_tr_length:
            raise ValueError(
                "tr_length disagrees with metadata TR geometry"
            )

        data = _readonly_array(
            self.data,
            dtype=np.int32,
            shape=(NPRODUCTS, navg2, tr_length),
            name="TR spectrum data",
        )
        product_present = _readonly_array(
            self.product_present,
            dtype=np.bool_,
            shape=(NPRODUCTS,),
            name="TR spectrum product_present",
        )
        if not np.any(product_present):
            raise ValueError("TR spectrum must contain at least one product")
        if np.any(~product_present) and np.any(data[~product_present] != 0):
            raise ValueError(
                "absent TR product backing planes must contain only zero"
            )

        unique_packet_id = _optional_uid(
            self.unique_packet_id, "unique_packet_id"
        )
        if unique_packet_id is None:
            raise ValueError("unique_packet_id must be present")
        raw_seconds = _optional_finite_float(self.raw_seconds, "raw_seconds")
        if self.metadata.unique_packet_id != unique_packet_id:
            raise ValueError("metadata UID disagrees with TR spectrum UID")
        if self.metadata.raw_seconds != raw_seconds:
            raise ValueError("metadata split time disagrees with raw_seconds")
        _validate_concrete_product_provenance(
            self.provenance,
            unique_packet_id=unique_packet_id,
            raw_seconds=raw_seconds,
        )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "product_present", product_present)
        object.__setattr__(self, "unique_packet_id", unique_packet_id)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "navg2", navg2)
        object.__setattr__(self, "tr_length", tr_length)


@dataclass(frozen=True, slots=True)
class ZoomSample:
    """One exact repaired-uncrater zoom-spectrum row."""

    data: np.ndarray
    unique_packet_id: int
    pfb_bin: int
    raw_seconds: float | None
    provenance: ProductProvenance
    component_labels: tuple[str, ...] = field(
        init=False, default=("AA", "BB", "ABR", "ABI")
    )
    units: str = field(init=False, default="unit_unestablished")
    representation: str = field(init=False, default="native_float32")

    def __post_init__(self) -> None:
        data = _readonly_array(
            self.data,
            dtype=np.float32,
            shape=(len(self.component_labels), ZOOM_BINS),
            name="zoom data",
        )
        if not np.all(np.isfinite(data)):
            raise ValueError("zoom data must contain only finite values")
        pfb_bin = _required_uint(self.pfb_bin, 16, "pfb_bin")
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "zoom",
        )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "pfb_bin", pfb_bin)
        object.__setattr__(self, "raw_seconds", raw_seconds)


@dataclass(frozen=True, slots=True)
class WaveformSample:
    """One exact signed waveform with mission and ADC time kept distinct."""

    data: np.ndarray
    channel: int
    unique_packet_id: int
    raw_seconds: float | None
    adc_timestamp: np.uint64
    provenance: ProductProvenance
    units: str = field(init=False, default="raw_count")
    representation: str = field(init=False, default="native_int16")

    def __post_init__(self) -> None:
        data = _readonly_array(
            self.data,
            dtype=np.int16,
            shape=(WAVEFORM_SAMPLES,),
            name="waveform data",
        )
        channel = _required_uint(self.channel, 2, "channel")
        if type(self.adc_timestamp) is not np.uint64:
            raise TypeError("adc_timestamp must be a numpy.uint64")
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "waveform",
        )
        if len(self.provenance.source_packets) != 2:
            raise ValueError(
                "waveform provenance must link one waveform and one metadata packet"
            )
        expected_roles = (
            f"waveform_channel_{channel}",
            "waveform_metadata",
        )
        source_roles = tuple(
            packet.role for packet in self.provenance.source_packets
        )
        if source_roles != expected_roles:
            raise ValueError(
                "waveform provenance must identify its channel packet followed "
                "by waveform metadata"
            )
        if self.provenance.uid_source_role != "waveform_metadata":
            raise ValueError("waveform UID must come from waveform metadata")
        if raw_seconds is None:
            raise ValueError("waveform mission time must come from waveform metadata")
        if self.provenance.time_source_role != "waveform_metadata":
            raise ValueError("waveform mission time must come from waveform metadata")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "channel", channel)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)


@dataclass(frozen=True, slots=True)
class GrimmSample:
    """One exact native-integer Grimm product with honest index axes."""

    data: np.ndarray
    unique_packet_id: int
    raw_seconds: float | None
    navg2: int
    provenance: ProductProvenance
    axis_labels: tuple[str, ...] = field(
        init=False,
        default=("average_index", "product_index", "grimm_value_index"),
    )
    value_axis_labels: tuple[str, ...] = field(
        init=False,
        default=("value_0", "value_1", "value_2", "value_3"),
    )
    units: str = field(init=False, default="unit_unestablished")
    representation: str = field(init=False, default="native_int32")

    def __post_init__(self) -> None:
        navg2 = _required_nonnegative(self.navg2, "navg2")
        if navg2 == 0:
            raise ValueError("navg2 must be positive")
        data = _readonly_array(
            self.data,
            dtype=np.int32,
            shape=(navg2, NPRODUCTS, len(self.value_axis_labels)),
            name="Grimm data",
        )
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "Grimm",
        )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "navg2", navg2)


@dataclass(frozen=True, slots=True)
class HKSample:
    """One type-aware housekeeping row with explicit field presence."""

    hk_type: int
    version: int
    unique_packet_id: int
    errors: int
    raw_seconds: float | None
    fields: Mapping[str, object | None]
    field_present: Mapping[str, bool]
    provenance: ProductProvenance

    def __post_init__(self) -> None:
        hk_type = _required_uint(self.hk_type, 16, "hk_type")
        if hk_type not in (0, 1, 2, 3, 100, 101):
            raise ValueError("hk_type must be one of 0, 1, 2, 3, 100, or 101")
        version = _required_uint(self.version, 16, "version")
        errors = _required_uint(self.errors, 32, "errors")
        fields, field_present = _normalized_fields(
            self.fields,
            self.field_present,
            name="housekeeping",
        )
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "housekeeping",
        )
        if hk_type not in (0, 2):
            if raw_seconds is not None:
                raise ValueError(
                    f"housekeeping type {hk_type} does not carry mission time"
                )
            if any(
                field_present.get(name, False)
                for name in ("time_32", "time_16")
            ):
                raise ValueError(
                    f"housekeeping type {hk_type} does not carry split time"
                )
        if len(self.provenance.source_packets) != 1:
            raise ValueError("housekeeping provenance must identify one packet")
        object.__setattr__(self, "hk_type", hk_type)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "field_present", field_present)


@dataclass(frozen=True, slots=True)
class CalibratorMetadataSample:
    """Normalized public calibrator metadata without decoder-owned objects."""

    unique_packet_id: int
    raw_seconds: float | None
    from_debug: bool
    fields: Mapping[str, object | None]
    field_present: Mapping[str, bool]
    provenance: ProductProvenance

    def __post_init__(self) -> None:
        from_debug = _required_bool(self.from_debug, "from_debug")
        fields, field_present = _normalized_fields(
            self.fields,
            self.field_present,
            name="calibrator metadata",
        )
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "calibrator metadata",
        )
        expected_packets = 8 if from_debug else 1
        if len(self.provenance.source_packets) != expected_packets:
            raise ValueError(
                "calibrator metadata provenance has the wrong packet count"
            )
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "from_debug", from_debug)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "field_present", field_present)


@dataclass(frozen=True, slots=True)
class CalibratorDataSample:
    """One complete three-page calibrator data group."""

    data: np.ndarray
    g_nacc: int
    gphase: np.ndarray
    unique_packet_id: int
    raw_seconds: float | None
    page_raw_seconds: np.ndarray
    provenance: ProductProvenance
    channel_labels: tuple[int, ...] = field(init=False, default=(0, 1, 2, 3))
    page_count: int = field(init=False, default=3)
    units: str = field(init=False, default="unit_unestablished")
    representation: str = field(init=False, default="native_complex128")

    def __post_init__(self) -> None:
        data = _readonly_array(
            self.data,
            dtype=np.complex128,
            shape=(len(self.channel_labels), 512),
            name="calibrator data",
        )
        if not np.all(np.isfinite(data)):
            raise ValueError("calibrator data must contain only finite values")
        g_nacc = _required_int32(self.g_nacc, "g_nacc")
        gphase = _readonly_array(
            self.gphase,
            dtype=np.int32,
            shape=(1024,),
            name="calibrator gphase",
        )
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "calibrator data",
        )
        page_raw_seconds = _readonly_array(
            self.page_raw_seconds,
            dtype=np.float64,
            shape=(self.page_count,),
            name="calibrator data page_raw_seconds",
        )
        if not np.all(np.isfinite(page_raw_seconds)):
            raise ValueError(
                "calibrator data page_raw_seconds must be finite"
            )
        if raw_seconds != float(page_raw_seconds[0]):
            raise ValueError(
                "calibrator data raw_seconds must identify page zero"
            )
        if len(self.provenance.source_packets) != self.page_count:
            raise ValueError("calibrator data provenance must identify three pages")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "g_nacc", g_nacc)
        object.__setattr__(self, "gphase", gphase)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "page_raw_seconds", page_raw_seconds)


@dataclass(frozen=True, slots=True)
class CalibratorRawPFBSample:
    """One complete eight-page complex raw-PFB calibrator group."""

    data: np.ndarray
    unique_packet_id: int
    raw_seconds: float | None
    page_raw_seconds: np.ndarray
    provenance: ProductProvenance
    channel_labels: tuple[int, ...] = field(init=False, default=(0, 1, 2, 3))
    page_count: int = field(init=False, default=8)
    units: str = field(init=False, default="unit_unestablished")
    representation: str = field(init=False, default="native_complex128")

    def __post_init__(self) -> None:
        data = _readonly_array(
            self.data,
            dtype=np.complex128,
            shape=(len(self.channel_labels), NCHANNELS),
            name="calibrator raw PFB data",
        )
        if not np.all(np.isfinite(data)):
            raise ValueError("calibrator raw PFB data must contain only finite values")
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "calibrator raw PFB",
        )
        page_raw_seconds = _readonly_array(
            self.page_raw_seconds,
            dtype=np.float64,
            shape=(self.page_count,),
            name="calibrator raw PFB page_raw_seconds",
        )
        if not np.all(np.isfinite(page_raw_seconds)):
            raise ValueError(
                "calibrator raw PFB page_raw_seconds must be finite"
            )
        if raw_seconds != float(page_raw_seconds[0]):
            raise ValueError(
                "calibrator raw PFB raw_seconds must identify page zero"
            )
        if len(self.provenance.source_packets) != self.page_count:
            raise ValueError(
                "calibrator raw PFB provenance must identify eight pages"
            )
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "page_raw_seconds", page_raw_seconds)


@dataclass(frozen=True, slots=True)
class CalibratorDebugPage:
    """One normalized public page in a calibrator debug group."""

    page: int
    fields: Mapping[str, object | None]
    field_present: Mapping[str, bool]

    def __post_init__(self) -> None:
        page = _required_uint(self.page, 3, "calibrator debug page")
        fields, field_present = _normalized_fields(
            self.fields,
            self.field_present,
            name=f"calibrator debug page {page}",
        )
        object.__setattr__(self, "page", page)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "field_present", field_present)


@dataclass(frozen=True, slots=True)
class CalibratorDebugSample:
    """One complete eight-page calibrator debug group."""

    pages: tuple[CalibratorDebugPage, ...]
    unique_packet_id: int
    raw_seconds: float | None
    page_raw_seconds: np.ndarray
    provenance: ProductProvenance
    page_count: int = field(init=False, default=8)

    def __post_init__(self) -> None:
        if type(self.pages) is not tuple or any(
            not isinstance(page, CalibratorDebugPage) for page in self.pages
        ):
            raise TypeError("pages must be a tuple of CalibratorDebugPage records")
        if len(self.pages) != self.page_count:
            raise ValueError("calibrator debug group must contain eight pages")
        if tuple(page.page for page in self.pages) != tuple(range(self.page_count)):
            raise ValueError("calibrator debug pages must be ordered 0 through 7")
        uid, raw_seconds = _auxiliary_identity(
            self.unique_packet_id,
            self.raw_seconds,
            self.provenance,
            "calibrator debug",
        )
        page_raw_seconds = _readonly_array(
            self.page_raw_seconds,
            dtype=np.float64,
            shape=(self.page_count,),
            name="calibrator debug page_raw_seconds",
        )
        if not np.all(np.isfinite(page_raw_seconds)):
            raise ValueError(
                "calibrator debug page_raw_seconds must be finite"
            )
        if raw_seconds != float(page_raw_seconds[0]):
            raise ValueError(
                "calibrator debug raw_seconds must identify page zero"
            )
        if len(self.provenance.source_packets) != self.page_count:
            raise ValueError(
                "calibrator debug provenance must identify eight pages"
            )
        object.__setattr__(self, "unique_packet_id", uid)
        object.__setattr__(self, "raw_seconds", raw_seconds)
        object.__setattr__(self, "page_raw_seconds", page_raw_seconds)


@dataclass(frozen=True, slots=True)
class DecodeProvenance:
    """Immutable decoder identity and canonical Collection report."""

    decoder_name: str | None
    distribution_version: str | None
    decoder_source_commit: str | None
    reported_schema_ids: tuple[int, ...]
    selected_schema_id: int | None
    binding_key: str | None
    schema_variant: str | None
    schema_assumed: bool
    binding_source_release: str | None
    binding_source_commit: str | None
    abi_fingerprint: str | None
    execution_mode: ExecutionMode | None
    input_packet_count: int
    valid_packet_count: int | None
    appid_counts: tuple[tuple[int, int], ...]
    issue_counts: tuple[tuple[str, int], ...]
    canonical_report_json: str | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        decoder_name = _optional_text(self.decoder_name, "decoder_name")
        distribution_version = _optional_text(
            self.distribution_version, "distribution_version"
        )
        decoder_source_commit = self._optional_sha(
            self.decoder_source_commit, 40, "decoder_source_commit"
        )
        reported = _schema_ids(self.reported_schema_ids, "reported_schema_ids")
        selected = self.selected_schema_id
        if selected is not None:
            if type(selected) is not int or not 0 <= selected <= 0xFFFF:
                raise ValueError("selected_schema_id must be an unsigned 16-bit value")
        binding_key = _optional_text(self.binding_key, "binding_key")
        schema_variant = _optional_text(self.schema_variant, "schema_variant")
        if type(self.schema_assumed) is not bool:
            raise TypeError("schema_assumed must be a boolean")
        binding_source_release = _optional_text(
            self.binding_source_release, "binding_source_release"
        )
        binding_source_commit = self._optional_sha(
            self.binding_source_commit, 40, "binding_source_commit"
        )
        abi_fingerprint = self._optional_sha(
            self.abi_fingerprint, 64, "abi_fingerprint"
        )
        mode = (
            None
            if self.execution_mode is None
            else ExecutionMode(self.execution_mode)
        )
        input_count = _required_nonnegative(
            self.input_packet_count, "input_packet_count"
        )
        valid_count = _optional_nonnegative(
            self.valid_packet_count, "valid_packet_count"
        )
        if valid_count is not None and valid_count > input_count:
            raise ValueError("valid_packet_count exceeds input_packet_count")
        appid_counts = _count_pairs(
            self.appid_counts, key_type=int, name="appid_counts"
        )
        issue_counts = _count_pairs(
            self.issue_counts, key_type=str, name="issue_counts"
        )
        if sum(count for _, count in appid_counts) != input_count:
            raise ValueError("appid_counts do not sum to input_packet_count")
        canonical_report = self.canonical_report_json
        if canonical_report is not None:
            try:
                decoded = json.loads(canonical_report)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("canonical_report_json must be valid JSON") from exc
            if not isinstance(decoded, dict):
                raise ValueError("canonical_report_json must encode an object")
            normalized = json.dumps(
                decoded,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            if canonical_report != normalized:
                raise ValueError("canonical_report_json must use canonical JSON encoding")
        unavailable_reason = _optional_text(
            self.unavailable_reason, "unavailable_reason"
        )
        if unavailable_reason is not None and (
            input_count
            or selected is not None
            or binding_key is not None
            or canonical_report is not None
        ):
            raise ValueError(
                "unavailable decoder provenance cannot contain decoded input"
            )
        if unavailable_reason is None:
            required_concrete = {
                "decoder_name": decoder_name,
                "selected_schema_id": selected,
                "binding_key": binding_key,
                "binding_source_release": binding_source_release,
                "binding_source_commit": binding_source_commit,
                "abi_fingerprint": abi_fingerprint,
                "execution_mode": mode,
                "valid_packet_count": valid_count,
                "canonical_report_json": canonical_report,
            }
            missing = [
                name for name, field_value in required_concrete.items()
                if field_value is None
            ]
            if missing:
                raise ValueError(
                    "available decoder provenance requires concrete field(s): "
                    + ", ".join(missing)
                )
        object.__setattr__(self, "decoder_name", decoder_name)
        object.__setattr__(self, "distribution_version", distribution_version)
        object.__setattr__(self, "decoder_source_commit", decoder_source_commit)
        object.__setattr__(self, "reported_schema_ids", reported)
        object.__setattr__(self, "binding_key", binding_key)
        object.__setattr__(self, "schema_variant", schema_variant)
        object.__setattr__(self, "binding_source_release", binding_source_release)
        object.__setattr__(self, "binding_source_commit", binding_source_commit)
        object.__setattr__(self, "abi_fingerprint", abi_fingerprint)
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "input_packet_count", input_count)
        object.__setattr__(self, "valid_packet_count", valid_count)
        object.__setattr__(self, "appid_counts", appid_counts)
        object.__setattr__(self, "issue_counts", issue_counts)
        object.__setattr__(self, "unavailable_reason", unavailable_reason)

    @staticmethod
    def _optional_sha(value: str | None, length: int, name: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not re.fullmatch(
            rf"[0-9a-f]{{{length}}}", value
        ):
            raise ValueError(f"{name} must be a lowercase {length}-hex digest")
        return value

    @classmethod
    def unavailable(
        cls,
        reason: str = "caller_constructed",
        *,
        execution_mode: ExecutionMode | str | None = None,
    ) -> "DecodeProvenance":
        """Return explicit provenance for products not built by a decoder."""
        return cls(
            decoder_name=None,
            distribution_version=None,
            decoder_source_commit=None,
            reported_schema_ids=(),
            selected_schema_id=None,
            binding_key=None,
            schema_variant=None,
            schema_assumed=False,
            binding_source_release=None,
            binding_source_commit=None,
            abi_fingerprint=None,
            execution_mode=(
                None if execution_mode is None else ExecutionMode(execution_mode)
            ),
            input_packet_count=0,
            valid_packet_count=None,
            appid_counts=(),
            issue_counts=(),
            canonical_report_json=None,
            unavailable_reason=reason,
        )

    @classmethod
    def from_report(
        cls,
        *,
        distribution_version: str | None,
        decoder_source_commit: str | None,
        reported_schema_ids: tuple[int, ...],
        selected_schema_id: int,
        binding_key: str,
        schema_variant: str | None,
        schema_assumed: bool,
        binding_source_release: str,
        binding_source_commit: str,
        abi_fingerprint: str,
        execution_mode: ExecutionMode | str,
        input_packet_count: int,
        valid_packet_count: int,
        appid_counts: tuple[tuple[int, int], ...],
        issue_counts: tuple[tuple[str, int], ...],
        canonical_report: Mapping[str, object],
    ) -> "DecodeProvenance":
        """Build one record from the reviewed public uncrater report."""
        canonical_report_json = json.dumps(
            canonical_report,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return cls(
            decoder_name="uncrater",
            distribution_version=distribution_version,
            decoder_source_commit=decoder_source_commit,
            reported_schema_ids=reported_schema_ids,
            selected_schema_id=selected_schema_id,
            binding_key=binding_key,
            schema_variant=schema_variant,
            schema_assumed=schema_assumed,
            binding_source_release=binding_source_release,
            binding_source_commit=binding_source_commit,
            abi_fingerprint=abi_fingerprint,
            execution_mode=ExecutionMode(execution_mode),
            input_packet_count=input_packet_count,
            valid_packet_count=valid_packet_count,
            appid_counts=appid_counts,
            issue_counts=issue_counts,
            canonical_report_json=canonical_report_json,
        )

    def canonical_report(self) -> dict[str, object] | None:
        """Return a fresh decoded copy of the canonical report."""
        if self.canonical_report_json is None:
            return None
        return json.loads(self.canonical_report_json)


@dataclass(frozen=True, slots=True)
class ValidatedCounts:
    """Counts at decode/product/write boundaries without conflating them."""

    input_packets: int = 0
    valid_packets: int | None = None
    product_rows: tuple[tuple[str, int], ...] = ()
    persisted_rows: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self) -> None:
        input_packets = _required_nonnegative(self.input_packets, "input_packets")
        valid_packets = _optional_nonnegative(self.valid_packets, "valid_packets")
        if valid_packets is not None and valid_packets > input_packets:
            raise ValueError("valid_packets exceeds input_packets")
        product_rows = _count_pairs(
            self.product_rows, key_type=str, name="product_rows"
        )
        persisted_rows = self.persisted_rows
        if persisted_rows is not None:
            persisted_rows = _count_pairs(
                persisted_rows, key_type=str, name="persisted_rows"
            )
        object.__setattr__(self, "input_packets", input_packets)
        object.__setattr__(self, "valid_packets", valid_packets)
        object.__setattr__(self, "product_rows", product_rows)
        object.__setattr__(self, "persisted_rows", persisted_rows)
