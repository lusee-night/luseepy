"""Immutable provenance and validation records for decoded ingest products."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .clock_reference import ClockSource


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
