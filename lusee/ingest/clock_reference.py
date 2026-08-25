"""Validated raw-clock references for layout-v4 ingestion."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

CLOCK_REFERENCE_FORMAT_VERSION = 1
_CLOCK_REFERENCE_KEYS = frozenset({
    "format_version",
    "reference_event",
    "clock_reference_isot",
    "time_scale",
    "clocks",
    "source",
    "assumed",
})
_CLOCK_REFERENCE_RECORD_KEYS = _CLOCK_REFERENCE_KEYS | frozenset({
    "source_sha256",
})
_CLOCK_KEYS = frozenset({"clock_reference_raw_seconds"})
_ISOT_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?$"
)


class ClockReferenceFormatError(ValueError):
    """An external clock-reference file violates its versioned contract."""


class UnsupportedClockSourceError(ValueError):
    """A caller named a clock outside the supported mission clock domains."""


class ClockReferenceUnavailableError(ValueError):
    """A supported clock has no anchor in this reference set."""


class ClockSource(StrEnum):
    """Clock domains that product records may identify."""

    SPECTROMETER = "spectrometer"
    DCB = "dcb"
    ADC = "adc"


_ANCHORABLE_CLOCK_SOURCES = frozenset({
    ClockSource.SPECTROMETER,
    ClockSource.DCB,
})
_ABSOLUTE_TIME_SCALES = frozenset({
    "tai", "tcb", "tcg", "tdb", "tt", "ut1", "utc",
})


@dataclass(frozen=True, slots=True)
class ClockReference:
    """One raw counter value measured at the shared reference event."""

    clock_source: ClockSource
    clock_reference_raw_seconds: float

    def __post_init__(self) -> None:
        source = ClockSource(self.clock_source)
        if source not in _ANCHORABLE_CLOCK_SOURCES:
            raise ValueError(f"clock {source.value!r} cannot be anchored in format 1")
        raw_seconds = _finite_number(
            self.clock_reference_raw_seconds,
            f"$.clocks.{source.value}.clock_reference_raw_seconds",
        )
        object.__setattr__(self, "clock_source", source)
        object.__setattr__(
            self,
            "clock_reference_raw_seconds",
            0.0 if raw_seconds == 0.0 else raw_seconds,
        )


@dataclass(frozen=True, slots=True)
class ClockReferenceSet:
    """Immutable, input-file-scoped raw-clock-to-absolute-time mapping."""

    format_version: int
    reference_event: str
    clock_reference_isot: str
    time_scale: str
    clocks: tuple[ClockReference, ...]
    source: str
    assumed: bool
    source_sha256: str

    def __post_init__(self) -> None:
        if type(self.format_version) is not int:
            raise TypeError("format_version must be an integer")
        if self.format_version != CLOCK_REFERENCE_FORMAT_VERSION:
            raise ValueError(
                "clock-reference format_version must be "
                f"{CLOCK_REFERENCE_FORMAT_VERSION}"
            )
        if self.reference_event != "landing":
            raise ValueError("reference_event must be 'landing'")
        _validate_isot(self.clock_reference_isot, self.time_scale)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a nonempty string")
        if type(self.assumed) is not bool:
            raise TypeError("assumed must be a boolean")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")

        clocks = tuple(self.clocks)
        if any(not isinstance(item, ClockReference) for item in clocks):
            raise TypeError("clocks must contain ClockReference records")
        normalized = tuple(sorted(clocks, key=lambda item: item.clock_source.value))
        sources = tuple(item.clock_source for item in normalized)
        if len(set(sources)) != len(sources):
            raise ValueError("clock-reference set contains duplicate clock sources")
        object.__setattr__(self, "clocks", normalized)

    def reference_for(
        self,
        clock_source: ClockSource | str,
    ) -> ClockReference | None:
        """Return one supported clock anchor, or None when it is unmapped."""
        source = _normalize_clock_source(clock_source)
        for reference in self.clocks:
            if reference.clock_source is source:
                return reference
        return None

    def require_reference(
        self,
        clock_source: ClockSource | str,
    ) -> ClockReference:
        """Return one clock anchor without borrowing another clock's anchor."""
        source = _normalize_clock_source(clock_source)
        reference = self.reference_for(source)
        if reference is None:
            raise ClockReferenceUnavailableError(
                f"clock reference does not cover {source.value!r}"
            )
        return reference

    def to_time(
        self,
        raw_seconds: float | np.ndarray,
        *,
        clock_source: ClockSource | str,
    ):
        """Convert raw seconds using the sole layout-v4 clock equation."""
        from astropy.time import Time, TimeDelta

        reference = self.require_reference(clock_source)
        raw = _finite_array(raw_seconds)
        absolute_reference = Time(
            self.clock_reference_isot,
            format="isot",
            scale=self.time_scale,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            delta_seconds = raw - reference.clock_reference_raw_seconds
        if not np.all(np.isfinite(delta_seconds)):
            raise ValueError("raw_seconds produces an unrepresentable time delta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = absolute_reference + TimeDelta(
                delta_seconds,
                format="sec",
            )
        if not (
            np.all(np.isfinite(result.jd1))
            and np.all(np.isfinite(result.jd2))
        ):
            raise ValueError("raw_seconds produces an unrepresentable absolute time")
        return result

    def to_mjd(
        self,
        raw_seconds: float | np.ndarray,
        *,
        clock_source: ClockSource | str,
    ) -> float | np.ndarray:
        """Derive MJD from the same absolute-time conversion."""
        return self.to_time(raw_seconds, clock_source=clock_source).mjd

    def as_payload(self) -> dict[str, object]:
        """Return the normalized external format-1 payload."""
        return {
            "format_version": self.format_version,
            "reference_event": self.reference_event,
            "clock_reference_isot": self.clock_reference_isot,
            "time_scale": self.time_scale,
            "clocks": {
                reference.clock_source.value: {
                    "clock_reference_raw_seconds": (
                        reference.clock_reference_raw_seconds
                    ),
                }
                for reference in self.clocks
            },
            "source": self.source,
            "assumed": self.assumed,
        }

    def canonical_json_bytes(self) -> bytes:
        """Return deterministic semantic bytes, excluding the source digest."""
        return json.dumps(
            self.as_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

    def as_record(self) -> dict[str, object]:
        """Return the normalized payload plus exact source-file provenance."""
        record = self.as_payload()
        record["source_sha256"] = self.source_sha256
        return record


@dataclass(frozen=True, slots=True)
class LegacyClockReferenceSet:
    """Explicit v2/v3 subtract-plus-MJD mapping, never a landing claim."""

    clock_reference_raw_seconds: float
    mjd_epoch_offset_days: float
    time_scale: str
    source: str
    assumed: bool
    mapping_sha256: str
    reference_event: str = "legacy_mjd_offset"
    verified_landing: bool = False

    def __post_init__(self) -> None:
        raw = _finite_number(
            self.clock_reference_raw_seconds,
            "clock_reference_raw_seconds",
        )
        mjd = _finite_number(
            self.mjd_epoch_offset_days,
            "mjd_epoch_offset_days",
        )
        if not isinstance(self.time_scale, str) or (
            self.time_scale.lower() not in _ABSOLUTE_TIME_SCALES
        ):
            raise ValueError("legacy time_scale is not an absolute time scale")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("legacy clock source must be nonempty")
        if type(self.assumed) is not bool:
            raise TypeError("legacy clock assumed must be a boolean")
        if not re.fullmatch(r"[0-9a-f]{64}", self.mapping_sha256):
            raise ValueError("mapping_sha256 must be a lowercase SHA-256 digest")
        if self.reference_event != "legacy_mjd_offset":
            raise ValueError("legacy reference_event must be 'legacy_mjd_offset'")
        if self.verified_landing is not False:
            raise ValueError("legacy clock mappings cannot verify landing")
        from astropy.time import Time, TimeDelta

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anchor = Time(
                    mjd,
                    format="mjd",
                    scale=self.time_scale.lower(),
                ) + TimeDelta(0.0, format="sec")
        except Exception as exc:
            raise ValueError(
                "legacy MJD anchor is not representable"
            ) from exc
        if not (
            np.all(np.isfinite(anchor.jd1))
            and np.all(np.isfinite(anchor.jd2))
        ):
            raise ValueError("legacy MJD anchor is not representable")
        object.__setattr__(self, "clock_reference_raw_seconds", raw)
        object.__setattr__(self, "mjd_epoch_offset_days", mjd)
        object.__setattr__(self, "time_scale", self.time_scale.lower())

    @property
    def clocks(self) -> tuple[ClockReference, ...]:
        """Expose the sole legacy spectrometer anchor without cross-clock reuse."""
        return (
            ClockReference(
                clock_source=ClockSource.SPECTROMETER,
                clock_reference_raw_seconds=self.clock_reference_raw_seconds,
            ),
        )

    def reference_for(
        self,
        clock_source: ClockSource | str,
    ) -> ClockReference | None:
        """Return the legacy spectrometer anchor and no other clock anchor."""
        source = _normalize_clock_source(clock_source)
        return self.clocks[0] if source is ClockSource.SPECTROMETER else None

    def require_reference(
        self,
        clock_source: ClockSource | str,
    ) -> ClockReference:
        """Require the sole legacy spectrometer mapping."""
        reference = self.reference_for(clock_source)
        if reference is None:
            source = _normalize_clock_source(clock_source)
            raise ClockReferenceUnavailableError(
                f"legacy clock reference does not cover {source.value!r}"
            )
        return reference

    def to_time(
        self,
        raw_seconds: float | np.ndarray,
        *,
        clock_source: ClockSource | str,
    ):
        """Apply the recorded legacy subtract-plus-MJD equation exactly."""
        from astropy.time import Time, TimeDelta

        self.require_reference(clock_source)
        raw = _finite_array(raw_seconds)
        with np.errstate(over="ignore", invalid="ignore"):
            delta = raw - self.clock_reference_raw_seconds
        if not np.all(np.isfinite(delta)):
            raise ValueError("raw_seconds produces an unrepresentable time delta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = Time(
                self.mjd_epoch_offset_days,
                format="mjd",
                scale=self.time_scale,
            ) + TimeDelta(delta, format="sec")
        if not (
            np.all(np.isfinite(result.jd1))
            and np.all(np.isfinite(result.jd2))
        ):
            raise ValueError("raw_seconds produces an unrepresentable absolute time")
        return result

    def to_mjd(
        self,
        raw_seconds: float | np.ndarray,
        *,
        clock_source: ClockSource | str,
    ) -> float | np.ndarray:
        """Return MJD from the explicit legacy equation."""
        return self.to_time(raw_seconds, clock_source=clock_source).mjd

    def as_record(self) -> dict[str, object]:
        """Return explicit legacy provenance without production-v4 labels."""
        return {
            "format": "layout_v2_v3_legacy_clock_mapping",
            "reference_event": self.reference_event,
            "verified_landing": self.verified_landing,
            "clock_source": ClockSource.SPECTROMETER.value,
            "clock_reference_raw_seconds": self.clock_reference_raw_seconds,
            "mjd_epoch_offset_days": self.mjd_epoch_offset_days,
            "time_scale": self.time_scale,
            "source": self.source,
            "assumed": self.assumed,
            "mapping_sha256": self.mapping_sha256,
        }


def load_clock_reference_set(path: Path | str) -> ClockReferenceSet:
    """Read and validate one explicit version-1 clock-reference JSON file."""
    source_path = Path(path)
    raw = source_path.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ClockReferenceFormatError("$: file must be valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ClockReferenceFormatError) as exc:
        if isinstance(exc, ClockReferenceFormatError):
            raise
        raise ClockReferenceFormatError(f"$: invalid JSON: {exc.msg}") from exc

    try:
        return _clock_reference_from_json(value, source_sha256)
    except ClockReferenceFormatError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise ClockReferenceFormatError(f"$: {exc}") from exc


def clock_reference_set_from_record(value: object) -> ClockReferenceSet:
    """Validate a normalized clock-reference record embedded in a manifest."""
    if not isinstance(value, dict):
        raise ClockReferenceFormatError("$: expected an object")
    _require_exact_keys(value, _CLOCK_REFERENCE_RECORD_KEYS, "$")
    payload = {key: value[key] for key in _CLOCK_REFERENCE_KEYS}
    try:
        references = _clock_reference_from_json(
            payload,
            value["source_sha256"],
        )
    except ClockReferenceFormatError:
        raise
    except (OverflowError, TypeError, ValueError) as exc:
        raise ClockReferenceFormatError(f"$: {exc}") from exc
    normalized = json.dumps(
        references.as_record(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    supplied = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if normalized != supplied:
        raise ClockReferenceFormatError(
            "$: embedded clock-reference record is not normalized"
        )
    return references


def _clock_reference_from_json(
    value: Any,
    source_sha256: str,
) -> ClockReferenceSet:
    if not isinstance(value, dict):
        raise ClockReferenceFormatError("$: expected an object")
    _require_exact_keys(value, _CLOCK_REFERENCE_KEYS, "$")

    version = value["format_version"]
    if type(version) is not int or version != CLOCK_REFERENCE_FORMAT_VERSION:
        raise ClockReferenceFormatError(
            f"$.format_version: expected integer {CLOCK_REFERENCE_FORMAT_VERSION}"
        )
    if value["reference_event"] != "landing":
        raise ClockReferenceFormatError(
            "$.reference_event: expected 'landing'"
        )
    reference_isot = value["clock_reference_isot"]
    time_scale = value["time_scale"]
    try:
        _validate_isot(reference_isot, time_scale)
    except (TypeError, ValueError) as exc:
        raise ClockReferenceFormatError(f"$.clock_reference_isot: {exc}") from exc
    source = value["source"]
    if not isinstance(source, str) or not source.strip():
        raise ClockReferenceFormatError("$.source: expected a nonempty string")
    assumed = value["assumed"]
    if type(assumed) is not bool:
        raise ClockReferenceFormatError("$.assumed: expected a boolean")

    raw_clocks = value["clocks"]
    if not isinstance(raw_clocks, dict):
        raise ClockReferenceFormatError("$.clocks: expected an object")
    clocks = []
    for name, raw_clock in raw_clocks.items():
        try:
            clock_source = ClockSource(name)
        except (TypeError, ValueError) as exc:
            raise ClockReferenceFormatError(
                f"$.clocks.{name}: unsupported clock source"
            ) from exc
        if clock_source not in _ANCHORABLE_CLOCK_SOURCES:
            raise ClockReferenceFormatError(
                f"$.clocks.{name}: clock cannot be anchored in format 1"
            )
        if not isinstance(raw_clock, dict):
            raise ClockReferenceFormatError(
                f"$.clocks.{name}: expected an object"
            )
        _require_exact_keys(raw_clock, _CLOCK_KEYS, f"$.clocks.{name}")
        try:
            clocks.append(ClockReference(
                clock_source=clock_source,
                clock_reference_raw_seconds=raw_clock[
                    "clock_reference_raw_seconds"
                ],
            ))
        except (TypeError, ValueError) as exc:
            raise ClockReferenceFormatError(str(exc)) from exc

    return ClockReferenceSet(
        format_version=version,
        reference_event="landing",
        clock_reference_isot=reference_isot,
        time_scale=time_scale,
        clocks=tuple(clocks),
        source=source,
        assumed=assumed,
        source_sha256=source_sha256,
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ClockReferenceFormatError(f"$: duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ClockReferenceFormatError(f"$: invalid JSON number {value}")


def _require_exact_keys(
    value: dict[str, Any],
    expected: frozenset[str],
    path: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise ClockReferenceFormatError(
            f"{path}: missing required field(s): {', '.join(missing)}"
        )
    if extra:
        raise ClockReferenceFormatError(
            f"{path}: unknown field(s): {', '.join(extra)}"
        )


def _validate_isot(reference_isot: object, time_scale: object) -> None:
    if not isinstance(time_scale, str) or time_scale not in _ABSOLUTE_TIME_SCALES:
        raise ValueError(
            "time_scale must be one of " + ", ".join(sorted(_ABSOLUTE_TIME_SCALES))
        )
    if not isinstance(reference_isot, str) or not _ISOT_RE.fullmatch(reference_isot):
        raise ValueError(
            "clock_reference_isot must use YYYY-MM-DDTHH:MM:SS[.fraction]"
        )
    from astropy.time import Time

    try:
        Time(reference_isot, format="isot", scale=time_scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("clock_reference_isot is invalid") from exc


def _normalize_clock_source(value: ClockSource | str) -> ClockSource:
    try:
        return ClockSource(value)
    except (TypeError, ValueError) as exc:
        raise UnsupportedClockSourceError(
            f"unsupported clock source {value!r}"
        ) from exc


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}: expected a numeric value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{path}: value must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: value must be finite")
    return result


def _finite_array(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind == "b":
        raise TypeError("raw_seconds must be numeric, not boolean")
    if raw.dtype.kind not in "iuf":
        raise TypeError("raw_seconds must be numeric")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("raw_seconds must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("raw_seconds must contain only finite values")
    return result
