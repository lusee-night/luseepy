"""Public boundary for the optional private telemetry decoder.

0x314 packets or the legacy binary sidecar
    -> optional lusee_telemetry
    -> one fixed 57-column TelemetryData
    -> session slice
    -> HDF5/FITS

No private decoder: warn once, return no TelemetryData, keep science output.
0x325 is not part of this path.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .clock_reference import ClockReferenceSet, ClockSource


TELEMETRY_APPID = 0x314
TELEMETRY_FIELD_COUNT = 57
LEGACY_TELEMETRY_SIDECAR_NAME = "DCB_telemetry.json"
RESULT_KEYS = frozenset({
    "field_names",
    "units",
    "source_indices",
    "mission_seconds",
    "lusee_subsecs",
    "raw_counts",
    "values",
    "valid",
})

try:
    import lusee_telemetry as private_decoder
except Exception as exc:  # noqa: BLE001
    private_decoder = None
    decoder_import_error = exc
else:
    decoder_import_error = None


def exact_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int | None, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ValueError(f"{name} has invalid shape {value.shape}")
    return value


@dataclass(frozen=True, slots=True)
class TelemetryData:
    """One fixed row-aligned 57-field DCB telemetry table."""

    source_kind: str
    field_names: tuple[str, ...]
    units: tuple[str, ...]
    source_indices: np.ndarray
    mission_seconds: np.ndarray
    lusee_subsecs: np.ndarray
    mjd_times: np.ndarray
    raw_counts: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.source_kind not in (
            "b01_0x314",
            "legacy_binary_sidecar",
        ):
            raise ValueError(f"unknown telemetry source {self.source_kind!r}")
        field_names = tuple(self.field_names)
        units = tuple(self.units)
        if (
            len(field_names) != TELEMETRY_FIELD_COUNT
            or any(not isinstance(name, str) or not name for name in field_names)
            or len(set(field_names)) != TELEMETRY_FIELD_COUNT
        ):
            raise ValueError("telemetry requires 57 unique nonempty field names")
        if (
            len(units) != TELEMETRY_FIELD_COUNT
            or any(not isinstance(unit, str) for unit in units)
        ):
            raise ValueError("telemetry requires one unit per field")
        for label, strings in (("field name", field_names), ("unit", units)):
            for value in strings:
                if "\x00" in value:
                    raise ValueError(f"telemetry {label}s must not contain NUL")
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        f"telemetry {label}s must be valid UTF-8"
                    ) from exc
        mission_seconds = exact_array(
            self.mission_seconds,
            name="telemetry mission_seconds",
            dtype=np.uint32,
            shape=(None,),
        )
        row_count = mission_seconds.shape[0]
        arrays = {
            "source_indices": exact_array(
                self.source_indices,
                name="telemetry source_indices",
                dtype=np.int64,
                shape=(row_count,),
            ),
            "mission_seconds": mission_seconds,
            "lusee_subsecs": exact_array(
                self.lusee_subsecs,
                name="telemetry lusee_subsecs",
                dtype=np.uint16,
                shape=(row_count,),
            ),
            "mjd_times": exact_array(
                self.mjd_times,
                name="telemetry mjd_times",
                dtype=np.float64,
                shape=(row_count,),
            ),
            "raw_counts": exact_array(
                self.raw_counts,
                name="telemetry raw_counts",
                dtype=np.uint16,
                shape=(row_count, TELEMETRY_FIELD_COUNT),
            ),
            "values": exact_array(
                self.values,
                name="telemetry values",
                dtype=np.float64,
                shape=(row_count, TELEMETRY_FIELD_COUNT),
            ),
            "valid": exact_array(
                self.valid,
                name="telemetry valid",
                dtype=np.bool_,
                shape=(row_count, TELEMETRY_FIELD_COUNT),
            ),
        }
        if np.any(arrays["source_indices"] < 0) or (
            row_count > 1
            and np.any(np.diff(arrays["source_indices"]) <= 0)
        ):
            raise ValueError(
                "telemetry source_indices must be nonnegative and increasing"
            )
        if not np.array_equal(
            np.isfinite(arrays["values"]),
            arrays["valid"],
        ):
            raise ValueError(
                "telemetry values must be finite exactly where valid is true"
            )
        if np.any(~arrays["valid"] & ~np.isnan(arrays["values"])):
            raise ValueError("invalid telemetry values must be NaN")
        if np.any(np.isinf(arrays["mjd_times"])):
            raise ValueError("telemetry mjd_times may contain only finite or NaN")
        object.__setattr__(self, "field_names", field_names)
        object.__setattr__(self, "units", units)

    @property
    def row_count(self) -> int:
        return self.mission_seconds.shape[0]

    @property
    def raw_seconds(self) -> np.ndarray:
        ticks = (
            self.mission_seconds.astype(np.uint64) * np.uint64(65536)
            + self.lusee_subsecs.astype(np.uint64)
        )
        return ticks.astype(np.float64) / 65536.0

    def slice_rows(self, selector: np.ndarray) -> TelemetryData:
        if (
            type(selector) is not np.ndarray
            or selector.dtype != np.dtype(np.bool_)
            or selector.shape != (self.row_count,)
        ):
            raise ValueError("telemetry row selector must be a matching bool array")
        return TelemetryData(
            source_kind=self.source_kind,
            field_names=self.field_names,
            units=self.units,
            source_indices=self.source_indices[selector],
            mission_seconds=self.mission_seconds[selector],
            lusee_subsecs=self.lusee_subsecs[selector],
            mjd_times=self.mjd_times[selector],
            raw_counts=self.raw_counts[selector],
            values=self.values[selector],
            valid=self.valid[selector],
        )

    def with_mjd_times(self, mjd_times: np.ndarray) -> TelemetryData:
        return replace(self, mjd_times=mjd_times)


def telemetry_from_result(
    result: object,
    *,
    source_kind: str,
) -> TelemetryData:
    if not isinstance(result, Mapping) or frozenset(result) != RESULT_KEYS:
        raise ValueError("private telemetry result has the wrong fields")
    mission_seconds = result["mission_seconds"]
    if type(mission_seconds) is not np.ndarray or mission_seconds.ndim != 1:
        raise TypeError("private telemetry mission_seconds must be a vector")
    return TelemetryData(
        source_kind=source_kind,
        field_names=result["field_names"],
        units=result["units"],
        source_indices=result["source_indices"],
        mission_seconds=mission_seconds,
        lusee_subsecs=result["lusee_subsecs"],
        mjd_times=np.full(mission_seconds.shape, np.nan, dtype=np.float64),
        raw_counts=result["raw_counts"],
        values=result["values"],
        valid=result["valid"],
    )


def selected_decoder(decoder: object | None) -> object | None:
    return private_decoder if decoder is None else decoder


def warn_skipped(source: str, reason: object) -> None:
    warnings.warn(
        f"{source} telemetry skipped: {reason}",
        stacklevel=3,
    )


def decode_b01_packets(
    logical_packets: Iterable[Any],
    *,
    decoder: object | None = None,
) -> TelemetryData | None:
    """Decode filtered logical 0x314 packets, or warn and return None."""
    packets = tuple(
        packet
        for packet in logical_packets
        if getattr(packet, "appid", None) == TELEMETRY_APPID
    )
    if not packets:
        return None
    loaded = selected_decoder(decoder)
    if loaded is None:
        warn_skipped("b01", decoder_import_error or "decoder is unavailable")
        return None
    try:
        result = loaded.decode_b01_packets(packets)
        return telemetry_from_result(result, source_kind="b01_0x314")
    except Exception as exc:  # noqa: BLE001
        warn_skipped("b01", exc)
        return None


def decode_legacy_sidecar(
    path: Path | str,
    *,
    decoder: object | None = None,
) -> TelemetryData | None:
    """Decode the recognized binary sidecar, or warn and return None."""
    loaded = selected_decoder(decoder)
    if loaded is None:
        warn_skipped(
            "legacy sidecar",
            decoder_import_error or "decoder is unavailable",
        )
        return None
    try:
        result = loaded.decode_legacy_sidecar(Path(path))
        return telemetry_from_result(
            result,
            source_kind="legacy_binary_sidecar",
        )
    except Exception as exc:  # noqa: BLE001
        warn_skipped("legacy sidecar", exc)
        return None


def find_legacy_sidecar(session_dir: Path | str) -> Path | None:
    """Return the recognized historic sidecar path when it exists."""
    path = Path(session_dir) / LEGACY_TELEMETRY_SIDECAR_NAME
    return path if path.is_file() else None


def map_dcb_absolute_time(
    telemetry: TelemetryData,
    *,
    clock_reference_set: ClockReferenceSet,
) -> TelemetryData:
    """Map integer DCB time through the explicit DCB clock reference."""
    if not isinstance(telemetry, TelemetryData):
        raise TypeError("telemetry must be TelemetryData")
    if not isinstance(clock_reference_set, ClockReferenceSet):
        raise TypeError("clock_reference_set must be ClockReferenceSet")
    dcb_reference = clock_reference_set.reference_for(ClockSource.DCB)
    if dcb_reference is None:
        if np.isfinite(telemetry.mjd_times).any():
            raise ValueError("telemetry has absolute time without a DCB reference")
        return telemetry
    mjd_times = np.asarray(
        clock_reference_set.to_mjd(
            telemetry.raw_seconds,
            clock_source=ClockSource.DCB,
        ),
        dtype=np.float64,
    )
    return telemetry.with_mjd_times(mjd_times)
