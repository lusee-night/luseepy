"""Versioned spectrometer averaging windows from the flight firmware."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FREQUENCY_WINDOW_CONTRACT_NAME = "coreloop_3r09_frequency_windows"
FREQUENCY_WINDOW_CONTRACT_VERSION = 1
FREQUENCY_WINDOW_SOURCE_COMMIT = "38770b94bd20dfdbf52d1d8b872a716537b851f3"
NATIVE_SPECTROMETER_BINS = 2048
FREQUENCY_COORDINATE_STATUS = "unresolved"
_FREQUENCY_WINDOWS = {
    1: (1, (0,), 1, 2048),
    2: (2, (0, 1), 2, 1024),
    3: (4, (0, 1, 2), 4, 512),
    4: (4, (0, 1, 2, 3), 4, 512),
}


class UnresolvedFrequencyCoordinateError(RuntimeError):
    """No reviewed MHz coordinate exists for the firmware averaging window."""


@dataclass(frozen=True, slots=True)
class FrequencyWindowContract:
    """Exact native-bin membership and firmware divisor for one Navgf mode."""

    navgf: int
    stride: int
    included_offsets: tuple[int, ...]
    divisor: int
    output_count: int
    native_count: int = NATIVE_SPECTROMETER_BINS
    contract_name: str = FREQUENCY_WINDOW_CONTRACT_NAME
    contract_version: int = FREQUENCY_WINDOW_CONTRACT_VERSION
    source_commit: str = FREQUENCY_WINDOW_SOURCE_COMMIT
    frequency_coordinate_status: str = FREQUENCY_COORDINATE_STATUS

    def __post_init__(self) -> None:
        if type(self.navgf) is not int or self.navgf not in (1, 2, 3, 4):
            raise ValueError("Navgf must be one of 1, 2, 3, or 4")
        expected = _FREQUENCY_WINDOWS[self.navgf]
        if type(self.stride) is not int or self.stride <= 0:
            raise ValueError("frequency-window stride must be a positive integer")
        if not self.included_offsets:
            raise ValueError("frequency window must include at least one native bin")
        if any(
            type(offset) is not int or offset < 0 or offset >= self.stride
            for offset in self.included_offsets
        ):
            raise ValueError("included offsets must lie within one stride")
        if tuple(sorted(set(self.included_offsets))) != self.included_offsets:
            raise ValueError("included offsets must be unique and increasing")
        if type(self.divisor) is not int or self.divisor <= 0:
            raise ValueError("firmware divisor must be a positive integer")
        if type(self.native_count) is not int or self.native_count <= 0:
            raise ValueError("native_count must be a positive integer")
        if type(self.output_count) is not int or self.output_count <= 0:
            raise ValueError("output_count must be a positive integer")
        if self.native_count % self.stride:
            raise ValueError("native_count must be divisible by stride")
        if self.output_count != self.native_count // self.stride:
            raise ValueError("output_count disagrees with native_count and stride")
        actual = (
            self.stride,
            self.included_offsets,
            self.divisor,
            self.output_count,
        )
        if actual != expected or self.native_count != NATIVE_SPECTROMETER_BINS:
            raise ValueError(
                f"frequency window does not match the reviewed Navgf={self.navgf} contract"
            )
        if (
            self.contract_name != FREQUENCY_WINDOW_CONTRACT_NAME
            or type(self.contract_version) is not int
            or self.contract_version != FREQUENCY_WINDOW_CONTRACT_VERSION
            or self.source_commit != FREQUENCY_WINDOW_SOURCE_COMMIT
        ):
            raise ValueError("frequency-window provenance constants are immutable")
        if self.frequency_coordinate_status != "unresolved":
            raise ValueError("format 1 has no resolved MHz coordinate")

    @property
    def nominal_response_weights(self) -> tuple[float, ...]:
        """Linear response coefficients, excluding integer-rounding details."""
        return tuple(1.0 / self.divisor for _ in self.included_offsets)

    def native_indices(self, output_index: int) -> tuple[int, ...]:
        """Return the native input-bin indices used by one output bin."""
        if type(output_index) is not int:
            raise TypeError("output_index must be an integer")
        if not 0 <= output_index < self.output_count:
            raise IndexError(
                f"output_index {output_index} outside [0, {self.output_count})"
            )
        start = self.stride * output_index
        return tuple(start + offset for offset in self.included_offsets)

    def all_native_indices(self) -> np.ndarray:
        """Return an ``(output_count, included_count)`` native-index table."""
        starts = self.stride * np.arange(self.output_count, dtype=np.int64)
        offsets = np.asarray(self.included_offsets, dtype=np.int64)
        result = starts[:, None] + offsets[None, :]
        result.flags.writeable = False
        return result

    def frequency_mhz(self) -> np.ndarray:
        """Refuse to invent a v4 MHz coordinate while FREQ-001 is open."""
        raise UnresolvedFrequencyCoordinateError(
            "the native-bin origin and averaged-bin MHz convention are unresolved"
        )

    def as_record(self) -> dict[str, object]:
        """Return the window provenance fields required in layout v4."""
        return {
            "contract_name": self.contract_name,
            "contract_version": self.contract_version,
            "source_commit": self.source_commit,
            "navgf": self.navgf,
            "native_count": self.native_count,
            "output_count": self.output_count,
            "stride": self.stride,
            "included_offsets": list(self.included_offsets),
            "firmware_divisor": self.divisor,
            "nominal_response_weights": list(self.nominal_response_weights),
            "integer_arithmetic": "averaging_mode_dependent",
            "frequency_coordinate_status": self.frequency_coordinate_status,
        }


def spectrometer_frequency_window(navgf: int) -> FrequencyWindowContract:
    """Return the reviewed 3r09 native-bin window for one Navgf value."""
    try:
        stride, offsets, divisor, output_count = _FREQUENCY_WINDOWS[navgf]
    except (KeyError, TypeError) as exc:
        raise ValueError("Navgf must be one of 1, 2, 3, or 4") from exc
    return FrequencyWindowContract(
        navgf=navgf,
        stride=stride,
        included_offsets=offsets,
        divisor=divisor,
        output_count=output_count,
    )
