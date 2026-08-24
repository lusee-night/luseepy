"""Factory that builds a :class:`lusee.Observation` (specifically an
:class:`IngestData`) from one or more HDF5 / FITS files written by
:mod:`lusee.ingest`.

Inputs may be:

* a single file path (HDF5 or FITS);
* a single directory; the directory is walked recursively and HDF5 files
  win over FITS when both are present (override with ``prefer_format``);
* an iterable of any of the above.

Sessions are loaded in a deterministic order using the recorded session-start
raw counter when available, then concatenated along the spectra time axis.
The raw counter order is not claimed to be global acquisition chronology.
The result is a single ``IngestData`` whose ``self.times`` carries the actual
irregular sample times (not a uniform synthetic grid), so
:class:`Observation`'s ``get_track_*`` methods work out of the box.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from lunarsky.time import Time as LunarTime

from lusee.Observation import Observation

from .clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockReferenceUnavailableError,
    ClockSource,
    LegacyClockReferenceSet,
)
from .constants import (
    BITSLICE_REFERENCE,
    DEFAULT_TIME_SCALE,
    INGEST_LAYOUT_VERSION,
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
from .decode import canonical_actual_bitslice, restore_bitsliced_spectra
from .dependencies import import_optional_dependency
from .frequency_contract import (
    FrequencyWindowContract,
    UnresolvedFrequencyCoordinateError,
    spectrometer_frequency_window,
)
from .issues import IngestIssue
from .layout_v4_reader import (
    LayoutV4ValidationError,
    decode_field_union_rows,
    read_layout_v4_fits,
    read_layout_v4_hdf5,
    validate_layout_v4_tree,
)
from .products import (
    CalibratorDataSample,
    CalibratorDebugPage,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    DataQuality,
    DecodeProvenance,
    GrimmSample,
    HKSample,
    ProductProvenance,
    SourcePacketProvenance,
    SpectrumMetadata,
    SpectrumSample,
    TRSpectrumSample,
    WaveformSample,
    ZoomSample,
)
from .write_request import (
    ALL_FAMILIES,
    FamilyStatus,
    InterpolationPolicy,
    RunProvenance,
)

log = logging.getLogger(__name__)

# Pixel→MHz conversion at full resolution (Navgf=1) -- mirrors uncrater.
_FREQ_STEP_MHZ_NAVGF1 = 0.025

# Antenna / ADC channels (auto products 0-3). Distinct from NCHANNELS, which
# is the number of frequency bins (2048); the gain model is per ADC channel.
_N_ADC_CHANNELS = 4

# Spectrometer gain_state enum (coreloop core_loop.h) -> gain-model level.
# GAIN_DISABLE (3) / GAIN_AUTO (4) have no L/M/H model and map to None
# (those channels come out NaN).
_GAIN_CODE_TO_LEVEL = {0: "L", 1: "M", 2: "H"}


class MixedFrequencyGridError(ValueError):
    """High-level indexing was requested for more than one native grid."""


class LegacyIngestWarning(RuntimeWarning):
    """A layout-v2/v3 compatibility path cannot verify repaired semantics."""


# ---------------------------------------------------------------------------
# Data layout
# ---------------------------------------------------------------------------

@dataclass
class SessionBundle:
    """In-memory mirror of one HDF5/FITS session, format-agnostic."""

    spectra: Optional[np.ndarray] = None
    spectra_unique_ids: Optional[np.ndarray] = None
    spectra_raw_times: Optional[np.ndarray] = None
    spectra_raw_time_valid: Optional[np.ndarray] = None
    spectra_mjd_times: Optional[np.ndarray] = None
    spectra_mjd_time_valid: Optional[np.ndarray] = None
    spectra_frequency_counts: Optional[np.ndarray] = None
    spectra_navgf: Optional[np.ndarray] = None
    spectra_frequency_window_index: Optional[np.ndarray] = None
    spectra_frequency_windows: Tuple[FrequencyWindowContract, ...] = ()
    spectra_metadata: Dict[str, np.ndarray] = field(default_factory=dict)
    spectra_metadata_present: Dict[str, np.ndarray] = field(default_factory=dict)
    spectra_units: Optional[str] = None
    spectra_representation: Optional[str] = None
    spectra_normalization_version: Optional[int] = None

    tr_spectra: Optional[np.ndarray] = None
    tr_unique_ids: Optional[np.ndarray] = None
    tr_raw_times: Optional[np.ndarray] = None
    tr_raw_time_valid: Optional[np.ndarray] = None
    tr_mjd_times: Optional[np.ndarray] = None
    tr_mjd_time_valid: Optional[np.ndarray] = None
    tr_navg2_per_sample: Optional[np.ndarray] = None
    tr_length_per_sample: Optional[np.ndarray] = None
    tr_metadata: Dict[str, np.ndarray] = field(default_factory=dict)

    zoom_spectra: Optional[np.ndarray] = None
    zoom_unique_ids: Optional[np.ndarray] = None
    zoom_pfb_indices: Optional[np.ndarray] = None
    zoom_pfb_bins: Optional[np.ndarray] = None
    zoom_raw_times: Optional[np.ndarray] = None
    zoom_raw_time_valid: Optional[np.ndarray] = None
    zoom_mjd_times: Optional[np.ndarray] = None
    zoom_mjd_time_valid: Optional[np.ndarray] = None

    grimm_spectra: Optional[np.ndarray] = None
    grimm_unique_ids: Optional[np.ndarray] = None
    grimm_raw_times: Optional[np.ndarray] = None
    grimm_raw_time_valid: Optional[np.ndarray] = None
    grimm_mjd_times: Optional[np.ndarray] = None
    grimm_mjd_time_valid: Optional[np.ndarray] = None
    grimm_navg2_per_sample: Optional[np.ndarray] = None
    grimm_average_valid: Optional[np.ndarray] = None

    waveforms: Dict[int, np.ndarray] = field(default_factory=dict)
    waveform_times: Dict[int, np.ndarray] = field(default_factory=dict)
    waveform_data: Optional[np.ndarray] = None
    waveform_channels: Optional[np.ndarray] = None
    waveform_unique_ids: Optional[np.ndarray] = None
    waveform_raw_times: Optional[np.ndarray] = None
    waveform_raw_time_valid: Optional[np.ndarray] = None
    waveform_mjd_times: Optional[np.ndarray] = None
    waveform_mjd_time_valid: Optional[np.ndarray] = None
    waveform_adc_timestamps: Optional[np.ndarray] = None
    waveform_adc_timestamp_valid: Optional[np.ndarray] = None

    housekeeping: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    housekeeping_unique_ids: Optional[np.ndarray] = None
    housekeeping_raw_times: Optional[np.ndarray] = None
    housekeeping_raw_time_valid: Optional[np.ndarray] = None
    housekeeping_mjd_times: Optional[np.ndarray] = None
    housekeeping_mjd_time_valid: Optional[np.ndarray] = None
    housekeeping_types: Optional[np.ndarray] = None
    housekeeping_versions: Optional[np.ndarray] = None
    housekeeping_firmware_errors: Optional[np.ndarray] = None
    housekeeping_fields: Tuple[Dict[str, object], ...] = ()
    housekeeping_field_present: Tuple[Dict[str, bool], ...] = ()

    calibrator: Dict[str, object] = field(default_factory=dict)

    dcb_fpga: Dict[str, np.ndarray] = field(default_factory=dict)
    dcb_encoder: Dict[str, np.ndarray] = field(default_factory=dict)
    interp_telemetry: Dict[str, np.ndarray] = field(default_factory=dict)

    session_invariants: Dict[str, Any] = field(default_factory=dict)
    # numeric calibration values plus string time-provenance attrs
    # (time_scale, clock_source, clock_epoch_isot)
    constants: Dict[str, Union[float, str]] = field(default_factory=dict)
    clock_reference_set: Optional[
        ClockReferenceSet | LegacyClockReferenceSet
    ] = None
    clock_reference_unavailable_reason: Optional[str] = None
    clock_reference_unavailable_reasons: Tuple[Optional[str], ...] = ()
    run_provenance: Dict[str, object] = field(default_factory=dict)
    decoder_provenance: Dict[str, object] = field(default_factory=dict)
    issues: Dict[str, object] = field(default_factory=dict)
    product_provenance: Dict[str, object] = field(default_factory=dict)
    product_records: Dict[str, Tuple[object, ...]] = field(default_factory=dict)
    family_status: Dict[str, object] = field(default_factory=dict)
    quality_status: Optional[str] = None
    execution_mode: Optional[str] = None
    legacy_unverified_families: Tuple[str, ...] = ()

    source_path: Optional[Path] = None
    source_paths: Tuple[Path, ...] = ()
    session_spectra_counts: Tuple[int, ...] = ()
    session_sources: Tuple[Optional[Path], ...] = ()
    layout_version: Optional[int] = None

    def frequency_window_for_row(self, row: int) -> FrequencyWindowContract:
        """Return the exact native-bin window used by one normal-spectrum row."""
        if self.spectra is None:
            raise ValueError("bundle has no normal spectra")
        row = int(row)
        if not 0 <= row < self.spectra.shape[0]:
            raise IndexError(row)
        if self.spectra_frequency_windows:
            if self.spectra_frequency_window_index is None:
                raise ValueError("frequency-window index is missing")
            index = int(self.spectra_frequency_window_index[row])
            try:
                return self.spectra_frequency_windows[index]
            except IndexError as exc:
                raise ValueError("frequency-window index is out of range") from exc
        navgf = _bundle_navgf(self)
        return spectrometer_frequency_window(int(navgf[row]))

    def frequency_for_row(self, row: int) -> np.ndarray:
        """Return a reviewed MHz grid, refusing unresolved layout-v4 coordinates."""
        contract = self.frequency_window_for_row(row)
        if self.layout_version == INGEST_LAYOUT_VERSION:
            return contract.frequency_mhz()
        warnings.warn(
            "layout-v2/v3 frequency coordinates are legacy_unverified; "
            "re-ingest after the physical MHz convention is resolved",
            LegacyIngestWarning,
            stacklevel=2,
        )
        return np.arange(contract.output_count, dtype=np.float64) * (
            _FREQ_STEP_MHZ_NAVGF1 * contract.stride
        )

    def split_by_frequency_grid(self) -> Tuple[SessionBundle, ...]:
        """Split normal rows by exact Navgf/window identity without resampling."""
        if self.spectra is None or self.spectra.shape[0] == 0:
            return ()
        navgf = _bundle_navgf(self)
        if self.spectra_frequency_window_index is None:
            window_index = np.asarray(navgf, dtype=np.int64) - 1
        else:
            window_index = np.asarray(
                self.spectra_frequency_window_index,
                dtype=np.int64,
            )
        keys = sorted({(int(n), int(w)) for n, w in zip(navgf, window_index)})
        groups = []
        for navgf_value, window_value in keys:
            rows = np.flatnonzero(
                (navgf == navgf_value) & (window_index == window_value)
            )
            groups.append(self._spectra_subset(rows))
        return tuple(groups)

    def _spectra_subset(self, rows: np.ndarray) -> SessionBundle:
        result = replace(self)
        aligned_names = (
            "spectra",
            "spectra_unique_ids",
            "spectra_raw_times",
            "spectra_raw_time_valid",
            "spectra_mjd_times",
            "spectra_mjd_time_valid",
            "spectra_frequency_counts",
            "spectra_navgf",
        )
        for name in aligned_names:
            value = getattr(self, name)
            if value is not None:
                setattr(result, name, np.asarray(value)[rows])
        result.spectra_metadata = {
            name: np.asarray(value)[rows]
            for name, value in self.spectra_metadata.items()
        }
        result.spectra_metadata_present = {
            name: np.asarray(value)[rows]
            for name, value in self.spectra_metadata_present.items()
        }
        if self.interp_telemetry:
            result.interp_telemetry = {
                name: np.asarray(value)[rows]
                for name, value in self.interp_telemetry.items()
            }
        contract = self.frequency_window_for_row(int(rows[0]))
        result.spectra_frequency_windows = (contract,)
        result.spectra_frequency_window_index = np.zeros(
            len(rows),
            dtype=np.uint8,
        )
        if self.session_spectra_counts:
            selected_counts = []
            start = 0
            for count in self.session_spectra_counts:
                stop = start + count
                selected_counts.append(
                    int(np.count_nonzero((rows >= start) & (rows < stop)))
                )
                start = stop
            if start != self.spectra.shape[0]:
                raise ValueError("session spectrum counts do not cover all rows")
            result.session_spectra_counts = tuple(selected_counts)
        result.product_records = dict(self.product_records)
        if "spectra" in self.product_records:
            result.product_records["spectra"] = tuple(
                self.product_records["spectra"][int(row)] for row in rows
            )
        original_provenance = self.product_provenance.get(
            "original_artifact",
            self.product_provenance,
        )
        result.product_provenance = {
            "scope": "frequency_grid_subset",
            "original_artifact": original_provenance,
            "records_by_family": result.product_records,
        }
        original_family_status = self.family_status.get(
            "original_artifact",
            self.family_status,
        )
        result.family_status = {
            "scope": "frequency_grid_subset",
            "original_artifact": original_family_status,
            "persisted_rows_by_family": _bundle_persisted_rows_by_family(
                result
            ),
        }
        return result


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_H5_EXTS = (".h5", ".hdf5")
_FITS_EXTS = (".fits", ".fit")


def _is_h5(p: Path) -> bool:
    return p.suffix.lower() in _H5_EXTS


def _is_fits(p: Path) -> bool:
    return p.suffix.lower() in _FITS_EXTS


def _resolve_one(target: Path, *, prefer_format: str) -> List[Path]:
    """Resolve a single path entry to a list of files."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(target)

    primary_exts = _H5_EXTS if prefer_format == "h5" else _FITS_EXTS
    fallback_exts = _FITS_EXTS if prefer_format == "h5" else _H5_EXTS

    primary = {
        p.relative_to(target).with_suffix(""): p
        for ext in primary_exts
        for p in target.rglob(f"*{ext}")
    }
    fallback = {
        p.relative_to(target).with_suffix(""): p
        for ext in fallback_exts
        for p in target.rglob(f"*{ext}")
    }

    resolved = dict(fallback)
    resolved.update(primary)
    if resolved:
        return sorted(resolved.values())
    raise FileNotFoundError(
        f"no .h5 / .hdf5 / .fits files found under {target}"
    )


def _resolve_paths(target, *, prefer_format: str = "h5") -> List[Path]:
    if isinstance(target, (str, Path)):
        items: List[Path] = [Path(target)]
    else:
        items = [Path(p) for p in target]

    out: List[Path] = []
    for t in items:
        out.extend(_resolve_one(t, prefer_format=prefer_format))
    if not out:
        raise FileNotFoundError("no input files resolved")
    return out


# ---------------------------------------------------------------------------
# Layout-v4 semantic reader
# ---------------------------------------------------------------------------

_V4_COMMON_DATASETS = {
    "unique_ids": np.uint32,
    "raw_seconds": np.float64,
    "raw_time_valid": np.bool_,
    "mjd_times": np.float64,
    "mjd_time_valid": np.bool_,
    "original_indices": np.uint64,
    "provenance_index": np.uint64,
}


def _v4_text_attr(validator, path: str, name: str) -> str:
    value = validator.attribute(path, name)
    if type(value) is not str or not value:
        raise LayoutV4ValidationError(
            f"layout-v4 attribute {path}@{name} must be nonempty text"
        )
    return value


def _v4_scalar_attr(validator, path: str, name: str, dtype) -> object:
    return validator.attribute(path, name, dtype=dtype)


def _v4_session_invariants(validator) -> Dict[str, object]:
    path = "/session_invariants"
    specs = {
        "software_version": np.uint32,
        "firmware_version": np.uint32,
        "firmware_id": np.uint32,
        "firmware_date": np.uint32,
        "firmware_time": np.uint32,
        "start_unique_packet_id": np.uint32,
        "start_time_32": np.uint32,
        "start_time_16": np.uint16,
        "start_raw_seconds": np.float64,
    }
    attrs = validator.group(path).attrs
    expected_names = {f"{name}_valid" for name in specs}
    expected_names.update(name for name in specs if name in attrs)
    if set(attrs) != expected_names:
        raise LayoutV4ValidationError(
            "layout-v4 session-invariant attributes disagree"
        )
    result: Dict[str, object] = {}
    for name, dtype in specs.items():
        valid_name = f"{name}_valid"
        valid_value = attrs[valid_name]
        array = np.asarray(valid_value)
        if array.shape != () or array.dtype != np.dtype(np.bool_):
            raise LayoutV4ValidationError(
                f"layout-v4 optional attribute {path}@{valid_name} is invalid"
            )
        if bool(array):
            if name not in attrs:
                raise LayoutV4ValidationError(
                    f"layout-v4 optional attribute {path}@{name} is missing"
                )
            value = np.asarray(attrs[name])
            if value.shape != () or value.dtype != np.dtype(dtype):
                raise LayoutV4ValidationError(
                    f"layout-v4 optional attribute {path}@{name} has the "
                    "wrong type"
                )
            result[name] = _scalarize(attrs[name])
        elif name in attrs:
            raise LayoutV4ValidationError(
                f"layout-v4 optional attribute {path}@{name} is unexpectedly present"
            )

    hello_names = tuple(specs)[:-1]
    hello_present = tuple(name in result for name in hello_names)
    if any(hello_present) and not all(hello_present):
        raise LayoutV4ValidationError(
            "layout-v4 Hello session invariants must be all present or all absent"
        )
    if not all(hello_present):
        if "start_raw_seconds" in result:
            raise LayoutV4ValidationError(
                "layout-v4 session start time requires Hello invariants"
            )
        return result
    if "start_raw_seconds" not in result:
        raise LayoutV4ValidationError(
            "layout-v4 Hello session invariants require start_raw_seconds"
        )
    combined = (int(result["start_time_16"]) << 32) + int(
        result["start_time_32"]
    )
    expected = (
        (combined >> MISSION_TIME_FRACT_SHIFT)
        / MISSION_TIME_FRACT_DIVISOR
    )
    if float(result["start_raw_seconds"]) != expected:
        raise LayoutV4ValidationError(
            "layout-v4 session start split time disagrees"
        )
    return result


def _v4_parse_clock_reference(validator) -> tuple[
    Optional[ClockReferenceSet], Optional[str]
]:
    available = bool(
        _v4_scalar_attr(validator, "/clock_reference", "available", np.bool_)
    )
    group = validator.group("/clock_reference")
    if not available:
        if set(group.attrs) != {"available", "unavailable_reason"}:
            raise LayoutV4ValidationError(
                "unavailable layout-v4 clock-reference attributes disagree"
            )
        reason = _v4_text_attr(
            validator, "/clock_reference", "unavailable_reason"
        )
        if group.children:
            raise LayoutV4ValidationError(
                "unavailable layout-v4 clock reference must not have datasets"
            )
        return None, reason

    expected_attrs = {
        "available",
        "format_version",
        "reference_event",
        "clock_reference_isot",
        "time_scale",
        "source",
        "assumed",
        "source_sha256",
        "canonical_record_json",
    }
    expected_children = {
        "clock_sources",
        "clock_reference_raw_seconds",
    }
    if (
        set(group.attrs) != expected_attrs
        or set(group.children) != expected_children
    ):
        raise LayoutV4ValidationError(
            "available layout-v4 clock-reference contract disagrees"
        )

    format_version = int(
        _v4_scalar_attr(
            validator, "/clock_reference", "format_version", np.uint16
        )
    )
    assumed = bool(
        _v4_scalar_attr(validator, "/clock_reference", "assumed", np.bool_)
    )
    sources = validator.dataset(
        "/clock_reference/clock_sources", utf8=True, ndim=1
    ).data
    raw = validator.dataset(
        "/clock_reference/clock_reference_raw_seconds",
        dtype=np.float64,
        utf8=False,
        shape=(sources.size,),
    ).data
    if not np.all(np.isfinite(raw)):
        raise LayoutV4ValidationError(
            "layout-v4 clock-reference anchors must be finite"
        )
    try:
        clocks = tuple(
            ClockReference(
                clock_source=ClockSource(str(source)),
                clock_reference_raw_seconds=float(value),
            )
            for source, value in zip(sources.tolist(), raw.tolist())
        )
        reference_set = ClockReferenceSet(
            format_version=format_version,
            reference_event=_v4_text_attr(
                validator, "/clock_reference", "reference_event"
            ),
            clock_reference_isot=_v4_text_attr(
                validator, "/clock_reference", "clock_reference_isot"
            ),
            time_scale=_v4_text_attr(
                validator, "/clock_reference", "time_scale"
            ),
            clocks=clocks,
            source=_v4_text_attr(validator, "/clock_reference", "source"),
            assumed=assumed,
            source_sha256=_v4_text_attr(
                validator, "/clock_reference", "source_sha256"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise LayoutV4ValidationError(
            f"layout-v4 clock reference is invalid: {exc}"
        ) from exc
    canonical_text = _v4_text_attr(
        validator, "/clock_reference", "canonical_record_json"
    )
    try:
        canonical_record = json.loads(canonical_text)
    except json.JSONDecodeError as exc:
        raise LayoutV4ValidationError(
            "layout-v4 clock-reference canonical JSON is invalid"
        ) from exc
    if canonical_record != reference_set.as_record():
        raise LayoutV4ValidationError(
            "layout-v4 clock-reference canonical record disagrees"
        )
    expected_canonical = json.dumps(
        reference_set.as_record(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if canonical_text != expected_canonical:
        raise LayoutV4ValidationError(
            "layout-v4 clock-reference record is not canonical"
        )
    if tuple(sources.tolist()) != tuple(
        item.clock_source.value for item in reference_set.clocks
    ):
        raise LayoutV4ValidationError(
            "layout-v4 clock sources are not unique and sorted"
        )
    return reference_set, None


def _v4_direct_arrays(validator, path: str) -> Dict[str, np.ndarray]:
    group = validator.group(path)
    return {
        name: np.asarray(child.data)
        for name, child in group.children.items()
        if hasattr(child, "data")
    }


def _v4_optional_text_attribute(validator, path: str, name: str) -> Optional[str]:
    valid = bool(_v4_scalar_attr(validator, path, f"{name}_valid", np.bool_))
    attrs = validator.group(path).attrs
    if not valid:
        if name in attrs:
            raise LayoutV4ValidationError(
                f"layout-v4 optional attribute {path}@{name} is unexpectedly present"
            )
        return None
    return _v4_text_attr(validator, path, name)


def _v4_parse_run_provenance(validator) -> Dict[str, object]:
    path = "/run_provenance"
    values = {
        name: _v4_optional_text_attribute(validator, path, name)
        for name in (
            "input_identity",
            "input_identity_kind",
            "input_identity_unavailable_reason",
            "source_kind",
            "source_path",
            "pipeline_version",
        )
    }
    try:
        record = RunProvenance(**values)
        interpolation = InterpolationPolicy(
            mode=_v4_text_attr(validator, path, "interpolation_mode"),
            maximum_gap_seconds=(
                float(
                    _v4_scalar_attr(
                        validator,
                        path,
                        "interpolation_maximum_gap_seconds",
                        np.float64,
                    )
                )
                if bool(
                    _v4_scalar_attr(
                        validator,
                        path,
                        "interpolation_maximum_gap_seconds_valid",
                        np.bool_,
                    )
                )
                else None
            ),
            extrapolate=bool(
                _v4_scalar_attr(
                    validator,
                    path,
                    "interpolation_extrapolate",
                    np.bool_,
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise LayoutV4ValidationError(
            f"layout-v4 run provenance is invalid: {exc}"
        ) from exc
    return {
        "record": record,
        "interpolation_policy": interpolation,
        "attributes": dict(validator.group(path).attrs),
    }


def _v4_parse_decoder_provenance(validator) -> Dict[str, object]:
    path = "/provenance/decoder"
    optional_names = (
        "decoder_name",
        "distribution_version",
        "decoder_source_commit",
        "binding_key",
        "schema_variant",
        "binding_source_release",
        "binding_source_commit",
        "abi_fingerprint",
        "canonical_report_json",
    )
    optional = {
        name: _v4_optional_text_attribute(validator, path, name)
        for name in optional_names
    }
    arrays = _v4_direct_arrays(validator, path)
    if set(arrays) != {
        "reported_schema_ids",
        "appids",
        "appid_counts",
        "issue_codes",
        "issue_code_counts",
    }:
        raise LayoutV4ValidationError(
            "layout-v4 decoder provenance datasets disagree"
        )
    reported = validator.dataset(
        f"{path}/reported_schema_ids", dtype=np.uint16, utf8=False, ndim=1
    ).data
    appids = validator.dataset(
        f"{path}/appids", dtype=np.uint16, utf8=False, ndim=1
    ).data
    appid_counts = validator.dataset(
        f"{path}/appid_counts",
        dtype=np.uint64,
        utf8=False,
        shape=(appids.size,),
    ).data
    issue_codes = validator.dataset(
        f"{path}/issue_codes", utf8=True, ndim=1
    ).data
    issue_counts = validator.dataset(
        f"{path}/issue_code_counts",
        dtype=np.uint64,
        utf8=False,
        shape=(issue_codes.size,),
    ).data
    try:
        record = DecodeProvenance(
            decoder_name=optional["decoder_name"],
            distribution_version=optional["distribution_version"],
            decoder_source_commit=optional["decoder_source_commit"],
            reported_schema_ids=tuple(int(value) for value in reported),
            selected_schema_id=int(
                _v4_scalar_attr(
                    validator, path, "selected_schema_id", np.uint16
                )
            ),
            binding_key=optional["binding_key"],
            schema_variant=optional["schema_variant"],
            schema_assumed=bool(
                _v4_scalar_attr(validator, path, "schema_assumed", np.bool_)
            ),
            binding_source_release=optional["binding_source_release"],
            binding_source_commit=optional["binding_source_commit"],
            abi_fingerprint=optional["abi_fingerprint"],
            execution_mode=_v4_text_attr(
                validator, path, "execution_mode"
            ),
            input_packet_count=int(
                _v4_scalar_attr(
                    validator, path, "input_packet_count", np.uint64
                )
            ),
            valid_packet_count=int(
                _v4_scalar_attr(
                    validator, path, "valid_packet_count", np.uint64
                )
            ),
            appid_counts=tuple(
                (int(appid), int(count))
                for appid, count in zip(appids, appid_counts)
            ),
            issue_counts=tuple(
                (str(code), int(count))
                for code, count in zip(issue_codes, issue_counts)
            ),
            canonical_report_json=optional["canonical_report_json"],
        )
    except (TypeError, ValueError) as exc:
        raise LayoutV4ValidationError(
            f"layout-v4 decoder provenance is invalid: {exc}"
        ) from exc
    return {
        "record": record,
        "attributes": dict(validator.group(path).attrs),
        "datasets": arrays,
    }


def _v4_parse_product_provenance(validator) -> Dict[str, object]:
    path = "/provenance/product_rows"
    specs = {
        "family": (None, True),
        "row_index": (np.uint64, False),
        "unique_ids": (np.uint32, False),
        "uid_source": (None, True),
        "uid_source_role": (None, True),
        "uid_source_role_valid": (np.bool_, False),
        "time_source": (None, True),
        "time_source_valid": (np.bool_, False),
        "time_source_role": (None, True),
        "time_source_role_valid": (np.bool_, False),
        "clock_source": (None, True),
        "clock_source_valid": (np.bool_, False),
        "time_valid": (np.bool_, False),
        "selected_schema_ids": (np.uint16, False),
    }
    group = validator.group(path)
    if set(group.children) != set(specs):
        raise LayoutV4ValidationError(
            "layout-v4 product provenance columns disagree"
        )
    count = validator.require_row_aligned(path, specs)
    rows: Dict[str, np.ndarray] = {}
    for name, (dtype, utf8) in specs.items():
        rows[name] = validator.dataset(
            f"{path}/{name}",
            dtype=dtype,
            utf8=utf8,
            shape=(count,),
        ).data
    if any(family not in ALL_FAMILIES for family in rows["family"].tolist()):
        raise LayoutV4ValidationError(
            "layout-v4 product provenance names an unknown family"
        )
    if np.any(np.asarray([not bool(value) for value in rows["uid_source"]])):
        raise LayoutV4ValidationError(
            "layout-v4 provenance uid_source must be nonempty"
        )
    for name in (
        "uid_source_role",
        "time_source",
        "time_source_role",
        "clock_source",
    ):
        valid = rows[f"{name}_valid"]
        present = np.asarray([bool(value) for value in rows[name]])
        if not np.array_equal(valid, present):
            raise LayoutV4ValidationError(
                f"layout-v4 provenance {name} validity disagrees"
            )
    if np.any(rows["time_valid"] != rows["time_source_valid"]) or np.any(
        rows["time_valid"] != rows["time_source_role_valid"]
    ) or np.any(rows["time_valid"] != rows["clock_source_valid"]):
        raise LayoutV4ValidationError(
            "layout-v4 provenance time-source validity disagrees"
        )

    issue_count = int(
        _v4_scalar_attr(validator, "/issues", "count", np.uint64)
    )
    reference_specs = {
        "schema_refs": (
            "/provenance/product_schema_refs",
            {"provenance_index": np.uint64, "schema_id": np.uint16},
        ),
        "issue_refs": (
            "/provenance/product_issue_refs",
            {"provenance_index": np.uint64, "issue_index": np.uint64},
        ),
    }
    result: Dict[str, object] = {"product_rows": rows}
    for key, (ref_path, ref_specs) in reference_specs.items():
        ref_count = validator.require_row_aligned(ref_path, ref_specs)
        refs = {
            name: validator.dataset(
                f"{ref_path}/{name}",
                dtype=dtype,
                utf8=False,
                shape=(ref_count,),
            ).data
            for name, dtype in ref_specs.items()
        }
        if np.any(refs["provenance_index"] >= count):
            raise LayoutV4ValidationError(
                f"layout-v4 {key} provenance reference is out of range"
            )
        if "issue_index" in refs and np.any(refs["issue_index"] >= issue_count):
            raise LayoutV4ValidationError(
                "layout-v4 product issue reference is out of range"
            )
        result[key] = refs

    packet_path = "/provenance/source_packets"
    packet_specs = {
        "provenance_index": (np.uint64, False),
        "source_order": (np.uint16, False),
        "role": (None, True),
        "original_appid": (np.uint16, False),
        "normalized_appid": (np.uint16, False),
        "normalized_appid_valid": (np.bool_, False),
        "packet_index": (np.uint64, False),
        "packet_index_valid": (np.bool_, False),
        "frame_start": (np.uint64, False),
        "frame_start_valid": (np.bool_, False),
        "frame_stop": (np.uint64, False),
        "frame_stop_valid": (np.bool_, False),
        "byte_offset_start": (np.uint64, False),
        "byte_offset_start_valid": (np.bool_, False),
        "byte_offset_stop": (np.uint64, False),
        "byte_offset_stop_valid": (np.bool_, False),
        "filename": (None, True),
        "filename_valid": (np.bool_, False),
        "bank": (None, True),
        "bank_valid": (np.bool_, False),
    }
    packet_group = validator.group(packet_path)
    if set(packet_group.children) != set(packet_specs):
        raise LayoutV4ValidationError(
            "layout-v4 source-packet provenance columns disagree"
        )
    packet_count = validator.require_row_aligned(packet_path, packet_specs)
    packets = {
        name: validator.dataset(
            f"{packet_path}/{name}",
            dtype=dtype,
            utf8=utf8,
            shape=(packet_count,),
        ).data
        for name, (dtype, utf8) in packet_specs.items()
    }
    if np.any(packets["provenance_index"] >= count):
        raise LayoutV4ValidationError(
            "layout-v4 source-packet provenance reference is invalid"
        )
    optional_packet_names = (
        "normalized_appid",
        "packet_index",
        "frame_start",
        "frame_stop",
        "byte_offset_start",
        "byte_offset_stop",
        "filename",
        "bank",
    )
    for name in optional_packet_names:
        valid = packets[f"{name}_valid"]
        if packets[name].dtype == np.dtype(object):
            placeholder = np.asarray(
                [not bool(value) for value in packets[name]], dtype=np.bool_
            )
        else:
            placeholder = packets[name] == 0
        if np.any(~valid & ~placeholder):
            raise LayoutV4ValidationError(
                f"layout-v4 source-packet {name} absence disagrees"
            )
    for provenance_index in range(count):
        source_orders = packets["source_order"][
            packets["provenance_index"] == provenance_index
        ]
        if not np.array_equal(
            source_orders,
            np.arange(source_orders.size, dtype=np.uint16),
        ):
            raise LayoutV4ValidationError(
                "layout-v4 source-packet order is not contiguous"
            )
    result["source_packets"] = packets
    issue_ids = validator.dataset(
        "/issues/issue_id", utf8=True, shape=(issue_count,)
    ).data
    schema_refs = result["schema_refs"]
    issue_refs = result["issue_refs"]
    records = []

    def optional_packet_value(name: str, row: int):
        if not packets[f"{name}_valid"][row]:
            return None
        value = packets[name][row]
        return str(value) if packets[name].dtype == np.dtype(object) else int(value)

    for provenance_index in range(count):
        packet_rows = np.flatnonzero(
            packets["provenance_index"] == provenance_index
        )
        source_packets = []
        try:
            for packet_row in packet_rows:
                source_packets.append(
                    SourcePacketProvenance(
                        role=str(packets["role"][packet_row]),
                        original_appid=int(
                            packets["original_appid"][packet_row]
                        ),
                        normalized_appid=optional_packet_value(
                            "normalized_appid", packet_row
                        ),
                        packet_index=optional_packet_value(
                            "packet_index", packet_row
                        ),
                        frame_start=optional_packet_value(
                            "frame_start", packet_row
                        ),
                        frame_stop=optional_packet_value(
                            "frame_stop", packet_row
                        ),
                        byte_offset_start=optional_packet_value(
                            "byte_offset_start", packet_row
                        ),
                        byte_offset_stop=optional_packet_value(
                            "byte_offset_stop", packet_row
                        ),
                        filename=optional_packet_value("filename", packet_row),
                        bank=optional_packet_value("bank", packet_row),
                    )
                )
            reported = tuple(
                int(value)
                for value in schema_refs["schema_id"][
                    schema_refs["provenance_index"] == provenance_index
                ]
            )
            decoder_issues = tuple(
                str(issue_ids[index])
                for index in issue_refs["issue_index"][
                    issue_refs["provenance_index"] == provenance_index
                ]
            )

            def optional_text(name: str):
                return (
                    str(rows[name][provenance_index])
                    if rows[f"{name}_valid"][provenance_index]
                    else None
                )

            record = ProductProvenance(
                source_packets=tuple(source_packets),
                uid=int(rows["unique_ids"][provenance_index]),
                uid_source=str(rows["uid_source"][provenance_index]),
                uid_source_role=optional_text("uid_source_role"),
                reported_schema_ids=reported,
                selected_schema_id=int(
                    rows["selected_schema_ids"][provenance_index]
                ),
                decoder_issue_ids=decoder_issues,
                time_source=optional_text("time_source"),
                time_source_role=optional_text("time_source_role"),
                clock_source=optional_text("clock_source"),
                time_valid=bool(rows["time_valid"][provenance_index]),
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 product provenance row {provenance_index} is "
                f"invalid: {exc}"
            ) from exc
        records.append(record)
    result["records"] = tuple(records)
    return result


def _v4_common_rows(
    validator,
    path: str,
    family: str,
    product_provenance: Dict[str, object],
    clock_reference_set: Optional[ClockReferenceSet],
) -> Dict[str, np.ndarray]:
    count = int(_v4_scalar_attr(validator, path, "count", np.uint64))
    common = {
        name: validator.dataset(
            f"{path}/{name}",
            dtype=dtype,
            utf8=False,
            shape=(count,),
        ).data
        for name, dtype in _V4_COMMON_DATASETS.items()
    }
    raw, raw_valid = common["raw_seconds"], common["raw_time_valid"]
    mjd, mjd_valid = common["mjd_times"], common["mjd_time_valid"]
    if (
        np.any(~np.isfinite(raw[raw_valid]))
        or np.any(~np.isnan(raw[~raw_valid]))
        or np.any(~np.isfinite(mjd[mjd_valid]))
        or np.any(~np.isnan(mjd[~mjd_valid]))
        or np.any(mjd_valid & ~raw_valid)
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 time validity disagrees at {path}"
        )
    if not np.array_equal(
        common["original_indices"], np.arange(count, dtype=np.uint64)
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 original row indices disagree at {path}"
        )

    provenance_rows = product_provenance["product_rows"]
    provenance_count = len(provenance_rows["family"])
    indices = common["provenance_index"]
    if np.any(indices >= provenance_count):
        raise LayoutV4ValidationError(
            f"layout-v4 product provenance index is out of range at {path}"
        )
    expected_rows = np.arange(count, dtype=np.uint64)
    if (
        not np.array_equal(provenance_rows["family"][indices], [family] * count)
        or not np.array_equal(provenance_rows["row_index"][indices], expected_rows)
        or not np.array_equal(
            provenance_rows["unique_ids"][indices], common["unique_ids"]
        )
        or not np.array_equal(
            provenance_rows["time_valid"][indices], raw_valid
        )
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 product provenance disagrees at {path}"
        )

    expected_mjd_valid = np.zeros(count, dtype=np.bool_)
    if clock_reference_set is not None:
        for row, provenance_index in enumerate(indices):
            if not raw_valid[row]:
                continue
            source = str(provenance_rows["clock_source"][provenance_index])
            if clock_reference_set.reference_for(source) is None:
                continue
            expected_mjd_valid[row] = True
            expected = float(
                clock_reference_set.to_mjd(
                    raw[row],
                    clock_source=source,
                )
            )
            if not np.isclose(mjd[row], expected, rtol=0.0, atol=1e-12):
                raise LayoutV4ValidationError(
                    f"layout-v4 calibrated MJD disagrees at {path} row {row}"
                )
    if not np.array_equal(mjd_valid, expected_mjd_valid):
        raise LayoutV4ValidationError(
            f"layout-v4 MJD coverage disagrees at {path}"
        )
    return common


def _v4_product_record(
    common: Dict[str, np.ndarray],
    row: int,
    product_provenance: Dict[str, object],
) -> ProductProvenance:
    provenance_index = int(common["provenance_index"][row])
    return product_provenance["records"][provenance_index]


def _field_rows_to_columns(
    rows: tuple[dict[str, object], ...],
    presence: tuple[dict[str, bool], ...],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if not rows:
        return {}, {}
    keys = tuple(sorted(rows[0]))
    columns: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}
    for name in keys:
        present = np.asarray(
            [row_mask[name] for row_mask in presence], dtype=np.bool_
        )
        values = [row[name] for row in rows]
        masks[name] = present
        if np.all(present):
            arrays = [np.asarray(value) for value in values]
            if len({(array.dtype.str, array.shape) for array in arrays}) == 1:
                columns[name] = np.stack(arrays)
                continue
        column = np.empty(len(rows), dtype=object)
        column[:] = values
        columns[name] = column
    return columns, masks


def _v4_spectrum_metadata(
    validator,
    path: str,
    count: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], tuple[SpectrumMetadata, ...]]:
    rows, presence = decode_field_union_rows(validator, path, count)
    init_names = tuple(item.name for item in dataclass_fields(SpectrumMetadata) if item.init)
    expected_names = set(init_names) | {
        "adc_statistics_valid",
        "current_fields_present",
    }
    if rows and set(rows[0]) != expected_names:
        raise LayoutV4ValidationError(
            f"layout-v4 spectrum metadata fields disagree at {path}"
        )
    records = []
    for row_index, (row, mask) in enumerate(zip(rows, presence)):
        try:
            record = SpectrumMetadata(
                **{
                    name: (
                        row[name].item()
                        if mask[name] and isinstance(row[name], np.generic)
                        else row[name] if mask[name] else None
                    )
                    for name in init_names
                }
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 spectrum metadata is invalid at {path} row "
                f"{row_index}: {exc}"
            ) from exc
        for derived_name in (
            "adc_statistics_valid",
            "current_fields_present",
        ):
            if not mask[derived_name] or not np.array_equal(
                np.asarray(row[derived_name]),
                np.asarray(getattr(record, derived_name)),
            ):
                raise LayoutV4ValidationError(
                    f"layout-v4 derived metadata {derived_name} disagrees at "
                    f"{path} row {row_index}"
                )
        records.append(record)
    columns, masks = _field_rows_to_columns(rows, presence)
    return columns, masks, tuple(records)


def _v4_frequency_windows(validator) -> Tuple[FrequencyWindowContract, ...]:
    path = "/spectra/frequency_windows"
    specs = {
        "navgf": (np.uint8, 1),
        "native_count": (np.uint16, 1),
        "output_count": (np.uint16, 1),
        "stride": (np.uint16, 1),
        "divisor": (np.uint16, 1),
        "included_offsets": (np.uint8, 2),
        "included_offset_valid": (np.bool_, 2),
        "nominal_response_weights": (np.float64, 2),
    }
    group = validator.group(path)
    if set(group.children) != set(specs):
        raise LayoutV4ValidationError(
            "layout-v4 frequency-window columns disagree"
        )
    count = validator.dataset(
        f"{path}/navgf", dtype=np.uint8, utf8=False, ndim=1
    ).shape[0]
    arrays = {
        name: validator.dataset(
            f"{path}/{name}",
            dtype=dtype,
            utf8=False,
            row_count=count,
            tail_shape=(() if rank == 1 else (4,)),
        ).data
        for name, (dtype, rank) in specs.items()
    }
    if count < 1 or not np.array_equal(
        arrays["navgf"], np.sort(np.unique(arrays["navgf"]))
    ):
        raise LayoutV4ValidationError(
            "layout-v4 frequency windows must be nonempty, unique, and sorted"
        )
    contracts = []
    for index, navgf_value in enumerate(arrays["navgf"]):
        try:
            expected = spectrometer_frequency_window(int(navgf_value))
        except ValueError as exc:
            raise LayoutV4ValidationError(
                "layout-v4 frequency window has invalid Navgf"
            ) from exc
        valid = arrays["included_offset_valid"][index]
        valid_count = len(expected.included_offsets)
        expected_valid = np.arange(4) < valid_count
        if (
            int(arrays["native_count"][index]) != expected.native_count
            or int(arrays["output_count"][index]) != expected.output_count
            or int(arrays["stride"][index]) != expected.stride
            or int(arrays["divisor"][index]) != expected.divisor
            or not np.array_equal(valid, expected_valid)
            or not np.array_equal(
                arrays["included_offsets"][index, :valid_count],
                expected.included_offsets,
            )
            or np.any(arrays["included_offsets"][index, valid_count:] != 0)
            or not np.array_equal(
                arrays["nominal_response_weights"][index, :valid_count],
                expected.nominal_response_weights,
            )
            or not np.all(
                np.isnan(
                    arrays["nominal_response_weights"][index, valid_count:]
                )
            )
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 Navgf={int(navgf_value)} window disagrees"
            )
        contracts.append(expected)
    first = contracts[0]
    expected_attrs = {
        "contract_name": first.contract_name,
        "contract_version": np.uint16(first.contract_version),
        "source_commit": first.source_commit,
        "frequency_coordinate_status": first.frequency_coordinate_status,
        "integer_arithmetic": first.as_record()["integer_arithmetic"],
    }
    for name, expected in expected_attrs.items():
        observed = validator.attribute(path, name)
        if type(expected) is np.uint16:
            if np.asarray(observed).shape != () or np.asarray(observed).dtype != np.dtype(
                np.uint16
            ) or int(observed) != int(expected):
                raise LayoutV4ValidationError(
                    f"layout-v4 frequency-window attribute {name} disagrees"
                )
        elif observed != expected:
            raise LayoutV4ValidationError(
                f"layout-v4 frequency-window attribute {name} disagrees"
            )
    return tuple(contracts)


def _v4_parse_spectra(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/spectra"
    common = _v4_common_rows(
        validator,
        path,
        "spectra",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    data_node = validator.dataset(
        f"{path}/data",
        dtype=np.float32,
        utf8=False,
        shape=(count, NPRODUCTS, NCHANNELS),
    )
    data = data_node.data
    frequency_counts = validator.dataset(
        f"{path}/frequency_counts",
        dtype=np.uint16,
        utf8=False,
        shape=(count,),
    ).data
    navgf = validator.dataset(
        f"{path}/navgf",
        dtype=np.uint8,
        utf8=False,
        shape=(count,),
    ).data
    window_index = validator.dataset(
        f"{path}/frequency_window_index",
        dtype=np.uint8,
        utf8=False,
        shape=(count,),
    ).data
    contracts = _v4_frequency_windows(validator)
    if np.any(window_index >= len(contracts)):
        raise LayoutV4ValidationError(
            "layout-v4 normal-spectrum window index is out of range"
        )
    for row in range(count):
        contract = contracts[int(window_index[row])]
        nfreq = int(frequency_counts[row])
        if int(navgf[row]) != contract.navgf or nfreq != contract.output_count:
            raise LayoutV4ValidationError(
                f"layout-v4 normal frequency contract disagrees at row {row}"
            )
        if not np.all(np.isnan(data[row, :, nfreq:])):
            raise LayoutV4ValidationError(
                f"layout-v4 normal-spectrum tail is not NaN at row {row}"
            )
        finite_product = np.all(np.isfinite(data[row, :, :nfreq]), axis=1)
        nan_product = np.all(np.isnan(data[row, :, :nfreq]), axis=1)
        if not np.all(finite_product | nan_product) or not np.any(finite_product):
            raise LayoutV4ValidationError(
                f"layout-v4 normal product planes are invalid at row {row}"
            )
    expected_data_attrs = {
        "units": SPECTRA_UNITS,
        "representation": SPECTRA_REPRESENTATION,
        "bitslice_restored": np.bool_(True),
        "bitslice_reference": np.uint8(BITSLICE_REFERENCE),
        "normalization_version": np.uint16(SPECTRA_NORMALIZATION_VERSION),
    }
    for name, expected in expected_data_attrs.items():
        observed = validator.attribute(f"{path}/data", name)
        if isinstance(expected, np.generic):
            array = np.asarray(observed)
            if array.shape != () or array.dtype != expected.dtype or observed != expected:
                raise LayoutV4ValidationError(
                    f"layout-v4 normal-spectrum attribute {name} disagrees"
                )
        elif observed != expected:
            raise LayoutV4ValidationError(
                f"layout-v4 normal-spectrum attribute {name} disagrees"
            )
    if validator.has(f"{path}/product_present") or validator.has(
        f"{path}/data_valid"
    ):
        raise LayoutV4ValidationError(
            "layout-v4 normal spectra must use NaN absence, not flag arrays"
        )
    metadata, metadata_present, metadata_records = _v4_spectrum_metadata(
        validator, f"{path}/metadata", count
    )
    strict_records = []
    for row, metadata_record in enumerate(metadata_records):
        expected_raw = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        if (
            metadata_record.unique_packet_id != int(common["unique_ids"][row])
            or metadata_record.navgf != int(navgf[row])
            or metadata_record.raw_seconds != expected_raw
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 normal metadata identity disagrees at row {row}"
            )
        contract = contracts[int(window_index[row])]
        nfreq = int(frequency_counts[row])
        product_present = np.all(
            np.isfinite(data[row, :, :nfreq]), axis=1
        )
        try:
            strict_records.append(
                SpectrumSample(
                    data=np.asarray(
                        data[row, :, :nfreq], dtype=np.float32
                    ),
                    product_present=np.asarray(
                        product_present, dtype=np.bool_
                    ),
                    navgf=int(navgf[row]),
                    frequency_contract=contract,
                    unique_packet_id=int(common["unique_ids"][row]),
                    raw_seconds=expected_raw,
                    metadata=metadata_record,
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 normal-spectrum row {row} is invalid: {exc}"
            ) from exc
    bundle.spectra = data
    bundle.spectra_unique_ids = common["unique_ids"]
    bundle.spectra_raw_times = common["raw_seconds"]
    bundle.spectra_raw_time_valid = common["raw_time_valid"]
    bundle.spectra_mjd_times = common["mjd_times"]
    bundle.spectra_mjd_time_valid = common["mjd_time_valid"]
    bundle.spectra_frequency_counts = frequency_counts
    bundle.spectra_navgf = navgf
    bundle.spectra_frequency_window_index = window_index
    bundle.spectra_frequency_windows = contracts
    bundle.spectra_metadata = metadata
    bundle.spectra_metadata_present = metadata_present
    bundle.spectra_units = SPECTRA_UNITS
    bundle.spectra_representation = SPECTRA_REPRESENTATION
    bundle.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION
    bundle.product_records["spectra"] = tuple(strict_records)


def _v4_parse_tr_spectra(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/tr_spectra"
    common = _v4_common_rows(
        validator,
        path,
        "tr_spectra",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    navg2 = int(_v4_scalar_attr(validator, path, "navg2", np.uint32))
    tr_length = int(
        _v4_scalar_attr(validator, path, "tr_length", np.uint32)
    )
    if navg2 < 1 or tr_length < 1:
        raise LayoutV4ValidationError("layout-v4 TR geometry must be positive")
    if (
        _v4_text_attr(validator, path, "native_dtype") != "int32"
        or _v4_text_attr(validator, path, "units") != "unit_unestablished"
        or _v4_text_attr(validator, path, "representation") != "native_int32"
    ):
        raise LayoutV4ValidationError("layout-v4 TR attributes disagree")
    data = validator.dataset(
        f"{path}/data",
        dtype=np.float64,
        utf8=False,
        shape=(count, NPRODUCTS, navg2, tr_length),
    ).data
    navg2_rows = validator.dataset(
        f"{path}/navg2_per_sample",
        dtype=np.uint32,
        utf8=False,
        shape=(count,),
    ).data
    length_rows = validator.dataset(
        f"{path}/tr_length_per_sample",
        dtype=np.uint32,
        utf8=False,
        shape=(count,),
    ).data
    if np.any(navg2_rows != navg2) or np.any(length_rows != tr_length):
        raise LayoutV4ValidationError(
            "layout-v4 TR per-sample geometry disagrees"
        )
    int32 = np.iinfo(np.int32)
    for row in range(count):
        finite_product = np.all(np.isfinite(data[row]), axis=(1, 2))
        nan_product = np.all(np.isnan(data[row]), axis=(1, 2))
        finite = data[row][np.isfinite(data[row])]
        if (
            not np.all(finite_product | nan_product)
            or not np.any(finite_product)
            or np.any(finite != np.trunc(finite))
            or np.any(finite < int32.min)
            or np.any(finite > int32.max)
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 TR values are invalid at row {row}"
            )
    if validator.has(f"{path}/product_present") or validator.has(
        f"{path}/data_valid"
    ):
        raise LayoutV4ValidationError(
            "layout-v4 TR spectra must use NaN absence, not flag arrays"
        )
    metadata, _, metadata_records = _v4_spectrum_metadata(
        validator, f"{path}/metadata", count
    )
    strict_records = []
    for row, metadata_record in enumerate(metadata_records):
        expected_raw = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        expected_navg2 = 1 << metadata_record.navg2_shift
        divisor = 1 << metadata_record.tr_average_shift
        width = metadata_record.tr_stop - metadata_record.tr_start
        if (
            metadata_record.unique_packet_id != int(common["unique_ids"][row])
            or metadata_record.raw_seconds != expected_raw
            or expected_navg2 != navg2
            or width <= 0
            or width % divisor
            or width // divisor != tr_length
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 TR metadata geometry disagrees at row {row}"
            )
        product_present = np.all(
            np.isfinite(data[row]), axis=(1, 2)
        )
        native_data = np.zeros(
            (NPRODUCTS, navg2, tr_length), dtype=np.int32
        )
        native_data[product_present] = data[row, product_present].astype(
            np.int32
        )
        try:
            strict_records.append(
                TRSpectrumSample(
                    data=native_data,
                    product_present=np.asarray(
                        product_present, dtype=np.bool_
                    ),
                    unique_packet_id=int(common["unique_ids"][row]),
                    raw_seconds=expected_raw,
                    navg2=navg2,
                    tr_length=tr_length,
                    metadata=metadata_record,
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 TR-spectrum row {row} is invalid: {exc}"
            ) from exc
    bundle.tr_spectra = data
    bundle.tr_unique_ids = common["unique_ids"]
    bundle.tr_raw_times = common["raw_seconds"]
    bundle.tr_raw_time_valid = common["raw_time_valid"]
    bundle.tr_mjd_times = common["mjd_times"]
    bundle.tr_mjd_time_valid = common["mjd_time_valid"]
    bundle.tr_navg2_per_sample = navg2_rows
    bundle.tr_length_per_sample = length_rows
    bundle.tr_metadata = metadata
    bundle.product_records["tr_spectra"] = tuple(strict_records)


def _v4_parse_zoom(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/calibrator/zoom_spectra"
    common = _v4_common_rows(
        validator,
        path,
        "zoom_spectra",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    data = validator.dataset(
        f"{path}/data",
        dtype=np.float32,
        utf8=False,
        shape=(count, 4, ZOOM_BINS),
    ).data
    pfb_bins = validator.dataset(
        f"{path}/pfb_bins",
        dtype=np.uint16,
        utf8=False,
        shape=(count,),
    ).data
    labels = np.asarray(
        validator.attribute(f"{path}/data", "component_labels")
    )
    if (
        not np.all(np.isfinite(data))
        or _v4_text_attr(validator, f"{path}/data", "units")
        != "unit_unestablished"
        or _v4_text_attr(validator, f"{path}/data", "representation")
        != "native_float32"
        or labels.dtype != np.dtype("S3")
        or not np.array_equal(labels, np.asarray([b"AA", b"BB", b"ABR", b"ABI"], dtype="S3"))
    ):
        raise LayoutV4ValidationError("layout-v4 zoom contract disagrees")
    records = []
    for row in range(count):
        raw_seconds = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        try:
            records.append(
                ZoomSample(
                    data=np.asarray(data[row], dtype=np.float32),
                    unique_packet_id=int(common["unique_ids"][row]),
                    pfb_bin=int(pfb_bins[row]),
                    raw_seconds=raw_seconds,
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 zoom-spectrum row {row} is invalid: {exc}"
            ) from exc
    bundle.zoom_spectra = data
    bundle.zoom_unique_ids = common["unique_ids"]
    bundle.zoom_pfb_bins = pfb_bins
    bundle.zoom_raw_times = common["raw_seconds"]
    bundle.zoom_raw_time_valid = common["raw_time_valid"]
    bundle.zoom_mjd_times = common["mjd_times"]
    bundle.zoom_mjd_time_valid = common["mjd_time_valid"]
    bundle.product_records["zoom_spectra"] = tuple(records)


def _v4_parse_waveforms(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/waveform"
    common = _v4_common_rows(
        validator,
        path,
        "waveforms",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    data = validator.dataset(
        f"{path}/data",
        dtype=np.int16,
        utf8=False,
        shape=(count, WAVEFORM_SAMPLES),
    ).data
    channels = validator.dataset(
        f"{path}/channel",
        dtype=np.uint8,
        utf8=False,
        shape=(count,),
    ).data
    adc_timestamps = validator.dataset(
        f"{path}/adc_timestamps",
        dtype=np.uint64,
        utf8=False,
        shape=(count,),
    ).data
    adc_valid = validator.dataset(
        f"{path}/adc_timestamp_valid",
        dtype=np.bool_,
        utf8=False,
        shape=(count,),
    ).data
    if (
        np.any(channels >= _N_ADC_CHANNELS)
        or not np.all(adc_valid)
        or _v4_text_attr(validator, path, "adc_clock_source")
        != ClockSource.ADC.value
        or _v4_text_attr(validator, f"{path}/data", "units") != "raw_count"
        or _v4_text_attr(validator, f"{path}/data", "representation")
        != "native_int16"
    ):
        raise LayoutV4ValidationError("layout-v4 waveform contract disagrees")
    records = []
    for row in range(count):
        raw_seconds = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        try:
            records.append(
                WaveformSample(
                    data=np.asarray(data[row], dtype=np.int16),
                    channel=int(channels[row]),
                    unique_packet_id=int(common["unique_ids"][row]),
                    raw_seconds=raw_seconds,
                    adc_timestamp=np.uint64(adc_timestamps[row]),
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 waveform row {row} is invalid: {exc}"
            ) from exc
    bundle.waveform_data = data
    bundle.waveform_channels = channels
    bundle.waveform_unique_ids = common["unique_ids"]
    bundle.waveform_raw_times = common["raw_seconds"]
    bundle.waveform_raw_time_valid = common["raw_time_valid"]
    bundle.waveform_mjd_times = common["mjd_times"]
    bundle.waveform_mjd_time_valid = common["mjd_time_valid"]
    bundle.waveform_adc_timestamps = adc_timestamps
    bundle.waveform_adc_timestamp_valid = adc_valid
    bundle.product_records["waveforms"] = tuple(records)


def _v4_parse_grimm(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/grimm_spectra"
    common = _v4_common_rows(
        validator,
        path,
        "grimm_spectra",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    navg2_rows = validator.dataset(
        f"{path}/navg2_per_sample",
        dtype=np.uint32,
        utf8=False,
        shape=(count,),
    ).data
    if count < 1 or np.any(navg2_rows < 1):
        raise LayoutV4ValidationError("layout-v4 Grimm Navg2 is invalid")
    navg2_max = int(np.max(navg2_rows))
    data = validator.dataset(
        f"{path}/data",
        dtype=np.int32,
        utf8=False,
        shape=(count, navg2_max, NPRODUCTS, 4),
    ).data
    average_valid = validator.dataset(
        f"{path}/average_valid",
        dtype=np.bool_,
        utf8=False,
        shape=(count, navg2_max),
    ).data
    expected_valid = np.arange(navg2_max)[None, :] < navg2_rows[:, None]
    axis_labels = np.asarray(validator.attribute(f"{path}/data", "axis_labels"))
    value_labels = np.asarray(
        validator.attribute(f"{path}/data", "value_axis_labels")
    )
    if (
        not np.array_equal(average_valid, expected_valid)
        or np.any(data[~average_valid] != 0)
        or _v4_text_attr(validator, f"{path}/data", "units")
        != "unit_unestablished"
        or _v4_text_attr(validator, f"{path}/data", "representation")
        != "native_int32"
        or axis_labels.dtype != np.dtype("S24")
        or not np.array_equal(
            axis_labels,
            np.asarray(
                ["average_index", "product_index", "grimm_value_index"],
                dtype="S24",
            ),
        )
        or value_labels.dtype != np.dtype("S16")
        or not np.array_equal(
            value_labels,
            np.asarray(["value_0", "value_1", "value_2", "value_3"], dtype="S16"),
        )
    ):
        raise LayoutV4ValidationError("layout-v4 Grimm contract disagrees")
    records = []
    for row in range(count):
        raw_seconds = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        row_navg2 = int(navg2_rows[row])
        try:
            records.append(
                GrimmSample(
                    data=np.asarray(
                        data[row, :row_navg2], dtype=np.int32
                    ),
                    unique_packet_id=int(common["unique_ids"][row]),
                    raw_seconds=raw_seconds,
                    navg2=row_navg2,
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 Grimm row {row} is invalid: {exc}"
            ) from exc
    bundle.grimm_spectra = data
    bundle.grimm_unique_ids = common["unique_ids"]
    bundle.grimm_raw_times = common["raw_seconds"]
    bundle.grimm_raw_time_valid = common["raw_time_valid"]
    bundle.grimm_mjd_times = common["mjd_times"]
    bundle.grimm_mjd_time_valid = common["mjd_time_valid"]
    bundle.grimm_navg2_per_sample = navg2_rows
    bundle.grimm_average_valid = average_valid
    bundle.product_records["grimm_spectra"] = tuple(records)


def _v4_parse_housekeeping(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    path = "/housekeeping"
    common = _v4_common_rows(
        validator,
        path,
        "housekeeping",
        product_provenance,
        bundle.clock_reference_set,
    )
    count = common["unique_ids"].size
    hk_types = validator.dataset(
        f"{path}/hk_type",
        dtype=np.uint16,
        utf8=False,
        shape=(count,),
    ).data
    versions = validator.dataset(
        f"{path}/version",
        dtype=np.uint16,
        utf8=False,
        shape=(count,),
    ).data
    firmware_errors = validator.dataset(
        f"{path}/firmware_errors",
        dtype=np.uint32,
        utf8=False,
        shape=(count,),
    ).data
    if np.any(~np.isin(hk_types, (0, 1, 2, 3, 100, 101))):
        raise LayoutV4ValidationError("layout-v4 housekeeping type is invalid")
    rows, presence = decode_field_union_rows(validator, path, count)
    records = []
    for row in range(count):
        raw_seconds = (
            float(common["raw_seconds"][row])
            if common["raw_time_valid"][row]
            else None
        )
        try:
            records.append(
                HKSample(
                    hk_type=int(hk_types[row]),
                    version=int(versions[row]),
                    unique_packet_id=int(common["unique_ids"][row]),
                    errors=int(firmware_errors[row]),
                    raw_seconds=raw_seconds,
                    fields=rows[row],
                    field_present=presence[row],
                    provenance=_v4_product_record(
                        common, row, product_provenance
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 housekeeping row {row} is invalid: {exc}"
            ) from exc
    bundle.housekeeping_unique_ids = common["unique_ids"]
    bundle.housekeeping_raw_times = common["raw_seconds"]
    bundle.housekeeping_raw_time_valid = common["raw_time_valid"]
    bundle.housekeeping_mjd_times = common["mjd_times"]
    bundle.housekeeping_mjd_time_valid = common["mjd_time_valid"]
    bundle.housekeeping_types = hk_types
    bundle.housekeeping_versions = versions
    bundle.housekeeping_firmware_errors = firmware_errors
    bundle.housekeeping_fields = rows
    bundle.housekeeping_field_present = presence
    bundle.product_records["housekeeping"] = tuple(records)


def _v4_validate_page_times(
    validator,
    path: str,
    common: Dict[str, np.ndarray],
    page_count: int,
    clock_reference_set: Optional[ClockReferenceSet],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = common["unique_ids"].size
    raw = validator.dataset(
        f"{path}/page_raw_seconds",
        dtype=np.float64,
        utf8=False,
        shape=(count, page_count),
    ).data
    mjd = validator.dataset(
        f"{path}/page_mjd_times",
        dtype=np.float64,
        utf8=False,
        shape=(count, page_count),
    ).data
    valid = validator.dataset(
        f"{path}/page_mjd_time_valid",
        dtype=np.bool_,
        utf8=False,
        shape=(count, page_count),
    ).data
    if (
        not np.all(np.isfinite(raw))
        or not np.array_equal(raw[:, 0], common["raw_seconds"])
    ):
        raise LayoutV4ValidationError(
            f"layout-v4 calibrator page raw times disagree at {path}"
        )
    expected_valid = np.zeros(raw.shape, dtype=np.bool_)
    if (
        clock_reference_set is not None
        and clock_reference_set.reference_for(ClockSource.SPECTROMETER) is not None
    ):
        expected_valid[:] = True
        expected_mjd = clock_reference_set.to_mjd(
            raw,
            clock_source=ClockSource.SPECTROMETER,
        )
        if not np.allclose(mjd, expected_mjd, rtol=0.0, atol=1e-12):
            raise LayoutV4ValidationError(
                f"layout-v4 calibrator page MJD disagrees at {path}"
            )
    elif not np.all(np.isnan(mjd)):
        raise LayoutV4ValidationError(
            f"layout-v4 uncalibrated page MJD must be NaN at {path}"
        )
    if not np.array_equal(valid, expected_valid):
        raise LayoutV4ValidationError(
            f"layout-v4 calibrator page validity disagrees at {path}"
        )
    return raw, mjd, valid


def _v4_parse_calibrator(
    validator,
    bundle: SessionBundle,
    product_provenance: Dict[str, object],
) -> None:
    family_paths = (
        ("metadata", "/calibrator/metadata", "calibrator_metadata"),
        ("data", "/calibrator/data", "calibrator_data"),
        ("raw_pfb", "/calibrator/raw_pfb", "calibrator_raw_pfb"),
        ("debug", "/calibrator/debug", "calibrator_debug"),
    )
    for public_name, path, family in family_paths:
        if not validator.has(path):
            continue
        common = _v4_common_rows(
            validator,
            path,
            family,
            product_provenance,
            bundle.clock_reference_set,
        )
        count = common["unique_ids"].size
        result: Dict[str, object] = dict(common)
        if public_name == "metadata":
            result["from_debug"] = validator.dataset(
                f"{path}/from_debug",
                dtype=np.bool_,
                utf8=False,
                shape=(count,),
            ).data
            rows, presence = decode_field_union_rows(validator, path, count)
            result["fields"] = rows
            result["field_present"] = presence
            records = []
            for row in range(count):
                raw_seconds = (
                    float(common["raw_seconds"][row])
                    if common["raw_time_valid"][row]
                    else None
                )
                try:
                    records.append(
                        CalibratorMetadataSample(
                            unique_packet_id=int(
                                common["unique_ids"][row]
                            ),
                            raw_seconds=raw_seconds,
                            from_debug=bool(result["from_debug"][row]),
                            fields=rows[row],
                            field_present=presence[row],
                            provenance=_v4_product_record(
                                common, row, product_provenance
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise LayoutV4ValidationError(
                        "layout-v4 calibrator metadata row "
                        f"{row} is invalid: {exc}"
                    ) from exc
            result["records"] = tuple(records)
        elif public_name in ("data", "raw_pfb"):
            page_count = 3 if public_name == "data" else 8
            channel_count = 512 if public_name == "data" else NCHANNELS
            stored_page_count = int(
                _v4_scalar_attr(validator, path, "page_count", np.uint8)
            )
            channel_labels = np.asarray(
                validator.attribute(path, "channel_labels")
            )
            if (
                stored_page_count != page_count
                or _v4_text_attr(validator, path, "page_clock_source")
                != ClockSource.SPECTROMETER.value
                or _v4_text_attr(validator, path, "units")
                != "unit_unestablished"
                or _v4_text_attr(validator, path, "representation")
                != "native_complex128"
                or channel_labels.dtype != np.dtype(np.uint8)
                or not np.array_equal(
                    channel_labels, np.arange(4, dtype=np.uint8)
                )
            ):
                raise LayoutV4ValidationError(
                    f"layout-v4 calibrator attributes disagree at {path}"
                )
            real = validator.dataset(
                f"{path}/data_real",
                dtype=np.float64,
                utf8=False,
                shape=(count, 4, channel_count),
            ).data
            imag = validator.dataset(
                f"{path}/data_imag",
                dtype=np.float64,
                utf8=False,
                shape=(count, 4, channel_count),
            ).data
            if not np.all(np.isfinite(real)) or not np.all(np.isfinite(imag)):
                raise LayoutV4ValidationError(
                    f"layout-v4 calibrator complex data is nonfinite at {path}"
                )
            result["data"] = real + 1j * imag
            page_raw, page_mjd, page_valid = _v4_validate_page_times(
                validator,
                path,
                common,
                page_count,
                bundle.clock_reference_set,
            )
            result["page_raw_seconds"] = page_raw
            result["page_mjd_times"] = page_mjd
            result["page_mjd_time_valid"] = page_valid
            if public_name == "data":
                result["g_nacc"] = validator.dataset(
                    f"{path}/g_nacc",
                    dtype=np.int32,
                    utf8=False,
                    shape=(count,),
                ).data
                result["gphase"] = validator.dataset(
                    f"{path}/gphase",
                    dtype=np.int32,
                    utf8=False,
                    shape=(count, 1024),
                ).data
            records = []
            for row in range(count):
                raw_seconds = (
                    float(common["raw_seconds"][row])
                    if common["raw_time_valid"][row]
                    else None
                )
                provenance = _v4_product_record(
                    common, row, product_provenance
                )
                try:
                    if public_name == "data":
                        record = CalibratorDataSample(
                            data=np.asarray(
                                result["data"][row], dtype=np.complex128
                            ),
                            g_nacc=int(result["g_nacc"][row]),
                            gphase=np.asarray(
                                result["gphase"][row], dtype=np.int32
                            ),
                            unique_packet_id=int(
                                common["unique_ids"][row]
                            ),
                            raw_seconds=raw_seconds,
                            page_raw_seconds=np.asarray(
                                page_raw[row], dtype=np.float64
                            ),
                            provenance=provenance,
                        )
                    else:
                        record = CalibratorRawPFBSample(
                            data=np.asarray(
                                result["data"][row], dtype=np.complex128
                            ),
                            unique_packet_id=int(
                                common["unique_ids"][row]
                            ),
                            raw_seconds=raw_seconds,
                            page_raw_seconds=np.asarray(
                                page_raw[row], dtype=np.float64
                            ),
                            provenance=provenance,
                        )
                except (TypeError, ValueError) as exc:
                    raise LayoutV4ValidationError(
                        f"layout-v4 calibrator {public_name} row {row} "
                        f"is invalid: {exc}"
                    ) from exc
                records.append(record)
            result["records"] = tuple(records)
        else:
            page_count = int(
                _v4_scalar_attr(validator, path, "page_count", np.uint8)
            )
            if (
                page_count != 8
                or _v4_text_attr(validator, path, "page_clock_source")
                != ClockSource.SPECTROMETER.value
            ):
                raise LayoutV4ValidationError(
                    "layout-v4 calibrator debug attributes disagree"
                )
            page_raw, page_mjd, page_valid = _v4_validate_page_times(
                validator,
                path,
                common,
                page_count,
                bundle.clock_reference_set,
            )
            result["page_raw_seconds"] = page_raw
            result["page_mjd_times"] = page_mjd
            result["page_mjd_time_valid"] = page_valid
            pages_group = validator.group(f"{path}/pages")
            expected_pages = {f"page_{index}" for index in range(page_count)}
            if set(pages_group.children) != expected_pages:
                raise LayoutV4ValidationError(
                    "layout-v4 calibrator debug pages disagree"
                )
            pages = []
            for page_index in range(page_count):
                page_path = f"{path}/pages/page_{page_index}"
                stored_index = int(
                    _v4_scalar_attr(
                        validator, page_path, "page_index", np.uint8
                    )
                )
                if stored_index != page_index:
                    raise LayoutV4ValidationError(
                        "layout-v4 calibrator debug page index disagrees"
                    )
                rows, presence = decode_field_union_rows(
                    validator, page_path, count
                )
                pages.append({"fields": rows, "field_present": presence})
            result["pages"] = tuple(pages)
            records = []
            for row in range(count):
                raw_seconds = (
                    float(common["raw_seconds"][row])
                    if common["raw_time_valid"][row]
                    else None
                )
                try:
                    records.append(
                        CalibratorDebugSample(
                            pages=tuple(
                                CalibratorDebugPage(
                                    page=page_index,
                                    fields=pages[page_index]["fields"][row],
                                    field_present=pages[page_index][
                                        "field_present"
                                    ][row],
                                )
                                for page_index in range(page_count)
                            ),
                            unique_packet_id=int(
                                common["unique_ids"][row]
                            ),
                            raw_seconds=raw_seconds,
                            page_raw_seconds=np.asarray(
                                page_raw[row], dtype=np.float64
                            ),
                            provenance=_v4_product_record(
                                common, row, product_provenance
                            ),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise LayoutV4ValidationError(
                        "layout-v4 calibrator debug row "
                        f"{row} is invalid: {exc}"
                    ) from exc
            result["records"] = tuple(records)
        bundle.calibrator[public_name] = result
        bundle.product_records[family] = result["records"]


def _v4_parse_issues(validator) -> Dict[str, object]:
    path = "/issues"
    count = int(_v4_scalar_attr(validator, path, "count", np.uint64))
    text_names = (
        "issue_id",
        "code",
        "stage",
        "message",
        "severity",
        "action",
        "input_identity",
        "bank",
        "session",
        "details_json",
    )
    integer_specs = {
        "byte_offset": np.uint64,
        "frame_index": np.uint64,
        "packet_index": np.uint64,
        "appid": np.uint16,
        "sequence_count": np.uint16,
        "uid": np.uint32,
    }
    expected = set(text_names)
    for name in ("input_identity", "bank", "session"):
        expected.add(f"{name}_valid")
    for name in integer_specs:
        expected.update((name, f"{name}_valid"))
    group = validator.group(path)
    if set(group.children) != expected:
        raise LayoutV4ValidationError("layout-v4 issue columns disagree")
    result = {
        name: validator.dataset(
            f"{path}/{name}", utf8=True, shape=(count,)
        ).data
        for name in text_names
    }
    for name, dtype in integer_specs.items():
        result[name] = validator.dataset(
            f"{path}/{name}",
            dtype=dtype,
            utf8=False,
            shape=(count,),
        ).data
        result[f"{name}_valid"] = validator.dataset(
            f"{path}/{name}_valid",
            dtype=np.bool_,
            utf8=False,
            shape=(count,),
        ).data
    for name in ("input_identity", "bank", "session"):
        result[f"{name}_valid"] = validator.dataset(
            f"{path}/{name}_valid",
            dtype=np.bool_,
            utf8=False,
            shape=(count,),
        ).data
    if len(set(result["issue_id"].tolist())) != count:
        raise LayoutV4ValidationError("layout-v4 issue IDs are not unique")
    if np.any(~np.isin(result["severity"], ("info", "warning", "error"))) or np.any(
        ~np.isin(
            result["action"],
            ("kept", "dropped", "rejected", "assumed", "overridden"),
        )
    ):
        raise LayoutV4ValidationError("layout-v4 issue enum value is invalid")
    for severity in ("info", "warning", "error"):
        stored = int(
            _v4_scalar_attr(
                validator, "/", f"{severity}_issue_count", np.uint64
            )
        )
        observed = int(np.count_nonzero(result["severity"] == severity))
        if stored != observed:
            raise LayoutV4ValidationError(
                f"layout-v4 {severity} issue count disagrees"
            )
    records = []
    integer_limits = {
        "byte_offset": (1 << 64) - 1,
        "frame_index": (1 << 64) - 1,
        "packet_index": (1 << 64) - 1,
        "appid": 0x7FF,
        "sequence_count": 0x3FFF,
        "uid": 0xFFFFFFFF,
    }
    for row, details in enumerate(result["details_json"]):
        for name in ("issue_id", "code", "stage", "message"):
            if not str(result[name][row]):
                raise LayoutV4ValidationError(
                    f"layout-v4 issue {name} is empty at row {row}"
                )
        try:
            value = json.loads(str(details))
        except json.JSONDecodeError as exc:
            raise LayoutV4ValidationError(
                "layout-v4 issue details JSON is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise LayoutV4ValidationError(
                "layout-v4 issue details JSON must be an object"
            )
        try:
            canonical = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 issue details JSON is invalid at row {row}"
            ) from exc
        if str(details) != canonical:
            raise LayoutV4ValidationError(
                f"layout-v4 issue details JSON is not canonical at row {row}"
            )

        def optional_text(name: str) -> Optional[str]:
            valid = bool(result[f"{name}_valid"][row])
            text = str(result[name][row])
            if valid != bool(text):
                raise LayoutV4ValidationError(
                    f"layout-v4 issue {name} validity disagrees at row {row}"
                )
            return text if valid else None

        def optional_integer(name: str) -> Optional[int]:
            valid = bool(result[f"{name}_valid"][row])
            integer = int(result[name][row])
            if not valid:
                if integer != 0:
                    raise LayoutV4ValidationError(
                        f"layout-v4 issue {name} absence disagrees at row {row}"
                    )
                return None
            if integer > integer_limits[name]:
                raise LayoutV4ValidationError(
                    f"layout-v4 issue {name} is out of range at row {row}"
                )
            return integer

        try:
            records.append(
                IngestIssue(
                    issue_id=str(result["issue_id"][row]),
                    code=str(result["code"][row]),
                    severity=str(result["severity"][row]),
                    stage=str(result["stage"][row]),
                    message=str(result["message"][row]),
                    action=str(result["action"][row]),
                    input_identity=optional_text("input_identity"),
                    bank=optional_text("bank"),
                    byte_offset=optional_integer("byte_offset"),
                    frame_index=optional_integer("frame_index"),
                    packet_index=optional_integer("packet_index"),
                    appid=optional_integer("appid"),
                    sequence_count=optional_integer("sequence_count"),
                    uid=optional_integer("uid"),
                    session=optional_text("session"),
                    details=tuple(value.items()),
                )
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 issue row {row} is invalid: {exc}"
            ) from exc
    result["records"] = tuple(records)
    return result


def _v4_parse_family_status(
    validator,
    family_counts: Dict[str, int],
    issue_ids: np.ndarray,
) -> Dict[str, object]:
    path = "/status/families"
    text_names = ("family", "coverage", "quality", "reason")
    count = len(ALL_FAMILIES)
    expected_families = tuple(sorted(ALL_FAMILIES))
    result: Dict[str, np.ndarray] = {
        name: validator.dataset(
            f"{path}/{name}", utf8=True, shape=(count,)
        ).data
        for name in text_names
    }
    for name, dtype in (
        ("supported", np.bool_),
        ("decoded_rows", np.uint64),
        ("persisted_rows", np.uint64),
        ("reason_valid", np.bool_),
    ):
        result[name] = validator.dataset(
            f"{path}/{name}",
            dtype=dtype,
            utf8=False,
            shape=(count,),
        ).data
    if tuple(result["family"].tolist()) != expected_families:
        raise LayoutV4ValidationError(
            "layout-v4 family status order or membership disagrees"
        )
    expected_persisted = np.asarray(
        [family_counts.get(name, 0) for name in expected_families], dtype=np.uint64
    )
    if not np.array_equal(result["persisted_rows"], expected_persisted):
        raise LayoutV4ValidationError(
            "layout-v4 family persisted counts disagree"
        )
    if np.any(
        (result["coverage"] == "persisted")
        != (result["persisted_rows"] > 0)
    ):
        raise LayoutV4ValidationError(
            "layout-v4 family coverage disagrees with persisted rows"
        )
    ref_path = "/status/family_issue_refs"
    ref_count = validator.require_row_aligned(
        ref_path, ("family_index", "issue_index")
    )
    family_index = validator.dataset(
        f"{ref_path}/family_index",
        dtype=np.uint64,
        utf8=False,
        shape=(ref_count,),
    ).data
    issue_index = validator.dataset(
        f"{ref_path}/issue_index",
        dtype=np.uint64,
        utf8=False,
        shape=(ref_count,),
    ).data
    if np.any(family_index >= count) or np.any(issue_index >= issue_ids.size):
        raise LayoutV4ValidationError(
            "layout-v4 family issue reference is out of range"
        )
    records = []
    for index, family in enumerate(expected_families):
        reason_valid = bool(result["reason_valid"][index])
        reason_text = str(result["reason"][index])
        if reason_valid == (not reason_text):
            raise LayoutV4ValidationError(
                f"layout-v4 family reason validity disagrees for {family}"
            )
        refs = issue_index[family_index == index]
        try:
            record = FamilyStatus(
                family=family,
                supported=bool(result["supported"][index]),
                coverage=str(result["coverage"][index]),
                quality=str(result["quality"][index]),
                decoded_rows=int(result["decoded_rows"][index]),
                issue_ids=tuple(str(issue_ids[ref]) for ref in refs),
                reason=reason_text if reason_valid else None,
            )
        except (TypeError, ValueError) as exc:
            raise LayoutV4ValidationError(
                f"layout-v4 family status for {family} is invalid: {exc}"
            ) from exc
        expected_rows = (
            record.decoded_rows
            if record.coverage.value == "persisted"
            else 0
        )
        if int(result["persisted_rows"][index]) != expected_rows:
            raise LayoutV4ValidationError(
                f"layout-v4 family persisted count disagrees for {family}"
            )
        records.append(record)
    return {
        **result,
        "issue_refs": {
            "family_index": family_index,
            "issue_index": issue_index,
        },
        "records": tuple(records),
    }


def _load_layout_v4_tree(root, path: Path) -> SessionBundle:
    validator = validate_layout_v4_tree(root)
    bundle = SessionBundle(
        source_path=path,
        source_paths=(path,),
        layout_version=INGEST_LAYOUT_VERSION,
    )
    bundle.quality_status = _v4_text_attr(validator, "/", "quality_status")
    bundle.execution_mode = _v4_text_attr(validator, "/", "execution_mode")
    bundle.constants = {
        name: float(
            _v4_scalar_attr(validator, "/constants", name, np.float64)
        )
        for name in ("lun_lat_deg", "lun_long_deg", "lun_height_m")
    }
    bundle.session_invariants = _v4_session_invariants(validator)
    bundle.clock_reference_set, bundle.clock_reference_unavailable_reason = (
        _v4_parse_clock_reference(validator)
    )
    bundle.clock_reference_unavailable_reasons = (
        bundle.clock_reference_unavailable_reason,
    )
    bundle.run_provenance = _v4_parse_run_provenance(validator)
    bundle.decoder_provenance = _v4_parse_decoder_provenance(validator)
    bundle.issues = _v4_parse_issues(validator)
    bundle.product_provenance = _v4_parse_product_provenance(validator)

    parsers = (
        ("/spectra", "spectra", _v4_parse_spectra),
        ("/tr_spectra", "tr_spectra", _v4_parse_tr_spectra),
        ("/calibrator/zoom_spectra", "zoom_spectra", _v4_parse_zoom),
        ("/waveform", "waveforms", _v4_parse_waveforms),
        ("/housekeeping", "housekeeping", _v4_parse_housekeeping),
        ("/grimm_spectra", "grimm_spectra", _v4_parse_grimm),
    )
    family_counts: Dict[str, int] = {}
    for family_path, family, parser in parsers:
        if validator.has(family_path):
            parser(validator, bundle, bundle.product_provenance)
            family_counts[family] = int(
                _v4_scalar_attr(validator, family_path, "count", np.uint64)
            )
    _v4_parse_calibrator(validator, bundle, bundle.product_provenance)
    for public_name, family in (
        ("metadata", "calibrator_metadata"),
        ("data", "calibrator_data"),
        ("raw_pfb", "calibrator_raw_pfb"),
        ("debug", "calibrator_debug"),
    ):
        if public_name in bundle.calibrator:
            family_counts[family] = len(
                bundle.calibrator[public_name]["unique_ids"]
            )
    bundle.family_status = _v4_parse_family_status(
        validator,
        family_counts,
        bundle.issues["issue_id"],
    )
    if bundle.quality_status == DataQuality.PARTIAL.value and not bundle.issues[
        "records"
    ]:
        raise LayoutV4ValidationError(
            "layout-v4 partial root quality requires a recorded issue"
        )
    if bundle.quality_status == DataQuality.CLEAN.value and any(
        status.supported and status.quality is not DataQuality.CLEAN
        for status in bundle.family_status["records"]
    ):
        raise LayoutV4ValidationError(
            "layout-v4 clean root quality contains a degraded supported family"
        )

    product_rows = bundle.product_provenance["product_rows"]
    if len(product_rows["family"]) != sum(family_counts.values()):
        raise LayoutV4ValidationError(
            "layout-v4 product provenance row total disagrees"
        )
    for family, count in zip(
        bundle.family_status["family"],
        bundle.family_status["persisted_rows"],
    ):
        decoded_attr = f"decoded_{family}_rows"
        persisted_attr = f"persisted_{family}_rows"
        decoded = int(
            _v4_scalar_attr(validator, "/", decoded_attr, np.uint64)
        )
        persisted = int(
            _v4_scalar_attr(validator, "/", persisted_attr, np.uint64)
        )
        status_index = tuple(bundle.family_status["family"]).index(family)
        if (
            decoded != int(bundle.family_status["decoded_rows"][status_index])
            or persisted != int(count)
        ):
            raise LayoutV4ValidationError(
                f"layout-v4 root family count disagrees for {family}"
            )
    return bundle


# ---------------------------------------------------------------------------
# HDF5 reader
# ---------------------------------------------------------------------------

def _require_supported_layout_version(value, path: Path) -> int:
    array = np.asarray(value)
    if value is None or array.shape != ():
        raise ValueError(
            f"{path}: ingest input requires an explicit scalar layout version"
        )
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(
        scalar,
        (int, np.integer),
    ):
        raise ValueError(f"{path}: ingest layout version is not an integer")
    version = int(scalar)
    if version not in (2, 3, INGEST_LAYOUT_VERSION):
        raise ValueError(
            f"{path}: unsupported ingest layout version {version}; "
            "expected 2, 3, or 4"
        )
    return version


def _load_h5(path: Path) -> SessionBundle:
    h5py = import_optional_dependency("h5py", "HDF5 ingest input")

    with h5py.File(path, "r") as source:
        layout_version = _require_supported_layout_version(
            source.attrs.get("layout_version"),
            path,
        )
        if layout_version == INGEST_LAYOUT_VERSION:
            return _load_layout_v4_tree(read_layout_v4_hdf5(path), path)

    bundle = SessionBundle(source_path=path)
    with h5py.File(path, "r") as f:
        if "layout_version" in f.attrs:
            bundle.layout_version = int(f.attrs["layout_version"])

        # session_invariants
        if "session_invariants" in f:
            for k, v in f["session_invariants"].attrs.items():
                bundle.session_invariants[k] = _scalarize(v)

        # constants: numeric values coerced to float; the provenance attrs
        # are strings by schema (key-based, so a numeric-looking value like
        # clock_source="2" stays a string)
        if "constants" in f:
            for k, v in f["constants"].attrs.items():
                sv = _scalarize(v)
                if isinstance(sv, bytes):
                    sv = sv.decode()
                if k in ("time_scale", "clock_source", "clock_epoch_isot"):
                    bundle.constants[k] = str(sv)
                    continue
                try:
                    bundle.constants[k] = float(sv)
                except (TypeError, ValueError):
                    bundle.constants[k] = str(sv)

        # spectra
        if "spectra" in f and "data" in f["spectra"]:
            sp = f["spectra"]
            data_ds = sp["data"]
            bundle.spectra = data_ds[...]
            if (bundle.spectra.ndim != 3
                    or bundle.spectra.shape[1] != NPRODUCTS):
                raise ValueError(
                    f"{path}: normal spectra must have shape "
                    f"(N, {NPRODUCTS}, Nfreq); got {bundle.spectra.shape}"
                )
            bundle.spectra_units = _attr_text(data_ds.attrs.get("units"))
            bundle.spectra_representation = _attr_text(
                data_ds.attrs.get("representation")
            )
            if "normalization_version" in data_ds.attrs:
                bundle.spectra_normalization_version = int(
                    data_ds.attrs["normalization_version"]
                )
            bundle.spectra_unique_ids = _read_or_none(sp, "unique_ids")
            bundle.spectra_raw_times = _read_or_none(sp, "raw_times")
            bundle.spectra_mjd_times = _read_or_none(sp, "mjd_times")
            bundle.spectra_frequency_counts = _read_or_none(
                sp, "frequency_counts"
            )
            if "metadata" in sp:
                for k, ds in sp["metadata"].items():
                    bundle.spectra_metadata[k] = ds[...]

            n_spectra = bundle.spectra.shape[0]
            actual_bitslice = bundle.spectra_metadata.get("actual_bitslice")
            if actual_bitslice is None:
                raise ValueError(
                    f"{path}: normal spectra have no actual_bitslice metadata"
                )
            if (bundle.layout_version == 3
                    and np.asarray(actual_bitslice).shape
                    != (n_spectra, NPRODUCTS)):
                raise ValueError(
                    f"{path}: layout-v3 actual_bitslice must have exact shape "
                    f"({n_spectra}, {NPRODUCTS}); got "
                    f"{np.asarray(actual_bitslice).shape}"
                )
            # Keep critical metadata in one stable shape for conversion and
            # auditing, even when older writers used (N,1,16).
            bundle.spectra_metadata["actual_bitslice"] = (
                canonical_actual_bitslice(actual_bitslice, n_spectra)
            )
            actual_gain = bundle.spectra_metadata.get("actual_gain")
            if bundle.layout_version == 3 and actual_gain is None:
                raise ValueError(
                    f"{path}: layout-v3 normal spectra have no actual_gain metadata"
                )
            if actual_gain is not None:
                actual_gain = np.asarray(actual_gain)
                if (bundle.layout_version == 3
                        and actual_gain.shape
                        != (n_spectra, _N_ADC_CHANNELS)):
                    raise ValueError(
                        f"{path}: layout-v3 actual_gain must have exact shape "
                        f"({n_spectra}, {_N_ADC_CHANNELS}); got "
                        f"{actual_gain.shape}"
                    )
                if actual_gain.size != n_spectra * _N_ADC_CHANNELS:
                    raise ValueError(
                        f"{path}: actual_gain must have shape "
                        f"({n_spectra}, {_N_ADC_CHANNELS}); got {actual_gain.shape}"
                    )
                bundle.spectra_metadata["actual_gain"] = actual_gain.reshape(
                    n_spectra, _N_ADC_CHANNELS
                )

            if bundle.layout_version == 2:
                bundle.spectra = restore_bitsliced_spectra(
                    bundle.spectra,
                    bundle.spectra_metadata["actual_bitslice"],
                )
                bundle.spectra_units = SPECTRA_UNITS
                bundle.spectra_representation = SPECTRA_REPRESENTATION
                bundle.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION
                warnings.warn(
                    f"{path}: layout-v2 spectra were bit-slice restored in memory; "
                    "re-ingest to produce an unambiguous layout-v3 file",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif bundle.layout_version == 3:
                if int(data_ds.attrs.get("bitslice_restored", 0)) != 1:
                    raise ValueError(
                        f"{path}: layout-v3 /spectra/data does not declare "
                        "bitslice_restored=1"
                    )
                if int(data_ds.attrs.get("bitslice_reference", -1)) != BITSLICE_REFERENCE:
                    raise ValueError(
                        f"{path}: unsupported layout-v3 bit-slice reference"
                    )
                if bundle.spectra_units != SPECTRA_UNITS:
                    raise ValueError(
                        f"{path}: layout-v3 spectra units must be {SPECTRA_UNITS!r}; "
                        f"got {bundle.spectra_units!r}"
                    )
                if bundle.spectra_representation != SPECTRA_REPRESENTATION:
                    raise ValueError(
                        f"{path}: layout-v3 spectra representation must be "
                        f"{SPECTRA_REPRESENTATION!r}; got "
                        f"{bundle.spectra_representation!r}"
                    )
                if (bundle.spectra_normalization_version
                        != SPECTRA_NORMALIZATION_VERSION):
                    raise ValueError(
                        f"{path}: layout-v3 normalization_version must be "
                        f"{SPECTRA_NORMALIZATION_VERSION}; got "
                        f"{bundle.spectra_normalization_version!r}"
                    )
            else:
                raise ValueError(
                    f"{path}: normal spectra require explicit layout_version 2 or 3; "
                    f"got {bundle.layout_version!r}"
                )

        # tr_spectra
        if "tr_spectra" in f and "data" in f["tr_spectra"]:
            tr = f["tr_spectra"]
            bundle.tr_spectra = tr["data"][...]
            bundle.tr_unique_ids = _read_or_none(tr, "unique_ids")
            bundle.tr_raw_times = _read_or_none(tr, "raw_times")
            bundle.tr_mjd_times = _read_or_none(tr, "mjd_times")
            bundle.tr_navg2_per_sample = _read_or_none(tr, "navg2_per_sample")
            bundle.tr_length_per_sample = _read_or_none(tr, "tr_length_per_sample")
            if "metadata" in tr:
                for k, ds in tr["metadata"].items():
                    bundle.tr_metadata[k] = ds[...]

        # zoom_spectra
        zs_path = "calibrator/zoom_spectra"
        if zs_path in f and "data" in f[zs_path]:
            zs = f[zs_path]
            bundle.zoom_spectra = zs["data"][...]
            bundle.zoom_unique_ids = _read_or_none(zs, "unique_ids")
            bundle.zoom_pfb_indices = _read_or_none(zs, "pfb_indices")
            bundle.zoom_raw_times = _read_or_none(zs, "raw_times")
            bundle.zoom_mjd_times = _read_or_none(zs, "mjd_times")

        # grimm
        if "grimm_spectra" in f and "data" in f["grimm_spectra"]:
            gr = f["grimm_spectra"]
            bundle.grimm_spectra = gr["data"][...]
            bundle.grimm_unique_ids = _read_or_none(gr, "unique_ids")
            bundle.grimm_raw_times = _read_or_none(gr, "raw_times")

        # waveform/channel_<N>/{waveforms, timestamps}
        if "waveform" in f:
            for ch_name, gch in f["waveform"].items():
                if not ch_name.startswith("channel_"):
                    continue
                ch = int(ch_name.split("_", 1)[1])
                if "waveforms" in gch:
                    bundle.waveforms[ch] = gch["waveforms"][...]
                if "timestamps" in gch:
                    bundle.waveform_times[ch] = gch["timestamps"][...]

        # housekeeping/type_<N>/<field>
        if "housekeeping" in f:
            for tname, gtype in f["housekeeping"].items():
                if not tname.startswith("type_"):
                    continue
                type_id = int(tname.split("_", 1)[1])
                bundle.housekeeping[type_id] = {
                    k: ds[...] for k, ds in gtype.items()
                }

        # DCB telemetry
        if "DCB_telemetry" in f:
            g = f["DCB_telemetry"]
            for k, ds in g.items():
                arr = ds[...]
                if k.startswith("fpga_"):
                    bundle.dcb_fpga[k[len("fpga_"):]] = arr
                elif k.startswith("encoder_"):
                    bundle.dcb_encoder[k[len("encoder_"):]] = arr
                elif k in ("enc_pos", "enc_status"):
                    bundle.dcb_encoder[k] = arr

        # spectra_interpolated_telemetry
        if "spectra_interpolated_telemetry" in f:
            for k, ds in f["spectra_interpolated_telemetry"].items():
                bundle.interp_telemetry[k] = ds[...]

    _mark_legacy_auxiliary(bundle, path)
    return bundle


def _read_or_none(g, name: str):
    return g[name][...] if name in g else None


def _scalarize(v):
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, np.ndarray):
        return v
    return v


def _attr_text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("ascii")
    return str(value)


def _mark_legacy_auxiliary(bundle: SessionBundle, path: Path) -> None:
    if bundle.layout_version not in (2, 3):
        return
    families = []
    if bundle.zoom_spectra is not None:
        families.append("zoom_spectra")
    if bundle.waveforms:
        families.append("waveforms")
    if bundle.housekeeping:
        families.append("housekeeping")
    if bundle.grimm_spectra is not None:
        families.append("grimm_spectra")
    bundle.legacy_unverified_families = tuple(families)
    if families:
        warnings.warn(
            f"{path}: layout-v{bundle.layout_version} auxiliary products "
            f"{families} are legacy_unverified; re-ingest from source telemetry",
            LegacyIngestWarning,
            stacklevel=3,
        )


# ---------------------------------------------------------------------------
# FITS reader
# ---------------------------------------------------------------------------

def _load_fits(path: Path) -> SessionBundle:
    from astropy.io import fits

    layout_version = _require_supported_layout_version(
        fits.getheader(path, 0).get("LAYOUTV"),
        path,
    )
    if layout_version == INGEST_LAYOUT_VERSION:
        return _load_layout_v4_tree(read_layout_v4_fits(path), path)

    bundle = SessionBundle(source_path=path)
    with fits.open(path) as hdul:
        primary = hdul[0]
        if "LAYOUTV" in primary.header:
            bundle.layout_version = int(primary.header["LAYOUTV"])

        # SESSION_INV / CONSTANTS: header keywords
        if "SESSION_INV" in [h.name for h in hdul]:
            inv = hdul["SESSION_INV"].header
            for k in ("SW_VERS", "FW_VERS", "FW_ID", "FW_DATE", "FW_TIME",
                     "ST_UPID", "ST_T32", "ST_T16"):
                if k in inv:
                    bundle.session_invariants[_inv_key_to_h5(k)] = int(inv[k])
        if "CONSTANTS" in [h.name for h in hdul]:
            cst = hdul["CONSTANTS"].header
            for k_fits, k_h5 in (
                ("LUN_LAT", "lun_lat_deg"),
                ("LUN_LON", "lun_long_deg"),
                ("LUN_HGT", "lun_height_m"),
                ("RAWSHFT", "raw_time_subtract_seconds"),
                ("MJDOFF", "mjd_epoch_offset_days"),
            ):
                if k_fits in cst:
                    bundle.constants[k_h5] = float(cst[k_fits])
            for k_fits, k_h5 in (
                ("TIMESYS", "time_scale"),
                ("CLKSRC", "clock_source"),
                ("CLKEPOCH", "clock_epoch_isot"),
            ):
                if k_fits in cst:
                    v = str(cst[k_fits])
                    # TIMESYS values are uppercase by FITS convention;
                    # astropy scales are lowercase
                    bundle.constants[k_h5] = (
                        v.lower() if k_h5 == "time_scale" else v
                    )

        names = [h.name for h in hdul]

        if "SPECTRA" in names:
            spectra_hdu = hdul["SPECTRA"]
            bundle.spectra = np.asarray(spectra_hdu.data, dtype=np.float32)
        if "SPECTRA_TIMES" in names:
            t = hdul["SPECTRA_TIMES"].data
            bundle.spectra_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.spectra_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.spectra_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
            if "NFREQ" in t.dtype.names:
                bundle.spectra_frequency_counts = np.asarray(t["NFREQ"])
        if "SPECTRA_META" in names:
            t = hdul["SPECTRA_META"].data
            for n in t.dtype.names:
                bundle.spectra_metadata[n.lower()] = np.asarray(t[n])

        if bundle.spectra is not None:
            spectra_hdu = hdul["SPECTRA"]
            if (bundle.spectra.ndim != 3
                    or bundle.spectra.shape[1] != NPRODUCTS):
                raise ValueError(
                    f"{path}: normal spectra must have shape "
                    f"(N, {NPRODUCTS}, Nfreq); got {bundle.spectra.shape}"
                )
            actual_key, actual_bitslice = _metadata_get_ci(
                bundle.spectra_metadata, "actual_bitslice"
            )
            if actual_bitslice is None:
                raise ValueError(
                    f"{path}: normal spectra have no actual_bitslice metadata"
                )
            canonical_bs = canonical_actual_bitslice(
                actual_bitslice, bundle.spectra.shape[0]
            )
            if actual_key != "actual_bitslice":
                bundle.spectra_metadata.pop(actual_key, None)
            bundle.spectra_metadata["actual_bitslice"] = canonical_bs

            gain_key, actual_gain = _metadata_get_ci(
                bundle.spectra_metadata, "actual_gain"
            )
            if bundle.layout_version == 3 and actual_gain is None:
                raise ValueError(
                    f"{path}: layout-v3 normal spectra have no actual_gain metadata"
                )
            if actual_gain is not None:
                actual_gain = np.asarray(actual_gain)
                expected = bundle.spectra.shape[0] * _N_ADC_CHANNELS
                if actual_gain.size != expected:
                    raise ValueError(
                        f"{path}: actual_gain has shape {actual_gain.shape}; "
                        f"expected ({bundle.spectra.shape[0]}, {_N_ADC_CHANNELS})"
                    )
                if gain_key != "actual_gain":
                    bundle.spectra_metadata.pop(gain_key, None)
                bundle.spectra_metadata["actual_gain"] = actual_gain.reshape(
                    bundle.spectra.shape[0], _N_ADC_CHANNELS
                )

            if bundle.layout_version == 2:
                bundle.spectra = restore_bitsliced_spectra(
                    bundle.spectra, canonical_bs
                )
                warnings.warn(
                    f"{path}: layout-v2 FITS spectra were bit-slice restored "
                    "in memory; re-ingest to layout v3",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif bundle.layout_version == 3:
                if not bool(spectra_hdu.header.get("BITSREST", False)):
                    raise ValueError(
                        f"{path}: layout-v3 SPECTRA does not declare BITSREST"
                    )
                if int(spectra_hdu.header.get("BITSREF", -1)) != BITSLICE_REFERENCE:
                    raise ValueError(f"{path}: unsupported SPECTRA bit reference")
                if int(spectra_hdu.header.get("NORMVER", -1)) \
                        != SPECTRA_NORMALIZATION_VERSION:
                    raise ValueError(f"{path}: unsupported SPECTRA normalization version")
                if str(spectra_hdu.header.get("BUNIT", "")) != SPECTRA_UNITS:
                    raise ValueError(f"{path}: SPECTRA units must be {SPECTRA_UNITS!r}")
                if str(spectra_hdu.header.get("REPRESENTATION", "")) \
                        != SPECTRA_REPRESENTATION:
                    raise ValueError(
                        f"{path}: SPECTRA representation must be "
                        f"{SPECTRA_REPRESENTATION!r}"
                    )
            else:
                raise ValueError(
                    f"{path}: normal spectra require explicit layout_version 2 or 3"
                )
            bundle.spectra_units = SPECTRA_UNITS
            bundle.spectra_representation = SPECTRA_REPRESENTATION
            bundle.spectra_normalization_version = SPECTRA_NORMALIZATION_VERSION

        if "TR_SPECTRA" in names:
            bundle.tr_spectra = np.asarray(hdul["TR_SPECTRA"].data)
        if "TR_TIMES" in names:
            t = hdul["TR_TIMES"].data
            bundle.tr_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.tr_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.tr_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
            if "NAVG2" in t.dtype.names:
                bundle.tr_navg2_per_sample = np.asarray(t["NAVG2"])
            if "TR_LEN" in t.dtype.names:
                bundle.tr_length_per_sample = np.asarray(t["TR_LEN"])
        if "TR_META" in names:
            t = hdul["TR_META"].data
            for n in t.dtype.names:
                bundle.tr_metadata[n.lower()] = np.asarray(t[n])

        if "ZOOM_DATA" in names:
            bundle.zoom_spectra = np.asarray(hdul["ZOOM_DATA"].data, dtype=np.float32)
        if "ZOOM_TIMES" in names:
            t = hdul["ZOOM_TIMES"].data
            bundle.zoom_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.zoom_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)
            if "MJD_TIME" in t.dtype.names:
                bundle.zoom_mjd_times = np.asarray(t["MJD_TIME"], dtype=np.float64)
            if "PFB_IDX" in t.dtype.names:
                bundle.zoom_pfb_indices = np.asarray(t["PFB_IDX"])

        if "GRIMM" in names:
            bundle.grimm_spectra = np.asarray(hdul["GRIMM"].data, dtype=np.float32)
        if "GRIMM_TIMES" in names:
            t = hdul["GRIMM_TIMES"].data
            bundle.grimm_unique_ids = np.asarray(t["UNIQUE_ID"])
            bundle.grimm_raw_times = np.asarray(t["RAW_TIME"], dtype=np.float64)

        # Waveforms: WF_CH<N>
        for h in hdul:
            if h.name.startswith("WF_CH"):
                ch = int(h.name[len("WF_CH"):])
                t = h.data
                if t is None:
                    continue
                bundle.waveforms[ch] = np.asarray(t["WAVEFORM"], dtype=np.int16)
                bundle.waveform_times[ch] = np.asarray(t["TIMESTAMP"], dtype=np.float64)

        # Housekeeping HK_T<N>
        for h in hdul:
            if h.name.startswith("HK_T"):
                try:
                    type_id = int(h.name[len("HK_T"):])
                except ValueError:
                    continue
                t = h.data
                if t is None:
                    continue
                bundle.housekeeping[type_id] = {
                    n.lower(): np.asarray(t[n]) for n in t.dtype.names
                }

        if "DCB_FPGA" in names:
            t = hdul["DCB_FPGA"].data
            for n in t.dtype.names:
                if n == "MS":
                    bundle.dcb_fpga["mission_seconds"] = np.asarray(t[n], dtype=np.float64)
                elif n == "SUBSEC":
                    bundle.dcb_fpga["lusee_subsecs"] = np.asarray(t[n], dtype=np.float64)
                else:
                    bundle.dcb_fpga[n] = np.asarray(t[n])

        if "DCB_ENC" in names:
            t = hdul["DCB_ENC"].data
            mapping = {"MS": "mission_seconds", "SUBSEC": "lusee_subsecs",
                       "ENC_POS": "enc_pos", "ENC_STAT": "enc_status"}
            for n in t.dtype.names:
                bundle.dcb_encoder[mapping.get(n, n)] = np.asarray(t[n])

        if "SPEC_INTERP" in names:
            t = hdul["SPEC_INTERP"].data
            for n in t.dtype.names:
                bundle.interp_telemetry[n] = np.asarray(t[n])

    _mark_legacy_auxiliary(bundle, path)
    return bundle


def _inv_key_to_h5(fits_key: str) -> str:
    """Map a SESSION_INV FITS keyword back to the HDF5 attr name."""
    return {
        "SW_VERS":  "software_version",
        "FW_VERS":  "firmware_version",
        "FW_ID":    "firmware_id",
        "FW_DATE":  "firmware_date",
        "FW_TIME":  "firmware_time",
        "ST_UPID":  "start_unique_packet_id",
        "ST_T32":   "start_time_32",
        "ST_T16":   "start_time_16",
    }.get(fits_key, fits_key)


def _metadata_get_ci(metadata: Dict[str, np.ndarray], wanted: str):
    """Return ``(real_key, value)`` from a case-insensitive FITS mapping."""
    for key, value in metadata.items():
        if key.lower() == wanted.lower():
            return key, value
    return wanted, None


def _bundle_navgf(bundle: SessionBundle) -> np.ndarray:
    if bundle.spectra is None:
        return np.empty(0, dtype=np.uint8)
    values = bundle.spectra_navgf
    if values is None:
        _, values = _metadata_get_ci(bundle.spectra_metadata, "navgf")
    if values is None:
        if bundle.layout_version == INGEST_LAYOUT_VERSION:
            raise ValueError("layout-v4 normal spectra are missing Navgf")
        values = np.ones(bundle.spectra.shape[0], dtype=np.uint8)
    values = np.asarray(values)
    if (
        values.shape != (bundle.spectra.shape[0],)
        or values.dtype.kind not in ("i", "u")
        or values.dtype == np.bool_
        or np.any(~np.isin(values, (1, 2, 3, 4)))
    ):
        raise ValueError("normal-spectrum Navgf must be integer rows in {1,2,3,4}")
    return values.astype(np.uint8, copy=False)


def _bundle_persisted_rows_by_family(
    bundle: SessionBundle,
) -> Dict[str, int]:
    """Count persisted payload rows even when legacy records are unavailable."""
    counts = {
        family: len(bundle.product_records.get(family, ()))
        for family in sorted(ALL_FAMILIES)
    }
    for family, data in (
        ("spectra", bundle.spectra),
        ("tr_spectra", bundle.tr_spectra),
        ("zoom_spectra", bundle.zoom_spectra),
        ("grimm_spectra", bundle.grimm_spectra),
    ):
        if data is not None:
            counts[family] = int(data.shape[0])

    if bundle.waveform_data is not None:
        counts["waveforms"] = int(bundle.waveform_data.shape[0])
    elif bundle.waveforms:
        counts["waveforms"] = sum(
            int(np.asarray(data).shape[0])
            for data in bundle.waveforms.values()
        )

    if bundle.housekeeping_unique_ids is not None:
        counts["housekeeping"] = int(
            bundle.housekeeping_unique_ids.shape[0]
        )
    elif bundle.housekeeping:
        counts["housekeeping"] = sum(
            int(np.asarray(next(iter(fields.values()))).shape[0])
            for fields in bundle.housekeeping.values()
            if fields
        )

    for public_name, family in (
        ("metadata", "calibrator_metadata"),
        ("data", "calibrator_data"),
        ("raw_pfb", "calibrator_raw_pfb"),
        ("debug", "calibrator_debug"),
    ):
        result = bundle.calibrator.get(public_name)
        if isinstance(result, dict) and "unique_ids" in result:
            counts[family] = int(np.asarray(result["unique_ids"]).shape[0])
    return counts


def _normal_frequency_counts(bundle: SessionBundle) -> np.ndarray:
    """Resolve and validate each normal row before any source padding."""
    if bundle.spectra is None:
        return np.empty(0, dtype=np.int64)
    n_rows = bundle.spectra.shape[0]
    stored_width = bundle.spectra.shape[2]
    navgf_values = _bundle_navgf(bundle)
    if bundle.spectra_frequency_counts is None:
        if bundle.layout_version == INGEST_LAYOUT_VERSION:
            raise ValueError("layout-v4 normal spectra are missing frequency counts")
        if bundle.layout_version in (2, 3):
            counts = np.asarray(
                [
                    spectrometer_frequency_window(int(navgf)).output_count
                    for navgf in navgf_values
                ],
                dtype=np.int64,
            )
        else:
            counts = np.full(n_rows, stored_width, dtype=np.int64)
    else:
        counts = np.asarray(bundle.spectra_frequency_counts)
        if counts.dtype.kind not in ("i", "u") or counts.dtype == np.bool_:
            raise ValueError("normal-spectrum frequency counts must be integers")
        if counts.shape != (n_rows,):
            raise ValueError(
                "normal-spectrum frequency counts must have shape "
                f"({n_rows},); got {counts.shape}"
            )
    if np.any(counts <= 0):
        raise ValueError("normal-spectrum frequency counts must be positive")
    if np.any(counts > stored_width):
        raise ValueError(
            "normal-spectrum frequency count exceeds source backing width "
            f"{stored_width}"
        )
    if bundle.layout_version in (2, 3, INGEST_LAYOUT_VERSION):
        expected = np.asarray(
            [
                spectrometer_frequency_window(int(navgf)).output_count
                for navgf in navgf_values
            ],
            dtype=np.int64,
        )
        mismatch = np.flatnonzero(counts != expected)
        if mismatch.size:
            row = int(mismatch[0])
            raise ValueError(
                f"normal-spectrum frequency count {int(counts[row])} "
                f"disagrees with Navgf={int(navgf_values[row])} "
                f"({int(expected[row])})"
            )
    return counts


def _homogeneous_normal_frequency_count(
    bundle: SessionBundle,
) -> Optional[int]:
    """Validate one public normal-spectrum grid and return its native width."""
    if bundle.spectra is None:
        return None

    n_rows = bundle.spectra.shape[0]
    navgf_values = _bundle_navgf(bundle)
    counts = _normal_frequency_counts(bundle)
    unique_counts = np.unique(counts)
    if unique_counts.size != 1:
        raise MixedFrequencyGridError(
            "high-level IngestData requires one homogeneous normal-spectrum "
            f"frequency count; got {unique_counts.tolist()}"
        )
    frequency_count = int(unique_counts[0])

    unique_navgf = np.unique(navgf_values)
    if unique_navgf.size != 1:
        raise MixedFrequencyGridError(
            "high-level IngestData requires one homogeneous normal-spectrum "
            f"Navgf; got {unique_navgf.tolist()}"
        )
    navgf = int(unique_navgf[0])
    expected_count = spectrometer_frequency_window(navgf).output_count
    if frequency_count != expected_count:
        raise ValueError(
            f"normal-spectrum frequency count {frequency_count} disagrees "
            f"with Navgf={navgf} ({expected_count})"
        )
    if bundle.spectra_frequency_window_index is not None:
        indices = np.asarray(bundle.spectra_frequency_window_index)
        if indices.shape != (n_rows,) or indices.dtype.kind not in ("i", "u"):
            raise ValueError("normal-spectrum frequency-window indices are invalid")
        if np.unique(indices).size != 1:
            raise MixedFrequencyGridError(
                "high-level IngestData requires one homogeneous frequency window"
            )
        window = bundle.frequency_window_for_row(0)
        if window.navgf != navgf:
            raise ValueError("frequency-window index disagrees with Navgf")

    return frequency_count


# ---------------------------------------------------------------------------
# Bundle loading dispatch
# ---------------------------------------------------------------------------

def _load_one(path: Path) -> SessionBundle:
    if _is_h5(path):
        bundle = _load_h5(path)
    elif _is_fits(path):
        bundle = _load_fits(path)
    else:
        raise ValueError(f"unrecognized file extension: {path}")
    if bundle.spectra is not None:
        bundle.spectra_navgf = _bundle_navgf(bundle)
    if not bundle.source_paths:
        bundle.source_paths = (path,)
    return bundle


def _bundle_sort_key(bundle: SessionBundle) -> tuple[float, int, str]:
    source = str(
        bundle.source_path
        or next(iter(bundle.source_paths), "")
    )
    session_start = bundle.session_invariants.get("start_raw_seconds")
    if session_start is not None:
        try:
            session_start = float(session_start)
        except (TypeError, ValueError):
            session_start = None
        if session_start is not None and np.isfinite(session_start):
            return session_start, 0, source
    for values in (
        bundle.spectra_raw_times,
        bundle.tr_raw_times,
        bundle.zoom_raw_times,
        bundle.grimm_raw_times,
        bundle.waveform_raw_times,
        bundle.housekeeping_raw_times,
    ):
        if values is None:
            continue
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return float(finite[0]), 1, source
    return float("inf"), 2, source


# ---------------------------------------------------------------------------
# Concatenation helpers
# ---------------------------------------------------------------------------

def _concat_optional(arrs: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [a for a in arrs if a is not None and a.size > 0]
    if not valid:
        return None
    return np.concatenate(valid, axis=0)


def _concat_row_aligned(
    arrays: Sequence[Optional[np.ndarray]],
    *,
    row_counts: Sequence[int],
    fill_value: object = np.nan,
    dtype: object | None = None,
) -> Optional[np.ndarray]:
    """Concatenate optional row arrays without silently dropping rows."""
    if len(arrays) != len(row_counts):
        raise ValueError("row-aligned arrays and row counts must have equal length")
    prototype = next((np.asarray(value) for value in arrays if value is not None), None)
    if prototype is None:
        return None
    tail_shape = prototype.shape[1:]
    if dtype is None:
        if all(value is not None for value in arrays):
            dtype = prototype.dtype
        else:
            dtype = (
                prototype.dtype
                if prototype.dtype.kind in "fc" or not np.isscalar(fill_value)
                else np.asarray(fill_value).dtype
            )
            if isinstance(fill_value, float) and np.isnan(fill_value):
                dtype = (
                    prototype.dtype
                    if prototype.dtype.kind in "fc"
                    else np.float64
                )
    chunks = []
    for value, count in zip(arrays, row_counts):
        if value is None:
            chunks.append(
                np.full((count, *tail_shape), fill_value, dtype=dtype)
            )
            continue
        array = np.asarray(value)
        if array.shape != (count, *tail_shape):
            raise ValueError(
                f"row-aligned array has shape {array.shape}; expected "
                f"{(count, *tail_shape)}"
            )
        chunks.append(array.astype(dtype, copy=False))
    return np.concatenate(chunks, axis=0)


def _concat_calibrator_results(
    results: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    """Concatenate one calibrator subproduct while preserving its public shape."""
    if not results:
        return {}
    keys = set(results[0])
    if any(set(result) != keys for result in results[1:]):
        raise ValueError("calibrator session fields disagree")
    merged: Dict[str, object] = {}
    for key in sorted(keys):
        values = [result[key] for result in results]
        if all(isinstance(value, np.ndarray) for value in values):
            merged[key] = np.concatenate(values, axis=0)
        elif key in ("fields", "field_present", "records"):
            merged[key] = tuple(item for value in values for item in value)
        elif key == "pages":
            page_count = len(values[0])
            if any(len(value) != page_count for value in values[1:]):
                raise ValueError("calibrator debug page counts disagree")
            merged[key] = tuple(
                _concat_calibrator_results(
                    [value[page_index] for value in values]
                )
                for page_index in range(page_count)
            )
        elif all(value == values[0] for value in values[1:]):
            merged[key] = values[0]
        else:
            raise ValueError(f"calibrator session field {key!r} disagrees")
    return merged


def _pad_to(arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Pad ``arr`` along axes 1..ndim to match ``target_shape`` with NaN."""
    if arr.shape[1:] == target_shape:
        return arr
    pad_widths = [(0, 0)] + [
        (0, t - s) for s, t in zip(arr.shape[1:], target_shape)
    ]
    return np.pad(arr, pad_widths, mode="constant", constant_values=np.nan)


def _concat_dict_arrays(
    dicts: Sequence[Dict[str, np.ndarray]],
    *,
    n_per_source: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Concatenate per-field dicts. Missing fields in a source are NaN-filled."""
    keys = sorted({k for d in dicts for k in d})
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        prototype = next(d[k] for d in dicts if k in d)
        tail = prototype.shape[1:]
        fill_dtype = (
            prototype.dtype
            if prototype.dtype.kind in {"f", "c"}
            else np.float64
        )
        chunks = []
        for d, n in zip(dicts, n_per_source):
            if k in d:
                chunks.append(d[k])
            else:
                fill = np.full((n,) + tail, np.nan, dtype=fill_dtype)
                chunks.append(fill)
        if chunks:
            out[k] = np.concatenate(chunks, axis=0)
    return out


def _concat_presence_dicts(
    dicts: Sequence[Dict[str, np.ndarray]],
    *,
    n_per_source: Sequence[int],
) -> Dict[str, np.ndarray]:
    keys = sorted({key for values in dicts for key in values})
    result = {}
    for key in keys:
        chunks = []
        for values, count in zip(dicts, n_per_source):
            if key in values:
                chunk = np.asarray(values[key], dtype=np.bool_)
                if chunk.shape != (count,):
                    raise ValueError(
                        f"presence field {key!r} has shape {chunk.shape}; "
                        f"expected ({count},)"
                    )
                chunks.append(chunk)
            else:
                chunks.append(np.zeros(count, dtype=np.bool_))
        result[key] = np.concatenate(chunks)
    return result


def _merge_housekeeping(
    bundles: Sequence[SessionBundle],
) -> Dict[int, Dict[str, np.ndarray]]:
    out: Dict[int, Dict[str, np.ndarray]] = {}
    type_ids = sorted({tid for b in bundles for tid in b.housekeeping})
    for tid in type_ids:
        sub_dicts = [b.housekeeping.get(tid, {}) for b in bundles]
        ns = [
            (next(iter(d.values())).shape[0] if d else 0)
            for d in sub_dicts
        ]
        out[tid] = _concat_dict_arrays(sub_dicts, n_per_source=ns)
    return out


def _concat_bundles(bundles: Sequence[SessionBundle]) -> SessionBundle:
    """Combine already-sorted SessionBundles after contract validation."""
    if len(bundles) == 1:
        return bundles[0]

    layout_versions = {bundle.layout_version for bundle in bundles}
    if len(layout_versions) != 1:
        raise ValueError(
            f"cannot concatenate ingest layout versions {sorted(layout_versions)}"
        )

    out = SessionBundle()
    out.source_path = None  # multi-source
    out.layout_version = next(iter(layout_versions))
    out.source_paths = tuple(
        path
        for bundle in bundles
        for path in (
            bundle.source_paths
            or (() if bundle.source_path is None else (bundle.source_path,))
        )
    )
    session_spectra_counts = []
    session_sources = []
    for bundle in bundles:
        if bundle.session_spectra_counts:
            counts = bundle.session_spectra_counts
            sources = bundle.session_sources
            if not sources:
                candidate_sources = bundle.source_paths
                if len(candidate_sources) == len(counts):
                    sources = tuple(candidate_sources)
                else:
                    sources = (bundle.source_path,) * len(counts)
        else:
            counts = (
                bundle.spectra.shape[0] if bundle.spectra is not None else 0,
            )
            sources = (bundle.source_path,)
        if len(sources) != len(counts):
            raise ValueError("session boundary provenance is inconsistent")
        session_spectra_counts.extend(int(count) for count in counts)
        session_sources.extend(sources)
    out.session_spectra_counts = tuple(session_spectra_counts)
    out.session_sources = tuple(session_sources)

    # Spectra
    units = {b.spectra_units for b in bundles if b.spectra is not None}
    representations = {
        b.spectra_representation for b in bundles if b.spectra is not None
    }
    if len(units) > 1 or len(representations) > 1:
        raise ValueError(
            "cannot concatenate normal spectra with different normalization "
            f"contracts: units={units}, representations={representations}"
        )
    out.spectra_units = next(iter(units), None)
    out.spectra_representation = next(iter(representations), None)
    normalization_versions = {
        b.spectra_normalization_version
        for b in bundles
        if b.spectra is not None
    }
    if len(normalization_versions) > 1:
        raise ValueError(
            "cannot concatenate different spectra normalization versions"
        )
    out.spectra_normalization_version = next(
        iter(normalization_versions),
        None,
    )
    spectra_bundles = [b for b in bundles if b.spectra is not None]
    frequency_count_chunks = [
        _normal_frequency_counts(bundle) for bundle in spectra_bundles
    ]
    if spectra_bundles:
        max_width = max(b.spectra.shape[2] for b in spectra_bundles)
        out.spectra = np.concatenate(
            [
                _pad_to(b.spectra, (NPRODUCTS, max_width))
                for b in spectra_bundles
            ],
            axis=0,
        )
    spectra_counts = [b.spectra.shape[0] for b in spectra_bundles]
    out.spectra_unique_ids = _concat_row_aligned(
        [b.spectra_unique_ids for b in spectra_bundles],
        row_counts=spectra_counts,
    )
    out.spectra_raw_times = _concat_row_aligned(
        [b.spectra_raw_times for b in spectra_bundles],
        row_counts=spectra_counts,
    )
    out.spectra_raw_time_valid = _concat_row_aligned(
        [
            b.spectra_raw_time_valid
            if b.spectra_raw_time_valid is not None
            else (
                np.isfinite(b.spectra_raw_times)
                if b.spectra_raw_times is not None
                else None
            )
            for b in spectra_bundles
        ],
        row_counts=spectra_counts,
        fill_value=False,
        dtype=np.bool_,
    )
    out.spectra_frequency_counts = _concat_optional(frequency_count_chunks)
    out.spectra_navgf = _concat_row_aligned(
        [_bundle_navgf(b) for b in spectra_bundles],
        row_counts=spectra_counts,
    )
    out.spectra_mjd_times = _concat_row_aligned(
        [b.spectra_mjd_times for b in spectra_bundles],
        row_counts=spectra_counts,
    )
    out.spectra_mjd_time_valid = _concat_row_aligned(
        [b.spectra_mjd_time_valid for b in spectra_bundles],
        row_counts=spectra_counts,
        fill_value=False,
        dtype=np.bool_,
    )

    if out.layout_version == INGEST_LAYOUT_VERSION and spectra_bundles:
        contracts = {
            contract.navgf: contract
            for bundle in spectra_bundles
            for contract in bundle.spectra_frequency_windows
        }
        out.spectra_frequency_windows = tuple(
            contracts[navgf] for navgf in sorted(contracts)
        )
        contract_indices = {
            contract.navgf: index
            for index, contract in enumerate(out.spectra_frequency_windows)
        }
        out.spectra_frequency_window_index = np.asarray(
            [contract_indices[int(value)] for value in out.spectra_navgf],
            dtype=np.uint8,
        )

    n_per = [
        b.spectra.shape[0] if b.spectra is not None else 0 for b in bundles
    ]
    out.spectra_metadata = _concat_dict_arrays(
        [b.spectra_metadata for b in bundles], n_per_source=n_per
    )
    out.spectra_metadata_present = _concat_presence_dicts(
        [b.spectra_metadata_present for b in bundles],
        n_per_source=n_per,
    )

    # TR spectra: each high-level bundle retains one exact session geometry
    if any(b.tr_spectra is not None for b in bundles):
        present = [b for b in bundles if b.tr_spectra is not None]
        geometries = {b.tr_spectra.shape[1:] for b in present}
        if len(geometries) != 1:
            raise ValueError(
                f"cannot concatenate TR session geometries {sorted(geometries)}"
            )
        out.tr_spectra = np.concatenate(
            [b.tr_spectra for b in present],
            axis=0,
        )
        n_tr = [b.tr_spectra.shape[0] for b in present]
        for name in (
            "tr_unique_ids",
            "tr_raw_times",
            "tr_mjd_times",
            "tr_navg2_per_sample",
            "tr_length_per_sample",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in present],
                    row_counts=n_tr,
                ),
            )
        out.tr_raw_time_valid = _concat_row_aligned(
            [
                b.tr_raw_time_valid
                if b.tr_raw_time_valid is not None
                else (
                    np.isfinite(b.tr_raw_times)
                    if b.tr_raw_times is not None
                    else None
                )
                for b in present
            ],
            row_counts=n_tr,
            fill_value=False,
            dtype=np.bool_,
        )
        out.tr_mjd_time_valid = _concat_row_aligned(
            [b.tr_mjd_time_valid for b in present],
            row_counts=n_tr,
            fill_value=False,
            dtype=np.bool_,
        )
        out.tr_metadata = _concat_dict_arrays(
            [b.tr_metadata for b in present], n_per_source=n_tr
        )

    # Zoom
    zoom_bundles = [b for b in bundles if b.zoom_spectra is not None]
    if zoom_bundles:
        zoom_counts = [b.zoom_spectra.shape[0] for b in zoom_bundles]
        out.zoom_spectra = np.concatenate(
            [b.zoom_spectra for b in zoom_bundles], axis=0
        )
        for name in (
            "zoom_unique_ids",
            "zoom_pfb_indices",
            "zoom_pfb_bins",
            "zoom_raw_times",
            "zoom_mjd_times",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in zoom_bundles],
                    row_counts=zoom_counts,
                ),
            )
        raw_valid_chunks = [
            b.zoom_raw_time_valid
            if b.zoom_raw_time_valid is not None
            else (
                np.isfinite(b.zoom_raw_times)
                if b.zoom_raw_times is not None
                else None
            )
            for b in zoom_bundles
        ]
        out.zoom_raw_time_valid = _concat_row_aligned(
            raw_valid_chunks,
            row_counts=zoom_counts,
            fill_value=False,
            dtype=np.bool_,
        )
        out.zoom_mjd_time_valid = _concat_row_aligned(
            [b.zoom_mjd_time_valid for b in zoom_bundles],
            row_counts=zoom_counts,
            fill_value=False,
            dtype=np.bool_,
        )

    # Grimm: preserve each native Navg2 and pad only the merged storage view
    grimm_bundles = [b for b in bundles if b.grimm_spectra is not None]
    if grimm_bundles:
        max_navg2 = max(b.grimm_spectra.shape[1] for b in grimm_bundles)
        grimm_dtype = np.result_type(
            *(b.grimm_spectra.dtype for b in grimm_bundles)
        )
        data_chunks = []
        valid_chunks = []
        for b in grimm_bundles:
            width = b.grimm_spectra.shape[1]
            source_valid = (
                b.grimm_average_valid
                if b.grimm_average_valid is not None
                else np.ones(
                    (b.grimm_spectra.shape[0], width), dtype=np.bool_
                )
            )
            if width == max_navg2:
                data_chunks.append(
                    b.grimm_spectra.astype(grimm_dtype, copy=False)
                )
                valid_chunks.append(source_valid)
                continue
            data = np.zeros(
                (b.grimm_spectra.shape[0], max_navg2, NPRODUCTS, 4),
                dtype=grimm_dtype,
            )
            valid = np.zeros(
                (b.grimm_spectra.shape[0], max_navg2), dtype=np.bool_
            )
            data[:, :width] = b.grimm_spectra
            valid[:, :width] = source_valid
            data_chunks.append(data)
            valid_chunks.append(valid)
        out.grimm_spectra = np.concatenate(data_chunks, axis=0)
        out.grimm_average_valid = np.concatenate(valid_chunks, axis=0)
        grimm_counts = [b.grimm_spectra.shape[0] for b in grimm_bundles]
        for name in (
            "grimm_unique_ids",
            "grimm_raw_times",
            "grimm_mjd_times",
            "grimm_navg2_per_sample",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in grimm_bundles],
                    row_counts=grimm_counts,
                ),
            )
        out.grimm_raw_time_valid = _concat_row_aligned(
            [
                b.grimm_raw_time_valid
                if b.grimm_raw_time_valid is not None
                else (
                    np.isfinite(b.grimm_raw_times)
                    if b.grimm_raw_times is not None
                    else None
                )
                for b in grimm_bundles
            ],
            row_counts=grimm_counts,
            fill_value=False,
            dtype=np.bool_,
        )
        out.grimm_mjd_time_valid = _concat_row_aligned(
            [b.grimm_mjd_time_valid for b in grimm_bundles],
            row_counts=grimm_counts,
            fill_value=False,
            dtype=np.bool_,
        )

    # Row-aligned v4 waveform view (legacy channel dictionaries remain below)
    waveform_bundles = [b for b in bundles if b.waveform_data is not None]
    if waveform_bundles:
        waveform_counts = [b.waveform_data.shape[0] for b in waveform_bundles]
        for name in (
            "waveform_data",
            "waveform_channels",
            "waveform_unique_ids",
            "waveform_raw_times",
            "waveform_mjd_times",
            "waveform_adc_timestamps",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in waveform_bundles],
                    row_counts=waveform_counts,
                ),
            )
        for name in (
            "waveform_raw_time_valid",
            "waveform_mjd_time_valid",
            "waveform_adc_timestamp_valid",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in waveform_bundles],
                    row_counts=waveform_counts,
                    fill_value=False,
                    dtype=np.bool_,
                ),
            )

    # Waveforms by channel
    chans = sorted({c for b in bundles for c in b.waveforms})
    for c in chans:
        channel_bundles = [b for b in bundles if c in b.waveforms]
        chunks = [b.waveforms[c] for b in channel_bundles]
        counts = [chunk.shape[0] for chunk in chunks]
        out.waveforms[c] = np.concatenate(chunks, axis=0)
        times = _concat_row_aligned(
            [b.waveform_times.get(c) for b in channel_bundles],
            row_counts=counts,
        )
        if times is not None:
            out.waveform_times[c] = times

    # Housekeeping
    out.housekeeping = _merge_housekeeping(bundles)
    housekeeping_bundles = [
        b for b in bundles if b.housekeeping_unique_ids is not None
    ]
    if housekeeping_bundles:
        housekeeping_counts = [
            len(b.housekeeping_unique_ids) for b in housekeeping_bundles
        ]
        for name in (
            "housekeeping_unique_ids",
            "housekeeping_raw_times",
            "housekeeping_mjd_times",
            "housekeeping_types",
            "housekeeping_versions",
            "housekeeping_firmware_errors",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in housekeeping_bundles],
                    row_counts=housekeeping_counts,
                ),
            )
        for name in (
            "housekeeping_raw_time_valid",
            "housekeeping_mjd_time_valid",
        ):
            setattr(
                out,
                name,
                _concat_row_aligned(
                    [getattr(b, name) for b in housekeeping_bundles],
                    row_counts=housekeeping_counts,
                    fill_value=False,
                    dtype=np.bool_,
                ),
            )
        out.housekeeping_fields = tuple(
            row for b in housekeeping_bundles for row in b.housekeeping_fields
        )
        out.housekeeping_field_present = tuple(
            row
            for b in housekeeping_bundles
            for row in b.housekeeping_field_present
        )

    calibrator_names = sorted(
        {name for bundle in bundles for name in bundle.calibrator}
    )
    out.calibrator = {
        name: _concat_calibrator_results(
            [
                bundle.calibrator[name]
                for bundle in bundles
                if name in bundle.calibrator
            ]
        )
        for name in calibrator_names
    }
    record_families = sorted(
        {family for bundle in bundles for family in bundle.product_records}
    )
    out.product_records = {
        family: tuple(
            record
            for bundle in bundles
            for record in bundle.product_records.get(family, ())
        )
        for family in record_families
    }
    out.decoder_provenance = {
        "sessions": tuple(bundle.decoder_provenance for bundle in bundles)
    }
    out.run_provenance = {
        "sessions": tuple(bundle.run_provenance for bundle in bundles)
    }
    out.issues = {
        "sessions": np.asarray(
            [bundle.issues for bundle in bundles],
            dtype=object,
        )
    }
    out.product_provenance = {
        "sessions": tuple(bundle.product_provenance for bundle in bundles),
        "records_by_family": out.product_records,
    }
    out.family_status = {
        "sessions": tuple(bundle.family_status for bundle in bundles)
    }

    # DCB telemetry
    n_fpga = [
        (b.dcb_fpga["mission_seconds"].size
         if "mission_seconds" in b.dcb_fpga else 0)
        for b in bundles
    ]
    out.dcb_fpga = _concat_dict_arrays(
        [b.dcb_fpga for b in bundles], n_per_source=n_fpga,
    )
    n_enc = [
        (b.dcb_encoder["mission_seconds"].size
         if "mission_seconds" in b.dcb_encoder else 0)
        for b in bundles
    ]
    out.dcb_encoder = _concat_dict_arrays(
        [b.dcb_encoder for b in bundles], n_per_source=n_enc,
    )

    out.interp_telemetry = _concat_dict_arrays(
        [b.interp_telemetry for b in bundles], n_per_source=n_per,
    )

    out.constants = dict(bundles[0].constants)
    if any(bundle.constants != out.constants for bundle in bundles[1:]):
        raise ValueError("input files have incompatible location/time constants")
    invariant_contract_names = (
        "software_version",
        "firmware_version",
        "firmware_id",
        "firmware_date",
        "firmware_time",
    )
    legacy_names = {
        "software_version": "sw_version",
        "firmware_version": "fw_version",
        "firmware_id": "fw_id",
        "firmware_date": "fw_date",
        "firmware_time": "fw_time",
    }

    def invariant_value(bundle: SessionBundle, name: str):
        return bundle.session_invariants.get(
            name, bundle.session_invariants.get(legacy_names[name])
        )

    invariant_contract = {
        name: invariant_value(bundles[0], name)
        for name in invariant_contract_names
    }
    for bundle in bundles[1:]:
        observed = {
            name: invariant_value(bundle, name)
            for name in invariant_contract_names
        }
        if observed != invariant_contract:
            raise ValueError("input files have incompatible firmware invariants")
    out.session_invariants = {
        **invariant_contract,
        "sessions": tuple(bundle.session_invariants for bundle in bundles),
    }
    out.clock_reference_set = bundles[0].clock_reference_set
    if any(
        bundle.clock_reference_set != out.clock_reference_set
        for bundle in bundles[1:]
    ):
        raise ValueError("input files have incompatible clock-reference sets")
    unavailable_reasons = tuple(
        reason
        for bundle in bundles
        for reason in (
            bundle.clock_reference_unavailable_reasons
            or (bundle.clock_reference_unavailable_reason,)
        )
    )
    out.clock_reference_unavailable_reasons = unavailable_reasons
    distinct_reasons = {
        reason for reason in unavailable_reasons if reason is not None
    }
    if len(distinct_reasons) == 1:
        out.clock_reference_unavailable_reason = next(iter(distinct_reasons))
    elif distinct_reasons:
        out.clock_reference_unavailable_reason = (
            "clock reference unavailable for multiple input sessions; see "
            "clock_reference_unavailable_reasons"
        )
    if any(bundle.quality_status == "partial" for bundle in bundles):
        out.quality_status = "partial"
    elif all(bundle.quality_status == "clean" for bundle in bundles):
        out.quality_status = "clean"
    else:
        out.quality_status = None
    execution_modes = {bundle.execution_mode for bundle in bundles}
    if len(execution_modes) != 1:
        raise ValueError("input files have incompatible execution modes")
    out.execution_mode = next(iter(execution_modes))
    out.legacy_unverified_families = tuple(
        sorted(
            {
                family
                for bundle in bundles
                for family in bundle.legacy_unverified_families
            }
        )
    )
    return out


# ---------------------------------------------------------------------------
# IngestData
# ---------------------------------------------------------------------------

# (i, j) -> (real_index, imag_index_or_None) inside the (16,) product axis.
_PRODUCT_INDEX = {
    (0, 0): (0, None),
    (1, 1): (1, None),
    (2, 2): (2, None),
    (3, 3): (3, None),
    (0, 1): (4, 5),
    (0, 2): (6, 7),
    (0, 3): (8, 9),
    (1, 2): (10, 11),
    (1, 3): (12, 13),
    (2, 3): (14, 15),
}


def _resolve_combination(comb) -> Tuple[int, int, str, float, bool]:
    """Translate a 'NNX' string or (i, j[, x]) tuple to a normalized record.

    Returns ``(i, j, mode, sign, conjugate)`` with ``i <= j`` (symmetric
    access ``i > j`` is mapped to its mirror plus sign/conjugate
    adjustments: real channels are unchanged, imaginary channels are
    negated, complex channels are conjugated).
    """
    sign = 1.0
    if isinstance(comb, str):
        s = comb
        if s.startswith("-"):
            sign = -1.0
            s = s[1:]
        if len(s) < 2:
            raise ValueError(f"combination must be at least 'NN': {comb!r}")
        i = int(s[0]); j = int(s[1])
        mode = s[2:] if len(s) >= 3 else ("R" if i == j else "C")
    else:
        if len(comb) == 2:
            i, j = comb
            mode = "R" if i == j else "C"
        elif len(comb) == 3:
            i, j, mode = comb
        else:
            raise ValueError(f"combination tuple must have 2 or 3 elements: {comb!r}")
        i, j = int(i), int(j)
    conjugate = False
    if i > j:
        i, j = j, i
        if mode.startswith("I"):
            sign = -sign
        elif mode.startswith("C"):
            conjugate = True
    return i, j, mode, sign, conjugate


class IngestData(Observation):
    """An :class:`Observation` populated from one or more lusee.ingest files.

    Indexing (mirrors :class:`lusee.Data`)::

        data[time_idx, "00",  freq_idx]   # autocorrelation 0 (real)
        data[time_idx, "01R", freq_idx]   # real part of cross 0x1
        data[time_idx, "01I", freq_idx]   # imag part of cross 0x1
        data[time_idx, "01C", freq_idx]   # complex (R + 1j*I)
        data[time_idx, "-01R", freq_idx]  # negated real

    Plotting helpers: :meth:`plot_waterfall`, :meth:`plot_mean_spectrum`,
    :meth:`plot_dcb`, :meth:`plot_adc_stats`.

    The list of underlying per-file bundles is exposed at
    ``data.bundles`` for power users that need lower-level access.

    Absolute time requires a complete stored or caller clock-reference
    record. ``assume_scale=`` may resolve only the scale of a complete legacy
    subtract-plus-MJD mapping; it never supplies a missing epoch or raw-clock
    anchor. ``self.time_provenance`` records the resolved scale, clock source,
    and assumption state.
    """

    def __init__(
        self,
        paths,
        *,
        prefer_format: str = "h5",
        time_source: str = "spectra",
        mission_epoch=None,
        assume_scale=None,
        clock_reference_set: ClockReferenceSet | None = None,
    ):
        if clock_reference_set is not None and not isinstance(
            clock_reference_set, ClockReferenceSet
        ):
            raise TypeError(
                "caller clock_reference_set must be a production "
                "ClockReferenceSet"
            )
        if isinstance(paths, SessionBundle):
            self.bundles = [paths]
        else:
            files = _resolve_paths(paths, prefer_format=prefer_format)
            log.info("loading %d file(s): %s", len(files),
                     [str(p) for p in files])
            self.bundles = [_load_one(path) for path in files]
        self.bundles.sort(key=_bundle_sort_key)
        bundle = _concat_bundles(self.bundles)
        normal_navgf = _bundle_navgf(bundle)
        normal_frequency_count = _homogeneous_normal_frequency_count(bundle)

        time_axis_mjd, time_axis_raw, source_used = _pick_time_axis(
            bundle, preferred=time_source,
        )
        if time_axis_mjd is None and time_axis_raw is None:
            raise ValueError("no time axis found in any input file")
        if bundle.spectra is not None:
            axis = time_axis_raw if time_axis_raw is not None else time_axis_mjd
            if axis is None or axis.shape != (bundle.spectra.shape[0],):
                raise ValueError(
                    f"time source {source_used!r} is not row-aligned with "
                    "normal spectra"
                )
            if source_used != "spectra":
                normal_axis = (
                    bundle.spectra_raw_times
                    if time_axis_raw is not None
                    else bundle.spectra_mjd_times
                )
                if (
                    normal_axis is None
                    or normal_axis.shape != axis.shape
                    or not np.allclose(
                        axis,
                        normal_axis,
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    )
                ):
                    raise ValueError(
                        f"time source {source_used!r} does not identify the "
                        "normal-spectrum rows used by IngestData indexing"
                    )
        mjd_valid, raw_valid = _time_validity(bundle, source_used)
        if assume_scale is not None:
            assume_scale = str(assume_scale).lower()
            if assume_scale == DEFAULT_TIME_SCALE:
                raise ValueError(
                    "assume_scale='unknown' is not a usable time scale"
                )
        scale_assumed = False
        reference_assumed = False
        clock_source = ClockSource.SPECTROMETER.value
        resolved_reference_set = None
        if bundle.layout_version == INGEST_LAYOUT_VERSION:
            if mission_epoch is not None:
                raise ValueError(
                    "mission_epoch is not a layout-v4 clock reference; pass "
                    "clock_reference_set instead"
                )
            stored_reference = bundle.clock_reference_set
            if stored_reference is not None and not isinstance(
                stored_reference, ClockReferenceSet
            ):
                raise TypeError(
                    "layout-v4 stored clock reference must be a production "
                    "ClockReferenceSet"
                )
            if (
                stored_reference is not None
                and clock_reference_set is not None
                and stored_reference != clock_reference_set
            ):
                raise ValueError(
                    "caller clock_reference_set contradicts the stored set"
                )
            reference_set = stored_reference or clock_reference_set
            resolved_reference_set = reference_set
            reference_assumed = bool(
                reference_set is not None and reference_set.assumed
            )
            if assume_scale is not None and reference_set is not None and (
                assume_scale != reference_set.time_scale
            ):
                raise ValueError(
                    f"assume_scale={assume_scale!r} contradicts the "
                    f"clock-reference scale {reference_set.time_scale!r}"
                )
            if (
                time_axis_mjd is not None
                and mjd_valid is not None
                and np.all(mjd_valid)
                and reference_set is not None
            ):
                if not np.all(np.isfinite(time_axis_mjd)):
                    raise ValueError("valid layout-v4 MJD rows must be finite")
                times_obj = LunarTime(
                    time_axis_mjd,
                    format="mjd",
                    scale=reference_set.time_scale,
                )
            else:
                if reference_set is None:
                    raise ClockReferenceUnavailableError(
                        "layout-v4 absolute time requires a stored or caller "
                        "ClockReferenceSet; assume_scale alone is insufficient"
                    )
                if (
                    time_axis_raw is None
                    or raw_valid is None
                    or not np.all(raw_valid)
                    or not np.all(np.isfinite(time_axis_raw))
                ):
                    raise ClockReferenceUnavailableError(
                        "layout-v4 high-level time requires valid raw time for "
                        "every selected row"
                    )
                times_obj = LunarTime(
                    reference_set.to_time(
                        time_axis_raw,
                        clock_source=clock_source,
                    )
                )
        else:
            if mission_epoch is not None:
                raise ValueError(
                    "mission_epoch implies an unverified zero raw-clock anchor; "
                    "pass clock_reference_set instead"
                )
            migrated_reference = _legacy_clock_reference_set(
                bundle, assume_scale
            )
            if (
                migrated_reference is not None
                and clock_reference_set is not None
                and not _legacy_reference_matches_production(
                    migrated_reference,
                    clock_reference_set,
                )
            ):
                raise ValueError(
                    "caller clock_reference_set contradicts the migrated "
                    "layout-v2/v3 clock constants"
                )
            resolved_reference_set = clock_reference_set or migrated_reference
            if resolved_reference_set is None:
                raise ClockReferenceUnavailableError(
                    "legacy raw time has no complete clock reference; "
                    "assume_scale alone cannot authorize absolute time"
                )
            if assume_scale is not None and (
                assume_scale != resolved_reference_set.time_scale
            ):
                raise ValueError(
                    f"assume_scale={assume_scale!r} contradicts the "
                    f"clock-reference scale {resolved_reference_set.time_scale!r}"
                )
            if (
                time_axis_raw is None
                or raw_valid is not None and not np.all(raw_valid)
                or not np.all(np.isfinite(time_axis_raw))
            ):
                raise ClockReferenceUnavailableError(
                    "legacy high-level time requires valid spectrometer raw "
                    "time for every selected row"
                )
            times_obj = LunarTime(
                resolved_reference_set.to_time(
                    time_axis_raw,
                    clock_source=ClockSource.SPECTROMETER,
                )
            )
            if (
                time_axis_mjd is not None
                and _mjd_is_calibrated(bundle)
                and not np.allclose(
                    time_axis_mjd,
                    times_obj.mjd,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise ValueError(
                    "legacy calibrated MJD contradicts the clock constants"
                )
            scale_assumed = resolved_reference_set.assumed

        # Compute median cadence from raw_times for Observation.deltaT.
        if time_axis_raw is not None and time_axis_raw.size > 1:
            dt_med = float(np.nanmedian(np.diff(time_axis_raw)))
            if not np.isfinite(dt_med) or dt_med <= 0:
                dt_med = 1.0
        else:
            dt_med = 1.0

        # Constants -> Observation.__init__
        c = bundle.constants
        # Observation does np.arange(t0, t1, deltaT).astype(Time) internally,
        # which leaks an astropy "missing unit" warning out of the parent's
        # TimeDelta arithmetic. Silence it: we're going to overwrite
        # self.times immediately afterwards anyway.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            super().__init__(
                time_range=(times_obj[0].isot, times_obj[-1].isot),
                lun_lat_deg=c.get("lun_lat_deg", self.default_lun_lat_deg),
                lun_long_deg=c.get("lun_long_deg", self.default_lun_long_deg),
                lun_height_m=c.get("lun_height_m", self.default_lun_height_m),
                deltaT_sec=dt_med,
            )
        # Override the synthetic uniform grid with the actual sample times.
        self.times = times_obj
        self.time_source = source_used
        self.time_provenance = {
            "scale": str(times_obj.scale),
            "source": clock_source,
            "assumed": scale_assumed or reference_assumed,
        }
        self._mission_epoch_override = False
        self.clock_reference_set = resolved_reference_set

        # Public attributes
        if bundle.spectra is None:
            self.spectra = None
        else:
            from lusee.LabeledArray import FRAME_TOPO, label
            self.spectra = label(
                bundle.spectra[:, :, :normal_frequency_count],
                units=bundle.spectra_units or SPECTRA_UNITS,
                frame=FRAME_TOPO,
            )
        self.unique_ids = bundle.spectra_unique_ids
        self.raw_times = bundle.spectra_raw_times
        self.raw_time_valid = bundle.spectra_raw_time_valid
        self.mjd_times = bundle.spectra_mjd_times
        self.mjd_time_valid = bundle.spectra_mjd_time_valid
        self.frequency_counts = bundle.spectra_frequency_counts
        self.navgf = normal_navgf
        self.frequency_window_index = bundle.spectra_frequency_window_index
        self.frequency_windows = bundle.spectra_frequency_windows
        self.metadata = bundle.spectra_metadata
        self.metadata_present = bundle.spectra_metadata_present

        self.tr_spectra = bundle.tr_spectra
        self.tr_unique_ids = bundle.tr_unique_ids
        self.tr_raw_times = bundle.tr_raw_times
        self.tr_raw_time_valid = bundle.tr_raw_time_valid
        self.tr_mjd_times = bundle.tr_mjd_times
        self.tr_mjd_time_valid = bundle.tr_mjd_time_valid
        self.tr_navg2_per_sample = bundle.tr_navg2_per_sample
        self.tr_length_per_sample = bundle.tr_length_per_sample
        self.tr_metadata = bundle.tr_metadata

        self.zoom_spectra = bundle.zoom_spectra
        self.zoom_unique_ids = bundle.zoom_unique_ids
        self.zoom_pfb_indices = bundle.zoom_pfb_indices
        self.zoom_pfb_bins = bundle.zoom_pfb_bins
        self.zoom_raw_times = bundle.zoom_raw_times
        self.zoom_raw_time_valid = bundle.zoom_raw_time_valid
        self.zoom_mjd_times = bundle.zoom_mjd_times
        self.zoom_mjd_time_valid = bundle.zoom_mjd_time_valid

        self.grimm_spectra = bundle.grimm_spectra
        self.grimm_unique_ids = bundle.grimm_unique_ids
        self.grimm_raw_times = bundle.grimm_raw_times
        self.grimm_raw_time_valid = bundle.grimm_raw_time_valid
        self.grimm_mjd_times = bundle.grimm_mjd_times
        self.grimm_mjd_time_valid = bundle.grimm_mjd_time_valid
        self.grimm_navg2_per_sample = bundle.grimm_navg2_per_sample
        self.grimm_average_valid = bundle.grimm_average_valid
        self.waveforms = bundle.waveforms
        self.waveform_times = bundle.waveform_times
        self.waveform_data = bundle.waveform_data
        self.waveform_channels = bundle.waveform_channels
        self.waveform_unique_ids = bundle.waveform_unique_ids
        self.waveform_raw_times = bundle.waveform_raw_times
        self.waveform_raw_time_valid = bundle.waveform_raw_time_valid
        self.waveform_mjd_times = bundle.waveform_mjd_times
        self.waveform_mjd_time_valid = bundle.waveform_mjd_time_valid
        self.waveform_adc_timestamps = bundle.waveform_adc_timestamps
        self.waveform_adc_timestamp_valid = (
            bundle.waveform_adc_timestamp_valid
        )
        self.housekeeping = bundle.housekeeping
        self.housekeeping_unique_ids = bundle.housekeeping_unique_ids
        self.housekeeping_raw_times = bundle.housekeeping_raw_times
        self.housekeeping_raw_time_valid = bundle.housekeeping_raw_time_valid
        self.housekeeping_mjd_times = bundle.housekeeping_mjd_times
        self.housekeeping_mjd_time_valid = bundle.housekeeping_mjd_time_valid
        self.housekeeping_types = bundle.housekeeping_types
        self.housekeeping_versions = bundle.housekeeping_versions
        self.housekeeping_firmware_errors = bundle.housekeeping_firmware_errors
        self.housekeeping_fields = bundle.housekeeping_fields
        self.housekeeping_field_present = bundle.housekeeping_field_present
        self.calibrator = bundle.calibrator
        self.dcb_telemetry = bundle.dcb_fpga
        self.encoder_telemetry = bundle.dcb_encoder
        self.interp_telemetry = bundle.interp_telemetry
        self.session_invariants = bundle.session_invariants
        self.run_provenance = bundle.run_provenance
        self.decoder_provenance = bundle.decoder_provenance
        self.product_provenance = bundle.product_provenance
        self.issues = bundle.issues
        self.family_status = bundle.family_status
        self.quality_status = bundle.quality_status
        self.execution_mode = bundle.execution_mode
        self.legacy_unverified_families = bundle.legacy_unverified_families
        self.bundle = bundle

        self.source_paths: List[Path] = list(
            bundle.source_paths
            or tuple(
                b.source_path for b in self.bundles if b.source_path is not None
            )
        )
        self.layout_version = bundle.layout_version

        # Convenience: count properties
        self.Nspectra = self.spectra.shape[0] if self.spectra is not None else 0
        self.Nfreq = self.spectra.shape[2] if self.spectra is not None else 0
        self.Nproducts = NPRODUCTS

        if self.spectra is None:
            self.frequency_window = None
            self.frequency_coordinate_status = None
            self.freq = None
        elif self.layout_version == INGEST_LAYOUT_VERSION:
            self.frequency_window = bundle.frequency_window_for_row(0)
            self.frequency_coordinate_status = (
                self.frequency_window.frequency_coordinate_status
            )
            self.freq = None
        else:
            self.frequency_window = bundle.frequency_window_for_row(0)
            self.frequency_coordinate_status = "legacy_unverified"
            self.freq = self._derive_freq()

    # -------------------- Indexing --------------------

    def __getitem__(self, req):
        """``data[time_idx, comb, freq_idx]`` -- see class docstring."""
        if self.spectra is None:
            raise ValueError("no /spectra in this IngestData")
        time_idx, comb, freq_idx = req
        i, j, mode, sign, conjugate = _resolve_combination(comb)
        try:
            re_idx, im_idx = _PRODUCT_INDEX[(i, j)]
        except KeyError as exc:
            raise KeyError(f"unknown channel pair ({i}, {j})") from exc

        head = mode[0] if mode else "R"
        if head == "R":
            data = self.spectra[time_idx, re_idx, freq_idx]
        elif head == "I":
            if im_idx is None:
                raise ValueError(f"no imaginary channel for autocorrelation ({i},{j})")
            data = self.spectra[time_idx, im_idx, freq_idx]
        elif head == "C":
            if im_idx is None:
                raise ValueError(f"no complex form for autocorrelation ({i},{j})")
            data = (self.spectra[time_idx, re_idx, freq_idx]
                    + 1j * self.spectra[time_idx, im_idx, freq_idx])
            if conjugate:
                # LabeledArray.conj() preserves the raw SDU decoration;
                # np.conj() intentionally returns a bare array.
                data = data.conj()
        else:
            raise ValueError(f"unknown combination mode {mode!r}")
        return sign * data

    def cross(self, i: int, j: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        """Convenience: complex cross-correlation as one ndarray."""
        return self[time_idx, (i, j, "C"), freq_idx]

    def auto(self, i: int, *, time_idx=slice(None), freq_idx=slice(None)) -> np.ndarray:
        return self[time_idx, (i, i, "R"), freq_idx]

    # -------------------- Physical-unit conversion --------------------

    # Telemetry channels the PCA gain model regresses on (see
    # lusee.GainModel.SpectrometerGain). All six live in
    # ``spectra_interpolated_telemetry`` once a session is ingested with
    # ``interpolate_telemetry=True``.
    _GAIN_TELEMETRY_KEYS = (
        "THERM_FPGA", "SPE_ADC0_T", "SPE_ADC1_T",
        "SPE_1VAD8_V", "VMON_1V2D", "SPE_1VAD8_C",
    )

    @staticmethod
    def _resolve_levels(levels):
        """Normalize a gain-level spec to a length-4 per-channel sequence.

        Accepts 'H' (one level for all four channels), 'HHHH' / 'LMHH'
        (one char per channel), a length-4 sequence, or a
        ``{channel: level}`` dict (passed through unchanged).
        """
        if isinstance(levels, str):
            s = levels.strip().upper()
            if len(s) == 1:
                return [s] * _N_ADC_CHANNELS
            if len(s) == _N_ADC_CHANNELS:
                return list(s)
            raise ValueError(
                f"string gain level must be 1 or {_N_ADC_CHANNELS} chars; got {levels!r}"
            )
        if isinstance(levels, dict):
            return levels
        levels = list(levels)
        if len(levels) != _N_ADC_CHANNELS:
            raise ValueError(f"need {_N_ADC_CHANNELS} per-channel levels; got {levels!r}")
        return levels

    def to_physical(
        self,
        levels=None,
        *,
        freqs_mhz=None,
        telemetry=None,
        gain=None,
        chunk_size=None,
    ):
        """Convert stored SDU spectra to nV/sqrt(Hz) on demand.

        The conversion is applied here, after loading; it is never stored
        in the HDF5. Bit-slice restoration has already happened during
        ingestion and is not an optional calibration switch. This method
        uses :class:`lusee.GainModel.SpectrometerGain` with explicit
        row-aligned telemetry (or a legacy file's interpolated telemetry).

        Per spectra time sample the model predicts a gain spectrum for each
        channel from that sample's telemetry, then maps counts to
        nV/sqrt(Hz): autos as ``sqrt(X / G)`` and crosses as
        ``sign(X) * sqrt(|X| / sqrt(Ga*Gb))``. Bins outside the model's
        anchor-frequency range (or with invalid gain/power) come out NaN.
        This cross convention is retained for compatibility and ASD sanity
        plots; use :meth:`to_physical_psd` for linear cross components.

        :param levels: Per-channel ADC gain setting. The default (None) reads
            the realized per-sample, per-channel gain from
            ``/spectra/metadata/actual_gain`` (GAIN_LOW/MED/HIGH -> 'L'/'M'/'H';
            disable/auto codes have no model and yield NaN for that channel).
            Pass an explicit level to override every sample: a single letter
            for all four channels, a length-4 string / sequence, or a
            ``{channel: level}`` dict.
        :param freqs_mhz: Reviewed frequency grid for the conversion. Layout
            v4 requires this explicitly while FREQ-001 remains unresolved.
        :param telemetry: Optional mapping of gain-model telemetry names to
            arrays with one value per normal-spectrum row. Required for v4.
        :param gain: An existing :class:`SpectrometerGain` to reuse; a
            cached one is created on first use otherwise.
        :param chunk_size: Optional positive number of time rows per gain-model
            batch. This bounds intermediate gain memory without changing the
            returned array or the model snapshot used for the call.
        :returns: :class:`lusee.LabeledArray` shaped like ``self.spectra``
            (Nspectra, NPRODUCTS, Nfreq), units "nV/sqrt(Hz)".

        Model-family selection is process-wide. ``lusee.GainModel.set_models``
        changes the selection used by the next call; ``convert_batch``
        snapshots that selection once so a concurrent update cannot split a
        conversion across two model configurations.
        """
        from lusee.LabeledArray import FRAME_TOPO, label

        gain, tel, level_rows, freqs = self._gain_conversion_inputs(
            levels, freqs_mhz, telemetry, gain
        )
        out = gain.convert_batch(
            np.asarray(self.spectra),
            tel,
            level_rows,
            freqs_mhz=freqs,
            chunk_size=chunk_size,
        )
        return label(out, frame=FRAME_TOPO)

    def to_physical_psd(
        self,
        levels=None,
        *,
        freqs_mhz=None,
        telemetry=None,
        gain=None,
        units="V^2/Hz",
        chunk_size=None,
    ):
        """Return the physically linear input-referred spectral-density cube.

        Autos are ``X/G`` and cross real/imaginary components are
        ``X/sqrt(Ga*Gb)``.  Unlike :meth:`to_physical`, this does not apply a
        signed square root to cross components.  ``units`` may be ``"V^2/Hz"``
        (default) or the gain model's native ``"nV^2/Hz"``.  The result is
        computed on demand and is never persisted in the ingest file.
        """
        from lusee.LabeledArray import FRAME_TOPO, label

        gain, tel, level_rows, freqs = self._gain_conversion_inputs(
            levels, freqs_mhz, telemetry, gain
        )
        out = gain.convert_batch_psd(
            np.asarray(self.spectra),
            tel,
            level_rows,
            freqs_mhz=freqs,
            units=units,
            chunk_size=chunk_size,
        )
        return label(out, frame=FRAME_TOPO)

    # Short, discoverable alias; the longer name makes the relationship with
    # the legacy ASD-returning to_physical() explicit in user-facing code.
    def to_psd(
        self,
        levels=None,
        *,
        freqs_mhz=None,
        telemetry=None,
        gain=None,
        units="V^2/Hz",
        chunk_size=None,
    ):
        return self.to_physical_psd(
            levels,
            freqs_mhz=freqs_mhz,
            telemetry=telemetry,
            gain=gain,
            units=units,
            chunk_size=chunk_size,
        )

    def _gain_conversion_inputs(self, levels, freqs_mhz, telemetry, gain):
        """Validate and prepare shared inputs for vectorized conversion."""
        from lusee.GainModel import SpectrometerGain

        if self.spectra is None:
            raise ValueError("no /spectra in this IngestData to convert")
        telemetry_source = self.interp_telemetry if telemetry is None else telemetry
        if not telemetry_source:
            raise ValueError(
                "no row-aligned gain telemetry is available; pass telemetry= "
                "with one value per spectrum row"
            )
        missing = [k for k in self._GAIN_TELEMETRY_KEYS
                   if k not in telemetry_source]
        if missing:
            raise ValueError(
                f"interpolated telemetry is missing gain-model channels {missing}"
            )
        if gain is None:
            gain = getattr(self, "_gain_model", None)
            if gain is None:
                gain = SpectrometerGain()
                self._gain_model = gain
        if freqs_mhz is None:
            if getattr(self, "layout_version", None) == INGEST_LAYOUT_VERSION:
                raise UnresolvedFrequencyCoordinateError(
                    "layout-v4 native averaging windows have no reviewed MHz "
                    "coordinate; pass an explicit reviewed freqs_mhz grid"
                )
            freqs = np.asarray(self.freq, dtype=float)
        else:
            freqs = np.asarray(freqs_mhz, dtype=float)
        if (
            freqs.shape != (self.Nfreq,)
            or not np.all(np.isfinite(freqs))
            or np.any(np.diff(freqs) <= 0.0)
        ):
            raise ValueError(
                f"freqs_mhz must be finite, strictly increasing, and have "
                f"shape ({self.Nfreq},)"
            )
        level_rows = np.asarray(self._level_rows(levels), dtype=object)
        telemetry = {
            k: np.asarray(telemetry_source[k], dtype=float)
            for k in self._GAIN_TELEMETRY_KEYS
        }
        for name, values in telemetry.items():
            if values.shape != (self.Nspectra,):
                raise ValueError(
                    f"telemetry channel {name!r} must have shape "
                    f"({self.Nspectra},); got {values.shape}"
                )
        return gain, telemetry, level_rows, freqs

    def _level_rows(self, levels):
        """Per-sample, per-channel gain levels as a list of NCH-long lists.

        ``levels=None`` reads the realized gain codes from
        ``/spectra/metadata/actual_gain`` (one row per spectra sample); any
        explicit ``levels`` is resolved once and broadcast to every sample.
        Entries are 'L'/'M'/'H', or None where the recorded code has no
        L/M/H model (disable/auto).
        """
        n = self.spectra.shape[0]
        if levels is not None:
            resolved = self._resolve_levels(levels)
            if isinstance(resolved, dict):
                row = [resolved[ch] for ch in range(_N_ADC_CHANNELS)]
            else:
                row = list(resolved)
            return [row] * n
        ag = self.metadata.get("actual_gain")
        if ag is None:
            raise ValueError(
                "levels=None auto-detects the gain from "
                "/spectra/metadata/actual_gain, which is absent; re-ingest with "
                "the current pipeline (it propagates the meta 'base' fields) or "
                "pass levels= explicitly"
            )
        ag = np.asarray(ag).reshape(n, -1)[:, -_N_ADC_CHANNELS:]
        return [[self._level_from_code(c) for c in ag[t]] for t in range(n)]

    @staticmethod
    def _level_from_code(code):
        """Map one actual_gain entry to 'L'/'M'/'H', or None if unmodeled.

        Accepts the integer gain_state enum (GAIN_LOW/MED/HIGH = 0/1/2) or a
        char code ('L'/'M'/'H'); disable/auto/NaN/unknown -> None.
        """
        if isinstance(code, (bytes, np.bytes_)):
            code = code.decode("ascii", "ignore")
        if isinstance(code, str):
            code = code.strip().upper()
            return code if code in ("L", "M", "H") else None
        try:
            numeric = float(code)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or numeric != np.rint(numeric):
            return None
        return _GAIN_CODE_TO_LEVEL.get(int(numeric))

    # -------------------- Misc helpers --------------------

    def session_boundaries(self) -> List[Tuple[int, int, Optional[Path]]]:
        """For multi-file loads: list of ``(start, end, source)`` tuples."""
        out = []
        cursor = 0
        if self.bundle.session_spectra_counts:
            for count, source in zip(
                self.bundle.session_spectra_counts,
                self.bundle.session_sources,
            ):
                out.append((cursor, cursor + count, source))
                cursor += count
            return out
        for b in self.bundles:
            n = b.spectra.shape[0] if b.spectra is not None else 0
            out.append((cursor, cursor + n, b.source_path))
            cursor += n
        return out

    def _derive_freq(self) -> np.ndarray:
        """Return only the explicit layout-v2/v3 compatibility coordinate."""
        warnings.warn(
            "layout-v2/v3 MHz coordinates are legacy_unverified; layout v4 "
            "refuses this origin-zero adapter while FREQ-001 is open",
            LegacyIngestWarning,
            stacklevel=3,
        )
        navgf = int(np.asarray(self.navgf)[0]) if self.navgf is not None else 1
        stride = spectrometer_frequency_window(navgf).stride
        return np.arange(self.Nfreq) * (_FREQ_STEP_MHZ_NAVGF1 * stride)

    def __repr__(self) -> str:
        n_sp = self.spectra.shape[0] if self.spectra is not None else 0
        n_tr = self.tr_spectra.shape[0] if self.tr_spectra is not None else 0
        n_zm = self.zoom_spectra.shape[0] if self.zoom_spectra is not None else 0
        n_src = len(self.source_paths)
        has_tlm = bool(self.dcb_telemetry)
        return (f"IngestData(N_spectra={n_sp}, N_tr={n_tr}, "
                f"N_zoom={n_zm}, sources={n_src}, telemetry={has_tlm})")

    # -------------------- Plotting --------------------

    def _plot_frequency_axis(self) -> tuple[np.ndarray, str]:
        if self.freq is None:
            return np.arange(self.Nfreq), "native-window output bin"
        return np.asarray(self.freq), "frequency (MHz; legacy_unverified)"

    def plot_waterfall(self, comb, *, ax=None, log=True, **imshow_kw):
        """Render an (n_time, n_freq) waterfall for one combination string."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        data = np.asarray(self[:, comb, :], dtype=np.float64)
        if np.iscomplexobj(data):
            data = np.abs(data)
        plot = data
        cbar_label = "value"
        if log:
            with np.errstate(invalid="ignore"):
                plot = np.where(data > 0, np.log10(np.abs(data)), np.nan)
            cbar_label = "log10|value|"
        kwargs = dict(aspect="auto", origin="lower", cmap="viridis")
        kwargs.update(imshow_kw)
        x, x_label = self._plot_frequency_axis()
        kwargs.setdefault(
            "extent",
            [x[0] if x.size else 0, x[-1] if x.size else self.Nfreq, 0, plot.shape[0]],
        )
        im = ax.imshow(plot, **kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel("sample index")
        ax.set_title(f"{comb} -- waterfall")
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
        return ax

    def plot_mean_spectrum(self, comb, *, ax=None, log=True):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        data = np.asarray(self[:, comb, :], dtype=np.float64)
        if np.iscomplexobj(data):
            data = np.abs(data)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(data, axis=0)
            std = np.nanstd(data, axis=0)
        x, x_label = self._plot_frequency_axis()
        if x.size != mean.size:
            x = np.arange(mean.size)
            x_label = "native-window output bin"
        if log:
            valid = (mean > 0) & np.isfinite(mean)
            ax.semilogy(x[valid], mean[valid], "b-", label="mean")
            ax.fill_between(x[valid],
                            np.maximum(mean[valid] - std[valid], 1e-12),
                            mean[valid] + std[valid],
                            alpha=0.3, color="blue")
        else:
            ax.plot(x, mean, "b-", label="mean")
            ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="blue")
            ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel(x_label)
        ax.set_ylabel("|value|")
        ax.set_title(f"{comb} -- time-averaged spectrum (N={data.shape[0]})")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_dcb(self, channels=None, *, ax=None):
        """Plot one or more DCB telemetry channels vs mission time."""
        import matplotlib.pyplot as plt
        if not self.dcb_telemetry:
            raise ValueError("no DCB telemetry available")
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        ms = self.dcb_telemetry.get("mission_seconds")
        if ms is None or ms.size == 0:
            raise ValueError("DCB telemetry has no time axis")
        ss = self.dcb_telemetry.get("lusee_subsecs")
        if ss is None:
            ss = np.zeros_like(ms)
        t = ms + ss * (1.0 / 65536.0)
        t = t - t[0]
        if channels is None:
            channels = ["THERM_FPGA", "THERM_DCB", "VMON_6V"]
        for ch in channels:
            if ch in self.dcb_telemetry:
                ax.plot(t, self.dcb_telemetry[ch], label=ch, lw=0.9)
            else:
                log.info("dcb channel %s not present", ch)
        ax.set_xlabel("seconds since first telemetry sample")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax

    def plot_adc_stats(self, *, fig=None):
        """Per-channel ADC min/max/mean/rms over time, 4 panels."""
        import matplotlib.pyplot as plt
        adc_min = self.metadata.get("adc_min")
        adc_max = self.metadata.get("adc_max")
        adc_mean = self.metadata.get("adc_mean")
        adc_rms = self.metadata.get("adc_rms")
        if any(a is None for a in (adc_min, adc_max, adc_mean, adc_rms)):
            raise ValueError("ADC stat metadata not present in this IngestData")
        names = ("adc_min", "adc_max", "adc_mean", "adc_rms")
        arrays = [
            np.asarray(array, dtype=np.float64).copy()
            for array in (adc_min, adc_max, adc_mean, adc_rms)
        ]
        valid = self.metadata.get("adc_statistics_valid")
        if valid is not None:
            valid = np.asarray(valid, dtype=np.bool_)
            for array in arrays:
                array[~valid] = np.nan
        for name, array in zip(names, arrays):
            present = self.metadata_present.get(name)
            if present is not None:
                array[~np.asarray(present, dtype=np.bool_)] = np.nan
        adc_min, adc_max, adc_mean, adc_rms = arrays
        if fig is None:
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        else:
            axes = np.array(fig.subplots(2, 2)).reshape(2, 2)
        x = self.raw_times - self.raw_times[0] if self.raw_times.size else np.arange(adc_min.shape[0])
        for ax, arr, title in zip(
            axes.flat,
            (adc_min, adc_max, adc_mean, adc_rms),
            ("ADC min", "ADC max", "ADC mean", "ADC rms"),
        ):
            for ch in range(arr.shape[1]):
                ax.plot(x, arr[:, ch], lw=0.8, label=f"ch{ch}")
            ax.set_title(title)
            ax.set_xlabel("seconds since session start")
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------

def _pick_time_axis(
    bundle: SessionBundle,
    *,
    preferred: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return (mjd_times, raw_times, source_name) by preference, else fallback ladder."""
    candidates = [
        ("spectra", bundle.spectra_mjd_times, bundle.spectra_raw_times),
        ("tr_spectra", bundle.tr_mjd_times, bundle.tr_raw_times),
        ("zoom_spectra", bundle.zoom_mjd_times, bundle.zoom_raw_times),
        (
            "grimm_spectra",
            bundle.grimm_mjd_times,
            bundle.grimm_raw_times,
        ),
    ]
    known = {name for name, _, _ in candidates}
    if preferred not in known:
        raise ValueError(
            f"unknown time_source {preferred!r}; expected one of "
            + ", ".join(sorted(known))
        )
    order = [preferred] + [name for name, _, _ in candidates if name != preferred]
    by_name = {name: (mjd, raw) for name, mjd, raw in candidates}

    for name in order:
        if name not in by_name:
            continue
        mjd, raw = by_name[name]
        if (raw is not None and raw.size > 0) or (mjd is not None and mjd.size > 0):
            return mjd, raw, name
    return None, None, preferred


def _time_validity(
    bundle: SessionBundle,
    source: str,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    fields = {
        "spectra": (
            bundle.spectra_mjd_time_valid,
            bundle.spectra_raw_time_valid,
        ),
        "tr_spectra": (
            bundle.tr_mjd_time_valid,
            bundle.tr_raw_time_valid,
        ),
        "zoom_spectra": (
            bundle.zoom_mjd_time_valid,
            bundle.zoom_raw_time_valid,
        ),
        "grimm_spectra": (
            bundle.grimm_mjd_time_valid,
            bundle.grimm_raw_time_valid,
        ),
    }
    mjd_valid, raw_valid = fields[source]
    return mjd_valid, raw_valid


def _mjd_is_calibrated(bundle: SessionBundle) -> bool:
    """MJD is meaningful only when the calibration constants are non-default."""
    raw_subtract = bundle.constants.get("raw_time_subtract_seconds", 0.0)
    mjd_offset = bundle.constants.get("mjd_epoch_offset_days", 0.0)
    return bool(raw_subtract or mjd_offset)


def _legacy_clock_reference_set(
    bundle: SessionBundle,
    assume_scale: str | None,
) -> LegacyClockReferenceSet | None:
    constants = bundle.constants
    required = ("raw_time_subtract_seconds", "mjd_epoch_offset_days")
    if any(name not in constants for name in required):
        return None
    raw_anchor = float(constants["raw_time_subtract_seconds"])
    mjd_offset = float(constants["mjd_epoch_offset_days"])
    if not np.isfinite(raw_anchor) or not np.isfinite(mjd_offset):
        raise ValueError("legacy clock constants must be finite")
    if raw_anchor == 0.0 and mjd_offset == 0.0:
        return None
    file_scale = str(constants.get("time_scale", DEFAULT_TIME_SCALE)).lower()
    scale = assume_scale if file_scale == DEFAULT_TIME_SCALE else file_scale
    if scale is None:
        return None
    if assume_scale is not None and file_scale != DEFAULT_TIME_SCALE and (
        assume_scale != file_scale
    ):
        raise ValueError(
            f"assume_scale={assume_scale!r} contradicts the recorded "
            f"time_scale={file_scale!r}"
        )
    digest_payload = (
        f"layout={bundle.layout_version};raw={raw_anchor!r};"
        f"mjd={mjd_offset!r};scale={scale}"
    ).encode("ascii")
    warnings.warn(
        "layout-v2/v3 subtract-plus-MJD time mapping is legacy_unverified; "
        "it is not a verified landing reference",
        LegacyIngestWarning,
        stacklevel=3,
    )
    return LegacyClockReferenceSet(
        clock_reference_raw_seconds=raw_anchor,
        mjd_epoch_offset_days=mjd_offset,
        time_scale=scale,
        source="layout-v2/v3 compatibility adapter",
        assumed=file_scale == DEFAULT_TIME_SCALE,
        mapping_sha256=hashlib.sha256(digest_payload).hexdigest(),
    )


def _legacy_reference_matches_production(
    legacy: LegacyClockReferenceSet,
    production: ClockReferenceSet,
) -> bool:
    if legacy.time_scale != production.time_scale:
        return False
    production_anchor = production.reference_for(ClockSource.SPECTROMETER)
    if production_anchor is None or (
        production_anchor.clock_reference_raw_seconds
        != legacy.clock_reference_raw_seconds
    ):
        return False
    production_mjd = float(
        production.to_mjd(
            production_anchor.clock_reference_raw_seconds,
            clock_source=ClockSource.SPECTROMETER,
        )
    )
    return bool(
        np.isclose(
            production_mjd,
            legacy.mjd_epoch_offset_days,
            rtol=0.0,
            atol=1e-12,
        )
    )


# ---------------------------------------------------------------------------
# Convenience free function
# ---------------------------------------------------------------------------

def load_bundle(
    target,
    *,
    prefer_format: str = "h5",
) -> SessionBundle:
    """Read one or more ingest files without constructing absolute time."""
    if isinstance(target, SessionBundle):
        return target
    paths = _resolve_paths(target, prefer_format=prefer_format)
    bundles = sorted((_load_one(path) for path in paths), key=_bundle_sort_key)
    return _concat_bundles(bundles)


def load(
    paths,
    *,
    prefer_format: str = "h5",
    time_source: str = "spectra",
    mission_epoch=None,
    assume_scale=None,
    clock_reference_set: ClockReferenceSet | None = None,
) -> IngestData:
    """Build an :class:`IngestData` from one or more files / directories.

    See :class:`IngestData` for the full kwarg description.
    """
    return IngestData(
        paths,
        prefer_format=prefer_format,
        time_source=time_source,
        mission_epoch=mission_epoch,
        assume_scale=assume_scale,
        clock_reference_set=clock_reference_set,
    )


PathsLike = Union[str, Path, Iterable[Union[str, Path]]]
