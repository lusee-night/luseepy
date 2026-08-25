"""LuSEE-Night downlink ingest pipeline.

This subpackage converts raw CCSDS binary downlinks (or already-extracted
"uncrater session" directories) into HDF5 science products and
sanity-check plots.

Public API:

  Top-level orchestrators
    process_flash    -- raw FLASH_TLMFS dir -> sessions on disk + HDF5 + plots
    process_session  -- existing uncrater session dir -> HDF5 + plots
    parse_flash      -- Stages 1..4 only; returns Session objects in memory

  Building blocks (no disk writes unless documented otherwise)
    parse_stream, parse_bank_file        -- Stage 1 (CCSDS framing)
    reassemble_logical_packets           -- Stage 2
    assign_identities                    -- Stage 3
    split_sessions                       -- Stage 4
    write_uncrater_session               -- Stage 5 (writes a session dir)
    read_uncrater_session                -- Stage 6
    write_hdf5                           -- validated WriteRequest -> layout-v4 HDF5

  Manifest
    SessionResult, write_manifest

  Visualization
    plot_session, plot_spectra_waterfall, plot_spectra_mean,
    plot_adc_stats, plot_dcb_telemetry

Most callers will only need ``process_flash`` or ``process_session``.
"""

from __future__ import annotations

from importlib import import_module

from .ccsds import (
    CcsdsFrame,
    FrameLocation,
    FramingResult,
    PrimaryHeader,
    crc16_ccitt,
    parse_bank_file,
    parse_bank_file_diagnostic,
    parse_primary_header,
    parse_stream,
    parse_stream_diagnostic,
)
from .dependencies import MissingIngestExtraError
from .issues import (
    IngestIssue,
    IngestIssueError,
    IssueAction,
    IssueCollector,
    IssuePolicy,
    IssueSeverity,
)
from .reassembly import LogicalPacket, reassemble_logical_packets

_LAZY_EXPORTS = {
    # Stage 3 identity assignment
    "assign_identities": ("collation", "assign_identities"),
    "is_dropped_appid": ("collation", "is_dropped_appid"),
    "is_uid_derived": ("collation", "is_uid_derived"),
    "is_uid_prefixed": ("collation", "is_uid_prefixed"),
    "is_uid_typed": ("collation", "is_uid_typed"),
    # Session split and persistence
    "Session": ("session", "Session"),
    "raw_seconds_from_split_time": ("session", "raw_seconds_from_split_time"),
    "split_sessions": ("session", "split_sessions"),
    "write_uncrater_session": ("session", "write_uncrater_session"),
    # Optional private-decoder boundary and typed public telemetry
    "TelemetryBlock": ("telemetry", "TelemetryBlock"),
    "TelemetryCollection": ("telemetry", "TelemetryCollection"),
    "TelemetryCounts": ("telemetry", "TelemetryCounts"),
    "TelemetryCoverage": ("telemetry", "TelemetryCoverage"),
    "TelemetryDecodeResult": ("telemetry", "TelemetryDecodeResult"),
    "TelemetryDecoderInfo": ("telemetry", "TelemetryDecoderInfo"),
    "TelemetryDecoderStatus": ("telemetry", "TelemetryDecoderStatus"),
    "TelemetryFieldMetadata": ("telemetry", "TelemetryFieldMetadata"),
    "TelemetryInputState": ("telemetry", "TelemetryInputState"),
    "decode_b01_packets": ("telemetry", "decode_b01_packets"),
    "decode_legacy_sidecar": ("telemetry", "decode_legacy_sidecar"),
    "field_groups": ("telemetry", "field_groups"),
    "find_legacy_sidecar": ("telemetry", "find_legacy_sidecar"),
    "has_decoder": ("telemetry", "has_decoder"),
    "parse_b01_packets": ("telemetry", "parse_b01_packets"),
    "parse_legacy_sidecar": ("telemetry", "parse_legacy_sidecar"),
    "slice_arrays_by_window": ("telemetry", "slice_arrays_by_window"),
    "telemetry_apids": ("telemetry", "telemetry_apids"),
    # Decoder products
    "CalDataSample": ("decode", "CalDataSample"),
    "Products": ("decode", "Products"),
    "read_uncrater_session": ("decode", "read_uncrater_session"),
    # Validated product/decode provenance
    "CalibratorDataSample": ("products", "CalibratorDataSample"),
    "CalibratorDebugPage": ("products", "CalibratorDebugPage"),
    "CalibratorDebugSample": ("products", "CalibratorDebugSample"),
    "CalibratorMetadataSample": ("products", "CalibratorMetadataSample"),
    "CalibratorRawPFBSample": ("products", "CalibratorRawPFBSample"),
    "DataQuality": ("products", "DataQuality"),
    "DecodeProvenance": ("products", "DecodeProvenance"),
    "ExecutionMode": ("products", "ExecutionMode"),
    "GrimmSample": ("products", "GrimmSample"),
    "HKSample": ("products", "HKSample"),
    "ProductProvenance": ("products", "ProductProvenance"),
    "SourcePacketProvenance": ("products", "SourcePacketProvenance"),
    "SpectrumMetadata": ("products", "SpectrumMetadata"),
    "SpectrumSample": ("products", "SpectrumSample"),
    "TRSpectrumSample": ("products", "TRSpectrumSample"),
    "ValidatedCounts": ("products", "ValidatedCounts"),
    "WaveformSample": ("products", "WaveformSample"),
    "ZoomSample": ("products", "ZoomSample"),
    # Versioned frequency-window and clock-reference contracts
    "ClockReference": ("clock_reference", "ClockReference"),
    "ClockReferenceFormatError": (
        "clock_reference", "ClockReferenceFormatError"
    ),
    "ClockReferenceSet": ("clock_reference", "ClockReferenceSet"),
    "LegacyClockReferenceSet": (
        "clock_reference", "LegacyClockReferenceSet"
    ),
    "ClockReferenceUnavailableError": (
        "clock_reference", "ClockReferenceUnavailableError"
    ),
    "ClockSource": ("clock_reference", "ClockSource"),
    "UnsupportedClockSourceError": (
        "clock_reference", "UnsupportedClockSourceError"
    ),
    "load_clock_reference_set": (
        "clock_reference", "load_clock_reference_set"
    ),
    "clock_reference_set_from_record": (
        "clock_reference", "clock_reference_set_from_record"
    ),
    "FrequencyWindowContract": (
        "frequency_contract", "FrequencyWindowContract"
    ),
    "UnresolvedFrequencyCoordinateError": (
        "frequency_contract", "UnresolvedFrequencyCoordinateError"
    ),
    "spectrometer_frequency_window": (
        "frequency_contract", "spectrometer_frequency_window"
    ),
    # Validated layout-v4 writer request
    "FamilyCoverage": ("write_request", "FamilyCoverage"),
    "FamilyStatus": ("write_request", "FamilyStatus"),
    "InterpolationPolicy": ("write_request", "InterpolationPolicy"),
    "LunarLocation": ("write_request", "LunarLocation"),
    "RunProvenance": ("write_request", "RunProvenance"),
    "WriteRequest": ("write_request", "WriteRequest"),
    "family_statuses_for_products": (
        "write_request", "family_statuses_for_products"
    ),
    "family_status_for_telemetry": (
        "write_request", "family_status_for_telemetry"
    ),
    # Writers and orchestration
    "write_hdf5": ("hdf5_writer", "write_hdf5"),
    "write_fits": ("fits_writer", "write_fits"),
    "SessionResult": ("pipeline", "SessionResult"),
    "parse_flash": ("pipeline", "parse_flash"),
    "process_flash": ("pipeline", "process_flash"),
    "process_session": ("pipeline", "process_session"),
    "write_manifest": ("pipeline", "write_manifest"),
    # Visualization
    "plot_adc_stats": ("viz", "plot_adc_stats"),
    "plot_dcb_telemetry": ("viz", "plot_dcb_telemetry"),
    "plot_session": ("viz", "plot_session"),
    "plot_spectra_mean": ("viz", "plot_spectra_mean"),
    "plot_spectra_waterfall": ("viz", "plot_spectra_waterfall"),
    # Reader/factory
    "IngestData": ("obs_factory", "IngestData"),
    "LegacyIngestWarning": ("obs_factory", "LegacyIngestWarning"),
    "MixedFrequencyGridError": ("obs_factory", "MixedFrequencyGridError"),
    "load": ("obs_factory", "load"),
    "load_bundle": ("obs_factory", "load_bundle"),
    "SessionBundle": ("obs_factory", "SessionBundle"),
    "LayoutV4ValidationError": (
        "layout_v4_reader", "LayoutV4ValidationError"
    ),
    # Decoder boundary provenance
    "DecoderInfo": ("uncrater_adapter", "DecoderInfo"),
    "IncompatibleUncraterError": (
        "uncrater_adapter", "IncompatibleUncraterError"
    ),
    "UncraterBindingInfo": ("uncrater_adapter", "UncraterBindingInfo"),
    "binding_info": ("uncrater_adapter", "binding_info"),
    "decoder_info": ("uncrater_adapter", "decoder_info"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'lusee.ingest' has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    # ccsds
    "CcsdsFrame", "FrameLocation", "FramingResult", "PrimaryHeader",
    "crc16_ccitt", "parse_bank_file", "parse_bank_file_diagnostic",
    "parse_primary_header", "parse_stream", "parse_stream_diagnostic",
    # issues
    "IngestIssue", "IngestIssueError", "IssueAction", "IssueCollector",
    "IssuePolicy", "IssueSeverity", "MissingIngestExtraError",
    # collation
    "LogicalPacket", "assign_identities",
    "is_uid_prefixed", "is_uid_typed", "is_uid_derived", "is_dropped_appid",
    "reassemble_logical_packets",
    # session
    "Session", "raw_seconds_from_split_time",
    "split_sessions", "write_uncrater_session",
    # typed telemetry plus established loose-array compatibility wrappers
    "TelemetryBlock", "TelemetryCollection", "TelemetryCounts",
    "TelemetryCoverage", "TelemetryDecodeResult", "TelemetryDecoderInfo",
    "TelemetryDecoderStatus", "TelemetryFieldMetadata", "TelemetryInputState",
    "decode_b01_packets", "decode_legacy_sidecar", "field_groups",
    "find_legacy_sidecar", "has_decoder", "parse_b01_packets",
    "parse_legacy_sidecar",
    "slice_arrays_by_window", "telemetry_apids",
    # decode
    "CalDataSample", "Products", "read_uncrater_session",
    # product/decode provenance
    "CalibratorDataSample", "CalibratorDebugPage", "CalibratorDebugSample",
    "CalibratorMetadataSample", "CalibratorRawPFBSample", "DataQuality",
    "DecodeProvenance", "ExecutionMode", "GrimmSample", "HKSample",
    "ProductProvenance", "SourcePacketProvenance", "SpectrumMetadata",
    "SpectrumSample", "TRSpectrumSample", "ValidatedCounts",
    "WaveformSample", "ZoomSample",
    # v4 clock and frequency contracts
    "ClockReference", "ClockReferenceFormatError", "ClockReferenceSet",
    "LegacyClockReferenceSet",
    "ClockReferenceUnavailableError", "ClockSource",
    "UnsupportedClockSourceError", "load_clock_reference_set",
    "clock_reference_set_from_record",
    "FrequencyWindowContract", "UnresolvedFrequencyCoordinateError",
    "spectrometer_frequency_window",
    # layout-v4 writer request
    "FamilyCoverage", "FamilyStatus", "InterpolationPolicy",
    "LunarLocation", "RunProvenance", "WriteRequest",
    "family_status_for_telemetry", "family_statuses_for_products",
    # hdf5
    "write_hdf5",
    # fits
    "write_fits",
    # pipeline
    "SessionResult", "parse_flash", "process_flash", "process_session",
    "write_manifest",
    # viz
    "plot_session", "plot_spectra_waterfall", "plot_spectra_mean",
    "plot_adc_stats", "plot_dcb_telemetry",
    # obs_factory (lazy)
    "IngestData", "LegacyIngestWarning", "LayoutV4ValidationError",
    "MixedFrequencyGridError", "SessionBundle", "load", "load_bundle",
    # uncrater adapter provenance (lazy)
    "DecoderInfo", "IncompatibleUncraterError", "UncraterBindingInfo",
    "binding_info", "decoder_info",
]
