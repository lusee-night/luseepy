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
    write_hdf5                           -- Stage 7

  Manifest
    SessionResult, write_manifest

  Visualization
    plot_session, plot_spectra_waterfall, plot_spectra_mean,
    plot_adc_stats, plot_dcb_telemetry

Most callers will only need ``process_flash`` or ``process_session``.
"""

from __future__ import annotations

from .ccsds import (
    CcsdsFrame,
    PrimaryHeader,
    crc16_ccitt,
    parse_bank_file,
    parse_primary_header,
    parse_stream,
)
from .collation import (
    LogicalPacket,
    assign_identities,
    is_dropped_appid,
    is_uid_derived,
    is_uid_prefixed,
    is_uid_typed,
    reassemble_logical_packets,
)
from .decode import (
    CalDataSample,
    HKSample,
    Products,
    SpectrumSample,
    TRSpectrumSample,
    WaveformSample,
    ZoomSample,
    read_uncrater_session,
)
from .fits_writer import write_fits
from .hdf5_writer import write_hdf5
from .pipeline import (
    SessionResult,
    parse_flash,
    process_flash,
    process_session,
    write_manifest,
)
from .session import (
    Session,
    raw_seconds_from_split_time,
    split_sessions,
    write_uncrater_session,
)
from .telemetry import (
    field_groups,
    find_legacy_sidecar,
    has_decoder,
    parse_b01_packets,
    parse_legacy_sidecar,
    slice_arrays_by_window,
    telemetry_apids,
)

# Visualization is optional (matplotlib may be absent in some envs).
try:
    from .viz import (
        plot_adc_stats,
        plot_dcb_telemetry,
        plot_session,
        plot_spectra_mean,
        plot_spectra_waterfall,
    )
except ImportError:    # pragma: no cover
    plot_session = None    # type: ignore[assignment]
    plot_spectra_waterfall = None    # type: ignore[assignment]
    plot_spectra_mean = None    # type: ignore[assignment]
    plot_adc_stats = None    # type: ignore[assignment]
    plot_dcb_telemetry = None    # type: ignore[assignment]


# IngestData / load are lazily imported via __getattr__ below: their
# implementation lives in obs_factory.py, which depends on lusee.Observation
# (and thus astropy + lunarsky + a SPICE-kernel download). Importing
# lusee.ingest itself stays light; the heavy chain only triggers when a
# user actually accesses lusee.ingest.IngestData or lusee.ingest.load.

_LAZY_FROM_OBS_FACTORY = ("IngestData", "load", "SessionBundle")


def __getattr__(name):
    if name in _LAZY_FROM_OBS_FACTORY:
        from . import obs_factory
        value = getattr(obs_factory, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'lusee.ingest' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_FROM_OBS_FACTORY))


__all__ = [
    # ccsds
    "CcsdsFrame", "PrimaryHeader", "crc16_ccitt",
    "parse_bank_file", "parse_primary_header", "parse_stream",
    # collation
    "LogicalPacket", "assign_identities",
    "is_uid_prefixed", "is_uid_typed", "is_uid_derived", "is_dropped_appid",
    "reassemble_logical_packets",
    # session
    "Session", "raw_seconds_from_split_time",
    "split_sessions", "write_uncrater_session",
    # telemetry (thin proxy to the private lusee_telemetry decoder)
    "field_groups", "find_legacy_sidecar", "has_decoder",
    "parse_b01_packets", "parse_legacy_sidecar",
    "slice_arrays_by_window", "telemetry_apids",
    # decode
    "CalDataSample", "HKSample", "Products", "SpectrumSample",
    "TRSpectrumSample", "WaveformSample", "ZoomSample",
    "read_uncrater_session",
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
    "IngestData", "load", "SessionBundle",
]
