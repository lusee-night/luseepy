"""Stages 4-5: session splitting and uncrater-session persistence.

A session is the run of logical packets between two Hello (startup)
packets. This module splits the sorted, identity-assigned stream into
``Session`` objects, decodes their start times from the first Hello's
mission-time fields, assigns DCB / encoder telemetry rows to the
session whose mission-time window covers them (when a private telemetry
decoder is loaded; see :mod:`lusee.ingest.telemetry`), and writes each
session out as a directory of ``NNNNN_XXXX.bin`` files (the "uncrater
session" format).
"""

from __future__ import annotations

import ctypes
import errno
import logging
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .clock_reference import ClockReferenceSet, ClockSource
from .constants import (
    FILENAME_APID_HEX_WIDTH,
    FILENAME_PACKET_INDEX_DEFAULT_WIDTH,
    FILENAME_PACKET_INDEX_WIDE_THRESHOLD,
    MISSION_TIME_FRACT_DIVISOR,
    MISSION_TIME_FRACT_SHIFT,
)
from .issues import IssueAction, IssueCollector, IssueSeverity
from .packet_map import (
    PACKET_MAP_FILENAME,
    PacketMapError,
    build_packet_map,
    read_packet_map,
    write_packet_map,
)
from .reassembly import LogicalPacket
from .telemetry import (
    TelemetryCoverage,
    TelemetryDecodeResult,
    TelemetryDecoderStatus,
    map_dcb_absolute_time,
)
from .uncrater_adapter import load_uncrater, read_packet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session record
# ---------------------------------------------------------------------------

@dataclass
class Session:
    ordinal: int
    packets: List[LogicalPacket] = field(default_factory=list)
    start_raw_seconds: Optional[float] = None    # None if no Hello in this session
    start_unique_packet_id: Optional[int] = None
    sw_version: Optional[int] = None
    fw_version: Optional[int] = None
    fw_id: Optional[int] = None
    fw_date: Optional[int] = None
    fw_time: Optional[int] = None
    start_time_32: Optional[int] = None
    start_time_16: Optional[int] = None
    telemetry: TelemetryDecodeResult | None = None

    @property
    def has_startup(self) -> bool:
        return self.start_raw_seconds is not None


# ---------------------------------------------------------------------------
# Mission-time decoding
# ---------------------------------------------------------------------------

def raw_seconds_from_split_time(time_32: int, time_16: int) -> float:
    """Combine the split-time pair into seconds since mission epoch.

    Formula from spec section 6.1:
        raw_seconds = ((((time_16 & 0xFFFF) << 32) + time_32) >> 4) / 4096
    """
    combined = ((int(time_16) & 0xFFFF) << 32) + int(time_32)
    return (combined >> MISSION_TIME_FRACT_SHIFT) / MISSION_TIME_FRACT_DIVISOR


def _read_hello(
    blob: bytes,
    sw_version: Optional[int] = None,
    *,
    issue_collector: IssueCollector | None = None,
    packet: LogicalPacket | None = None,
    packet_index: int | None = None,
    session_ordinal: int | None = None,
) -> Optional[dict]:
    """Decode a Hello packet's identity / time fields. Returns None on failure."""
    decoder = load_uncrater()
    try:
        pkt = decoder.Packet(
            int(decoder.id.AppID_uC_Start), blob=blob, version=sw_version
        )
        read_packet(pkt)
    except Exception as exc:    # noqa: BLE001
        message = f"failed to decode Hello: {exc}"
        if issue_collector is not None:
            issue_collector.record(
                code="session_start.hello_decode_failed",
                severity=IssueSeverity.WARNING,
                stage="session_start",
                message=message,
                action=IssueAction.KEPT,
                bank=None if packet is None else packet.bank,
                packet_index=packet_index,
                appid=(
                    int(decoder.id.AppID_uC_Start)
                    if packet is None
                    else packet.appid
                ),
                sequence_count=None if packet is None else packet.seq,
                uid=None if packet is None else packet.unique_packet_id,
                session=(
                    None
                    if session_ordinal is None
                    else f"ordinal:{session_ordinal}"
                ),
                details={
                    "exception_type": type(exc).__name__,
                    "session_ordinal": session_ordinal,
                    "sw_version": sw_version,
                },
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None

    out = {}
    for name in ("SW_version", "FW_Version", "FW_ID", "FW_Date", "FW_Time",
                 "unique_packet_id", "time_32", "time_16"):
        v = getattr(pkt, name, None)
        if v is not None:
            out[name] = int(v)
    return out


def _populate_session_start(
    session: Session,
    *,
    issue_collector: IssueCollector | None = None,
) -> None:
    """If the session has a first Hello, read it and fill start-* fields."""
    decoder = load_uncrater()
    for fallback_index, p in enumerate(session.packets):
        if decoder.appid_is_hello(p.appid):
            packet_index = (
                p.file_index
                if type(p.file_index) is int and p.file_index >= 0
                else fallback_index
            )
            fields = _read_hello(
                p.blob,
                sw_version=session.sw_version,
                issue_collector=issue_collector,
                packet=p,
                packet_index=packet_index,
                session_ordinal=session.ordinal,
            )
            if fields is None:
                return
            session.sw_version = fields.get("SW_version")
            session.fw_version = fields.get("FW_Version")
            session.fw_id = fields.get("FW_ID")
            session.fw_date = fields.get("FW_Date")
            session.fw_time = fields.get("FW_Time")
            session.start_unique_packet_id = fields.get("unique_packet_id")
            session.start_time_32 = fields.get("time_32")
            session.start_time_16 = fields.get("time_16")
            if (session.start_time_32 is not None
                    and session.start_time_16 is not None):
                session.start_raw_seconds = raw_seconds_from_split_time(
                    session.start_time_32, session.start_time_16
                )
            return


# ---------------------------------------------------------------------------
# Stage 4a: session splitting
# ---------------------------------------------------------------------------

def split_sessions(
    packets: Sequence[LogicalPacket],
    *,
    issue_collector: IssueCollector | None = None,
) -> List[Session]:
    """Split a sorted, identity-assigned packet stream into sessions.

    Rules (spec section 6.1):
      * A new session begins at a Hello that follows at least one
        non-Hello packet in the current session.
      * Consecutive Hellos fold into the current session.
      * Packets preceding any Hello form session 0 with no startup metadata.
    """
    if issue_collector is not None and not isinstance(
        issue_collector, IssueCollector
    ):
        raise TypeError("issue_collector must be an IssueCollector or None")
    sessions: List[Session] = []
    current: Optional[Session] = None
    seen_non_hello = False
    decoder = load_uncrater()

    for p in packets:
        if decoder.appid_is_hello(p.appid):
            if current is None or seen_non_hello:
                current = Session(ordinal=len(sessions))
                sessions.append(current)
                seen_non_hello = False
            current.packets.append(p)
        else:
            if current is None:
                current = Session(ordinal=0)
                sessions.append(current)
            current.packets.append(p)
            seen_non_hello = True

    for s in sessions:
        _populate_session_start(s, issue_collector=issue_collector)
    return sessions


# ---------------------------------------------------------------------------
# Stage 4b: telemetry-to-session assignment
# ---------------------------------------------------------------------------

def assign_telemetry_to_sessions(
    sessions: List[Session],
    telemetry: TelemetryDecodeResult,
    *,
    clock_reference_set: ClockReferenceSet,
    issue_collector: IssueCollector,
) -> TelemetryDecodeResult | None:
    """Assign b01 rows using shared elapsed DCB/spectrometer coordinates.

    Rows that cannot be associated remain in ``unassigned_fpga``. The same
    explicitly unassigned block is carried by each session result so a
    per-session output cannot silently lose source telemetry.
    """
    if not isinstance(telemetry, TelemetryDecodeResult):
        raise TypeError("telemetry must be a TelemetryDecodeResult")
    if not isinstance(clock_reference_set, ClockReferenceSet):
        raise TypeError("clock_reference_set must be a ClockReferenceSet")
    if not isinstance(issue_collector, IssueCollector):
        raise TypeError("issue_collector must be an IssueCollector")
    if telemetry.input_source not in (None, "b01"):
        raise ValueError("session-boundary assignment accepts only b01 telemetry")
    if telemetry.input_source is None:
        return None
    if telemetry.decoder_status is not TelemetryDecoderStatus.AVAILABLE:
        for session in sessions:
            session.telemetry = telemetry
        return None
    if telemetry.fpga is None:
        raise ValueError("available telemetry has no FPGA block")

    full_block = telemetry.fpga
    empty = np.zeros(full_block.row_count, dtype=np.bool_)
    if full_block.row_count == 0:
        for session in sessions:
            session.telemetry = telemetry.with_blocks(
                fpga=full_block.slice_rows(empty),
            )
        return None

    def keep_unassigned(code: str, message: str, details: dict[str, object]):
        issue = issue_collector.record(
            code=code,
            severity=IssueSeverity.WARNING,
            stage="telemetry_assignment",
            message=message,
            action=IssueAction.KEPT,
            details=details,
        )
        issues = (*telemetry.issues, issue)
        unassigned = telemetry.with_blocks(
            fpga=full_block.slice_rows(empty),
            unassigned_fpga=full_block,
            issues=issues,
            coverage=TelemetryCoverage.PARTIAL,
        )
        for session in sessions:
            session.telemetry = unassigned
        return unassigned

    spectrometer_reference = clock_reference_set.reference_for(
        ClockSource.SPECTROMETER
    )
    dcb_reference = clock_reference_set.reference_for(ClockSource.DCB)
    if dcb_reference is not None:
        telemetry = map_dcb_absolute_time(
            telemetry,
            clock_reference_set=clock_reference_set,
            issue_collector=issue_collector,
        )
        assert telemetry.fpga is not None
        full_block = telemetry.fpga
    missing_clocks = [
        name
        for name, reference in (
            (ClockSource.SPECTROMETER.value, spectrometer_reference),
            (ClockSource.DCB.value, dcb_reference),
        )
        if reference is None
    ]
    if missing_clocks:
        return keep_unassigned(
            "telemetry_assignment.missing_clock_reference",
            "b01 telemetry was retained unassigned because a clock reference "
            "is missing",
            {"missing_clock_sources": missing_clocks},
        )

    if not sessions:
        return keep_unassigned(
            "telemetry_assignment.no_sessions",
            "b01 telemetry was retained unassigned because no science session exists",
            {"row_count": full_block.row_count},
        )
    ordinals = [session.ordinal for session in sessions]
    if len(set(ordinals)) != len(ordinals):
        return keep_unassigned(
            "telemetry_assignment.invalid_session_boundaries",
            "b01 telemetry was retained unassigned because session ordinals repeat",
            {"session_ordinals": ordinals},
        )
    ordered_sessions = sorted(sessions, key=lambda session: session.ordinal)
    starts = [session.start_raw_seconds for session in ordered_sessions]
    if any(start is None for start in starts):
        return keep_unassigned(
            "telemetry_assignment.invalid_session_boundaries",
            "b01 telemetry was retained unassigned because a session start is missing",
            {"session_ordinals": [session.ordinal for session in ordered_sessions]},
        )
    start_values = np.asarray(starts, dtype=np.float64)
    if not np.all(np.isfinite(start_values)) or (
        start_values.size > 1 and np.any(np.diff(start_values) <= 0.0)
    ):
        return keep_unassigned(
            "telemetry_assignment.invalid_session_boundaries",
            "b01 telemetry was retained unassigned because session starts "
            "are not monotone",
            {"session_ordinals": [session.ordinal for session in ordered_sessions]},
        )

    assert spectrometer_reference is not None
    assert dcb_reference is not None
    mapped_block = full_block
    session_elapsed = (
        start_values - spectrometer_reference.clock_reference_raw_seconds
    )
    telemetry_elapsed = (
        mapped_block.raw_seconds - dcb_reference.clock_reference_raw_seconds
    )
    assignments = np.searchsorted(
        session_elapsed,
        telemetry_elapsed,
        side="right",
    ) - 1
    pre_session = assignments < 0
    issues = telemetry.issues
    coverage = telemetry.coverage
    unassigned_block = None
    if np.any(pre_session):
        issue = issue_collector.record(
            code="telemetry_assignment.pre_session_rows",
            severity=IssueSeverity.WARNING,
            stage="telemetry_assignment",
            message=(
                "b01 telemetry rows before the first session were retained "
                "unassigned"
            ),
            action=IssueAction.KEPT,
            details={"row_count": int(np.count_nonzero(pre_session))},
        )
        issues = (*issues, issue)
        coverage = TelemetryCoverage.PARTIAL
        unassigned_block = mapped_block.slice_rows(pre_session)
    for index, session in enumerate(ordered_sessions):
        session.telemetry = telemetry.with_blocks(
            fpga=mapped_block.slice_rows(assignments == index),
            unassigned_fpga=unassigned_block,
            issues=issues,
            coverage=coverage,
        )
    if unassigned_block is None:
        return None
    return telemetry.with_blocks(
        fpga=mapped_block.slice_rows(empty),
        unassigned_fpga=unassigned_block,
        issues=issues,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Stage 5: persistence
# ---------------------------------------------------------------------------

def _index_width(n_packets: int) -> int:
    return (FILENAME_PACKET_INDEX_DEFAULT_WIDTH
            if n_packets < FILENAME_PACKET_INDEX_WIDE_THRESHOLD
            else 6)


def packet_filename(index: int, appid: int, *, width: int = FILENAME_PACKET_INDEX_DEFAULT_WIDTH) -> str:
    return f"{index:0{width}d}_{appid:0{FILENAME_APID_HEX_WIDTH}x}.bin"


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically rename one directory without replacing an existing target."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        renamex_np = libc.renamex_np
        renamex_np.argtypes = (
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(target), 0x4)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace directory installation is unavailable",
                target,
            )
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            0x1,
        )
    elif os.name == "nt":
        os.rename(source, target)
        return
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic no-replace directory installation is unavailable",
            target,
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), target)


def write_uncrater_session(
    session: Session,
    dest_dir: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a Session to disk in uncrater session format.

    The directory layout produced is "Layout B" (cdi_output/ subdirectory)
    -- ``dest_dir/cdi_output/NNNNN_XXXX.bin`` plus a versioned
    ``dest_dir/packet_map.json``. The complete tree is first written to a
    sibling staging directory, then installed. An existing ``dest_dir`` is
    refused and left untouched unless ``overwrite=True``;
    replacement removes that directory only after the complete staged tree is
    validated. No telemetry sidecar is written.

    Returns the path of the cdi_output/ subdirectory.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    dest = Path(dest_dir)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            raise FileExistsError(dest)
        if dest.is_symlink() or not dest.is_dir():
            raise NotADirectoryError(dest)
    width = _index_width(len(session.packets))
    filenames = [
        packet_filename(index, packet.appid, width=width)
        for index, packet in enumerate(session.packets)
    ]
    normalize_appid = load_uncrater().normalize_dcb_appid
    packet_map = build_packet_map(
        session.packets,
        filenames,
        normalize_appid=normalize_appid,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=dest.parent,
    ))
    try:
        staging = staging_root / dest.name
        staging.mkdir()
        cdi = staging / "cdi_output"
        cdi.mkdir()
        for p, filename in zip(session.packets, filenames):
            fn = cdi / filename
            with fn.open("wb") as fh:
                fh.write(p.blob)
        write_packet_map(packet_map, staging / PACKET_MAP_FILENAME)
        installed_map = read_packet_map(
            staging,
            cdi,
            normalize_appid=normalize_appid,
        )
        if installed_map != packet_map:
            raise PacketMapError(
                "staged packet map disagrees with retained packet provenance"
            )
        if dest.exists() or dest.is_symlink():
            if not overwrite:
                raise FileExistsError(dest)
            if dest.is_symlink() or not dest.is_dir():
                raise NotADirectoryError(dest)
            shutil.rmtree(dest)
        _rename_noreplace(staging, dest)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    installed_cdi = dest / "cdi_output"
    log.info("wrote %d packets to %s", len(session.packets), installed_cdi)
    return installed_cdi
