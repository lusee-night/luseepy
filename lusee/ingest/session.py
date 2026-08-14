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

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .collation import APID_HELLO, LogicalPacket
from .constants import (
    FILENAME_APID_HEX_WIDTH,
    FILENAME_PACKET_INDEX_DEFAULT_WIDTH,
    FILENAME_PACKET_INDEX_WIDE_THRESHOLD,
    MISSION_TIME_FRACT_DIVISOR,
    MISSION_TIME_FRACT_SHIFT,
)

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
    # Per-field arrays produced by the private telemetry decoder, then
    # sliced to this session's mission-time window. Empty when no decoder
    # is loaded or no records fell into the window.
    fpga_telemetry: Dict[str, np.ndarray] = field(default_factory=dict)
    encoder_telemetry: Dict[str, np.ndarray] = field(default_factory=dict)

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


def _read_hello(blob: bytes, sw_version: Optional[int] = None) -> Optional[dict]:
    """Decode a Hello packet's identity / time fields. Returns None on failure."""
    try:
        from uncrater import Packet  # type: ignore[import-not-found]
    except ImportError:
        warnings.warn(
            "uncrater not available -- Hello decoding skipped (session start "
            "time and firmware metadata will be absent)",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        pkt = Packet(APID_HELLO, blob=blob, version=sw_version)
        pkt._read()
    except Exception as exc:    # noqa: BLE001
        warnings.warn(f"failed to decode Hello: {exc}", RuntimeWarning, stacklevel=2)
        return None

    out = {}
    for name in ("SW_version", "FW_Version", "FW_ID", "FW_Date", "FW_Time",
                 "unique_packet_id", "time_32", "time_16"):
        v = getattr(pkt, name, None)
        if v is not None:
            out[name] = int(v)
    return out


def _populate_session_start(session: Session) -> None:
    """If the session has a first Hello, read it and fill start-* fields."""
    for p in session.packets:
        if p.appid == APID_HELLO:
            fields = _read_hello(p.blob, sw_version=session.sw_version)
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

def split_sessions(packets: Sequence[LogicalPacket]) -> List[Session]:
    """Split a sorted, identity-assigned packet stream into sessions.

    Rules (spec section 6.1):
      * A new session begins at a Hello that follows at least one
        non-Hello packet in the current session.
      * Consecutive Hellos fold into the current session.
      * Packets preceding any Hello form session 0 with no startup metadata.
    """
    sessions: List[Session] = []
    current: Optional[Session] = None
    seen_non_hello = False

    for p in packets:
        if p.appid == APID_HELLO:
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
        _populate_session_start(s)
    return sessions


# ---------------------------------------------------------------------------
# Stage 4b: telemetry-to-session assignment
# ---------------------------------------------------------------------------

def assign_telemetry_to_sessions(
    sessions: List[Session],
    fpga_arrays: Dict[str, np.ndarray],
    encoder_arrays: Dict[str, np.ndarray],
    *,
    time_key: str = "mission_seconds",
) -> None:
    """Distribute per-field telemetry arrays across sessions by mission time.

    Each row of ``fpga_arrays[time_key]`` is assigned to the most recent
    preceding session (``bisect_right(starts, t) - 1``, with rows before
    the first session bumped to it). The per-session slices are stored
    on ``session.fpga_telemetry`` / ``session.encoder_telemetry`` as
    sub-dicts of arrays.
    """
    if not sessions:
        return

    indexed = sorted(enumerate(sessions),
                     key=lambda e: e[1].start_raw_seconds or 0.0)
    sorted_starts = np.array(
        [s.start_raw_seconds or 0.0 for _, s in indexed], dtype=np.float64
    )
    sorted_to_orig = [orig for orig, _ in indexed]

    def assign(arrays: Dict[str, np.ndarray], target_attr: str) -> None:
        if not arrays or time_key not in arrays:
            return
        times = np.asarray(arrays[time_key], dtype=np.float64)
        if times.size == 0:
            return
        # bisect_right(starts, t) - 1, with -1 bumped to 0.
        idxs = np.searchsorted(sorted_starts, times, side="right") - 1
        idxs = np.maximum(idxs, 0)
        for sorted_i in range(len(sorted_to_orig)):
            mask = idxs == sorted_i
            if not mask.any():
                continue
            sess = sessions[sorted_to_orig[sorted_i]]
            sess_arrays = {k: v[mask] for k, v in arrays.items()}
            setattr(sess, target_attr, sess_arrays)

    assign(fpga_arrays, "fpga_telemetry")
    assign(encoder_arrays, "encoder_telemetry")


# ---------------------------------------------------------------------------
# Stage 5: persistence
# ---------------------------------------------------------------------------

def _index_width(n_packets: int) -> int:
    return (FILENAME_PACKET_INDEX_DEFAULT_WIDTH
            if n_packets < FILENAME_PACKET_INDEX_WIDE_THRESHOLD
            else 6)


def packet_filename(index: int, appid: int, *, width: int = FILENAME_PACKET_INDEX_DEFAULT_WIDTH) -> str:
    return f"{index:0{width}d}_{appid:0{FILENAME_APID_HEX_WIDTH}x}.bin"


def write_uncrater_session(session: Session, dest_dir: Path | str) -> Path:
    """Write a Session to disk in uncrater session format.

    The directory layout produced is "Layout B" (cdi_output/ subdirectory)
    -- ``dest_dir/cdi_output/NNNNN_XXXX.bin``. ``dest_dir`` is created if
    it does not exist. No telemetry sidecar is written.

    Returns the path of the cdi_output/ subdirectory.
    """
    dest = Path(dest_dir)
    cdi = dest / "cdi_output"
    cdi.mkdir(parents=True, exist_ok=True)

    width = _index_width(len(session.packets))
    for i, p in enumerate(session.packets):
        fn = cdi / packet_filename(i, p.appid, width=width)
        with fn.open("wb") as fh:
            fh.write(p.blob)
    log.info("wrote %d packets to %s", len(session.packets), cdi)
    return cdi
