"""Stages 4-5: session splitting and uncrater-session persistence.

A session is the run of logical packets between two Hello (startup)
packets. This module splits the sorted, identity-assigned stream into
``Session`` objects, decodes their start times from the first Hello's
mission-time fields, assigns decoded DCB rows to disjoint session windows,
and writes each session as ``NNNNN_XXXX.bin`` files.
"""

from __future__ import annotations

import logging
import shutil
import warnings
from tempfile import TemporaryDirectory
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
    build_packet_map,
    write_packet_map,
)
from .reassembly import LogicalPacket
from .telemetry import TelemetryData, map_dcb_absolute_time
from .uncrater_adapter import load_uncrater, read_packet, packet_schema_options, schema_record

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
    telemetry: TelemetryData | None = None
    schema_resolution: object | None = None

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
    schema_resolution=None,
) -> Optional[dict]:
    """Decode a Hello packet's identity / time fields. Returns None on failure."""
    decoder = load_uncrater()
    try:
        pkt = decoder.Packet(
            int(decoder.id.AppID_uC_Start), blob=blob, version=sw_version,
            **packet_schema_options(schema_resolution),
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
                schema_resolution=session.schema_resolution,
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
    schema_variant: str | None = None,
    schema_resolution=None,
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
        s.schema_resolution = schema_resolution
        _populate_session_start(s, issue_collector=issue_collector)
    associate_session_waveforms(sessions, schema_variant=schema_variant)
    return sessions


def waveform_context(packets):
    """Original bank order, optionally with complete-input metadata decisions."""
    by_source = {packet.file_index: index for index, packet in enumerate(packets)}
    checked = any(packet.waveform_association_checked for packet in packets)
    decoder = load_uncrater()
    if checked and any(decoder.appid_is_raw_adc(packet.appid)
                       and not packet.waveform_association_checked for packet in packets):
        raise ValueError("waveform associations must cover the complete input")
    context = {}
    for index, packet in enumerate(packets):
        entry = {"order": index if packet.file_index is None else packet.file_index,
                 "stream": packet.bank or "",
                 "start_sequence_count": packet.start_seq,
                 "last_sequence_count": packet.seq}
        if checked:
            target = packet.waveform_metadata_source_order
            entry["metadata_packet_index"] = None if target is None else by_source[target]
        context[index] = entry
    return context


def mark_waveform_transport_loss(packets, issues):
    """Do not infer a capture mode from a bank with observed relevant loss."""
    decoder = load_uncrater()
    uncertain_banks = set()
    for issue in issues:
        if issue.stage == "framing":
            # A failed CRC does not establish the header's AppID either
            uncertain_banks.add(issue.bank)
        elif issue.stage == "reassembly":
            appids = (issue.appid, dict(issue.details).get("new_appid", issue.appid))
            if any(appid is None or decoder.appid_is_raw_adc(appid)
                   or decoder.appid_is_raw_adc_metadata(appid) for appid in appids):
                uncertain_banks.add(issue.bank)
    for packet in packets:
        if decoder.appid_is_raw_adc(packet.appid):
            packet.waveform_transport_uncertain = (
                None in uncertain_banks or packet.bank in uncertain_banks
            )


def associate_session_waveforms(sessions, *, schema_variant=None):
    """Route invariant full-input matches to their metadata's session.

    Bank concatenation is not chronology. Only same-bank Hello/EOS positions
    delimit waveform matching. Unresolved packets retain their heuristic
    session placement and an explicit null metadata reference.
    """
    decoder = load_uncrater()
    packets = [packet for session in sessions for packet in session.packets]
    if not any(decoder.appid_is_raw_adc(packet.appid) for packet in packets):
        return
    for index, packet in enumerate(packets):
        if packet.file_index is None:
            packet.file_index = index
    # Include all schema evidence, but avoid copying large spectra into this pass
    selected = [packet for packet in packets if (
        decoder.appid_is_raw_adc(packet.appid)
        or decoder.appid_is_raw_adc_metadata(packet.appid)
        or decoder.appid_is_hello(packet.appid)
        or packet.appid == int(decoder.id.AppID_End_Of_Sequence)
        or decoder.appid_is_housekeeping(packet.appid)
        or decoder.appid_is_metadata(packet.appid)
        or decoder.appid_is_cal_metadata(packet.appid)
    )]
    with TemporaryDirectory(prefix="lusee-waveform-association-") as directory:
        for index, packet in enumerate(selected):
            (Path(directory) / packet_filename(index, packet.appid)).write_bytes(packet.blob)
        collection = decoder.Collection(directory, schema_variant=schema_variant,
                                        schema_resolution=sessions[0].schema_resolution,
                                        waveform_packet_context=waveform_context(selected))
    targets = {
        selected[packet.packet_index].file_index: selected[group["meta"].packet_index].file_index
        for group in collection.waveform_groups for packet in group["packets"].values()
        if not selected[packet.packet_index].waveform_transport_uncertain
    }
    owners = {packet.file_index: session for session in sessions for packet in session.packets}
    for session in sessions:
        session.packets = []
    for packet in packets:
        if decoder.appid_is_raw_adc(packet.appid):
            packet.waveform_association_checked = True
            packet.waveform_metadata_source_order = targets.get(packet.file_index)
        owner = owners[targets.get(packet.file_index, packet.file_index)]
        owner.packets.append(packet)
    sessions[:] = [session for session in sessions if session.packets]
    for ordinal, session in enumerate(sessions):
        session.ordinal = ordinal


# ---------------------------------------------------------------------------
# Stage 4b: telemetry-to-session assignment
# ---------------------------------------------------------------------------

def assign_telemetry_to_sessions(
    sessions: List[Session],
    telemetry: TelemetryData,
    *,
    clock_reference_set: ClockReferenceSet,
) -> TelemetryData | None:
    """Map b01 time and assign every usable row to one science session."""
    if not isinstance(telemetry, TelemetryData):
        raise TypeError("telemetry must be TelemetryData")
    if not isinstance(clock_reference_set, ClockReferenceSet):
        raise TypeError("clock_reference_set must be ClockReferenceSet")
    if telemetry.source_kind != "b01_0x314":
        raise ValueError("session assignment accepts only b01 telemetry")

    empty = np.zeros(telemetry.row_count, dtype=np.bool_)
    if telemetry.row_count == 0:
        for session in sessions:
            session.telemetry = telemetry.slice_rows(empty)
        return telemetry

    spectrometer_reference = clock_reference_set.reference_for(
        ClockSource.SPECTROMETER
    )
    dcb_reference = clock_reference_set.reference_for(ClockSource.DCB)
    missing_clocks = [
        name
        for name, reference in (
            (ClockSource.SPECTROMETER.value, spectrometer_reference),
            (ClockSource.DCB.value, dcb_reference),
        )
        if reference is None
    ]
    if missing_clocks:
        warnings.warn(
            "b01 telemetry omitted because the following clock reference(s) "
            f"are missing: {', '.join(missing_clocks)}",
            stacklevel=2,
        )
        return None

    if not sessions:
        warnings.warn(
            "b01 telemetry omitted because there are no science sessions",
            stacklevel=2,
        )
        return None
    ordinals = [session.ordinal for session in sessions]
    if len(set(ordinals)) != len(ordinals):
        warnings.warn("b01 telemetry omitted: session ordinals repeat", stacklevel=2)
        return None
    ordered_sessions = sorted(sessions, key=lambda session: session.ordinal)
    starts = [session.start_raw_seconds for session in ordered_sessions]
    if any(start is None for start in starts):
        warnings.warn(
            "b01 telemetry omitted because a session start is missing",
            stacklevel=2,
        )
        return None
    start_values = np.asarray(starts, dtype=np.float64)
    if not np.all(np.isfinite(start_values)) or (
        start_values.size > 1 and np.any(np.diff(start_values) <= 0.0)
    ):
        warnings.warn(
            "b01 telemetry omitted because session starts are not monotone",
            stacklevel=2,
        )
        return None

    assert spectrometer_reference is not None
    assert dcb_reference is not None
    mapped = map_dcb_absolute_time(
        telemetry,
        clock_reference_set=clock_reference_set,
    )
    session_elapsed = (
        start_values - spectrometer_reference.clock_reference_raw_seconds
    )
    telemetry_elapsed = (
        mapped.raw_seconds - dcb_reference.clock_reference_raw_seconds
    )
    assignments = np.searchsorted(
        session_elapsed,
        telemetry_elapsed,
        side="right",
    ) - 1
    pre_session = assignments < 0
    if np.any(pre_session):
        warnings.warn(
            f"{int(np.count_nonzero(pre_session))} b01 telemetry row(s) before "
            "the first science session were omitted",
            stacklevel=2,
        )
    for index, session in enumerate(ordered_sessions):
        session.telemetry = mapped.slice_rows(assignments == index)
    return mapped


# ---------------------------------------------------------------------------
# Stage 5: persistence
# ---------------------------------------------------------------------------

def _index_width(n_packets: int) -> int:
    return (FILENAME_PACKET_INDEX_DEFAULT_WIDTH
            if n_packets < FILENAME_PACKET_INDEX_WIDE_THRESHOLD
            else 6)


def packet_filename(index: int, appid: int, *, width: int = FILENAME_PACKET_INDEX_DEFAULT_WIDTH) -> str:
    return f"{index:0{width}d}_{appid:0{FILENAME_APID_HEX_WIDTH}x}.bin"


def write_uncrater_session(
    session: Session,
    dest_dir: Path | str,
    *,
    overwrite: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
) -> Path:
    """Write a Session to disk in uncrater session format.

    The directory layout produced is "Layout B" (cdi_output/ subdirectory)
    -- ``dest_dir/cdi_output/NNNNN_XXXX.bin`` plus a versioned
    ``dest_dir/packet_map.json``. An existing ``dest_dir`` is refused unless
    ``overwrite=True``. No telemetry sidecar is written.

    Returns the path of the cdi_output/ subdirectory.
    """
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be bool")
    dest = Path(dest_dir)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(dest)
        shutil.rmtree(dest)
    width = _index_width(len(session.packets))
    filenames = [
        packet_filename(index, packet.appid, width=width)
        for index, packet in enumerate(session.packets)
    ]
    decoder = load_uncrater()
    cdi = dest / "cdi_output"
    cdi.mkdir(parents=True)
    for packet, filename in zip(session.packets, filenames):
        with (cdi / filename).open("wb") as output:
            output.write(packet.blob)
    if any(decoder.appid_is_raw_adc(packet.appid) for packet in session.packets):
        for index, packet in enumerate(session.packets):
            if packet.file_index is None:
                packet.file_index = index
        collection = decoder.Collection(
            str(cdi), waveform_packet_context=waveform_context(session.packets),
            diagnostic_override=diagnostic_override, schema_variant=schema_variant,
            schema_resolution=session.schema_resolution,
        )
        waveform_targets = {
            packet.packet_index: session.packets[group["meta"].packet_index].file_index
            for group in collection.waveform_groups
            for packet in group["packets"].values()
        }
        waveform_uids = {
            packet.packet_index: group["meta"].unique_packet_id
            for group in collection.waveform_groups
            for packet in group["packets"].values()
        }
        # Repair identities after ordering; never sort by these repaired UIDs
        for index, packet in enumerate(session.packets):
            if decoder.appid_is_raw_adc(packet.appid):
                packet.unique_packet_id = waveform_uids.get(index, 0)
                packet.waveform_association_checked = True
                packet.waveform_metadata_source_order = waveform_targets.get(index)
    packet_map = build_packet_map(
        session.packets,
        filenames,
        normalize_appid=decoder.normalize_dcb_appid,
        schema_resolution=schema_record(session.schema_resolution),
    )
    write_packet_map(packet_map, dest / PACKET_MAP_FILENAME)
    log.info("wrote %d packets to %s", len(session.packets), cdi)
    return cdi
