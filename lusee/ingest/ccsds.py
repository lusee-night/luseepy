"""Stage 1: CCSDS frame recovery.

A bank file is a concatenation of CCSDS Space Packets interleaved with
``0xA5`` padding bytes. Each packet is preceded by the two-byte sync word
``0xECA0`` and a 6-byte primary header, and followed by a CRC-16-CCITT
trailer. This module implements the byte-stream state machine that
extracts validated packets from the raw bank stream and yields them as
``CcsdsFrame`` records.

The CRC behavior is **warn and resync**: a checksum failure causes the
offending packet to be dropped (with a warning) and the parser returns
to sync-word search. This differs from the original LuSEE pipeline,
which aborted on the first CRC failure.
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Iterator

from .constants import (
    CRC_INIT,
    CRC_LEN,
    CRC_POLY,
    PADDING_BYTE,
    PRIMARY_HEADER_LEN,
    SYNC_WORD,
)
from .issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary header
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrimaryHeader:
    """CCSDS Space Packet primary header (6 bytes, big-endian).

    Field meanings follow CCSDS 133.0-B-2; see spec section 3.2.
    """

    version: int        # 3 bits, always 0 in this system
    packet_type: int    # 1 bit, 0 = telemetry, 1 = command
    secheaderflag: int  # 1 bit, secondary-header presence (unused downstream)
    appid: int          # 11 bits, application id
    groupflags: int     # 2 bits, segmentation flag (see collation.py)
    sequence_cnt: int   # 14 bits, mod-16384 transmission counter
    packetlen: int      # 16 bits; data field is (packetlen + 1) bytes long


def parse_primary_header(buf: bytes) -> PrimaryHeader:
    if len(buf) != PRIMARY_HEADER_LEN:
        raise ValueError(f"primary header must be {PRIMARY_HEADER_LEN} bytes")
    h0 = (buf[0] << 8) | buf[1]
    h1 = (buf[2] << 8) | buf[3]
    h2 = (buf[4] << 8) | buf[5]
    return PrimaryHeader(
        version=(h0 >> 13) & 0x7,
        packet_type=(h0 >> 12) & 0x1,
        secheaderflag=(h0 >> 11) & 0x1,
        appid=h0 & 0x7FF,
        groupflags=(h1 >> 14) & 0x3,
        sequence_cnt=h1 & 0x3FFF,
        packetlen=h2,
    )


# ---------------------------------------------------------------------------
# CRC-16-CCITT
# ---------------------------------------------------------------------------

def crc16_ccitt(data: bytes, init: int = CRC_INIT) -> int:
    """CRC-16-CCITT (poly=0x1021, init=0xFFFF, MSB-first, no XorOut).

    See spec section 3.3.
    """
    crc = init
    for b in data:
        crc ^= (b << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


# ---------------------------------------------------------------------------
# Frame record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CcsdsFrame:
    """One validated CCSDS frame extracted from a bank stream."""

    header: PrimaryHeader
    head_bytes: bytes   # the 6-byte primary header bytes (kept for diagnostics)
    payload: bytes      # data field, CRC stripped


@dataclass(frozen=True)
class FrameLocation:
    """Source location of one accepted frame in a diagnostic result."""

    accepted_index: int
    candidate_index: int
    source_offset: int


@dataclass(frozen=True)
class FramingResult:
    """Replayable accepted frames plus source and integrity diagnostics."""

    frames: tuple[CcsdsFrame, ...]
    locations: tuple[FrameLocation, ...]
    issues: tuple[IngestIssue, ...]
    source: str
    input_size_bytes: int
    input_sha256: str

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.locations):
            raise ValueError("frames and locations must have the same length")


@dataclass(frozen=True)
class _AcceptedFrameEvent:
    frame: CcsdsFrame
    location: FrameLocation


@dataclass(frozen=True)
class _FramingIssueEvent:
    code: str
    message: str
    warning_message: str
    emit_warning: bool
    byte_offset: int
    frame_index: int
    appid: int | None = None
    sequence_count: int | None = None
    details: tuple[tuple[str, object], ...] = ()


# ---------------------------------------------------------------------------
# Stream parser
# ---------------------------------------------------------------------------

_FINDING_SYNC = 0
_READING_HEADER = 1
_READING_BODY = 2


def _iter_framing_events(
    stream: bytes,
) -> Iterator[_AcceptedFrameEvent | _FramingIssueEvent]:
    state = _FINDING_SYNC
    sync_window = 0
    previous_search_offset: int | None = None
    head = bytearray()
    body = bytearray()
    packetlen = 0
    hdr: PrimaryHeader | None = None
    source_offset = 0
    candidate_index = 0
    current_candidate_index = 0
    accepted_index = 0

    n = len(stream)
    i = 0
    while i < n:
        v = stream[i]
        i += 1

        if state == _FINDING_SYNC:
            if v == PADDING_BYTE:
                continue
            sync_window = ((sync_window << 8) | v) & 0xFFFF
            if sync_window == SYNC_WORD:
                if previous_search_offset is None:  # pragma: no cover
                    raise AssertionError("sync word has no first-byte offset")
                source_offset = previous_search_offset
                current_candidate_index = candidate_index
                candidate_index += 1
                state = _READING_HEADER
                sync_window = 0
                previous_search_offset = None
                head.clear()
                body.clear()
            else:
                previous_search_offset = i - 1

        elif state == _READING_HEADER:
            head.append(v)
            if len(head) == PRIMARY_HEADER_LEN:
                try:
                    hdr = parse_primary_header(bytes(head))
                except ValueError as exc:
                    yield _FramingIssueEvent(
                        code="framing.malformed_header",
                        message=f"malformed CCSDS header: {exc}",
                        warning_message=f"malformed CCSDS header at byte {i}: {exc}",
                        emit_warning=True,
                        byte_offset=source_offset,
                        frame_index=current_candidate_index,
                        details=(("header_end_offset", i),),
                    )
                    state = _FINDING_SYNC
                    sync_window = 0
                    previous_search_offset = None
                    continue
                packetlen = hdr.packetlen
                state = _READING_BODY

        elif state == _READING_BODY:
            body.append(v)
            if len(body) >= packetlen + 1 + CRC_LEN:
                payload = bytes(body[: packetlen + 1])
                pktcrc = (body[packetlen + 1] << 8) | body[packetlen + 2]
                head_bytes = bytes(head)
                computed = crc16_ccitt(head_bytes + payload)
                if computed == pktcrc:
                    frame = CcsdsFrame(
                        header=hdr,
                        head_bytes=head_bytes,
                        payload=payload,
                    )
                    yield _AcceptedFrameEvent(
                        frame=frame,
                        location=FrameLocation(
                            accepted_index=accepted_index,
                            candidate_index=current_candidate_index,
                            source_offset=source_offset,
                        ),
                    )
                    accepted_index += 1
                else:
                    warning_message = (
                        f"CRC mismatch (apid=0x{hdr.appid:03x}, "
                        f"seq={hdr.sequence_cnt}, got=0x{pktcrc:04x}, "
                        f"computed=0x{computed:04x}); dropping and resyncing"
                    )
                    yield _FramingIssueEvent(
                        code="framing.crc_mismatch",
                        message=warning_message,
                        warning_message=warning_message,
                        emit_warning=True,
                        byte_offset=source_offset,
                        frame_index=current_candidate_index,
                        appid=hdr.appid,
                        sequence_count=hdr.sequence_cnt,
                        details=(
                            ("computed_crc", computed),
                            ("transmitted_crc", pktcrc),
                        ),
                    )
                state = _FINDING_SYNC
                sync_window = 0
                previous_search_offset = None
                head.clear()
                body.clear()

    if state == _READING_HEADER:
        yield _FramingIssueEvent(
            code="framing.truncated_header",
            message="stream ended with a truncated CCSDS primary header",
            warning_message=(
                f"stream ended mid-packet (state={state}); discarding partial"
            ),
            emit_warning=False,
            byte_offset=source_offset,
            frame_index=current_candidate_index,
            details=(
                ("available_header_bytes", len(head)),
                ("expected_header_bytes", PRIMARY_HEADER_LEN),
            ),
        )
    elif state == _READING_BODY:
        if hdr is None:  # pragma: no cover
            raise AssertionError("body state has no parsed header")
        yield _FramingIssueEvent(
            code="framing.truncated_body",
            message="stream ended with a truncated CCSDS packet body",
            warning_message=(
                f"stream ended mid-packet (state={state}); discarding partial"
            ),
            emit_warning=False,
            byte_offset=source_offset,
            frame_index=current_candidate_index,
            appid=hdr.appid,
            sequence_count=hdr.sequence_cnt,
            details=(
                ("available_body_bytes", len(body)),
                ("expected_body_bytes", packetlen + 1 + CRC_LEN),
            ),
        )


def _record_framing_issue(
    event: _FramingIssueEvent,
    *,
    collector: IssueCollector,
    input_identity: str,
    bank: str | None,
) -> IngestIssue:
    return collector.record(
        code=event.code,
        severity=IssueSeverity.WARNING,
        stage="framing",
        message=event.message,
        action=IssueAction.DROPPED,
        input_identity=input_identity,
        bank=bank,
        byte_offset=event.byte_offset,
        frame_index=event.frame_index,
        appid=event.appid,
        sequence_count=event.sequence_count,
        details=dict(event.details),
    )


def _emit_framing_visibility(event: _FramingIssueEvent, source: str) -> None:
    if event.emit_warning:
        warnings.warn(
            f"{source}: {event.warning_message}",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        log.info("%s: %s", source, event.warning_message)


def parse_stream(
    stream: bytes,
    *,
    source: str = "<stream>",
    bank: str | None = None,
    issue_collector: IssueCollector | None = None,
) -> Iterator[CcsdsFrame]:
    """Yield ``CcsdsFrame`` records extracted from a bank byte stream.

    On CRC mismatch, the bad packet is dropped with a warning and the
    parser resyncs to the next ``0xECA0`` boundary. ``source`` is used
    only in warnings / log messages. Supplying an issue collector adds
    structured diagnostics without changing iterator behavior.
    """
    input_identity = ""
    if issue_collector is not None:
        input_identity = f"sha256:{hashlib.sha256(stream).hexdigest()}"
    for event in _iter_framing_events(stream):
        if isinstance(event, _AcceptedFrameEvent):
            yield event.frame
            continue
        if issue_collector is not None:
            _record_framing_issue(
                event,
                collector=issue_collector,
                input_identity=input_identity,
                bank=bank,
            )
        _emit_framing_visibility(event, source)


def parse_stream_diagnostic(
    stream: bytes,
    *,
    source: str = "<stream>",
    bank: str | None = None,
    issue_collector: IssueCollector | None = None,
) -> FramingResult:
    """Return replayable accepted frames and structured framing diagnostics."""
    collector = issue_collector
    if collector is None:
        collector = IssueCollector()
    marker = collector.mark()
    digest = hashlib.sha256(stream).hexdigest()
    input_identity = f"sha256:{digest}"
    frames = []
    locations = []
    for event in _iter_framing_events(stream):
        if isinstance(event, _AcceptedFrameEvent):
            frames.append(event.frame)
            locations.append(event.location)
            continue
        _record_framing_issue(
            event,
            collector=collector,
            input_identity=input_identity,
            bank=bank,
        )
        _emit_framing_visibility(event, source)
    return FramingResult(
        frames=tuple(frames),
        locations=tuple(locations),
        issues=collector.since(marker),
        source=source,
        input_size_bytes=len(stream),
        input_sha256=digest,
    )


def parse_bank_file(
    path,
    *,
    bank: str | None = None,
    issue_collector: IssueCollector | None = None,
) -> Iterator[CcsdsFrame]:
    """Convenience wrapper: read ``path`` and stream frames out of it."""
    from pathlib import Path
    p = Path(path)
    with p.open("rb") as fh:
        data = fh.read()
    yield from parse_stream(
        data,
        source=str(p),
        bank=bank,
        issue_collector=issue_collector,
    )


def parse_bank_file_diagnostic(
    path,
    *,
    bank: str | None = None,
    issue_collector: IssueCollector | None = None,
) -> FramingResult:
    """Read one bank file and return its complete framing result."""
    from pathlib import Path
    p = Path(path)
    with p.open("rb") as fh:
        data = fh.read()
    return parse_stream_diagnostic(
        data,
        source=str(p),
        bank=bank,
        issue_collector=issue_collector,
    )
