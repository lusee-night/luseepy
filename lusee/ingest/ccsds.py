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


# ---------------------------------------------------------------------------
# Stream parser
# ---------------------------------------------------------------------------

_FINDING_SYNC = 0
_READING_HEADER = 1
_READING_BODY = 2


def parse_stream(stream: bytes, *, source: str = "<stream>") -> Iterator[CcsdsFrame]:
    """Yield ``CcsdsFrame`` records extracted from a bank byte stream.

    On CRC mismatch, the bad packet is dropped with a warning and the
    parser resyncs to the next ``0xECA0`` boundary. ``source`` is used
    only in warnings / log messages.
    """
    state = _FINDING_SYNC
    sync_window = 0
    head = bytearray()
    body = bytearray()
    packetlen = 0

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
                state = _READING_HEADER
                sync_window = 0
                head.clear()
                body.clear()

        elif state == _READING_HEADER:
            head.append(v)
            if len(head) == PRIMARY_HEADER_LEN:
                try:
                    hdr = parse_primary_header(bytes(head))
                except ValueError as exc:
                    warnings.warn(
                        f"{source}: malformed CCSDS header at byte {i}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    state = _FINDING_SYNC
                    sync_window = 0
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
                    yield CcsdsFrame(header=hdr, head_bytes=head_bytes, payload=payload)
                else:
                    warnings.warn(
                        f"{source}: CRC mismatch (apid=0x{hdr.appid:03x}, "
                        f"seq={hdr.sequence_cnt}, got=0x{pktcrc:04x}, "
                        f"computed=0x{computed:04x}); dropping and resyncing",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                state = _FINDING_SYNC
                sync_window = 0
                head.clear()
                body.clear()

    if state != _FINDING_SYNC:
        # Truncated tail: warn but do not raise, real captures sometimes end mid-frame.
        log.info("%s: stream ended mid-packet (state=%d); discarding partial", source, state)


def parse_bank_file(path) -> Iterator[CcsdsFrame]:
    """Convenience wrapper: read ``path`` and stream frames out of it."""
    from pathlib import Path
    p = Path(path)
    with p.open("rb") as fh:
        data = fh.read()
    yield from parse_stream(data, source=str(p))
