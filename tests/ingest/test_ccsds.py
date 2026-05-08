"""Unit tests for CCSDS framing (Stage 1)."""

from __future__ import annotations

import struct

import pytest

from lusee.ingest.ccsds import (
    crc16_ccitt,
    parse_primary_header,
    parse_stream,
)


# ---------------------------------------------------------------------------
# CRC-16-CCITT known-answer vector
# ---------------------------------------------------------------------------

def test_crc16_known_answer_123456789():
    """CRC-16/XMODEM (poly 0x1021, init 0x0000) gives 0x31C3 for "123456789".

    Our implementation has init=0xFFFF (CRC-16/CCITT-FALSE). Verify against
    its known value, 0x29B1, for the same input.
    """
    data = b"123456789"
    assert crc16_ccitt(data) == 0x29B1


def test_crc16_empty():
    assert crc16_ccitt(b"") == 0xFFFF


# ---------------------------------------------------------------------------
# Primary header round-trip
# ---------------------------------------------------------------------------

def _make_header(appid: int, groupflags: int, seq: int, packetlen: int) -> bytes:
    h0 = (0 << 13) | (0 << 12) | (0 << 11) | (appid & 0x7FF)
    h1 = ((groupflags & 0x3) << 14) | (seq & 0x3FFF)
    h2 = packetlen & 0xFFFF
    return struct.pack(">HHH", h0, h1, h2)


def test_parse_primary_header_round_trip():
    raw = _make_header(0x209, 3, 0xABC, 12)
    hdr = parse_primary_header(raw)
    assert hdr.version == 0
    assert hdr.packet_type == 0
    assert hdr.appid == 0x209
    assert hdr.groupflags == 3
    assert hdr.sequence_cnt == 0xABC
    assert hdr.packetlen == 12


def test_parse_primary_header_appid_max():
    raw = _make_header(0x7FF, 1, 0, 0)
    assert parse_primary_header(raw).appid == 0x7FF


def test_parse_primary_header_wrong_size():
    with pytest.raises(ValueError):
        parse_primary_header(b"\x00" * 5)


# ---------------------------------------------------------------------------
# Stream parser end-to-end on a synthetic frame
# ---------------------------------------------------------------------------

def _frame_bytes(appid: int, payload: bytes, *, gf: int = 3, seq: int = 1) -> bytes:
    """Build SYNC || header || payload || crc."""
    if len(payload) < 1:
        raise ValueError("payload must be non-empty")
    head = _make_header(appid, gf, seq, len(payload) - 1)
    crc = crc16_ccitt(head + payload).to_bytes(2, "big")
    return b"\xec\xa0" + head + payload + crc


def test_parse_stream_one_frame():
    payload = bytes(range(8))
    stream = b"\xa5\xa5\xa5" + _frame_bytes(0x209, payload, seq=42) + b"\xa5"
    frames = list(parse_stream(stream))
    assert len(frames) == 1
    f = frames[0]
    assert f.header.appid == 0x209
    assert f.header.sequence_cnt == 42
    assert f.payload == payload


def test_parse_stream_skips_padding_between_frames():
    p1 = bytes([0x10, 0x20, 0x30, 0x40])
    p2 = bytes([0x55] * 6)
    stream = (
        _frame_bytes(0x210, p1, seq=1)
        + b"\xa5" * 7
        + _frame_bytes(0x211, p2, seq=2)
    )
    frames = list(parse_stream(stream))
    assert [f.header.appid for f in frames] == [0x210, 0x211]
    assert frames[0].payload == p1
    assert frames[1].payload == p2


def test_parse_stream_warn_and_resync_on_crc(recwarn):
    payload = bytes([1, 2, 3, 4])
    head = _make_header(0x209, 3, 0, len(payload) - 1)
    bad_crc = b"\xff\xff"   # almost certainly wrong
    bad_frame = b"\xec\xa0" + head + payload + bad_crc

    good_payload = bytes([5, 6, 7, 8])
    good_frame = _frame_bytes(0x209, good_payload, seq=2)

    stream = bad_frame + good_frame
    frames = list(parse_stream(stream))
    # Bad frame is dropped, good one comes through
    assert len(frames) == 1
    assert frames[0].payload == good_payload
    assert any("CRC mismatch" in str(w.message) for w in recwarn.list)
