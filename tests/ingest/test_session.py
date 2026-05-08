"""Unit tests for session-time decoding and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from lusee.ingest.session import (
    Session,
    packet_filename,
    raw_seconds_from_split_time,
    write_uncrater_session,
)
from lusee.ingest.collation import APID_HELLO, APID_SPECTRA_HIGH, LogicalPacket


def test_raw_seconds_zero():
    assert raw_seconds_from_split_time(0, 0) == 0.0


def test_raw_seconds_known():
    # ((time_16 << 32) + time_32) >> 4 / 4096
    # take time_32 = 0x10000 (=65536), time_16 = 0
    # combined = 65536 ; shift 4 -> 4096 ; / 4096 -> 1.0 second
    assert raw_seconds_from_split_time(0x10000, 0) == 1.0


def test_packet_filename_padding():
    assert packet_filename(0, 0x209) == "00000_0209.bin"
    assert packet_filename(42, 0x2FA) == "00042_02fa.bin"
    assert packet_filename(0, 0x2F0, width=6) == "000000_02f0.bin"


def test_write_uncrater_session(tmp_path: Path):
    sess = Session(ordinal=0)
    for i in range(3):
        sess.packets.append(LogicalPacket(
            appid=APID_HELLO if i == 0 else APID_SPECTRA_HIGH,
            start_seq=i, seq=i,
            blob=bytes([i, i + 1, i + 2, i + 3]),
            single_packet=True,
            unique_packet_id=100 + i,
        ))
    cdi = write_uncrater_session(sess, tmp_path)
    assert cdi == tmp_path / "cdi_output"
    files = sorted(p.name for p in cdi.iterdir())
    assert files == ["00000_0209.bin", "00001_0210.bin", "00002_0210.bin"]
    assert (cdi / "00000_0209.bin").read_bytes() == bytes([0, 1, 2, 3])
