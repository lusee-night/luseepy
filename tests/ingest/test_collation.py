"""Unit tests for Stage 2 reassembly and Stage 3 identity assignment."""

from __future__ import annotations

import struct

import pytest

from lusee.ingest.ccsds import CcsdsFrame, PrimaryHeader
from lusee.ingest.collation import (
    APID_HELLO,
    APID_RAW_ADC,
    APID_SPECTRA_HIGH,
    LogicalPacket,
    assign_identities,
    is_uid_derived,
    is_uid_prefixed,
    is_uid_typed,
    reassemble_logical_packets,
)


def _frame(appid: int, payload: bytes, *, gf: int = 3, seq: int = 0) -> CcsdsFrame:
    hdr = PrimaryHeader(
        version=0, packet_type=0, secheaderflag=0,
        appid=appid, groupflags=gf, sequence_cnt=seq,
        packetlen=max(0, len(payload) - 1),
    )
    return CcsdsFrame(header=hdr, head_bytes=b"\x00" * 6, payload=payload)


def test_byteswap_unsegmented_science():
    payload = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    out = list(reassemble_logical_packets([_frame(APID_SPECTRA_HIGH, payload)],
                                           byteswap_pairs=True))
    assert len(out) == 1
    assert out[0].blob == bytes([0xBB, 0xAA, 0xDD, 0xCC])
    assert out[0].appid == APID_SPECTRA_HIGH
    assert out[0].single_packet is True


def test_no_byteswap_telemetry():
    payload = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    out = list(reassemble_logical_packets([_frame(0x314, payload)],
                                           byteswap_pairs=False))
    assert out[0].blob == payload


def test_segmented_terminates_on_groupflags_1():
    p2 = bytes([0xAA, 0xBB])
    p0a = bytes([0xCC, 0xDD])
    p1 = bytes([0xEE, 0xFF])
    frames = [
        _frame(APID_SPECTRA_HIGH, p2, gf=2, seq=10),
        _frame(APID_SPECTRA_HIGH, p0a, gf=0, seq=11),
        _frame(APID_SPECTRA_HIGH, p1, gf=1, seq=12),
    ]
    out = list(reassemble_logical_packets(frames, byteswap_pairs=True))
    assert len(out) == 1
    assert out[0].single_packet is False
    # Each pair byte-swapped independently then concatenated
    assert out[0].blob == bytes([0xBB, 0xAA, 0xDD, 0xCC, 0xFF, 0xEE])
    assert out[0].start_seq == 10
    assert out[0].seq == 12


def test_uid_source_predicates():
    assert is_uid_prefixed(APID_SPECTRA_HIGH)
    assert is_uid_prefixed(APID_SPECTRA_HIGH + 7)
    assert not is_uid_prefixed(APID_HELLO)
    assert is_uid_typed(APID_HELLO)
    assert not is_uid_derived(APID_HELLO)
    assert is_uid_derived(APID_RAW_ADC)
    assert is_uid_derived(APID_RAW_ADC + 3)
    assert not is_uid_derived(APID_RAW_ADC + 4)


def test_assign_identities_uid_prefixed():
    upid = 0x12345678
    blob = upid.to_bytes(4, "little") + bytes(8)
    p = LogicalPacket(appid=APID_SPECTRA_HIGH, start_seq=0, seq=0,
                      blob=blob, single_packet=True)
    out = assign_identities([p], sw_version=None,
                            typed_uid_extractor=lambda *a, **kw: None)
    assert len(out) == 1
    assert out[0].unique_packet_id == upid


def test_assign_identities_uid_derived():
    parent = LogicalPacket(
        appid=APID_SPECTRA_HIGH, start_seq=0, seq=0,
        blob=(0x42).to_bytes(4, "little") + b"\x00" * 8,
        single_packet=True,
    )
    child = LogicalPacket(
        appid=APID_RAW_ADC, start_seq=1, seq=1,
        blob=b"\x11" * 16, single_packet=True,
    )
    orphan = LogicalPacket(
        appid=APID_RAW_ADC, start_seq=2, seq=2,
        blob=b"\x22" * 16, single_packet=True,
    )
    # parent + child inherit; orphan placed before parent has no preceding id
    out = assign_identities([orphan, parent, child], sw_version=None,
                            typed_uid_extractor=lambda *a, **kw: None)
    upids = {p.appid: p.unique_packet_id for p in out}
    assert upids.get(APID_SPECTRA_HIGH) == 0x42
    # Orphan should be dropped; child kept and inherited
    raw_appids = [p.appid for p in out]
    assert APID_RAW_ADC in raw_appids
    assert raw_appids.count(APID_RAW_ADC) == 1


def test_assign_identities_drops_heartbeat():
    from lusee.ingest.collation import APID_HEARTBEAT
    p = LogicalPacket(appid=APID_HEARTBEAT, start_seq=0, seq=0,
                      blob=b"\x00" * 16, single_packet=True)
    out = assign_identities([p], sw_version=None,
                            typed_uid_extractor=lambda *a, **kw: 999)
    assert out == []
