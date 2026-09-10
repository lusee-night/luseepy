from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lusee.ingest import decode
from lusee.ingest.decode import HKSample, Products
from lusee.ingest.packet_map import (
    PACKET_MAP_FILENAME,
    PacketMapError,
    build_packet_map,
    read_packet_map,
    write_packet_map,
)
from lusee.ingest.products import ProductProvenance, SourcePacketProvenance
from lusee.ingest.reassembly import LogicalPacket


def normalize_appid(appid: int) -> int:
    return 0x2F0 if appid == 0x4F0 else appid


def packet(*, appid=0x4F0, blob=b"packet", uid=7, bank="b05"):
    return LogicalPacket(
        appid=appid,
        start_seq=3,
        seq=5,
        blob=blob,
        single_packet=False,
        unique_packet_id=uid,
        bank=bank,
    )


def write_tree(tmp_path):
    session = tmp_path / "session"
    cdi = session / "cdi_output"
    cdi.mkdir(parents=True)
    filename = "00000_04f0.bin"
    logical_packet = packet()
    (cdi / filename).write_bytes(logical_packet.blob)
    packet_map = build_packet_map(
        [logical_packet],
        [filename],
        normalize_appid=normalize_appid,
    )
    write_packet_map(packet_map, session / PACKET_MAP_FILENAME)
    return session, cdi, packet_map


def test_packet_map_round_trip_validates_inventory_and_normalized_appid(tmp_path):
    session, cdi, expected = write_tree(tmp_path)

    loaded = read_packet_map(session, cdi, normalize_appid=normalize_appid)

    assert loaded == expected
    entry = loaded.entries[0]
    assert entry.original_appid == 0x4F0
    assert entry.normalized_appid == 0x2F0
    assert entry.source_bank == "b05"
    assert entry.terminal_groupflag == 1


@pytest.mark.parametrize("damage", ["hash", "inventory", "schema"])
def test_present_packet_map_damage_fails_closed(tmp_path, damage):
    session, cdi, _packet_map = write_tree(tmp_path)
    if damage == "hash":
        (cdi / "00000_04f0.bin").write_bytes(b"changed")
    elif damage == "inventory":
        (cdi / "00001_020f.bin").write_bytes(b"extra")
    else:
        path = session / PACKET_MAP_FILENAME
        document = json.loads(path.read_text("ascii"))
        document["format_version"] = 3
        path.write_text(json.dumps(document), encoding="ascii")

    with pytest.raises(PacketMapError):
        read_packet_map(session, cdi, normalize_appid=normalize_appid)


@pytest.mark.parametrize(
    "damage",
    ["hash", "missing", "extra", "packet_symlink", "map_symlink"],
)
def test_invalid_map_is_rejected_before_collection_construction(
    tmp_path,
    monkeypatch,
    damage,
):
    session, cdi, _packet_map = write_tree(tmp_path)
    packet_path = cdi / "00000_04f0.bin"
    if damage == "hash":
        packet_path.write_bytes(b"changed")
    elif damage == "missing":
        packet_path.unlink()
    elif damage == "extra":
        (cdi / "00001_020f.bin").write_bytes(b"extra")
    elif damage == "packet_symlink":
        target = tmp_path / "packet-target.bin"
        target.write_bytes(packet_path.read_bytes())
        packet_path.unlink()
        packet_path.symlink_to(target)
    else:
        map_path = session / PACKET_MAP_FILENAME
        map_path.unlink()
        map_path.symlink_to(tmp_path / "missing-packet-map.json")
    monkeypatch.setattr(
        decode,
        "load_uncrater",
        lambda: SimpleNamespace(normalize_dcb_appid=normalize_appid),
    )

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("Collection must not be constructed")

    monkeypatch.setattr(decode, "make_collection", unexpected_collection)

    with pytest.raises(PacketMapError):
        decode.read_uncrater_session(session)


def test_duplicate_json_keys_are_rejected(tmp_path):
    session, cdi, _packet_map = write_tree(tmp_path)
    path = session / PACKET_MAP_FILENAME
    path.write_text(
        path.read_text("ascii").replace(
            '"format_version": 2,',
            '"format_version": 2, "format_version": 2,',
            1,
        ),
        encoding="ascii",
    )

    with pytest.raises(PacketMapError, match="duplicate key 'format_version'"):
        read_packet_map(session, cdi, normalize_appid=normalize_appid)


def test_packet_map_postprocess_enriches_only_available_source_fields():
    packet_map = build_packet_map(
        [packet(appid=0x206, uid=9, bank="b07")],
        ["00000_0206.bin"],
        normalize_appid=normalize_appid,
    )
    source = SourcePacketProvenance(
        role="housekeeping",
        packet_index=0,
        original_appid=0x206,
        normalized_appid=0x206,
    )
    provenance = ProductProvenance(
        source_packets=(source,),
        uid=9,
        uid_source="Packet_Housekeep.unique_packet_id",
        uid_source_role="housekeeping",
        selected_schema_id=0x307,
    )
    row = HKSample(
        hk_type=0,
        version=0x307,
        unique_packet_id=9,
        errors=0,
        fields={"sentinel": np.int32(1)},
        provenance=provenance,
    )
    products = Products(
        housekeeping=[row],
        packet_map_status="verified",
        packet_map_format_version=2,
        raw_flash_provenance_unavailable_reason=None,
    )

    decode._enrich_products_from_packet_map(products, packet_map)

    enriched = products.housekeeping[0].provenance.source_packets[0]
    assert enriched.filename == "00000_0206.bin"
    assert enriched.bank == "b07"
    assert enriched.original_appid == 0x206
    assert enriched.normalized_appid == 0x206
    assert enriched.frame_start is None
    assert enriched.frame_stop is None
    assert enriched.byte_offset_start is None
    assert enriched.byte_offset_stop is None

    mismatched_map = replace(
        packet_map,
        entries=(replace(packet_map.entries[0], unique_packet_id=10),),
    )
    mismatched_products = Products(
        housekeeping=[row],
        packet_map_status="verified",
        packet_map_format_version=2,
        raw_flash_provenance_unavailable_reason=None,
    )
    with pytest.raises(PacketMapError, match="product UID disagrees"):
        decode._enrich_products_from_packet_map(
            mismatched_products,
            mismatched_map,
        )


def test_products_record_legacy_packet_map_absence_without_an_issue():
    products = Products()

    assert products.packet_map_status == "unavailable"
    assert products.packet_map_format_version is None
    assert (
        products.raw_flash_provenance_unavailable_reason
        == "packet_map_not_loaded"
    )
    assert products.issues == ()
