from __future__ import annotations

import hashlib
import json
import os
import stat

import pytest

from lusee.ingest.reassembly import LogicalPacket
from lusee.ingest.session import Session, write_uncrater_session


def make_packet(appid: int, blob: bytes, index: int) -> LogicalPacket:
    return LogicalPacket(
        appid=appid,
        start_seq=index,
        seq=index,
        blob=blob,
        single_packet=True,
        unique_packet_id=100 + index,
        bank="b05",
    )


def make_session() -> Session:
    return Session(
        ordinal=0,
        packets=[
            make_packet(0x001, b"first", 0),
            make_packet(0x20F, b"second", 1),
            make_packet(0x7FF, b"third", 2),
        ],
    )


def test_write_uncrater_session_writes_complete_tree(tmp_path):
    dest = tmp_path / "nested" / "session"

    cdi = write_uncrater_session(make_session(), dest)

    assert cdi == dest / "cdi_output"
    assert (cdi / "00000_0001.bin").read_bytes() == b"first"
    assert (cdi / "00001_020f.bin").read_bytes() == b"second"
    assert (cdi / "00002_07ff.bin").read_bytes() == b"third"
    assert sorted(path.name for path in dest.iterdir()) == [
        "cdi_output",
        "packet_map.json",
    ]


def test_packet_map_records_only_retained_packet_provenance(tmp_path):
    session = make_session()
    session.packets[0].single_packet = False
    cdi = write_uncrater_session(session, tmp_path / "session")

    document = json.loads((cdi.parent / "packet_map.json").read_text("ascii"))

    assert document["format_version"] == 2
    assert document["reassembly_profile"] == "legacy"
    assert document["packet_order"] == {
        "chronological": False,
        "key": ["ordering_uid", "last_sequence_count"],
        "kind": "uid_sequence_heuristic",
    }
    assert set(document["provenance_limits"]) == {
        "contributing_frame_byte_offsets",
        "contributing_frame_flags",
        "contributing_frame_ordinals",
        "packet_issue_references",
        "uid_source",
        "cross_bank_boundary_placement",
        "undetectable_packet_loss",
    }
    first = document["packets"][0]
    assert first == {
        "content_sha256": hashlib.sha256(b"first").hexdigest(),
        "last_sequence_count": 0,
        "normalized_appid": 0x001,
        "original_appid": 0x001,
        "output_filename": "00000_0001.bin",
        "output_index": 0,
        "source_bank": "b05",
        "source_order": 0,
        "waveform_metadata_source_order": None,
        "start_sequence_count": 0,
        "terminal_groupflag": 1,
        "unavailable_fields": {},
        "unique_packet_id": 100,
    }


def test_write_uncrater_session_refuses_and_preserves_existing_destination(
    tmp_path,
):
    dest = tmp_path / "session"
    old_cdi = dest / "cdi_output"
    old_cdi.mkdir(parents=True)
    sentinel = old_cdi / "99999_0001.bin"
    sentinel.write_bytes(b"old data")

    with pytest.raises(FileExistsError) as caught:
        write_uncrater_session(make_session(), dest)

    assert caught.value.args == (dest,)
    assert sentinel.read_bytes() == b"old data"
    assert sorted(path.name for path in old_cdi.iterdir()) == [sentinel.name]


def test_write_uncrater_session_explicitly_replaces_complete_tree(tmp_path):
    dest = tmp_path / "session"
    old_cdi = dest / "cdi_output"
    old_cdi.mkdir(parents=True)
    stale = old_cdi / "99999_0001.bin"
    stale.write_bytes(b"stale data")
    (dest / "stale-sidecar.json").write_text("stale", encoding="ascii")

    cdi = write_uncrater_session(make_session(), dest, overwrite=True)

    assert cdi == dest / "cdi_output"
    assert sorted(path.name for path in dest.iterdir()) == [
        "cdi_output",
        "packet_map.json",
    ]
    assert sorted(path.name for path in cdi.iterdir()) == [
        "00000_0001.bin",
        "00001_020f.bin",
        "00002_07ff.bin",
    ]
    assert not stale.exists()


@pytest.mark.parametrize("overwrite", [None, 0, 1, "true"])
def test_write_uncrater_session_requires_exact_bool_overwrite(
    tmp_path,
    overwrite,
):
    dest = tmp_path / "session"

    with pytest.raises(TypeError, match="overwrite must be bool"):
        write_uncrater_session(make_session(), dest, overwrite=overwrite)

    assert not dest.exists()


def test_write_uncrater_session_preserves_mkdir_umask_mode(tmp_path):
    dest = tmp_path / "session"
    previous_umask = os.umask(0o022)
    try:
        write_uncrater_session(make_session(), dest)
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(dest.stat().st_mode) == 0o755


def test_write_uncrater_session_filenames_are_deterministic(tmp_path):
    first = write_uncrater_session(make_session(), tmp_path / "session_a")
    second = write_uncrater_session(make_session(), tmp_path / "session_b")

    first_names = sorted(path.name for path in first.iterdir())
    second_names = sorted(path.name for path in second.iterdir())
    assert first_names == second_names == [
        "00000_0001.bin",
        "00001_020f.bin",
        "00002_07ff.bin",
    ]
    assert (first.parent / "packet_map.json").read_bytes() == (
        second.parent / "packet_map.json"
    ).read_bytes()
