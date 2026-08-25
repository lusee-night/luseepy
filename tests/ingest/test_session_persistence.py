from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from lusee.ingest import session as session_mod
from lusee.ingest.reassembly import LogicalPacket
from lusee.ingest.session import Session, write_uncrater_session


def make_packet(appid: int, blob: bytes, index: int) -> LogicalPacket:
    return LogicalPacket(
        appid=appid,
        start_seq=index,
        seq=index,
        blob=blob,
        single_packet=True,
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


def staging_paths(dest: Path) -> list[Path]:
    return sorted(dest.parent.glob(f".{dest.name}.*.tmp"))


def test_write_uncrater_session_atomically_installs_complete_tree(
    tmp_path,
    monkeypatch,
):
    dest = tmp_path / "nested" / "session"
    observed = []
    real_rename = session_mod._rename_noreplace

    def inspect_rename(source, target):
        source = Path(source)
        target = Path(target)
        assert source.parent.parent == dest.parent
        assert target == dest
        assert not target.exists()
        assert sorted(path.name for path in (source / "cdi_output").iterdir()) == [
            "00000_0001.bin",
            "00001_020f.bin",
            "00002_07ff.bin",
        ]
        observed.append((source, target))
        real_rename(source, target)

    monkeypatch.setattr(session_mod, "_rename_noreplace", inspect_rename)

    cdi = write_uncrater_session(make_session(), dest)

    assert cdi == dest / "cdi_output"
    assert observed and observed[0][1] == dest
    assert (cdi / "00000_0001.bin").read_bytes() == b"first"
    assert (cdi / "00001_020f.bin").read_bytes() == b"second"
    assert (cdi / "00002_07ff.bin").read_bytes() == b"third"
    assert list(dest.iterdir()) == [cdi]
    assert staging_paths(dest) == []


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
    assert staging_paths(dest) == []


def test_write_uncrater_session_cleans_staging_after_install_failure(
    tmp_path,
    monkeypatch,
):
    dest = tmp_path / "session"

    def fail_rename(source, target):
        raise OSError("injected install failure")

    monkeypatch.setattr(session_mod, "_rename_noreplace", fail_rename)

    with pytest.raises(OSError, match="injected install failure"):
        write_uncrater_session(make_session(), dest)

    assert not dest.exists()
    assert staging_paths(dest) == []


def test_write_uncrater_session_does_not_replace_raced_destination(
    tmp_path,
    monkeypatch,
):
    dest = tmp_path / "session"
    real_rename = session_mod._rename_noreplace

    def race_rename(source, target):
        target.mkdir()
        real_rename(source, target)

    monkeypatch.setattr(session_mod, "_rename_noreplace", race_rename)

    with pytest.raises(FileExistsError):
        write_uncrater_session(make_session(), dest)

    assert dest.is_dir()
    assert list(dest.iterdir()) == []
    assert staging_paths(dest) == []


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
