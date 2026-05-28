"""End-to-end test: process_flash writes an in-session manifest,
process_session re-derives telemetry from the flash backreference.

Skipped when uncrater isn't importable or the FLASH_TLMFS fixture
isn't present.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path


import numpy as np
import pytest


pytest.importorskip("h5py")
pytest.importorskip("uncrater")

# Resolve the FLASH_TLMFS fixture; skip the whole module when absent so
# this test doesn't break dev environments that don't have data.
_FIXTURE = Path(
    "/Users/anigmetov/code/lusee_night/luseepy/ingest_playground/new_data/"
    "20251105_112220/fs/FLASH_TLMFS"
)
if not _FIXTURE.is_dir():
    pytest.skip("FLASH_TLMFS fixture not available", allow_module_level=True)


def _bytes_in_dataset(h5: h5py.File, path: str):
    if path in h5:
        return h5[path][...]
    return None


def test_process_flash_writes_in_session_manifest(tmp_path: Path):
    from lusee.ingest import process_flash

    sessions_root = tmp_path / "sessions"
    h5_dir = tmp_path / "h5"
    results = process_flash(
        _FIXTURE,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
        manifest_dir=h5_dir,
    )
    assert len(results) >= 1, "expected at least one session"
    for r in results:
        sd = Path(r.session_dir)
        manifest_path = sd / "session.json"
        assert manifest_path.is_file(), f"missing in-session manifest at {manifest_path}"
        with manifest_path.open() as fh:
            body = json.load(fh)
        # The four new fields must be populated on the flash path.
        assert body["flash_source_path"] == str(_FIXTURE.resolve())
        assert isinstance(body["flash_source_fingerprint"], dict)
        assert "b01/FFFFFFFE" in body["flash_source_fingerprint"]
        # start_raw_seconds is allowed to be 0.0 / None for this fixture
        # (the SW version doesn't expose Hello time fields), but
        # next_session_start_raw_seconds must agree across consecutive
        # sessions.
    # Cross-session consistency: each manifest's telemetry_window_upper
    # equals the next session's start. The first session's lower is None
    # (-inf); subsequent sessions' lower is their own start.
    for i, (r0, r1) in enumerate(zip(results, results[1:])):
        manifest_path0 = Path(r0.session_dir) / "session.json"
        with manifest_path0.open() as fh:
            body0 = json.load(fh)
        assert body0["telemetry_window_upper_raw_seconds"] == r1.start_raw_seconds
        if i == 0:
            # First session in the run owns everything before session 1's start.
            assert body0["telemetry_window_lower_raw_seconds"] is None
        else:
            assert body0["telemetry_window_lower_raw_seconds"] == r0.start_raw_seconds


def test_process_session_rederives_telemetry_from_flash(tmp_path: Path):
    from lusee.ingest import process_flash, process_session

    sessions_root = tmp_path / "sessions"
    h5_dir = tmp_path / "h5"
    results = process_flash(
        _FIXTURE,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
        manifest_dir=h5_dir,
    )
    # Pick a session that has telemetry on the single-pass run.
    pick = next((r for r in results if r.has_telemetry), None)
    if pick is None:
        pytest.skip("fixture has no telemetry session")

    # Now reprocess that session via process_session into a fresh out dir.
    out_h5 = tmp_path / "h5_reprocessed"
    re_result = process_session(
        Path(pick.session_dir),
        h5_dir=out_h5,
        ordinal=pick.session_ordinal,
        name=pick.session_name,
    )
    assert re_result.has_telemetry, (
        "expected telemetry to be re-derived from the flash backreference"
    )
    assert re_result.telemetry_source == "flash"
    assert re_result.flash_source_path == str(_FIXTURE.resolve())

    # The new HDF5 must have /DCB_telemetry/ with the same shape as the
    # original single-pass HDF5.
    orig = Path(pick.h5_path)
    re_h5 = out_h5 / f"{pick.session_name}.h5"
    with h5py.File(orig, "r") as f1, h5py.File(re_h5, "r") as f2:
        assert "DCB_telemetry" in f1, "single-pass HDF5 missing /DCB_telemetry"
        assert "DCB_telemetry" in f2, "re-derived HDF5 missing /DCB_telemetry"
        # Check a representative channel
        for ch in ("fpga_THERM_FPGA", "fpga_VMON_6V", "fpga_mission_seconds"):
            a1 = _bytes_in_dataset(f1["DCB_telemetry"], ch.split("fpga_")[1] and ch)
            a2 = _bytes_in_dataset(f2["DCB_telemetry"], ch.split("fpga_")[1] and ch)
            if a1 is None or a2 is None:
                continue
            assert a1.shape == a2.shape, f"shape mismatch on {ch}: {a1.shape} vs {a2.shape}"
            np.testing.assert_array_equal(a1, a2)


def test_process_session_warns_on_fingerprint_mismatch(tmp_path: Path):
    from lusee.ingest import process_flash, process_session

    # Run process_flash against a *copy* of the fixture so we can tamper
    # with the bank file mtime and not leak side effects to other tests.
    flash_copy = tmp_path / "flash_copy"
    shutil.copytree(_FIXTURE, flash_copy)
    sessions_root = tmp_path / "sessions"
    h5_dir = tmp_path / "h5"
    results = process_flash(
        flash_copy,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
    )
    pick = next((r for r in results if r.has_telemetry), None)
    if pick is None:
        pytest.skip("fixture has no telemetry session")

    # Bump the b01 mtime so the fingerprint disagrees, but leave content
    # alone so the slice still produces the same telemetry.
    b01 = flash_copy / "b01" / "FFFFFFFE"
    st = b01.stat()
    os.utime(b01, (st.st_atime, st.st_mtime + 60.0))

    out_h5 = tmp_path / "h5_reprocessed"
    with pytest.warns(RuntimeWarning, match="fingerprint"):
        result = process_session(
            Path(pick.session_dir),
            h5_dir=out_h5,
            ordinal=pick.session_ordinal,
            name=pick.session_name,
        )
    # The re-derive still succeeds despite the warning.
    assert result.has_telemetry
    assert result.telemetry_source == "flash"


def test_no_rederive_telemetry_flag(tmp_path: Path):
    from lusee.ingest import process_flash, process_session

    flash_copy = tmp_path / "flash_copy"
    shutil.copytree(_FIXTURE, flash_copy)
    sessions_root = tmp_path / "sessions"
    h5_dir = tmp_path / "h5"
    results = process_flash(
        flash_copy,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
    )
    pick = next((r for r in results if r.has_telemetry), None)
    if pick is None:
        pytest.skip("fixture has no telemetry session")

    out_h5 = tmp_path / "h5_no_rederive"
    result = process_session(
        Path(pick.session_dir),
        h5_dir=out_h5,
        ordinal=pick.session_ordinal,
        name=pick.session_name,
        rederive_telemetry=False,
    )
    # No flash re-derive, no sidecar -> no telemetry.
    assert not result.has_telemetry
    assert result.telemetry_source is None

    re_h5 = out_h5 / f"{pick.session_name}.h5"
    with h5py.File(re_h5, "r") as f:
        assert "DCB_telemetry" not in f


def test_flash_root_override(tmp_path: Path):
    from lusee.ingest import process_flash, process_session

    flash_copy = tmp_path / "original_flash"
    shutil.copytree(_FIXTURE, flash_copy)
    sessions_root = tmp_path / "sessions"
    h5_dir = tmp_path / "h5"
    results = process_flash(
        flash_copy,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
    )
    pick = next((r for r in results if r.has_telemetry), None)
    if pick is None:
        pytest.skip("fixture has no telemetry session")

    # "Move" the flash by renaming the directory; the manifest's recorded
    # path is now stale.
    new_flash = tmp_path / "moved_flash"
    flash_copy.rename(new_flash)

    out_h5 = tmp_path / "h5_moved"
    result = process_session(
        Path(pick.session_dir),
        h5_dir=out_h5,
        ordinal=pick.session_ordinal,
        name=pick.session_name,
        flash_root=new_flash,
    )
    assert result.has_telemetry
    assert result.telemetry_source == "flash"
    assert Path(result.flash_source_path) == new_flash
