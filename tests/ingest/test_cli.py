from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from lusee.ingest import cli, pipeline
from lusee.ingest.issues import IssuePolicy


def test_status_exit_codes():
    assert cli.status_exit_code("clean") == 0
    assert cli.status_exit_code("partial") == 2
    assert cli.status_exit_code("failed") == 1
    with pytest.raises(ValueError, match="unknown ingest status"):
        cli.status_exit_code("unknown")


def test_process_session_routes_options_and_prints_paths(
    tmp_path,
    monkeypatch,
    capsys,
):
    seen = {}
    h5_path = tmp_path / "h5" / "chosen.h5"
    manifest_path = tmp_path / "manifests" / "chosen.json"

    def process_session(session_dir, **kwargs):
        seen["session_dir"] = session_dir
        seen.update(kwargs)
        return pipeline.SessionResult(
            session_ordinal=4,
            session_name="chosen",
            source_path=str(session_dir),
            source_kind="session",
            session_dir=str(Path(session_dir).resolve()),
            h5_path=str(h5_path),
            manifest_path=str(manifest_path),
        )

    monkeypatch.setattr(cli.pipeline, "process_session", process_session)
    session_dir = tmp_path / "input"
    landing = tmp_path / "landing.json"
    flash_root = tmp_path / "flash"
    result = cli.main([
        "process-session",
        str(session_dir),
        "--landing-time-file",
        str(landing),
        "--h5-dir",
        str(tmp_path / "h5"),
        "--fits-dir",
        str(tmp_path / "fits"),
        "--plots-dir",
        str(tmp_path / "plots"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
        "--name",
        "chosen",
        "--ordinal",
        "4",
        "--flash-root",
        str(flash_root),
        "--no-rederive-telemetry",
        "--overwrite",
        "--issue-policy",
        "strict",
        "--decoder-strict",
        "--schema-variant",
        "early",
    ])

    assert result == 0
    assert seen["session_dir"] == session_dir
    assert seen["landing_time_file"] == landing
    assert seen["name"] == "chosen"
    assert seen["ordinal"] == 4
    assert seen["flash_root"] == flash_root
    assert seen["rederive_telemetry"] is False
    assert seen["overwrite"] is True
    assert seen["issue_collector"].policy is IssuePolicy.STRICT
    assert seen["decoder_strict"] is True
    assert seen["schema_variant"] == "early"
    output = capsys.readouterr().out
    assert "status: clean" in output
    assert f"hdf5: {h5_path}" in output
    assert f"manifest: {manifest_path}" in output
    assert "issues: none" in output


def test_process_flash_partial_returns_two_and_summarizes_issues(
    tmp_path,
    monkeypatch,
    capsys,
):
    seen = {}
    manifest = tmp_path / "sessions" / "flash.json"

    def process_flash(flash_dir, **kwargs):
        seen["flash_dir"] = flash_dir
        seen.update(kwargs)
        return pipeline.FlashResult(
            flash_result_id=f"flash-{'a' * 16}",
            input_identity_sha256="b" * 64,
            source_path=str(flash_dir),
            source_fingerprint={},
            status="partial",
            issue_counts={"z.issue": 1, "a.issue": 2},
            status_issue_codes=["a.issue", "z.issue"],
            manifest_paths=[str(manifest)],
        )

    monkeypatch.setattr(cli.pipeline, "process_flash", process_flash)
    result = cli.main([
        "process-flash",
        str(tmp_path / "flash"),
        "--landing-time-file",
        str(tmp_path / "landing.json"),
        "--sessions-root",
        str(tmp_path / "sessions"),
        "--issue-policy",
        "collect",
        "--decoder-strict",
        "--schema-variant",
        "final",
    ])

    assert result == 2
    assert seen["decoder_strict"] is True
    assert seen["schema_variant"] == "final"
    assert seen["issue_collector"].policy is IssuePolicy.COLLECT
    output = capsys.readouterr().out
    assert "status: partial" in output
    assert f"manifest: {manifest}" in output
    assert "issues: a.issue=2, z.issue=1" in output


def test_command_error_returns_one_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        cli.pipeline,
        "process_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("synthetic failure")
        ),
    )
    result = cli.main([
        "process-session",
        str(tmp_path / "input"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
    ])

    assert result == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: ValueError: synthetic failure\n"


def test_argument_error_returns_one(tmp_path, capsys):
    result = cli.main([
        "process-flash",
        str(tmp_path / "flash"),
        "--sessions-root",
        str(tmp_path / "sessions"),
    ])

    assert result == 1
    assert "--landing-time-file" in capsys.readouterr().err


def test_validate_routes_layout_v4_and_preserves_partial_exit(
    tmp_path,
    monkeypatch,
    capsys,
):
    from lusee.ingest import obs_factory

    path = tmp_path / "product.h5"
    path.write_bytes(b"synthetic")
    seen = []
    monkeypatch.setattr(
        obs_factory,
        "load_bundle",
        lambda value: seen.append(value) or SimpleNamespace(
            layout_version=4,
            quality_status="partial",
        ),
    )

    assert cli.main(["validate", str(path)]) == 2
    assert seen == [path]
    output = capsys.readouterr().out
    assert "layout-v4 output" in output
    assert "status: partial" in output


def test_validate_session_manifest_uses_strict_linkage(
    tmp_path,
    monkeypatch,
    capsys,
):
    session_dir = tmp_path / "session_000"
    session_dir.mkdir()
    manifest = session_dir / "session.json"
    document = {
        "manifest_kind": "session",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "status": "clean",
    }
    manifest.write_text(json.dumps(document), encoding="ascii")
    seen = []
    monkeypatch.setattr(
        cli.pipeline,
        "_read_in_session_manifest",
        lambda path, *, strict: seen.append((path, strict)) or document,
    )

    assert cli.main(["validate", str(manifest)]) == 0
    assert seen == [(session_dir, True)]
    assert "session manifest" in capsys.readouterr().out


def test_validate_empty_flash_manifest(tmp_path, capsys):
    manifest = tmp_path / "flash.json"
    manifest.write_text(json.dumps({
        "manifest_kind": "flash",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": "flash.json",
            "external_copies_are_mirrors": True,
        },
        "status": "failed",
        "sessions": [],
    }), encoding="ascii")

    assert cli.main(["validate", str(manifest)]) == 1
    output = capsys.readouterr().out
    assert "FLASH manifest" in output
    assert "status: failed" in output


def test_validate_flash_rejects_unlinked_session(tmp_path, capsys):
    session_dir = tmp_path / "session_000"
    session_dir.mkdir()
    (session_dir / "session.json").write_text(json.dumps({
        "manifest_kind": "session",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "status": "clean",
    }), encoding="ascii")
    (tmp_path / "flash.json").write_text(json.dumps({
        "manifest_kind": "flash",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": "flash.json",
            "external_copies_are_mirrors": True,
        },
        "status": "clean",
        "sessions": [{
            "session_ordinal": 0,
            "session_name": "session_000",
            "session_dir": "session_000",
            "status": "clean",
        }],
    }), encoding="ascii")

    assert cli.main(["validate", str(tmp_path / "flash.json")]) == 1
    assert "not linked" in capsys.readouterr().err


def write_linked_flash_fixture(
    tmp_path,
    *,
    swap_locators,
    run_status="clean",
    session_statuses=("clean", "clean"),
):
    flash_result_id = f"flash-{'a' * 16}"
    source_fingerprint = {
        "b02/FFFFFFFE": {
            "size_bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        },
    }
    input_identity = pipeline._flash_source_identity_sha256(
        source_fingerprint
    )
    clock_reference = {"source_sha256": "d" * 64}
    locators = ["session_001", "session_000"] if swap_locators else [
        "session_000",
        "session_001",
    ]
    document = {
        "manifest_kind": "flash",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": "flash.json",
            "external_copies_are_mirrors": True,
        },
        "flash_result_id": flash_result_id,
        "input_identity_sha256": input_identity,
        "input_identity_unavailable_reason": None,
        "source_fingerprint": source_fingerprint,
        "clock_reference": clock_reference,
        "status": run_status,
        "sessions": [
            {
                "session_ordinal": ordinal,
                "session_name": f"session_{ordinal:03d}",
                "session_dir": locators[ordinal],
                "status": session_statuses[ordinal],
            }
            for ordinal in range(2)
        ],
    }
    flash_path = tmp_path / "flash.json"
    payload = json.dumps(document).encode("ascii")
    flash_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    for ordinal in range(2):
        session_dir = tmp_path / f"session_{ordinal:03d}"
        session_dir.mkdir()
        (session_dir / "session.json").write_text(json.dumps({
            "manifest_kind": "session",
            "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
            "status": session_statuses[ordinal],
            "flash_result_id": flash_result_id,
            "flash_manifest_sha256": digest,
            "flash_input_identity_sha256": input_identity,
            "flash_source_fingerprint": source_fingerprint,
            "clock_reference": clock_reference,
            "session_ordinal": ordinal,
            "session_name": f"session_{ordinal:03d}",
        }), encoding="ascii")
    return flash_path


def test_validate_flash_rejects_swapped_session_locators(tmp_path, capsys):
    flash_path = write_linked_flash_fixture(tmp_path, swap_locators=True)

    assert cli.main(["validate", str(flash_path)]) == 1
    assert "locator does not match" in capsys.readouterr().err


def test_validate_flash_accepts_linked_sessions(tmp_path, capsys):
    flash_path = write_linked_flash_fixture(tmp_path, swap_locators=False)

    assert cli.main(["validate", str(flash_path)]) == 0
    assert "FLASH manifest" in capsys.readouterr().out


def test_validate_flash_rejects_usable_empty_run(tmp_path, capsys):
    path = tmp_path / "flash.json"
    path.write_text(json.dumps({
        "manifest_kind": "flash",
        "manifest_schema_version": pipeline.MANIFEST_SCHEMA_VERSION,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": "flash.json",
            "external_copies_are_mirrors": True,
        },
        "status": "clean",
        "sessions": [],
    }), encoding="ascii")

    assert cli.main(["validate", str(path)]) == 1
    assert "usable FLASH run has no sessions" in capsys.readouterr().err


def test_validate_flash_rejects_clean_run_with_partial_session(
    tmp_path,
    capsys,
):
    path = write_linked_flash_fixture(
        tmp_path,
        swap_locators=False,
        run_status="clean",
        session_statuses=("partial", "clean"),
    )

    assert cli.main(["validate", str(path)]) == 1
    assert "clean FLASH run contains" in capsys.readouterr().err


def test_validate_flash_rejects_partial_run_with_all_failed_sessions(
    tmp_path,
    capsys,
):
    path = write_linked_flash_fixture(
        tmp_path,
        swap_locators=False,
        run_status="partial",
        session_statuses=("failed", "failed"),
    )

    assert cli.main(["validate", str(path)]) == 1
    assert "all-failed FLASH sessions" in capsys.readouterr().err


def test_console_entrypoint_is_declared():
    project = tomllib.loads(Path("pyproject.toml").read_text("utf-8"))
    assert project["project"]["scripts"]["lusee-ingest"] == (
        "lusee.ingest.cli:main"
    )
