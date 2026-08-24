from __future__ import annotations

from lusee.ingest import pipeline
from lusee.ingest.constants import BANK_FILENAME, SCIENCE_BANKS, TELEMETRY_BANK
from lusee.ingest.issues import IssueCollector


def make_bank(flash_dir, bank):
    bank_dir = flash_dir / bank
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")


def test_parse_flash_threads_one_collector_through_science_and_telemetry(
    tmp_path,
    monkeypatch,
):
    science_bank = SCIENCE_BANKS[0]
    make_bank(tmp_path, science_bank)
    make_bank(tmp_path, TELEMETRY_BANK)
    collector = IssueCollector()
    calls = []

    def fake_parse_bank_file(path, *, bank=None, issue_collector=None):
        calls.append((path, bank, issue_collector))
        return iter(())

    monkeypatch.setattr(pipeline, "parse_bank_file", fake_parse_bank_file)
    monkeypatch.setattr(pipeline, "detect_sw_version", lambda packets: None)
    monkeypatch.setattr(
        pipeline,
        "assign_identities",
        lambda packets, *, sw_version: packets,
    )
    monkeypatch.setattr(pipeline, "split_sessions", lambda packets: [])
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "parse_b01_packets",
        lambda packets: ({}, {}),
    )
    monkeypatch.setattr(
        pipeline,
        "assign_telemetry_to_sessions",
        lambda sessions, fpga, encoder: None,
    )

    result = pipeline.parse_flash(tmp_path, issue_collector=collector)

    assert result == ([], {}, {})
    assert [call[1] for call in calls] == [science_bank, TELEMETRY_BANK]
    assert all(call[2] is collector for call in calls)


def test_process_flash_preserves_caller_collector_identity(tmp_path, monkeypatch):
    flash_dir = tmp_path / "flash"
    sessions_root = tmp_path / "sessions"
    flash_dir.mkdir()
    collector = IssueCollector()
    seen = []

    def fake_parse_flash(path, *, issue_collector=None):
        seen.append((path, issue_collector))
        return [], {}, {}

    monkeypatch.setattr(pipeline, "parse_flash", fake_parse_flash)

    result = pipeline.process_flash(
        flash_dir,
        sessions_root=sessions_root,
        issue_collector=collector,
    )

    assert result == []
    assert seen == [(flash_dir.resolve(), collector)]


def test_telemetry_rederive_uses_the_shared_collector(tmp_path, monkeypatch):
    make_bank(tmp_path, TELEMETRY_BANK)
    collector = IssueCollector()
    seen = []

    def fake_parse_bank_file(path, *, bank=None, issue_collector=None):
        seen.append((path, bank, issue_collector))
        return iter(())

    monkeypatch.setattr(pipeline, "parse_bank_file", fake_parse_bank_file)
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "parse_b01_packets",
        lambda packets: ({}, {}),
    )

    result = pipeline._rederive_telemetry_from_flash(
        tmp_path,
        window_lower_raw_seconds=None,
        window_upper_raw_seconds=None,
        issue_collector=collector,
    )

    assert result == (None, None)
    assert len(seen) == 1
    assert seen[0][1] == TELEMETRY_BANK
    assert seen[0][2] is collector
