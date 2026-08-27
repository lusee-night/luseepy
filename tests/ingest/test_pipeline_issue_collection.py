from __future__ import annotations

import json

import pytest

from lusee.ingest import collation, pipeline
from lusee.ingest.constants import BANK_FILENAME, SCIENCE_BANKS, TELEMETRY_BANK
from lusee.ingest.decode import Products
from lusee.ingest.issues import IssueCollector
from lusee.ingest.reassembly import LogicalPacket


def write_landing_reference(tmp_path):
    path = tmp_path / "landing.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "reference_event": "landing",
        "clock_reference_isot": "2027-05-01T00:00:00",
        "time_scale": "utc",
        "clocks": {
            "spectrometer": {"clock_reference_raw_seconds": 0.0},
        },
        "source": "synthetic pipeline test",
        "assumed": True,
    }), encoding="utf-8")
    return path


def make_bank(flash_dir, bank):
    bank_dir = flash_dir / bank
    bank_dir.mkdir(parents=True)
    (bank_dir / BANK_FILENAME).write_bytes(b"")


def test_failed_hello_sw_version_is_recorded_once(tmp_path, monkeypatch):
    class SyntheticDecoder:
        @staticmethod
        def appid_is_hello(appid):
            return appid == 0x100

        @staticmethod
        def Packet(appid, *, blob):
            raise ValueError("synthetic failed Hello")

    source = LogicalPacket(
        appid=0x100,
        start_seq=4,
        seq=4,
        blob=(41).to_bytes(4, "little"),
        single_packet=True,
        bank=SCIENCE_BANKS[0],
        file_index=8,
    )
    make_bank(tmp_path, SCIENCE_BANKS[0])
    monkeypatch.setattr(
        pipeline,
        "parse_bank_file",
        lambda *args, **kwargs: iter(()),
    )
    monkeypatch.setattr(
        pipeline,
        "reassemble_logical_packets",
        lambda *args, **kwargs: iter((source,)),
    )
    monkeypatch.setattr(
        collation,
        "load_uncrater",
        lambda: SyntheticDecoder(),
    )
    monkeypatch.setattr(collation, "is_dropped_appid", lambda appid: False)
    monkeypatch.setattr(collation, "is_uid_prefixed", lambda appid: True)
    monkeypatch.setattr(pipeline, "is_dropped_appid", lambda appid: False)
    monkeypatch.setattr(
        pipeline,
        "split_sessions",
        lambda packets, *, issue_collector=None: [],
    )
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning, match="failed to read SW_version") as seen:
        sessions, _telemetry = pipeline.parse_flash(
            tmp_path,
            landing_time_file=write_landing_reference(tmp_path),
            issue_collector=collector,
        )

    assert len(seen) == 1
    assert sessions == []
    assert [issue.code for issue in collector.issues] == [
        "identity.hello_sw_version_decode_failed",
    ]


def test_parse_flash_isolates_telemetry_issues_from_science_quality(
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
        if bank == TELEMETRY_BANK:
            issue_collector.record(
                code="framing.synthetic_b01_damage",
                severity="warning",
                stage="framing",
                message="synthetic b01 framing damage",
                action="dropped",
                bank=bank,
            )
        return iter(())

    monkeypatch.setattr(pipeline, "parse_bank_file", fake_parse_bank_file)
    monkeypatch.setattr(
        pipeline,
        "detect_sw_version",
        lambda packets, *, issue_collector=None: None,
    )
    monkeypatch.setattr(
        pipeline,
        "assign_identities",
        lambda packets, **kwargs: packets,
    )
    monkeypatch.setattr(
        pipeline,
        "split_sessions",
        lambda packets, *, issue_collector=None: [],
    )
    telemetry_seen = []
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "decode_b01_packets",
        lambda packets: telemetry_seen.append(tuple(packets)) or None,
    )

    assert pipeline.parse_flash(
        tmp_path,
        landing_time_file=write_landing_reference(tmp_path),
        issue_collector=collector,
    ) == ([], None)

    science_call, telemetry_call = calls
    assert science_call[1:] == (science_bank, collector)
    assert telemetry_call[1] == TELEMETRY_BANK
    assert telemetry_call[2] is not collector
    assert collector.issues == ()
    assert telemetry_seen == [()]


def test_process_flash_preserves_caller_collector_identity(tmp_path, monkeypatch):
    flash_dir = tmp_path / "flash"
    sessions_root = tmp_path / "sessions"
    flash_dir.mkdir()
    collector = IssueCollector()
    parse_seen = []
    decode_seen = []
    worker_seen = []

    def fake_parse_flash(
        path,
        *,
        clock_reference_set=None,
        issue_collector=None,
        capture=None,
    ):
        parse_seen.append((path, issue_collector))
        return (
            [pipeline.Session(ordinal=0)],
            None,
        )

    def fake_process_one_session(**kwargs):
        worker_seen.append(kwargs)
        return pipeline.SessionResult(
            session_ordinal=kwargs["ordinal"],
            session_name=kwargs["name"],
            source_path=str(kwargs["source_path"]),
            source_kind=kwargs["source_kind"],
        )

    def fake_read(path, *, issue_collector=None, **kwargs):
        decode_seen.append((path, issue_collector, kwargs))
        return Products()

    monkeypatch.setattr(pipeline, "_parse_flash_loaded", fake_parse_flash)
    monkeypatch.setattr(
        pipeline,
        "write_uncrater_session",
        lambda session, session_dir: session_dir,
    )
    monkeypatch.setattr(pipeline, "read_uncrater_session", fake_read)
    monkeypatch.setattr(pipeline, "_binding_identity", lambda products: ("307",))
    monkeypatch.setattr(pipeline, "_process_one_session", fake_process_one_session)

    result = pipeline.process_flash(
        flash_dir,
        landing_time_file=write_landing_reference(tmp_path),
        sessions_root=sessions_root,
        issue_collector=collector,
        decoder_strict=True,
        schema_variant="early",
    )

    assert len(result) == 1
    assert parse_seen == [(flash_dir.resolve(), collector)]
    assert decode_seen == [(
        sessions_root / "session_000",
        collector,
        {"strict": True, "schema_variant": "early"},
    )]
    assert len(worker_seen) == 1
    assert worker_seen[0]["issue_collector"] is collector


def test_process_session_forwards_decoder_policy(tmp_path, monkeypatch):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    collector = IssueCollector()
    seen = []

    def fake_process_one_session(**kwargs):
        seen.append(kwargs)
        return pipeline.SessionResult(
            session_ordinal=kwargs["ordinal"],
            session_name=kwargs["name"],
            source_path=str(kwargs["source_path"]),
            source_kind=kwargs["source_kind"],
        )

    monkeypatch.setattr(pipeline, "_process_one_session", fake_process_one_session)
    monkeypatch.setattr(
        pipeline.telemetry_mod,
        "find_legacy_sidecar",
        lambda path: None,
    )

    pipeline.process_session(
        session_dir,
        decoder_strict=True,
        diagnostic_override=True,
        schema_variant="early",
        issue_collector=collector,
    )

    assert len(seen) == 1
    assert seen[0]["decoder_strict"] is True
    assert seen[0]["diagnostic_override"] is True
    assert seen[0]["schema_variant"] == "early"
    assert seen[0]["issue_collector"] is collector


def test_session_worker_forwards_decoder_policy_to_uncrater(
    tmp_path,
    monkeypatch,
):
    collector = IssueCollector()
    seen = []

    def fake_read(path, **kwargs):
        seen.append((path, kwargs))
        return Products()

    monkeypatch.setattr(pipeline, "read_uncrater_session", fake_read)
    pipeline._process_one_session(
        session_dir=tmp_path,
        name="session",
        ordinal=0,
        h5_dir=None,
        plots_dir=None,
        manifest_dir=None,
        issue_collector=collector,
        decoder_strict=True,
        diagnostic_override=True,
        schema_variant="final",
    )

    assert seen == [
        (
            tmp_path,
            {
                "strict": True,
                "diagnostic_override": True,
                "schema_variant": "final",
                "issue_collector": collector,
            },
        )
    ]
