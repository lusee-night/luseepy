from __future__ import annotations

from types import SimpleNamespace

import pytest

from lusee.ingest import collation, session
from lusee.ingest.ccsds import CcsdsFrame, PrimaryHeader
from lusee.ingest.issues import IssueCollector
from lusee.ingest.reassembly import LogicalPacket, reassemble_logical_packets


def frame(
    appid: int,
    sequence: int,
    groupflags: int,
    payload: bytes,
) -> CcsdsFrame:
    header = PrimaryHeader(
        version=0,
        packet_type=0,
        secheaderflag=0,
        appid=appid,
        groupflags=groupflags,
        sequence_cnt=sequence,
        packetlen=len(payload) - 1,
    )
    return CcsdsFrame(header=header, head_bytes=b"\x00" * 6, payload=payload)


def packet(
    appid: int,
    sequence: int,
    blob: bytes,
    *,
    file_index: int | None = None,
) -> LogicalPacket:
    return LogicalPacket(
        appid=appid,
        start_seq=sequence,
        seq=sequence,
        blob=blob,
        single_packet=True,
        bank="b05",
        file_index=file_index,
    )


def test_reassembly_apid_change_keeps_output_and_records_context():
    frames = [
        frame(0x201, 5, 0, b"\x01\x02"),
        frame(0x202, 6, 1, b"\x03\x04"),
    ]

    with pytest.warns(RuntimeWarning) as plain_warnings:
        plain = list(
            reassemble_logical_packets(
                frames,
                byteswap_pairs=False,
                bank="b05",
            )
        )
    collector = IssueCollector()
    with pytest.warns(RuntimeWarning) as diagnostic_warnings:
        diagnostic = list(
            reassemble_logical_packets(
                frames,
                byteswap_pairs=False,
                bank="b05",
                issue_collector=collector,
            )
        )

    assert diagnostic == plain
    assert str(diagnostic_warnings[0].message) == str(plain_warnings[0].message)
    assert len(collector.issues) == 1
    issue = collector.issues[0]
    assert issue.code == "reassembly.apid_changed_mid_packet"
    assert issue.stage == "reassembly"
    assert issue.action.value == "kept"
    assert issue.bank == "b05"
    assert issue.frame_index == 1
    assert issue.packet_index == 0
    assert issue.appid == 0x201
    assert issue.sequence_count == 6
    assert issue.as_dict()["details"] == {"new_appid": 0x202}


@pytest.mark.parametrize(
    "frames, code, frame_index, sequence_count, details",
    [
        (
            [
                frame(0x201, 5, 0, b"\x01\x02"),
                frame(0x201, 6, 3, b"\x03"),
            ],
            "reassembly.invalid_science_payload_length",
            1,
            6,
            {"accumulated_payload_bytes": 2, "frame_payload_bytes": 1},
        ),
        (
            [frame(0x201, 7, 0, b"\x01\x02")],
            "reassembly.trailing_partial_packet",
            0,
            7,
            {"accumulated_payload_bytes": 2},
        ),
    ],
)
def test_reassembly_drops_are_structured_and_still_warn(
    frames,
    code,
    frame_index,
    sequence_count,
    details,
):
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning):
        result = list(
            reassemble_logical_packets(
                frames,
                byteswap_pairs=True,
                bank="b06",
                issue_collector=collector,
            )
        )

    assert result == []
    assert len(collector.issues) == 1
    issue = collector.issues[0]
    assert issue.code == code
    assert issue.action.value == "dropped"
    assert issue.bank == "b06"
    assert issue.frame_index == frame_index
    assert issue.packet_index == 0
    assert issue.appid == 0x201
    assert issue.sequence_count == sequence_count
    assert issue.as_dict()["details"] == details


def patch_identity_classes(monkeypatch):
    monkeypatch.setattr(
        collation,
        "is_uid_prefixed",
        lambda appid: appid in (0x201, 0x206),
    )
    monkeypatch.setattr(collation, "is_uid_typed", lambda appid: appid == 0x202)
    monkeypatch.setattr(
        collation,
        "is_uid_derived",
        lambda appid: appid in (0x203, 0x207),
    )
    monkeypatch.setattr(
        collation,
        "is_dropped_appid",
        lambda appid: appid == 0x204,
    )


def test_identity_collector_does_not_change_uid_sort_heuristic(monkeypatch):
    patch_identity_classes(monkeypatch)

    def inputs():
        return [
            packet(0x206, 2, (20).to_bytes(4, "little")),
            packet(0x207, 1, b"derived"),
            packet(0x206, 9, (10).to_bytes(4, "little")),
        ]

    plain = collation.assign_identities(inputs(), sw_version=307)
    collector = IssueCollector()
    diagnostic = collation.assign_identities(
        inputs(),
        sw_version=307,
        issue_collector=collector,
    )

    assert [
        (item.appid, item.seq, item.unique_packet_id) for item in diagnostic
    ] == [
        (item.appid, item.seq, item.unique_packet_id) for item in plain
    ] == [
        (0x206, 9, 10),
        (0x207, 1, 20),
        (0x206, 2, 20),
    ]
    assert collector.issues == ()


def test_identity_drops_have_specific_reasons_without_changing_association(
    monkeypatch,
):
    patch_identity_classes(monkeypatch)
    packets = [
        packet(0x201, 0, b"\x01"),
        packet(0x202, 1, b"typed"),
        packet(0x203, 2, b"derived-too-early"),
        packet(0x204, 3, b"heartbeat"),
        packet(0x205, 4, b"unknown"),
        packet(0x206, 8, (9).to_bytes(4, "little")),
        packet(0x207, 7, b"derived"),
    ]
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning, match="too short"):
        kept = collation.assign_identities(
            packets,
            sw_version=307,
            typed_uid_extractor=lambda appid, blob, version: None,
            sort=False,
            issue_collector=collector,
        )

    assert kept == packets[-2:]
    assert [item.unique_packet_id for item in kept] == [9, 9]
    assert [issue.code for issue in collector.issues] == [
        "identity.uid_prefix_too_short",
        "identity.packet_without_uid_dropped",
        "identity.packet_without_uid_dropped",
        "identity.packet_without_uid_dropped",
        "identity.appid_intentionally_dropped",
        "identity.packet_without_uid_dropped",
    ]
    assert [
        issue.as_dict()["details"].get("reason")
        for issue in collector.issues[1:]
    ] == [
        "invalid_uid_prefix",
        "typed_uid_unavailable",
        "no_preceding_unique_packet_id",
        "intentionally_filtered_appid",
        "unrecognized_appid",
    ]
    assert [issue.packet_index for issue in collector.issues] == [0, 0, 1, 2, 3, 4]
    assert collector.issues[0].action.value == "rejected"
    assert collector.issues[4].severity.value == "info"


def test_default_typed_uid_failure_records_rejection_and_drop(
    monkeypatch,
):
    class BrokenDecoder:
        def Packet(self, appid, *, blob, version):
            raise ValueError("bad typed header")

    monkeypatch.setattr(collation, "load_uncrater", lambda: BrokenDecoder())
    monkeypatch.setattr(collation, "is_dropped_appid", lambda appid: False)
    monkeypatch.setattr(collation, "is_uid_prefixed", lambda appid: False)
    monkeypatch.setattr(collation, "is_uid_typed", lambda appid: True)
    monkeypatch.setattr(collation, "is_uid_derived", lambda appid: False)
    source = packet(0x210, 33, b"bad", file_index=12)
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning, match="failed to extract"):
        kept = collation.assign_identities(
            [source],
            sw_version=307,
            issue_collector=collector,
        )

    assert kept == []
    assert [issue.code for issue in collector.issues] == [
        "identity.typed_uid_extraction_failed",
        "identity.packet_without_uid_dropped",
    ]
    assert [issue.action.value for issue in collector.issues] == [
        "rejected",
        "dropped",
    ]
    assert [issue.packet_index for issue in collector.issues] == [12, 12]
    assert all(issue.appid == 0x210 for issue in collector.issues)
    assert all(issue.sequence_count == 33 for issue in collector.issues)


def test_hello_version_failure_is_kept_and_structured(monkeypatch):
    class BrokenDecoder:
        @staticmethod
        def appid_is_hello(appid):
            return True

        def Packet(self, appid, *, blob):
            raise ValueError("bad Hello")

    monkeypatch.setattr(collation, "load_uncrater", lambda: BrokenDecoder())
    source = packet(0x100, 4, b"bad", file_index=8)
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning, match="failed to read SW_version"):
        version = collation.detect_sw_version(
            [source],
            issue_collector=collector,
        )

    assert version is None
    assert len(collector.issues) == 1
    issue = collector.issues[0]
    assert issue.code == "identity.hello_sw_version_decode_failed"
    assert issue.action.value == "kept"
    assert issue.packet_index == 8
    assert issue.appid == 0x100
    assert issue.sequence_count == 4


def test_session_start_decode_failure_keeps_session_and_records_context(
    monkeypatch,
):
    class BrokenDecoder:
        id = SimpleNamespace(AppID_uC_Start=0x100)

        @staticmethod
        def appid_is_hello(appid):
            return appid == 0x100

        def Packet(self, appid, *, blob, version):
            raise ValueError("bad session Hello")

    monkeypatch.setattr(session, "load_uncrater", lambda: BrokenDecoder())
    hello = packet(0x100, 12, b"bad", file_index=7)
    hello.unique_packet_id = 42
    collector = IssueCollector()

    with pytest.warns(RuntimeWarning, match="failed to decode Hello"):
        sessions = session.split_sessions(
            [hello],
            issue_collector=collector,
        )

    assert len(sessions) == 1
    assert sessions[0].packets == [hello]
    assert sessions[0].start_raw_seconds is None
    assert len(collector.issues) == 1
    issue = collector.issues[0]
    assert issue.code == "session_start.hello_decode_failed"
    assert issue.action.value == "kept"
    assert issue.bank == "b05"
    assert issue.packet_index == 7
    assert issue.appid == 0x100
    assert issue.sequence_count == 12
    assert issue.uid == 42
    assert issue.session == "ordinal:0"
    assert issue.as_dict()["details"] == {
        "exception_type": "ValueError",
        "session_ordinal": 0,
        "sw_version": None,
    }
