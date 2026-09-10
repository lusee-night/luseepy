from __future__ import annotations

import hashlib
import logging
import warnings
from dataclasses import FrozenInstanceError

import pytest

import lusee.ingest.ccsds as ccsds
from lusee.ingest.ccsds import (
    crc16_ccitt,
    parse_bank_file,
    parse_bank_file_diagnostic,
    parse_primary_header,
    parse_stream,
    parse_stream_diagnostic,
)
from lusee.ingest.issues import IngestIssueError, IssueCollector, IssuePolicy


SYNC_BYTES = b"\xec\xa0"


def make_header(
    *,
    payload_size: int,
    appid: int = 0x209,
    sequence: int = 0,
    groupflags: int = 3,
    version: int = 0,
    packet_type: int = 0,
    secondary_header: int = 0,
) -> bytes:
    assert payload_size >= 1
    h0 = (
        (version << 13)
        | (packet_type << 12)
        | (secondary_header << 11)
        | appid
    )
    h1 = (groupflags << 14) | sequence
    return (
        h0.to_bytes(2, "big")
        + h1.to_bytes(2, "big")
        + (payload_size - 1).to_bytes(2, "big")
    )


def make_frame(
    payload: bytes,
    *,
    appid: int = 0x209,
    sequence: int = 0,
    groupflags: int = 3,
    corrupt_crc: bool = False,
) -> bytes:
    header = make_header(
        payload_size=len(payload),
        appid=appid,
        sequence=sequence,
        groupflags=groupflags,
    )
    crc = crc16_ccitt(header + payload)
    if corrupt_crc:
        crc ^= 1
    return SYNC_BYTES + header + payload + crc.to_bytes(2, "big")


def test_primary_header_fields_and_payload_length_convention():
    header_bytes = make_header(
        payload_size=7,
        appid=0x345,
        sequence=0x1234,
        groupflags=2,
        version=5,
        packet_type=1,
        secondary_header=1,
    )
    header = parse_primary_header(header_bytes)

    assert header.version == 5
    assert header.packet_type == 1
    assert header.secheaderflag == 1
    assert header.appid == 0x345
    assert header.groupflags == 2
    assert header.sequence_cnt == 0x1234
    assert header.packetlen == 6


def test_crc_known_vector_and_frame_coverage():
    assert crc16_ccitt(b"123456789") == 0x29B1
    packet = make_frame(b"payload", sequence=4)
    header = packet[2:8]
    payload = packet[8:-2]
    transmitted = int.from_bytes(packet[-2:], "big")
    assert transmitted == crc16_ccitt(header + payload)
    assert transmitted != crc16_ccitt(packet[:-2])


def test_diagnostic_result_matches_legacy_acceptance_and_records_crc_drop():
    first = make_frame(b"first", appid=0x201, sequence=10)
    bad = make_frame(
        b"bad",
        appid=0x202,
        sequence=11,
        corrupt_crc=True,
    )
    last = make_frame(b"last", appid=0x203, sequence=12)
    stream = b"\xa5\x00" + first + bad + last

    with pytest.warns(RuntimeWarning) as legacy_warnings:
        legacy_frames = tuple(parse_stream(stream, source="legacy"))
    with pytest.warns(RuntimeWarning) as diagnostic_warnings:
        result = parse_stream_diagnostic(stream, source="diagnostic", bank="b05")

    assert result.frames == legacy_frames
    assert [frame.payload for frame in result.frames] == [b"first", b"last"]
    assert len(legacy_warnings) == len(diagnostic_warnings) == 1
    bad_transmitted_crc = int.from_bytes(bad[-2:], "big")
    bad_computed_crc = crc16_ccitt(bad[2:-2])
    assert str(legacy_warnings[0].message) == (
        f"legacy: CRC mismatch (apid=0x202, seq=11, "
        f"got=0x{bad_transmitted_crc:04x}, computed=0x{bad_computed_crc:04x}); "
        f"dropping and resyncing"
    )
    assert str(diagnostic_warnings[0].message).startswith(
        "diagnostic: CRC mismatch"
    )
    assert result.locations[0].accepted_index == 0
    assert result.locations[0].candidate_index == 0
    assert result.locations[0].source_offset == 2
    assert result.locations[1].accepted_index == 1
    assert result.locations[1].candidate_index == 2
    assert result.locations[1].source_offset == 2 + len(first) + len(bad)

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "framing.crc_mismatch"
    assert issue.issue_id == "issue-00000001"
    assert issue.action.value == "dropped"
    assert issue.input_identity == f"sha256:{hashlib.sha256(stream).hexdigest()}"
    assert issue.bank == "b05"
    assert issue.byte_offset == 2 + len(first)
    assert issue.frame_index == 1
    assert issue.appid == 0x202
    assert issue.sequence_count == 11
    assert issue.as_dict()["details"] == {
        "computed_crc": bad_computed_crc,
        "transmitted_crc": bad_transmitted_crc,
    }


@pytest.mark.parametrize(
    "stream, code, details",
    [
        (
            SYNC_BYTES + b"\x00\x01",
            "framing.truncated_header",
            {"available_header_bytes": 2, "expected_header_bytes": 6},
        ),
        (
            SYNC_BYTES + make_header(payload_size=4) + b"\x01\x02",
            "framing.truncated_body",
            {"available_body_bytes": 2, "expected_body_bytes": 6},
        ),
    ],
)
def test_truncated_header_and_body_are_structured_but_keep_info_visibility(
    stream,
    code,
    details,
    caplog,
):
    caplog.set_level(logging.INFO, logger="lusee.ingest.ccsds")
    with warnings.catch_warnings(record=True) as caught:
        result = parse_stream_diagnostic(stream, source="tail")

    assert caught == []
    assert result.frames == ()
    assert len(result.issues) == 1
    assert result.issues[0].code == code
    assert result.issues[0].byte_offset == 0
    assert result.issues[0].frame_index == 0
    assert result.issues[0].as_dict()["details"] == details
    assert "tail: stream ended mid-packet" in caplog.text


def test_padding_split_sync_and_payload_padding_preserve_current_behavior():
    payload = b"\x01\xa5\x02"
    packet_without_sync = make_frame(payload)[2:]
    stream = b"\x99\xec\xa5\xa0" + packet_without_sync

    result = parse_stream_diagnostic(stream)

    assert [frame.payload for frame in result.frames] == [payload]
    assert result.locations[0].source_offset == 1
    assert result.issues == ()


def test_false_sync_candidate_swallows_embedded_sync_until_declared_end():
    embedded = make_frame(b"embedded", appid=0x210, sequence=1)
    bad_outer = make_frame(
        embedded,
        appid=0x211,
        sequence=2,
        corrupt_crc=True,
    )
    following = make_frame(b"following", appid=0x212, sequence=3)

    with pytest.warns(RuntimeWarning):
        result = parse_stream_diagnostic(bad_outer + following)

    assert [frame.payload for frame in result.frames] == [b"following"]
    assert result.locations[0].candidate_index == 1
    assert result.locations[0].source_offset == len(bad_outer)
    assert [issue.code for issue in result.issues] == ["framing.crc_mismatch"]


def test_lone_trailing_sync_prefix_is_not_a_truncation_issue():
    result = parse_stream_diagnostic(b"noise\xec")
    assert result.frames == ()
    assert result.issues == ()


def test_input_hash_is_exact_and_independent_of_source_label():
    stream = make_frame(b"same")
    first = parse_stream_diagnostic(stream, source="first")
    second = parse_stream_diagnostic(stream, source="second")

    assert first.input_size_bytes == len(stream)
    assert first.input_sha256 == hashlib.sha256(stream).hexdigest()
    assert second.input_sha256 == first.input_sha256
    assert first.source == "first"
    assert second.source == "second"


def test_result_and_location_are_frozen_and_replayable():
    result = parse_stream_diagnostic(make_frame(b"frame"))
    assert isinstance(result.frames, tuple)
    assert isinstance(result.locations, tuple)
    assert tuple(result.frames) == result.frames
    with pytest.raises(FrozenInstanceError):
        result.source = "changed"
    with pytest.raises(FrozenInstanceError):
        result.locations[0].source_offset = 99


def test_public_iterator_remains_lazy_past_first_accepted_frame():
    good = make_frame(b"good")
    later_bad = make_frame(b"bad", corrupt_crc=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iterator = parse_stream(good + later_bad, source="lazy")
        assert caught == []
        assert next(iterator).payload == b"good"
        assert caught == []
        iterator.close()
    assert caught == []


def test_explicit_collector_receives_legacy_iterator_issues():
    collector = IssueCollector()
    stream = make_frame(b"bad", corrupt_crc=True)

    with pytest.warns(RuntimeWarning):
        assert list(
            parse_stream(
                stream,
                bank="b09",
                issue_collector=collector,
            )
        ) == []

    assert collector.issues[0].bank == "b09"
    assert collector.issues[0].code == "framing.crc_mismatch"


def test_strict_collector_raises_on_first_framing_issue():
    collector = IssueCollector(IssuePolicy.STRICT)
    stream = make_frame(b"bad", corrupt_crc=True)

    with pytest.raises(IngestIssueError) as caught:
        parse_stream_diagnostic(stream, issue_collector=collector)

    assert caught.value.issue is collector.issues[0]
    assert caught.value.issue.code == "framing.crc_mismatch"


def test_file_diagnostic_matches_stream_diagnostic(tmp_path):
    stream = make_frame(b"one") + make_frame(b"two", sequence=1)
    path = tmp_path / "FFFFFFFE"
    path.write_bytes(stream)

    legacy = tuple(parse_bank_file(path))
    from_file = parse_bank_file_diagnostic(path)
    from_stream = parse_stream_diagnostic(stream, source=str(path))

    assert from_file == from_stream
    assert from_file.frames == legacy


def test_malformed_header_branch_has_matching_warning_and_issue(monkeypatch):
    def reject_header(_buf):
        raise ValueError("synthetic rejection")

    monkeypatch.setattr(ccsds, "parse_primary_header", reject_header)
    stream = SYNC_BYTES + b"\x00" * 6

    with pytest.warns(RuntimeWarning, match="malformed CCSDS header"):
        result = parse_stream_diagnostic(stream, source="malformed")

    assert result.frames == ()
    assert result.issues[0].code == "framing.malformed_header"
    assert result.issues[0].byte_offset == 0
