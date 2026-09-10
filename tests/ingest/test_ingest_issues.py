from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from lusee.ingest.issues import (
    IngestIssue,
    IngestIssueError,
    IssueAction,
    IssueCollector,
    IssuePolicy,
    IssueSeverity,
)


def test_collect_assigns_stable_ids_and_deterministic_records():
    collector = IssueCollector()
    marker = collector.mark()
    details = {"nested": {"values": [1, 2]}, "answer": 42}

    first = collector.record(
        code="framing.test",
        severity="warning",
        stage="framing",
        message="test issue",
        action="dropped",
        input_identity="sha256:abc",
        bank="b05",
        byte_offset=12,
        frame_index=3,
        appid=0x209,
        sequence_count=7,
        details=details,
    )
    second = collector.record(
        code="framing.test",
        severity=IssueSeverity.INFO,
        stage="framing",
        message="second issue",
        action=IssueAction.KEPT,
    )

    details["nested"]["values"].append(3)
    assert first.issue_id == "issue-00000001"
    assert second.issue_id == "issue-00000002"
    assert first.severity is IssueSeverity.WARNING
    assert first.action is IssueAction.DROPPED
    assert first.as_dict()["details"] == {
        "answer": 42,
        "nested": {"values": [1, 2]},
    }
    assert collector.since(marker) == (first, second)
    assert collector.issues == (first, second)
    assert collector.counts() == {"framing.test": 2}
    json.dumps(first.as_dict(), sort_keys=True)


def test_issue_and_nested_details_are_immutable():
    issue = IngestIssue(
        issue_id="issue-00000001",
        code="test.immutable",
        severity=IssueSeverity.WARNING,
        stage="test",
        message="immutable",
        action=IssueAction.KEPT,
        details=(("values", [1, {"two": 2}]),),
    )

    with pytest.raises(FrozenInstanceError):
        issue.message = "changed"
    with pytest.raises(AttributeError):
        issue.details[0][1].items += (3,)
    assert issue.as_dict()["details"] == {"values": [1, {"two": 2}]}


def test_strict_policy_records_then_raises_the_same_issue():
    collector = IssueCollector(IssuePolicy.STRICT)

    with pytest.raises(IngestIssueError) as caught:
        collector.record(
            code="framing.bad",
            severity="error",
            stage="framing",
            message="bad frame",
            action="rejected",
        )

    assert collector.issues == (caught.value.issue,)
    assert caught.value.issue.issue_id == "issue-00000001"
    assert "framing.bad" in str(caught.value)


def test_collectors_do_not_share_state():
    first = IssueCollector()
    second = IssueCollector()

    first_issue = first.record(
        code="one",
        severity="info",
        stage="test",
        message="one",
        action="kept",
    )
    second_issue = second.record(
        code="two",
        severity="info",
        stage="test",
        message="two",
        action="kept",
    )

    assert first.issues == (first_issue,)
    assert second.issues == (second_issue,)
    assert first_issue.issue_id == second_issue.issue_id == "issue-00000001"


@pytest.mark.parametrize(
    "details, exception",
    [
        ({"bad": float("nan")}, ValueError),
        ({"bad": b"bytes"}, TypeError),
        ({1: "non-string key"}, TypeError),
    ],
)
def test_non_json_issue_details_are_rejected(details, exception):
    collector = IssueCollector()
    with pytest.raises(exception):
        collector.record(
            code="bad.details",
            severity="warning",
            stage="test",
            message="bad details",
            action="rejected",
            details=details,
        )


def test_issue_marker_must_belong_to_current_collector_extent():
    collector = IssueCollector()
    with pytest.raises(ValueError, match="invalid issue marker"):
        collector.since(1)
