"""Structured issues shared by the downlink ingest stages."""

from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class IssuePolicy(str, Enum):
    """How an issue collector responds when an issue is recorded."""

    COLLECT = "collect"
    STRICT = "strict"


class IssueSeverity(str, Enum):
    """Visibility level of an ingest issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IssueAction(str, Enum):
    """Action taken on the smallest unit affected by an issue."""

    KEPT = "kept"
    DROPPED = "dropped"
    REJECTED = "rejected"
    ASSUMED = "assumed"
    OVERRIDDEN = "overridden"


DetailScalar = bool | int | float | str | None


@dataclass(frozen=True)
class _FrozenMapping:
    items: tuple[tuple[str, "FrozenDetail"], ...]


@dataclass(frozen=True)
class _FrozenSequence:
    items: tuple["FrozenDetail", ...]


FrozenDetail = DetailScalar | _FrozenMapping | _FrozenSequence


def _freeze_detail(value: object) -> FrozenDetail:
    if isinstance(value, (_FrozenMapping, _FrozenSequence)):
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("issue detail floats must be finite")
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("issue detail mapping keys must be strings")
            items.append((key, _freeze_detail(item)))
        return _FrozenMapping(tuple(sorted(items)))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _FrozenSequence(tuple(_freeze_detail(item) for item in value))
    raise TypeError(
        "issue details must contain only JSON-compatible immutable values"
    )


def _thaw_detail(value: FrozenDetail) -> object:
    if isinstance(value, _FrozenMapping):
        return {key: _thaw_detail(item) for key, item in value.items}
    if isinstance(value, _FrozenSequence):
        return [_thaw_detail(item) for item in value.items]
    return value


def _freeze_details(
    details: Mapping[str, object] | None,
) -> tuple[tuple[str, FrozenDetail], ...]:
    if details is None:
        return ()
    frozen = []
    for key, value in details.items():
        if not isinstance(key, str):
            raise TypeError("issue detail keys must be strings")
        frozen.append((key, _freeze_detail(value)))
    return tuple(sorted(frozen))


@dataclass(frozen=True)
class IngestIssue:
    """One immutable ingest anomaly with stable run-local identity."""

    issue_id: str
    code: str
    severity: IssueSeverity
    stage: str
    message: str
    action: IssueAction
    input_identity: str | None = None
    bank: str | None = None
    byte_offset: int | None = None
    frame_index: int | None = None
    packet_index: int | None = None
    appid: int | None = None
    sequence_count: int | None = None
    uid: int | None = None
    session: str | None = None
    details: tuple[tuple[str, FrozenDetail], ...] = ()

    def __post_init__(self) -> None:
        if not self.issue_id:
            raise ValueError("issue id must not be empty")
        if not self.code:
            raise ValueError("issue code must not be empty")
        if not self.stage:
            raise ValueError("issue stage must not be empty")
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(self, "action", IssueAction(self.action))
        normalized_details = []
        seen_keys = set()
        for key, value in self.details:
            if not isinstance(key, str):
                raise TypeError("issue detail keys must be strings")
            if key in seen_keys:
                raise ValueError(f"duplicate issue detail key {key!r}")
            seen_keys.add(key)
            normalized_details.append((key, _freeze_detail(value)))
        object.__setattr__(self, "details", tuple(sorted(normalized_details)))

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible representation."""
        return {
            "issue_id": self.issue_id,
            "code": self.code,
            "severity": self.severity.value,
            "stage": self.stage,
            "message": self.message,
            "action": self.action.value,
            "input_identity": self.input_identity,
            "bank": self.bank,
            "byte_offset": self.byte_offset,
            "frame_index": self.frame_index,
            "packet_index": self.packet_index,
            "appid": self.appid,
            "sequence_count": self.sequence_count,
            "uid": self.uid,
            "session": self.session,
            "details": {
                key: _thaw_detail(value) for key, value in self.details
            },
        }


class IngestIssueError(RuntimeError):
    """Raised by an issue collector using the explicit strict policy."""

    def __init__(self, issue: IngestIssue):
        self.issue = issue
        super().__init__(f"{issue.issue_id} {issue.code}: {issue.message}")


class IssueCollector:
    """Collect ordered issues, or raise after recording the first issue."""

    def __init__(self, policy: IssuePolicy | str = IssuePolicy.COLLECT):
        self.policy = IssuePolicy(policy)
        self._issues: list[IngestIssue] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._issues)

    @property
    def issues(self) -> tuple[IngestIssue, ...]:
        with self._lock:
            return tuple(self._issues)

    def mark(self) -> int:
        """Return a marker that can later select issues from this point."""
        return len(self)

    def since(self, marker: int) -> tuple[IngestIssue, ...]:
        """Return the immutable issue slice recorded since ``marker``."""
        with self._lock:
            if marker < 0 or marker > len(self._issues):
                raise ValueError(f"invalid issue marker {marker}")
            return tuple(self._issues[marker:])

    def counts(self) -> dict[str, int]:
        """Return occurrence counts keyed by stable issue code."""
        with self._lock:
            counts = Counter(issue.code for issue in self._issues)
        return dict(sorted(counts.items()))

    def record(
        self,
        *,
        code: str,
        severity: IssueSeverity | str,
        stage: str,
        message: str,
        action: IssueAction | str,
        input_identity: str | None = None,
        bank: str | None = None,
        byte_offset: int | None = None,
        frame_index: int | None = None,
        packet_index: int | None = None,
        appid: int | None = None,
        sequence_count: int | None = None,
        uid: int | None = None,
        session: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> IngestIssue:
        """Record one issue and return the assigned immutable record."""
        if not code:
            raise ValueError("issue code must not be empty")
        if not stage:
            raise ValueError("issue stage must not be empty")
        frozen_details = _freeze_details(details)
        with self._lock:
            issue = IngestIssue(
                issue_id=f"issue-{len(self._issues) + 1:08d}",
                code=code,
                severity=IssueSeverity(severity),
                stage=stage,
                message=message,
                action=IssueAction(action),
                input_identity=input_identity,
                bank=bank,
                byte_offset=byte_offset,
                frame_index=frame_index,
                packet_index=packet_index,
                appid=appid,
                sequence_count=sequence_count,
                uid=uid,
                session=session,
                details=frozen_details,
            )
            self._issues.append(issue)
        if self.policy is IssuePolicy.STRICT:
            raise IngestIssueError(issue)
        return issue
