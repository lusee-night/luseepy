"""Stage 2 logical-packet reassembly with no decoder dependency."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .ccsds import CcsdsFrame
from .issues import IssueAction, IssueCollector, IssueSeverity


@dataclass
class LogicalPacket:
    appid: int
    start_seq: int
    seq: int
    blob: bytes
    single_packet: bool
    unique_packet_id: Optional[int] = None
    bank: Optional[str] = None
    file_index: Optional[int] = None
    waveform_association_checked: bool = False
    waveform_transport_uncertain: bool = False
    waveform_metadata_source_order: Optional[int] = None


def _byteswap16(payload: bytes) -> bytes:
    if len(payload) % 2:
        raise ValueError(
            f"science payload must be even-length for 16-bit byteswap "
            f"(got {len(payload)})"
        )
    out = bytearray(len(payload))
    out[0::2] = payload[1::2]
    out[1::2] = payload[0::2]
    return bytes(out)


def reassemble_logical_packets(
    frames: Iterable[CcsdsFrame],
    *,
    byteswap_pairs: bool,
    bank: Optional[str] = None,
    issue_collector: IssueCollector | None = None,
) -> Iterator[LogicalPacket]:
    """Stage 2: turn CCSDS frames into logical packets.

    ``byteswap_pairs`` is True for science banks (b05..b09) and False for
    the DCB telemetry bank (b01). Termination follows the current LuSEE
    mission profile: a logical packet ends on ``groupflags == 1`` or ``== 3``.
    """
    buf = bytearray()
    start_seq: Optional[int] = None
    current_appid: Optional[int] = None
    logical_packet_index = 0
    last_frame_index: int | None = None
    last_sequence_count: int | None = None

    if issue_collector is not None and not isinstance(
        issue_collector, IssueCollector
    ):
        raise TypeError("issue_collector must be an IssueCollector or None")

    def record_issue(
        *,
        code: str,
        message: str,
        action: IssueAction,
        frame_index: int | None,
        appid: int | None,
        sequence_count: int | None,
        details: dict[str, object] | None = None,
    ) -> None:
        if issue_collector is None:
            return
        issue_collector.record(
            code=code,
            severity=IssueSeverity.WARNING,
            stage="reassembly",
            message=message,
            action=action,
            bank=bank,
            frame_index=frame_index,
            packet_index=logical_packet_index,
            appid=appid,
            sequence_count=sequence_count,
            details=details,
        )

    def take_payload(payload: bytes) -> bytes:
        return _byteswap16(payload) if byteswap_pairs else payload

    for frame_index, frame in enumerate(frames):
        hdr = frame.header
        last_frame_index = frame_index
        last_sequence_count = hdr.sequence_cnt
        if start_seq is None:
            start_seq = hdr.sequence_cnt
            current_appid = hdr.appid
        elif hdr.appid != current_appid:
            message = (
                f"APID changed mid logical packet "
                f"(was 0x{current_appid:03x}, now 0x{hdr.appid:03x}); "
                f"continuing accumulation"
            )
            record_issue(
                code="reassembly.apid_changed_mid_packet",
                message=message,
                action=IssueAction.KEPT,
                frame_index=frame_index,
                appid=current_appid,
                sequence_count=hdr.sequence_cnt,
                details={"new_appid": hdr.appid},
            )
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            buf.extend(take_payload(frame.payload))
        except ValueError as exc:
            message = (
                f"discarding logical packet (bank={bank}, "
                f"appid=0x{(current_appid or 0):03x}): {exc}"
            )
            record_issue(
                code="reassembly.invalid_science_payload_length",
                message=message,
                action=IssueAction.DROPPED,
                frame_index=frame_index,
                appid=current_appid,
                sequence_count=hdr.sequence_cnt,
                details={
                    "accumulated_payload_bytes": len(buf),
                    "frame_payload_bytes": len(frame.payload),
                },
            )
            warnings.warn(
                message,
                RuntimeWarning,
                stacklevel=2,
            )
            buf.clear()
            start_seq = None
            current_appid = None
            logical_packet_index += 1
            continue

        groupflags = hdr.groupflags
        if groupflags in (1, 3):
            yield LogicalPacket(
                appid=current_appid,  # type: ignore[arg-type]
                start_seq=start_seq,  # type: ignore[arg-type]
                seq=hdr.sequence_cnt,
                blob=bytes(buf),
                single_packet=(groupflags == 3),
                bank=bank,
            )
            buf.clear()
            start_seq = None
            current_appid = None
            logical_packet_index += 1

    if buf:
        message = (
            f"trailing partial logical packet (bank={bank}, "
            f"appid=0x{(current_appid or 0):03x}); discarding"
        )
        record_issue(
            code="reassembly.trailing_partial_packet",
            message=message,
            action=IssueAction.DROPPED,
            frame_index=last_frame_index,
            appid=current_appid,
            sequence_count=last_sequence_count,
            details={"accumulated_payload_bytes": len(buf)},
        )
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=2,
        )
