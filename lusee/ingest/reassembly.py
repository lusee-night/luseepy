"""Stage 2 logical-packet reassembly with no decoder dependency."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from .ccsds import CcsdsFrame


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
) -> Iterator[LogicalPacket]:
    """Stage 2: turn CCSDS frames into logical packets.

    ``byteswap_pairs`` is True for science banks (b05..b09) and False for
    the DCB telemetry bank (b01). Termination follows the current LuSEE
    mission profile: a logical packet ends on ``groupflags == 1`` or ``== 3``.
    """
    buf = bytearray()
    start_seq: Optional[int] = None
    current_appid: Optional[int] = None

    def take_payload(payload: bytes) -> bytes:
        return _byteswap16(payload) if byteswap_pairs else payload

    for frame in frames:
        hdr = frame.header
        if start_seq is None:
            start_seq = hdr.sequence_cnt
            current_appid = hdr.appid
        elif hdr.appid != current_appid:
            warnings.warn(
                f"APID changed mid logical packet "
                f"(was 0x{current_appid:03x}, now 0x{hdr.appid:03x}); "
                f"continuing accumulation",
                RuntimeWarning,
                stacklevel=2,
            )
        try:
            buf.extend(take_payload(frame.payload))
        except ValueError as exc:
            warnings.warn(
                f"discarding logical packet (bank={bank}, "
                f"appid=0x{(current_appid or 0):03x}): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            buf.clear()
            start_seq = None
            current_appid = None
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

    if buf:
        warnings.warn(
            f"trailing partial logical packet (bank={bank}, "
            f"appid=0x{(current_appid or 0):03x}); discarding",
            RuntimeWarning,
            stacklevel=2,
        )
