"""Stages 2-3: logical packet reassembly and identity assignment.

A logical packet is the multi-CCSDS-frame entity that the producer
emits as a single unit. Stage 2 walks per-bank streams of CCSDS frames
and concatenates segmented payloads, applying a 16-bit byte-swap on
science banks to compensate for an FPGA byte-order defect. Stage 3
assigns ``unique_packet_id`` to each logical packet by where its uid
comes from:

* uid-prefixed: blob[0:4] is the uid as little-endian uint32.
* uid-typed:    a typed C-struct header carries the uid; uncrater
                decodes it.
* uid-derived:  no embedded uid; inherit from the most recent preceding
                uid-prefixed or uid-typed packet.

AppID constants and the per-APID predicates come from the ``uncrater``
package (which itself wraps ``pycoreloop`` and honors ``CORELOOP_DIR``).
This is the single coreloop integration point for the whole pipeline.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, List, Optional

from uncrater import (
    Packet,
    appid_is_cal_any,
    appid_is_grimm_spectrum,
    appid_is_heartbeat,
    appid_is_hello,
    appid_is_housekeeping,
    appid_is_metadata,
    appid_is_raw_adc,
    appid_is_spectrum,
    appid_is_tr_spectrum,
    appid_is_watchdog,
    appid_is_zoom_spectrum,
)
from uncrater.coreloop import pycoreloop

appId = pycoreloop.appId

from .ccsds import CcsdsFrame

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AppID constants -- sourced from coreloop via uncrater
# ---------------------------------------------------------------------------

APID_HK            = appId.AppID_uC_Housekeeping
APID_EOS           = appId.AppID_End_Of_Sequence
APID_BOOTLOADER    = appId.AppID_uC_Bootloader
APID_HELLO         = appId.AppID_uC_Start
APID_HEARTBEAT     = appId.AppID_uC_Heartbeat
APID_WATCHDOG      = appId.AppID_Watchdog
APID_METADATA      = appId.AppID_MetaData
APID_SPECTRA_HIGH  = appId.AppID_SpectraHigh
APID_SPECTRA_MED   = appId.AppID_SpectraMed
APID_SPECTRA_LOW   = appId.AppID_SpectraLow
APID_TR_HIGH       = appId.AppID_SpectraTRHigh
APID_TR_MED        = appId.AppID_SpectraTRMed
APID_TR_LOW        = appId.AppID_SpectraTRLow
APID_ZOOM          = appId.AppID_ZoomSpectra
APID_CAL_METADATA  = appId.AppID_Calibrator_MetaData
APID_CAL_DATA      = appId.AppID_Calibrator_Data
APID_CAL_RAW_PFB   = appId.AppID_Calibrator_RawPFB
APID_CAL_DEBUG     = appId.AppID_Calibrator_Debug
APID_GRIMM         = appId.AppID_SpectraGrimm
APID_RAW_ADC       = appId.AppID_RawADC
APID_RAW_ADC_META  = appId.AppID_RawADC_Meta


# ---------------------------------------------------------------------------
# uid-source predicates -- compositions over uncrater's appid_is_* helpers
# ---------------------------------------------------------------------------

def is_uid_prefixed(appid: int) -> bool:
    """APIDs whose blob[0:4] is a little-endian uint32 unique_packet_id."""
    return (
        appid_is_spectrum(appid)
        or appid_is_tr_spectrum(appid)
        or appid_is_zoom_spectrum(appid)
        or appid_is_grimm_spectrum(appid)
        or appid_is_cal_any(appid)
    )


def is_uid_typed(appid: int) -> bool:
    """APIDs whose blob starts with a typed C-struct header carrying the uid."""
    return (
        appid_is_hello(appid)
        or appid_is_housekeeping(appid)
        or appid_is_metadata(appid)
        or appid == APID_CAL_METADATA
        or appid == APID_RAW_ADC_META
    )


def is_uid_derived(appid: int) -> bool:
    """APIDs whose uid is inherited from a preceding uid-prefixed/typed packet."""
    return (
        appid_is_raw_adc(appid)
        or appid_is_watchdog(appid)
        or appid == APID_BOOTLOADER
        or appid == APID_EOS
    )


def is_dropped_appid(appid: int) -> bool:
    """APIDs deliberately dropped from the science stream."""
    return appid_is_heartbeat(appid)


# ---------------------------------------------------------------------------
# Logical packet record
# ---------------------------------------------------------------------------

@dataclass
class LogicalPacket:
    appid: int
    start_seq: int
    seq: int
    blob: bytes
    single_packet: bool
    unique_packet_id: Optional[int] = None
    bank: Optional[str] = None    # "b05".."b09" or "b01"; informational
    file_index: Optional[int] = None  # index within the source bank stream


# ---------------------------------------------------------------------------
# Stage 2: reassembly
# ---------------------------------------------------------------------------

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
    the DCB telemetry bank (b01). Termination follows the standard CCSDS
    rule: a logical packet ends on ``groupflags == 1`` or ``== 3``.
    """
    buf = bytearray()
    start_seq: Optional[int] = None
    current_appid: Optional[int] = None

    def take_payload(p: bytes) -> bytes:
        return _byteswap16(p) if byteswap_pairs else p

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

        gf = hdr.groupflags
        if gf in (1, 3):
            yield LogicalPacket(
                appid=current_appid,    # type: ignore[arg-type]
                start_seq=start_seq,    # type: ignore[arg-type]
                seq=hdr.sequence_cnt,
                blob=bytes(buf),
                single_packet=(gf == 3),
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


# ---------------------------------------------------------------------------
# Stage 3: identity assignment
# ---------------------------------------------------------------------------

TypedUidExtractor = Callable[[int, bytes, Optional[int]], Optional[int]]
"""Signature: (appid, blob, sw_version) -> unique_packet_id or None."""


def _uncrater_typed_uid_extractor(appid: int, blob: bytes, sw_version: Optional[int]) -> Optional[int]:
    """Default uid-typed extractor; reads the typed C-struct via uncrater."""
    try:
        pkt = Packet(appid, blob=blob, version=sw_version)
        pkt._read()
        return int(getattr(pkt, "unique_packet_id"))
    except Exception as exc:    # noqa: BLE001
        warnings.warn(
            f"failed to extract unique_packet_id from uid-typed packet "
            f"(appid=0x{appid:03x}): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def detect_sw_version(packets: Iterable[LogicalPacket]) -> Optional[int]:
    """Scan packets for the first Hello and report its SW_version.

    Used to seed uid-typed extraction. Returns None if no Hello is present.
    """
    for p in packets:
        if p.appid == APID_HELLO:
            try:
                hello = Packet(p.appid, blob=p.blob)
                hello._read()
                return int(getattr(hello, "SW_version"))
            except Exception as exc:    # noqa: BLE001
                warnings.warn(
                    f"failed to read SW_version from Hello: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
    return None


def assign_identities(
    packets: List[LogicalPacket],
    *,
    sw_version: Optional[int] = None,
    typed_uid_extractor: Optional[TypedUidExtractor] = None,
    sort: bool = True,
) -> List[LogicalPacket]:
    """Stage 3: assign ``unique_packet_id`` to each logical packet.

    Pass 1 extracts ids from uid-prefixed and uid-typed packets; pass 2
    derives ids onto uid-derived packets from the most recent preceding
    uid-prefixed/typed packet. Packets with no extractable id, dropped
    APIDs (heartbeat), or unrecognised APIDs are removed from the
    returned list.

    With ``sort=True`` (default), the returned list is sorted by
    ``(unique_packet_id, seq)`` -- the canonical chronological order.
    """
    if typed_uid_extractor is None:
        typed_uid_extractor = _uncrater_typed_uid_extractor
    if sw_version is None:
        sw_version = detect_sw_version(packets)

    # Pass 1: explicit extraction
    for pkt in packets:
        if is_dropped_appid(pkt.appid):
            pkt.unique_packet_id = None
            continue
        if is_uid_prefixed(pkt.appid):
            if len(pkt.blob) < 4:
                warnings.warn(
                    f"uid-prefixed packet too short for u32 uid "
                    f"(appid=0x{pkt.appid:03x}, len={len(pkt.blob)})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            pkt.unique_packet_id = int.from_bytes(pkt.blob[0:4], "little")
        elif is_uid_typed(pkt.appid):
            pkt.unique_packet_id = typed_uid_extractor(pkt.appid, pkt.blob, sw_version)

    # Pass 2: derive uid for uid-derived packets
    last_id: Optional[int] = None
    for pkt in packets:
        if pkt.unique_packet_id is not None:
            last_id = pkt.unique_packet_id
        elif is_uid_derived(pkt.appid) and last_id is not None:
            pkt.unique_packet_id = last_id
        # otherwise leave None -> filtered below

    kept = [p for p in packets if p.unique_packet_id is not None]
    n_dropped = len(packets) - len(kept)
    if n_dropped:
        log.info("dropped %d packet(s) with no extractable unique_packet_id", n_dropped)

    if sort:
        kept.sort(key=lambda p: (p.unique_packet_id, p.seq))
    return kept
