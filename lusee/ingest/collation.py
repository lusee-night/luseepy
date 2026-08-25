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

AppID constants and the per-APID predicates come from the public ``uncrater``
API through :mod:`lusee.ingest.uncrater_adapter`.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Iterable, List, Optional

from .issues import IssueAction, IssueCollector, IssueSeverity
from .reassembly import LogicalPacket, reassemble_logical_packets
from .uncrater_adapter import load_uncrater, read_packet

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility AppID constants -- resolved lazily from public uncrater.id
# ---------------------------------------------------------------------------

_APID_ATTRIBUTES = {
    "APID_HK": "AppID_uC_Housekeeping",
    "APID_EOS": "AppID_End_Of_Sequence",
    "APID_BOOTLOADER": "AppID_uC_Bootloader",
    "APID_HELLO": "AppID_uC_Start",
    "APID_HEARTBEAT": "AppID_uC_Heartbeat",
    "APID_WATCHDOG": "AppID_Watchdog",
    "APID_METADATA": "AppID_MetaData",
    "APID_SPECTRA_HIGH": "AppID_SpectraHigh",
    "APID_SPECTRA_MED": "AppID_SpectraMed",
    "APID_SPECTRA_LOW": "AppID_SpectraLow",
    "APID_TR_HIGH": "AppID_SpectraTRHigh",
    "APID_TR_MED": "AppID_SpectraTRMed",
    "APID_TR_LOW": "AppID_SpectraTRLow",
    "APID_ZOOM": "AppID_ZoomSpectra",
    "APID_CAL_METADATA": "AppID_Calibrator_MetaData",
    "APID_CAL_DATA": "AppID_Calibrator_Data",
    "APID_CAL_RAW_PFB": "AppID_Calibrator_RawPFB",
    "APID_CAL_DEBUG": "AppID_Calibrator_Debug",
    "APID_GRIMM": "AppID_SpectraGrimm",
    "APID_RAW_ADC": "AppID_RawADC",
    "APID_RAW_ADC_META": "AppID_RawADC_Meta",
}


def __getattr__(name: str):
    attribute = _APID_ATTRIBUTES.get(name)
    if attribute is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = int(getattr(load_uncrater().id, attribute))
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_APID_ATTRIBUTES))


# ---------------------------------------------------------------------------
# uid-source predicates -- compositions over uncrater's appid_is_* helpers
# ---------------------------------------------------------------------------

def is_uid_prefixed(appid: int) -> bool:
    """APIDs whose blob[0:4] is a little-endian uint32 unique_packet_id."""
    decoder = load_uncrater()
    return (
        decoder.appid_is_spectrum(appid)
        or decoder.appid_is_tr_spectrum(appid)
        or decoder.appid_is_zoom_spectrum(appid)
        or decoder.appid_is_grimm_spectrum(appid)
        or decoder.appid_is_cal_segmented_payload(appid)
    )


def is_uid_typed(appid: int) -> bool:
    """APIDs whose blob starts with a typed C-struct header carrying the uid."""
    decoder = load_uncrater()
    return (
        decoder.appid_is_hello(appid)
        or decoder.appid_is_housekeeping(appid)
        or decoder.appid_is_metadata(appid)
        or decoder.appid_is_cal_metadata(appid)
        or decoder.appid_is_raw_adc_metadata(appid)
    )


def is_uid_derived(appid: int) -> bool:
    """APIDs whose uid is inherited from a preceding uid-prefixed/typed packet."""
    decoder = load_uncrater()
    return (
        decoder.appid_is_raw_adc(appid)
        or decoder.appid_is_watchdog(appid)
        or appid == int(decoder.id.AppID_uC_Bootloader)
        or appid == int(decoder.id.AppID_End_Of_Sequence)
    )


def is_dropped_appid(appid: int) -> bool:
    """APIDs deliberately dropped from the science stream."""
    return load_uncrater().appid_is_heartbeat(appid)


# ---------------------------------------------------------------------------
# Stage 3: identity assignment
# ---------------------------------------------------------------------------

TypedUidExtractor = Callable[[int, bytes, Optional[int]], Optional[int]]
"""Signature: (appid, blob, sw_version) -> unique_packet_id or None."""


def _packet_index(packet: LogicalPacket, fallback: int) -> int:
    if type(packet.file_index) is int and packet.file_index >= 0:
        return packet.file_index
    return fallback


def _record_identity_issue(
    issue_collector: IssueCollector | None,
    *,
    code: str,
    severity: IssueSeverity,
    message: str,
    action: IssueAction,
    packet: LogicalPacket | None = None,
    packet_index: int | None = None,
    appid: int | None = None,
    sequence_count: int | None = None,
    details: dict[str, object] | None = None,
) -> None:
    if issue_collector is None:
        return
    issue_collector.record(
        code=code,
        severity=severity,
        stage="identity",
        message=message,
        action=action,
        bank=None if packet is None else packet.bank,
        packet_index=packet_index,
        appid=appid if packet is None else packet.appid,
        sequence_count=(
            sequence_count if packet is None else packet.seq
        ),
        uid=None if packet is None else packet.unique_packet_id,
        details=details,
    )


def _uncrater_typed_uid_extractor(
    appid: int,
    blob: bytes,
    sw_version: Optional[int],
    *,
    issue_collector: IssueCollector | None = None,
    packet: LogicalPacket | None = None,
    packet_index: int | None = None,
) -> Optional[int]:
    """Default uid-typed extractor; reads the typed C-struct via uncrater."""
    try:
        pkt = load_uncrater().Packet(appid, blob=blob, version=sw_version)
        read_packet(pkt)
        return int(getattr(pkt, "unique_packet_id"))
    except Exception as exc:    # noqa: BLE001
        message = (
            f"failed to extract unique_packet_id from uid-typed packet "
            f"(appid=0x{appid:03x}): {exc}"
        )
        _record_identity_issue(
            issue_collector,
            code="identity.typed_uid_extraction_failed",
            severity=IssueSeverity.WARNING,
            message=message,
            action=IssueAction.REJECTED,
            packet=packet,
            packet_index=packet_index,
            appid=appid,
            details={
                "exception_type": type(exc).__name__,
                "sw_version": sw_version,
            },
        )
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def detect_sw_version(
    packets: Iterable[LogicalPacket],
    *,
    issue_collector: IssueCollector | None = None,
) -> Optional[int]:
    """Scan packets for the first Hello and report its SW_version.

    Used to seed uid-typed extraction. Returns None if no Hello is present.
    """
    if issue_collector is not None and not isinstance(
        issue_collector, IssueCollector
    ):
        raise TypeError("issue_collector must be an IssueCollector or None")
    decoder = load_uncrater()
    for fallback_index, p in enumerate(packets):
        if decoder.appid_is_hello(p.appid):
            try:
                hello = decoder.Packet(p.appid, blob=p.blob)
                read_packet(hello)
                return int(getattr(hello, "SW_version"))
            except Exception as exc:    # noqa: BLE001
                message = f"failed to read SW_version from Hello: {exc}"
                _record_identity_issue(
                    issue_collector,
                    code="identity.hello_sw_version_decode_failed",
                    severity=IssueSeverity.WARNING,
                    message=message,
                    action=IssueAction.KEPT,
                    packet=p,
                    packet_index=_packet_index(p, fallback_index),
                    details={"exception_type": type(exc).__name__},
                )
                warnings.warn(
                    message,
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
    return None


def assign_identities(
    packets: List[LogicalPacket],
    *,
    sw_version: Optional[int] = None,
    auto_detect_sw_version: bool = True,
    typed_uid_extractor: Optional[TypedUidExtractor] = None,
    sort: bool = True,
    issue_collector: IssueCollector | None = None,
) -> List[LogicalPacket]:
    """Stage 3: assign ``unique_packet_id`` to each logical packet.

    Pass 1 extracts ids from uid-prefixed and uid-typed packets; pass 2
    derives ids onto uid-derived packets from the most recent preceding
    uid-prefixed/typed packet. Packets with no extractable id, dropped
    APIDs (heartbeat), or unrecognised APIDs are removed from the
    returned list.

    With ``sort=True`` (default), the returned list is sorted by
    ``(unique_packet_id, seq)`` using the existing deterministic heuristic.
    This is not a canonical or global chronological order.

    Set ``auto_detect_sw_version=False`` when detection was already attempted
    and returned ``None``.
    """
    if issue_collector is not None and not isinstance(
        issue_collector, IssueCollector
    ):
        raise TypeError("issue_collector must be an IssueCollector or None")
    use_default_typed_extractor = typed_uid_extractor is None
    if sw_version is None and auto_detect_sw_version:
        sw_version = detect_sw_version(
            packets,
            issue_collector=issue_collector,
        )

    drop_reasons: list[str | None] = [None] * len(packets)

    # Pass 1: explicit extraction
    for fallback_index, pkt in enumerate(packets):
        packet_index = _packet_index(pkt, fallback_index)
        if is_dropped_appid(pkt.appid):
            pkt.unique_packet_id = None
            drop_reasons[fallback_index] = "intentionally_filtered_appid"
            continue
        if is_uid_prefixed(pkt.appid):
            if len(pkt.blob) < 4:
                message = (
                    f"uid-prefixed packet too short for u32 uid "
                    f"(appid=0x{pkt.appid:03x}, len={len(pkt.blob)})"
                )
                _record_identity_issue(
                    issue_collector,
                    code="identity.uid_prefix_too_short",
                    severity=IssueSeverity.WARNING,
                    message=message,
                    action=IssueAction.REJECTED,
                    packet=pkt,
                    packet_index=packet_index,
                    details={"blob_length": len(pkt.blob), "required_length": 4},
                )
                warnings.warn(
                    message,
                    RuntimeWarning,
                    stacklevel=2,
                )
                drop_reasons[fallback_index] = "invalid_uid_prefix"
                continue
            pkt.unique_packet_id = int.from_bytes(pkt.blob[0:4], "little")
        elif is_uid_typed(pkt.appid):
            if use_default_typed_extractor:
                pkt.unique_packet_id = _uncrater_typed_uid_extractor(
                    pkt.appid,
                    pkt.blob,
                    sw_version,
                    issue_collector=issue_collector,
                    packet=pkt,
                    packet_index=packet_index,
                )
            else:
                assert typed_uid_extractor is not None
                pkt.unique_packet_id = typed_uid_extractor(
                    pkt.appid, pkt.blob, sw_version
                )
            if pkt.unique_packet_id is None:
                drop_reasons[fallback_index] = "typed_uid_unavailable"

    # Pass 2: derive uid for uid-derived packets
    last_id: Optional[int] = None
    for fallback_index, pkt in enumerate(packets):
        if pkt.unique_packet_id is not None:
            last_id = pkt.unique_packet_id
        elif is_uid_derived(pkt.appid):
            if last_id is not None:
                pkt.unique_packet_id = last_id
                drop_reasons[fallback_index] = None
            else:
                drop_reasons[fallback_index] = "no_preceding_unique_packet_id"
        elif drop_reasons[fallback_index] is None:
            drop_reasons[fallback_index] = "unrecognized_appid"
        # otherwise leave None -> filtered below

    kept = [p for p in packets if p.unique_packet_id is not None]
    n_dropped = len(packets) - len(kept)
    if n_dropped:
        log.info("dropped %d packet(s) with no extractable unique_packet_id", n_dropped)
    for fallback_index, (pkt, reason) in enumerate(
        zip(packets, drop_reasons, strict=True)
    ):
        if pkt.unique_packet_id is not None:
            continue
        packet_index = _packet_index(pkt, fallback_index)
        if reason == "intentionally_filtered_appid":
            _record_identity_issue(
                issue_collector,
                code="identity.appid_intentionally_dropped",
                severity=IssueSeverity.INFO,
                message=(
                    f"packet AppID 0x{pkt.appid:03x} was intentionally dropped "
                    "from the science stream"
                ),
                action=IssueAction.DROPPED,
                packet=pkt,
                packet_index=packet_index,
                details={"reason": reason},
            )
            continue
        _record_identity_issue(
            issue_collector,
            code="identity.packet_without_uid_dropped",
            severity=IssueSeverity.WARNING,
            message=(
                f"packet AppID 0x{pkt.appid:03x} was dropped because no "
                "unique_packet_id could be assigned"
            ),
            action=IssueAction.DROPPED,
            packet=pkt,
            packet_index=packet_index,
            details={"reason": reason},
        )

    if sort:
        kept.sort(key=lambda p: (p.unique_packet_id, p.seq))
    return kept
