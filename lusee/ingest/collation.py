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


def _uncrater_typed_uid_extractor(appid: int, blob: bytes, sw_version: Optional[int]) -> Optional[int]:
    """Default uid-typed extractor; reads the typed C-struct via uncrater."""
    try:
        pkt = load_uncrater().Packet(appid, blob=blob, version=sw_version)
        read_packet(pkt)
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
    decoder = load_uncrater()
    for p in packets:
        if decoder.appid_is_hello(p.appid):
            try:
                hello = decoder.Packet(p.appid, blob=p.blob)
                read_packet(hello)
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
