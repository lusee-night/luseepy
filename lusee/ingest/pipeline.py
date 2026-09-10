"""End-to-end orchestrators for the LuSEE-Night downlink pipeline.

Two top-level entry points:

* :func:`process_flash` -- Stage 1..7 single-pass: walks a FLASH_TLMFS
  directory, recovers sessions, persists them as uncrater session
  directories, decodes them in-memory, and writes one HDF5 per session
  (plus optional plots and a manifest).
* :func:`process_session` -- Stages 6..7: reads an existing uncrater
  session directory (with an optional legacy ``DCB_telemetry.json``
  sidecar) and writes one HDF5 (plus optional plots and a manifest).

All output paths are caller-supplied. ``lusee.ingest`` does not invent
on-disk layout; the example scripts pick a layout and supply paths.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import shutil
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import telemetry as telemetry_mod
from .ccsds import parse_bank_file, parse_bank_file_diagnostic
from .clock_reference import (
    ClockReferenceSet,
    ClockSource,
    clock_reference_set_from_record,
    load_clock_reference_set,
)
from .collation import (
    assign_identities,
    detect_sw_version,
    is_dropped_appid,
)
from .constants import (
    BANK_FILENAME,
    DEFAULT_LUN_HEIGHT_M,
    DEFAULT_LUN_LAT_DEG,
    DEFAULT_LUN_LONG_DEG,
    DEFAULT_SESSION_NAME_FMT,
    INGEST_LAYOUT_VERSION,
    SCIENCE_BANKS,
    SESSION_NAME_NO_TIME_FMT,
    SESSION_TIMESTAMP_FMT,
    SPECTRA_NORMALIZATION_VERSION,
    TELEMETRY_BANK,
)
from .decode import Products, read_uncrater_session
from .issues import (
    IngestIssue,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from .packet_map import PACKET_MAP_FILENAME
from .reassembly import LogicalPacket, reassemble_logical_packets
from .session import (
    Session,
    assign_telemetry_to_sessions,
    split_sessions,
    mark_waveform_transport_loss,
    write_uncrater_session,
)
from .write_request import (
    FAMILY_TYPES,
    LunarLocation,
    RunProvenance,
    WriteRequest,
    family_statuses_for_products,
)

log = logging.getLogger(__name__)

# Schema version of session.json -- bump when the manifest layout changes.
MANIFEST_SCHEMA_VERSION = 3

# Filename of the in-session manifest written into each session directory.
IN_SESSION_MANIFEST_NAME = "session.json"
FLASH_MANIFEST_NAME = "flash.json"
LEGACY_TELEMETRY_SIDECAR_NAME = "DCB_telemetry.json"
FLASH_FINGERPRINT_FORMAT_VERSION = 1
REASSEMBLY_PROFILE = "legacy"


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_overwrite(overwrite: object) -> bool:
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be a boolean")
    return overwrite


def _validate_flash_fingerprint(
    fingerprint: object,
) -> Dict[str, Dict[str, object]]:
    if not isinstance(fingerprint, dict):
        raise ValueError("flash source fingerprint must be a JSON object")
    normalized: Dict[str, Dict[str, object]] = {}
    for relative_path, raw_record in fingerprint.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("flash fingerprint paths must be safe relative paths")
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "size_bytes",
            "sha256",
        }:
            raise ValueError(
                "flash fingerprint records require size_bytes and sha256"
            )
        size = raw_record["size_bytes"]
        digest = raw_record["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError("flash fingerprint size_bytes must be nonnegative")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("flash fingerprint sha256 must be lowercase 64-hex")
        normalized[relative_path] = {
            "size_bytes": size,
            "sha256": digest,
        }
    return dict(sorted(normalized.items()))


def _validate_legacy_flash_fingerprint(
    fingerprint: object,
) -> Dict[str, Dict[str, object]]:
    if not isinstance(fingerprint, dict):
        raise ValueError("legacy flash fingerprint must be a JSON object")
    normalized: Dict[str, Dict[str, object]] = {}
    for relative_path, raw_record in fingerprint.items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError(
                "legacy flash fingerprint paths must be safe relative paths"
            )
        if not isinstance(raw_record, dict) or set(raw_record) != {
            "size",
            "mtime",
        }:
            raise ValueError(
                "legacy flash fingerprint records require size and mtime"
            )
        size = raw_record["size"]
        mtime = raw_record["mtime"]
        if type(size) is not int or size < 0:
            raise ValueError("legacy flash fingerprint size must be nonnegative")
        if type(mtime) not in (int, float) or not np.isfinite(mtime):
            raise ValueError("legacy flash fingerprint mtime must be finite")
        normalized[relative_path] = {
            "size": size,
            "mtime": float(mtime),
        }
    return dict(sorted(normalized.items()))


# ---------------------------------------------------------------------------
# SessionResult
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    """Summary of one processed session, intended for manifest serialization."""

    session_ordinal: int
    session_name: str
    source_path: str
    source_kind: str   # "flash" or "session"

    start_time_utc: Optional[str] = None
    start_unique_packet_id: Optional[int] = None
    software_version: Optional[int] = None
    firmware_version: Optional[int] = None

    input_packet_count: int = 0
    decoder_input_packet_count: int = 0
    decoder_valid_packet_count: int = 0
    decoder_invalid_packet_count: int = 0
    n_product_rows: int = 0
    decoded_rows_by_family: Dict[str, int] = field(default_factory=dict)
    persisted_rows_by_family: Dict[str, int] = field(default_factory=dict)

    # Retained on the Python object for compatibility; manifest v3 uses the
    # unambiguous n_product_rows field instead.
    n_packets: int = 0
    n_spectra: int = 0
    n_tr_spectra: int = 0
    n_zoom_spectra: int = 0
    n_grimm_spectra: int = 0
    n_waveforms: int = 0
    n_housekeeping: int = 0

    n_warnings: int = 0
    warnings_summary: List[str] = field(default_factory=list)
    status: str = "clean"
    status_issue_codes: List[str] = field(default_factory=list)
    issue_counts: Dict[str, int] = field(default_factory=dict)
    issues: List[Dict[str, object]] = field(default_factory=list)
    stage_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    family_statuses: List[Dict[str, object]] = field(default_factory=list)
    decoder_provenance: Dict[str, object] = field(default_factory=dict)
    output_artifacts: Dict[str, Dict[str, object]] = field(default_factory=dict)
    contracts: Dict[str, object] = field(default_factory=dict)
    failure: Optional[Dict[str, object]] = None

    h5_path: Optional[str] = None
    fits_path: Optional[str] = None
    session_dir: Optional[str] = None
    plot_paths: List[str] = field(default_factory=list)
    manifest_path: Optional[str] = None

    # Raw FLASH provenance retained for traceability
    flash_source_path: Optional[str] = None
    flash_source_fingerprint: Dict[str, Dict[str, object]] = field(
        default_factory=dict
    )
    flash_result_id: Optional[str] = None
    flash_manifest_sha256: Optional[str] = None
    start_raw_seconds: Optional[float] = None
    telemetry_source: Optional[str] = None
    telemetry_status: str = "absent"
    telemetry_reason: Optional[str] = None
    n_telemetry_rows: int = 0

    processed_at_utc: str = ""
    pipeline_version: str = ""
    clock_reference: Optional[Dict[str, object]] = None
    packet_map_status: str = "unavailable"
    packet_map_format_version: Optional[int] = None
    raw_flash_provenance_unavailable_reason: Optional[str] = None
    overwrite: bool = False
    committed_artifacts: List[str] = field(default_factory=list)
    flash_input_identity_sha256: Optional[str] = None
    flash_input_identity_unavailable_reason: Optional[str] = None


@dataclass
class FlashResult:
    """Run-level result for one FLASH input, sequence-compatible by session."""

    flash_result_id: str
    input_identity_sha256: Optional[str]
    source_path: str
    source_fingerprint: Dict[str, Dict[str, object]]
    input_identity_unavailable_reason: Optional[str] = None
    session_results: List[SessionResult] = field(default_factory=list)
    status: str = "clean"
    status_issue_codes: List[str] = field(default_factory=list)
    issue_counts: Dict[str, int] = field(default_factory=dict)
    issues: List[Dict[str, object]] = field(default_factory=list)
    stage_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    decoder_provenance: List[Dict[str, object]] = field(default_factory=list)
    clock_reference: Optional[Dict[str, object]] = None
    reassembly_profile: str = REASSEMBLY_PROFILE
    pipeline_version: str = ""
    processed_at_utc: str = ""
    manifest_paths: List[str] = field(default_factory=list)
    manifest_sha256: Optional[str] = None
    failure: Optional[Dict[str, object]] = None
    overwrite: bool = False
    source_identity_unavailable_reasons: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.session_results)

    def __iter__(self):
        return iter(self.session_results)

    def __getitem__(self, index):
        return self.session_results[index]


@dataclass
class _FlashParseCapture:
    source_fingerprint: Dict[str, Dict[str, object]] = field(default_factory=dict)
    source_identity_complete: bool = True
    source_identity_unavailable_reasons: List[str] = field(default_factory=list)
    input_bank_files: int = 0
    input_bytes: int = 0
    recovered_frames: int = 0
    science_logical_packets: int = 0
    telemetry_logical_packets: int = 0
    telemetry_0x314_packets: int = 0
    identity_input_packets: int = 0
    identity_kept_packets: int = 0
    identity_policy_dropped_packets: int = 0
    session_count: int = 0
    session_input_packets: int = 0

    def record_bank(
        self,
        *,
        bank: str,
        size_bytes: int,
        sha256: str,
        recovered_frames: int,
    ) -> None:
        key = f"{bank}/{BANK_FILENAME}"
        self.source_fingerprint[key] = {
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
        self.input_bank_files += 1
        self.input_bytes += size_bytes
        self.recovered_frames += recovered_frames

    def record_unreadable_bank(self, *, bank: str, reason: str) -> None:
        self.source_identity_complete = False
        value = f"{bank}:{reason}"
        if value not in self.source_identity_unavailable_reasons:
            self.source_identity_unavailable_reasons.append(value)

    def stage_count_record(
        self,
        issues: Sequence[IngestIssue],
    ) -> Dict[str, Dict[str, int]]:
        issue_actions = Counter(
            (issue.stage, issue.action.value) for issue in issues
        )
        issue_stages = Counter(issue.stage for issue in issues)
        return {
            "input": {
                "bank_files": self.input_bank_files,
                "bytes": self.input_bytes,
            },
            "framing": {
                "recovered_frames": self.recovered_frames,
                "issues": issue_stages.get("framing", 0),
                "dropped_frames": issue_actions.get(("framing", "dropped"), 0),
            },
            "reassembly": {
                "logical_packets": (
                    self.science_logical_packets
                    + self.telemetry_logical_packets
                ),
                "science_logical_packets": self.science_logical_packets,
                "telemetry_logical_packets": self.telemetry_logical_packets,
                "dropped_packets": issue_actions.get(
                    ("reassembly", "dropped"), 0
                ),
            },
            "identity": {
                "input_packets": self.identity_input_packets,
                "kept_packets": self.identity_kept_packets,
                "dropped_packets": (
                    self.identity_input_packets - self.identity_kept_packets
                ),
                "policy_dropped_packets": self.identity_policy_dropped_packets,
            },
            "session_split": {
                "sessions": self.session_count,
                "input_packets": self.session_input_packets,
            },
        }


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def default_session_name(
    ordinal: int,
    start_raw_seconds: Optional[float],
    clock_reference_set: ClockReferenceSet | None = None,
) -> str:
    """Default UTC session name when its spectrometer mapping is known."""
    if start_raw_seconds is None or clock_reference_set is None:
        return SESSION_NAME_NO_TIME_FMT.format(ord=ordinal)
    ts = clock_reference_set.to_time(
        start_raw_seconds,
        clock_source=ClockSource.SPECTROMETER,
    ).utc.strftime(
        SESSION_TIMESTAMP_FMT,
    )
    return DEFAULT_SESSION_NAME_FMT.format(ord=ordinal, ts=ts)


SessionNamer = Callable[[int, Optional[float], ClockReferenceSet], str]


# ---------------------------------------------------------------------------
# Flash directory parsing (Stages 1-4)
# ---------------------------------------------------------------------------

def _bank_path(flash_dir: Path, bank: str) -> Path:
    return flash_dir / bank / BANK_FILENAME


def _frames_from_bank(
    path: Path,
    *,
    bank: str,
    issue_collector: IssueCollector,
    capture: _FlashParseCapture | None,
):
    if capture is None:
        return parse_bank_file(
            path,
            bank=bank,
            issue_collector=issue_collector,
        )
    try:
        diagnostic = parse_bank_file_diagnostic(
            path,
            bank=bank,
            issue_collector=issue_collector,
        )
    except Exception as exc:
        capture.record_unreadable_bank(
            bank=bank,
            reason=type(exc).__name__,
        )
        raise
    capture.record_bank(
        bank=bank,
        size_bytes=diagnostic.input_size_bytes,
        sha256=diagnostic.input_sha256,
        recovered_frames=len(diagnostic.frames),
    )
    return diagnostic.frames


def _input_path_present(path: Path) -> bool:
    return path.is_symlink() or path.exists()


def _load_landing_reference(path: Path | str) -> ClockReferenceSet:
    reference_set = load_clock_reference_set(path)
    reference_set.require_reference(ClockSource.SPECTROMETER)
    return reference_set


def parse_flash(
    flash_dir: Path | str,
    *,
    landing_time_file: Path | str,
    issue_collector: IssueCollector | None = None,
) -> Tuple[
    List[Session],
    telemetry_mod.TelemetryData | None,
]:
    """Parse a FLASH_TLMFS directory through Stage 4.

    Returns the sessions and decoded source telemetry. Per-session disjoint
    slices are stored on each ``Session``.
    """
    clock_reference_set = _load_landing_reference(landing_time_file)
    return _parse_flash_loaded(
        Path(flash_dir),
        clock_reference_set=clock_reference_set,
        issue_collector=issue_collector,
    )


def _parse_flash_loaded(
    flash_dir: Path,
    *,
    clock_reference_set: ClockReferenceSet | None,
    issue_collector: IssueCollector | None,
    capture: _FlashParseCapture | None = None,
    schema_variant: str | None = None,
) -> Tuple[
    List[Session],
    telemetry_mod.TelemetryData | None,
]:
    """Parse a flash after the public entry point validates mission time."""
    flash_dir = Path(flash_dir)
    if issue_collector is None:
        issue_collector = IssueCollector()
    parse_marker = issue_collector.mark()
    if not flash_dir.is_dir():
        raise NotADirectoryError(f"FLASH input is not a directory: {flash_dir}")
    science_packets: List[LogicalPacket] = []
    telem_packets: List[LogicalPacket] = []

    for bank in SCIENCE_BANKS:
        path = _bank_path(flash_dir, bank)
        if not _input_path_present(path):
            log.info("skipping missing science bank %s", path)
            continue
        if not path.is_file():
            if capture is not None:
                capture.record_unreadable_bank(
                    bank=bank,
                    reason="not_regular_file",
                )
            issue_collector.record(
                code="source.science_bank_unreadable",
                severity=IssueSeverity.ERROR,
                stage="input",
                message=(
                    "a present science bank is not a readable regular file"
                ),
                action=IssueAction.REJECTED,
                bank=bank,
                details={"reason": "not_regular_file"},
            )
            continue
        log.info("reading science bank %s", path)
        try:
            bank_packets = list(reassemble_logical_packets(
                _frames_from_bank(
                    path,
                    bank=bank,
                    issue_collector=issue_collector,
                    capture=capture,
                ),
                byteswap_pairs=True,
                bank=bank,
                issue_collector=issue_collector,
            ))
        except OSError as exc:
            if capture is not None:
                capture.record_unreadable_bank(
                    bank=bank,
                    reason=type(exc).__name__,
                )
            issue_collector.record(
                code="source.science_bank_unreadable",
                severity=IssueSeverity.ERROR,
                stage="input",
                message="a present science bank could not be read",
                action=IssueAction.REJECTED,
                bank=bank,
                details={"error_type": type(exc).__name__},
            )
            continue
        if capture is not None:
            capture.science_logical_packets += len(bank_packets)
        for lp in bank_packets:
            if not is_dropped_appid(lp.appid):
                science_packets.append(lp)
                continue
            issue_collector.record(
                code="identity.policy_dropped_appid",
                severity=IssueSeverity.INFO,
                stage="identity",
                message="a configured non-science AppID was dropped",
                action=IssueAction.DROPPED,
                bank=lp.bank,
                appid=lp.appid,
                sequence_count=lp.seq,
                details={"policy": "mission_heartbeat"},
            )
            if capture is not None:
                capture.identity_policy_dropped_packets += 1

    tpath = _bank_path(flash_dir, TELEMETRY_BANK)
    telemetry_present = _input_path_present(tpath)
    telemetry_issue_collector = IssueCollector()
    telemetry_read_error = None
    if telemetry_present and tpath.is_file():
        log.info("reading telemetry bank %s", tpath)
        try:
            telem_packets.extend(reassemble_logical_packets(
                _frames_from_bank(
                    tpath,
                    bank=TELEMETRY_BANK,
                    issue_collector=telemetry_issue_collector,
                    capture=capture,
                ),
                byteswap_pairs=False,
                bank=TELEMETRY_BANK,
                issue_collector=telemetry_issue_collector,
            ))
            if capture is not None:
                capture.telemetry_logical_packets += len(telem_packets)
                capture.telemetry_0x314_packets += sum(
                    packet.appid == telemetry_mod.TELEMETRY_APPID
                    for packet in telem_packets
                )
        except OSError as exc:
            telemetry_read_error = type(exc).__name__
            if capture is not None:
                capture.record_unreadable_bank(
                    bank=TELEMETRY_BANK,
                    reason=telemetry_read_error,
                )
    elif telemetry_present:
        telemetry_read_error = "not_regular_file"
        if capture is not None:
            capture.record_unreadable_bank(
                bank=TELEMETRY_BANK,
                reason=telemetry_read_error,
            )
    else:
        log.info("no telemetry bank at %s", tpath)

    if capture is not None:
        capture.identity_input_packets = (
            len(science_packets) + capture.identity_policy_dropped_packets
        )
    sw_version = detect_sw_version(
        science_packets,
        issue_collector=issue_collector,
    )
    science_packets = assign_identities(
        science_packets,
        sw_version=sw_version,
        auto_detect_sw_version=False,
        issue_collector=issue_collector,
    )
    if capture is not None:
        capture.identity_kept_packets = len(science_packets)

    mark_waveform_transport_loss(science_packets, issue_collector.since(parse_marker))
    sessions = split_sessions(
        science_packets,
        **({"schema_variant": schema_variant} if schema_variant is not None else {}),
        issue_collector=issue_collector,
    )
    if capture is not None:
        capture.session_count = len(sessions)
        capture.session_input_packets = sum(
            len(session.packets) for session in sessions
        )

    if telemetry_read_error is not None:
        warnings.warn(
            "b01 telemetry skipped because its bank could not be read: "
            f"{telemetry_read_error}",
            stacklevel=2,
        )
        telemetry = None
    else:
        telemetry = telemetry_mod.decode_b01_packets(telem_packets)
    if telemetry is not None:
        if clock_reference_set is None:
            warnings.warn(
                "b01 telemetry omitted because no clock reference is available",
                stacklevel=2,
            )
            telemetry = None
        else:
            telemetry = assign_telemetry_to_sessions(
                sessions,
                telemetry,
                clock_reference_set=clock_reference_set,
            )
    return sessions, telemetry


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _canonical_manifest_bytes(body: Dict[str, object]) -> bytes:
    return (
        json.dumps(
            body,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_bytes(
    payload: bytes,
    dest: Path,
    *,
    overwrite: bool,
) -> Path:
    _validate_overwrite(overwrite)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(dest)
    dest.write_bytes(payload)
    return dest


def _relative_manifest_path(value: object, *, parent: Path) -> object:
    if value is None:
        return None
    return os.path.relpath(Path(str(value)), parent)


def _normalized_manifest_locator(
    value: object,
    *,
    parent: Path,
    field: str,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"linked FLASH session {field} locator is invalid")
    return (parent / value).resolve()


def _normalized_manifest_locator_list(
    value: object,
    *,
    parent: Path,
    field: str,
) -> tuple[Path, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"linked FLASH session {field} locators are invalid")
    if any(
        not isinstance(item, str)
        or not item
        or Path(item).is_absolute()
        for item in value
    ):
        raise ValueError(f"linked FLASH session {field} locators are invalid")
    return tuple((parent / item).resolve() for item in value)


def _session_manifest_body(
    result: SessionResult,
    *,
    destination: Path,
) -> Dict[str, object]:
    body = asdict(result)
    body.pop("manifest_path", None)
    body.pop("n_packets", None)
    for name in (
        "source_path",
        "h5_path",
        "fits_path",
        "session_dir",
        "flash_source_path",
    ):
        body[name] = _relative_manifest_path(
            body.get(name),
            parent=destination.parent,
        )
    body["plot_paths"] = [
        _relative_manifest_path(path, parent=destination.parent)
        for path in body["plot_paths"]
    ]
    body["manifest_kind"] = "session"
    body["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    return body


def _flash_manifest_body(
    result: FlashResult,
    *,
    path_parent: Path | None = None,
) -> Dict[str, object]:
    if path_parent is None and result.manifest_paths:
        path_parent = Path(result.manifest_paths[-1]).parent

    def located(value: object) -> object:
        if path_parent is None:
            return value
        return _relative_manifest_path(value, parent=path_parent)

    return {
        "manifest_kind": "flash",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "locator_contract": {
            "version": 1,
            "base": "sessions_root",
            "canonical_manifest": FLASH_MANIFEST_NAME,
            "external_copies_are_mirrors": True,
        },
        "flash_result_id": result.flash_result_id,
        "input_identity_sha256": result.input_identity_sha256,
        "input_identity_unavailable_reason": (
            result.input_identity_unavailable_reason
        ),
        "source_identity_unavailable_reasons": (
            result.source_identity_unavailable_reasons
        ),
        "source_fingerprint_format_version": FLASH_FINGERPRINT_FORMAT_VERSION,
        "source_fingerprint": result.source_fingerprint,
        "status": result.status,
        "status_issue_codes": result.status_issue_codes,
        "issue_counts": result.issue_counts,
        "issues": result.issues,
        "stage_counts": result.stage_counts,
        "decoder_provenance": result.decoder_provenance,
        "clock_reference": result.clock_reference,
        "reassembly_profile": result.reassembly_profile,
        "pipeline_version": result.pipeline_version,
        "processed_at_utc": result.processed_at_utc,
        "failure": result.failure,
        "overwrite": result.overwrite,
        "sessions": [
            {
                "session_ordinal": session.session_ordinal,
                "session_name": session.session_name,
                "status": session.status,
                "status_issue_codes": session.status_issue_codes,
                "issue_counts": session.issue_counts,
                "issues": session.issues,
                "stage_counts": session.stage_counts,
                "family_statuses": session.family_statuses,
                "input_packet_count": session.input_packet_count,
                "decoder_input_packet_count": session.decoder_input_packet_count,
                "decoder_valid_packet_count": session.decoder_valid_packet_count,
                "decoder_invalid_packet_count": session.decoder_invalid_packet_count,
                "decoded_rows_by_family": session.decoded_rows_by_family,
                "persisted_rows_by_family": session.persisted_rows_by_family,
                "decoder_provenance": session.decoder_provenance,
                "start_time_utc": session.start_time_utc,
                "start_raw_seconds": session.start_raw_seconds,
                "telemetry_status": session.telemetry_status,
                "telemetry_reason": session.telemetry_reason,
                "telemetry_source": session.telemetry_source,
                "n_telemetry_rows": session.n_telemetry_rows,
                "contracts": session.contracts,
                "output_artifacts": session.output_artifacts,
                "committed_artifacts": session.committed_artifacts,
                "packet_map_status": session.packet_map_status,
                "packet_map_format_version": session.packet_map_format_version,
                "raw_flash_provenance_unavailable_reason": (
                    session.raw_flash_provenance_unavailable_reason
                ),
                "session_dir": located(session.session_dir),
                "h5_path": located(session.h5_path),
                "fits_path": located(session.fits_path),
                "manifest_path": located(session.manifest_path),
                "plot_paths": [located(path) for path in session.plot_paths],
                "failure": session.failure,
            }
            for session in result.session_results
        ],
    }


def write_manifest(result: SessionResult, dest_path: Path | str) -> Path:
    """Serialize one final session result as strict ASCII JSON."""
    if not isinstance(result, SessionResult):
        raise TypeError("session manifest requires a SessionResult")
    dest = Path(dest_path)
    return _write_bytes(
        _canonical_manifest_bytes(
            _session_manifest_body(result, destination=dest)
        ),
        dest,
        overwrite=result.overwrite,
    )


def write_flash_manifest(
    result: FlashResult,
    dest_path: Path | str,
    *,
    sessions_root: Path | str,
    overwrite: bool = False,
) -> Path:
    """Serialize a canonical FLASH manifest under its sessions root."""
    if not isinstance(result, FlashResult):
        raise TypeError("FLASH manifest requires a FlashResult")
    _validate_overwrite(overwrite)
    destination = Path(dest_path).resolve()
    root = Path(sessions_root).resolve()
    if destination != root / FLASH_MANIFEST_NAME:
        raise ValueError(
            "FLASH manifest destination must be sessions_root/flash.json"
        )
    return _write_bytes(
        _canonical_manifest_bytes(
            _flash_manifest_body(result, path_parent=root)
        ),
        destination,
        overwrite=overwrite,
    )


def _artifact_record(path: Path) -> Dict[str, object]:
    size_bytes, digest = _sha256_file(path)
    return {
        "path": path.name,
        "size_bytes": size_bytes,
        "sha256": digest,
    }


def _status_from_issues(
    issues: Sequence[IngestIssue],
    *,
    failed: bool = False,
) -> str:
    if failed:
        return "failed"
    if any(issue.severity is not IssueSeverity.INFO for issue in issues):
        return "partial"
    return "clean"


def _issue_manifest_fields(
    issues: Sequence[IngestIssue],
) -> tuple[List[Dict[str, object]], Dict[str, int], List[str]]:
    records = [issue.as_dict() for issue in issues]
    counts = dict(sorted(Counter(issue.code for issue in issues).items()))
    degrading = sorted({
        issue.code
        for issue in issues
        if issue.severity is not IssueSeverity.INFO
    })
    return records, counts, degrading


def _merge_issues(*groups: Sequence[IngestIssue]) -> tuple[IngestIssue, ...]:
    by_id: Dict[str, IngestIssue] = {}
    order: List[str] = []
    for group in groups:
        for issue in group:
            existing = by_id.get(issue.issue_id)
            if existing is not None:
                if existing != issue:
                    raise ValueError(
                        "ingest issue ID refers to conflicting run records"
                    )
                continue
            by_id[issue.issue_id] = issue
            order.append(issue.issue_id)
    return tuple(by_id[issue_id] for issue_id in order)


def _flash_stage_counts(
    capture: _FlashParseCapture,
    issues: Sequence[IngestIssue],
    *,
    products: Sequence[Products] = (),
    results: Sequence[SessionResult] = (),
) -> Dict[str, Dict[str, int]]:
    counts = capture.stage_count_record(issues)
    decode_input = sum(
        product.decode_provenance.input_packet_count for product in products
    )
    decode_valid = sum(
        product.decode_provenance.valid_packet_count or 0
        for product in products
    )
    counts["decode"] = {
        "input_packets": decode_input,
        "valid_packets": decode_valid,
        "invalid_packets": decode_input - decode_valid,
    }
    counts["products"] = {
        family: sum(len(getattr(product, family)) for product in products)
        for family, _ in FAMILY_TYPES
    }
    persistence_families = {
        family for family, _family_type in FAMILY_TYPES
    }
    persistence_families.update(
        family
        for result in results
        for family in result.persisted_rows_by_family
    )
    counts["persistence"] = {
        family: sum(
            result.persisted_rows_by_family.get(family, 0)
            for result in results
        )
        for family in sorted(persistence_families)
    }
    return counts


def _flash_source_identity_sha256(
    source_fingerprint: Dict[str, Dict[str, object]],
) -> str:
    source_payload = _canonical_manifest_bytes({
        "source_fingerprint_format_version": (
            FLASH_FINGERPRINT_FORMAT_VERSION
        ),
        "source_fingerprint": source_fingerprint,
    })
    return hashlib.sha256(source_payload).hexdigest()


def _flash_identity(
    source_fingerprint: Dict[str, Dict[str, object]],
    clock_reference_set: ClockReferenceSet,
    binding_identity: tuple[object, ...] | None = None,
    *,
    source_identity_complete: bool = True,
    source_identity_unavailable_reasons: Sequence[str] = (),
) -> tuple[str, Optional[str], Optional[str]]:
    normalized_fingerprint = _validate_flash_fingerprint(source_fingerprint)
    normalized_unavailable_reasons = tuple(sorted(set(
        source_identity_unavailable_reasons
    )))
    if any(
        not isinstance(reason, str) or not reason
        for reason in normalized_unavailable_reasons
    ):
        raise ValueError("source identity unavailable reasons must be text")
    input_identity_unavailable_reason = None
    if not normalized_fingerprint:
        input_identity_sha256 = None
        input_identity_unavailable_reason = "no_captured_bank_sources"
    elif not source_identity_complete:
        input_identity_sha256 = None
        input_identity_unavailable_reason = "source_bank_unreadable"
    else:
        input_identity_sha256 = _flash_source_identity_sha256(
            normalized_fingerprint
        )
    run_payload = _canonical_manifest_bytes({
        "clock_reference_sha256": clock_reference_set.source_sha256,
        "decoder_binding_identity": binding_identity,
        "input_identity_sha256": input_identity_sha256,
        "input_identity_unavailable_reason": (
            input_identity_unavailable_reason
        ),
        "source_identity_unavailable_reasons": (
            normalized_unavailable_reasons
        ),
        "reassembly_profile": REASSEMBLY_PROFILE,
        "source_fingerprint_format_version": (
            FLASH_FINGERPRINT_FORMAT_VERSION
        ),
        "source_fingerprint": normalized_fingerprint,
    })
    run_digest = hashlib.sha256(run_payload).hexdigest()
    return (
        f"flash-{run_digest[:16]}",
        input_identity_sha256,
        input_identity_unavailable_reason,
    )


def _now_utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _start_utc_iso(
    raw_seconds: Optional[float],
    clock_reference_set: ClockReferenceSet | None,
) -> Optional[str]:
    if raw_seconds is None or clock_reference_set is None:
        return None
    utc = clock_reference_set.to_time(
        raw_seconds,
        clock_source=ClockSource.SPECTROMETER,
    ).utc
    utc.precision = 9
    return f"{utc.isot}Z"


def _pipeline_version_string() -> str:
    try:
        from .. import __version__ as luv
    except Exception:    # noqa: BLE001
        luv = "unknown"
    return f"lusee={luv},ingest_schema={MANIFEST_SCHEMA_VERSION}"


# ---------------------------------------------------------------------------
# Session result helpers
# ---------------------------------------------------------------------------

def _decoder_provenance_manifest(products: Products) -> Dict[str, object]:
    provenance = products.decode_provenance
    return {
        "decoder_name": provenance.decoder_name,
        "distribution_version": provenance.distribution_version,
        "decoder_source_commit": provenance.decoder_source_commit,
        "reported_schema_ids": list(provenance.reported_schema_ids),
        "selected_schema_id": provenance.selected_schema_id,
        "binding_key": provenance.binding_key,
        "schema_variant": provenance.schema_variant,
        "schema_assumed": provenance.schema_assumed,
        "binding_source_release": provenance.binding_source_release,
        "binding_source_commit": provenance.binding_source_commit,
        "abi_fingerprint": provenance.abi_fingerprint,
        "execution_mode": (
            provenance.execution_mode.value
            if provenance.execution_mode is not None
            else None
        ),
        "input_packet_count": provenance.input_packet_count,
        "valid_packet_count": provenance.valid_packet_count,
        "appid_counts": [
            {"appid": appid, "count": count}
            for appid, count in provenance.appid_counts
        ],
        "issue_counts": dict(provenance.issue_counts),
        "unavailable_reason": provenance.unavailable_reason,
    }


def _family_status_manifest(
    statuses,
    issues: Sequence[IngestIssue],
    *,
    persisted_rows: Dict[str, int] | None = None,
) -> List[Dict[str, object]]:
    issue_by_id = {issue.issue_id: issue for issue in issues}
    records = []
    for status in statuses:
        persisted = (
            persisted_rows.get(status.family, 0)
            if persisted_rows is not None
            else status.decoded_rows
            if status.coverage.value == "persisted"
            else 0
        )
        coverage = status.coverage.value
        if coverage == "persisted" and persisted == 0:
            coverage = "decoded_not_persisted"
        records.append({
            "family": status.family,
            "supported": status.supported,
            "coverage": coverage,
            "quality": status.quality.value,
            "decoded_rows": status.decoded_rows,
            "persisted_rows": persisted,
            "issue_ids": list(status.issue_ids),
            "issue_codes": sorted({
                issue_by_id[issue_id].code
                for issue_id in status.issue_ids
            }),
            "reason": status.reason,
        })
    return records


# ---------------------------------------------------------------------------
# Process one session (existing-session mode)
# ---------------------------------------------------------------------------

def _process_one_session(
    *,
    session_dir: Path,
    name: str | None,
    ordinal: int,
    h5_dir: Optional[Path],
    plots_dir: Optional[Path],
    manifest_dir: Optional[Path],
    issue_collector: IssueCollector,
    clock_reference_set: ClockReferenceSet | None = None,
    fits_dir: Optional[Path] = None,
    telemetry: telemetry_mod.TelemetryData | None = None,
    telemetry_source: str | None = None,
    telemetry_reason: str | None = None,
    source_path: Optional[Path] = None,
    source_kind: str = "session",
    plot_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
    products: Products | None = None,
    context_issues: Sequence[IngestIssue] = (),
    input_packet_count: int | None = None,
    input_identity: str | None = None,
    input_identity_kind: str | None = None,
    input_identity_unavailable_reason: str | None = None,
    flash_result_id: str | None = None,
) -> SessionResult:
    _validate_overwrite(overwrite)
    if (
        input_identity is not None
        and input_identity_unavailable_reason is not None
    ):
        raise ValueError(
            "available input identity cannot have an unavailable reason"
        )
    if products is None:
        products = read_uncrater_session(
            session_dir,
            issue_collector=issue_collector,
            strict=decoder_strict,
            diagnostic_override=diagnostic_override,
            schema_variant=schema_variant,
        )

    if telemetry is not None:
        if telemetry_source not in (None, telemetry.source_kind):
            raise ValueError("telemetry source disagrees with TelemetryData")
        if telemetry_reason is not None:
            raise ValueError("decoded telemetry cannot have a skip reason")
        telemetry_status = "decoded"
        telemetry_source = telemetry.source_kind
    elif telemetry_source is not None:
        if telemetry_source not in (
            "b01_0x314",
            "legacy_binary_sidecar",
        ):
            raise ValueError("unknown telemetry source")
        if not isinstance(telemetry_reason, str) or not telemetry_reason:
            raise ValueError("skipped telemetry requires a reason")
        telemetry_status = "skipped"
    else:
        if telemetry_reason is not None:
            raise ValueError("absent telemetry cannot have a reason")
        telemetry_status = "absent"
    name = _validate_session_output_name(
        name or default_session_name(
            ordinal,
            products.start_raw_seconds,
            clock_reference_set,
        )
    )
    resolved_source = (source_path or session_dir).resolve()

    context_by_id: Dict[str, IngestIssue] = {}
    for issue in context_issues:
        if not isinstance(issue, IngestIssue):
            raise TypeError("context_issues must contain IngestIssue records")
        existing = context_by_id.get(issue.issue_id)
        if existing is not None and existing != issue:
            raise ValueError("context issues reuse an ID with different records")
        context_by_id[issue.issue_id] = issue
    normalized_context_issues = tuple(
        context_by_id[issue_id] for issue_id in sorted(context_by_id)
    )
    product_issues = list(products.issues)
    if (
        products.quality_status is not None
        and products.quality_status.value == "failed"
        and not any(
            issue.code == "decode.no_usable_products"
            for issue in product_issues
        )
    ):
        product_issues.append(issue_collector.record(
            code="decode.no_usable_products",
            severity=IssueSeverity.ERROR,
            stage="decode",
            message="the decoder produced no usable products",
            action=IssueAction.REJECTED,
            session=name,
            details={
                "input_packets": products.decode_provenance.input_packet_count,
                "valid_packets": products.decode_provenance.valid_packet_count,
            },
        ))
    issue_by_id: Dict[str, IngestIssue] = {}
    for issue in (
        *product_issues,
        *normalized_context_issues,
    ):
        existing = issue_by_id.get(issue.issue_id)
        if existing is not None and existing != issue:
            raise ValueError("ingest issue sources disagree for one issue ID")
        issue_by_id[issue.issue_id] = issue
    request_issues = tuple(
        issue_by_id[issue_id] for issue_id in sorted(issue_by_id)
    )
    family_issue_ids = {}
    for family, _ in FAMILY_TYPES:
        issue_ids = set(products.family_issue_ids.get(family, ()))
        issue_ids.update(
            issue_id
            for row in getattr(products, family)
            for issue_id in row.provenance.decoder_issue_ids
        )
        family_issue_ids[family] = tuple(sorted(issue_ids))
    family_statuses = family_statuses_for_products(
        products,
        family_issue_ids=family_issue_ids,
    )
    decoded_rows = {
        status.family: status.decoded_rows for status in family_statuses
    }
    persistable_rows = {
        status.family: (
            status.decoded_rows
            if status.coverage.value == "persisted"
            else 0
        )
        for status in family_statuses
    }
    persisted_rows = {family: 0 for family in persistable_rows}
    product_row_count = sum(
        len(getattr(products, family)) for family, _ in FAMILY_TYPES
    )
    decoder_input_count = products.decode_provenance.input_packet_count
    decoder_valid_count = products.decode_provenance.valid_packet_count or 0
    session_input_count = (
        decoder_input_count
        if input_packet_count is None
        else input_packet_count
    )
    if type(session_input_count) is not int or session_input_count < 0:
        raise ValueError("session input packet count must be nonnegative")
    issue_records, issue_counts, status_issue_codes = _issue_manifest_fields(
        request_issues
    )
    status = _status_from_issues(request_issues)
    if products.quality_status is not None:
        if products.quality_status.value == "failed":
            status = "failed"
        elif products.quality_status.value == "partial" and status == "clean":
            status = "partial"

    result = SessionResult(
        session_ordinal=ordinal,
        session_name=name,
        source_path=str(resolved_source),
        source_kind=source_kind,
        start_time_utc=_start_utc_iso(
            products.start_raw_seconds,
            clock_reference_set,
        ),
        start_unique_packet_id=products.start_unique_packet_id,
        software_version=products.sw_version,
        firmware_version=products.fw_version,
        input_packet_count=session_input_count,
        decoder_input_packet_count=decoder_input_count,
        decoder_valid_packet_count=decoder_valid_count,
        decoder_invalid_packet_count=decoder_input_count - decoder_valid_count,
        n_product_rows=product_row_count,
        decoded_rows_by_family=decoded_rows,
        persisted_rows_by_family=persisted_rows,
        n_packets=product_row_count,
        n_spectra=len(products.spectra),
        n_tr_spectra=len(products.tr_spectra),
        n_zoom_spectra=len(products.zoom_spectra),
        n_grimm_spectra=len(products.grimm_spectra),
        n_waveforms=len(products.waveforms),
        n_housekeeping=len(products.housekeeping),
        n_warnings=sum(
            issue.severity is IssueSeverity.WARNING
            for issue in request_issues
        ),
        warnings_summary=[
            issue.message
            for issue in request_issues
            if issue.severity is IssueSeverity.WARNING
        ],
        status=status,
        status_issue_codes=status_issue_codes,
        issue_counts=issue_counts,
        issues=issue_records,
        stage_counts={
            "session_input": {"logical_packets": session_input_count},
            "decode": {
                "input_packets": decoder_input_count,
                "valid_packets": decoder_valid_count,
                "invalid_packets": decoder_input_count - decoder_valid_count,
            },
            "products": decoded_rows,
            "persistence": persisted_rows,
        },
        family_statuses=_family_status_manifest(
            family_statuses,
            request_issues,
            persisted_rows=persisted_rows,
        ),
        decoder_provenance=_decoder_provenance_manifest(products),
        contracts={
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "output_layout_version": INGEST_LAYOUT_VERSION,
            "spectra_normalization_version": SPECTRA_NORMALIZATION_VERSION,
            "frequency_coordinate_status": "unresolved",
            "clock_reference_sha256": (
                clock_reference_set.source_sha256
                if clock_reference_set is not None
                else None
            ),
        },
        session_dir=str(session_dir.resolve()),
        processed_at_utc=_now_utc_iso(),
        pipeline_version=_pipeline_version_string(),
        clock_reference=(
            clock_reference_set.as_record()
            if clock_reference_set is not None
            else None
        ),
        packet_map_status=products.packet_map_status,
        packet_map_format_version=products.packet_map_format_version,
        raw_flash_provenance_unavailable_reason=(
            products.raw_flash_provenance_unavailable_reason
        ),
        flash_result_id=flash_result_id,
        overwrite=overwrite,
        telemetry_status=telemetry_status,
        telemetry_reason=telemetry_reason,
        telemetry_source=telemetry_source,
        n_telemetry_rows=0 if telemetry is None else telemetry.row_count,
    )
    def mark_products_persisted() -> None:
        result.persisted_rows_by_family = dict(persistable_rows)
        result.stage_counts["persistence"] = dict(persistable_rows)
        result.family_statuses = _family_status_manifest(
            family_statuses,
            request_issues,
            persisted_rows=persistable_rows,
        )

    h5_path = h5_dir / f"{name}.h5" if h5_dir is not None else None
    fits_path = fits_dir / f"{name}.fits" if fits_dir is not None else None
    manifest_path = (
        manifest_dir / f"{name}.json"
        if manifest_dir is not None
        else None
    )
    plot_dest = (
        plots_dir / name
        if plots_dir is not None and h5_path is not None
        else None
    )
    if manifest_path is not None:
        result.manifest_path = str(manifest_path.resolve())
    if plot_dest is not None and plot_dest.exists():
        if not overwrite:
            raise FileExistsError(plot_dest)
        shutil.rmtree(plot_dest)

    request = None
    if h5_path is not None or fits_path is not None:
        if clock_reference_set is None:
            raise ValueError(
                "layout-v4 output requires a verified landing-time reference"
            )
        request = WriteRequest(
            products=products,
            clock_reference_set=clock_reference_set,
            clock_reference_unavailable_reason=None,
            location=LunarLocation(
                latitude_deg=DEFAULT_LUN_LAT_DEG,
                longitude_deg=DEFAULT_LUN_LONG_DEG,
                height_m=DEFAULT_LUN_HEIGHT_M,
            ),
            run_provenance=RunProvenance(
                input_identity=input_identity,
                input_identity_kind=input_identity_kind,
                input_identity_unavailable_reason=(
                    input_identity_unavailable_reason
                    or "portable_input_identity_not_available"
                    if input_identity is None
                    else None
                ),
                source_kind=source_kind,
                source_path=None,
                pipeline_version=_pipeline_version_string(),
            ),
            issues=request_issues,
            family_statuses=family_statuses,
            context_issues=normalized_context_issues,
            telemetry=telemetry,
            overwrite=overwrite,
        )

    if h5_path is not None:
        from .hdf5_writer import write_hdf5

        write_hdf5(request, h5_path)
        result.h5_path = str(h5_path.resolve())
        if h5_path.is_file():
            result.output_artifacts["hdf5"] = _artifact_record(h5_path)
        result.committed_artifacts.append("hdf5")
        mark_products_persisted()

    if fits_path is not None:
        from .fits_writer import write_fits

        write_fits(request, fits_path)
        result.fits_path = str(fits_path.resolve())
        if fits_path.is_file():
            result.output_artifacts["fits"] = _artifact_record(fits_path)
        result.committed_artifacts.append("fits")
        mark_products_persisted()

    if plot_dest is not None:
        from . import viz as viz_mod

        plot_dest.mkdir(parents=True, exist_ok=True)
        result.committed_artifacts.append("plot_directory")
        plot_paths = viz_mod.plot_session(h5_path, plot_dest, plots=plot_names)
        result.plot_paths = [str(p.resolve()) for p in plot_paths]
        for index, path in enumerate(plot_paths):
            if path.is_file():
                result.output_artifacts[f"plot_{index:03d}"] = _artifact_record(
                    path
                )

    return result


def _binding_identity(products: Products) -> tuple[object, ...]:
    """Return only the selected frozen binding identity."""
    provenance = products.decode_provenance
    if provenance.unavailable_reason is not None:
        raise RuntimeError(
            "FLASH binding preflight requires concrete decoder provenance"
        )
    return (
        provenance.selected_schema_id,
        provenance.binding_key,
        provenance.schema_variant,
        provenance.binding_source_release,
        provenance.binding_source_commit,
        provenance.abi_fingerprint,
    )


# ---------------------------------------------------------------------------
# Public: process_session (existing-session mode)
# ---------------------------------------------------------------------------

def _manifest_object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"session manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _validate_manifest_telemetry_fields(record: Mapping[str, Any]) -> None:
    """Validate the new flat diagnostics while accepting older v3 records."""
    fields = {
        "telemetry_status",
        "telemetry_reason",
        "telemetry_source",
        "n_telemetry_rows",
    }
    legacy_markers = {
        "telemetry_decoder_status",
        "telemetry_coverage",
        "telemetry_input_sources",
        "telemetry_provenance",
    }
    discriminators = {"telemetry_status", "telemetry_reason"}
    if not discriminators.intersection(record):
        if legacy_markers.intersection(record):
            return
        raise ValueError("current manifest is missing telemetry diagnostics")
    if not fields.issubset(record):
        raise ValueError("current manifest is missing telemetry diagnostics")

    status = record["telemetry_status"]
    reason = record["telemetry_reason"]
    source = record["telemetry_source"]
    row_count = record["n_telemetry_rows"]
    if status not in ("absent", "decoded", "skipped"):
        raise ValueError("manifest has invalid telemetry_status")
    if source not in (None, "b01_0x314", "legacy_binary_sidecar"):
        raise ValueError("manifest has invalid telemetry_source")
    if type(row_count) is not int or row_count < 0:
        raise ValueError("manifest has invalid n_telemetry_rows")

    if status == "absent":
        coherent = reason is None and source is None and row_count == 0
    elif status == "decoded":
        coherent = reason is None and source is not None
    else:
        coherent = (
            isinstance(reason, str)
            and bool(reason.strip())
            and source is not None
            and row_count == 0
        )
    if not coherent:
        raise ValueError("manifest telemetry diagnostics are inconsistent")


def _read_in_session_manifest(
    session_dir: Path,
    *,
    strict: bool,
    issue_collector: IssueCollector | None = None,
) -> Optional[Dict[str, Any]]:
    p = session_dir / IN_SESSION_MANIFEST_NAME
    if p.is_symlink() or (p.exists() and not p.is_file()):
        if strict:
            raise ValueError("session.json must be a regular file")
        return None
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="ascii") as fh:
            value = json.load(
                fh,
                object_pairs_hook=_manifest_object_without_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        if not isinstance(value, dict):
            raise ValueError("session manifest must be a JSON object")
        version = value.get("manifest_schema_version")
        if version is not None and (
            type(version) is not int or version not in (2, MANIFEST_SCHEMA_VERSION)
        ):
            raise ValueError("session manifest has unsupported schema version")
        kind = value.get("manifest_kind")
        if kind is not None and kind != "session":
            raise ValueError("session manifest has invalid manifest_kind")
        if version == MANIFEST_SCHEMA_VERSION:
            _validate_manifest_telemetry_fields(value)
        flash_result_id = value.get("flash_result_id")
        if flash_result_id is not None and (
            not isinstance(flash_result_id, str)
            or not flash_result_id.startswith("flash-")
            or len(flash_result_id) != 22
            or any(
                char not in "0123456789abcdef"
                for char in flash_result_id[6:]
            )
        ):
            raise ValueError("session manifest has invalid flash_result_id")
        flash_manifest_sha256 = value.get("flash_manifest_sha256")
        if flash_manifest_sha256 is not None and (
            not isinstance(flash_manifest_sha256, str)
            or len(flash_manifest_sha256) != 64
            or any(
                char not in "0123456789abcdef"
                for char in flash_manifest_sha256
            )
        ):
            raise ValueError(
                "session manifest has invalid flash_manifest_sha256"
            )
        if (flash_result_id is None) != (flash_manifest_sha256 is None):
            raise ValueError(
                "session manifest FLASH linkage must include both result ID "
                "and manifest digest"
            )
        flash_input_identity = value.get("flash_input_identity_sha256")
        if flash_input_identity is not None and (
            not isinstance(flash_input_identity, str)
            or len(flash_input_identity) != 64
            or any(
                char not in "0123456789abcdef"
                for char in flash_input_identity
            )
        ):
            raise ValueError(
                "session manifest has invalid flash_input_identity_sha256"
            )
        identity_unavailable = value.get(
            "flash_input_identity_unavailable_reason"
        )
        if identity_unavailable is not None and (
            not isinstance(identity_unavailable, str)
            or not identity_unavailable
        ):
            raise ValueError(
                "session manifest has invalid FLASH identity unavailable reason"
            )
        if flash_input_identity is not None and identity_unavailable is not None:
            raise ValueError(
                "available FLASH input identity cannot have an unavailable reason"
            )
        if flash_result_id is not None and (
            (flash_input_identity is None) == (identity_unavailable is None)
        ):
            raise ValueError(
                "linked FLASH input requires either its source identity or "
                "an unavailable reason"
            )
        if flash_result_id is None and (
            flash_input_identity is not None or identity_unavailable is not None
        ):
            raise ValueError(
                "FLASH input identity requires a linked FLASH result"
            )
        fingerprint_present = "flash_source_fingerprint" in value
        if fingerprint_present:
            fingerprint = value["flash_source_fingerprint"]
            try:
                value["flash_source_fingerprint"] = (
                    _validate_flash_fingerprint(fingerprint)
                )
            except ValueError:
                value["flash_source_fingerprint"] = (
                    _validate_legacy_flash_fingerprint(fingerprint)
                )
                value["_legacy_flash_source_fingerprint"] = True
        elif flash_result_id is not None:
            raise ValueError(
                "linked FLASH input requires its source fingerprint"
            )
        if flash_input_identity is not None:
            normalized_fingerprint = value["flash_source_fingerprint"]
            if value.get("_legacy_flash_source_fingerprint") or not (
                normalized_fingerprint
            ):
                raise ValueError(
                    "FLASH source identity requires a complete SHA-256 fingerprint"
                )
            expected_input_identity = _flash_source_identity_sha256(
                normalized_fingerprint
            )
            if flash_input_identity != expected_input_identity:
                raise ValueError(
                    "FLASH source identity does not match its fingerprint"
                )
        sibling_flash_manifest = session_dir.parent / FLASH_MANIFEST_NAME
        if flash_manifest_sha256 is not None and _input_path_present(
            sibling_flash_manifest
        ):
            if (
                sibling_flash_manifest.is_symlink()
                or not sibling_flash_manifest.is_file()
            ):
                raise ValueError("linked FLASH manifest must be a regular file")
            sibling_payload = sibling_flash_manifest.read_bytes()
            if hashlib.sha256(sibling_payload).hexdigest() != flash_manifest_sha256:
                raise ValueError("linked FLASH manifest digest does not match")
            sibling = json.loads(
                sibling_payload.decode("ascii"),
                object_pairs_hook=_manifest_object_without_duplicate_keys,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
            expected_locator_contract = {
                "version": 1,
                "base": "sessions_root",
                "canonical_manifest": FLASH_MANIFEST_NAME,
                "external_copies_are_mirrors": True,
            }
            if (
                not isinstance(sibling, dict)
                or sibling.get("manifest_kind") != "flash"
                or sibling.get("manifest_schema_version")
                != MANIFEST_SCHEMA_VERSION
                or sibling.get("flash_result_id") != flash_result_id
                or sibling.get("input_identity_sha256")
                != flash_input_identity
                or sibling.get("input_identity_unavailable_reason")
                != identity_unavailable
                or sibling.get("source_fingerprint")
                != value.get("flash_source_fingerprint", {})
                or sibling.get("clock_reference")
                != value.get("clock_reference")
                or sibling.get("locator_contract")
                != expected_locator_contract
            ):
                raise ValueError("linked FLASH manifest identity does not match")
            session_name = value.get("session_name")
            session_ordinal = value.get("session_ordinal")
            linked_sessions = sibling.get("sessions")
            if (
                not isinstance(session_name, str)
                or type(session_ordinal) is not int
                or not isinstance(linked_sessions, list)
            ):
                raise ValueError("linked FLASH session identity is incomplete")
            matches = [
                record
                for record in linked_sessions
                if isinstance(record, dict)
                and record.get("session_name") == session_name
                and record.get("session_ordinal") == session_ordinal
            ]
            if len(matches) != 1:
                raise ValueError("linked FLASH session identity does not match")
            linked_locator = matches[0].get("session_dir")
            if (
                not isinstance(linked_locator, str)
                or Path(linked_locator).is_absolute()
                or (session_dir.parent / linked_locator).resolve()
                != session_dir.resolve()
            ):
                raise ValueError("linked FLASH session locator does not match")
            linked_session = matches[0]
            for field in ("h5_path", "fits_path"):
                linked_output = _normalized_manifest_locator(
                    linked_session.get(field),
                    parent=session_dir.parent,
                    field=field,
                )
                session_output = _normalized_manifest_locator(
                    value.get(field),
                    parent=session_dir,
                    field=field,
                )
                if linked_output != session_output:
                    raise ValueError(
                        f"linked FLASH session {field} locator does not match"
                    )
            if _normalized_manifest_locator_list(
                linked_session.get("plot_paths"),
                parent=session_dir.parent,
                field="plot_paths",
            ) != _normalized_manifest_locator_list(
                value.get("plot_paths"),
                parent=session_dir,
                field="plot_paths",
            ):
                raise ValueError(
                    "linked FLASH session plot_paths locator does not match"
                )
            shared_fields = (
                "status",
                "status_issue_codes",
                "issue_counts",
                "issues",
                "stage_counts",
                "family_statuses",
                "input_packet_count",
                "decoder_input_packet_count",
                "decoder_valid_packet_count",
                "decoder_invalid_packet_count",
                "decoded_rows_by_family",
                "persisted_rows_by_family",
                "decoder_provenance",
                "start_time_utc",
                "start_raw_seconds",
                "telemetry_status",
                "telemetry_reason",
                "telemetry_source",
                "n_telemetry_rows",
                "contracts",
                "output_artifacts",
                "committed_artifacts",
                "packet_map_status",
                "packet_map_format_version",
                "raw_flash_provenance_unavailable_reason",
                "failure",
            )
            if any(
                linked_session.get(field) != value.get(field)
                for field in shared_fields
            ):
                raise ValueError("linked FLASH session metadata does not match")
        return value
    except Exception as exc:    # noqa: BLE001
        if issue_collector is not None:
            issue_collector.record(
                code="manifest.session_read_failed",
                severity=IssueSeverity.ERROR,
                stage="manifest_read",
                message="the in-session manifest could not be validated",
                action=IssueAction.REJECTED,
                details={"error_type": type(exc).__name__},
            )
        if strict:
            raise ValueError(f"cannot read in-session manifest {p}: {exc}") from exc
        warnings.warn(
            f"failed to read in-session manifest {p}: {exc}; ignoring",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _process_session_impl(
    session_dir: Path | str,
    *,
    landing_time_file: Path | str | None = None,
    h5_dir: Optional[Path | str] = None,
    fits_dir: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    manifest_dir: Optional[Path | str] = None,
    name: Optional[str] = None,
    ordinal: int = 0,
    plot_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
    issue_marker: int | None = None,
) -> SessionResult:
    """Process one extracted session using only its optional sidecar."""
    session_dir = Path(session_dir).resolve()
    h5_dir = Path(h5_dir) if h5_dir else None
    fits_dir = Path(fits_dir) if fits_dir else None
    plots_dir = Path(plots_dir) if plots_dir else None
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    output_requested = any(
        path is not None
        for path in (h5_dir, fits_dir, plots_dir, manifest_dir)
    )
    if issue_collector is None:
        issue_collector = IssueCollector()
    if issue_marker is None:
        issue_marker = issue_collector.mark()

    in_session_manifest = _read_in_session_manifest(
        session_dir,
        strict=output_requested or landing_time_file is not None,
        issue_collector=issue_collector,
    )
    embedded_reference = None
    embedded_record = (in_session_manifest or {}).get("clock_reference")
    if embedded_record is not None:
        embedded_reference = clock_reference_set_from_record(embedded_record)
        embedded_reference.require_reference(ClockSource.SPECTROMETER)
    elif (
        output_requested
        and (in_session_manifest or {}).get("manifest_schema_version") == 3
    ):
        raise ValueError("session manifest v3 is missing clock_reference")

    supplied_reference = (
        _load_landing_reference(landing_time_file)
        if landing_time_file is not None
        else None
    )
    if (
        embedded_reference is not None
        and supplied_reference is not None
        and embedded_reference != supplied_reference
    ):
        raise ValueError(
            "supplied landing-time file contradicts the embedded clock reference"
        )
    clock_reference_set = supplied_reference or embedded_reference
    if output_requested and clock_reference_set is None:
        raise ValueError(
            "operational session output requires landing_time_file or a "
            "verified embedded clock reference"
        )

    telemetry = None
    telemetry_source = None
    telemetry_reason = None
    sidecar = telemetry_mod.find_legacy_sidecar(session_dir)
    if sidecar is not None:
        telemetry_source = "legacy_binary_sidecar"
        log.info("reading legacy DCB_telemetry sidecar at %s", sidecar)
        telemetry = telemetry_mod.decode_legacy_sidecar(sidecar)
        if telemetry is None:
            telemetry_reason = (
                "private telemetry decoder unavailable, failed, or returned "
                "invalid data"
            )
        elif clock_reference_set is not None:
            telemetry = telemetry_mod.map_dcb_absolute_time(
                telemetry,
                clock_reference_set=clock_reference_set,
            )

    flash_result_id = (in_session_manifest or {}).get("flash_result_id")
    flash_input_identity = (in_session_manifest or {}).get(
        "flash_input_identity_sha256"
    )
    flash_identity_unavailable = (in_session_manifest or {}).get(
        "flash_input_identity_unavailable_reason"
    )
    result = _process_one_session(
        session_dir=session_dir,
        name=name,
        ordinal=ordinal,
        h5_dir=h5_dir,
        fits_dir=fits_dir,
        plots_dir=plots_dir,
        manifest_dir=manifest_dir,
        issue_collector=issue_collector,
        clock_reference_set=clock_reference_set,
        telemetry=telemetry,
        telemetry_source=telemetry_source,
        telemetry_reason=telemetry_reason,
        source_path=session_dir,
        source_kind="session",
        plot_names=plot_names,
        overwrite=overwrite,
        decoder_strict=decoder_strict,
        diagnostic_override=diagnostic_override,
        schema_variant=schema_variant,
        context_issues=issue_collector.since(issue_marker),
        input_identity=flash_input_identity,
        input_identity_kind=(
            "flash_source_sha256"
            if flash_input_identity is not None
            else None
        ),
        input_identity_unavailable_reason=flash_identity_unavailable,
        flash_result_id=flash_result_id,
    )
    manifest_flash = (in_session_manifest or {}).get("flash_source_path")
    if isinstance(manifest_flash, str) and manifest_flash:
        flash_path = Path(manifest_flash)
        if not flash_path.is_absolute():
            flash_path = session_dir / flash_path
        result.flash_source_path = str(flash_path.resolve())
    result.flash_source_fingerprint = (in_session_manifest or {}).get(
        "flash_source_fingerprint", {}
    )
    result.flash_manifest_sha256 = (in_session_manifest or {}).get(
        "flash_manifest_sha256"
    )
    result.flash_input_identity_sha256 = flash_input_identity
    result.flash_input_identity_unavailable_reason = (
        flash_identity_unavailable
    )
    if manifest_dir is not None and result.manifest_path:
        _validate_session_manifest_destination(
            session_dir,
            Path(result.manifest_path),
        )
        write_manifest(result, result.manifest_path)
    return result


def _validate_session_manifest_destination(
    session_dir: Path,
    destination: Path,
) -> None:
    protected_inputs = {
        (session_dir / IN_SESSION_MANIFEST_NAME).resolve(),
        (session_dir / PACKET_MAP_FILENAME).resolve(),
        (session_dir / LEGACY_TELEMETRY_SIDECAR_NAME).resolve(),
    }
    if destination.resolve() in protected_inputs:
        raise ValueError(
            "session manifest destination aliases an ingestion input"
        )


def process_session(
    session_dir: Path | str,
    *,
    landing_time_file: Path | str | None = None,
    h5_dir: Optional[Path | str] = None,
    fits_dir: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    manifest_dir: Optional[Path | str] = None,
    name: Optional[str] = None,
    ordinal: int = 0,
    plot_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
) -> SessionResult:
    """Process one extracted session and finalize its manifest."""
    _validate_overwrite(overwrite)
    collector = (
        issue_collector
        if issue_collector is not None
        else IssueCollector()
    )
    marker = collector.mark()
    resolved_session = Path(session_dir).resolve()
    if manifest_dir is not None and name is not None:
        protected_name = _validate_session_output_name(name)
        manifest_destination = (
            Path(manifest_dir).resolve() / f"{protected_name}.json"
        ).resolve()
        _validate_session_manifest_destination(
            resolved_session,
            manifest_destination,
        )
    return _process_session_impl(
        resolved_session,
        landing_time_file=landing_time_file,
        h5_dir=h5_dir,
        fits_dir=fits_dir,
        plots_dir=plots_dir,
        manifest_dir=manifest_dir,
        name=name,
        ordinal=ordinal,
        plot_names=plot_names,
        overwrite=overwrite,
        issue_collector=collector,
        decoder_strict=decoder_strict,
        diagnostic_override=diagnostic_override,
        schema_variant=schema_variant,
        issue_marker=marker,
    )


# ---------------------------------------------------------------------------
# Public: process_flash (raw flash mode)
# ---------------------------------------------------------------------------

def _flash_manifest_destinations(
    sessions_root: Path,
    manifest_dir: Path | None,
) -> List[Path]:
    canonical = (sessions_root / FLASH_MANIFEST_NAME).resolve()
    destinations: List[Path] = []
    if manifest_dir is not None:
        external = (manifest_dir / FLASH_MANIFEST_NAME).resolve()
        if external != canonical:
            destinations.append(external)
    destinations.append(canonical)
    return destinations


def _validate_session_output_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("session_name must return a nonempty string")
    candidate = Path(value)
    if candidate.name != value or value in (".", ".."):
        raise ValueError("session_name must return one safe path component")
    return value


def _preflight_flash_outputs(
    *,
    names: Sequence[str],
    sessions_root: Path,
    h5_dir: Path | None,
    fits_dir: Path | None,
    plots_dir: Path | None,
    manifest_dir: Path | None,
    run_manifest_destinations: Sequence[Path],
    protected_inputs: Sequence[Path],
) -> None:
    if len(set(names)) != len(names):
        raise ValueError("session_name returned duplicate output names")
    candidates: List[tuple[str, Path]] = [
        ("run_manifest", Path(path))
        for path in run_manifest_destinations
    ]
    for name in names:
        session_dir = sessions_root / name
        candidates.extend((
            ("session_directory", session_dir),
            (
                "in_session_manifest",
                session_dir / IN_SESSION_MANIFEST_NAME,
            ),
        ))
        if h5_dir is not None:
            candidates.append(("hdf5", h5_dir / f"{name}.h5"))
        if fits_dir is not None:
            candidates.append(("fits", fits_dir / f"{name}.fits"))
        if manifest_dir is not None:
            candidates.append((
                "session_manifest",
                manifest_dir / f"{name}.json",
            ))
        if plots_dir is not None and h5_dir is not None:
            candidates.append((
                "plot_directory",
                plots_dir / name,
            ))

    claimed: Dict[Path, tuple[str, Path]] = {}
    for label, destination in candidates:
        resolved = destination.resolve()
        previous = claimed.get(resolved)
        if previous is not None:
            previous_label, previous_path = previous
            raise ValueError(
                "FLASH output destinations alias each other: "
                f"{previous_label}={previous_path} and {label}={destination}"
            )
        claimed[resolved] = (label, destination)

    for protected_input in protected_inputs:
        protected = protected_input.resolve()
        for label, destination in candidates:
            resolved = destination.resolve()
            if resolved == protected or resolved in protected.parents:
                raise ValueError(
                    "FLASH output destination overlaps a protected input: "
                    f"{label}={destination}, input={protected_input}"
                )


def _set_flash_session_metadata(
    result: SessionResult,
    *,
    session: Session,
    flash_dir: Path,
    flash_fingerprint: Dict[str, Dict[str, object]],
    flash_result_id: str,
    input_identity_sha256: str | None,
    input_identity_unavailable_reason: str | None,
) -> None:
    result.flash_source_path = str(flash_dir)
    result.flash_source_fingerprint = flash_fingerprint
    result.flash_result_id = flash_result_id
    result.flash_input_identity_sha256 = input_identity_sha256
    result.flash_input_identity_unavailable_reason = (
        input_identity_unavailable_reason
    )
    result.start_raw_seconds = session.start_raw_seconds


def _issues_for_session(
    issues: Sequence[IngestIssue],
    *,
    session: Session,
    session_name: str,
) -> tuple[IngestIssue, ...]:
    session_labels = {
        session_name,
        str(session.ordinal),
        f"ordinal:{session.ordinal}",
    }
    session_uids = {
        packet.unique_packet_id
        for packet in session.packets
        if packet.unique_packet_id is not None
    }
    selected = []
    for issue in issues:
        if issue.session is not None:
            if issue.session in session_labels:
                selected.append(issue)
        elif issue.uid is not None and issue.uid in session_uids:
            selected.append(issue)
    return tuple(selected)


def _write_flash_manifests(
    result: FlashResult,
    destinations: Sequence[Path],
    *,
    payload: bytes,
    overwrite: bool,
) -> None:
    entries: List[tuple[Path, bytes]] = []
    for session in result.session_results:
        session_destinations = []
        if session.manifest_path is not None:
            session_destinations.append(Path(session.manifest_path))
        if session.session_dir is not None:
            session_destinations.append(
                Path(session.session_dir) / IN_SESSION_MANIFEST_NAME
            )
        for destination in session_destinations:
            entries.append((
                destination,
                _canonical_manifest_bytes(
                    _session_manifest_body(
                        session,
                        destination=destination,
                    )
                ),
            ))
    entries.extend((Path(destination), payload) for destination in destinations)
    for destination, manifest_payload in entries:
        _write_bytes(
            manifest_payload,
            destination,
            overwrite=overwrite,
        )

def process_flash(
    flash_dir: Path | str,
    *,
    landing_time_file: Path | str,
    sessions_root: Path | str,
    h5_dir: Optional[Path | str] = None,
    fits_dir: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    manifest_dir: Optional[Path | str] = None,
    session_name: Optional[SessionNamer] = None,
    plot_names: Optional[Sequence[str]] = None,
    overwrite: bool = False,
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    schema_variant: str | None = None,
) -> FlashResult:
    """Run the single-pass FLASH pipeline and return its run-level result."""
    _validate_overwrite(overwrite)
    landing_time_path = Path(landing_time_file).resolve()
    clock_reference_set = _load_landing_reference(landing_time_path)
    flash_dir = Path(flash_dir).resolve()
    sessions_root = Path(sessions_root).resolve()
    h5_dir = Path(h5_dir).resolve() if h5_dir else None
    fits_dir = Path(fits_dir).resolve() if fits_dir else None
    plots_dir = Path(plots_dir).resolve() if plots_dir else None
    manifest_dir = Path(manifest_dir).resolve() if manifest_dir else None
    if session_name is None:
        session_name = default_session_name
    collector = (
        issue_collector
        if issue_collector is not None
        else IssueCollector()
    )

    return _process_flash_run(
        flash_dir=flash_dir,
        clock_reference_set=clock_reference_set,
        landing_time_path=landing_time_path,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
        fits_dir=fits_dir,
        plots_dir=plots_dir,
        manifest_dir=manifest_dir,
        session_name=session_name,
        plot_names=plot_names,
        overwrite=overwrite,
        issue_collector=collector,
        decoder_strict=decoder_strict,
        schema_variant=schema_variant,
    )


def _flash_run_issues(
    issue_collector: IssueCollector,
    marker: int,
    *,
    prepared: Sequence[tuple[Session, str, Path, Products]],
) -> tuple[IngestIssue, ...]:
    groups: List[Sequence[IngestIssue]] = [issue_collector.since(marker)]
    groups.extend(product.issues for *_, product in prepared)
    return _merge_issues(*groups)


def _flash_decoder_provenance(
    prepared: Sequence[tuple[Session, str, Path, Products]],
) -> List[Dict[str, object]]:
    return [
        {
            "session_ordinal": session.ordinal,
            "session_name": name,
            **_decoder_provenance_manifest(products),
        }
        for session, name, _session_dir, products in prepared
    ]


def _process_flash_run(
    *,
    flash_dir: Path,
    clock_reference_set: ClockReferenceSet,
    landing_time_path: Path,
    sessions_root: Path,
    h5_dir: Path | None,
    fits_dir: Path | None,
    plots_dir: Path | None,
    manifest_dir: Path | None,
    session_name: SessionNamer,
    plot_names: Sequence[str] | None,
    overwrite: bool,
    issue_collector: IssueCollector,
    decoder_strict: bool,
    schema_variant: str | None,
) -> FlashResult:
    run_marker = issue_collector.mark()
    capture = _FlashParseCapture()
    results: List[SessionResult] = []
    prepared: List[tuple[Session, str, Path, Products]] = []
    sessions: List[Session] = []
    names: List[str] = []
    expected_binding: tuple[object, ...] | None = None
    flash_result_id, input_identity_sha256, identity_unavailable_reason = (
        _flash_identity(
            {},
            clock_reference_set,
        )
    )
    flash_fingerprint: Dict[str, Dict[str, object]] = {}
    parse_issues: tuple[IngestIssue, ...] = ()
    run_manifest_destinations = _flash_manifest_destinations(
        sessions_root,
        manifest_dir,
    )

    decoder_options = {} if schema_variant is None else {"schema_variant": schema_variant}
    sessions, _source_telemetry = _parse_flash_loaded(
        flash_dir,
        clock_reference_set=clock_reference_set,
        issue_collector=issue_collector,
        capture=capture,
        **decoder_options,
    )
    flash_fingerprint = _validate_flash_fingerprint(
        capture.source_fingerprint
    )
    parse_issues = issue_collector.since(run_marker)
    for session in sessions:
        names.append(_validate_session_output_name(session_name(
            session.ordinal,
            session.start_raw_seconds,
            clock_reference_set,
        )))
    _preflight_flash_outputs(
        names=names,
        sessions_root=sessions_root,
        h5_dir=h5_dir,
        fits_dir=fits_dir,
        plots_dir=plots_dir,
        manifest_dir=manifest_dir,
        run_manifest_destinations=run_manifest_destinations,
        protected_inputs=(
            flash_dir,
            landing_time_path,
            *(
                _bank_path(flash_dir, bank).resolve()
                for bank in (*SCIENCE_BANKS, TELEMETRY_BANK)
                if _input_path_present(_bank_path(flash_dir, bank))
            ),
        ),
    )
    for session, name in zip(sessions, names):
        session_dir = sessions_root / name
        if overwrite:
            write_uncrater_session(
                session,
                session_dir,
                overwrite=True,
                **decoder_options,
            )
        else:
            write_uncrater_session(session, session_dir, **decoder_options)
        products = read_uncrater_session(
            session_dir,
            strict=decoder_strict,
            schema_variant=schema_variant,
            issue_collector=issue_collector,
        )
        prepared.append((session, name, session_dir, products))
        binding = _binding_identity(products)
        if expected_binding is None:
            expected_binding = binding
        elif binding != expected_binding:
            raise RuntimeError(
                "derived FLASH sessions selected different decoder "
                "bindings; refusing all product writes. This is a "
                "conservative guard, not a forced input-wide binding"
            )

    (
        flash_result_id,
        input_identity_sha256,
        identity_unavailable_reason,
    ) = _flash_identity(
        flash_fingerprint,
        clock_reference_set,
        expected_binding,
        source_identity_complete=capture.source_identity_complete,
        source_identity_unavailable_reasons=(
            capture.source_identity_unavailable_reasons
        ),
    )
    telemetry_source = (
        "b01_0x314" if capture.telemetry_0x314_packets else None
    )
    for session, name, session_dir, products in prepared:
        telemetry_reason = (
            "private telemetry decoder unavailable, failed, returned invalid "
            "data, or telemetry could not be assigned to sessions"
            if telemetry_source is not None and session.telemetry is None
            else None
        )
        result = _process_one_session(
            session_dir=session_dir,
            name=name,
            ordinal=session.ordinal,
            h5_dir=h5_dir,
            fits_dir=fits_dir,
            plots_dir=plots_dir,
            manifest_dir=manifest_dir,
            issue_collector=issue_collector,
            clock_reference_set=clock_reference_set,
            telemetry=session.telemetry,
            telemetry_source=telemetry_source,
            telemetry_reason=telemetry_reason,
            source_path=flash_dir,
            source_kind="flash",
            plot_names=plot_names,
            overwrite=overwrite,
            products=products,
            context_issues=_issues_for_session(
                parse_issues,
                session=session,
                session_name=name,
            ),
            input_packet_count=len(session.packets),
            input_identity=input_identity_sha256,
            input_identity_kind=(
                "flash_source_sha256"
                if input_identity_sha256 is not None
                else None
            ),
            input_identity_unavailable_reason=(
                identity_unavailable_reason
            ),
            flash_result_id=flash_result_id,
        )
        _set_flash_session_metadata(
            result,
            session=session,
            flash_dir=flash_dir,
            flash_fingerprint=flash_fingerprint,
            flash_result_id=flash_result_id,
            input_identity_sha256=input_identity_sha256,
            input_identity_unavailable_reason=(
                identity_unavailable_reason
            ),
        )
        results.append(result)

    if not results:
        issue_collector.record(
            code="pipeline.no_usable_sessions",
            severity=IssueSeverity.ERROR,
            stage="session_split",
            message="the FLASH input produced no usable sessions",
            action=IssueAction.REJECTED,
        )
    run_issues = _flash_run_issues(
        issue_collector,
        run_marker,
        prepared=prepared,
    )
    issue_records, issue_counts, status_issue_codes = (
        _issue_manifest_fields(run_issues)
    )
    run_status = _status_from_issues(run_issues)
    if not results or all(result.status == "failed" for result in results):
        run_status = "failed"
    elif any(result.status != "clean" for result in results):
        run_status = "partial"
    if any(result.status != "clean" for result in results):
        status_issue_codes = sorted({
            *status_issue_codes,
            *(
                code
                for result in results
                for code in result.status_issue_codes
            ),
        })
    flash_result = FlashResult(
        flash_result_id=flash_result_id,
        input_identity_sha256=input_identity_sha256,
        input_identity_unavailable_reason=identity_unavailable_reason,
        source_path=str(flash_dir),
        source_fingerprint=flash_fingerprint,
        session_results=results,
        status=run_status,
        status_issue_codes=status_issue_codes,
        issue_counts=issue_counts,
        issues=issue_records,
        stage_counts=_flash_stage_counts(
            capture,
            run_issues,
            products=[product for *_, product in prepared],
            results=results,
        ),
        decoder_provenance=_flash_decoder_provenance(prepared),
        clock_reference=clock_reference_set.as_record(),
        pipeline_version=_pipeline_version_string(),
        processed_at_utc=_now_utc_iso(),
        manifest_paths=[str(path) for path in run_manifest_destinations],
        overwrite=overwrite,
        source_identity_unavailable_reasons=list(
            capture.source_identity_unavailable_reasons
        ),
    )
    manifest_payload = _canonical_manifest_bytes(
        _flash_manifest_body(
            flash_result,
            path_parent=sessions_root,
        )
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    flash_result.manifest_sha256 = manifest_sha256
    for result in results:
        result.flash_manifest_sha256 = manifest_sha256

    _write_flash_manifests(
        flash_result,
        run_manifest_destinations,
        payload=manifest_payload,
        overwrite=overwrite,
    )
    return flash_result
