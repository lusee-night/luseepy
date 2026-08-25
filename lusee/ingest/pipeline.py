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
import tempfile
import warnings
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    IngestIssueError,
    IssueAction,
    IssueCollector,
    IssueSeverity,
)
from .packet_map import PACKET_MAP_FILENAME, PACKET_MAP_FORMAT_VERSION
from .reassembly import LogicalPacket, reassemble_logical_packets
from .session import (
    Session,
    assign_telemetry_to_sessions,
    split_sessions,
    write_uncrater_session,
)
from .write_request import (
    FAMILY_TYPES,
    InterpolationPolicy,
    LunarLocation,
    RunProvenance,
    WriteRequest,
    family_statuses_for_products,
)

log = logging.getLogger(__name__)

# Schema version of session.json -- bump when the manifest layout changes.
MANIFEST_SCHEMA_VERSION = 3
TELEMETRY_ASSIGNMENT_CONTRACT_VERSION = 2

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

    has_telemetry: bool = False
    has_legacy_sidecar: bool = False

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

    # Flash backreference: lets a later process_session run re-derive
    # /DCB_telemetry/ from the original raw bank without writing a binary
    # sidecar. None / empty for sessions produced by process_session.
    flash_source_path: Optional[str] = None
    flash_source_fingerprint: Dict[str, Dict[str, object]] = field(
        default_factory=dict
    )
    flash_result_id: Optional[str] = None
    flash_manifest_sha256: Optional[str] = None
    # Shared elapsed-time window used to re-derive b01 telemetry. These
    # bounds are spectrometer-session starts relative to the landing event;
    # DCB rows are compared only after subtracting their separate DCB anchor.
    start_raw_seconds: Optional[float] = None
    telemetry_window_lower_elapsed_seconds: Optional[float] = None
    telemetry_window_upper_elapsed_seconds: Optional[float] = None
    telemetry_assignment_mode: Optional[str] = None
    telemetry_assignment_contract_version: int = (
        TELEMETRY_ASSIGNMENT_CONTRACT_VERSION
    )
    telemetry_unassigned_upper_elapsed_seconds: Optional[float] = None
    telemetry_assignment_issues: List[Dict[str, object]] = field(
        default_factory=list
    )

    telemetry_source: Optional[str] = None
    """How telemetry reached the HDF5 for this session: 'flash' (single-pass
    or re-derived via flash backreference), 'sidecar' (legacy
    DCB_telemetry.json), or None."""

    processed_at_utc: str = ""
    pipeline_version: str = ""
    clock_reference: Optional[Dict[str, object]] = None
    packet_map_status: str = "unavailable"
    packet_map_format_version: Optional[int] = None
    raw_flash_provenance_unavailable_reason: Optional[str] = None
    overwrite: bool = False
    committed_artifacts: List[str] = field(default_factory=list)
    telemetry_input_sources: List[str] = field(default_factory=list)
    telemetry_decoder_status: str = "not_needed"
    telemetry_coverage: str = "absent"
    n_telemetry_rows: int = 0
    n_unassigned_telemetry_rows: int = 0
    telemetry_provenance: Optional[Dict[str, object]] = None
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
    telemetry_provenance: Optional[Dict[str, object]] = None
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


def _broken_b01_telemetry(
    issue_collector: IssueCollector,
    *,
    issue_marker: int,
    error_type: str,
) -> telemetry_mod.TelemetryDecodeResult:
    issue_collector.record(
        code="telemetry_input.b01_unreadable",
        severity="error",
        stage="telemetry_input",
        message=(
            "the present b01 telemetry source could not be read; independent "
            "science data was retained"
        ),
        action="rejected",
        bank=TELEMETRY_BANK,
        details={"error_type": error_type},
    )
    return telemetry_mod.TelemetryDecodeResult(
        input_source="b01",
        input_state=telemetry_mod.TelemetryInputState.PRESENT,
        decoder_status=telemetry_mod.TelemetryDecoderStatus.BROKEN,
        coverage=telemetry_mod.TelemetryCoverage.BROKEN,
        issues=issue_collector.since(issue_marker),
    )


def _unreachable_b01_telemetry(
    issue_collector: IssueCollector,
    *,
    rederive_enabled: bool,
) -> telemetry_mod.TelemetryDecodeResult:
    marker = issue_collector.mark()
    reason = (
        "flash_backreference_unreachable"
        if rederive_enabled
        else "telemetry_rederive_disabled"
    )
    issue_collector.record(
        code="telemetry_input.b01_unavailable",
        severity="error",
        stage="telemetry_input",
        message=(
            "the session records a b01 telemetry input, but it could not be "
            "re-derived; independent extracted science data was retained"
        ),
        action="rejected",
        bank=TELEMETRY_BANK,
        details={"reason": reason},
    )
    return telemetry_mod.TelemetryDecodeResult(
        input_source="b01",
        input_state=telemetry_mod.TelemetryInputState.PRESENT,
        decoder_status=telemetry_mod.TelemetryDecoderStatus.BROKEN,
        coverage=telemetry_mod.TelemetryCoverage.BROKEN,
        issues=issue_collector.since(marker),
    )


def _recorded_telemetry_sources(
    manifest: Dict[str, Any] | None,
) -> set[str]:
    if manifest is None:
        return set()
    value = manifest.get("telemetry_input_sources", [])
    if not isinstance(value, list) or any(
        source not in ("b01", "legacy_sidecar") for source in value
    ):
        raise ValueError("session manifest has invalid telemetry_input_sources")
    sources = set(value)
    fingerprint = manifest.get("flash_source_fingerprint", {})
    if isinstance(fingerprint, dict) and (
        f"{TELEMETRY_BANK}/{BANK_FILENAME}" in fingerprint
    ):
        sources.add("b01")
    return sources


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
    telemetry_mod.TelemetryDecodeResult,
    telemetry_mod.TelemetryDecodeResult | None,
]:
    """Parse a FLASH_TLMFS directory through Stage 4.

    Returns the sessions, the source-scoped b01 result, and any rows that
    remain explicitly unassigned. Per-session typed slices are stored on
    each ``Session``.
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
) -> Tuple[
    List[Session],
    telemetry_mod.TelemetryDecodeResult,
    telemetry_mod.TelemetryDecodeResult | None,
]:
    """Parse a flash after the public entry point validates mission time."""
    flash_dir = Path(flash_dir)
    if issue_collector is None:
        issue_collector = IssueCollector()
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
    telemetry_issue_marker = issue_collector.mark()
    telemetry_read_error = None
    if telemetry_present and tpath.is_file():
        log.info("reading telemetry bank %s", tpath)
        try:
            telem_packets.extend(reassemble_logical_packets(
                _frames_from_bank(
                    tpath,
                    bank=TELEMETRY_BANK,
                    issue_collector=issue_collector,
                    capture=capture,
                ),
                byteswap_pairs=False,
                bank=TELEMETRY_BANK,
                issue_collector=issue_collector,
            ))
            if capture is not None:
                capture.telemetry_logical_packets += len(telem_packets)
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

    sessions = split_sessions(
        science_packets,
        issue_collector=issue_collector,
    )
    if capture is not None:
        capture.session_count = len(sessions)
        capture.session_input_packets = sum(
            len(session.packets) for session in sessions
        )

    if telemetry_read_error is not None:
        telemetry = _broken_b01_telemetry(
            issue_collector,
            issue_marker=telemetry_issue_marker,
            error_type=telemetry_read_error,
        )
    else:
        telemetry = telemetry_mod.decode_b01_packets(
            telem_packets if telemetry_present else None,
            issue_collector=issue_collector,
        )
        if telemetry_present:
            telemetry = telemetry.with_issues(
                issue_collector.since(telemetry_issue_marker)
            )
    unassigned = assign_telemetry_to_sessions(
        sessions,
        telemetry,
        clock_reference_set=clock_reference_set,
        issue_collector=issue_collector,
    )
    return sessions, telemetry, unassigned


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


def _write_bytes_atomic(
    payload: bytes,
    dest: Path,
    *,
    overwrite: bool,
) -> Path:
    _validate_overwrite(overwrite)
    dest.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f".{dest.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary, dest)
        else:
            os.link(temporary, dest)
    finally:
        if temporary.exists():
            temporary.unlink()
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
        "telemetry_provenance": result.telemetry_provenance,
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
                "telemetry_input_sources": session.telemetry_input_sources,
                "telemetry_decoder_status": session.telemetry_decoder_status,
                "telemetry_coverage": session.telemetry_coverage,
                "n_telemetry_rows": session.n_telemetry_rows,
                "n_unassigned_telemetry_rows": (
                    session.n_unassigned_telemetry_rows
                ),
                "telemetry_provenance": session.telemetry_provenance,
                "telemetry_assignment_mode": (
                    session.telemetry_assignment_mode
                ),
                "start_raw_seconds": session.start_raw_seconds,
                "telemetry_window_lower_elapsed_seconds": (
                    session.telemetry_window_lower_elapsed_seconds
                ),
                "telemetry_window_upper_elapsed_seconds": (
                    session.telemetry_window_upper_elapsed_seconds
                ),
                "telemetry_assignment_contract_version": (
                    session.telemetry_assignment_contract_version
                ),
                "telemetry_unassigned_upper_elapsed_seconds": (
                    session.telemetry_unassigned_upper_elapsed_seconds
                ),
                "telemetry_assignment_issues": (
                    session.telemetry_assignment_issues
                ),
                "telemetry_source": session.telemetry_source,
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
    """Atomically serialize one final session result as strict ASCII JSON."""
    if not isinstance(result, SessionResult):
        raise TypeError("session manifest requires a SessionResult")
    dest = Path(dest_path)
    return _write_bytes_atomic(
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
    return _write_bytes_atomic(
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
    telemetry: telemetry_mod.TelemetryDecodeResult | None = None,
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
    counts["products"]["dcb_telemetry"] = (
        telemetry.fpga.row_count
        if telemetry is not None and telemetry.fpga is not None
        else 0
    )
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
        for family in sorted(persistence_families - {"dcb_telemetry"})
    }
    persisted_telemetry_results = [
        result
        for result in results
        if result.persisted_rows_by_family.get("dcb_telemetry", 0) > 0
    ]
    counts["persistence"]["dcb_telemetry"] = (
        sum(result.n_telemetry_rows for result in persisted_telemetry_results)
        + max(
            (
                result.n_unassigned_telemetry_rows
                for result in persisted_telemetry_results
            ),
            default=0,
        )
    )
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


def _telemetry_provenance_manifest(
    telemetry: telemetry_mod.TelemetryDecodeResult | None,
    interpolation_policy: InterpolationPolicy,
) -> Optional[Dict[str, object]]:
    if telemetry is None:
        return None
    decoder = telemetry.decoder_info
    counts = telemetry.counts
    return {
        "selected_source": telemetry.input_source,
        "input_state": telemetry.input_state.value,
        "decoder_status": telemetry.decoder_status.value,
        "coverage": telemetry.coverage.value,
        "counts_scope": telemetry.counts_scope,
        "decoder": (
            {
                "api_version": decoder.api_version,
                "decoder_name": decoder.decoder_name,
                "decoder_version": decoder.decoder_version,
                "claimed_appids": list(decoder.claimed_appids),
            }
            if decoder is not None
            else None
        ),
        "counts": (
            {
                "kind": counts.source,
                "scalar_counts": dict(counts.scalar_counts),
                "claimed_appid_counts": [
                    {"appid": appid, "packet_count": count}
                    for appid, count in counts.claimed_appid_counts
                ],
                "unclaimed_appid_counts": [
                    {"appid": appid, "packet_count": count}
                    for appid, count in counts.unclaimed_appid_counts
                ],
            }
            if counts is not None
            else None
        ),
        "field_metadata": [asdict(metadata) for metadata in telemetry.field_metadata],
        "interpolation_policy": {
            "mode": interpolation_policy.mode,
            "maximum_gap_seconds": interpolation_policy.maximum_gap_seconds,
            "extrapolate": interpolation_policy.extrapolate,
        },
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
    telemetry: telemetry_mod.TelemetryDecodeResult | None = None,
    has_legacy_sidecar: bool = False,
    telemetry_input_sources: Sequence[str] = (),
    telemetry_decoder_status: str = "not_needed",
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
    result_sink: List[SessionResult] | None = None,
) -> SessionResult:
    _validate_overwrite(overwrite)
    if (
        input_identity is not None
        and input_identity_unavailable_reason is not None
    ):
        raise ValueError(
            "available input identity cannot have an unavailable reason"
        )
    if result_sink is not None and not isinstance(result_sink, list):
        raise TypeError("result_sink must be a list or None")
    if products is None:
        products = read_uncrater_session(
            session_dir,
            issue_collector=issue_collector,
            strict=decoder_strict,
            diagnostic_override=diagnostic_override,
            schema_variant=schema_variant,
        )

    has_telemetry = (
        telemetry is not None
        and telemetry.input_state is not telemetry_mod.TelemetryInputState.ABSENT
    )
    telemetry_input_sources = tuple(sorted(set(telemetry_input_sources)))
    if any(
        source not in ("b01", "legacy_sidecar")
        for source in telemetry_input_sources
    ):
        raise ValueError("unknown telemetry input source")
    if telemetry_decoder_status not in (
        "available",
        "unavailable",
        "broken",
        "incompatible",
        "not_needed",
    ):
        raise ValueError("invalid telemetry decoder status")
    if telemetry_input_sources and telemetry_decoder_status == "not_needed":
        raise ValueError("present telemetry requires an explicit decoder status")
    if telemetry_input_sources and (
        telemetry is None
        or telemetry.input_state is telemetry_mod.TelemetryInputState.ABSENT
    ):
        raise ValueError(
            "present telemetry requires its typed decoder result"
        )
    if not telemetry_input_sources and telemetry_decoder_status != "not_needed":
        raise ValueError("absent telemetry must use decoder status not_needed")
    if telemetry is not None:
        if telemetry_decoder_status != telemetry.decoder_status.value:
            raise ValueError("telemetry result and decoder status disagree")
        expected_source = (
            "legacy_sidecar"
            if telemetry.input_source == "legacy_sidecar"
            else telemetry.input_source
        )
        if (
            expected_source is not None
            and expected_source not in telemetry_input_sources
        ):
            raise ValueError("telemetry result and input sources disagree")
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
    request_telemetry = telemetry or telemetry_mod.TelemetryDecodeResult.absent()
    interpolation_policy = InterpolationPolicy()
    issue_by_id: Dict[str, IngestIssue] = {}
    for issue in (
        *product_issues,
        *request_telemetry.issues,
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
        telemetry=request_telemetry,
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
        has_telemetry=has_telemetry,
        has_legacy_sidecar=has_legacy_sidecar,
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
            "telemetry_assignment_contract_version": (
                TELEMETRY_ASSIGNMENT_CONTRACT_VERSION
            ),
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
        telemetry_input_sources=list(telemetry_input_sources),
        telemetry_decoder_status=telemetry_decoder_status,
        telemetry_coverage=(
            telemetry.coverage.value
            if telemetry is not None
            else telemetry_mod.TelemetryCoverage.ABSENT.value
        ),
        n_telemetry_rows=(
            telemetry.fpga.row_count
            if telemetry is not None and telemetry.fpga is not None
            else 0
        ),
        n_unassigned_telemetry_rows=(
            telemetry.unassigned_fpga.row_count
            if telemetry is not None and telemetry.unassigned_fpga is not None
            else 0
        ),
        telemetry_provenance=_telemetry_provenance_manifest(
            request_telemetry,
            interpolation_policy,
        ),
    )
    if result_sink is not None:
        result_sink.append(result)

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
    if manifest_path is not None:
        result.manifest_path = str(manifest_path.resolve())
    if not overwrite:
        for destination in (h5_path, fits_path, manifest_path):
            if destination is not None and _input_path_present(destination):
                raise FileExistsError(destination)

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
            telemetry=request_telemetry,
            interpolation_policy=interpolation_policy,
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

    if plots_dir is not None and h5_path is not None:
        from . import viz as viz_mod

        plot_dest = plots_dir / name
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
                "telemetry_input_sources",
                "telemetry_decoder_status",
                "telemetry_coverage",
                "n_telemetry_rows",
                "n_unassigned_telemetry_rows",
                "telemetry_provenance",
                "telemetry_assignment_mode",
                "start_raw_seconds",
                "telemetry_window_lower_elapsed_seconds",
                "telemetry_window_upper_elapsed_seconds",
                "telemetry_assignment_contract_version",
                "telemetry_unassigned_upper_elapsed_seconds",
                "telemetry_assignment_issues",
                "telemetry_source",
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


def _manifest_optional_float(
    manifest: Dict[str, Any],
    name: str,
) -> float | None:
    value = manifest.get(name)
    if value is None:
        return None
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(f"session manifest has invalid {name}")
    return float(value)


def _telemetry_assignment_from_manifest(
    manifest: Dict[str, Any],
    *,
    clock_reference_set: ClockReferenceSet | None,
) -> tuple[
    float | None,
    float | None,
    str | None,
    float | None,
    list[Dict[str, object]],
]:
    """Read current assignment fields or convert the original v3 window."""
    assignment_issues = manifest.get("telemetry_assignment_issues", [])
    if not isinstance(assignment_issues, list) or any(
        not isinstance(issue, dict) for issue in assignment_issues
    ):
        raise ValueError(
            "session manifest has invalid telemetry_assignment_issues"
        )

    current_names = (
        "telemetry_window_lower_elapsed_seconds",
        "telemetry_window_upper_elapsed_seconds",
        "telemetry_assignment_mode",
        "telemetry_unassigned_upper_elapsed_seconds",
    )
    legacy_names = (
        "telemetry_window_lower_raw_seconds",
        "telemetry_window_upper_raw_seconds",
    )
    has_current = any(name in manifest for name in current_names)
    has_legacy = any(name in manifest for name in legacy_names)
    if has_current and has_legacy:
        raise ValueError("session manifest mixes telemetry assignment contracts")

    contract_version = manifest.get("telemetry_assignment_contract_version")
    if contract_version is not None and type(contract_version) is not int:
        raise ValueError(
            "session manifest has invalid telemetry assignment contract version"
        )
    if contract_version is None:
        contract_version = 2 if has_current else 1 if has_legacy else 2
    if contract_version not in (1, TELEMETRY_ASSIGNMENT_CONTRACT_VERSION):
        raise ValueError(
            "session manifest has unsupported telemetry assignment contract"
        )
    if contract_version == 1 and has_current:
        raise ValueError(
            "session manifest assignment contract version disagrees with its fields"
        )
    if contract_version == TELEMETRY_ASSIGNMENT_CONTRACT_VERSION and has_legacy:
        raise ValueError(
            "session manifest assignment contract version disagrees with its fields"
        )

    if contract_version == TELEMETRY_ASSIGNMENT_CONTRACT_VERSION:
        mode = manifest.get("telemetry_assignment_mode")
        if mode not in (
            None,
            "assigned",
            "assigned_with_pre_session",
            "all_unassigned",
        ):
            raise ValueError("session manifest has invalid telemetry_assignment_mode")
        return (
            _manifest_optional_float(
                manifest, "telemetry_window_lower_elapsed_seconds"
            ),
            _manifest_optional_float(
                manifest, "telemetry_window_upper_elapsed_seconds"
            ),
            mode,
            _manifest_optional_float(
                manifest, "telemetry_unassigned_upper_elapsed_seconds"
            ),
            assignment_issues,
        )

    lower_raw = _manifest_optional_float(
        manifest, "telemetry_window_lower_raw_seconds"
    )
    upper_raw = _manifest_optional_float(
        manifest, "telemetry_window_upper_raw_seconds"
    )
    start_raw = _manifest_optional_float(manifest, "start_raw_seconds")
    if lower_raw is not None and start_raw is not None and lower_raw != start_raw:
        raise ValueError(
            "legacy session manifest telemetry lower bound disagrees with "
            "start_raw_seconds"
        )
    if clock_reference_set is None or start_raw is None:
        return None, None, None, None, assignment_issues
    reference = clock_reference_set.require_reference(ClockSource.SPECTROMETER)
    anchor = reference.clock_reference_raw_seconds
    lower = (lower_raw if lower_raw is not None else start_raw) - anchor
    upper = None if upper_raw is None else upper_raw - anchor
    if upper is not None and upper <= lower:
        raise ValueError("legacy session manifest has an empty telemetry window")
    if lower_raw is None:
        assignment_issues = [
            *assignment_issues,
            {
                "code": "telemetry_assignment.legacy_window_converted",
                "severity": "warning",
                "message": (
                    "pre-session rows from the original v3 raw-time window "
                    "were retained as explicitly unassigned telemetry"
                ),
                "action": "kept",
                "details": {"assignment_contract_from": 1},
            },
        ]
        return (
            lower,
            upper,
            "assigned_with_pre_session",
            lower,
            assignment_issues,
        )
    return lower, upper, "assigned", None, assignment_issues


def _rederive_telemetry_from_flash(
    flash_dir: Path,
    *,
    window_lower_elapsed_seconds: Optional[float],
    window_upper_elapsed_seconds: Optional[float],
    clock_reference_set: ClockReferenceSet | None,
    assignment_mode: str | None = None,
    unassigned_upper_elapsed_seconds: float | None = None,
    assignment_issues: Sequence[Dict[str, object]] = (),
    issue_collector: IssueCollector | None = None,
    expected_source_fingerprint: object | None = None,
    allow_changed_flash_source: bool = False,
) -> telemetry_mod.TelemetryDecodeResult:
    """Re-parse b01 and slice it in the shared landing-event coordinate."""
    tpath = _bank_path(flash_dir, TELEMETRY_BANK)
    if not _input_path_present(tpath):
        return telemetry_mod.TelemetryDecodeResult.absent()
    if issue_collector is None:
        issue_collector = IssueCollector()
    telemetry_issue_marker = issue_collector.mark()
    if not tpath.is_file():
        return _broken_b01_telemetry(
            issue_collector,
            issue_marker=telemetry_issue_marker,
            error_type="not_regular_file",
        )
    log.info("re-deriving telemetry from %s", tpath)
    try:
        frames = None
        if expected_source_fingerprint is not None:
            key = f"{TELEMETRY_BANK}/{BANK_FILENAME}"
            try:
                expected = _validate_flash_fingerprint(
                    expected_source_fingerprint
                )
            except ValueError as exc:
                issue_collector.record(
                    code="source.flash_fingerprint_invalid",
                    severity=IssueSeverity.ERROR,
                    stage="source_validation",
                    message=(
                        "the recorded FLASH fingerprint cannot authorize "
                        "telemetry re-derivation"
                    ),
                    action=IssueAction.REJECTED,
                    bank=TELEMETRY_BANK,
                    details={"error_type": type(exc).__name__},
                )
                return telemetry_mod.TelemetryDecodeResult(
                    input_source="b01",
                    input_state=telemetry_mod.TelemetryInputState.PRESENT,
                    decoder_status=telemetry_mod.TelemetryDecoderStatus.BROKEN,
                    coverage=telemetry_mod.TelemetryCoverage.BROKEN,
                    issues=issue_collector.since(telemetry_issue_marker),
                )
            diagnostic = parse_bank_file_diagnostic(
                tpath,
                bank=TELEMETRY_BANK,
                issue_collector=issue_collector,
            )
            observed_record = {
                "size_bytes": diagnostic.input_size_bytes,
                "sha256": diagnostic.input_sha256,
            }
            if expected.get(key) != observed_record:
                action = (
                    IssueAction.OVERRIDDEN
                    if allow_changed_flash_source
                    else IssueAction.REJECTED
                )
                issue_collector.record(
                    code="source.flash_changed",
                    severity=IssueSeverity.WARNING if allow_changed_flash_source else IssueSeverity.ERROR,
                    stage="source_validation",
                    message=(
                        "the b01 bytes no longer match the recorded FLASH "
                        "source identity"
                    ),
                    action=action,
                    bank=TELEMETRY_BANK,
                    details={
                        "expected": expected.get(key),
                        "observed": observed_record,
                    },
                )
                if not allow_changed_flash_source:
                    return telemetry_mod.TelemetryDecodeResult(
                        input_source="b01",
                        input_state=telemetry_mod.TelemetryInputState.PRESENT,
                        decoder_status=(
                            telemetry_mod.TelemetryDecoderStatus.BROKEN
                        ),
                        coverage=telemetry_mod.TelemetryCoverage.BROKEN,
                        issues=issue_collector.since(telemetry_issue_marker),
                    )
            frames = diagnostic.frames
        if frames is None:
            frames = parse_bank_file(
                tpath,
                bank=TELEMETRY_BANK,
                issue_collector=issue_collector,
            )
        telem_packets = list(reassemble_logical_packets(
            frames,
            byteswap_pairs=False,
            bank=TELEMETRY_BANK,
            issue_collector=issue_collector,
        ))
    except OSError as exc:
        return _broken_b01_telemetry(
            issue_collector,
            issue_marker=telemetry_issue_marker,
            error_type=type(exc).__name__,
        )
    result = telemetry_mod.decode_b01_packets(
        telem_packets,
        issue_collector=issue_collector,
    )
    result = result.with_issues(
        issue_collector.since(telemetry_issue_marker)
    )
    if (
        result.decoder_status is not telemetry_mod.TelemetryDecoderStatus.AVAILABLE
        or result.fpga is None
        or result.fpga.row_count == 0
    ):
        return result
    dcb_reference = (
        clock_reference_set.reference_for(ClockSource.DCB)
        if clock_reference_set is not None
        else None
    )
    mapped = result
    if dcb_reference is not None:
        mapped = telemetry_mod.map_dcb_absolute_time(
            result,
            clock_reference_set=clock_reference_set,
            issue_collector=issue_collector,
        )
    assert mapped.fpga is not None

    def replay_assignment_issues():
        replayed = []
        for record in assignment_issues:
            if not isinstance(record, dict):
                raise ValueError(
                    "telemetry_assignment_issues must contain JSON objects"
                )
            code = record.get("code")
            message = record.get("message")
            details = record.get("details", {})
            if (
                not isinstance(code, str)
                or not code.startswith("telemetry_assignment.")
                or not isinstance(message, str)
                or not isinstance(details, dict)
            ):
                raise ValueError("invalid recorded telemetry assignment issue")
            replayed.append(issue_collector.record(
                code=code,
                severity=record.get("severity", "warning"),
                stage="telemetry_assignment",
                message=message,
                action=record.get("action", "kept"),
                details=details,
            ))
        return tuple(replayed)

    def keep_all_unassigned(*, missing: Sequence[str]):
        replayed = replay_assignment_issues()
        if missing or not replayed:
            replayed = (*replayed, issue_collector.record(
                code="telemetry_assignment.rederive_unassigned",
                severity="warning",
                stage="telemetry_assignment",
                message=(
                    "re-derived b01 telemetry was retained unassigned because "
                    "it could not be placed safely from the recorded policy"
                ),
                action="kept",
                details={"missing": list(missing)},
            ))
        empty = np.zeros(mapped.fpga.row_count, dtype=np.bool_)
        return mapped.with_blocks(
            fpga=mapped.fpga.slice_rows(empty),
            unassigned_fpga=mapped.fpga,
            issues=(*mapped.issues, *replayed),
            coverage=telemetry_mod.TelemetryCoverage.PARTIAL,
        )

    if assignment_mode == "all_unassigned":
        return keep_all_unassigned(missing=())
    if assignment_mode not in ("assigned", "assigned_with_pre_session"):
        return keep_all_unassigned(missing=("telemetry_assignment_mode",))
    if dcb_reference is None or window_lower_elapsed_seconds is None:
        missing = []
        if dcb_reference is None:
            missing.append(ClockSource.DCB.value)
        if clock_reference_set is None:
            missing.append("clock_reference_set")
        if window_lower_elapsed_seconds is None:
            missing.append("session_elapsed_lower_bound")
        return keep_all_unassigned(missing=missing)
    elapsed = mapped.fpga.raw_seconds - dcb_reference.clock_reference_raw_seconds
    selector = elapsed >= window_lower_elapsed_seconds
    if window_upper_elapsed_seconds is not None:
        selector &= elapsed < window_upper_elapsed_seconds
    unassigned = None
    if assignment_mode == "assigned_with_pre_session":
        if unassigned_upper_elapsed_seconds is None:
            return keep_all_unassigned(
                missing=("telemetry_unassigned_upper_elapsed_seconds",)
            )
        unassigned = mapped.fpga.slice_rows(
            elapsed < unassigned_upper_elapsed_seconds
        )
    replayed = replay_assignment_issues()
    return mapped.with_blocks(
        fpga=mapped.fpga.slice_rows(selector),
        unassigned_fpga=unassigned,
        issues=(*mapped.issues, *replayed),
        coverage=(
            telemetry_mod.TelemetryCoverage.PARTIAL
            if replayed
            else mapped.coverage
        ),
    )


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
    flash_root: Optional[Path | str] = None,
    rederive_telemetry: bool = True,
    allow_changed_flash_source: bool = False,
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
    issue_marker: int | None = None,
    result_sink: List[SessionResult] | None = None,
) -> SessionResult:
    """Process one already-extracted uncrater session directory.

    Telemetry resolution order:

    1. If ``rederive_telemetry`` and the in-session ``session.json``
       carries a reachable ``flash_source_path`` (or ``flash_root`` is
       given as an override), re-parse the b01 bank and slice records to
       the session's mission-time window. This is the preferred source.
    2. Else, if a legacy ``DCB_telemetry.json`` sidecar exists, read it.
    3. Else, no telemetry.

    Writes HDF5 / FITS / plots / manifest to the caller-supplied parent
    directories. Each output is opt-in (pass None to skip).

    ``decoder_strict``, ``diagnostic_override``, and ``schema_variant`` are
    forwarded to the one uncrater Collection for this standalone session.
    """
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
    if type(allow_changed_flash_source) is not bool:
        raise TypeError("allow_changed_flash_source must be a boolean")

    telemetry: telemetry_mod.TelemetryDecodeResult | None = None
    telemetry_source: Optional[str] = None

    in_session_manifest = _read_in_session_manifest(
        session_dir,
        strict=output_requested or landing_time_file is not None,
        issue_collector=issue_collector,
    )
    telemetry_input_sources = _recorded_telemetry_sources(in_session_manifest)
    legacy_flash_fingerprint = bool(
        (in_session_manifest or {}).get(
            "_legacy_flash_source_fingerprint",
            False,
        )
    )
    public_sidecar = session_dir / LEGACY_TELEMETRY_SIDECAR_NAME
    if _input_path_present(public_sidecar):
        telemetry_input_sources.add("legacy_sidecar")
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
    flash_used: Optional[Path] = None

    manifest_flash = (in_session_manifest or {}).get("flash_source_path")
    if flash_root is not None:
        candidate = Path(flash_root).resolve()
    elif manifest_flash:
        candidate = Path(manifest_flash)
        if not candidate.is_absolute():
            candidate = (session_dir / candidate).resolve()
    else:
        candidate = None
    if (
        candidate is not None
        and candidate.is_dir()
        and _input_path_present(_bank_path(candidate, TELEMETRY_BANK))
    ):
        telemetry_input_sources.add("b01")

    b01_failure = None
    if rederive_telemetry:
        if candidate is not None and candidate.is_dir():
            recorded_fp = (in_session_manifest or {}).get(
                "flash_source_fingerprint", {}
            )
            (
                lower,
                upper,
                assignment_mode,
                unassigned_upper,
                assignment_issues,
            ) = _telemetry_assignment_from_manifest(
                in_session_manifest or {},
                clock_reference_set=clock_reference_set,
            )
            telemetry = _rederive_telemetry_from_flash(
                candidate,
                window_lower_elapsed_seconds=lower,
                window_upper_elapsed_seconds=upper,
                clock_reference_set=clock_reference_set,
                assignment_mode=assignment_mode,
                unassigned_upper_elapsed_seconds=unassigned_upper,
                assignment_issues=assignment_issues,
                issue_collector=issue_collector,
                expected_source_fingerprint=recorded_fp,
                allow_changed_flash_source=allow_changed_flash_source,
            )
            if telemetry.input_source is not None:
                flash_used = candidate
                if (
                    telemetry.decoder_status
                    is telemetry_mod.TelemetryDecoderStatus.AVAILABLE
                ):
                    telemetry_source = "flash"
                else:
                    b01_failure = telemetry
                    telemetry = None
        elif manifest_flash:
            log.info("flash backreference %s is not reachable; falling back",
                     manifest_flash)

    if (
        telemetry_source is None
        and b01_failure is None
        and "b01" in telemetry_input_sources
    ):
        b01_failure = _unreachable_b01_telemetry(
            issue_collector,
            rederive_enabled=rederive_telemetry,
        )

    sidecar_was_recorded = "legacy_sidecar" in telemetry_input_sources
    sidecar = telemetry_mod.find_legacy_sidecar(session_dir)
    if sidecar is not None:
        telemetry_input_sources.add("legacy_sidecar")
    sidecar_failure = None
    if sidecar is None and sidecar_was_recorded:
        marker = issue_collector.mark()
        sidecar_failure = telemetry_mod.failed_input_result(
            input_source="legacy_sidecar",
            error="recognized sidecar is no longer reachable",
            issue_collector=issue_collector,
            marker=marker,
        )
    if telemetry_source is None and sidecar is not None:
        log.info("reading legacy DCB_telemetry sidecar at %s", sidecar)
        telemetry = telemetry_mod.decode_legacy_sidecar(
            sidecar,
            issue_collector=issue_collector,
        )
        if clock_reference_set is not None:
            telemetry = telemetry_mod.map_dcb_absolute_time(
                telemetry,
                clock_reference_set=clock_reference_set,
                issue_collector=issue_collector,
            )
        telemetry_source = "sidecar"
        if b01_failure is not None:
            fallback_issue = issue_collector.record(
                code="telemetry_input.source_fallback",
                severity="warning",
                stage="telemetry_input",
                message=(
                    "the legacy telemetry sidecar was selected because the "
                    "recorded b01 source was unusable"
                ),
                action="kept",
                details={
                    "selected_source": "legacy_sidecar",
                    "unusable_source": "b01",
                    "selection_policy": "prefer_usable_b01_else_sidecar",
                },
            )
            telemetry = telemetry.with_issues(
                (*b01_failure.issues, *telemetry.issues, fallback_issue)
            )
    elif telemetry_source == "flash" and sidecar is not None:
        log.info("ignoring legacy sidecar %s in favor of flash backreference",
                 sidecar)
        selection_issue = issue_collector.record(
            code="telemetry_input.source_not_selected",
            severity="warning",
            stage="telemetry_input",
            message=(
                "a legacy telemetry sidecar was not decoded because the "
                "recorded b01 source has precedence"
            ),
            action="rejected",
            details={
                "selected_source": "b01",
                "unselected_source": "legacy_sidecar",
                "selection_policy": "prefer_b01",
            },
        )
        telemetry = telemetry.with_issues((*telemetry.issues, selection_issue))
    elif telemetry_source == "flash" and sidecar_failure is not None:
        telemetry = telemetry.with_issues(
            (*telemetry.issues, *sidecar_failure.issues)
        )
    elif telemetry_source is None and sidecar_failure is not None:
        telemetry = sidecar_failure
        telemetry_source = "sidecar"
        if b01_failure is not None:
            fallback_issue = issue_collector.record(
                code="telemetry_input.source_fallback",
                severity="warning",
                stage="telemetry_input",
                message=(
                    "the recorded legacy telemetry sidecar was selected after "
                    "the b01 source was unusable, but the sidecar was also "
                    "unreachable"
                ),
                action="kept",
                details={
                    "selected_source": "legacy_sidecar",
                    "unusable_source": "b01",
                    "selection_policy": "prefer_usable_b01_else_sidecar",
                },
            )
            telemetry = telemetry.with_issues(
                (*b01_failure.issues, *telemetry.issues, fallback_issue)
            )
    elif telemetry_source is None and b01_failure is not None:
        telemetry = b01_failure
        telemetry_source = "flash"

    telemetry_decoder_status = (
        telemetry.decoder_status.value
        if telemetry is not None
        else "not_needed"
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
        has_legacy_sidecar="legacy_sidecar" in telemetry_input_sources,
        telemetry_input_sources=tuple(telemetry_input_sources),
        telemetry_decoder_status=telemetry_decoder_status,
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
        result_sink=result_sink,
    )
    if telemetry_source is not None:
        result.telemetry_source = telemetry_source
    if flash_used is not None:
        result.flash_source_path = str(flash_used)
        result.flash_source_fingerprint = (
            {}
            if legacy_flash_fingerprint
            else (in_session_manifest or {}).get(
                "flash_source_fingerprint", {}
            )
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
    flash_root: Optional[Path | str] = None,
    rederive_telemetry: bool = True,
    allow_changed_flash_source: bool = False,
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
) -> SessionResult:
    """Process one extracted session and atomically finalize its manifest."""
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
    partial_results: List[SessionResult] = []
    try:
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
            flash_root=flash_root,
            rederive_telemetry=rederive_telemetry,
            allow_changed_flash_source=allow_changed_flash_source,
            issue_collector=collector,
            decoder_strict=decoder_strict,
            diagnostic_override=diagnostic_override,
            schema_variant=schema_variant,
            issue_marker=marker,
            result_sink=partial_results,
        )
    except Exception as exc:
        failure_result = partial_results[-1] if partial_results else None
        failure_issue = None
        try:
            failure_issue = collector.record(
                code="pipeline.session_failed",
                severity=IssueSeverity.ERROR,
                stage="finalization",
                message="session processing aborted before finalization",
                action=IssueAction.REJECTED,
                details={"error_type": type(exc).__name__},
            )
        except IngestIssueError as strict_exc:
            failure_issue = strict_exc.issue
        if manifest_dir is not None:
            failure_issues = collector.since(marker)
            records, counts, codes = _issue_manifest_fields(failure_issues)
            try:
                failure_name = _validate_session_output_name(
                    name or f"session_{ordinal:03d}"
                )
            except ValueError:
                failure_name = f"session_{ordinal:03d}"
            failed = failure_result
            if failed is None:
                family_issue_ids = (
                    {
                        family: (failure_issue.issue_id,)
                        for family, _family_type in FAMILY_TYPES
                    }
                    if failure_issue is not None
                    else {}
                )
                failure_clock_reference = None
                if landing_time_file is not None:
                    try:
                        failure_clock_reference = _load_landing_reference(
                            Path(landing_time_file).resolve()
                        )
                    except Exception:  # noqa: BLE001
                        pass
                try:
                    failed = _process_one_session(
                        session_dir=resolved_session,
                        name=failure_name,
                        ordinal=ordinal,
                        h5_dir=None,
                        fits_dir=None,
                        plots_dir=None,
                        manifest_dir=None,
                        issue_collector=collector,
                        clock_reference_set=failure_clock_reference,
                        source_path=resolved_session,
                        source_kind="session",
                        products=Products(
                            family_issue_ids=family_issue_ids
                        ),
                        context_issues=failure_issues,
                        overwrite=overwrite,
                    )
                    failed_families = {
                        family for family, _family_type in FAMILY_TYPES
                    }
                    for family_status in failed.family_statuses:
                        if family_status["family"] in failed_families:
                            family_status["reason"] = (
                                "stage_failed_before_decode"
                            )
                except Exception:  # noqa: BLE001
                    failed = SessionResult(
                        session_ordinal=ordinal,
                        session_name=failure_name,
                        source_path=str(resolved_session),
                        source_kind="session",
                    )
            failed.status = "failed"
            failed.status_issue_codes = codes
            failed.issue_counts = counts
            failed.issues = records
            failed.failure = {
                "stage": "session_processing",
                "error_type": type(exc).__name__,
                "message": "session processing aborted before finalization",
            }
            failed.processed_at_utc = _now_utc_iso()
            failed.pipeline_version = _pipeline_version_string()
            failed.overwrite = overwrite
            failure_path = (
                Path(failed.manifest_path)
                if failed.manifest_path is not None
                else Path(manifest_dir) / f"{failed.session_name}.json"
            )
            failed.manifest_path = str(failure_path.resolve())
            failure_result = failed
            try:
                _write_bytes_atomic(
                    _canonical_manifest_bytes(
                        _session_manifest_body(
                            failed,
                            destination=failure_path,
                        )
                    ),
                    failure_path,
                    overwrite=overwrite,
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to write session failure manifest")
        if failure_result is not None:
            setattr(exc, "ingest_result", failure_result)
        raise


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
    overwrite: bool,
) -> None:
    if len(set(names)) != len(names):
        raise ValueError("session_name returned duplicate output names")
    candidates: List[tuple[str, Path, str]] = [
        ("run_manifest", Path(path), "file")
        for path in run_manifest_destinations
    ]
    for name in names:
        session_dir = sessions_root / name
        candidates.extend((
            ("session_directory", session_dir, "directory"),
            (
                "in_session_manifest",
                session_dir / IN_SESSION_MANIFEST_NAME,
                "file",
            ),
        ))
        if h5_dir is not None:
            candidates.append(("hdf5", h5_dir / f"{name}.h5", "file"))
        if fits_dir is not None:
            candidates.append(("fits", fits_dir / f"{name}.fits", "file"))
        if manifest_dir is not None:
            candidates.append((
                "session_manifest",
                manifest_dir / f"{name}.json",
                "file",
            ))
        if plots_dir is not None and h5_dir is not None:
            candidates.append((
                "plot_directory",
                plots_dir / name,
                "directory",
            ))

    claimed: Dict[Path, tuple[str, Path]] = {}
    for label, destination, kind in candidates:
        resolved = destination.resolve()
        previous = claimed.get(resolved)
        if previous is not None:
            previous_label, previous_path = previous
            raise ValueError(
                "FLASH output destinations alias each other: "
                f"{previous_label}={previous_path} and {label}={destination}"
            )
        claimed[resolved] = (label, destination)
        if not _input_path_present(destination):
            continue
        if not overwrite:
            raise FileExistsError(destination)
        if kind == "directory" and (
            destination.is_symlink() or not destination.is_dir()
        ):
            raise ValueError(
                f"session destination is not a directory: {destination}"
            )
        if kind == "file" and (
            destination.is_symlink() or not destination.is_file()
        ):
            raise ValueError(
                f"output destination is not a regular file: {destination}"
            )

    for index, (label, destination, _kind) in enumerate(candidates):
        resolved = destination.resolve()
        for other_label, other_destination, _other_kind in candidates[index + 1:]:
            other_resolved = other_destination.resolve()
            if resolved not in other_resolved.parents and (
                other_resolved not in resolved.parents
            ):
                continue
            intentional_session_manifest = (
                label == "session_directory"
                and other_label == "in_session_manifest"
                and other_resolved.parent == resolved
            ) or (
                other_label == "session_directory"
                and label == "in_session_manifest"
                and resolved.parent == other_resolved
            )
            if intentional_session_manifest:
                continue
            raise ValueError(
                "FLASH output destinations have an ancestor conflict: "
                f"{label}={destination} and "
                f"{other_label}={other_destination}"
            )

    for protected_input in protected_inputs:
        protected = protected_input.resolve()
        for label, destination, _kind in candidates:
            resolved = destination.resolve()
            if (
                resolved == protected
                or resolved in protected.parents
                or protected in resolved.parents
            ):
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
    lower_elapsed_seconds: float | None,
    upper_elapsed_seconds: float | None,
    first_session_elapsed_seconds: float | None,
    telemetry_all: telemetry_mod.TelemetryDecodeResult,
) -> None:
    result.flash_source_path = str(flash_dir)
    result.flash_source_fingerprint = flash_fingerprint
    result.flash_result_id = flash_result_id
    result.flash_input_identity_sha256 = input_identity_sha256
    result.flash_input_identity_unavailable_reason = (
        input_identity_unavailable_reason
    )
    result.start_raw_seconds = session.start_raw_seconds
    result.telemetry_window_lower_elapsed_seconds = lower_elapsed_seconds
    result.telemetry_window_upper_elapsed_seconds = upper_elapsed_seconds
    session_telemetry = session.telemetry
    if (
        session_telemetry is not None
        and session_telemetry.decoder_status
        is telemetry_mod.TelemetryDecoderStatus.AVAILABLE
        and session_telemetry.fpga is not None
    ):
        assigned_rows = session_telemetry.fpga.row_count
        unassigned_rows = (
            session_telemetry.unassigned_fpga.row_count
            if session_telemetry.unassigned_fpga is not None
            else 0
        )
        full_rows = (
            telemetry_all.fpga.row_count
            if telemetry_all.fpga is not None
            else 0
        )
        if unassigned_rows and assigned_rows == 0 and unassigned_rows == full_rows:
            result.telemetry_assignment_mode = "all_unassigned"
        elif unassigned_rows:
            result.telemetry_assignment_mode = "assigned_with_pre_session"
            result.telemetry_unassigned_upper_elapsed_seconds = (
                first_session_elapsed_seconds
            )
        else:
            result.telemetry_assignment_mode = "assigned"
        result.telemetry_assignment_issues = [
            issue.as_dict()
            for issue in session_telemetry.issues
            if issue.stage == "telemetry_assignment"
        ]
    if result.has_telemetry:
        result.telemetry_source = "flash"


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
        _write_bytes_atomic(
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
    telemetry: telemetry_mod.TelemetryDecodeResult | None,
    sessions: Sequence[Session],
    prepared: Sequence[tuple[Session, str, Path, Products]],
) -> tuple[IngestIssue, ...]:
    groups: List[Sequence[IngestIssue]] = [issue_collector.since(marker)]
    if telemetry is not None:
        groups.append(telemetry.issues)
    groups.extend(product.issues for *_, product in prepared)
    groups.extend(
        session.telemetry.issues
        for session in sessions
        if session.telemetry is not None
    )
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


def _write_flash_failure_manifests(
    result: FlashResult,
    destinations: Sequence[Path],
    *,
    payload: bytes,
    overwrite: bool,
) -> None:
    try:
        _write_flash_manifests(
            result,
            destinations,
            payload=payload,
            overwrite=overwrite,
        )
    except Exception:  # noqa: BLE001
        log.exception("failed to write linked FLASH failure manifests")


def _failed_flash_session_result(
    *,
    session: Session,
    name: str,
    session_dir: Path,
    installed: bool,
    products: Products | None,
    failure_issue: IngestIssue | None,
    run_issues: Sequence[IngestIssue],
    failure_stage: str,
    error_type: str,
    flash_dir: Path,
    flash_fingerprint: Dict[str, Dict[str, object]],
    flash_result_id: str,
    input_identity_sha256: str | None,
    input_identity_unavailable_reason: str | None,
    clock_reference_set: ClockReferenceSet,
    issue_collector: IssueCollector,
    telemetry_all: telemetry_mod.TelemetryDecodeResult,
    telemetry_input_sources: Sequence[str],
    telemetry_decoder_status: str,
    lower_elapsed_seconds: float | None,
    upper_elapsed_seconds: float | None,
    first_session_elapsed_seconds: float | None,
    manifest_dir: Path | None,
    overwrite: bool,
) -> SessionResult:
    decode_failed = products is None
    if products is None:
        family_issue_ids = (
            {
                family: (failure_issue.issue_id,)
                for family, _family_type in FAMILY_TYPES
            }
            if failure_issue is not None
            else {}
        )
        products = Products(family_issue_ids=family_issue_ids)
        if installed and (session_dir / PACKET_MAP_FILENAME).is_file():
            products.packet_map_status = "verified"
            products.packet_map_format_version = PACKET_MAP_FORMAT_VERSION
            products.raw_flash_provenance_unavailable_reason = None
    context_issues = list(_issues_for_session(
        run_issues,
        session=session,
        session_name=name,
    ))
    if failure_issue is not None and failure_issue not in context_issues:
        context_issues.append(failure_issue)
    result = _process_one_session(
        session_dir=session_dir,
        name=name,
        ordinal=session.ordinal,
        h5_dir=None,
        fits_dir=None,
        plots_dir=None,
        manifest_dir=None,
        issue_collector=issue_collector,
        clock_reference_set=clock_reference_set,
        telemetry=session.telemetry,
        has_legacy_sidecar=False,
        telemetry_input_sources=telemetry_input_sources,
        telemetry_decoder_status=telemetry_decoder_status,
        source_path=flash_dir,
        source_kind="flash",
        overwrite=overwrite,
        products=products,
        context_issues=context_issues,
        input_packet_count=len(session.packets),
        input_identity=input_identity_sha256,
        input_identity_kind=(
            "flash_source_sha256"
            if input_identity_sha256 is not None
            else None
        ),
        input_identity_unavailable_reason=(
            input_identity_unavailable_reason
        ),
        flash_result_id=flash_result_id,
    )
    if decode_failed:
        decoded_families = {family for family, _family_type in FAMILY_TYPES}
        for family_status in result.family_statuses:
            if family_status["family"] in decoded_families:
                family_status["reason"] = "stage_failed_before_decode"
    _set_flash_session_metadata(
        result,
        session=session,
        flash_dir=flash_dir,
        flash_fingerprint=flash_fingerprint,
        flash_result_id=flash_result_id,
        input_identity_sha256=input_identity_sha256,
        input_identity_unavailable_reason=(
            input_identity_unavailable_reason
        ),
        lower_elapsed_seconds=lower_elapsed_seconds,
        upper_elapsed_seconds=upper_elapsed_seconds,
        first_session_elapsed_seconds=first_session_elapsed_seconds,
        telemetry_all=telemetry_all,
    )
    if installed:
        packet_map_path = session_dir / PACKET_MAP_FILENAME
        if packet_map_path.is_file():
            result.output_artifacts["packet_map"] = _artifact_record(
                packet_map_path
            )
    else:
        result.session_dir = None
    if manifest_dir is not None:
        result.manifest_path = str(
            (manifest_dir / f"{name}.json").resolve()
        )
    result.status = "failed"
    result.failure = {
        "stage": failure_stage,
        "error_type": error_type,
        "message": "session was not finalized before the FLASH run aborted",
    }
    return result


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
    extracted: List[tuple[Session, str, Path]] = []
    sessions: List[Session] = []
    names: List[str] = []
    telemetry_all: telemetry_mod.TelemetryDecodeResult | None = None
    expected_binding: tuple[object, ...] | None = None
    flash_result_id, input_identity_sha256, identity_unavailable_reason = (
        _flash_identity(
            {},
            clock_reference_set,
        )
    )
    flash_fingerprint: Dict[str, Dict[str, object]] = {}
    parse_issues: tuple[IngestIssue, ...] = ()
    telemetry_input_sources: tuple[str, ...] = ()
    telemetry_decoder_status = "not_needed"
    win_lower: Dict[int, Optional[float]] = {}
    win_upper: Dict[int, Optional[float]] = {}
    first_session_elapsed: float | None = None
    outputs_preflighted = False
    run_manifest_destinations = _flash_manifest_destinations(
        sessions_root,
        manifest_dir,
    )
    if not overwrite:
        for destination in run_manifest_destinations:
            if _input_path_present(destination):
                raise FileExistsError(destination)

    failure_stage = "input_parse"
    try:
        sessions, telemetry_all, _unassigned_telemetry = _parse_flash_loaded(
            flash_dir,
            clock_reference_set=clock_reference_set,
            issue_collector=issue_collector,
            capture=capture,
        )
        flash_fingerprint = _validate_flash_fingerprint(
            capture.source_fingerprint
        )
        parse_issues = issue_collector.since(run_marker)
        telemetry_input_sources = (
            ("b01",)
            if telemetry_all.input_source == "b01"
            else ()
        )
        telemetry_decoder_status = telemetry_all.decoder_status.value

        sorted_sessions = sorted(sessions, key=lambda item: item.ordinal)
        spectrometer_reference = clock_reference_set.require_reference(
            ClockSource.SPECTROMETER
        )
        for index, session in enumerate(sorted_sessions):
            next_session = (
                sorted_sessions[index + 1]
                if index + 1 < len(sorted_sessions)
                else None
            )
            win_lower[session.ordinal] = (
                None
                if session.start_raw_seconds is None
                else session.start_raw_seconds
                - spectrometer_reference.clock_reference_raw_seconds
            )
            win_upper[session.ordinal] = (
                None
                if next_session is None
                or next_session.start_raw_seconds is None
                else next_session.start_raw_seconds
                - spectrometer_reference.clock_reference_raw_seconds
            )
        first_session_elapsed = (
            win_lower.get(sorted_sessions[0].ordinal)
            if sorted_sessions
            else None
        )
        for session in sessions:
            names.append(_validate_session_output_name(session_name(
                session.ordinal,
                session.start_raw_seconds,
                clock_reference_set,
            )))
        failure_stage = "output_preflight"
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
            overwrite=overwrite,
        )
        outputs_preflighted = True

        failure_stage = "decoder_preflight"
        for session, name in zip(sessions, names):
            session_dir = sessions_root / name
            if overwrite:
                write_uncrater_session(
                    session,
                    session_dir,
                    overwrite=True,
                )
            else:
                write_uncrater_session(session, session_dir)
            extracted.append((session, name, session_dir))
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
        failure_stage = "product_write"
        for session, name, session_dir, products in prepared:
            partial_results: List[SessionResult] = []
            try:
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
                    has_legacy_sidecar=False,
                    telemetry_input_sources=telemetry_input_sources,
                    telemetry_decoder_status=telemetry_decoder_status,
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
                    result_sink=partial_results,
                )
            except Exception as exc:
                if partial_results:
                    partial = partial_results[-1]
                    _set_flash_session_metadata(
                        partial,
                        session=session,
                        flash_dir=flash_dir,
                        flash_fingerprint=flash_fingerprint,
                        flash_result_id=flash_result_id,
                        input_identity_sha256=input_identity_sha256,
                        input_identity_unavailable_reason=(
                            identity_unavailable_reason
                        ),
                        lower_elapsed_seconds=win_lower.get(session.ordinal),
                        upper_elapsed_seconds=win_upper.get(session.ordinal),
                        first_session_elapsed_seconds=first_session_elapsed,
                        telemetry_all=telemetry_all,
                    )
                    partial.status = "failed"
                    partial.failure = {
                        "stage": "product_write",
                        "error_type": type(exc).__name__,
                        "message": (
                            "session product writing aborted before finalization"
                        ),
                    }
                    results.append(partial)
                raise
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
                lower_elapsed_seconds=win_lower.get(session.ordinal),
                upper_elapsed_seconds=win_upper.get(session.ordinal),
                first_session_elapsed_seconds=first_session_elapsed,
                telemetry_all=telemetry_all,
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
            telemetry=telemetry_all,
            sessions=sessions,
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
                telemetry=telemetry_all,
            ),
            decoder_provenance=_flash_decoder_provenance(prepared),
            telemetry_provenance=_telemetry_provenance_manifest(
                telemetry_all,
                InterpolationPolicy(),
            ),
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

        failure_stage = "manifest_finalization"
        _write_flash_manifests(
            flash_result,
            run_manifest_destinations,
            payload=manifest_payload,
            overwrite=overwrite,
        )
        return flash_result
    except Exception as exc:
        failure_issue = None
        try:
            failure_issue = issue_collector.record(
                code="pipeline.flash_failed",
                severity=IssueSeverity.ERROR,
                stage="finalization",
                message="FLASH processing aborted before finalization",
                action=IssueAction.REJECTED,
                details={
                    "error_type": type(exc).__name__,
                    "failure_stage": failure_stage,
                },
            )
        except IngestIssueError as strict_exc:
            failure_issue = strict_exc.issue
        if failure_issue is not None:
            for result in results:
                if result.status != "failed":
                    continue
                result.status_issue_codes = sorted({
                    *result.status_issue_codes,
                    failure_issue.code,
                })
                result.issue_counts[failure_issue.code] = (
                    result.issue_counts.get(failure_issue.code, 0) + 1
                )
                result.issues.append(failure_issue.as_dict())
        failure_issues = _flash_run_issues(
            issue_collector,
            run_marker,
            telemetry=telemetry_all,
            sessions=sessions,
            prepared=prepared,
        )
        issue_records, issue_counts, status_issue_codes = (
            _issue_manifest_fields(failure_issues)
        )
        source_fingerprint = _validate_flash_fingerprint(
            capture.source_fingerprint
        )
        (
            flash_result_id,
            input_identity_sha256,
            identity_unavailable_reason,
        ) = _flash_identity(
            source_fingerprint,
            clock_reference_set,
            expected_binding,
            source_identity_complete=capture.source_identity_complete,
            source_identity_unavailable_reasons=(
                capture.source_identity_unavailable_reasons
            ),
        )
        represented = {
            (result.session_ordinal, result.session_name)
            for result in results
        }
        prepared_products = {
            (session.ordinal, name): products
            for session, name, _session_dir, products in prepared
        }
        extracted_sessions = {
            (session.ordinal, name)
            for session, name, _session_dir in extracted
        }
        failure_telemetry = (
            telemetry_all
            if telemetry_all is not None
            else telemetry_mod.TelemetryDecodeResult.absent()
        )
        failure_sources = (
            telemetry_input_sources if telemetry_all is not None else ()
        )
        failure_decoder_status = (
            telemetry_decoder_status
            if failure_sources
            else "not_needed"
        )
        for index, session in enumerate(sessions):
            name = (
                names[index]
                if index < len(names)
                else f"session_{session.ordinal:03d}"
            )
            if (session.ordinal, name) in represented:
                continue
            session_dir = sessions_root / name
            products = prepared_products.get((session.ordinal, name))
            session_result = _failed_flash_session_result(
                session=session,
                name=name,
                session_dir=session_dir,
                installed=(session.ordinal, name) in extracted_sessions,
                products=products,
                failure_issue=failure_issue,
                run_issues=failure_issues,
                failure_stage=failure_stage,
                error_type=type(exc).__name__,
                flash_dir=flash_dir,
                flash_fingerprint=source_fingerprint,
                flash_result_id=flash_result_id,
                input_identity_sha256=input_identity_sha256,
                input_identity_unavailable_reason=(
                    identity_unavailable_reason
                ),
                clock_reference_set=clock_reference_set,
                issue_collector=issue_collector,
                telemetry_all=failure_telemetry,
                telemetry_input_sources=failure_sources,
                telemetry_decoder_status=failure_decoder_status,
                lower_elapsed_seconds=win_lower.get(session.ordinal),
                upper_elapsed_seconds=win_upper.get(session.ordinal),
                first_session_elapsed_seconds=first_session_elapsed,
                manifest_dir=manifest_dir,
                overwrite=overwrite,
            )
            results.append(session_result)
            represented.add((session.ordinal, name))
        failed_result = FlashResult(
            flash_result_id=flash_result_id,
            input_identity_sha256=input_identity_sha256,
            input_identity_unavailable_reason=identity_unavailable_reason,
            source_path=str(flash_dir),
            source_fingerprint=source_fingerprint,
            session_results=results,
            status="failed",
            status_issue_codes=status_issue_codes,
            issue_counts=issue_counts,
            issues=issue_records,
            stage_counts=_flash_stage_counts(
                capture,
                failure_issues,
                products=[product for *_, product in prepared],
                results=results,
                telemetry=telemetry_all,
            ),
            decoder_provenance=_flash_decoder_provenance(prepared),
            telemetry_provenance=_telemetry_provenance_manifest(
                telemetry_all,
                InterpolationPolicy(),
            ),
            clock_reference=clock_reference_set.as_record(),
            pipeline_version=_pipeline_version_string(),
            processed_at_utc=_now_utc_iso(),
            manifest_paths=[str(path) for path in run_manifest_destinations],
            failure={
                "stage": failure_stage,
                "error_type": type(exc).__name__,
                "message": "FLASH processing aborted before finalization",
            },
            overwrite=overwrite,
            source_identity_unavailable_reasons=list(
                capture.source_identity_unavailable_reasons
            ),
        )
        failure_payload = _canonical_manifest_bytes(
            _flash_manifest_body(
                failed_result,
                path_parent=sessions_root,
            )
        )
        failed_result.manifest_sha256 = hashlib.sha256(
            failure_payload
        ).hexdigest()
        for result in results:
            result.flash_manifest_sha256 = failed_result.manifest_sha256
        setattr(exc, "ingest_result", failed_result)
        if outputs_preflighted:
            _write_flash_failure_manifests(
                failed_result,
                run_manifest_destinations,
                payload=failure_payload,
                overwrite=overwrite,
            )
        raise
