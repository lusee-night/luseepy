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
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import telemetry as telemetry_mod
from .ccsds import parse_bank_file
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
    SCIENCE_BANKS,
    SESSION_NAME_NO_TIME_FMT,
    SESSION_TIMESTAMP_FMT,
    TELEMETRY_BANK,
)
from .decode import Products, read_uncrater_session
from .issues import IssueCollector
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
LEGACY_TELEMETRY_SIDECAR_NAME = "DCB_telemetry.json"


def _fingerprint_flash(flash_dir: Path) -> Dict[str, Dict[str, float]]:
    """Per-bank size+mtime fingerprint of a FLASH_TLMFS directory.

    Returned dict is keyed by ``"<bank>/FFFFFFFE"`` and carries
    ``{"size": int, "mtime": float}``. Missing banks are omitted.
    """
    out: Dict[str, Dict[str, float]] = {}
    for bank in (TELEMETRY_BANK,) + SCIENCE_BANKS:
        path = _bank_path(flash_dir, bank)
        if not path.is_file():
            continue
        st = path.stat()
        out[f"{bank}/{BANK_FILENAME}"] = {
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    return out


def _fingerprint_matches(
    recorded: Dict[str, Dict[str, float]],
    current: Dict[str, Dict[str, float]],
) -> List[str]:
    """Return the list of bank names whose fingerprints disagree."""
    diffs: List[str] = []
    for k, v in recorded.items():
        if k not in current:
            diffs.append(k)
            continue
        if int(current[k]["size"]) != int(v["size"]) or abs(
            float(current[k]["mtime"]) - float(v["mtime"])
        ) > 1e-3:
            diffs.append(k)
    return diffs


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

    h5_path: Optional[str] = None
    fits_path: Optional[str] = None
    session_dir: Optional[str] = None
    plot_paths: List[str] = field(default_factory=list)
    manifest_path: Optional[str] = None

    # Flash backreference: lets a later process_session run re-derive
    # /DCB_telemetry/ from the original raw bank without writing a binary
    # sidecar. None / empty for sessions produced by process_session.
    flash_source_path: Optional[str] = None
    flash_source_fingerprint: Dict[str, Dict[str, float]] = field(default_factory=dict)
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
    telemetry_input_sources: List[str] = field(default_factory=list)
    telemetry_decoder_status: str = "not_needed"
    telemetry_coverage: str = "absent"
    n_telemetry_rows: int = 0
    n_unassigned_telemetry_rows: int = 0


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
) -> Tuple[
    List[Session],
    telemetry_mod.TelemetryDecodeResult,
    telemetry_mod.TelemetryDecodeResult | None,
]:
    """Parse a flash after the public entry point validates mission time."""
    flash_dir = Path(flash_dir)
    if issue_collector is None:
        issue_collector = IssueCollector()
    science_packets: List[LogicalPacket] = []
    telem_packets: List[LogicalPacket] = []

    for bank in SCIENCE_BANKS:
        path = _bank_path(flash_dir, bank)
        if not path.is_file():
            log.info("skipping missing science bank %s", path)
            continue
        log.info("reading science bank %s", path)
        for lp in reassemble_logical_packets(
            parse_bank_file(
                path,
                bank=bank,
                issue_collector=issue_collector,
            ),
            byteswap_pairs=True,
            bank=bank,
        ):
            if not is_dropped_appid(lp.appid):
                science_packets.append(lp)

    tpath = _bank_path(flash_dir, TELEMETRY_BANK)
    telemetry_present = _input_path_present(tpath)
    telemetry_issue_marker = issue_collector.mark()
    telemetry_read_error = None
    if telemetry_present and tpath.is_file():
        log.info("reading telemetry bank %s", tpath)
        try:
            for lp in reassemble_logical_packets(
                parse_bank_file(
                    tpath,
                    bank=TELEMETRY_BANK,
                    issue_collector=issue_collector,
                ),
                byteswap_pairs=False,
                bank=TELEMETRY_BANK,
            ):
                telem_packets.append(lp)
        except OSError as exc:
            telemetry_read_error = type(exc).__name__
    elif telemetry_present:
        telemetry_read_error = "not_regular_file"
    else:
        log.info("no telemetry bank at %s", tpath)

    sw_version = detect_sw_version(science_packets)
    science_packets = assign_identities(science_packets, sw_version=sw_version)

    sessions = split_sessions(science_packets)

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

def write_manifest(result: SessionResult, dest_path: Path | str) -> Path:
    """Serialize ``result`` to ``dest_path`` as ASCII JSON."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = asdict(result)
    body["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    with dest.open("w", encoding="ascii") as fh:
        json.dump(body, fh, indent=2, sort_keys=True)
    return dest


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
# Per-session warning capture
# ---------------------------------------------------------------------------

class _WarningCapture:
    """Catch warnings.warn calls into a list while still emitting them."""

    def __init__(self) -> None:
        self.records: List[str] = []
        self._token: Optional[Any] = None
        self._old_showwarning = None

    def __enter__(self) -> "_WarningCapture":
        self._old_showwarning = warnings.showwarning

        def show(message, category, filename, lineno, file=None, line=None):
            text = warnings.formatwarning(message, category, filename, lineno, line)
            self.records.append(text.strip())
            try:
                self._old_showwarning(message, category, filename, lineno, file, line)
            except Exception:    # noqa: BLE001
                pass

        warnings.showwarning = show
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        warnings.showwarning = self._old_showwarning


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
) -> SessionResult:
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
    name = name or default_session_name(
        ordinal,
        products.start_raw_seconds,
        clock_reference_set,
    )
    resolved_source = (source_path or session_dir).resolve()

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
        n_packets=(
            len(products.spectra) + len(products.tr_spectra)
            + len(products.zoom_spectra) + len(products.grimm_spectra)
            + len(products.waveforms) + len(products.housekeeping)
            + len(products.cal_data)
        ),
        n_spectra=len(products.spectra),
        n_tr_spectra=len(products.tr_spectra),
        n_zoom_spectra=len(products.zoom_spectra),
        n_grimm_spectra=len(products.grimm_spectra),
        n_waveforms=len(products.waveforms),
        n_housekeeping=len(products.housekeeping),
        has_telemetry=has_telemetry,
        has_legacy_sidecar=has_legacy_sidecar,
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
    )

    h5_path = h5_dir / f"{name}.h5" if h5_dir is not None else None
    fits_path = fits_dir / f"{name}.fits" if fits_dir is not None else None
    if not overwrite:
        for destination in (h5_path, fits_path):
            if destination is not None and destination.exists():
                raise FileExistsError(destination)

    request = None
    if h5_path is not None or fits_path is not None:
        if clock_reference_set is None:
            raise ValueError(
                "layout-v4 output requires a verified landing-time reference"
            )
        request_telemetry = telemetry or telemetry_mod.TelemetryDecodeResult.absent()
        issue_by_id = {
            issue.issue_id: issue
            for issue in (*products.issues, *request_telemetry.issues)
        }
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
                input_identity=None,
                input_identity_kind=None,
                input_identity_unavailable_reason=(
                    "portable_input_identity_not_available"
                ),
                source_kind=source_kind,
                source_path=str(resolved_source),
                pipeline_version=_pipeline_version_string(),
            ),
            issues=request_issues,
            family_statuses=family_statuses_for_products(
                products,
                family_issue_ids=family_issue_ids,
                telemetry=request_telemetry,
            ),
            telemetry=request_telemetry,
            interpolation_policy=InterpolationPolicy(),
            overwrite=overwrite,
        )

    if h5_path is not None:
        from .hdf5_writer import write_hdf5

        write_hdf5(request, h5_path)
        result.h5_path = str(h5_path.resolve())

    if fits_path is not None:
        from .fits_writer import write_fits

        write_fits(request, fits_path)
        result.fits_path = str(fits_path.resolve())

    if plots_dir is not None and h5_path is not None:
        from . import viz as viz_mod

        plot_dest = plots_dir / name
        plot_dest.mkdir(parents=True, exist_ok=True)
        plot_paths = viz_mod.plot_session(h5_path, plot_dest, plots=plot_names)
        result.plot_paths = [str(p.resolve()) for p in plot_paths]

    if manifest_dir is not None:
        mpath = manifest_dir / f"{name}.json"
        result.manifest_path = str(mpath.resolve())
        write_manifest(result, mpath)

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
            )
        if not isinstance(value, dict):
            raise ValueError("session manifest must be a JSON object")
        return value
    except Exception as exc:    # noqa: BLE001
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
        telem_packets = list(reassemble_logical_packets(
            parse_bank_file(
                tpath,
                bank=TELEMETRY_BANK,
                issue_collector=issue_collector,
            ),
            byteswap_pairs=False,
            bank=TELEMETRY_BANK,
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
    issue_collector: IssueCollector | None = None,
    decoder_strict: bool = False,
    diagnostic_override: bool = False,
    schema_variant: str | None = None,
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

    telemetry: telemetry_mod.TelemetryDecodeResult | None = None
    telemetry_source: Optional[str] = None

    in_session_manifest = _read_in_session_manifest(
        session_dir,
        strict=output_requested or landing_time_file is not None,
    )
    telemetry_input_sources = _recorded_telemetry_sources(in_session_manifest)
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
    candidate = (Path(flash_root) if flash_root
                 else (Path(manifest_flash) if manifest_flash else None))
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
            if recorded_fp:
                current_fp = _fingerprint_flash(candidate)
                diffs = _fingerprint_matches(recorded_fp, current_fp)
                if diffs:
                    warnings.warn(
                        f"flash bank fingerprint(s) changed since session "
                        f"extraction: {diffs}; proceeding with time-window "
                        f"slice (records outside the session window are "
                        f"ignored)",
                        RuntimeWarning,
                        stacklevel=2,
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

    with _WarningCapture() as cap:
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
        )
    result.warnings_summary = cap.records
    result.n_warnings = len(cap.records)
    if telemetry_source is not None:
        result.telemetry_source = telemetry_source
    if flash_used is not None:
        result.flash_source_path = str(flash_used)
    if manifest_dir is not None and result.manifest_path:
        write_manifest(result, result.manifest_path)
    return result


# ---------------------------------------------------------------------------
# Public: process_flash (raw flash mode)
# ---------------------------------------------------------------------------

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
) -> List[SessionResult]:
    """Run the single-pass flash-to-session and product pipeline."""
    clock_reference_set = _load_landing_reference(landing_time_file)
    flash_dir = Path(flash_dir).resolve()
    telemetry_input_sources = (
        ("b01",)
        if _input_path_present(_bank_path(flash_dir, TELEMETRY_BANK))
        else ()
    )
    sessions_root = Path(sessions_root)
    h5_dir = Path(h5_dir) if h5_dir else None
    fits_dir = Path(fits_dir) if fits_dir else None
    plots_dir = Path(plots_dir) if plots_dir else None
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    if session_name is None:
        session_name = default_session_name
    if issue_collector is None:
        issue_collector = IssueCollector()

    results: List[SessionResult] = []
    sessions, telemetry_all, _unassigned_telemetry = _parse_flash_loaded(
        flash_dir,
        clock_reference_set=clock_reference_set,
        issue_collector=issue_collector,
    )
    telemetry_decoder_status = telemetry_all.decoder_status.value

    # Fingerprint the source flash dir once for all sessions in this run.
    flash_fingerprint = _fingerprint_flash(flash_dir)
    flash_path_str = str(flash_dir)

    # Pre-compute shared elapsed session windows. The DCB source later
    # subtracts its own anchor before comparison with these boundaries.
    sorted_sessions = sorted(
        sessions,
        key=lambda s: s.ordinal,
    )
    win_lower: Dict[int, Optional[float]] = {}
    win_upper: Dict[int, Optional[float]] = {}
    spectrometer_reference = clock_reference_set.require_reference(
        ClockSource.SPECTROMETER
    )
    for i, s in enumerate(sorted_sessions):
        nxt = sorted_sessions[i + 1] if i + 1 < len(sorted_sessions) else None
        win_lower[s.ordinal] = (
            None
            if s.start_raw_seconds is None
            else s.start_raw_seconds
            - spectrometer_reference.clock_reference_raw_seconds
        )
        win_upper[s.ordinal] = (
            None
            if nxt is None or nxt.start_raw_seconds is None
            else nxt.start_raw_seconds
            - spectrometer_reference.clock_reference_raw_seconds
        )
    first_session_elapsed = (
        win_lower.get(sorted_sessions[0].ordinal)
        if sorted_sessions
        else None
    )

    prepared: list[tuple[Session, str, Path, Products, list[str]]] = []
    expected_binding: tuple[object, ...] | None = None
    for session in sessions:
        name = session_name(
            session.ordinal,
            session.start_raw_seconds,
            clock_reference_set,
        )
        session_dir = sessions_root / name
        write_uncrater_session(session, session_dir)

        with _WarningCapture() as cap:
            products = read_uncrater_session(
                session_dir,
                issue_collector=issue_collector,
            )
        binding = _binding_identity(products)
        if expected_binding is None:
            expected_binding = binding
        elif binding != expected_binding:
            raise RuntimeError(
                "derived FLASH sessions selected different decoder bindings; "
                "refusing all product writes. This is a conservative guard, "
                "not a forced input-wide binding"
            )
        prepared.append((session, name, session_dir, products, cap.records))

    for session, name, session_dir, products, decode_warnings in prepared:
        with _WarningCapture() as cap:
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
            )
        result.warnings_summary = [*decode_warnings, *cap.records]
        result.n_warnings = len(result.warnings_summary)

        # Backreference to the source flash dir + the time window for
        # this session, so a later process_session run can re-derive the
        # telemetry without writing a binary sidecar.
        result.flash_source_path = flash_path_str
        result.flash_source_fingerprint = flash_fingerprint
        result.start_raw_seconds = session.start_raw_seconds
        result.telemetry_window_lower_elapsed_seconds = win_lower.get(
            session.ordinal
        )
        result.telemetry_window_upper_elapsed_seconds = win_upper.get(
            session.ordinal
        )
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
            if unassigned_rows and (
                assigned_rows == 0 and unassigned_rows == full_rows
            ):
                result.telemetry_assignment_mode = "all_unassigned"
            elif unassigned_rows:
                result.telemetry_assignment_mode = "assigned_with_pre_session"
                result.telemetry_unassigned_upper_elapsed_seconds = (
                    first_session_elapsed
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

        # Write a copy of the manifest into the session dir so
        # process_session can find it without depending on manifest_dir.
        in_session_manifest = session_dir / IN_SESSION_MANIFEST_NAME
        write_manifest(result, in_session_manifest)
        if manifest_dir is not None and result.manifest_path:
            write_manifest(result, result.manifest_path)
        results.append(result)
    return results
