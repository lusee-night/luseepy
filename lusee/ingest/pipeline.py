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
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import telemetry as telemetry_mod
from .ccsds import parse_bank_file
from .collation import (
    APID_HELLO,
    LogicalPacket,
    assign_identities,
    detect_sw_version,
    is_dropped_appid,
    reassemble_logical_packets,
)
from .constants import (
    BANK_FILENAME,
    DEFAULT_LUN_HEIGHT_M,
    DEFAULT_LUN_LAT_DEG,
    DEFAULT_LUN_LONG_DEG,
    DEFAULT_MJD_EPOCH_OFFSET_DAYS,
    DEFAULT_RAW_TIME_SUBTRACT_SECONDS,
    DEFAULT_SESSION_NAME_FMT,
    SCIENCE_BANKS,
    SESSION_NAME_NO_TIME_FMT,
    SESSION_TIMESTAMP_FMT,
    TELEMETRY_BANK,
)
from .decode import Products, read_uncrater_session
from .fits_writer import write_fits
from .hdf5_writer import write_hdf5
from .session import (
    Session,
    assign_telemetry_to_sessions,
    split_sessions,
    write_uncrater_session,
)

try:
    from . import viz as _viz
except Exception:    # noqa: BLE001
    _viz = None    # matplotlib may not be available in some test envs

log = logging.getLogger(__name__)

# Schema version of session.json -- bump when the manifest layout changes.
MANIFEST_SCHEMA_VERSION = 2

# Filename of the in-session manifest written into each session directory.
IN_SESSION_MANIFEST_NAME = "session.json"


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
    # Mission-time window used to slice telemetry when re-deriving from
    # the flash bank. The window is
    # ``[telemetry_window_lower_raw_seconds, telemetry_window_upper_raw_seconds)``,
    # with None standing in for -inf / +inf at the ends. This mirrors the
    # bisect_right rule in assign_telemetry_to_sessions: every telemetry
    # record falls into exactly one session, the most recent preceding
    # session's window. The first session in a flash dump owns everything
    # earlier than the second session's start, hence its lower bound is
    # None (-inf) -- not its own start_raw_seconds. For every other
    # session the lower bound equals its own start_raw_seconds.
    start_raw_seconds: Optional[float] = None
    telemetry_window_lower_raw_seconds: Optional[float] = None
    telemetry_window_upper_raw_seconds: Optional[float] = None

    telemetry_source: Optional[str] = None
    """How telemetry reached the HDF5 for this session: 'flash' (single-pass
    or re-derived via flash backreference), 'sidecar' (legacy
    DCB_telemetry.json), or None."""

    processed_at_utc: str = ""
    pipeline_version: str = ""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def default_session_name(ordinal: int, start_raw_seconds: Optional[float]) -> str:
    """Default ``session_NNN_YYYYMMDD_HHMMSS`` (UTC) or ``session_NNN``."""
    if start_raw_seconds is None or start_raw_seconds <= 0:
        return SESSION_NAME_NO_TIME_FMT.format(ord=ordinal)
    ts = _dt.datetime.fromtimestamp(start_raw_seconds, tz=_dt.timezone.utc).strftime(
        SESSION_TIMESTAMP_FMT
    )
    return DEFAULT_SESSION_NAME_FMT.format(ord=ordinal, ts=ts)


SessionNamer = Callable[[int, Optional[float]], str]


# ---------------------------------------------------------------------------
# Flash directory parsing (Stages 1-4)
# ---------------------------------------------------------------------------

def _bank_path(flash_dir: Path, bank: str) -> Path:
    return flash_dir / bank / BANK_FILENAME


def parse_flash(
    flash_dir: Path | str,
) -> Tuple[List[Session], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Parse a FLASH_TLMFS directory through Stage 4.

    Returns ``(sessions, fpga_arrays, encoder_arrays)``. The two arrays
    dicts hold all b01 telemetry decoded by the private telemetry
    decoder, if one is loaded; otherwise both are empty. Per-session
    slices are also already stored on each Session via
    ``assign_telemetry_to_sessions``.
    """
    flash_dir = Path(flash_dir)
    science_packets: List[LogicalPacket] = []
    telem_packets: List[LogicalPacket] = []

    for bank in SCIENCE_BANKS:
        path = _bank_path(flash_dir, bank)
        if not path.is_file():
            log.info("skipping missing science bank %s", path)
            continue
        log.info("reading science bank %s", path)
        for lp in reassemble_logical_packets(
            parse_bank_file(path),
            byteswap_pairs=True,
            bank=bank,
        ):
            if not is_dropped_appid(lp.appid):
                science_packets.append(lp)

    tpath = _bank_path(flash_dir, TELEMETRY_BANK)
    if tpath.is_file():
        log.info("reading telemetry bank %s", tpath)
        for lp in reassemble_logical_packets(
            parse_bank_file(tpath),
            byteswap_pairs=False,
            bank=TELEMETRY_BANK,
        ):
            telem_packets.append(lp)
    else:
        log.info("no telemetry bank at %s", tpath)

    sw_version = detect_sw_version(science_packets)
    science_packets = assign_identities(science_packets, sw_version=sw_version)

    sessions = split_sessions(science_packets)

    fpga_arrays, encoder_arrays = telemetry_mod.parse_b01_packets(telem_packets)
    assign_telemetry_to_sessions(sessions, fpga_arrays, encoder_arrays)
    return sessions, fpga_arrays, encoder_arrays


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


def _start_utc_iso(raw_seconds: Optional[float]) -> Optional[str]:
    if raw_seconds is None or raw_seconds <= 0:
        return None
    return _dt.datetime.fromtimestamp(raw_seconds, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


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
    name: str,
    ordinal: int,
    h5_dir: Optional[Path],
    plots_dir: Optional[Path],
    manifest_dir: Optional[Path],
    fits_dir: Optional[Path] = None,
    fpga_arrays: Optional[Dict[str, np.ndarray]] = None,
    encoder_arrays: Optional[Dict[str, np.ndarray]] = None,
    has_legacy_sidecar: bool = False,
    source_path: Optional[Path] = None,
    source_kind: str = "session",
    interpolate_telemetry: bool = False,
    interpolation_mode: str = "normalized",
    plot_names: Optional[Sequence[str]] = None,
    constants_kwargs: Optional[Dict[str, object]] = None,
) -> SessionResult:
    products = read_uncrater_session(session_dir)
    constants_kwargs = dict(constants_kwargs or {})

    has_telemetry = bool(fpga_arrays) or bool(encoder_arrays)

    result = SessionResult(
        session_ordinal=ordinal,
        session_name=name,
        source_path=str((source_path or session_dir).resolve()),
        source_kind=source_kind,
        start_time_utc=_start_utc_iso(products.start_raw_seconds),
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
    )

    if h5_dir is not None:
        h5_dir.mkdir(parents=True, exist_ok=True)
        h5_path = h5_dir / f"{name}.h5"
        write_hdf5(
            products,
            h5_path,
            cdi_directory=session_dir,
            fpga_telemetry=fpga_arrays,
            encoder_telemetry=encoder_arrays,
            interpolate_telemetry=interpolate_telemetry,
            interpolation_mode=interpolation_mode,
            **constants_kwargs,
        )
        result.h5_path = str(h5_path.resolve())
    else:
        h5_path = None

    if fits_dir is not None:
        fits_dir.mkdir(parents=True, exist_ok=True)
        fits_path = fits_dir / f"{name}.fits"
        write_fits(
            products,
            fits_path,
            cdi_directory=session_dir,
            fpga_telemetry=fpga_arrays,
            encoder_telemetry=encoder_arrays,
            interpolate_telemetry=interpolate_telemetry,
            interpolation_mode=interpolation_mode,
            **constants_kwargs,
        )
        result.fits_path = str(fits_path.resolve())

    if plots_dir is not None and h5_path is not None and _viz is not None:
        plot_dest = plots_dir / name
        plot_dest.mkdir(parents=True, exist_ok=True)
        plot_paths = _viz.plot_session(h5_path, plot_dest, plots=plot_names)
        result.plot_paths = [str(p.resolve()) for p in plot_paths]

    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        mpath = manifest_dir / f"{name}.json"
        write_manifest(result, mpath)
        result.manifest_path = str(mpath.resolve())

    return result


# ---------------------------------------------------------------------------
# Public: process_session (existing-session mode)
# ---------------------------------------------------------------------------

def _read_in_session_manifest(session_dir: Path) -> Optional[Dict[str, Any]]:
    p = session_dir / IN_SESSION_MANIFEST_NAME
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="ascii") as fh:
            return json.load(fh)
    except Exception as exc:    # noqa: BLE001
        warnings.warn(
            f"failed to read in-session manifest {p}: {exc}; ignoring",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _rederive_telemetry_from_flash(
    flash_dir: Path,
    *,
    window_lower_raw_seconds: Optional[float],
    window_upper_raw_seconds: Optional[float],
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, np.ndarray]]]:
    """Re-parse b01 telemetry and slice it to one session's mission-time window.

    Window is ``[window_lower_raw_seconds, window_upper_raw_seconds)``,
    with None at either end meaning -inf / +inf. See
    ``SessionResult.telemetry_window_*`` for the semantics. Returns
    ``(None, None)`` when no decoder is available or the b01 bank is
    absent.
    """
    tpath = _bank_path(flash_dir, TELEMETRY_BANK)
    if not tpath.is_file():
        return None, None
    log.info("re-deriving telemetry from %s", tpath)
    telem_packets = list(reassemble_logical_packets(
        parse_bank_file(tpath),
        byteswap_pairs=False,
        bank=TELEMETRY_BANK,
    ))
    fpga_arrays, encoder_arrays = telemetry_mod.parse_b01_packets(telem_packets)
    fpga_arrays = telemetry_mod.slice_arrays_by_window(
        fpga_arrays,
        window_lower_raw_seconds=window_lower_raw_seconds,
        window_upper_raw_seconds=window_upper_raw_seconds,
    )
    encoder_arrays = telemetry_mod.slice_arrays_by_window(
        encoder_arrays,
        window_lower_raw_seconds=window_lower_raw_seconds,
        window_upper_raw_seconds=window_upper_raw_seconds,
    )
    return (fpga_arrays or None, encoder_arrays or None)


def process_session(
    session_dir: Path | str,
    *,
    h5_dir: Optional[Path | str] = None,
    fits_dir: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    manifest_dir: Optional[Path | str] = None,
    name: Optional[str] = None,
    ordinal: int = 0,
    interpolate_telemetry: bool = False,
    interpolation_mode: str = "normalized",
    plot_names: Optional[Sequence[str]] = None,
    constants_kwargs: Optional[Dict[str, object]] = None,
    flash_root: Optional[Path | str] = None,
    rederive_telemetry: bool = True,
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
    """
    session_dir = Path(session_dir).resolve()
    h5_dir = Path(h5_dir) if h5_dir else None
    fits_dir = Path(fits_dir) if fits_dir else None
    plots_dir = Path(plots_dir) if plots_dir else None
    manifest_dir = Path(manifest_dir) if manifest_dir else None

    fpga_arrays: Optional[Dict[str, np.ndarray]] = None
    encoder_arrays: Optional[Dict[str, np.ndarray]] = None
    telemetry_source: Optional[str] = None

    in_session_manifest = _read_in_session_manifest(session_dir)
    flash_used: Optional[Path] = None

    if rederive_telemetry:
        manifest_flash = (in_session_manifest or {}).get("flash_source_path")
        candidate = (Path(flash_root) if flash_root
                     else (Path(manifest_flash) if manifest_flash else None))
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
            lower = (in_session_manifest or {}).get(
                "telemetry_window_lower_raw_seconds"
            )
            upper = (in_session_manifest or {}).get(
                "telemetry_window_upper_raw_seconds"
            )
            fpga_arrays, encoder_arrays = _rederive_telemetry_from_flash(
                candidate,
                window_lower_raw_seconds=lower,
                window_upper_raw_seconds=upper,
            )
            if fpga_arrays or encoder_arrays:
                telemetry_source = "flash"
                flash_used = candidate
        elif manifest_flash:
            log.info("flash backreference %s is not reachable; falling back",
                     manifest_flash)

    sidecar = telemetry_mod.find_legacy_sidecar(session_dir)
    if telemetry_source is None and sidecar is not None:
        log.info("reading legacy DCB_telemetry sidecar at %s", sidecar)
        fpga_arrays = telemetry_mod.parse_legacy_sidecar(sidecar) or None
        telemetry_source = "sidecar"
    elif telemetry_source == "flash" and sidecar is not None:
        log.info("ignoring legacy sidecar %s in favor of flash backreference",
                 sidecar)

    name = name or default_session_name(ordinal, None)

    with _WarningCapture() as cap:
        result = _process_one_session(
            session_dir=session_dir,
            name=name,
            ordinal=ordinal,
            h5_dir=h5_dir,
            fits_dir=fits_dir,
            plots_dir=plots_dir,
            manifest_dir=manifest_dir,
            fpga_arrays=fpga_arrays,
            encoder_arrays=encoder_arrays,
            has_legacy_sidecar=sidecar is not None,
            source_path=session_dir,
            source_kind="session",
            interpolate_telemetry=interpolate_telemetry,
            interpolation_mode=interpolation_mode,
            plot_names=plot_names,
            constants_kwargs=constants_kwargs,
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
    sessions_root: Path | str,
    h5_dir: Optional[Path | str] = None,
    fits_dir: Optional[Path | str] = None,
    plots_dir: Optional[Path | str] = None,
    manifest_dir: Optional[Path | str] = None,
    session_name: Optional[SessionNamer] = None,
    interpolate_telemetry: bool = False,
    interpolation_mode: str = "normalized",
    plot_names: Optional[Sequence[str]] = None,
    constants_kwargs: Optional[Dict[str, object]] = None,
) -> List[SessionResult]:
    """Single-pass: raw flash directory -> sessions on disk + HDF5/FITS/plots/manifests."""
    flash_dir = Path(flash_dir).resolve()
    sessions_root = Path(sessions_root)
    sessions_root.mkdir(parents=True, exist_ok=True)
    h5_dir = Path(h5_dir) if h5_dir else None
    fits_dir = Path(fits_dir) if fits_dir else None
    plots_dir = Path(plots_dir) if plots_dir else None
    manifest_dir = Path(manifest_dir) if manifest_dir else None
    if session_name is None:
        session_name = default_session_name

    results: List[SessionResult] = []
    sessions, _fpga_all, _enc_all = parse_flash(flash_dir)

    # Fingerprint the source flash dir once for all sessions in this run.
    flash_fingerprint = _fingerprint_flash(flash_dir)
    flash_path_str = str(flash_dir)

    # Pre-compute the telemetry window per session, mirroring the
    # bisect_right rule in assign_telemetry_to_sessions:
    #   * first session: window = [-inf, second_session.start)
    #   * other session: window = [own.start, next_session.start)  (or +inf at end)
    sorted_sessions = sorted(
        sessions,
        key=lambda s: (s.start_raw_seconds if s.start_raw_seconds is not None else float("-inf")),
    )
    win_lower: Dict[int, Optional[float]] = {}
    win_upper: Dict[int, Optional[float]] = {}
    for i, s in enumerate(sorted_sessions):
        nxt = sorted_sessions[i + 1] if i + 1 < len(sorted_sessions) else None
        win_lower[s.ordinal] = None if i == 0 else s.start_raw_seconds
        win_upper[s.ordinal] = (nxt.start_raw_seconds if nxt is not None else None)

    for session in sessions:
        name = session_name(session.ordinal, session.start_raw_seconds)
        session_dir = sessions_root / name
        write_uncrater_session(session, session_dir)

        # Per-session arrays were already produced by parse_flash and
        # sliced onto the Session via assign_telemetry_to_sessions.
        fpga_arrays = session.fpga_telemetry or None
        encoder_arrays = session.encoder_telemetry or None

        with _WarningCapture() as cap:
            result = _process_one_session(
                session_dir=session_dir,
                name=name,
                ordinal=session.ordinal,
                h5_dir=h5_dir,
                fits_dir=fits_dir,
                plots_dir=plots_dir,
                manifest_dir=manifest_dir,
                fpga_arrays=fpga_arrays,
                encoder_arrays=encoder_arrays,
                has_legacy_sidecar=False,
                source_path=flash_dir,
                source_kind="flash",
                interpolate_telemetry=interpolate_telemetry,
                interpolation_mode=interpolation_mode,
                plot_names=plot_names,
                constants_kwargs=constants_kwargs,
            )
        result.warnings_summary = cap.records
        result.n_warnings = len(cap.records)

        # Backreference to the source flash dir + the time window for
        # this session, so a later process_session run can re-derive the
        # telemetry without writing a binary sidecar.
        result.flash_source_path = flash_path_str
        result.flash_source_fingerprint = flash_fingerprint
        result.start_raw_seconds = session.start_raw_seconds
        result.telemetry_window_lower_raw_seconds = win_lower.get(session.ordinal)
        result.telemetry_window_upper_raw_seconds = win_upper.get(session.ordinal)
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
