#!/usr/bin/env python3
"""Generate deterministic baseline/final reports for ``lusee.ingest``.

The tracked script contains no corpus paths or expected values. A private JSON
configuration supplies portable target IDs, source paths, the external landing
reference, and observed product-family counts. Science derivatives are written
only below a temporary directory and removed after their semantic summaries are
captured.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import textwrap
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np


REPORT_SCHEMA_VERSION = 1
CONFIG_FORMAT_VERSION = 1
FAMILIES = (
    "normal",
    "tr",
    "zoom",
    "waveform",
    "grimm",
    "telemetry",
    "housekeeping",
    "calibrator",
)
COVERAGE_STAGES = (
    "observed_input",
    "decoded",
    "hdf5",
    "fits",
    "reader",
    "plotted",
)
COVERAGE_STATES = {
    "present",
    "present_empty",
    "absent_in_input",
    "invalid_or_dropped",
    "unsupported",
    "stage_failed",
}
TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class ObservedFamilyEvidence:
    """Independent inventory meaning attached to one observed family count."""

    unit: str
    basis: str
    input_state: Literal["absent", "present", "present_empty"]


@dataclass(frozen=True)
class TargetConfig:
    """One private corpus target selected by a portable ID."""

    target_id: str
    kind: Literal["cdi", "raw"]
    source_path: Path
    observed_families: Mapping[str, int]
    observed_family_metadata: Mapping[str, ObservedFamilyEvidence] = field(
        default_factory=dict
    )
    telemetry_sidecar: Path | None = None
    reassembly_profile: Literal["legacy"] = "legacy"


@dataclass(frozen=True)
class QualificationConfig:
    """Validated report configuration loaded from an external JSON file."""

    format_version: int
    run_id: str
    subject_commit: str
    landing_time_file: Path
    corpus_manifest_paths: tuple[Path, ...]
    targets: tuple[TargetConfig, ...]
    spectrometer_clock_source: str = "spectrometer"
    dcb_clock_source: str = "dcb"
    expected_source_commits: Mapping[str, str] = field(default_factory=dict)
    config_digest: str = ""


@dataclass(frozen=True)
class BaselineClockAdapter:
    """Report coordinates derived from one explicit clock reference."""

    landing_time_file: Path
    reference_isot: str
    time_scale: str
    spectrometer_clock_source: str
    spectrometer_raw_seconds: float
    dcb_clock_source: str
    dcb_raw_seconds: float
    mjd_epoch_offset_days: float
    assumed: bool
    source_digest: str

@dataclass(frozen=True)
class SessionArtifacts:
    """Temporary products and decoded counts for one attempted session."""

    target_id: str
    session_id: str
    decoded_summary: Mapping[str, int]
    h5_path: Path | None
    fits_path: Path | None
    telemetry_state: str = "absent"
    issues: tuple[Mapping[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class QualificationResult:
    """Top-level outcome returned after every configured target is attempted."""

    status: Literal["complete", "complete_with_issues", "failed"]
    attempted_target_ids: tuple[str, ...]
    output_dir: Path


@dataclass(frozen=True)
class ReaderView:
    bundle: object
    frequency_mhz: np.ndarray | None


@dataclass(frozen=True)
class ReaderSemanticSummary:
    metrics: tuple[Mapping[str, object], ...]
    issues: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class PlotSession:
    """One HDF5 public-reader bundle retained only until its target is plotted."""

    session_id: str
    bundle: object
    frequency_mhz: np.ndarray | None
    source_format: Literal["hdf5", "fits"] = "hdf5"


@dataclass(frozen=True)
class FamilyPage:
    """Prepared family page and the evidence used to freeze its selection."""

    family: str
    count: int
    selection: object
    figure: object
    issues: tuple[Mapping[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TargetOutcome:
    """Portable evidence retained after one target's temporary files are removed."""

    metrics: tuple[Mapping[str, object], ...]
    issues: tuple[Mapping[str, object], ...]
    coverage: tuple[Mapping[str, object], ...]
    selections: Mapping[str, object]
    reader_layout_versions: tuple[str, ...]
    reader_source_formats: tuple[str, ...]


TargetExecutor = Callable[
    [TargetConfig, Path, BaselineClockAdapter], Sequence[SessionArtifacts]
]


def canonical_json(value: object) -> str:
    """Return the canonical single-line JSON representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_path(path: Path) -> str:
    """Hash one file or directory tree without recording its private path."""

    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(b"symlink\0")
        digest.update(os.readlink(path).encode("utf-8"))
        return digest.hexdigest()
    if path.is_file():
        digest.update(b"file\0")
        digest.update(hash_file(path).encode("ascii"))
        return digest.hexdigest()
    if not path.is_dir():
        raise ValueError("qualification source is neither a file nor a directory")
    digest.update(b"directory\0")
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        if child.is_symlink():
            digest.update(b"link\0" + relative + b"\0")
            digest.update(os.readlink(child).encode("utf-8"))
        elif child.is_file():
            digest.update(b"file\0" + relative + b"\0")
            digest.update(hash_file(child).encode("ascii"))
        elif child.is_dir():
            digest.update(b"dir\0" + relative + b"\0")
        else:
            digest.update(b"other\0" + relative + b"\0")
    return digest.hexdigest()


def hash_python_source_tree(path: Path) -> str:
    """Hash Python source names and contents while ignoring runtime caches."""

    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*.py") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for child in files:
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(b"python\0" + relative + b"\0")
        digest.update(hash_file(child).encode("ascii"))
    return digest.hexdigest()


def resolve_config_path(value: object, name: str, base_dir: Path | None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def validate_existing_file(
    value: object, name: str, base_dir: Path | None = None
) -> Path:
    path = resolve_config_path(value, name, base_dir)
    if not path.is_file():
        raise ValueError(f"{name} is not a file")
    return path


def validate_existing_path(
    value: object, name: str, base_dir: Path | None = None
) -> Path:
    path = resolve_config_path(value, name, base_dir)
    if not path.exists():
        raise ValueError(f"{name} does not exist")
    return path


def parse_target(value: object, index: int, base_dir: Path) -> TargetConfig:
    if not isinstance(value, dict):
        raise ValueError(f"targets[{index}] must be an object")
    target_id = value.get("target_id")
    if not isinstance(target_id, str) or not TARGET_ID_RE.fullmatch(target_id):
        raise ValueError(f"targets[{index}].target_id is not portable")
    kind = value.get("kind")
    if kind not in ("cdi", "raw"):
        raise ValueError(f"targets[{index}].kind must be 'cdi' or 'raw'")
    profile = value.get("reassembly_profile", "legacy")
    if profile != "legacy":
        raise ValueError(
            "the baseline reporter supports only the explicit legacy "
            "reassembly profile"
        )
    observed = value.get("observed_families")
    if not isinstance(observed, dict):
        raise ValueError(f"targets[{index}].observed_families must be an object")
    missing = sorted(set(FAMILIES) - set(observed))
    if missing:
        raise ValueError(
            f"targets[{index}].observed_families is missing {missing}"
        )
    unknown = sorted(set(observed) - set(FAMILIES))
    if unknown:
        raise ValueError(
            f"targets[{index}].observed_families has unknown families {unknown}"
        )
    normalized: dict[str, int] = {}
    for family, count in observed.items():
        if not isinstance(family, str) or not family:
            raise ValueError("observed family names must be nonempty strings")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"observed count for {family!r} must be nonnegative")
        normalized[family] = count
    metadata_value = value.get("observed_family_metadata")
    if not isinstance(metadata_value, dict):
        raise ValueError(
            f"targets[{index}].observed_family_metadata must be an object"
        )
    if set(metadata_value) != set(FAMILIES):
        raise ValueError(
            f"targets[{index}].observed_family_metadata must contain exactly "
            f"{list(FAMILIES)}"
        )
    metadata: dict[str, ObservedFamilyEvidence] = {}
    for family in FAMILIES:
        evidence = metadata_value[family]
        if not isinstance(evidence, dict):
            raise ValueError(
                f"observed metadata for {family!r} must be an object"
            )
        unit = evidence.get("unit")
        basis = evidence.get("basis")
        input_state = evidence.get("input_state")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"observed metadata unit for {family!r} is missing")
        if not isinstance(basis, str) or not basis:
            raise ValueError(f"observed metadata basis for {family!r} is missing")
        if input_state not in ("absent", "present", "present_empty"):
            raise ValueError(
                f"observed metadata input_state for {family!r} is invalid"
            )
        count = normalized[family]
        if input_state == "present" and count == 0:
            raise ValueError(
                f"observed metadata for {family!r} says present with zero count"
            )
        if input_state in ("absent", "present_empty") and count != 0:
            raise ValueError(
                f"observed metadata for {family!r} has count incompatible with "
                f"input_state={input_state}"
            )
        metadata[family] = ObservedFamilyEvidence(
            unit=unit,
            basis=basis,
            input_state=input_state,
        )
    sidecar_value = value.get("telemetry_sidecar")
    sidecar = (
        validate_existing_file(
            sidecar_value, f"targets[{index}].telemetry_sidecar", base_dir
        )
        if sidecar_value is not None
        else None
    )
    return TargetConfig(
        target_id=target_id,
        kind=kind,
        source_path=validate_existing_path(
            value.get("source_path"), f"targets[{index}].source_path", base_dir
        ),
        observed_families=normalized,
        observed_family_metadata=metadata,
        telemetry_sidecar=sidecar,
        reassembly_profile="legacy",
    )


def load_config(path: Path | str) -> QualificationConfig:
    """Load and validate one external report selection file."""

    config_path = Path(path).expanduser().resolve()
    base_dir = config_path.parent
    raw = config_path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("qualification config must be a JSON object")
    if value.get("format_version") != CONFIG_FORMAT_VERSION:
        raise ValueError(
            f"qualification config format_version must be {CONFIG_FORMAT_VERSION}"
        )
    run_id = value.get("run_id")
    subject_commit = value.get("subject_commit")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(subject_commit, str) or not subject_commit:
        raise ValueError("subject_commit must be a nonempty string")
    if re.fullmatch(r"[0-9a-f]{40}", subject_commit) is None:
        raise ValueError("subject_commit must be an exact full Git SHA")
    spectrometer_clock_source = value.get("spectrometer_clock_source")
    if not isinstance(spectrometer_clock_source, str) or not spectrometer_clock_source:
        raise ValueError("spectrometer_clock_source must be a nonempty string")
    dcb_clock_source = value.get("dcb_clock_source")
    if not isinstance(dcb_clock_source, str) or not dcb_clock_source:
        raise ValueError("dcb_clock_source must be a nonempty string")
    expected_source_commits = value.get("expected_source_commits")
    if not isinstance(expected_source_commits, dict):
        raise ValueError("expected_source_commits must be an object")
    required_sources = {"uncrater", "lusee_telemetry"}
    if set(expected_source_commits) != required_sources:
        raise ValueError(
            "expected_source_commits must contain exactly uncrater and "
            "lusee_telemetry"
        )
    for name, commit in expected_source_commits.items():
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError(f"expected source commit for {name} must be a full SHA")
    manifests = value.get("corpus_manifest_paths")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("corpus_manifest_paths must be a nonempty list")
    manifest_paths = tuple(
        validate_existing_file(item, f"corpus_manifest_paths[{index}]", base_dir)
        for index, item in enumerate(manifests)
    )
    target_values = value.get("targets")
    if not isinstance(target_values, list) or not target_values:
        raise ValueError("targets must be a nonempty list")
    targets = tuple(
        parse_target(item, index, base_dir)
        for index, item in enumerate(target_values)
    )
    ids = [target.target_id for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("target IDs must be unique")
    return QualificationConfig(
        format_version=CONFIG_FORMAT_VERSION,
        run_id=run_id,
        subject_commit=subject_commit,
        landing_time_file=validate_existing_file(
            value.get("landing_time_file"), "landing_time_file", base_dir
        ),
        corpus_manifest_paths=manifest_paths,
        targets=tuple(sorted(targets, key=lambda target: target.target_id)),
        spectrometer_clock_source=spectrometer_clock_source,
        dcb_clock_source=dcb_clock_source,
        expected_source_commits=dict(sorted(expected_source_commits.items())),
        config_digest=hashlib.sha256(raw).hexdigest(),
    )


def load_baseline_clock_adapter(config: QualificationConfig) -> BaselineClockAdapter:
    """Validate the report-only landing reference and build plot coordinates."""

    from astropy.time import Time

    path = config.landing_time_file
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("landing reference must be a JSON object")
    if value.get("format_version") != 1:
        raise ValueError("landing reference format_version must be 1")
    if value.get("reference_event") != "landing":
        raise ValueError("landing reference_event must be 'landing'")
    if value.get("assumed") is not True:
        raise ValueError("baseline qualification requires assumed=true")
    source = value.get("source")
    if not isinstance(source, str) or not source:
        raise ValueError("landing source must be a nonempty string")
    reference_isot = value.get("clock_reference_isot")
    time_scale = value.get("time_scale")
    clocks = value.get("clocks")
    if not isinstance(reference_isot, str) or not reference_isot:
        raise ValueError("clock_reference_isot must be a nonempty string")
    if not isinstance(time_scale, str) or not time_scale:
        raise ValueError("time_scale must be a nonempty string")
    if not isinstance(clocks, dict):
        raise ValueError("landing clocks must be an object")

    def raw_anchor(source: str) -> float:
        clock = clocks.get(source)
        if not isinstance(clock, dict):
            raise ValueError(f"landing reference has no clock {source!r}")
        raw_seconds = clock.get("clock_reference_raw_seconds")
        if isinstance(raw_seconds, bool) or not isinstance(raw_seconds, (int, float)):
            raise ValueError(f"clock {source!r} has no numeric raw anchor")
        result = float(raw_seconds)
        if not math.isfinite(result):
            raise ValueError(f"clock {source!r} raw anchor is not finite")
        return result

    spectrometer_raw = raw_anchor(config.spectrometer_clock_source)
    dcb_raw = raw_anchor(config.dcb_clock_source)
    reference = Time(reference_isot, format="isot", scale=time_scale.lower())
    return BaselineClockAdapter(
        landing_time_file=path,
        reference_isot=reference_isot,
        time_scale=time_scale.lower(),
        spectrometer_clock_source=config.spectrometer_clock_source,
        spectrometer_raw_seconds=spectrometer_raw,
        dcb_clock_source=config.dcb_clock_source,
        dcb_raw_seconds=dcb_raw,
        mjd_epoch_offset_days=float(reference.mjd),
        assumed=True,
        source_digest=hash_file(path),
    )


def issue_record(
    target_id: str,
    session_id: str,
    stage: str,
    code: str,
    message: str,
    *,
    severity: str = "warning",
    details: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "session_id": session_id,
        "stage": stage,
        "code": code,
        "severity": severity,
        "message": message,
        "details": dict(details or {}),
        "occurrence_count": 1,
    }


def clean_message(message: object, private_paths: Sequence[Path]) -> str:
    result = str(message)
    for path in sorted(private_paths, key=lambda item: len(str(item)), reverse=True):
        result = result.replace(str(path), "<private-path>")
    return result


def clean_structured_value(
    value: object,
    private_paths: Sequence[Path],
) -> object:
    if isinstance(value, str):
        return clean_message(value, private_paths)
    if isinstance(value, Mapping):
        return {
            key: clean_structured_value(item, private_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [clean_structured_value(item, private_paths) for item in value]
    return value


def capture_call(
    function: Callable[[], Any],
    *,
    target_id: str,
    session_id: str,
    stage: str,
    private_paths: Sequence[Path],
) -> tuple[bool, Any, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    stdout = io.StringIO()
    stderr = io.StringIO()
    caught: list[warnings.WarningMessage] = []
    result: Any = None
    failure: Exception | None = None
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            caught = captured
            try:
                result = function()
            except Exception as exc:  # noqa: BLE001
                failure = exc
    except Exception as exc:  # noqa: BLE001
        failure = exc

    for warning in caught:
        records.append(issue_record(
            target_id,
            session_id,
            stage,
            f"warning.{warning.category.__name__}",
            clean_message(warning.message, private_paths),
        ))
    for stream_name, text in (
        ("stdout", stdout.getvalue()),
        ("stderr", stderr.getvalue()),
    ):
        for line in text.splitlines():
            if not line.lstrip().startswith("Warning:"):
                continue
            records.append(issue_record(
                target_id,
                session_id,
                stage,
                f"diagnostic.{stream_name}",
                clean_message(line.strip(), private_paths),
            ))
    if failure is not None:
        records.append(issue_record(
            target_id,
            session_id,
            stage,
            f"stage_failed.{type(failure).__name__}",
            clean_message(failure, private_paths),
            severity="error",
        ))
        return False, None, records
    return True, result, records


def products_summary(
    products: object, telemetry_state: str = "absent"
) -> dict[str, int]:
    calibrator_count = sum(
        len(getattr(products, name, ()))
        for name in (
            "cal_data",
            "calibrator_metadata",
            "calibrator_data",
            "calibrator_raw_pfb",
            "calibrator_debug",
        )
    )
    return {
        "normal": len(getattr(products, "spectra", ())),
        "tr": len(getattr(products, "tr_spectra", ())),
        "zoom": len(getattr(products, "zoom_spectra", ())),
        "waveform": len(getattr(products, "waveforms", ())),
        "grimm": len(getattr(products, "grimm_spectra", ())),
        "telemetry": int(telemetry_state == "decoded"),
        "housekeeping": len(getattr(products, "housekeeping", ())),
        "calibrator": calibrator_count,
    }


def ingest_issue_union(
    products: object,
    telemetry: object | None,
    context_issues: Sequence[object] = (),
) -> tuple[object, ...]:
    issue_by_id = {}
    for issue in (
        *products.issues,
        *(() if telemetry is None else telemetry.issues),
        *context_issues,
    ):
        existing = issue_by_id.get(issue.issue_id)
        if existing is not None and existing != issue:
            raise ValueError("request issue sources reuse an issue ID")
        issue_by_id[issue.issue_id] = issue
    return tuple(issue_by_id[key] for key in sorted(issue_by_id))


def append_ingest_issue_records(
    output: list[dict[str, object]],
    issues: Sequence[object],
    target_id: str,
    session_id: str,
    reported_issue_ids: set[str],
    private_paths: Sequence[Path],
) -> None:
    for issue in issues:
        if issue.issue_id in reported_issue_ids:
            continue
        output.append(qualification_issue_record(
            issue,
            target_id,
            session_id,
            private_paths,
        ))
        reported_issue_ids.add(issue.issue_id)


def make_write_request(
    target: TargetConfig,
    products: object,
    clock_adapter: BaselineClockAdapter,
    telemetry: object | None,
    context_issues: Sequence[object] = (),
):
    from lusee.ingest import (
        InterpolationPolicy,
        LunarLocation,
        RunProvenance,
        TelemetryDecodeResult,
        WriteRequest,
        family_statuses_for_products,
        load_clock_reference_set,
    )
    from lusee.ingest.constants import (
        DEFAULT_LUN_HEIGHT_M,
        DEFAULT_LUN_LAT_DEG,
        DEFAULT_LUN_LONG_DEG,
    )
    from lusee.ingest.write_request import FAMILY_TYPES

    request_telemetry = telemetry or TelemetryDecodeResult.absent()
    normalized_context_issues = tuple(context_issues)
    issues = ingest_issue_union(
        products,
        request_telemetry,
        normalized_context_issues,
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
    return WriteRequest(
        products=products,
        clock_reference_set=load_clock_reference_set(
            clock_adapter.landing_time_file
        ),
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
                "qualification_input_identity_not_recorded"
            ),
            source_kind="session" if target.kind == "cdi" else "flash",
            source_path=None,
            pipeline_version=None,
        ),
        issues=issues,
        family_statuses=family_statuses,
        telemetry=request_telemetry,
        interpolation_policy=InterpolationPolicy(),
        context_issues=normalized_context_issues,
    )


def telemetry_state(result: object | None) -> str:
    from lusee.ingest import (
        TelemetryCoverage,
        TelemetryDecoderStatus,
        TelemetryInputState,
    )

    if result is None or result.input_state is TelemetryInputState.ABSENT:
        return "absent"
    if result.decoder_status is TelemetryDecoderStatus.UNAVAILABLE:
        return "decoder_unavailable"
    if result.decoder_status in (
        TelemetryDecoderStatus.BROKEN,
        TelemetryDecoderStatus.INCOMPATIBLE,
    ):
        return "decoder_broken"
    if result.coverage is TelemetryCoverage.PRESENT_EMPTY:
        return "present_empty"
    return "decoded"


def qualification_issue_record(
    issue: object,
    target_id: str,
    session_id: str,
    private_paths: Sequence[Path],
) -> dict[str, object]:
    record = {
        "target_id": target_id,
        "session_id": session_id,
        **issue.as_dict(),
        "occurrence_count": 1,
    }
    cleaned = clean_structured_value(record, private_paths)
    if not isinstance(cleaned, dict):
        raise TypeError("structured issue redaction did not return a mapping")
    return cleaned


def write_one_session(
    target: TargetConfig,
    session_id: str,
    products: object,
    work_dir: Path,
    clock_adapter: BaselineClockAdapter,
    *,
    telemetry: object | None = None,
    context_issues: Sequence[object] = (),
    telemetry_state: str = "absent",
    initial_issues: Sequence[Mapping[str, object]] = (),
    reported_issue_ids: set[str] | None = None,
) -> SessionArtifacts:
    from lusee.ingest import write_fits, write_hdf5

    paths = [target.source_path, work_dir]
    if target.telemetry_sidecar is not None:
        paths.append(target.telemetry_sidecar)
    issues = [dict(item) for item in initial_issues]
    reported = reported_issue_ids if reported_issue_ids is not None else set()
    reported.update(
        str(item["issue_id"])
        for item in issues
        if item.get("issue_id") is not None
    )
    append_ingest_issue_records(
        issues,
        ingest_issue_union(products, telemetry, context_issues),
        target.target_id,
        session_id,
        reported,
        paths,
    )
    h5_path = work_dir / f"{session_id}.h5"
    fits_path = work_dir / f"{session_id}.fits"
    ok_h5, _, found = capture_call(
        lambda: write_hdf5(
            make_write_request(
                target,
                products,
                clock_adapter,
                telemetry,
                context_issues,
            ),
            h5_path,
        ),
        target_id=target.target_id,
        session_id=session_id,
        stage="hdf5",
        private_paths=paths,
    )
    issues.extend(found)
    ok_fits, _, found = capture_call(
        lambda: write_fits(
            make_write_request(
                target,
                products,
                clock_adapter,
                telemetry,
                context_issues,
            ),
            fits_path,
        ),
        target_id=target.target_id,
        session_id=session_id,
        stage="fits",
        private_paths=paths,
    )
    issues.extend(found)
    return SessionArtifacts(
        target_id=target.target_id,
        session_id=session_id,
        decoded_summary=products_summary(
            products, telemetry_state
        ),
        h5_path=h5_path if ok_h5 else None,
        fits_path=fits_path if ok_fits else None,
        telemetry_state=telemetry_state,
        issues=tuple(issues),
    )


def execute_target_default(
    target: TargetConfig,
    work_dir: Path,
    clock_adapter: BaselineClockAdapter,
) -> Sequence[SessionArtifacts]:
    """Execute current public ingest stages without changing production code."""

    from lusee.ingest import (
        IssueCollector,
        decode_legacy_sidecar,
        load_clock_reference_set,
        parse_flash,
        read_uncrater_session,
        write_uncrater_session,
    )
    from lusee.ingest.telemetry import map_dcb_absolute_time
    from lusee.ingest.pipeline import _issues_for_session

    work_dir.mkdir(parents=True, exist_ok=True)
    issue_collector = IssueCollector()
    if target.kind == "cdi":
        ok, products, decode_issues = capture_call(
            lambda: read_uncrater_session(
                target.source_path,
                issue_collector=issue_collector,
            ),
            target_id=target.target_id,
            session_id="session_000",
            stage="decoded",
            private_paths=[target.source_path, work_dir],
        )
        if not ok:
            return [SessionArtifacts(
                target_id=target.target_id,
                session_id="session_000",
                decoded_summary={family: 0 for family in FAMILIES},
                h5_path=None,
                fits_path=None,
                issues=tuple(decode_issues),
            )]
        telemetry = None
        state = "absent"
        sidecar_issues: list[dict[str, object]] = list(decode_issues)
        if target.telemetry_sidecar is not None:
            def decode_sidecar():
                value = decode_legacy_sidecar(
                    target.telemetry_sidecar,
                    issue_collector=issue_collector,
                )
                return map_dcb_absolute_time(
                    value,
                    clock_reference_set=load_clock_reference_set(
                        clock_adapter.landing_time_file
                    ),
                    issue_collector=issue_collector,
                )

            ok, value, found = capture_call(
                decode_sidecar,
                target_id=target.target_id,
                session_id="session_000",
                stage="telemetry_decode",
                private_paths=[
                    target.source_path, target.telemetry_sidecar, work_dir
                ],
            )
            sidecar_issues.extend(found)
            if ok:
                telemetry = value
                state = telemetry_state(value)
            else:
                state = "decoder_broken"
        return [write_one_session(
            target,
            "session_000",
            products,
            work_dir,
            clock_adapter,
            telemetry=telemetry,
            telemetry_state=state,
            initial_issues=sidecar_issues,
        )]

    raw_context_marker = issue_collector.mark()
    ok, parsed, parse_issues = capture_call(
        lambda: parse_flash(
            target.source_path,
            landing_time_file=clock_adapter.landing_time_file,
            issue_collector=issue_collector,
        ),
        target_id=target.target_id,
        session_id="target",
        stage="raw_reassembly",
        private_paths=[target.source_path, work_dir],
    )
    if not ok:
        return [SessionArtifacts(
            target_id=target.target_id,
            session_id="target",
            decoded_summary={family: 0 for family in FAMILIES},
            h5_path=None,
            fits_path=None,
            issues=tuple(parse_issues),
        )]
    sessions, _, _ = parsed
    raw_context_issues = issue_collector.since(raw_context_marker)
    ordered_sessions = sorted(sessions, key=lambda item: item.ordinal)
    session_contexts = [
        _issues_for_session(
            raw_context_issues,
            session=session,
            session_name=f"session_{index:03d}",
        )
        for index, session in enumerate(ordered_sessions)
    ]
    selected_context_ids = {
        issue.issue_id
        for context in session_contexts
        for issue in context
    }
    global_issues = tuple(
        issue
        for issue in raw_context_issues
        if issue.issue_id not in selected_context_ids
    )
    reported_issue_ids: set[str] = set()
    append_ingest_issue_records(
        parse_issues,
        global_issues,
        target.target_id,
        "target",
        reported_issue_ids,
        [target.source_path, work_dir],
    )
    if not ordered_sessions:
        return [] if not parse_issues else [SessionArtifacts(
            target_id=target.target_id,
            session_id="target",
            decoded_summary={family: 0 for family in FAMILIES},
            h5_path=None,
            fits_path=None,
            issues=tuple(parse_issues),
        )]
    artifacts: list[SessionArtifacts] = []
    for index, (session, context_issues) in enumerate(zip(
        ordered_sessions,
        session_contexts,
    )):
        session_id = f"session_{index:03d}"
        session_dir = work_dir / session_id
        session_issues: list[dict[str, object]] = (
            list(parse_issues) if index == 0 else []
        )
        append_ingest_issue_records(
            session_issues,
            context_issues,
            target.target_id,
            session_id,
            reported_issue_ids,
            [target.source_path, work_dir],
        )
        ok, _, found = capture_call(
            lambda session=session, session_dir=session_dir: write_uncrater_session(
                session, session_dir
            ),
            target_id=target.target_id,
            session_id=session_id,
            stage="session_extract",
            private_paths=[target.source_path, work_dir],
        )
        session_issues.extend(found)
        if not ok:
            artifacts.append(SessionArtifacts(
                target_id=target.target_id,
                session_id=session_id,
                decoded_summary={family: 0 for family in FAMILIES},
                h5_path=None,
                fits_path=None,
                issues=tuple(session_issues),
            ))
            continue
        ok, products, found = capture_call(
            lambda session_dir=session_dir: read_uncrater_session(
                session_dir,
                issue_collector=issue_collector,
            ),
            target_id=target.target_id,
            session_id=session_id,
            stage="decoded",
            private_paths=[target.source_path, work_dir],
        )
        session_issues.extend(found)
        if not ok:
            artifacts.append(SessionArtifacts(
                target_id=target.target_id,
                session_id=session_id,
                decoded_summary={family: 0 for family in FAMILIES},
                h5_path=None,
                fits_path=None,
                issues=tuple(session_issues),
            ))
            continue
        artifacts.append(write_one_session(
            target,
            session_id,
            products,
            work_dir,
            clock_adapter,
            telemetry=session.telemetry,
            context_issues=context_issues,
            telemetry_state=telemetry_state(session.telemetry),
            initial_issues=session_issues,
            reported_issue_ids=reported_issue_ids,
        ))
    return artifacts


def semantic_name(name: object) -> str:
    key = str(name).casefold()
    aliases = {
        "upid": "unique_packet_id",
        "raw_time": "raw_seconds",
        "act_gain": "actual_gain",
        "wgt_ndx": "weight_ndx",
    }
    return aliases.get(key, key)


def add_mapping(
    output: dict[str, object], prefix: str, values: Mapping[object, object]
) -> None:
    for name in sorted(values, key=lambda item: str(item).casefold()):
        output[f"{prefix}/{semantic_name(name)}"] = values[name]


def bundle_semantics(
    bundle: object,
    frequency_mhz: np.ndarray | None = None,
    clock_adapter: BaselineClockAdapter | None = None,
) -> dict[str, object]:
    """Flatten the public ``SessionBundle`` surface into canonical fields."""

    output: dict[str, object] = {}
    scalar_fields = {
        "layout_version": getattr(bundle, "layout_version", None),
        "normal/units": getattr(bundle, "spectra_units", None),
        "normal/representation": getattr(bundle, "spectra_representation", None),
        "normal/normalization_version": getattr(
            bundle, "spectra_normalization_version", None
        ),
    }
    for path, value in scalar_fields.items():
        if value is not None:
            output[path] = value
    add_mapping(output, "session", getattr(bundle, "session_invariants", {}))
    add_mapping(output, "constants", getattr(bundle, "constants", {}))

    fields = {
        "normal/data": "spectra",
        "normal/unique_ids": "spectra_unique_ids",
        "normal/raw_times": "spectra_raw_times",
        "normal/mjd_times": "spectra_mjd_times",
        "tr/data": "tr_spectra",
        "tr/unique_ids": "tr_unique_ids",
        "tr/raw_times": "tr_raw_times",
        "tr/mjd_times": "tr_mjd_times",
        "tr/navg2": "tr_navg2_per_sample",
        "tr/length": "tr_length_per_sample",
        "zoom/data": "zoom_spectra",
        "zoom/unique_ids": "zoom_unique_ids",
        "zoom/pfb_indices": "zoom_pfb_indices",
        "zoom/raw_times": "zoom_raw_times",
        "zoom/mjd_times": "zoom_mjd_times",
        "grimm/data": "grimm_spectra",
        "grimm/unique_ids": "grimm_unique_ids",
        "grimm/raw_times": "grimm_raw_times",
    }
    for path, attribute in fields.items():
        value = getattr(bundle, attribute, None)
        if value is not None:
            output[path] = value
    add_mapping(output, "normal/metadata", getattr(bundle, "spectra_metadata", {}))
    add_mapping(output, "tr/metadata", getattr(bundle, "tr_metadata", {}))
    for channel, value in sorted(getattr(bundle, "waveforms", {}).items()):
        output[f"waveform/channel_{channel}/data"] = value
    for channel, value in sorted(getattr(bundle, "waveform_times", {}).items()):
        output[f"waveform/channel_{channel}/times"] = value
    for type_id, values in sorted(getattr(bundle, "housekeeping", {}).items()):
        add_mapping(output, f"housekeeping/type_{type_id}", values)
    add_mapping(output, "telemetry/fpga", getattr(bundle, "dcb_fpga", {}))
    add_mapping(output, "telemetry/encoder", getattr(bundle, "dcb_encoder", {}))
    add_mapping(output, "telemetry/interpolated", getattr(bundle, "interp_telemetry", {}))
    if frequency_mhz is not None:
        output["normal/frequency_mhz"] = frequency_mhz
    fpga = getattr(bundle, "dcb_fpga", {})
    encoder = getattr(bundle, "dcb_encoder", {})
    time_values = fpga
    time_source = "fpga"
    if "mission_seconds" not in time_values:
        time_values = encoder
        time_source = "encoder"
    if (
        clock_adapter is not None
        and "mission_seconds" in time_values
        and "lusee_subsecs" in time_values
    ):
        raw_seconds = (
            np.asarray(time_values["mission_seconds"], dtype=np.float64)
            + np.asarray(time_values["lusee_subsecs"], dtype=np.float64) / 65536.0
        )
        output["telemetry/display/assumed_mjd"] = (
            (raw_seconds - clock_adapter.dcb_raw_seconds) / 86400.0
            + clock_adapter.mjd_epoch_offset_days
        )
        output["telemetry/display/time_source"] = time_source
    return output


def field_units(field_path: str, bundle: object | None = None) -> str:
    if field_path == "normal/data":
        return str(getattr(bundle, "spectra_units", None) or "SDU")
    if field_path == "constants/clock_epoch_isot":
        return "qualification absolute-time assumption (TIME-004)"
    if field_path == "constants/mjd_epoch_offset_days":
        return "MJD day (qualification assumption; TIME-004)"
    if field_path == "constants/raw_time_subtract_seconds":
        return "raw clock second anchor (qualification assumption; TIME-004)"
    if field_path in ("constants/clock_source", "constants/time_scale"):
        return "qualification clock convention (TIME-004)"
    if field_path == "telemetry/display/time_source":
        return "qualification display provenance (TIME-004)"
    if field_path.endswith("frequency_mhz"):
        return "MHz (legacy reader-derived; FREQ-001 open)"
    if field_path.endswith("mjd_times") or field_path.endswith("assumed_mjd"):
        return "MJD day (qualification assumption; TIME-004)"
    if (
        field_path.endswith("raw_times")
        or field_path.endswith("raw_seconds")
        or field_path.endswith("mission_seconds")
    ):
        return "raw clock second"
    if field_path.endswith("lusee_subsecs"):
        return "raw 1/65536-second tick"
    if field_path.startswith("housekeeping/") and time_field_path(field_path):
        return "raw seconds (native HK counter)"
    if field_path.startswith("waveform/") and time_field_path(field_path):
        return "raw ADC clock second (unmapped; not absolute time)"
    if field_path.startswith(("tr/", "zoom/", "waveform/", "grimm/")):
        return "raw count (UNITS-001 open)"
    if field_path.startswith(("telemetry/", "housekeeping/")):
        return "unit unestablished (UNITS-001 open)"
    return "not applicable or unestablished"


def canonical_dtype(array: np.ndarray) -> str:
    dtype = array.dtype
    if dtype.kind in "SU":
        return f"{dtype.kind}{dtype.itemsize}"
    return f"{dtype.kind}{dtype.itemsize}"


def canonical_array_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    if value.dtype.kind == "O":
        return canonical_json(value.tolist()).encode("utf-8")
    if value.dtype.kind == "U":
        return canonical_json(value.tolist()).encode("utf-8")
    if value.dtype.kind in "fc":
        value = value.copy()
        value[np.isnan(value)] = np.nan
    if value.dtype.byteorder == ">" or (
        value.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        value = value.astype(value.dtype.newbyteorder("<"), copy=False)
    return np.ascontiguousarray(value).tobytes()


def array_digest(value: object) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(canonical_dtype(array).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(canonical_array_bytes(array))
    return digest.hexdigest()


def json_number(value: object) -> int | float | None:
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def categorical_field_path(field_path: str) -> bool:
    """Return whether a flattened reader field has categorical semantics."""

    name = field_path.rsplit("/", 1)[-1].casefold()
    if field_path.startswith("telemetry/encoder/"):
        return name in ("enc_status", "status") or "status" in name
    if not field_path.startswith(("telemetry/", "housekeeping/")):
        return False
    return (
        any(
            token in name
            for token in ("status", "error", "gain", "checksum", "version")
        )
        or name == "ok"
        or name == "id"
        or name.endswith(("_id", "_ids"))
    )


def time_field_path(field_path: str) -> bool:
    """Return whether one flattened field is a time coordinate."""

    name = field_path.rsplit("/", 1)[-1].casefold()
    return (
        name in {
            "time",
            "times",
            "timestamp",
            "timestamps",
            "raw_seconds",
            "mission_seconds",
            "raw_times",
            "mjd_times",
            "assumed_mjd",
        }
        or name.endswith(("_time", "_times", "_timestamp", "_timestamps"))
    )


def categorical_statistics(value: object) -> dict[str, object]:
    """Return deterministic counts and mode for a categorical array."""

    array = np.asarray(value)
    flat = array.reshape(-1)
    if array.dtype.kind in "fc":
        valid_mask = np.isfinite(flat)
        valid = flat[valid_mask]
        missing = int(flat.size - np.count_nonzero(valid_mask))
    else:
        valid = flat
        missing = 0
    counts = Counter(str(item) for item in valid.tolist())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    mode, mode_count = ranked[0] if ranked else (None, 0)
    return {
        "valid_count": int(valid.size),
        "missing_count": missing,
        "counts": dict(sorted(counts.items())),
        "mode": mode,
        "mode_count": mode_count,
    }


def array_metric(
    target_id: str,
    session_id: str,
    field_path: str,
    value: object,
    source_format: str,
    *,
    units: str | None = None,
) -> dict[str, object]:
    array = np.asarray(value)
    categorical = categorical_field_path(field_path)
    metric: dict[str, object] = {
        "record_type": "array",
        "target_id": target_id,
        "session_id": session_id,
        "field_path": field_path,
        "source_format": source_format,
        "shape": list(array.shape),
        "dtype": canonical_dtype(array),
        "digest": array_digest(array),
        "valid_count": int(array.size),
        "missing_count": 0,
        "units": units or field_units(field_path),
    }
    if categorical:
        metric.update(categorical_statistics(array))
    elif array.dtype.kind in "fc":
        valid = np.isfinite(array)
        metric["valid_count"] = int(np.count_nonzero(valid))
        metric["missing_count"] = int(array.size - np.count_nonzero(valid))
        values = array[valid]
        if values.size and array.dtype.kind == "c":
            magnitude = np.abs(values)
            phase = np.angle(values)
            metric.update({
                "magnitude_min": json_number(np.min(magnitude)),
                "magnitude_max": json_number(np.max(magnitude)),
                "magnitude_mean": json_number(np.mean(magnitude)),
                "phase_min": json_number(np.min(phase)),
                "phase_max": json_number(np.max(phase)),
            })
        elif values.size:
            metric.update({
                "minimum": json_number(np.min(values)),
                "maximum": json_number(np.max(values)),
                "mean": json_number(np.mean(values)),
            })
    elif array.dtype.kind in "iub" and array.size:
        metric.update({
            "minimum": json_number(np.min(array)),
            "maximum": json_number(np.max(array)),
            "mean": json_number(np.mean(array.astype(np.float64))),
        })
    elif array.dtype.kind in "SUO" and array.size:
        metric.update(categorical_statistics(array))
    if (
        not categorical
        and array.ndim == 1
        and array.dtype.kind in "fiu"
        and array.size
        and time_field_path(field_path)
    ):
        finite = np.asarray(array, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            ordered = np.sort(finite)
            metric["time_minimum"] = json_number(ordered[0])
            metric["time_maximum"] = json_number(ordered[-1])
            if ordered.size > 1:
                gaps = np.diff(ordered)
                metric["sampling_gap_median"] = json_number(np.median(gaps))
                metric["sampling_gap_maximum"] = json_number(np.max(gaps))
    return metric


def semantic_equal(left: object, right: object) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in "fc" or right_array.dtype.kind in "fc":
        try:
            return bool(np.array_equal(left_array, right_array, equal_nan=True))
        except TypeError:
            return False
    return bool(np.array_equal(left_array, right_array))


def compare_semantics(
    hdf5_semantics: Mapping[str, object],
    fits_semantics: Mapping[str, object],
    *,
    target_id: str = "",
    session_id: str = "",
) -> list[dict[str, object]]:
    """Compare the union of canonical fields returned by both readers."""

    records: list[dict[str, object]] = []
    for field_path in sorted(set(hdf5_semantics) | set(fits_semantics)):
        base: dict[str, object] = {
            "record_type": "parity",
            "target_id": target_id,
            "session_id": session_id,
            "field_path": field_path,
        }
        if field_path not in hdf5_semantics:
            records.append({**base, "status": "missing_hdf5"})
            continue
        if field_path not in fits_semantics:
            records.append({**base, "status": "missing_fits"})
            continue
        left = np.asarray(hdf5_semantics[field_path])
        right = np.asarray(fits_semantics[field_path])
        if left.shape != right.shape:
            records.append({
                **base,
                "status": "shape_mismatch",
                "hdf5_shape": list(left.shape),
                "fits_shape": list(right.shape),
            })
            continue
        if canonical_dtype(left) != canonical_dtype(right):
            records.append({
                **base,
                "status": "dtype_mismatch",
                "hdf5_dtype": canonical_dtype(left),
                "fits_dtype": canonical_dtype(right),
            })
            continue
        if semantic_equal(left, right):
            records.append({**base, "status": "equal"})
            continue
        mismatch: dict[str, object] = {**base, "status": "value_mismatch"}
        if left.size and right.size:
            flat_left = left.reshape(-1)
            flat_right = right.reshape(-1)
            for index in range(min(flat_left.size, flat_right.size)):
                if not semantic_equal(flat_left[index], flat_right[index]):
                    mismatch["first_index"] = index
                    mismatch["hdf5_value"] = str(flat_left[index])
                    mismatch["fits_value"] = str(flat_right[index])
                    break
        records.append(mismatch)
    return records


def read_public_bundle(path: Path, prefer_format: str) -> ReaderView:
    from lusee.ingest import load_bundle

    bundle = load_bundle(path, prefer_format=prefer_format)
    frequency_mhz = None
    if (
        getattr(bundle, "layout_version", None) in (2, 3)
        and getattr(bundle, "spectra", None) is not None
    ):
        groups = bundle.split_by_frequency_grid()
        if len(groups) != 1:
            raise ValueError(
                "qualification reader expected one homogeneous frequency grid"
            )
        frequency_mhz = np.asarray(groups[0].frequency_for_row(0))
    return ReaderView(
        bundle=bundle,
        frequency_mhz=frequency_mhz,
    )


def family_counts_from_bundle(bundle: object) -> dict[str, int]:
    def rows(value: object) -> int:
        return int(np.asarray(value).shape[0]) if value is not None else 0

    family_status = getattr(bundle, "family_status", {})
    status_families = np.asarray(family_status.get("family", ()))
    persisted_rows = np.asarray(family_status.get("persisted_rows", ()))
    if status_families.size and status_families.shape == persisted_rows.shape:
        by_family = {
            str(family): int(count)
            for family, count in zip(status_families, persisted_rows)
        }
        return {
            "normal": by_family.get("spectra", 0),
            "tr": by_family.get("tr_spectra", 0),
            "zoom": by_family.get("zoom_spectra", 0),
            "waveform": by_family.get("waveforms", 0),
            "grimm": by_family.get("grimm_spectra", 0),
            "telemetry": int(by_family.get("dcb_telemetry", 0) > 0),
            "housekeeping": by_family.get("housekeeping", 0),
            "calibrator": sum(
                by_family.get(family, 0)
                for family in (
                    "calibrator_metadata",
                    "calibrator_data",
                    "calibrator_raw_pfb",
                    "calibrator_debug",
                )
            ),
        }

    return {
        "normal": rows(getattr(bundle, "spectra", None)),
        "tr": rows(getattr(bundle, "tr_spectra", None)),
        "zoom": rows(getattr(bundle, "zoom_spectra", None)),
        "waveform": sum(
            rows(value) for value in getattr(bundle, "waveforms", {}).values()
        ),
        "grimm": rows(getattr(bundle, "grimm_spectra", None)),
        "telemetry": int(bool(getattr(bundle, "dcb_fpga", {})) or bool(getattr(bundle, "dcb_encoder", {}))),
        "housekeeping": sum(
            rows(next(iter(values.values()))) if values else 0
            for values in getattr(bundle, "housekeeping", {}).values()
        ),
        "calibrator": sum(
            rows(values.get("unique_ids"))
            for values in getattr(bundle, "calibrator", {}).values()
        ),
    }


def empty_family_counts() -> dict[str, int]:
    return {family: 0 for family in FAMILIES}


def observed_family_evidence(
    target: TargetConfig,
    family: str,
) -> ObservedFamilyEvidence:
    """Return explicit inventory metadata or a labelled direct-test fallback."""

    evidence = target.observed_family_metadata.get(family)
    if evidence is not None:
        return evidence
    count = int(target.observed_families.get(family, 0))
    return ObservedFamilyEvidence(
        unit="unspecified item",
        basis="direct TargetConfig without external inventory metadata",
        input_state="present" if count > 0 else "absent",
    )


def add_family_counts(
    total: dict[str, int], values: Mapping[str, int] | None
) -> None:
    if values is None:
        return
    for family in FAMILIES:
        total[family] += int(values.get(family, 0))


def coverage_state(
    observed_count: int,
    count: int,
    *,
    input_state: str | None = None,
    failed: bool = False,
    supported: bool = True,
) -> str:
    resolved_input_state = input_state or (
        "present" if observed_count > 0 else "absent"
    )
    if count > 0:
        return "present"
    if resolved_input_state == "absent":
        return "absent_in_input"
    if failed:
        return "stage_failed"
    if not supported:
        return "unsupported"
    if resolved_input_state == "present_empty":
        return "present_empty"
    return "invalid_or_dropped"


def coverage_consistency_issues(
    target: TargetConfig,
    coverage: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Turn inventory contradictions and silent drops into run issues."""

    issues: list[dict[str, object]] = []
    for row in coverage:
        stage = str(row.get("stage", ""))
        if stage == "observed_input":
            continue
        family = str(row.get("family", ""))
        count = int(row.get("count", 0))
        state = str(row.get("state", ""))
        input_state = str(row.get("observed_input_state", ""))
        if input_state in ("absent", "present_empty") and count > 0:
            issues.append(issue_record(
                target.target_id,
                "all",
                "coverage",
                "coverage.unexpected_presence",
                f"{stage} contains {family} although inventory records it "
                f"{input_state}",
                details={"family": family, "stage": stage, "count": count},
            ))
        if state == "invalid_or_dropped":
            issues.append(issue_record(
                target.target_id,
                "all",
                "coverage",
                "coverage.invalid_or_dropped",
                f"{family} was present in inventory but has no {stage} evidence",
                details={"family": family, "stage": stage, "count": count},
            ))
    return issues


def build_target_coverage(
    target: TargetConfig,
    decoded_counts: Mapping[str, int],
    h5_counts: Mapping[str, int],
    fits_counts: Mapping[str, int],
    reader_counts: Mapping[str, int],
    plotted_counts: Mapping[str, int],
    failed_stages: set[str],
    *,
    telemetry_state: str = "absent",
    failed_plot_families: set[str] | None = None,
) -> list[dict[str, object]]:
    failed_plot_families = failed_plot_families or set()
    output: list[dict[str, object]] = []
    for family in sorted(target.observed_families):
        observed = int(target.observed_families[family])
        evidence = observed_family_evidence(target, family)
        decoded_supported = not (
            family == "telemetry" and telemetry_state == "decoder_unavailable"
        )
        decoded_failed = (
            "target_execute" in failed_stages
            or "raw_reassembly" in failed_stages
            or "decoded" in failed_stages
            or "target_report" in failed_stages
            or (family == "telemetry" and telemetry_state == "decoder_broken")
        )
        both_writers_failed = (
            "hdf5" in failed_stages and "fits" in failed_stages
        )
        reader_failed = (
            decoded_failed
            or both_writers_failed
            or "reader" in failed_stages
            or "reader_evidence" in failed_stages
        )
        plotted_failed = (
            reader_failed
            or family in failed_plot_families
        )
        stages = {
            "observed_input": (observed, False, True),
            "decoded": (
                int(decoded_counts.get(family, 0)),
                decoded_failed,
                decoded_supported,
            ),
            "hdf5": (
                int(h5_counts.get(family, 0)),
                decoded_failed or "hdf5" in failed_stages,
                True,
            ),
            "fits": (
                int(fits_counts.get(family, 0)),
                decoded_failed or "fits" in failed_stages,
                True,
            ),
            "reader": (
                int(reader_counts.get(family, 0)),
                reader_failed,
                True,
            ),
            "plotted": (
                int(plotted_counts.get(family, 0)),
                plotted_failed,
                False,
            ),
        }
        for stage in COVERAGE_STAGES:
            count, failed, supported = stages[stage]
            state = coverage_state(
                observed,
                count,
                input_state=evidence.input_state,
                failed=failed,
                supported=supported,
            )
            if state not in COVERAGE_STATES:
                raise AssertionError(f"unexpected coverage state {state}")
            output.append({
                "record_type": "coverage",
                "target_id": target.target_id,
                "session_id": "all",
                "family": family,
                "stage": stage,
                "state": state,
                "count": count,
                "observed_input_state": evidence.input_state,
                "observed_unit": evidence.unit,
                "observed_basis": evidence.basis,
            })
    return output


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def scalar_text(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summary_text(value: object, units: str) -> str:
    metric = array_metric("", "", "", value, "hdf5", units=units)
    parts = [
        f"shape={tuple(metric['shape'])}",
        f"dtype={metric['dtype']}",
        f"units={units}",
        f"valid={metric['valid_count']}",
        f"missing={metric['missing_count']}",
    ]
    for name in ("minimum", "maximum", "mean", "magnitude_min", "magnitude_max", "magnitude_mean", "phase_min", "phase_max"):
        if name in metric:
            parts.append(f"{name}={scalar_text(metric[name])}")
    return "  ".join(parts)


def time_summary_text(value: object, label: str, units: str) -> str:
    """Summarize coverage and sampling gaps for one time coordinate."""

    metric = array_metric("", "", f"{label}/time", value, "hdf5", units=units)
    return "  ".join((
        f"{label}",
        f"units={units}",
        f"valid={metric['valid_count']}",
        f"missing={metric['missing_count']}",
        f"first={scalar_text(metric.get('time_minimum'))}",
        f"last={scalar_text(metric.get('time_maximum'))}",
        f"sampling_gap_median={scalar_text(metric.get('sampling_gap_median'))}",
        f"sampling_gap_maximum={scalar_text(metric.get('sampling_gap_maximum'))}",
    ))


def family_time_summary(
    sessions: Sequence[PlotSession],
    fields: Sequence[tuple[str, str, str]],
) -> str:
    """Summarize each available reader time field across selected sessions."""

    lines: list[str] = []
    for attribute, label, units in fields:
        arrays = [
            np.asarray(value).reshape(-1)
            for session in sessions
            if (value := getattr(session.bundle, attribute, None)) is not None
        ]
        lines.append(
            time_summary_text(np.concatenate(arrays), label, units)
            if arrays
            else f"{label}: unavailable; sampling gaps unavailable"
        )
    return "\n".join(lines)


def categorical_summary_text(value: object, units: str) -> str:
    """Summarize categorical/status values with counts and mode, never a mean."""

    array = np.asarray(value)
    statistics = categorical_statistics(array)
    counts = statistics["counts"]
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    shown = ",".join(f"{name}:{count}" for name, count in ordered[:12])
    if len(ordered) > 12:
        shown += f",...({len(ordered) - 12} more)"
    return "  ".join((
        f"shape={array.shape}",
        f"dtype={canonical_dtype(array)}",
        f"units={units}",
        f"valid={statistics['valid_count']}",
        f"missing={statistics['missing_count']}",
        f"mode={statistics['mode'] if statistics['mode'] is not None else 'n/a'}",
        f"mode_count={statistics['mode_count']}",
        f"counts={{{shown}}}",
    ))


def text_figure(title: str, header: Sequence[str], lines: Sequence[str]):
    plt = import_pyplot()
    figure = plt.figure(figsize=(11, 8.5))
    figure.suptitle(title, fontsize=10 if len(title) > 75 else 14, y=0.98)
    figure.text(0.04, 0.935, "\n".join(header), fontsize=7.2, va="top")
    columns = 3 if len(lines) > 70 else 2 if len(lines) > 35 else 1
    wrapped: list[str] = []
    for _ in range(3):
        wrap_width = {1: 140, 2: 76, 3: 48}[columns]
        wrapped = []
        for line in lines:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=wrap_width,
                    subsequent_indent="  ",
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]
            )
        next_columns = 3 if len(wrapped) > 70 else 2 if len(wrapped) > 35 else 1
        if next_columns == columns:
            break
        columns = next_columns
    block = max(1, math.ceil(len(wrapped) / columns))
    for column in range(columns):
        chunk = wrapped[column * block:(column + 1) * block]
        figure.text(
            0.04 + column * (0.92 / columns),
            0.86,
            "\n".join(chunk),
            family="monospace",
            fontsize=5.6 if columns == 3 else 6.4,
            va="top",
        )
    return figure


def source_identity(session: PlotSession, family: str, row: int) -> dict[str, object]:
    uid_attribute = {
        "tr": "tr_unique_ids",
        "zoom": "zoom_unique_ids",
        "grimm": "grimm_unique_ids",
    }.get(family)
    identity: dict[str, object] = {
        "session_id": session.session_id,
        "reader_row": row,
    }
    if uid_attribute is not None:
        values = getattr(session.bundle, uid_attribute, None)
        if values is not None and row < len(values):
            identity["unique_packet_id"] = int(np.asarray(values)[row])
    return identity


def selection_matches(candidate: Mapping[str, object], frozen: object) -> bool:
    if not isinstance(frozen, dict):
        return False
    if candidate.get("session_id") != frozen.get("session_id"):
        return False
    if "unique_packet_id" in frozen:
        return candidate.get("unique_packet_id") == frozen.get("unique_packet_id")
    return candidate.get("reader_row") == frozen.get("reader_row")


def unavailable_selection_page(
    target: TargetConfig, family: str, frozen: object
) -> FamilyPage:
    issue = issue_record(
        target.target_id,
        "all",
        "selection",
        "selection_unavailable",
        f"frozen {family} source is unavailable",
        details={"family": family, "requested": frozen},
    )
    figure = text_figure(
        f"{target.target_id}: {family}",
        ["Frozen source selection unavailable; no replacement was chosen."],
        [canonical_json(frozen)],
    )
    return FamilyPage(family, 0, frozen, figure, (issue,))


def normal_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
) -> FamilyPage | None:
    groups: dict[tuple[object, ...], list[tuple[PlotSession, np.ndarray]]] = {}
    for session in sessions:
        data = getattr(session.bundle, "spectra", None)
        if data is None:
            continue
        array = np.asarray(data)
        if array.ndim != 3 or array.shape[1] != 16:
            continue
        key = (
            array.shape[1:],
            str(getattr(session.bundle, "spectra_units", None)),
            str(getattr(session.bundle, "spectra_representation", None)),
        )
        groups.setdefault(key, []).append((session, array))
    if not groups:
        return None
    key = sorted(
        groups,
        key=lambda item: (-sum(array.shape[0] for _, array in groups[item]), str(item)),
    )[0]
    selected_group = groups[key]
    data = np.concatenate([array for _, array in selected_group], axis=0)
    issues: list[dict[str, object]] = []
    if len(groups) > 1:
        issues.append(issue_record(
            target.target_id,
            "all",
            "plot",
            "plot.mixed_normal_layout",
            "normal rows have multiple legacy shapes/representations; the largest compatible group is plotted",
        ))
    bundle = selected_group[0][0].bundle
    units = str(getattr(bundle, "spectra_units", None) or "SDU")
    navgf_values: list[int] = []
    frequency_arrays: list[np.ndarray] = []
    for session, array in selected_group:
        navgf = getattr(session.bundle, "spectra_metadata", {}).get("Navgf")
        if navgf is not None:
            navgf_values.extend(int(item) for item in np.asarray(navgf).reshape(-1))
        if session.frequency_mhz is not None:
            frequency_arrays.append(np.asarray(session.frequency_mhz))
    homogeneous_navgf = len(set(navgf_values)) <= 1
    frequency_ok = (
        homogeneous_navgf
        and frequency_arrays
        and all(array.shape == frequency_arrays[0].shape for array in frequency_arrays)
        and frequency_arrays[0].size == data.shape[-1]
        and all(np.array_equal(array, frequency_arrays[0]) for array in frequency_arrays[1:])
    )
    if frequency_ok:
        x = frequency_arrays[0]
        x_label = "legacy reader-derived MHz (FREQ-001 open)"
        short_x_label = "legacy MHz*"
    else:
        x = np.arange(data.shape[-1])
        x_label = "stored bin index (FREQ-001 open; no common verified MHz grid)"
        short_x_label = "stored bin index*"
        issues.append(issue_record(
            target.target_id,
            "all",
            "plot",
            "plot.frequency_axis_unresolved",
            "normal rows cannot share one verified MHz coordinate",
        ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(data, axis=0)
        low = np.nanpercentile(data, 10, axis=0)
        high = np.nanpercentile(data, 90, axis=0)
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 5, figsize=(11, 8.5), sharex=True)
    figure.subplots_adjust(top=0.84, bottom=0.28, hspace=0.38, wspace=0.28)
    products = [
        ("00", 0, None), ("11", 1, None), ("22", 2, None), ("33", 3, None),
        ("01", 4, 5), ("02", 6, 7), ("03", 8, 9),
        ("12", 10, 11), ("13", 12, 13), ("23", 14, 15),
    ]
    for axis, (label, real_index, imag_index) in zip(axes.reshape(-1), products):
        axis.plot(x, median[real_index], lw=0.7, label="auto" if imag_index is None else "real")
        axis.fill_between(x, low[real_index], high[real_index], alpha=0.18)
        if imag_index is not None:
            axis.plot(x, median[imag_index], lw=0.7, label="imag")
            axis.fill_between(x, low[imag_index], high[imag_index], alpha=0.12)
            axis.legend(fontsize=5, loc="best")
        axis.set_title(label, fontsize=8)
        axis.tick_params(labelsize=5)
    for axis in axes[-1]:
        axis.set_xlabel(short_x_label, fontsize=5.5)
    figure.suptitle(f"{target.target_id}: normal spectra", fontsize=13, y=0.98)
    figure.text(
        0.04,
        0.925,
        f"source={target.target_id}; sessions={','.join(item.session_id for item, _ in selected_group)}; "
        f"layout={getattr(bundle, 'layout_version', None)}; selected schema unavailable in layout-v3 reader; "
        f"representation={getattr(bundle, 'spectra_representation', None)}\n"
        f"aggregate rows={data.shape[0]}; bands=10--90 percent; {summary_text(data, units)}\n"
        + family_time_summary(
            [session for session, _ in selected_group],
            (
                ("spectra_raw_times", "raw time coverage", "raw seconds"),
                (
                    "spectra_mjd_times",
                    "assumed absolute MJD coverage (qualification adapter; TIME-004)",
                    "MJD days (assumed; TIME-004)",
                ),
            ),
        ),
        fontsize=6.2,
        va="top",
    )
    product_lines = []
    for index in range(16):
        metric = array_metric("", "", "", data[:, index], "hdf5", units=units)
        product_lines.append(
            f"p{index:02d} valid={metric['valid_count']:7d} missing={metric['missing_count']:7d} "
            f"min={scalar_text(metric.get('minimum')):>9} max={scalar_text(metric.get('maximum')):>9} "
            f"mean={scalar_text(metric.get('mean')):>9}"
        )
    figure.text(0.04, 0.22, f"* {x_label}", fontsize=5.2, va="top")
    figure.text(0.04, 0.19, "\n".join(product_lines[:8]), family="monospace", fontsize=5.1, va="top")
    figure.text(0.52, 0.19, "\n".join(product_lines[8:]), family="monospace", fontsize=5.1, va="top")
    return FamilyPage(
        "normal",
        int(data.shape[0]),
        {"mode": "all_valid_rows"},
        figure,
        tuple(issues),
    )


def row_candidates(
    sessions: Sequence[PlotSession], family: str, attribute: str
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for session in sorted(sessions, key=lambda item: item.session_id):
        value = getattr(session.bundle, attribute, None)
        if value is None:
            continue
        array = np.asarray(value)
        for row in range(array.shape[0]):
            identity = source_identity(session, family, row)
            candidates.append({
                **identity,
                "session": session,
                "data": array[row],
            })
    return candidates


def select_candidate(
    candidates: Sequence[Mapping[str, object]],
    frozen: object,
    sort_key: Callable[[Mapping[str, object]], object],
) -> Mapping[str, object] | None:
    if frozen is not None:
        return next(
            (candidate for candidate in candidates if selection_matches(candidate, frozen)),
            None,
        )
    return min(candidates, key=sort_key) if candidates else None


def heatmap_page(
    target: TargetConfig,
    family: str,
    candidate: Mapping[str, object],
    data: np.ndarray,
    units: str,
    detail: str,
) -> FamilyPage:
    plt = import_pyplot()
    figure, axis = plt.subplots(figsize=(11, 8.5))
    figure.subplots_adjust(top=0.82, bottom=0.12, left=0.1, right=0.92)
    image = axis.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
    figure.colorbar(image, ax=axis, shrink=0.8, label=units)
    axis.set_xlabel("native inner index (not physical frequency)")
    axis.set_ylabel("native accumulation index")
    identity = {name: candidate[name] for name in ("session_id", "reader_row", "unique_packet_id") if name in candidate}
    figure.suptitle(f"{target.target_id}: {family}", fontsize=13, y=0.98)
    figure.text(
        0.04,
        0.92,
        f"source={target.target_id}; identity={canonical_json(identity)}; selected schema unavailable in layout-v3 reader\n"
        f"{detail}; {summary_text(data, units)}",
        fontsize=6.4,
        va="top",
    )
    return FamilyPage(family, 1, identity, figure)


def tr_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    frozen: object,
) -> FamilyPage | None:
    candidates = row_candidates(sessions, "tr", "tr_spectra")
    for candidate in candidates:
        array = np.asarray(candidate["data"])
        candidate["valid_products"] = int(
            np.count_nonzero(np.any(np.isfinite(array).reshape(array.shape[0], -1), axis=1))
        )
    selected = select_candidate(
        candidates,
        frozen,
        lambda item: (-int(item["valid_products"]), str(item["session_id"]), int(item["reader_row"])),
    )
    if not candidates:
        return None
    if selected is None:
        return unavailable_selection_page(target, "tr", frozen)
    array = np.asarray(selected["data"])
    valid_products = np.flatnonzero(
        np.any(np.isfinite(array).reshape(array.shape[0], -1), axis=1)
    )
    if valid_products.size == 0:
        return unavailable_selection_page(target, "tr", frozen)
    product = int(valid_products[0])
    session = selected["session"]
    row = int(selected["reader_row"])
    navg2_values = getattr(session.bundle, "tr_navg2_per_sample", None)
    length_values = getattr(session.bundle, "tr_length_per_sample", None)
    navg2 = int(navg2_values[row]) if navg2_values is not None else array.shape[-2]
    length = int(length_values[row]) if length_values is not None else array.shape[-1]
    page = heatmap_page(
        target,
        "tr",
        selected,
        array[product, :navg2, :length],
        "raw count (UNITS-001 open)",
        f"product={product}; Navg2={navg2}; native length={length}; axes unestablished\n"
        + family_time_summary(
            sessions,
            (
                ("tr_raw_times", "raw time coverage", "raw seconds"),
                (
                    "tr_mjd_times",
                    "assumed absolute MJD coverage (qualification adapter; TIME-004)",
                    "MJD days (assumed; TIME-004)",
                ),
            ),
        ),
    )
    return FamilyPage(page.family, page.count, {**page.selection, "product": product}, page.figure)


def zoom_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    frozen: object,
) -> FamilyPage | None:
    candidates = [
        item for item in row_candidates(sessions, "zoom", "zoom_spectra")
        if np.asarray(item["data"]).shape == (4, 64)
        and np.any(np.isfinite(np.asarray(item["data"])))
    ]
    selected = select_candidate(
        candidates,
        frozen,
        lambda item: (str(item["session_id"]), int(item["reader_row"])),
    )
    if not candidates:
        return None
    if selected is None:
        return unavailable_selection_page(target, "zoom", frozen)
    array = np.asarray(selected["data"])
    session = selected["session"]
    row = int(selected["reader_row"])
    pfb_values = getattr(session.bundle, "zoom_pfb_indices", None)
    pfb_index = int(pfb_values[row]) if pfb_values is not None else None
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex=True)
    figure.subplots_adjust(top=0.82, hspace=0.35)
    for axis, values, label in zip(axes.reshape(-1), array, ("AA", "BB", "ABR", "ABI")):
        axis.plot(np.arange(values.size), values, lw=0.8)
        axis.set_title(label)
        axis.set_xlabel("native zoom bin")
        axis.set_ylabel("raw count")
    identity = {name: selected[name] for name in ("session_id", "reader_row", "unique_packet_id") if name in selected}
    figure.suptitle(f"{target.target_id}: zoom", fontsize=13, y=0.98)
    figure.text(
        0.04,
        0.92,
        f"source={target.target_id}; identity={canonical_json(identity)}; legacy pfb_index={pfb_index}; "
        "selected schema unavailable in layout-v3 reader\n"
        f"native components=(AA,BB,ABR,ABI); {summary_text(array, 'raw count (UNITS-001 open)')}\n"
        + family_time_summary(
            sessions,
            (
                ("zoom_raw_times", "raw time coverage", "raw seconds"),
                (
                    "zoom_mjd_times",
                    "assumed absolute MJD coverage (qualification adapter; TIME-004)",
                    "MJD days (assumed; TIME-004)",
                ),
            ),
        ),
        fontsize=6.4,
        va="top",
    )
    return FamilyPage("zoom", 1, identity, figure)


def waveform_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    frozen: object,
) -> FamilyPage | None:
    available = [session for session in sessions if getattr(session.bundle, "waveforms", {})]
    if not available:
        return None
    if frozen is not None:
        if not isinstance(frozen, dict):
            return unavailable_selection_page(target, "waveform", frozen)
        session = next((item for item in available if item.session_id == frozen.get("session_id")), None)
        if session is None:
            return unavailable_selection_page(target, "waveform", frozen)
    else:
        session = sorted(available, key=lambda item: item.session_id)[0]
    waveforms = getattr(session.bundle, "waveforms", {})
    times = getattr(session.bundle, "waveform_times", {})
    requested_times = frozen.get("raw_times", {}) if isinstance(frozen, dict) else {}
    rows: dict[int, int] = {}
    for channel in sorted(waveforms):
        row = 0
        if str(channel) in requested_times:
            candidates = np.flatnonzero(
                np.asarray(times.get(channel, ())) == requested_times[str(channel)]
            )
            if candidates.size == 0:
                return unavailable_selection_page(target, "waveform", frozen)
            row = int(candidates[0])
        if len(waveforms[channel]) > row:
            rows[channel] = row
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    figure.subplots_adjust(top=0.82, bottom=0.16, hspace=0.35)
    raw_times: dict[str, float] = {}
    for channel, axis in enumerate(axes.reshape(-1)):
        if channel not in rows:
            axis.text(0.5, 0.5, f"channel {channel} absent", ha="center", va="center")
            axis.set_axis_off()
            continue
        row = rows[channel]
        values = np.asarray(waveforms[channel][row])
        stride = max(1, values.size // 4096)
        axis.plot(np.arange(0, values.size, stride), values[::stride], lw=0.55)
        axis.set_title(f"channel {channel}")
        axis.set_xlabel("ADC sample index")
        axis.set_ylabel("raw ADC count")
        if channel in times and len(times[channel]) > row:
            raw_times[str(channel)] = float(times[channel][row])
    selection = {
        "session_id": session.session_id,
        "raw_times": raw_times,
        "reader_rows": {str(channel): row for channel, row in rows.items()},
    }
    waveform_time_arrays = [
        np.asarray(times[channel]).reshape(-1)
        for channel in sorted(times)
    ]
    time_detail = (
        time_summary_text(
            np.unique(np.concatenate(waveform_time_arrays)),
            "waveform raw time coverage",
            "raw ADC clock second (unmapped; not absolute time)",
        )
        if waveform_time_arrays
        else "waveform raw time coverage: unavailable; sampling gaps unavailable"
    )
    figure.suptitle(f"{target.target_id}: waveform", fontsize=13, y=0.98)
    figure.text(
        0.04,
        0.92,
        f"source={target.target_id}; selection={canonical_json(selection)}; layout={getattr(session.bundle, 'layout_version', None)}\n"
        "cross-channel metadata association and waveform UID are unavailable in layout v3; units=raw count (UNITS-001 open)\n"
        f"{time_detail}",
        fontsize=6.4,
        va="top",
    )
    summary_lines = [
        f"channel {channel}: {summary_text(waveforms[channel][row], 'raw ADC count')}"
        for channel, row in sorted(rows.items())
    ]
    figure.text(0.04, 0.075, "\n".join(summary_lines), family="monospace", fontsize=4.9, va="top")
    issue = issue_record(
        target.target_id,
        session.session_id,
        "plot",
        "plot.waveform_identity_unavailable",
        "layout-v3 reader does not preserve waveform UID or metadata association",
    )
    return FamilyPage("waveform", int(bool(rows)), selection, figure, (issue,))


def grimm_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    frozen: object,
) -> FamilyPage | None:
    candidates = [
        item for item in row_candidates(sessions, "grimm", "grimm_spectra")
        if np.any(np.isfinite(np.asarray(item["data"])))
    ]
    selected = select_candidate(
        candidates,
        frozen,
        lambda item: (str(item["session_id"]), int(item["reader_row"])),
    )
    if not candidates:
        return None
    if selected is None:
        return unavailable_selection_page(target, "grimm", frozen)
    array = np.asarray(selected["data"])
    panels: list[np.ndarray] | None = None
    if array.ndim == 3 and array.shape[0] == 4 and array.shape[-1] == 16:
        panels = [array[index] for index in range(4)]
    elif array.ndim == 3 and array.shape[-2:] == (16, 4):
        panels = [array[..., index] for index in range(4)]
    identity = {name: selected[name] for name in ("session_id", "reader_row", "unique_packet_id") if name in selected}
    time_detail = family_time_summary(
        sessions,
        (("grimm_raw_times", "raw time coverage", "raw seconds"),),
    )
    if panels is None:
        issue = issue_record(
            target.target_id,
            str(selected["session_id"]),
            "plot",
            "plot.grimm_layout_unsupported",
            f"native Grimm row shape {array.shape} is not a verified four-panel Navg2 x 16 layout",
        )
        figure = text_figure(
            f"{target.target_id}: Grimm",
            [
                f"identity={canonical_json(identity)}",
                "No reshape was attempted; axes and units remain unestablished (UNITS-001).",
                time_detail,
            ],
            [summary_text(array, "raw count (UNITS-001 open)")],
        )
        return FamilyPage("grimm", 1, identity, figure, (issue,))
    plt = import_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    figure.subplots_adjust(top=0.82, hspace=0.35)
    for index, (axis, panel) in enumerate(zip(axes.reshape(-1), panels)):
        image = axis.imshow(panel, aspect="auto", origin="lower", interpolation="nearest")
        figure.colorbar(image, ax=axis, shrink=0.7)
        axis.set_title(f"component {index} (labels unverified)")
        axis.set_xlabel("native 16-bin index")
        axis.set_ylabel("Navg2 index")
    figure.suptitle(f"{target.target_id}: Grimm", fontsize=13, y=0.98)
    figure.text(
        0.04,
        0.92,
        f"source={target.target_id}; identity={canonical_json(identity)}; axes are native indices, not frequency; "
        f"{summary_text(array, 'raw count (UNITS-001 open)')}\n{time_detail}",
        fontsize=6.4,
        va="top",
    )
    return FamilyPage("grimm", 1, identity, figure)


def telemetry_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    frozen: object,
    clock_adapter: BaselineClockAdapter,
) -> FamilyPage | None:
    available = [
        session for session in sessions
        if getattr(session.bundle, "dcb_fpga", {}) or getattr(session.bundle, "dcb_encoder", {})
    ]
    if not available:
        return None
    if frozen is not None:
        if not isinstance(frozen, dict):
            return unavailable_selection_page(target, "telemetry", frozen)
        session = next((item for item in available if item.session_id == frozen.get("session_id")), None)
        if session is None:
            return unavailable_selection_page(target, "telemetry", frozen)
    else:
        session = sorted(available, key=lambda item: item.session_id)[0]
    fpga = getattr(session.bundle, "dcb_fpga", {})
    encoder = getattr(session.bundle, "dcb_encoder", {})
    fpga_mission = np.asarray(fpga.get("mission_seconds", ()), dtype=np.float64)
    fpga_subsecs = np.asarray(
        fpga.get("lusee_subsecs", np.zeros(fpga_mission.shape)),
        dtype=np.float64,
    )
    if fpga_mission.size and fpga_subsecs.shape == fpga_mission.shape:
        fpga_elapsed = (
            fpga_mission + fpga_subsecs / 65536.0 - clock_adapter.dcb_raw_seconds
        )
        x = fpga_elapsed - fpga_elapsed[0]
        x_label = "seconds since first displayed DCB FPGA sample"
    else:
        longest = max((np.asarray(value).size for value in fpga.values()), default=0)
        x = np.arange(longest)
        x_label = "sample index; DCB FPGA time unavailable"

    time_details: list[str] = []
    for time_source, time_mapping in (("FPGA", fpga), ("encoder", encoder)):
        if not time_mapping:
            continue
        mission = np.asarray(
            time_mapping.get("mission_seconds", ()),
            dtype=np.float64,
        )
        subsecs = np.asarray(
            time_mapping.get("lusee_subsecs", np.zeros(mission.shape)),
            dtype=np.float64,
        )
        if mission.size and subsecs.shape == mission.shape:
            assumed_elapsed = (
                mission + subsecs / 65536.0 - clock_adapter.dcb_raw_seconds
            )
            time_details.append(time_summary_text(
                assumed_elapsed,
                f"assumed DCB {time_source} seconds from landing "
                "(display only; TIME-004)",
                "s",
            ))
        else:
            time_details.append(f"DCB {time_source} display time unavailable")
    if not time_details:
        time_details.append("DCB FPGA and encoder display time unavailable")
    groups = {
        "temperature-like fields": [name for name in fpga if "THERM" in name.upper() or name.upper().endswith("_T") or "TEMP" in name.upper()],
        "voltage-like fields": [name for name in fpga if "VMON" in name.upper() or name.upper().endswith("_V")],
        "current-like fields": [name for name in fpga if name.upper().endswith("_C") or name.upper().endswith("_A")],
    }
    plt = import_pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(11, 8.5))
    figure.subplots_adjust(top=0.79, bottom=0.53, wspace=0.3)
    for axis, (title, names) in zip(axes, groups.items()):
        for name in sorted(names):
            values = np.asarray(fpga[name])
            count = min(x.size, values.size)
            if count >= 2:
                axis.plot(x[:count], values[:count], lw=0.55, label=name)
        axis.set_title(title, fontsize=8)
        axis.set_xlabel(x_label, fontsize=6)
        axis.set_ylabel("unit unestablished", fontsize=6)
        if axis.lines:
            axis.legend(fontsize=3.7, ncol=2, loc="best")
        else:
            axis.text(0.5, 0.5, "fewer than two valid times", ha="center", va="center")
    field_lines = ["All public-reader fields; native uncropped statistics; units unestablished"]
    for name in sorted(fpga):
        field_path = f"telemetry/fpga/{semantic_name(name)}"
        detail = (
            categorical_summary_text(fpga[name], field_units(field_path))
            if categorical_field_path(field_path)
            else summary_text(fpga[name], field_units(field_path))
        )
        field_lines.append(
            f"FPGA {name}: {detail}"
        )
    field_lines.append("Encoder/status (TELEMETRY-003 open)")
    for name in sorted(encoder):
        field_path = f"telemetry/encoder/{semantic_name(name)}"
        detail = (
            categorical_summary_text(encoder[name], field_units(field_path))
            if categorical_field_path(field_path)
            else summary_text(encoder[name], field_units(field_path))
        )
        field_lines.append(
            f"ENC {name}: {detail}"
        )
    if not encoder:
        field_lines.append("ENC absent")
    wrapped_lines: list[str] = []
    for line in field_lines:
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=66,
                subsequent_indent="  ",
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
        )
    columns = min(4, max(1, math.ceil(len(wrapped_lines) / 28)))
    block = max(1, math.ceil(len(wrapped_lines) / columns))
    for column in range(columns):
        chunk = wrapped_lines[column * block:(column + 1) * block]
        figure.text(
            0.025 + column * (0.95 / columns),
            0.47,
            "\n".join(chunk),
            family="monospace",
            fontsize=3.7,
            va="top",
        )
    selection = {"session_id": session.session_id}
    figure.suptitle(f"{target.target_id}: telemetry", fontsize=13, y=0.98)
    header_lines: list[str] = []
    for line in (
        f"source={target.target_id}; selection={canonical_json(selection)}; "
        f"layout={getattr(session.bundle, 'layout_version', None)}; field-name "
        "groups are presentation-only; units unestablished (UNITS-001)",
        *time_details,
        "display equation: assumed_mjd=(mission_seconds+lusee_subsecs/65536-"
        "dcb_anchor)/86400+reference_mjd; ADC time unmapped",
    ):
        header_lines.extend(
            textwrap.wrap(
                line,
                width=178,
                subsequent_indent="  ",
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
        )
    figure.text(
        0.04,
        0.92,
        "\n".join(header_lines),
        fontsize=5.8,
        va="top",
    )
    count = max((np.asarray(value).size for value in fpga.values()), default=0)
    count = max(count, max((np.asarray(value).size for value in encoder.values()), default=0))
    return FamilyPage("telemetry", count, selection, figure)


def housekeeping_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
) -> FamilyPage | None:
    lines: list[str] = []
    total = 0
    for session in sorted(sessions, key=lambda item: item.session_id):
        for type_id, values in sorted(getattr(session.bundle, "housekeeping", {}).items()):
            record_count = max(
                (
                    int(np.asarray(value).shape[0])
                    if np.asarray(value).ndim
                    else 1
                    for value in values.values()
                ),
                default=0,
            )
            total += record_count
            for name, value in sorted(values.items()):
                array = np.asarray(value)
                field_path = f"housekeeping/type_{type_id}/{semantic_name(name)}"
                if time_field_path(field_path):
                    detail = time_summary_text(
                        array,
                        f"{name} coverage",
                        "raw seconds (native HK counter)",
                    )
                elif categorical_field_path(field_path):
                    detail = categorical_summary_text(
                        array,
                        "unit unestablished (UNITS-001 open)",
                    )
                else:
                    detail = summary_text(array, "unit unestablished (UNITS-001 open)")
                lines.append(f"{session.session_id} type={type_id} {name}: {detail}")
    if not lines:
        return None
    figure = text_figure(
        f"{target.target_id}: housekeeping",
        [
            f"source={target.target_id}; all public-reader fields; selected schema unavailable in layout v3",
            "Categorical/status fields use mode/count. Other fields retain native uncropped statistics; units unestablished (UNITS-001).",
        ],
        lines,
    )
    return FamilyPage("housekeeping", total, {"mode": "all_fields"}, figure)


def configuration_page(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
) -> FamilyPage | None:
    """Render public-reader invariants, constants, and scalar configuration."""

    lines: list[str] = []
    scalar_attributes = {
        "layout_version": "layout_version",
        "normal/units": "spectra_units",
        "normal/representation": "spectra_representation",
        "normal/normalization_version": "spectra_normalization_version",
    }
    for session in sorted(sessions, key=lambda item: item.session_id):
        values: dict[str, object] = {}
        for path, attribute in scalar_attributes.items():
            value = getattr(session.bundle, attribute, None)
            if value is not None:
                values[path] = value
        add_mapping(values, "session", getattr(session.bundle, "session_invariants", {}))
        add_mapping(values, "constants", getattr(session.bundle, "constants", {}))
        for field_path, value in sorted(values.items()):
            array = np.asarray(value)
            units = field_units(field_path, session.bundle)
            if array.dtype.kind in "SUO" or categorical_field_path(field_path):
                detail = categorical_summary_text(array, units)
            else:
                detail = summary_text(array, units)
            if array.size <= 12:
                detail += f"  values={canonical_json(array.tolist())}"
            lines.append(
                f"{session.session_id} {session.source_format} public-reader "
                f"{field_path}: {detail}"
            )
    if not lines:
        return None
    figure = text_figure(
        f"{target.target_id}: configuration and scalar evidence",
        [
            f"source={target.target_id}; provenance=canonical public reader "
            "(HDF5 preferred, FITS fallback)",
            "All exposed session invariants, constants, and scalar layout fields; no private paths.",
            "Absolute-time constants are qualification assumptions (TIME-004); native raw counters remain unshifted evidence.",
        ],
        lines,
    )
    return FamilyPage(
        "configuration",
        len(lines),
        {"mode": "all_public_reader_scalar_fields"},
        figure,
    )


def build_family_pages(
    target: TargetConfig,
    sessions: Sequence[PlotSession],
    requested: Mapping[str, object],
    clock_adapter: BaselineClockAdapter,
) -> tuple[list[FamilyPage], dict[str, object], list[dict[str, object]], set[str]]:
    builders: dict[str, Callable[[], FamilyPage | None]] = {
        "normal": lambda: normal_page(target, sessions),
        "tr": lambda: tr_page(target, sessions, requested.get(f"{target.target_id}/tr")),
        "zoom": lambda: zoom_page(target, sessions, requested.get(f"{target.target_id}/zoom")),
        "waveform": lambda: waveform_page(target, sessions, requested.get(f"{target.target_id}/waveform")),
        "grimm": lambda: grimm_page(target, sessions, requested.get(f"{target.target_id}/grimm")),
        "telemetry": lambda: telemetry_page(target, sessions, requested.get(f"{target.target_id}/telemetry"), clock_adapter),
        "housekeeping": lambda: housekeeping_page(target, sessions),
    }
    pages: list[FamilyPage] = []
    selections: dict[str, object] = {}
    issues: list[dict[str, object]] = []
    failed_families: set[str] = set()
    for family in FAMILIES:
        if family not in builders:
            continue
        try:
            page = builders[family]()
        except Exception as exc:  # noqa: BLE001
            failed_families.add(family)
            issue = issue_record(
                target.target_id,
                "all",
                "plot",
                f"stage_failed.{type(exc).__name__}",
                clean_message(exc, [target.source_path]),
                severity="error",
                details={"family": family},
            )
            issues.append(issue)
            pages.append(FamilyPage(
                family,
                0,
                None,
                text_figure(
                    f"{target.target_id}: {family}",
                    ["Plot construction failed; later families were still attempted."],
                    [issue["message"]],
                ),
                (issue,),
            ))
            continue
        if page is None:
            continue
        pages.append(page)
        issues.extend(dict(item) for item in page.issues)
        selections[f"{target.target_id}/{family}"] = page.selection
        if page.count == 0:
            failed_families.add(family)
    try:
        scalar_page = configuration_page(target, sessions)
    except Exception as exc:  # noqa: BLE001
        issue = issue_record(
            target.target_id,
            "all",
            "plot",
            f"stage_failed.{type(exc).__name__}",
            clean_message(exc, [target.source_path]),
            severity="error",
            details={"family": "configuration"},
        )
        issues.append(issue)
        pages.append(FamilyPage(
            "configuration",
            0,
            None,
            text_figure(
                f"{target.target_id}: configuration and scalar evidence",
                ["Table construction failed; product-family pages remain available."],
                [issue["message"]],
            ),
            (issue,),
        ))
    else:
        if scalar_page is not None:
            pages.append(scalar_page)
    return pages, selections, issues, failed_families


def make_coverage_figure(
    target: TargetConfig,
    coverage: Sequence[Mapping[str, object]],
    issues: Sequence[Mapping[str, object]],
):
    """Build a readable first page with compact stages and one evidence row."""

    plt = import_pyplot()
    figure = plt.figure(figsize=(11, 8.5))
    figure.suptitle(
        f"{target.target_id}: coverage",
        fontsize=10 if len(target.target_id) > 55 else 14,
        y=0.98,
    )
    figure.text(
        0.04,
        0.925,
        "observed input -> decoded -> HDF5 -> FITS -> public reader -> plotted",
        fontsize=8,
        va="top",
    )
    by_family = {
        family: sorted(
            (row for row in coverage if row.get("family") == family),
            key=lambda row: COVERAGE_STAGES.index(str(row["stage"])),
        )
        for family in FAMILIES
    }
    family_columns = (FAMILIES[:4], FAMILIES[4:])
    for column, families in enumerate(family_columns):
        lines = ["family / independent observed-input evidence", "  stage           state                 count"]
        for family in families:
            rows = by_family[family]
            if not rows:
                continue
            first = rows[0]
            evidence = (
                f"{family}: input={first.get('observed_input_state')} "
                f"observed={target.observed_families.get(family, 0)} "
                f"{first.get('observed_unit')}; basis={first.get('observed_basis')}"
            )
            lines.extend(
                textwrap.wrap(
                    evidence,
                    width=76,
                    subsequent_indent="  ",
                )
            )
            for row in rows:
                lines.append(
                    f"  {str(row['stage']):15} {str(row['state']):21} "
                    f"{int(row['count']):6d}"
                )
        figure.text(
            0.04 + 0.49 * column,
            0.86,
            "\n".join(lines),
            family="monospace",
            fontsize=5.1,
            va="top",
        )

    issue_counts: Counter[tuple[str, str]] = Counter()
    for record in issues:
        issue_counts[(
            str(record.get("stage", "")),
            str(record.get("code", "")),
        )] += int(record.get("occurrence_count", 1))
    issue_lines = [
        f"{count:5d}  {stage}: {code}"
        for (stage, code), count in sorted(issue_counts.items())
    ]
    shown_issue_lines = issue_lines[:20]
    if len(issue_lines) > len(shown_issue_lines):
        shown_issue_lines.append(
            f"... {len(issue_lines) - len(shown_issue_lines)} more issue codes"
        )
    figure.text(
        0.04,
        0.305,
        "Issue-code counts (full records in issues.jsonl)\n"
        + ("\n".join(shown_issue_lines) if shown_issue_lines else "none"),
        family="monospace",
        fontsize=5.2,
        va="top",
    )
    return figure


def write_target_pdf(
    path: Path,
    target: TargetConfig,
    coverage: Sequence[Mapping[str, object]],
    issues: Sequence[Mapping[str, object]],
    pages: Sequence[FamilyPage],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    first = make_coverage_figure(target, coverage, issues)
    figures = [first, *(page.figure for page in pages)]
    try:
        with PdfPages(path, metadata={"Title": target.target_id, "CreationDate": None, "ModDate": None}) as pdf:
            for figure in figures:
                pdf.savefig(figure)
    finally:
        plt = import_pyplot()
        for figure in figures:
            plt.close(figure)


def metric_sort_key(record: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(record.get(name, "")) for name in (
        "target_id",
        "session_id",
        "record_type",
        "family",
        "stage",
        "field_path",
        "status",
    ))


def issue_sort_key(record: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(record.get(name, "")) for name in (
        "target_id", "session_id", "stage", "code", "message"
    ))


def coalesce_issues(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    combined: dict[str, dict[str, object]] = {}
    for record in records:
        base = dict(record)
        base["occurrence_count"] = 1
        key = canonical_json(base)
        if key in combined:
            combined[key]["occurrence_count"] = int(combined[key]["occurrence_count"]) + 1
        else:
            combined[key] = base
    return sorted(combined.values(), key=issue_sort_key)


def read_requested_selections(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    manifest_path = path / "run_manifest.json" if path.is_dir() else path
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    selections = value.get("selections", {}) if isinstance(value, dict) else {}
    if not isinstance(selections, dict):
        raise ValueError("selection manifest has no selections object")
    return selections


def comparison_root(path: Path) -> Path:
    """Return the run directory for a compare-to directory or metrics file."""

    return path if path.is_dir() else path.parent


def verify_run_artifact(
    root: Path,
    value: object,
    name: str,
) -> tuple[Path, str]:
    """Verify one path-safe artifact entry from a run manifest."""

    if not isinstance(value, dict):
        raise ValueError(f"comparison manifest has no {name} artifact")
    relative_text = value.get("path")
    digest = value.get("sha256")
    if not isinstance(relative_text, str) or not relative_text:
        raise ValueError(f"comparison {name} artifact has no path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"comparison {name} artifact has no exact SHA-256")
    relative = Path(relative_text)
    if relative.is_absolute():
        raise ValueError(f"comparison {name} artifact path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"comparison {name} artifact escapes its run directory")
    if not path.is_file():
        raise ValueError(f"comparison {name} artifact is missing")
    if hash_file(path) != digest:
        raise ValueError(f"comparison {name} artifact digest mismatch")
    return path, digest


def resolve_comparison_files(
    path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    """Validate a previous run and return its metrics, issues, and manifest."""

    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        metrics_path = candidate / "metrics.jsonl"
    else:
        if candidate.name != "metrics.jsonl":
            raise ValueError("--compare-to file must be metrics.jsonl")
        metrics_path = candidate
    manifest_path = metrics_path.parent / "run_manifest.json"
    if not metrics_path.is_file():
        raise ValueError("--compare-to metrics.jsonl is missing")
    if not manifest_path.is_file():
        raise ValueError("--compare-to requires the sibling run_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("--compare-to run_manifest.json must contain an object")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("--compare-to run manifest has no artifacts object")
    verified_metrics, _ = verify_run_artifact(
        metrics_path.parent,
        artifacts.get("metrics"),
        "metrics",
    )
    if verified_metrics != metrics_path.resolve():
        raise ValueError("--compare-to path is not the manifest metrics artifact")
    verified_issues, _ = verify_run_artifact(
        metrics_path.parent,
        artifacts.get("issues"),
        "issues",
    )
    verify_run_artifact(
        metrics_path.parent,
        artifacts.get("summary_pdf"),
        "summary PDF",
    )
    target_reports = artifacts.get("target_reports")
    if not isinstance(target_reports, dict):
        raise ValueError("--compare-to run manifest has no target_reports artifacts")
    manifest_targets = manifest.get("targets")
    if not isinstance(manifest_targets, list):
        raise ValueError("--compare-to run manifest has no targets list")
    if set(target_reports) != set(str(item) for item in manifest_targets):
        raise ValueError("--compare-to target report artifacts do not match targets")
    for target_id, artifact in sorted(target_reports.items()):
        verify_run_artifact(
            metrics_path.parent,
            artifact,
            f"target report {target_id}",
        )
    return metrics_path, verified_issues, manifest_path, manifest


def read_comparison_selections(path: Path) -> dict[str, object]:
    """Read selections from the sibling run manifest, never from JSONL metrics."""

    _, _, _, manifest = resolve_comparison_files(path)
    selections = manifest.get("selections", {})
    if not isinstance(selections, dict):
        raise ValueError("comparison run manifest has no selections object")
    return selections


def comparison_run_reference(
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Extract exact, portable identity and artifact digests for one run."""

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("comparison run manifest has no artifacts object")
    target_reports = artifacts.get("target_reports")
    if not isinstance(target_reports, dict):
        raise ValueError("comparison run manifest has no target report artifacts")

    def artifact_reference(name: str) -> dict[str, object]:
        value = artifacts.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"comparison run manifest has no {name} artifact")
        return {
            "path": value.get("path"),
            "sha256": value.get("sha256"),
        }

    return {
        "run_id": manifest.get("run_id"),
        "subject_commit": manifest.get("subject_commit"),
        "report_tool_commit": manifest.get("report_tool_commit"),
        "report_source_digest": manifest.get("report_source_digest"),
        "ingest_source_digest": manifest.get("ingest_source_digest"),
        "config_digest": manifest.get("config_digest"),
        "source_commits": manifest.get("source_commits"),
        "landing_source_digest": (
            manifest.get("landing_reference", {}).get("source_digest")
            if isinstance(manifest.get("landing_reference"), dict)
            else None
        ),
        "corpus_manifest_digests": manifest.get("corpus_manifest_digests"),
        "run_manifest_sha256": hash_file(manifest_path),
        "metrics": artifact_reference("metrics"),
        "issues": artifact_reference("issues"),
        "summary_pdf": artifact_reference("summary_pdf"),
        "target_reports": {
            str(target_id): {
                "path": value.get("path"),
                "sha256": value.get("sha256"),
            }
            for target_id, value in sorted(target_reports.items())
            if isinstance(value, dict)
        },
    }


def metric_identity(record: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(record.get(name, "")) for name in (
        "record_type", "target_id", "session_id", "family", "stage", "field_path"
    ))


def compare_metric_sets(
    previous: Sequence[Mapping[str, object]],
    current: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    old = {
        metric_identity(record): record
        for record in previous
        if not str(record.get("record_type", "")).endswith("comparison")
    }
    new = {
        metric_identity(record): record
        for record in current
        if not str(record.get("record_type", "")).endswith("comparison")
    }
    output: list[dict[str, object]] = []
    for identity in sorted(set(old) | set(new)):
        if identity not in old:
            status = "added"
        elif identity not in new:
            status = "removed"
        else:
            status = "unchanged" if canonical_json(old[identity]) == canonical_json(new[identity]) else "changed"
        record: dict[str, object] = {
            "record_type": "comparison",
            "target_id": identity[1],
            "session_id": identity[2],
            "family": identity[3],
            "stage": identity[4],
            "field_path": identity[5],
            "status": status,
        }
        if identity in old:
            record["previous"] = old[identity]
        if identity in new:
            record["current"] = new[identity]
        output.append(record)
    return output


def compare_issue_sets(
    previous: Sequence[Mapping[str, object]],
    current: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    def counts(records: Sequence[Mapping[str, object]]) -> Counter[tuple[str, ...]]:
        result: Counter[tuple[str, ...]] = Counter()
        for record in records:
            identity = (
                str(record.get("target_id", "")),
                str(record.get("stage", "")),
                str(record.get("code", "")),
                str(record.get("severity", "")),
            )
            result[identity] += int(record.get("occurrence_count", 1))
        return result

    old = counts(previous)
    new = counts(current)
    output: list[dict[str, object]] = []
    for identity in sorted(set(old) | set(new)):
        before = old.get(identity, 0)
        after = new.get(identity, 0)
        output.append({
            "record_type": "issue_comparison",
            "target_id": identity[0],
            "stage": identity[1],
            "code": identity[2],
            "severity": identity[3],
            "previous_count": before,
            "current_count": after,
            "status": "unchanged" if before == after else "changed",
        })
    return output


def compare_selections(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for key in sorted(set(previous) | set(current)):
        if key not in previous:
            status = "added"
        elif key not in current:
            status = "removed"
        else:
            status = (
                "unchanged"
                if canonical_json(previous[key]) == canonical_json(current[key])
                else "changed"
            )
        target_id, _, family = key.partition("/")
        record: dict[str, object] = {
            "record_type": "selection_comparison",
            "target_id": target_id,
            "family": family,
            "field_path": key,
            "status": status,
        }
        if key in previous:
            record["previous"] = previous[key]
        if key in current:
            record["current"] = current[key]
        output.append(record)
    return output


def comparison_evidence(record: Mapping[str, object]) -> dict[str, object]:
    """Keep the numerical and semantic evidence needed in the comparison PDF."""

    names = (
        "record_type",
        "state",
        "count",
        "status",
        "shape",
        "dtype",
        "units",
        "valid_count",
        "missing_count",
        "counts",
        "mode",
        "mode_count",
        "minimum",
        "maximum",
        "mean",
        "magnitude_min",
        "magnitude_max",
        "magnitude_mean",
        "phase_min",
        "phase_max",
        "time_minimum",
        "time_maximum",
        "sampling_gap_median",
        "sampling_gap_maximum",
        "digest",
        "hdf5_shape",
        "fits_shape",
        "hdf5_dtype",
        "fits_dtype",
    )
    return {name: record[name] for name in names if name in record}


def build_comparison_summary_lines(
    metric_records: Sequence[Mapping[str, object]],
    issue_records: Sequence[Mapping[str, object]],
    selection_records: Sequence[Mapping[str, object]],
    targets: Sequence[TargetConfig],
    baseline_reference: Mapping[str, object],
    final_reference: Mapping[str, object],
) -> list[str]:
    """Build a human-readable baseline/final evidence index without private paths."""

    status_counts = Counter(str(record.get("status", "")) for record in metric_records)
    lines = [
        "Metric comparison summary",
        "  " + "  ".join(
            f"{status}={status_counts.get(status, 0)}"
            for status in ("added", "removed", "changed", "unchanged")
        ),
        "Changed metric evidence: presence/counts; shapes/dtypes; units/statistics/digests",
    ]
    changed_metrics = [
        record for record in metric_records
        if record.get("status") != "unchanged"
    ]
    if not changed_metrics:
        lines.append("no metric changes")
    for record in changed_metrics:
        identity = "/".join(
            str(record.get(name, ""))
            for name in ("target_id", "session_id", "family", "stage", "field_path")
        )
        lines.append(f"[{record.get('status')}] {identity}")
        if isinstance(record.get("previous"), dict):
            lines.append(
                "  baseline=" + canonical_json(
                    comparison_evidence(record["previous"])
                )
            )
        else:
            lines.append("  baseline=absent")
        if isinstance(record.get("current"), dict):
            lines.append(
                "  final=" + canonical_json(
                    comparison_evidence(record["current"])
                )
            )
        else:
            lines.append("  final=absent")

    lines.extend(("", "Issue-code count changes"))
    changed_issues = [
        record for record in issue_records
        if record.get("status") != "unchanged"
    ]
    if not changed_issues:
        lines.append("no issue-code count changes")
    for record in changed_issues:
        lines.append(
            f"[{record.get('status')}] {record.get('severity')} "
            f"{record.get('target_id')}/{record.get('stage')}/{record.get('code')}: "
            f"baseline={record.get('previous_count')} final={record.get('current_count')}"
        )

    lines.extend(("", "Frozen-selection changes"))
    changed_selections = [
        record for record in selection_records
        if record.get("status") != "unchanged"
    ]
    if not changed_selections:
        lines.append("no frozen-selection changes")
    for record in changed_selections:
        lines.append(
            f"[{record.get('status')}] {record.get('field_path')}: "
            f"baseline={canonical_json(record.get('previous')) if 'previous' in record else 'absent'}; "
            f"final={canonical_json(record.get('current')) if 'current' in record else 'absent'}"
        )
    lines.extend(("", "Exact run identities"))
    for label, reference in (
        ("baseline", baseline_reference),
        ("final", final_reference),
    ):
        lines.append(label)
        for name in (
            "run_id",
            "subject_commit",
            "report_tool_commit",
            "report_source_digest",
            "ingest_source_digest",
            "config_digest",
            "run_manifest_sha256",
            "landing_source_digest",
            "corpus_manifest_digests",
            "source_commits",
        ):
            lines.append(f"  {name}: {reference.get(name)}")
        for name in ("metrics", "issues", "summary_pdf"):
            artifact = reference.get(name)
            lines.append(
                f"  {name}: {canonical_json(artifact) if isinstance(artifact, dict) else 'missing'}"
            )
    lines.extend((
        "",
        "Report PDF references",
    ))
    baseline_summary = baseline_reference.get("summary_pdf")
    final_summary = final_reference.get("summary_pdf")
    lines.append(
        "baseline corpus PDF: <compare-to>/"
        f"{baseline_summary.get('path')} sha256={baseline_summary.get('sha256')}"
        if isinstance(baseline_summary, dict)
        else "baseline corpus PDF: missing"
    )
    lines.append(
        "final corpus PDF: ./"
        f"{final_summary.get('path')} sha256={final_summary.get('sha256')}"
        if isinstance(final_summary, dict)
        else "final corpus PDF: missing"
    )
    baseline_reports = baseline_reference.get("target_reports", {})
    final_reports = final_reference.get("target_reports", {})
    for target in sorted(targets, key=lambda item: item.target_id):
        baseline_artifact = (
            baseline_reports.get(target.target_id, {})
            if isinstance(baseline_reports, dict)
            else {}
        )
        final_artifact = (
            final_reports.get(target.target_id, {})
            if isinstance(final_reports, dict)
            else {}
        )
        lines.append(
            f"{target.target_id}: baseline <compare-to>/"
            f"{baseline_artifact.get('path')} sha256={baseline_artifact.get('sha256')}; "
            f"final ./{final_artifact.get('path')} sha256={final_artifact.get('sha256')}"
        )
    return lines


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    text = "".join(canonical_json(record) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def target_input_identity(
    target: TargetConfig,
    config: QualificationConfig,
    clock_adapter: BaselineClockAdapter,
    corpus_manifest_digests: Sequence[str],
    source_commits: Mapping[str, str],
    report_tool_commit: str | None,
    ingest_source_digest: str,
    dependency_snapshot: Mapping[str, str],
    requested_selections: Mapping[str, object],
) -> dict[str, object]:
    """Return portable content and code identity for run provenance."""

    return {
        "target_id": target.target_id,
        "target_kind": target.kind,
        "target_source_digest": hash_path(target.source_path),
        "telemetry_sidecar_digest": (
            hash_path(target.telemetry_sidecar)
            if target.telemetry_sidecar is not None
            else None
        ),
        "observed_families": dict(sorted(target.observed_families.items())),
        "observed_family_metadata": {
            family: {
                "unit": observed_family_evidence(target, family).unit,
                "basis": observed_family_evidence(target, family).basis,
                "input_state": observed_family_evidence(target, family).input_state,
            }
            for family in sorted(target.observed_families)
        },
        "landing_source_digest": clock_adapter.source_digest,
        "corpus_manifest_digests": sorted(corpus_manifest_digests),
        "subject_commit": config.subject_commit,
        "report_tool_commit": report_tool_commit,
        "ingest_source_digest": ingest_source_digest,
        "source_commits": dict(sorted(source_commits.items())),
        "expected_source_commits": dict(
            sorted(config.expected_source_commits.items())
        ),
        "dependencies": dict(sorted(dependency_snapshot.items())),
        "requested_selections": {
            key: requested_selections[key]
            for key in sorted(requested_selections)
            if key.startswith(f"{target.target_id}/")
        },
    }


def emergency_target_coverage(target: TargetConfig) -> list[dict[str, object]]:
    """Conservatively mark every downstream stage failed after an uncaught error."""

    records: list[dict[str, object]] = []
    for family in sorted(target.observed_families):
        observed = int(target.observed_families[family])
        evidence = observed_family_evidence(target, family)
        for stage in COVERAGE_STAGES:
            if evidence.input_state == "absent":
                state = "absent_in_input"
            elif stage == "observed_input":
                state = (
                    "present_empty"
                    if evidence.input_state == "present_empty"
                    else "present"
                )
            elif family == "calibrator" and stage in {"fits", "reader", "plotted"}:
                state = "unsupported"
            else:
                state = "stage_failed"
            records.append({
                "record_type": "coverage",
                "target_id": target.target_id,
                "session_id": "all",
                "family": family,
                "stage": stage,
                "state": state,
                "count": observed if stage == "observed_input" else 0,
                "observed_input_state": evidence.input_state,
                "observed_unit": evidence.unit,
                "observed_basis": evidence.basis,
            })
    return records


def write_summary_csv(path: Path, coverage: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "target_id",
                "session_id",
                "family",
                "stage",
                "state",
                "count",
                "observed_input_state",
                "observed_unit",
                "observed_basis",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sorted(coverage, key=metric_sort_key):
            writer.writerow({name: record.get(name, "") for name in writer.fieldnames})


def write_text_pdf(
    path: Path,
    title: str,
    lines: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(
            textwrap.wrap(
                line,
                width=132,
                subsequent_indent="  ",
                replace_whitespace=False,
                drop_whitespace=False,
            ) or [""]
        )
    page_size = 54
    pages = [
        wrapped[index:index + page_size]
        for index in range(0, len(wrapped), page_size)
    ] or [[]]
    with PdfPages(
        path,
        metadata={"Title": title, "CreationDate": None, "ModDate": None},
    ) as pdf:
        for page_number, page in enumerate(pages, start=1):
            figure = plt.figure(figsize=(11, 8.5))
            figure.text(0.04, 0.96, title, fontsize=14, va="top")
            figure.text(
                0.04,
                0.92,
                "\n".join(page),
                family="monospace",
                fontsize=7.5,
                va="top",
            )
            figure.text(0.96, 0.03, str(page_number), ha="right", fontsize=8)
            pdf.savefig(figure)
            plt.close(figure)


def repository_state() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def source_commit(module_name: str) -> str | None:
    """Return the Git commit containing an imported dependency, if available."""

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin).resolve()
    try:
        root = subprocess.run(
            ["git", "-C", str(origin.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def verify_source_commits(expected: Mapping[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, required in sorted(expected.items()):
        commit = source_commit(name)
        if commit is None:
            raise RuntimeError(f"cannot verify source commit for {name}")
        if commit != required:
            raise RuntimeError(
                f"source commit mismatch for {name}: expected {required}, got {commit}"
            )
        observed[name] = commit
    return observed


def verify_subject_commit(subject_commit: str) -> None:
    root = Path(__file__).resolve().parents[1]
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{subject_commit}^{{commit}}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", subject_commit, "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "subject_commit is unavailable or is not an ancestor of the report tool"
        ) from exc


def dependency_versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("astropy", "h5py", "matplotlib", "lusee", "uncrater", "lusee_telemetry"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def summarize_reader_semantics(
    target_id: str,
    session_id: str,
    h5_view: ReaderView | None,
    fits_view: ReaderView | None,
    clock_adapter: BaselineClockAdapter,
) -> ReaderSemanticSummary:
    """Collect one session's reader metrics without mutating target state."""

    canonical_view = h5_view or fits_view
    if canonical_view is None:
        raise ValueError("no public-reader bundle is available")
    h5_semantics = bundle_semantics(
        h5_view.bundle,
        h5_view.frequency_mhz,
        clock_adapter,
    ) if h5_view else {}
    fits_semantics = bundle_semantics(
        fits_view.bundle,
        fits_view.frequency_mhz,
        clock_adapter,
    ) if fits_view else {}
    canonical = h5_semantics or fits_semantics
    source_format = "hdf5" if h5_semantics else "fits"
    metrics = [
        array_metric(
            target_id,
            session_id,
            field_path,
            canonical[field_path],
            source_format,
            units=field_units(field_path, canonical_view.bundle),
        )
        for field_path in sorted(canonical)
    ]
    issues: list[dict[str, object]] = []
    if h5_semantics and fits_semantics:
        parity = compare_semantics(
            h5_semantics,
            fits_semantics,
            target_id=target_id,
            session_id=session_id,
        )
        metrics.extend(parity)
        for record in parity:
            if record["status"] != "equal":
                issues.append(issue_record(
                    target_id,
                    session_id,
                    "parity",
                    f"parity.{record['status']}",
                    f"reader parity differs at {record['field_path']}",
                    details={"field_path": record["field_path"]},
                ))
    return ReaderSemanticSummary(
        tuple(metrics),
        tuple(issues),
    )


def process_target_report(
    target: TargetConfig,
    target_dir: Path,
    target_work: Path,
    executor: TargetExecutor,
    clock_adapter: BaselineClockAdapter,
    requested: Mapping[str, object],
    output: Path,
) -> TargetOutcome:
    """Run and summarize one target inside the outer corpus failure boundary."""

    private_paths = [target.source_path, target_work, output]
    if target.telemetry_sidecar is not None:
        private_paths.append(target.telemetry_sidecar)
    ok, artifacts_value, found = capture_call(
        lambda: executor(target, target_work, clock_adapter),
        target_id=target.target_id,
        session_id="target",
        stage="target_execute",
        private_paths=private_paths,
    )
    artifacts = list(artifacts_value or ()) if ok else []
    if not artifacts:
        artifacts = [SessionArtifacts(
            target_id=target.target_id,
            session_id="target",
            decoded_summary=empty_family_counts(),
            h5_path=None,
            fits_path=None,
        )]

    target_issues: list[dict[str, object]] = [dict(item) for item in found]
    target_metrics: list[dict[str, object]] = []
    decoded_total = empty_family_counts()
    h5_total = empty_family_counts()
    fits_total = empty_family_counts()
    reader_total = empty_family_counts()
    failed_stages = {
        str(item.get("stage"))
        for item in found
        if str(item.get("code", "")).startswith("stage_failed.")
    }
    telemetry_states: list[str] = []
    plot_sessions: list[PlotSession] = []
    target_layout_versions: set[str] = set()
    target_source_formats: set[str] = set()
    any_reader_attempt = False
    any_reader_success = False
    for artifact in artifacts:
        target_issues.extend(dict(item) for item in artifact.issues)
        add_family_counts(decoded_total, artifact.decoded_summary)
        telemetry_states.append(artifact.telemetry_state)
        failed_stages.update(
            str(item.get("stage"))
            for item in artifact.issues
            if str(item.get("code", "")).startswith("stage_failed.")
        )
        h5_view: ReaderView | None = None
        fits_view: ReaderView | None = None

        if artifact.h5_path is not None:
            any_reader_attempt = True
            ok_reader, value, found = capture_call(
                lambda path=artifact.h5_path: read_public_bundle(path, "h5"),
                target_id=target.target_id,
                session_id=artifact.session_id,
                stage="reader_hdf5",
                private_paths=private_paths,
            )
            target_issues.extend(found)
            h5_view = value if ok_reader else None
            any_reader_success = any_reader_success or ok_reader
            if not ok_reader:
                failed_stages.add("hdf5")
                failed_stages.add("reader_hdf5")
            else:
                ok_counts, value, found = capture_call(
                    lambda: family_counts_from_bundle(h5_view.bundle),
                    target_id=target.target_id,
                    session_id=artifact.session_id,
                    stage="hdf5_inspect",
                    private_paths=private_paths,
                )
                target_issues.extend(found)
                if ok_counts:
                    add_family_counts(h5_total, value)
                else:
                    failed_stages.add("hdf5")
        if artifact.fits_path is not None:
            any_reader_attempt = True
            ok_reader, value, found = capture_call(
                lambda path=artifact.fits_path: read_public_bundle(path, "fits"),
                target_id=target.target_id,
                session_id=artifact.session_id,
                stage="reader_fits",
                private_paths=private_paths,
            )
            target_issues.extend(found)
            fits_view = value if ok_reader else None
            any_reader_success = any_reader_success or ok_reader
            if not ok_reader:
                failed_stages.add("fits")
                failed_stages.add("reader_fits")
            else:
                ok_counts, value, found = capture_call(
                    lambda: family_counts_from_bundle(fits_view.bundle),
                    target_id=target.target_id,
                    session_id=artifact.session_id,
                    stage="fits_inspect",
                    private_paths=private_paths,
                )
                target_issues.extend(found)
                if ok_counts:
                    add_family_counts(fits_total, value)
                else:
                    failed_stages.add("fits")

        canonical_view = h5_view or fits_view
        if canonical_view is not None:
            source_format: Literal["hdf5", "fits"] = (
                "hdf5" if h5_view is not None else "fits"
            )
            target_layout_versions.add(
                str(getattr(canonical_view.bundle, "layout_version", None))
            )
            target_source_formats.add(source_format)
            plot_sessions.append(PlotSession(
                artifact.session_id,
                canonical_view.bundle,
                canonical_view.frequency_mhz,
                source_format,
            ))
            ok_counts, value, found = capture_call(
                lambda: family_counts_from_bundle(canonical_view.bundle),
                target_id=target.target_id,
                session_id=artifact.session_id,
                stage="reader_evidence",
                private_paths=private_paths,
            )
            target_issues.extend(found)
            if ok_counts:
                add_family_counts(reader_total, value)
            else:
                failed_stages.add("reader_evidence")
            ok_semantic, value, found = capture_call(
                lambda: summarize_reader_semantics(
                    target.target_id,
                    artifact.session_id,
                    h5_view,
                    fits_view,
                    clock_adapter,
                ),
                target_id=target.target_id,
                session_id=artifact.session_id,
                stage="semantic",
                private_paths=private_paths,
            )
            target_issues.extend(found)
            if not ok_semantic:
                failed_stages.add("semantic")
                continue
            summary: ReaderSemanticSummary = value
            target_metrics.extend(dict(item) for item in summary.metrics)
            target_issues.extend(dict(item) for item in summary.issues)

    if any_reader_attempt and not any_reader_success:
        failed_stages.add("reader")
    pages, target_selections, plot_issues, failed_plot_families = (
        build_family_pages(
            target,
            plot_sessions,
            requested,
            clock_adapter,
        )
    )
    if not plot_sessions:
        failed_plot_families.update(
            family
            for family in FAMILIES
            if family != "calibrator" and reader_total.get(family, 0) > 0
        )
    target_issues.extend(plot_issues)
    plotted_total = empty_family_counts()
    for page in pages:
        if page.family in plotted_total and page.family not in failed_plot_families:
            plotted_total[page.family] += int(page.count)

    telemetry_state = "absent"
    for candidate_state in (
        "decoded",
        "decoder_broken",
        "decoder_unavailable",
        "present_empty",
    ):
        if candidate_state in telemetry_states:
            telemetry_state = candidate_state
            break
    target_coverage = build_target_coverage(
        target,
        decoded_total,
        h5_total,
        fits_total,
        reader_total,
        plotted_total,
        failed_stages,
        telemetry_state=telemetry_state,
        failed_plot_families=failed_plot_families,
    )
    target_issues.extend(coverage_consistency_issues(target, target_coverage))

    write_target_pdf(
        target_dir / "report.pdf",
        target,
        target_coverage,
        target_issues,
        pages,
    )
    return TargetOutcome(
        tuple(sorted(target_metrics, key=metric_sort_key)),
        tuple(sorted(target_issues, key=issue_sort_key)),
        tuple(sorted(target_coverage, key=metric_sort_key)),
        dict(sorted(target_selections.items())),
        tuple(sorted(target_layout_versions)),
        tuple(sorted(target_source_formats)),
    )


def run_qualification(
    config: QualificationConfig,
    output_dir: Path | str,
    *,
    selection_manifest: Path | None = None,
    compare_to: Path | None = None,
    target_executor: TargetExecutor | None = None,
) -> QualificationResult:
    """Attempt every configured target and write deterministic reports."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty")
    output.mkdir(parents=True, exist_ok=True)
    clock_adapter = load_baseline_clock_adapter(config)
    executor = target_executor or execute_target_default
    previous_metrics_path: Path | None = None
    previous_issues_path: Path | None = None
    previous_manifest_path: Path | None = None
    previous_manifest: dict[str, object] = {}
    if compare_to is not None:
        (
            previous_metrics_path,
            previous_issues_path,
            previous_manifest_path,
            previous_manifest,
        ) = resolve_comparison_files(compare_to)
    requested = read_requested_selections(selection_manifest)
    if compare_to is not None and not requested:
        requested = read_comparison_selections(compare_to)

    metrics: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    attempted: list[str] = []
    reader_layout_versions: set[str] = set()
    reader_source_formats: set[str] = set()

    verified_source_commits: dict[str, str] = {}
    if target_executor is None:
        verify_subject_commit(config.subject_commit)
        verified_source_commits = verify_source_commits(
            config.expected_source_commits
        )

    report_tool_commit, tracked_dirty = repository_state()
    dependency_snapshot = dependency_versions()
    ingest_source_digest = hash_python_source_tree(
        Path(__file__).resolve().parents[1] / "lusee" / "ingest"
    )
    corpus_manifest_digests = sorted(
        hash_file(path) for path in config.corpus_manifest_paths
    )
    target_input_identities: dict[str, dict[str, object]] = {}

    with tempfile.TemporaryDirectory(prefix="lusee-ingest-report-") as temp_name:
        temp_root = Path(temp_name)
        for target in sorted(config.targets, key=lambda item: item.target_id):
            attempted.append(target.target_id)
            target_dir = (
                output
                / ("trees" if target.kind == "cdi" else "raw")
                / target.target_id
            )
            input_identity = target_input_identity(
                target,
                config,
                clock_adapter,
                corpus_manifest_digests,
                verified_source_commits,
                report_tool_commit,
                ingest_source_digest,
                dependency_snapshot,
                requested,
            )
            target_input_identities[target.target_id] = input_identity
            target_work = temp_root / target.target_id
            private_paths = [target.source_path, target_work, output]
            if target.telemetry_sidecar is not None:
                private_paths.append(target.telemetry_sidecar)
            ok_target, outcome_value, boundary_issues = capture_call(
                lambda target=target, target_dir=target_dir, target_work=target_work: process_target_report(
                    target,
                    target_dir,
                    target_work,
                    executor,
                    clock_adapter,
                    requested,
                    output,
                ),
                target_id=target.target_id,
                session_id="all",
                stage="target_report",
                private_paths=private_paths,
            )
            if ok_target:
                outcome = outcome_value
                target_metrics = [dict(item) for item in outcome.metrics]
                target_issues = [dict(item) for item in outcome.issues]
                target_coverage = [dict(item) for item in outcome.coverage]
                target_selections = dict(outcome.selections)
                target_layout_versions = set(outcome.reader_layout_versions)
                target_source_formats = set(outcome.reader_source_formats)
                target_issues.extend(boundary_issues)
            else:
                target_metrics = []
                target_issues = [dict(item) for item in boundary_issues]
                target_coverage = emergency_target_coverage(target)
                target_selections = {}
                target_layout_versions = set()
                target_source_formats = set()
                try:
                    import_pyplot().close("all")
                except Exception:  # noqa: BLE001
                    pass
                ok_fallback, _, fallback_issues = capture_call(
                    lambda: write_text_pdf(
                        target_dir / "report.pdf",
                        f"{target.target_id}: target report failure",
                        [
                            "The target-level report boundary caught a failure.",
                            "Later targets were still attempted.",
                            "",
                            *(
                                f"{item.get('stage')}: {item.get('code')}: {item.get('message')}"
                                for item in target_issues
                            ),
                        ],
                    ),
                    target_id=target.target_id,
                    session_id="all",
                    stage="pdf_fallback",
                    private_paths=private_paths,
                )
                target_issues.extend(fallback_issues)
                if not ok_fallback:
                    target_coverage = emergency_target_coverage(target)

            metrics.extend(target_metrics)
            coverage.extend(target_coverage)
            issues.extend(target_issues)
            selections.update(target_selections)
            reader_layout_versions.update(target_layout_versions)
            reader_source_formats.update(target_source_formats)

    target_report_artifacts: dict[str, dict[str, str]] = {}
    for target in sorted(config.targets, key=lambda item: item.target_id):
        report_path = (
            output
            / ("trees" if target.kind == "cdi" else "raw")
            / target.target_id
            / "report.pdf"
        )
        if not report_path.is_file():
            raise RuntimeError(
                f"target report is missing after full corpus attempt: {target.target_id}"
            )
        target_report_artifacts[target.target_id] = {
            "path": report_path.relative_to(output).as_posix(),
            "sha256": hash_file(report_path),
        }

    metrics.extend(coverage)
    issues_final = coalesce_issues(issues)
    metric_comparisons: list[dict[str, object]] = []
    issue_comparisons: list[dict[str, object]] = []
    selection_comparisons: list[dict[str, object]] = []
    if compare_to is not None:
        if previous_metrics_path is None:
            raise AssertionError("comparison metrics were not resolved")
        previous = [
            json.loads(line)
            for line in previous_metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metric_comparisons = compare_metric_sets(previous, metrics)
        metrics.extend(metric_comparisons)
        if previous_issues_path is None:
            raise AssertionError("comparison issues were not resolved")
        previous_issues = [
            json.loads(line)
            for line in previous_issues_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        issue_comparisons = compare_issue_sets(previous_issues, issues_final)
        metrics.extend(issue_comparisons)
        previous_selections = previous_manifest.get("selections", {})
        if not isinstance(previous_selections, dict):
            raise ValueError("comparison run manifest has no selections object")
        selection_comparisons = compare_selections(
            previous_selections,
            selections,
        )
        metrics.extend(selection_comparisons)
    metrics = sorted(metrics, key=metric_sort_key)

    metrics_path = output / "metrics.jsonl"
    issues_path = output / "issues.jsonl"
    csv_path = output / "corpus_summary.csv"
    write_jsonl(metrics_path, metrics)
    write_jsonl(issues_path, issues_final)
    write_summary_csv(csv_path, coverage)

    summary_counts = Counter(str(row["state"]) for row in coverage)
    summary_lines = [f"run_id: {config.run_id}", f"targets attempted: {len(attempted)}", ""]
    summary_lines.extend(
        f"{state:24} {summary_counts.get(state, 0):6d}"
        for state in sorted(COVERAGE_STATES)
    )
    summary_lines.extend(("", f"issues: {len(issues_final)}"))
    summary_pdf_path = output / "corpus_summary.pdf"
    write_text_pdf(summary_pdf_path, "lusee.ingest qualification", summary_lines)

    report_source_digest = hash_file(Path(__file__).resolve())
    current_reference = {
        "run_id": config.run_id,
        "subject_commit": config.subject_commit,
        "report_tool_commit": report_tool_commit,
        "report_source_digest": report_source_digest,
        "ingest_source_digest": ingest_source_digest,
        "config_digest": config.config_digest,
        "source_commits": verified_source_commits,
        "landing_source_digest": clock_adapter.source_digest,
        "corpus_manifest_digests": corpus_manifest_digests,
        "run_manifest_sha256": None,
        "metrics": {"path": metrics_path.name, "sha256": hash_file(metrics_path)},
        "issues": {"path": issues_path.name, "sha256": hash_file(issues_path)},
        "summary_pdf": {
            "path": summary_pdf_path.name,
            "sha256": hash_file(summary_pdf_path),
        },
        "target_reports": target_report_artifacts,
    }

    comparison_pdf_path: Path | None = None
    baseline_reference: dict[str, object] | None = None
    if compare_to is not None:
        if previous_manifest_path is None:
            raise AssertionError("comparison manifest was not resolved")
        baseline_reference = comparison_run_reference(
            previous_manifest_path,
            previous_manifest,
        )
        comparison_pdf_path = output / "baseline_final_comparison.pdf"
        write_text_pdf(
            comparison_pdf_path,
            "lusee.ingest baseline/final comparison",
            build_comparison_summary_lines(
                metric_comparisons,
                issue_comparisons,
                selection_comparisons,
                config.targets,
                baseline_reference,
                current_reference,
            ),
        )

    status: Literal["complete", "complete_with_issues", "failed"] = (
        "complete_with_issues" if issues_final else "complete"
    )
    manifest = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "status": status,
        "subject_commit": config.subject_commit,
        "subject_commit_verified": target_executor is None,
        "report_tool_commit": report_tool_commit,
        "report_source_digest": report_source_digest,
        "tracked_dirty": tracked_dirty,
        "ingest_source_digest": ingest_source_digest,
        "config_digest": config.config_digest,
        "landing_reference": {
            "adapter": "baseline_legacy_clock_adapter",
            "equation": "mjd=(raw-clock_reference_raw_seconds)/86400+reference_mjd",
            "source_digest": clock_adapter.source_digest,
            "reference_isot": clock_adapter.reference_isot,
            "time_scale": clock_adapter.time_scale,
            "spectrometer_clock_source": clock_adapter.spectrometer_clock_source,
            "spectrometer_raw_seconds": clock_adapter.spectrometer_raw_seconds,
            "dcb_clock_source": clock_adapter.dcb_clock_source,
            "dcb_raw_seconds": clock_adapter.dcb_raw_seconds,
            "dcb_display_equation": "assumed_mjd=(mission_seconds+lusee_subsecs/65536-dcb_reference_raw_seconds)/86400+reference_mjd",
            "dcb_display_only": True,
            "adc_clock_mapped": False,
            "assumed": clock_adapter.assumed,
        },
        "corpus_manifest_digests": corpus_manifest_digests,
        "target_input_digests": {
            target_id: hashlib.sha256(
                canonical_json(identity).encode("utf-8")
            ).hexdigest()
            for target_id, identity in sorted(target_input_identities.items())
        },
        "dependencies": dependency_snapshot,
        "source_commits": verified_source_commits,
        "targets": attempted,
        "reassembly_profile": "legacy",
        "semantic_options": {
            "plot_source_policy": "hdf5_public_reader_preferred_fits_fallback",
            "actual_plot_source_formats": sorted(reader_source_formats),
            "telemetry_interpolation": False,
            "frequency_coordinate": "legacy_reader_derived_or_stored_bin_index",
            "reader_layout_versions": sorted(reader_layout_versions),
            "selected_decoder_schema": "unavailable_in_layout_v3_reader",
        },
        "selections": selections,
        "artifacts": {
            "metrics": current_reference["metrics"],
            "issues": current_reference["issues"],
            "summary_csv": {"path": "corpus_summary.csv", "sha256": hash_file(csv_path)},
            "summary_pdf": current_reference["summary_pdf"],
            "target_reports": target_report_artifacts,
        },
    }
    if comparison_pdf_path is not None:
        manifest["artifacts"]["comparison_pdf"] = {
            "path": comparison_pdf_path.name,
            "sha256": hash_file(comparison_pdf_path),
            "baseline_corpus_pdf": "<compare-to>/corpus_summary.pdf",
            "final_corpus_pdf": "./corpus_summary.pdf",
        }
        manifest["comparison_baseline_reference"] = baseline_reference
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return QualificationResult(status, tuple(attempted), output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a private lusee.ingest qualification report"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_qualification(
            load_config(args.config),
            args.output_dir,
            selection_manifest=args.selection_manifest,
            compare_to=args.compare_to,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"qualification failed: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1
    print(f"output: {result.output_dir}")
    print(f"manifest: {result.output_dir / 'run_manifest.json'}")
    print(f"status: {result.status}")
    return 0 if result.status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
