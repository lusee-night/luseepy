"""Command-line interface for the LuSEE downlink ingest pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from . import pipeline
from .issues import IssueCollector, IssuePolicy


EXIT_CLEAN = 0
EXIT_FAILED = 1
EXIT_PARTIAL = 2
VALID_STATUSES = ("clean", "partial", "failed")


def status_exit_code(status: str) -> int:
    """Map one final ingest status to the command exit contract."""
    if status == "clean":
        return EXIT_CLEAN
    if status == "partial":
        return EXIT_PARTIAL
    if status == "failed":
        return EXIT_FAILED
    raise ValueError(f"unknown ingest status {status!r}")


def print_issue_summary(
    issue_counts: dict[str, int],
    status_issue_codes: Sequence[str],
) -> None:
    if issue_counts:
        summary = ", ".join(
            f"{code}={count}" for code, count in sorted(issue_counts.items())
        )
    elif status_issue_codes:
        summary = ", ".join(sorted(status_issue_codes))
    else:
        summary = "none"
    print(f"issues: {summary}")


def print_session_result(result: pipeline.SessionResult) -> None:
    print(f"status: {result.status}")
    print(f"session: {result.session_name}")
    for label, path in (
        ("session_directory", result.session_dir),
        ("hdf5", result.h5_path),
        ("fits", result.fits_path),
        ("manifest", result.manifest_path),
    ):
        if path is not None:
            print(f"{label}: {path}")
    for path in result.plot_paths:
        print(f"plot: {path}")
    print_issue_summary(result.issue_counts, result.status_issue_codes)


def print_flash_result(result: pipeline.FlashResult) -> None:
    print(f"status: {result.status}")
    for path in result.manifest_paths:
        print(f"manifest: {path}")
    for session in result.session_results:
        print(f"session[{session.session_name}]: {session.status}")
        for label, path in (
            ("directory", session.session_dir),
            ("hdf5", session.h5_path),
            ("fits", session.fits_path),
            ("manifest", session.manifest_path),
        ):
            if path is not None:
                print(f"session[{session.session_name}].{label}: {path}")
        for path in session.plot_paths:
            print(f"session[{session.session_name}].plot: {path}")
    print_issue_summary(result.issue_counts, result.status_issue_codes)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="ascii") as stream:
        document = json.load(
            stream,
            object_pairs_hook=object_without_duplicate_keys,
            parse_constant=reject_json_constant,
        )
    if not isinstance(document, dict):
        raise ValueError("manifest must be a JSON object")
    if document.get("manifest_schema_version") != pipeline.MANIFEST_SCHEMA_VERSION:
        raise ValueError("manifest has unsupported schema version")
    if document.get("status") not in VALID_STATUSES:
        raise ValueError("manifest has invalid status")
    return document


def validate_manifest(
    path: Path,
) -> tuple[str, str]:
    document = load_manifest(path)
    kind = document.get("manifest_kind")
    if kind == "session":
        if path.name != pipeline.IN_SESSION_MANIFEST_NAME:
            raise ValueError(
                "session validation requires the in-session session.json"
            )
        if document.get("flash_manifest_sha256") is not None and not (
            path.parent.parent / pipeline.FLASH_MANIFEST_NAME
        ).is_file():
            raise ValueError("linked FLASH manifest is missing")
        if pipeline._read_in_session_manifest(path.parent, strict=True) is None:
            raise ValueError("session manifest is missing")
        return "session manifest", str(document["status"])
    if kind != "flash":
        raise ValueError("manifest has invalid manifest_kind")
    if path.name != pipeline.FLASH_MANIFEST_NAME:
        raise ValueError("FLASH validation requires the canonical flash.json")

    expected_locator_contract = {
        "version": 1,
        "base": "sessions_root",
        "canonical_manifest": pipeline.FLASH_MANIFEST_NAME,
        "external_copies_are_mirrors": True,
    }
    if document.get("locator_contract") != expected_locator_contract:
        raise ValueError("FLASH manifest has invalid locator contract")
    sessions = document.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("FLASH manifest sessions must be a list")

    root = path.parent.resolve()
    manifest_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    seen_sessions = set()
    session_statuses = []
    for record in sessions:
        if not isinstance(record, dict):
            raise ValueError("FLASH session records must be JSON objects")
        identity = (record.get("session_ordinal"), record.get("session_name"))
        if type(identity[0]) is not int or not isinstance(identity[1], str):
            raise ValueError("FLASH session identity is invalid")
        if identity in seen_sessions:
            raise ValueError("FLASH manifest contains a duplicate session")
        seen_sessions.add(identity)
        session_status = record.get("status")
        if session_status not in VALID_STATUSES:
            raise ValueError("FLASH session status is invalid")
        session_statuses.append(session_status)
        locator = record.get("session_dir")
        if locator is None:
            if session_status != "failed":
                raise ValueError("usable FLASH session has no session_dir")
            continue
        locator_path = Path(locator) if isinstance(locator, str) else None
        if (
            locator_path is None
            or not locator
            or locator_path.name != locator
            or locator in (".", "..")
        ):
            raise ValueError("FLASH session_dir locator is invalid")
        session_dir = root / locator
        session_manifest = pipeline._read_in_session_manifest(
            session_dir,
            strict=True,
        )
        if session_manifest is None:
            raise ValueError(f"session manifest is missing for {identity!r}")
        if (
            session_manifest.get("flash_result_id")
            != document.get("flash_result_id")
            or session_manifest.get("flash_manifest_sha256")
            != manifest_digest
            or session_manifest.get("flash_input_identity_sha256")
            != document.get("input_identity_sha256")
            or session_manifest.get(
                "flash_input_identity_unavailable_reason"
            )
            != document.get("input_identity_unavailable_reason")
            or session_manifest.get("session_ordinal") != identity[0]
            or session_manifest.get("session_name") != identity[1]
        ):
            raise ValueError(
                f"session manifest is not linked to FLASH record {identity!r}"
            )
    run_status = str(document["status"])
    if run_status in ("clean", "partial") and not session_statuses:
        raise ValueError("usable FLASH run has no sessions")
    if run_status == "clean" and any(
        status != "clean" for status in session_statuses
    ):
        raise ValueError("clean FLASH run contains a non-clean session")
    if (
        session_statuses
        and all(status == "failed" for status in session_statuses)
        and run_status != "failed"
    ):
        raise ValueError("all-failed FLASH sessions require a failed run")
    return "FLASH manifest", run_status


def validate_path(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise ValueError("validate requires one existing file")
    if path.suffix.lower() == ".json":
        return validate_manifest(path)
    from .obs_factory import load_bundle

    bundle = load_bundle(path)
    if bundle.layout_version != 4:
        raise ValueError("validate accepts only current layout-v4 outputs")
    if bundle.quality_status not in VALID_STATUSES:
        raise ValueError("layout-v4 output has invalid quality status")
    return "layout-v4 output", bundle.quality_status


def process_session_command(args: argparse.Namespace) -> int:
    if args.ordinal < 0:
        raise ValueError("--ordinal must be nonnegative")
    if args.plots_dir is not None and args.h5_dir is None:
        raise ValueError("--plots-dir requires --h5-dir")
    collector = IssueCollector(args.issue_policy)
    result = pipeline.process_session(
        args.session_dir,
        landing_time_file=args.landing_time_file,
        h5_dir=args.h5_dir,
        fits_dir=args.fits_dir,
        plots_dir=args.plots_dir,
        manifest_dir=args.manifest_dir,
        name=args.name,
        ordinal=args.ordinal,
        overwrite=args.overwrite,
        flash_root=args.flash_root,
        rederive_telemetry=not args.no_rederive_telemetry,
        issue_collector=collector,
        decoder_strict=args.decoder_strict,
        schema_variant=args.schema_variant,
    )
    print_session_result(result)
    return status_exit_code(result.status)


def process_flash_command(args: argparse.Namespace) -> int:
    if args.plots_dir is not None and args.h5_dir is None:
        raise ValueError("--plots-dir requires --h5-dir")
    collector = IssueCollector(args.issue_policy)
    result = pipeline.process_flash(
        args.flash_dir,
        landing_time_file=args.landing_time_file,
        sessions_root=args.sessions_root,
        h5_dir=args.h5_dir,
        fits_dir=args.fits_dir,
        plots_dir=args.plots_dir,
        manifest_dir=args.manifest_dir,
        overwrite=args.overwrite,
        issue_collector=collector,
        decoder_strict=args.decoder_strict,
        schema_variant=args.schema_variant,
    )
    print_flash_result(result)
    return status_exit_code(result.status)


def validate_command(args: argparse.Namespace) -> int:
    kind, status = validate_path(args.path)
    print(f"valid: {args.path.resolve()} ({kind})")
    print(f"status: {status}")
    return status_exit_code(status)


class IngestArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(f"argument error: {message}")


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--h5-dir", type=Path)
    parser.add_argument("--fits-dir", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--issue-policy",
        choices=tuple(policy.value for policy in IssuePolicy),
        default=IssuePolicy.COLLECT.value,
    )
    parser.add_argument("--decoder-strict", action="store_true")
    parser.add_argument("--schema-variant")


def build_parser() -> argparse.ArgumentParser:
    parser = IngestArgumentParser(prog="lusee-ingest")
    commands = parser.add_subparsers(dest="command", required=True)

    session_parser = commands.add_parser(
        "process-session",
        help="process one extracted uncrater session",
    )
    session_parser.add_argument("session_dir", type=Path)
    session_parser.add_argument("--landing-time-file", type=Path)
    session_parser.add_argument("--manifest-dir", type=Path, required=True)
    session_parser.add_argument("--name")
    session_parser.add_argument("--ordinal", type=int, default=0)
    session_parser.add_argument("--flash-root", type=Path)
    session_parser.add_argument("--no-rederive-telemetry", action="store_true")
    session_parser.add_argument("--h5-dir", type=Path)
    session_parser.add_argument("--fits-dir", type=Path)
    session_parser.add_argument("--plots-dir", type=Path)
    session_parser.add_argument("--overwrite", action="store_true")
    session_parser.add_argument(
        "--issue-policy",
        choices=tuple(policy.value for policy in IssuePolicy),
        default=IssuePolicy.COLLECT.value,
    )
    session_parser.add_argument("--decoder-strict", action="store_true")
    session_parser.add_argument("--schema-variant")
    session_parser.set_defaults(handler=process_session_command)

    flash_parser = commands.add_parser(
        "process-flash",
        help="process one raw FLASH_TLMFS directory",
    )
    flash_parser.add_argument("flash_dir", type=Path)
    flash_parser.add_argument("--landing-time-file", type=Path, required=True)
    flash_parser.add_argument("--sessions-root", type=Path, required=True)
    add_output_arguments(flash_parser)
    flash_parser.set_defaults(handler=process_flash_command)

    validate_parser = commands.add_parser(
        "validate",
        help="validate a layout-v4 output or current manifest",
    )
    validate_parser.add_argument("path", type=Path)
    validate_parser.set_defaults(handler=validate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        return args.handler(args)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
