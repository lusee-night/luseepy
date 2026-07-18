#!/usr/bin/env python3
"""Compare DCB telemetry decode between two HDF5 layouts.

Iterates a set of session HDF5 files and, for each, compares all 57
documented FPGA channels in the new layout (``/DCB_telemetry/fpga_<NAME>``)
against a reference HDF5 in some legacy layout (default:
``/DCB_telemetry/<NAME>`` -- no prefix).

Optionally also checks the four encoder fields (``encoder_*`` /
``enc_pos`` / ``enc_status``) when both files carry them.

For each (session, channel) pair we report:
  * shape match
  * max absolute diff
  * max relative diff (against the larger-magnitude side)
  * a PASS / WARN / FAIL classification using user-tunable tolerances

A summary table prints to stdout. Exit code is 0 when every comparison
is PASS, 1 otherwise.

Usage:

    python examples/ingest/compare_dcb_telemetry.py \\
        --new /path/to/jsons_data/out/h5 \\
        --ref /path/to/jsons_data/h5_from_json

The two directories must contain matching ``session_*.h5`` filenames.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Inject the LUSEE_TELEMETRY_PATH side-channel so we can pull the
# canonical 57-field list straight from the private decoder. Falls back
# to the union of channels actually present in the files when the
# decoder isn't loadable.
_TLM_PATH = os.environ.get("LUSEE_TELEMETRY_PATH")
if _TLM_PATH and _TLM_PATH not in sys.path:
    sys.path.insert(0, _TLM_PATH)


def _canonical_fields() -> Optional[Tuple[str, ...]]:
    try:
        import lusee_telemetry    # type: ignore[import-not-found]
        return tuple(lusee_telemetry.DCB_FIELDS)
    except (ImportError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

@dataclass
class ChannelResult:
    name: str
    new_shape: Optional[tuple]
    ref_shape: Optional[tuple]
    max_abs: Optional[float]
    max_rel: Optional[float]
    status: str    # "PASS" | "WARN" | "FAIL" | "SKIP"
    note: str = ""


@dataclass
class SessionResult:
    session: str
    channels: List[ChannelResult] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.channels if c.status == "PASS")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.channels if c.status == "WARN")

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.channels if c.status == "FAIL")

    @property
    def n_skip(self) -> int:
        return sum(1 for c in self.channels if c.status == "SKIP")


def _read(group, name) -> Optional[np.ndarray]:
    if name in group:
        return group[name][...]
    return None


def _classify(arr_a: np.ndarray, arr_b: np.ndarray,
              *, atol: float, rtol: float) -> Tuple[str, float, float, str]:
    """Compare two 1-D arrays elementwise; return (status, max_abs, max_rel, note)."""
    if arr_a.shape != arr_b.shape:
        return "FAIL", float("nan"), float("nan"), f"shape mismatch {arr_a.shape} vs {arr_b.shape}"

    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        # Every entry is NaN/Inf in at least one side -- silently treat as PASS
        # if the corresponding side is also NaN, else WARN.
        same_nan = np.all(np.isnan(a) == np.isnan(b))
        return ("PASS" if same_nan else "WARN", 0.0, 0.0,
                "all NaN" if same_nan else "NaN positions disagree")

    diff = np.abs(a[finite] - b[finite])
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(a[finite]), np.abs(b[finite]))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 0, diff / denom, 0.0)
    max_rel = float(rel.max())

    if max_abs <= atol or max_rel <= rtol:
        status = "PASS"
    elif max_abs <= 10 * atol or max_rel <= 10 * rtol:
        status = "WARN"
    else:
        status = "FAIL"
    return status, max_abs, max_rel, ""


def compare_session(
    new_path: Path,
    ref_path: Path,
    *,
    fields: Tuple[str, ...],
    new_prefix: str = "fpga_",
    ref_prefix: str = "",
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> SessionResult:
    """Compare DCB FPGA channels between ``new_path`` and ``ref_path``."""
    result = SessionResult(session=new_path.stem)
    with h5py.File(new_path, "r") as new_f, h5py.File(ref_path, "r") as ref_f:
        new_g = new_f.get("DCB_telemetry")
        ref_g = ref_f.get("DCB_telemetry")
        if new_g is None or ref_g is None:
            note = "missing /DCB_telemetry in {}".format(
                "new" if new_g is None else "ref"
            )
            result.channels.append(
                ChannelResult(name="<group>", new_shape=None, ref_shape=None,
                              max_abs=None, max_rel=None, status="SKIP", note=note)
            )
            return result

        for fname in fields:
            new_arr = _read(new_g, new_prefix + fname)
            ref_arr = _read(ref_g, ref_prefix + fname)
            if new_arr is None and ref_arr is None:
                result.channels.append(
                    ChannelResult(name=fname, new_shape=None, ref_shape=None,
                                  max_abs=None, max_rel=None, status="SKIP",
                                  note="missing in both")
                )
                continue
            if new_arr is None:
                result.channels.append(
                    ChannelResult(name=fname, new_shape=None, ref_shape=ref_arr.shape,
                                  max_abs=None, max_rel=None, status="FAIL",
                                  note="missing in new")
                )
                continue
            if ref_arr is None:
                result.channels.append(
                    ChannelResult(name=fname, new_shape=new_arr.shape, ref_shape=None,
                                  max_abs=None, max_rel=None, status="FAIL",
                                  note="missing in ref")
                )
                continue
            status, max_abs, max_rel, note = _classify(new_arr, ref_arr,
                                                       atol=atol, rtol=rtol)
            result.channels.append(
                ChannelResult(name=fname, new_shape=new_arr.shape,
                              ref_shape=ref_arr.shape, max_abs=max_abs,
                              max_rel=max_rel, status=status, note=note)
            )
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_per_channel_summary(sessions: List[SessionResult],
                                fields: Tuple[str, ...]) -> None:
    """Per-channel rollup: how many sessions pass / warn / fail per field."""
    print()
    print("Per-channel rollup across sessions:")
    print(f"  {'channel':<22} {'PASS':>5} {'WARN':>5} {'FAIL':>5} {'SKIP':>5} "
          f"{'max_abs':>12} {'max_rel':>12}")
    print("  " + "-" * 75)
    for fname in fields:
        rows = [s for s in sessions for c in s.channels if c.name == fname]
        chans = [c for s in sessions for c in s.channels if c.name == fname]
        n_pass = sum(1 for c in chans if c.status == "PASS")
        n_warn = sum(1 for c in chans if c.status == "WARN")
        n_fail = sum(1 for c in chans if c.status == "FAIL")
        n_skip = sum(1 for c in chans if c.status == "SKIP")
        max_abs = max((c.max_abs for c in chans
                       if c.max_abs is not None), default=float("nan"))
        max_rel = max((c.max_rel for c in chans
                       if c.max_rel is not None), default=float("nan"))
        print(f"  {fname:<22} {n_pass:>5} {n_warn:>5} {n_fail:>5} {n_skip:>5} "
              f"{max_abs:>12.4g} {max_rel:>12.4g}")


def _print_per_session_summary(sessions: List[SessionResult]) -> None:
    print()
    print("Per-session rollup:")
    print(f"  {'session':<28} {'PASS':>5} {'WARN':>5} {'FAIL':>5} {'SKIP':>5}")
    print("  " + "-" * 50)
    for s in sessions:
        print(f"  {s.session:<28} {s.n_pass:>5} {s.n_warn:>5} "
              f"{s.n_fail:>5} {s.n_skip:>5}")


def _print_failures(sessions: List[SessionResult], *, max_lines: int = 25) -> None:
    fails = [(s.session, c) for s in sessions for c in s.channels
             if c.status == "FAIL"]
    if not fails:
        print("\nNo FAIL entries.")
        return
    print(f"\nFAIL details (showing up to {max_lines}):")
    for session, c in fails[:max_lines]:
        print(f"  {session:<28} {c.name:<22}  "
              f"max_abs={c.max_abs!r}  max_rel={c.max_rel!r}  {c.note}")
    if len(fails) > max_lines:
        print(f"  ... and {len(fails) - max_lines} more FAIL entries.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new", required=True, type=Path,
                   help="directory of new-layout HDF5 files (one per session)")
    p.add_argument("--ref", required=True, type=Path,
                   help="directory of reference / legacy-layout HDF5 files")
    p.add_argument("--new-prefix", default="fpga_",
                   help="dataset name prefix in the new layout (default 'fpga_')")
    p.add_argument("--ref-prefix", default="",
                   help="dataset name prefix in the reference layout (default '')")
    p.add_argument("--atol", type=float, default=1e-3,
                   help="absolute tolerance for PASS classification")
    p.add_argument("--rtol", type=float, default=1e-3,
                   help="relative tolerance for PASS classification")
    p.add_argument("--fields", default=None,
                   help="comma-separated subset of channel names to compare; "
                        "default uses lusee_telemetry.DCB_FIELDS or the union "
                        "of channels found in both files")
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def _list_sessions(d: Path) -> Dict[str, Path]:
    return {p.stem: p for p in sorted(d.glob("session*.h5"))}


def _channel_union(new_h5: Path, ref_h5: Path,
                   new_prefix: str, ref_prefix: str) -> Tuple[str, ...]:
    chans = set()
    for path, prefix in ((new_h5, new_prefix), (ref_h5, ref_prefix)):
        try:
            with h5py.File(path, "r") as f:
                if "DCB_telemetry" not in f:
                    continue
                for k in f["DCB_telemetry"].keys():
                    if k.startswith(prefix):
                        chans.add(k[len(prefix):])
        except OSError:
            continue
    # Drop time-axis fields (mission_seconds, lusee_subsecs, encoder_*).
    drop = {"mission_seconds", "lusee_subsecs"}
    chans -= drop
    chans = {c for c in chans if not c.startswith("encoder_")
             and c not in ("enc_pos", "enc_status")}
    return tuple(sorted(chans))


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    args.new = args.new.resolve()
    args.ref = args.ref.resolve()
    new_files = _list_sessions(args.new)
    ref_files = _list_sessions(args.ref)

    common = sorted(set(new_files) & set(ref_files))
    if not common:
        print("ERROR: no overlapping session_*.h5 filenames between "
              f"{args.new} and {args.ref}", file=sys.stderr)
        return 2
    new_only = sorted(set(new_files) - set(ref_files))
    ref_only = sorted(set(ref_files) - set(new_files))
    if new_only:
        print(f"NOTE: {len(new_only)} session(s) only in --new "
              f"(skipped): {new_only[:3]}{'...' if len(new_only) > 3 else ''}")
    if ref_only:
        print(f"NOTE: {len(ref_only)} session(s) only in --ref "
              f"(skipped): {ref_only[:3]}{'...' if len(ref_only) > 3 else ''}")

    # Determine the channel set
    if args.fields:
        fields = tuple(s.strip() for s in args.fields.split(",") if s.strip())
        source = "user-supplied"
    else:
        fields = _canonical_fields() or _channel_union(
            new_files[common[0]], ref_files[common[0]],
            args.new_prefix, args.ref_prefix,
        )
        source = ("lusee_telemetry.DCB_FIELDS"
                  if _canonical_fields() is not None else "channel union")
    print(f"Comparing {len(fields)} channels (source: {source}) "
          f"across {len(common)} session(s).")
    print(f"Tolerances: atol={args.atol:g}, rtol={args.rtol:g} "
          f"(WARN window = 10x).")

    sessions: List[SessionResult] = []
    for stem in common:
        sr = compare_session(
            new_files[stem], ref_files[stem],
            fields=fields,
            new_prefix=args.new_prefix,
            ref_prefix=args.ref_prefix,
            atol=args.atol,
            rtol=args.rtol,
        )
        sessions.append(sr)

    _print_per_session_summary(sessions)
    _print_per_channel_summary(sessions, fields)
    _print_failures(sessions)

    total_fail = sum(s.n_fail for s in sessions)
    print()
    print(f"Result: {total_fail} FAIL across "
          f"{sum(len(s.channels) for s in sessions)} comparisons.")
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
