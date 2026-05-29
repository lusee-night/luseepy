#!/usr/bin/env python3
"""Walk a directory of uncrater session dirs and convert each to HDF5+plots.

Usage:
    python examples/ingest/process_session_dir.py \\
        --input /path/to/dir/with/session/dirs \\
        --out /path/to/out

A directory is treated as an uncrater session if it contains either:

  * a ``cdi_output/`` subdirectory of ``NNNNN_XXXX.bin`` files (Layout B), or
  * a flat set of ``NNNNN_XXXX.bin`` files at its top level (Layout A).

Output layout (the same as ``process_flash_dir.py``):

    out/
      h5/<session_name>.h5
      h5/<session_name>.json
      plots/<session_name>/<plot>.png

Sessions are named after the input directory's basename. If the input
itself is a single session directory, only that one is processed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

from lusee import ingest as li


def _has_bin_files(d: Path) -> bool:
    if not d.is_dir():
        return False
    for entry in d.iterdir():
        name = entry.name
        if entry.is_file() and name.endswith(".bin") and "_" in name:
            return True
    return False


def is_session_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    if _has_bin_files(d / "cdi_output"):
        return True
    return _has_bin_files(d)


def find_session_dirs(start: Path) -> List[Path]:
    if is_session_dir(start):
        return [start]
    return sorted(p for p in start.iterdir() if is_session_dir(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="single session dir or a parent dir containing several")
    p.add_argument("--out", required=True, type=Path,
                   help="output root; h5/ and plots/ subdirs are created")
    p.add_argument("--no-plots", action="store_true", help="skip plot generation")
    p.add_argument("--no-manifest", action="store_true", help="skip session.json")
    p.add_argument("--fits", action="store_true",
                   help="also write a FITS file per session (mirrors HDF5 layout)")
    p.add_argument("--interpolate-telemetry", action="store_true",
                   help="also write /spectra_interpolated_telemetry")
    p.add_argument(
        "--flash-root", type=Path, default=None,
        help=("override the FLASH_TLMFS path recorded in session.json; "
              "useful when sessions were moved between machines"),
    )
    p.add_argument(
        "--no-rederive-telemetry", action="store_true",
        help=("skip the b01 telemetry re-read even when a flash backreference "
              "is available; falls back to legacy sidecar / no-telemetry"),
    )
    p.add_argument("--verbose", "-v", action="count", default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.input = args.input.resolve()
    args.out = args.out.resolve()
    session_dirs = find_session_dirs(args.input)
    if not session_dirs:
        print(f"no session directories found under {args.input}", file=sys.stderr)
        return 2

    out_h5 = args.out / "h5"
    out_fits = args.out / "fits" if args.fits else None
    out_plots = None if args.no_plots else args.out / "plots"
    out_manifests = None if args.no_manifest else args.out / "h5"

    for ordinal, sd in enumerate(session_dirs):
        name = sd.name
        print(f"=== processing {sd} (-> {name}) ===")
        result = li.process_session(
            sd,
            h5_dir=out_h5,
            fits_dir=out_fits,
            plots_dir=out_plots,
            manifest_dir=out_manifests,
            name=name,
            ordinal=ordinal,
            interpolate_telemetry=args.interpolate_telemetry,
            flash_root=args.flash_root,
            rederive_telemetry=not args.no_rederive_telemetry,
        )
        telem = result.telemetry_source or "none"
        print(f"  spectra={result.n_spectra}, tr={result.n_tr_spectra}, "
              f"zoom={result.n_zoom_spectra}, wf={result.n_waveforms}, "
              f"hk={result.n_housekeeping}, telem={telem}, "
              f"warnings={result.n_warnings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
