#!/usr/bin/env python3
"""Walk a directory of FLASH_TLMFS roots and convert each to HDF5+plots.

Usage:
    python examples/ingest/process_flash_dir.py \\
        --input /path/to/dir/with/flash/dirs \\
        --out /path/to/out

Each FLASH_TLMFS root produces N session directories (one per detected
session). The output layout is:

    out/
      sessions/<session_name>/cdi_output/NNNNN_XXXX.bin
      h5/<session_name>.h5
      h5/<session_name>.json
      plots/<session_name>/<plot>.png

Input may be either a single FLASH_TLMFS directory or a parent directory
containing several. Detection is by the presence of at least one
``bNN/FFFFFFFE`` file inside.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from lusee import ingest as li


SCIENCE_BANKS = ("b05", "b06", "b07", "b08", "b09")
TELEMETRY_BANK = "b01"
ALL_BANKS = (TELEMETRY_BANK,) + SCIENCE_BANKS


def is_flash_root(d: Path) -> bool:
    """A directory is a FLASH_TLMFS root if it has at least one bNN/FFFFFFFE file."""
    if not d.is_dir():
        return False
    for bank in ALL_BANKS:
        if (d / bank / "FFFFFFFE").is_file():
            return True
    return False


def find_flash_roots(start: Path) -> List[Path]:
    if is_flash_root(start):
        return [start]
    return sorted(p for p in start.iterdir() if is_flash_root(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="FLASH_TLMFS root or a parent dir containing multiple")
    p.add_argument("--out", required=True, type=Path,
                   help="output root; sessions/, h5/, plots/ subdirs are created")
    p.add_argument("--no-plots", action="store_true", help="skip plot generation")
    p.add_argument("--no-manifest", action="store_true", help="skip session.json manifests")
    p.add_argument("--fits", action="store_true",
                   help="also write a FITS file per session (mirrors HDF5 layout)")
    p.add_argument("--interpolate-telemetry", action="store_true",
                   help="also write /spectra_interpolated_telemetry")
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
    flash_roots = find_flash_roots(args.input)
    if not flash_roots:
        print(f"no FLASH_TLMFS roots found under {args.input}", file=sys.stderr)
        return 2

    out_sessions = args.out / "sessions"
    out_h5 = args.out / "h5"
    out_fits = args.out / "fits" if args.fits else None
    out_plots = None if args.no_plots else args.out / "plots"
    out_manifests = None if args.no_manifest else args.out / "h5"   # adjacent to .h5

    for flash in flash_roots:
        print(f"=== processing {flash} ===")
        results = li.process_flash(
            flash,
            sessions_root=out_sessions,
            h5_dir=out_h5,
            fits_dir=out_fits,
            plots_dir=out_plots,
            manifest_dir=out_manifests,
            interpolate_telemetry=args.interpolate_telemetry,
        )
        for r in results:
            print(f"  -> {r.session_name} "
                  f"(spectra={r.n_spectra}, tr={r.n_tr_spectra}, "
                  f"telem={r.has_telemetry}, warnings={r.n_warnings})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
