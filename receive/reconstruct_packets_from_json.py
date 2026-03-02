import argparse
import sys
import types
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


def _install_hexdump_stub_if_needed() -> None:
    """
    uncrater imports `hexdump` for debug dumps.
    Provide a tiny fallback so the processing pipeline can run in minimal envs.
    """
    try:
        import hexdump  # noqa: F401
        return
    except ImportError:
        pass

    mod = types.ModuleType("hexdump")

    def _hexdump(data, result=None):
        b = bytes(data)
        lines = []
        for i in range(0, len(b), 16):
            chunk = b[i:i + 16]
            hx = " ".join(f"{v:02x}" for v in chunk)
            asc = "".join(chr(v) if 32 <= v < 127 else "." for v in chunk)
            lines.append(f"{i:08x}  {hx:<47}  {asc}")
        out = "\n".join(lines)
        if result == "return":
            return out
        print(out)
        return None

    mod.hexdump = _hexdump
    sys.modules["hexdump"] = mod


_install_hexdump_stub_if_needed()

import uncrater as unc
from const_storage import Constants
from dcb_decode import decode_packets
from hdf5_writer import save_to_hdf5


def _iter_sessions(spec_root: Path) -> Iterable[Tuple[Path, Path, Path]]:
    for session_dir in sorted(spec_root.glob("session_*")):
        if not session_dir.is_dir():
            continue
        cdi_dir = session_dir / "cdi_output"
        telem_file = session_dir / "DCB_telemetry.json"
        if cdi_dir.is_dir() and telem_file.exists():
            yield session_dir, cdi_dir, telem_file


def _extract_session_start_seconds(coll: unc.Collection) -> Optional[int]:
    for pkt in coll.cont:
        if isinstance(pkt, unc.Packet_Hello):
            pkt._read()
            return int(pkt.time)
    return None


def _extract_spectra_time_span(coll: unc.Collection) -> Tuple[Optional[int], Optional[int]]:
    times = []
    for sp_dict in coll.spectra:
        meta = sp_dict.get("meta")
        if meta is None:
            continue
        if hasattr(meta, "time"):
            times.append(int(meta.time))
    if not times:
        return None, None
    return min(times), max(times)


def _print_time_check(
    session_dir: Path,
    session_start_sec: Optional[int],
    spectra_min_sec: Optional[int],
    spectra_max_sec: Optional[int],
    telemetry: dict,
) -> None:
    mission_seconds = telemetry["fpga_mission_seconds"]
    if len(mission_seconds) == 0:
        print(
            f"[{session_dir.name}] telemetry packets: 0 | "
            f"session_start={session_start_sec} | spectra=[{spectra_min_sec}, {spectra_max_sec}]"
        )
        return

    telem_first = int(mission_seconds[0])
    telem_last = int(mission_seconds[-1])
    telem_min = int(np.min(mission_seconds))
    telem_max = int(np.max(mission_seconds))

    delta_first = None if session_start_sec is None else (telem_first - session_start_sec)
    start_in_telem = (
        None
        if session_start_sec is None
        else (telem_min <= session_start_sec <= telem_max)
    )

    print(
        f"[{session_dir.name}] "
        f"session_start={session_start_sec} "
        f"spectra=[{spectra_min_sec}, {spectra_max_sec}] "
        f"telemetry_first={telem_first} telemetry_last={telem_last} "
        f"telemetry_range=[{telem_min}, {telem_max}] "
        f"delta_first={delta_first} start_in_telemetry_range={start_in_telem}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process existing session_* directories (with cdi_output + DCB_telemetry.json) "
            "and write one HDF5 per session."
        )
    )
    parser.add_argument(
        "--spec-root",
        default="graham_data/20250923_155948/spec",
        help="Root containing session_* directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for per-session HDF5 files (default: spec root).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N sessions (debug helper).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing if output HDF5 already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    spec_root = Path(args.spec_root).resolve()
    if not spec_root.exists():
        raise FileNotFoundError(f"Spec root not found: {spec_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else spec_root
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions = list(_iter_sessions(spec_root))
    if args.limit is not None:
        sessions = sessions[:args.limit]

    if not sessions:
        print(f"No session_* directories with cdi_output + DCB_telemetry.json under {spec_root}")
        return 0

    print(f"Found {len(sessions)} sessions under {spec_root}")

    for idx, (session_dir, cdi_dir, telem_file) in enumerate(sessions, 1):
        output_h5 = output_dir / f"{session_dir.name}.h5"
        if args.skip_existing and output_h5.exists():
            print(f"[{idx}] Skipping existing {output_h5}")
            continue

        print(f"[{idx}] Loading cdi_output: {cdi_dir}")
        coll = unc.Collection(str(cdi_dir))
        session_start_sec = _extract_session_start_seconds(coll)
        spectra_min_sec, spectra_max_sec = _extract_spectra_time_span(coll)

        print(f"[{idx}] Parsing telemetry binary: {telem_file}")
        fpga_telemetry = decode_packets(telem_file)
        _print_time_check(
            session_dir=session_dir,
            session_start_sec=session_start_sec,
            spectra_min_sec=spectra_min_sec,
            spectra_max_sec=spectra_max_sec,
            telemetry=fpga_telemetry,
        )

        print(f"[{idx}] Writing HDF5: {output_h5}")
        save_to_hdf5(
            cdi_dir=str(cdi_dir),
            output_file=str(output_h5),
            consts=Constants(),
            dcb_fpga_telemetry=fpga_telemetry,
            dcb_encoder_telemetry=None,
        )
        print(f"[{idx}] Saved {output_h5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
