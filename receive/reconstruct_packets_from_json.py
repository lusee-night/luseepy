import argparse
import sys
import types
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib
import numpy as np
from icecream import ic

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
from telemetry_utils import interpolate_telemetry_to_spectra_times


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


def _extract_spectra_times(coll: unc.Collection) -> np.ndarray:
    times = []
    for sp_dict in coll.spectra:
        meta = sp_dict.get("meta")
        if meta is None:
            continue
        if hasattr(meta, "time"):
            times.append(float(meta.time))
    return np.asarray(times, dtype=np.float64)


def _relative_times(times: np.ndarray) -> list:
    arr = np.asarray(times, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    return arr - arr[0]


def _pick_telemetry_plot_fields(fpga_telemetry: dict) -> Tuple[list, list]:
    keys = list(fpga_telemetry.keys())
    exclude = {"fpga_mission_seconds", "fpga_lusee_subsecs"}

    voltage_fields = []
    temp_fields = []

    preferred_voltage = [
        "VMON_6V",
        "VMON_3V7",
        "VMON_1V8",
        "VMON_3V3D",
        "SPE_P5_V",
        "SPE_N5_V",
    ]
    preferred_temp = [
        "THERM_FPGA",
        "THERM_DCB",
        "SPE_FPGA_T",
        "SPE_ADC0_T",
        "SPE_ADC1_T",
    ]

    for k in preferred_voltage:
        if k in fpga_telemetry:
            voltage_fields.append(k)
        if len(voltage_fields) == 2:
            break

    for k in preferred_temp:
        if k in fpga_telemetry:
            temp_fields.append(k)
        if len(temp_fields) == 2:
            break

    if len(voltage_fields) < 2:
        for k in sorted(keys):
            if k in exclude or k in voltage_fields:
                continue
            if "V" in k:
                voltage_fields.append(k)
            if len(voltage_fields) == 2:
                break

    if len(temp_fields) < 2:
        for k in sorted(keys):
            if k in exclude or k in temp_fields:
                continue
            if "THERM" in k or k.endswith("_T"):
                temp_fields.append(k)
            if len(temp_fields) == 2:
                break

    return voltage_fields, temp_fields


def plot_telemetry(
    fpga_telemetry: dict,
    voltage_fields: list,
    temp_fields: list,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    mission_seconds = np.asarray(fpga_telemetry.get("fpga_mission_seconds", []), dtype=np.float64).reshape(-1)
    if mission_seconds.size == 0:
        return

    x = mission_seconds - mission_seconds[0]
    nrows = int(bool(voltage_fields)) + int(bool(temp_fields))
    if nrows == 0:
        return

    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    ax_i = 0
    if voltage_fields:
        ax = axes[ax_i]
        for field in voltage_fields:
            y = np.asarray(fpga_telemetry[field]).reshape(-1)
            ax.plot(x[:len(y)], y, label=field, linewidth=1.2)
        ax.set_ylabel("Voltage / current")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax_i += 1

    if temp_fields:
        ax = axes[ax_i]
        for field in temp_fields:
            y = np.asarray(fpga_telemetry[field]).reshape(-1)
            ax.plot(x[:len(y)], y, label=field, linewidth=1.2)
        ax.set_ylabel("Temperature [C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Telemetry relative time [s]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


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
    parser.set_defaults(interp_use_normalized_relative_position=True)
    parser.add_argument(
        "--interp-use-normalized-relative-position",
        dest="interp_use_normalized_relative_position",
        action="store_true",
        help="Interpolate telemetry on normalized [0, 1] relative session position (default).",
    )
    parser.add_argument(
        "--interp-use-absolute-time",
        dest="interp_use_normalized_relative_position",
        action="store_false",
        help="Interpolate telemetry directly on absolute time axes.",
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
        spectra_times_abs = _extract_spectra_times(coll)
        spectra_rel_times = _relative_times(spectra_times_abs)

        print(f"[{idx}] Parsing telemetry binary: {telem_file}")
        fpga_telemetry = decode_packets(telem_file)
        telem_rel_times = _relative_times(fpga_telemetry["fpga_mission_seconds"])
        telem_subsecs = np.asarray(fpga_telemetry["fpga_lusee_subsecs"]).reshape(-1)
        ic(spectra_rel_times, telem_rel_times, telem_subsecs)
        spectra_interpolated_telemetry = interpolate_telemetry_to_spectra_times(
            spectra_times=spectra_times_abs,
            telemetry=fpga_telemetry,
            telemetry_time_key="fpga_mission_seconds",
            telemetry_subseconds_key="fpga_lusee_subsecs",
            use_normalized_relative_position=args.interp_use_normalized_relative_position,
        )
        voltage_fields, temp_fields = _pick_telemetry_plot_fields(fpga_telemetry)
        session_id = session_dir.name.removeprefix("session_")
        telem_plot_path = output_dir / f"session_{session_id}_telem_plot.png"
        plot_telemetry(
            fpga_telemetry=fpga_telemetry,
            voltage_fields=voltage_fields,
            temp_fields=temp_fields,
            output_path=telem_plot_path,
            title=f"{session_dir.name} telemetry sanity",
        )
        print(f"[{idx}] Saved telemetry sanity plot: {telem_plot_path}")
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
            spectra_interpolated_telemetry=spectra_interpolated_telemetry,
        )
        print(f"[{idx}] Saved {output_h5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
