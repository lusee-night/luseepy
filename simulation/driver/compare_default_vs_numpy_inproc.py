#!/usr/bin/env python3
"""In-process default-vs-numpy comparison with warmup + timed second run.

Key property:
- Each engine is instantiated once.
- Warmup simulate() is run first.
- Timing is measured on the second simulate() call on the SAME simulator object.
"""

import argparse
import copy
import os
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import fitsio
import hydra
import jax
import numpy as np
from omegaconf import OmegaConf

from sim_driver import SimDriver


def parse_args() -> argparse.Namespace:
    default_cfg = Path(__file__).resolve().parent.parent / "config" / "realistic_example.yaml"
    parser = argparse.ArgumentParser(
        description="Compare default(jax) vs numpy engines with in-process warmup/timing."
    )
    parser.add_argument("--config-path", default=str(default_cfg), help="Path to hydra yaml config")
    parser.add_argument("--lmax", type=int, default=8, help="Override observation.lmax")
    parser.add_argument("--freq-end", type=int, default=3, help="Override observation.freq.end")
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=float(os.environ.get("ABS_TOL", "1e-9")),
        help="Absolute tolerance for numeric FITS diffs",
    )
    parser.add_argument(
        "--jax-enable-x64",
        choices=["true", "false"],
        default=None,
        help="Override simulation.jax_enable_x64 for this run",
    )
    return parser.parse_args()


def compose_resolved_cfg(config_path: Path, overrides: list[str]) -> dict:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = hydra.compose(config_name=config_path.stem, overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


def build_simulator_and_times(driver: SimDriver):
    lusee = driver._lusee
    od = driver["observation"]
    obs = lusee.Observation(
        od["lunar_day"],
        deltaT_sec=driver.dt,
        lun_lat_deg=od["lat"],
        lun_long_deg=od["long"],
    )
    combs = od["combinations"]
    if isinstance(combs, str) and combs == "all":
        combs = []
        for i in range(driver.Nbeams):
            for j in range(i, driver.Nbeams):
                combs.append((i, j))

    engine = driver._normalize_engine(driver)
    if engine == "luseepy":
        sim = lusee.DefaultSimulator(
            obs,
            driver.beams,
            driver.sky,
            Tground=od["Tground"],
            combinations=combs,
            freq=driver.freq,
            lmax=driver.lmax,
            cross_power=driver.couplings,
            extra_opts=driver["simulation"],
        )
    elif engine == "numpy":
        sim = lusee.NumpySimulator(
            obs,
            driver.beams,
            driver.sky,
            Tground=od["Tground"],
            combinations=combs,
            freq=driver.freq,
            lmax=driver.lmax,
            cross_power=driver.couplings,
            extra_opts=driver["simulation"],
        )
    else:
        raise ValueError(f"Only default/luseepy and numpy engines are supported here, got {engine!r}")
    return sim, obs.times


def _block_ready(x):
    try:
        return jax.block_until_ready(x)
    except Exception:
        return x


def run_engine(base_cfg: dict, engine: str, output_name: str, cache_name: str):
    cfg = copy.deepcopy(base_cfg)
    cfg["engine"] = engine
    cfg.setdefault("simulation", {})
    cfg["simulation"]["output"] = output_name
    cfg["simulation"]["cache_transform"] = cache_name

    driver = SimDriver(cfg)
    sim, times = build_simulator_and_times(driver)

    # Warmup pass (compile/traces on JAX engine).
    warm = sim.simulate(times=times)
    _block_ready(warm)

    # Timed pass on same simulator object.
    t0 = time.perf_counter()
    out = sim.simulate(times=times)
    _block_ready(out)
    elapsed = time.perf_counter() - t0

    warm_np = np.asarray(warm)
    out_np = np.asarray(out)
    repeat_max_abs = float(np.max(np.abs(out_np - warm_np)))

    out_path = Path(driver.outdir) / output_name
    sim.write_fits(str(out_path))

    return elapsed, out_path, int(len(driver.freq)), repeat_max_abs


def load_by_ext(path: Path) -> dict:
    out = {}
    with fitsio.FITS(str(path)) as f:
        for i in range(len(f)):
            name = f[i].get_extname() or f"EXT{i}"
            out[name] = f[i].read()
    return out


def compare_fits(jax_path: Path, numpy_path: Path, abs_tol: float) -> bool:
    print("Comparing FITS outputs...")
    a = load_by_ext(jax_path)
    b = load_by_ext(numpy_path)

    ok = True
    if set(a) != set(b):
        print("FAIL: Extension name mismatch")
        print("jax only:", sorted(set(a) - set(b)))
        print("numpy only:", sorted(set(b) - set(a)))
        return False

    for name in sorted(a):
        x = a[name]
        y = b[name]
        if x.shape != y.shape:
            print(f"FAIL [{name}]: shape mismatch {x.shape} vs {y.shape}")
            ok = False
            continue
        if np.array_equal(x, y):
            print(f"OK   [{name}]: exact match")
            continue
        if np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
            max_abs = float(np.max(np.abs(x - y)))
            if max_abs <= abs_tol:
                print(
                    f"OK   [{name}]: within tolerance, "
                    f"max_abs_diff={max_abs:.6e} <= {abs_tol:.6e}"
                )
            else:
                print(f"FAIL [{name}]: not equal, max_abs_diff={max_abs:.6e} > {abs_tol:.6e}")
                ok = False
        else:
            print(f"FAIL [{name}]: non-numeric mismatch")
            ok = False

    if ok:
        print("PASS: all FITS extensions are exactly identical within tolerance")
    return ok


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_path).resolve()
    if not config_path.exists():
        print(f"ERROR: config path does not exist: {config_path}")
        return 1
    if not os.environ.get("LUSEE_DRIVE_DIR"):
        print("ERROR: LUSEE_DRIVE_DIR is not set.")
        return 1
    if not os.environ.get("LUSEE_OUTPUT_DIR"):
        print("ERROR: LUSEE_OUTPUT_DIR is not set.")
        return 1

    quick_overrides = [
        "observation.lunar_day='2025-02-01 13:00:00 to 2025-02-01 14:00:00'",
        "observation.dt=3600",
        f"observation.lmax={args.lmax}",
        f"observation.freq.end={args.freq_end}",
    ]
    base_cfg = compose_resolved_cfg(config_path, quick_overrides)
    if args.jax_enable_x64 is not None:
        base_cfg.setdefault("simulation", {})
        base_cfg["simulation"]["jax_enable_x64"] = (args.jax_enable_x64 == "true")

    print("Warmup+timed run: default (jax) engine...")
    jax_sec, jax_out, n_freq, jax_repeat_max = run_engine(
        base_cfg,
        engine="default",
        output_name="sim_output_quick_jax.fits",
        cache_name="quick_2step_jax.pickle",
    )
    print("Warmup+timed run: numpy engine...")
    np_sec, np_out, _, np_repeat_max = run_engine(
        base_cfg,
        engine="numpy",
        output_name="sim_output_quick_numpy.fits",
        cache_name="quick_2step_numpy.pickle",
    )

    ok = compare_fits(jax_out, np_out, args.abs_tol)

    print()
    print("Summary:")
    print(f"  lmax: {args.lmax}")
    print(f"  freq bins: {n_freq}")
    print(f"  jax_enable_x64: {base_cfg.get('simulation', {}).get('jax_enable_x64')}")
    print(f"  jax real time (s): {jax_sec:.6f}")
    print(f"  numpy real time (s): {np_sec:.6f}")
    print(f"  jax repeat max abs (warmup vs timed): {jax_repeat_max:.6e}")
    print(f"  numpy repeat max abs (warmup vs timed): {np_repeat_max:.6e}")
    print(f"  result: {'PASS' if ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
