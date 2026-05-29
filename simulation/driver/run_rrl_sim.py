#!/usr/bin/env python3
"""
RRL + ULSA Croissant driver driven by YAML (fine frequency grid, beam resampling).

Staged analysis (default): ULSA + smooth catalog envelope → beam convolution at
beam frequency resolution → resample to ``rrl_analysis.fine_step_khz`` (0.5 kHz)
→ add Rydberg line spectrum. Positions from the RRL FITS only.

Example::
    run this from simulation/ folder:
    python driver/run_rrl_sim.py config/rrl_fine_1khz.yaml

"""

from __future__ import annotations

import argparse
import os
import sys


def _resolve_path(val: str) -> str:
    if isinstance(val, str) and val.startswith("$"):
        key = val[1:]
        if key in os.environ:
            return os.environ[key]
    return val


def main(argv: list[str] | None = None) -> int:
    import yaml
    from yaml.loader import SafeLoader

    import lusee as lu
    from lusee.beam_fine_frequency import linear_resample_beam_freq_mhz
    from lusee.frequencies import observation_frequency_mhz_from_config

    argv = argv if argv is not None else sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.normpath(os.path.join(here, "..", "config", "rrl_fine_1khz.yaml"))
    ap = argparse.ArgumentParser(description="RRL + ULSA Croissant simulation from YAML")
    ap.add_argument(
        "config",
        nargs="?",
        default=default_cfg,
        help=f"YAML config (default: {default_cfg})",
    )
    args = ap.parse_args(argv)

    if lu.CroSimulator is None:
        print("CroSimulator requires croissant and s2fft.", file=sys.stderr)
        return 1

    cfg_path = args.config
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)

    paths = cfg.get("paths") or {}
    drive = _resolve_path(paths.get("lusee_drive_dir", "$LUSEE_DRIVE_DIR"))
    if not drive or not os.path.isdir(drive):
        print("Set paths.lusee_drive_dir / LUSEE_DRIVE_DIR to your LUSEE drive root.", file=sys.stderr)
        return 1

    out = _resolve_path(paths.get("output_dir", "$LUSEE_OUTPUT_DIR"))
    if not out:
        out = "."
    os.makedirs(out, exist_ok=True)
    out_fits = os.path.join(out, (cfg.get("output") or {}).get("fits", "rrl_sim_waterfall.fits"))

    od = cfg["observation"]
    freq_beam = observation_frequency_mhz_from_config(od["freq"])
    lmax = int(od["lmax"])

    sky_kw = cfg.get("sky") or {}
    rrl_kw = cfg.get("rrl_analysis") or {}
    fine_step_khz = float(rrl_kw.get("fine_step_khz", 0.5))
    env_kw = dict(rrl_kw.get("envelope") or {})
    gas_case = str(env_kw.pop("gas_case", "cold")).lower()
    envelope_params = lu.vydula2024_envelope_params_from_config(gas_case, env_kw)

    pipeline = lu.build_rrl_analysis_pipeline(
        lmax,
        lusee_drive_dir=drive,
        fine_step_khz=fine_step_khz,
        spot_sigma_deg=float(sky_kw.get("spot_sigma_deg", 0.35)),
        gas_case=gas_case,
        envelope_params=envelope_params,
        envelope_nu_ref_mhz=float(env_kw.get("nu_ref_mhz", 12.5)),
        envelope_sigma_mhz=float(env_kw.get("sigma_mhz", 3.0)),
        envelope_amplitude_k=float(env_kw.get("amplitude_k", 0.5)),
        rrl_sigma_mhz=float(
            sky_kw.get("rrl_sigma_mhz", lu.RRL_DEFAULT_LINE_SIGMA_MHZ)
        ),
        rrl_peak_k=float(sky_kw.get("rrl_peak_k", lu.RRL_DEFAULT_LINE_PEAK_K)),
    )

    bd = cfg.get("beam") or {}
    btype = str(bd.get("type", "gauss")).lower()
    if btype == "gauss":
        beam0 = lu.BeamGauss(
            alt_deg=float(bd["alt_deg"]),
            az_deg=float(bd.get("az_deg", 0.0)),
            sigma_deg=float(bd["sigma_deg"]),
            one_over_freq_scaling=bool(bd.get("one_over_freq_scaling", False)),
            id=bd.get("id", "rrl_demo"),
        )
    elif btype == "fits":
        broot = os.path.join(drive, paths.get("beam_dir", "Simulations/BeamModels/"))
        bf = os.path.join(broot, bd["file"])
        beam0 = lu.Beam(bf, id=bd.get("id", "beam"))
        ang = float(bd.get("angle", 0.0))
        if ang != 0.0:
            beam0 = beam0.rotate(ang)
    else:
        print(f"Unknown beam.type: {btype}", file=sys.stderr)
        return 1

    beam = linear_resample_beam_freq_mhz(beam0, freq_beam)

    obs = lu.Observation(
        od["lunar_day"],
        deltaT_sec=float(od["dt"]),
        lun_lat_deg=float(od["lat"]),
        lun_long_deg=float(od["long"]),
    )

    combs = od.get("combinations", [[0, 0]])
    combs = [tuple(int(x) for x in c) for c in combs]

    print(f"Config: {cfg_path}")
    print(
        f"Staged RRL pipeline: beam ν grid ({len(freq_beam)} ch), "
        f"fine output Δν={fine_step_khz} kHz"
    )
    print(f"Simulating {len(obs.times)} times …")
    pipeline.run(
        obs,
        beam,
        CroSimulator=lu.CroSimulator,
        Tground=float(od.get("Tground", 0.0)),
        combinations=combs,
        times=obs.times,
    )
    pipeline.write_final_fits(out_fits, obs)
    st = pipeline.stages
    print(
        f"Wrote {out_fits}  shape={st.waterfall_final.shape}  "
        f"freq {st.freq_fine_mhz[0]:.4f}–{st.freq_fine_mhz[-1]:.4f} MHz"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
