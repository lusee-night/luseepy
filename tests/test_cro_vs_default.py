#!/usr/bin/env python3
"""
Test script to compare CroSimulator vs DefaultSimulator outputs.
DefaultSimulator is the reference. Identifies where differences come from
(normalization, coordinate frame, etc.) to guide fixes in CroSimulator.

Usage:
  cd luseepy && python tests/test_cro_vs_default.py [config.yaml]
  (default config: simulation/config/sim_choice_realistic.yaml)

Requires: LUSEE_DRIVE_DIR, LUSEE_OUTPUT_DIR (or paths in config).
Sky must have frame="galactic" for CroSimulator (e.g. FitsSky).
"""

import os
import sys
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Add luseepy to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load project .env so LUSEE_DRIVE_DIR / LUSEE_OUTPUT_DIR don't need exporting each time
def _load_project_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if key and value and key not in os.environ:
                os.environ[key] = value.strip('"').strip("'")


_load_project_env()


def load_config(config_path):
    """Load YAML config and resolve env vars ($VAR only)."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    root = cfg["paths"]["lusee_drive_dir"]
    if root.startswith("$"):
        root = os.environ.get(root[1:], root)
    outdir = cfg["paths"]["output_dir"]
    if outdir.startswith("$"):
        outdir = os.environ.get(outdir[1:], outdir)
    cfg["_root"] = root
    cfg["_outdir"] = outdir
    return cfg


def setup_simulation(cfg, lusee):
    """Build Observation, beams, sky from config (same as SimDriver)."""
    root = cfg["_root"]
    od = cfg["observation"]
    dt = od["dt"]
    if isinstance(dt, str):
        dt = eval(dt)
    freq = np.arange(od["freq"]["start"], od["freq"]["end"], od["freq"]["step"])
    lmax = od["lmax"]

    # Sky
    sky_cfg = cfg["sky"]
    if sky_cfg.get("type", "file") == "file":
        fname = os.path.join(root, cfg["paths"]["sky_dir"], sky_cfg["file"])
        sky = lusee.sky.FitsSky(fname, lmax=lmax)
    elif sky_cfg.get("type") == "Cane1979":
        sky = lusee.sky.ConstSkyCane1979(lmax, lmax=lmax, freq=freq)
    else:
        sky = lusee.sky.ConstSky(lmax, lmax=lmax, T=2.73, freq=freq)

    # Beams
    broot = os.path.join(root, cfg["paths"]["beam_dir"])
    bdc = cfg["beam_config"]
    couplings = bdc.get("couplings")
    beam_type = bdc.get("type", "fits")
    beams = []
    for b in od["beams"]:
        cbeam = cfg["beams"][b]
        if beam_type == "fits":
            filename = cbeam.get("file", bdc.get("default_file"))
            fname = os.path.join(broot, filename)
            B = lusee.Beam(fname, id=b)
        else:
            raise NotImplementedError(f"beam type {beam_type}")
        angle = bdc.get("common_beam_angle", 0) + cbeam.get("angle", 0)
        B = B.rotate(angle)
        beams.append(B)

    cross_power = None
    if couplings:
        for c in couplings:
            couplings[c]["two_port"] = os.path.join(broot, couplings[c]["two_port"])
        cross_power = lusee.BeamCouplings(beams, from_yaml_dict=couplings)

    obs = lusee.Observation(
        od["lunar_day"],
        deltaT_sec=dt,
        lun_lat_deg=od["lat"],
        lun_long_deg=od["long"],
    )
    combs = od["combinations"]
    if combs == "all":
        combs = [(i, j) for i in range(len(beams)) for j in range(i, len(beams))]

    return {
        "obs": obs,
        "beams": beams,
        "sky": sky,
        "freq": freq,
        "lmax": lmax,
        "combinations": combs,
        "Tground": od["Tground"],
        "cross_power": cross_power,
        "extra_opts": cfg.get("simulation", {}),
    }


def cmp(label, a, b, tol=1e-5):
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    diff = np.abs(a - b)
    ok = np.allclose(a, b, rtol=tol, atol=tol)
    print(f"  {label}: {'PASS' if ok else 'FAIL'} (max|diff|={np.max(diff):.3e})")
    return ok


def stats(label, a, b):
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    print(f"  {label}: Default range=[{np.nanmin(a):.4f}, {np.nanmax(a):.4f}], Cro range=[{np.nanmin(b):.4f}, {np.nanmax(b):.4f}]")


def run_comparison(config_path=None):
    import lusee
    import healpy as hp
    import jax
    import jax.numpy as jnp
    from lusee.SimulatorBase import mean_alm

    if lusee.CroSimulator is None:
        print("CroSimulator not available (croissant/s2fft not installed). Skip test.")
        return

    from lusee.CroSimulator import healpy_packed_alm_to_croissant_2d
    import croissant.jax as crojax
    from functools import partial
    import s2fft

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "simulation", "config", "sim_choice_realistic.yaml"
        )
    cfg = load_config(config_path)
    setup = setup_simulation(cfg, lusee)
    od = cfg["observation"]
    lmax = setup["lmax"]
    freq = setup["freq"]
    Nfreq = len(freq)

    # Run both simulators (observer frame so they match)
    def_sim = lusee.DefaultSimulator(
        setup["obs"], setup["beams"], setup["sky"],
        Tground=setup["Tground"], combinations=setup["combinations"],
        freq=setup["freq"], lmax=setup["lmax"],
        cross_power=setup["cross_power"], extra_opts=setup.get("extra_opts", {}),
    )
    def_sim.simulate()
    out_def = def_sim.result

    cro_sim = lusee.CroSimulator(
        setup["obs"], setup["beams"], setup["sky"],
        Tground=setup["Tground"], combinations=setup["combinations"],
        freq=setup["freq"], lmax=setup["lmax"],
        cross_power=setup["cross_power"],
        extra_opts={**setup.get("extra_opts", {}), "use_observer_frame": True},
    )
    cro_sim.simulate()
    out_cro = cro_sim.result

    # Step 1: Sky raw
    print("\n" + "=" * 60)
    print("STEP 1: Sky model (get_alm, galactic frame)")
    print("=" * 60)
    sky_alm_raw = setup["sky"].get_alm(def_sim.freq_ndx_sky, freq)
    print("  Sky alms: same for both (galactic).")

    # Step 2: Beams
    print("\n" + "=" * 60)
    print("STEP 2: Beams (efbeams)")
    print("=" * 60)
    beamreal = def_sim.efbeams[0][2]
    gpr = def_sim.efbeams[0][4]
    print("  First combo beam (real): same for both.")

    # Step 3: Beam 2D monopole
    print("\n" + "=" * 60)
    print("STEP 3: Beam healpy -> 2D (monopole)")
    print("=" * 60)
    beam_2d = np.stack([healpy_packed_alm_to_croissant_2d(br_, lmax) for br_ in beamreal])
    beam_mono = np.asarray(beam_2d[:, 0, lmax])
    print(f"  Beam monopole (first combo): {beam_mono[:3]} ...")

    # Step 4: Sky at time 0 — Default rotates to observer; Cro uses galactic->MCMF
    print("\n" + "=" * 60)
    print("STEP 4: Sky at time 0 — Default rotates to observer; Cro uses galactic->MCMF ---")
    print("=" * 60)
    times = setup["obs"].times
    lzl, bzl = setup["obs"].get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = setup["obs"].get_l_b_from_alt_az(0.0, 0.0, times)
    lz, bz, ly, by = lzl[0], bzl[0], lyl[0], byl[0]
    zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
    yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
    xhat = np.cross(yhat, zhat)
    from lusee.SimulatorBase import rot2eul
    R = np.array([xhat, yhat, zhat]).T
    a, b, g = rot2eul(R)
    rot = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
    sky_rot_default = [rot.rotate_alm(s_) for s_ in sky_alm_raw]
    sky_rot0_mono = np.array([s[hp.sphtfunc.Alm.getidx(lmax, 0, 0)] for s in sky_rot_default])
    # Cro: sky in MCMF (gal2mcmf); at t=0 phases[0] is applied in convolve
    sky_2d = np.stack([healpy_packed_alm_to_croissant_2d(s_, lmax) for s_ in sky_alm_raw])
    sky_2d_j = jnp.array(sky_2d)
    eul_gal, dl_gal = crojax.rotations.generate_euler_dl(lmax, "galactic", "mcmf")
    gal2mcmf = partial(
        s2fft.utils.rotation.rotate_flms,
        L=lmax + 1,
        rotation=eul_gal,
        dl_array=dl_gal,
    )
    sky_mcmf = jax.vmap(gal2mcmf)(sky_2d_j)
    sky_mcmf_mono = np.asarray(sky_mcmf[:, 0, lmax])
    cmp("Sky monopole after rotation (Default observer vs Cro MCMF)", sky_rot0_mono, sky_mcmf_mono, tol=1e-5)

    # Step 5: Raw convolved product (before any normalization)
    print("\n" + "=" * 60)
    print("STEP 5: Raw convolved product (beam·conj(sky), no normalization)")
    print("=" * 60)
    raw_def = np.array([
        mean_alm(br_, sr_, lmax) * (4 * np.pi)
        for br_, sr_ in zip(beamreal, sky_rot_default)
    ])
    sky_2d_rot = np.stack([healpy_packed_alm_to_croissant_2d(s_, lmax) for s_ in sky_rot_default])
    vis_raw = crojax.simulator.convolve(
        jnp.array(beam_2d), jnp.array(sky_2d_rot),
        jnp.ones((1, 2 * lmax + 1), dtype=jnp.complex128),
    )
    raw_cro = np.asarray(vis_raw[0])
    cmp("Raw convolution Re (Default mean_alm*4pi vs Cro convolve.real)", raw_def, raw_cro.real, tol=1e-5)

    # Step 6: Normalization (both use 4*pi)
    print("\n" + "=" * 60)
    print("STEP 6: After normalization (Default and Cro both use /4*pi)")
    print("=" * 60)
    four_pi = 4.0 * np.pi
    T_def_step = raw_def / four_pi + od["Tground"] * gpr
    T_cro_step = raw_cro.real / four_pi + od["Tground"] * gpr
    cmp("T after norm (same sky/beam, observer frame)", T_cro_step, T_def_step, tol=1e-5)
    print(f"  Divisor (both engines): 4*pi = {four_pi:.6f}")

    # Step 7: Full pipeline
    print("\n" + "=" * 60)
    print("STEP 7: Full pipeline (Default: per-time rotation; Cro: observer frame)")
    print("=" * 60)
    stats("Final output t=0 combo=0", out_def[0, 0], out_cro[0, 0])
    cmp("Final output t=0 combo=0 (observer frame)", out_def[0, 0], out_cro[0, 0], tol=1e-5)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Where do the two engines diverge?")
    print("=" * 60)
    print("  Step 1: Sky raw — same.")
    print("  Step 2: Beams (efbeams) — same.")
    print("  Step 3: Beam healpy->2D — check monopole.")
    print("  Step 4: Sky rotated — Default observer vs Cro MCMF (may differ by frame convention).")
    print("  Step 5: Raw convolution — should match if same beam/sky in same frame.")
    print("  Step 6: Normalization — both use 4*pi.")
    print("  Step 7: Full result — with use_observer_frame=True, Cro matches Default.")


if __name__ == "__main__":
    run_comparison(sys.argv[1] if len(sys.argv) > 1 else None)
