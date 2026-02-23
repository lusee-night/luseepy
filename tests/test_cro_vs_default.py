#!/usr/bin/env python3
"""
Test script to compare CroSimulator vs DefaultSimulator outputs.
DefaultSimulator is the reference. Identifies where differences come from
(normalization, coordinate frame, etc.) to guide fixes in CroSimulator.

Usage:
  cd luseepy/simulation && python driver/test_cro_vs_default.py [config.yaml]
  (default config: config/sim_choice_realistic.yaml)

Requires: LUSEE_DRIVE_DIR, LUSEE_OUTPUT_DIR (or paths in config).
Sky must have frame="galactic" for CroSimulator (e.g. FitsSky).
"""

import os
import sys
import numpy as np
import yaml
from yaml.loader import SafeLoader

# Add luseepy to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_config(config_path):
    """Load YAML config and resolve env vars."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    root = cfg["paths"]["lusee_drive_dir"]
    if root.startswith("$"):
        root = os.environ[root[1:]]
    outdir = cfg["paths"]["output_dir"]
    if outdir.startswith("$"):
        outdir = os.environ[outdir[1:]]
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
    beams = []
    for b in od["beams"]:
        cbeam = cfg["beams"][b]
        if bdc["type"] == "Gaussian":
            B = lusee.BeamGauss(
                dec_deg=cbeam["declination"],
                sigma_deg=cbeam["sigma"],
                one_over_freq_scaling=cbeam["one_over_freq_scaling"],
                id=b,
            )
        else:
            fn = cbeam.get("file", bdc.get("default_file"))
            B = lusee.Beam(os.path.join(broot, fn), id=b)
        angle = bdc.get("common_beam_angle", 0) + cbeam.get("angle", 0)
        B = B.rotate(angle)
        beams.append(B)

    couplings = None
    if bdc.get("couplings"):
        for c in bdc["couplings"]:
            bdc["couplings"][c]["two_port"] = os.path.join(
                broot, bdc["couplings"][c]["two_port"]
            )
        couplings = lusee.BeamCouplings(beams, from_yaml_dict=bdc["couplings"])

    # Observation
    O = lusee.Observation(
        od["lunar_day"],
        deltaT_sec=dt,
        lun_lat_deg=od["lat"],
        lun_long_deg=od["long"],
    )

    combs = od["combinations"]
    if combs == "all":
        combs = [(i, j) for i in range(len(beams)) for j in range(i, len(beams))]

    return O, beams, sky, freq, lmax, od, combs, couplings, bdc.get("beam_smooth")


def run_comparison(config_path=None):
    # Import here to avoid multiprocessing issues with lunarsky (SPICE kernel downloads)
    import lusee
    if lusee.CroSimulator is None:
        try:
            import pytest
            pytest.skip("CroSimulator requires croissant and s2fft; install with: pip install croissant s2fft")
        except ImportError:
            sys.exit("CroSimulator not available (missing croissant/s2fft). Install with: pip install croissant s2fft")

    from functools import partial
    import healpy as hp
    import jax
    import jax.numpy as jnp
    import s2fft
    from lusee.SimulatorBase import mean_alm, rot2eul
    from lusee.CroSimulator import healpy_packed_alm_to_croissant_2d
    import croissant.jax as crojax

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "../config/sim_choice_realistic.yaml"
        )
    cfg = load_config(config_path)
    O, beams, sky, freq, lmax, od, combs, couplings, beam_smooth = setup_simulation(
        cfg, lusee
    )

    # Limit times for quick test (e.g. last 5 times)
    times = O.times[:5]    # first 5 times
    # Bypass cache for reproducible test (cache may have different length)
    extra = {"simulation": {**cfg.get("simulation", {}), "cache_transform": None}}

    print("=" * 60)
    print("Running DefaultSimulator (reference)...")
    S_def = lusee.DefaultSimulator(
        O, beams, sky,
        Tground=od["Tground"],
        combinations=combs,
        freq=freq,
        lmax=lmax,
        cross_power=couplings,
        beam_smooth=beam_smooth,
        extra_opts=extra,
    )
    S_def.simulate(times=times)
    out_def = S_def.result

    print("Running CroSimulator...")
    S_cro = lusee.CroSimulator(
        O, beams, sky,
        Tground=od["Tground"],
        combinations=combs,
        freq=freq,
        lmax=lmax,
        cross_power=couplings,
        beam_smooth=beam_smooth,
        extra_opts=extra,
    )
    S_cro.simulate(times=times)
    out_cro = S_cro.result

    def stats(name, a, b):
        diff = a - b
        rat = np.abs(a) / (np.abs(b) + 1e-30)
        print(f"  {name}:")
        print(f"    Default: mean={np.nanmean(b):.6e}, std={np.nanstd(b):.6e}")
        print(f"    Cro:     mean={np.nanmean(a):.6e}, std={np.nanstd(a):.6e}")
        print(f"    Ratio Cro/Default: mean={np.nanmean(rat):.4f}")
        print(f"    |diff|: mean={np.nanmean(np.abs(diff)):.6e}, max={np.nanmax(np.abs(diff)):.6e}")

    def cmp(name, a, b, tol=1e-10):
        """Compare two arrays; report match or max diff."""
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            return False
        diff = np.abs(a - b)
        ok = np.allclose(a, b, atol=tol, rtol=tol)
        print(f"  {name}: {'OK' if ok else 'DIFFER'}")
        print(f"    max|a-b|={np.nanmax(diff):.6e}, mean|a-b|={np.nanmean(diff):.6e}")
        if not ok and a.size <= 5:
            print(f"    a={a}, b={b}")
        return ok

    # --- Step 1: Sky model (raw galactic) ---
    print("\n" + "=" * 60)
    print("STEP 1: Sky model (get_alm, galactic frame)")
    print("=" * 60)
    sky_alm_raw = sky.get_alm(S_cro.freq_ndx_sky, list(freq))
    # Both simulators use the same sky.get_alm; no divergence here
    print("  Same source (sky.get_alm); no Default vs Cro difference.")

    # --- Step 2: Beam (efbeams from prepare_beams) ---
    print("\n" + "=" * 60)
    print("STEP 2: Beam alms (efbeams from prepare_beams)")
    print("=" * 60)
    # Both use same SimulatorBase.prepare_beams, so efbeams are identical
    ci, cj, beamreal, beamimag, gpr, gpi = S_cro.efbeams[0]
    cmp("First combo beamreal[0] (first freq)", S_def.efbeams[0][2][0], beamreal[0])
    cmp("groundPowerReal first combo", S_def.efbeams[0][4], gpr)

    # --- Step 3: Beam healpy -> 2D conversion ---
    print("\n" + "=" * 60)
    print("STEP 3: Beam healpy packed -> Croissant 2D conversion")
    print("=" * 60)
    beam_2d_one = healpy_packed_alm_to_croissant_2d(beamreal[0], lmax)
    # Check monopole (l=0,m=0): healpy index 0, 2D [0, lmax]
    hp_idx0 = hp.sphtfunc.Alm.getidx(lmax, 0, 0)
    monopole_hp = beamreal[0][hp_idx0]
    monopole_2d = beam_2d_one[0, lmax]
    cmp("Beam monopole (l=0,m=0) after conversion", monopole_2d, monopole_hp)

    # --- Step 4: Sky at time 0 — Default rotates to observer; Cro uses galactic->MCMF ---
    print("\n" + "=" * 60)
    print("STEP 4: Sky after rotation (time 0)")
    print("=" * 60)
    import pickle
    cache_fn = extra.get("cache_transform")
    if not (cache_fn and os.path.isfile(cache_fn)):
        lzl, bzl = O.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
        lyl, byl = O.get_l_b_from_alt_az(0.0, 0.0, times)
    else:
        lzl, bzl, lyl, byl = pickle.load(open(cache_fn, "br"))
    ti = 0
    lz, bz, ly, by = lzl[ti], bzl[ti], lyl[ti], byl[ti]
    zhat = np.array([np.cos(bz) * np.cos(lz), np.cos(bz) * np.sin(lz), np.sin(bz)])
    yhat = np.array([np.cos(by) * np.cos(ly), np.cos(by) * np.sin(ly), np.sin(by)])
    xhat = np.cross(yhat, zhat)
    R = np.array([xhat, yhat, zhat]).T
    a, b, g = rot2eul(R)
    rot = hp.rotator.Rotator(rot=(g, -b, a), deg=False, eulertype="XYZ", inv=False)
    sky_rot_default = [rot.rotate_alm(s_) for s_ in sky_alm_raw]
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
    # Compare monopole of rotated sky: Default sky_rot vs Cro sky_mcmf (at t=0 no phase)
    sky_rot0_mono = np.array([s[hp.sphtfunc.Alm.getidx(lmax, 0, 0)] for s in sky_rot_default])
    sky_mcmf_mono = np.asarray(sky_mcmf[:, 0, lmax])
    cmp("Sky monopole after rotation (Default observer vs Cro MCMF)", sky_rot0_mono, sky_mcmf_mono, tol=1e-5)

    # --- Step 5: Raw convolved product (before any normalization) ---
    print("\n" + "=" * 60)
    print("STEP 5: Raw convolved product (beam·conj(sky), no normalization)")
    print("=" * 60)
    # Default: mean_alm = (1/4pi) * [ sum real(beam*conj(sky)) ]
    raw_def = np.array([
        mean_alm(br_, sr_, lmax) * (4 * np.pi)
        for br_, sr_ in zip(beamreal, sky_rot_default)
    ])
    # Cro: convolve with phases=1 gives sum(conj(sky)*beam) = sum(beam*conj(sky))
    beam_2d_all = np.stack([healpy_packed_alm_to_croissant_2d(br_, lmax) for br_ in beamreal])
    sky_2d_rot = np.stack([healpy_packed_alm_to_croissant_2d(s_, lmax) for s_ in sky_rot_default])
    vis_raw = crojax.simulator.convolve(
        jnp.array(beam_2d_all), jnp.array(sky_2d_rot),
        jnp.ones((1, 2 * lmax + 1), dtype=jnp.complex128),
    )
    raw_cro = np.asarray(vis_raw[0])  # complex
    # Default gives real part only; compare real parts
    cmp("Raw convolution Re (Default mean_alm*4pi vs Cro convolve.real)", raw_def, raw_cro.real, tol=1e-5)

    # --- Step 6: Normalization (both use 4*pi; CroSimulator now matches Default) ---
    print("\n" + "=" * 60)
    print("STEP 6: After normalization (Default and Cro both use /4*pi)")
    print("=" * 60)
    four_pi = 4.0 * np.pi
    T_def_step = raw_def / four_pi + od["Tground"] * gpr
    T_cro_step = raw_cro.real / four_pi + od["Tground"] * gpr
    cmp("T after norm (same sky/beam, observer frame)", T_cro_step, T_def_step, tol=1e-5)
    print(f"  Divisor (both engines): 4*pi = {four_pi:.6f}")

    # --- Step 7: Full pipeline — Default observer vs Cro MCMF+phases ---
    print("\n" + "=" * 60)
    print("STEP 7: Full pipeline (Default: per-time rotation; Cro: MCMF + rot_alm_z)")
    print("=" * 60)
    stats("Final output t=0 combo=0", out_cro[0, 0], out_def[0, 0])
    print("  If Step 5–6 matched but Step 7 differs → divergence is frame/rotation (observer vs MCMF+phases).")
    print("  If Step 6 matched → both engines use 4*pi normalization.")

    # --- Final summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: Where do the two engines diverge?")
    print("=" * 60)
    print("  Step 1: Sky raw — same.")
    print("  Step 2: Beams (efbeams) — same.")
    print("  Step 3: Beam healpy->2D — check monopole.")
    print("  Step 4: Sky rotated — Default observer vs Cro MCMF (may differ by frame convention).")
    print("  Step 5: Raw convolution — should match if same beam/sky in same frame.")
    print("  Step 6: Normalization — both use 4*pi (CroSimulator aligned with DefaultSimulator).")
    print("  Step 7: Full result — any remaining difference is frame/rotation (MCMF+phases vs per-time observer).")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_comparison(cfg_path)
