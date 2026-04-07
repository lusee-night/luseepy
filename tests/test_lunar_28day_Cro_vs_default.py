#!/usr/bin/env python3
"""
28-day lunar test:
Single source simulation in the topo frame (DefaultSimulator), the JAX topo
frame (JaxSimulator), and the MEPA frame (CroSimulator).

Usage:
  python tests/test_lunar_day_sims.py
  pytest tests/test_lunar_day_sims.py -v
"""

import os
import sys
import numpy as np
import pytest
import time
import platform
from importlib import metadata as importlib_metadata

os.environ["JAX_ENABLE_X64"] = "True"

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp
from pathlib import Path
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic
from astropy import units as u

# luseepy package root (parent of tests/)
_LUSEEPY_ROOT = Path(__file__).resolve().parent.parent


def _debug_enabled():
    return os.environ.get("LUSEE_JAX_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _print_debug(message):
    if _debug_enabled():
        print(f"lunar28 debug: {message}", flush=True)


def _print_array_summary(label, value, max_items=8):
    if not _debug_enabled():
        return
    arr = np.asarray(value)
    flat = arr.reshape(-1) if arr.shape else arr.reshape(1)
    sample = flat[:max_items].tolist()
    _print_debug(
        f"{label}: shape={arr.shape} dtype={arr.dtype} size={arr.size} sample={sample}"
    )


def _print_runtime_summary():
    if not _debug_enabled():
        return
    _print_debug(f"python={sys.version}")
    _print_debug(f"platform={platform.platform()}")
    _print_debug(f"uname={platform.uname()}")
    for key in [
        "JAX_ENABLE_X64",
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "JAX_DISABLE_JIT",
        "JAX_LOG_COMPILES",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONFAULTHANDLER",
        "LUSEE_JAX_DEBUG",
        "LUSEE_JAX_DEBUG_HLO_CHARS",
    ]:
        _print_debug(f"env[{key}]={os.environ.get(key)}")
    for pkg in [
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "s2fft",
        "croissant-sim",
        "healpy",
        "astropy",
        "fitsio",
        "lunarsky",
    ]:
        try:
            version = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            version = "<missing>"
        _print_debug(f"{pkg}={version}")
    _print_debug(f"jax.default_backend={jax.default_backend()}")
    _print_debug(f"jax.device_count={jax.device_count()} process_count={jax.process_count()}")
    _print_debug(f"jax.devices={jax.devices()}")
    _print_debug(f"jax.local_devices={jax.local_devices()}")
    _print_debug(f"jax_enable_x64={jax.config.jax_enable_x64}")


def test_lunar_day_28_single_source():
    """run sim for 28 days, for a single pixel source.

    Note: Output is the same at every frequency because (1) SingleSourceHealpixSky
    uses the same map at all frequencies, and
    (2) BeamGauss with one_over_freq_scaling=False uses the same beam at all
    frequencies. Frequency is handled correctly in both simulators; use a
    frequency-dependent sky
    """
    import lusee
    import matplotlib
    matplotlib.use("Agg")
    _print_runtime_summary()

    # Source on the inertial ecliptic plane (lat=0); conversion uses mean obliquity, not geocentric frame
    ecl_lon_deg, ecl_lat_deg = 90.0, 00.0
    c = SkyCoord(lon=ecl_lon_deg * u.deg, lat=ecl_lat_deg * u.deg, frame=GeocentricMeanEcliptic)
    ra_deg, dec_deg = c.icrs.ra.deg, c.icrs.dec.deg
    l_deg, b_deg = c.galactic.l.deg, c.galactic.b.deg

    time_start = "2025-03-01 00:00:00"
    time_end = "2025-03-29 00:00:00"  # 28 days later
    deltaT_sec = 7200.0  # 2 hours
    obs = lusee.Observation(
        f"{time_start} to {time_end}",
        deltaT_sec=deltaT_sec,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    nside = 32
    lmax = 3*nside - 1
    sigma_deg = 20.0
    Tground = 0.0
    beam = lusee.BeamGauss(
        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=sigma_deg,
        one_over_freq_scaling=False,
    )
    freq = beam.freq[::5]

    # Single-pixel sky (equatorial) wrapped to galactic
    sky = lusee.sky.SingleSourceHealpixSky(l_deg=l_deg, b_deg=b_deg, Nside=nside, freq=freq)
    np_sky = lusee.NpWrapper(sky)
    np_beam = lusee.NpWrapper(beam)
    beams = [beam]
    np_beams = [np_beam]
    debug_extra_opts = {
        "plot_sky_and_beam": False,
        "profile_timing": _debug_enabled(),
        "debug_jax": _debug_enabled(),
        "debug_hlo_chars": int(os.environ.get("LUSEE_JAX_DEBUG_HLO_CHARS", "2500")),
    }

    _print_debug(
        f"test config: ra_deg={ra_deg:.6f} dec_deg={dec_deg:.6f} "
        f"l_deg={l_deg:.6f} b_deg={b_deg:.6f}"
    )
    _print_debug(
        f"observation: start={time_start} end={time_end} deltaT_sec={deltaT_sec} "
        f"ntimes={len(times)} nside={nside} lmax={lmax} sigma_deg={sigma_deg}"
    )
    if len(times) > 0:
        _print_debug(f"time range: first={times[0]} last={times[-1]}")
    _print_array_summary("freq", freq)
    _print_array_summary("beam.freq", beam.freq)
    _print_debug(
        f"beam metadata: id={beam.id} "
        f"one_over_freq_scaling={getattr(beam, 'one_over_freq_scaling', '<missing>')}"
    )
    _print_debug(
        f"sky metadata: class={type(sky).__name__} frame={sky.frame} Nside={sky.Nside}"
    )

    # Run DefaultSimulator (topo frame)
    _print_debug("starting DefaultSimulator")
    def_sim = lusee.DefaultSimulator(
        obs, np_beams, np_sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"plot_sky_and_beam": False},
    )
    t0 = time.perf_counter()
    def_sim.simulate(times=times)
    def_time = time.perf_counter() - t0
    _print_debug(f"DefaultSimulator complete in {def_time:.3f} s")
    _print_array_summary("def_sim.result", def_sim.result)

    # Run JaxSimulator (topo frame)
    _print_debug("starting JaxSimulator")
    jax_sim = lusee.JaxSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts=debug_extra_opts,
    )
    t0 = time.perf_counter()
    _print_debug("calling jax_sim.simulate")
    jax_result = jax_sim.simulate(times=times)
    _print_debug("jax_sim.simulate returned, blocking on result")
    jax.block_until_ready(jax_result)
    jax_time = time.perf_counter() - t0
    _print_debug(f"JaxSimulator complete in {jax_time:.3f} s")
    _print_array_summary("jax_sim.result", jax_sim.result)

    # Run CroSimulator (MEPA frame)
    if lusee.CroSimulator is None:
        pytest.skip("CroSimulator requires optional croissant and s2fft dependencies")
    _print_debug("starting CroSimulator")
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations= [(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts=debug_extra_opts,
    )
    t0 = time.perf_counter()
    cro_result = cro_sim.simulate(times=times)
    _print_debug("cro_sim.simulate returned, blocking on result")
    jax.block_until_ready(cro_result)
    cro_time = time.perf_counter() - t0
    _print_debug(f"CroSimulator complete in {cro_time:.3f} s")
    _print_array_summary("cro_sim.result", cro_sim.result)

    print(
        f"n_times={len(times)} "
        f"default_sec={def_time:.3f} "
        f"jaxsim_sec={jax_time:.3f} "
        f"croissant_sec={cro_time:.3f}"
    , flush=True)

    assert cro_sim.result.shape == def_sim.result.shape
    assert jax_sim.result.shape == def_sim.result.shape

    # to avoid OOM on Github, delete the memory

    np_cro_result = np.asarray(cro_sim.result)
    del cro_sim
    np_def_result = np.asarray(def_sim.result)
    del def_sim
    np_jax_result = np.asarray(jax_sim.result)
    del jax_sim
    _print_array_summary("np_cro_result", np_cro_result)
    _print_array_summary("np_def_result", np_def_result)
    _print_array_summary("np_jax_result", np_jax_result)

    diff_norm = np.linalg.norm(np_jax_result - np_def_result)
    rel = diff_norm / np.linalg.norm(np_def_result)
    _print_debug(f"relative diff jax vs default = {rel:.6e}")
    assert rel < 1e-9

    diff_norm = np.linalg.norm(np_cro_result - np_def_result)
    rel = diff_norm / np.linalg.norm(np_def_result)
    _print_debug(f"relative diff cro vs default = {rel:.6e}")
    assert rel < 5e-3

    diff_norm = np.linalg.norm(np_cro_result - np_jax_result)
    rel = diff_norm / np.linalg.norm(np_jax_result)
    _print_debug(f"relative diff cro vs jax = {rel:.6e}")
    assert rel < 5e-3


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
