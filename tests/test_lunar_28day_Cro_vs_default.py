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
from datetime import datetime, timedelta

os.environ["JAX_ENABLE_X64"] = "True"

import jax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp
from pathlib import Path
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic
from astropy import units as u
from lusee.frequencies import canonical_frequencies, canonical_frequency_indices

# luseepy package root (parent of tests/)
_LUSEEPY_ROOT = Path(__file__).resolve().parent.parent


def _env_positive_float(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive float, got {raw!r}") from exc
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _env_nonnegative_float(name, default):
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative float, got {raw!r}") from exc
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _time_range_from_days(start_text, days):
    start = datetime.fromisoformat(start_text)
    end = start + timedelta(days=days)
    return f"{start:%Y-%m-%d %H:%M:%S} to {end:%Y-%m-%d %H:%M:%S}"


_LUNAR_REGRESSION_DAYS = _env_positive_float("LUSEE_LUNAR_DAY_REGRESSION_DAYS", 0.25)
_LUNAR_REGRESSION_START_OFFSET_DAYS = _env_nonnegative_float(
    "LUSEE_LUNAR_DAY_REGRESSION_START_OFFSET_DAYS",
    0.0,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("days", "start_offset_days"),
    [
        pytest.param(
            _LUNAR_REGRESSION_DAYS,
            _LUNAR_REGRESSION_START_OFFSET_DAYS,
            id=f"{_LUNAR_REGRESSION_DAYS:g}d_offset{_LUNAR_REGRESSION_START_OFFSET_DAYS:g}",
        )
    ],
)
def test_lunar_day_28_single_source(days, start_offset_days):
    """run sim for 28 days, for a single pixel source.

    Note: Output is the same at every frequency because (1) SingleSourceHealpixSky
    uses the same map at all frequencies, and
    (2) BeamGauss with one_over_freq_scaling=False uses the same beam at all
    frequencies. Frequency is handled correctly in both simulators; use a
    frequency-dependent sky.

    Default CI/local regression length is controlled by
    ``LUSEE_LUNAR_DAY_REGRESSION_DAYS`` and defaults to 0.25 days.

    The start offset is controlled by
    ``LUSEE_LUNAR_DAY_REGRESSION_START_OFFSET_DAYS`` and defaults to 0.
    Run the original 28-day case with ``LUSEE_LUNAR_DAY_REGRESSION_DAYS=28``.
    """
    import lusee
    import matplotlib
    matplotlib.use("Agg")

    # Source on the inertial ecliptic plane (lat=0); conversion uses mean obliquity, not geocentric frame
    ecl_lon_deg, ecl_lat_deg = 90.0, 00.0
    c = SkyCoord(lon=ecl_lon_deg * u.deg, lat=ecl_lat_deg * u.deg, frame=GeocentricMeanEcliptic)
    ra_deg, dec_deg = c.icrs.ra.deg, c.icrs.dec.deg
    l_deg, b_deg = c.galactic.l.deg, c.galactic.b.deg

    time_start = datetime.fromisoformat("2025-03-01 00:00:00") + timedelta(
        days=start_offset_days
    )
    deltaT_sec = 7200.0  # 2 hours
    obs = lusee.Observation(
        _time_range_from_days(time_start.strftime("%Y-%m-%d %H:%M:%S"), days),
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
    freq = canonical_frequencies(canonical_frequency_indices(start_idx=0, stop_idx=50, step_idx=5))

    # Single-pixel sky (equatorial) wrapped to galactic
    sky = lusee.sky.SingleSourceHealpixSky(l_deg=l_deg, b_deg=b_deg, Nside=nside, freq=freq)
    np_sky = lusee.NpWrapper(sky)
    np_beam = lusee.NpWrapper(beam)
    beams = [beam]
    np_beams = [np_beam]

    # Run DefaultSimulator (topo frame)
    def_sim = lusee.TopoNumpySimulator(
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

    # Run JaxSimulator (topo frame)
    jax_sim = lusee.TopoJaxSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={
            "plot_sky_and_beam": False,
            "time_batch_size": None,
        },
    )
    t0 = time.perf_counter()
    jax_result = jax_sim.simulate(times=times)
    jax.block_until_ready(jax_result)
    jax_time = time.perf_counter() - t0

    # Run CroSimulator (MEPA frame)
    if lusee.CroSimulator is None:
        pytest.skip(
            "CroSimulator requires optional croissant-sim, s2fft, and spiceypy dependencies"
        )
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations= [(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={
            "plot_sky_and_beam": False,
        },
    )
    t0 = time.perf_counter()
    cro_result = cro_sim.simulate(times=times)
    jax.block_until_ready(cro_result)
    cro_time = time.perf_counter() - t0

    print(
        f"days={days:g} "
        f"start_offset_days={start_offset_days:g} "
        f"n_times={len(times)} "
        f"default_sec={def_time:.3f} "
        f"jaxsim_sec={jax_time:.3f} "
        f"croissant_sec={cro_time:.3f}"
    )

    assert cro_sim.result.shape == def_sim.result.shape
    assert jax_sim.result.shape == def_sim.result.shape

    np_cro_result = np.asarray(cro_sim.result)
    np_def_result = np.asarray(def_sim.result)
    np_jax_result = np.asarray(jax_sim.result)

    diff_norm = np.linalg.norm(np_jax_result - np_def_result)
    rel = diff_norm / np.linalg.norm(np_def_result)
    assert rel < 1e-9

    diff_norm = np.linalg.norm(np_cro_result - np_def_result)
    rel = diff_norm / np.linalg.norm(np_def_result)
    assert rel < 5e-3

    diff_norm = np.linalg.norm(np_cro_result - np_jax_result)
    rel = diff_norm / np.linalg.norm(np_jax_result)
    assert rel < 5e-3


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
