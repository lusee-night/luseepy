#!/usr/bin/env python3
"""
28-day lunar test: 
Single source simulation in the topo frame (DefaultSimulator) and the MEPA frame (CroSimulator).

Usage:
  python tests/test_lunar_day_sims.py
  pytest tests/test_lunar_day_sims.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp
from pathlib import Path
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic
from astropy import units as u

# luseepy package root (parent of tests/)
_LUSEEPY_ROOT = Path(__file__).resolve().parent.parent


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
    import matplotlib.pyplot as plt

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
    freq = np.arange(1, 51, 5, dtype=float)

    #combinations
    Nbeams = 4
    combs = []
    for i in range(Nbeams):
        for j in range(i, Nbeams):
            combs.append((i, j))


    # Single-pixel sky (equatorial) wrapped to galactic
    sky = lusee.sky.SingleSourceHealpixSky(l_deg=l_deg, b_deg=b_deg, Nside=nside, freq=freq)

    beam = lusee.BeamGauss(

        alt_deg=90.0,
        az_deg=0.0,
        sigma_deg=sigma_deg,
        one_over_freq_scaling=False,
    )
    beams = [beam]

    # Run DefaultSimulator (topo frame)
    def_sim = lusee.DefaultSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={
            "plot_sky_and_beam": True,
            "freq_idx_plot": 5,
            "plot_dir": str(_LUSEEPY_ROOT / "simulation" / "output" / "figures"),
            "plot_filename": "sky_beam_healpix_default_single_pixel.png",
        },
    )
    def_sim.simulate(times=times)

    # Run CroSimulator (MEPA frame)
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
    )
    cro_result = cro_sim.simulate()
    cro_sim.plot_sky_beam(
        freq_idx=5,
        save_dir=str(_LUSEEPY_ROOT / "simulation" / "output" / "figures"),
        save_filename="sky_beam_healpix_cro_single_pixel.png",
    )

    out_dir = str(_LUSEEPY_ROOT / "simulation" / "output")
    cro_sim.write_fits(os.path.join(out_dir, "sim_output_cro_singlepixel_28days.fits"), result=cro_result)
    def_sim.write_fits(os.path.join(out_dir, "sim_output_default_singlepixel_28days.fits"))

    np_cro_result = np.asarray(cro_result)
    np_def_result = np.asarray(def_sim.result)
    assert np_cro_result.shape == np_def_result.shape
    diff_norm = np.linalg.norm(np_cro_result - np_def_result)
    rel = diff_norm / np.linalg.norm(np_def_result)
    assert rel < 5e-3


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
