#!/usr/bin/env python3
"""
Transit single-source lunar simulations.

Six combinations of beam pointing and source position; both DefaultSimulator
and CroSimulator are run for each case.  A separate FITS pair is written per case.

Cases
-----
i)   beam dec=30, phi=90,  source ecl_lon=70, ecl_lat=0
ii)  beam dec=60, phi=90,  source ecl_lon=70, ecl_lat=0
iii) beam dec=60, phi=270, source ecl_lon=70, ecl_lat=0
iv)  beam dec=30, phi=270, source ecl_lon=70, ecl_lat=0
v)   beam dec=60, phi=0,   source ecl_lon=70, ecl_lat=30
vi)  beam dec=60, phi=180, source ecl_lon=70, ecl_lat=-30

Usage
-----
  python tests/test_transit_cases.py
  pytest tests/test_transit_cases.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic
from astropy import units as u
from lusee.frequencies import canonical_frequencies, canonical_frequency_indices


CASES = [
    dict(name="i",   beam_dec=30, beam_phi=90,  ecl_lon=70, ecl_lat=0),
    dict(name="ii",  beam_dec=60, beam_phi=90,  ecl_lon=70, ecl_lat=0),
    dict(name="iii", beam_dec=60, beam_phi=270, ecl_lon=70, ecl_lat=0),
    dict(name="iv",  beam_dec=30, beam_phi=270, ecl_lon=70, ecl_lat=0),
    dict(name="v",   beam_dec=60, beam_phi=0,   ecl_lon=70, ecl_lat=30),
    dict(name="vi",  beam_dec=60, beam_phi=180, ecl_lon=70, ecl_lat=-30),
]

pytestmark = pytest.mark.manual

TIME_START   = "2025-03-01 00:00:00"
TIME_END     = "2025-03-29 00:00:00"
DELTA_T_SEC  = 7200.0
NSIDE        = 32
LMAX         = 3 * NSIDE - 1
SIGMA_DEG    = 20.0
TGROUND      = 0.0
FREQ         = canonical_frequencies(canonical_frequency_indices(start_idx=0, stop_idx=50, step_idx=5))
MIN_TRANSIT_AMP = 0.0025


def _ecl_to_gal(ecl_lon_deg, ecl_lat_deg):
    c = SkyCoord(
        lon=ecl_lon_deg * u.deg,
        lat=ecl_lat_deg * u.deg,
        frame=GeocentricMeanEcliptic,
    )
    return c.galactic.l.deg, c.galactic.b.deg


def test_transit_cases():
    import lusee
    import matplotlib
    matplotlib.use("Agg")

    if lusee.CroSimulator is None:
        pytest.skip("croissant-sim, s2fft, and spiceypy are not installed")

    obs = lusee.Observation(
        f"{TIME_START} to {TIME_END}",
        deltaT_sec=DELTA_T_SEC,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    times = obs.times
    out_dir = Path(__file__).resolve().parent

    for case in CASES:
        name = case["name"]
        print(f"\n--- Case {name}: beam(alt={case['beam_dec']}, az={case['beam_phi']})  "
              f"src(ecl_lon={case['ecl_lon']}, ecl_lat={case['ecl_lat']}) ---")

        l_deg, b_deg = _ecl_to_gal(case["ecl_lon"], case["ecl_lat"])
        sky = lusee.sky.SingleSourceHealpixSky(
            l_deg=l_deg, b_deg=b_deg, Nside=NSIDE, freq=FREQ
        )
        beam = lusee.BeamGauss(
            alt_deg=float(case["beam_dec"]),
            az_deg=float(case["beam_phi"]),
            sigma_deg=SIGMA_DEG,
            one_over_freq_scaling=False,
            id="beam",
        )
        beams = [beam]

        # --- DefaultSimulator ---
        def_sim = lusee.TopoNumpySimulator(
            obs, beams, sky,
            Tground=TGROUND,
            combinations=[(0, 0)],
            freq=FREQ,
            lmax=LMAX,
        )
        def_sim.simulate(times=times)
        def_path = str(out_dir / f"sim_output_default_case_{name}.fits")
        def_sim.write_fits(def_path)

        # --- CroSimulator ---
        cro_sim = lusee.CroSimulator(
            obs, beams, sky,
            Tground=TGROUND,
            combinations=[(0, 0)],
            freq=FREQ,
            lmax=LMAX,
        )
        cro_sim.simulate(times=times)
        cro_path = str(out_dir / f"sim_output_cro_case_{name}.fits")
        cro_sim.write_fits(cro_path)

        # freq-averaged time stream: shape (Ntimes,)
        def_freq_avg = np.mean(def_sim.result[:, 0, :], axis=1)
        cro_freq_avg = np.mean(cro_sim.result[:, 0, :], axis=1)

        def_max = def_freq_avg.max()
        cro_max = cro_freq_avg.max()
        print(f"  default max={def_max:.4f}  cro max={cro_max:.4f}")

        if not def_max > MIN_TRANSIT_AMP:
            print(f"Case {name}: Default peak {def_max:.4f} < {MIN_TRANSIT_AMP}")
        if not cro_max > MIN_TRANSIT_AMP:
            print(f"Case {name}: Cro peak {cro_max:.4f} < {MIN_TRANSIT_AMP}")



if __name__ == "__main__":
    test_transit_cases()
    print("test_transit_cases: passed.")
