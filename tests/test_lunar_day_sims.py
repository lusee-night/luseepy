#!/usr/bin/env python3
"""
28-day lunar test: compute and plot source (alt, az) from obs, times, gal_dir only.
No simulation output used. Observer-frame and MCMF-frame conventions via get_R_gal_to_topo
and get_rot_mat / get_topo_z_rotation_angles. Observation at lunar latitude 0 deg.

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
from astropy.coordinates import SkyCoord
from astropy import units as u




def _ra_dec_to_galactic_rad(ra_deg, dec_deg):
    """Return (l_rad, b_rad) for the given (ra_deg, dec_deg) in equatorial."""
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    return c.galactic.l.rad, c.galactic.b.rad



# Mean obliquity of the ecliptic (J2000.0), radians. Defines the inertial ecliptic plane
# (same plane for any observer; not tied to Earth or Moon).
_OBLIQUITY_DEG = 23.4392811


def _ecliptic_lon_lat_to_ra_dec_deg(lon_deg, lat_deg):
    """Convert ecliptic (lon, lat) to equatorial (ra, dec) in degrees. Lat=0 is the ecliptic plane.
    Uses the inertial ecliptic plane and mean obliquity (J2000); valid for any observer (Earth or Moon)."""
    lam = np.radians(lon_deg)
    be = np.radians(lat_deg)
    eps = np.radians(_OBLIQUITY_DEG)
    # Ecliptic cartesian (x toward vernal equinox, z north ecliptic pole)
    x_ecl = np.cos(be) * np.cos(lam)
    y_ecl = np.cos(be) * np.sin(lam)
    z_ecl = np.sin(be)
    # Rotate by obliquity: equatorial x = ecliptic x, equatorial y,z from ecliptic y,z
    x_eq = x_ecl
    y_eq = y_ecl * np.cos(eps) - z_ecl * np.sin(eps)
    z_eq = y_ecl * np.sin(eps) + z_ecl * np.cos(eps)
    ra_rad = np.arctan2(y_eq, x_eq)
    dec_rad = np.arcsin(np.clip(z_eq, -1, 1))
    return np.degrees(ra_rad), np.degrees(dec_rad)


def _galactic_dir(l_rad, b_rad):
    """Unit vector in galactic coordinates: (x, y, z) = (cos(b)cos(l), cos(b)sin(l), sin(b))."""
    return np.array([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad),
    ])


class GalacticSkyAdapter:
    """Wraps equatorial HealpixSky and exposes galactic alms (eq->gal rotation)."""
    def __init__(self, equatorial_sky, lmax):
        self._sky = equatorial_sky
        self._lmax = lmax
        self.frame = "galactic"
        self.freq = equatorial_sky.freq
        self.Nside = equatorial_sky.Nside
        self._rot = hp.rotator.Rotator(coord=["C", "G"])
        self._alm_size = hp.sphtfunc.Alm.getsize(lmax)

    def get_alm(self, ndx, freq):
        alms = self._sky.get_alm(ndx, freq)
        alms = np.atleast_2d(alms)
        out = []
        for a in alms:
            r = self._rot.rotate_alm(a)
            out.append(r[: self._alm_size].copy())
        return out



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
    ecl_lon_deg, ecl_lat_deg = 45.0, 0.0
    ra_deg, dec_deg = _ecliptic_lon_lat_to_ra_dec_deg(ecl_lon_deg, ecl_lat_deg)
    l_rad, b_rad = _ra_dec_to_galactic_rad(ra_deg, dec_deg)
    gal_dir = _galactic_dir(l_rad, b_rad)

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
    lmax = 32
    sigma_deg = 10.0
    Tground = 0.0
    freq = np.arange(1, 51, 1, dtype=float)

    #combinations
    Nbeams = 4
    combs = []
    for i in range(Nbeams):
        for j in range(i, Nbeams):
            combs.append((i, j))


    # Single-pixel sky (equatorial) wrapped to galactic
    sky_eq = lusee.sky.SingleSourceHealpixSky(ra_deg, dec_deg, Nside=nside, freq=freq)
    sky = GalacticSkyAdapter(sky_eq, lmax=lmax)

    beam = lusee.BeamGauss(
        dec_deg=90.0,
        sigma_deg=sigma_deg,
        phi_deg=90.0,
        one_over_freq_scaling=False,
        id="beam",
    )
    beams = [beam]

    # Run DefaultSimulator (topo frame)
    def_sim = lusee.DefaultSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"plot_sky_and_beam": True},
    )
    def_sim.simulate(times=times)

    # Run CroSimulator (MCMF frame)
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations= [(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"plot_sky_and_beam": True},
    )
    cro_sim.simulate(times=times)

    out_dir = Path(__file__).resolve().parent
    cro_sim.write_fits(str(out_dir / "sim_output_cro_singlepixel_28days.fits"))
    def_sim.write_fits(str(out_dir / "sim_output_default_singlepixel_28days.fits"))


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
