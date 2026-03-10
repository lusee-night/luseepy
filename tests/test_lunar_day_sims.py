#!/usr/bin/env python3
"""
28-day lunar simulation with a single-pixel source: run DefaultSimulator (observer frame)
and CroSimulator (MCMF frame), then plot the source position (altitude) relative to the
horizon for both frame conventions over 28 days. Observation at lunar latitude 0 deg.

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

# Optional: astropy for (ra,dec) -> galactic
try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    _HAS_ASTROPY = True
except ImportError:
    _HAS_ASTROPY = False


def _ra_dec_to_galactic_rad(ra_deg, dec_deg):
    """Return (l_rad, b_rad) for the given (ra_deg, dec_deg) in equatorial."""
    if _HAS_ASTROPY:
        c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        return c.galactic.l.rad, c.galactic.b.rad
    rot = hp.rotator.Rotator(coord=["C", "G"])
    theta_eq = np.radians(90.0 - dec_deg)
    phi_eq = np.radians(ra_deg)
    theta_gal, phi_gal = rot(theta_eq, phi_eq)
    b_rad = np.pi / 2 - theta_gal
    l_rad = phi_gal
    return float(l_rad), float(b_rad)


def _galactic_dir(l_rad, b_rad):
    """Unit vector in galactic coordinates: (x, y, z) = (cos(b)cos(l), cos(b)sin(l), sin(b))."""
    return np.array([
        np.cos(b_rad) * np.cos(l_rad),
        np.cos(b_rad) * np.sin(l_rad),
        np.sin(b_rad),
    ])


def _topo_to_alt_az(vec_topo):
    """Convert unit vector in topo (x=east, y=north, z=zenith) to (alt_rad, az_rad)."""
    z = vec_topo[2]
    x, y = vec_topo[0], vec_topo[1]
    alt = np.arcsin(np.clip(z, -1, 1))
    az = np.arctan2(x, y)
    return alt, az


def _R_z(phi):
    """Rotation matrix about z-axis by phi (radians)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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


def compute_observer_frame_alt_az(obs, times, gal_dir):
    """Source altitude/azimuth (rad) at each time in observer (topo) frame: R_gal_topo(t) @ gal_dir."""
    from lusee.SimulatorBase import get_R_gal_to_topo
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    alt_rad = np.zeros(len(times))
    az_rad = np.zeros(len(times))
    for i in range(len(times)):
        R = get_R_gal_to_topo(lzl[i], bzl[i], lyl[i], byl[i])
        vec_topo = R @ gal_dir
        alt_rad[i], az_rad[i] = _topo_to_alt_az(vec_topo)
    return alt_rad, az_rad


def compute_mcmf_frame_alt_az(obs, times, gal_dir):
    """Source altitude/azimuth (rad) at each time in MCMF convention: R_mcmf_to_topo(t) @ R_z(phi(t)) @ R_gal_mcmf @ gal_dir."""
    from lusee.SimulatorBase import get_topo_z_rotation_angles
    from croissant.rotations import get_rot_mat
    from lunarsky import LunarTopo

    # gal_dir is unit vector in galactic; R_gal_mcmf takes gal to MCMF
    R_gal_mcmf = get_rot_mat("galactic", "mcmf")
    phi_rad = get_topo_z_rotation_angles(obs, times)

    alt_rad = np.zeros(len(times))
    az_rad = np.zeros(len(times))
    for i in range(len(times)):
        R_topo_to_mcmf = get_rot_mat(LunarTopo(obstime=times[i], location=obs.loc), "mcmf")
        R_mcmf_to_topo = R_topo_to_mcmf.T
        source_mcmf = _R_z(phi_rad[i]) @ R_gal_mcmf @ gal_dir
        vec_topo = R_mcmf_to_topo @ source_mcmf
        alt_rad[i], az_rad[i] = _topo_to_alt_az(vec_topo)
    return alt_rad, az_rad


def test_lunar_day_28_single_source():
    """Single-pixel sky, lunar lat 0°, 28 days; run Default (observer) and Cro (MCMF); plot source alt vs time."""
    import lusee
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Source position (equatorial); SingleSourceHealpixSky uses (ra, dec)
    ra_deg, dec_deg = 45.0, 10.0
    l_rad, b_rad = _ra_dec_to_galactic_rad(ra_deg, dec_deg)
    gal_dir = _galactic_dir(l_rad, b_rad)

    # 28 days, 2-hour steps → 336 points
    time_start = "2025-03-01 00:00:00"
    time_end = "2025-03-29 00:00:00"  # 28 days later
    deltaT_sec = 7200.0  # 2 hours
    obs = lusee.Observation(
        f"{time_start} to {time_end}",
        deltaT_sec=deltaT_sec,
        lun_lat_deg=0.0,
        lun_long_deg=30.0,
    )
    times = obs.times
    nside = 32
    lmax = 32
    sigma_deg = 10.0
    Tground = 0.0
    freq = np.array([25.0])

    # Single-pixel sky (equatorial) wrapped to galactic
    sky_eq = lusee.sky.SingleSourceHealpixSky(ra_deg, dec_deg, Nside=nside, freq=freq)
    sky = GalacticSkyAdapter(sky_eq, lmax=lmax)

    beam = lusee.BeamGauss(
        dec_deg=90.0,
        sigma_deg=sigma_deg,
        phi_deg=0.0,
        one_over_freq_scaling=False,
        id="beam",
    )
    beams = [beam]

    # Run DefaultSimulator (observer frame)
    def_sim = lusee.DefaultSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={},
    )
    def_sim.simulate(times=times)

    # Run CroSimulator (MCMF frame)
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"use_observer_frame": False},
    )
    cro_sim.simulate(times=times)

    # Source position (alt) for both frame conventions
    alt_obs, az_obs = compute_observer_frame_alt_az(obs, times, gal_dir)
    alt_mcmf, az_mcmf = compute_mcmf_frame_alt_az(obs, times, gal_dir)

    # Time in days from start (uniform step)
    time_days = np.arange(len(times)) * (deltaT_sec / 86400.0)
    sec_per_day = 86400.0
    days_above_obs = (alt_obs > 0).sum() * (deltaT_sec / sec_per_day)
    days_above_mcmf = (alt_mcmf > 0).sum() * (deltaT_sec / sec_per_day)

    # Single subplot: altitude (deg) vs time (days) for both
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(time_days, np.degrees(alt_obs), label=f"Observer frame (Default) ({days_above_obs:.1f} d above horizon)", alpha=0.8)
    ax.plot(time_days, np.degrees(alt_mcmf), label=f"MCMF frame (Cro) ({days_above_mcmf:.1f} d above horizon)", alpha=0.8)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time (days from start)")
    ax.set_ylabel("Source altitude (deg)")
    ax.set_title("Single-pixel source position over 28 days (lunar lat 0°)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_dir = Path(__file__).resolve().parent
    plot_path = plot_dir / "lunar_day_28_source_alt.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")

    # Sanity: both simulators produced output
    assert def_sim.result.shape[0] == len(times)
    assert cro_sim.result.shape[0] == len(times)


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
