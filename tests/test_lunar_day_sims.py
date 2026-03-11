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


def compute_source_alt_az_default_rotations(obs, times, gal_dir):
    """
    Source (alt, az) at each time using the same rotations as DefaultSimulator.
    DefaultSimulator at each t: gets (lz,bz,ly,by) from obs.get_l_b_from_alt_az(zenith, north),
    builds R = [xhat,yhat,zhat].T with zhat=zenith, yhat=north, then rotates sky by R (gal→topo).
    Same R is get_R_gal_to_topo(lz,bz,ly,by). So source_topo = R @ gal_dir.
    """
    from lusee.SimulatorBase import get_R_gal_to_topo

    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)   # zenith (same as DefaultSimulator)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)         # north, az=0 (same as DefaultSimulator)
    alt_rad = np.zeros(len(times))
    az_rad = np.zeros(len(times))
    for i in range(len(times)):
        R = get_R_gal_to_topo(lzl[i], bzl[i], lyl[i], byl[i])   # same R as DefaultSimulator
        vec_topo = R @ gal_dir
        alt_rad[i], az_rad[i] = _topo_to_alt_az(vec_topo)
    return alt_rad, az_rad


def compute_source_alt_az_mcmf(obs, times, gal_dir):
    """
    Source (alt, az) at each time for MCMF time evolution but in same topo as observer
    so both curves align at t=0: vec_topo = R_gal_topo(t) @ R_gal_mcmf.T @ R_z(phi(t)) @ R_gal_mcmf @ gal_dir.
    """
    from lusee.SimulatorBase import get_R_gal_to_topo, get_topo_z_rotation_angles
    from croissant.rotations import get_rot_mat

    R_gal_mcmf = get_rot_mat("galactic", "mcmf")
    phi_rad = get_topo_z_rotation_angles(obs, times)
    lzl, bzl = obs.get_l_b_from_alt_az(np.pi / 2, 0.0, times)
    lyl, byl = obs.get_l_b_from_alt_az(0.0, 0.0, times)
    alt_rad = np.zeros(len(times))
    az_rad = np.zeros(len(times))
    for i in range(len(times)):
        R_gal_topo_t = get_R_gal_to_topo(lzl[i], bzl[i], lyl[i], byl[i])  # same topo as observer
        source_mcmf = _R_z(phi_rad[i]) @ R_gal_mcmf @ gal_dir
        vec_topo = R_gal_topo_t @ (R_gal_mcmf.T @ source_mcmf)
        alt_rad[i], az_rad[i] = _topo_to_alt_az(vec_topo)
    return alt_rad, az_rad


def test_lunar_day_28_single_source():
    """Compute and plot source (alt, az) over 28 days from obs, times, gal_dir."""
    import lusee
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Source on the inertial ecliptic plane (lat=0); conversion uses mean obliquity, not geocentric frame
    ecl_lon_deg, ecl_lat_deg = 45.0, 0.0
    ra_deg, dec_deg = _ecliptic_lon_lat_to_ra_dec_deg(ecl_lon_deg, ecl_lat_deg)
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
        lun_long_deg=0.0,
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
        phi_deg=90.0,
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
        extra_opts={},
    )
    cro_sim.simulate(times=times)

    # Source (alt, az) from obs, times, gal_dir (for comparison with simulated output)
    alt_obs, az_obs = compute_source_alt_az_default_rotations(obs, times, gal_dir)
    alt_mcmf, az_mcmf = compute_source_alt_az_mcmf(obs, times, gal_dir)

    # Simulated visibility (first combo, first freq): peak ≈ source in beam
    default_vis = np.asarray(def_sim.result[:, 0, 0])
    cro_vis = np.asarray(cro_sim.result[:, 0, 0])
    idx_peak_default = int(np.argmax(default_vis))
    idx_peak_cro = int(np.argmax(cro_vis))
    print("  Source position at time of peak simulated visibility:")
    print("    Default: t_peak = {:.2f} d  →  alt = {:.2f}°, az = {:.2f}° (observer)".format(
        idx_peak_default * (deltaT_sec / 86400.0), np.degrees(alt_obs[idx_peak_default]), np.degrees(az_obs[idx_peak_default])))
    print("    Cro:     t_peak = {:.2f} d  →  alt = {:.2f}°, az = {:.2f}° (MCMF/same topo)".format(
        idx_peak_cro * (deltaT_sec / 86400.0), np.degrees(alt_mcmf[idx_peak_cro]), np.degrees(az_mcmf[idx_peak_cro])))

    # Time in days from start (uniform step)
    time_days = np.arange(len(times)) * (deltaT_sec / 86400.0)
    sec_per_day = 86400.0
    days_above_obs = (alt_obs > 0).sum() * (deltaT_sec / sec_per_day)
    days_above_mcmf = (alt_mcmf > 0).sum() * (deltaT_sec / sec_per_day)

    # Two subplots: (1) geometric source alt, (2) simulated visibility (check source in beam)
    fig, (ax_alt, ax_sim) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_alt.plot(time_days, np.degrees(alt_obs), label=f"Observer frame (Default) ({days_above_obs:.1f} d above horizon)", alpha=0.8)
    ax_alt.plot(time_days, np.degrees(alt_mcmf), label=f"MCMF frame (Cro) ({days_above_mcmf:.1f} d above horizon)", alpha=0.8)
    ax_alt.axhline(0, color="k", linestyle="--", linewidth=0.5)
    #ax_alt.axvline(time_days[idx_peak_default], color="C0", linestyle=":", alpha=0.7, label="Default peak vis")
    #ax_alt.axvline(time_days[idx_peak_cro], color="C1", linestyle=":", alpha=0.7, label="Cro peak vis")
    ax_alt.set_ylabel("Source altitude (deg)")
    #ax_alt.set_title("Geometric source position (obs, times, gal_dir)")
    ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)

    ax_sim.plot(time_days, default_vis, label="DefaultSimulator", alpha=0.8)
    ax_sim.plot(time_days, cro_vis, label="CroSimulator (MCMF)", alpha=0.8)
    ax_sim.axvline(time_days[idx_peak_default], color="C0", linestyle=":", alpha=0.7)
    ax_sim.axvline(time_days[idx_peak_cro], color="C1", linestyle=":", alpha=0.7)
    ax_sim.set_xlabel("Time (days from start)")
    ax_sim.set_ylabel("Visibility / T (a.u.)")
    ax_sim.set_title("Simulated output (peak ≈ source in beam)")
    ax_sim.legend()
    ax_sim.grid(True, alpha=0.3)

    fig.suptitle("Source position over 28 days (lunar lat 0°)", y=1.02)
    plt.tight_layout()
    plot_dir = Path(__file__).resolve().parent
    plot_path = plot_dir / "lunar_day_28_source_alt.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")

    assert def_sim.result.shape[0] == len(times)
    assert cro_sim.result.shape[0] == len(times)


if __name__ == "__main__":
    test_lunar_day_28_single_source()
    print("test_lunar_day_sims: passed.")
