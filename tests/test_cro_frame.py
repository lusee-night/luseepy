#!/usr/bin/env python3
"""
First-order test: Gaussian beam + sky with a single pixel at known (ra, dec).
Simulate with CroSimulator and check that the response is consistent with
that source position (compare with track from known ra, dec).

Usage:
  python tests/test_cro_frame.py [-v]
  pytest tests/test_cro_frame.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp
from astropy.coordinates import SkyCoord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PointSourceSky:
    """Sky with a single pixel at (l_deg, b_deg) in galactic, temperature T."""

    def __init__(self, l_deg, b_deg, Nside, lmax, T=1.0, freq=None):
        self.Nside = Nside
        self.Npix = Nside ** 2 * 12
        self.lmax = lmax
        self._T = np.atleast_1d(np.asarray(T, dtype=float))
        self.freq = np.array([1.0]) if freq is None else np.atleast_1d(freq)
        self.frame = "galactic"
        theta = np.pi / 2 - np.deg2rad(b_deg)
        phi = np.deg2rad(l_deg)
        if phi < 0:
            phi += 2 * np.pi
        pix = hp.ang2pix(Nside, theta, phi)
        Tmap = np.zeros(self.Npix)
        Tmap[pix] = self._T[0] if self._T.size == 1 else 1.0
        self.mapalm = hp.map2alm(Tmap, lmax=lmax)
        if self._T.size > 1:
            self.mapalm = [self.mapalm * t for t in self._T]

    def get_alm(self, ndx, freq=None):
        if np.isscalar(ndx):
            ndx = [ndx]
        if hasattr(self.mapalm, "__len__") and not isinstance(self.mapalm, np.ndarray):
            return [self.mapalm[i] for i in ndx]
        return [self.mapalm] * len(ndx)


def ra_dec_to_galactic(ra_deg, dec_deg):
    """(l_deg, b_deg) galactic for ICRS (ra_deg, dec_deg)."""
    c = SkyCoord(ra=ra_deg, dec=dec_deg, frame="icrs", unit="deg")
    g = c.galactic
    return float(g.l.deg), float(g.b.deg)


def plot_visual_check(ra_deg, dec_deg, ra_track, dec_track, simulated, outpath="test_cro_frame_visual.png"):
    """
    Two subplots, both with x=RA (deg), y=Dec (deg):
    (1) Single dot at the selected input (ra, dec).
    (2) Simulated sky (beam convolved): track of the source in ra-dec, colored by simulated response.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    # Subplot 1: input position (one dot)
    ax1.scatter(ra_deg, dec_deg, s=80, c="red", edgecolors="black", zorder=2, label="Input (ra, dec)")
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.set_title("Input sky: single pixel at (ra, dec)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Subplot 2: track in ra-dec colored by simulated response (beam convolved)
    sc = ax2.scatter(ra_track, dec_track, c=simulated, s=40, cmap="viridis")
    ax2.set_xlabel("RA (deg)")
    ax2.set_ylabel("Dec (deg)")
    ax2.set_title("Simulated output (beam convolved): source track colored by response")
    plt.colorbar(sc, ax=ax2, label="Response")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal", adjustable="box")
    # Match x and y limits to first plot
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())

    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"  Visual check plot saved to {outpath}")


def test_cro_single_source():
    """Gaussian beam, single pixel at known (ra, dec); run CroSimulator and compare with known position."""
    import lusee

    if lusee.CroSimulator is None:
        import pytest
        pytest.skip("CroSimulator requires croissant and s2fft")

    # Known source position (ICRS)
    ra_deg, dec_deg = 45.0, 10.0
    nside = 32
    lmax = 32
    sigma_deg = 8.0
    Tground = 0.0

    obs = lusee.Observation(
        "2025-03-01 12:00:00 to 2025-03-01 16:00:00",
        deltaT_sec=1800.0,
        lun_lat_deg=-20.0,
        lun_long_deg=30.0,
    )
    times = obs.times[:9]
    freq = np.array([10.0])

    # Sky: single pixel at (ra_deg, dec_deg) in galactic
    l_deg, b_deg = ra_dec_to_galactic(ra_deg, dec_deg)
    sky = PointSourceSky(l_deg, b_deg, nside, lmax, T=1.0, freq=freq)

    # Beam at zenith
    beam = lusee.BeamGauss(
        dec_deg=90.0,
        sigma_deg=sigma_deg,
        phi_deg=0.0,
        one_over_freq_scaling=False,
        id="beam",
    )
    beams = [beam]

    # Simulate with CroSimulator
    sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
    )
    sim.simulate(times=times)
    out = sim.result  # (N_times, N_combos, N_freq)
    simulated = out[:, 0, 0]

    # Track for the known (ra, dec): alt, az in radians at each time
    alt, az = obs.get_track_ra_dec(ra_deg, dec_deg, times=times)

    # At the time when the simulated response peaks, (alt, az) should correspond to our known (ra, dec)
    t_max_sim = int(np.argmax(simulated))
    alt_peak = float(alt[t_max_sim])
    az_peak = float(az[t_max_sim])
    ra_rad, dec_rad = obs.get_ra_dec_from_alt_az(alt_peak, az_peak, times=[times[t_max_sim]])
    ra_recovered = np.rad2deg(ra_rad[0])
    dec_recovered = np.rad2deg(dec_rad[0])

    print("  Input (ra, dec) [deg]:     ({:.4f}, {:.4f})".format(ra_deg, dec_deg))
    print("  Recovered (ra, dec) [deg]: ({:.4f}, {:.4f})".format(ra_recovered, dec_recovered))

    # Compare recovered ra, dec to known (ra_deg, dec_deg)
    np.testing.assert_allclose(
        [ra_recovered, dec_recovered],
        [ra_deg, dec_deg],
        atol=1.0,
        err_msg=f"Recovered (ra, dec) = ({ra_recovered}, {dec_recovered}) should match known ({ra_deg}, {dec_deg})",
    )

    # (ra, dec) at each time for the track (for plot)
    ra_track = np.zeros(len(times))
    dec_track = np.zeros(len(times))
    for i in range(len(times)):
        ra_rad_i, dec_rad_i = obs.get_ra_dec_from_alt_az(float(alt[i]), float(az[i]), times=[times[i]])
        ra_track[i] = np.rad2deg(ra_rad_i[0])
        dec_track[i] = np.rad2deg(dec_rad_i[0])
    plot_visual_check(ra_deg, dec_deg, ra_track, dec_track, simulated)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = p.parse_args()
    test_cro_single_source()
    print("test_cro_frame: passed.")
