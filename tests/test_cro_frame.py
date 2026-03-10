#!/usr/bin/env python3
"""
Single-pixel sky test using SingleSourceHealpixSky: run DefaultSimulator and CroSimulator,
write sim_output_singlepixel.fits and sim_output_cro_singlepixel.fits.

SingleSourceHealpixSky is frame="equatorial"; both simulators require frame="galactic", so the test
wraps the sky with a test-only adapter that rotates alms equatorial -> galactic.

Usage:
  python tests/test_cro_frame.py [-v]
  pytest tests/test_cro_frame.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp
from pathlib import Path

# Output FITS files in luseepy/simulation/output/
out_dir = Path(__file__).resolve().parent.parent / "simulation" / "output"
OUT_DEFAULT = out_dir / "sim_output_singlepixel.fits"
OUT_CRO = out_dir / "sim_output_cro_singlepixel.fits"


class GalacticSkyAdapter:
    """Test-only: wraps an equatorial HealpixSky and exposes galactic alms via fixed eq->gal rotation.
    Truncates alms to lmax so they match the simulator/beam (SingleSourceHealpixSky uses 3*Nside-1)."""
    def __init__(self, equatorial_sky, lmax):
        self._sky = equatorial_sky
        self._lmax = lmax
        self.frame = "galactic"
        self.freq = equatorial_sky.freq
        self.Nside = equatorial_sky.Nside
        self._rot = hp.rotator.Rotator(coord=["C", "G"])
        self._alm_size = hp.Alm.getsize(lmax)

    def get_alm(self, ndx, freq):
        alms = self._sky.get_alm(ndx, freq)
        alms = np.atleast_2d(alms)
        out = []
        for a in alms:
            r = self._rot.rotate_alm(a)
            out.append(r[: self._alm_size].copy())
        return out


def test_cro_single_source():
    """SingleSourceHealpixSky at (ra, dec) [equatorial]; wrap to galactic; run Default and Cro; write FITS."""
    import lusee

    # Known source position (RA, Dec in degrees). SingleSourceHealpixSky uses equatorial frame.
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
    # Use full obs.times so written FITS has same number of rows as Observation expects
    # (lusee.Data asserts len(self.times) == self.data.shape[0]).
    times = obs.times
    freq = np.arange(1,51,1)

    sky_equatorial = lusee.sky.SingleSourceHealpixSky(ra_deg, dec_deg, Nside=nside, freq=freq)
    # Simulators require galactic; adapter rotates alms eq->gal and truncates to lmax (sky uses 3*Nside-1)
    sky = GalacticSkyAdapter(sky_equatorial, lmax=lmax)

    beam = lusee.BeamGauss(
        dec_deg=90.0,
        sigma_deg=sigma_deg,
        phi_deg=0.0,
        one_over_freq_scaling=False,
        id="beam",
    )
    beams = [beam]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_default = str(OUT_DEFAULT)
    out_cro = str(OUT_CRO)
    for p in (out_default, out_cro):
        if os.path.exists(p):
            os.remove(p)

    # DefaultSimulator
    def_sim = lusee.DefaultSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"plot_sky_and_beam": True},
    )
    def_sim.simulate(times=times)
    def_sim.write_fits(out_default)
    print(f"  Wrote {out_default}")

    default_out = def_sim.result[:, 0, 0]

    # CroSimulator
    cro_sim = lusee.CroSimulator(
        obs, beams, sky,
        Tground=Tground,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
        extra_opts={"plot_sky_and_beam": True, "use_observer_frame": True},
    )
    cro_sim.simulate(times=times)
    cro_sim.write_fits(out_cro)
    print(f"  Wrote {out_cro}")

    cro_out = cro_sim.result[:, 0, 0]

    # Compare: difference is expected from coordinate transformations (test_cro_vs_default)
    diff = np.abs(default_out - cro_out)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(default_out != 0, cro_out / default_out, np.nan)
    print("  Default vs Cro: max |diff| = {:.6e}, ratio (Cro/Default) range = [{:.4f}, {:.4f}]".format(
        np.nanmax(diff), np.nanmin(ratio), np.nanmax(ratio)))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    args = p.parse_args()
    test_cro_single_source()
    print("test_cro_frame: passed.")
