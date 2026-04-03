"""HarmonicPointSourceSky tests."""
import numpy as np
import healpy as hp
import pytest
from lusee.SkyModels import HarmonicPointSourceSky


def _pixel_point_source_alm(theta, phi, nside, lmax):
    """Reference: hot-pixel map2alm, normalized by pixel area."""
    pix = hp.ang2pix(nside, theta, phi)
    m = np.zeros(hp.nside2npix(nside))
    m[pix] = 1.0
    alm = hp.map2alm(m, lmax=lmax)
    pixel_area = 4 * np.pi / hp.nside2npix(nside)
    return alm / pixel_area


def _apply_gaussian_beam(alm, sigma_rad, lmax):
    """Multiply each a_lm by b_l = exp(-l(l+1)σ²/2)."""
    out = alm.copy()
    for l in range(lmax + 1):
        bl = np.exp(-l * (l + 1) * sigma_rad ** 2 / 2)
        for m in range(l + 1):
            out[hp.Alm.getidx(lmax, l, m)] *= bl
    return out


def test_beam_convolved_matches_pixel_version():
    """Gaussian beam applied to harmonic vs pixel point source → same map."""
    lmax = 64
    sigma_rad = np.radians(8.0)
    l_deg, b_deg = 30.0, 45.0
    theta = np.radians(90.0 - b_deg)
    phi = np.radians(l_deg)

    sky = HarmonicPointSourceSky(lmax=lmax, freq=[10.0], l_deg=l_deg, b_deg=b_deg)
    alm_h = sky.get_alm([0])[0]
    alm_p = _pixel_point_source_alm(theta, phi, nside=512, lmax=lmax)

    map_h = hp.alm2map(_apply_gaussian_beam(alm_h, sigma_rad, lmax),
                       nside=64, lmax=lmax, verbose=False)
    map_p = hp.alm2map(_apply_gaussian_beam(alm_p, sigma_rad, lmax),
                       nside=64, lmax=lmax, verbose=False)

    corr = np.corrcoef(map_h, map_p)[0, 1]
    assert corr > 0.999, f"map correlation {corr:.6f} too low"
    np.testing.assert_allclose(np.max(map_h), np.max(map_p), rtol=0.01)


def test_a00_equals_Y00():
    """a_00 should be Y*_00 = 1/sqrt(4π) regardless of position."""
    for l_deg, b_deg in [(0, 90), (180, -45), (42, 7)]:
        sky = HarmonicPointSourceSky(lmax=8, freq=[1.0], l_deg=l_deg, b_deg=b_deg)
        a00 = sky.get_alm([0])[0][0]
        np.testing.assert_allclose(a00.real, 1 / np.sqrt(4 * np.pi), atol=1e-14)


def test_equatorial_vs_galactic_frame():
    sky_eq = HarmonicPointSourceSky(lmax=8, freq=[1.0], ra_deg=0, dec_deg=0)
    sky_gal = HarmonicPointSourceSky(lmax=8, freq=[1.0], l_deg=0, b_deg=0)
    assert sky_eq.frame == "equatorial"
    assert sky_gal.frame == "galactic"


def test_rejects_ambiguous_coordinates():
    with pytest.raises(ValueError, match="either"):
        HarmonicPointSourceSky(lmax=8, freq=[1.0], ra_deg=0, dec_deg=0, l_deg=0, b_deg=0)
    with pytest.raises(ValueError, match="either"):
        HarmonicPointSourceSky(lmax=8, freq=[1.0])


def test_frequency_scaling():
    sky = HarmonicPointSourceSky(lmax=4, freq=[10.0, 20.0], T=[2.0, 5.0], l_deg=0, b_deg=45)
    alms = sky.get_alm([0, 1])
    ratio = np.abs(alms[1][0]) / np.abs(alms[0][0])
    np.testing.assert_allclose(ratio, 2.5, atol=1e-14)
