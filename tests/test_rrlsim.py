"""Tests for Rydberg RRL helpers, catalog I/O, ULSA+RRL sky, and RRLSimulator."""

from __future__ import annotations

import os

import numpy as np
import pytest

import lusee
from lusee.frequencies import canonical_frequencies, frequency_indices_from_values
from lusee.RRLSkyModels import (
    ULSAPlusRRLSky,
    hydrogen_rrl_alpha_quantum_numbers_from_frequency_mhz,
    hydrogen_rrl_alpha_transitions_in_frequency_band_mhz,
    hydrogen_rrl_frequency_mhz,
    load_rrl_region_positions_gal_deg,
    spectral_axis_numpy_index_from_header,
)


def test_linear_resample_beam_gauss_matches_length():
    import jax.numpy as jnp

    from lusee.beam_fine_frequency import linear_resample_beam_freq_mhz

    b0 = lusee.BeamGauss(
        alt_deg=90.0,
        sigma_deg=20.0,
        one_over_freq_scaling=False,
        id="t",
    )
    fine = jnp.linspace(5.0, 8.0, 31)
    b1 = linear_resample_beam_freq_mhz(b0, fine)
    assert b1.Nfreq == 31
    assert b1.Etheta.shape[0] == 31


def test_alpha_quantum_numbers_round_trip():
    n1, n2 = 3527, 3526
    nu = hydrogen_rrl_frequency_mhz(n1, n2)
    assert hydrogen_rrl_alpha_quantum_numbers_from_frequency_mhz(nu) == (n1, n2)


def test_alpha_transitions_default_band_inside_1_50_mhz():
    from lusee.frequencies import CANONICAL_FREQ_START_MHZ, CANONICAL_FREQ_STOP_MHZ

    trans = hydrogen_rrl_alpha_transitions_in_frequency_band_mhz()
    assert len(trans) > 50
    for n1, n2 in (trans[0], trans[len(trans) // 2], trans[-1]):
        f = hydrogen_rrl_frequency_mhz(n1, n2)
        assert CANONICAL_FREQ_START_MHZ - 1e-9 <= f <= CANONICAL_FREQ_STOP_MHZ + 1e-9


def test_alpha_transitions_narrow_band_frequencies():
    trans = hydrogen_rrl_alpha_transitions_in_frequency_band_mhz(24.0, 26.0)
    assert 5 < len(trans) < 5000
    for n1, n2 in (trans[0], trans[len(trans) // 2], trans[-1]):
        f = hydrogen_rrl_frequency_mhz(n1, n2)
        assert 24.0 - 1e-6 <= f <= 26.0 + 1e-6


def test_hydrogen_rrl_frequency_ordering():
    """Higher n → lower transition frequency for nα (adjacent rungs)."""
    f168 = hydrogen_rrl_frequency_mhz(168, 167)
    f167 = hydrogen_rrl_frequency_mhz(167, 166)
    f166 = hydrogen_rrl_frequency_mhz(166, 165)
    assert f168 < f167 < f166
    assert np.isfinite([f168, f167, f166]).all()


def test_rrl_default_line_gaussian_fwhm_and_peak():
    sig = lusee.RRL_DEFAULT_LINE_SIGMA_MHZ
    fwhm_khz = 1000.0 * sig * (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.isclose(fwhm_khz, lusee.RRL_DEFAULT_LINE_FWHM_KHZ)
    assert lusee.RRL_DEFAULT_LINE_PEAK_K == 0.5


def test_hydrogen_rrl_frequency_matches_astropy_scalar():
    pytest.importorskip("astropy")
    from astropy import constants as const
    import astropy.units as u

    ni, nf = 401, 400
    mu = const.m_e * const.m_p / (const.m_e + const.m_p)
    r_inf = getattr(const, "R_inf", const.Ryd)
    r_m = r_inf * mu / const.m_e
    inv = 1.0 / (nf * nf) - 1.0 / (ni * ni)
    expected = float((const.c * r_m * inv).to(u.MHz).value)
    assert np.isclose(hydrogen_rrl_frequency_mhz(ni, nf), expected, rtol=1e-9)


def test_load_rrl_region_positions_from_temp_fits(tmp_path):
    import fitsio

    lon = np.array([10.0, 20.0], dtype=np.float32)
    lat = np.array([-5.0, 2.0], dtype=np.float32)
    tbl_path = tmp_path / "rrl_cat.fits"
    fitsio.write(str(tbl_path), {"GLON": lon, "GLAT": lat}, extname="SOURCES")

    glon, glat = load_rrl_region_positions_gal_deg(str(tbl_path))
    np.testing.assert_array_almost_equal(glon, lon)
    np.testing.assert_array_almost_equal(glat, lat)


def test_observation_frequency_mhz_from_config_khz_grid():
    from lusee.frequencies import observation_frequency_mhz_from_config

    f = observation_frequency_mhz_from_config(
        {"start_mhz": 10.0, "stop_mhz": 10.003, "step_khz": 1.0}
    )
    assert len(f) == 4
    assert float(f[0]) == 10.0
    assert float(f[-1]) == 10.003


def test_ulsa_plus_rrl_get_alm_shape(tmp_path):
    import fitsio
    import healpy as hp

    nside = 8
    npix = 12 * nside * nside
    nfreq = 3
    maps = np.full((nfreq, npix), 50.0, dtype=np.float32)
    ulsa_path = tmp_path / "ulsa_like.fits"
    fitsio.write(
        str(ulsa_path),
        maps,
        header={"freq_start": 10.0, "freq_end": 12.0, "freq_step": 1.0},
    )

    rrl_path = tmp_path / "rrl_src.fits"
    fitsio.write(
        str(rrl_path),
        {"GLON": np.array([0.0]), "GLAT": np.array([0.0])},
        extname="SRC",
    )

    lmax = 3 * nside - 1
    sky = ULSAPlusRRLSky(
        str(ulsa_path),
        str(rrl_path),
        lmax,
        alpha_transitions=((168, 167), (167, 166)),
        rrl_sigma_mhz=500.0,
        rrl_peak_k=1.0,
        spot_sigma_deg=15.0,
    )
    assert sky.rrl_spot_map.shape == (npix,)
    assert float(np.max(sky.rrl_spot_map)) == pytest.approx(1.0)
    freq = canonical_frequencies(frequency_indices_from_values([10.0, 11.0, 12.0]))
    assert np.allclose(np.asarray(sky.freq), np.asarray(freq))

    ndx = np.array([0, 2], dtype=np.int32)
    alm = sky.get_alm(ndx)
    nalm = hp.Alm.getsize(lmax)
    assert alm.shape == (2, nalm)
    assert np.all(np.isfinite(np.asarray(alm)))

    fine = np.array([10.0, 10.0005, 10.001, 11.0, 12.0], dtype=np.float64)
    sky2 = ULSAPlusRRLSky(
        str(ulsa_path),
        str(rrl_path),
        lmax,
        sim_freq_mhz=fine,
        alpha_transitions=((168, 167), (167, 166)),
        rrl_sigma_mhz=500.0,
        rrl_peak_k=1.0,
        spot_sigma_deg=15.0,
    )
    assert sky2.mapalm_ulsa.shape[0] == len(fine)
    assert np.allclose(np.asarray(sky2.freq), fine)
    alm2 = sky2.get_alm(np.array([0, 4], dtype=np.int32))
    assert alm2.shape == (2, nalm)


def test_rrl_brightness_map_K_shape_and_nonzero_near_line(tmp_path):
    import fitsio
    import healpy as hp

    nside = 8
    lmax = 3 * nside - 1
    npix = 12 * nside**2
    f0, f1, df = 10.0, 12.0, 1.0
    nfreq = int(round((f1 - f0) / df)) + 1
    maps = np.full((nfreq, npix), 50.0, dtype=np.float32)
    ulsa_path = tmp_path / "ulsa_like.fits"
    fitsio.write(
        str(ulsa_path),
        maps,
        header={"freq_start": f0, "freq_end": f1, "freq_step": df},
    )
    rrl_path = tmp_path / "rrl_src.fits"
    fitsio.write(
        str(rrl_path),
        {"GLON": np.array([0.0]), "GLAT": np.array([0.0])},
        extname="SRC",
    )
    n1, n2 = hydrogen_rrl_alpha_quantum_numbers_from_frequency_mhz(11.0)
    sky = ULSAPlusRRLSky(
        str(ulsa_path),
        str(rrl_path),
        lmax,
        alpha_transitions=((n1, n2),),
        rrl_sigma_mhz=0.5,
        rrl_peak_k=2.0,
        spot_sigma_deg=20.0,
    )
    m = sky.rrl_brightness_map_K(1)
    assert m.shape == (hp.nside2npix(sky.Nside),)
    assert np.isfinite(m).all()
    assert float(np.nanmax(np.abs(m))) > 1e-6


@pytest.mark.integration
def test_rrlsimulator_with_drive_data(drive_dir):
    if lusee.RRLSimulator is None:
        pytest.skip("croissant/s2fft not installed")

    ulsa = os.path.join(drive_dir, "Simulations", "SkyModels", "ULSA_32_ddi_smooth.fits")
    rrl = os.path.join(
        drive_dir,
        "Simulations",
        "SkyModels",
        "RRL_maps",
        "RRL_H166-167-168a_HIPASS+ZOA_lbv.fits",
    )
    if not (os.path.isfile(ulsa) and os.path.isfile(rrl)):
        pytest.skip("ULSA or RRL catalog FITS not found on drive")

    lmax = 32
    sky = ULSAPlusRRLSky(ulsa, rrl, lmax, rrl_sigma_mhz=0.1, rrl_peak_k=0.01)
    freq = canonical_frequencies(frequency_indices_from_values([15.0, 20.0, 25.0]))

    beam = lusee.BeamGauss(
        alt_deg=90.0,
        sigma_deg=30.0,
        one_over_freq_scaling=False,
        id="b0",
    )
    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 06:00:00",
        deltaT_sec=7200.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )
    sim = lusee.RRLSimulator(
        obs,
        [beam],
        sky,
        Tground=0.0,
        combinations=[(0, 0)],
        freq=freq,
        lmax=lmax,
    )
    out = sim.simulate(times=obs.times)
    assert out.shape == (len(obs.times), 1, len(freq))
    assert np.all(np.isfinite(np.asarray(out)))


def test_spectral_axis_numpy_index_from_header():
    """FITS numpy axis order is (NAXIS3, NAXIS2, NAXIS1); spectral on axis 3 → axis 0."""
    from astropy.io.fits import Header

    h = Header()
    h["NAXIS"] = 3
    h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = 201, 51, 51
    h["CTYPE1"] = "GLON-CAR"
    h["CTYPE2"] = "GLAT-CAR"
    h["CTYPE3"] = "VELO-LSR"
    assert spectral_axis_numpy_index_from_header(h) == 0

    h2 = Header()
    h2["NAXIS"] = 3
    h2["NAXIS1"], h2["NAXIS2"], h2["NAXIS3"] = 10, 20, 30
    h2["CTYPE1"] = "FREQ"
    h2["CTYPE2"] = "GLON-CAR"
    h2["CTYPE3"] = "GLAT-CAR"
    assert spectral_axis_numpy_index_from_header(h2) == 2

    h3 = Header()
    h3["NAXIS"] = 3
    h3["CTYPE1"] = "RA---TAN"
    h3["CTYPE2"] = "DEC--TAN"
    h3["CTYPE3"] = "STOKES"
    assert spectral_axis_numpy_index_from_header(h3) == 0

    h4 = Header()
    h4["NAXIS"] = 3
    h4["CTYPE1"] = "GLON-CAR"
    h4["CTYPE2"] = "GLAT-CAR"
    h4["CTYPE3"] = "XXXX"
    assert spectral_axis_numpy_index_from_header(h4) is None


def test_rrlsimulator_smoke_matches_cro(tmp_path):
    if lusee.RRLSimulator is None:
        pytest.skip("croissant/s2fft not installed")

    import fitsio
    import healpy as hp

    nside = 8
    npix = 12 * nside * nside
    maps = np.ones((3, npix), dtype=np.float32)
    ulsa_path = tmp_path / "u.fits"
    fitsio.write(
        str(ulsa_path),
        maps,
        header={"freq_start": 10.0, "freq_end": 12.0, "freq_step": 1.0},
    )
    rrl_path = tmp_path / "r.fits"
    fitsio.write(
        str(rrl_path),
        {"GLON": np.array([45.0]), "GLAT": np.array([0.0])},
        extname="SRC",
    )
    lmax = 3 * nside - 1
    sky = ULSAPlusRRLSky(
        str(ulsa_path),
        str(rrl_path),
        lmax,
        alpha_transitions=((700, 699),),
        rrl_sigma_mhz=1e6,
        rrl_peak_k=0.0,
    )

    freq = canonical_frequencies(frequency_indices_from_values([10.0, 11.0]))
    beam = lusee.BeamGauss(
        alt_deg=90.0,
        sigma_deg=25.0,
        one_over_freq_scaling=False,
        id="b",
    )
    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 03:00:00",
        deltaT_sec=3600.0,
        lun_lat_deg=0.0,
        lun_long_deg=0.0,
    )

    rsim = lusee.RRLSimulator(
        obs, [beam], sky, Tground=0.0, combinations=[(0, 0)], freq=freq, lmax=lmax
    )
    csim = lusee.CroSimulator(
        obs, [beam], sky, Tground=0.0, combinations=[(0, 0)], freq=freq, lmax=lmax
    )
    np.testing.assert_allclose(
        np.asarray(rsim.simulate(times=obs.times)),
        np.asarray(csim.simulate(times=obs.times)),
        rtol=0.0,
        atol=0.0,
    )
