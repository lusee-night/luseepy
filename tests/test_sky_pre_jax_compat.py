import os

os.environ["JAX_ENABLE_X64"] = "True"
os.environ.setdefault("LUSEE_DRIVE_DIR", "/Users/anigmetov/Data/lusee")
os.environ.setdefault("LUSEE_OUTPUT_DIR", "/Users/anigmetov/code/lusee_night/luseepy/out/luseepy_simulator_ng")

import astropy.units as u
import fitsio
import healpy as hp
import jax
import numpy as np
import pytest

import lusee.MonoSkyModels as monosky
import lusee.SkyModels as sky
import lusee.pre_jax.MonoSkyModels as pre_monosky
import lusee.pre_jax.SkyModels as pre_sky


RTOL = 1e-9
ATOL = 1e-9

assert jax.config.jax_enable_x64


def to_numpy(value):
    if isinstance(value, dict):
        return {key: to_numpy(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool)):
        return value
    if hasattr(value, "unit") and hasattr(value, "value"):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return value


def assert_same_value(old_value, new_value, label):
    old_value = to_numpy(old_value)
    new_value = to_numpy(new_value)

    if isinstance(old_value, dict):
        assert set(old_value) == set(new_value), label
        for key in sorted(old_value):
            assert_same_value(old_value[key], new_value[key], f"{label}.{key}")
        return

    if isinstance(old_value, list):
        assert len(old_value) == len(new_value), label
        for idx, (old_item, new_item) in enumerate(zip(old_value, new_value)):
            assert_same_value(old_item, new_item, f"{label}[{idx}]")
        return

    if isinstance(old_value, tuple):
        assert len(old_value) == len(new_value), label
        for idx, (old_item, new_item) in enumerate(zip(old_value, new_value)):
            assert_same_value(old_item, new_item, f"{label}[{idx}]")
        return

    if isinstance(old_value, u.Quantity) or isinstance(new_value, u.Quantity):
        assert old_value.unit == new_value.unit, label
        np.testing.assert_allclose(
            old_value.value,
            new_value.to_value(old_value.unit),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
            err_msg=label,
        )
        return

    if isinstance(old_value, np.ndarray) or isinstance(new_value, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(old_value),
            np.asarray(new_value),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
            err_msg=label,
        )
        return

    if np.isscalar(old_value) and np.isscalar(new_value) and not isinstance(old_value, (str, bytes, bool)):
        np.testing.assert_allclose(old_value, new_value, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=label)
        return

    assert old_value == new_value, label


def assert_same_members(old_obj, new_obj):
    assert set(old_obj.__dict__) == set(new_obj.__dict__)
    for name in sorted(old_obj.__dict__):
        assert_same_value(old_obj.__dict__[name], new_obj.__dict__[name], f"attr:{name}")


def assert_same_monosky_functions():
    freq = np.array([1.5, 3.0, 10.0, 25.0, 40.0])
    brightness = np.array([1e-20, 3e-20, 8e-20])

    for name, args in [
        ("B_NB", (freq,)),
        ("T_NB", (freq,)),
        ("B_C", (freq,)),
        ("T_C", (freq,)),
        ("T_J", (freq.copy(),)),
        ("B2V", (brightness, np.array([5.0, 10.0, 20.0]), 2.5, 0.7)),
        ("T_DarkAges", (freq,)),
        ("T_DarkAges_Scaled", (freq, 17.2, 12.5, 0.05)),
    ]:
        assert_same_value(getattr(pre_monosky, name)(*args), getattr(monosky, name)(*args), name)

    assert_same_value(pre_monosky.T2B(2500.0, freq), monosky.T2B(2500.0, freq), "T2B")
    assert_same_value(pre_monosky.B2T(brightness, np.array([5.0, 10.0, 20.0])), monosky.B2T(brightness, np.array([5.0, 10.0, 20.0])), "B2T")


def assert_same_constsky_methods(old_sky, new_sky, ndx):
    assert_same_members(old_sky, new_sky)
    assert_same_value(old_sky.T(ndx), new_sky.T(ndx), "T")
    assert_same_value(old_sky.get_alm(ndx), new_sky.get_alm(ndx), "get_alm")


def assert_same_healpixsky_methods(old_sky, new_sky, ndx, freq):
    assert_same_members(old_sky, new_sky)
    assert_same_value(old_sky.get_alm(ndx), new_sky.get_alm(ndx), "get_alm")
    assert_same_value(old_sky.get_alm(ndx, freq=freq), new_sky.get_alm(ndx, freq=freq), "get_alm_with_freq")


def test_monosky_models_match_pre_jax_reference():
    assert_same_monosky_functions()


def test_skymodels_match_pre_jax_reference(tmp_path):
    freq = np.array([10.0, 20.0])
    ndx = np.array([0, 1])
    nside = 8
    lmax = 6
    npix = hp.nside2npix(nside)

    old_const = pre_sky.ConstSky(nside, lmax, T=np.array([120.0, 140.0]), freq=freq, zero_cone=True)
    new_const = sky.ConstSky(nside, lmax, T=np.array([120.0, 140.0]), freq=freq, zero_cone=True)
    assert_same_constsky_methods(old_const, new_const, ndx)

    old_cane = pre_sky.ConstSkyCane1979(nside, lmax, freq=freq)
    new_cane = sky.ConstSkyCane1979(nside, lmax, freq=freq)
    assert_same_constsky_methods(old_cane, new_cane, ndx)

    old_dark = pre_sky.DarkAgesMonopole(nside, lmax, scaled=True, nu_min=17.0, nu_rms=12.0, A=0.05, freq=freq)
    new_dark = sky.DarkAgesMonopole(nside, lmax, scaled=True, nu_min=17.0, nu_rms=12.0, A=0.05, freq=freq)
    assert_same_constsky_methods(old_dark, new_dark, ndx)

    old_gal = pre_sky.GalCenter(nside, lmax, T=175.0, freq=freq)
    new_gal = sky.GalCenter(nside, lmax, T=175.0, freq=freq)
    assert_same_members(old_gal, new_gal)
    assert_same_value(old_gal.get_alm(ndx), new_gal.get_alm(ndx), "GalCenter.get_alm")

    maps = np.vstack([
        np.linspace(0.0, 1.0, npix),
        np.linspace(1.0, 0.0, npix),
    ])
    old_hp = pre_sky.HealpixSky(nside, lmax, maps, freq=freq, frame="galactic")
    new_hp = sky.HealpixSky(nside, lmax, maps, freq=freq, frame="galactic")
    assert_same_healpixsky_methods(old_hp, new_hp, ndx, freq)

    old_src = pre_sky.SingleSourceHealpixSky(Nside=nside, freq=freq, T=[2.0, 5.0], l_deg=30.0, b_deg=45.0)
    new_src = sky.SingleSourceHealpixSky(Nside=nside, freq=freq, T=[2.0, 5.0], l_deg=30.0, b_deg=45.0)
    assert_same_members(old_src, new_src)
    assert_same_value(old_src.get_alm(ndx, freq=freq), new_src.get_alm(ndx, freq=freq), "SingleSourceHealpixSky.get_alm")

    old_harm = pre_sky.HarmonicPointSourceSky(lmax=lmax, freq=freq, T=[1.5, 4.0], l_deg=20.0, b_deg=10.0)
    new_harm = sky.HarmonicPointSourceSky(lmax=lmax, freq=freq, T=[1.5, 4.0], l_deg=20.0, b_deg=10.0)
    assert_same_members(old_harm, new_harm)
    assert_same_value(old_harm.get_alm(ndx, freq=freq), new_harm.get_alm(ndx, freq=freq), "HarmonicPointSourceSky.get_alm")

    fits_maps = maps.astype(np.float64)
    fits_path = tmp_path / "test_sky.fits"
    fits = fitsio.FITS(str(fits_path), "rw", clobber=True)
    fits.write(fits_maps)
    fits[0].write_key("freq_start", float(freq[0]))
    fits[0].write_key("freq_end", float(freq[-1]))
    fits[0].write_key("freq_step", float(freq[1] - freq[0]))
    fits.close()

    old_fits = pre_sky.FitsSky(str(fits_path), lmax=lmax)
    new_fits = sky.FitsSky(str(fits_path), lmax=lmax)
    assert_same_healpixsky_methods(old_fits, new_fits, ndx, freq)


@pytest.mark.integration
def test_bale_plasma_effects_matches_pre_jax_reference(drive_dir):
    fname = os.path.join(drive_dir, "Simulations/SkyModels/PlasmaEffects", "R1_L300cm_n500t2k6v0nf300.sav")
    if not os.path.isfile(fname):
        pytest.skip(f"missing plasma effects file: {fname}")

    old_model = pre_monosky.BalePlasmaEffects()
    new_model = monosky.BalePlasmaEffects()

    assert_same_value(old_model._interp.x, new_model._interp.x, "BalePlasmaEffects._interp.x")
    assert_same_value(old_model._interp.y, new_model._interp.y, "BalePlasmaEffects._interp.y")
    sample_freq = np.linspace(old_model._interp.x[0], old_model._interp.x[-1], 5)
    assert_same_value(old_model(sample_freq), new_model(sample_freq), "BalePlasmaEffects.__call__")
