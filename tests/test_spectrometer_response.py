import numpy as np
import pytest

import lusee
from lusee import SpectrometerResponse as sr


def test_functions_are_exposed_on_package():
    assert lusee.spectrometer_response is sr.spectrometer_response
    assert lusee.spectrometer_response_zoom is sr.spectrometer_response_zoom


def test_load_cache_is_cached_and_well_formed():
    sr._load_cache.cache_clear()
    cache = sr._load_cache()
    # lru_cache returns the identical object on repeat calls (no re-read).
    assert sr._load_cache() is cache

    assert set(cache["response"]) == {0, 4, 16, 64}
    assert len(cache["response_zoom"]) == 64
    assert cache["freq"].ndim == 1


@pytest.mark.parametrize("notch", [0, 4, 16, 64])
def test_notch_response_scalar_and_array(notch):
    scalar = sr.spectrometer_response(0.0, notch=notch)
    assert np.isscalar(scalar) or scalar.shape == ()

    arr = sr.spectrometer_response(np.array([0.0, 10.0, 100.0]), notch=notch)
    assert arr.shape == (3,)
    assert np.all(np.isfinite(arr))


def test_notch_default_matches_zero():
    x = np.linspace(-100, 100, 21)
    assert np.array_equal(sr.spectrometer_response(x), sr.spectrometer_response(x, notch=0))


def test_response_is_zero_outside_grid():
    freq = sr._load_cache()["freq"]
    far = freq.max() * 10 + 1.0
    assert sr.spectrometer_response(far, notch=0) == 0.0
    assert sr.spectrometer_response_zoom(far, zoom_bin=0) == 0.0


def test_invalid_notch_raises():
    with pytest.raises(ValueError):
        sr.spectrometer_response(0.0, notch=3)


@pytest.mark.parametrize("zoom_bin", [0, 1, 32, 63, -1, -32])
def test_zoom_valid_bins(zoom_bin):
    val = sr.spectrometer_response_zoom(0.0, zoom_bin=zoom_bin)
    assert np.isfinite(val)


def test_zoom_negative_index_wraps():
    x = np.linspace(-50, 50, 11)
    # -32 wraps to 32, -1 wraps to 63.
    assert np.array_equal(
        sr.spectrometer_response_zoom(x, zoom_bin=-32), sr.spectrometer_response_zoom(x, zoom_bin=32)
    )
    assert np.array_equal(
        sr.spectrometer_response_zoom(x, zoom_bin=-1), sr.spectrometer_response_zoom(x, zoom_bin=63)
    )


@pytest.mark.parametrize("zoom_bin", [-33, 64, 100])
def test_zoom_out_of_range_raises(zoom_bin):
    with pytest.raises(ValueError):
        sr.spectrometer_response_zoom(0.0, zoom_bin=zoom_bin)
