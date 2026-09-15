"""Explicit zoom scaling with native FITS/HDF5 and bundle preservation."""

from dataclasses import replace
import hashlib

import h5py
import numpy as np
import pytest
from test_layout_v4_hdf5 import make_all_family_request

from lusee.ingest import (
    IngestData,
    load,
    load_bundle,
    normalize_zoom,
    white_noise_notch_correction,
    write_fits,
    write_hdf5,
)


@pytest.fixture(scope="module")
def zoom_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("zoom-normalization")
    request = make_all_family_request()
    powers = np.asarray([1.0, 2.0, -0.5, 0.25])[:, None]
    expected = powers * np.ones((4, 64))
    native = (expected * (64 * 2**31)).astype(np.float32)
    request.products.zoom_spectra[0] = replace(
        request.products.zoom_spectra[0], data=native,
    )
    h5 = directory / "session.h5"
    fits = directory / "session.fits"
    write_hdf5(request, h5)
    write_fits(request, fits)
    return (h5, fits), native[None], expected[None]


@pytest.mark.parametrize("format_index", [0, 1])
def test_opt_in_and_repeated_calls_preserve_files_and_bundles(zoom_files, format_index):
    files, native, expected = zoom_files
    path = files[format_index]
    before = hashlib.sha256(path.read_bytes()).digest()
    data = load(path)
    normal = data.spectra.copy()
    assert data.zoom_representation == "native_float32"
    np.testing.assert_array_equal(data.zoom_spectra, native)
    assert data.normalize_zoom() is data
    assert data.zoom_representation == "bit31_sdu"
    np.testing.assert_array_equal(data.zoom_spectra, expected)
    assert data.zoom_spectra.dtype == np.float64
    normalized_array = data.zoom_spectra
    data.normalize_zoom()
    assert data.zoom_spectra is normalized_array
    np.testing.assert_array_equal(data.spectra, normal)
    for bundle in [data.bundle, *data.bundles]:
        np.testing.assert_array_equal(bundle.zoom_spectra, native)
        assert not np.shares_memory(data.zoom_spectra, bundle.zoom_spectra)
        np.testing.assert_array_equal(
            bundle.product_records["zoom_spectra"][0].data, native[0],
        )
    for constructor in (load, IngestData):
        converted = constructor(path, normalize_zoom=True)
        np.testing.assert_array_equal(converted.zoom_spectra, expected)
    np.testing.assert_array_equal(load_bundle(path).zoom_spectra, native)
    assert hashlib.sha256(path.read_bytes()).digest() == before


def test_switching_conventions_restarts_from_native_values(zoom_files):
    files, native, expected = zoom_files
    data = load(files[0], normalize_zoom=True)
    data.normalize_zoom(convention="pfb")
    assert data.zoom_representation == "pfb_power"
    np.testing.assert_array_equal(data.zoom_spectra, expected * 2**31)
    data.normalize_zoom()
    np.testing.assert_array_equal(data.zoom_spectra, expected)
    np.testing.assert_array_equal(data.bundle.zoom_spectra, native)
    with pytest.raises(ValueError, match="convention"):
        data.normalize_zoom(convention="unknown")
    np.testing.assert_array_equal(data.zoom_spectra, expected)


def test_multiple_files_and_absent_zoom(zoom_files):
    files, _, expected = zoom_files
    data = load(files, normalize_zoom=True)
    np.testing.assert_array_equal(data.zoom_spectra, np.concatenate([expected] * 2))
    empty = replace(load_bundle(files[0]), zoom_spectra=None)
    data = IngestData(empty, normalize_zoom=True)
    assert data.normalize_zoom() is data
    assert data.zoom_spectra is None
    assert data.zoom_representation == "native_float32"


def test_native_storage_declaration_stays_unchanged(zoom_files):
    files, native, _ = zoom_files
    with h5py.File(files[0]) as h5:
        data = h5["calibrator/zoom_spectra/data"]
        np.testing.assert_array_equal(data[...], native)
        assert data.attrs["representation"] == "native_float32"
        assert data.attrs["units"] == "unit_unestablished"
        assert "zoom_normalized" not in data.attrs


def test_normalization_flag_requires_a_boolean(zoom_files):
    with pytest.raises(TypeError, match="boolean"):
        load(zoom_files[0][0], normalize_zoom=64)


@pytest.mark.parametrize("convention,divisor", [("sdu", 64 * 2**31), ("pfb", 64)])
def test_array_conversion_known_values(zoom_files, convention, divisor):
    _, native, _ = zoom_files
    np.testing.assert_array_equal(
        normalize_zoom(native, convention=convention), native.astype(float) / divisor,
    )


def test_array_conversion_input_contract():
    values = np.full((4, 64), np.nextafter(np.float32(0), np.float32(1)))
    assert normalize_zoom(values)[0, 0] == 2.0**-186
    with pytest.raises(ValueError, match="shape"):
        normalize_zoom(np.zeros((64, 4)))
    with pytest.raises(ValueError, match="convention"):
        normalize_zoom(values, convention="unknown")
    with pytest.raises(TypeError, match="real numeric"):
        normalize_zoom(values.astype(complex))


def test_notch_factor_honors_subtraction_and_detector_bits():
    metadata = np.array([0, 2, 4, 6, 4 | 16, 6 | 16, 4 | 32], dtype=np.uint8)
    expected = [1, 4 / 3, 16 / 15, 64 / 63, 1, 1, 16 / 15]
    np.testing.assert_array_equal(white_noise_notch_correction(metadata), expected)
    assert white_noise_notch_correction(0) == 1
    with pytest.raises(TypeError, match="integer"):
        white_noise_notch_correction(4.0)
    with pytest.raises(ValueError, match="mode"):
        white_noise_notch_correction(3)
    with pytest.raises(ValueError, match="255"):
        white_noise_notch_correction(-1)
