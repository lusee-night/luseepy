"""Restored-SDU contracts for layout v4 and legacy normalization readers."""

from __future__ import annotations

import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
from test_layout_v4_hdf5 import make_all_family_request
from test_layout_v4_hdf5_products import make_metadata

from lusee.ingest.constants import (
    BITSLICE_REFERENCE,
    NCHANNELS,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    SPECTRA_REPRESENTATION,
    SPECTRA_UNITS,
)
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.layout_v4_reader import LayoutV4ValidationError
from lusee.ingest.obs_factory import (
    IngestData,
    _concat_dict_arrays,
    _load_h5,
    load_bundle,
)
from lusee.ingest.products import SpectrumMetadata
from lusee.LabeledArray import label

h5py = pytest.importorskip("h5py")


def normalized_request():
    """Return one strict request with product-varying realized bit slices."""
    request = make_all_family_request()
    original = request.products.spectra[0]
    bitslices = np.arange(
        BITSLICE_REFERENCE,
        BITSLICE_REFERENCE - NPRODUCTS,
        -1,
        dtype=np.uint8,
    )
    gains = np.array([2, 2, 1, 1], dtype=np.uint8)
    decoded = np.empty_like(original.data)
    for product in range(NPRODUCTS):
        decoded[product] = np.float32(4.0 * (product + 1))
    expected = np.ldexp(
        decoded,
        bitslices.astype(np.int16)[:, None] - BITSLICE_REFERENCE,
    )
    metadata = replace(
        original.metadata,
        actual_bitslice=bitslices,
        actual_gain=gains,
    )
    row = replace(
        original,
        data=expected,
        product_present=np.ones(NPRODUCTS, dtype=np.bool_),
        metadata=metadata,
    )
    request.products.spectra[0] = row
    request.validate()
    return request, decoded, bitslices, gains, expected


def spectrum_metadata_values() -> dict[str, object]:
    metadata = make_metadata(101, 100.0)
    return {
        item.name: getattr(metadata, item.name)
        for item in fields(metadata)
        if item.init
    }


def legacy_spectra_arrays():
    decoded = np.empty((2, NPRODUCTS, NCHANNELS), dtype=np.float32)
    for row in range(decoded.shape[0]):
        for product in range(NPRODUCTS):
            decoded[row, product] = np.float32(4.0 * (1 + row + product))
    bitslices = np.array(
        [
            np.arange(31, 15, -1),
            np.arange(16, 32),
        ],
        dtype=np.uint8,
    )
    gains = np.array([[2, 2, 2, 2], [1, 2, 1, 2]], dtype=np.uint8)
    expected = np.ldexp(
        decoded,
        bitslices.astype(np.int16)[:, :, None] - BITSLICE_REFERENCE,
    )
    return decoded, bitslices, gains, expected


def write_legacy_spectra(
    path: Path,
    layout_version: int,
    decoded: np.ndarray,
    bitslices: np.ndarray,
    gains: np.ndarray,
    expected: np.ndarray,
) -> None:
    """Write a direct layout-v2/v3 fixture without a production writer."""
    with h5py.File(path, "w") as handle:
        handle.attrs["layout_version"] = np.uint16(layout_version)
        spectra = handle.create_group("spectra")
        data = spectra.create_dataset(
            "data",
            data=(decoded if layout_version == 2 else expected),
        )
        if layout_version == 3:
            data.attrs["units"] = SPECTRA_UNITS
            data.attrs["representation"] = SPECTRA_REPRESENTATION
            data.attrs["bitslice_restored"] = np.uint8(1)
            data.attrs["bitslice_reference"] = np.uint8(BITSLICE_REFERENCE)
            data.attrs["normalization_version"] = np.uint16(
                SPECTRA_NORMALIZATION_VERSION
            )
        metadata = spectra.create_group("metadata")
        metadata.create_dataset(
            "actual_bitslice",
            data=(bitslices[:, None, :] if layout_version == 2 else bitslices),
        )
        metadata.create_dataset(
            "actual_gain",
            data=(gains[:, None, :] if layout_version == 2 else gains),
        )


def test_v4_writer_persists_restored_sdu_exactly_once(tmp_path: Path):
    request, _, bitslices, gains, expected = normalized_request()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"

    request.products.spectra[0].restore_bitslice()
    write_hdf5(request, first)
    write_hdf5(request, second)

    for path in (first, second):
        with h5py.File(path, "r") as handle:
            assert int(handle.attrs["layout_version"]) == 4
            data = handle["spectra/data"]
            np.testing.assert_array_equal(data[0, :, : expected.shape[1]], expected)
            assert np.isnan(data[0, :, expected.shape[1] :]).all()
            assert data.attrs["units"] == SPECTRA_UNITS
            assert data.attrs["representation"] == SPECTRA_REPRESENTATION
            assert data.attrs["bitslice_restored"] == np.bool_(True)
            assert int(data.attrs["bitslice_reference"]) == BITSLICE_REFERENCE
            assert (
                int(data.attrs["normalization_version"])
                == SPECTRA_NORMALIZATION_VERSION
            )

            metadata = handle["spectra/metadata/fields"]
            np.testing.assert_array_equal(
                metadata["actual_bitslice/variant_000/data"][0],
                bitslices,
            )
            np.testing.assert_array_equal(
                metadata["actual_gain/variant_000/data"][0],
                gains,
            )

    np.testing.assert_array_equal(request.products.spectra[0].data, expected)
    request.products.spectra[0].restore_bitslice()


@pytest.mark.parametrize("missing", ("actual_bitslice", "actual_gain"))
def test_missing_realized_metadata_fails_at_strict_record_boundary(missing: str):
    values = spectrum_metadata_values()
    del values[missing]

    with pytest.raises(TypeError, match=missing):
        SpectrumMetadata(**values)


@pytest.mark.parametrize(
    ("field", "bad_value", "error", "message"),
    (
        (
            "actual_bitslice",
            np.full(NPRODUCTS, 31.5),
            TypeError,
            "dtype uint8",
        ),
        (
            "actual_bitslice",
            np.full(NPRODUCTS, 32, dtype=np.uint8),
            ValueError,
            "actual_bitslice values",
        ),
        (
            "actual_bitslice",
            np.full(NPRODUCTS - 1, 16, dtype=np.uint8),
            ValueError,
            "shape",
        ),
        ("actual_gain", None, TypeError, "numpy.ndarray"),
        (
            "actual_gain",
            np.array([0, 1, 2], dtype=np.uint8),
            ValueError,
            "shape",
        ),
    ),
)
def test_malformed_realized_metadata_fails_at_strict_record_boundary(
    field: str,
    bad_value,
    error,
    message: str,
):
    values = spectrum_metadata_values()
    values[field] = bad_value

    with pytest.raises(error, match=message):
        SpectrumMetadata(**values)


def test_layout_v2_and_v3_load_to_identical_in_memory_sdu(tmp_path: Path):
    decoded, bitslices, gains, expected = legacy_spectra_arrays()
    v2 = tmp_path / "v2.h5"
    v3 = tmp_path / "v3.h5"
    write_legacy_spectra(v2, 2, decoded, bitslices, gains, expected)
    write_legacy_spectra(v3, 3, decoded, bitslices, gains, expected)

    with pytest.warns(
        RuntimeWarning,
        match="layout-v2 spectra were bit-slice restored",
    ):
        v2_bundle = _load_h5(v2)
    v3_bundle = _load_h5(v3)

    np.testing.assert_array_equal(v2_bundle.spectra, expected)
    np.testing.assert_array_equal(v3_bundle.spectra, expected)
    np.testing.assert_array_equal(v2_bundle.spectra, v3_bundle.spectra)
    assert v2_bundle.spectra_units == v3_bundle.spectra_units == SPECTRA_UNITS
    assert (
        v2_bundle.spectra_representation
        == v3_bundle.spectra_representation
        == SPECTRA_REPRESENTATION
    )
    assert v2_bundle.spectra_metadata["actual_bitslice"].shape == (2, NPRODUCTS)
    assert v2_bundle.spectra_metadata["actual_gain"].shape == (2, 4)


@pytest.mark.parametrize(
    ("attribute", "bad_value", "message"),
    (
        ("bitslice_restored", np.bool_(False), "bitslice_restored disagrees"),
        (
            "normalization_version",
            np.uint16(99),
            "normalization_version disagrees",
        ),
    ),
)
def test_layout_v4_rejects_corrupt_normalization_declaration(
    tmp_path: Path,
    attribute: str,
    bad_value,
    message: str,
):
    valid = tmp_path / "valid.h5"
    corrupt = tmp_path / f"corrupt-{attribute}.h5"
    request, *_ = normalized_request()
    write_hdf5(request, valid)
    shutil.copy2(valid, corrupt)
    with h5py.File(corrupt, "r+") as handle:
        handle["spectra/data"].attrs[attribute] = bad_value

    with pytest.raises(LayoutV4ValidationError, match=message):
        load_bundle(corrupt)


def test_lazy_conversion_and_indexing_keep_distinct_unit_decorations():
    data = object.__new__(IngestData)
    nrow = 2
    freqs = np.array([
        0.1, 0.7, 1.1, 3.1, 5.1, 10.1, 15.1, 20.1,
        25.1, 30.1, 35.1, 40.1, 45.1, 50.1, 60.1, 70.1,
    ])
    raw = np.ones((nrow, NPRODUCTS, freqs.size), dtype=np.float32)
    raw[:, 4] = 3.0
    raw[:, 5] = -4.0
    data.spectra = label(raw, units=SPECTRA_UNITS, frame="topo")
    data.layout_version = 3
    data.Nspectra = nrow
    data.Nfreq = freqs.size
    data.freq = freqs
    data.metadata = {
        "actual_gain": np.full((nrow, 4), 2, dtype=np.int16),
    }
    telemetry_values = {
        "THERM_FPGA": 30.4,
        "SPE_ADC0_T": 29.8,
        "SPE_ADC1_T": 28.5,
        "SPE_1VAD8_V": 1.799,
        "VMON_1V2D": 1.201,
        "SPE_1VAD8_C": 0.045,
    }
    data.interp_telemetry = {
        key: np.full(nrow, value) for key, value in telemetry_values.items()
    }

    assert data[:, (0, 1, "C"), :].units == SPECTRA_UNITS
    assert data[:, (1, 0, "C"), :].units == SPECTRA_UNITS

    asd = data.to_physical(telemetry=data.interp_telemetry, chunk_size=1)
    psd = data.to_physical_psd(
        telemetry=data.interp_telemetry,
        chunk_size=1,
    )
    scalar_asd = data.to_physical(telemetry=telemetry_values, chunk_size=1)
    scalar_psd = data.to_physical_psd(
        telemetry=telemetry_values,
        chunk_size=1,
    )
    native_psd = data.to_psd(units="nV^2/Hz")
    assert asd.units == "nV/sqrt(Hz)" and asd.frame == "topo"
    assert psd.units == "V^2/Hz" and psd.frame == "topo"
    assert native_psd.units == "nV^2/Hz" and native_psd.frame == "topo"
    np.testing.assert_array_equal(scalar_asd, asd)
    np.testing.assert_array_equal(scalar_psd, psd)
    np.testing.assert_array_equal(np.asarray(data.spectra), raw)
    assert data.spectra.units == SPECTRA_UNITS

    data.interp_telemetry = {}
    for convert in (data.to_physical, data.to_physical_psd):
        with pytest.raises(ValueError, match="no row-aligned gain telemetry"):
            convert()


def test_realized_gain_codes_are_not_silently_rounded():
    assert IngestData._level_from_code(0) == "L"
    assert IngestData._level_from_code(1.0) == "M"
    assert IngestData._level_from_code(b"H") == "H"
    assert IngestData._level_from_code(1.5) is None
    assert IngestData._level_from_code(np.nan) is None
    assert IngestData._level_from_code(4) is None


def test_concatenated_metadata_nan_fills_missing_sources_with_field_shape():
    combined = _concat_dict_arrays(
        [
            {"actual_bitslice": np.ones((2, 16), dtype=np.int16)},
            {"actual_gain": np.ones((3, 4), dtype=np.int16)},
        ],
        n_per_source=[2, 3],
    )
    assert combined["actual_bitslice"].shape == (5, 16)
    assert combined["actual_gain"].shape == (5, 4)
    assert np.isnan(combined["actual_bitslice"][2:]).all()
    assert np.isnan(combined["actual_gain"][:2]).all()
