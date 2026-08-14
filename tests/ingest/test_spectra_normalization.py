"""Focused tests for the layout-v3 normal-spectrum SDU invariant."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.decode import Products, SpectrumSample
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.obs_factory import IngestData, _concat_dict_arrays, _load_h5
from lusee.LabeledArray import label


def _products_with_varying_bitslices():
    # A product-constant cube makes a wrong row/product broadcast immediately
    # visible while keeping the fixture compact and exactly representable.
    decoded = np.empty((2, NPRODUCTS, NCHANNELS), dtype=np.float32)
    for row in range(decoded.shape[0]):
        for product in range(NPRODUCTS):
            decoded[row, product] = 4.0 * (1 + row + product)

    bitslices = np.array([
        np.arange(31, 15, -1),
        np.arange(16, 32),
    ], dtype=np.int16)
    gains = np.array([[2, 2, 2, 2], [1, 2, 1, 2]], dtype=np.int16)

    products = Products()
    for row in range(decoded.shape[0]):
        products.spectra.append(SpectrumSample(
            data=decoded[row].copy(),
            unique_packet_id=100 + row,
            raw_seconds=1000.0 + row,
            metadata={
                "actual_bitslice": bitslices[row].copy(),
                "bitslice": np.full(NPRODUCTS, 15, dtype=np.int16),
                "actual_gain": gains[row].copy(),
                "Navgf": 1,
            },
        ))
    expected = np.ldexp(decoded, bitslices[:, :, None] - 31)
    return products, decoded, bitslices, gains, expected


def test_v3_writer_normalizes_once_and_records_unambiguous_attrs(tmp_path: Path):
    products, _, bitslices, gains, expected = _products_with_varying_bitslices()
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"

    write_hdf5(products, first)
    # Reusing an in-memory Products object must not apply the power of two a
    # second time. This is the easiest regression path for double scaling.
    write_hdf5(products, second)

    for path in (first, second):
        with h5py.File(path, "r") as handle:
            assert int(handle.attrs["layout_version"]) == 3
            data = handle["spectra/data"]
            np.testing.assert_array_equal(data[...], expected)
            assert data.attrs["units"] == "SDU"
            assert data.attrs["representation"] == "gain_model_input_sdu"
            assert int(data.attrs["bitslice_restored"]) == 1
            assert int(data.attrs["bitslice_reference"]) == 31
            assert int(data.attrs["normalization_version"]) == 1

            actual_bitslice = handle["spectra/metadata/actual_bitslice"]
            assert actual_bitslice.shape == (2, NPRODUCTS)
            np.testing.assert_array_equal(actual_bitslice[...], bitslices)
            assert int(actual_bitslice.attrs["applied_to_spectra"]) == 1
            assert int(actual_bitslice.attrs["reference_bit"]) == 31

            actual_gain = handle["spectra/metadata/actual_gain"]
            assert actual_gain.shape == (2, 4)
            np.testing.assert_array_equal(actual_gain[...], gains)


def test_missing_actual_bitslice_fails_before_creating_a_file(tmp_path: Path):
    products = Products(spectra=[SpectrumSample(
        data=np.ones((NPRODUCTS, NCHANNELS), dtype=np.float32),
        unique_packet_id=1,
        raw_seconds=1.0,
        # A requested bitslice is intentionally not an acceptable substitute
        # for the realized value.
        metadata={"bitslice": np.full(NPRODUCTS, 16, dtype=np.int16)},
    )])
    destination = tmp_path / "must-not-exist.h5"

    with pytest.raises(ValueError, match="no actual_bitslice"):
        write_hdf5(products, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "bad_bitslice, message",
    [
        (np.full(NPRODUCTS, 31.5), "integral"),
        (np.full(NPRODUCTS, 32), r"\[0, 31\]"),
        (np.full(NPRODUCTS - 1, 16), "16 values"),
    ],
)
def test_malformed_actual_bitslice_is_a_hard_error(
    tmp_path: Path, bad_bitslice, message
):
    products = Products(spectra=[SpectrumSample(
        data=np.ones((NPRODUCTS, NCHANNELS), dtype=np.float32),
        unique_packet_id=1,
        raw_seconds=1.0,
        metadata={"actual_bitslice": bad_bitslice},
    )])
    destination = tmp_path / "bad.h5"

    with pytest.raises(ValueError, match=message):
        write_hdf5(products, destination)
    assert not destination.exists()


@pytest.mark.parametrize("gain", [None, np.array([0, 1, 2])])
def test_missing_or_malformed_actual_gain_fails_before_file_creation(
    tmp_path: Path, gain
):
    metadata = {
        "actual_bitslice": np.full(NPRODUCTS, 16, dtype=np.int16),
    }
    if gain is not None:
        metadata["actual_gain"] = gain
    products = Products(spectra=[SpectrumSample(
        data=np.ones((NPRODUCTS, NCHANNELS), dtype=np.float32),
        unique_packet_id=1,
        raw_seconds=1.0,
        metadata=metadata,
    )])
    destination = tmp_path / "bad-gain.h5"

    with pytest.raises(ValueError, match="actual_gain"):
        write_hdf5(products, destination)
    assert not destination.exists()


def test_layout_v2_and_v3_load_to_identical_in_memory_sdu(tmp_path: Path):
    products, decoded, bitslices, gains, expected = _products_with_varying_bitslices()
    v3 = tmp_path / "v3.h5"
    v2 = tmp_path / "v2.h5"
    write_hdf5(products, v3)

    # Construct the exact old representation: decoded bit-sliced values,
    # layout_version=2, no v3 representation attrs, and legacy metadata shapes.
    shutil.copy2(v3, v2)
    with h5py.File(v2, "r+") as handle:
        handle.attrs["layout_version"] = np.int64(2)
        data = handle["spectra/data"]
        data[...] = decoded
        for name in (
            "units", "representation", "bitslice_restored",
            "bitslice_reference", "normalization_version",
        ):
            if name in data.attrs:
                del data.attrs[name]
        metadata = handle["spectra/metadata"]
        del metadata["actual_bitslice"]
        metadata.create_dataset("actual_bitslice", data=bitslices[:, None, :])
        del metadata["actual_gain"]
        metadata.create_dataset("actual_gain", data=gains[:, None, :])

    v3_bundle = _load_h5(v3)
    with pytest.warns(RuntimeWarning, match="layout-v2 spectra were bit-slice restored"):
        v2_bundle = _load_h5(v2)

    np.testing.assert_array_equal(v3_bundle.spectra, expected)
    np.testing.assert_array_equal(v2_bundle.spectra, expected)
    np.testing.assert_array_equal(v2_bundle.spectra, v3_bundle.spectra)
    assert v2_bundle.spectra_units == v3_bundle.spectra_units == "SDU"
    assert (
        v2_bundle.spectra_representation
        == v3_bundle.spectra_representation
        == "gain_model_input_sdu"
    )
    assert v2_bundle.spectra_metadata["actual_bitslice"].shape == (2, NPRODUCTS)
    assert v2_bundle.spectra_metadata["actual_gain"].shape == (2, 4)


def test_layout_v3_refuses_missing_or_false_normalization_declaration(tmp_path: Path):
    products, *_ = _products_with_varying_bitslices()
    valid = tmp_path / "valid.h5"
    corrupt = tmp_path / "corrupt-bitslice.h5"
    wrong_version = tmp_path / "corrupt-version.h5"
    write_hdf5(products, valid)
    shutil.copy2(valid, corrupt)
    shutil.copy2(valid, wrong_version)

    with h5py.File(corrupt, "r+") as handle:
        handle["spectra/data"].attrs["bitslice_restored"] = np.int64(0)

    with pytest.raises(ValueError, match="bitslice_restored=1"):
        _load_h5(corrupt)

    with h5py.File(wrong_version, "r+") as handle:
        handle["spectra/data"].attrs["normalization_version"] = np.int64(99)

    with pytest.raises(ValueError, match="normalization_version"):
        _load_h5(wrong_version)


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
    data.spectra = label(raw, units="SDU", frame="topo")
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

    # Both forward and conjugated/reversed raw cross indexing remain SDU.
    assert data[:, (0, 1, "C"), :].units == "SDU"
    assert data[:, (1, 0, "C"), :].units == "SDU"

    asd = data.to_physical(chunk_size=1)
    psd = data.to_physical_psd(chunk_size=1)
    native_psd = data.to_psd(units="nV^2/Hz")
    assert asd.units == "nV/sqrt(Hz)" and asd.frame == "topo"
    assert psd.units == "V^2/Hz" and psd.frame == "topo"
    assert native_psd.units == "nV^2/Hz" and native_psd.frame == "topo"
    # Lazy views do not alter the normalized stored values or decoration.
    np.testing.assert_array_equal(np.asarray(data.spectra), raw)
    assert data.spectra.units == "SDU"


def test_realized_gain_codes_are_not_silently_rounded():
    assert IngestData._level_from_code(0) == "L"
    assert IngestData._level_from_code(1.0) == "M"
    assert IngestData._level_from_code(b"H") == "H"
    assert IngestData._level_from_code(1.5) is None
    assert IngestData._level_from_code(np.nan) is None
    assert IngestData._level_from_code(4) is None  # firmware auto-gain state


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
