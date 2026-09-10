"""Strict generic reader tests for HDF5 and FITS ingest layout v4."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits
from test_layout_v4_hdf5 import make_all_family_request, make_request

from lusee.ingest.fits_writer import _decode_attrs, _encode_attrs, write_fits
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.layout_v4_reader import (
    LayoutV4ValidationError,
    read_layout_v4,
    read_layout_v4_fits,
    read_layout_v4_hdf5,
    validate_layout_v4_tree,
)
from lusee.ingest.layout_v4_tree import (
    LayoutDataset,
    LayoutGroup,
    assert_layout_trees_equal,
)


@pytest.fixture(scope="module")
def all_family_files(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("layout-v4-reader")
    request = make_all_family_request()
    hdf5_path = directory / "all-products.h5"
    fits_path = directory / "all-products.fits"
    write_hdf5(request, hdf5_path)
    write_fits(request, fits_path)
    return hdf5_path, fits_path


def test_hdf5_and_fits_readers_return_the_same_validated_tree(all_family_files):
    hdf5_path, fits_path = all_family_files

    hdf5_tree = read_layout_v4_hdf5(hdf5_path)
    fits_tree = read_layout_v4_fits(fits_path)

    assert_layout_trees_equal(
        hdf5_tree,
        fits_tree,
        context="strict layout-v4 readers",
    )
    assert_layout_trees_equal(
        hdf5_tree,
        read_layout_v4(hdf5_path),
        context="detected HDF5 layout-v4 reader",
    )
    assert_layout_trees_equal(
        fits_tree,
        read_layout_v4(fits_path),
        context="detected FITS layout-v4 reader",
    )


def test_hdf5_reader_preserves_utf8_and_fixed_byte_dtypes(tmp_path: Path):
    request = make_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(
        row,
        fields={
            **row.fields,
            "greeting": "Gruesse λ",
            "raw_bytes": np.asarray([b"a\x00b"], dtype="S3"),
        },
        field_present={
            **row.field_present,
            "greeting": True,
            "raw_bytes": True,
        },
    )
    destination = tmp_path / "strings.h5"
    write_hdf5(request, destination)

    tree = read_layout_v4_hdf5(destination)
    greeting = tree["housekeeping/fields/greeting/variant_000/data"]
    raw_bytes = tree["housekeeping/fields/raw_bytes/variant_000/data"]

    assert isinstance(greeting, LayoutDataset)
    assert greeting.is_utf8
    assert greeting.dtype == np.dtype(object)
    assert greeting.data.tolist() == ["Gruesse λ"]
    assert isinstance(raw_bytes, LayoutDataset)
    assert not raw_bytes.is_utf8
    assert raw_bytes.dtype == np.dtype("S3")
    np.testing.assert_array_equal(
        raw_bytes.data,
        np.asarray([[b"a\x00b"]], dtype="S3"),
    )


def test_validator_exposes_exact_shape_dtype_and_row_primitives(all_family_files):
    hdf5_path, _ = all_family_files
    validator = validate_layout_v4_tree(read_layout_v4_hdf5(hdf5_path))

    spectra = validator.dataset(
        "/spectra/data",
        dtype=np.float32,
        utf8=False,
        shape=(1, 16, 2048),
        row_count=1,
        tail_shape=(16, 2048),
    )
    assert spectra.data.ndim == 3
    assert validator.require_row_aligned(
        "/spectra",
        ("data", "unique_ids", "frequency_counts"),
    ) == 1

    with pytest.raises(LayoutV4ValidationError, match="aligned to 2 rows"):
        validator.dataset("/spectra/data", row_count=2)
    with pytest.raises(LayoutV4ValidationError, match="expected float64"):
        validator.dataset("/spectra/data", dtype=np.float64)


def test_validator_rejects_missing_required_provenance(all_family_files):
    hdf5_path, _ = all_family_files
    tree = read_layout_v4_hdf5(hdf5_path)
    run_provenance = tree["run_provenance"]
    assert isinstance(run_provenance, LayoutGroup)
    del run_provenance.attrs["source_kind_valid"]

    with pytest.raises(LayoutV4ValidationError, match="source_kind_valid"):
        validate_layout_v4_tree(tree)


def test_hdf5_reader_rejects_wrong_layout_version(tmp_path: Path):
    destination = tmp_path / "wrong-layout.h5"
    write_hdf5(make_request(), destination)
    with h5py.File(destination, "r+") as h5:
        h5.attrs.modify("layout_version", np.uint16(3))

    with pytest.raises(LayoutV4ValidationError, match="not ingest layout version 4"):
        read_layout_v4_hdf5(destination)


def test_fits_reader_rejects_wrong_layout_with_valid_checksums(tmp_path: Path):
    destination = tmp_path / "wrong-layout.fits"
    write_fits(make_request(), destination)
    with fits.open(destination, mode="update", uint=True, memmap=False) as hdul:
        hdul[0].header["LAYOUTV"] = 3
        for hdu in hdul:
            hdu.add_checksum()
        hdul.flush(output_verify="exception")

    with pytest.raises(ValueError, match="primary contract disagrees"):
        read_layout_v4_fits(destination)


def test_fits_reader_requires_valid_checksum_and_datasum(tmp_path: Path):
    destination = tmp_path / "bad-checksum.fits"
    write_fits(make_request(), destination)
    with fits.open(destination, mode="readonly", uint=True, memmap=False) as hdul:
        data_offset = next(
            hdu.fileinfo()["datLoc"]
            for hdu in hdul[1:]
            if hdu.data is not None and hdu.fileinfo()["datSpan"] > 0
        )
    with destination.open("r+b") as stream:
        stream.seek(data_offset)
        original = stream.read(1)
        stream.seek(data_offset)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(LayoutV4ValidationError, match="checksum contract failed"):
        read_layout_v4_fits(destination)


def test_readers_reject_out_of_contract_lunar_location(tmp_path: Path):
    hdf5_path = tmp_path / "bad-location.h5"
    write_hdf5(make_request(), hdf5_path)
    with h5py.File(hdf5_path, "r+") as h5:
        h5["constants"].attrs.modify("lun_lat_deg", np.float64(999.0))

    with pytest.raises(LayoutV4ValidationError, match="latitude_deg"):
        read_layout_v4_hdf5(hdf5_path)

    fits_path = tmp_path / "bad-location.fits"
    write_fits(make_request(), fits_path)
    with fits.open(fits_path, mode="update", uint=True, memmap=False) as hdul:
        constants = next(
            hdu
            for hdu in hdul[1:]
            if hdu.header.get("LUSEEPTH") == "/constants"
        )
        attributes = _decode_attrs(constants.header["ATTRJSON"])
        attributes["lun_lat_deg"] = np.float64(999.0)
        constants.header["ATTRJSON"] = _encode_attrs(attributes)
        for hdu in hdul:
            hdu.add_checksum()
        hdul.flush(output_verify="exception")

    with pytest.raises(LayoutV4ValidationError, match="latitude_deg"):
        read_layout_v4_fits(fits_path)


def test_fits_reader_regenerates_and_checks_exact_transport_headers(tmp_path: Path):
    destination = tmp_path / "wrong-transport-header.fits"
    write_fits(make_request(), destination)
    with fits.open(destination, mode="update", uint=True, memmap=False) as hdul:
        target = next(hdu for hdu in hdul[1:] if isinstance(hdu, fits.BinTableHDU))
        target.header["TUNIT1"] = "invented-unit"
        for hdu in hdul:
            hdu.add_checksum()
        hdul.flush(output_verify="exception")

    with pytest.raises(ValueError, match="header TUNIT1 disagrees"):
        read_layout_v4_fits(destination)


def test_generic_reader_rejects_unknown_file_signature(tmp_path: Path):
    destination = tmp_path / "not-layout-v4.dat"
    destination.write_bytes(b"not an ingest file")

    with pytest.raises(LayoutV4ValidationError, match="neither HDF5 nor FITS"):
        read_layout_v4(destination)
