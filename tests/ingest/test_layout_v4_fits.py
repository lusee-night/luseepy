"""Layout-v4 FITS transport, parity, and validation tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import fitsio
import h5py
import numpy as np
import pytest
from astropy.io import fits
from test_layout_v4_hdf5 import make_all_family_request, make_request

from lusee.ingest import fits_writer
from lusee.ingest.constants import INGEST_LAYOUT_VERSION, NCHANNELS, NPRODUCTS
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.layout_v4_tree import (
    LayoutDataset,
    LayoutGroup,
    assert_layout_trees_equal,
)


def hdu_for_path(hdul, path: str):
    matches = [
        hdu
        for hdu in hdul[1:]
        if hdu.header.get("LUSEEPTH") == path
        and hdu.header.get("LUSEEKND") == "table"
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.fixture(scope="module")
def parity_files(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("layout-v4-parity")
    request = make_all_family_request()
    hdf5_path = directory / "all-products.h5"
    fits_path = directory / "all-products.fits"
    write_hdf5(request, hdf5_path)
    fits_writer.write_fits(request, fits_path)
    return request, hdf5_path, fits_path


def test_fits_dense_normal_and_tr_storage(parity_files):
    request, _, fits_path = parity_files
    with fits.open(fits_path, uint=True, memmap=False) as hdul:
        assert hdul[0].header["LAYOUTV"] == INGEST_LAYOUT_VERSION
        assert hdul[0].header["FITSFMT"] == 1
        assert hdul[0].header["QUALITY"] == "clean"

        spectra = hdu_for_path(hdul, "/spectra")
        normal = spectra.data["data"]
        assert normal.shape == (1, NPRODUCTS, NCHANNELS)
        assert normal.dtype == np.dtype(">f4")
        native_count = request.products.spectra[0].nfreq
        np.testing.assert_array_equal(
            normal[0, 3, :native_count],
            request.products.spectra[0].data[3],
        )
        assert np.isnan(normal[0, 3, native_count:]).all()
        assert np.isnan(normal[0, 0]).all()
        assert "product_present" not in spectra.columns.names
        assert "data_valid" not in spectra.columns.names

        tr_spectra = hdu_for_path(hdul, "/tr_spectra")
        tr = tr_spectra.data["data"]
        assert tr.shape == (1, NPRODUCTS, 4, 4)
        assert tr.dtype == np.dtype(">f8")
        assert tr[0, 7, 0, 0] == np.iinfo(np.int32).min
        assert tr[0, 7, 0, 1] == np.iinfo(np.int32).max
        assert np.isnan(tr[0, 0]).all()
        assert "product_present" not in tr_spectra.columns.names
        assert "data_valid" not in tr_spectra.columns.names


def test_hdf5_and_fits_layout_trees_are_semantically_identical(parity_files):
    _, hdf5_path, fits_path = parity_files
    expected = hdf5_tree(hdf5_path)
    with fits.open(fits_path, uint=True, memmap=False) as hdul:
        observed = fits_writer._read_layout_tree(hdul, fits)
    assert_layout_trees_equal(
        expected,
        observed,
        context="HDF5/FITS parity",
    )


def test_uint64_adc_timestamp_uses_k_tzero_and_round_trips(parity_files):
    _, _, fits_path = parity_files
    with fits.open(fits_path, uint=True, memmap=False) as hdul:
        waveform = hdu_for_path(hdul, "/waveform")
        column_index = waveform.columns.names.index("adc_timestamps") + 1
        assert waveform.header[f"TFORM{column_index}"].strip() == "K"
        assert waveform.header[f"TZERO{column_index}"] == 2**63
        assert waveform.header.get(f"TSCAL{column_index}") in (None, 1)
        observed = waveform.data["adc_timestamps"]
        assert observed.dtype == np.dtype(np.uint64)
        np.testing.assert_array_equal(
            observed,
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        )

    with fitsio.FITS(fits_path) as hdul:
        extension = next(
            index
            for index in range(1, len(hdul))
            if hdul[index].read_header().get("LUSEEPTH") == "/waveform"
        )
        observed = hdul[extension].read(columns=["adc_timestamps"])
        assert observed["adc_timestamps"].dtype == np.dtype(">u8")
        assert observed["adc_timestamps"][0] == np.iinfo(np.uint64).max


def test_groups_over_fits_column_limit_use_deterministic_partitions():
    root = LayoutGroup(
        path="",
        attrs={"quality_status": "clean", "execution_mode": "strict"},
    )
    group = LayoutGroup(path="/many")
    group.children.update(
        {
            f"field_{index:04d}": LayoutDataset(
                np.asarray([index], dtype=np.uint16)
            )
            for index in range(1000)
        }
    )
    root.children["many"] = group

    hdul = fits_writer._tree_to_hdul(root, fits)
    try:
        assert [hdu.header["LUSEEKND"] for hdu in hdul[1:]] == [
            "group",
            "table-part",
            "table-part",
        ]
        assert [len(hdu.columns) for hdu in hdul[2:]] == [999, 1]
        observed = fits_writer._read_layout_tree(hdul, fits)
    finally:
        hdul.close()
    assert_layout_trees_equal(root, observed, context="FITS column partitions")


def test_utf8_and_fixed_bytes_field_values_round_trip(tmp_path: Path):
    request = make_request()
    row = request.products.housekeeping[0]
    long_name = "a" * 1000
    request.products.housekeeping[0] = replace(
        row,
        fields={
            **row.fields,
            "greeting": "Gruesse λ",
            "empty_text": np.empty((0,), dtype="U1"),
            "raw_bytes": np.asarray([b"a\x00b"], dtype="S3"),
            long_name: np.int16(7),
        },
        field_present={
            **row.field_present,
            "greeting": True,
            "empty_text": True,
            "raw_bytes": True,
            long_name: True,
        },
    )
    destination = tmp_path / "text-and-bytes.fits"
    hdf5_path = tmp_path / "text-and-bytes.h5"

    fits_writer.write_fits(request, destination)
    write_hdf5(request, hdf5_path)

    with fits.open(destination, uint=True, memmap=False) as hdul:
        tree = fits_writer._read_layout_tree(hdul, fits)
    assert_layout_trees_equal(
        hdf5_tree(hdf5_path),
        tree,
        context="HDF5/FITS field union parity",
    )
    housekeeping = tree["housekeeping"]
    assert isinstance(housekeeping, LayoutGroup)
    greeting = housekeeping["fields/greeting/variant_000/data"]
    empty_text = housekeeping["fields/empty_text/variant_000/data"]
    raw_bytes = housekeeping["fields/raw_bytes/variant_000/data"]
    assert isinstance(greeting, LayoutDataset)
    assert isinstance(empty_text, LayoutDataset)
    assert isinstance(raw_bytes, LayoutDataset)
    assert greeting.data.tolist() == ["Gruesse λ"]
    assert empty_text.is_utf8
    assert empty_text.data.shape == (1, 0)
    long_presence = housekeeping[f"field_present/{long_name}"]
    assert isinstance(long_presence, LayoutDataset)
    assert long_presence.data.tolist() == [True]
    np.testing.assert_array_equal(
        raw_bytes.data,
        np.asarray([[b"a\x00b"]], dtype="S3"),
    )
    with fits.open(destination, uint=True, memmap=False) as hdul:
        presence_hdu = hdu_for_path(hdul, "/housekeeping/field_present")
        assert max(map(len, presence_hdu.columns.names)) <= 60
        assert long_name in presence_hdu.header["COLJSON"]


def test_writer_requires_write_request(tmp_path: Path):
    destination = tmp_path / "invalid.fits"

    with pytest.raises(TypeError, match="validated WriteRequest"):
        fits_writer.write_fits(make_request().products, destination)

    assert not destination.exists()


def test_existing_destination_is_refused_by_default(tmp_path: Path):
    destination = tmp_path / "existing.fits"
    original = b"do not replace"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError):
        fits_writer.write_fits(make_request(), destination)

    assert destination.read_bytes() == original


def test_overwrite_is_explicit_and_recorded(tmp_path: Path):
    destination = tmp_path / "existing.fits"
    destination.write_bytes(b"old contents")

    fits_writer.write_fits(make_request(overwrite=True), destination)

    with fits.open(destination, uint=True, memmap=False) as hdul:
        tree = fits_writer._read_layout_tree(hdul, fits)
    run = tree["run_provenance"]
    assert isinstance(run, LayoutGroup)
    assert run.attrs["overwrite_requested"] == np.bool_(True)
    assert run.attrs["destination_preexisted"] == np.bool_(True)


def test_close_time_verifier_rejects_valid_checksum_schema_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "malformed.fits"
    write_fits_file = fits_writer._write_fits_file

    def write_then_corrupt(path, root, fits_module):
        write_fits_file(path, root, fits_module)
        with fits_module.open(path, mode="update", uint=True, memmap=False) as hdul:
            target = hdu_for_path(hdul, "/spectra")
            del target.header["COLJSON"]
            for hdu in hdul:
                hdu.add_checksum()
            hdul.flush(output_verify="exception")

    monkeypatch.setattr(fits_writer, "_write_fits_file", write_then_corrupt)

    with pytest.raises(ValueError, match="FITS.*COLJSON"):
        fits_writer.write_fits(make_all_family_request(), destination)

    assert destination.exists()


@pytest.mark.parametrize(
    ("keyword", "value"),
    (("TNULL", np.iinfo(np.int32).min), ("TUNIT", "metre")),
)
def test_close_time_verifier_rejects_unexpected_column_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    keyword: str,
    value: object,
):
    destination = tmp_path / f"malformed-{keyword.lower()}.fits"
    write_fits_file = fits_writer._write_fits_file

    def write_then_corrupt(path, root, fits_module):
        write_fits_file(path, root, fits_module)
        with fits_module.open(path, mode="update", uint=True, memmap=False) as hdul:
            target = hdu_for_path(hdul, "/housekeeping")
            column_index = target.columns.names.index("firmware_errors") + 1
            target.header[f"{keyword}{column_index}"] = value
            for hdu in hdul:
                hdu.add_checksum()
            hdul.flush(output_verify="exception")

    monkeypatch.setattr(fits_writer, "_write_fits_file", write_then_corrupt)

    with pytest.raises(ValueError, match=f"FITS output header {keyword}"):
        fits_writer.write_fits(make_all_family_request(), destination)

    assert destination.exists()


def test_close_time_verifier_rejects_duplicate_contract_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "duplicate-contract-card.fits"
    write_fits_file = fits_writer._write_fits_file

    def write_then_corrupt(path, root, fits_module):
        write_fits_file(path, root, fits_module)
        with fits_module.open(path, mode="update", uint=True, memmap=False) as hdul:
            target = hdu_for_path(hdul, "/housekeeping")
            target.header.append(("COLJSON", "{}"))
            for hdu in hdul:
                hdu.add_checksum()
            hdul.flush(output_verify="exception")

    monkeypatch.setattr(fits_writer, "_write_fits_file", write_then_corrupt)

    with pytest.raises(ValueError, match="FITS output header COLJSON"):
        fits_writer.write_fits(make_all_family_request(), destination)

    assert destination.exists()


def hdf5_tree(path: Path) -> LayoutGroup:
    with h5py.File(path, "r") as h5:
        return hdf5_group(h5, "")


def hdf5_group(group, path: str) -> LayoutGroup:
    result = LayoutGroup(
        path=path,
        attrs={name: group.attrs[name] for name in group.attrs},
    )
    for name, child in group.items():
        child_path = f"{path}/{name}" if path else f"/{name}"
        if isinstance(child, h5py.Group):
            result.children[name] = hdf5_group(child, child_path)
            continue
        string_info = h5py.check_string_dtype(child.dtype)
        is_utf8 = (
            string_info is not None
            and string_info.encoding == "utf-8"
            and string_info.length is None
        )
        data = child.asstr()[:] if is_utf8 else child[:]
        result.children[name] = LayoutDataset(
            data=np.asarray(data, dtype=object if is_utf8 else None),
            is_utf8=is_utf8,
            attrs={attr_name: child.attrs[attr_name] for attr_name in child.attrs},
        )
    return result
