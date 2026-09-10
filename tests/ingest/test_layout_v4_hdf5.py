"""Atomic layout-v4 HDF5 writer contract tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
from test_layout_v4_hdf5_products import make_products as make_all_family_products

from lusee.ingest import hdf5_writer
from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.decode import Products
from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    HKSample,
    ProductProvenance,
    SourcePacketProvenance,
    ValidatedCounts,
)
from lusee.ingest.write_request import (
    ALL_FAMILIES,
    LunarLocation,
    RunProvenance,
    WriteRequest,
    family_statuses_for_products,
)


def make_products() -> Products:
    row_provenance = ProductProvenance(
        source_packets=(
            SourcePacketProvenance(
                role="housekeeping",
                original_appid=0x212,
                packet_index=0,
            ),
        ),
        uid=7,
        uid_source="housekeeping.unique_packet_id",
        uid_source_role="housekeeping",
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        time_valid=False,
    )
    housekeeping = HKSample(
        hk_type=1,
        version=0x307,
        unique_packet_id=7,
        errors=0,
        raw_seconds=None,
        fields={"temperature": 21.5},
        field_present={"temperature": True},
        provenance=row_provenance,
    )
    decode_provenance = DecodeProvenance.from_report(
        distribution_version="1.2.3",
        decoder_source_commit="a" * 40,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant=None,
        schema_assumed=False,
        binding_source_release="3r09",
        binding_source_commit="b" * 40,
        abi_fingerprint="c" * 64,
        execution_mode="collect",
        input_packet_count=1,
        valid_packet_count=1,
        appid_counts=((0x212, 1),),
        issue_counts=(),
        canonical_report={"fixture": "strict"},
    )
    return Products(
        housekeeping=[housekeeping],
        decode_provenance=decode_provenance,
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=1,
            valid_packets=1,
            product_rows=(("housekeeping", 1),),
        ),
        issues=(),
    )


def make_request(*, overwrite: bool = False) -> WriteRequest:
    products = make_products()
    return WriteRequest(
        products=products,
        clock_reference_set=None,
        clock_reference_unavailable_reason="fixture has no clock anchor",
        location=LunarLocation(-23.814, 182.258, 0.0),
        run_provenance=RunProvenance(
            input_identity="fixture-sha256",
            input_identity_kind="sha256",
            input_identity_unavailable_reason=None,
            source_kind="uncrater_session",
        ),
        issues=products.issues,
        family_statuses=family_statuses_for_products(
            products,
            family_issue_ids={},
        ),
        overwrite=overwrite,
    )


def make_all_family_request() -> WriteRequest:
    products = make_all_family_products()
    clock_reference_set = ClockReferenceSet(
        format_version=1,
        reference_event="landing",
        clock_reference_isot="2026-01-01T00:00:00",
        time_scale="utc",
        clocks=(
            ClockReference(
                clock_source=ClockSource.SPECTROMETER,
                clock_reference_raw_seconds=100.0,
            ),
        ),
        source="test fixture",
        assumed=False,
        source_sha256="d" * 64,
    )
    return WriteRequest(
        products=products,
        clock_reference_set=clock_reference_set,
        clock_reference_unavailable_reason=None,
        location=LunarLocation(-23.814, 182.258, 0.0),
        run_provenance=RunProvenance(
            input_identity="e" * 64,
            input_identity_kind="sha256",
            input_identity_unavailable_reason=None,
            source_kind="synthetic_fixture",
        ),
        issues=products.issues,
        family_statuses=family_statuses_for_products(
            products,
            family_issue_ids={},
        ),
        hdf5_compression=None,
        hdf5_compression_level=None,
    )


def test_writer_requires_write_request(tmp_path: Path):
    destination = tmp_path / "invalid.h5"

    with pytest.raises(TypeError, match="validated WriteRequest"):
        hdf5_writer.write_hdf5(make_products(), destination)

    assert not destination.exists()


def test_writer_revalidates_mutated_session_invariants_before_output(tmp_path: Path):
    request = make_request()
    products = request.products
    products.sw_version = 0x307
    products.fw_version = 1
    products.fw_id = 2
    products.fw_date = 3
    products.fw_time = 4
    products.start_unique_packet_id = 5
    products.start_time_32 = 65536
    products.start_time_16 = 0
    products.start_raw_seconds = float("nan")
    destination = tmp_path / "invalid-session.h5"

    with pytest.raises(ValueError, match="start_raw_seconds must be finite"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_mutated_string_quality_before_output(tmp_path: Path):
    request = make_request()
    request.products.quality_status = "clean"
    destination = tmp_path / "invalid-quality.h5"

    with pytest.raises(ValueError, match="clean or partial product quality"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_nonportable_field_dtype_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(
        row,
        fields={**row.fields, "invalid_time": np.datetime64("2026-01-01")},
        field_present={**row.field_present, "invalid_time": True},
    )
    destination = tmp_path / "invalid-field-dtype.h5"

    with pytest.raises(TypeError, match="not portable for layout v4"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_excessive_field_rank_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(
        row,
        fields={**row.fields, "excessive_rank": np.zeros((1,) * 32, np.int8)},
        field_present={**row.field_present, "excessive_rank": True},
    )
    destination = tmp_path / "excessive-field-rank.h5"

    with pytest.raises(ValueError, match="at most 31 dimensions"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_dtype_metadata_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    enum_dtype = h5py.enum_dtype({"\ud800": 0}, basetype="i4")
    request.products.housekeeping[0] = replace(
        row,
        fields={**row.fields, "invalid_enum": np.asarray(0, dtype=enum_dtype)},
        field_present={**row.field_present, "invalid_enum": True},
    )
    destination = tmp_path / "invalid-field-metadata.h5"

    with pytest.raises(TypeError, match="dtype metadata is not portable"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_non_native_field_dtype_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(
        row,
        fields={
            **row.fields,
            "non_native": np.asarray([1], dtype=">i4"),
        },
        field_present={**row.field_present, "non_native": True},
    )
    destination = tmp_path / "non-native-field-dtype.h5"

    with pytest.raises(TypeError, match="non-native dtype"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_non_utf8_field_text_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(
        row,
        fields={**row.fields, "invalid_text": "\ud800"},
        field_present={**row.field_present, "invalid_text": True},
    )
    destination = tmp_path / "invalid-field-text.h5"

    with pytest.raises(ValueError, match="valid UTF-8"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_non_utf8_decoder_issue_code_before_output(tmp_path: Path):
    request = make_all_family_request()
    request.products.decode_provenance = replace(
        request.products.decode_provenance,
        issue_counts=(("\ud800", 1),),
    )
    destination = tmp_path / "invalid-decoder-issue-code.h5"

    with pytest.raises(ValueError, match="valid UTF-8"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_unrepresentable_clock_conversion_before_output(
    tmp_path: Path,
):
    request = make_all_family_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping[0] = replace(row, raw_seconds=1e308)
    destination = tmp_path / "unrepresentable-clock.h5"

    with pytest.raises(ValueError, match="unrepresentable"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_writer_rejects_unrepresentable_page_clock_before_output(tmp_path: Path):
    request = make_all_family_request()
    row = request.products.calibrator_data[0]
    page_raw_seconds = row.page_raw_seconds.copy()
    page_raw_seconds[-1] = 1e308
    request.products.calibrator_data[0] = replace(
        row,
        page_raw_seconds=page_raw_seconds,
    )
    destination = tmp_path / "unrepresentable-page-clock.h5"

    with pytest.raises(ValueError, match="unrepresentable"):
        hdf5_writer.write_hdf5(request, destination)

    assert not destination.exists()


def test_optional_uid_source_role_has_validity_mask(tmp_path: Path):
    request = make_request()
    row = request.products.housekeeping[0]
    request.products.housekeeping = [
        replace(
            row,
            provenance=replace(row.provenance, uid_source_role=None),
        )
    ]
    destination = tmp_path / "optional-role.h5"

    hdf5_writer.write_hdf5(request, destination)

    with h5py.File(destination, "r") as h5:
        provenance = h5["provenance/product_rows"]
        assert provenance["uid_source_role"].asstr()[:].tolist() == [""]
        assert provenance["uid_source_role_valid"][:].tolist() == [False]


def test_layout_v4_core_provenance_and_housekeeping_union(tmp_path: Path):
    destination = tmp_path / "ingest.h5"

    returned = hdf5_writer.write_hdf5(make_request(), destination)

    assert returned == destination
    with h5py.File(destination, "r") as h5:
        assert int(h5.attrs["layout_version"]) == 4
        assert h5.attrs["quality_status"] == "clean"
        assert h5.attrs["execution_mode"] == "collect"
        assert int(h5.attrs["input_packet_count"]) == 1
        assert int(h5.attrs["valid_packet_count"]) == 1
        assert int(h5.attrs["persisted_housekeeping_rows"]) == 1
        assert h5["constants"].attrs["lun_lat_deg"] == pytest.approx(-23.814)
        assert h5["clock_reference"].attrs["available"] == np.bool_(False)
        assert h5["clock_reference"].attrs["unavailable_reason"] == (
            "fixture has no clock anchor"
        )
        run = h5["run_provenance"]
        assert run.attrs["input_identity"] == "fixture-sha256"
        assert run.attrs["input_identity_kind"] == "sha256"
        assert run.attrs["source_kind"] == "uncrater_session"
        assert run.attrs["overwrite_requested"] == np.bool_(False)
        assert run.attrs["destination_preexisted"] == np.bool_(False)
        decoder = h5["provenance/decoder"]
        assert decoder.attrs["decoder_name"] == "uncrater"
        assert int(decoder.attrs["selected_schema_id"]) == 0x307
        assert decoder.attrs["canonical_report_json"] == ('{"fixture":"strict"}')
        product_rows = h5["provenance/product_rows"]
        assert product_rows["family"].asstr()[:].tolist() == ["housekeeping"]
        assert product_rows["unique_ids"][:].tolist() == [7]
        families = h5["status/families/family"].asstr()[:].tolist()
        assert families == sorted(ALL_FAMILIES)
        housekeeping = h5["housekeeping"]
        assert housekeeping["hk_type"][:].tolist() == [1]
        assert housekeeping["raw_time_valid"][:].tolist() == [False]
        assert housekeeping["field_present/temperature"][:].tolist() == [True]
        temperature = housekeeping["fields/temperature"]
        assert temperature.attrs["kind"] == "array_variants"
        np.testing.assert_allclose(
            temperature["variant_000/data"][:],
            np.array([21.5], dtype=np.float64),
        )


def test_existing_destination_is_refused_by_default(tmp_path: Path):
    destination = tmp_path / "existing.h5"
    original = b"do not replace"
    destination.write_bytes(original)

    with pytest.raises(FileExistsError):
        hdf5_writer.write_hdf5(make_request(), destination)

    assert destination.read_bytes() == original


def test_overwrite_is_explicit_and_recorded(tmp_path: Path):
    destination = tmp_path / "existing.h5"
    destination.write_bytes(b"old contents")

    hdf5_writer.write_hdf5(make_request(overwrite=True), destination)

    with h5py.File(destination, "r") as h5:
        run = h5["run_provenance"]
        assert run.attrs["overwrite_requested"] == np.bool_(True)
        assert run.attrs["destination_preexisted"] == np.bool_(True)


@pytest.mark.parametrize(
    ("corruption", "target"),
    (
        ("delete", "issues/details_json"),
        ("delete_root_attr", "info_issue_count"),
        ("delete_root_attr", "decoded_fw_direct_spectra_rows"),
        ("delete_attr", "clock_reference@available"),
        ("delete_attr", "provenance/decoder@selected_schema_id"),
        ("wrong_attr_dtype", "clock_reference@format_version"),
        ("change_string", "status/families/quality"),
        ("delete", "provenance/source_packets/role"),
        ("wrong_dtype", "status/families/decoded_rows"),
        ("delete", "spectra/frequency_counts"),
        ("add_forbidden_mask", "spectra"),
        ("delete", "tr_spectra/navg2_per_sample"),
        ("delete", "calibrator/zoom_spectra/pfb_bins"),
        ("wrong_dtype", "waveform/adc_timestamps"),
        ("wrong_shape", "grimm_spectra/average_valid"),
        ("delete", "housekeeping/firmware_errors"),
        ("wrong_dtype", "housekeeping/field_present/adc_min"),
        ("delete", "calibrator/metadata/from_debug"),
        ("wrong_shape", "calibrator/data/gphase"),
        ("delete", "calibrator/raw_pfb/data_imag"),
        ("delete", "calibrator/debug/pages/page_0"),
    ),
)
def test_close_time_verifier_rejects_malformed_all_family_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    target: str,
):
    request = make_all_family_request()
    destination = tmp_path / "malformed.h5"
    write_layout_v4 = hdf5_writer._write_layout_v4

    def write_then_corrupt(*args, **kwargs):
        write_layout_v4(*args, **kwargs)
        path = args[0]
        with h5py.File(path, "a") as h5:
            if corruption == "delete_root_attr":
                del h5.attrs[target]
                return
            if corruption == "delete_attr":
                group_name, attr_name = target.split("@", maxsplit=1)
                del h5[group_name].attrs[attr_name]
                return
            if corruption == "wrong_attr_dtype":
                group_name, attr_name = target.split("@", maxsplit=1)
                value = h5[group_name].attrs[attr_name]
                del h5[group_name].attrs[attr_name]
                h5[group_name].attrs[attr_name] = np.int64(value)
                return
            if corruption == "change_string":
                h5[target][0] = "partial"
                return
            if corruption == "delete":
                del h5[target]
                return
            if corruption == "add_forbidden_mask":
                h5[target].create_dataset(
                    "product_present",
                    data=np.ones((1, 16), dtype=np.bool_),
                )
                return
            data = h5[target][:]
            del h5[target]
            if corruption == "wrong_dtype":
                h5.create_dataset(target, data=data.astype(np.int64))
                return
            if corruption == "wrong_shape":
                h5.create_dataset(target, data=data.reshape(-1))
                return
            raise AssertionError(f"unknown corruption {corruption}")

    monkeypatch.setattr(hdf5_writer, "_write_layout_v4", write_then_corrupt)

    with pytest.raises(ValueError, match="HDF5"):
        hdf5_writer.write_hdf5(request, destination)

    assert destination.exists()
