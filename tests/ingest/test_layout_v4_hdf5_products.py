"""All-family native-storage contract for layout-v4 HDF5 output."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from lusee.ingest.clock_reference import (
    ClockReference,
    ClockReferenceSet,
    ClockSource,
)
from lusee.ingest.constants import (
    BITSLICE_REFERENCE,
    NCHANNELS,
    NPRODUCTS,
    WAVEFORM_SAMPLES,
    ZOOM_BINS,
)
from lusee.ingest.decode import Products
from lusee.ingest.frequency_contract import spectrometer_frequency_window
from lusee.ingest.hdf5_writer import write_hdf5
from lusee.ingest.products import (
    CalibratorDataSample,
    CalibratorDebugPage,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    DataQuality,
    DecodeProvenance,
    GrimmSample,
    HKSample,
    ProductProvenance,
    SourcePacketProvenance,
    SpectrumMetadata,
    SpectrumSample,
    TRSpectrumSample,
    ValidatedCounts,
    WaveformSample,
    ZoomSample,
)
from lusee.ingest.write_request import (
    FAMILY_TYPES,
    FamilyCoverage,
    LunarLocation,
    RunProvenance,
    WriteRequest,
    family_statuses_for_products,
)


def make_provenance(
    uid: int,
    raw_seconds: float | None,
    roles: tuple[str, ...],
    *,
    uid_source_role: str | None = None,
    time_source_role: str | None = None,
) -> ProductProvenance:
    uid_role = roles[0] if uid_source_role is None else uid_source_role
    time_role = uid_role if time_source_role is None else time_source_role
    return ProductProvenance(
        source_packets=tuple(
            SourcePacketProvenance(
                role=role,
                original_appid=0x270,
                normalized_appid=0x270,
                filename=f"packet_{uid}_{order}.bin",
                packet_index=uid * 16 + order,
                bank="A",
                frame_start=uid * 2 + order,
                frame_stop=uid * 2 + order,
                byte_offset_start=uid * 128 + order * 16,
                byte_offset_stop=uid * 128 + order * 16 + 15,
            )
            for order, role in enumerate(roles)
        ),
        uid=uid,
        uid_source="fixture.unique_packet_id",
        uid_source_role=uid_role,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        time_source=("fixture.time_32_time_16" if raw_seconds is not None else None),
        time_source_role=(time_role if raw_seconds is not None else None),
        clock_source=(
            ClockSource.SPECTROMETER.value if raw_seconds is not None else None
        ),
        time_valid=raw_seconds is not None,
    )


def make_metadata(
    uid: int,
    raw_seconds: float,
    *,
    navgf: int = 2,
) -> SpectrumMetadata:
    valid = np.array([4, 3, 2, 1], dtype=np.int64)
    invalid_max = np.array([1, 0, 1, 0], dtype=np.int64)
    invalid_min = np.array([0, 1, 0, 1], dtype=np.int64)
    return SpectrumMetadata(
        version=0x307,
        unique_packet_id=uid,
        uc_time=123456789,
        time_32=int(raw_seconds * 65536),
        time_16=0,
        tvs_sensors=np.array([16000, 28800, 40000, 39000], dtype=np.uint16),
        requested_gain=np.array([0, 1, 2, 3], dtype=np.uint8),
        gain_auto_min=np.array([1, 2, 3, 4], dtype=np.uint16),
        gain_auto_mult=np.array([4, 3, 2, 1], dtype=np.uint16),
        route_plus=np.array([0, 1, 2, 3], dtype=np.uint8),
        route_minus=np.array([3, 2, 1, 0], dtype=np.uint8),
        navg1_shift=1,
        navg2_shift=2,
        notch=0,
        navgf=navgf,
        high_fraction=7,
        medium_fraction=5,
        requested_bitslice=np.full(NPRODUCTS, BITSLICE_REFERENCE, dtype=np.uint8),
        bitslice_keep_bits=8,
        output_format=0,
        reject_ratio=3,
        reject_max_bad=4,
        tr_start=2,
        tr_stop=10,
        tr_average_shift=1,
        errors=0,
        correlation_products_mask=0xFFFF,
        actual_gain=np.array([1, 2, 1, 2], dtype=np.uint8),
        actual_bitslice=np.full(NPRODUCTS, BITSLICE_REFERENCE, dtype=np.uint8),
        spectrum_overflow=0,
        notch_overflow=0,
        adc_min=np.array([-5, -4, -3, -2], dtype=np.int64),
        adc_max=np.array([5, 4, 3, 2], dtype=np.int64),
        adc_valid_count=valid,
        adc_invalid_count_max=invalid_max,
        adc_invalid_count_min=invalid_min,
        adc_total_count=valid + invalid_max + invalid_min,
        adc_mean=np.array([0.5, 0.0, -0.5, 1.0], dtype=np.float64),
        adc_rms=np.array([1.5, 1.0, 2.5, 3.5], dtype=np.float64),
        spectrometer_enable=True,
        calibrator_enable=True,
        random_state=0xAABBCCDD,
        weight=8,
        weight_current=9,
        telemetry_v1_0=1.0,
        telemetry_v1_8=1.8,
        telemetry_v2_5=2.5,
        telemetry_t_fpga=30.25,
    )


def make_products() -> Products:
    normal_uid = 101
    normal_raw = 100.0
    normal_contract = spectrometer_frequency_window(2)
    normal_data = np.full(
        (NPRODUCTS, normal_contract.output_count),
        np.nan,
        dtype=np.float32,
    )
    normal_data[3] = np.linspace(
        -4.0, 4.0, normal_contract.output_count, dtype=np.float32
    )
    normal_present = np.zeros(NPRODUCTS, dtype=np.bool_)
    normal_present[3] = True
    normal_metadata = make_metadata(normal_uid, normal_raw)
    normal = SpectrumSample(
        data=normal_data,
        product_present=normal_present,
        navgf=2,
        frequency_contract=normal_contract,
        unique_packet_id=normal_uid,
        raw_seconds=normal_raw,
        metadata=normal_metadata,
        provenance=make_provenance(
            normal_uid,
            normal_raw,
            ("metadata", "normal_product_03"),
        ),
    )

    tr_uid = 102
    tr_raw = 101.0
    tr_metadata = make_metadata(tr_uid, tr_raw)
    navg2 = 1 << tr_metadata.navg2_shift
    tr_length = (tr_metadata.tr_stop - tr_metadata.tr_start) // (
        1 << tr_metadata.tr_average_shift
    )
    tr_data = np.zeros((NPRODUCTS, navg2, tr_length), dtype=np.int32)
    tr_data[7] = np.arange(navg2 * tr_length, dtype=np.int32).reshape(navg2, tr_length)
    tr_data[7, 0, 0] = np.iinfo(np.int32).min
    tr_data[7, 0, 1] = np.iinfo(np.int32).max
    tr_present = np.zeros(NPRODUCTS, dtype=np.bool_)
    tr_present[7] = True
    tr = TRSpectrumSample(
        data=tr_data,
        product_present=tr_present,
        unique_packet_id=tr_uid,
        raw_seconds=tr_raw,
        navg2=navg2,
        tr_length=tr_length,
        metadata=tr_metadata,
        provenance=make_provenance(
            tr_uid,
            tr_raw,
            ("metadata", "tr_product_07"),
        ),
    )

    zoom = ZoomSample(
        data=np.arange(4 * ZOOM_BINS, dtype=np.float32).reshape(4, ZOOM_BINS),
        unique_packet_id=103,
        pfb_bin=65535,
        raw_seconds=102.0,
        provenance=make_provenance(103, 102.0, ("zoom", "metadata")),
    )
    waveform = WaveformSample(
        data=np.arange(WAVEFORM_SAMPLES, dtype=np.int16),
        channel=3,
        unique_packet_id=104,
        raw_seconds=103.0,
        adc_timestamp=np.uint64(2**64 - 1),
        provenance=make_provenance(
            104,
            103.0,
            ("waveform_channel_3", "waveform_metadata"),
            uid_source_role="waveform_metadata",
            time_source_role="waveform_metadata",
        ),
    )
    grimm = GrimmSample(
        data=np.arange(2 * NPRODUCTS * 4, dtype=np.int32).reshape(2, NPRODUCTS, 4),
        unique_packet_id=105,
        raw_seconds=104.0,
        navg2=2,
        provenance=make_provenance(105, 104.0, ("grimm",)),
    )

    housekeeping = [
        HKSample(
            hk_type=0,
            version=0x307,
            unique_packet_id=106,
            errors=0xAABBCCDD,
            raw_seconds=105.0,
            fields={
                "adc_min": np.array([-4, -3, -2, -1], dtype=np.int64),
                "optional_gain": None,
                "telemetry": {
                    "state": np.uint8(2),
                    "v1_0": np.float64(1.0),
                },
            },
            field_present={
                "adc_min": True,
                "optional_gain": False,
                "telemetry": True,
            },
            provenance=make_provenance(106, 105.0, ("housekeeping",)),
        ),
        HKSample(
            hk_type=1,
            version=0x307,
            unique_packet_id=107,
            errors=0,
            raw_seconds=None,
            fields={
                "adc_min": None,
                "optional_gain": np.array([1, 2, 3, 4], dtype=np.uint8),
                "telemetry": None,
            },
            field_present={
                "adc_min": False,
                "optional_gain": True,
                "telemetry": False,
            },
            provenance=make_provenance(107, None, ("housekeeping",)),
        ),
    ]

    calibrator_metadata = [
        CalibratorMetadataSample(
            unique_packet_id=108,
            raw_seconds=106.0,
            from_debug=False,
            fields={
                "drift": np.linspace(-1.0, 1.0, 8, dtype=np.float64),
                "mode": None,
            },
            field_present={"drift": True, "mode": False},
            provenance=make_provenance(108, 106.0, ("calibrator_metadata",)),
        ),
        CalibratorMetadataSample(
            unique_packet_id=109,
            raw_seconds=107.0,
            from_debug=False,
            fields={"drift": None, "mode": np.uint16(7)},
            field_present={"drift": False, "mode": True},
            provenance=make_provenance(109, 107.0, ("calibrator_metadata",)),
        ),
    ]

    cal_real = np.arange(4 * 512, dtype=np.float64).reshape(4, 512)
    cal_data_values = cal_real.astype(np.complex128)
    cal_data_values.imag = -cal_real
    calibrator_data = CalibratorDataSample(
        data=cal_data_values,
        g_nacc=np.iinfo(np.int32).min,
        gphase=np.arange(1024, dtype=np.int32),
        unique_packet_id=110,
        raw_seconds=108.0,
        page_raw_seconds=np.array([108.0, 108.25, 108.5], dtype=np.float64),
        provenance=make_provenance(
            110,
            108.0,
            ("cal_data_page0", "cal_data_page1", "cal_data_page2"),
        ),
    )

    raw_pfb_values = np.zeros((4, NCHANNELS), dtype=np.complex128)
    raw_pfb_values[3, -1] = 3.0 + 4.0j
    raw_pfb_times = 109.0 + np.arange(8, dtype=np.float64) / 8.0
    calibrator_raw_pfb = CalibratorRawPFBSample(
        data=raw_pfb_values,
        unique_packet_id=111,
        raw_seconds=109.0,
        page_raw_seconds=raw_pfb_times,
        provenance=make_provenance(
            111,
            109.0,
            tuple(f"cal_pfb_page{page}" for page in range(8)),
        ),
    )

    debug_pages = tuple(
        CalibratorDebugPage(
            page=page,
            fields={
                "optional_word": (np.uint16(page) if page % 2 == 0 else None),
                "values": np.full((2, 3), page, dtype=np.int32),
            },
            field_present={
                "optional_word": page % 2 == 0,
                "values": True,
            },
        )
        for page in range(8)
    )
    debug_times = 110.0 + np.arange(8, dtype=np.float64) / 16.0
    calibrator_debug = CalibratorDebugSample(
        pages=debug_pages,
        unique_packet_id=112,
        raw_seconds=110.0,
        page_raw_seconds=debug_times,
        provenance=make_provenance(
            112,
            110.0,
            tuple(f"cal_debug_page{page}" for page in range(8)),
        ),
    )

    rows = {
        "spectra": [normal],
        "tr_spectra": [tr],
        "zoom_spectra": [zoom],
        "waveforms": [waveform],
        "housekeeping": housekeeping,
        "grimm_spectra": [grimm],
        "calibrator_metadata": calibrator_metadata,
        "calibrator_data": [calibrator_data],
        "calibrator_raw_pfb": [calibrator_raw_pfb],
        "calibrator_debug": [calibrator_debug],
    }
    source_packet_count = sum(
        len(row.provenance.source_packets)
        for family_rows in rows.values()
        for row in family_rows
    )
    decoder = DecodeProvenance.from_report(
        distribution_version="2.0.0",
        decoder_source_commit="a" * 40,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant="production",
        schema_assumed=False,
        binding_source_release="3r09",
        binding_source_commit="b" * 40,
        abi_fingerprint="c" * 64,
        execution_mode="collect",
        input_packet_count=source_packet_count,
        valid_packet_count=source_packet_count,
        appid_counts=((0x270, source_packet_count),),
        issue_counts=(),
        canonical_report={
            "fixture": "layout-v4-all-products",
            "packet_count": source_packet_count,
        },
    )
    product_counts = tuple(
        sorted((family, len(family_rows)) for family, family_rows in rows.items())
    )
    return Products(
        **rows,
        sw_version=0x102,
        fw_version=0x307,
        fw_id=7,
        fw_date=0x20260824,
        fw_time=0x010203,
        start_unique_packet_id=100,
        start_time_32=int(99.0 * 65536),
        start_time_16=0,
        start_raw_seconds=99.0,
        decode_provenance=decoder,
        quality_status=DataQuality.CLEAN,
        validated_counts=ValidatedCounts(
            input_packets=source_packet_count,
            valid_packets=source_packet_count,
            product_rows=product_counts,
        ),
        issues=(),
    )


@pytest.fixture(scope="module")
def all_products_request() -> WriteRequest:
    products = make_products()
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


@pytest.fixture(scope="module")
def all_products_file(
    tmp_path_factory: pytest.TempPathFactory,
    all_products_request: WriteRequest,
):
    path = tmp_path_factory.mktemp("layout-v4") / "all-products.h5"
    write_hdf5(all_products_request, path)
    return path


def test_dense_normal_and_tr_storage_contracts(
    all_products_file,
    all_products_request,
):
    products = all_products_request.products
    with h5py.File(all_products_file, "r") as h5:
        normal = h5["spectra/data"]
        assert normal.shape == (1, NPRODUCTS, NCHANNELS)
        assert normal.dtype == np.dtype(np.float32)
        native_count = products.spectra[0].nfreq
        np.testing.assert_array_equal(
            normal[0, 3, :native_count],
            products.spectra[0].data[3],
        )
        assert np.isnan(normal[0, 3, native_count:]).all()
        assert np.isnan(normal[0, 0]).all()
        assert h5["spectra/frequency_counts"][:].tolist() == [native_count]
        assert h5["spectra/frequency_windows"].attrs["integer_arithmetic"] == (
            "averaging_mode_dependent"
        )
        assert "product_present" not in h5["spectra"]
        assert "data_valid" not in h5["spectra"]
        assert h5["spectra/metadata/field_present/loop_count_min"][:].tolist() == [
            False
        ]
        assert (
            h5["spectra/metadata/fields/loop_count_min"].attrs["kind"]
            == "untyped_absent"
        )

        tr = h5["tr_spectra/data"]
        assert tr.shape == (1, NPRODUCTS, 4, 4)
        assert tr.dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(
            tr[0, 7],
            products.tr_spectra[0].data[7].astype(np.float64),
        )
        assert tr[0, 7, 0, 0] == np.iinfo(np.int32).min
        assert tr[0, 7, 0, 1] == np.iinfo(np.int32).max
        assert np.isnan(tr[0, 0]).all()
        assert "product_present" not in h5["tr_spectra"]
        assert "data_valid" not in h5["tr_spectra"]
        assert h5["tr_spectra/metadata/field_present/loop_count_min"][:].tolist() == [
            False
        ]
        assert (
            h5["tr_spectra/metadata/fields/loop_count_min"].attrs["kind"]
            == "untyped_absent"
        )


def test_auxiliary_native_arrays_presence_unions_and_page_clocks(
    all_products_file,
    all_products_request,
):
    products = all_products_request.products
    clock_reference_set = all_products_request.clock_reference_set
    assert clock_reference_set is not None
    with h5py.File(all_products_file, "r") as h5:
        zoom = h5["calibrator/zoom_spectra/data"]
        assert zoom.shape == (1, 4, ZOOM_BINS)
        assert zoom.dtype == np.dtype(np.float32)

        waveform = h5["waveform/data"]
        assert waveform.shape == (1, WAVEFORM_SAMPLES)
        assert waveform.dtype == np.dtype(np.int16)
        timestamps = h5["waveform/adc_timestamps"]
        assert timestamps.dtype == np.dtype(np.uint64)
        assert timestamps[0] == np.uint64(2**64 - 1)
        assert h5["waveform/adc_timestamp_valid"][:].tolist() == [True]

        grimm = h5["grimm_spectra/data"]
        assert grimm.shape == (1, 2, NPRODUCTS, 4)
        assert grimm.dtype == np.dtype(np.int32)
        assert h5["grimm_spectra/average_valid"][:].tolist() == [[True, True]]

        np.testing.assert_array_equal(
            h5["housekeeping/field_present/adc_min"][:],
            np.array([True, False]),
        )
        np.testing.assert_array_equal(
            h5["housekeeping/fields/adc_min/variant_000/data"][0],
            np.array([-4, -3, -2, -1], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            h5["housekeeping/field_present/optional_gain"][:],
            np.array([False, True]),
        )
        assert h5["housekeeping/raw_time_valid"][:].tolist() == [True, False]
        assert np.isnan(h5["housekeeping/raw_seconds"][1])
        assert h5["housekeeping/fields/telemetry"].attrs["kind"] == "mapping"

        np.testing.assert_array_equal(
            h5["calibrator/metadata/field_present/drift"][:],
            np.array([True, False]),
        )
        np.testing.assert_array_equal(
            h5["calibrator/metadata/field_present/mode"][:],
            np.array([False, True]),
        )

        cal_data = products.calibrator_data[0]
        assert h5["calibrator/data"].attrs["units"] == cal_data.units
        assert h5["calibrator/data"].attrs["representation"] == (
            cal_data.representation
        )
        assert int(h5["calibrator/data"].attrs["page_count"]) == 3
        np.testing.assert_array_equal(
            h5["calibrator/data"].attrs["channel_labels"],
            np.arange(4, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            h5["calibrator/data/data_real"][0], cal_data.data.real
        )
        np.testing.assert_array_equal(
            h5["calibrator/data/data_imag"][0], cal_data.data.imag
        )
        assert h5["calibrator/data/data_real"].dtype == np.dtype(np.float64)
        assert h5["calibrator/data/data_imag"].dtype == np.dtype(np.float64)
        assert h5["calibrator/data/g_nacc"][0] == np.iinfo(np.int32).min

        raw_pfb = products.calibrator_raw_pfb[0]
        assert h5["calibrator/raw_pfb"].attrs["units"] == raw_pfb.units
        assert h5["calibrator/raw_pfb"].attrs["representation"] == (
            raw_pfb.representation
        )
        assert int(h5["calibrator/raw_pfb"].attrs["page_count"]) == 8
        assert int(h5["calibrator/debug"].attrs["page_count"]) == 8
        np.testing.assert_array_equal(
            h5["calibrator/raw_pfb/data_real"][0], raw_pfb.data.real
        )
        np.testing.assert_array_equal(
            h5["calibrator/raw_pfb/data_imag"][0], raw_pfb.data.imag
        )
        assert h5["calibrator/raw_pfb/data_real"].dtype == np.dtype(np.float64)
        assert h5["calibrator/raw_pfb/data_imag"].dtype == np.dtype(np.float64)

        page_contracts = (
            ("calibrator/data", products.calibrator_data[0].page_raw_seconds),
            (
                "calibrator/raw_pfb",
                products.calibrator_raw_pfb[0].page_raw_seconds,
            ),
            (
                "calibrator/debug",
                products.calibrator_debug[0].page_raw_seconds,
            ),
        )
        for path, raw_seconds in page_contracts:
            assert h5[path].attrs["page_clock_source"] == (
                ClockSource.SPECTROMETER.value
            )
            np.testing.assert_array_equal(
                h5[f"{path}/page_raw_seconds"][0], raw_seconds
            )
            expected_mjd = clock_reference_set.to_mjd(
                raw_seconds,
                clock_source=ClockSource.SPECTROMETER,
            )
            np.testing.assert_allclose(
                h5[f"{path}/page_mjd_times"][0],
                expected_mjd,
                rtol=0.0,
                atol=1e-12,
            )
            assert h5[f"{path}/page_mjd_time_valid"][0].all()

        assert h5["calibrator/debug/pages/page_0/field_present/optional_word"][
            :
        ].tolist() == [True]
        assert h5["calibrator/debug/pages/page_1/field_present/optional_word"][
            :
        ].tolist() == [False]
        np.testing.assert_array_equal(
            h5["calibrator/debug/pages/page_3/fields/values/variant_000/data"][0],
            np.full((2, 3), 3, dtype=np.int32),
        )


def test_product_source_provenance_and_family_status_tables(
    all_products_file,
    all_products_request,
):
    products = all_products_request.products
    expected_families = [
        family for family, _ in FAMILY_TYPES for _ in getattr(products, family)
    ]
    expected_uids = [
        row.unique_packet_id
        for family, _ in FAMILY_TYPES
        for row in getattr(products, family)
    ]
    expected_source_count = sum(
        len(row.provenance.source_packets)
        for family, _ in FAMILY_TYPES
        for row in getattr(products, family)
    )
    with h5py.File(all_products_file, "r") as h5:
        product_rows = h5["provenance/product_rows"]
        assert product_rows["family"].asstr()[:].tolist() == expected_families
        assert product_rows["unique_ids"][:].tolist() == expected_uids
        assert product_rows["row_index"][:].tolist() == [
            row_index
            for family, _ in FAMILY_TYPES
            for row_index in range(len(getattr(products, family)))
        ]

        source_packets = h5["provenance/source_packets"]
        assert len(source_packets["role"]) == expected_source_count
        assert source_packets["packet_index_valid"][:].all()
        assert source_packets["normalized_appid_valid"][:].all()
        assert source_packets["filename_valid"][:].all()
        assert source_packets["bank_valid"][:].all()
        assert source_packets["frame_start_valid"][:].all()
        assert source_packets["byte_offset_start_valid"][:].all()
        assert source_packets["provenance_index"][:].max() == (
            len(expected_families) - 1
        )
        assert len(h5["provenance/product_issue_refs/issue_index"]) == 0

        status_group = h5["status/families"]
        status_families = status_group["family"].asstr()[:].tolist()
        assert status_families == [
            status.family for status in all_products_request.family_statuses
        ]
        for index, status in enumerate(all_products_request.family_statuses):
            assert bool(status_group["supported"][index]) is status.supported
            assert status_group["coverage"].asstr()[index] == status.coverage.value
            assert status_group["quality"].asstr()[index] == status.quality.value
            assert int(status_group["decoded_rows"][index]) == status.decoded_rows
            expected_persisted = (
                status.decoded_rows
                if status.coverage is FamilyCoverage.PERSISTED
                else 0
            )
            assert int(status_group["persisted_rows"][index]) == (expected_persisted)
        assert len(h5["status/family_issue_refs/issue_index"]) == 0
