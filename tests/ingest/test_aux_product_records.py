"""Strict in-memory contracts for repaired-uncrater auxiliary products."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from lusee.ingest.constants import NCHANNELS, NPRODUCTS, WAVEFORM_SAMPLES, ZOOM_BINS
from lusee.ingest.products import (
    CalibratorDataSample,
    CalibratorDebugPage,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    GrimmSample,
    HKSample,
    ProductProvenance,
    SourcePacketProvenance,
    WaveformSample,
    ZoomSample,
)

UID = 0x12345678
RAW_SECONDS = 12.5


def provenance(
    *roles: str,
    uid: int = UID,
    raw_seconds: float | None = RAW_SECONDS,
    uid_source_role: str | None = None,
    time_source_role: str | None = None,
) -> ProductProvenance:
    if not roles:
        roles = ("product",)
    uid_role = roles[0] if uid_source_role is None else uid_source_role
    time_role = uid_role if time_source_role is None else time_source_role
    return ProductProvenance(
        source_packets=tuple(
            SourcePacketProvenance(
                role=role,
                original_appid=0x270 + index,
                packet_index=100 + index,
            )
            for index, role in enumerate(roles)
        ),
        uid=uid,
        uid_source="packet.unique_packet_id",
        uid_source_role=uid_role,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        time_source=("packet.time_32_time_16" if raw_seconds is not None else None),
        time_source_role=(time_role if raw_seconds is not None else None),
        clock_source=("spectrometer" if raw_seconds is not None else None),
        time_valid=raw_seconds is not None,
    )


def test_public_auxiliary_exports_are_strict_records():
    import lusee.ingest as ingest

    assert ingest.ZoomSample is ZoomSample
    assert ingest.WaveformSample is WaveformSample
    assert ingest.GrimmSample is GrimmSample
    assert ingest.HKSample is HKSample
    assert ingest.CalibratorMetadataSample is CalibratorMetadataSample
    assert ingest.CalibratorDataSample is CalibratorDataSample
    assert ingest.CalibratorRawPFBSample is CalibratorRawPFBSample
    assert ingest.CalibratorDebugPage is CalibratorDebugPage
    assert ingest.CalibratorDebugSample is CalibratorDebugSample


def test_zoom_preserves_component_order_pfb_bin_and_immutable_float32_data():
    values = np.arange(4 * ZOOM_BINS, dtype=np.float32).reshape(4, ZOOM_BINS)
    sample = ZoomSample(
        data=values,
        unique_packet_id=UID,
        pfb_bin=65535,
        raw_seconds=RAW_SECONDS,
        provenance=provenance("zoom", "metadata"),
    )

    assert sample.component_labels == ("AA", "BB", "ABR", "ABI")
    assert sample.pfb_bin == 65535
    assert sample.data.dtype == np.float32
    values[0, 0] = -1
    assert sample.data[0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        sample.data[0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        sample.pfb_bin = 4


@pytest.mark.parametrize(
    ("data", "pfb_bin", "error"),
    [
        (np.zeros((4, ZOOM_BINS), dtype=np.float64), 1, TypeError),
        (np.zeros((4, ZOOM_BINS - 1), dtype=np.float32), 1, ValueError),
        (np.zeros((4, ZOOM_BINS), dtype=np.float32), 1 << 16, ValueError),
    ],
)
def test_zoom_rejects_wrong_native_contract(data, pfb_bin, error):
    with pytest.raises(error):
        ZoomSample(
            data=data,
            unique_packet_id=UID,
            pfb_bin=pfb_bin,
            raw_seconds=None,
            provenance=provenance("zoom", raw_seconds=None),
        )


def test_waveform_preserves_signed_samples_and_full_uint64_adc_time():
    values = np.arange(WAVEFORM_SAMPLES, dtype=np.int16)
    sample = WaveformSample(
        data=values,
        channel=3,
        unique_packet_id=UID,
        raw_seconds=RAW_SECONDS,
        adc_timestamp=np.uint64(2**64 - 1),
        provenance=provenance(
            "waveform_channel_3",
            "waveform_metadata",
            uid_source_role="waveform_metadata",
            time_source_role="waveform_metadata",
        ),
    )

    assert sample.data.dtype == np.int16
    assert sample.data[-1] == np.int16(WAVEFORM_SAMPLES - 1)
    assert type(sample.adc_timestamp) is np.uint64
    assert sample.adc_timestamp == np.uint64(2**64 - 1)
    values[0] = -8
    assert sample.data[0] == 0
    with pytest.raises(ValueError, match="read-only"):
        sample.data[0] = 2


@pytest.mark.parametrize(
    ("uid_source_role", "time_source_role"),
    [
        ("waveform_channel_0", "waveform_metadata"),
        ("waveform_metadata", "waveform_channel_0"),
    ],
)
def test_waveform_requires_uid_and_time_from_attached_metadata(
    uid_source_role,
    time_source_role,
):
    with pytest.raises(ValueError, match="waveform (UID|mission time)"):
        WaveformSample(
            data=np.zeros(WAVEFORM_SAMPLES, dtype=np.int16),
            channel=0,
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            adc_timestamp=np.uint64(1),
            provenance=provenance(
                "waveform_channel_0",
                "waveform_metadata",
                uid_source_role=uid_source_role,
                time_source_role=time_source_role,
            ),
        )


@pytest.mark.parametrize(
    ("data", "channel", "adc_timestamp", "error"),
    [
        (np.zeros(WAVEFORM_SAMPLES, dtype=np.int32), 0, np.uint64(1), TypeError),
        (np.zeros(WAVEFORM_SAMPLES - 1, dtype=np.int16), 0, np.uint64(1), ValueError),
        (np.zeros(WAVEFORM_SAMPLES, dtype=np.int16), 4, np.uint64(1), ValueError),
        (np.zeros(WAVEFORM_SAMPLES, dtype=np.int16), 0, 1, TypeError),
    ],
)
def test_waveform_rejects_wrong_native_contract(
    data, channel, adc_timestamp, error
):
    with pytest.raises(error):
        WaveformSample(
            data=data,
            channel=channel,
            unique_packet_id=UID,
            raw_seconds=None,
            adc_timestamp=adc_timestamp,
            provenance=provenance(
                "waveform_ch0",
                "waveform_metadata",
                raw_seconds=None,
                uid_source_role="waveform_metadata",
            ),
        )


@pytest.mark.parametrize("navg2", [1, 3, 8])
def test_grimm_preserves_native_integer_geometry_and_index_labels(navg2):
    values = np.arange(navg2 * NPRODUCTS * 4, dtype=np.int32).reshape(
        navg2, NPRODUCTS, 4
    )
    sample = GrimmSample(
        data=values,
        unique_packet_id=UID,
        raw_seconds=None,
        navg2=navg2,
        provenance=provenance("grimm", raw_seconds=None),
    )

    assert sample.data.dtype == np.int32
    assert sample.data.shape == (navg2, NPRODUCTS, 4)
    assert sample.axis_labels == (
        "average_index",
        "product_index",
        "grimm_value_index",
    )
    assert sample.value_axis_labels == ("value_0", "value_1", "value_2", "value_3")
    values[...] = -1
    assert sample.data[0, 0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        sample.data[0, 0, 0] = 2


def test_grimm_rejects_wrong_dtype_shape_and_navg2():
    with pytest.raises(TypeError):
        GrimmSample(
            np.zeros((2, NPRODUCTS, 4), dtype=np.int64),
            UID,
            None,
            2,
            provenance("grimm", raw_seconds=None),
        )
    with pytest.raises(ValueError, match="shape"):
        GrimmSample(
            np.zeros((2, NPRODUCTS, 3), dtype=np.int32),
            UID,
            None,
            2,
            provenance("grimm", raw_seconds=None),
        )
    with pytest.raises(ValueError, match="positive"):
        GrimmSample(
            np.zeros((0, NPRODUCTS, 4), dtype=np.int32),
            UID,
            None,
            0,
            provenance("grimm", raw_seconds=None),
        )


def test_housekeeping_fields_and_presence_are_deeply_immutable():
    adc_min = np.array([-4, -3, -2, -1], dtype=np.int64)
    nested = {"v1_0": np.float64(1.0)}
    sample = HKSample(
        hk_type=0,
        version=0x307,
        unique_packet_id=UID,
        errors=0xAABBCCDD,
        raw_seconds=RAW_SECONDS,
        fields={
            "adc_min": adc_min,
            "telemetry": nested,
            "actual_gain": None,
        },
        field_present={
            "adc_min": True,
            "telemetry": True,
            "actual_gain": False,
        },
        provenance=provenance("housekeeping"),
    )

    assert sample.hk_type == 0
    assert sample.version == 0x307
    assert sample.unique_packet_id == UID
    assert sample.errors == 0xAABBCCDD
    assert sample.field_present["actual_gain"] is False
    assert sample.fields["actual_gain"] is None
    adc_min[0] = 99
    nested["v1_0"] = np.float64(2.0)
    assert sample.fields["adc_min"][0] == -4
    assert sample.fields["telemetry"]["v1_0"] == np.float64(1.0)
    with pytest.raises(TypeError):
        sample.fields["new"] = 1
    with pytest.raises(TypeError):
        sample.fields["telemetry"]["v1_0"] = np.float64(3.0)
    with pytest.raises(ValueError, match="read-only"):
        sample.fields["adc_min"][0] = 1


@pytest.mark.parametrize("hk_type", [0, 1, 2, 3, 100, 101])
def test_every_supported_housekeeping_type_is_explicit(hk_type):
    sample = HKSample(
        hk_type=hk_type,
        version=0x307,
        unique_packet_id=UID,
        errors=hk_type,
        raw_seconds=None,
        fields={"value": np.int32(hk_type)},
        field_present={"value": True},
        provenance=provenance("housekeeping", raw_seconds=None),
    )
    assert (sample.hk_type, sample.errors) == (hk_type, hk_type)


@pytest.mark.parametrize("hk_type", [1, 3, 100, 101])
def test_housekeeping_types_without_clocks_reject_mission_time(hk_type):
    with pytest.raises(ValueError, match="does not carry mission time"):
        HKSample(
            hk_type=hk_type,
            version=0x307,
            unique_packet_id=UID,
            errors=0,
            raw_seconds=RAW_SECONDS,
            fields={"time_32": None, "time_16": None},
            field_present={"time_32": False, "time_16": False},
            provenance=provenance("housekeeping"),
        )
    with pytest.raises(ValueError, match="does not carry split time"):
        HKSample(
            hk_type=hk_type,
            version=0x307,
            unique_packet_id=UID,
            errors=0,
            raw_seconds=None,
            fields={"time_32": np.uint32(0), "time_16": np.uint16(0)},
            field_present={"time_32": True, "time_16": True},
            provenance=provenance("housekeeping", raw_seconds=None),
        )


def test_normalized_field_presence_and_values_fail_closed():
    base = dict(
        hk_type=1,
        version=0x307,
        unique_packet_id=UID,
        errors=0,
        raw_seconds=None,
        provenance=provenance("housekeeping", raw_seconds=None),
    )
    with pytest.raises(ValueError, match="identical keys"):
        HKSample(fields={"adc_min": None}, field_present={}, **base)
    with pytest.raises(ValueError, match="presence disagrees"):
        HKSample(
            fields={"adc_min": None},
            field_present={"adc_min": True},
            **base,
        )
    with pytest.raises(TypeError, match="normalized"):
        HKSample(
            fields={"actual_gain": ["L", "M", "H", "L"]},
            field_present={"actual_gain": True},
            **base,
        )
    with pytest.raises(ValueError, match="hk_type"):
        HKSample(fields={}, field_present={}, **{**base, "hk_type": 4})


def test_calibrator_metadata_uses_normalized_fields_not_decoder_objects():
    drift = np.linspace(-1.0, 1.0, 1024, dtype=np.float64)
    sample = CalibratorMetadataSample(
        unique_packet_id=UID,
        raw_seconds=RAW_SECONDS,
        from_debug=False,
        fields={"drift": drift, "optional_status": None},
        field_present={"drift": True, "optional_status": False},
        provenance=provenance("calibrator_metadata"),
    )

    drift[0] = 9.0
    assert sample.fields["drift"][0] == -1.0
    with pytest.raises(ValueError, match="read-only"):
        sample.fields["drift"][0] = 0.0
    with pytest.raises(TypeError, match="normalized"):
        CalibratorMetadataSample(
            unique_packet_id=UID,
            raw_seconds=None,
            from_debug=False,
            fields={"packet": SimpleNamespace(value=1)},
            field_present={"packet": True},
            provenance=provenance("calibrator_metadata", raw_seconds=None),
        )


def test_calibrator_data_and_raw_pfb_keep_exact_group_geometry():
    data = np.arange(4 * 512, dtype=np.float64).reshape(4, 512).astype(np.complex128)
    data.imag = -data.real
    gphase = np.arange(1024, dtype=np.int32)
    data_sample = CalibratorDataSample(
        data=data,
        g_nacc=-7,
        gphase=gphase,
        unique_packet_id=UID,
        raw_seconds=RAW_SECONDS,
        page_raw_seconds=np.full(3, RAW_SECONDS, dtype=np.float64),
        provenance=provenance("cal_data_page0", "cal_data_page1", "cal_data_page2"),
    )

    pfb = np.zeros((4, NCHANNELS), dtype=np.complex128)
    pfb[3, -1] = 3 + 4j
    pfb_sample = CalibratorRawPFBSample(
        data=pfb,
        unique_packet_id=UID,
        raw_seconds=RAW_SECONDS,
        page_raw_seconds=np.full(8, RAW_SECONDS, dtype=np.float64),
        provenance=provenance(*(f"cal_pfb_page{i}" for i in range(8))),
    )

    assert data_sample.data.shape == (4, 512)
    assert data_sample.data.dtype == np.complex128
    assert data_sample.g_nacc == -7
    assert data_sample.gphase.dtype == np.int32
    assert data_sample.page_count == 3
    np.testing.assert_array_equal(
        data_sample.page_raw_seconds,
        np.full(3, RAW_SECONDS),
    )
    assert pfb_sample.data.shape == (4, NCHANNELS)
    assert pfb_sample.data.dtype == np.complex128
    assert pfb_sample.page_count == 8
    np.testing.assert_array_equal(
        pfb_sample.page_raw_seconds,
        np.full(8, RAW_SECONDS),
    )
    data[0, 0] = 99
    gphase[0] = 99
    pfb[3, -1] = 0
    assert data_sample.data[0, 0] == 0j
    assert data_sample.gphase[0] == 0
    assert pfb_sample.data[3, -1] == 3 + 4j
    with pytest.raises(ValueError, match="read-only"):
        data_sample.page_raw_seconds[0] = 0.0


def test_calibrator_array_records_reject_wrong_native_types_and_pages():
    with pytest.raises(TypeError):
        CalibratorDataSample(
            data=np.zeros((4, 512), dtype=np.complex64),
            g_nacc=1,
            gphase=np.zeros(1024, dtype=np.int32),
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=np.full(3, RAW_SECONDS, dtype=np.float64),
            provenance=provenance(
                "cal_data_page0",
                "cal_data_page1",
                "cal_data_page2",
            ),
        )
    with pytest.raises(ValueError, match="eight pages"):
        CalibratorRawPFBSample(
            data=np.zeros((4, NCHANNELS), dtype=np.complex128),
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=np.full(8, RAW_SECONDS, dtype=np.float64),
            provenance=provenance("cal_pfb_page0"),
        )
    with pytest.raises(ValueError, match="page zero"):
        CalibratorDataSample(
            data=np.zeros((4, 512), dtype=np.complex128),
            g_nacc=1,
            gphase=np.zeros(1024, dtype=np.int32),
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=np.array(
                [RAW_SECONDS + 1, RAW_SECONDS, RAW_SECONDS],
                dtype=np.float64,
            ),
            provenance=provenance(
                "cal_data_page0",
                "cal_data_page1",
                "cal_data_page2",
            ),
        )


def test_calibrator_debug_keeps_eight_ordered_immutable_normalized_pages():
    source_arrays = [np.full((3, 8), page, dtype=np.int32) for page in range(8)]
    pages = tuple(
        CalibratorDebugPage(
            page=page,
            fields={"values": source_arrays[page]},
            field_present={"values": True},
        )
        for page in range(8)
    )
    page_raw_seconds = RAW_SECONDS + np.arange(8, dtype=np.float64)
    sample = CalibratorDebugSample(
        pages=pages,
        unique_packet_id=UID,
        raw_seconds=RAW_SECONDS,
        page_raw_seconds=page_raw_seconds,
        provenance=provenance(*(f"cal_debug_page{i}" for i in range(8))),
    )

    assert sample.page_count == 8
    assert tuple(page.page for page in sample.pages) == tuple(range(8))
    np.testing.assert_array_equal(sample.page_raw_seconds, page_raw_seconds)
    source_arrays[0][0, 0] = 99
    assert sample.pages[0].fields["values"][0, 0] == 0
    with pytest.raises(ValueError, match="read-only"):
        sample.pages[0].fields["values"][0, 0] = 2
    with pytest.raises(FrozenInstanceError):
        sample.pages[0].page = 3


def test_calibrator_debug_rejects_missing_reordered_or_decoder_owned_pages():
    pages = tuple(
        CalibratorDebugPage(page=i, fields={}, field_present={}) for i in range(8)
    )
    debug_provenance = provenance(
        *(f"cal_debug_page{i}" for i in range(8))
    )
    page_raw_seconds = np.full(8, RAW_SECONDS, dtype=np.float64)
    with pytest.raises(ValueError, match="eight pages"):
        CalibratorDebugSample(
            pages=pages[:-1],
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=page_raw_seconds,
            provenance=debug_provenance,
        )
    with pytest.raises(ValueError, match="ordered"):
        CalibratorDebugSample(
            pages=(pages[1], pages[0], *pages[2:]),
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=page_raw_seconds,
            provenance=debug_provenance,
        )
    with pytest.raises(TypeError, match="tuple"):
        CalibratorDebugSample(
            pages=list(pages),
            unique_packet_id=UID,
            raw_seconds=RAW_SECONDS,
            page_raw_seconds=page_raw_seconds,
            provenance=debug_provenance,
        )
    with pytest.raises(TypeError, match="normalized"):
        CalibratorDebugPage(
            page=0,
            fields={"packet": SimpleNamespace(value=1)},
            field_present={"packet": True},
        )


def test_auxiliary_identity_time_and_provenance_must_agree():
    data = np.zeros((4, ZOOM_BINS), dtype=np.float32)
    with pytest.raises(ValueError, match="UID disagrees"):
        ZoomSample(
            data,
            UID,
            1,
            None,
            provenance("zoom", uid=UID + 1, raw_seconds=None),
        )
    with pytest.raises(ValueError, match="time_valid"):
        ZoomSample(
            data,
            UID,
            1,
            RAW_SECONDS,
            provenance("zoom", raw_seconds=None),
        )
    with pytest.raises(ValueError, match="concrete provenance"):
        ZoomSample(
            data,
            UID,
            1,
            None,
            ProductProvenance.unavailable(),
        )
    with pytest.raises(ValueError, match="selected schema"):
        unselected = ProductProvenance(
            source_packets=(
                SourcePacketProvenance(
                    role="zoom", original_appid=0x270, packet_index=1
                ),
            ),
            uid=UID,
            uid_source="packet.unique_packet_id",
            selected_schema_id=None,
            time_valid=False,
        )
        ZoomSample(data, UID, 1, None, unselected)
