"""Known-value repaired-uncrater tests for auxiliary ingest adapters."""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path

import numpy as np
import pytest

from lusee.ingest.decode import read_uncrater_session
from lusee.ingest.issues import IssueAction
from lusee.ingest.products import (
    CalibratorDataSample,
    CalibratorDebugSample,
    CalibratorMetadataSample,
    CalibratorRawPFBSample,
    DataQuality,
    GrimmSample,
    HKSample,
    WaveformSample,
    ZoomSample,
)
from lusee.ingest.session import raw_seconds_from_split_time
from lusee.ingest.write_request import (
    FamilyCoverage,
    family_statuses_for_products,
)

uncrater = pytest.importorskip("uncrater")
schema_registry = pytest.importorskip("uncrater.schema_registry")
c_utils = pytest.importorskip("uncrater.c_utils")

BINDING = schema_registry.LATEST_BINDING
SCHEMA_ID = BINDING.canonical_schema_id


def bytes_of(value: ctypes.Structure) -> bytes:
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def cdi_padded(value: ctypes.Structure) -> bytes:
    payload = bytes_of(value)
    return payload + b"\0" * (-len(payload) % 4)


def write_packet(directory: Path, index: int, appid: int, blob: bytes) -> None:
    (directory / f"{index:05d}_{appid:04x}.bin").write_bytes(blob)


def metadata_blob(
    uid: int,
    *,
    time_32: int,
    time_16: int,
) -> bytes:
    value = BINDING.pystruct.meta_data()
    value.version = SCHEMA_ID
    value.unique_packet_id = uid
    value.base.Navgf = 1
    value.base.time_32 = time_32
    value.base.time_16 = time_16
    return cdi_padded(value)


def zoom_blob(uid: int, pfb_bin: int, values: np.ndarray) -> bytes:
    assert values.dtype == np.dtype("<f4")
    assert values.shape == (4, 64)
    return struct.pack("<IH", uid, pfb_bin) + values.tobytes(order="C")


def waveform_blob(values: np.ndarray) -> bytes:
    assert values.dtype == np.dtype("<u2")
    assert values.shape == (16_384,)
    return values.tobytes(order="C")


def waveform_metadata_blob(
    uid: int,
    *,
    time_32: int,
    time_16: int,
    timestamp: int,
) -> bytes:
    value = BINDING.pystruct.waveform_metadata()
    value.unique_packet_id = uid
    value.time_32 = time_32
    value.time_16 = time_16
    value.timestamp = timestamp
    return cdi_padded(value)


def grimm_blob(uid: int, values: np.ndarray) -> bytes:
    assert values.dtype == np.dtype(np.int32)
    assert values.ndim == 3 and values.shape[1:] == (16, 4)
    words = np.concatenate(
        [c_utils.encode_4_into_5(chunk) for chunk in values.reshape(-1, 4)]
    ).astype("<u2")
    return struct.pack("<I", uid) + words.tobytes(order="C")


def page_blob(
    uid: int,
    values: np.ndarray | bytes | bytearray,
    *,
    time_32: int,
    time_16: int,
) -> bytes:
    payload = values if isinstance(values, (bytes, bytearray)) else values.tobytes()
    return struct.pack("<III", uid, time_32, time_16) + payload


def calibrator_metadata_value(
    uid: int,
    *,
    time_32: int,
    time_16: int,
    mode: int,
) -> ctypes.Structure:
    value = BINDING.pystruct.calibrator_metadata()
    value.version = SCHEMA_ID
    value.unique_packet_id = uid
    value.time_32 = time_32
    value.time_16 = time_16
    value.mode = mode
    value.SNRon = 0x10203040
    value.SNRoff = 0x50607080
    value.errors = 0xA1B2C3D4
    value.bitslicer_errors = 0x13579BDF
    value.drift_shift = 1
    value.drift[3] = 17
    value.drift[91] = -23
    return value


def historical_calibrator_metadata_value(
    binding: object,
    uid: int,
    *,
    time_32: int,
    time_16: int,
) -> ctypes.Structure:
    value = binding.pystruct.calibrator_metadata()
    value.version = binding.canonical_schema_id
    value.unique_packet_id = uid
    value.time_32 = time_32
    value.time_16 = time_16
    value.have_lock[2] = 13
    if binding.binding_key == "203":
        value.SNR_max = -111
        value.SNR_min = 222
        value.state.mode = 5
        value.state.weight_ndx = 0x3456
        value.drift[3] = 17
        value.error_regs[4] = 0xA1B2C3D4
        return value

    value.SNRon = 0x10203040
    value.SNRoff = 0x50607080
    value.stats.SNR_max[1] = 0x12345678
    value.stats.FD_min[2] = -44
    value.stats.SD_positive_count[3] = 55
    value.drift_shift = 1
    value.drift[3] = 17
    if binding.binding_key in {"305", "306-early"}:
        value.error_regs[4] = 0xA1B2C3D4
    else:
        value.mode = 9
        value.error_reg.cal_phaser_err[1] = 0x11223344
        value.error_reg.check = 0x55667788
    return value


def assert_product_provenance(
    sample: object,
    *,
    uid: int,
    packet_indices: tuple[int, ...],
    roles: tuple[str, ...],
    raw_seconds: float | None,
    reported_schema_ids: tuple[int, ...] = (SCHEMA_ID,),
) -> None:
    provenance = sample.provenance
    assert provenance.uid == uid
    assert provenance.reported_schema_ids == reported_schema_ids
    assert provenance.selected_schema_id == SCHEMA_ID
    assert tuple(packet.packet_index for packet in provenance.source_packets) == (
        packet_indices
    )
    assert tuple(packet.role for packet in provenance.source_packets) == roles
    assert provenance.time_valid is (raw_seconds is not None)
    assert sample.raw_seconds == raw_seconds


def set_adc_sentinel(stats: ctypes.Array, channel: int) -> None:
    stats[channel].min = 8188
    stats[channel].max = 8196
    stats[channel].valid_count = 1
    stats[channel].invalid_count_max = 7
    stats[channel].invalid_count_min = 9
    stats[channel].sumv = 8193
    stats[channel].sumv2 = 8193**2


def housekeeping_blob(hk_type: int, uid: int) -> bytes:
    value = getattr(BINDING.pystruct, f"housekeeping_data_{hk_type}")()
    value.base.version = SCHEMA_ID
    value.base.unique_packet_id = uid
    value.base.errors = 0xC0000000 | hk_type
    value.base.housekeeping_type = hk_type
    if hk_type == 0:
        value.core_state.base.time_32 = 0x10010
        value.core_state.base.time_16 = 0
        set_adc_sentinel(value.core_state.base.ADC_stat, 2)
        value.core_state.base.TVS_sensors[:] = (16000, 8000, 4000, 32000)
    elif hk_type == 1:
        set_adc_sentinel(value.ADC_stat, 1)
        value.actual_gain[:] = (2, 0, 1, 2)
    elif hk_type == 2:
        value.heartbeat.time_32 = 0x20020
        value.heartbeat.time_16 = 0
        value.heartbeat.TVS_sensors[:] = (4000, 8000, 12000, 32000)
        value.heartbeat.magic = b"BRNMRL"
    elif hk_type == 3:
        value.checksum = 0x11223344
        value.weight_ndx = 0x5566
    elif hk_type == 100:
        value.meta_valid[:] = (1, 0, 1, 1, 0, 1)
        value.size[:] = (10, 20, 30, 40, 50, 60)
        value.checksum_meta[:] = (
            0x101,
            0x202,
            0x303,
            0x404,
            0x505,
            0x606,
        )
        value.checksum_data[:] = (
            0x111,
            0x222,
            0x333,
            0x444,
            0x555,
            0x666,
        )
    elif hk_type == 101:
        value.report.region_1 = -7
        value.report.region_2 = 13
        value.report.size_1 = 0x12345678
        value.report.size_2 = 0x87654321
        value.report.checksum_1_meta = 0x11112222
        value.report.checksum_1_data = 0x33334444
        value.report.checksum_2_meta = 0x55556666
        value.report.checksum_2_data = 0x77778888
        value.report.status = -19
    return cdi_padded(value)


def debug_payload(page: int, embedded_metadata: ctypes.Structure) -> bytes:
    values = np.zeros((3, 1024), dtype="<i4")
    if page == 0:
        values[1, 5] = 1 << 20
        values[2, 7] = 333
    elif page == 1:
        values[0, 11] = 401
        values[1, 12] = 402
        values[2, 13] = 403
    elif page == 2:
        values[0, 21] = 501
        values[1, 22] = 502
        values[2, 23] = 503
    elif page == 3:
        values[0, 31] = 601
        values[1, 32] = -602
        values[2, 33] = 603
    elif page == 4:
        values[0, 41] = -701
        values[1, 42] = 702
        values[2, 43] = -703
    elif page == 5:
        values[0, 51] = 801
        values[1, 52] = -802
        values[2, 53] = 803
    elif page == 6:
        values[0, 61] = -901
        values[1, 62] = 902
        values[2, 63] = 160
    elif page == 7:
        values[0, 71] = 176
        values[1, 72] = 192
        values[2, 73] = 208
    payload = bytearray(values.tobytes(order="C"))
    if page == 0:
        encoded_metadata = bytes_of(embedded_metadata)
        payload[2048:2048 + len(encoded_metadata)] = encoded_metadata
    return bytes(payload)


def test_zoom_uses_exact_component_order_uid_time_and_packet_provenance(tmp_path):
    uid = 0x10203040
    time_32 = 0x10010
    time_16 = 0
    values = np.zeros((4, 64), dtype="<f4")
    values[0, 3] = np.float32(1.25)
    values[1, 17] = np.float32(-2.5)
    values[2, 29] = np.float32(3.75)
    values[3, 63] = np.float32(-4.125)
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_MetaData,
        metadata_blob(uid, time_32=time_32, time_16=time_16),
    )
    write_packet(
        tmp_path,
        1,
        BINDING.appids.AppID_ZoomSpectra,
        zoom_blob(uid, 0xBEEF, values),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.zoom_spectra) == 1
    sample = products.zoom_spectra[0]
    assert type(sample) is ZoomSample
    assert sample.data.shape == (4, 64)
    assert sample.data.dtype == np.float32
    assert sample.component_labels == ("AA", "BB", "ABR", "ABI")
    assert sample.pfb_bin == 0xBEEF
    assert sample.unique_packet_id == uid
    assert sample.data[0, 3] == np.float32(1.25)
    assert sample.data[1, 17] == np.float32(-2.5)
    assert sample.data[2, 29] == np.float32(3.75)
    assert sample.data[3, 63] == np.float32(-4.125)
    raw_seconds = raw_seconds_from_split_time(time_32, time_16)
    assert_product_provenance(
        sample,
        uid=uid,
        packet_indices=(1, 0),
        roles=("zoom", "time_metadata"),
        raw_seconds=raw_seconds,
    )


def test_hello_preserves_valid_zero_session_fields(tmp_path):
    value = BINDING.pystruct.startup_hello()
    value.SW_version = SCHEMA_ID
    value.FW_Version = 0
    value.FW_ID = 0
    value.FW_Date = 0
    value.FW_Time = 0
    value.unique_packet_id = 0
    value.time_32 = 0
    value.time_16 = 0
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_uC_Start,
        cdi_padded(value),
    )

    products = read_uncrater_session(tmp_path)

    assert products.sw_version == SCHEMA_ID
    assert products.fw_version == 0
    assert products.fw_id == 0
    assert products.fw_date == 0
    assert products.fw_time == 0
    assert products.start_unique_packet_id == 0
    assert products.start_time_32 == 0
    assert products.start_time_16 == 0
    assert products.start_raw_seconds == 0.0


def test_waveform_uses_attached_metadata_and_keeps_unsigned_adc_time(tmp_path):
    uid = 0x22334455
    time_32 = 0x30030
    time_16 = 0
    timestamp = 2**63 + 0x123456789
    codes = np.zeros(16_384, dtype="<u2")
    codes[[0, 10, 8192, 16_383]] = (17, 8192, 16_383, 4095)
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_RawADC + 2,
        waveform_blob(codes),
    )
    write_packet(
        tmp_path,
        1,
        BINDING.appids.AppID_RawADC_Meta,
        waveform_metadata_blob(
            uid,
            time_32=time_32,
            time_16=time_16,
            timestamp=timestamp,
        ),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.waveforms) == 1
    sample = products.waveforms[0]
    assert type(sample) is WaveformSample
    assert sample.data.shape == (16_384,)
    assert sample.data.dtype == np.int16
    assert sample.channel == 2
    assert sample.unique_packet_id == uid
    assert type(sample.adc_timestamp) is np.uint64
    assert sample.adc_timestamp == np.uint64(timestamp)
    np.testing.assert_array_equal(
        sample.data[[0, 10, 8192, 16_383]],
        np.array([17, -8192, -1, 4095], dtype=np.int16),
    )
    assert_product_provenance(
        sample,
        uid=uid,
        packet_indices=(0, 1),
        roles=("waveform_channel_2", "waveform_metadata"),
        raw_seconds=raw_seconds_from_split_time(time_32, time_16),
        reported_schema_ids=(),
    )
    assert sample.provenance.uid_source_role == "waveform_metadata"
    assert sample.provenance.time_source_role == "waveform_metadata"


def test_every_housekeeping_type_preserves_native_fields_and_missing_time(
    tmp_path,
):
    for index, hk_type in enumerate((0, 1, 2, 3, 100, 101)):
        write_packet(
            tmp_path,
            index,
            BINDING.appids.AppID_uC_Housekeeping,
            housekeeping_blob(hk_type, 0x3000 + hk_type),
        )

    products = read_uncrater_session(tmp_path)

    assert len(products.housekeeping) == 6
    by_type = {sample.hk_type: sample for sample in products.housekeeping}
    assert set(by_type) == {0, 1, 2, 3, 100, 101}
    assert all(type(sample) is HKSample for sample in by_type.values())
    for index, hk_type in enumerate((0, 1, 2, 3, 100, 101)):
        sample = by_type[hk_type]
        expected_time = {
            0: raw_seconds_from_split_time(0x10010, 0),
            2: raw_seconds_from_split_time(0x20020, 0),
        }.get(hk_type)
        assert sample.unique_packet_id == 0x3000 + hk_type
        assert sample.version == SCHEMA_ID
        assert sample.errors == 0xC0000000 | hk_type
        assert_product_provenance(
            sample,
            uid=0x3000 + hk_type,
            packet_indices=(index,),
            roles=("housekeeping",),
            raw_seconds=expected_time,
        )
        assert sample.field_present["time_32"] is (hk_type in (0, 2))
        if hk_type == 0:
            assert sample.fields["time_32"] == 0x10010
            assert sample.fields["time_16"] == 0
        elif hk_type == 2:
            assert sample.fields["time_32"] == 0x20020
            assert sample.fields["time_16"] == 0
        else:
            assert sample.fields["time_32"] is None
            assert sample.fields["time_16"] is None

    hk0 = by_type[0]
    assert hk0.fields["adc_min"][2] == -3
    assert hk0.fields["adc_max"][2] == 5
    assert hk0.fields["adc_valid_count"][2] == 1
    assert hk0.fields["adc_invalid_count_max"][2] == 7
    assert hk0.fields["adc_invalid_count_min"][2] == 9
    assert hk0.fields["adc_total_count"][2] == 17
    assert hk0.fields["adc_mean"][2] == 2.0
    assert hk0.fields["adc_rms"][2] == 0.0
    assert hk0.fields["adc_min"].shape == (4,)
    assert hk0.fields["adc_min"].dtype == np.int64
    assert hk0.fields["adc_mean"].shape == (4,)
    assert hk0.fields["adc_mean"].dtype == np.float64
    assert hk0.fields["adc_statistics_valid"].dtype == np.bool_
    assert hk0.fields["adc_statistics_valid"][2]
    assert hk0.fields["telemetry_v1_0"] == 1.0
    assert hk0.fields["telemetry_v1_8"] == 0.5
    assert hk0.fields["telemetry_v2_5"] == 0.25
    assert hk0.fields["telemetry_t_fpga"] == pytest.approx(-23.15)

    hk1 = by_type[1]
    assert hk1.raw_seconds is None
    assert hk1.fields["actual_gain"] == ("H", "L", "M", "H")
    assert hk1.fields["adc_min"][1] == -3

    hk2 = by_type[2]
    assert hk2.fields["ok"] is True
    assert hk2.fields["telemetry_v1_0"] == 0.25
    assert hk2.fields["telemetry_v2_5"] == 0.75

    hk3 = by_type[3]
    assert hk3.raw_seconds is None
    assert hk3.fields["checksum"] == 0x11223344
    assert hk3.fields["weight_ndx"] == 0x5566

    hk100 = by_type[100]
    np.testing.assert_array_equal(
        hk100.fields["meta_valid"],
        np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8),
    )
    assert hk100.fields["size"].dtype == np.uint32
    assert hk100.fields["size"][4] == 50
    assert hk100.fields["checksum_data"][5] == 0x666

    hk101 = by_type[101]
    assert hk101.fields["region_1"] == -7
    assert hk101.fields["region_2"] == 13
    assert hk101.fields["size_1"] == 0x12345678
    assert hk101.fields["checksum_2_data"] == 0x77778888
    assert hk101.fields["status"] == -19


def test_binding_305_housekeeping_crc_normalizes_without_inventing_time(
    tmp_path,
):
    binding = schema_registry.binding_for_key("305")
    value = binding.pystruct.housekeeping_data_3()
    value.base.version = binding.canonical_schema_id
    value.base.unique_packet_id = 0x3500
    value.base.errors = 0xAABBCCDD
    value.base.housekeeping_type = 3
    value.crc = 0x1234ABCD
    value.weight_ndx = 0x4567
    write_packet(
        tmp_path,
        0,
        binding.appids.AppID_uC_Housekeeping,
        cdi_padded(value),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.housekeeping) == 1
    sample = products.housekeeping[0]
    assert type(sample) is HKSample
    assert sample.hk_type == 3
    assert sample.unique_packet_id == 0x3500
    assert sample.fields["checksum"] == 0x1234ABCD
    assert sample.field_present["checksum"] is True
    assert sample.fields["weight_ndx"] == 0x4567
    assert sample.raw_seconds is None
    assert sample.fields["time_32"] is None
    assert sample.fields["time_16"] is None
    assert sample.field_present["time_32"] is False
    assert sample.provenance.time_valid is False
    assert sample.provenance.reported_schema_ids == (0x305,)
    assert sample.provenance.selected_schema_id == 0x305
    assert products.decode_provenance.binding_key == "305"


@pytest.mark.parametrize("navg2", (1, 3, 7))
def test_grimm_keeps_native_integer_geometry_and_exact_uid_time(tmp_path, navg2):
    uid = 0x4000 + navg2
    time_32 = 0x40040 + 16 * navg2
    time_16 = 0
    values = (np.arange(navg2 * 16 * 4, dtype=np.int32) - 73).reshape(
        navg2, 16, 4
    )
    values[0, 0, 0] = -1234
    values[-1, 15, 3] = 2345
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_MetaData,
        metadata_blob(uid, time_32=time_32, time_16=time_16),
    )
    write_packet(
        tmp_path,
        1,
        BINDING.appids.AppID_SpectraGrimm,
        grimm_blob(uid, values),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.grimm_spectra) == 1
    sample = products.grimm_spectra[0]
    assert type(sample) is GrimmSample
    assert sample.data.dtype == np.int32
    assert sample.data.shape == (navg2, 16, 4)
    assert sample.navg2 == navg2
    np.testing.assert_array_equal(sample.data, values)
    assert_product_provenance(
        sample,
        uid=uid,
        packet_indices=(1, 0),
        roles=("grimm", "time_metadata"),
        raw_seconds=raw_seconds_from_split_time(time_32, time_16),
    )


def test_calibrator_metadata_data_raw_pfb_and_debug_groups_are_exact(tmp_path):
    metadata_uid = 0x5100
    metadata_time = (0x50050, 0)
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_Calibrator_MetaData,
        cdi_padded(calibrator_metadata_value(
            metadata_uid,
            time_32=metadata_time[0],
            time_16=metadata_time[1],
            mode=7,
        )),
    )

    data_uid = 0x5200
    data_time = (0x60060, 0)
    real = np.zeros((4, 512), dtype="<i4")
    imaginary = np.zeros((4, 512), dtype="<i4")
    real[0, 0] = 111
    real[3, 511] = -222
    imaginary[1, 17] = -333
    gphase = np.zeros(1024, dtype="<i4")
    gphase[321] = 987654
    page2 = np.concatenate((np.array([-17], dtype="<i4"), gphase))
    for page, values in enumerate((real, imaginary, page2), start=0):
        write_packet(
            tmp_path,
            1 + page,
            BINDING.appids.AppID_Calibrator_Data + page,
            page_blob(
                data_uid,
                values,
                time_32=data_time[0],
                time_16=data_time[1],
            ),
        )

    pfb_uid = 0x5300
    pfb_time = (0x70070, 0)
    pfb_pages = [np.zeros(2048, dtype="<i4") for _ in range(8)]
    pfb_pages[0][5] = 101
    pfb_pages[1][5] = -202
    pfb_pages[6][-1] = 303
    pfb_pages[7][-1] = 404
    for page, values in enumerate(pfb_pages):
        write_packet(
            tmp_path,
            4 + page,
            BINDING.appids.AppID_Calibrator_RawPFB + page,
            page_blob(
                pfb_uid,
                values,
                time_32=pfb_time[0],
                time_16=pfb_time[1],
            ),
        )

    debug_uid = 0x5400
    debug_time = (0x80080, 0)
    embedded = calibrator_metadata_value(
        debug_uid,
        time_32=debug_time[0],
        time_16=debug_time[1],
        mode=9,
    )
    for page in range(8):
        write_packet(
            tmp_path,
            12 + page,
            BINDING.appids.AppID_Calibrator_Debug + page,
            page_blob(
                debug_uid,
                debug_payload(page, embedded),
                time_32=debug_time[0],
                time_16=debug_time[1],
            ),
        )

    products = read_uncrater_session(tmp_path)

    direct = next(
        sample
        for sample in products.calibrator_metadata
        if sample.unique_packet_id == metadata_uid and not sample.from_debug
    )
    assert type(direct) is CalibratorMetadataSample
    assert direct.fields["version"] == SCHEMA_ID
    assert direct.fields["mode"] == 7
    assert direct.fields["snr_on"] == 0x10203040
    assert direct.fields["snr_off"] == 0x50607080
    assert direct.fields["drift_raw"].shape == (1024,)
    assert direct.fields["drift_raw"].dtype == np.int64
    assert direct.fields["drift"].shape == (1024,)
    assert direct.fields["drift"].dtype == np.float64
    assert direct.fields["drift_raw"][24] == 34
    assert direct.fields["drift_raw"][728] == -46
    assert direct.fields["drift"][24] == pytest.approx(34 * np.pi / (1 << 30))
    assert_product_provenance(
        direct,
        uid=metadata_uid,
        packet_indices=(0,),
        roles=("calibrator_metadata",),
        raw_seconds=raw_seconds_from_split_time(*metadata_time),
    )

    assert len(products.calibrator_data) == 1
    data = products.calibrator_data[0]
    assert type(data) is CalibratorDataSample
    assert data.data.dtype == np.complex128
    assert data.data.shape == (4, 512)
    assert data.data[0, 0] == 111 + 0j
    assert data.data[1, 17] == -333j
    assert data.data[3, 511] == -222 + 0j
    assert data.g_nacc == -17
    assert data.gphase.dtype == np.int32
    assert data.gphase.shape == (1024,)
    assert data.gphase[321] == 987654
    np.testing.assert_array_equal(
        data.page_raw_seconds,
        np.full(3, raw_seconds_from_split_time(*data_time)),
    )
    assert_product_provenance(
        data,
        uid=data_uid,
        packet_indices=(1, 2, 3),
        roles=(
            "calibrator_data_page_0",
            "calibrator_data_page_1",
            "calibrator_data_page_2",
        ),
        raw_seconds=raw_seconds_from_split_time(*data_time),
    )

    assert len(products.calibrator_raw_pfb) == 1
    pfb = products.calibrator_raw_pfb[0]
    assert type(pfb) is CalibratorRawPFBSample
    assert pfb.data.dtype == np.complex128
    assert pfb.data.shape == (4, 2048)
    assert pfb.data[0, 5] == 101 - 202j
    assert pfb.data[3, -1] == 303 + 404j
    np.testing.assert_array_equal(
        pfb.page_raw_seconds,
        np.full(8, raw_seconds_from_split_time(*pfb_time)),
    )
    assert_product_provenance(
        pfb,
        uid=pfb_uid,
        packet_indices=tuple(range(4, 12)),
        roles=tuple(f"calibrator_raw_pfb_page_{page}" for page in range(8)),
        raw_seconds=raw_seconds_from_split_time(*pfb_time),
    )

    assert len(products.calibrator_debug) == 1
    debug = products.calibrator_debug[0]
    assert type(debug) is CalibratorDebugSample
    assert tuple(page.page for page in debug.pages) == tuple(range(8))
    np.testing.assert_array_equal(
        debug.page_raw_seconds,
        np.full(8, raw_seconds_from_split_time(*debug_time)),
    )
    assert debug.pages[0].fields["powertop0"].shape == (1024,)
    assert debug.pages[0].fields["powertop0"].dtype == np.int64
    assert debug.pages[0].fields["drift"].shape == (1024,)
    assert debug.pages[0].fields["drift"].dtype == np.float64
    assert debug.pages[6].fields["snr0"].shape == (1024,)
    assert debug.pages[6].fields["snr0"].dtype == np.float64
    assert debug.pages[0].fields["powertop0"][7] == 333
    assert debug.pages[0].fields["drift"][5] == pytest.approx(np.pi / 1024)
    assert debug.pages[1].fields["powertop1"][11] == 401
    assert debug.pages[3].fields["fd0"][32] == -602
    assert debug.pages[4].fields["sd0"][43] == -703
    assert debug.pages[6].fields["fdx"][61] == -901
    assert debug.pages[6].fields["snr0"][63] == 10.0
    assert debug.pages[7].fields["snr3"][73] == 13.0
    embedded_metadata = debug.pages[0].fields["embedded_metadata"]
    assert embedded_metadata["fields"]["mode"] == 9
    assert embedded_metadata["field_present"]["mode"] is True
    assert embedded_metadata["raw_seconds"] == raw_seconds_from_split_time(
        *debug_time
    )
    assert_product_provenance(
        debug,
        uid=debug_uid,
        packet_indices=tuple(range(12, 20)),
        roles=tuple(f"calibrator_debug_page_{page}" for page in range(8)),
        raw_seconds=raw_seconds_from_split_time(*debug_time),
    )


@pytest.mark.parametrize(
    (
        "binding_key",
        "outer_mode_present",
        "error_regs_present",
        "typed_errors_present",
        "state_present",
        "stats_present",
    ),
    (
        ("203", False, True, False, True, False),
        ("305", False, True, False, False, True),
        ("306-early", False, True, False, False, True),
        ("306-final", True, False, True, False, True),
        ("307", True, False, True, False, True),
    ),
)
def test_direct_calibrator_metadata_normalizes_each_historical_binding(
    tmp_path,
    binding_key,
    outer_mode_present,
    error_regs_present,
    typed_errors_present,
    state_present,
    stats_present,
):
    binding = schema_registry.binding_for_key(binding_key)
    uid = 0x6000 + len(binding_key)
    time_32 = 0x90090
    time_16 = 0
    value = historical_calibrator_metadata_value(
        binding,
        uid,
        time_32=time_32,
        time_16=time_16,
    )
    write_packet(
        tmp_path,
        0,
        binding.appids.AppID_Calibrator_MetaData,
        cdi_padded(value),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.calibrator_metadata) == 1
    sample = products.calibrator_metadata[0]
    assert type(sample) is CalibratorMetadataSample
    assert sample.unique_packet_id == uid
    assert sample.fields["version"] == binding.canonical_schema_id
    assert sample.fields["have_lock"].dtype == np.uint16
    assert sample.fields["have_lock"][2] == 13
    assert sample.field_present["mode"] is outer_mode_present
    assert sample.field_present["error_regs"] is error_regs_present
    assert sample.field_present["error_check"] is typed_errors_present
    assert sample.field_present["state_mode"] is state_present
    assert sample.field_present["stats_snr_max"] is stats_present
    assert sample.fields["drift_raw"].dtype == np.int64
    assert sample.fields["drift"].dtype == np.float64
    assert sample.fields["drift_raw"].shape == (1024,)
    assert sample.fields["drift"].shape == (1024,)
    drift_index = 3 if binding_key == "203" else 24
    expected_raw = 17 if binding_key == "203" else 34
    assert sample.fields["drift_raw"][drift_index] == expected_raw
    assert sample.fields["drift"][drift_index] == pytest.approx(
        expected_raw * np.pi / (1 << 30)
    )
    if binding_key != "203":
        np.testing.assert_array_equal(
            sample.fields["drift_raw"][24:32],
            np.full(8, 34, dtype=np.int64),
        )
    if binding_key == "203":
        assert sample.fields["snr_max"] == -111
        assert sample.fields["snr_min"] == 222
        assert sample.fields["state_mode"] == 5
        assert sample.fields["state_weight_ndx"] == 0x3456
        assert sample.fields["error_regs"].shape == (30,)
        assert sample.fields["error_regs"].dtype == np.uint32
        assert sample.fields["error_regs"][4] == 0xA1B2C3D4
    else:
        assert sample.fields["snr_on"] == 0x10203040
        assert sample.fields["snr_off"] == 0x50607080
        assert sample.fields["stats_snr_max"][1] == 0x12345678
        assert sample.fields["stats_fd_min"][2] == -44
        assert sample.fields["stats_sd_positive_count"][3] == 55
        if error_regs_present:
            assert sample.fields["error_regs"][4] == 0xA1B2C3D4
        else:
            assert sample.fields["mode"] == 9
            assert sample.fields["error_cal_phaser"].shape == (2,)
            assert sample.fields["error_cal_phaser"].dtype == np.uint32
            assert sample.fields["error_cal_phaser"][1] == 0x11223344
            assert sample.fields["error_check"] == 0x55667788
    assert sample.raw_seconds == raw_seconds_from_split_time(time_32, time_16)
    assert sample.provenance.reported_schema_ids == (
        binding.canonical_schema_id,
    )
    assert sample.provenance.selected_schema_id == binding.canonical_schema_id
    assert products.decode_provenance.binding_key == binding_key
    assert sample.provenance.source_packets[0].packet_index == 0
    assert sample.provenance.source_packets[0].role == "calibrator_metadata"


def test_calibrator_data_group_preserves_distinct_page_times(tmp_path):
    uid = 0x7000
    pages = (
        np.zeros((4, 512), dtype="<i4"),
        np.zeros((4, 512), dtype="<i4"),
        np.zeros(1025, dtype="<i4"),
    )
    times = (0x10010, 0x20020, 0x10010)
    for page, (values, time_32) in enumerate(zip(pages, times)):
        write_packet(
            tmp_path,
            page,
            BINDING.appids.AppID_Calibrator_Data + page,
            page_blob(
                uid,
                values,
                time_32=time_32,
                time_16=0,
            ),
        )

    products = read_uncrater_session(tmp_path)

    assert len(products.calibrator_data) == 1
    sample = products.calibrator_data[0]
    expected_times = np.array(
        [raw_seconds_from_split_time(time_32, 0) for time_32 in times],
        dtype=np.float64,
    )
    assert sample.raw_seconds == expected_times[0]
    np.testing.assert_array_equal(sample.page_raw_seconds, expected_times)
    assert not any(
        issue.code == "decode_adapter.invalid_calibrator_data"
        for issue in products.issues
    )


def test_malformed_orphan_and_duplicate_inputs_create_no_zero_rows(tmp_path):
    zoom_values = np.ones((4, 64), dtype="<f4")
    write_packet(
        tmp_path,
        0,
        BINDING.appids.AppID_ZoomSpectra,
        zoom_blob(1, 2, zoom_values)[:-4],
    )
    malformed_waveform = np.zeros(16_383, dtype="<u2").tobytes()
    write_packet(
        tmp_path,
        1,
        BINDING.appids.AppID_RawADC,
        malformed_waveform,
    )
    write_packet(
        tmp_path,
        2,
        BINDING.appids.AppID_RawADC_Meta,
        waveform_metadata_blob(2, time_32=16, time_16=0, timestamp=3),
    )
    duplicate = np.zeros(16_384, dtype="<u2")
    write_packet(
        tmp_path,
        3,
        BINDING.appids.AppID_RawADC + 1,
        waveform_blob(duplicate),
    )
    write_packet(
        tmp_path,
        4,
        BINDING.appids.AppID_RawADC + 1,
        waveform_blob(duplicate),
    )
    write_packet(
        tmp_path,
        5,
        BINDING.appids.AppID_RawADC_Meta,
        waveform_metadata_blob(4, time_32=16, time_16=0, timestamp=5),
    )
    malformed_grimm = grimm_blob(
        6, np.ones((1, 16, 4), dtype=np.int32)
    ) + b"\x34\x12"
    write_packet(
        tmp_path,
        6,
        BINDING.appids.AppID_SpectraGrimm,
        malformed_grimm,
    )
    orphan = np.zeros((4, 512), dtype="<i4")
    write_packet(
        tmp_path,
        7,
        BINDING.appids.AppID_Calibrator_Data + 1,
        page_blob(7, orphan, time_32=16, time_16=0),
    )
    write_packet(
        tmp_path,
        8,
        BINDING.appids.AppID_Calibrator_RawPFB,
        page_blob(
            8,
            np.zeros(2048, dtype="<i4"),
            time_32=16,
            time_16=0,
        ),
    )
    debug_meta = calibrator_metadata_value(
        9,
        time_32=16,
        time_16=0,
        mode=1,
    )
    write_packet(
        tmp_path,
        9,
        BINDING.appids.AppID_Calibrator_Debug,
        page_blob(
            9,
            debug_payload(0, debug_meta),
            time_32=16,
            time_16=0,
        ),
    )

    products = read_uncrater_session(tmp_path)

    assert products.zoom_spectra == []
    assert products.waveforms == []
    assert products.grimm_spectra == []
    assert products.calibrator_data == []
    assert products.calibrator_raw_pfb == []
    assert products.calibrator_debug == []
    codes = {issue.code for issue in products.issues}
    assert "decode.bad_blob_length" in codes
    assert "decode.duplicate_waveform_channel" in codes
    assert "decode.payload_decode_failed" in codes
    assert "decode.orphan_multipart_page" in codes
    assert "decode.missing_multipart_page" in codes


@pytest.mark.parametrize(
    ("dropped_family", "surviving_family"),
    (
        ("calibrator_metadata", "grimm_spectra"),
        ("grimm_spectra", "calibrator_metadata"),
    ),
)
def test_fatal_auxiliary_packet_is_attributed_with_another_family_present(
    tmp_path,
    dropped_family,
    surviving_family,
):
    if dropped_family == "calibrator_metadata":
        uid = 0x6100
        write_packet(
            tmp_path,
            0,
            BINDING.appids.AppID_MetaData,
            metadata_blob(uid, time_32=0x10010, time_16=0),
        )
        write_packet(
            tmp_path,
            1,
            BINDING.appids.AppID_SpectraGrimm,
            grimm_blob(uid, np.ones((1, 16, 4), dtype=np.int32)),
        )
        damaged_index = 2
        damaged = cdi_padded(calibrator_metadata_value(
            0x6200,
            time_32=0x20020,
            time_16=0,
            mode=1,
        ))[:-4]
        write_packet(
            tmp_path,
            damaged_index,
            BINDING.appids.AppID_Calibrator_MetaData,
            damaged,
        )
    else:
        write_packet(
            tmp_path,
            0,
            BINDING.appids.AppID_Calibrator_MetaData,
            cdi_padded(calibrator_metadata_value(
                0x6300,
                time_32=0x30030,
                time_16=0,
                mode=1,
            )),
        )
        damaged_index = 1
        damaged = grimm_blob(
            0x6400,
            np.ones((1, 16, 4), dtype=np.int32),
        ) + b"\x34\x12"
        write_packet(
            tmp_path,
            damaged_index,
            BINDING.appids.AppID_SpectraGrimm,
            damaged,
        )

    products = read_uncrater_session(tmp_path)

    assert getattr(products, dropped_family) == []
    assert len(getattr(products, surviving_family)) == 1
    issue = next(
        issue
        for issue in products.issues
        if issue.packet_index == damaged_index
        and issue.action is IssueAction.DROPPED
    )
    assert products.family_issue_ids[dropped_family] == (issue.issue_id,)
    assert surviving_family not in products.family_issue_ids
    assert products.quality_status.value == "partial"
    statuses = {
        status.family: status
        for status in family_statuses_for_products(
            products,
            family_issue_ids=products.family_issue_ids,
        )
    }
    assert statuses[dropped_family].coverage is FamilyCoverage.INVALID_OR_DROPPED
    assert statuses[dropped_family].quality is DataQuality.FAILED
    assert statuses[surviving_family].coverage is FamilyCoverage.PERSISTED
    assert statuses[surviving_family].quality is DataQuality.CLEAN
