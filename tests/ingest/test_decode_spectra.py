from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from lusee.ingest.decode import (
    _extract_spectrum_metadata,
    read_uncrater_session,
)
from lusee.ingest.issues import (
    IngestIssueError,
    IssueAction,
    IssueCollector,
    IssuePolicy,
)
from lusee.ingest.products import (
    SpectrumSample,
    TRSpectrumSample,
)

uncrater = pytest.importorskip("uncrater")
schema_registry = pytest.importorskip("uncrater.schema_registry")

UC_TIME = 2**63 + 123


def metadata_blob(
    binding,
    *,
    uid: int = 42,
    navgf: int = 3,
    navg2_shift: int = 1,
    tr_start: int = 0,
    tr_stop: int = 4,
    tr_average_shift: int = 1,
) -> bytes:
    record = binding.pystruct.meta_data()
    record.version = binding.canonical_schema_id
    record.unique_packet_id = uid
    base = record.base
    base.uC_time = UC_TIME
    base.time_32 = 16
    base.time_16 = 0
    for index, value in enumerate((16000, 16001, 16002, 16003)):
        base.TVS_sensors[index] = value
    for index in range(4):
        base.gain[index] = index
        base.gain_auto_min[index] = 10 + index
        base.gain_auto_mult[index] = 20 + index
        base.route[index].plus = index
        base.route[index].minus = 3 - index
        base.actual_gain[index] = index + 1
    base.Navg1_shift = 2
    base.Navg2_shift = navg2_shift
    base.notch = 1
    base.Navgf = navgf
    base.hi_frac = 7
    base.med_frac = 9
    for product in range(16):
        base.bitslice[product] = 20
        base.actual_bitslice[product] = 31
    base.bitslice[0] = 0xFF
    base.actual_bitslice[4] = 30
    base.bitslice_keep_bits = 12
    base.format = binding.pystruct.OUTPUT_32BIT
    base.reject_ratio = 4
    base.reject_maxbad = 5
    base.tr_start = tr_start
    base.tr_stop = tr_stop
    base.tr_avg_shift = tr_average_shift
    base.errors = 0x10
    base.corr_products_mask = 0xFFFF
    base.spec_overflow = 2
    base.notch_overflow = 3
    base.spectrometer_enable = True
    base.calibrator_enable = False
    base.rand_state = 0x12345678
    if binding.binding_key == "203":
        base.weight_previous = 2
    else:
        base.loop_count_min = 1
        base.loop_count_max = 2
        base.grimm_enable = 1
        base.averaging_mode = 2
        base.weight = 2
        base.num_bad_min_current = 3
        base.num_bad_max_current = 4
        base.num_bad_min = 5
        base.num_bad_max = 6
    base.weight_current = 3
    blob = bytes(record)
    padding = (-len(blob)) % 4
    return blob + b"\x00" * padding


def write_packet(directory: Path, index: int, appid: int, blob: bytes) -> None:
    (directory / f"{index:05d}_{appid:04x}.bin").write_bytes(blob)


def spectrum_blob(uid: int, values: np.ndarray) -> bytes:
    payload = values.tobytes(order="C")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack("<II", uid, crc) + payload


def tr_blob(uid: int, encoded_values: np.ndarray) -> bytes:
    payload = np.asarray(encoded_values, dtype="<u2").tobytes(order="C")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return struct.pack("<II", uid, crc) + payload


def write_partial_normal_tr_session(directory: Path) -> None:
    binding = schema_registry.LATEST_BINDING
    uid = 42
    write_packet(directory, 0, binding.appids.AppID_MetaData, metadata_blob(binding))
    write_packet(
        directory,
        1,
        binding.appids.AppID_SpectraHigh,
        spectrum_blob(uid, np.full(512, 4, dtype="<u4")),
    )
    # Product 1 is structurally rejected and must remain absent, not zero
    write_packet(
        directory,
        2,
        binding.appids.AppID_SpectraHigh + 1,
        spectrum_blob(uid, np.full(7, 6, dtype="<u4")),
    )
    write_packet(
        directory,
        3,
        binding.appids.AppID_SpectraHigh + 4,
        spectrum_blob(uid, np.full(512, -8, dtype="<i4")),
    )
    write_packet(
        directory,
        4,
        binding.appids.AppID_SpectraTRHigh,
        tr_blob(uid, np.array([0x8019, 0x8039, 0x8019, 0x8039])),
    )


@pytest.mark.parametrize(
    "binding_key",
    ("203", "305", "306-early", "306-final", "307"),
)
def test_metadata_field_map_covers_every_reviewed_binding(binding_key):
    binding = schema_registry.binding_for_key(binding_key)
    packet = uncrater.Packet_Metadata(
        binding.appids.AppID_MetaData,
        blob=metadata_blob(binding),
        schema=binding,
        reported_version=binding.canonical_schema_id,
        strict=False,
    )

    metadata = _extract_spectrum_metadata(
        packet,
        binding_key=binding_key,
    )

    assert metadata.version == binding.canonical_schema_id
    assert metadata.unique_packet_id == 42
    assert metadata.uc_time == UC_TIME
    assert metadata.raw_seconds == 1 / 4096
    assert metadata.navgf == 3
    assert metadata.weight == 2
    assert metadata.requested_bitslice[0] == 0xFF
    np.testing.assert_array_equal(metadata.route_plus, np.arange(4))
    np.testing.assert_array_equal(metadata.route_minus, np.arange(3, -1, -1))
    assert metadata.current_fields_present is (binding_key != "203")
    assert metadata.loop_count_min == (None if binding_key == "203" else 1)


def test_real_collection_adapts_partial_normal_and_tr_rows_exactly(tmp_path):
    write_partial_normal_tr_session(tmp_path)

    collector = IssueCollector()
    products = read_uncrater_session(
        tmp_path,
        issue_collector=collector,
        strict=False,
    )

    assert len(products.spectra) == 1
    normal = products.spectra[0]
    assert type(normal) is SpectrumSample
    assert normal.data.shape == (16, 512)
    assert normal.data.dtype == np.float32
    assert normal.product_present[[0, 4]].tolist() == [True, True]
    assert normal.product_present.sum() == 2
    np.testing.assert_array_equal(normal.data[0], np.full(512, 4, np.float32))
    np.testing.assert_array_equal(normal.data[4], np.full(512, -4, np.float32))
    assert np.isnan(normal.data[1]).all()
    assert np.isnan(normal.data[15]).all()

    assert len(products.tr_spectra) == 1
    tr = products.tr_spectra[0]
    assert type(tr) is TRSpectrumSample
    assert tr.data.shape == (16, 2, 2)
    assert tr.data.dtype == np.int32
    assert tr.product_present.sum() == 1
    np.testing.assert_array_equal(tr.data[0], [[64, -64], [64, -64]])
    np.testing.assert_array_equal(tr.data[1], np.zeros((2, 2), np.int32))

    assert products.validated_counts.product_rows == (
        ("spectra", 1),
        ("tr_spectra", 1),
    )
    assert products.quality_status.value == "partial"
    fatal = next(
        issue for issue in products.issues if issue.code == "decode.bad_blob_length"
    )
    assert fatal.action is IssueAction.DROPPED
    assert fatal.issue_id in normal.provenance.decoder_issue_ids
    assert products.issues == collector.issues


def test_duplicate_product_is_nan_while_other_product_survives(tmp_path):
    binding = schema_registry.LATEST_BINDING
    uid = 43
    write_packet(
        tmp_path,
        0,
        binding.appids.AppID_MetaData,
        metadata_blob(binding, uid=uid),
    )
    write_packet(
        tmp_path,
        1,
        binding.appids.AppID_SpectraHigh,
        spectrum_blob(uid, np.full(512, 1, dtype="<u4")),
    )
    write_packet(
        tmp_path,
        2,
        binding.appids.AppID_SpectraHigh,
        spectrum_blob(uid, np.full(512, 2, dtype="<u4")),
    )
    write_packet(
        tmp_path,
        3,
        binding.appids.AppID_SpectraHigh + 1,
        spectrum_blob(uid, np.full(512, 3, dtype="<u4")),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.spectra) == 1
    row = products.spectra[0]
    assert row.product_present[:2].tolist() == [False, True]
    assert np.isnan(row.data[0]).all()
    np.testing.assert_array_equal(row.data[1], np.full(512, 3, np.float32))
    duplicate = next(
        issue
        for issue in products.issues
        if issue.code == "decode_adapter.duplicate_normal_product"
    )
    assert duplicate.action is IssueAction.DROPPED
    assert duplicate.as_dict()["details"]["packet_indices"] == [1, 2]
    assert duplicate.issue_id in row.provenance.decoder_issue_ids


def test_fatal_normal_candidate_does_not_suppress_valid_sibling(tmp_path):
    binding = schema_registry.LATEST_BINDING
    write_packet(
        tmp_path, 0, binding.appids.AppID_MetaData, metadata_blob(binding)
    )
    write_packet(
        tmp_path,
        1,
        binding.appids.AppID_SpectraHigh,
        spectrum_blob(42, np.full(7, 1, dtype="<u4")),
    )
    write_packet(
        tmp_path,
        2,
        binding.appids.AppID_SpectraHigh,
        spectrum_blob(42, np.full(512, 5, dtype="<u4")),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.spectra) == 1
    row = products.spectra[0]
    assert row.product_present[0]
    np.testing.assert_array_equal(row.data[0], np.full(512, 5, np.float32))
    assert not any(
        issue.code == "decode_adapter.duplicate_normal_product"
        for issue in products.issues
    )
    fatal = next(
        issue
        for issue in products.issues
        if issue.code == "decode.bad_blob_length"
        and issue.packet_index == 1
    )
    assert fatal.issue_id in row.provenance.decoder_issue_ids


def test_fatal_tr_candidate_does_not_suppress_valid_sibling(tmp_path):
    binding = schema_registry.LATEST_BINDING
    write_packet(
        tmp_path, 0, binding.appids.AppID_MetaData, metadata_blob(binding)
    )
    write_packet(
        tmp_path,
        1,
        binding.appids.AppID_SpectraTRHigh,
        tr_blob(42, np.array([0x8019])),
    )
    write_packet(
        tmp_path,
        2,
        binding.appids.AppID_SpectraTRHigh,
        tr_blob(42, np.array([0x8019, 0x8039, 0x8019, 0x8039])),
    )

    products = read_uncrater_session(tmp_path)

    assert len(products.tr_spectra) == 1
    row = products.tr_spectra[0]
    assert row.product_present[0]
    np.testing.assert_array_equal(row.data[0], [[64, -64], [64, -64]])
    assert not any(
        issue.code == "decode_adapter.duplicate_tr_product"
        for issue in products.issues
    )
    fatal = next(
        issue
        for issue in products.issues
        if issue.code == "decode.bad_blob_length"
        and issue.packet_index == 1
    )
    assert fatal.issue_id in row.provenance.decoder_issue_ids


def test_strict_issue_collector_raises_on_adapter_duplicate(tmp_path):
    binding = schema_registry.LATEST_BINDING
    write_packet(tmp_path, 0, binding.appids.AppID_MetaData, metadata_blob(binding))
    packet = spectrum_blob(42, np.full(512, 1, dtype="<u4"))
    write_packet(tmp_path, 1, binding.appids.AppID_SpectraHigh, packet)
    write_packet(tmp_path, 2, binding.appids.AppID_SpectraHigh, packet)

    with pytest.raises(IngestIssueError) as caught:
        read_uncrater_session(
            tmp_path,
            issue_collector=IssueCollector(IssuePolicy.STRICT),
            strict=False,
        )

    assert caught.value.issue.code == "decode_adapter.duplicate_normal_product"


def test_metadata_only_group_does_not_fabricate_science_row(tmp_path):
    binding = schema_registry.LATEST_BINDING
    write_packet(tmp_path, 0, binding.appids.AppID_MetaData, metadata_blob(binding))

    products = read_uncrater_session(tmp_path)

    assert products.spectra == []
    assert products.tr_spectra == []
    assert products.validated_counts.product_rows == ()
    assert products.issues == ()
    assert products.quality_status.value == "failed"
