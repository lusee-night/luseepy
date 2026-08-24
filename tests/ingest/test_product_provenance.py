"""Focused validation tests for ingest product/decode provenance records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from lusee.ingest.products import (
    DataQuality,
    DecodeProvenance,
    ExecutionMode,
    ProductProvenance,
    SourcePacketProvenance,
    ValidatedCounts,
)
from lusee.ingest.decode import (
    HKSample,
    Products,
    SpectrumSample,
    WaveformSample,
    _finite_seconds_or_none,
    _meta_raw_seconds,
    _usable_product_count,
    _waveform_adc_timestamp,
)


DECODER_COMMIT = "0" * 40
BINDING_COMMIT = "1" * 40
ABI_FINGERPRINT = "2" * 64


def source_packet(role: str = "metadata") -> SourcePacketProvenance:
    return SourcePacketProvenance(
        role=role,
        filename="00042_020f_123.bin",
        packet_index=42,
        original_appid=0x20F,
        normalized_appid=0x20F,
        bank="b05",
        frame_start=10,
        frame_stop=12,
        byte_offset_start=100,
        byte_offset_stop=500,
    )


def test_product_provenance_is_immutable_and_validates_row_identity():
    provenance = ProductProvenance(
        source_packets=(source_packet(),),
        uid=123,
        uid_source="metadata.unique_packet_id",
        uid_source_role="metadata",
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        decoder_issue_ids=("decoder-0001",),
        time_source="metadata.time_32_time_16",
        time_source_role="metadata",
        clock_source="spectrometer",
        time_valid=True,
    )

    provenance.validate_product_identity(unique_packet_id=123, raw_seconds=10.5)
    with pytest.raises(FrozenInstanceError):
        provenance.uid = 5
    with pytest.raises(ValueError, match="UID disagrees"):
        provenance.validate_product_identity(unique_packet_id=124, raw_seconds=10.5)
    with pytest.raises(ValueError, match="time_valid"):
        provenance.validate_product_identity(unique_packet_id=123, raw_seconds=None)


def test_missing_time_is_none_and_never_an_implicit_zero():
    provenance = ProductProvenance(
        source_packets=(source_packet(),),
        uid=123,
        uid_source="metadata.unique_packet_id",
        selected_schema_id=0x307,
        clock_source="spectrometer",
        time_valid=False,
    )

    provenance.validate_product_identity(unique_packet_id=123, raw_seconds=None)
    with pytest.raises(ValueError, match="time_valid"):
        provenance.validate_product_identity(unique_packet_id=123, raw_seconds=0.0)


def test_multi_packet_roles_link_uid_and_time_sources():
    provenance = ProductProvenance(
        source_packets=(
            source_packet("waveform_metadata"),
            SourcePacketProvenance(
                role="waveform_ch0",
                filename=None,
                packet_index=44,
                original_appid=0x2F0,
            ),
        ),
        uid=77,
        uid_source="metadata.unique_packet_id",
        uid_source_role="waveform_metadata",
        selected_schema_id=0x307,
        time_source="metadata.split_mission_time",
        time_source_role="waveform_metadata",
        clock_source="spectrometer",
        time_valid=True,
    )

    assert provenance.source_packets[1].filename is None
    provenance.validate_product_identity(unique_packet_id=77, raw_seconds=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "", "original_appid": 0x20F},
        {"role": "meta", "original_appid": -1},
        {"role": "meta", "original_appid": 0x800},
        {"role": "meta", "original_appid": 0x20F, "filename": "a/b.bin"},
        {"role": "meta", "original_appid": 0x20F, "filename": "a\\b.bin"},
        {"role": "meta", "original_appid": 0x20F, "packet_index": -1},
        {"role": "meta", "original_appid": 0x20F, "frame_start": 1},
        {"role": "meta", "original_appid": 0x20F, "frame_start": 2, "frame_stop": 1},
        {"role": "meta", "original_appid": 0x20F},
    ],
)
def test_invalid_source_packet_provenance_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SourcePacketProvenance(**kwargs)


def test_product_provenance_rejects_broken_references_and_duplicates():
    with pytest.raises(ValueError, match="valid product time"):
        ProductProvenance(time_valid=True)
    with pytest.raises(ValueError, match="does not identify"):
        ProductProvenance(
            source_packets=(source_packet(),),
            uid=1,
            uid_source="metadata.unique_packet_id",
            uid_source_role="other",
        )
    with pytest.raises(ValueError, match="duplicates"):
        ProductProvenance(decoder_issue_ids=("issue-1", "issue-1"))
    with pytest.raises(ValueError, match="duplicate schema"):
        ProductProvenance(reported_schema_ids=(0x307, 0x307))
    with pytest.raises(ValueError, match="cannot accompany"):
        ProductProvenance(
            source_packets=(source_packet(),),
            unavailable_reason="unknown",
        )
    with pytest.raises(ValueError, match="recorded together"):
        ProductProvenance(uid=1)
    with pytest.raises(ValueError, match="recorded together"):
        ProductProvenance(uid_source="metadata.unique_packet_id")
    with pytest.raises(ValueError, match="clock_source"):
        ProductProvenance(
            clock_source="spectromter",
            unavailable_reason="test",
        )
    with pytest.raises(ValueError, match="requires uid"):
        ProductProvenance(source_packets=(source_packet(),))


def test_unavailable_provenance_is_explicit_not_none():
    provenance = ProductProvenance.unavailable()

    assert provenance.unavailable_reason == "caller_constructed"
    assert provenance.source_packets == ()
    assert provenance.time_valid is False


def test_decode_provenance_canonicalizes_counts_and_report():
    provenance = DecodeProvenance.from_report(
        distribution_version="1.0.0",
        decoder_source_commit=DECODER_COMMIT,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant=None,
        schema_assumed=False,
        binding_source_release="3r09",
        binding_source_commit=BINDING_COMMIT,
        abi_fingerprint=ABI_FINGERPRINT,
        execution_mode="collect",
        input_packet_count=3,
        valid_packet_count=2,
        appid_counts=((0x210, 2), (0x20F, 1)),
        issue_counts=(("crc_mismatch", 1),),
        canonical_report={"z": 1, "a": [2, 3]},
    )

    assert provenance.execution_mode is ExecutionMode.COLLECT
    assert provenance.appid_counts == ((0x20F, 1), (0x210, 2))
    assert provenance.canonical_report_json == '{"a":[2,3],"z":1}'
    report = provenance.canonical_report()
    assert report == {"a": [2, 3], "z": 1}
    report["a"].append(4)
    assert provenance.canonical_report() == {"a": [2, 3], "z": 1}
    with pytest.raises(FrozenInstanceError):
        provenance.schema_assumed = True


def test_decode_provenance_rejects_count_and_identity_mismatches():
    kwargs = dict(
        decoder_name="uncrater",
        distribution_version="1.0.0",
        decoder_source_commit=DECODER_COMMIT,
        reported_schema_ids=(0x307,),
        selected_schema_id=0x307,
        binding_key="307",
        schema_variant=None,
        schema_assumed=False,
        binding_source_release="3r09",
        binding_source_commit=BINDING_COMMIT,
        abi_fingerprint=ABI_FINGERPRINT,
        execution_mode=ExecutionMode.COLLECT,
        input_packet_count=3,
        valid_packet_count=2,
        appid_counts=((0x20F, 1), (0x210, 2)),
        issue_counts=(),
        canonical_report_json='{"a":1}',
    )
    with pytest.raises(ValueError, match="sum"):
        DecodeProvenance(**{**kwargs, "appid_counts": ((0x20F, 1),)})
    with pytest.raises(ValueError, match="exceeds"):
        DecodeProvenance(**{**kwargs, "valid_packet_count": 4})
    with pytest.raises(ValueError, match="40-hex"):
        DecodeProvenance(**{**kwargs, "decoder_source_commit": "ABC"})
    with pytest.raises(ValueError, match="canonical JSON"):
        DecodeProvenance(**{**kwargs, "canonical_report_json": '{"a": 1}'})
    with pytest.raises(ValueError, match="concrete field"):
        DecodeProvenance(
            decoder_name="uncrater",
            distribution_version=None,
            decoder_source_commit=None,
            reported_schema_ids=(),
            selected_schema_id=None,
            binding_key=None,
            schema_variant=None,
            schema_assumed=True,
            binding_source_release=None,
            binding_source_commit=None,
            abi_fingerprint=None,
            execution_mode=None,
            input_packet_count=0,
            valid_packet_count=None,
            appid_counts=(),
            issue_counts=(),
            canonical_report_json=None,
        )


def test_validated_counts_keep_decode_product_and_write_boundaries_separate():
    counts = ValidatedCounts(
        input_packets=5,
        valid_packets=4,
        product_rows=(("waveform", 1), ("spectra", 2)),
        persisted_rows=None,
    )

    assert counts.product_rows == (("spectra", 2), ("waveform", 1))
    assert counts.persisted_rows is None
    assert DataQuality.CLEAN.value == "clean"
    assert DataQuality.PARTIAL.value == "partial"
    assert DataQuality.FAILED.value == "failed"


def test_compatibility_products_use_explicit_unknown_provenance():
    products = Products()
    row = HKSample(hk_type=0, version=0x307, unique_packet_id=1, errors=0)

    assert products.decode_provenance is not None
    assert products.decode_provenance.decoder_name is None
    assert products.decode_provenance.unavailable_reason == "caller_constructed"
    assert products.execution_mode is None
    assert products.quality_status is None
    assert products.validated_counts.persisted_rows is None
    assert row.provenance.unavailable_reason == "legacy_adapter_provenance_pending"


def test_metadata_only_nan_spectrum_is_not_a_usable_product():
    products = Products(spectra=[SpectrumSample(
        data=np.full((16, 2048), np.nan, dtype=np.float32),
        unique_packet_id=1,
        raw_seconds=0.0,
    )])

    assert _usable_product_count(products) == 0


def test_missing_mission_time_is_none_while_zero_is_valid():
    assert _meta_raw_seconds(SimpleNamespace()) is None
    assert _meta_raw_seconds(SimpleNamespace(_time_32=0, _time_16=0)) == 0.0
    assert _meta_raw_seconds(SimpleNamespace(time=np.nan)) is None
    assert _meta_raw_seconds(SimpleNamespace(time=np.inf)) is None
    assert _finite_seconds_or_none(-np.inf) is None
    assert _finite_seconds_or_none(True) is None


def test_legacy_waveform_keeps_adc_timestamp_separate_from_mission_time():
    waveform = WaveformSample(
        data=np.zeros(16384, dtype=np.int16),
        channel=0,
        unique_packet_id=1,
        raw_seconds=None,
        adc_timestamp=np.uint64(2**64 - 1),
    )

    assert waveform.raw_seconds is None
    assert waveform.adc_timestamp == np.uint64(2**64 - 1)


def test_waveform_adc_sentinel_requires_attached_metadata():
    sentinel = 2**64 - 1

    assert _waveform_adc_timestamp(SimpleNamespace(timestamp=sentinel)) is None
    assert _waveform_adc_timestamp(SimpleNamespace(
        timestamp=sentinel,
        meta=SimpleNamespace(),
    )) == np.uint64(sentinel)
    assert _waveform_adc_timestamp(SimpleNamespace(
        timestamp=-1,
        meta=SimpleNamespace(),
    )) is None
    assert _waveform_adc_timestamp(SimpleNamespace(
        timestamp=1.5,
        meta=SimpleNamespace(),
    )) is None
    assert _waveform_adc_timestamp(SimpleNamespace(
        timestamp="5",
        meta=SimpleNamespace(),
    )) is None
