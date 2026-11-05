"""Strict in-memory contracts for normal and time-resolved spectra."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from lusee.ingest.constants import (
    BITSLICE_REFERENCE,
    NPRODUCTS,
    SPECTRA_NORMALIZATION_VERSION,
    SPECTRA_REPRESENTATION,
    SPECTRA_UNITS,
)
from lusee.ingest.frequency_contract import spectrometer_frequency_window
from lusee.ingest.products import (
    ProductProvenance,
    SourcePacketProvenance,
    SpectrumMetadata,
    SpectrumSample,
    TRSpectrumSample,
)

UID = 0x12345678
RAW_SECONDS = 10.0


def test_public_spectrum_exports_are_strict_records():
    import lusee.ingest as ingest

    assert ingest.SpectrumMetadata is SpectrumMetadata
    assert ingest.SpectrumSample is SpectrumSample
    assert ingest.TRSpectrumSample is TRSpectrumSample


def metadata_values(**changes):
    valid = np.array([4, 0, 2, 1], dtype=np.int64)
    invalid_max = np.array([1, 2, 0, 0], dtype=np.int64)
    invalid_min = np.array([0, 3, 1, 0], dtype=np.int64)
    values = {
        "version": 0x307,
        "unique_packet_id": UID,
        "uc_time": 123456789,
        "time_32": int(RAW_SECONDS * 65536),
        "time_16": 0,
        "tvs_sensors": np.array([16000, 28800, 40000, 39000], dtype=np.uint16),
        "requested_gain": np.array([0, 1, 2, 3], dtype=np.uint8),
        "gain_auto_min": np.array([1, 2, 3, 4], dtype=np.uint16),
        "gain_auto_mult": np.array([4, 3, 2, 1], dtype=np.uint16),
        "route_plus": np.array([0, 1, 2, 3], dtype=np.uint8),
        "route_minus": np.array([3, 2, 1, 0], dtype=np.uint8),
        "navg1_shift": 2,
        "navg2_shift": 2,
        "notch": 1,
        "navgf": 2,
        "high_fraction": 7,
        "medium_fraction": 5,
        "requested_bitslice": np.full(
            NPRODUCTS, BITSLICE_REFERENCE - 1, dtype=np.uint8
        ),
        "bitslice_keep_bits": 8,
        "output_format": 0,
        "reject_ratio": 3,
        "reject_max_bad": 4,
        "tr_start": 2,
        "tr_stop": 10,
        "tr_average_shift": 1,
        "errors": 0,
        "correlation_products_mask": 0xFFFF,
        "actual_gain": np.array([1, 2, 1, 2], dtype=np.uint8),
        "actual_bitslice": np.full(
            NPRODUCTS, BITSLICE_REFERENCE, dtype=np.uint8
        ),
        "spectrum_overflow": 2,
        "notch_overflow": 1,
        "adc_min": np.array([-5, -4, -3, -2], dtype=np.int64),
        "adc_max": np.array([5, 4, 3, 2], dtype=np.int64),
        "adc_valid_count": valid,
        "adc_invalid_count_max": invalid_max,
        "adc_invalid_count_min": invalid_min,
        "adc_total_count": valid + invalid_max + invalid_min,
        "adc_mean": np.array([0.5, 0.0, -0.5, 1.0], dtype=np.float64),
        "adc_rms": np.array([1.5, 0.0, 2.5, 3.5], dtype=np.float64),
        "spectrometer_enable": True,
        "calibrator_enable": False,
        "random_state": 0xAABBCCDD,
        "weight": 8,
        "weight_current": 9,
        "telemetry_v1_0": 1.0,
        "telemetry_v1_8": 1.8,
        "telemetry_v2_5": 2.5,
        "telemetry_t_fpga": 30.25,
        "loop_count_min": 10,
        "loop_count_max": 20,
        "grimm_enable": 1,
        "averaging_mode": 2,
        "num_bad_min_current": 1,
        "num_bad_max_current": 2,
        "num_bad_min": 3,
        "num_bad_max": 4,
    }
    values.update(changes)
    return values


def make_metadata(**changes) -> SpectrumMetadata:
    return SpectrumMetadata(**metadata_values(**changes))


def make_provenance(
    *,
    uid: int = UID,
    raw_seconds: float | None = RAW_SECONDS,
    selected_schema_id: int | None = 0x307,
    clock_source: str = "spectrometer",
) -> ProductProvenance:
    return ProductProvenance(
        source_packets=(
            SourcePacketProvenance(
                role="metadata",
                original_appid=0x20F,
                packet_index=10,
            ),
            SourcePacketProvenance(
                role="normal_or_tr_product_00",
                original_appid=0x210,
                packet_index=11,
            ),
        ),
        uid=uid,
        uid_source="metadata.unique_packet_id",
        uid_source_role="metadata",
        reported_schema_ids=(0x307,),
        selected_schema_id=selected_schema_id,
        time_source=(
            "metadata.base.time_32_time_16"
            if raw_seconds is not None
            else None
        ),
        time_source_role="metadata" if raw_seconds is not None else None,
        clock_source=clock_source,
        time_valid=raw_seconds is not None,
    )


def normal_values(*, navgf: int = 2, product: int = 4):
    contract = spectrometer_frequency_window(navgf)
    data = np.full(
        (NPRODUCTS, contract.output_count), np.nan, dtype=np.float32
    )
    data[product] = np.linspace(
        -4.0, 4.0, contract.output_count, dtype=np.float32
    )
    present = np.zeros(NPRODUCTS, dtype=np.bool_)
    present[product] = True
    metadata = make_metadata(navgf=navgf)
    return {
        "data": data,
        "product_present": present,
        "navgf": navgf,
        "frequency_contract": contract,
        "unique_packet_id": UID,
        "raw_seconds": RAW_SECONDS,
        "metadata": metadata,
        "provenance": make_provenance(),
    }


def tr_values(*, product: int = 3):
    metadata = make_metadata()
    navg2 = 1 << metadata.navg2_shift
    tr_length = (
        (metadata.tr_stop - metadata.tr_start)
        // (1 << metadata.tr_average_shift)
    )
    data = np.zeros((NPRODUCTS, navg2, tr_length), dtype=np.int32)
    data[product] = np.arange(navg2 * tr_length, dtype=np.int32).reshape(
        navg2, tr_length
    ) - 8
    present = np.zeros(NPRODUCTS, dtype=np.bool_)
    present[product] = True
    return {
        "data": data,
        "product_present": present,
        "unique_packet_id": UID,
        "raw_seconds": RAW_SECONDS,
        "navg2": navg2,
        "tr_length": tr_length,
        "metadata": metadata,
        "provenance": make_provenance(),
    }


def test_metadata_is_explicit_and_deeply_immutable():
    values = metadata_values()
    metadata = SpectrumMetadata(**values)

    assert metadata.raw_seconds == RAW_SECONDS
    assert metadata.current_fields_present is True
    np.testing.assert_array_equal(
        metadata.adc_statistics_valid,
        np.array([True, False, True, True]),
    )
    assert metadata.spectrum_overflow == 2
    assert metadata.notch_overflow == 1
    assert metadata.route_plus.tolist() == [0, 1, 2, 3]
    metadata_mapping = metadata.as_mapping()
    assert metadata_mapping["unique_packet_id"] == UID
    assert dict(metadata.items())["navgf"] == 2
    assert metadata_mapping["actual_gain"] is metadata.actual_gain

    values["actual_gain"][0] = 99
    assert metadata.actual_gain[0] == 1
    with pytest.raises(FrozenInstanceError):
        metadata.weight = 4
    with pytest.raises(ValueError, match="read-only"):
        metadata.actual_gain[0] = 4
    with pytest.raises(ValueError):
        metadata.actual_gain.setflags(write=True)
    with pytest.raises(ValueError):
        metadata.adc_statistics_valid.setflags(write=True)
    with pytest.raises(TypeError):
        metadata_mapping["weight"] = 4


def test_metadata_accepts_requested_auto_bitslice():
    requested = np.full(NPRODUCTS, BITSLICE_REFERENCE, dtype=np.uint8)
    requested[0] = 0xFF

    metadata = make_metadata(requested_bitslice=requested)

    assert metadata.requested_bitslice[0] == 0xFF


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (
            {"requested_gain": np.zeros(4, dtype=np.int16)},
            TypeError,
            "dtype uint8",
        ),
        (
            {"route_plus": np.zeros(3, dtype=np.uint8)},
            ValueError,
            "shape",
        ),
        (
            {
                "actual_bitslice": np.full(
                    NPRODUCTS, BITSLICE_REFERENCE + 1, dtype=np.uint8
                )
            },
            ValueError,
            "actual_bitslice",
        ),
        (
            {
                "requested_bitslice": np.full(
                    NPRODUCTS, BITSLICE_REFERENCE + 1, dtype=np.uint8
                )
            },
            ValueError,
            "requested_bitslice",
        ),
        (
            {"adc_total_count": np.zeros(4, dtype=np.int64)},
            ValueError,
            "adc_total_count",
        ),
        ({"loop_count_min": None}, ValueError, "all present or all absent"),
        ({"time_16": None}, ValueError, "present together"),
        ({"navgf": 5}, ValueError, "navgf"),
        ({"telemetry_v1_0": np.inf}, ValueError, "finite"),
    ],
)
def test_metadata_rejects_coercion_and_inconsistent_fields(
    change, error, message
):
    with pytest.raises(error, match=message):
        make_metadata(**change)


def test_historical_optional_metadata_group_is_explicitly_absent():
    absent = {
        "loop_count_min": None,
        "loop_count_max": None,
        "grimm_enable": None,
        "averaging_mode": None,
        "num_bad_min_current": None,
        "num_bad_max_current": None,
        "num_bad_min": None,
        "num_bad_max": None,
    }
    metadata = make_metadata(version=0x203, **absent)

    assert metadata.current_fields_present is False
    assert metadata.loop_count_min is None


def test_normal_record_keeps_native_frequency_count_and_is_read_only():
    values = normal_values(navgf=2)
    original_data = values["data"]
    original_mask = values["product_present"]
    sample = SpectrumSample(**values)

    assert sample.data.shape == (NPRODUCTS, 1024)
    assert sample.nfreq == 1024
    assert sample.data.dtype == np.float32
    assert sample.product_present.dtype == np.bool_
    assert sample.product_present.tolist() == [False] * 4 + [True] + [False] * 11
    assert sample.units == SPECTRA_UNITS
    assert sample.representation == SPECTRA_REPRESENTATION
    assert sample.normalization_version == SPECTRA_NORMALIZATION_VERSION
    assert sample.bitslice_reference == BITSLICE_REFERENCE
    assert not hasattr(sample, "bitslice_restored")
    restored_data = sample.data
    sample.restore_bitslice()
    # A repeated call remains an invariant check, not a state transition
    sample.restore_bitslice()
    assert sample.data is restored_data

    original_data[4, 0] = 12345.0
    original_mask[4] = False
    assert sample.data[4, 0] == -4.0
    assert sample.product_present[4]
    with pytest.raises(ValueError, match="read-only"):
        sample.data[4, 0] = 0.0
    with pytest.raises(ValueError):
        sample.data.setflags(write=True)
    with pytest.raises(ValueError):
        sample.product_present.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        sample.raw_seconds = 11.0


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda values: values.update(data=values["data"].astype(np.float64)),
            TypeError,
            "dtype float32",
        ),
        (
            lambda values: values.update(
                data=np.full((NPRODUCTS, 2048), np.nan, dtype=np.float32)
            ),
            ValueError,
            "shape",
        ),
        (
            lambda values: values.update(
                product_present=values["product_present"].astype(np.uint8)
            ),
            TypeError,
            "dtype bool",
        ),
        (
            lambda values: values["data"].__setitem__((0, 0), 0.0),
            ValueError,
            "absent normal-spectrum",
        ),
        (
            lambda values: values["data"].__setitem__((4, 0), np.nan),
            ValueError,
            "present normal-spectrum",
        ),
        (
            lambda values: values["product_present"].fill(False),
            ValueError,
            "at least one",
        ),
    ],
)
def test_normal_record_rejects_cast_reshape_padding_and_mask_fabrication(
    mutate, error, message
):
    values = normal_values(navgf=2)
    mutate(values)
    with pytest.raises(error, match=message):
        SpectrumSample(**values)


def test_normal_record_rejects_contract_metadata_and_identity_mismatches():
    values = normal_values(navgf=2)
    values["frequency_contract"] = spectrometer_frequency_window(1)
    with pytest.raises(ValueError, match="navgf disagrees"):
        SpectrumSample(**values)

    values = normal_values(navgf=2)
    values["metadata"] = make_metadata(navgf=1)
    with pytest.raises(ValueError, match="metadata navgf"):
        SpectrumSample(**values)

    values = normal_values(navgf=2)
    values["unique_packet_id"] = UID + 1
    with pytest.raises(ValueError, match="metadata UID"):
        SpectrumSample(**values)

    values = normal_values(navgf=2)
    values["provenance"] = ProductProvenance.unavailable("test")
    with pytest.raises(ValueError, match="concrete provenance"):
        SpectrumSample(**values)

    values = normal_values(navgf=2)
    values["provenance"] = make_provenance(selected_schema_id=None)
    with pytest.raises(ValueError, match="selected schema"):
        SpectrumSample(**values)

    values = normal_values(navgf=2)
    values["provenance"] = make_provenance(clock_source="dcb")
    with pytest.raises(ValueError, match="spectrometer clock"):
        SpectrumSample(**values)


def test_normal_record_allows_explicitly_missing_time_only_when_consistent():
    values = normal_values(navgf=2)
    values["metadata"] = make_metadata(time_32=None, time_16=None)
    values["raw_seconds"] = None
    values["provenance"] = make_provenance(raw_seconds=None)
    sample = SpectrumSample(**values)

    assert sample.raw_seconds is None
    assert sample.provenance.time_valid is False

    values = normal_values(navgf=2)
    values["raw_seconds"] = None
    values["provenance"] = make_provenance(raw_seconds=None)
    with pytest.raises(ValueError, match="split time"):
        SpectrumSample(**values)


def test_tr_record_preserves_native_int32_geometry_and_mask():
    values = tr_values()
    original_data = values["data"]
    sample = TRSpectrumSample(**values)

    assert sample.data.shape == (NPRODUCTS, 4, 4)
    assert sample.data.dtype == np.int32
    assert sample.product_present.dtype == np.bool_
    assert sample.navg2 == 4
    assert sample.tr_length == 4
    assert sample.representation == "native_int32"
    assert sample.units == "unit_unestablished"
    assert not hasattr(sample, "normalization_version")
    assert not hasattr(sample, "restore_bitslice")
    assert sample.data[3, 0, 0] == -8

    original_data[3, 0, 0] = 999
    assert sample.data[3, 0, 0] == -8
    with pytest.raises(ValueError, match="read-only"):
        sample.data[3, 0, 0] = 1
    with pytest.raises(ValueError):
        sample.data.setflags(write=True)


def test_tr_mask_is_authoritative_when_a_present_product_is_all_zero():
    values = tr_values()
    values["data"][3] = 0
    sample = TRSpectrumSample(**values)

    assert sample.product_present[3]
    assert np.all(sample.data[3] == 0)


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda values: values.update(data=values["data"].astype(np.int64)),
            TypeError,
            "dtype int32",
        ),
        (
            lambda values: values.update(data=values["data"].reshape(-1)),
            ValueError,
            "shape",
        ),
        (
            lambda values: values.update(
                product_present=values["product_present"].astype(np.uint8)
            ),
            TypeError,
            "dtype bool",
        ),
        (
            lambda values: values["data"].__setitem__((0, 0, 0), 7),
            ValueError,
            "absent TR",
        ),
        (
            lambda values: values["product_present"].fill(False),
            ValueError,
            "at least one",
        ),
        (
            lambda values: values.update(navg2=2),
            ValueError,
            "navg2 disagrees",
        ),
        (
            lambda values: values.update(tr_length=3),
            ValueError,
            "tr_length disagrees",
        ),
    ],
)
def test_tr_record_rejects_cast_reshape_padding_and_geometry_changes(
    mutate, error, message
):
    values = tr_values()
    mutate(values)
    with pytest.raises(error, match=message):
        TRSpectrumSample(**values)


def test_tr_record_rejects_nondivisible_metadata_geometry():
    values = tr_values()
    values["metadata"] = make_metadata(tr_stop=9)

    with pytest.raises(ValueError, match="divisible"):
        TRSpectrumSample(**values)
