"""Unit tests for the per-hk_type HDF5 housekeeping layout."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from lusee.ingest.decode import HKSample, Products
from lusee.ingest.hdf5_writer import write_hdf5


def _hk(hk_type: int, **kwargs):
    fields = kwargs.pop("fields", {})
    return HKSample(
        hk_type=hk_type,
        version=kwargs.pop("version", 0x307),
        unique_packet_id=kwargs.pop("unique_packet_id", 0),
        errors=kwargs.pop("errors", 0),
        raw_seconds=kwargs.pop("raw_seconds", 0.0),
        fields=fields,
    )


def test_per_type_tables(tmp_path: Path):
    products = Products()
    # Two type_1 records with deliberately out-of-order raw_seconds.
    products.housekeeping.append(_hk(
        1, unique_packet_id=11, errors=1, raw_seconds=200.0,
        fields={
            "adc_min": np.array([-100, -110, -120, -130], dtype=np.int16),
            "adc_max": np.array([100, 110, 120, 130], dtype=np.int16),
            "adc_mean": np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32),
            "adc_rms": np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32),
            "actual_gain": np.array([b"L", b"L", b"M", b"H"], dtype="S1"),
        },
    ))
    products.housekeeping.append(_hk(
        1, unique_packet_id=10, errors=0, raw_seconds=100.0,
        fields={
            "adc_min": np.array([-50, -60, -70, -80], dtype=np.int16),
            "adc_max": np.array([50, 60, 70, 80], dtype=np.int16),
            "adc_mean": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            "adc_rms": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            "actual_gain": np.array([b"L", b"L", b"L", b"L"], dtype="S1"),
        },
    ))
    products.housekeeping.append(_hk(
        2, unique_packet_id=20, raw_seconds=300.0,
        fields={"time": 300.0, "ok": True,
                "telemetry_T_FPGA": 42.0, "telemetry_V1_8": 1.83},
    ))
    products.housekeeping.append(_hk(
        2, unique_packet_id=21, raw_seconds=310.0,
        fields={"time": 310.0, "ok": False,
                "telemetry_V1_8": 1.84, "telemetry_V2_5": 2.51},
    ))
    products.housekeeping.append(_hk(
        3, unique_packet_id=30, raw_seconds=400.0,
        fields={"checksum": 0xABCD1234, "weight_ndx": 7},
    ))

    h5_path = tmp_path / "out.h5"
    write_hdf5(products, h5_path)

    with h5py.File(h5_path, "r") as f:
        hk = f["housekeeping"]
        assert int(hk.attrs["count"]) == 5

        # type_1: sorted by raw_seconds ascending
        t1 = hk["type_1"]
        assert int(t1.attrs["count"]) == 2
        assert list(t1["unique_packet_id"][...]) == [10, 11]
        assert t1["raw_seconds"][...].tolist() == [100.0, 200.0]
        assert t1["adc_min"].dtype == np.int16
        assert t1["adc_mean"].dtype == np.float32
        assert t1["actual_gain"].dtype == np.dtype("S1")
        assert t1["adc_min"].shape == (2, 4)
        np.testing.assert_array_equal(
            t1["adc_min"][0], np.array([-50, -60, -70, -80], dtype=np.int16)
        )

        # type_2: per-channel telemetry_* columns + NaN where missing
        t2 = hk["type_2"]
        assert int(t2.attrs["count"]) == 2
        # The first row had T_FPGA + V1_8; the second had V1_8 + V2_5.
        assert "telemetry_T_FPGA" in t2
        assert "telemetry_V1_8" in t2
        assert "telemetry_V2_5" in t2
        assert t2["telemetry_T_FPGA"].dtype == np.float32
        assert np.isfinite(t2["telemetry_T_FPGA"][0])    # row 0 had it
        assert np.isnan(t2["telemetry_T_FPGA"][1])       # row 1 didn't
        assert np.isnan(t2["telemetry_V2_5"][0])         # row 0 didn't have it
        assert np.isfinite(t2["telemetry_V2_5"][1])      # row 1 did
        assert t2["ok"].dtype == np.uint8
        np.testing.assert_array_equal(t2["ok"][...], [1, 0])

        # type_3: simple scalars
        t3 = hk["type_3"]
        assert int(t3.attrs["count"]) == 1
        assert int(t3["checksum"][0]) == 0xABCD1234
        assert int(t3["weight_ndx"][0]) == 7

        # No legacy /housekeeping/packet_<i> groups should exist.
        for k in hk:
            assert k.startswith("type_"), f"unexpected child: /housekeeping/{k}"


def test_no_housekeeping_no_group(tmp_path: Path):
    products = Products()
    h5_path = tmp_path / "empty.h5"
    write_hdf5(products, h5_path)
    with h5py.File(h5_path, "r") as f:
        assert "housekeeping" not in f
