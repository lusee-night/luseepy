"""Round-trip tests for the FITS writer.

The FITS layout mirrors the HDF5 schema; these tests build a small
``Products`` instance, write both formats, and check structural parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.decode import (
    HKSample,
    Products,
    SpectrumSample,
    TRSpectrumSample,
    WaveformSample,
    ZoomSample,
)
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.hdf5_writer import write_hdf5


def _make_products():
    products = Products(
        sw_version=0x307,
        fw_version=0x1234,
        fw_id=42,
        fw_date=20240101,
        fw_time=120000,
        start_unique_packet_id=100,
        start_time_32=0x10000,
        start_time_16=0,
        start_raw_seconds=1.0,
    )
    # Two spectra rows; the second is all-NaN so it should be retained as
    # filtered out (drop-NaN-row rule).
    spec0 = SpectrumSample(
        data=np.full((NPRODUCTS, NCHANNELS), 1.5, dtype=np.float32),
        unique_packet_id=101, raw_seconds=10.0,
        metadata={
            "Navg1_shift": 4,
            "adc_min": np.array([-1, -2, -3, -4], dtype=np.int16),
            "adc_mean": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        },
    )
    spec_nan = SpectrumSample(
        data=np.full((NPRODUCTS, NCHANNELS), np.nan, dtype=np.float32),
        unique_packet_id=102, raw_seconds=11.0,
        metadata={
            "Navg1_shift": 4,
            "adc_min": np.array([-5, -6, -7, -8], dtype=np.int16),
            "adc_mean": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
        },
    )
    products.spectra.extend([spec0, spec_nan])

    products.tr_spectra.append(TRSpectrumSample(
        data=np.full((NPRODUCTS, 4, 8), 2.5, dtype=np.float32),
        unique_packet_id=200, raw_seconds=20.0,
        navg2=4, tr_length=8,
        metadata={"Navg2_shift": 2, "tr_avg_shift": 1},
    ))

    products.zoom_spectra.append(ZoomSample(
        data=np.full((4, 64), 3.0, dtype=np.float32),
        unique_packet_id=300, pfb_index=42, raw_seconds=30.0,
    ))

    products.waveforms.extend([
        WaveformSample(
            data=np.arange(16384, dtype=np.int16) % 100,
            channel=0, unique_packet_id=400, raw_seconds=40.0,
        ),
        WaveformSample(
            data=np.arange(16384, dtype=np.int16) % 50,
            channel=2, unique_packet_id=401, raw_seconds=40.5,
        ),
    ])

    products.housekeeping.extend([
        HKSample(hk_type=1, version=0x307, unique_packet_id=500,
                 errors=0, raw_seconds=50.0,
                 fields={
                     "adc_min": np.array([-10, -20, -30, -40], dtype=np.int16),
                     "adc_max": np.array([10, 20, 30, 40], dtype=np.int16),
                     "adc_mean": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                     "adc_rms": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                     "actual_gain": np.array([b"L", b"M", b"H", b"L"], dtype="S1"),
                 }),
        HKSample(hk_type=2, version=0x307, unique_packet_id=501,
                 errors=0, raw_seconds=51.0,
                 fields={"time": 51.0, "ok": True,
                         "telemetry_T_FPGA": 42.0}),
        HKSample(hk_type=3, version=0x307, unique_packet_id=502,
                 errors=0, raw_seconds=52.0,
                 fields={"checksum": 0xDEADBEEF, "weight_ndx": 7}),
    ])

    fpga = {
        "mission_seconds": np.array([100.0, 200.0, 300.0], dtype=np.float64),
        "lusee_subsecs": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        "THERM_FPGA": np.array([42.0, 43.0, 44.0], dtype=np.float64),
        "VMON_6V": np.array([6.05, 6.04, 6.03], dtype=np.float64),
    }
    encoder = {
        "mission_seconds": np.array([110.0, 210.0], dtype=np.float64),
        "lusee_subsecs": np.array([0.0, 0.0], dtype=np.float64),
        "enc_pos": np.array([1234, 5678], dtype=np.int64),
        "enc_status": np.array([0, 1], dtype=np.int64),
    }
    return products, fpga, encoder


def test_fits_basic_roundtrip(tmp_path: Path):
    products, fpga, encoder = _make_products()
    fits_path = tmp_path / "out.fits"
    write_fits(products, fits_path,
               cdi_directory=tmp_path / "session_x",
               fpga_telemetry=fpga, encoder_telemetry=encoder)
    with fits.open(fits_path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert names[0] == "PRIMARY"
        for required in (
            "SESSION_INV", "CONSTANTS",
            "SPECTRA", "SPECTRA_TIMES", "SPECTRA_META",
            "TR_SPECTRA", "TR_TIMES", "TR_META",
            "ZOOM_DATA", "ZOOM_TIMES",
            "WF_CH0", "WF_CH2",
            "HK_T1", "HK_T2", "HK_T3",
            "DCB_FPGA", "DCB_ENC",
        ):
            assert required in names, f"missing HDU: {required}"

        # Session invariants should land as header keywords.
        si = hdul["SESSION_INV"]
        assert si.data is None    # NAXIS=0
        assert si.header["SW_VERS"] == 0x307
        assert si.header["FW_VERS"] == 0x1234
        assert si.header["ST_T32"] == 0x10000

        # Constants header.
        c = hdul["CONSTANTS"]
        assert c.data is None
        assert c.header["MJDOFF"] == 0.0

        # Spectra cube: NaN-row dropped, shape (1, 16, NCHANNELS).
        sp = hdul["SPECTRA"]
        assert sp.data.shape == (1, NPRODUCTS, NCHANNELS)
        # FITS stores big-endian by convention; check kind+itemsize, not endianness.
        assert sp.data.dtype.kind == "f" and sp.data.dtype.itemsize == 4
        np.testing.assert_allclose(sp.data[0, 0, 0], 1.5)

        # SPECTRA_TIMES has the documented columns.
        st = hdul["SPECTRA_TIMES"].data
        assert st["UNIQUE_ID"].tolist() == [101]
        assert st["RAW_TIME"][0] == 10.0
        assert st["ORIG_IDX"][0] == 0     # original index of the kept row

        # SPECTRA_META: per-row metadata as columns
        sm = hdul["SPECTRA_META"].data
        assert "Navg1_shift" in sm.dtype.names
        assert sm["adc_min"].shape == (1, 4)

        # TR cube
        tr = hdul["TR_SPECTRA"]
        assert tr.data.shape == (1, NPRODUCTS, 4, 8)
        assert tr.data.dtype.kind == "f" and tr.data.dtype.itemsize == 4

        # Zoom
        zm = hdul["ZOOM_DATA"]
        assert zm.data.shape == (1, 4, 64)

        # Waveforms (per channel, BINTABLE with WAVEFORM column)
        wf0 = hdul["WF_CH0"].data
        assert wf0.shape == (1,)
        assert wf0["WAVEFORM"].shape == (1, 16384)
        assert wf0["WAVEFORM"].dtype.kind == "i" and wf0["WAVEFORM"].dtype.itemsize == 2
        assert wf0["TIMESTAMP"][0] == 40.0
        assert hdul["WF_CH0"].header["CHANNEL"] == 0

        # Housekeeping per type
        t1 = hdul["HK_T1"].data
        assert t1.shape == (1,)
        assert t1["UPID"][0] == 500
        assert t1["ADC_MIN"].shape == (1, 4)
        np.testing.assert_array_equal(
            t1["ADC_MIN"][0], np.array([-10, -20, -30, -40], dtype=np.int16)
        )
        # actual_gain joined to a 4-char ASCII string per row.
        # astropy returns FITS A-format columns as Python str.
        gain = t1["ACT_GAIN"][0]
        if isinstance(gain, bytes):
            gain = gain.decode("ascii")
        assert gain == "LMHL"

        t2 = hdul["HK_T2"].data
        assert t2["TIME"][0] == 51.0
        assert t2["OK"][0] == 1
        assert "telemetry_T_FPGA" in t2.dtype.names
        assert abs(t2["telemetry_T_FPGA"][0] - 42.0) < 1e-5

        t3 = hdul["HK_T3"].data
        assert int(t3["CHECKSUM"][0]) == 0xDEADBEEF
        assert int(t3["WGT_NDX"][0]) == 7

        # FPGA telemetry: BINTABLE with one column per channel
        fp = hdul["DCB_FPGA"].data
        assert fp.shape == (3,)
        assert "THERM_FPGA" in fp.dtype.names
        assert "VMON_6V" in fp.dtype.names
        np.testing.assert_allclose(fp["THERM_FPGA"], [42.0, 43.0, 44.0])
        np.testing.assert_allclose(fp["MS"], [100.0, 200.0, 300.0])

        # Encoder
        enc = hdul["DCB_ENC"].data
        assert enc.shape == (2,)
        assert enc["ENC_POS"].tolist() == [1234, 5678]


def test_fits_matches_hdf5_row_counts(tmp_path: Path):
    """The drop-NaN-row retention rule must produce identical N in both formats."""
    import h5py

    products, fpga, encoder = _make_products()
    h5_path = tmp_path / "out.h5"
    fits_path = tmp_path / "out.fits"
    write_hdf5(products, h5_path, fpga_telemetry=fpga, encoder_telemetry=encoder)
    write_fits(products, fits_path, fpga_telemetry=fpga, encoder_telemetry=encoder)

    with h5py.File(h5_path, "r") as hf, fits.open(fits_path) as ff:
        # /spectra/data and SPECTRA must agree in (N, 16, NCHANNELS).
        h5_n = hf["spectra"]["data"].shape[0]
        fits_n = ff["SPECTRA"].data.shape[0]
        assert h5_n == fits_n == 1

        # /tr_spectra and TR_SPECTRA agree.
        h5_tr_n = hf["tr_spectra"]["data"].shape[0]
        fits_tr_n = ff["TR_SPECTRA"].data.shape[0]
        assert h5_tr_n == fits_tr_n == 1

        # Housekeeping types
        for t in (1, 2, 3):
            hk_h5_n = int(hf[f"housekeeping/type_{t}"].attrs["count"])
            hk_fits_n = ff[f"HK_T{t}"].data.shape[0]
            assert hk_h5_n == hk_fits_n


def test_fits_no_telemetry_no_dcb_hdus(tmp_path: Path):
    products, _, _ = _make_products()
    fits_path = tmp_path / "out.fits"
    write_fits(products, fits_path)    # no telemetry kwargs
    with fits.open(fits_path) as hdul:
        names = [hdu.name for hdu in hdul]
        assert "DCB_FPGA" not in names
        assert "DCB_ENC" not in names
