"""Time-provenance attrs: writer round-trip and the reader's scale ladder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.decode import (
    HKSample,
    Products,
    SpectrumSample,
    WaveformSample,
)
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.hdf5_writer import _interpolate_telemetry, write_hdf5
from lusee.ingest.obs_factory import IngestData


def _minimal_products(nrows: int = 3, offset: int = 0) -> Products:
    products = Products()
    for row in range(nrows):
        products.spectra.append(SpectrumSample(
            data=np.full((NPRODUCTS, NCHANNELS), 4.0, dtype=np.float32),
            unique_packet_id=100 + offset + row,
            raw_seconds=1.75e9 + float(offset + row),
            metadata={
                "actual_bitslice": np.full(NPRODUCTS, 31, dtype=np.int16),
                "bitslice": np.full(NPRODUCTS, 31, dtype=np.int16),
                "actual_gain": np.full(4, 1, dtype=np.int16),
                "Navgf": 1,
            },
        ))
    return products


def test_hdf5_writer_records_provenance(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(
        _minimal_products(), dest,
        time_scale="utc",
        clock_source="mission_counter",
        clock_epoch_isot="2025-09-01T00:00:00",
    )
    with h5py.File(dest, "r") as f:
        c = f["constants"].attrs
        assert c["time_scale"] == "utc"
        assert c["clock_source"] == "mission_counter"
        assert c["clock_epoch_isot"] == "2025-09-01T00:00:00"


def test_hdf5_writer_default_is_unknown(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest)
    with h5py.File(dest, "r") as f:
        c = f["constants"].attrs
        assert c["time_scale"] == "unknown"
        assert c["clock_source"] == "unknown"
        assert "clock_epoch_isot" not in c


def test_hdf5_writer_rejects_bad_scale(tmp_path: Path):
    with pytest.raises(ValueError, match="time_scale"):
        write_hdf5(_minimal_products(), tmp_path / "s.h5", time_scale="GPS")


def test_fits_writer_records_provenance(tmp_path: Path):
    from astropy.io import fits
    dest = tmp_path / "s.fits"
    write_fits(
        _minimal_products(), dest,
        time_scale="tai",
        clock_source="mission_counter",
    )
    with fits.open(dest) as hdul:
        hdr = hdul["CONSTANTS"].header
        assert hdr["TIMESYS"] == "TAI"
        assert hdr["CLKSRC"] == "mission_counter"
        assert "CLKEPOCH" not in hdr


def test_fits_reader_round_trips_provenance(tmp_path: Path):
    dest = tmp_path / "s.fits"
    write_fits(_minimal_products(), dest, time_scale="tai",
               clock_source="mission_counter")
    data = IngestData(dest, prefer_format="fits")
    assert str(data.times.scale) == "tai"
    assert data.time_provenance == {
        "scale": "tai", "source": "mission_counter", "assumed": False,
    }


def test_reader_uses_recorded_scale(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest, time_scale="utc",
               clock_source="mission_counter")
    data = IngestData(dest)
    assert str(data.times.scale) == "utc"
    assert data.time_provenance == {
        "scale": "utc", "source": "mission_counter", "assumed": False,
    }


def test_reader_refuses_to_guess(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest)
    with pytest.raises(ValueError, match="assume_scale"):
        IngestData(dest)


def test_reader_accepts_assume_scale(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest)
    data = IngestData(dest, assume_scale="utc")
    assert str(data.times.scale) == "utc"
    assert data.time_provenance["assumed"] is True


def test_reader_rejects_contradicting_assume_scale(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest, time_scale="utc")
    with pytest.raises(ValueError, match="contradicts"):
        IngestData(dest, assume_scale="tai")


def test_reader_scale_on_calibrated_mjd(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest, time_scale="tai",
               raw_time_subtract_seconds=1.75e9,
               mjd_epoch_offset_days=60950.0)
    data = IngestData(dest)
    assert str(data.times.scale) == "tai"
    assert data.time_provenance["assumed"] is False


def test_reader_rejects_unknown_and_normalizes_case(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest)
    with pytest.raises(ValueError, match="unknown"):
        IngestData(dest, assume_scale="unknown")
    data = IngestData(dest, assume_scale="UTC")
    assert str(data.times.scale) == "utc"


def test_concat_contradicting_scales_raises(tmp_path: Path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    write_hdf5(_minimal_products(), a, time_scale="utc")
    write_hdf5(_minimal_products(offset=10), b, time_scale="tai")
    with pytest.raises(ValueError, match="contradicting time scales"):
        IngestData([a, b])


def test_concat_with_scaleless_file_degrades_to_unknown(tmp_path: Path):
    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    write_hdf5(_minimal_products(), a, time_scale="utc")
    write_hdf5(_minimal_products(offset=10), b)
    with pytest.raises(ValueError, match="assume_scale"):
        IngestData([a, b])
    data = IngestData([a, b], assume_scale="utc")
    assert data.time_provenance["assumed"] is True
    # the assumption must still agree with the scale file a recorded
    with pytest.raises(ValueError, match="contradicts"):
        IngestData([a, b], assume_scale="tai")


def test_mission_epoch_contradicting_file_scale_raises(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest, time_scale="tai")
    with pytest.raises(ValueError, match="mission_epoch"):
        # an ISO-string epoch defaults to the utc scale
        IngestData(dest, mission_epoch="2025-01-01T00:00:00")


def test_mission_epoch_scale_is_not_marked_assumed(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest)
    data = IngestData(dest, mission_epoch="2025-01-01T00:00:00")
    assert data.time_provenance == {
        "scale": "utc", "source": "unknown", "assumed": False,
    }


def test_legacy_file_without_attrs_requires_assume_scale(tmp_path: Path):
    dest = tmp_path / "s.h5"
    write_hdf5(_minimal_products(), dest, time_scale="utc")
    # simulate a pre-provenance file
    with h5py.File(dest, "a") as f:
        del f["constants"].attrs["time_scale"]
        del f["constants"].attrs["clock_source"]
    with pytest.raises(ValueError, match="assume_scale"):
        IngestData(dest)
    data = IngestData(dest, assume_scale="utc")
    assert str(data.times.scale) == "utc"
    assert data.time_provenance == {
        "scale": "utc", "source": "unknown", "assumed": True,
    }


def test_missing_housekeeping_time_is_nan_and_sorts_last(tmp_path: Path):
    products = Products(housekeeping=[
        HKSample(
            hk_type=3,
            version=0x307,
            unique_packet_id=2,
            errors=0,
            raw_seconds=None,
        ),
        HKSample(
            hk_type=3,
            version=0x307,
            unique_packet_id=1,
            errors=0,
            raw_seconds=5.0,
        ),
    ])
    h5_path = tmp_path / "hk.h5"
    fits_path = tmp_path / "hk.fits"

    write_hdf5(products, h5_path)
    write_fits(products, fits_path)

    with h5py.File(h5_path, "r") as handle:
        np.testing.assert_array_equal(
            handle["housekeeping/type_3/unique_packet_id"][...],
            [1, 2],
        )
        raw = handle["housekeeping/type_3/raw_seconds"][...]
        assert raw[0] == 5.0 and np.isnan(raw[1])
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        np.testing.assert_array_equal(hdul["HK_T3"].data["UPID"], [1, 2])
        raw = hdul["HK_T3"].data["RAW_TIME"]
        assert raw[0] == 5.0 and np.isnan(raw[1])


def test_missing_spectrum_time_does_not_poison_telemetry_interpolation():
    result = _interpolate_telemetry(
        {
            "mission_seconds": np.array([0.0, 10.0]),
            "lusee_subsecs": np.zeros(2),
            "temperature": np.array([1.0, 3.0]),
        },
        np.array([0.0, np.nan, 10.0]),
    )

    np.testing.assert_allclose(
        result["temperature"],
        np.array([1.0, np.nan, 3.0]),
        equal_nan=True,
    )


def test_waveform_adc_timestamp_is_persisted_separately(tmp_path: Path):
    products = Products(waveforms=[
        WaveformSample(
            data=np.zeros(16384, dtype=np.int16),
            channel=0,
            unique_packet_id=1,
            raw_seconds=None,
            adc_timestamp=np.uint64(2**64 - 1),
        ),
        WaveformSample(
            data=np.ones(16384, dtype=np.int16),
            channel=0,
            unique_packet_id=2,
            raw_seconds=5.0,
            adc_timestamp=None,
        ),
    ])
    h5_path = tmp_path / "waveform.h5"
    fits_path = tmp_path / "waveform.fits"

    write_hdf5(products, h5_path)
    write_fits(products, fits_path)

    with h5py.File(h5_path, "r") as handle:
        group = handle["waveform/channel_0"]
        np.testing.assert_array_equal(
            group["adc_timestamps"][...],
            np.array([2**64 - 1, 0], dtype=np.uint64),
        )
        np.testing.assert_array_equal(
            group["adc_timestamp_valid"][...],
            [True, False],
        )
        assert np.isnan(group["timestamps"][0])
        assert group["timestamps"][1] == 5.0

    from astropy.io import fits

    with fits.open(fits_path, uint=True) as hdul:
        table = hdul["WF_CH0"].data
        np.testing.assert_array_equal(
            table["ADC_TIME"],
            np.array([2**64 - 1, 0], dtype=np.uint64),
        )
        np.testing.assert_array_equal(table["ADC_VALID"], [1, 0])
        assert np.isnan(table["TIMESTAMP"][0])
        assert table["TIMESTAMP"][1] == 5.0


def test_waveform_uint64_timestamp_roundtrips_fitsio(tmp_path: Path):
    fitsio = pytest.importorskip("fitsio")
    products = Products(waveforms=[WaveformSample(
        data=np.zeros(16384, dtype=np.int16),
        channel=0,
        unique_packet_id=1,
        raw_seconds=None,
        adc_timestamp=np.uint64(2**64 - 1),
    )])
    path = tmp_path / "waveform-fitsio.fits"

    write_fits(products, path)

    with fitsio.FITS(path) as handle:
        table = handle["WF_CH0"].read()
    assert table["ADC_TIME"].dtype.kind == "u"
    assert table["ADC_TIME"][0] == np.uint64(2**64 - 1)
    assert table["ADC_VALID"][0] == 1
