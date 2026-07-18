"""Time-provenance attrs: writer round-trip and the reader's scale ladder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.decode import Products, SpectrumSample
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.hdf5_writer import write_hdf5
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
