"""Round-trip tests for the IngestData factory.

Builds a synthetic Products, writes both HDF5 and FITS, then reloads via
the factory and checks shapes, indexing, multi-file concatenation,
HDF5-vs-FITS precedence, and the time-source fallback ladder.

These tests need lunarsky (for the LunarTime axis on Observation). If
lunarsky cannot be imported (e.g. in a sandbox that blocks the SPICE
kernel download), the tests are skipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# IngestData triggers lunarsky -> SPICE-kernel download on first import.
# pytest.importorskip catches ImportError but not PermissionError raised by
# astropy's parallel-download path under restrictive sandboxes. Try the
# import explicitly and skip the whole module on any failure.
try:
    import lunarsky.time    # noqa: F401
except Exception as _exc:    # noqa: BLE001
    pytest.skip(
        f"lunarsky.time unavailable in this environment: {_exc!s}",
        allow_module_level=True,
    )

from lusee.ingest.constants import NCHANNELS, NPRODUCTS
from lusee.ingest.decode import (
    HKSample,
    Products,
    SpectrumSample,
    TRSpectrumSample,
    WaveformSample,
    ZoomSample,
)
pytest.importorskip("h5py")
from lusee.ingest.fits_writer import write_fits
from lusee.ingest.hdf5_writer import write_hdf5


def _make_products(n_spectra: int = 3, raw_t0: float = 100.0):
    products = Products(
        sw_version=0x307,
        fw_version=0x1234,
        start_unique_packet_id=100,
        start_time_32=0x10000,
        start_time_16=0,
        start_raw_seconds=1.0,
    )
    for i in range(n_spectra):
        cube = np.full((NPRODUCTS, NCHANNELS), float(i + 1), dtype=np.float32)
        products.spectra.append(SpectrumSample(
            data=cube,
            unique_packet_id=200 + i,
            raw_seconds=raw_t0 + i,
            metadata={
                "Navg1_shift": 4,
                "Navgf": 1,
                "adc_min": np.array([-1, -2, -3, -4], dtype=np.int16),
                "adc_max": np.array([1, 2, 3, 4], dtype=np.int16),
                "adc_mean": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "adc_rms": np.array([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            },
        ))

    products.tr_spectra.append(TRSpectrumSample(
        data=np.full((NPRODUCTS, 2, 4), 7.0, dtype=np.float32),
        unique_packet_id=300, raw_seconds=raw_t0 + 0.5,
        navg2=2, tr_length=4,
        metadata={"Navg2_shift": 1},
    ))
    products.zoom_spectra.append(ZoomSample(
        data=np.full((4, 64), 9.0, dtype=np.float32),
        unique_packet_id=400, pfb_index=11, raw_seconds=raw_t0 + 0.25,
    ))
    products.waveforms.append(WaveformSample(
        data=np.arange(16384, dtype=np.int16) % 100,
        channel=0, unique_packet_id=500, raw_seconds=raw_t0 + 0.1,
    ))
    products.housekeeping.append(HKSample(
        hk_type=1, version=0x307, unique_packet_id=600,
        errors=0, raw_seconds=raw_t0 + 0.2,
        fields={
            "adc_min": np.array([-10, -20, -30, -40], dtype=np.int16),
            "adc_max": np.array([10, 20, 30, 40], dtype=np.int16),
            "adc_mean": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "adc_rms": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "actual_gain": np.array([b"L", b"M", b"H", b"L"], dtype="S1"),
        },
    ))

    fpga = {
        "mission_seconds": np.array([raw_t0, raw_t0 + 1, raw_t0 + 2], dtype=np.float64),
        "lusee_subsecs": np.zeros(3, dtype=np.float64),
        "THERM_FPGA": np.array([42.0, 43.0, 44.0], dtype=np.float64),
        "VMON_6V": np.array([6.05, 6.04, 6.03], dtype=np.float64),
    }
    return products, fpga


def test_load_h5_basic(tmp_path: Path):
    from lusee.ingest import load
    products, fpga = _make_products()
    h5 = tmp_path / "session.h5"
    write_hdf5(products, h5, fpga_telemetry=fpga)
    data = load(h5)

    assert data.spectra.shape == (3, NPRODUCTS, NCHANNELS)
    assert data.spectra.dtype == np.float32
    assert data.unique_ids.tolist() == [200, 201, 202]
    assert data.raw_times.tolist() == [100.0, 101.0, 102.0]
    assert data.tr_spectra.shape == (1, NPRODUCTS, 2, 4)
    assert data.zoom_spectra.shape == (1, 4, 64)
    assert data.dcb_telemetry["THERM_FPGA"].size == 3
    assert 1 in data.housekeeping and "adc_min" in data.housekeeping[1]
    assert data.session_invariants["software_version"] == 0x307
    assert data.layout_version == 2
    assert len(data.bundles) == 1
    assert len(data.source_paths) == 1


def test_load_fits_basic(tmp_path: Path):
    from lusee.ingest import load
    products, fpga = _make_products()
    fits_path = tmp_path / "session.fits"
    write_fits(products, fits_path, fpga_telemetry=fpga)
    data = load(fits_path)

    assert data.spectra.shape == (3, NPRODUCTS, NCHANNELS)
    assert data.tr_spectra.shape == (1, NPRODUCTS, 2, 4)
    assert data.zoom_spectra.shape == (1, 4, 64)
    assert data.dcb_telemetry["THERM_FPGA"].size == 3
    assert 1 in data.housekeeping
    assert data.session_invariants["software_version"] == 0x307


def test_h5_takes_precedence_over_fits(tmp_path: Path):
    from lusee.ingest import load
    products, _ = _make_products()
    h5_dir = tmp_path / "out"
    h5_dir.mkdir()
    # Write FITS first then HDF5; confirm directory load picks h5.
    write_fits(products, h5_dir / "session.fits")
    write_hdf5(products, h5_dir / "session.h5")
    data = load(h5_dir)
    src = data.source_paths[0]
    assert src.suffix == ".h5", f"expected .h5 precedence, got {src}"


def test_prefer_format_fits(tmp_path: Path):
    from lusee.ingest import load
    products, _ = _make_products()
    h5_dir = tmp_path / "out"
    h5_dir.mkdir()
    write_hdf5(products, h5_dir / "session.h5")
    write_fits(products, h5_dir / "session.fits")
    data = load(h5_dir, prefer_format="fits")
    assert data.source_paths[0].suffix == ".fits"


def test_indexing(tmp_path: Path):
    from lusee.ingest import load
    products, _ = _make_products()
    # Set distinct values per product for indexing checks.
    for s_i, s in enumerate(products.spectra):
        for p in range(NPRODUCTS):
            s.data[p, :] = float(s_i * 100 + p)
    h5 = tmp_path / "s.h5"
    write_hdf5(products, h5)
    data = load(h5)

    # Autocorrelation 0 across all time, all freq -> p=0 plane.
    auto0 = data[:, "00", :]
    assert auto0.shape == (3, NCHANNELS)
    np.testing.assert_array_equal(auto0[0], 0.0)
    np.testing.assert_array_equal(auto0[1], 100.0)
    np.testing.assert_array_equal(auto0[2], 200.0)

    # Cross 0x1 real -> product index 4
    cross_re = data[:, "01R", :]
    np.testing.assert_array_equal(cross_re[0], 4.0)

    # Cross 0x1 imag -> product index 5
    cross_im = data[:, "01I", :]
    np.testing.assert_array_equal(cross_im[0], 5.0)

    # Cross 0x1 complex -> 4 + 1j*5
    cross_c = data[0, "01C", 0]
    assert cross_c == 4.0 + 1j * 5.0

    # Negation prefix
    neg_auto = data[:, "-00", :]
    np.testing.assert_array_equal(neg_auto[0], -0.0)

    # Symmetric access (i > j) -> conjugate of (j, i)
    via_swap = data[0, "10C", 0]
    assert via_swap == 4.0 - 1j * 5.0


def test_multi_file_concat(tmp_path: Path):
    from lusee.ingest import load
    p_a, _ = _make_products(n_spectra=2, raw_t0=100.0)
    p_b, _ = _make_products(n_spectra=3, raw_t0=500.0)
    write_hdf5(p_a, tmp_path / "a.h5")
    write_hdf5(p_b, tmp_path / "b.h5")
    data = load([tmp_path / "a.h5", tmp_path / "b.h5"])
    assert data.spectra.shape[0] == 5
    # Order is by start time -> a first (100), then b (500).
    np.testing.assert_array_equal(
        data.raw_times, [100.0, 101.0, 500.0, 501.0, 502.0]
    )
    bounds = data.session_boundaries()
    assert len(bounds) == 2
    assert bounds[0][1] - bounds[0][0] == 2
    assert bounds[1][1] - bounds[1][0] == 3


def test_load_directory(tmp_path: Path):
    from lusee.ingest import load
    products, _ = _make_products()
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    write_hdf5(products, h5_dir / "session.h5")
    data = load(h5_dir)    # directory path, not file
    assert data.spectra.shape[0] == 3


def test_repr(tmp_path: Path):
    from lusee.ingest import load
    products, fpga = _make_products()
    write_hdf5(products, tmp_path / "s.h5", fpga_telemetry=fpga)
    data = load(tmp_path / "s.h5")
    text = repr(data)
    assert "IngestData" in text
    assert "N_spectra=3" in text
    assert "telemetry=True" in text


def test_empty_input_errors(tmp_path: Path):
    from lusee.ingest import load
    with pytest.raises(FileNotFoundError):
        load(tmp_path)    # empty dir


def test_lusee_ingest_lazy_load(tmp_path: Path):
    """`import lusee.ingest` must not pull lunarsky / lusee.Observation.

    We run this in a subprocess because the parent test runner has long
    since polluted sys.modules with lunarsky and astropy machinery; only a
    fresh interpreter can answer the "first import" question honestly.
    """
    import os
    import subprocess
    import sys

    # Build a small bootstrap that mirrors what tests/ingest/conftest.py does:
    # if UNCRATER_PATH is set, prepend it to sys.path. (The subprocess does
    # not load conftest because we run a `-c` snippet, not pytest.)
    code = (
        "import os, sys; "
        "_u = os.environ.get('UNCRATER_PATH'); "
        "_t = os.environ.get('LUSEE_TELEMETRY_PATH'); "
        "[sys.path.insert(0, p) for p in (_u, _t) if p]; "
        "import lusee.ingest; "
        "assert 'lunarsky' not in sys.modules, list(sys.modules); "
        "assert 'lusee.Observation' not in sys.modules, list(sys.modules); "
        "print('OK')"
    )
    env = os.environ.copy()
    # Inherit PYTHONPATH from the parent so the subprocess sees the same
    # packages (uncrater, lusee, h5py, etc.).
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=60,
    )
    if proc.returncode != 0:
        pytest.fail(f"lazy-load subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}")
    assert "OK" in proc.stdout
