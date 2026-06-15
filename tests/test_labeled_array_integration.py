"""Integration tests: units/frame labels on the simulator output and Data interface."""
import os

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import jax
import pytest

import lusee
from lusee.LabeledArray import LabeledArray
from lusee.frequencies import canonical_frequencies, frequency_indices_from_values


def _small_sim(tmp_path):
    obs = lusee.Observation(
        "2025-03-01 00:00:00 to 2025-03-01 01:00:00",
        deltaT_sec=3600.0, lun_lat_deg=0.0, lun_long_deg=0.0,
    )
    freq = canonical_frequencies(frequency_indices_from_values([10.0]))
    lmax = 8
    sky = lusee.sky.HarmonicPointSourceSky(lmax=lmax, l_deg=0.0, b_deg=0.0, freq=freq)
    beam = lusee.BeamGauss(alt_deg=90.0, az_deg=0.0, sigma_deg=20.0,
                           one_over_freq_scaling=False, id="lbl")
    sim = lusee.TopoNumpySimulator(
        obs, [lusee.NpWrapper(beam)], lusee.NpWrapper(sky),
        Tground=0.0, combinations=[(0, 0)], freq=freq, lmax=lmax,
        extra_opts={"cache_transform": str(tmp_path / "cache")},
    )
    return sim, obs.times, freq


def test_simulate_returns_bare_array(tmp_path):
    """simulate() must return a bare array (MapMaker jit/grad contract)."""
    sim, times, _ = _small_sim(tmp_path)
    res = sim.simulate(times=times)
    assert not isinstance(res, LabeledArray)
    assert isinstance(res, (np.ndarray, jax.Array))
    assert res.shape == (3, 1, 1)


def test_result_labeled_property(tmp_path):
    sim, times, _ = _small_sim(tmp_path)
    assert sim.result_labeled is None  # before simulate
    sim.simulate(times=times)
    rl = sim.result_labeled
    assert isinstance(rl, LabeledArray)
    assert rl.units == "K"
    assert rl.frame == lusee.FRAME_TOPO == "topo"
    np.testing.assert_array_equal(np.asarray(rl), np.asarray(sim.result))


class _StubThroughput:
    """Minimal throughput so Data() works without LUSEE_DRIVE_DIR."""
    def T2Vsq(self, freq):
        return np.ones(len(freq))


def test_data_getitem_is_labeled(tmp_path):
    sim, times, freq = _small_sim(tmp_path)
    sim.simulate(times=times)
    fits_path = str(tmp_path / "sim.fits")
    sim.write_fits(fits_path)

    D = lusee.Data(fits_path, throughput=_StubThroughput())

    raw = D[0, "00R", :]
    assert isinstance(raw, LabeledArray)
    assert raw.units == "K" and raw.frame == "topo"
    # drop-in: numpy conversion matches the bare FITS slice
    np.testing.assert_array_equal(np.asarray(raw), D.data[0, 0, :])

    volts = D[0, "00RV", :]
    assert isinstance(volts, LabeledArray)
    assert volts.units == "V"
    # T2Vsq stub is unity, so V values equal K values here
    np.testing.assert_allclose(np.asarray(volts), np.asarray(raw))


def test_data_units_attribute_override(tmp_path):
    sim, times, freq = _small_sim(tmp_path)
    sim.simulate(times=times)
    fits_path = str(tmp_path / "sim2.fits")
    sim.write_fits(fits_path)
    D = lusee.Data(fits_path, throughput=_StubThroughput())
    assert D.data_units == "K"
    # a future ingest path may relabel raw data units
    D.data_units = "counts"
    assert D[0, "00R", :].units == "counts"
