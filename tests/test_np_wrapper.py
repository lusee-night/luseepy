import os

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np

from lusee.BeamGauss import BeamGauss
from lusee.NpWrapper import NpWrapper
from lusee.SkyModels import ConstSky, HarmonicPointSourceSky
from lusee.frequencies import canonical_frequencies, frequency_indices_from_values


def test_np_wrapper_exposes_numpy_beam_fields_and_results():
    beam = BeamGauss(alt_deg=37.0, az_deg=140.0, sigma_deg=12.0, id="gauss")
    wrapped = NpWrapper(beam)

    assert isinstance(wrapped.Etheta, np.ndarray)
    assert isinstance(wrapped.Ephi, np.ndarray)
    assert isinstance(wrapped.freq, np.ndarray)
    assert isinstance(wrapped.gain_conv, np.ndarray)

    assert isinstance(wrapped.power(), np.ndarray)
    power_stokes = wrapped.power_stokes()
    assert isinstance(power_stokes, list)
    assert all(isinstance(item, np.ndarray) for item in power_stokes)
    assert isinstance(wrapped.get_healpix_alm(lmax=4, freq_ndx=0), np.ndarray)


def test_np_wrapper_unwraps_other_wrappers_for_method_calls():
    beam_a = NpWrapper(BeamGauss(alt_deg=37.0, az_deg=140.0, sigma_deg=12.0, id="a"))
    beam_b = NpWrapper(BeamGauss(alt_deg=37.0, az_deg=230.0, sigma_deg=12.0, id="b"))

    cross = beam_a.cross_power(beam_b)
    assert isinstance(cross, np.ndarray)


def test_np_wrapper_exposes_numpy_sky_results():
    freq = canonical_frequencies(frequency_indices_from_values([10.0, 20.0]))
    sky = NpWrapper(ConstSky(Nside=8, lmax=6, T=np.array([120.0, 140.0]), freq=freq))
    assert isinstance(sky.mapalm, np.ndarray)
    assert isinstance(sky.get_alm([0, 1]), np.ndarray)

    point = NpWrapper(HarmonicPointSourceSky(lmax=6, freq=freq, T=[2.0, 5.0], l_deg=0.0, b_deg=45.0))
    assert isinstance(point._alm, np.ndarray)
    assert isinstance(point.get_alm([0, 1]), np.ndarray)
