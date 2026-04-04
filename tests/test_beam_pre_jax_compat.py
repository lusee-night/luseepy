import copy
import os

os.environ["JAX_ENABLE_X64"] = "True"

import jax
import numpy as np
import pytest

from lusee.Beam import Beam
from lusee.BeamGauss import BeamGauss
from lusee.pre_jax.Beam import Beam as PreJaxBeam
from lusee.pre_jax.BeamGauss import BeamGauss as PreJaxBeamGauss


RTOL = 1e-10
ATOL = 1e-10

assert jax.config.jax_enable_x64


def to_numpy(value):
    """Normalize numpy/jax-like containers to numpy-backed Python objects."""
    if isinstance(value, dict):
        return {key: to_numpy(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_numpy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool)):
        return value
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return value


def assert_same_value(old_value, new_value, label):
    old_value = to_numpy(old_value)
    new_value = to_numpy(new_value)

    if isinstance(old_value, dict):
        assert set(old_value) == set(new_value), label
        for key in sorted(old_value):
            assert_same_value(old_value[key], new_value[key], f"{label}.{key}")
        return

    if isinstance(old_value, list):
        assert len(old_value) == len(new_value), label
        for idx, (old_item, new_item) in enumerate(zip(old_value, new_value)):
            assert_same_value(old_item, new_item, f"{label}[{idx}]")
        return

    if isinstance(old_value, tuple):
        assert len(old_value) == len(new_value), label
        for idx, (old_item, new_item) in enumerate(zip(old_value, new_value)):
            assert_same_value(old_item, new_item, f"{label}[{idx}]")
        return

    if isinstance(old_value, np.ndarray) or isinstance(new_value, np.ndarray):
        np.testing.assert_allclose(
            np.asarray(old_value),
            np.asarray(new_value),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
            err_msg=label,
        )
        return

    if np.isscalar(old_value) and np.isscalar(new_value) and not isinstance(old_value, (str, bytes, bool)):
        np.testing.assert_allclose(old_value, new_value, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=label)
        return

    assert old_value == new_value, label


def assert_same_members(old_beam, new_beam):
    assert set(old_beam.__dict__) == set(new_beam.__dict__)
    for name in sorted(old_beam.__dict__):
        assert_same_value(old_beam.__dict__[name], new_beam.__dict__[name], f"attr:{name}")


def assert_same_beam_or_value(old_value, new_value, label):
    if hasattr(old_value, "__dict__") and hasattr(new_value, "__dict__"):
        assert_same_members(old_value, new_value)
    else:
        assert_same_value(old_value, new_value, label)


def assert_same_interpolators(old_beam, new_beam):
    old_interp_theta, old_interp_phi = old_beam.get_Efield_interpolator()
    new_interp_theta, new_interp_phi = new_beam.get_Efield_interpolator()

    scalar_args = (np.deg2rad(35.0), np.deg2rad(123.0), 12.5)
    assert_same_value(old_interp_theta(*scalar_args), new_interp_theta(*scalar_args), "interp_theta_scalar")
    assert_same_value(old_interp_phi(*scalar_args), new_interp_phi(*scalar_args), "interp_phi_scalar")

    vector_args = (
        np.deg2rad(np.array([15.0, 40.0])),
        np.deg2rad(np.array([45.0, 210.0])),
        np.array([5.0, 17.5]),
    )
    assert_same_value(old_interp_theta(*vector_args), new_interp_theta(*vector_args), "interp_theta_vector")
    assert_same_value(old_interp_phi(*vector_args), new_interp_phi(*vector_args), "interp_phi_vector")


def assert_same_public_methods(old_beam, new_beam):
    old_peer = old_beam.rotate(90)
    new_peer = new_beam.rotate(90)

    assert_same_beam_or_value(old_beam.copy_beam(), new_beam.copy_beam(), "copy_beam")
    assert_same_beam_or_value(old_beam.rotate(0), new_beam.rotate(0), "rotate_zero")
    assert_same_beam_or_value(old_peer, new_peer, "rotate_90")

    assert_same_value(old_beam.power(), new_beam.power(), "power")
    assert_same_value(old_beam.power_stokes(), new_beam.power_stokes(), "power_stokes")
    assert_same_value(old_beam.power_stokes(cross=old_peer), new_beam.power_stokes(cross=new_peer), "power_stokes_cross")
    assert_same_value(old_beam.cross_power(old_peer), new_beam.cross_power(new_peer), "cross_power")
    assert_same_value(old_beam.sky_fraction(), new_beam.sky_fraction(), "sky_fraction")
    assert_same_value(old_beam.sky_fraction(cross=old_peer), new_beam.sky_fraction(cross=new_peer), "sky_fraction_cross")
    assert_same_value(old_beam.ground_fraction(), new_beam.ground_fraction(), "ground_fraction")
    assert_same_value(old_beam.ground_fraction(cross=old_peer), new_beam.ground_fraction(cross=new_peer), "ground_fraction_cross")

    assert_same_value(old_beam.power_hp(ellmax=4, Nside=8, freq_ndx=0), new_beam.power_hp(ellmax=4, Nside=8, freq_ndx=0), "power_hp")
    assert_same_value(
        old_beam.power_hp(ellmax=4, Nside=8, freq_ndx=0, stokes=True),
        new_beam.power_hp(ellmax=4, Nside=8, freq_ndx=0, stokes=True),
        "power_hp_stokes",
    )
    assert_same_value(old_beam.get_healpix_alm(lmax=4, freq_ndx=0), new_beam.get_healpix_alm(lmax=4, freq_ndx=0), "healpix_alm_I")
    assert_same_value(
        old_beam.get_healpix_alm(lmax=4, freq_ndx=0, other=old_peer, return_I_stokes_only=False),
        new_beam.get_healpix_alm(lmax=4, freq_ndx=0, other=new_peer, return_I_stokes_only=False),
        "healpix_alm_stokes_cross",
    )
    assert_same_value(
        old_beam.get_healpix_alm(lmax=4, freq_ndx=0, return_complex_components=True),
        new_beam.get_healpix_alm(lmax=4, freq_ndx=0, return_complex_components=True),
        "healpix_alm_complex_components",
    )

    assert_same_interpolators(old_beam, new_beam)

    old_tapered = copy.deepcopy(old_beam).taper_and_smooth(taper=0.03, beam_smooth=0.5)
    new_tapered = copy.deepcopy(new_beam).taper_and_smooth(taper=0.03, beam_smooth=0.5)
    assert_same_beam_or_value(old_tapered, new_tapered, "taper_and_smooth")


@pytest.mark.integration
def test_beam_matches_pre_jax_reference(drive_dir):
    fname = os.path.join(
        drive_dir,
        "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits",
    )
    old_beam = PreJaxBeam(fname=fname, id="pre_jax")
    new_beam = Beam(fname=fname, id="pre_jax")

    assert_same_members(old_beam, new_beam)
    assert_same_public_methods(old_beam, new_beam)


def test_beam_gauss_matches_pre_jax_reference():
    ctor_kwargs = dict(
        alt_deg=37.0,
        az_deg=140.0,
        sigma_deg=12.0,
        one_over_freq_scaling=False,
        id="gauss_pre_jax",
    )
    old_beam = PreJaxBeamGauss(**ctor_kwargs)
    new_beam = BeamGauss(**ctor_kwargs)

    assert_same_members(old_beam, new_beam)
    assert_same_public_methods(old_beam, new_beam)


def test_beam_gauss_pytree_roundtrip():
    beam = BeamGauss(
        alt_deg=37.0,
        az_deg=140.0,
        sigma_deg=12.0,
        one_over_freq_scaling=False,
        id="gauss_tree",
    )
    leaves, treedef = jax.tree_util.tree_flatten(beam)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, BeamGauss)
    assert_same_value(beam.Etheta, rebuilt.Etheta, "pytree.Etheta")
    assert_same_value(beam.Ephi, rebuilt.Ephi, "pytree.Ephi")
    assert_same_value(beam.ZRe, rebuilt.ZRe, "pytree.ZRe")
    assert_same_value(beam.ZIm, rebuilt.ZIm, "pytree.ZIm")
    assert_same_value(beam.freq, rebuilt.freq, "pytree.freq")
    assert_same_value(beam.gain_conv, rebuilt.gain_conv, "pytree.gain_conv")
    assert_same_value(beam.theta, rebuilt.theta, "pytree.theta")
    assert_same_value(beam.phi, rebuilt.phi, "pytree.phi")
    assert_same_value(beam.ground_fraction(), rebuilt.ground_fraction(), "pytree.ground_fraction")
