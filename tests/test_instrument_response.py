"""Tests for FITS-v3 four-port instrument responses."""

import jax.numpy as jnp
import fitsio
import numpy as np
import pytest

from beam_conversion.common import (
    ResponseArrays,
    compute_sky_moon_resistance,
    embedded_fields_to_bare,
    write_response_fits,
)
from lusee.InstrumentResponse import InstrumentResponse, PORT_PAIRS


def make_response_arrays(freq=(10.0, 20.0)):
    """Build a small mwss-compatible physical response fixture."""
    freq = np.asarray(freq, dtype=np.float64)
    theta = np.arange(0.0, 91.0, 45.0)
    phi = np.arange(0.0, 361.0, 45.0)
    tt, pp = np.meshgrid(np.radians(theta), np.radians(phi), indexing="ij")
    H_theta = np.empty((4, freq.size, theta.size, phi.size), dtype=np.complex128)
    H_phi = np.empty_like(H_theta)
    for port in range(4):
        frequency_scale = (1.0 + 0.1 * np.arange(freq.size))[None, :, None, None]
        phase = np.exp(1j * port * pp)[None, None]
        H_theta[port : port + 1] = (
            0.05
            * (port + 1)
            * frequency_scale
            * np.cos(tt)[None, None]
            * phase
        )
        H_phi[port : port + 1] = (
            0.03
            * frequency_scale
            * np.sin(tt)[None, None]
            * np.exp(-1j * (port + 1) * pp)[None, None]
        )
    ZA = np.broadcast_to(
        (30.0 + 5.0j) * np.eye(4)[None],
        (freq.size, 4, 4),
    ).copy()
    Rsky, Rmoon = compute_sky_moon_resistance(
        freq, theta, phi, H_theta, H_phi, ZA
    )
    return ResponseArrays(
        freq,
        theta,
        phi,
        H_theta,
        H_phi,
        ZA,
        Rsky,
        Rmoon,
        metadata={
            "SOURCE": "synthetic",
            "SOURCE_ROOT": "pytest",
            "INPUT_KIND": "bare",
            "FIELD_KIND": "effective-length",
            "AMP_CONV": "RMS",
            "ZA_SOURCE": "analytic",
            "GIT_SHA": "test",
            "TIMECONV": "e+jwt",
            "COORDSYS": "instrument-topocentric",
            "THETADEF": "colatitude-from-+z",
            "PHIDEF": "right-handed-about-+z",
            "OMEGADEF": "source-arrival-direction",
            "POLBASIS": "e_theta,e_phi",
            "PHASEREF": "analytic-origin",
            "PORTS": "0123",
        },
    )


def test_response_fits_round_trip_preserves_float64_grid_and_units(tmp_path):
    arrays = make_response_arrays()
    filename = tmp_path / "response.fits"
    write_response_fits(filename, arrays, dtype="float32", validated=True)
    response = InstrumentResponse(filename)
    assert response.freq.dtype == np.float64
    assert response.H_theta.dtype == jnp.complex64
    assert response.H_theta.shape == arrays.H_theta.shape
    assert response.ZA.shape == (2, 4, 4)
    assert response.sky_coupling_check()["physical"]
    np.testing.assert_allclose(
        response.Rsky_native,
        arrays.Rsky,
        rtol=2e-6,
        atol=2e-6,
    )


def test_response_loader_rejects_unvalidated_by_default(tmp_path):
    arrays = make_response_arrays()
    filename = tmp_path / "unvalidated.fits"
    write_response_fits(filename, arrays, validated=False)
    with pytest.raises(ValueError, match="VALIDATED=False"):
        InstrumentResponse(filename)
    response = InstrumentResponse(filename, require_validated=False)
    assert not response.validated


def test_response_loader_rejects_noncanonical_validated_metadata(tmp_path):
    arrays = make_response_arrays()
    filename = tmp_path / "bad_convention.fits"
    write_response_fits(filename, arrays)
    with fitsio.FITS(filename, "rw") as fits:
        fits["H_theta_real"].write_key("POLBASIS", "mystery-basis")
    with pytest.raises(ValueError, match="unsupported POLBASIS"):
        InstrumentResponse(filename)


def test_response_loader_rechecks_validated_physical_matrices(tmp_path):
    arrays = make_response_arrays()
    filename = tmp_path / "bad_physics.fits"
    write_response_fits(filename, arrays, dtype="float64")
    with fitsio.FITS(filename, "rw") as fits:
        moon = fits["Rmoon_real"].read()
        sky = fits["Rsky_real"].read()
        shift = 2.0 * np.max(np.linalg.eigvalsh(arrays.Rmoon[0]))
        moon[0, 0, 0] -= shift
        sky[0, 0, 0] += shift
        fits["Rmoon_real"].write(moon)
        fits["Rsky_real"].write(sky)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        InstrumentResponse(filename)


def test_response_loader_uses_stored_content_hash_without_rehashing(
    tmp_path,
    monkeypatch,
):
    arrays = make_response_arrays()
    filename = tmp_path / "content_hash.fits"
    write_response_fits(filename, arrays)
    import importlib

    module = importlib.import_module("lusee.InstrumentResponse")

    def unexpected_hash(values):
        raise AssertionError("stored CONTENT must bypass fallback hashing")

    monkeypatch.setattr(module, "_content_hash", unexpected_hash)
    response = InstrumentResponse(filename)
    assert response.content_hash


def test_validated_writer_requires_explicit_provenance(tmp_path):
    arrays = make_response_arrays()
    arrays.metadata = {"SOURCE": "synthetic"}
    with pytest.raises(ValueError, match="explicit response provenance"):
        write_response_fits(tmp_path / "missing.fits", arrays)


def test_validated_writer_rejects_noncanonical_provenance(tmp_path):
    arrays = make_response_arrays()
    arrays.metadata["POLBASIS"] = "mystery-basis"
    with pytest.raises(ValueError, match="unsupported POLBASIS"):
        write_response_fits(tmp_path / "bad_convention.fits", arrays)


def test_writer_argument_controls_validated_header(tmp_path):
    arrays = make_response_arrays()
    arrays.metadata["VALIDATED"] = False
    validated_filename = tmp_path / "validated.fits"
    write_response_fits(validated_filename, arrays, validated=True)
    assert bool(fitsio.read_header(validated_filename)["VALIDATED"])

    arrays.metadata["VALIDATED"] = True
    unvalidated_filename = tmp_path / "unvalidated.fits"
    write_response_fits(unvalidated_filename, arrays, validated=False)
    assert not bool(fitsio.read_header(unvalidated_filename)["VALIDATED"])


def test_validated_writer_requires_horizon_and_phi_wrap(tmp_path):
    arrays = make_response_arrays()
    arrays.theta_deg = arrays.theta_deg[:-1]
    arrays.H_theta = arrays.H_theta[:, :, :-1]
    arrays.H_phi = arrays.H_phi[:, :, :-1]
    with pytest.raises(ValueError, match="span 0 through 90"):
        write_response_fits(tmp_path / "short_theta.fits", arrays)

    arrays = make_response_arrays()
    arrays.phi_deg = arrays.phi_deg[:-1]
    arrays.H_theta = arrays.H_theta[..., :-1]
    arrays.H_phi = arrays.H_phi[..., :-1]
    with pytest.raises(ValueError, match="retain the 0/360 wrap"):
        write_response_fits(tmp_path / "no_wrap.fits", arrays)

    arrays = make_response_arrays()
    phi_indices = [0, 2, 4, 6, 8]
    arrays.phi_deg = arrays.phi_deg[phi_indices]
    arrays.H_theta = arrays.H_theta[..., phi_indices]
    arrays.H_phi = arrays.H_phi[..., phi_indices]
    with pytest.raises(ValueError, match="Nphi-1"):
        write_response_fits(tmp_path / "incompatible_mwss.fits", arrays)


def test_validated_loader_rechecks_horizon_geometry(tmp_path):
    arrays = make_response_arrays()
    filename = tmp_path / "bad_theta.fits"
    write_response_fits(filename, arrays)
    with fitsio.FITS(filename, "rw") as fits:
        theta = fits["theta"].read()
        theta[-1] = 80.0
        fits["theta"].write(theta)
    with pytest.raises(ValueError, match="end at 90"):
        InstrumentResponse(filename)

    arrays = make_response_arrays()
    with pytest.raises(ValueError, match="three stored phi"):
        InstrumentResponse.from_arrays(
            arrays.freq_mhz,
            arrays.theta_deg,
            [0.0, 360.0],
            arrays.H_theta[..., [0, -1]],
            arrays.H_phi[..., [0, -1]],
            arrays.ZA,
            arrays.Rsky,
            arrays.Rmoon,
            validated=True,
            metadata=arrays.metadata,
        )


def test_validated_writer_rejects_negative_moon_resistance(tmp_path):
    arrays = make_response_arrays()
    arrays.Rmoon = arrays.Rmoon.copy()
    arrays.Rsky = arrays.Rsky.copy()
    shift = 2.0 * np.max(np.linalg.eigvalsh(arrays.Rmoon[0]))
    arrays.Rmoon[0, 0, 0] -= shift
    arrays.Rsky[0, 0, 0] += shift
    with pytest.raises(ValueError, match="negative eigenvalue"):
        write_response_fits(tmp_path / "negative_moon.fits", arrays)


def test_embedded_basis_right_solve_recovers_noncommuting_bare_fields():
    rng = np.random.default_rng(4)
    nfreq = 2
    ZA = (
        rng.normal(size=(nfreq, 4, 4))
        + 1j * rng.normal(size=(nfreq, 4, 4))
        + 8 * np.eye(4)[None]
    )
    Vsource = (
        rng.normal(size=(nfreq, 4, 4))
        + 1j * rng.normal(size=(nfreq, 4, 4))
        + 3 * np.eye(4)[None]
    )
    Zref = np.broadcast_to(np.asarray([40.0, 50.0, 60.0, 70.0]), (nfreq, 4))
    load = np.zeros_like(ZA)
    diagonal = np.arange(4)
    load[:, diagonal, diagonal] = Zref
    currents = np.linalg.solve(ZA + load, Vsource)
    bare_theta = rng.normal(size=(4, nfreq, 2, 3)) + 1j * rng.normal(
        size=(4, nfreq, 2, 3)
    )
    bare_phi = rng.normal(size=(4, nfreq, 2, 3)) + 1j * rng.normal(
        size=(4, nfreq, 2, 3)
    )

    def embed(bare):
        rows = np.moveaxis(bare, (0, 1), (-1, 0))
        embedded = np.einsum("ftpa,fae->ftpe", rows, currents)
        return np.moveaxis(embedded, (0, -1), (1, 0))

    recovered_theta, recovered_phi, recovered_currents = (
        embedded_fields_to_bare(
            embed(bare_theta),
            embed(bare_phi),
            ZA,
            Zref,
            Vsource,
        )
    )
    np.testing.assert_allclose(recovered_currents, currents, rtol=1e-12)
    np.testing.assert_allclose(recovered_theta, bare_theta, rtol=1e-12)
    np.testing.assert_allclose(recovered_phi, bare_phi, rtol=1e-12)


def test_pair_stokes_maps_obey_baseline_conjugation():
    arrays = make_response_arrays()
    response = InstrumentResponse.from_arrays(
        arrays.freq_mhz,
        arrays.theta_deg,
        arrays.phi_deg,
        arrays.H_theta,
        arrays.H_phi,
        arrays.ZA,
        arrays.Rsky,
        arrays.Rmoon,
    )
    for a, b in PORT_PAIRS:
        forward = response.pair_stokes_maps(a, b)
        reverse = response.pair_stokes_maps(b, a)
        assert jnp.allclose(reverse, forward.conjugate())


def test_pair_stokes_positive_v_sign_matches_exp_plus_i_fixture():
    arrays = make_response_arrays()
    shape = arrays.H_theta.shape
    H_theta = np.zeros(shape, dtype=np.complex128)
    H_phi = np.zeros(shape, dtype=np.complex128)
    H_theta[0] = 1.0 / np.sqrt(2.0)
    H_phi[0] = -1.0j / np.sqrt(2.0)
    H_theta[1] = 1.0 / np.sqrt(2.0)
    H_phi[1] = 1.0j / np.sqrt(2.0)
    response = InstrumentResponse.from_arrays(
        arrays.freq_mhz,
        arrays.theta_deg,
        arrays.phi_deg,
        H_theta,
        H_phi,
        arrays.ZA,
        arrays.Rsky,
        arrays.Rmoon,
    )
    matched = response.pair_stokes_maps(0, 0)
    rejected = response.pair_stokes_maps(1, 1)
    assert jnp.allclose(matched[:, 0], 1.0)
    assert jnp.allclose(matched[:, 3], 1.0)
    assert jnp.allclose(rejected[:, 0], 1.0)
    assert jnp.allclose(rejected[:, 3], -1.0)


def test_target_pair_alms_preserve_order_duplicates_and_unique_sht_endpoints():
    arrays = make_response_arrays()
    response = InstrumentResponse.from_arrays(
        arrays.freq_mhz,
        arrays.theta_deg,
        arrays.phi_deg,
        arrays.H_theta,
        arrays.H_phi,
        arrays.ZA,
        arrays.Rsky,
        arrays.Rmoon,
    )
    target = np.asarray([17.5, 10.0, 17.5])
    alms, frequency_map = response.pair_stokes_alms(2, target)
    assert alms.shape == (10, 3, 4, 3, 5)
    assert frequency_map.source_indices.tolist() == [0, 1]
    assert jnp.array_equal(alms[:, 0], alms[:, 2])
    assert response.native_transform_count == 2
    response.pair_stokes_alms(2, [10.0])
    assert response.native_transform_count == 2


def test_rotation_moves_all_ports_and_keeps_wraparound():
    arrays = make_response_arrays()
    response = InstrumentResponse.from_arrays(
        arrays.freq_mhz,
        arrays.theta_deg,
        arrays.phi_deg,
        arrays.H_theta,
        arrays.H_phi,
        arrays.ZA,
        arrays.Rsky,
        arrays.Rmoon,
    )
    rotated = response.rotate(45.0)
    assert jnp.array_equal(rotated.H_theta[..., 0], rotated.H_theta[..., -1])
    assert jnp.array_equal(rotated.H_theta[..., 0], response.H_theta[..., 1])
