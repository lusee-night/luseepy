"""Independent analytic and angular-quadrature four-port physics oracles."""

import numpy as np
from scipy.constants import c, physical_constants

import croissant as cro

from lusee.InstrumentResponse import InstrumentResponse
from lusee.ResponsePhysics import compute_sky_moon_resistance


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
FREQUENCY_MHZ = 13.0
EFFECTIVE_LENGTH_M = 0.07


def dipole_directions():
    """Return four non-orthogonal unit vectors."""
    tilt = np.radians(67.0)
    azimuth = np.radians([0.0, 71.0, 193.0, 287.0])
    return np.stack(
        (
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            np.full(4, np.cos(tilt)),
        ),
        axis=-1,
    )


def hertzian_fields(theta_intervals):
    """Evaluate bare effective lengths on a valid upper MWSS grid."""
    theta_deg = np.linspace(0.0, 90.0, theta_intervals + 1)
    phi_deg = np.linspace(0.0, 360.0, 4 * theta_intervals + 1)
    theta, phi = np.meshgrid(
        np.radians(theta_deg),
        np.radians(phi_deg),
        indexing="ij",
    )
    e_theta = np.stack(
        (
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ),
        axis=-1,
    )
    e_phi = np.stack(
        (
            -np.sin(phi),
            np.cos(phi),
            np.zeros_like(phi),
        ),
        axis=-1,
    )
    directions = dipole_directions()
    H_theta = EFFECTIVE_LENGTH_M * np.einsum(
        "ak,tpk->atp",
        directions,
        e_theta,
    )
    H_phi = EFFECTIVE_LENGTH_M * np.einsum(
        "ak,tpk->atp",
        directions,
        e_phi,
    )
    return (
        theta_deg,
        phi_deg,
        H_theta[:, None].astype(np.complex128),
        H_phi[:, None].astype(np.complex128),
    )


def analytic_upper_hemisphere_resistance():
    """Evaluate the closed-form Hertzian transverse-projector integral."""
    wavelength_m = c / (FREQUENCY_MHZ * 1e6)
    gram = dipole_directions() @ dipole_directions().T
    return (
        VACUUM_IMPEDANCE_OHM
        * np.pi
        * EFFECTIVE_LENGTH_M**2
        / (3.0 * wavelength_m**2)
        * gram
    )


def test_hertzian_resistance_matches_closed_form_across_grids():
    expected = analytic_upper_hemisphere_resistance()
    ZA = 2.0 * expected + 5.0j * np.eye(4)
    Rloss = np.zeros((1, 4, 4), dtype=np.complex128)
    for theta_intervals in (2, 4, 8, 16):
        theta, phi, H_theta, H_phi = hertzian_fields(theta_intervals)
        Rsky, Rmoon = compute_sky_moon_resistance(
            [FREQUENCY_MHZ],
            theta,
            phi,
            H_theta,
            H_phi,
            ZA[None],
            Rloss,
        )
        np.testing.assert_allclose(
            Rsky[0],
            expected,
            rtol=2e-12,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            Rmoon[0],
            expected,
            rtol=2e-12,
            atol=2e-14,
        )


def smooth_unpolarized_sky(theta, phi):
    """Return a positive localized sky with power beyond low l."""
    source_theta = np.radians(48.0)
    source_phi = np.radians(37.0)
    direction_cosine = (
        np.cos(theta) * np.cos(source_theta)
        + np.sin(theta)
        * np.sin(source_theta)
        * np.cos(phi - source_phi)
    )
    return 90.0 + 70.0 * np.exp(5.0 * (direction_cosine - 1.0))


def direct_upper_hemisphere_integral():
    """Integrate eta/lambda^2 H H^T I with independent quadrature."""
    mu_nodes, mu_weights = np.polynomial.legendre.leggauss(160)
    mu = 0.5 * (mu_nodes + 1.0)
    mu_weights = 0.5 * mu_weights
    phi = np.linspace(0.0, 2 * np.pi, 512, endpoint=False)
    theta = np.arccos(mu)[:, None]
    phi_grid = phi[None]
    normal = np.stack(
        (
            np.sin(theta) * np.cos(phi_grid) + np.zeros_like(theta),
            np.sin(theta) * np.sin(phi_grid) + np.zeros_like(theta),
            np.cos(theta) + np.zeros_like(phi_grid),
        ),
        axis=-1,
    )
    directions = dipole_directions()
    directional_cosines = np.einsum(
        "ak,mpk->amp",
        directions,
        normal,
    )
    projected_gram = (
        (directions @ directions.T)[:, :, None, None]
        - np.einsum(
            "amp,bmp->abmp",
            directional_cosines,
            directional_cosines,
        )
    )
    sky = smooth_unpolarized_sky(theta, phi_grid)
    angular_integral = np.einsum(
        "m,abmp,mp->ab",
        mu_weights,
        projected_gram,
        sky,
    ) * (2 * np.pi / phi.size)
    wavelength_m = c / (FREQUENCY_MHZ * 1e6)
    return (
        VACUUM_IMPEDANCE_OHM
        / wavelength_m**2
        * EFFECTIVE_LENGTH_M**2
        * angular_integral
    )


def make_harmonic_oracle_inputs(theta_intervals=32):
    theta, phi, H_theta, H_phi = hertzian_fields(theta_intervals)
    Rsky = analytic_upper_hemisphere_resistance()
    response = InstrumentResponse.from_arrays(
        [FREQUENCY_MHZ],
        theta,
        phi,
        H_theta,
        H_phi,
        (2.0 * Rsky + 5.0j * np.eye(4))[None],
        Rsky[None],
        Rsky[None],
        np.zeros((1, 4, 4), dtype=np.complex128),
        validated=False,
        metadata={"LOSSMODEL": "PEC"},
    )
    theta_full = np.linspace(0.0, np.pi, 2 * theta_intervals + 1)
    phi_full = np.linspace(
        0.0,
        2 * np.pi,
        4 * theta_intervals,
        endpoint=False,
    )
    theta_grid, phi_grid = np.meshgrid(
        theta_full,
        phi_full,
        indexing="ij",
    )
    stokes = np.zeros(
        (1, 4, theta_full.size, phi_full.size),
        dtype=np.float64,
    )
    stokes[:, 0] = smooth_unpolarized_sky(theta_grid, phi_grid)
    sky = cro.PolarizedSky(
        stokes,
        [FREQUENCY_MHZ],
        sampling="mwss",
        coord="topo",
        frame="topo",
        convention="IAU",
    )
    return response, sky


def harmonic_contraction(response, sky, lmax):
    pair_alms, _ = response.pair_stokes_alms(
        lmax,
        [FREQUENCY_MHZ],
    )
    sky_alms = sky.compute_alm(lmax=lmax)
    pair_values = np.einsum(
        "fclm,pfclm->p",
        np.asarray(sky_alms).conjugate(),
        np.asarray(pair_alms),
    )
    result = np.zeros((4, 4), dtype=np.complex128)
    for value, (a, b) in zip(pair_values, PORT_PAIRS):
        result[a, b] = value
        result[b, a] = value.conjugate()
    return result


def test_harmonic_contraction_converges_to_direct_quadrature():
    response, sky = make_harmonic_oracle_inputs()
    direct = direct_upper_hemisphere_integral()
    errors = []
    for lmax in (2, 4, 8):
        harmonic = harmonic_contraction(response, sky, lmax)
        np.testing.assert_allclose(harmonic, harmonic.conjugate().T)
        errors.append(
            np.linalg.norm(harmonic - direct) / np.linalg.norm(direct)
        )
    assert errors[1] < 0.1 * errors[0]
    assert errors[2] < 0.05 * errors[1]
    assert errors[2] < 5e-5
