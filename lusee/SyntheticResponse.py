"""Analytic four-port response fixture independent of converter normalization."""

import numpy as np
from scipy.constants import c, physical_constants

from .InstrumentResponse import InstrumentResponse


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]


def synthetic_four_port_response(
    freq_mhz=(10.0, 20.0),
    *,
    effective_length_m=0.05,
    tilt_deg=75.0,
    angular_step_deg=45.0,
):
    """Construct four ideal short dipoles with analytic radiation resistance."""
    freq = np.asarray(freq_mhz, dtype=np.float64)
    theta = np.arange(0.0, 90.0 + angular_step_deg, angular_step_deg)
    phi = np.arange(0.0, 360.0 + angular_step_deg, angular_step_deg)
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)
    tt, pp = np.meshgrid(theta_rad, phi_rad, indexing="ij")
    e_theta = np.stack(
        (
            np.cos(tt) * np.cos(pp),
            np.cos(tt) * np.sin(pp),
            -np.sin(tt),
        ),
        axis=-1,
    )
    e_phi = np.stack(
        (
            -np.sin(pp),
            np.cos(pp),
            np.zeros_like(pp),
        ),
        axis=-1,
    )
    tilt = np.radians(tilt_deg)
    azimuth = np.radians([0.0, 90.0, 180.0, 270.0])
    directions = np.stack(
        (
            np.sin(tilt) * np.cos(azimuth),
            np.sin(tilt) * np.sin(azimuth),
            np.full(4, np.cos(tilt)),
        ),
        axis=-1,
    )
    Htheta_one = effective_length_m * np.einsum(
        "ak,txk->atx",
        directions,
        e_theta,
    )
    Hphi_one = effective_length_m * np.einsum(
        "ak,txk->atx",
        directions,
        e_phi,
    )
    Htheta = np.broadcast_to(
        Htheta_one[:, None],
        (4, freq.size) + Htheta_one.shape[1:],
    ).astype(np.complex128).copy()
    Hphi = np.broadcast_to(
        Hphi_one[:, None],
        (4, freq.size) + Hphi_one.shape[1:],
    ).astype(np.complex128).copy()

    wavelength = c / (freq * 1e6)
    gram = directions @ directions.T
    radiation_scale = (
        VACUUM_IMPEDANCE_OHM
        * 2
        * np.pi
        * effective_length_m**2
        / (3 * wavelength**2)
    )
    radiation = radiation_scale[:, None, None] * gram[None]
    reactance = (
        5.0 * np.eye(4)[None]
        + 0.25 * gram[None]
    )
    ZA = radiation + 1j * reactance
    Rsky = 0.5 * radiation
    Rmoon = 0.5 * radiation
    return InstrumentResponse.from_arrays(
        freq,
        theta,
        phi,
        Htheta,
        Hphi,
        ZA,
        Rsky,
        Rmoon,
        metadata={
            "SOURCE": "analytic-short-dipoles",
            "SOURCE_ROOT": "lusee.SyntheticResponse",
            "ZA_SOURCE": "analytic-radiation-resistance",
            "INPUT_KIND": "bare",
            "FIELD_KIND": "effective-length",
            "AMP_CONV": "RMS",
            "TIMECONV": "e+jwt",
            "GIT_SHA": "analytic-fixture",
            "COORDSYS": "instrument-topocentric",
            "THETADEF": "colatitude-from-+z",
            "PHIDEF": "right-handed-about-+z",
            "OMEGADEF": "source-arrival-direction",
            "POLBASIS": "e_theta,e_phi",
            "PHASEREF": "analytic-origin",
            "PORTS": "0123",
            "VALIDATED": True,
        },
    )
