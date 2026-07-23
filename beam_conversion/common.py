"""Shared conversion and FITS-v3 response utilities."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import fitsio
import numpy as np
from scipy.constants import Boltzmann, c, physical_constants


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]
K_BOLTZMANN = Boltzmann
CANONICAL_PORT_PAIRS = tuple(
    (a, b) for a in range(4) for b in range(a, 4)
)
REQUIRED_PROVENANCE_KEYS = (
    "SOURCE",
    "SOURCE_ROOT",
    "INPUT_KIND",
    "FIELD_KIND",
    "AMP_CONV",
    "TIMECONV",
    "ZA_SOURCE",
    "GIT_SHA",
    "COORDSYS",
    "THETADEF",
    "PHIDEF",
    "OMEGADEF",
    "POLBASIS",
    "PHASEREF",
    "PORTS",
)
CANONICAL_CONVENTIONS = {
    "TIMECONV": {"e+jwt"},
    "COORDSYS": {"instrument-topocentric"},
    "THETADEF": {"colatitude-from-+z"},
    "PHIDEF": {"right-handed-about-+z"},
    "OMEGADEF": {"source-arrival-direction"},
    "POLBASIS": {"e_theta,e_phi"},
    "PORTS": {"0123"},
    "INPUT_KIND": {"bare", "embedded"},
    "FIELD_KIND": {"re", "r_e", "effective-length", "effective_length"},
    "AMP_CONV": {"rms", "peak"},
}


@dataclass
class ResponseArrays:
    """Numerical and metadata payload for one four-port response."""

    freq_mhz: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    H_theta: np.ndarray
    H_phi: np.ndarray
    ZA: np.ndarray
    Rsky: np.ndarray | None = None
    Rmoon: np.ndarray | None = None
    Vsource: np.ndarray | None = None
    Zref: np.ndarray | None = None
    metadata: dict | None = None


def _as_frequency_grid(freq_mhz):
    freq = np.asarray(freq_mhz, dtype=np.float64).reshape(-1)
    if freq.size == 0 or not np.all(np.isfinite(freq)):
        raise ValueError("Frequency grid must be nonempty and finite.")
    if freq.size > 1 and not np.all(np.diff(freq) > 0):
        raise ValueError("Frequency grid must be strictly increasing.")
    return freq


def _grid_quadrature(theta_deg, phi_deg):
    theta = np.radians(np.asarray(theta_deg, dtype=np.float64))
    phi = np.radians(np.asarray(phi_deg, dtype=np.float64))
    if theta.ndim != 1 or phi.ndim != 1:
        raise ValueError("theta and phi must be one-dimensional.")
    if theta.size < 2 or phi.size < 2:
        raise ValueError("theta and phi grids require at least two samples.")
    if not np.all(np.diff(theta) > 0) or not np.all(np.diff(phi) > 0):
        raise ValueError("theta and phi grids must be strictly increasing.")

    has_wrap = np.isclose(phi[-1] - phi[0], 2 * np.pi)
    phi_use = phi[:-1] if has_wrap else phi
    if phi_use.size < 2:
        raise ValueError("phi grid has too few unique samples.")
    dphi = 2 * np.pi / phi_use.size if has_wrap else np.mean(np.diff(phi_use))
    if not np.allclose(np.diff(phi_use), dphi):
        raise ValueError("Only uniform phi grids are supported.")

    theta_weights = np.empty_like(theta)
    theta_weights[0] = 0.5 * (theta[1] - theta[0])
    theta_weights[-1] = 0.5 * (theta[-1] - theta[-2])
    if theta.size > 2:
        theta_weights[1:-1] = 0.5 * (theta[2:] - theta[:-2])
    theta_weights *= np.sin(theta)
    weights = theta_weights[:, None] * dphi
    return weights, phi_use.size


def compute_sky_moon_resistance(
    freq_mhz,
    theta_deg,
    phi_deg,
    H_theta,
    H_phi,
    ZA,
):
    """Compute native Rsky and Rmoon with endpoint-aware quadrature."""
    freq = _as_frequency_grid(freq_mhz)
    H_theta = np.asarray(H_theta)
    H_phi = np.asarray(H_phi)
    ZA = np.asarray(ZA)
    expected = (4, freq.size, len(theta_deg), len(phi_deg))
    if H_theta.shape != expected or H_phi.shape != expected:
        raise ValueError(
            f"Field arrays must have shape {expected}; got "
            f"{H_theta.shape} and {H_phi.shape}."
        )
    if ZA.shape != (freq.size, 4, 4):
        raise ValueError(
            f"ZA must have shape {(freq.size, 4, 4)}; got {ZA.shape}."
        )

    weights, nphi = _grid_quadrature(theta_deg, phi_deg)
    Ht = H_theta[..., :nphi]
    Hp = H_phi[..., :nphi]
    pair_integral = np.einsum(
        "aftx,bftx,tx->fab",
        Ht,
        Ht.conjugate(),
        weights,
        optimize=True,
    )
    pair_integral += np.einsum(
        "aftx,bftx,tx->fab",
        Hp,
        Hp.conjugate(),
        weights,
        optimize=True,
    )
    wavelength_m = c / (freq * 1e6)
    Rsky = (
        VACUUM_IMPEDANCE_OHM
        / (4.0 * wavelength_m**2)
    )[:, None, None] * pair_integral
    dissipative = 0.5 * (ZA + np.swapaxes(ZA.conjugate(), -1, -2))
    Rmoon = dissipative - Rsky
    return Rsky, Rmoon


def embedded_fields_to_bare(
    E_theta,
    E_phi,
    ZA,
    Zref,
    Vsource,
):
    """Remove the embedded excitation-current basis using right-side solves.

    Input field layout is ``(excitation, frequency, theta, phi)``. The output
    layout is ``(bare_port, frequency, theta, phi)``.
    """
    E_theta = np.asarray(E_theta)
    E_phi = np.asarray(E_phi)
    ZA = np.asarray(ZA)
    Vsource = np.asarray(Vsource)
    nfreq, nport, _ = ZA.shape
    if nport != 4 or ZA.shape != (nfreq, nport, nport):
        raise ValueError("ZA must have shape (frequency, 4, 4).")
    if Vsource.shape != ZA.shape:
        raise ValueError("Vsource must have the same shape as ZA.")

    Zref = np.asarray(Zref)
    if Zref.ndim == 0:
        Zref = np.full((nfreq, nport), Zref)
    elif Zref.ndim == 1 and Zref.size == nport:
        Zref = np.broadcast_to(Zref[None], (nfreq, nport))
    if Zref.shape != (nfreq, nport):
        raise ValueError("Zref must be scalar, per-port, or frequency-by-port.")

    load = np.zeros_like(ZA)
    diagonal = np.arange(nport)
    load[:, diagonal, diagonal] = Zref
    I_sim = np.linalg.solve(ZA + load, Vsource)

    def right_solve(fields):
        if fields.shape[:2] != (nport, nfreq):
            raise ValueError(
                "Embedded field arrays must have shape "
                "(4, frequency, theta, phi)."
            )
        rows = np.moveaxis(fields, (0, 1), (-1, 0))
        lhs = np.swapaxes(I_sim, -1, -2)[:, None, None]
        solved = np.linalg.solve(lhs, rows[..., None])[..., 0]
        return np.moveaxis(solved, (0, -1), (1, 0))

    return right_solve(E_theta), right_solve(E_phi), I_sim


def convert_fields_to_effective_length(
    fields,
    freq_mhz,
    *,
    field_kind,
    amplitude_convention,
):
    """Convert solver field samples to RMS effective lengths in meters."""
    field_kind = str(field_kind).lower().replace("-", "_")
    amplitude = str(amplitude_convention).lower()
    if field_kind not in {"re", "r_e", "effective_length"}:
        raise ValueError(
            "field_kind must be 'rE' or 'effective-length'."
        )
    if amplitude not in {"rms", "peak"}:
        raise ValueError("amplitude_convention must be 'rms' or 'peak'.")
    values = np.asarray(fields)
    if amplitude == "peak":
        values = values / np.sqrt(2.0)
    if field_kind == "effective_length":
        return values
    freq = _as_frequency_grid(freq_mhz)
    wave_number = 2 * np.pi * freq * 1e6 / c
    scale = -4 * np.pi / (1j * wave_number * VACUUM_IMPEDANCE_OHM)
    shape = [1] * values.ndim
    shape[1] = freq.size
    return values * scale.reshape(shape)


def response_content_hash(response):
    """Return a stable content hash for cache/provenance keys."""
    digest = hashlib.sha256()
    for value in (
        response.freq_mhz,
        response.theta_deg,
        response.phi_deg,
        response.H_theta,
        response.H_phi,
        response.ZA,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _response_header(response, validated):
    metadata = {
        str(key).upper(): value
        for key, value in dict(response.metadata or {}).items()
    }
    if validated:
        unknown_values = {"", "unknown", "unspecified", "none"}
        missing = [
            key
            for key in REQUIRED_PROVENANCE_KEYS
            if key not in metadata
            or str(metadata[key]).strip().lower() in unknown_values
        ]
        if missing:
            raise ValueError(
                "VALIDATED=True requires explicit response provenance for: "
                + ", ".join(missing)
            )
        for key, allowed in CANONICAL_CONVENTIONS.items():
            value = str(metadata[key]).strip().lower()
            if value not in allowed:
                raise ValueError(
                    f"VALIDATED=True has unsupported {key}={metadata[key]!r}; "
                    f"expected one of {sorted(allowed)}."
                )
    required = {
        "SOURCE": metadata.pop("SOURCE", "UNKNOWN"),
        "SOURCE_ROOT": metadata.pop("SOURCE_ROOT", ""),
        "INPUT_KIND": metadata.pop("INPUT_KIND", "bare"),
        "FIELD_KIND": metadata.pop("FIELD_KIND", "effective-length"),
        "AMP_CONV": metadata.pop("AMP_CONV", "RMS"),
        "TIMECONV": metadata.pop("TIMECONV", "e+jwt"),
        "ZA_SOURCE": metadata.pop("ZA_SOURCE", "UNKNOWN"),
        "GIT_SHA": metadata.pop("GIT_SHA", "UNKNOWN"),
        "COORDSYS": metadata.pop("COORDSYS", "instrument-topocentric"),
        "THETADEF": metadata.pop("THETADEF", "colatitude-from-+z"),
        "PHIDEF": metadata.pop("PHIDEF", "right-handed-about-+z"),
        "OMEGADEF": metadata.pop("OMEGADEF", "source-arrival-direction"),
        "POLBASIS": metadata.pop("POLBASIS", "e_theta,e_phi"),
        "PHASEREF": metadata.pop("PHASEREF", "solver-origin"),
    }
    header = {
        "VERSION": 3,
        "PORTS": metadata.pop("PORTS", "0123"),
        "VALIDATED": bool(validated),
        "FREQ_N": response.freq_mhz.size,
        "FREQ_MIN": float(response.freq_mhz.min()),
        "FREQ_MAX": float(response.freq_mhz.max()),
        "THETA_N": response.theta_deg.size,
        "THETA_MIN": float(response.theta_deg.min()),
        "THETA_MAX": float(response.theta_deg.max()),
        "PHI_N": response.phi_deg.size,
        "PHI_MIN": float(response.phi_deg.min()),
        "PHI_MAX": float(response.phi_deg.max()),
        **required,
    }
    reserved_keys = set(header) | {"CONTENT"}
    for key, value in metadata.items():
        header_key = str(key).upper()
        if (
            header_key not in reserved_keys
            and isinstance(value, (str, int, float, bool, np.generic))
        ):
            header[header_key] = (
                value.item() if isinstance(value, np.generic) else value
            )
    return header


def _validate_response(response, *, validated):
    response.freq_mhz = _as_frequency_grid(response.freq_mhz)
    response.theta_deg = np.asarray(response.theta_deg, dtype=np.float64)
    response.phi_deg = np.asarray(response.phi_deg, dtype=np.float64)
    if validated:
        if response.theta_deg.size < 2 or response.phi_deg.size < 3:
            raise ValueError(
                "Validated response grids need at least two theta samples "
                "and three stored phi samples including the wrap."
            )
        if (
            not np.isclose(response.theta_deg[0], 0.0)
            or not np.isclose(response.theta_deg[-1], 90.0)
        ):
            raise ValueError(
                "Validated response theta grid must span 0 through 90 deg."
            )
        if not np.allclose(
            np.diff(response.theta_deg),
            np.diff(response.theta_deg)[0],
        ):
            raise ValueError(
                "Validated response theta grid must be uniform."
            )
        if (
            not np.isclose(response.phi_deg[0], 0.0)
            or not np.isclose(response.phi_deg[-1], 360.0)
        ):
            raise ValueError(
                "Validated response phi grid must retain the 0/360 wrap."
            )
        if not np.allclose(
            np.diff(response.phi_deg),
            np.diff(response.phi_deg)[0],
        ):
            raise ValueError(
                "Validated response phi grid must be uniform."
            )
        expected_phi_count = 4 * (response.theta_deg.size - 1) + 1
        if response.phi_deg.size != expected_phi_count:
            raise ValueError(
                "Validated response angular grid must satisfy "
                "Nphi-1 == 4*(Ntheta-1) for full-sphere MWSS padding."
            )
    expected = (
        4,
        response.freq_mhz.size,
        response.theta_deg.size,
        response.phi_deg.size,
    )
    response.H_theta = np.asarray(response.H_theta)
    response.H_phi = np.asarray(response.H_phi)
    response.ZA = np.asarray(response.ZA)
    if response.H_theta.shape != expected or response.H_phi.shape != expected:
        raise ValueError(f"Response fields must have shape {expected}.")
    if response.ZA.shape != (response.freq_mhz.size, 4, 4):
        raise ValueError("ZA must have shape (frequency, 4, 4).")
    if not np.all(np.isfinite(response.H_theta)):
        raise ValueError("H_theta contains non-finite values.")
    if not np.all(np.isfinite(response.H_phi)):
        raise ValueError("H_phi contains non-finite values.")

    computed_rsky, computed_rmoon = compute_sky_moon_resistance(
        response.freq_mhz,
        response.theta_deg,
        response.phi_deg,
        response.H_theta,
        response.H_phi,
        response.ZA,
    )
    if response.Rsky is None:
        response.Rsky = computed_rsky
    if response.Rmoon is None:
        response.Rmoon = computed_rmoon
    response.Rsky = np.asarray(response.Rsky)
    response.Rmoon = np.asarray(response.Rmoon)
    matrix_shape = (response.freq_mhz.size, 4, 4)
    if response.Rsky.shape != matrix_shape or response.Rmoon.shape != matrix_shape:
        raise ValueError("Rsky and Rmoon must have shape (frequency, 4, 4).")
    if not np.all(np.isfinite(response.ZA)):
        raise ValueError("ZA contains non-finite values.")
    if not np.all(np.isfinite(response.Rsky)):
        raise ValueError("Rsky contains non-finite values.")
    if not np.all(np.isfinite(response.Rmoon)):
        raise ValueError("Rmoon contains non-finite values.")
    dissipative = 0.5 * (
        response.ZA
        + np.swapaxes(response.ZA.conjugate(), -1, -2)
    )
    if not np.allclose(
        response.Rsky + response.Rmoon,
        dissipative,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(
            "Rsky + Rmoon does not equal the dissipative part of ZA."
        )
    for name, matrix in (
        ("Rsky", response.Rsky),
        ("Rmoon", response.Rmoon),
    ):
        if not np.allclose(
            matrix,
            np.swapaxes(matrix.conjugate(), -1, -2),
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(f"{name} must be Hermitian.")
    moon_hermitian = 0.5 * (
        response.Rmoon
        + np.swapaxes(response.Rmoon.conjugate(), -1, -2)
    )
    minimum_moon_eigenvalue = float(
        np.min(np.linalg.eigvalsh(moon_hermitian))
    )
    scale = max(
        1.0,
        float(np.max(np.abs(moon_hermitian))),
    )
    if minimum_moon_eigenvalue < -1e-8 * scale:
        message = (
            "Rmoon has a negative eigenvalue "
            f"({minimum_moon_eigenvalue:.6g} Ohm); check response "
            "normalization and ZA provenance."
        )
        if validated:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return response


def write_response_fits(
    filename,
    response,
    *,
    dtype="float32",
    validated=True,
):
    """Write one machine-tagged instrument response FITS v3 file."""
    response = _validate_response(response, validated=validated)
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'.")
    real_dtype = np.dtype(dtype)
    header = _response_header(response, validated)
    header["CONTENT"] = response_content_hash(response)

    filename = str(Path(filename))
    fits = fitsio.FITS(filename, "rw", clobber=True)

    def write_complex(name, value, units, primary_header=None):
        value = np.asarray(value)
        real_header = {"BUNIT": units}
        if primary_header:
            real_header.update(primary_header)
        fits.write(
            value.real.astype(real_dtype),
            extname=f"{name}_real",
            header=real_header,
        )
        fits.write(
            value.imag.astype(real_dtype),
            extname=f"{name}_imag",
            header={"BUNIT": units},
        )

    write_complex("H_theta", response.H_theta, "m", header)
    write_complex("H_phi", response.H_phi, "m")
    write_complex("ZA", response.ZA, "Ohm")
    write_complex("Rsky", response.Rsky, "Ohm")
    write_complex("Rmoon", response.Rmoon, "Ohm")
    fits.write(
        np.asarray(response.freq_mhz, dtype=np.float64),
        extname="freq",
        header={"BUNIT": "MHz"},
    )
    fits.write(
        np.asarray(response.theta_deg, dtype=np.float64),
        extname="theta",
        header={"BUNIT": "deg"},
    )
    fits.write(
        np.asarray(response.phi_deg, dtype=np.float64),
        extname="phi",
        header={"BUNIT": "deg"},
    )
    if response.Vsource is not None:
        write_complex("Vsource", response.Vsource, "V")
    if response.Zref is not None:
        fits.write(
            np.asarray(response.Zref, dtype=real_dtype),
            extname="Zref",
            header={"BUNIT": "Ohm"},
        )
    provenance = json.dumps(response.metadata or {}, sort_keys=True).encode(
        "utf-8"
    )
    fits.write(
        np.frombuffer(provenance, dtype=np.uint8),
        extname="provenance_json",
        header={"BUNIT": "1"},
    )
    fits.close()
    return filename
