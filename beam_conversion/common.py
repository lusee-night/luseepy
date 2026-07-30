"""Shared conversion and FITS-v3 response utilities."""

from dataclasses import dataclass
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
    "FIELD_UNIT",
    "FIELD_AMP",
    "NORM_KIND",
    "NORM_UNIT",
    "NORM_AMP",
    "CANONICAL",
    "LOSSMODEL",
    "RLOSSSRC",
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
    "FIELD_KIND": {"re", "re-per-current", "effective-length"},
    "FIELD_UNIT": {"v", "mv", "v/a", "mv/a", "m"},
    "FIELD_AMP": {"rms", "peak", "ratio"},
    "NORM_KIND": {
        "vsource",
        "current",
        "already-per-ampere",
        "already-effective-length",
    },
    "NORM_UNIT": {"v", "mv", "a", "ma", "not-applicable"},
    "NORM_AMP": {"rms", "peak", "ratio"},
    "CANONICAL": {"h[m],si-rms"},
    "LOSSMODEL": {"pec", "lossy"},
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
    Rloss: np.ndarray | None = None
    Vsource: np.ndarray | None = None
    Inorm: np.ndarray | None = None
    Zref: np.ndarray | None = None
    metadata: dict | None = None


def _as_frequency_grid(freq_mhz):
    freq = np.asarray(freq_mhz, dtype=np.float64).reshape(-1)
    if freq.size == 0 or not np.all(np.isfinite(freq)):
        raise ValueError("Frequency grid must be nonempty and finite.")
    if freq.size > 1 and not np.all(np.diff(freq) > 0):
        raise ValueError("Frequency grid must be strictly increasing.")
    return freq


def compute_sky_moon_resistance(
    freq_mhz,
    theta_deg,
    phi_deg,
    H_theta,
    H_phi,
    ZA,
    Rloss=None,
):
    """Compute native matrices with the simulator's harmonic operator."""
    from lusee.ResponsePhysics import compute_sky_moon_resistance as derive

    return derive(
        freq_mhz,
        theta_deg,
        phi_deg,
        H_theta,
        H_phi,
        ZA,
        Rloss,
    )


def _phasor_to_si_rms(values, *, units, amplitude_convention, factors, name):
    units_key = str(units).strip().lower()
    amplitude = str(amplitude_convention).strip().lower()
    if units_key not in factors:
        raise ValueError(
            f"{name} units must be one of {sorted(factors)}; got {units!r}."
        )
    if amplitude not in {"rms", "peak"}:
        raise ValueError(
            f"{name} amplitude convention must be 'rms' or 'peak'."
        )
    scale = factors[units_key]
    if amplitude == "peak":
        scale /= np.sqrt(2.0)
    return np.asarray(values) * scale


def voltage_phasor_to_si_rms(values, *, units, amplitude_convention):
    """Convert a voltage-like phasor to RMS volts."""
    return _phasor_to_si_rms(
        values,
        units=units,
        amplitude_convention=amplitude_convention,
        factors={"v": 1.0, "mv": 1e-3},
        name="Voltage phasor",
    )


def current_phasor_to_si_rms(values, *, units, amplitude_convention):
    """Convert a current phasor to RMS amperes."""
    return _phasor_to_si_rms(
        values,
        units=units,
        amplitude_convention=amplitude_convention,
        factors={"a": 1.0, "ma": 1e-3},
        name="Current phasor",
    )


def embedded_fields_to_bare(
    E_theta,
    E_phi,
    ZA,
    Zref,
    Vsource,
    *,
    field_units,
    field_amplitude_convention,
    vsource_units,
    vsource_amplitude_convention,
):
    """Recover bare ``rE/I`` in V/A from embedded solver phasors.

    Input field layout is ``(excitation, frequency, theta, phi)``. Field and
    Thevenin-source phasors are independently converted to SI RMS before the
    current solve. The output layout is
    ``(bare_port, frequency, theta, phi)``.
    """
    E_theta = voltage_phasor_to_si_rms(
        E_theta,
        units=field_units,
        amplitude_convention=field_amplitude_convention,
    )
    E_phi = voltage_phasor_to_si_rms(
        E_phi,
        units=field_units,
        amplitude_convention=field_amplitude_convention,
    )
    ZA = np.asarray(ZA)
    Vsource = voltage_phasor_to_si_rms(
        Vsource,
        units=vsource_units,
        amplitude_convention=vsource_amplitude_convention,
    )
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


def bare_fields_to_per_current(
    E_theta,
    E_phi,
    normalization_current,
    *,
    field_units,
    field_amplitude_convention,
    current_units,
    current_amplitude_convention,
):
    """Normalize direct bare ``rE`` exports to SI ``rE/I`` in V/A."""
    E_theta = voltage_phasor_to_si_rms(
        E_theta,
        units=field_units,
        amplitude_convention=field_amplitude_convention,
    )
    E_phi = voltage_phasor_to_si_rms(
        E_phi,
        units=field_units,
        amplitude_convention=field_amplitude_convention,
    )
    current = current_phasor_to_si_rms(
        normalization_current,
        units=current_units,
        amplitude_convention=current_amplitude_convention,
    )
    if E_theta.shape != E_phi.shape or E_theta.ndim != 4:
        raise ValueError(
            "Bare field arrays must share shape "
            "(4, frequency, theta, phi)."
        )
    nport, nfreq = E_theta.shape[:2]
    if nport != 4 or current.shape != (nfreq, nport):
        raise ValueError(
            "normalization_current must have shape (frequency, 4)."
        )
    if np.any(current == 0):
        raise ValueError("normalization_current must be nonzero.")
    denominator = np.swapaxes(current, 0, 1)[..., None, None]
    return E_theta / denominator, E_phi / denominator, current


def convert_fields_to_effective_length(
    fields,
    freq_mhz,
    *,
    field_kind,
    field_units,
):
    """Convert canonical ``rE/I`` or effective lengths to meters."""
    field_kind = str(field_kind).strip().lower().replace("_", "-")
    unit = str(field_units).strip().lower()
    if field_kind not in {"re-per-current", "effective-length"}:
        raise ValueError(
            "field_kind must be 'rE-per-current' or 'effective-length'."
        )
    values = np.asarray(fields)
    if field_kind == "effective-length":
        if unit != "m":
            raise ValueError("Effective-length inputs must have units 'm'.")
        return values
    factors = {"v/a": 1.0, "mv/a": 1e-3}
    if unit not in factors:
        raise ValueError(
            "rE-per-current inputs must have units 'V/A' or 'mV/A'."
        )
    values = values * factors[unit]
    freq = _as_frequency_grid(freq_mhz)
    wave_number = 2 * np.pi * freq * 1e6 / c
    scale = -4 * np.pi / (1j * wave_number * VACUUM_IMPEDANCE_OHM)
    shape = [1] * values.ndim
    shape[1] = freq.size
    return values * scale.reshape(shape)


def response_content_hash(response, *, metadata=None, real_dtype=None):
    """Return the canonical persisted-payload hash."""
    from lusee.ResponsePhysics import response_payload_hash

    return response_payload_hash(
        freq=response.freq_mhz,
        theta_deg=response.theta_deg,
        phi_deg=response.phi_deg,
        H_theta=response.H_theta,
        H_phi=response.H_phi,
        ZA=response.ZA,
        Rsky=response.Rsky,
        Rmoon=response.Rmoon,
        Rloss=response.Rloss,
        Vsource=response.Vsource,
        Inorm=response.Inorm,
        Zref=response.Zref,
        metadata=response.metadata if metadata is None else metadata,
        real_dtype=real_dtype,
    )


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
        from lusee.ResponsePhysics import validate_conversion_metadata

        validate_conversion_metadata(metadata)
    required = {
        "SOURCE": metadata.pop("SOURCE", "UNKNOWN"),
        "SOURCE_ROOT": metadata.pop("SOURCE_ROOT", ""),
        "INPUT_KIND": metadata.pop("INPUT_KIND", "bare"),
        "FIELD_KIND": metadata.pop("FIELD_KIND", "effective-length"),
        "FIELD_UNIT": metadata.pop("FIELD_UNIT", "UNKNOWN"),
        "FIELD_AMP": metadata.pop("FIELD_AMP", "UNKNOWN"),
        "NORM_KIND": metadata.pop("NORM_KIND", "UNKNOWN"),
        "NORM_UNIT": metadata.pop("NORM_UNIT", "UNKNOWN"),
        "NORM_AMP": metadata.pop("NORM_AMP", "UNKNOWN"),
        "CANONICAL": metadata.pop("CANONICAL", "UNKNOWN"),
        "LOSSMODEL": metadata.pop("LOSSMODEL", "UNKNOWN"),
        "RLOSSSRC": metadata.pop("RLOSSSRC", "UNKNOWN"),
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

    metadata = {
        str(key).upper(): value
        for key, value in dict(response.metadata or {}).items()
    }
    loss_model = str(metadata.get("LOSSMODEL", "")).strip().lower()
    if loss_model not in {"pec", "lossy"}:
        raise ValueError(
            "Response metadata must declare LOSSMODEL='PEC' or 'lossy'."
        )
    if response.Rloss is None:
        if loss_model != "pec":
            raise ValueError(
                "A lossy response requires an explicit Rloss matrix."
            )
        response.Rloss = np.zeros_like(response.ZA)
    response.Rloss = np.asarray(response.Rloss)

    computed_rsky, computed_rmoon = compute_sky_moon_resistance(
        response.freq_mhz,
        response.theta_deg,
        response.phi_deg,
        response.H_theta,
        response.H_phi,
        response.ZA,
        response.Rloss,
    )
    if response.Rsky is None:
        response.Rsky = computed_rsky
    if response.Rmoon is None:
        response.Rmoon = computed_rmoon
    response.Rsky = np.asarray(response.Rsky)
    response.Rmoon = np.asarray(response.Rmoon)
    matrix_shape = (response.freq_mhz.size, 4, 4)
    for name in ("Rsky", "Rmoon", "Rloss"):
        if getattr(response, name).shape != matrix_shape:
            raise ValueError(
                f"{name} must have shape (frequency, 4, 4)."
            )
    if not np.all(np.isfinite(response.ZA)):
        raise ValueError("ZA contains non-finite values.")
    if not np.all(np.isfinite(response.Rsky)):
        raise ValueError("Rsky contains non-finite values.")
    if not np.all(np.isfinite(response.Rmoon)):
        raise ValueError("Rmoon contains non-finite values.")
    if not np.all(np.isfinite(response.Rloss)):
        raise ValueError("Rloss contains non-finite values.")
    if validated:
        from lusee.ResponsePhysics import (
            validate_normalization_payload,
            validate_response_matrices,
        )

        if "NORM_KIND" in metadata:
            validate_normalization_payload(
                metadata,
                Vsource=response.Vsource,
                Inorm=response.Inorm,
                Zref=response.Zref,
                nfrequency=response.freq_mhz.size,
            )
        validate_response_matrices(
            response.ZA,
            response.Rsky,
            response.Rmoon,
            response.Rloss,
            field_rsky=computed_rsky,
            loss_model=loss_model,
        )
        return response

    dissipative = 0.5 * (
        response.ZA
        + np.swapaxes(response.ZA.conjugate(), -1, -2)
    )
    if not np.allclose(
        response.Rsky + response.Rmoon + response.Rloss,
        dissipative,
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(
            "Rsky + Rmoon + Rloss does not equal the dissipative part of ZA."
        )
    for name, matrix in (
        ("Rsky", response.Rsky),
        ("Rmoon", response.Rmoon),
        ("Rloss", response.Rloss),
    ):
        if not np.allclose(
            matrix,
            np.swapaxes(matrix.conjugate(), -1, -2),
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(f"{name} must be Hermitian.")
    for name, matrix in (
        ("Rmoon", response.Rmoon),
        ("Rloss", response.Rloss),
    ):
        hermitian = 0.5 * (
            matrix + np.swapaxes(matrix.conjugate(), -1, -2)
        )
        minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(hermitian))
        )
        scale = max(1.0, float(np.max(np.abs(hermitian))))
        if minimum_eigenvalue < -1e-8 * scale:
            warnings.warn(
                f"{name} has a negative eigenvalue "
                f"({minimum_eigenvalue:.6g} Ohm); check response "
                "normalization and ZA provenance.",
                RuntimeWarning,
                stacklevel=2,
            )
    if loss_model == "pec" and not np.allclose(
        response.Rloss,
        0.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("LOSSMODEL='PEC' requires Rloss=0.")
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
    header.pop("MAX_ICOND", None)
    from lusee.ResponsePhysics import normalization_condition_numbers

    def persisted_complex(value):
        if value is None:
            return None
        value = np.asarray(value)
        return (
            value.real.astype(real_dtype)
            + 1j * value.imag.astype(real_dtype)
        )

    condition_numbers = normalization_condition_numbers(
        header,
        ZA=persisted_complex(response.ZA),
        Vsource=persisted_complex(response.Vsource),
        Zref=(
            None
            if response.Zref is None
            else np.asarray(response.Zref).astype(real_dtype)
        ),
    )
    if condition_numbers is not None:
        header["MAX_ICOND"] = float(np.max(condition_numbers))
    header["CONTENT"] = response_content_hash(
        response,
        metadata=header,
        real_dtype=real_dtype,
    )

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
    write_complex("Rloss", response.Rloss, "Ohm")
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
    if response.Inorm is not None:
        write_complex("Inorm", response.Inorm, "A")
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
