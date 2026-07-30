"""Shared physical validation for four-port response artifacts."""

import hashlib
import json
import os

import jax
import numpy as np
from scipy.constants import c, physical_constants


VACUUM_IMPEDANCE_OHM = physical_constants[
    "characteristic impedance of vacuum"
][0]
PORT_PAIRS = tuple((a, b) for a in range(4) for b in range(a, 4))
RESPONSE_HASH_METADATA_KEYS = (
    "VERSION",
    "VALIDATED",
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
    "MAX_ICOND",
)
DEFAULT_RESPONSE_TRANSFORM_MAX_BYTES = 512 * 2**20


def validate_conversion_metadata(metadata):
    """Validate the physical quantity, units, and normalization contract."""
    values = {
        str(key).upper(): str(value).strip().lower().replace("_", "-")
        for key, value in dict(metadata).items()
    }
    input_kind = values.get("INPUT_KIND")
    field_kind = values.get("FIELD_KIND")
    field_unit = values.get("FIELD_UNIT")
    field_amplitude = values.get("FIELD_AMP")
    normalization_kind = values.get("NORM_KIND")
    normalization_unit = values.get("NORM_UNIT")
    normalization_amplitude = values.get("NORM_AMP")
    canonical = values.get("CANONICAL")

    if canonical != "h[m],si-rms":
        raise ValueError(
            "Validated response CANONICAL must be 'H[m],SI-RMS'."
        )
    if input_kind == "embedded":
        expected = {
            "FIELD_KIND": (field_kind, {"re"}),
            "FIELD_UNIT": (field_unit, {"v", "mv"}),
            "FIELD_AMP": (field_amplitude, {"rms", "peak"}),
            "NORM_KIND": (normalization_kind, {"vsource"}),
            "NORM_UNIT": (normalization_unit, {"v", "mv"}),
            "NORM_AMP": (normalization_amplitude, {"rms", "peak"}),
        }
    elif input_kind == "bare" and field_kind == "re":
        expected = {
            "FIELD_UNIT": (field_unit, {"v", "mv"}),
            "FIELD_AMP": (field_amplitude, {"rms", "peak"}),
            "NORM_KIND": (normalization_kind, {"current"}),
            "NORM_UNIT": (normalization_unit, {"a", "ma"}),
            "NORM_AMP": (normalization_amplitude, {"rms", "peak"}),
        }
    elif input_kind == "bare" and field_kind == "re-per-current":
        expected = {
            "FIELD_UNIT": (field_unit, {"v/a", "mv/a"}),
            "FIELD_AMP": (field_amplitude, {"ratio"}),
            "NORM_KIND": (
                normalization_kind,
                {"already-per-ampere"},
            ),
            "NORM_UNIT": (normalization_unit, {"not-applicable"}),
            "NORM_AMP": (normalization_amplitude, {"ratio"}),
        }
    elif input_kind == "bare" and field_kind == "effective-length":
        expected = {
            "FIELD_UNIT": (field_unit, {"m"}),
            "FIELD_AMP": (field_amplitude, {"ratio"}),
            "NORM_KIND": (
                normalization_kind,
                {"already-effective-length"},
            ),
            "NORM_UNIT": (normalization_unit, {"not-applicable"}),
            "NORM_AMP": (normalization_amplitude, {"ratio"}),
        }
    else:
        raise ValueError(
            "Validated response has an unsupported INPUT_KIND/FIELD_KIND "
            f"combination: {input_kind!r}/{field_kind!r}."
        )

    for key, (actual, allowed) in expected.items():
        if actual not in allowed:
            raise ValueError(
                f"Validated response has inconsistent {key}={actual!r}; "
                f"expected one of {sorted(allowed)} for "
                f"{input_kind}/{field_kind}."
            )


def validate_normalization_payload(
    metadata,
    *,
    Vsource,
    Inorm,
    Zref,
    nfrequency,
):
    """Bind normalization provenance to the numerical payload."""
    normalization_kind = str(
        dict(metadata).get("NORM_KIND", "")
    ).strip().lower()
    if normalization_kind == "vsource":
        if Vsource is None or Zref is None or Inorm is not None:
            raise ValueError(
                "NORM_KIND='vsource' requires Vsource and Zref and forbids "
                "Inorm."
            )
        Vsource = np.asarray(Vsource)
        if Vsource.shape != (nfrequency, 4, 4):
            raise ValueError(
                "Vsource must have shape (frequency, 4, 4)."
            )
        if not np.all(np.isfinite(Vsource)):
            raise ValueError("Vsource contains non-finite values.")
        Zref = np.asarray(Zref)
        if Zref.shape != (nfrequency, 4):
            raise ValueError("Zref must have shape (frequency, 4).")
        if not np.all(np.isfinite(Zref)) or np.any(Zref <= 0):
            raise ValueError("Zref must be finite and positive.")
    elif normalization_kind == "current":
        if Inorm is None or Vsource is not None or Zref is not None:
            raise ValueError(
                "NORM_KIND='current' requires Inorm and forbids Vsource "
                "and Zref."
            )
        Inorm = np.asarray(Inorm)
        if Inorm.shape != (nfrequency, 4):
            raise ValueError(
                "Inorm must have shape (frequency, 4)."
            )
        if not np.all(np.isfinite(Inorm)) or np.any(Inorm == 0):
            raise ValueError("Inorm must be finite and nonzero.")
    elif normalization_kind in {
        "already-per-ampere",
        "already-effective-length",
    }:
        if Vsource is not None or Inorm is not None or Zref is not None:
            raise ValueError(
                f"NORM_KIND={normalization_kind!r} forbids Vsource and "
                "Inorm and Zref payloads."
            )
    else:
        raise ValueError(
            f"Unsupported NORM_KIND={normalization_kind!r}."
        )


def _validate_mwss_upper_grid(theta_deg, phi_deg):
    theta = np.asarray(theta_deg, dtype=np.float64)
    phi = np.asarray(phi_deg, dtype=np.float64)
    if theta.ndim != 1 or phi.ndim != 1:
        raise ValueError("theta and phi grids must be one-dimensional.")
    if theta.size < 2 or phi.size < 3:
        raise ValueError("Response angular grids are too short.")
    if (
        not np.isclose(theta[0], 0.0)
        or not np.isclose(theta[-1], 90.0)
        or not np.isclose(phi[0], 0.0)
        or not np.isclose(phi[-1], 360.0)
    ):
        raise ValueError(
            "Response grid must span theta=0..90 and phi=0..360 degrees."
        )
    if not np.allclose(np.diff(theta), np.diff(theta)[0]):
        raise ValueError("Response theta grid must be uniform.")
    if not np.allclose(np.diff(phi), np.diff(phi)[0]):
        raise ValueError("Response phi grid must be uniform.")
    if phi.size - 1 != 4 * (theta.size - 1):
        raise ValueError(
            "Response grid must satisfy Nphi-1 == 4*(Ntheta-1)."
        )
    return theta, phi


def _response_transform_budget():
    raw = os.environ.get(
        "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES",
        str(DEFAULT_RESPONSE_TRANSFORM_MAX_BYTES),
    )
    try:
        result = int(raw)
    except ValueError as error:
        raise ValueError(
            "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES must be an integer."
        ) from error
    if result <= 0:
        raise ValueError(
            "LUSEE_RESPONSE_TRANSFORM_MAX_BYTES must be positive."
        )
    return result


def _response_transform_chunk_shape(
    *,
    nfrequency,
    npair,
    full_theta,
    unique_phi,
    itemsize,
):
    native_lmax = full_theta - 2
    map_elements = 4 * full_theta * unique_phi
    alm_elements = 4 * (native_lmax + 1) * (2 * native_lmax + 1)
    estimated_per_pair_frequency = (
        map_elements + alm_elements
    ) * itemsize * 6
    budget = _response_transform_budget()
    pair_chunk = min(
        npair,
        max(1, budget // max(estimated_per_pair_frequency, 1)),
    )
    frequency_chunk = min(
        nfrequency,
        max(
            1,
            budget
            // max(estimated_per_pair_frequency * pair_chunk, 1),
        ),
    )
    return int(pair_chunk), int(frequency_chunk)


def compute_sky_moon_resistance(
    freq_mhz,
    theta_deg,
    phi_deg,
    H_theta,
    H_phi,
    ZA,
    Rloss=None,
):
    """Derive native sky and Moon resistances with the simulator's SHT.

    The l=0 coefficient is formed by the same Croissant MWSS operator used
    later by :class:`lusee.InstrumentResponse`. This avoids assigning a
    validated resistance with one quadrature and simulating with another.

    ``Rloss`` is the antenna-metal loss matrix. Passing ``None`` is
    equivalent to an explicit zero matrix for analytic PEC calculations.
    """
    import croissant as cro

    freq = np.asarray(freq_mhz, dtype=np.float64)
    if freq.ndim != 1 or freq.size == 0 or not np.all(np.isfinite(freq)):
        raise ValueError("Frequency grid must be nonempty, finite, and 1-D.")
    if freq.size > 1 and not np.all(np.diff(freq) > 0):
        raise ValueError("Frequency grid must be strictly increasing.")
    theta, phi = _validate_mwss_upper_grid(theta_deg, phi_deg)
    H_theta = np.asarray(H_theta)
    H_phi = np.asarray(H_phi)
    ZA = np.asarray(ZA)
    expected = (4, freq.size, theta.size, phi.size)
    if H_theta.shape != expected or H_phi.shape != expected:
        raise ValueError(
            f"Field arrays must have shape {expected}; got "
            f"{H_theta.shape} and {H_phi.shape}."
        )
    if ZA.shape != (freq.size, 4, 4):
        raise ValueError(
            f"ZA must have shape {(freq.size, 4, 4)}; got {ZA.shape}."
        )
    if Rloss is None:
        Rloss = np.zeros_like(ZA)
    else:
        Rloss = np.asarray(Rloss)
    if Rloss.shape != ZA.shape:
        raise ValueError(
            f"Rloss must have shape {ZA.shape}; got {Rloss.shape}."
        )
    if not np.all(np.isfinite(H_theta)) or not np.all(np.isfinite(H_phi)):
        raise ValueError("Response fields contain non-finite values.")
    if not np.all(np.isfinite(ZA)):
        raise ValueError("ZA contains non-finite values.")
    if not np.all(np.isfinite(Rloss)):
        raise ValueError("Rloss contains non-finite values.")

    unique_phi = phi.size - 1
    full_theta = 2 * (theta.size - 1) + 1
    pair_dtype = np.result_type(H_theta.dtype, H_phi.dtype)
    if (
        np.dtype(pair_dtype).itemsize >= 16
        and not jax.config.x64_enabled
    ):
        raise RuntimeError(
            "Validated complex128 response integration requires "
            "JAX_ENABLE_X64=1."
        )
    wavelength_m = c / (freq * 1e6)
    scale = VACUUM_IMPEDANCE_OHM / wavelength_m**2
    Rsky = np.zeros((freq.size, 4, 4), dtype=pair_dtype)
    pair_chunk, frequency_chunk = _response_transform_chunk_shape(
        nfrequency=freq.size,
        npair=len(PORT_PAIRS),
        full_theta=full_theta,
        unique_phi=unique_phi,
        itemsize=np.dtype(pair_dtype).itemsize,
    )
    horizon = np.ones((full_theta, unique_phi), dtype=bool)
    for frequency_start in range(0, freq.size, frequency_chunk):
        frequency_slice = slice(
            frequency_start,
            min(frequency_start + frequency_chunk, freq.size),
        )
        for pair_start in range(0, len(PORT_PAIRS), pair_chunk):
            selected_pairs = PORT_PAIRS[
                pair_start : pair_start + pair_chunk
            ]
            pair_i = np.stack(
                [
                    (
                        H_theta[
                            a,
                            frequency_slice,
                            :,
                            :unique_phi,
                        ]
                        * H_theta[
                            b,
                            frequency_slice,
                            :,
                            :unique_phi,
                        ].conjugate()
                        + H_phi[
                            a,
                            frequency_slice,
                            :,
                            :unique_phi,
                        ]
                        * H_phi[
                            b,
                            frequency_slice,
                            :,
                            :unique_phi,
                        ].conjugate()
                    )
                    for a, b in selected_pairs
                ],
                axis=0,
            )
            full_i = np.zeros(
                pair_i.shape[:-2] + (full_theta, unique_phi),
                dtype=pair_i.dtype,
            )
            full_i[..., : theta.size, :] = pair_i
            stokes_maps = np.stack(
                (
                    full_i,
                    np.zeros_like(full_i),
                    np.zeros_like(full_i),
                    np.zeros_like(full_i),
                ),
                axis=2,
            )
            pair_beam = cro.PairStokesBeam(
                stokes_maps,
                freq[frequency_slice],
                selected_pairs,
                sampling="mwss",
                convention="IAU",
                units="m^2",
                frame="topo",
                tangent_basis="theta-phi",
                baseline_direction="a<=b",
                visibility_definition="<v_a v_b*>",
                horizon=horizon,
            )
            monopole = np.asarray(
                pair_beam.compute_alm(lmax=0)
            )[:, :, 0, 0, 0]
            pair_rsky = (
                0.25
                * monopole
                * scale[frequency_slice][None]
                * np.sqrt(4 * np.pi)
            )
            for pair_position, (a, b) in enumerate(selected_pairs):
                Rsky[frequency_slice, a, b] = pair_rsky[pair_position]
                Rsky[frequency_slice, b, a] = (
                    pair_rsky[pair_position].conjugate()
                )
    dissipative = 0.5 * (ZA + np.swapaxes(ZA.conjugate(), -1, -2))
    Rmoon = dissipative - Rsky - Rloss
    return Rsky, Rmoon


def _validation_tolerances(*arrays):
    real_itemsize = min(
        np.asarray(array).real.dtype.itemsize for array in arrays
    )
    if real_itemsize <= 4:
        return 5e-5, 5e-7
    return 1e-9, 1e-11


def _matrix_scale(value):
    return np.max(np.abs(value), axis=(-2, -1))


def _relative_residual(residual, scale):
    return np.divide(
        residual,
        scale,
        out=np.zeros_like(residual, dtype=np.float64),
        where=scale > 0,
    )


def response_matrix_diagnostics(ZA, Rsky, Rmoon, Rloss):
    """Return per-frequency response-matrix residuals and eigenvalues."""
    ZA = np.asarray(ZA)
    Rsky = np.asarray(Rsky)
    Rmoon = np.asarray(Rmoon)
    Rloss = np.asarray(Rloss)
    if not (
        ZA.shape == Rsky.shape == Rmoon.shape == Rloss.shape
        and ZA.ndim == 3
        and ZA.shape[-2:] == (4, 4)
    ):
        raise ValueError(
            "ZA, Rsky, Rmoon, and Rloss must share shape "
            "(frequency, 4, 4)."
        )

    adjoint = lambda value: np.swapaxes(value.conjugate(), -1, -2)
    dissipative = 0.5 * (ZA + adjoint(ZA))
    closure = Rsky + Rmoon + Rloss - dissipative
    za_scale = _matrix_scale(ZA)
    resistance_scale = np.maximum.reduce(
        (
            _matrix_scale(dissipative),
            _matrix_scale(Rsky),
            _matrix_scale(Rmoon),
            _matrix_scale(Rloss),
        )
    )
    reciprocity = _matrix_scale(ZA - np.swapaxes(ZA, -1, -2))
    closure_error = _matrix_scale(closure)
    result = {
        "za_condition_number": np.linalg.cond(ZA),
        "za_reciprocity_error": reciprocity,
        "za_reciprocity_relative": _relative_residual(
            reciprocity,
            za_scale,
        ),
        "resistance_closure_error": closure_error,
        "resistance_closure_relative": _relative_residual(
            closure_error,
            resistance_scale,
        ),
        "passivity_eigenvalues": np.linalg.eigvalsh(dissipative),
    }
    for label, value in (
        ("sky", Rsky),
        ("moon", Rmoon),
        ("loss", Rloss),
    ):
        scale = _matrix_scale(value)
        hermitian_error = _matrix_scale(value - adjoint(value))
        result[f"{label}_hermitian_error"] = hermitian_error
        result[f"{label}_hermitian_relative"] = _relative_residual(
            hermitian_error,
            scale,
        )
        result[f"{label}_eigenvalues"] = np.linalg.eigvalsh(
            0.5 * (value + adjoint(value))
        )
    return result


def normalization_condition_numbers(
    metadata,
    *,
    ZA,
    Vsource,
    Zref,
):
    """Return embedded-current unmixing condition numbers, when applicable."""
    normalization_kind = str(
        dict(metadata).get("NORM_KIND", "")
    ).strip().lower()
    if normalization_kind != "vsource":
        return None
    ZA = np.asarray(ZA)
    Vsource = np.asarray(Vsource)
    Zref = np.asarray(Zref)
    if (
        ZA.ndim != 3
        or ZA.shape[-2:] != (4, 4)
        or Vsource.shape != ZA.shape
        or Zref.shape != ZA.shape[:2]
    ):
        raise ValueError(
            "Embedded condition diagnostics require ZA/Vsource shape "
            "(frequency, 4, 4) and Zref shape (frequency, 4)."
        )
    load = np.zeros_like(ZA)
    diagonal = np.arange(4)
    load[:, diagonal, diagonal] = Zref
    currents = np.linalg.solve(ZA + load, Vsource)
    return np.linalg.cond(currents)


def validate_response_matrices(
    ZA,
    Rsky,
    Rmoon,
    Rloss,
    *,
    field_rsky=None,
    loss_model=None,
):
    """Validate physical matrices and optionally bind them to the fields."""
    ZA = np.asarray(ZA)
    Rsky = np.asarray(Rsky)
    Rmoon = np.asarray(Rmoon)
    Rloss = np.asarray(Rloss)
    matrices = (
        ("ZA", ZA),
        ("Rsky", Rsky),
        ("Rmoon", Rmoon),
        ("Rloss", Rloss),
    )
    for name, value in matrices:
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values.")
    rtol, atol = _validation_tolerances(ZA, Rsky, Rmoon, Rloss)
    for name, value in (
        ("Rsky", Rsky),
        ("Rmoon", Rmoon),
        ("Rloss", Rloss),
    ):
        if not np.allclose(
            value,
            np.swapaxes(value.conjugate(), -1, -2),
            rtol=rtol,
            atol=atol,
        ):
            raise ValueError(f"{name} must be Hermitian.")
    difference = float(
        np.max(np.abs(ZA - np.swapaxes(ZA, -1, -2)))
    )
    reciprocity_scale = float(np.max(np.abs(ZA)))
    if difference > atol + rtol * reciprocity_scale:
        raise ValueError(
            "ZA must be reciprocal (ZA == ZA.T); maximum residual is "
            f"{difference:.6g} Ohm."
        )
    dissipative = 0.5 * (ZA + np.swapaxes(ZA.conjugate(), -1, -2))
    if not np.allclose(
        Rsky + Rmoon + Rloss,
        dissipative,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(
            "Rsky + Rmoon + Rloss does not equal the dissipative part of ZA."
        )
    for name, value in (
        ("Herm(ZA)", dissipative),
        ("Rsky", Rsky),
        ("Rmoon", Rmoon),
        ("Rloss", Rloss),
    ):
        scale = max(1.0, float(np.max(np.abs(value))))
        minimum = float(
            np.min(
                np.linalg.eigvalsh(
                    0.5
                    * (
                        value
                        + np.swapaxes(value.conjugate(), -1, -2)
                    )
                )
            )
        )
        if minimum < -max(1e-8, 10 * atol) * scale:
            raise ValueError(
                f"{name} has a negative eigenvalue "
                f"({minimum:.6g} Ohm); response is not physically validated."
            )
    if str(loss_model).strip().lower() == "pec" and not np.allclose(
        Rloss,
        0.0,
        rtol=0.0,
        atol=atol,
    ):
        maximum = float(np.max(np.abs(Rloss)))
        raise ValueError(
            "LOSSMODEL='PEC' requires Rloss=0; "
            f"maximum magnitude is {maximum:.6g} Ohm."
        )
    if field_rsky is not None:
        field_rsky = np.asarray(field_rsky)
        if not np.allclose(
            Rsky,
            field_rsky,
            rtol=rtol,
            atol=atol,
        ):
            difference = float(np.max(np.abs(Rsky - field_rsky)))
            raise ValueError(
                "Stored Rsky is inconsistent with H_theta/H_phi under "
                f"the simulator SHT (max difference {difference:.6g} Ohm)."
            )


def _canonical_array(array):
    value = np.asarray(array)
    dtype = value.dtype.newbyteorder("<")
    return np.ascontiguousarray(value.astype(dtype, copy=False))


def _update_array_hash(digest, name, value):
    digest.update(name.encode("ascii"))
    if value is None:
        digest.update(b"<absent>")
        return
    array = _canonical_array(value)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def response_payload_hash(
    *,
    freq,
    theta_deg,
    phi_deg,
    H_theta,
    H_phi,
    ZA,
    Rsky,
    Rmoon,
    Rloss,
    metadata,
    Vsource=None,
    Inorm=None,
    Zref=None,
    real_dtype=None,
):
    """Hash canonical persisted numerical content and semantic metadata."""
    if real_dtype is not None:
        real_dtype = np.dtype(real_dtype)

        def persisted_complex(value):
            value = np.asarray(value)
            return (
                value.real.astype(real_dtype)
                + 1j * value.imag.astype(real_dtype)
            )

        H_theta = persisted_complex(H_theta)
        H_phi = persisted_complex(H_phi)
        ZA = persisted_complex(ZA)
        Rsky = persisted_complex(Rsky)
        Rmoon = persisted_complex(Rmoon)
        Rloss = persisted_complex(Rloss)
        if Vsource is not None:
            Vsource = persisted_complex(Vsource)
        if Inorm is not None:
            Inorm = persisted_complex(Inorm)
        if Zref is not None:
            Zref = np.asarray(Zref).astype(real_dtype)

    digest = hashlib.sha256()
    for name, value in (
        ("freq", np.asarray(freq, dtype=np.float64)),
        ("theta_deg", np.asarray(theta_deg, dtype=np.float64)),
        ("phi_deg", np.asarray(phi_deg, dtype=np.float64)),
        ("H_theta", H_theta),
        ("H_phi", H_phi),
        ("ZA", ZA),
        ("Rsky", Rsky),
        ("Rmoon", Rmoon),
        ("Rloss", Rloss),
        ("Vsource", Vsource),
        ("Inorm", Inorm),
        ("Zref", Zref),
    ):
        _update_array_hash(digest, name, value)
    normalized_metadata = {}
    metadata = {str(key).upper(): value for key, value in metadata.items()}
    for key in RESPONSE_HASH_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        normalized_metadata[key] = value
    digest.update(
        json.dumps(
            normalized_metadata,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()
