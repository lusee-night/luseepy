"""Four-port open-circuit and loaded covariance kernels."""

import jax
import jax.numpy as jnp
from scipy.constants import Boltzmann

from .InstrumentResponse import PORT_PAIRS, assemble_pair_matrix
from .ReceiverImpedance import loading_matrix


K_BOLTZMANN = Boltzmann


def default_product_labels():
    """Return four autos and real/imaginary channels for six crosses."""
    labels = []
    for a in range(4):
        for b in range(a, 4):
            if a == b:
                labels.append(f"{a}{b}R")
            else:
                labels.extend((f"{a}{b}R", f"{a}{b}I"))
    return tuple(labels)


def normalize_products(products):
    """Normalize ``all`` or pair/channel requests to explicit labels."""
    if products == "all":
        return default_product_labels()
    labels = []
    for product in products:
        if isinstance(product, str):
            label = product.upper()
            if len(label) == 2:
                a, b = int(label[0]), int(label[1])
                labels.extend(
                    (f"{a}{b}R",)
                    if a == b
                    else (f"{a}{b}R", f"{a}{b}I")
                )
            elif len(label) == 3 and label[2] in {"R", "I"}:
                labels.append(label)
            else:
                raise ValueError(f"Invalid product label {product!r}.")
        else:
            a, b = (int(value) for value in product)
            labels.extend(
                (f"{a}{b}R",)
                if a == b
                else (f"{a}{b}R", f"{a}{b}I")
            )
    for label in labels:
        a, b = int(label[0]), int(label[1])
        if not (0 <= a <= b < 4):
            raise ValueError(f"Invalid product ports in {label!r}.")
        if a == b and label[2] == "I":
            raise ValueError("Auto products do not have an imaginary channel.")
    return tuple(labels)


@jax.jit
def assemble_open_covariance(
    pair_sky_integrals,
    Rmoon,
    Rloss,
    *,
    T_moon,
    T_ant,
):
    """Assemble open covariance from sky, Moon, and antenna-metal terms."""
    pair_sky_integrals = jnp.asarray(pair_sky_integrals)
    K_sky = K_BOLTZMANN * assemble_pair_matrix(
        pair_sky_integrals,
        PORT_PAIRS,
    )
    K_moon = 4 * K_BOLTZMANN * jnp.asarray(T_moon) * jnp.asarray(Rmoon)
    K_ant = 4 * K_BOLTZMANN * jnp.asarray(T_ant) * jnp.asarray(Rloss)
    return K_sky + K_moon[None] + K_ant[None]


@jax.jit
def apply_receiver_loading(open_covariance, ZA, ZL):
    """Apply receiver loading and retain the unprojected covariance."""
    M = loading_matrix(ZA, ZL)
    covariance = jnp.einsum(
        "fab,tfbc,fdc->tfad",
        M,
        open_covariance,
        M.conjugate(),
    )
    return covariance, M


@jax.jit
def project_hermitian(matrix):
    """Project a matrix batch onto its Hermitian part."""
    return 0.5 * (
        matrix + jnp.swapaxes(matrix.conjugate(), -1, -2)
    )


@jax.jit
def covariance_projection_diagnostics(unprojected_covariance):
    """Measure projection residuals and eigenvalues without PSD clipping."""
    adjoint = jnp.swapaxes(
        unprojected_covariance.conjugate(),
        -1,
        -2,
    )
    absolute = jnp.max(
        jnp.abs(unprojected_covariance - adjoint),
        axis=(-2, -1),
    )
    scale = jnp.max(
        jnp.abs(unprojected_covariance),
        axis=(-2, -1),
    )
    safe_scale = jnp.where(scale > 0, scale, 1.0)
    relative = jnp.where(scale > 0, absolute / safe_scale, 0.0)
    eigenvalues = jnp.linalg.eigvalsh(
        project_hermitian(unprojected_covariance)
    )
    eigenvalue_scale = jnp.max(jnp.abs(eigenvalues), axis=-1)
    safe_eigenvalue_scale = jnp.where(
        eigenvalue_scale > 0,
        eigenvalue_scale,
        1.0,
    )
    minimum_eigenvalue_ratio = jnp.where(
        eigenvalue_scale > 0,
        eigenvalues[..., 0] / safe_eigenvalue_scale,
        0.0,
    )
    return {
        "antihermitian_absolute": absolute,
        "antihermitian_relative": relative,
        "eigenvalues": eigenvalues,
        "minimum_eigenvalue_ratio": minimum_eigenvalue_ratio,
    }


@jax.jit
def matrix_condition_number(matrix):
    """Return the spectral condition number of each final-axis matrix."""
    return jnp.linalg.cond(matrix)


@jax.jit
def load_covariance(open_covariance, ZA, ZL):
    """Apply receiver loading and project the result onto Hermitian matrices."""
    covariance, M = apply_receiver_loading(open_covariance, ZA, ZL)
    covariance = project_hermitian(covariance)
    return covariance, M


@jax.jit
def blackbody_normalization(ZA, M):
    """Return the covariance response to a one-kelvin blackbody enclosure."""
    dissipative = 0.5 * (
        ZA + jnp.swapaxes(ZA.conjugate(), -1, -2)
    )
    return 4 * K_BOLTZMANN * jnp.einsum(
        "fab,fbc,fdc->fad",
        M,
        dissipative,
        M.conjugate(),
    )


def pack_covariance(covariance, products="all"):
    """Pack Hermitian covariance into real science channels."""
    labels = normalize_products(products)
    channels = []
    for label in labels:
        a, b = int(label[0]), int(label[1])
        value = covariance[..., a, b]
        channels.append(value.real if label[2] == "R" else value.imag)
    return jnp.stack(channels, axis=-2), labels
