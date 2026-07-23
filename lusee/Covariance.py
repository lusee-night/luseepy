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
    T_moon,
):
    """Assemble open-circuit covariance from pair integrals and Moon term."""
    pair_sky_integrals = jnp.asarray(pair_sky_integrals)
    K_sky = K_BOLTZMANN * assemble_pair_matrix(
        pair_sky_integrals,
        PORT_PAIRS,
    )
    K_moon = 4 * K_BOLTZMANN * jnp.asarray(T_moon) * jnp.asarray(Rmoon)
    return K_sky + K_moon[None]


@jax.jit
def load_covariance(open_covariance, ZA, ZL):
    """Apply receiver loading as ``M K M^dagger``."""
    M = loading_matrix(ZA, ZL)
    covariance = jnp.einsum(
        "fab,tfbc,fdc->tfad",
        M,
        open_covariance,
        M.conjugate(),
    )
    covariance = 0.5 * (
        covariance + jnp.swapaxes(covariance.conjugate(), -1, -2)
    )
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
