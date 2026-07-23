"""Tests for receiver loading and four-port covariance kernels."""

import jax
import jax.numpy as jnp
import numpy as np

from lusee.Covariance import (
    assemble_open_covariance,
    blackbody_normalization,
    default_product_labels,
    load_covariance,
    pack_covariance,
)
from lusee.MapMaker import compute_radiometric_noise
from lusee.InstrumentResponse import PORT_PAIRS
from lusee.ReceiverImpedance import (
    IdealCapacitorReceiver,
    JFETReceiver,
    MeasuredReceiver,
    loading_matrix,
)


def random_well_conditioned_matrices(seed=2, nfreq=3):
    rng = np.random.default_rng(seed)
    ZA = rng.normal(size=(nfreq, 4, 4)) + 1j * rng.normal(
        size=(nfreq, 4, 4)
    )
    ZL = rng.normal(size=(nfreq, 4, 4)) + 1j * rng.normal(
        size=(nfreq, 4, 4)
    )
    ZA += 10 * np.eye(4)[None]
    ZL += 20 * np.eye(4)[None]
    return jnp.asarray(ZA), jnp.asarray(ZL)


def test_loading_matrix_is_right_solve_for_noncommuting_batches():
    ZA, ZL = random_well_conditioned_matrices()
    M = loading_matrix(ZA, ZL)
    assert jnp.allclose(jnp.einsum("fab,fbc->fac", M, ZA + ZL), ZL)
    assert not jnp.allclose(
        M,
        jnp.linalg.solve(ZA + ZL, ZL),
    )


def test_receiver_models_accept_arbitrary_target_arrays():
    target = np.asarray([17.5, 10.0, 17.5])
    assert JFETReceiver().Z(target).shape == (3, 4, 4)
    assert IdealCapacitorReceiver().Z(target).shape == (3, 4, 4)
    native = np.asarray([10.0, 20.0])
    values = np.stack((10 * np.eye(4), 20 * np.eye(4))).astype(complex)
    measured = MeasuredReceiver(native, values)
    result = measured.Z(target)
    assert jnp.allclose(result[0], 17.5 * jnp.eye(4))
    assert jnp.array_equal(result[0], result[2])


def test_blackbody_identity_native_and_off_grid():
    nfreq = 3
    temperature = 247.0
    ZA, ZL = random_well_conditioned_matrices(nfreq=nfreq)
    dissipative = 0.5 * (ZA + jnp.swapaxes(ZA.conjugate(), -1, -2))
    Rsky = 0.35 * dissipative
    Rmoon = dissipative - Rsky
    pair_values = []
    for a, b in PORT_PAIRS:
        pair_values.append(4 * temperature * Rsky[:, a, b])
    pair_values = jnp.stack(pair_values, axis=-1)[None]
    open_covariance = assemble_open_covariance(
        pair_values,
        Rmoon,
        temperature,
    )
    covariance, M = load_covariance(open_covariance, ZA, ZL)
    expected = temperature * blackbody_normalization(ZA, M)
    assert jnp.allclose(covariance[0], expected, rtol=1e-11, atol=1e-25)


def test_covariance_is_hermitian_and_packs_16_real_channels():
    ZA, ZL = random_well_conditioned_matrices(nfreq=2)
    Rmoon = jnp.broadcast_to(jnp.eye(4)[None], (2, 4, 4))
    pair_values = jnp.zeros((3, 2, 10), dtype=jnp.complex128)
    open_covariance = assemble_open_covariance(pair_values, Rmoon, 250.0)
    covariance, _ = load_covariance(open_covariance, ZA, ZL)
    packed, labels = pack_covariance(covariance)
    assert packed.shape == (3, 16, 2)
    assert len(labels) == 16
    assert jnp.isrealobj(packed)
    assert jnp.allclose(
        covariance,
        jnp.swapaxes(covariance.conjugate(), -1, -2),
    )


def test_receiver_parameter_gradient_is_finite():
    freq = jnp.asarray([10.0, 17.5])
    ZA = jnp.broadcast_to((30.0 + 4.0j) * jnp.eye(4)[None], (2, 4, 4))

    def loss(C_pf):
        receiver = JFETReceiver(C_pf=C_pf)
        M = loading_matrix(ZA, receiver.Z(freq))
        return jnp.real(jnp.sum(jnp.abs(M) ** 2))

    gradient = jax.grad(loss)(jnp.asarray([35.0, 36.0, 37.0, 38.0]))
    assert gradient.shape == (4,)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(jnp.abs(gradient) > 0)
    epsilon = 1e-3
    base = jnp.asarray([35.0, 36.0, 37.0, 38.0])
    direction = jnp.asarray([1.0, -0.5, 0.25, -0.75])
    finite_difference = (
        loss(base + epsilon * direction)
        - loss(base - epsilon * direction)
    ) / (2 * epsilon)
    autodiff_directional = jnp.vdot(gradient, direction)
    assert jnp.allclose(
        autodiff_directional,
        finite_difference,
        rtol=2e-5,
        atol=2e-8,
    )


def test_mapmaker_noise_uses_response_v3_product_order():
    labels = default_product_labels()
    data = np.zeros((1, len(labels), 1))
    channel = {label: index for index, label in enumerate(labels)}
    for port, value in enumerate((2.0, 3.0, 5.0, 7.0)):
        data[:, channel[f"{port}{port}R"], :] = value
    data[:, channel["01R"], :] = 0.4
    data[:, channel["01I"], :] = -0.3

    sigma = compute_radiometric_noise(
        data,
        delta_f_hz=2.0,
        delta_t_sec=5.0,
    )
    np.testing.assert_allclose(
        sigma[:, channel["00R"], :],
        2.0 / np.sqrt(10.0),
    )
    cross = np.sqrt((2.0 * 3.0 + 0.4**2 + 0.3**2) / 40.0)
    np.testing.assert_allclose(sigma[:, channel["01R"], :], cross)
    np.testing.assert_allclose(sigma[:, channel["01I"], :], cross)
