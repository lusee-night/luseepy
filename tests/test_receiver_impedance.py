"""Tests for the JFET receiver channel maps and the preamp fit tables."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lusee.preamp_fits import (
    FLIGHT_CHANNEL_TO_PREAMP,
    FLIGHT_PREAMP_FITS,
    SPARE_PREAMP_FITS,
)
from lusee.ReceiverImpedance import (
    PREAMP_FITS,
    JFETReceiver,
    receiver_from_config,
    spare_preamp_average_zload,
)

PARAMETERS = ("C_pf", "L_nh", "Rs_ohm", "n", "a", "Rp_ohm")


def assert_parameters_from(receiver, preamps):
    for name in PARAMETERS:
        expected = [PREAMP_FITS[preamp][name] for preamp in preamps]
        np.testing.assert_allclose(
            np.asarray(getattr(receiver, name)), expected, rtol=1e-6
        )


def test_default_channel_map_is_flight():
    receiver = JFETReceiver()
    expected = tuple(FLIGHT_CHANNEL_TO_PREAMP[ch] for ch in range(4))
    assert receiver.channel_map == expected
    assert receiver.channel_map == ("fmpre6", "fmpre3", "fmpre4", "fmpre1")
    assert_parameters_from(receiver, expected)
    assert set(expected) == set(FLIGHT_PREAMP_FITS)


@pytest.mark.parametrize("name", ["spare", "Spare", "SPARE"])
def test_spare_channel_map(name):
    receiver = JFETReceiver(channel_map=name)
    assert receiver.channel_map == ("fmpre0", "fmpre2", "fmpre5", "fmpre7")
    assert_parameters_from(receiver, tuple(SPARE_PREAMP_FITS))


def test_explicit_preamp_tuple_selects_per_channel_fits():
    preamps = ("fmpre7", "fmpre6", "fmpre0", "fmpre1")
    receiver = JFETReceiver(channel_map=preamps)
    assert receiver.channel_map == preamps
    assert_parameters_from(receiver, preamps)


def test_explicit_parameters_override_fit_defaults():
    receiver = JFETReceiver(C_pf=30.0, L_nh=(1.0, 2.0, 3.0, 4.0))
    np.testing.assert_allclose(np.asarray(receiver.C_pf), 30.0)
    np.testing.assert_allclose(np.asarray(receiver.L_nh), [1, 2, 3, 4])
    np.testing.assert_allclose(
        np.asarray(receiver.Rp_ohm),
        [PREAMP_FITS[p]["Rp_ohm"] for p in receiver.channel_map],
    )


def test_unknown_labels_need_explicit_parameters():
    labels = ("a", "b", "c", "d")
    with pytest.raises(ValueError, match="No impedance fit"):
        JFETReceiver(channel_map=labels)
    values = {name: PREAMP_FITS["fmpre6"][name] for name in PARAMETERS}
    receiver = JFETReceiver(channel_map=labels, **values)
    assert receiver.channel_map == labels


@pytest.mark.parametrize(
    "channel_map", ["flatsat", ("fmpre0", "fmpre2", "fmpre5")]
)
def test_invalid_channel_maps_raise(channel_map):
    with pytest.raises(ValueError):
        JFETReceiver(channel_map=channel_map)


def test_receiver_matches_four_term_fit_in_band():
    freq = np.linspace(1.0, 50.0, 50)
    omega = 2 * np.pi * freq * 1e6
    for channel_map in ("flight", "spare"):
        receiver = JFETReceiver(channel_map=channel_map)
        Z = np.asarray(receiver.Z(freq))
        for port, preamp in enumerate(receiver.channel_map):
            fit = PREAMP_FITS[preamp]
            C = fit["C_pf"] * 1e-12
            four_term = (
                1 / (omega**2 * fit["Rp_ohm"] * C**2)
                + fit["Rs_ohm"]
                + fit["a"] * omega ** fit["n"]
                + 1j * (omega * fit["L_nh"] * 1e-9 - 1 / (omega * C))
            )
            np.testing.assert_allclose(Z[:, port, port], four_term, rtol=3e-4)
        off_diagonal = Z * (1 - np.eye(4))[None]
        assert np.all(off_diagonal == 0)


def test_receiver_is_a_pytree_under_jit():
    receiver = JFETReceiver(channel_map="spare")
    freq = jnp.asarray([5.0, 20.0])
    result = jax.jit(lambda r: r.Z(freq))(receiver)
    np.testing.assert_allclose(result, receiver.Z(freq), rtol=1e-6)
    leaves, treedef = jax.tree_util.tree_flatten(receiver)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.channel_map == receiver.channel_map


def test_receiver_from_config_channel_map():
    receiver = receiver_from_config({"model": "jfet"})
    assert receiver.channel_map == JFETReceiver().channel_map
    receiver = receiver_from_config({"model": "jfet", "channel_map": "spare"})
    assert receiver.channel_map == ("fmpre0", "fmpre2", "fmpre5", "fmpre7")
    receiver = receiver_from_config(
        {"model": "jfet", "params": {"channel_map": ["fmpre6"] * 4}}
    )
    assert receiver.channel_map == ("fmpre6",) * 4


def test_hfss_zload_reproduces_notebook_values():
    # Literal SI values from resources/HFSS/ReceiveMatrix.ipynb, which built
    # the ZL baked into the legacy Receive_Matrix_Fields_* exports; note the
    # fmpre7 L = 8.5 nH typo, which must be kept to unload those exports.
    notebook = [
        (33.79e-12, 35.0e-9, 0.21, 1.39, 5.2e-12, 1101e3),
        (35.32e-12, 36.4e-9, 2.16, 2.26, 6.9e-20, 331e3),
        (34.89e-12, 34.7e-9, 3.71, 4.35, 2.2e-38, 288e3),
        (36.85e-12, 8.5e-9, 1.65, 1.88, 1.9e-16, 375e3),
    ]
    freq = np.asarray([1.0, 35.0, 50.0])
    omega = 2 * np.pi * freq * 1e6
    real = sum(
        1 / (omega**2 * Rp * C**2) + Rs + a * omega**n
        for C, L, Rs, n, a, Rp in notebook
    ) / 4
    imag = sum(omega * L - 1 / (omega * C) for C, L, *_ in notebook) / 4
    zload = spare_preamp_average_zload(freq)
    np.testing.assert_allclose(zload[:, 0, 0], real + 1j * imag, rtol=1e-14)
    np.testing.assert_allclose(
        zload[:, 0, 0],
        [
            52.8086010819862 - 4524.025046838699j,
            2.849444813416083 - 122.96253765030426j,
            3.5683732128511485 - 81.48343824942025j,
        ],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        zload, zload[:, :1, :1] * np.eye(4)[None]
    )
