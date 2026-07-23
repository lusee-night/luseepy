"""Differentiable four-port receiver impedance models."""

import jax
import jax.numpy as jnp
import numpy as np

from .frequencies import FrequencyMap


def _four_vector(value, name):
    array = jnp.asarray(value)
    if array.ndim == 0:
        array = jnp.repeat(array[None], 4)
    if array.shape != (4,):
        raise ValueError(f"{name} must be scalar or have shape (4,).")
    return array


@jax.tree_util.register_pytree_node_class
class JFETReceiver:
    """Diagonal JFET load model with differentiable per-channel parameters."""

    def __init__(
        self,
        C_pf=(35.4, 34.2, 36.1, 36.8),
        L_nh=(37.0, 35.5, 38.2, 39.0),
        Rs_ohm=(1.0, 0.7, 1.5, 2.0),
        n=(2.0, 2.0, 2.0, 2.0),
        a=(0.0, 0.0, 0.0, 0.0),
        Rp_ohm=(6.0e5, 7.0e5, 5.0e5, 8.0e5),
        channel_map=("fmpre0", "fmpre2", "fmpre5", "fmpre7"),
    ):
        self.C_pf = _four_vector(C_pf, "C_pf")
        self.L_nh = _four_vector(L_nh, "L_nh")
        self.Rs_ohm = _four_vector(Rs_ohm, "Rs_ohm")
        self.n = _four_vector(n, "n")
        self.a = _four_vector(a, "a")
        self.Rp_ohm = _four_vector(Rp_ohm, "Rp_ohm")
        self.channel_map = tuple(str(value) for value in channel_map)
        if len(self.channel_map) != 4:
            raise ValueError("channel_map must contain four entries.")

    def Z(self, freq_mhz):
        """Evaluate the complex load matrix at arbitrary MHz frequencies."""
        freq = jnp.atleast_1d(jnp.asarray(freq_mhz))
        omega = 2 * jnp.pi * freq[:, None] * 1e6
        C = self.C_pf[None] * 1e-12
        L = self.L_nh[None] * 1e-9
        Rp = self.Rp_ohm[None]
        parallel = Rp / (1.0 + 1j * omega * Rp * C)
        diagonal = (
            self.Rs_ohm[None]
            + self.a[None] * omega ** self.n[None]
            + 1j * omega * L
            + parallel
        )
        result = jnp.zeros((freq.size, 4, 4), dtype=diagonal.dtype)
        indices = jnp.arange(4)
        return result.at[:, indices, indices].set(diagonal)

    @property
    def params(self):
        """Return numerical model parameters without static channel metadata."""
        return {
            "C_pf": self.C_pf,
            "L_nh": self.L_nh,
            "Rs_ohm": self.Rs_ohm,
            "n": self.n,
            "a": self.a,
            "Rp_ohm": self.Rp_ohm,
        }

    def tree_flatten(self):
        children = (
            self.C_pf,
            self.L_nh,
            self.Rs_ohm,
            self.n,
            self.a,
            self.Rp_ohm,
        )
        return children, self.channel_map

    @classmethod
    def tree_unflatten(cls, channel_map, children):
        obj = cls.__new__(cls)
        (
            obj.C_pf,
            obj.L_nh,
            obj.Rs_ohm,
            obj.n,
            obj.a,
            obj.Rp_ohm,
        ) = children
        obj.channel_map = channel_map
        return obj


@jax.tree_util.register_pytree_node_class
class IdealCapacitorReceiver:
    """Four independent ideal series capacitors."""

    def __init__(self, C_pf=30.0):
        self.C_pf = _four_vector(C_pf, "C_pf")
        self.channel_map = ("0", "1", "2", "3")

    def Z(self, freq_mhz):
        """Evaluate ``1/(j omega C)`` on the supplied target grid."""
        try:
            host_freq = np.asarray(freq_mhz)
        except jax.errors.TracerArrayConversionError:
            host_freq = None
        if host_freq is not None and np.any(host_freq <= 0):
            raise ValueError("Ideal capacitor frequencies must be positive.")
        freq = jnp.atleast_1d(jnp.asarray(freq_mhz))
        omega = 2 * jnp.pi * freq[:, None] * 1e6
        diagonal = 1.0 / (1j * omega * self.C_pf[None] * 1e-12)
        result = jnp.zeros((freq.size, 4, 4), dtype=diagonal.dtype)
        indices = jnp.arange(4)
        return result.at[:, indices, indices].set(diagonal)

    @property
    def params(self):
        return {"C_pf": self.C_pf}

    def tree_flatten(self):
        return (self.C_pf,), self.channel_map

    @classmethod
    def tree_unflatten(cls, channel_map, children):
        obj = cls.__new__(cls)
        (obj.C_pf,) = children
        obj.channel_map = channel_map
        return obj


@jax.tree_util.register_pytree_node_class
class MeasuredReceiver:
    """Measured dense receiver matrix with FrequencyMap interpolation."""

    def __init__(self, freq_mhz, impedance_ohm, *, source=None):
        freq = np.asarray(freq_mhz, dtype=np.float64).reshape(-1)
        if freq.size == 0 or not np.all(np.isfinite(freq)):
            raise ValueError("Measured receiver frequencies must be finite.")
        if freq.size > 1 and not np.all(np.diff(freq) > 0):
            raise ValueError(
                "Measured receiver frequencies must be strictly increasing."
            )
        impedance = jnp.asarray(impedance_ohm)
        if impedance.shape != (freq.size, 4, 4):
            raise ValueError(
                "Measured impedance must have shape (frequency, 4, 4)."
            )
        self.freq = freq
        self.impedance = impedance
        self.source = None if source is None else str(source)
        self.channel_map = ("0", "1", "2", "3")

    def Z(self, freq_mhz):
        """Interpolate the measured matrix without extrapolation."""
        frequency_map = FrequencyMap.build(freq_mhz, self.freq)
        return frequency_map.from_native(self.impedance)

    @property
    def params(self):
        return {}

    def tree_flatten(self):
        children = (self.impedance,)
        aux = (tuple(self.freq.tolist()), self.source, self.channel_map)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        freq, source, channel_map = aux
        obj = cls.__new__(cls)
        (obj.impedance,) = children
        obj.freq = np.asarray(freq, dtype=np.float64)
        obj.source = source
        obj.channel_map = channel_map
        return obj


@jax.jit
def loading_matrix(ZA, ZL):
    """Compute ``ZL (ZA + ZL)^-1`` with a batched right-side solve."""
    ZA = jnp.asarray(ZA)
    ZL = jnp.asarray(ZL)
    if ZA.shape != ZL.shape or ZA.shape[-2:] != (4, 4):
        raise ValueError("ZA and ZL must have matching (..., 4, 4) shapes.")
    left_transpose = jnp.swapaxes(ZA + ZL, -1, -2)
    right_transpose = jnp.swapaxes(ZL, -1, -2)
    return jnp.swapaxes(
        jnp.linalg.solve(left_transpose, right_transpose),
        -1,
        -2,
    )


def receiver_from_config(config):
    """Construct one receiver model from the breaking response config."""
    model = str(config.get("model", "jfet")).lower()
    params = dict(config.get("params", {}))
    if model == "jfet":
        if "channel_map" in config:
            params["channel_map"] = config["channel_map"]
        return JFETReceiver(**params)
    if model in {"capacitor", "ideal_capacitor"}:
        return IdealCapacitorReceiver(**params)
    if model in {"file", "measured"}:
        if "freq_mhz" not in params or "impedance_ohm" not in params:
            raise ValueError(
                "Measured receiver config requires freq_mhz and impedance_ohm."
            )
        return MeasuredReceiver(**params)
    raise ValueError(f"Unsupported receiver model {model!r}.")
