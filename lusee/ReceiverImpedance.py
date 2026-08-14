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


# Measured fit parameters for the four spare flight-model preamps
# (fmpre0/2/5/7), from the JFET impedance Bode-sweep analysis reproduced in
# resources/HFSS/ReceiveMatrix.ipynb. Model:
#   Re Z = 1/(w^2 Rp C^2) + Rs + a w^n,   Im Z = w L - 1/(w C)
SPARE_PREAMP_FITS = {
    "fmpre0": {"C_pf": 33.79, "L_nh": 35.0, "Rs_ohm": 0.21, "n": 1.39,
               "a": 5.2e-12, "Rp_ohm": 1101e3},
    "fmpre2": {"C_pf": 35.32, "L_nh": 36.4, "Rs_ohm": 2.16, "n": 2.26,
               "a": 6.9e-20, "Rp_ohm": 331e3},
    "fmpre5": {"C_pf": 34.89, "L_nh": 34.7, "Rs_ohm": 3.71, "n": 4.35,
               "a": 2.2e-38, "Rp_ohm": 288e3},
    "fmpre7": {"C_pf": 36.85, "L_nh": 8.5, "Rs_ohm": 1.65, "n": 1.88,
               "a": 1.9e-16, "Rp_ohm": 375e3},
}


def spare_preamp_average_zload(freq_mhz):
    """Averaged spare-preamp load matrix used by the HFSS receive exports.

    Evaluates the exact four-term model of ``ReceiveMatrix.ipynb`` for each
    measured spare preamp, averages the real and imaginary parts across the
    four devices, and returns the scalar-diagonal ``(frequency, 4, 4)``
    matrix that the notebook applied as ``ZL (ZA + ZL)^-1``. This is the
    ZLoad payload required to unload the ``Receive_Matrix_Fields_*`` CSVs.
    """
    freq = np.atleast_1d(np.asarray(freq_mhz, dtype=np.float64))
    omega = 2 * np.pi * freq * 1e6
    real = np.zeros_like(omega)
    imag = np.zeros_like(omega)
    for fit in SPARE_PREAMP_FITS.values():
        C = fit["C_pf"] * 1e-12
        L = fit["L_nh"] * 1e-9
        real += (
            1.0 / (omega**2 * fit["Rp_ohm"] * C**2)
            + fit["Rs_ohm"]
            + fit["a"] * omega ** fit["n"]
        )
        imag += omega * L - 1.0 / (omega * C)
    scalar = (real + 1j * imag) / len(SPARE_PREAMP_FITS)
    result = np.zeros((freq.size, 4, 4), dtype=np.complex128)
    indices = np.arange(4)
    result[:, indices, indices] = scalar[:, None]
    return result


@jax.tree_util.register_pytree_node_class
class JFETReceiver:
    """Diagonal JFET load model with differentiable per-channel parameters.

    Defaults are the measured Bode-sweep fits for the spare preamps
    fmpre0/2/5/7 (see ``SPARE_PREAMP_FITS``), evaluated with the
    parallel-RC form ``Rs + a w^n + jwL + Rp/(1 + jw Rp C)``, whose in-band
    difference from the notebook's four-term expansion is below 1e-4.
    """

    def __init__(
        self,
        C_pf=(33.79, 35.32, 34.89, 36.85),
        L_nh=(35.0, 36.4, 34.7, 8.5),
        Rs_ohm=(0.21, 2.16, 3.71, 1.65),
        n=(1.39, 2.26, 4.35, 1.88),
        a=(5.2e-12, 6.9e-20, 2.2e-38, 1.9e-16),
        Rp_ohm=(1101e3, 331e3, 288e3, 375e3),
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
        frequency_map = FrequencyMap.build(
            freq_mhz,
            self.freq,
            policy="linear",
        )
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
