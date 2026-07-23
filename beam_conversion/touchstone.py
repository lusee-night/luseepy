"""Touchstone S-parameter parsing and full-matrix S-to-Z conversion."""

from pathlib import Path

import numpy as np


_FREQUENCY_SCALE = {
    "HZ": 1e-6,
    "KHZ": 1e-3,
    "MHZ": 1.0,
    "GHZ": 1e3,
}


def s_to_z(s_parameters, zref):
    """Convert batched full S matrices to Z without discarding coupling."""
    s_parameters = np.asarray(s_parameters)
    if s_parameters.ndim != 3 or s_parameters.shape[-1] != s_parameters.shape[-2]:
        raise ValueError("S parameters must have shape (frequency, port, port).")
    nfreq, nport, _ = s_parameters.shape
    zref = np.asarray(zref, dtype=np.float64)
    if zref.ndim == 0:
        zref = np.full((nfreq, nport), zref)
    elif zref.ndim == 1 and zref.size == nport:
        zref = np.broadcast_to(zref[None], (nfreq, nport))
    if zref.shape != (nfreq, nport):
        raise ValueError("zref must be scalar, per-port, or frequency-by-port.")
    if np.any(zref <= 0):
        raise ValueError("Reference impedances must be positive.")

    identity = np.eye(nport, dtype=s_parameters.dtype)[None]
    sqrt_z = np.zeros_like(s_parameters)
    diagonal = np.arange(nport)
    sqrt_z[:, diagonal, diagonal] = np.sqrt(zref)
    middle = np.swapaxes(
        np.linalg.solve(
            np.swapaxes(identity - s_parameters, -1, -2),
            np.swapaxes(identity + s_parameters, -1, -2),
        ),
        -1,
        -2,
    )
    return sqrt_z @ middle @ sqrt_z


def read_touchstone(path):
    """Read a Touchstone 1.x file with RI, MA, or DB value encoding."""
    path = Path(path)
    suffix = path.suffix.lower()
    if not suffix.startswith(".s") or not suffix.endswith("p"):
        raise ValueError(f"Cannot infer port count from {path.name!r}.")
    nport = int(suffix[2:-1])
    option = ["GHZ", "S", "MA", "R", "50"]
    numeric = []
    pending = []
    for raw in path.read_text().splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#"):
            option = line[1:].upper().split()
            continue
        pending.extend(float(token) for token in line.split())
        row_size = 1 + 2 * nport * nport
        while len(pending) >= row_size:
            numeric.append(pending[:row_size])
            pending = pending[row_size:]
    if pending:
        raise ValueError("Incomplete Touchstone data row.")
    values = np.asarray(numeric, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Touchstone file contains no network data.")

    freq_unit = option[0]
    parameter = option[1]
    encoding = option[2]
    if parameter != "S":
        raise ValueError("Only S-parameter Touchstone files are supported.")
    if freq_unit not in _FREQUENCY_SCALE:
        raise ValueError(f"Unsupported frequency unit {freq_unit!r}.")
    pairs = values[:, 1:].reshape(values.shape[0], nport * nport, 2)
    if encoding == "RI":
        complex_values = pairs[..., 0] + 1j * pairs[..., 1]
    elif encoding == "MA":
        complex_values = pairs[..., 0] * np.exp(
            1j * np.radians(pairs[..., 1])
        )
    elif encoding == "DB":
        complex_values = 10 ** (pairs[..., 0] / 20.0) * np.exp(
            1j * np.radians(pairs[..., 1])
        )
    else:
        raise ValueError(f"Unsupported Touchstone encoding {encoding!r}.")

    # Touchstone lists port pairs column-major: S11, S21, ..., S12, ...
    matrices = complex_values.reshape(values.shape[0], nport, nport)
    matrices = np.swapaxes(matrices, -1, -2)
    try:
        reference = float(option[option.index("R") + 1])
    except (ValueError, IndexError):
        reference = 50.0
    freq_mhz = values[:, 0] * _FREQUENCY_SCALE[freq_unit]
    return freq_mhz, matrices, reference


def read_touchstone_z(path):
    """Read a Touchstone S file and return frequency, full Z, and zref."""
    freq_mhz, scattering, zref = read_touchstone(path)
    return freq_mhz, s_to_z(scattering, zref), zref
