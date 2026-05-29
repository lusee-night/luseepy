import numpy as np
import jax.numpy as jnp


CANONICAL_FREQ_START_MHZ = 1.0
CANONICAL_FREQ_STOP_MHZ = 50.0
CANONICAL_FREQ_COUNT = 50

ALL_FREQUENCY_INDICES = jnp.arange(CANONICAL_FREQ_COUNT, dtype=jnp.int32)
# float64 reference grid: JAX linspace defaults can be float32 and break exact
# matches to FITS-derived MHz values in :class:`FitsSky`.
ALL_FREQUENCIES_MHZ_NP = np.linspace(
    CANONICAL_FREQ_START_MHZ,
    CANONICAL_FREQ_STOP_MHZ,
    CANONICAL_FREQ_COUNT,
    dtype=np.float64,
)
ALL_FREQUENCIES_MHZ = jnp.asarray(ALL_FREQUENCIES_MHZ_NP, dtype=jnp.float64)


def canonical_frequency_indices(indices=None, *, start_idx=0, stop_idx=None, step_idx=1):
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int32)
    else:
        if stop_idx is None:
            stop_idx = CANONICAL_FREQ_COUNT
        idx = np.arange(start_idx, stop_idx, step_idx, dtype=np.int32)
    if np.any(idx < 0) or np.any(idx >= CANONICAL_FREQ_COUNT):
        raise ValueError("Frequency indices must lie within the canonical 0..49 grid")
    return idx


def canonical_frequencies(indices=None, *, as_jax=False):
    idx = canonical_frequency_indices(indices)
    if as_jax:
        return ALL_FREQUENCIES_MHZ[jnp.asarray(idx, dtype=jnp.int32)]
    return ALL_FREQUENCIES_MHZ_NP[idx]


def frequency_indices_from_values(freq_values, *, atol=1e-8, rtol=1e-8):
    freq_arr = np.asarray(freq_values, dtype=float).reshape(-1)
    indices = []
    for value in freq_arr:
        matches = np.nonzero(
            np.isclose(ALL_FREQUENCIES_MHZ_NP, value, atol=atol, rtol=rtol)
        )[0]
        if matches.size == 0:
            raise ValueError(
                f"Frequency {value} MHz is not on the canonical simulator grid"
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int32)


def canonicalize_frequencies(
    freq_values, *, as_jax=False, atol=1e-8, rtol=1e-8
):
    """Map MHz samples to the canonical simulator grid (see :func:`frequency_indices_from_values`)."""
    return canonical_frequencies(
        frequency_indices_from_values(freq_values, atol=atol, rtol=rtol),
        as_jax=as_jax,
    )


def fine_uniform_frequency_mhz(
    start_mhz: float = CANONICAL_FREQ_START_MHZ,
    stop_mhz: float = CANONICAL_FREQ_STOP_MHZ,
    *,
    step_khz: float = 1.0,
) -> jnp.ndarray:
    """
    Uniform frequency axis in MHz (not restricted to the 50-bin canonical grid).

    *step_khz* is the channel spacing in kHz (e.g. ``1.0`` → 1 kHz). The last
    sample is the greatest frequency ≤ *stop_mhz* reachable from *start_mhz*.
    """
    step_mhz = float(step_khz) / 1000.0
    if step_mhz <= 0.0:
        raise ValueError("step_khz must be positive")
    n = int(np.floor((float(stop_mhz) - float(start_mhz)) / step_mhz)) + 1
    if n < 1:
        raise ValueError("no frequency samples with given start/stop/step_khz")
    vals = float(start_mhz) + step_mhz * np.arange(n, dtype=np.float64)
    return jnp.asarray(vals, dtype=jnp.float64)


def observation_frequency_mhz_from_config(freq_cfg: dict) -> jnp.ndarray:
    """
    Build the simulator frequency list (MHz) from an ``observation.freq`` dict.

    Supported shapes:

    - **Uniform kHz grid** (RRL / high-resolution runs)::

        freq: { start_mhz: 1.0, stop_mhz: 50.0, step_khz: 1.0 }

    - **Canonical 0..49 indices** (existing drivers)::

        freq: { start_idx: 0, stop_idx: 50, step_idx: 1 }

    - **Legacy** ``start`` / ``end`` / ``step`` in MHz on the canonical 1–50 MHz grid.
    """
    if not isinstance(freq_cfg, dict):
        raise TypeError("freq_cfg must be a dict")
    if all(k in freq_cfg for k in ("start_mhz", "stop_mhz", "step_khz")):
        return fine_uniform_frequency_mhz(
            float(freq_cfg["start_mhz"]),
            float(freq_cfg["stop_mhz"]),
            step_khz=float(freq_cfg["step_khz"]),
        )
    if "indices" in freq_cfg or any(
        k in freq_cfg for k in ("start_idx", "stop_idx", "step_idx")
    ):
        return canonical_frequencies(frequency_indices_from_config(freq_cfg), as_jax=True)
    if all(k in freq_cfg for k in ("start", "end", "step")):
        return canonical_frequencies(frequency_indices_from_config(freq_cfg), as_jax=True)
    raise ValueError(
        "freq config must define start_mhz/stop_mhz/step_khz, indices or "
        "start_idx/stop_idx/step_idx, or legacy start/end/step (MHz on canonical grid)"
    )


def frequency_indices_from_config(freq_cfg):
    if "indices" in freq_cfg:
        return canonical_frequency_indices(freq_cfg["indices"])
    if any(key in freq_cfg for key in ("start_idx", "stop_idx", "step_idx")):
        return canonical_frequency_indices(
            start_idx=int(freq_cfg.get("start_idx", 0)),
            stop_idx=int(freq_cfg.get("stop_idx", CANONICAL_FREQ_COUNT)),
            step_idx=int(freq_cfg.get("step_idx", 1)),
        )
    if all(key in freq_cfg for key in ("start", "end", "step")):
        freq_values = np.arange(
            float(freq_cfg["start"]),
            float(freq_cfg["end"]),
            float(freq_cfg["step"]),
            dtype=float,
        )
        return frequency_indices_from_values(freq_values)
    raise ValueError(
        "Frequency config must define either indices, start_idx/stop_idx/step_idx, "
        "or legacy start/end/step"
    )
