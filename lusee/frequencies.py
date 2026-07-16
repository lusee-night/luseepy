import numpy as np
import jax.numpy as jnp


CANONICAL_FREQ_START_MHZ = 1.0
CANONICAL_FREQ_STOP_MHZ = 50.0
CANONICAL_FREQ_COUNT = 50

ALL_FREQUENCY_INDICES = jnp.arange(CANONICAL_FREQ_COUNT, dtype=jnp.int32)
# Reference grid in float64 NumPy so FITS/header MHz values (e.g. 12.0) match
# even when JAX uses float32 by default; `np.asarray(jnp.linspace(...))` can
# be ~1e-6 MHz off integer MHz and fail tight isclose checks.
ALL_FREQUENCIES_MHZ_NP = np.linspace(
    CANONICAL_FREQ_START_MHZ,
    CANONICAL_FREQ_STOP_MHZ,
    CANONICAL_FREQ_COUNT,
    dtype=np.float64,
)
ALL_FREQUENCIES_MHZ = jnp.asarray(ALL_FREQUENCIES_MHZ_NP)


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


def frequency_indices_from_values(
    freq_values, *, atol=1e-5, rtol=1e-5, nearest_max_mhz=0.05
):
    """Map MHz values to canonical indices (0..49).

    Defaults tolerate small float noise from FITS headers and ``np.arange``.
    If no ``isclose`` hit, the nearest canonical bin is used when within
    ``nearest_max_mhz`` MHz (otherwise ``ValueError``).
    """
    freq_arr = np.asarray(freq_values, dtype=np.float64).reshape(-1)
    indices = []
    for value in freq_arr:
        matches = np.nonzero(
            np.isclose(ALL_FREQUENCIES_MHZ_NP, value, atol=atol, rtol=rtol)
        )[0]
        if matches.size > 0:
            indices.append(int(matches[0]))
            continue
        j = int(np.argmin(np.abs(ALL_FREQUENCIES_MHZ_NP - value)))
        err_mhz = float(abs(ALL_FREQUENCIES_MHZ_NP[j] - value))
        if err_mhz <= nearest_max_mhz:
            indices.append(j)
        else:
            raise ValueError(
                f"Frequency {value} MHz is not on the canonical simulator grid "
                f"(nearest bin {float(ALL_FREQUENCIES_MHZ_NP[j]):.6f} MHz is "
                f"{err_mhz:.6f} MHz away; limit {nearest_max_mhz} MHz)."
            )
    return np.asarray(indices, dtype=np.int32)


def canonicalize_frequencies(freq_values, *, as_jax=False):
    return canonical_frequencies(frequency_indices_from_values(freq_values), as_jax=as_jax)


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
