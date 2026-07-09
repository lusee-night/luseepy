from typing import NamedTuple

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


class FrequencyMap(NamedTuple):
    """Linear-interpolation map from a target grid to a source grid.

    For target frequency i, the interpolated value is
        val[i] = (1 - alpha[i]) * unique_vals[lo_in_unique[i]]
               +      alpha[i]  * unique_vals[hi_in_unique[i]]
    where ``unique_vals = native_array[unique_native_idx]``.

    Snap-on-match semantics: when a target frequency falls within
    ``(atol, rtol)`` of a source point, the helper sets ``lo == hi`` to
    that native index and ``alpha == 0.0`` exactly, so plain indexing is
    recovered with no floating-point garbage.
    """

    unique_native_idx: np.ndarray
    lo_in_unique: np.ndarray
    hi_in_unique: np.ndarray
    alpha: np.ndarray


def interpolation_weights(target_freqs, source_freqs, *, atol=1e-6, rtol=1e-9):
    """Build a linear-interpolation map from ``target_freqs`` to ``source_freqs``.

    :param target_freqs: requested frequencies, 1-D array-like in MHz.
    :param source_freqs: native frequencies of the data being interpolated,
        1-D strictly-increasing array-like in MHz.
    :param atol: absolute tolerance for snap-on-match and boundary checks.
    :param rtol: relative tolerance for the same.
    :returns: a :class:`FrequencyMap` with int32 index arrays and float64 alpha.
    :raises ValueError: if ``source_freqs`` is not strictly increasing, or any
        target frequency lies outside ``[source_freqs.min(), source_freqs.max()]``
        beyond the tolerance.
    """
    target = np.asarray(target_freqs, dtype=np.float64).reshape(-1)
    source = np.asarray(source_freqs, dtype=np.float64).reshape(-1)

    if source.size < 1:
        raise ValueError("source_freqs must contain at least one frequency")
    if source.size >= 2 and not np.all(np.diff(source) > 0):
        raise ValueError("source_freqs must be strictly increasing")

    src_min = float(source[0])
    src_max = float(source[-1])
    boundary_atol = atol + rtol * max(abs(src_min), abs(src_max))
    too_low = target < (src_min - boundary_atol)
    too_high = target > (src_max + boundary_atol)
    out_of_range = too_low | too_high
    if np.any(out_of_range):
        offenders = target[out_of_range].tolist()
        raise ValueError(
            f"target_freqs out of range [{src_min}, {src_max}] MHz: {offenders}"
        )

    if source.size == 1:
        # Degenerate single-point source: every target must snap to index 0.
        n = target.size
        zeros = np.zeros(n, dtype=np.int32)
        return FrequencyMap(
            unique_native_idx=np.asarray([0], dtype=np.int32),
            lo_in_unique=zeros,
            hi_in_unique=zeros,
            alpha=np.zeros(n, dtype=np.float64),
        )

    insertion = np.searchsorted(source, target, side="left")
    hi = np.clip(insertion, 0, source.size - 1)
    lo = np.clip(insertion - 1, 0, source.size - 1)

    src_lo = source[lo]
    src_hi = source[hi]
    lo_match = np.isclose(target, src_lo, atol=atol, rtol=rtol)
    hi_match = np.isclose(target, src_hi, atol=atol, rtol=rtol) & ~lo_match

    denom = src_hi - src_lo
    safe_denom = np.where(denom == 0.0, 1.0, denom)
    alpha = (target - src_lo) / safe_denom

    new_lo = np.where(lo_match, lo, np.where(hi_match, hi, lo))
    new_hi = np.where(lo_match, lo, np.where(hi_match, hi, hi))
    new_alpha = np.where(lo_match | hi_match, 0.0, alpha)

    all_idx = np.concatenate([new_lo, new_hi])
    unique_idx, inverse = np.unique(all_idx, return_inverse=True)
    lo_in_unique = inverse[: target.size].astype(np.int32)
    hi_in_unique = inverse[target.size :].astype(np.int32)

    return FrequencyMap(
        unique_native_idx=unique_idx.astype(np.int32),
        lo_in_unique=lo_in_unique,
        hi_in_unique=hi_in_unique,
        alpha=new_alpha.astype(np.float64),
    )


def interp1d(freq_map, native_array):
    """Apply a :class:`FrequencyMap` to a full native-grid array along axis 0.

    ``native_array`` must be indexed by the *native* source grid (the same grid
    passed as ``source_freqs`` to :func:`interpolation_weights`). Use this for
    cheap arrays already held on the full native grid (gains, impedances,
    couplings). For arrays that were computed only at ``unique_native_idx`` (the
    expensive beam/sky alm products), use :func:`interp_from_unique` instead.

    ``native_array`` may be numpy or JAX; the return type follows the input.
    Other axes broadcast unchanged.
    """
    is_jax = isinstance(native_array, jnp.ndarray)
    arr = jnp.asarray(native_array) if is_jax else np.asarray(native_array)
    unique_vals = arr[freq_map.unique_native_idx]
    return interp_from_unique(freq_map, unique_vals)


def interp_from_unique(freq_map, unique_array):
    """Apply a :class:`FrequencyMap` to an array already reduced to unique indices.

    ``unique_array`` must be indexed positionally by ``freq_map.unique_native_idx``
    -- i.e. row ``k`` holds the value at native index ``unique_native_idx[k]``.
    This is exactly what ``get_healpix_alm(freq_ndx=freq_map.unique_native_idx)``
    and ``sky.get_alm(freq_map.unique_native_idx)`` return, so the expensive alm
    products are computed once per unique bracket endpoint and blended here.

    ``unique_array`` may be numpy or JAX; the return type follows the input.
    Other axes broadcast unchanged.
    """
    is_jax = isinstance(unique_array, jnp.ndarray)
    arr = jnp.asarray(unique_array) if is_jax else np.asarray(unique_array)
    lo_vals = arr[freq_map.lo_in_unique]
    hi_vals = arr[freq_map.hi_in_unique]
    a = jnp.asarray(freq_map.alpha) if is_jax else freq_map.alpha
    shape = (a.shape[0],) + (1,) * (lo_vals.ndim - 1)
    return (1.0 - a.reshape(shape)) * lo_vals + a.reshape(shape) * hi_vals


def frequencies_from_config(freq_cfg):
    """Parse a YAML ``freq`` block into a numpy array of MHz values.

    Accepted forms::

        freq: { values: [10.0, 20.0, 30.0] }
        freq: { start: 1.0, end: 50.0, step: 1.0 }   # inclusive end
        freq: { start: 1.0, end: 75.0, n: 75 }       # linspace(start, end, n)

    Legacy index-based forms (``indices``, ``start_idx``/``stop_idx``/``step_idx``)
    are no longer supported and produce a :class:`ValueError` naming the new keys.
    """
    legacy_keys = {"indices", "start_idx", "stop_idx", "step_idx"}
    found_legacy = legacy_keys & set(freq_cfg.keys())
    if found_legacy:
        raise ValueError(
            f"freq config keys {sorted(found_legacy)} are no longer supported. "
            "Use 'values', 'start/end/step', or 'start/end/n' in MHz units."
        )

    if "values" in freq_cfg:
        return np.asarray(freq_cfg["values"], dtype=float)

    has_start = "start" in freq_cfg
    has_end = "end" in freq_cfg
    has_step = "step" in freq_cfg
    has_n = "n" in freq_cfg
    if has_start and has_end and has_step and has_n:
        raise ValueError("freq config: specify 'step' or 'n', not both")
    if has_start and has_end and has_step:
        a = float(freq_cfg["start"])
        b = float(freq_cfg["end"])
        s = float(freq_cfg["step"])
        if s <= 0:
            raise ValueError("freq config: 'step' must be positive")
        return np.arange(a, b + 0.5 * s, s, dtype=float)
    if has_start and has_end and has_n:
        return np.linspace(
            float(freq_cfg["start"]),
            float(freq_cfg["end"]),
            int(freq_cfg["n"]),
        )
    raise ValueError(
        "freq config must be one of {values: [...]}, {start, end, step}, "
        f"or {{start, end, n}}. Got keys: {sorted(freq_cfg.keys())}"
    )


# Compatibility shims kept for the pre_jax/* modules and a few legacy tests.
# New code should use ``frequencies_from_config`` and ``interpolation_weights``.

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
