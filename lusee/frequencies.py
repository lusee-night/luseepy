import numpy as np
import jax
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


@jax.tree_util.register_pytree_node_class
class FrequencyMap:
    """Native-index lookup + linear interpolator from a target grid to a source grid.

    A single object bundles the two things every simulator needs to evaluate an
    off-grid frequency set:

    1. **Index lookup** (:attr:`source_indices`): the unique native source-grid
       indices bracketed by the target frequencies. Expensive per-frequency
       products (beam and sky a_lm transforms) are evaluated *only* at these
       indices -- ``get_healpix_alm(freq_ndx=fmap.source_indices)`` and
       ``sky.get_alm(fmap.source_indices)`` -- so nothing is recomputed at
       shared bracket endpoints or at exact on-grid hits.

    2. **Interpolator**: linear blend weights mapping either the unique samples
       (:meth:`from_unique`) or a full native-grid array (:meth:`from_native`)
       onto the target grid. For target frequency i::

           val[i] = (1 - alpha[i]) * unique_vals[lo_in_unique[i]]
                  +      alpha[i]  * unique_vals[hi_in_unique[i]]

       where ``unique_vals[k]`` is the value at native index
       ``source_indices[k]``.

    Snap-on-match: a target within ``(atol, rtol)`` of a source point sets
    ``lo == hi`` and ``alpha == 0.0`` exactly, so plain indexing is recovered
    with no floating-point garbage.

    Registered as a JAX pytree (children: the three index arrays and ``alpha``)
    so an instance can cross ``jit``/``grad``/``vmap`` boundaries. Gradients w.r.t.
    interpolated *values* flow through :meth:`from_unique`/:meth:`from_native`;
    the discrete brackets are built on the host in numpy, so there is no gradient
    w.r.t. the frequencies themselves -- intended, since observing channels are
    fixed. The two-point linear stencil lives entirely behind these methods, so a
    higher-order scheme can be swapped in later without touching call sites.
    """

    def __init__(self, unique_native_idx, lo_in_unique, hi_in_unique, alpha):
        self.unique_native_idx = unique_native_idx
        self.lo_in_unique = lo_in_unique
        self.hi_in_unique = hi_in_unique
        self.alpha = alpha

    @classmethod
    def build(cls, target_freqs, source_freqs, *, atol=1e-6, rtol=1e-9):
        """Construct a map from ``target_freqs`` onto ``source_freqs``.

        :param target_freqs: requested frequencies, 1-D array-like in MHz.
        :param source_freqs: native frequencies of the data being interpolated,
            1-D strictly-increasing array-like in MHz.
        :param atol: absolute tolerance for snap-on-match and boundary checks.
        :param rtol: relative tolerance for the same.
        :returns: a :class:`FrequencyMap` with int32 index arrays and float64 alpha.
        :raises ValueError: if either grid is None or contains non-finite values,
            ``source_freqs`` is not strictly increasing, or any target frequency
            lies outside ``[source.min(), source.max()]`` beyond the tolerance.
        """
        if target_freqs is None:
            raise ValueError("target_freqs is None; expected a 1-D array of MHz values")
        if source_freqs is None:
            raise ValueError(
                "source_freqs is None; the data being interpolated has no native "
                "frequency grid (a sky model without one must implement get_alm_at_freq)"
            )
        target = np.asarray(target_freqs, dtype=np.float64).reshape(-1)
        source = np.asarray(source_freqs, dtype=np.float64).reshape(-1)

        if target.size == 0:
            raise ValueError("target_freqs is empty; expected at least one frequency")
        if not np.all(np.isfinite(target)):
            offenders = target[~np.isfinite(target)].tolist()
            raise ValueError(f"target_freqs contains non-finite values: {offenders}")
        if not np.all(np.isfinite(source)):
            offenders = source[~np.isfinite(source)].tolist()
            raise ValueError(f"source_freqs contains non-finite values: {offenders}")
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
            return cls(
                np.asarray([0], dtype=np.int32),
                zeros,
                zeros,
                np.zeros(n, dtype=np.float64),
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

        return cls(
            unique_idx.astype(np.int32),
            lo_in_unique,
            hi_in_unique,
            new_alpha.astype(np.float64),
        )

    @property
    def source_indices(self):
        """Unique native source-grid indices the expensive products must be evaluated at."""
        return self.unique_native_idx

    def __len__(self):
        """Number of target frequencies this map produces."""
        return int(np.asarray(self.alpha).shape[0])

    def from_unique(self, unique_array):
        """Interpolate an array already reduced to :attr:`source_indices`.

        ``unique_array`` must be indexed positionally by :attr:`source_indices`
        -- row ``k`` holds the value at native index ``source_indices[k]``. This
        is exactly what ``get_healpix_alm(freq_ndx=fmap.source_indices)`` and
        ``sky.get_alm(fmap.source_indices)`` return, so the expensive alm
        products are computed once per unique bracket endpoint and blended here.

        Numpy or JAX in; return type follows the input. Other axes broadcast.
        """
        is_jax = isinstance(unique_array, jnp.ndarray)
        arr = jnp.asarray(unique_array) if is_jax else np.asarray(unique_array)
        lo_vals = arr[self.lo_in_unique]
        hi_vals = arr[self.hi_in_unique]
        a = jnp.asarray(self.alpha) if is_jax else np.asarray(self.alpha)
        shape = (a.shape[0],) + (1,) * (lo_vals.ndim - 1)
        return (1.0 - a.reshape(shape)) * lo_vals + a.reshape(shape) * hi_vals

    def from_native(self, native_array):
        """Interpolate a full native-grid array (indexed by the source grid).

        Use for cheap arrays already held on the full native grid (gains,
        impedances, couplings). Selects :attr:`source_indices` then blends, so
        it is equivalent to ``from_unique(native_array[source_indices])``.

        Numpy or JAX in; return type follows the input. Other axes broadcast.
        """
        is_jax = isinstance(native_array, jnp.ndarray)
        arr = jnp.asarray(native_array) if is_jax else np.asarray(native_array)
        return self.from_unique(arr[self.unique_native_idx])

    def tree_flatten(self):
        children = (self.unique_native_idx, self.lo_in_unique, self.hi_in_unique, self.alpha)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self):
        n_offgrid = int(np.count_nonzero(np.asarray(self.alpha) != 0.0))
        return (
            f"FrequencyMap(n_target={len(self)}, "
            f"n_source_touched={np.asarray(self.unique_native_idx).shape[0]}, "
            f"n_offgrid={n_offgrid})"
        )


def frequencies_from_config(freq_cfg):
    """Parse a YAML ``freq`` block into a numpy array of MHz values.

    Accepted forms::

        freq: { values: [10.0, 20.0, 30.0] }
        freq: { start: 1.0, end: 50.0, step: 1.0 }   # arange: end is EXCLUSIVE
        freq: { start: 1.0, end: 75.0, n: 75 }       # linspace: end is inclusive

    The ``step`` form follows numpy.arange semantics (half-open interval,
    matching the pre-interpolation parser). Float steps inherit arange's
    endpoint rounding caveats; for a grid that must contain the endpoint,
    prefer the ``n`` (linspace) or ``values`` forms.

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

    has_start = "start" in freq_cfg
    has_end = "end" in freq_cfg
    has_step = "step" in freq_cfg
    has_n = "n" in freq_cfg
    if "values" in freq_cfg:
        freq = np.asarray(freq_cfg["values"], dtype=float)
    elif has_start and has_end and has_step and has_n:
        raise ValueError("freq config: specify 'step' or 'n', not both")
    elif has_start and has_end and has_step:
        a = float(freq_cfg["start"])
        b = float(freq_cfg["end"])
        s = float(freq_cfg["step"])
        if s <= 0:
            raise ValueError("freq config: 'step' must be positive")
        freq = np.arange(a, b, s, dtype=float)
    elif has_start and has_end and has_n:
        freq = np.linspace(
            float(freq_cfg["start"]),
            float(freq_cfg["end"]),
            int(freq_cfg["n"]),
        )
    else:
        raise ValueError(
            "freq config must be one of {values: [...]}, {start, end, step}, "
            f"or {{start, end, n}}. Got keys: {sorted(freq_cfg.keys())}"
        )
    if freq.size == 0:
        raise ValueError(f"freq config produced an empty frequency grid: {freq_cfg}")
    return freq


# Legacy canonical-grid shims kept for the pre_jax/* modules and a few tests.
# These predate off-grid interpolation and are slated for removal once the
# notebooks are migrated. New code should use ``frequencies_from_config`` and
# ``FrequencyMap``.

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
