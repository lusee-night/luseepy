from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import jax
import jax.numpy as jnp


CANONICAL_FREQ_START_MHZ = 1.0
CANONICAL_FREQ_STOP_MHZ = 50.0
CANONICAL_FREQ_COUNT = 50
FREQUENCY_SNAP_ATOL_MHZ = 1e-6
FREQUENCY_SNAP_RTOL = 1e-9

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
# sky models alias this array as their default native grid; freeze it so an
# in-place edit (e.g. sky.freq *= 1e6) fails loudly instead of corrupting
# the canonical grid process-wide
ALL_FREQUENCIES_MHZ_NP.setflags(write=False)
ALL_FREQUENCIES_MHZ = jnp.asarray(ALL_FREQUENCIES_MHZ_NP)


def frequency_grids_match(left, right):
    """Whether two native MHz grids agree under the standard snap tolerance."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return left.shape == right.shape and bool(
        np.allclose(
            left,
            right,
            atol=FREQUENCY_SNAP_ATOL_MHZ,
            rtol=FREQUENCY_SNAP_RTOL,
        )
    )


class FrequencyPolicy(StrEnum):
    """Allowed construction-time contracts for frequency alignment."""

    IDENTITY = "identity"
    EXACT = "exact"
    LINEAR = "linear"

    @classmethod
    def normalize(cls, value):
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(repr(policy.value) for policy in cls)
            raise ValueError(
                f"Unknown frequency policy {value!r}; expected one of {choices}"
            ) from exc


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FrequencyMap:
    """Static frequency-alignment plan compiled from target and source grids.

    ``policy`` is the construction-time contract: ``identity`` requires the
    grids to align one-for-one, ``exact`` permits integer selection only, and
    ``linear`` permits interpolation. The resolved ``mode`` is one of
    ``identity``, ``gather``, or ``linear`` and controls the runtime fast path.

    All floating-point lookup and snap-on-match handling happens in
    :meth:`build`. Runtime methods use only the compiled integer indices and
    weights. Policy and mode are static JAX pytree metadata.
    """

    unique_native_idx: object
    lo_in_unique: object
    hi_in_unique: object
    alpha: object
    policy: FrequencyPolicy
    mode: str
    source_size: int

    def __init__(
        self,
        unique_native_idx,
        lo_in_unique,
        hi_in_unique,
        alpha,
        *,
        policy=FrequencyPolicy.LINEAR,
        mode="linear",
        source_size,
    ):
        unique_native_idx = self._metadata_array(
            unique_native_idx, name="unique_native_idx", integer=True
        )
        lo_in_unique = self._metadata_array(
            lo_in_unique, name="lo_in_unique", integer=True
        )
        hi_in_unique = self._metadata_array(
            hi_in_unique, name="hi_in_unique", integer=True
        )
        alpha = self._metadata_array(alpha, name="alpha", integer=False)
        policy = FrequencyPolicy.normalize(policy)
        if mode not in {"identity", "gather", "linear"}:
            raise ValueError(f"Unknown frequency map mode {mode!r}")
        source_size = int(source_size)
        if source_size < 1:
            raise ValueError("FrequencyMap source_size must be positive")

        self._validate_compiled_metadata(
            unique_native_idx,
            lo_in_unique,
            hi_in_unique,
            alpha,
            policy=policy,
            mode=mode,
            source_size=source_size,
        )
        object.__setattr__(self, "unique_native_idx", unique_native_idx)
        object.__setattr__(self, "lo_in_unique", lo_in_unique)
        object.__setattr__(self, "hi_in_unique", hi_in_unique)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source_size", source_size)

    @staticmethod
    def _metadata_array(value, *, name, integer):
        if getattr(value, "ndim", None) != 1:
            raise ValueError(f"FrequencyMap {name} must be one-dimensional")
        if isinstance(value, (jax.Array, jax.core.Tracer)):
            return value
        array = np.asarray(value)
        if integer and not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"FrequencyMap {name} must contain integers")
        array = np.array(
            array,
            dtype=np.int32 if integer else np.float64,
            copy=True,
        )
        array.setflags(write=False)
        return array

    @staticmethod
    def _validate_compiled_metadata(
        unique_native_idx,
        lo_in_unique,
        hi_in_unique,
        alpha,
        *,
        policy,
        mode,
        source_size,
    ):
        n_unique = unique_native_idx.shape[0]
        n_target = alpha.shape[0]
        if n_unique < 1 or n_target < 1:
            raise ValueError("FrequencyMap source and target metadata must not be empty")
        if lo_in_unique.shape != alpha.shape or hi_in_unique.shape != alpha.shape:
            raise ValueError("FrequencyMap target metadata lengths must match")

        # Pytree reconstruction under jit receives tracers from metadata that
        # was already validated when build() created the map.
        values = (unique_native_idx, lo_in_unique, hi_in_unique, alpha)
        if any(isinstance(value, jax.core.Tracer) for value in values):
            return

        unique = np.asarray(unique_native_idx)
        lo = np.asarray(lo_in_unique)
        hi = np.asarray(hi_in_unique)
        weights = np.asarray(alpha)
        if not all(
            np.issubdtype(value.dtype, np.integer) for value in (unique, lo, hi)
        ):
            raise TypeError("FrequencyMap stencil indices must be integers")
        if np.any(unique < 0) or np.any(unique >= source_size):
            raise ValueError("FrequencyMap native indices are out of bounds")
        if unique.size > 1 and not np.all(np.diff(unique) > 0):
            raise ValueError("FrequencyMap native indices must be strictly increasing")
        if np.any(lo < 0) or np.any(lo >= n_unique):
            raise ValueError("FrequencyMap lower stencil indices are out of bounds")
        if np.any(hi < 0) or np.any(hi >= n_unique):
            raise ValueError("FrequencyMap upper stencil indices are out of bounds")
        if not np.all(np.isfinite(weights)):
            raise ValueError("FrequencyMap weights must all be finite")
        if np.any(weights < 0.0) or np.any(weights > 1.0):
            raise ValueError("FrequencyMap interpolation weights must lie in [0, 1]")

        direct = np.array_equal(lo, hi) and np.all(weights == 0.0)
        identity = (
            direct
            and n_target == source_size
            and n_unique == source_size
            and np.array_equal(unique, np.arange(source_size))
            and np.array_equal(lo, np.arange(source_size))
        )
        if mode == "identity" and not identity:
            raise ValueError("FrequencyMap identity mode does not match its stencil")
        if mode == "gather" and not direct:
            raise ValueError("FrequencyMap gather mode requires exact channel indices")
        if mode == "linear" and direct:
            raise ValueError("FrequencyMap linear mode requires an interpolating stencil")
        if policy is FrequencyPolicy.IDENTITY and mode != "identity":
            raise ValueError("FrequencyMap identity policy requires identity mode")
        if policy is FrequencyPolicy.EXACT and mode == "linear":
            raise ValueError("FrequencyMap exact policy cannot use linear mode")

    @classmethod
    def build(
        cls,
        target_freqs,
        source_freqs,
        *,
        policy=FrequencyPolicy.EXACT,
        atol=FREQUENCY_SNAP_ATOL_MHZ,
        rtol=FREQUENCY_SNAP_RTOL,
    ):
        """Construct a map from ``target_freqs`` onto ``source_freqs``.

        :param target_freqs: requested frequencies, 1-D array-like in MHz.
        :param source_freqs: native frequencies of the data being interpolated,
            1-D strictly-increasing array-like in MHz.
        :param policy: ``identity`` requires one-for-one alignment, ``exact``
            allows exact channel selection, and ``linear`` allows off-grid
            interpolation.
        :param atol: absolute tolerance for snap-on-match and boundary checks.
        :param rtol: relative tolerance for the same.
        :returns: a :class:`FrequencyMap` with a static execution mode.
        :raises ValueError: if either grid is None or contains non-finite values,
            ``source_freqs`` is not strictly increasing, or any target frequency
            violates the requested policy.
        """
        policy = FrequencyPolicy.normalize(policy)
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

        if policy is FrequencyPolicy.IDENTITY:
            if target.shape != source.shape:
                raise ValueError(
                    "frequency_policy='identity' requires target and source grids "
                    f"to have the same shape; got {target.shape} and {source.shape}"
                )
            matches = np.isclose(target, source, atol=atol, rtol=rtol)
            if not np.all(matches):
                offenders = target[~matches].tolist()
                raise ValueError(
                    "frequency_policy='identity' requires one-for-one native "
                    f"channel alignment; mismatched targets: {offenders}"
                )
            indices = np.arange(source.size, dtype=np.int32)
            return cls(
                indices,
                indices.copy(),
                indices.copy(),
                np.zeros(target.size, dtype=np.float64),
                policy=policy,
                mode="identity",
                source_size=source.size,
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
                policy=policy,
                mode="identity" if n == 1 else "gather",
                source_size=1,
            )

        insertion = np.searchsorted(source, target, side="left")
        hi = np.clip(insertion, 0, source.size - 1)
        lo = np.clip(insertion - 1, 0, source.size - 1)

        src_lo = source[lo]
        src_hi = source[hi]
        lo_close = np.isclose(target, src_lo, atol=atol, rtol=rtol)
        hi_close = np.isclose(target, src_hi, atol=atol, rtol=rtol)
        # Dense grids can put both neighbors inside the snap tolerance. Choose
        # the physically closest native bin; an exact distance tie goes lower.
        hi_match = hi_close & (
            ~lo_close | (np.abs(target - src_hi) < np.abs(target - src_lo))
        )
        lo_match = lo_close & ~hi_match

        denom = src_hi - src_lo
        safe_denom = np.where(denom == 0.0, 1.0, denom)
        alpha = (target - src_lo) / safe_denom

        new_lo = np.where(lo_match, lo, np.where(hi_match, hi, lo))
        new_hi = np.where(lo_match, lo, np.where(hi_match, hi, hi))
        new_alpha = np.where(lo_match | hi_match, 0.0, alpha)

        off_grid = new_lo != new_hi
        if policy is FrequencyPolicy.EXACT and np.any(off_grid):
            offenders = target[off_grid].tolist()
            raise ValueError(
                "frequency_policy='exact' rejected off-grid target frequencies "
                f"{offenders}; no native channel is within tolerance. "
                "Use policy='linear' (frequency_policy='linear' on a simulator) "
                "to interpolate."
            )

        all_idx = np.concatenate([new_lo, new_hi])
        unique_idx, inverse = np.unique(all_idx, return_inverse=True)
        lo_in_unique = inverse[: target.size].astype(np.int32)
        hi_in_unique = inverse[target.size :].astype(np.int32)

        direct = not np.any(off_grid)
        identity = (
            direct
            and target.size == source.size
            and np.array_equal(new_lo, np.arange(source.size))
        )
        return cls(
            unique_idx.astype(np.int32),
            lo_in_unique,
            hi_in_unique,
            new_alpha.astype(np.float64),
            policy=policy,
            mode="identity" if identity else ("gather" if direct else "linear"),
            source_size=source.size,
        )

    @property
    def source_indices(self):
        """Unique native source-grid indices the expensive products must be evaluated at."""
        return self.unique_native_idx

    def per_target_indices(self):
        """One native source-grid index per target frequency, in target order.

        The pre-interpolation index contract: duplicates preserved, aligned
        with the target grid. Only defined when every target snaps to a
        single native bin.

        :returns: int32 numpy array of length ``len(self)``.
        :raises ValueError: if any target is genuinely interpolated between
            two native bins.
        """
        lo = np.asarray(self.lo_in_unique)
        hi = np.asarray(self.hi_in_unique)
        if not np.array_equal(lo, hi):
            n_off = int(np.count_nonzero(lo != hi))
            raise ValueError(
                f"{n_off} target frequencies are off-grid (interpolated between "
                "two native bins), so per-target indices are undefined; use "
                "from_native/from_unique instead"
            )
        return np.asarray(self.unique_native_idx)[lo]

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
        expected = self.unique_native_idx.shape[0]
        if arr.ndim < 1 or arr.shape[0] != expected:
            raise ValueError(
                "unique array leading dimension must match source_indices "
                f"({expected}), got shape {arr.shape}"
            )
        if self.mode == "identity":
            return arr
        lo_vals = arr[self.lo_in_unique]
        if self.mode == "gather":
            return lo_vals
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
        if arr.ndim < 1 or arr.shape[0] != self.source_size:
            raise ValueError(
                f"native array leading dimension must be {self.source_size}, "
                f"got shape {arr.shape}"
            )
        if self.mode == "identity":
            return arr
        if self.mode == "gather":
            return arr[self.unique_native_idx[self.lo_in_unique]]
        return self.from_unique(arr[self.unique_native_idx])

    def tree_flatten(self):
        children = (self.unique_native_idx, self.lo_in_unique, self.hi_in_unique, self.alpha)
        return children, (self.policy.value, self.mode, self.source_size)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        policy, mode, source_size = aux_data
        return cls(
            *children,
            policy=policy,
            mode=mode,
            source_size=source_size,
        )

    def __repr__(self):
        n_offgrid = int(np.count_nonzero(np.asarray(self.alpha) != 0.0))
        return (
            f"FrequencyMap(policy={self.policy.value!r}, mode={self.mode!r}, "
            f"n_target={len(self)}, "
            f"n_source_touched={np.asarray(self.unique_native_idx).shape[0]}, "
            f"n_offgrid={n_offgrid})"
        )


def frequency_policy_from_config(freq_cfg):
    """Return the frequency policy in a YAML ``freq`` block (default: exact)."""
    return FrequencyPolicy.normalize(
        freq_cfg.get("policy", FrequencyPolicy.EXACT.value)
    )


def frequencies_from_config(freq_cfg):
    """Parse a YAML ``freq`` block into a numpy array of MHz values.

    Accepted forms::

        freq: { policy: exact, values: [10.0, 20.0, 30.0] }
        freq: { policy: exact, start: 1.0, end: 50.0, step: 1.0 }
        freq: { policy: linear, start: 1.0, end: 75.0, n: 75 }

    The ``step`` form follows numpy.arange semantics (half-open interval,
    matching the pre-interpolation parser). Float steps inherit arange's
    endpoint rounding caveats; for a grid that must contain the endpoint,
    prefer the ``n`` (linspace) or ``values`` forms.

    Legacy index-based forms (``indices``, ``start_idx``/``stop_idx``/``step_idx``)
    are no longer supported and produce a :class:`ValueError` naming the new keys.
    """
    frequency_policy_from_config(freq_cfg)
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
