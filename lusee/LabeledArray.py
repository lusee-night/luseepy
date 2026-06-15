"""Thin units+frame label on a (JAX or NumPy) array.

A :class:`LabeledArray` decorates a single array with two free-form ASCII
string labels:

* ``units`` -- physical units, e.g. ``"V"``, ``"K"``, ``"V^2/Hz"``,
  ``"1"`` (dimensionless).
* ``frame`` -- coordinate frame, e.g. ``"galactic"``, ``"equatorial"``,
  ``"MCMF"``, ``"topo"``, ``"mepa"``.  See the ``FRAME_*`` constants.

The labels are purely informational: there are NO unit/frame checks.
They exist so that an array can be printed and inspected during
debugging ("which frame is this a_lm in, and what are its units?").

Drop-in behaviour
-----------------
``LabeledArray`` forwards the common array operations (arithmetic,
indexing, ``.shape``/``.dtype``/``.real``/``.imag``/``.conj()``,
``np.asarray`` / ``jnp.asarray`` conversion), so most code that expects a
plain array keeps working.  Binary operations and slicing carry the
label of the left/self operand forward; they do NOT attempt unit algebra.

It is registered as a JAX pytree with a single array child and
``(units, frame)`` as static aux data.  This means it can be passed
through ``jax.jit`` / ``grad`` / ``vmap`` -- but note that inside a
transform the pytree is flattened to the bare array and the label is
dropped until the result is re-wrapped at the Python boundary.  In
practice: keep the hot jitted kernels operating on bare arrays and
re-attach labels at the function boundaries (see ``label`` / ``relabel``).
NumPy ufuncs and ``jax.numpy`` functions likewise return bare arrays --
the label survives only the operators implemented here.
"""

import numpy as np
import jax


# Canonical ASCII frame labels used across luseepy.  These are conventions
# only; the frame field accepts any string (no validation is performed).
FRAME_GALACTIC = "galactic"      # astropy Galactic (l, b)
FRAME_EQUATORIAL = "equatorial"  # ICRS (RA, Dec)
FRAME_MCMF = "MCMF"              # lunarsky Moon-Centred Moon-Fixed
FRAME_TOPO = "topo"             # lunarsky LunarTopo (instrument/topocentric)
FRAME_MEPA = "mepa"             # croissant Moon Ephemeris Pole Axis


@jax.tree_util.register_pytree_node_class
class LabeledArray:
    """An array decorated with informational ``units`` and ``frame`` labels.

    :param array: The wrapped array (JAX or NumPy) or any array-like value.
    :param units: Optional units label (free-form ASCII string).
    :param frame: Optional coordinate-frame label (free-form ASCII string).
    """

    __slots__ = ("array", "units", "frame")

    def __init__(self, array, units=None, frame=None):
        if isinstance(array, LabeledArray):
            if units is None:
                units = array.units
            if frame is None:
                frame = array.frame
            array = array.array
        self.array = array
        self.units = units
        self.frame = frame

    # -- JAX pytree ---------------------------------------------------------
    def tree_flatten(self):
        return (self.array,), (self.units, self.frame)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        units, frame = aux_data
        (array,) = children
        return cls(array, units, frame)

    # -- pickle (explicit, so __slots__ + __getattr__ play nicely) ----------
    def __getstate__(self):
        return (self.array, self.units, self.frame)

    def __setstate__(self, state):
        self.array, self.units, self.frame = state

    # -- array / numpy / jax protocols --------------------------------------
    def __jax_array__(self):
        return self.array

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self.array, dtype=dtype)
        if copy:
            arr = arr.copy()
        return arr

    # -- attribute forwarding ----------------------------------------------
    def __getattr__(self, name):
        # __getattr__ runs only when normal lookup fails.  Guard the slots
        # and dunder names to avoid infinite recursion (e.g. during unpickle,
        # before ``array`` is set) and to let copy/pickle protocol fall back.
        if name.startswith("__") or name in ("array", "units", "frame"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "array"), name)

    # -- shape-like metadata (forwarded) ------------------------------------
    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def real(self):
        return LabeledArray(self.array.real, self.units, self.frame)

    @property
    def imag(self):
        return LabeledArray(self.array.imag, self.units, self.frame)

    @property
    def T(self):
        return LabeledArray(self.array.T, self.units, self.frame)

    # -- label-preserving views/methods -------------------------------------
    def _relabel(self, result):
        return LabeledArray(result, self.units, self.frame)

    def conj(self):
        return self._relabel(self.array.conj())

    conjugate = conj

    def ravel(self):
        return self._relabel(self.array.ravel())

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._relabel(self.array.reshape(*shape))

    def astype(self, dtype):
        return self._relabel(self.array.astype(dtype))

    # -- container protocol -------------------------------------------------
    def __getitem__(self, idx):
        return self._relabel(self.array[idx])

    def __len__(self):
        return len(self.array)

    def __iter__(self):
        # yield bare rows so numpy-style consumers keep working
        return iter(self.array)

    def __bool__(self):
        return bool(self.array)

    # -- arithmetic (carry self's label forward; no unit algebra) -----------
    def __neg__(self):
        return self._relabel(-self.array)

    def __pos__(self):
        return self._relabel(+self.array)

    def __abs__(self):
        return self._relabel(abs(self.array))

    def __add__(self, other):
        return self._relabel(self.array + _unwrap(other))

    def __radd__(self, other):
        return self._relabel(_unwrap(other) + self.array)

    def __sub__(self, other):
        return self._relabel(self.array - _unwrap(other))

    def __rsub__(self, other):
        return self._relabel(_unwrap(other) - self.array)

    def __mul__(self, other):
        return self._relabel(self.array * _unwrap(other))

    def __rmul__(self, other):
        return self._relabel(_unwrap(other) * self.array)

    def __truediv__(self, other):
        return self._relabel(self.array / _unwrap(other))

    def __rtruediv__(self, other):
        return self._relabel(_unwrap(other) / self.array)

    def __floordiv__(self, other):
        return self._relabel(self.array // _unwrap(other))

    def __rfloordiv__(self, other):
        return self._relabel(_unwrap(other) // self.array)

    def __mod__(self, other):
        return self._relabel(self.array % _unwrap(other))

    def __pow__(self, other):
        return self._relabel(self.array ** _unwrap(other))

    def __rpow__(self, other):
        return self._relabel(_unwrap(other) ** self.array)

    def __matmul__(self, other):
        return self._relabel(self.array @ _unwrap(other))

    def __rmatmul__(self, other):
        return self._relabel(_unwrap(other) @ self.array)

    # -- comparisons (elementwise, return bare arrays like numpy/jax) -------
    def __eq__(self, other):
        return self.array == _unwrap(other)

    def __ne__(self, other):
        return self.array != _unwrap(other)

    def __lt__(self, other):
        return self.array < _unwrap(other)

    def __le__(self, other):
        return self.array <= _unwrap(other)

    def __gt__(self, other):
        return self.array > _unwrap(other)

    def __ge__(self, other):
        return self.array >= _unwrap(other)

    __hash__ = None  # mutable, array-backed: not hashable (matches np/jax)

    # -- printing -----------------------------------------------------------
    def __repr__(self):
        try:
            shape = tuple(self.array.shape)
            dtype = self.array.dtype
        except AttributeError:
            shape = None
            dtype = type(self.array).__name__
        return (f"LabeledArray(units={self.units!r}, frame={self.frame!r}, "
                f"shape={shape}, dtype={dtype})")


def _unwrap(x):
    """Return the bare array if ``x`` is a LabeledArray, else ``x`` unchanged."""
    return x.array if isinstance(x, LabeledArray) else x


def label(array, units=None, frame=None):
    """Wrap ``array`` with ``units``/``frame`` labels.

    Idempotent: if ``array`` is already a LabeledArray, only the labels you
    pass are overridden; unspecified labels are kept.

    :param array: Array (or LabeledArray) to label.
    :param units: Units label, or None to keep existing/none.
    :param frame: Frame label, or None to keep existing/none.
    :returns: A LabeledArray.
    """
    return LabeledArray(array, units, frame)


# ``relabel`` reads naturally at call sites that change an existing label.
relabel = label


def asarray(x):
    """Return the bare underlying array (unwrap a LabeledArray, else pass through)."""
    return _unwrap(x)


def units_of(x):
    """Return the units label of ``x``, or None if it is not a LabeledArray."""
    return x.units if isinstance(x, LabeledArray) else None


def frame_of(x):
    """Return the frame label of ``x``, or None if it is not a LabeledArray."""
    return x.frame if isinstance(x, LabeledArray) else None


def is_labeled(x):
    """Return True if ``x`` is a LabeledArray."""
    return isinstance(x, LabeledArray)


def describe(x):
    """Return a short human-readable description of an array for debugging.

    Works on LabeledArray (shows units/frame) and on bare arrays.
    """
    if isinstance(x, LabeledArray):
        return repr(x)
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", type(x).__name__)
    return f"array(shape={tuple(shape) if shape is not None else None}, dtype={dtype}) [unlabeled]"
