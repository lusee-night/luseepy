"""Unit tests for lusee.LabeledArray (units+frame array wrapper)."""
import pickle

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from lusee.LabeledArray import (
    LabeledArray, label, relabel, asarray, units_of, frame_of, is_labeled,
    describe, FRAME_GALACTIC, FRAME_MCMF,
)


def test_construction_and_attributes():
    a = jnp.arange(4.0)
    la = LabeledArray(a, units="V", frame="galactic")
    assert la.units == "V"
    assert la.frame == "galactic"
    assert la.shape == (4,)
    assert la.ndim == 1
    assert la.size == 4
    assert la.dtype == a.dtype


def test_repr_is_informative_and_ascii():
    la = LabeledArray(jnp.zeros((2, 3), dtype=jnp.complex64), units="K", frame="MCMF")
    r = repr(la)
    assert "LabeledArray" in r and "units='K'" in r and "frame='MCMF'" in r
    assert "(2, 3)" in r
    assert r.isascii()


def test_no_double_wrap():
    la = LabeledArray(jnp.arange(3), units="V", frame="topo")
    la2 = LabeledArray(la)  # keep labels
    assert not isinstance(la2.array, LabeledArray)
    assert la2.units == "V" and la2.frame == "topo"
    la3 = LabeledArray(la, units="K")  # override one
    assert la3.units == "K" and la3.frame == "topo"


def test_label_helper_idempotent():
    la = label(np.arange(3), units="V", frame="galactic")
    assert is_labeled(la) and la.units == "V"
    la2 = relabel(la, units="V^2/Hz")
    assert la2.units == "V^2/Hz" and la2.frame == "galactic"
    assert not is_labeled(la2.array)


def test_numpy_conversion():
    base = np.array([1.0, 2.0, 3.0])
    la = LabeledArray(base, units="K")
    out = np.asarray(la)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, base)
    # dtype kwarg honoured
    out2 = np.asarray(la, dtype=np.complex128)
    assert out2.dtype == np.complex128


def test_jax_conversion_and_ops():
    la = LabeledArray(jnp.arange(4.0), units="V", frame="galactic")
    j = jnp.asarray(la)
    assert isinstance(j, jax.Array)
    np.testing.assert_array_equal(np.asarray(j), np.arange(4.0))
    # jnp functions coerce via __jax_array__ and return bare arrays
    s = jnp.sum(la)
    assert not isinstance(s, LabeledArray)
    assert float(s) == 6.0


def test_arithmetic_carries_label():
    la = LabeledArray(jnp.array([1.0, 2.0, 3.0]), units="V", frame="galactic")
    r = la * 2
    assert isinstance(r, LabeledArray) and r.units == "V" and r.frame == "galactic"
    np.testing.assert_array_equal(np.asarray(r), [2.0, 4.0, 6.0])
    # reflected op
    r2 = 10.0 - la
    assert isinstance(r2, LabeledArray) and r2.units == "V"
    np.testing.assert_array_equal(np.asarray(r2), [9.0, 8.0, 7.0])
    # operate with a bare jax array
    r3 = la + jnp.ones(3)
    np.testing.assert_array_equal(np.asarray(r3), [2.0, 3.0, 4.0])
    # operate with another LabeledArray (left label wins)
    other = LabeledArray(jnp.ones(3), units="dimensionless", frame="topo")
    r4 = la + other
    assert r4.units == "V" and r4.frame == "galactic"
    # negation / abs preserve label
    assert (-la).units == "V"
    assert abs(LabeledArray(jnp.array([-1.0]), units="K")).units == "K"


def test_indexing_and_views_preserve_label():
    la = LabeledArray(jnp.arange(6.0).reshape(2, 3), units="K", frame="MCMF")
    row = la[0]
    assert isinstance(row, LabeledArray) and row.units == "K" and row.frame == "MCMF"
    np.testing.assert_array_equal(np.asarray(row), [0.0, 1.0, 2.0])
    assert la.real.units == "K"
    assert la.T.shape == (3, 2) and la.T.units == "K"
    assert la.ravel().shape == (6,) and la.ravel().units == "K"
    assert la.reshape(3, 2).units == "K"
    assert la.reshape((3, 2)).shape == (3, 2)
    assert la.astype(jnp.complex64).dtype == jnp.complex64


def test_complex_conj_and_imag():
    la = LabeledArray(jnp.array([1 + 2j, 3 - 1j]), units="1", frame="equatorial")
    assert la.imag.units == "1"
    c = la.conj()
    assert isinstance(c, LabeledArray) and c.units == "1"
    np.testing.assert_array_equal(np.asarray(c), [1 - 2j, 3 + 1j])


def test_iteration_yields_bare_rows():
    la = LabeledArray(jnp.arange(6.0).reshape(3, 2), units="K")
    rows = list(la)
    assert len(rows) == 3
    assert not any(isinstance(r, LabeledArray) for r in rows)


def test_comparisons_return_arrays():
    la = LabeledArray(jnp.array([1.0, 2.0, 3.0]), units="V")
    mask = la > 1.5
    assert not isinstance(mask, LabeledArray)
    np.testing.assert_array_equal(np.asarray(mask), [False, True, True])


def test_getattr_forwards_unknown_methods():
    la = LabeledArray(jnp.array([1.0, 2.0, 3.0]), units="V")
    # .mean is not explicitly defined -> forwarded to the bare array
    assert float(la.mean()) == 2.0


def test_pytree_flatten_unflatten_roundtrip():
    la = LabeledArray(jnp.arange(4.0), units="V", frame="galactic")
    leaves, treedef = jax.tree_util.tree_flatten(la)
    assert len(leaves) == 1
    np.testing.assert_array_equal(np.asarray(leaves[0]), np.arange(4.0))
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, LabeledArray)
    assert rebuilt.units == "V" and rebuilt.frame == "galactic"


def test_pytree_through_jit_preserves_labels():
    la = LabeledArray(jnp.arange(4.0), units="V", frame="galactic")

    @jax.jit
    def double(box):
        # inside jit the wrapper is flattened; rebuild explicitly
        return LabeledArray(box.array * 2, box.units, box.frame)

    out = double(la)
    assert isinstance(out, LabeledArray)
    assert out.units == "V" and out.frame == "galactic"
    np.testing.assert_array_equal(np.asarray(out), [0.0, 2.0, 4.0, 6.0])


def test_grad_through_child():
    la = LabeledArray(jnp.array([1.0, 2.0, 3.0]), units="V")
    g = jax.grad(lambda box: jnp.sum(box.array ** 2))(la)
    # gradient is returned as the same pytree (LabeledArray) with same aux
    assert isinstance(g, LabeledArray)
    assert g.units == "V"
    np.testing.assert_array_equal(np.asarray(g), [2.0, 4.0, 6.0])


def test_vmap_over_child():
    batch = LabeledArray(jnp.arange(6.0).reshape(3, 2), units="K", frame="MCMF")
    out = jax.vmap(lambda box: jnp.sum(box.array))(batch)
    np.testing.assert_array_equal(np.asarray(out), [1.0, 5.0, 9.0])


def test_pickle_roundtrip():
    la = LabeledArray(np.arange(5.0), units="V^2/Hz", frame="topo")
    blob = pickle.dumps(la)
    back = pickle.loads(blob)
    assert isinstance(back, LabeledArray)
    assert back.units == "V^2/Hz" and back.frame == "topo"
    np.testing.assert_array_equal(np.asarray(back), np.arange(5.0))


def test_helpers():
    la = LabeledArray(jnp.arange(3), units="V", frame="galactic")
    assert asarray(la) is la.array
    assert asarray(5) == 5
    assert units_of(la) == "V" and frame_of(la) == "galactic"
    assert units_of(jnp.arange(3)) is None and frame_of(np.arange(3)) is None
    assert is_labeled(la) and not is_labeled(np.arange(3))
    assert "units='V'" in describe(la)
    assert "unlabeled" in describe(np.arange(3))


def test_frame_constants():
    assert FRAME_GALACTIC == "galactic" and FRAME_MCMF == "MCMF"


def test_bool_on_scalar():
    assert bool(LabeledArray(jnp.array(True)))
    assert not bool(LabeledArray(jnp.array(0.0)))
