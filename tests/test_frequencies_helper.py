import numpy as np
import pytest

from lusee.frequencies import (
    FrequencyMap,
    frequencies_from_config,
)


def _check_map(fmap, source, expected_target_vals, atol):
    """Interpolate the source grid through the map and compare to expected."""
    reconstructed = fmap.from_native(np.asarray(source))
    np.testing.assert_allclose(reconstructed, expected_target_vals, atol=atol)


def test_exact_match_snaps_to_alpha_zero():
    source = np.linspace(1.0, 50.0, 50)
    target = np.asarray([1.0, 10.0, 25.0, 50.0])
    fmap = FrequencyMap.build(target, source)

    assert isinstance(fmap, FrequencyMap)
    assert np.all(fmap.alpha == 0.0)
    assert np.all(fmap.lo_in_unique == fmap.hi_in_unique)
    _check_map(fmap, source, target, atol=0.0)


def test_near_match_within_atol_snaps():
    # Regression for the original canonicalization bug: np.arange(1,51,5) drifts
    # vs np.linspace(1,50,50)[::5]. The map must snap with alpha = 0.0 exactly.
    source = np.linspace(1.0, 50.0, 50)
    target = source + 1e-10
    fmap = FrequencyMap.build(target, source, atol=1e-6, rtol=1e-9)

    assert np.all(fmap.alpha == 0.0)
    assert np.all(fmap.lo_in_unique == fmap.hi_in_unique)


def test_midpoint_interpolation():
    source = np.linspace(0.0, 10.0, 11)
    target = np.asarray([0.5, 1.5, 9.5])
    fmap = FrequencyMap.build(target, source)

    np.testing.assert_allclose(fmap.alpha, [0.5, 0.5, 0.5])
    _check_map(fmap, source, target, atol=1e-12)


def test_out_of_range_raises():
    source = np.linspace(1.0, 50.0, 50)

    with pytest.raises(ValueError, match=r"out of range"):
        FrequencyMap.build([55.0], source)

    with pytest.raises(ValueError, match=r"out of range"):
        FrequencyMap.build([0.5], source)


def test_boundary_snap_at_endpoints():
    source = np.linspace(1.0, 50.0, 50)

    # Exactly at endpoints: snap, no error.
    fmap_lo = FrequencyMap.build([1.0], source)
    fmap_hi = FrequencyMap.build([50.0], source)
    assert fmap_lo.alpha[0] == 0.0
    assert fmap_hi.alpha[0] == 0.0
    assert fmap_lo.lo_in_unique[0] == fmap_lo.hi_in_unique[0]
    assert fmap_hi.lo_in_unique[0] == fmap_hi.hi_in_unique[0]
    assert int(fmap_lo.source_indices[fmap_lo.lo_in_unique[0]]) == 0
    assert int(fmap_hi.source_indices[fmap_hi.lo_in_unique[0]]) == 49

    # Within boundary atol on the outside: snap, no error.
    fmap_eps = FrequencyMap.build([50.0 + 1e-12], source, atol=1e-6)
    assert fmap_eps.alpha[0] == 0.0


def test_source_indices_deduplication():
    source = np.linspace(1.0, 50.0, 50)
    target = np.asarray([10.0, 10.0, 10.0, 20.5])
    fmap = FrequencyMap.build(target, source)

    # 10.0 snaps to index 9; 20.5 brackets indices 19 and 20.
    assert sorted(np.asarray(fmap.source_indices).tolist()) == [9, 19, 20]
    # First three targets all reference the snapped index for 10.0.
    assert fmap.lo_in_unique[0] == fmap.lo_in_unique[1] == fmap.lo_in_unique[2]
    assert fmap.alpha[0] == fmap.alpha[1] == fmap.alpha[2] == 0.0
    # Fourth target is genuinely interpolated.
    np.testing.assert_allclose(fmap.alpha[3], 0.5)


def test_non_increasing_source_raises():
    with pytest.raises(ValueError, match=r"strictly increasing"):
        FrequencyMap.build([5.0], np.asarray([10.0, 5.0, 1.0]))


def test_none_grids_raise():
    source = np.linspace(1.0, 50.0, 50)

    with pytest.raises(ValueError, match=r"target_freqs is None"):
        FrequencyMap.build(None, source)

    # e.g. a sky model constructed with freq=None and no get_alm_at_freq
    with pytest.raises(ValueError, match=r"source_freqs is None"):
        FrequencyMap.build([5.0], None)


def test_empty_target_raises():
    source = np.linspace(1.0, 50.0, 50)

    with pytest.raises(ValueError, match=r"target_freqs is empty"):
        FrequencyMap.build([], source)


def test_non_finite_frequencies_raise():
    source = np.linspace(1.0, 50.0, 50)

    with pytest.raises(ValueError, match=r"non-finite"):
        FrequencyMap.build([np.nan], source)

    with pytest.raises(ValueError, match=r"non-finite"):
        FrequencyMap.build([np.inf], source)

    with pytest.raises(ValueError, match=r"non-finite"):
        FrequencyMap.build([5.0], np.asarray([1.0, np.nan, 50.0]))


def test_from_native_recovers_native_values_on_snap():
    source = np.linspace(1.0, 50.0, 50)
    data = np.cos(source) + 2.0
    fmap = FrequencyMap.build(source, source)
    result = fmap.from_native(data)
    np.testing.assert_allclose(result, data, atol=0.0)


def test_from_native_multidim_broadcasts_along_first_axis():
    source = np.linspace(0.0, 10.0, 11)
    data = np.arange(11 * 3 * 2, dtype=float).reshape(11, 3, 2)
    target = np.asarray([0.5, 5.5])
    fmap = FrequencyMap.build(target, source)
    result = fmap.from_native(data)

    expected = 0.5 * data[[0, 5], ...] + 0.5 * data[[1, 6], ...]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_from_unique_matches_from_native_on_full_grid():
    # from_native(full) and from_unique(full[source_indices]) must agree: this
    # is the exact split the simulators rely on (cheap full arrays via
    # from_native vs expensive alm products pre-reduced to source_indices via
    # from_unique).
    source = np.linspace(1.0, 50.0, 50)
    data = np.cos(source)[:, None] * np.arange(1, 4)[None, :]
    target = np.asarray([12.5, 12.5, 30.0, 50.0])
    fmap = FrequencyMap.build(target, source)

    via_full = fmap.from_native(data)
    via_unique = fmap.from_unique(data[fmap.source_indices])
    np.testing.assert_allclose(via_full, via_unique, atol=0.0)


def test_interpolation_is_differentiable_wrt_values():
    # Differentiability is the point of the JAX path: gradients must flow
    # through the off-grid blend back to the beam/sky values.
    import jax
    import jax.numpy as jnp

    source = np.linspace(1.0, 50.0, 50)
    target = np.asarray([12.5, 30.0])  # one genuinely off-grid, one on-grid
    fmap = FrequencyMap.build(target, source)

    unique_vals = jnp.asarray(
        np.random.default_rng(0).standard_normal((fmap.source_indices.size, 3))
    )

    def loss(vals):
        return jnp.sum(fmap.from_unique(vals) ** 2)

    grad = jax.grad(loss)(unique_vals)
    assert grad.shape == unique_vals.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_frequencymap_class_api():
    source = np.linspace(1.0, 50.0, 50)
    data = np.cos(source)[:, None] * np.arange(1, 4)[None, :]
    target = np.asarray([12.5, 12.5, 30.0, 50.0])
    fmap = FrequencyMap.build(target, source)

    assert len(fmap) == len(target)
    # source_indices is the dedup'd lookup table (12.5 brackets 11/12, 30/50 snap).
    np.testing.assert_array_equal(np.asarray(fmap.source_indices), [11, 12, 29, 49])
    np.testing.assert_allclose(
        fmap.from_unique(data[fmap.source_indices]), fmap.from_native(data), atol=0.0
    )
    assert "n_target=4" in repr(fmap)


def test_frequencymap_is_jax_pytree():
    import jax
    import jax.numpy as jnp

    source = np.linspace(1.0, 50.0, 50)
    fmap = FrequencyMap.build(np.asarray([12.5, 30.0]), source)

    leaves, treedef = jax.tree_util.tree_flatten(fmap)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(np.asarray(rebuilt.alpha), np.asarray(fmap.alpha))

    # Passing the map as a traced argument through jit must work, and gradients
    # w.r.t. the interpolated values must flow.
    vals = jnp.asarray(
        np.random.default_rng(0).standard_normal((fmap.source_indices.shape[0], 2))
    )

    @jax.jit
    def loss(m, v):
        return jnp.sum(m.from_unique(v) ** 2)

    grad = jax.grad(loss, argnums=1)(fmap, vals)
    assert grad.shape == vals.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_frequencies_from_config_values():
    freq = frequencies_from_config({"values": [10.0, 20.0, 30.0]})
    np.testing.assert_allclose(freq, [10.0, 20.0, 30.0])


def test_frequencies_from_config_start_end_step_inclusive():
    freq = frequencies_from_config({"start": 1.0, "end": 5.0, "step": 1.0})
    np.testing.assert_allclose(freq, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_frequencies_from_config_start_end_n():
    freq = frequencies_from_config({"start": 1.0, "end": 75.0, "n": 75})
    assert freq.shape == (75,)
    assert freq[0] == 1.0
    assert freq[-1] == 75.0


def test_frequencies_from_config_rejects_legacy_keys():
    for legacy in [
        {"indices": [0, 1]},
        {"start_idx": 0, "stop_idx": 5},
        {"step_idx": 2},
    ]:
        with pytest.raises(ValueError, match=r"no longer supported"):
            frequencies_from_config(legacy)


def test_frequencies_from_config_step_and_n_conflict():
    with pytest.raises(ValueError, match=r"'step' or 'n'"):
        frequencies_from_config({"start": 1.0, "end": 5.0, "step": 1.0, "n": 5})


def test_frequencies_from_config_invalid_step():
    with pytest.raises(ValueError, match=r"positive"):
        frequencies_from_config({"start": 1.0, "end": 5.0, "step": 0.0})


def test_frequencies_from_config_empty_raises():
    with pytest.raises(ValueError, match=r"one of"):
        frequencies_from_config({"foo": "bar"})

