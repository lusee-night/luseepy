import numpy as np
import pytest

from lusee.frequencies import (
    FrequencyMap,
    frequencies_from_config,
    interp1d,
    interpolation_weights,
)


def _check_map(fmap, source, expected_target_vals, atol):
    """Reconstruct interpolated values from the map and compare to expected."""
    unique_vals = np.asarray(source)[fmap.unique_native_idx]
    lo_vals = unique_vals[fmap.lo_in_unique]
    hi_vals = unique_vals[fmap.hi_in_unique]
    reconstructed = (1.0 - fmap.alpha) * lo_vals + fmap.alpha * hi_vals
    np.testing.assert_allclose(reconstructed, expected_target_vals, atol=atol)


def test_exact_match_snaps_to_alpha_zero():
    source = np.linspace(1.0, 50.0, 50)
    target = np.asarray([1.0, 10.0, 25.0, 50.0])
    fmap = interpolation_weights(target, source)

    assert isinstance(fmap, FrequencyMap)
    assert np.all(fmap.alpha == 0.0)
    assert np.all(fmap.lo_in_unique == fmap.hi_in_unique)
    _check_map(fmap, source, target, atol=0.0)


def test_near_match_within_atol_snaps():
    # Regression for the original canonicalization bug: np.arange(1,51,5) drifts
    # vs np.linspace(1,50,50)[::5]. The map must snap with alpha = 0.0 exactly.
    source = np.linspace(1.0, 50.0, 50)
    target = source + 1e-10
    fmap = interpolation_weights(target, source, atol=1e-6, rtol=1e-9)

    assert np.all(fmap.alpha == 0.0)
    assert np.all(fmap.lo_in_unique == fmap.hi_in_unique)


def test_midpoint_interpolation():
    source = np.linspace(0.0, 10.0, 11)
    target = np.asarray([0.5, 1.5, 9.5])
    fmap = interpolation_weights(target, source)

    np.testing.assert_allclose(fmap.alpha, [0.5, 0.5, 0.5])
    _check_map(fmap, source, target, atol=1e-12)


def test_out_of_range_raises():
    source = np.linspace(1.0, 50.0, 50)

    with pytest.raises(ValueError, match=r"out of range"):
        interpolation_weights([55.0], source)

    with pytest.raises(ValueError, match=r"out of range"):
        interpolation_weights([0.5], source)


def test_boundary_snap_at_endpoints():
    source = np.linspace(1.0, 50.0, 50)

    # Exactly at endpoints: snap, no error.
    fmap_lo = interpolation_weights([1.0], source)
    fmap_hi = interpolation_weights([50.0], source)
    assert fmap_lo.alpha[0] == 0.0
    assert fmap_hi.alpha[0] == 0.0
    assert fmap_lo.lo_in_unique[0] == fmap_lo.hi_in_unique[0]
    assert fmap_hi.lo_in_unique[0] == fmap_hi.hi_in_unique[0]
    assert int(fmap_lo.unique_native_idx[fmap_lo.lo_in_unique[0]]) == 0
    assert int(fmap_hi.unique_native_idx[fmap_hi.lo_in_unique[0]]) == 49

    # Within boundary atol on the outside: snap, no error.
    fmap_eps = interpolation_weights([50.0 + 1e-12], source, atol=1e-6)
    assert fmap_eps.alpha[0] == 0.0


def test_unique_native_idx_deduplication():
    source = np.linspace(1.0, 50.0, 50)
    target = np.asarray([10.0, 10.0, 10.0, 20.5])
    fmap = interpolation_weights(target, source)

    # 10.0 snaps to index 9; 20.5 brackets indices 19 and 20.
    assert sorted(fmap.unique_native_idx.tolist()) == [9, 19, 20]
    # First three targets all reference the snapped index for 10.0.
    assert fmap.lo_in_unique[0] == fmap.lo_in_unique[1] == fmap.lo_in_unique[2]
    assert fmap.alpha[0] == fmap.alpha[1] == fmap.alpha[2] == 0.0
    # Fourth target is genuinely interpolated.
    np.testing.assert_allclose(fmap.alpha[3], 0.5)


def test_non_increasing_source_raises():
    with pytest.raises(ValueError, match=r"strictly increasing"):
        interpolation_weights([5.0], np.asarray([10.0, 5.0, 1.0]))


def test_interp1d_recovers_native_values_on_snap():
    source = np.linspace(1.0, 50.0, 50)
    data = np.cos(source) + 2.0
    fmap = interpolation_weights(source, source)
    result = interp1d(fmap, data)
    np.testing.assert_allclose(result, data, atol=0.0)


def test_interp1d_multidim_broadcasts_along_first_axis():
    source = np.linspace(0.0, 10.0, 11)
    data = np.arange(11 * 3 * 2, dtype=float).reshape(11, 3, 2)
    target = np.asarray([0.5, 5.5])
    fmap = interpolation_weights(target, source)
    result = interp1d(fmap, data)

    expected = 0.5 * data[[0, 5], ...] + 0.5 * data[[1, 6], ...]
    np.testing.assert_allclose(result, expected, atol=1e-12)


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
