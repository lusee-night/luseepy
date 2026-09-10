"""Flight-firmware spectrometer averaging-window contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from lusee.ingest.frequency_contract import (
    FREQUENCY_WINDOW_CONTRACT_VERSION,
    FREQUENCY_WINDOW_SOURCE_COMMIT,
    FrequencyWindowContract,
    UnresolvedFrequencyCoordinateError,
    spectrometer_frequency_window,
)


@pytest.mark.parametrize(
    ("navgf", "count", "stride", "offsets", "divisor"),
    [
        (1, 2048, 1, (0,), 1),
        (2, 1024, 2, (0, 1), 2),
        (3, 512, 4, (0, 1, 2), 4),
        (4, 512, 4, (0, 1, 2, 3), 4),
    ],
)
def test_reviewed_coreloop_windows(
    navgf: int,
    count: int,
    stride: int,
    offsets: tuple[int, ...],
    divisor: int,
):
    contract = spectrometer_frequency_window(navgf)

    assert contract.output_count == count
    assert contract.native_count == 2048
    assert contract.stride == stride
    assert contract.included_offsets == offsets
    assert contract.divisor == divisor
    assert contract.native_indices(0) == offsets
    assert contract.native_indices(7) == tuple(7 * stride + x for x in offsets)
    assert contract.native_indices(count - 1) == tuple(
        (count - 1) * stride + x for x in offsets
    )
    assert contract.all_native_indices().shape == (count, len(offsets))
    assert contract.contract_version == FREQUENCY_WINDOW_CONTRACT_VERSION
    assert contract.source_commit == FREQUENCY_WINDOW_SOURCE_COMMIT


def test_navgf_three_skips_fourth_bin_and_retains_divisor_four():
    three = spectrometer_frequency_window(3)
    four = spectrometer_frequency_window(4)

    assert three.output_count == four.output_count == 512
    assert three.native_indices(11) == (44, 45, 46)
    assert four.native_indices(11) == (44, 45, 46, 47)
    assert three.nominal_response_weights == (0.25, 0.25, 0.25)
    assert sum(three.nominal_response_weights) == 0.75
    assert sum(four.nominal_response_weights) == 1.0
    assert three.as_record()["integer_arithmetic"] == "averaging_mode_dependent"


def test_all_native_indices_are_exact_and_read_only():
    contract = spectrometer_frequency_window(2)
    indices = contract.all_native_indices()

    np.testing.assert_array_equal(
        indices[:4],
        np.array([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.int64),
    )
    assert indices[-1].tolist() == [2046, 2047]
    with pytest.raises(ValueError, match="read-only"):
        indices[0, 0] = 99


@pytest.mark.parametrize("navgf", [0, 5, -1, True, 1.0, "1", None])
def test_invalid_navgf_is_rejected(navgf):
    with pytest.raises(ValueError, match="Navgf"):
        spectrometer_frequency_window(navgf)


def test_invalid_output_index_is_rejected():
    contract = spectrometer_frequency_window(4)
    with pytest.raises(TypeError, match="integer"):
        contract.native_indices(True)
    with pytest.raises(IndexError, match="outside"):
        contract.native_indices(-1)
    with pytest.raises(IndexError, match="outside"):
        contract.native_indices(contract.output_count)


def test_mhz_coordinate_remains_explicitly_unresolved():
    contract = spectrometer_frequency_window(3)

    assert contract.frequency_coordinate_status == "unresolved"
    assert contract.as_record()["frequency_coordinate_status"] == "unresolved"
    with pytest.raises(UnresolvedFrequencyCoordinateError, match="unresolved"):
        contract.frequency_mhz()


@pytest.mark.parametrize(
    "change",
    [
        {"included_offsets": (0, 1, 3)},
        {"divisor": 3},
        {"output_count": 256},
        {"output_count": 512.0},
        {"native_count": 4096},
        {"contract_version": 99},
        {"contract_version": True},
        {"contract_version": 1.0},
        {"source_commit": "not-coreloop"},
    ],
)
def test_direct_construction_cannot_forge_firmware_contract(change):
    values = {
        "navgf": 3,
        "stride": 4,
        "included_offsets": (0, 1, 2),
        "divisor": 4,
        "output_count": 512,
    }
    values.update(change)

    with pytest.raises(ValueError):
        FrequencyWindowContract(**values)
