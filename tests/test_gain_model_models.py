"""Regression tests for process-wide gain-model family selection.

The oracle below reads ``alpha_refit.csv`` directly instead of using any
coefficient tables compiled by :mod:`lusee.GainModel`.  This makes the tests
useful for catching both selector mistakes and stale model-dependent caches.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import lusee.GainModel as gm


TELEMETRY = {
    "THERM_FPGA": 30.4,
    "SPE_ADC0_T": 29.8,
    "SPE_ADC1_T": 28.5,
    "SPE_1VAD8_V": 1.799,
    "VMON_1V2D": 1.201,
    "SPE_1VAD8_C": 0.045,
}


# Frozen representative outputs from alpha_refit.csv.  In particular, the
# negative 0.1-MHz values in three linear-containing H3 models are expected;
# gain-to-ASD conversion masks non-positive gain rather than changing the fit.
H3_GOLDEN = {
    ("linear", "linear"): np.array([
        -0.4239864291, 1.1911091274, 1.2270033922, 1.1139011835,
        1.0999954407, 1.0615511175, 0.9785145884, 0.8392555688,
        0.7743174389, 0.8117779535, 0.8515531661, 0.9473737947,
        1.0789495190, 1.0165654802, 0.5210602283, 0.2520915407,
    ]),
    ("linear", "quadratic"): np.array([
        0.2842812261, 1.2334924634, 1.2569455479, 1.2114391928,
        1.1841723516, 1.1048277704, 1.0025436704, 0.8879198148,
        0.8153601874, 0.8355617241, 0.8633738771, 0.9109388154,
        0.9505701123, 0.8638846827, 0.4674357855, 0.2134198558,
    ]),
    ("quadratic", "linear"): np.array([
        -0.3043603165, 3.2585947144, 3.2126785123, 2.8600259693,
        2.9548927064, 2.8932898860, 2.7219057159, 2.5788800723,
        2.5547639224, 2.8116882026, 3.1587327732, 3.5389577729,
        3.7620643279, 3.4095481779, 1.9018803554, 1.1237094819,
    ]),
    ("quadratic", "quadratic"): np.array([
        0.4039073388, 3.3009780504, 3.2426206680, 2.9575639785,
        3.0390696173, 2.9365665389, 2.7459347979, 2.6275443183,
        2.5958066709, 2.8354719732, 3.1705534842, 3.5025227937,
        3.6336849212, 3.2568673803, 1.8482559126, 1.0850377970,
    ]),
}


@pytest.fixture(autouse=True)
def restore_process_models():
    """Do not let process-wide selector state leak into another test."""
    before = gm.get_models()
    try:
        yield
    finally:
        gm.set_models(**before)


def _features(channel: int) -> dict[str, float]:
    adc_key = "SPE_ADC0_T" if channel <= 1 else "SPE_ADC1_T"
    t = TELEMETRY["THERM_FPGA"]
    adc = TELEMETRY[adc_key]
    return {
        "1": 1.0,
        "THERM_FPGA": t,
        adc_key: adc,
        "SPE_1VAD8_V": TELEMETRY["SPE_1VAD8_V"],
        "VMON_1V2D": TELEMETRY["VMON_1V2D"],
        "SPE_1VAD8_C": TELEMETRY["SPE_1VAD8_C"],
        "THERM_FPGA*THERM_FPGA": t * t,
        f"{adc_key}*{adc_key}": adc * adc,
        f"THERM_FPGA*{adc_key}": t * adc,
    }


def _reference_gain(level: str, channel: int, pc1: str, pc2: str) -> np.ndarray:
    """Evaluate the CSV and PCA files without GainModel's compiled tables."""
    data_dir = Path(gm.__file__).resolve().parent / "data" / "gain"
    key = f"{level}{channel}"
    selected = {"PC1": pc1, "PC2": pc2}
    features = _features(channel)
    scores = {"PC1": 0.0, "PC2": 0.0}

    with (data_dir / "alpha_refit.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            component = row["component"]
            if (row["gain_setting"] == key
                    and row["model"] == selected[component]):
                scores[component] += (
                    float(row["alpha_refit"]) * features[row["term"]]
                )

    mean = np.load(data_dir / f"{key}_mean.npy")
    eigvecs = np.load(data_dir / f"{key}_eigvecs.npy")
    return mean + scores["PC1"] * eigvecs[:, 0] + scores["PC2"] * eigvecs[:, 1]


@pytest.mark.parametrize("pc1", ["linear", "quadratic"])
@pytest.mark.parametrize("pc2", ["linear", "quadratic"])
def test_all_family_combinations_match_independent_reference(pc1, pc2):
    selected = gm.set_models(pc1=pc1.upper(), pc2=f"  {pc2}  ")
    assert selected == {"pc1": pc1, "pc2": pc2}
    assert gm.get_models() == selected

    model = gm.SpectrometerGain()
    for level in ("L", "M", "H"):
        for channel in range(4):
            expected = _reference_gain(level, channel, pc1, pc2)
            actual = model.predict_gain(level, channel, TELEMETRY)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-13)

    np.testing.assert_allclose(
        model.predict_gain("H", 3, TELEMETRY),
        H3_GOLDEN[(pc1, pc2)],
        rtol=0,
        atol=5e-10,
    )


def test_existing_instance_does_not_reuse_stale_model_selection():
    """The same object must observe every process-wide selector transition."""
    model = gm.SpectrometerGain()

    gm.set_models(pc1="quadratic", pc2="quadratic")
    qq = model.predict_gain("H", 3, TELEMETRY)

    gm.set_models(pc1="quadratic", pc2="linear")
    ql = model.predict_gain("H", 3, TELEMETRY)
    assert not np.allclose(ql, qq, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        ql, H3_GOLDEN[("quadratic", "linear")], rtol=0, atol=5e-10
    )

    # This catches a cache keyed only by gain level/channel: returning to Q/Q
    # must reproduce the first answer even though Q/L was evaluated in between.
    gm.set_models(pc1="quadratic", pc2="quadratic")
    qq_again = model.predict_gain("H", 3, TELEMETRY)
    np.testing.assert_array_equal(qq_again, qq)


def test_set_models_is_atomic_and_returned_dicts_are_not_live_state():
    returned = gm.set_models(pc1="quadratic", pc2="linear")
    returned["pc1"] = "linear"
    observed = gm.get_models()
    assert observed == {"pc1": "quadratic", "pc2": "linear"}
    observed["pc2"] = "quadratic"
    assert gm.get_models() == {"pc1": "quadratic", "pc2": "linear"}

    with pytest.raises(ValueError):
        gm.set_models(pc1="not-a-model", pc2="quadratic")
    assert gm.get_models() == {"pc1": "quadratic", "pc2": "linear"}

