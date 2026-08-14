"""Tests for lusee.GainModel (spectrometer counts -> nV/sqrt(Hz) conversion)."""
import numpy as np
import pytest

import lusee
from lusee.GainModel import (
    SpectrometerGain, counts_to_nv_auto, counts_to_nv_cross,
    counts_to_nv2_auto, counts_to_nv2_cross,
    asd_to_psd, psd_to_asd,
    bin_frequencies, CHANNEL_BIN_MHZ,
    NV_PER_SQRT_HZ, NV2_PER_HZ, V2_PER_HZ,
    CROSS_PRODUCT_CHANNELS,
)
from lusee.LabeledArray import LabeledArray, label


# Telemetry used to capture the golden anchor gains from the reference
# gain_model implementation (spectrometer_gain.SpectrometerGain).
TELE = {
    "THERM_FPGA": 30.4, "SPE_ADC0_T": 29.8, "SPE_ADC1_T": 28.5,
    "SPE_1VAD8_V": 1.799, "VMON_1V2D": 1.201, "SPE_1VAD8_C": 0.045,
}

# Golden anchor gain spectra (from gain_model/scripts/spectrometer_gain.py).
GOLDEN = {
    ("H", 3): [0.40390734, 3.30097805, 3.24262067, 2.95756398, 3.03906962,
               2.93656654, 2.7459348, 2.62754432, 2.59580667, 2.83547197,
               3.17055348, 3.50252279, 3.63368492, 3.25686738, 1.84825591, 1.0850378],
    ("L", 0): [0.00064701, 0.00074811, 0.00074319, 0.00074968, 0.00075748,
               0.00072934, 0.0007128, 0.0007059, 0.0006882, 0.00067795,
               0.00064809, 0.00057659, 0.00047874, 0.00035648, 0.0001708, 8.094e-05],
    ("M", 2): [0.00887006, 0.05437538, 0.05390339, 0.05113203, 0.0519566,
               0.0500867, 0.04737336, 0.04590235, 0.04644503, 0.05230481,
               0.0602209, 0.0694016, 0.07497182, 0.07054102, 0.04694652, 0.02852607],
}
ANCHOR_FREQS = [0.1, 0.7, 1.1, 3.1, 5.1, 10.1, 15.1, 20.1, 25.1, 30.1,
                35.1, 40.1, 45.1, 50.1, 60.1, 70.1]


@pytest.fixture(scope="module")
def sg():
    return SpectrometerGain()  # uses vendored lusee/data/gain


def test_predict_gain_matches_golden(sg):
    """Lock the PCA + quadratic-regression math against the production model."""
    for (lvl, ch), expected in GOLDEN.items():
        gain = sg.predict_gain(lvl, ch, TELE)
        assert gain.shape == (16,)
        np.testing.assert_allclose(gain, expected, rtol=0, atol=1e-7)


def test_anchor_freqs(sg):
    np.testing.assert_allclose(sg.anchor_freqs("H", 3), ANCHOR_FREQS)


def test_all_gain_keys_load(sg):
    for lvl in ("L", "M", "H"):
        for ch in range(4):
            g = sg.predict_gain(lvl, ch, TELE)
            assert g.shape == (16,) and np.all(np.isfinite(g))


def test_spline_passes_through_anchors(sg):
    """Interpolating at the anchor frequencies returns the anchor gains."""
    anchors = np.array(ANCHOR_FREQS)
    at_anchors = sg.predict_gain("H", 3, TELE, freqs_mhz=anchors)
    np.testing.assert_allclose(at_anchors, GOLDEN[("H", 3)], rtol=0, atol=1e-7)


def test_interp_out_of_range_is_nan(sg):
    # below first anchor (0.1) and above last (70.1) -> NaN (no extrapolation)
    g = sg.predict_gain("H", 3, TELE, freqs_mhz=np.array([0.0, 10.1, 100.0]))
    assert np.isnan(g[0]) and np.isnan(g[2]) and np.isfinite(g[1])


def test_invalid_level_or_channel(sg):
    with pytest.raises(ValueError):
        sg.predict_gain("X", 0, TELE)
    with pytest.raises(ValueError):
        sg.predict_gain("H", 9, TELE)


def test_missing_telemetry_raises(sg):
    with pytest.raises(ValueError):
        sg.predict_gain("H", 0, {"THERM_FPGA": 30.0})  # missing ADC0 etc.


def test_channel_selects_adc_key(sg):
    # channels 0,1 use SPE_ADC0_T; 2,3 use SPE_ADC1_T. Changing the unused
    # ADC temp must not change the channel-0 gain, but changing ADC0 must.
    base = sg.predict_gain("M", 0, TELE)
    other_adc1 = dict(TELE, SPE_ADC1_T=TELE["SPE_ADC1_T"] + 5.0)
    np.testing.assert_allclose(sg.predict_gain("M", 0, other_adc1), base)
    other_adc0 = dict(TELE, SPE_ADC0_T=TELE["SPE_ADC0_T"] + 5.0)
    assert not np.allclose(sg.predict_gain("M", 0, other_adc0), base)


# -- conversion formulas -------------------------------------------------

def test_counts_to_nv_auto_formula():
    counts = np.array([4.0, 9.0, 0.0, -1.0])
    gain = np.array([1.0, 9.0, 2.0, 1.0])
    out = counts_to_nv_auto(counts, gain)
    # sqrt(4/1)=2, sqrt(9/9)=1, sqrt(0/2)=0, negative power -> NaN
    np.testing.assert_array_equal(out[:3], [2.0, 1.0, 0.0])
    assert np.isnan(out[3])


def test_counts_to_nv_auto_masks_bad_gain():
    out = counts_to_nv_auto(np.array([4.0, 4.0]), np.array([0.0, -1.0]))
    assert np.all(np.isnan(out))


def test_counts_to_nv_cross_sign_preserved():
    counts = np.array([4.0, -4.0, 0.0])
    ga = np.array([1.0, 1.0, 1.0])
    gb = np.array([1.0, 1.0, 1.0])
    out = counts_to_nv_cross(counts, ga, gb)
    # geom gain = 1 -> sign(X)*sqrt(|X|): +2, -2, 0
    np.testing.assert_array_equal(out, [2.0, -2.0, 0.0])


def test_counts_to_nv_cross_geom_gain():
    out = counts_to_nv_cross(np.array([16.0]), np.array([2.0]), np.array([8.0]))
    # g_geom = sqrt(2*8)=4 ; sqrt(16/4)=2
    np.testing.assert_allclose(out, [2.0])


def test_counts_to_nv_auto_broadcasts():
    # scalar gain broadcast over a vector of counts
    out = counts_to_nv_auto([4.0, 9.0, 16.0], 4.0)
    np.testing.assert_allclose(out, [1.0, 1.5, 2.0])
    # (ntime, nfreq) counts with per-freq gain
    counts = np.array([[4.0, 9.0], [16.0, 25.0]])
    gain = np.array([1.0, 1.0])
    np.testing.assert_allclose(counts_to_nv_auto(counts, gain),
                               [[2.0, 3.0], [4.0, 5.0]])


def test_counts_to_nv_cross_broadcasts():
    out = counts_to_nv_cross([4.0, -4.0], 1.0, 1.0)  # scalar gains
    np.testing.assert_array_equal(out, [2.0, -2.0])
    counts = np.array([[16.0, -16.0]])
    np.testing.assert_allclose(counts_to_nv_cross(counts, np.array([2.0, 2.0]),
                                                  np.array([8.0, 8.0])),
                               [[2.0, -2.0]])


def test_counts_to_nv2_helpers_are_linear_and_mask_invalid_auto():
    auto = counts_to_nv2_auto(
        np.array([8.0, -1.0, 4.0]), np.array([2.0, 2.0, 0.0])
    )
    assert auto[0] == 4.0
    assert np.isnan(auto[1]) and np.isnan(auto[2])

    cross = counts_to_nv2_cross(
        np.array([8.0, -8.0]), np.array([2.0, 2.0]), np.array([8.0, 8.0])
    )
    np.testing.assert_array_equal(cross, [2.0, -2.0])


def test_convert_row_matches_per_product_and_reuses_gains(sg):
    nbins = 50
    rng = np.random.default_rng(0)
    spectra = rng.normal(size=(16, nbins)) ** 2  # non-negative-ish
    levels = ["H", "M", "L", "H"]

    row = sg.convert_row(spectra, TELE, levels)
    assert isinstance(row, LabeledArray)
    assert row.units == NV_PER_SQRT_HZ
    assert row.shape == (16, nbins)

    # matches calling convert_product per product
    for p in range(16):
        per = np.asarray(sg.convert_product(p, spectra[p], TELE, levels))
        np.testing.assert_allclose(np.asarray(row)[p], per, equal_nan=True)

    # only 4 channel-gain predictions for the whole row
    calls = {"n": 0}
    orig = sg.predict_gain
    sg.predict_gain = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                       orig(*a, **k))[1]
    try:
        sg.convert_row(spectra, TELE, levels)
    finally:
        sg.predict_gain = orig
    assert calls["n"] == 4


def test_vectorized_batch_chunking_and_psd_match_scalar_rows(sg):
    rng = np.random.default_rng(42)
    nrow = 5
    freqs = np.asarray(ANCHOR_FREQS)
    spectra = rng.normal(size=(nrow, 16, freqs.size))
    spectra[:, :4] = np.abs(spectra[:, :4])
    levels = np.asarray([
        list("HHHH"), list("LMHL"), list("MMLH"),
        list("LLMM"), list("HMLH"),
    ], dtype=object)
    telemetry = {
        key: np.linspace(value, value + 0.2, nrow)
        for key, value in TELE.items()
    }

    scalar = np.stack([
        np.asarray(sg.convert_row(
            spectra[row],
            {key: value[row] for key, value in telemetry.items()},
            levels[row],
            freqs_mhz=freqs,
        ))
        for row in range(nrow)
    ])
    full = sg.convert_batch(
        label(spectra, units="SDU", frame="topo"),
        telemetry,
        levels,
        freqs_mhz=freqs,
    )
    chunked = sg.convert_batch(
        spectra, telemetry, levels, freqs_mhz=freqs, chunk_size=2
    )

    assert full.units == NV_PER_SQRT_HZ and full.frame == "topo"
    np.testing.assert_allclose(np.asarray(full), scalar, rtol=2e-13, atol=2e-13)
    np.testing.assert_array_equal(np.asarray(chunked), np.asarray(full))

    native_psd = sg.convert_batch_psd(
        spectra, telemetry, levels, freqs_mhz=freqs, units=NV2_PER_HZ,
        chunk_size=2,
    )
    si_psd = sg.convert_batch_psd(
        spectra, telemetry, levels, freqs_mhz=freqs, chunk_size=3
    )
    assert native_psd.units == NV2_PER_HZ
    assert si_psd.units == V2_PER_HZ
    np.testing.assert_allclose(
        np.asarray(si_psd), np.asarray(native_psd) * 1e-18,
        rtol=2e-15, atol=0.0, equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(si_psd), np.asarray(asd_to_psd(full)),
        rtol=2e-15, atol=0.0, equal_nan=True,
    )

    with pytest.raises(ValueError, match="chunk_size"):
        sg.convert_batch(spectra, telemetry, levels, freqs_mhz=freqs, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size"):
        sg.convert_batch(spectra, telemetry, levels, freqs_mhz=freqs, chunk_size=True)


def test_bin_frequencies():
    f = bin_frequencies(2048)
    assert f.shape == (2048,) and f[0] == 0.0
    np.testing.assert_allclose(f[1], CHANNEL_BIN_MHZ)


def test_convert_product_auto_returns_labeled(sg):
    nbins = 200
    counts = np.full(nbins, 4.0)
    levels = ["H", "H", "H", "H"]
    out = sg.convert_product(0, counts, TELE, levels)
    assert isinstance(out, LabeledArray)
    assert out.units == NV_PER_SQRT_HZ
    assert out.shape == (nbins,)
    # within the anchor range the conversion is finite
    bf = bin_frequencies(nbins)
    gain = sg.predict_gain("H", 0, TELE, freqs_mhz=bf)
    np.testing.assert_allclose(np.asarray(out), counts_to_nv_auto(counts, gain),
                               equal_nan=True)


def test_convert_product_cross_uses_correct_channels(sg):
    nbins = 100
    counts = np.full(nbins, -4.0)
    levels = {0: "H", 1: "M", 2: "L", 3: "H"}
    # product 8 -> channels (0, 3)
    assert CROSS_PRODUCT_CHANNELS[8] == (0, 3)
    out = sg.convert_product(8, counts, TELE, levels)
    bf = bin_frequencies(nbins)
    ga = sg.predict_gain("H", 0, TELE, freqs_mhz=bf)
    gb = sg.predict_gain("H", 3, TELE, freqs_mhz=bf)
    np.testing.assert_allclose(np.asarray(out), counts_to_nv_cross(counts, ga, gb),
                               equal_nan=True)
    assert out.units == NV_PER_SQRT_HZ


def test_asd_to_psd_scale_and_units():
    # 2 nV/sqrt(Hz) -> (2e-9)^2 = 4e-18 V^2/Hz
    out = asd_to_psd(np.array([2.0]))
    assert isinstance(out, LabeledArray) and out.units == V2_PER_HZ
    np.testing.assert_allclose(np.asarray(out), [4e-18])


def test_psd_to_asd_scale_and_units():
    # 4e-18 V^2/Hz -> sqrt(4e-18)/1e-9 = 2 nV/sqrt(Hz)
    out = psd_to_asd(np.array([4e-18]))
    assert isinstance(out, LabeledArray) and out.units == NV_PER_SQRT_HZ
    np.testing.assert_allclose(np.asarray(out), [2.0])


def test_asd_psd_round_trip_with_sign():
    asd = np.array([-3.0, 0.0, 5.0, 12.5])
    back = psd_to_asd(asd_to_psd(asd))
    assert back.units == NV_PER_SQRT_HZ
    np.testing.assert_allclose(np.asarray(back), asd)
    # and the other direction
    psd = np.array([-4e-18, 0.0, 9e-18])
    back2 = asd_to_psd(psd_to_asd(psd))
    np.testing.assert_allclose(np.asarray(back2), psd)


def test_asd_psd_preserve_frame_and_accept_labeled():
    la = label(np.array([2.0]), units=NV_PER_SQRT_HZ, frame="topo")
    out = asd_to_psd(la)
    assert out.frame == "topo" and out.units == V2_PER_HZ
    np.testing.assert_allclose(np.asarray(out), [4e-18])
    # bare input -> frame None
    assert asd_to_psd(np.array([1.0])).frame is None


def test_asd_psd_reject_complex():
    with pytest.raises(ValueError):
        asd_to_psd(np.array([1.0 + 2.0j]))
    with pytest.raises(ValueError):
        psd_to_asd(np.array([1.0 + 0.0j]))


def test_exported_from_package():
    assert lusee.SpectrometerGain is SpectrometerGain
    assert lusee.NV_PER_SQRT_HZ == "nV/sqrt(Hz)"
    assert lusee.V2_PER_HZ == "V^2/Hz"
    assert lusee.NV2_PER_HZ == "nV^2/Hz"
    assert lusee.counts_to_nv2_auto is counts_to_nv2_auto
    assert lusee.counts_to_nv2_cross is counts_to_nv2_cross
    assert lusee.asd_to_psd is asd_to_psd and lusee.psd_to_asd is psd_to_asd
