"""Tests for staged RRL analysis pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from lusee.RRLAnalysis import resample_waterfall_frequency
from lusee.RRLSkyModels import (
    COLD_GAS_VYDULA2024,
    HOT_GAS_VYDULA2024,
    carbon_rrl_alpha_transitions_in_frequency_band_mhz,
    carbon_rrl_frequency_mhz,
    rrl_envelope_T_rrl_k_mhz,
    rrl_smooth_envelope_weight_mhz,
    rydberg_line_spectrum_mhz,
    hydrogen_rrl_frequency_mhz,
)


def test_rrl_vydula2024_cold_envelope_absorption_at_12mhz():
    nu = np.array([12.0])
    t = rrl_envelope_T_rrl_k_mhz(nu, COLD_GAS_VYDULA2024)
    assert t.shape == (1,)
    assert float(t[0]) < 0.0


def test_rrl_vydula2024_hot_envelope_emission_at_12mhz():
    nu = np.array([12.0])
    t = rrl_envelope_T_rrl_k_mhz(nu, HOT_GAS_VYDULA2024)
    assert float(t[0]) > 0.0


def test_rrl_smooth_envelope_gaussian_legacy():
    nu = np.linspace(10.0, 15.0, 11)
    w = rrl_smooth_envelope_weight_mhz(
        nu, nu_ref_mhz=12.5, sigma_mhz=2.0, amplitude_k=0.5, gas_case="gaussian"
    )
    assert w.shape == nu.shape
    assert np.all(w >= 0.0)
    assert np.all(np.isfinite(w))


def test_carbon_rrl_band_has_many_alpha_lines_10_15_mhz():
    tr = carbon_rrl_alpha_transitions_in_frequency_band_mhz(10.0, 15.0)
    assert len(tr) > 50
    for n1, n2 in (tr[0], tr[-1]):
        assert n2 == n1 - 1
        f = carbon_rrl_frequency_mhz(n1, n2)
        assert 10.0 <= f <= 15.0


def test_rydberg_line_spectrum_carbon_peaks_near_rest_freq():
    nu = np.linspace(12.45, 12.55, 401)
    tr = carbon_rrl_alpha_transitions_in_frequency_band_mhz(12.45, 12.55)
    assert len(tr) >= 1
    n1, n2 = tr[len(tr) // 2]
    nu0 = carbon_rrl_frequency_mhz(n1, n2)
    spec = rydberg_line_spectrum_mhz(
        nu, ((n1, n2),), species="carbon", sigma_mhz=0.01, peak_k=1.0
    )
    assert float(spec[np.argmin(np.abs(nu - nu0))]) == pytest.approx(1.0, rel=1e-3)


def test_rydberg_line_spectrum_auto_fills_carbon_band():
    nu = np.linspace(10.0, 15.0, 5001)
    spec = rydberg_line_spectrum_mhz(nu, None, sigma_mhz=0.05, peak_k=0.5)
    assert spec.shape == nu.shape
    assert np.any(spec > 0.0)
    n_lines = len(carbon_rrl_alpha_transitions_in_frequency_band_mhz(10.0, 15.0))
    assert n_lines > 0


def test_resample_waterfall_frequency_shape():
    fin = np.array([10.0, 11.0, 12.0])
    fout = np.linspace(10.0, 12.0, 5)
    w = np.ones((2, 1, 3))
    w[:, 0, 1] = 2.0
    out = resample_waterfall_frequency(w, fin, fout)
    assert out.shape == (2, 1, 5)
    assert np.allclose(out[:, 0, 0], 1.0)
    assert np.allclose(out[:, 0, -1], 2.0)
