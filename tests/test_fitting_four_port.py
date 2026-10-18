"""The fitting framework driven by the coupled four-port response.

``lusee.mapmaker.build_instrument`` dispatches a response-v3 file to
``FullStokesCroSimulator``, so ``lusee.Fitting.Experiment`` has to work against
a simulator whose output is 16 packed covariance channels in V^2/Hz rather than
per-antenna temperatures.  These tests use a synthetic response so they need no
Drive data.
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"  # must precede the first jax import

import numpy as np
import jax
import jax.numpy as jnp
import healpy as hp
import pytest

import lusee
from lusee.Fitting import BeamModule, Experiment, InstrumentModule, linear_fisher
from lusee.ReceiverImpedance import IdealCapacitorReceiver
from lusee.SpectralSky import SpectralHealpixSky, SpectralSkyModule
from lusee.SeparableSky import SeparableHealpixSky
from lusee.SyntheticResponse import synthetic_four_port_response


FREQ = np.array([10.0, 15.0, 20.0], dtype=float)
LMAX = 3
NSIDE = 2


def _four_port_simulator():
    response = synthetic_four_port_response(freq_mhz=(10.0, 15.0, 20.0))
    obs = lusee.Observation(
        "2025-02-01 00:00:00 to 2025-02-01 12:00:00",
        deltaT_sec=4 * 3600.0,
        lun_lat_deg=-10.0,
        lun_long_deg=180.0,
    )
    sky_dummy = lusee.HealpixSky(
        NSIDE, LMAX, maps=[np.ones(12 * NSIDE**2) for _ in FREQ],
        freq=FREQ, frame="galactic",
    )
    sim = lusee.CroSimulator(
        obs, response, sky_dummy, IdealCapacitorReceiver(),
        T_moon=0.0, T_ant=0.0, products="all", freq=FREQ, lmax=LMAX,
    )
    return sim, obs


def _truth_sky(rng):
    flux_map = 1.0e4 * (1.0 + 0.3 * rng.standard_normal(12 * NSIDE**2))
    flux_alm = SpectralHealpixSky.flux_alm_from_map(flux_map, LMAX, NSIDE)
    beta = np.full(12 * NSIDE**2, -2.4)
    sky = SpectralHealpixSky(flux_alm, beta, Nside=NSIDE, lmax=LMAX, freq=FREQ,
                             f_fid=15.0, beta_nside=NSIDE)
    return sky, np.asarray(flux_alm), beta


def test_build_instrument_dispatches_response_v3_to_full_stokes():
    sim, _ = _four_port_simulator()
    assert type(sim).__name__ == "FullStokesCroSimulator"
    # channel metadata must be available before simulate() has ever run
    assert lusee.mapmaker.channel_names(sim) == (
        "00R", "01R", "01I", "02R", "02I", "03R", "03I",
        "11R", "12R", "12I", "13R", "13I", "22R", "23R", "23I", "33R",
    )
    assert lusee.mapmaker.channel_metadata(sim) == {
        "products": lusee.mapmaker.channel_names(sim)
    }


def test_channel_metadata_falls_back_to_legacy_combinations():
    class LegacySim:
        combinations = [(0, 0), (1, 1), (0, 1)]

    assert lusee.mapmaker.channel_metadata(LegacySim()) == {
        "combinations": ((0, 0), (1, 1), (0, 1))
    }
    assert lusee.mapmaker.channel_names(LegacySim()) == (
        "00R", "11R", "01R", "01I",
    )


def test_spectral_sky_freq_stays_host_side_inside_jit():
    """Regression: ``jnp.asarray`` on freq made it a tracer inside a trace,
    which the four-port sky dispatch (``np.asarray(sky.freq)``) cannot read."""
    nalm = (LMAX + 1) * (LMAX + 2) // 2
    seen = {}

    def build(alm):
        sky = SpectralHealpixSky(alm, jnp.full(12 * NSIDE**2, -2.5),
                                 Nside=NSIDE, lmax=LMAX, freq=FREQ,
                                 f_fid=15.0, beta_nside=NSIDE)
        seen["freq"] = sky.freq
        # must be readable by the simulators' host-side frequency dispatch
        seen["as_numpy"] = np.asarray(sky.freq)
        return jnp.sum(jnp.abs(sky.get_alm_at_freq(FREQ)))

    jax.jit(build)(jnp.zeros(nalm, dtype=complex))
    assert not isinstance(seen["freq"], jax.core.Tracer)
    np.testing.assert_allclose(seen["as_numpy"], FREQ)


def test_get_alm_at_freq_matches_get_alm_on_own_grid():
    rng = np.random.default_rng(0)
    sky, _, _ = _truth_sky(rng)
    np.testing.assert_allclose(
        np.asarray(sky.get_alm_at_freq(FREQ)),
        np.asarray(sky.get_alm(np.arange(len(FREQ)))),
        rtol=1e-12, atol=1e-12,
    )


def test_spectral_sky_get_alm_at_freq_is_closed_form_off_grid():
    """The power-law model is exact at any frequency, not just its own grid."""
    rng = np.random.default_rng(1)
    sky, _, _ = _truth_sky(rng)
    off = np.array([12.5])
    got = np.asarray(sky.get_alm_at_freq(off))[0]
    on = np.asarray(sky.get_alm_at_freq(np.array([15.0])))[0]
    # flux(f) = flux(f_fid) * (f/f_fid)**beta with a constant beta map
    np.testing.assert_allclose(got, on * (12.5 / 15.0) ** -2.4,
                               rtol=1e-10, atol=1e-10)


def test_separable_sky_rejects_foreign_frequency_grid():
    nalm = (LMAX + 1) * (LMAX + 2) // 2
    sky = SeparableHealpixSky(np.zeros((2, nalm), dtype=complex),
                              np.ones((2, len(FREQ))), lmax=LMAX, freq=FREQ)
    np.testing.assert_allclose(
        np.asarray(sky.get_alm_at_freq(FREQ)),
        np.asarray(sky.get_alm(np.arange(len(FREQ)))),
    )
    with pytest.raises(ValueError, match="its own grid"):
        sky.get_alm_at_freq(np.array([11.0, 16.0, 21.0]))


@pytest.mark.slow
def test_four_port_fixed_beta_flux_recovery():
    """End-to-end: Wiener solve of the linear flux block through the four-port
    forward model recovers the input sky, with calibrated Fisher errors."""
    sim, obs = _four_port_simulator()
    rng = np.random.default_rng(7)
    sky, flux_true, beta_true = _truth_sky(rng)

    data_clean = sim.simulate(sky=sky)
    assert np.asarray(data_clean).shape == (len(obs.times), 16, len(FREQ))

    sigma = lusee.mapmaker.compute_radiometric_noise(
        data_clean, delta_f_hz=1e6, delta_t_sec=4 * 3600.0,
        **lusee.mapmaker.channel_metadata(sim))
    assert np.all(np.asarray(sigma) > 0)
    # keep the SNR finite so the prior is not entirely irrelevant
    snr0 = float(jnp.std(data_clean)) / float(jnp.median(sigma))
    sigma = sigma * (snr0 / 3e3)
    data = data_clean + sigma * jax.random.normal(
        jax.random.PRNGKey(3), data_clean.shape)
    N_inv = 1.0 / jnp.asarray(sigma) ** 2

    exp = Experiment(
        sim,
        sky=SpectralSkyModule(lmax=LMAX, Nside=NSIDE, freq=FREQ, f_fid=15.0,
                              beta_nside=NSIDE,
                              cl_flux=hp.alm2cl(flux_true),
                              beta_fixed=beta_true),
        beam=BeamModule(), instrument=InstrumentModule(),
        data=data, N_inv=N_inv)
    assert exp.paramset.nonlinear == []

    lin = exp.linear_solve(method="dense")
    flux_hat = np.asarray(lin["sky.flux"])

    fisher = linear_fisher(exp.predict, exp.paramset, data, N_inv, {},
                           method="dense")
    block = exp.paramset.linear[0].reparam
    from lusee.Fitting import snr_weighted_recovery
    rec = snr_weighted_recovery(block.natural_to_theta(flux_hat),
                                block.natural_to_theta(flux_true), fisher)
    assert rec["rho_w"] > 0.99
    assert rec["resid_frac"] < 0.05


@pytest.mark.slow
def test_four_port_varpro_improves_on_a_wrong_beta():
    """The outer (non-linear) VarPro loop drives beta toward the truth."""
    sim, _ = _four_port_simulator()
    rng = np.random.default_rng(11)
    sky, flux_true, beta_true = _truth_sky(rng)

    data = sim.simulate(sky=sky)
    sigma = lusee.mapmaker.compute_radiometric_noise(
        data, delta_f_hz=1e6, delta_t_sec=4 * 3600.0,
        **lusee.mapmaker.channel_metadata(sim))
    N_inv = 1.0 / jnp.asarray(sigma) ** 2

    exp = Experiment(
        sim,
        sky=SpectralSkyModule(lmax=LMAX, Nside=NSIDE, freq=FREQ, f_fid=15.0,
                              beta_nside=NSIDE, cl_flux=hp.alm2cl(flux_true),
                              beta_init=-3.0),
        beam=BeamModule(), instrument=InstrumentModule(),
        data=data, N_inv=N_inv)

    out = exp.optimize(maxiter=200, inner_method="dense", verbose=False)
    beta_hat = np.asarray(out["nonlinear"]["sky.beta"])
    # Three frequencies over three timesteps constrain the *mean* spectral
    # index well and individual coarse pixels only weakly, so test the mode
    # the data actually measure.
    start_err = abs(-3.0 - beta_true.mean())
    end_err = abs(beta_hat.mean() - beta_true.mean())
    assert end_err < 0.25 * start_err
    assert out["chi2"] < 1e-4 * out["chi2_history"][0]
