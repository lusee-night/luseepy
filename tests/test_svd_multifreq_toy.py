"""Toy test of multifrequency SVD-subspace Wiener solve.

Stub out the CroSimulator with a fixed random linear operator per frequency.
Inject a ground-truth sky that lives exactly in the span of K frequency
templates and check the solver recovers the beta coefficients.
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import healpy as hp
import pytest


class _FakeSim:
    """Minimal sim.simulate(sky=...) returning a linear function of sky.mapalm."""
    def __init__(self, nfreq, nalm, ndata_per_freq, seed=0):
        rng = np.random.default_rng(seed)
        # Real linear map from (Re, Im) alm parts to real data.
        self.Jr = jnp.asarray(rng.standard_normal((nfreq, ndata_per_freq, nalm)))
        self.Ji = jnp.asarray(rng.standard_normal((nfreq, ndata_per_freq, nalm)))
        self.ndata_per_freq = ndata_per_freq
        self.nfreq = nfreq

    def simulate(self, sky):
        alm = sky.mapalm  # (nfreq, nalm) complex
        re = jnp.real(alm)
        im = jnp.imag(alm)
        # data[f, d] = sum_a Jr[f, d, a]*re[f, a] + Ji[f, d, a]*im[f, a]
        out = jnp.einsum("fda,fa->fd", self.Jr, re) + jnp.einsum("fda,fa->fd", self.Ji, im)
        return out[:, None, :]  # shape (nfreq, 1, ndata_per_freq) -> we ravel later


def test_svd_multifreq_recovers_beta():
    from lusee.SkyModels import HealpixSky
    from lusee import MapMaker as mm

    lmax = 4
    nalm = hp.Alm.getsize(lmax)
    Nside = 8
    nfreq = 16
    K = 3
    npix = 12 * Nside**2

    # Random freq templates (orthonormal)
    rng = np.random.default_rng(7)
    F_raw = rng.standard_normal((nfreq, K))
    F, _ = np.linalg.qr(F_raw)          # (nfreq, K)

    # Ground-truth beta (K, nalm) complex, real a_{l,0}
    beta_true = jnp.asarray(rng.standard_normal((K, nalm)) +
                            1j * rng.standard_normal((K, nalm)))
    # zero imag parts at m=0
    m0 = np.array([hp.Alm.getidx(lmax, l, 0) for l in range(lmax + 1)])
    beta_true = beta_true.at[:, m0].set(jnp.real(beta_true[:, m0]))

    # Assemble true multi-freq alm
    alm_true = jnp.asarray(F) @ beta_true   # (nfreq, nalm)

    # Build a sky template with correct shape
    sky_tmpl = HealpixSky(
        Nside, lmax,
        maps=[np.zeros(npix) for _ in range(nfreq)],
        freq=np.arange(nfreq) + 1.0, frame="galactic",
    )
    sky_tmpl.mapalm = alm_true  # irrelevant for template, but keeps shape

    # Fake forward model
    ndata_per_freq = 40
    sim = _FakeSim(nfreq, nalm, ndata_per_freq, seed=3)

    # Generate data from ground truth + small noise
    sky_truth = HealpixSky(
        Nside, lmax,
        maps=[np.zeros(npix) for _ in range(nfreq)],
        freq=np.arange(nfreq) + 1.0, frame="galactic",
    )
    sky_truth.mapalm = alm_true
    data_clean = sim.simulate(sky=sky_truth)
    noise_sigma = 0.01
    data = data_clean + noise_sigma * jnp.asarray(
        rng.standard_normal(data_clean.shape))

    # Solve in SVD subspace
    signal_prior = 1e-4 * jnp.ones((K, nalm))  # loose Tikhonov
    beta_hat = mm.solve_svd_multifreq(
        sim, data, sky_tmpl, noise_sigma, F,
        signal_prior=signal_prior, lmax=lmax,
        maxiter=2000, tol=1e-10,
    )

    # With linear forward, loose prior, low noise, beta_hat ≈ beta_true
    err = jnp.linalg.norm(beta_hat - beta_true) / jnp.linalg.norm(beta_true)
    assert float(err) < 0.05, f"beta relerr {float(err):.3e}"


if __name__ == "__main__":
    test_svd_multifreq_recovers_beta()
    print("OK")
