import pytest
import numpy as np
import os
import lusee

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def observation():
    return lusee.Observation('2025-02-01 13:00:00 to 2025-03-01 13:00:00', deltaT_sec=24 * 3600, lun_lat_deg=0.0)


@pytest.fixture(scope="module")
def fits_beam(drive_dir):
    return lusee.Beam(
        os.path.join(drive_dir, 'Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith/hfss_lbl_3m_75deg.fits')
    )


def test_sim_fits_beams(observation, fits_beam):
    beams = []
    for ofs, c in enumerate(["N", "E", "S", "W"]):
        cB = fits_beam.rotate(-90 * ofs)
        cB.id = c
        beams.append(cB)

    lmax = 64
    freq = [10, 12, 30]
    sky = lusee.sky.ConstSky(Nside=32, lmax=lmax, freq=freq, T=200)
    S = lusee.DefaultSimulator(
        observation, beams, sky, freq=freq, lmax=lmax,
        combinations=[(0, 0), (1, 1), (1, 3)], Tground=200.0,
    )
    WF = S.simulate(times=observation.times)
    assert np.allclose(WF[:, :2, :], 200)


def test_sim_gauss_beam(observation):
    BG = lusee.BeamGauss(dec_deg=50, sigma_deg=6, phi_deg=0)
    beams = [BG]

    lmax = 64
    freq = [1, 5, 10]
    sky = lusee.sky.ConstSky(Nside=32, lmax=lmax, freq=freq, T=200)
    S = lusee.DefaultSimulator(
        observation, beams, sky, freq=freq, lmax=lmax,
        combinations=[(0, 0)], Tground=0.0,
    )
    WF = S.simulate(times=observation.times)
    assert np.allclose(WF[:, :2, :], 200, rtol=5e-4)
