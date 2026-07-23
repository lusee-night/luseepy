"""End-to-end tests for four-port full-Stokes covariance simulators."""

from astropy.time import Time
import fitsio
import jax
import jax.numpy as jnp
import numpy as np

import croissant as cro
import lusee
from beam_conversion.common import (
    compute_sky_moon_resistance,
)
from lusee.Data import Data
from lusee.FullStokesSimulator import (
    FullStokesCroSimulator,
    FullStokesTopoJaxSimulator,
)
from lusee.FullStokesCalibrator import FullStokesCalibratorSimulator
from lusee.GainModel import V2_PER_HZ
from lusee.InstrumentResponse import InstrumentResponse
from lusee.LabeledArray import FRAME_TOPO, LabeledArray
from lusee.ReceiverImpedance import IdealCapacitorReceiver
from lusee.ReceiverImpedance import MeasuredReceiver


class SyntheticObservation:
    """Minimal observation metadata for an already-topocentric sky fixture."""

    def __init__(self, times):
        self.times = times
        self.time_range = "2028-01-01 00:00:00 to 2028-01-01 00:06:00"
        self.lun_lat_deg = -23.814
        self.lun_long_deg = 182.258
        self.lun_height_m = 0.0
        self.deltaT_sec = 60.0
        self.loc = None


def make_in_memory_response():
    freq = np.asarray([10.0, 20.0], dtype=np.float64)
    theta = np.arange(0.0, 91.0, 45.0)
    phi = np.arange(0.0, 361.0, 45.0)
    tt, pp = np.meshgrid(np.radians(theta), np.radians(phi), indexing="ij")
    Htheta = np.zeros((4, 2, 3, 9), dtype=np.complex128)
    Hphi = np.zeros_like(Htheta)
    for port in range(4):
        Htheta[port] = (
            0.02
            * (port + 1)
            * (1.0 + 0.1 * np.arange(2))[:, None, None]
            * np.cos(tt)[None]
            * np.exp(1j * port * pp)[None]
        )
        Hphi[port] = (
            0.01
            * (1.0 + 0.05 * np.arange(2))[:, None, None]
            * np.sin(tt)[None]
            * np.exp(-1j * (port + 1) * pp)[None]
        )
    ZA = np.broadcast_to(
        (30.0 + 4.0j) * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    Rsky, Rmoon = compute_sky_moon_resistance(
        freq, theta, phi, Htheta, Hphi, ZA
    )
    return InstrumentResponse.from_arrays(
        freq,
        theta,
        phi,
        Htheta,
        Hphi,
        ZA,
        Rsky,
        Rmoon,
        metadata={
            "SOURCE": "analytic",
            "SOURCE_ROOT": "pytest",
            "INPUT_KIND": "bare",
            "FIELD_KIND": "effective-length",
            "AMP_CONV": "RMS",
            "TIMECONV": "e+jwt",
            "ZA_SOURCE": "analytic",
            "GIT_SHA": "test",
            "COORDSYS": "instrument-topocentric",
            "THETADEF": "colatitude-from-+z",
            "PHIDEF": "right-handed-about-+z",
            "OMEGADEF": "source-arrival-direction",
            "POLBASIS": "e_theta,e_phi",
            "PHASEREF": "analytic-origin",
            "PORTS": "0123",
            "VALIDATED": True,
        },
    )


def make_topocentric_blackbody_sky(temperature):
    theta_count = 5
    phi_count = 8
    data = np.zeros((2, 4, theta_count, phi_count), dtype=np.float64)
    data[:, 0] = temperature
    return cro.PolarizedSky(
        data,
        [10.0, 20.0],
        sampling="mwss",
        coord="mcmf",
        frame="topo",
        convention="IAU",
    )


def make_galactic_anisotropic_sky():
    theta = np.linspace(0.0, np.pi, 5)
    phi = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    data = np.zeros((2, 4, 5, 8), dtype=np.float64)
    data[:, 0] = (
        180.0
        + 20.0 * np.cos(tt)[None]
        + 8.0 * np.sin(tt)[None] * np.cos(pp)[None]
    )
    data[:, 3] = 3.0 * np.cos(tt)[None]
    return cro.PolarizedSky(
        data,
        [10.0, 20.0],
        sampling="mwss",
        coord="galactic",
        frame="galactic",
        convention="IAU",
    )


def test_cro_simulator_blackbody_off_grid_timestamps_and_boundaries(monkeypatch):
    times = Time(
        [
            "2028-01-01T00:00:00",
            "2028-01-01T00:01:00",
            "2028-01-01T00:05:00",
        ],
        scale="utc",
    )
    monkeypatch.setattr(
        "lusee.SimulatorBase.get_topo_z_rotation_angles",
        lambda obs, supplied: np.zeros(len(supplied)),
    )
    temperature = 237.0
    simulator = FullStokesCroSimulator(
        SyntheticObservation(times),
        make_in_memory_response(),
        make_topocentric_blackbody_sky(temperature),
        IdealCapacitorReceiver(30.0),
        T_moon=temperature,
        freq=[17.5, 10.0, 17.5],
        lmax=2,
    )
    result = simulator.simulate(times)
    assert isinstance(result, jax.Array)
    assert result.shape == (3, 16, 3)
    assert simulator.result_times is not times
    assert np.array_equal(simulator.result_times.utc.mjd, times.utc.mjd)
    np.testing.assert_allclose(
        simulator.elapsed_tdb_seconds,
        [0.0, 60.0, 300.0],
        atol=1e-7,
    )
    assert jnp.array_equal(result[..., 0], result[..., 2])
    expected = temperature * simulator.blackbody_normalization
    assert jnp.allclose(
        simulator.covariance,
        expected[None],
        rtol=2e-5,
        atol=1e-25,
    )
    labeled = simulator.result_labeled
    assert isinstance(labeled, LabeledArray)
    assert labeled.units == V2_PER_HZ
    assert labeled.frame == FRAME_TOPO


def test_topo_and_cro_independent_contractions_agree_for_topo_sky():
    times = Time(
        ["2028-01-01T00:00:00", "2028-01-01T00:03:00"],
        scale="utc",
    )
    obs = SyntheticObservation(times)
    beam = make_in_memory_response()
    sky = make_topocentric_blackbody_sky(180.0)
    receiver = IdealCapacitorReceiver()
    kwargs = {
        "T_moon": 210.0,
        "freq": [12.5, 20.0],
        "lmax": 2,
    }
    cro_result = FullStokesCroSimulator(
        obs, beam, sky, receiver, **kwargs
    ).simulate(times)
    topo_result = FullStokesTopoJaxSimulator(
        obs, beam, sky, receiver, **kwargs
    ).simulate(times)
    assert jnp.allclose(cro_result, topo_result, rtol=2e-5, atol=1e-25)


def test_topo_and_cro_agree_for_celestial_sky_away_from_lunar_pole():
    observation = lusee.Observation(
        "2028-01-01 00:00:00 to 2028-01-01 06:00:00",
        lun_lat_deg=-23.814,
        lun_long_deg=182.258,
        deltaT_sec=3 * 3600.0,
    )
    times = Time(
        [
            "2028-01-01T00:00:00",
            "2028-01-01T03:00:00",
            "2028-01-01T06:00:00",
        ],
        scale="utc",
    )
    beam = make_in_memory_response()
    sky = make_galactic_anisotropic_sky()
    receiver = IdealCapacitorReceiver()
    kwargs = {
        "T_moon": 210.0,
        "freq": [12.5],
        "lmax": 2,
    }
    cro_result = FullStokesCroSimulator(
        observation, beam, sky, receiver, **kwargs
    ).simulate(times)
    topo_result = FullStokesTopoJaxSimulator(
        observation, beam, sky, receiver, **kwargs
    ).simulate(times)
    np.testing.assert_allclose(
        cro_result,
        topo_result,
        rtol=2e-4,
        atol=1e-24,
    )


def test_direct_constructor_defaults_to_common_frequency_interval():
    times = Time(["2028-01-01T00:00:00"], scale="utc")
    impedance = np.broadcast_to(
        40.0 * np.eye(4)[None],
        (2, 4, 4),
    ).astype(np.complex128)
    simulator = FullStokesCroSimulator(
        SyntheticObservation(times),
        make_in_memory_response(),
        make_topocentric_blackbody_sky(180.0),
        MeasuredReceiver([15.0, 25.0], impedance),
        lmax=2,
    )
    assert np.array_equal(simulator.freq, [20.0])
    assert np.array_equal(
        simulator.default_frequency_removals["receiver"],
        [10.0],
    )


def test_public_beam_and_simulator_facades_dispatch_v3(tmp_path):
    from beam_conversion.common import ResponseArrays, write_response_fits

    response = make_in_memory_response()
    payload = ResponseArrays(
        response.freq,
        response.theta_deg,
        response.phi_deg,
        np.asarray(response.H_theta),
        np.asarray(response.H_phi),
        np.asarray(response.ZA),
        np.asarray(response.Rsky_native),
        np.asarray(response.Rmoon_native),
        metadata=response.header,
    )
    filename = tmp_path / "public_response.fits"
    write_response_fits(filename, payload)
    beam = lusee.Beam(filename)
    assert isinstance(beam, InstrumentResponse)
    times = Time(["2028-01-01T00:00:00"], scale="utc")
    simulator = lusee.CroSimulator(
        SyntheticObservation(times),
        beam,
        make_topocentric_blackbody_sky(100.0),
        IdealCapacitorReceiver(),
        T_moon=100.0,
        freq=[12.5],
        lmax=2,
    )
    assert isinstance(simulator, FullStokesCroSimulator)


def test_mapmaker_builds_response_v3_instrument(tmp_path):
    from beam_conversion.common import ResponseArrays, write_response_fits

    response = make_in_memory_response()
    filename = tmp_path / "mapmaker_response.fits"
    write_response_fits(
        filename,
        ResponseArrays(
            response.freq,
            response.theta_deg,
            response.phi_deg,
            np.asarray(response.H_theta),
            np.asarray(response.H_phi),
            np.asarray(response.ZA),
            np.asarray(response.Rsky_native),
            np.asarray(response.Rmoon_native),
            metadata=response.header,
        ),
    )
    simulator, loaded, observation = lusee.mapmaker.build_instrument(
        filename,
        "2028-01-01 00:00:00 to 2028-01-01 00:01:00",
        np.asarray([17.5, 10.0, 17.5]),
        2,
        dt_sec=60.0,
        receiver=IdealCapacitorReceiver(),
    )
    assert isinstance(simulator, FullStokesCroSimulator)
    assert isinstance(loaded, InstrumentResponse)
    assert np.array_equal(simulator.freq, [17.5, 10.0, 17.5])
    assert simulator.obs is observation
    target = np.asarray([17.5, 10.0, 17.5])
    sky_template = lusee.sky.HealpixSky(
        8,
        2,
        maps=[np.ones(12 * 64) for _ in target],
        freq=target,
        frame="galactic",
    )
    data = simulator.simulate()
    solution = lusee.mapmaker.solve(
        simulator,
        data,
        sky_template,
        sigma=1.0,
        maxiter=1,
        tol=1e-5,
        precondition=False,
    )
    assert solution.shape == sky_template.mapalm.shape
    assert jnp.all(jnp.isfinite(solution))


def test_covariance_fits_data_round_trip_preserves_exact_time_and_units(
    tmp_path,
    monkeypatch,
):
    times = Time(
        [
            "2028-01-01T00:00:00",
            "2028-01-01T00:01:30",
            "2028-01-01T00:05:00",
        ],
        scale="utc",
    )
    monkeypatch.setattr(
        "lusee.SimulatorBase.get_topo_z_rotation_angles",
        lambda obs, supplied: np.zeros(len(supplied)),
    )
    temperature = 225.0
    simulator = FullStokesCroSimulator(
        SyntheticObservation(times),
        make_in_memory_response(),
        make_topocentric_blackbody_sky(temperature),
        IdealCapacitorReceiver(),
        T_moon=temperature,
        freq=[15.0, 10.0, 15.0],
        lmax=2,
    )
    simulator.simulate(times)
    filename = tmp_path / "covariance.fits"
    simulator.write_fits(filename)
    with fitsio.FITS(filename) as fits:
        header = fits["data"].read_header()
    assert header["RECMODEL"] == "IdealCapacitorReceiver"
    assert header["RECCHANS"] == "0,1,2,3"
    assert header["SKYMODEL"] == "PolarizedSky"
    assert header["SKYFRAME"] == "topo"
    assert header["LUSEEVER"] != ""
    assert header["CROVER"] != ""
    assert header["S2FFTVER"] != ""
    data = Data(filename)
    assert data.response_provenance["content_hash"] == (
        simulator.result_beam.content_hash
    )
    assert data.receiver_provenance["model"] == "IdealCapacitorReceiver"
    assert data.receiver_params["C_pf"] == [30.0, 30.0, 30.0, 30.0]
    assert data.sky_provenance["frame"] == "topo"
    assert data.software_versions["croissant"]
    assert np.array_equal(data.times.utc.mjd, times.utc.mjd)
    assert np.array_equal(data.freq, [15.0, 10.0, 15.0])
    raw = data[:, "00R", :]
    assert raw.units == V2_PER_HZ
    assert raw.frame == FRAME_TOPO
    np.testing.assert_allclose(
        raw.array,
        np.asarray(simulator.covariance[:, :, 0, 0]).real,
        rtol=2e-5,
    )
    equivalent = data[:, "00RK", :]
    assert equivalent.units == "K"
    np.testing.assert_allclose(
        equivalent.array,
        temperature,
        rtol=2e-5,
    )
    no_op = data[:, "00RV", :]
    assert no_op.units == V2_PER_HZ
    np.testing.assert_array_equal(no_op.array, raw.array)


def test_equivalent_utc_and_tdb_instants_match_and_preserve_scale(
    tmp_path,
    monkeypatch,
):
    utc = Time(
        ["2028-01-01T00:00:00", "2028-01-01T00:02:30"],
        scale="utc",
    )
    tdb = utc.tdb
    monkeypatch.setattr(
        "lusee.SimulatorBase.get_topo_z_rotation_angles",
        lambda obs, supplied: np.zeros(len(supplied)),
    )
    kwargs = {
        "beam": make_in_memory_response(),
        "sky_model": make_topocentric_blackbody_sky(190.0),
        "receiver": IdealCapacitorReceiver(),
        "T_moon": 210.0,
        "freq": [12.5],
        "lmax": 2,
    }
    utc_simulator = FullStokesCroSimulator(
        SyntheticObservation(utc),
        **kwargs,
    )
    tdb_simulator = FullStokesCroSimulator(
        SyntheticObservation(tdb),
        **kwargs,
    )
    utc_result = utc_simulator.simulate(utc)
    tdb_result = tdb_simulator.simulate(tdb)
    assert jnp.allclose(utc_result, tdb_result, rtol=1e-12, atol=1e-25)
    assert utc_simulator.result_epoch_tdb_jd == tdb_simulator.result_epoch_tdb_jd

    filename = tmp_path / "tdb_covariance.fits"
    tdb_simulator.write_fits(filename)
    stored = Data(filename)
    assert stored.times.scale == "tdb"
    np.testing.assert_array_equal(stored.times.mjd, tdb.mjd)


def test_calibrator_pair_kernel_matches_direct_loaded_response_for_iquv():
    beam = make_in_memory_response()
    receiver = IdealCapacitorReceiver()
    theta = np.radians(30.0)
    phi = np.radians(22.5)
    fixtures = (
        np.asarray([1.0, 0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.7, 0.0, 0.0]),
        np.asarray([1.0, 0.0, -0.6, 0.0]),
        np.asarray([1.0, 0.0, 0.0, 0.8]),
        np.asarray([2.0, 0.3, -0.2, 0.4]),
    )
    for stokes in fixtures:
        simulator = FullStokesCalibratorSimulator(
            beam,
            receiver,
            freq=[10.0],
        )
        simulator.simulate(theta, phi, stokes)
        direct = simulator.direct_native_covariance(
            0,
            theta,
            phi,
            stokes,
        )
        assert jnp.allclose(
            simulator.covariance[0, 0],
            direct,
            rtol=2e-11,
            atol=1e-25,
        )


def test_gradients_flow_through_off_grid_sky_and_response_overrides(
    monkeypatch,
):
    times = Time(["2028-01-01T00:00:00"], scale="utc")
    monkeypatch.setattr(
        "lusee.SimulatorBase.get_topo_z_rotation_angles",
        lambda obs, supplied: np.zeros(len(supplied)),
    )
    base_response = make_in_memory_response()
    base_sky = make_topocentric_blackbody_sky(100.0)
    simulator = FullStokesCroSimulator(
        SyntheticObservation(times),
        base_response,
        base_sky,
        IdealCapacitorReceiver(),
        T_moon=0.0,
        freq=[17.5],
        lmax=2,
    )

    def sky_loss(scale):
        sky = cro.PolarizedSky(
            base_sky.data.at[:, 0].multiply(scale),
            base_sky.freqs,
            sampling=base_sky.sampling,
            coord=base_sky.coord,
            frame=base_sky.frame,
        )
        return jnp.sum(simulator.simulate(times, sky=sky))

    sky_gradient = jax.grad(sky_loss)(jnp.asarray(1.0))
    assert jnp.isfinite(sky_gradient)
    assert jnp.abs(sky_gradient) > 0
    epsilon = 1e-4
    sky_finite_difference = (
        sky_loss(1.0 + epsilon) - sky_loss(1.0 - epsilon)
    ) / (2 * epsilon)
    assert jnp.allclose(
        sky_gradient,
        sky_finite_difference,
        rtol=2e-5,
        atol=1e-24,
    )

    def response_loss(scale):
        response = InstrumentResponse.from_arrays(
            base_response.freq,
            base_response.theta_deg,
            base_response.phi_deg,
            base_response.H_theta * scale,
            base_response.H_phi * scale,
            base_response.ZA,
            base_response.Rsky_native,
            base_response.Rmoon_native,
            metadata=base_response.header,
        )
        return jnp.sum(simulator.simulate(times, beam=response))

    response_gradient = jax.grad(response_loss)(jnp.asarray(1.0))
    assert jnp.isfinite(response_gradient)
    assert jnp.abs(response_gradient) > 0
    response_finite_difference = (
        response_loss(1.0 + epsilon)
        - response_loss(1.0 - epsilon)
    ) / (2 * epsilon)
    assert jnp.allclose(
        response_gradient,
        response_finite_difference,
        rtol=3e-5,
        atol=1e-24,
    )

    leaves = jax.tree_util.tree_leaves(base_response)
    assert not any(isinstance(leaf, LabeledArray) for leaf in leaves)
