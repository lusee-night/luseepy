"""Driver tests for the breaking four-port response configuration."""

import copy

import numpy as np

from beam_conversion.common import ResponseArrays, write_response_fits
from lusee.SyntheticResponse import synthetic_four_port_response
from simulation.driver.sim_driver import SimDriver


def test_driver_parses_response_receiver_and_value_frequencies(tmp_path):
    response = synthetic_four_port_response()
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
    write_response_fits(tmp_path / "response.fits", payload)
    config = {
        "paths": {
            "lusee_drive_dir": str(tmp_path),
            "output_dir": str(tmp_path),
        },
        "response": {
            "file": "response.fits",
            "rotation_deg": 0,
            "require_validated": True,
        },
        "receiver": {
            "model": "capacitor",
            "params": {"C_pf": 31.0},
        },
        "observation": {
            "time_range": (
                "2028-01-01 00:00:00 to 2028-01-01 00:05:00"
            ),
            "dt": 60,
            "lat": -23.814,
            "long": 182.258,
            "lmax": 2,
            "T_moon": 250.0,
            "products": "all",
            "freq": {"values": [17.5, 10.0, 17.5]},
        },
        "sky": {"type": "CMB"},
        "simulation": {
            "engine": "croissant",
            "output": "result.fits",
        },
    }
    driver = SimDriver(config)
    assert driver.new_response_schema
    assert np.array_equal(driver.freq, [17.5, 10.0, 17.5])
    assert driver.response.nports == 4
    assert driver.receiver.Z(driver.freq).shape == (3, 4, 4)

    default_config = copy.deepcopy(config)
    default_config["observation"].pop("freq")
    default_config["receiver"] = {
        "model": "measured",
        "params": {
            "freq_mhz": [15.0, 25.0],
            "impedance_ohm": np.broadcast_to(
                40.0 * np.eye(4)[None],
                (2, 4, 4),
            ).astype(complex),
        },
    }
    default_driver = SimDriver(default_config)
    assert np.array_equal(default_driver.freq, [20.0])
