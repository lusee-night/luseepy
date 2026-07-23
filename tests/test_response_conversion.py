"""Converter tests independent of production solver exports."""

import csv

import fitsio
import numpy as np
import pytest
from scipy.constants import c

from beam_conversion.common import (
    ResponseArrays,
    VACUUM_IMPEDANCE_OHM,
    convert_fields_to_effective_length,
    write_response_fits,
)
from beam_conversion.receive_csv import convert_receive_csvs, read_receive_csv
from beam_conversion.touchstone import s_to_z
from lusee.InstrumentResponse import InstrumentResponse
from lusee.SyntheticResponse import synthetic_four_port_response


def write_receive_csv(path, *, nonzero_below=False):
    fieldnames = [
        "freq_MHz",
        "phi_deg",
        "theta_deg",
        "re(rx_Phi)",
        "im(rx_Phi)",
        "re(rx_Theta)",
        "im(rx_Theta)",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frequency in (10.0, 20.0):
            for theta in (0.0, 90.0, 135.0):
                for phi in (0.0, 90.0, 180.0, 270.0, 360.0):
                    below = nonzero_below and theta > 90.0
                    writer.writerow(
                        {
                            "freq_MHz": frequency,
                            "phi_deg": phi,
                            "theta_deg": theta,
                            "re(rx_Phi)": 1.0 if below else 0.0,
                            "im(rx_Phi)": 0.0,
                            "re(rx_Theta)": (
                                frequency + theta / 100 + phi / 1000
                                if theta <= 90.0
                                else 0.0
                            ),
                            "im(rx_Theta)": (
                                -0.25 if theta <= 90.0 else 0.0
                            ),
                        }
                    )


def test_streaming_receive_csv_grid_and_horizon_validation(tmp_path):
    filename = tmp_path / "receive.csv"
    write_receive_csv(filename)
    freq, theta, phi, theta_field, phi_field = read_receive_csv(filename)
    assert np.array_equal(freq, [10.0, 20.0])
    assert np.array_equal(theta, [0.0, 90.0])
    assert np.array_equal(phi, [0.0, 90.0, 180.0, 270.0, 360.0])
    assert theta_field.shape == (2, 2, 5)
    assert phi_field.shape == theta_field.shape
    assert theta_field[1, 1, 2] == pytest.approx(21.08 - 0.25j)

    invalid = tmp_path / "receive_nonzero_below.csv"
    write_receive_csv(invalid, nonzero_below=True)
    with pytest.raises(ValueError, match="nonzero below"):
        read_receive_csv(invalid)


def test_peak_and_re_conversion_are_applied_exactly_once():
    fields = np.ones((4, 2, 1, 1), dtype=np.complex128)
    freq = np.asarray([10.0, 20.0])
    rms_length = convert_fields_to_effective_length(
        fields,
        freq,
        field_kind="rE",
        amplitude_convention="peak",
    )
    wave_number = 2 * np.pi * freq * 1e6 / c
    expected = (
        -4
        * np.pi
        / (1j * wave_number * VACUUM_IMPEDANCE_OHM)
        / np.sqrt(2.0)
    )
    np.testing.assert_allclose(
        rms_length[:, :, 0, 0],
        np.broadcast_to(expected[None], (4, 2)),
    )


def test_converter_frequency_selection_keeps_float64_native_grid(tmp_path):
    paths = []
    for port in range(4):
        path = tmp_path / f"receive_{port}.csv"
        write_receive_csv(path)
        paths.append(path)
    ZA = np.broadcast_to(
        1.0e6 * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    filename = tmp_path / "selected_response.fits"
    convert_receive_csvs(
        paths,
        filename,
        za=ZA,
        input_kind="bare",
        field_kind="effective-length",
        amplitude_convention="rms",
        freq_select=[20.0],
        metadata=synthetic_four_port_response().header,
    )
    response = InstrumentResponse(filename)
    assert response.freq.dtype == np.float64
    assert np.array_equal(response.freq, [20.0])


def test_full_matrix_s_to_z_keeps_noncommuting_off_diagonals():
    rng = np.random.default_rng(9)
    scattering = 0.08 * (
        rng.normal(size=(3, 4, 4))
        + 1j * rng.normal(size=(3, 4, 4))
    )
    zref = np.asarray([40.0, 50.0, 60.0, 70.0])
    result = s_to_z(scattering, zref)
    sqrt_z = np.diag(np.sqrt(zref))
    expected = np.stack(
        [
            sqrt_z
            @ (np.eye(4) + matrix)
            @ np.linalg.inv(np.eye(4) - matrix)
            @ sqrt_z
            for matrix in scattering
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
    assert np.any(np.abs(result[:, ~np.eye(4, dtype=bool)]) > 0)


def test_response_loader_rejects_contradictory_machine_unit(tmp_path):
    synthetic = synthetic_four_port_response()
    response = ResponseArrays(
        synthetic.freq,
        synthetic.theta_deg,
        synthetic.phi_deg,
        np.asarray(synthetic.H_theta),
        np.asarray(synthetic.H_phi),
        np.asarray(synthetic.ZA),
        np.asarray(synthetic.Rsky_native),
        np.asarray(synthetic.Rmoon_native),
        metadata=synthetic.header,
    )
    filename = tmp_path / "bad_unit.fits"
    write_response_fits(filename, response)
    with fitsio.FITS(filename, "rw") as fits:
        fits["freq"].write_key("BUNIT", "Hz")
    with pytest.raises(ValueError, match="expected 'MHz'"):
        InstrumentResponse(filename)
