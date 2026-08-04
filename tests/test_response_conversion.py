"""Converter tests independent of production solver exports."""

import csv

import fitsio
import numpy as np
import pytest
from scipy.constants import c

from beam_conversion.common import (
    ResponseArrays,
    VACUUM_IMPEDANCE_OHM,
    bare_fields_to_per_current,
    convert_fields_to_effective_length,
    embedded_fields_to_bare,
    write_response_fits,
)
from beam_conversion.receive_csv import (
    _canonical_conversion_contract,
    convert_receive_csvs,
    read_receive_csv,
)
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


def embed_bare_fields(bare, currents):
    rows = np.moveaxis(bare, (0, 1), (-1, 0))
    embedded = np.einsum("ftpa,fae->ftpe", rows, currents)
    return np.moveaxis(embedded, (0, -1), (1, 0))


def test_hfss_mv_peak_fields_and_rms_thevenin_source_recover_si_length():
    rng = np.random.default_rng(13)
    freq = np.asarray([10.0, 20.0])
    ZA = np.broadcast_to(
        (30.0 + 4.0j) * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    Zref = 50.0
    Vsource_rms = np.sqrt(2.0) * np.broadcast_to(
        np.eye(4)[None],
        (2, 4, 4),
    )
    currents = np.linalg.solve(
        ZA + Zref * np.eye(4)[None],
        Vsource_rms,
    )
    bare = rng.normal(size=(4, 2, 2, 3)) + 1j * rng.normal(
        size=(4, 2, 2, 3)
    )
    embedded_rms = embed_bare_fields(bare, currents)
    hfss_re_peak_mv = embedded_rms * (1e3 * np.sqrt(2.0))
    recovered_theta, recovered_phi, recovered_currents = (
        embedded_fields_to_bare(
            hfss_re_peak_mv,
            hfss_re_peak_mv,
            ZA,
            Zref,
            Vsource_rms,
            field_units="mV",
            field_amplitude_convention="peak",
            vsource_units="V",
            vsource_amplitude_convention="rms",
        )
    )
    np.testing.assert_allclose(recovered_currents, currents, rtol=1e-12)
    np.testing.assert_allclose(recovered_theta, bare, rtol=1e-12)
    np.testing.assert_allclose(recovered_phi, bare, rtol=1e-12)

    effective_length = convert_fields_to_effective_length(
        recovered_theta,
        freq,
        field_kind="rE-per-current",
        field_units="V/A",
    )
    wave_number = 2 * np.pi * freq * 1e6 / c
    factor = -4 * np.pi / (
        1j * wave_number * VACUUM_IMPEDANCE_OHM
    )
    np.testing.assert_allclose(
        effective_length,
        bare * factor[None, :, None, None],
        rtol=1e-12,
    )


def test_embedded_conversion_is_peak_rms_representation_invariant():
    rng = np.random.default_rng(17)
    ZA = np.broadcast_to(
        (27.0 + 3.0j) * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    Vsource_rms = np.sqrt(2.0) * np.broadcast_to(
        np.eye(4)[None],
        (2, 4, 4),
    )
    currents = np.linalg.solve(
        ZA + 50.0 * np.eye(4)[None],
        Vsource_rms,
    )
    bare = rng.normal(size=(4, 2, 1, 2)) + 1j * rng.normal(
        size=(4, 2, 1, 2)
    )
    embedded_rms = embed_bare_fields(bare, currents)
    mixed, _, _ = embedded_fields_to_bare(
        embedded_rms * (1e3 * np.sqrt(2.0)),
        embedded_rms * (1e3 * np.sqrt(2.0)),
        ZA,
        50.0,
        Vsource_rms,
        field_units="mV",
        field_amplitude_convention="peak",
        vsource_units="V",
        vsource_amplitude_convention="rms",
    )
    all_peak, _, _ = embedded_fields_to_bare(
        embedded_rms * np.sqrt(2.0),
        embedded_rms * np.sqrt(2.0),
        ZA,
        50.0,
        Vsource_rms * np.sqrt(2.0),
        field_units="V",
        field_amplitude_convention="peak",
        vsource_units="V",
        vsource_amplitude_convention="peak",
    )
    np.testing.assert_allclose(mixed, bare, rtol=1e-12)
    np.testing.assert_allclose(all_peak, bare, rtol=1e-12)
    np.testing.assert_allclose(all_peak, mixed, rtol=1e-12)


def test_direct_bare_re_uses_complex_current_normalization_consistently():
    rng = np.random.default_rng(19)
    ratio = rng.normal(size=(4, 2, 1, 2)) + 1j * rng.normal(
        size=(4, 2, 1, 2)
    )
    current_rms = np.asarray(
        [
            [0.2 + 0.03j, 0.3 - 0.02j, 0.4 + 0.05j, 0.5 - 0.01j],
            [0.25 - 0.04j, 0.35 + 0.06j, 0.45 - 0.02j, 0.55 + 0.03j],
        ]
    )
    field_rms = ratio * np.swapaxes(
        current_rms,
        0,
        1,
    )[..., None, None]
    recovered_theta, recovered_phi, recovered_current = (
        bare_fields_to_per_current(
            field_rms * (1e3 * np.sqrt(2.0)),
            field_rms * (1e3 * np.sqrt(2.0)),
            current_rms * (1e3 * np.sqrt(2.0)),
            field_units="mV",
            field_amplitude_convention="peak",
            current_units="mA",
            current_amplitude_convention="peak",
        )
    )
    np.testing.assert_allclose(recovered_theta, ratio, rtol=1e-12)
    np.testing.assert_allclose(recovered_phi, ratio, rtol=1e-12)
    np.testing.assert_allclose(recovered_current, current_rms, rtol=1e-12)


def test_preformed_re_ratio_and_effective_length_receive_no_phasor_scaling():
    rng = np.random.default_rng(23)
    freq = np.asarray([10.0, 20.0])
    ratio_v_per_a = rng.normal(size=(4, 2, 1, 2)) + 1j * rng.normal(
        size=(4, 2, 1, 2)
    )
    from_volts = convert_fields_to_effective_length(
        ratio_v_per_a,
        freq,
        field_kind="rE-per-current",
        field_units="V/A",
    )
    from_millivolts = convert_fields_to_effective_length(
        1e3 * ratio_v_per_a,
        freq,
        field_kind="rE-per-current",
        field_units="mV/A",
    )
    np.testing.assert_allclose(from_volts, from_millivolts, rtol=1e-12)

    effective_length = rng.normal(size=(4, 2, 1, 2))
    unchanged = convert_fields_to_effective_length(
        effective_length,
        freq,
        field_kind="effective-length",
        field_units="m",
    )
    np.testing.assert_array_equal(unchanged, effective_length)


def test_embedded_effective_length_contract_is_rejected():
    with pytest.raises(ValueError, match="Embedded inputs must be raw rE"):
        _canonical_conversion_contract(
            input_kind="embedded",
            field_kind="effective-length",
            field_units="m",
            field_amplitude_convention="ratio",
            vsource_units="V",
            vsource_amplitude_convention="rms",
            normalization_current_units=None,
            normalization_current_amplitude_convention=None,
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
        field_units="m",
        field_amplitude_convention="ratio",
        freq_select=[20.0],
        metadata=synthetic_four_port_response().header,
    )
    response = InstrumentResponse(filename)
    assert response.freq.dtype == np.float64
    assert np.array_equal(response.freq, [20.0])


def test_embedded_converter_persists_original_and_canonical_normalization(
    tmp_path,
):
    paths = []
    for port in range(4):
        path = tmp_path / f"embedded_{port}.csv"
        write_receive_csv(path)
        paths.append(path)
    ZA = np.broadcast_to(
        1.0e3 * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    Vsource_rms = np.sqrt(2.0) * np.broadcast_to(
        np.eye(4)[None],
        (2, 4, 4),
    )
    filename = tmp_path / "embedded_response.fits"
    convert_receive_csvs(
        paths,
        filename,
        za=ZA,
        input_kind="embedded",
        field_kind="rE",
        field_units="mV",
        field_amplitude_convention="peak",
        zref=50.0,
        vsource=Vsource_rms,
        vsource_units="V",
        vsource_amplitude_convention="rms",
        metadata=synthetic_four_port_response().header,
    )
    response = InstrumentResponse(filename)
    assert response.header["FIELD_UNIT"] == "mV"
    assert response.header["FIELD_AMP"] == "peak"
    assert response.header["NORM_KIND"] == "vsource"
    assert response.header["NORM_UNIT"] == "V"
    assert response.header["NORM_AMP"] == "rms"
    assert response.header["CANONICAL"] == "H[m],SI-RMS"
    assert response.header["MAX_ICOND"] == pytest.approx(1.0)
    np.testing.assert_allclose(response.Vsource, Vsource_rms)
    np.testing.assert_allclose(response.Zref, 50.0)
    diagnostics = response.response_diagnostics()
    np.testing.assert_allclose(
        diagnostics["normalization_condition_number"],
        1.0,
    )
    assert diagnostics["max_normalization_condition_number"] == (
        pytest.approx(1.0)
    )

    with fitsio.FITS(filename, "rw") as fits:
        fits["H_theta_real"].write_key("MAX_ICOND", 2.0)
    with pytest.raises(ValueError, match="CONTENT hash"):
        InstrumentResponse(filename)


def test_direct_bare_converter_persists_si_normalization_current(tmp_path):
    paths = []
    for port in range(4):
        path = tmp_path / f"bare_{port}.csv"
        write_receive_csv(path)
        paths.append(path)
    ZA = np.broadcast_to(
        1.0e6 * np.eye(4)[None],
        (2, 4, 4),
    ).copy()
    current_peak_ma = 1e3 * np.sqrt(2.0) * np.ones(
        (2, 4),
        dtype=np.complex128,
    )
    filename = tmp_path / "bare_response.fits"
    convert_receive_csvs(
        paths,
        filename,
        za=ZA,
        input_kind="bare",
        field_kind="rE",
        field_units="V",
        field_amplitude_convention="rms",
        normalization_current=current_peak_ma,
        normalization_current_units="mA",
        normalization_current_amplitude_convention="peak",
        metadata=synthetic_four_port_response().header,
    )
    response = InstrumentResponse(filename)
    assert response.header["NORM_KIND"] == "current"
    assert response.header["NORM_UNIT"] == "mA"
    assert response.header["NORM_AMP"] == "peak"
    np.testing.assert_allclose(response.Inorm, 1.0)
    assert response.Vsource is None


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
        np.asarray(synthetic.Rloss_native),
        metadata=synthetic.header,
    )
    filename = tmp_path / "bad_unit.fits"
    write_response_fits(filename, response)
    with fitsio.FITS(filename, "rw") as fits:
        fits["freq"].write_key("BUNIT", "Hz")
    with pytest.raises(ValueError, match="expected 'MHz'"):
        InstrumentResponse(filename)


def _write_loaded_csvs(tmp_path, R_theta, R_phi, freq, theta, phi):
    fieldnames = [
        "freq_MHz",
        "phi_deg",
        "theta_deg",
        "re(rx_Phi)",
        "im(rx_Phi)",
        "re(rx_Theta)",
        "im(rx_Theta)",
    ]
    paths = []
    for port in range(4):
        path = tmp_path / f"loaded_{port}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for fi, frequency in enumerate(freq):
                for ti, tv in enumerate(theta):
                    for pi, pv in enumerate(phi):
                        writer.writerow(
                            {
                                "freq_MHz": frequency,
                                "phi_deg": pv,
                                "theta_deg": tv,
                                "re(rx_Phi)": R_phi[port, fi, ti, pi].real,
                                "im(rx_Phi)": R_phi[port, fi, ti, pi].imag,
                                "re(rx_Theta)": R_theta[port, fi, ti, pi].real,
                                "im(rx_Theta)": R_theta[port, fi, ti, pi].imag,
                            }
                        )
        paths.append(path)
    return paths


def test_loaded_receive_fields_unload_to_bare_effective_length(tmp_path):
    rng = np.random.default_rng(7)
    freq = np.asarray([10.0, 20.0])
    theta = np.asarray([0.0, 90.0])
    phi = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
    shape = (4, freq.size, theta.size, phi.size)
    H_theta = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    H_phi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    # keep the periodic wrap column consistent
    H_theta[..., -1] = H_theta[..., 0]
    H_phi[..., -1] = H_phi[..., 0]
    base = rng.standard_normal((freq.size, 4, 4))
    ZA = 30 * np.eye(4)[None] + 2.0 * (base + np.swapaxes(base, -1, -2)) \
        - 5j * np.eye(4)[None]
    ZL = np.zeros((freq.size, 4, 4), dtype=complex)
    ZL[:, np.arange(4), np.arange(4)] = 2.0 - 150.0j
    mismatch = np.einsum(
        "fab,fbc->fac",
        ZL,
        np.linalg.inv(ZA + ZL),
    )
    R_theta = np.einsum("fab,bfij->afij", mismatch, H_theta)
    R_phi = np.einsum("fab,bfij->afij", mismatch, H_phi)
    paths = _write_loaded_csvs(tmp_path, R_theta, R_phi, freq, theta, phi)
    output = tmp_path / "loaded_response.fits"
    convert_receive_csvs(
        paths,
        output,
        za=ZA,
        input_kind="loaded",
        field_kind="effective-length",
        field_units="m",
        field_amplitude_convention="ratio",
        rloss=np.zeros_like(ZA),
        zload=ZL,
        dtype="float64",
        metadata={"LOSSMODEL": "PEC", "RLOSSSRC": "test"},
        allow_unvalidated=True,
    )
    with fitsio.FITS(output, "r") as fits:
        recovered = (
            fits["H_theta_real"].read() + 1j * fits["H_theta_imag"].read()
        )
        recovered_phi = (
            fits["H_phi_real"].read() + 1j * fits["H_phi_imag"].read()
        )
        stored_zl = fits["ZLoad_real"].read() + 1j * fits["ZLoad_imag"].read()
        header = dict(fits[0].read_header())
    np.testing.assert_allclose(recovered, H_theta, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(recovered_phi, H_phi, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(stored_zl, ZL, rtol=1e-12, atol=0)
    assert header["INPUT_KIND"] == "loaded"
    assert header["NORM_KIND"] == "unloaded-zl"
    response = InstrumentResponse(output, require_validated=False)
    assert response.ZLoad is not None
    np.testing.assert_allclose(np.asarray(response.ZLoad), ZL)


def test_loaded_conversion_requires_zload(tmp_path):
    rng = np.random.default_rng(3)
    freq = np.asarray([10.0])
    theta = np.asarray([0.0, 90.0])
    phi = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
    shape = (4, freq.size, theta.size, phi.size)
    fields = rng.standard_normal(shape) + 0j
    paths = _write_loaded_csvs(tmp_path, fields, fields, freq, theta, phi)
    with pytest.raises(ValueError, match="require the solver-side ZLoad"):
        convert_receive_csvs(
            paths,
            tmp_path / "out.fits",
            za=np.broadcast_to(50 * np.eye(4), (1, 4, 4)),
            input_kind="loaded",
            field_kind="effective-length",
            field_units="m",
            field_amplitude_convention="ratio",
            metadata={"LOSSMODEL": "PEC", "RLOSSSRC": "test"},
            allow_unvalidated=True,
        )


def test_phi_source_zero_rolls_maps_to_east_zero():
    from beam_conversion.receive_csv import _roll_phi_to_enu

    phi = np.asarray([0.0, 90.0, 180.0, 270.0, 360.0])
    field = np.zeros((4, 1, 2, phi.size), dtype=complex)
    field[..., 0] = 1.0  # feature on the solver's phi=0 axis
    field[..., -1] = field[..., 0]
    rolled_theta, rolled_phi = _roll_phi_to_enu(field, field, phi, 180.0)
    # solver phi=0 pointed West (ENU azimuth 180): feature must move there
    assert np.all(rolled_theta[..., 2] == 1.0)
    assert np.all(rolled_theta[..., 0] == 0.0)
    # wraparound column mirrors the new first column
    np.testing.assert_array_equal(
        rolled_theta[..., -1],
        rolled_theta[..., 0],
    )
    np.testing.assert_array_equal(rolled_phi, rolled_theta)
