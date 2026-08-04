"""Streaming parser and command-line converter for receive-matrix CSV files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import (
    ResponseArrays,
    bare_fields_to_per_current,
    convert_fields_to_effective_length,
    embedded_fields_to_bare,
    voltage_phasor_to_si_rms,
    write_response_fits,
)
from .touchstone import read_touchstone_z


_COLUMN_ALIASES = {
    "freq": ("freq_MHz", "frequency_MHz", "freq"),
    "phi": ("phi_deg", "phi"),
    "theta": ("theta_deg", "theta"),
    "phi_real": ("re(rx_Phi)", "rx_Phi_real", "phi_real"),
    "phi_imag": ("im(rx_Phi)", "rx_Phi_imag", "phi_imag"),
    "theta_real": ("re(rx_Theta)", "rx_Theta_real", "theta_real"),
    "theta_imag": ("im(rx_Theta)", "rx_Theta_imag", "theta_imag"),
}


def _frequency_selection(freq, requested):
    if requested is None:
        return np.arange(freq.size, dtype=np.int32)
    requested = np.asarray(requested, dtype=np.float64).reshape(-1)
    if requested.size == 0:
        raise ValueError("freq_select must contain at least one frequency.")
    indices = []
    for value in requested:
        matches = np.flatnonzero(np.isclose(freq, value, rtol=0.0, atol=1e-9))
        if matches.size != 1:
            raise ValueError(
                f"Selected frequency {value} MHz is not one native CSV "
                "frequency."
            )
        indices.append(int(matches[0]))
    if len(set(indices)) != len(indices):
        raise ValueError("freq_select must not contain duplicate frequencies.")
    return np.asarray(indices, dtype=np.int32)


def _resolve_columns(fieldnames):
    resolved = {}
    available = set(fieldnames or ())
    for logical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in available:
                resolved[logical] = alias
                break
        else:
            raise ValueError(
                f"Missing CSV column for {logical!r}; tried {aliases}."
            )
    return resolved


def read_receive_csv(path, *, theta_max=90.0, zero_tolerance=1e-12):
    """Read one solver receive CSV in two streaming passes."""
    path = Path(path)
    axes = {"freq": set(), "theta": set(), "phi": set()}
    columns = None
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = _resolve_columns(reader.fieldnames)
        for row in reader:
            axes["freq"].add(float(row[columns["freq"]]))
            axes["theta"].add(float(row[columns["theta"]]))
            axes["phi"].add(float(row[columns["phi"]]))
    freq = np.asarray(sorted(axes["freq"]), dtype=np.float64)
    theta_all = np.asarray(sorted(axes["theta"]), dtype=np.float64)
    phi = np.asarray(sorted(axes["phi"]), dtype=np.float64)
    theta = theta_all[theta_all <= theta_max + 1e-12]
    theta_field = np.empty((freq.size, theta.size, phi.size), dtype=np.complex128)
    phi_field = np.empty_like(theta_field)
    theta_field.fill(np.nan)
    phi_field.fill(np.nan)
    indices = {
        "freq": {value: index for index, value in enumerate(freq)},
        "theta": {value: index for index, value in enumerate(theta)},
        "phi": {value: index for index, value in enumerate(phi)},
    }
    max_below_horizon = 0.0
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            fval = float(row[columns["freq"]])
            tval = float(row[columns["theta"]])
            pval = float(row[columns["phi"]])
            theta_value = complex(
                float(row[columns["theta_real"]]),
                float(row[columns["theta_imag"]]),
            )
            phi_value = complex(
                float(row[columns["phi_real"]]),
                float(row[columns["phi_imag"]]),
            )
            if tval > theta_max + 1e-12:
                max_below_horizon = max(
                    max_below_horizon,
                    abs(theta_value),
                    abs(phi_value),
                )
                continue
            index = (
                indices["freq"][fval],
                indices["theta"][tval],
                indices["phi"][pval],
            )
            theta_field[index] = theta_value
            phi_field[index] = phi_value
    if max_below_horizon > zero_tolerance:
        raise ValueError(
            "Receive CSV is nonzero below the requested horizon: "
            f"maximum amplitude {max_below_horizon}."
        )
    if np.any(~np.isfinite(theta_field)) or np.any(~np.isfinite(phi_field)):
        raise ValueError("Receive CSV does not cover a complete regular grid.")
    return freq, theta, phi, theta_field, phi_field


def _roll_phi_to_enu(Etheta, Ephi, phi_deg, phi_source_zero_deg):
    """Roll the periodic phi axis so phi=0 is local East in ENU.

    ``phi_source_zero_deg`` is the ENU azimuth of the solver grid's phi=0
    axis (for the LuSEE HFSS exports the +x axis points West, so 180). The
    wraparound phi bin is preserved.
    """
    offset = float(phi_source_zero_deg) % 360.0
    if offset == 0.0:
        return Etheta, Ephi
    phi = np.asarray(phi_deg, dtype=np.float64)
    if phi.size < 2 or not np.isclose(phi[-1] - phi[0], 360.0):
        raise ValueError(
            "phi_source_zero_deg requires a periodic phi grid with the "
            "0/360 wraparound bin."
        )
    step = float(phi[1] - phi[0])
    if not np.allclose(np.diff(phi), step):
        raise ValueError("phi grid must be uniform to roll the azimuth zero.")
    bins = offset / step
    if not np.isclose(bins, round(bins)):
        raise ValueError(
            f"phi_source_zero_deg must be a multiple of the phi step {step}."
        )
    bins = int(round(bins))

    def roll(field):
        unique = field[..., :-1]
        rolled = np.roll(unique, shift=bins, axis=-1)
        return np.concatenate((rolled, rolled[..., :1]), axis=-1)

    return roll(Etheta), roll(Ephi)


def _unload_receive_fields(R_theta, R_phi, ZA, ZLoad):
    """Recover bare fields from loaded receive fields.

    The solver-side loading is ``R = ZL (ZA + ZL)^-1 H``, so
    ``H = (ZA + ZL) ZL^-1 R`` evaluated with batched solves.
    """
    ZA = np.asarray(ZA)
    ZLoad = np.asarray(ZLoad)
    if ZA.shape != ZLoad.shape or ZA.shape[-2:] != (4, 4):
        raise ValueError("ZA and ZLoad must share shape (frequency, 4, 4).")
    unload = (ZA + ZLoad) @ np.linalg.solve(
        ZLoad,
        np.broadcast_to(np.eye(4), ZLoad.shape),
    )

    def apply(field):
        if field.shape[:2] != (4, ZA.shape[0]):
            raise ValueError(
                "Loaded field arrays must have shape (4, frequency, ...)."
            )
        return np.einsum("fab,bf...->af...", unload, field)

    return apply(R_theta), apply(R_phi)


def _canonical_conversion_contract(
    *,
    input_kind,
    field_kind,
    field_units,
    field_amplitude_convention,
    vsource_units,
    vsource_amplitude_convention,
    normalization_current_units,
    normalization_current_amplitude_convention,
):
    input_kind = str(input_kind).strip().lower()
    field_kind = str(field_kind).strip().lower().replace("_", "-")
    field_units = str(field_units).strip()
    field_amplitude = str(field_amplitude_convention).strip().lower()
    if input_kind == "embedded":
        if field_kind != "re":
            raise ValueError(
                "Embedded inputs must be raw rE; embedded effective-length "
                "or rE-per-current inputs are undefined."
            )
        if vsource_units is None or vsource_amplitude_convention is None:
            raise ValueError(
                "Embedded inputs require explicit Vsource units and "
                "amplitude convention."
            )
        metadata = {
            "NORM_KIND": "vsource",
            "NORM_UNIT": str(vsource_units).strip(),
            "NORM_AMP": str(vsource_amplitude_convention).strip().lower(),
        }
    elif input_kind == "bare" and field_kind == "re":
        if (
            normalization_current_units is None
            or normalization_current_amplitude_convention is None
        ):
            raise ValueError(
                "Direct bare rE inputs require explicit normalization-current "
                "units and amplitude convention."
            )
        metadata = {
            "NORM_KIND": "current",
            "NORM_UNIT": str(normalization_current_units).strip(),
            "NORM_AMP": str(
                normalization_current_amplitude_convention
            ).strip().lower(),
        }
    elif input_kind == "bare" and field_kind == "re-per-current":
        metadata = {
            "NORM_KIND": "already-per-ampere",
            "NORM_UNIT": "not-applicable",
            "NORM_AMP": "ratio",
        }
    elif input_kind == "bare" and field_kind == "effective-length":
        metadata = {
            "NORM_KIND": "already-effective-length",
            "NORM_UNIT": "not-applicable",
            "NORM_AMP": "ratio",
        }
    elif input_kind == "loaded" and field_kind == "effective-length":
        # Solver-side loaded receive fields ZL (ZA + ZL)^-1 H in meters.
        # The converter recovers bare H with the supplied ZLoad matrix.
        metadata = {
            "NORM_KIND": "unloaded-zl",
            "NORM_UNIT": "Ohm",
            "NORM_AMP": "ratio",
        }
    else:
        raise ValueError(
            "input_kind must be 'embedded', 'bare', or 'loaded', with a "
            "supported field_kind."
        )
    metadata.update(
        {
            "INPUT_KIND": input_kind,
            "FIELD_KIND": field_kind,
            "FIELD_UNIT": field_units,
            "FIELD_AMP": field_amplitude,
            "CANONICAL": "H[m],SI-RMS",
        }
    )
    from lusee.ResponsePhysics import validate_conversion_metadata

    validate_conversion_metadata(metadata)
    return metadata


def convert_receive_csvs(
    csv_paths,
    output,
    *,
    za,
    input_kind,
    field_kind,
    field_units,
    field_amplitude_convention,
    rloss=None,
    zref=None,
    vsource=None,
    vsource_units=None,
    vsource_amplitude_convention=None,
    normalization_current=None,
    normalization_current_units=None,
    normalization_current_amplitude_convention=None,
    zload=None,
    phi_source_zero_deg=0.0,
    dtype="float32",
    freq_select=None,
    metadata=None,
    allow_unvalidated=False,
):
    """Convert four port-ordered receive CSVs to one response FITS."""
    if len(csv_paths) != 4:
        raise ValueError("Exactly four receive CSV paths are required.")
    contract = _canonical_conversion_contract(
        input_kind=input_kind,
        field_kind=field_kind,
        field_units=field_units,
        field_amplitude_convention=field_amplitude_convention,
        vsource_units=vsource_units,
        vsource_amplitude_convention=vsource_amplitude_convention,
        normalization_current_units=normalization_current_units,
        normalization_current_amplitude_convention=(
            normalization_current_amplitude_convention
        ),
    )
    input_kind = contract["INPUT_KIND"]
    field_kind = contract["FIELD_KIND"]
    loaded = [read_receive_csv(path) for path in csv_paths]
    freq, theta, phi = loaded[0][:3]
    for other in loaded[1:]:
        if not (
            np.array_equal(freq, other[0])
            and np.array_equal(theta, other[1])
            and np.array_equal(phi, other[2])
        ):
            raise ValueError("All receive CSVs must share one grid.")
    Etheta = np.stack([entry[3] for entry in loaded])
    Ephi = np.stack([entry[4] for entry in loaded])
    Etheta, Ephi = _roll_phi_to_enu(
        Etheta,
        Ephi,
        phi,
        phi_source_zero_deg,
    )
    ZA = np.asarray(za)
    native_nfrequency = freq.size
    selection = _frequency_selection(freq, freq_select)
    freq = freq[selection]
    Etheta = Etheta[:, selection]
    Ephi = Ephi[:, selection]
    if ZA.shape[0] != native_nfrequency:
        raise ValueError("ZA and receive CSV frequency axes have different lengths.")
    ZA = ZA[selection]
    if rloss is not None:
        rloss = np.asarray(rloss)
        if rloss.shape[0] != native_nfrequency:
            raise ValueError(
                "Rloss and receive CSV frequency axes have different lengths."
            )
        rloss = rloss[selection]
    if vsource is not None:
        vsource = np.asarray(vsource)
        if vsource.shape != (native_nfrequency, 4, 4):
            raise ValueError("Vsource must have shape (frequency, 4, 4).")
        vsource = vsource[selection]
    if normalization_current is not None:
        normalization_current = np.asarray(normalization_current)
        if normalization_current.shape != (native_nfrequency, 4):
            raise ValueError(
                "normalization_current must have shape (frequency, 4)."
            )
        normalization_current = normalization_current[selection]
    if zload is not None:
        zload = np.asarray(zload)
        if zload.shape != (native_nfrequency, 4, 4):
            raise ValueError("ZLoad must have shape (frequency, 4, 4).")
        zload = zload[selection]
    if zref is not None:
        zref_array = np.asarray(zref)
        if zref_array.ndim == 0:
            zref = np.full((freq.size, 4), zref_array)
        elif zref_array.shape == (4,):
            zref = np.broadcast_to(zref_array[None], (freq.size, 4))
        elif zref_array.shape == (native_nfrequency, 4):
            zref = zref_array[selection]
        else:
            raise ValueError(
                "Zref must be scalar, per-port, or frequency-by-port."
            )
    canonical_vsource = None
    canonical_inorm = None
    canonical_zload = None
    if input_kind == "loaded":
        if zload is None:
            raise ValueError(
                "Loaded inputs require the solver-side ZLoad matrix."
            )
        if any(
            value is not None
            for value in (
                vsource,
                vsource_units,
                vsource_amplitude_convention,
                normalization_current,
                normalization_current_units,
                normalization_current_amplitude_convention,
                zref,
            )
        ):
            raise ValueError(
                "Loaded inputs cannot supply Vsource, Zref, or "
                "normalization_current."
            )
        Etheta, Ephi = _unload_receive_fields(Etheta, Ephi, ZA, zload)
        canonical_zload = zload
        canonical_field_kind = "effective-length"
        canonical_field_units = field_units
    elif input_kind == "embedded":
        if zref is None or vsource is None:
            raise ValueError("Embedded inputs require zref and vsource.")
        if any(
            value is not None
            for value in (
                normalization_current,
                normalization_current_units,
                normalization_current_amplitude_convention,
            )
        ):
            raise ValueError(
                "Embedded inputs cannot also supply a normalization current."
            )
        Etheta, Ephi, _ = embedded_fields_to_bare(
            Etheta,
            Ephi,
            ZA,
            zref,
            vsource,
            field_units=field_units,
            field_amplitude_convention=field_amplitude_convention,
            vsource_units=vsource_units,
            vsource_amplitude_convention=(
                vsource_amplitude_convention
            ),
        )
        canonical_vsource = voltage_phasor_to_si_rms(
            vsource,
            units=vsource_units,
            amplitude_convention=vsource_amplitude_convention,
        )
        canonical_field_kind = "rE-per-current"
        canonical_field_units = "V/A"
    elif field_kind == "re":
        if normalization_current is None:
            raise ValueError(
                "Direct bare rE inputs require normalization_current."
            )
        if any(
            value is not None
            for value in (
                vsource,
                vsource_units,
                vsource_amplitude_convention,
                zref,
            )
        ):
            raise ValueError("Bare inputs cannot supply Vsource or Zref.")
        Etheta, Ephi, canonical_inorm = bare_fields_to_per_current(
            Etheta,
            Ephi,
            normalization_current,
            field_units=field_units,
            field_amplitude_convention=field_amplitude_convention,
            current_units=normalization_current_units,
            current_amplitude_convention=(
                normalization_current_amplitude_convention
            ),
        )
        canonical_field_kind = "rE-per-current"
        canonical_field_units = "V/A"
    else:
        if any(
            value is not None
            for value in (
                vsource,
                vsource_units,
                vsource_amplitude_convention,
                normalization_current,
                normalization_current_units,
                normalization_current_amplitude_convention,
                zref,
            )
        ):
            raise ValueError(
                "Already-normalized bare inputs cannot supply Vsource, "
                "Zref, or normalization_current."
            )
        canonical_field_kind = field_kind
        canonical_field_units = field_units
    Htheta = convert_fields_to_effective_length(
        Etheta,
        freq,
        field_kind=canonical_field_kind,
        field_units=canonical_field_units,
    )
    Hphi = convert_fields_to_effective_length(
        Ephi,
        freq,
        field_kind=canonical_field_kind,
        field_units=canonical_field_units,
    )
    meta = dict(metadata or {})
    meta.update(contract)
    meta["SRCAZ0"] = float(phi_source_zero_deg) % 360.0
    response = ResponseArrays(
        freq,
        theta,
        phi,
        Htheta,
        Hphi,
        ZA,
        Rloss=rloss,
        Vsource=canonical_vsource,
        Inorm=canonical_inorm,
        Zref=zref,
        ZLoad=canonical_zload,
        metadata=meta,
    )
    return write_response_fits(
        output,
        response,
        dtype=dtype,
        validated=not allow_unvalidated,
    )


def read_zmatrix_csv(path):
    """Read a dense complex Z matrix CSV (freq_Hz, freq_MHz, re/im(Zij))."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    freq_mhz = np.asarray(rows["freq_MHz"], dtype=np.float64)
    Z = np.empty((freq_mhz.size, 4, 4), dtype=np.complex128)
    for a in range(4):
        for b in range(4):
            Z[:, a, b] = (
                rows[f"reZ{a + 1}{b + 1}"] + 1j * rows[f"imZ{a + 1}{b + 1}"]
            )
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z matrix CSV contains non-finite values.")
    return freq_mhz, Z


def main(argv=None):
    """Command-line entry point for the four-CSV response converter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs=4)
    parser.add_argument("--output", required=True)
    za_group = parser.add_mutually_exclusive_group(required=True)
    za_group.add_argument("--touchstone")
    za_group.add_argument(
        "--zmatrix-csv",
        help=(
            "Dense complex Z-matrix CSV with freq_Hz,freq_MHz,re/im(Zij) "
            "columns (skrf-converted s4p export)"
        ),
    )
    loss_group = parser.add_mutually_exclusive_group(required=True)
    loss_group.add_argument(
        "--pec",
        action="store_true",
        help="Declare an explicit zero antenna-metal loss matrix",
    )
    loss_group.add_argument(
        "--rloss-npy",
        help="Dense complex (frequency,4,4) antenna-metal loss matrix",
    )
    loss_group.add_argument(
        "--za-pec-npy",
        help=(
            "Dense complex PEC impedance matrix; derive "
            "Rloss=Herm(ZA_lossy)-Herm(ZA_PEC)"
        ),
    )
    parser.add_argument(
        "--vsource-npy",
        help=(
            "Full complex (frequency,4,4) solver source-voltage matrix; "
            "required for embedded inputs"
        ),
    )
    parser.add_argument(
        "--vsource-units",
        choices=("V", "mV"),
        help="Units of --vsource-npy",
    )
    parser.add_argument(
        "--vsource-amplitude",
        choices=("rms", "peak"),
        help="Phasor convention of --vsource-npy",
    )
    parser.add_argument(
        "--normalization-current-npy",
        help=(
            "Complex (frequency,4) current for direct bare rE patterns"
        ),
    )
    parser.add_argument(
        "--normalization-current-units",
        choices=("A", "mA"),
    )
    parser.add_argument(
        "--normalization-current-amplitude",
        choices=("rms", "peak"),
    )
    parser.add_argument(
        "--input-kind",
        choices=("embedded", "bare", "loaded"),
        required=True,
    )
    parser.add_argument(
        "--zload-npy",
        help=(
            "Dense complex (frequency,4,4) solver-side load matrix used to "
            "produce loaded receive fields; required for --input-kind loaded"
        ),
    )
    parser.add_argument(
        "--phi-source-zero-deg",
        type=float,
        default=0.0,
        help=(
            "ENU azimuth of the solver grid's phi=0 axis; the converter "
            "rolls maps so stored phi=0 is local East (LuSEE HFSS: 180)"
        ),
    )
    parser.add_argument(
        "--field-kind",
        choices=("rE", "rE-per-current", "effective-length"),
        required=True,
    )
    parser.add_argument(
        "--field-units",
        choices=("V", "mV", "V/A", "mV/A", "m"),
        required=True,
    )
    parser.add_argument(
        "--field-amplitude",
        choices=("rms", "peak", "ratio"),
        required=True,
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--freq-select",
        nargs="+",
        type=float,
        help="Native CSV frequencies in MHz to retain",
    )
    parser.add_argument(
        "--provenance-json",
        help=(
            "JSON object containing the explicitly reviewed solver and "
            "coordinate provenance required for VALIDATED=True"
        ),
    )
    parser.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args(argv)
    if args.touchstone is not None:
        freq_za, ZA, zref = read_touchstone_z(args.touchstone)
        za_source = str(args.touchstone)
    else:
        freq_za, ZA = read_zmatrix_csv(args.zmatrix_csv)
        zref = None
        za_source = str(args.zmatrix_csv)
    freq_csv = read_receive_csv(args.csv[0])[0]
    if not np.allclose(freq_za, freq_csv):
        raise ValueError("Z-matrix and receive CSV frequency grids differ.")
    if args.input_kind == "loaded" and args.zload_npy is None:
        raise ValueError("Loaded conversion requires --zload-npy.")
    if args.input_kind == "embedded" and args.zmatrix_csv is not None:
        raise ValueError(
            "Embedded conversion requires --touchstone for its Zref."
        )
    if args.input_kind == "embedded" and any(
        value is None
        for value in (
            args.vsource_npy,
            args.vsource_units,
            args.vsource_amplitude,
        )
    ):
        raise ValueError(
            "Embedded conversion requires --vsource-npy, --vsource-units, "
            "and --vsource-amplitude."
        )
    if args.input_kind == "bare" and args.field_kind == "rE" and any(
        value is None
        for value in (
            args.normalization_current_npy,
            args.normalization_current_units,
            args.normalization_current_amplitude,
        )
    ):
        raise ValueError(
            "Direct bare rE conversion requires "
            "--normalization-current-npy, --normalization-current-units, "
            "and --normalization-current-amplitude."
        )
    vsource = (
        np.load(args.vsource_npy)
        if args.vsource_npy is not None
        else None
    )
    normalization_current = (
        np.load(args.normalization_current_npy)
        if args.normalization_current_npy is not None
        else None
    )
    if args.provenance_json is None:
        metadata = {}
    else:
        with Path(args.provenance_json).open() as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError("--provenance-json must contain a JSON object.")
    metadata.setdefault("SOURCE", ",".join(args.csv))
    metadata.setdefault("ZA_SOURCE", za_source)
    if args.pec:
        rloss = np.zeros_like(ZA)
        loss_model = "PEC"
        loss_source = "explicit-cli-pec"
    elif args.rloss_npy is not None:
        rloss = np.load(args.rloss_npy)
        loss_model = "lossy"
        loss_source = str(args.rloss_npy)
    else:
        ZA_pec = np.asarray(np.load(args.za_pec_npy))
        if ZA_pec.shape != ZA.shape:
            raise ValueError("--za-pec-npy must have the same shape as ZA.")
        hermitian = lambda value: 0.5 * (
            value + np.swapaxes(value.conjugate(), -1, -2)
        )
        rloss = hermitian(ZA) - hermitian(ZA_pec)
        loss_model = "lossy"
        loss_source = str(args.za_pec_npy)
    declared_loss_model = str(
        metadata.get("LOSSMODEL", loss_model)
    ).strip().lower()
    if declared_loss_model != loss_model.lower():
        raise ValueError(
            "Provenance LOSSMODEL contradicts the selected loss input."
        )
    metadata["LOSSMODEL"] = loss_model
    metadata.setdefault("RLOSSSRC", loss_source)
    return convert_receive_csvs(
        args.csv,
        args.output,
        za=ZA,
        input_kind=args.input_kind,
        field_kind=args.field_kind,
        field_units=args.field_units,
        field_amplitude_convention=args.field_amplitude,
        rloss=rloss,
        zref=zref if args.input_kind == "embedded" else None,
        vsource=vsource,
        vsource_units=args.vsource_units,
        vsource_amplitude_convention=args.vsource_amplitude,
        normalization_current=normalization_current,
        normalization_current_units=args.normalization_current_units,
        normalization_current_amplitude_convention=(
            args.normalization_current_amplitude
        ),
        zload=(
            np.load(args.zload_npy)
            if args.zload_npy is not None
            else None
        ),
        phi_source_zero_deg=args.phi_source_zero_deg,
        dtype=args.dtype,
        freq_select=args.freq_select,
        allow_unvalidated=args.allow_unvalidated,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
