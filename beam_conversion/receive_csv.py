"""Streaming parser and command-line converter for receive-matrix CSV files."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import (
    ResponseArrays,
    convert_fields_to_effective_length,
    embedded_fields_to_bare,
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


def convert_receive_csvs(
    csv_paths,
    output,
    *,
    za,
    input_kind,
    field_kind,
    amplitude_convention,
    zref=None,
    vsource=None,
    dtype="float32",
    freq_select=None,
    metadata=None,
    allow_unvalidated=False,
):
    """Convert four port-ordered receive CSVs to one response FITS."""
    if len(csv_paths) != 4:
        raise ValueError("Exactly four receive CSV paths are required.")
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
    ZA = np.asarray(za)
    selection = _frequency_selection(freq, freq_select)
    freq = freq[selection]
    Etheta = Etheta[:, selection]
    Ephi = Ephi[:, selection]
    if ZA.shape[0] != loaded[0][0].size:
        raise ValueError("ZA and receive CSV frequency axes have different lengths.")
    ZA = ZA[selection]
    if vsource is not None:
        vsource = np.asarray(vsource)[selection]
    if zref is not None:
        zref_array = np.asarray(zref)
        if zref_array.ndim >= 2:
            zref = zref_array[selection]
    I_sim = None
    if input_kind == "embedded":
        if zref is None or vsource is None:
            raise ValueError("Embedded inputs require zref and vsource.")
        Etheta, Ephi, I_sim = embedded_fields_to_bare(
            Etheta, Ephi, ZA, zref, vsource
        )
    elif input_kind != "bare":
        raise ValueError("input_kind must be 'embedded' or 'bare'.")
    Htheta = convert_fields_to_effective_length(
        Etheta,
        freq,
        field_kind=field_kind,
        amplitude_convention=amplitude_convention,
    )
    Hphi = convert_fields_to_effective_length(
        Ephi,
        freq,
        field_kind=field_kind,
        amplitude_convention=amplitude_convention,
    )
    meta = dict(metadata or {})
    meta.update(
        {
            "INPUT_KIND": input_kind,
            "FIELD_KIND": field_kind,
            "AMP_CONV": amplitude_convention,
        }
    )
    if I_sim is not None:
        meta["MAX_ICOND"] = float(
            np.max(np.linalg.cond(I_sim))
        )
    response = ResponseArrays(
        freq,
        theta,
        phi,
        Htheta,
        Hphi,
        ZA,
        Vsource=vsource,
        Zref=zref,
        metadata=meta,
    )
    return write_response_fits(
        output,
        response,
        dtype=dtype,
        validated=not allow_unvalidated,
    )


def main(argv=None):
    """Command-line entry point for the four-CSV response converter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--touchstone", required=True)
    parser.add_argument(
        "--vsource-npy",
        help=(
            "Full complex (frequency,4,4) solver source-voltage matrix; "
            "required for embedded inputs"
        ),
    )
    parser.add_argument("--input-kind", choices=("embedded", "bare"), required=True)
    parser.add_argument(
        "--field-kind",
        choices=("rE", "effective-length"),
        required=True,
    )
    parser.add_argument(
        "--amplitude-convention",
        choices=("rms", "peak"),
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
    freq_za, ZA, zref = read_touchstone_z(args.touchstone)
    freq_csv = read_receive_csv(args.csv[0])[0]
    if not np.allclose(freq_za, freq_csv):
        raise ValueError("Touchstone and receive CSV frequency grids differ.")
    if args.input_kind == "embedded" and args.vsource_npy is None:
        raise ValueError("Embedded conversion requires --vsource-npy.")
    vsource = (
        np.load(args.vsource_npy)
        if args.vsource_npy is not None
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
    metadata.setdefault("ZA_SOURCE", str(args.touchstone))
    return convert_receive_csvs(
        args.csv,
        args.output,
        za=ZA,
        input_kind=args.input_kind,
        field_kind=args.field_kind,
        amplitude_convention=args.amplitude_convention,
        zref=zref if args.input_kind == "embedded" else None,
        vsource=vsource,
        dtype=args.dtype,
        freq_select=args.freq_select,
        allow_unvalidated=args.allow_unvalidated,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
