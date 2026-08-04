"""Convert the LuSEE_BGL_V16 HFSS receive-matrix exports to response FITS v3.

The ``Receive_Matrix_Fields_{N,E,S,W}.csv`` exports produced by
``resources/HFSS/ReceiveMatrix.ipynb`` are *loaded* receive fields

    R = ZL (ZA + ZL)^-1 H            [meters]

where ``ZA`` is the bare antenna impedance from the stitched ``.s4p``
S-parameter sweep (exported as ``Complex_Z_Matrix.csv``, Zref = 50 Ohm) and
``ZL`` is the scalar-diagonal average of the four measured spare preamps
(fmpre0/2/5/7). This driver rebuilds that exact ``ZL``, unloads the fields
back to the bare effective-length matrix ``H``, rolls the azimuth so the
stored phi=0 is local East (HFSS +x points West, so 180 degrees), and writes
one validated instrument response file.

Solver coordinate provenance (Kaja R., 2026-08): LuSEE_BGL_V16 exports use
theta 0..180 (colatitude from +z, zenith), phi 0..360 from +x toward +y,
west antenna along +x, south antenna along +y, rE in mV peak with
``e^{+j omega t}`` phasors, ports ordered N, E, S, W. The metal/regolith
loss split is not available from this export, so it is declared PEC:
all sub-horizon and ohmic dissipation is attributed to Rmoon at T_moon.

Future *bare* exports (``Bare_Effective_Length_Fields_{pole}.csv`` from
the notebook's ``export_fields_to_csv``, columns ``re/im(h_Theta)`` and
``re/im(h_Phi)``) need no unloading; convert them with the generic CLI::

    python -m beam_conversion.receive_csv Bare_..._{N,E,S,W}.csv \
        --zmatrix-csv Complex_Z_Matrix.csv --input-kind bare \
        --field-kind effective-length --field-units m \
        --field-amplitude ratio --pec --phi-source-zero-deg 180 ...

and the result should match this driver's output H to numerical precision.
"""

import argparse
import subprocess
from pathlib import Path

from .receive_csv import convert_receive_csvs, read_zmatrix_csv


PORT_ORDER = ("N", "E", "S", "W")


def build_provenance(root, za_source):
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
    return {
        "SOURCE": ",".join(
            f"Receive_Matrix_Fields_{port}.csv" for port in PORT_ORDER
        ),
        "SOURCE_ROOT": str(root),
        "ZA_SOURCE": str(za_source),
        "GIT_SHA": git_sha,
        "TIMECONV": "e+jwt",
        "COORDSYS": "instrument-topocentric",
        "THETADEF": "colatitude-from-+z",
        "PHIDEF": "right-handed-about-+z",
        "OMEGADEF": "source-arrival-direction",
        "POLBASIS": "e_theta,e_phi",
        "PHASEREF": "solver-origin",
        "PORTS": "0123",
        "PORTNAMES": "N,E,S,W",
        "LOSSMODEL": "PEC",
        "RLOSSSRC": "not-separated;lossy/PEC-pair-unavailable",
        "SIMULATION": "LuSEE_BGL_V16_terminal run 02",
        "ZL_MODEL": "spare-preamp-average;fmpre0/2/5/7;4-term-fit",
        "SRCXAXIS": "west",
        "SRCYAXIS": "south",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receive-dir",
        required=True,
        help="Directory containing Receive_Matrix_Fields_{N,E,S,W}.csv",
    )
    parser.add_argument(
        "--zmatrix-csv",
        required=True,
        help="Complex_Z_Matrix.csv exported from the stitched s4p files",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--freq-select",
        nargs="+",
        type=float,
        help="Optional subset of native frequencies in MHz",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--allow-unvalidated", action="store_true")
    args = parser.parse_args(argv)

    from lusee.ReceiverImpedance import spare_preamp_average_zload

    root = Path(args.receive_dir)
    csv_paths = [
        root / f"Receive_Matrix_Fields_{port}.csv" for port in PORT_ORDER
    ]
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(str(path))
    freq_mhz, ZA = read_zmatrix_csv(args.zmatrix_csv)
    zload = spare_preamp_average_zload(freq_mhz)
    return convert_receive_csvs(
        [str(path) for path in csv_paths],
        args.output,
        za=ZA,
        input_kind="loaded",
        field_kind="effective-length",
        field_units="m",
        field_amplitude_convention="ratio",
        rloss=None,
        zload=zload,
        phi_source_zero_deg=180.0,
        dtype=args.dtype,
        freq_select=args.freq_select,
        metadata=build_provenance(root, args.zmatrix_csv),
        allow_unvalidated=args.allow_unvalidated,
    )


if __name__ == "__main__":
    main()
