"""Drop the off-diagonal terms of ZA in a four-port response FITS.

This produces a deliberately UNPHYSICAL diagnostic instrument: the four
monopoles keep their real (mutually coupled) embedded patterns H, but the
terminal network is forced to be uncoupled,

    ZA_diag[i, j] = ZA[i, i] * delta_ij ,

so the loading matrix M = ZL (ZA + ZL)^-1 becomes diagonal and no longer
mixes ports. It isolates how much of the observed cross-correlation
structure comes from the terminal network versus the field overlap in
Rsky.

Because H is untouched, Rsky is unchanged while
Rmoon = Herm(ZA_diag) - Rsky picks up off-diagonals equal to -Rsky's, and
there is no reason for it to stay positive semi-definite. The output is
therefore written with VALIDATED=False and must never be used by a
flight-like configuration; it exists only to be compared against the real
response.
"""
import argparse

import fitsio
import numpy as np


def diagonalize_za(ZA):
    """Return ZA with every off-diagonal entry set to zero."""
    ZA_diag = np.zeros_like(ZA)
    for i in range(ZA.shape[-1]):
        ZA_diag[:, i, i] = ZA[:, i, i]
    return ZA_diag


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    args = parser.parse_args(argv)

    from .common import ResponseArrays, write_response_fits

    f = fitsio.FITS(args.input, "r")
    header = dict(f[0].read_header())

    def cplx(name):
        return f[f"{name}_real"].read() + 1j * f[f"{name}_imag"].read()

    freq = f["freq"].read()
    theta = f["theta"].read()
    phi = f["phi"].read()
    H_theta = cplx("H_theta")
    H_phi = cplx("H_phi")
    ZA = cplx("ZA")
    ZLoad = cplx("ZLoad") if "zload_real" in {
        h.get_extname().lower() for h in f
    } else None
    f.close()

    ZA_diag = diagonalize_za(ZA)

    metadata = {
        key: header[key]
        for key in header
        if isinstance(header.get(key), (str, int, float, bool))
        and key not in {"CONTENT", "VALIDATED", "VERSION", "MAX_ICOND"}
        and not key.startswith(("FREQ_", "THETA_", "PHI_"))
    }
    metadata["ZADIAG"] = "off-diagonal-ZA-dropped"
    metadata["ZADIAGSR"] = str(args.input)
    metadata["SOURCE"] = (
        "ZA-diagonalized:" + str(metadata.get("SOURCE", "unknown"))
    )

    response = ResponseArrays(
        freq,
        theta,
        phi,
        H_theta,
        H_phi,
        ZA_diag,
        Rloss=None,
        ZLoad=ZLoad,
        metadata=metadata,
    )
    # validated=False: Rmoon is not guaranteed PSD once the mutual terms
    # are removed, which is itself part of the result.
    return write_response_fits(
        args.output,
        response,
        dtype=args.dtype,
        validated=False,
    )


if __name__ == "__main__":
    main()
