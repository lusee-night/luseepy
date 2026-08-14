"""C4-symmetrize a four-port response FITS (group-average over rotations).

The LuSEE ports N, E, S, W sit at ENU azimuths 90, 0, 270, 180 degrees, so
the stored port order is cyclic (clockwise) in azimuth. Averaging over the
C4 rotation group makes each monopole the rotation-average of all four and
makes ZA exactly circulant:

  H_N^sym(phi) = 1/4 sum_k H_k(phi - 90k)          [k = 0..3 in N,E,S,W]
  H_k^sym(phi) = H_N^sym(phi + 90k)
  ZA^sym[i, j] = c_{(j-i) mod 4},  c1 = c3 = adjacent, c2 = opposite

This is the reference instrument for quantifying how much the lander (and
any other asymmetry) breaks the four-fold symmetry. The dissipative budget
stays physical: Rsky^sym <= avg of relabeled Rsky (Cauchy-Schwarz) while
Herm(ZA^sym) is the exact relabel average, so Rmoon^sym >= avg of relabeled
Rmoon >= 0.
"""
import argparse

import fitsio
import numpy as np


def _roll_phi(field, bins):
    """Roll the periodic phi axis (wraparound bin preserved)."""
    unique = field[..., :-1]
    rolled = np.roll(unique, shift=bins, axis=-1)
    return np.concatenate((rolled, rolled[..., :1]), axis=-1)


def c4_symmetrize(H_theta, H_phi, ZA, phi_step_deg=1.0):
    """Return the C4 group-averaged fields and circulant ZA."""
    bins_per_90 = int(round(90.0 / phi_step_deg))

    def symmetrize_field(field):
        # estimate of port N's pattern from every port, rotated into slot N
        n_sym = sum(
            _roll_phi(field[k], bins_per_90 * k) for k in range(4)
        ) / 4.0
        # redistribute to all four slots
        return np.stack(
            [_roll_phi(n_sym, -bins_per_90 * k) for k in range(4)]
        )

    coeff = {}
    for delta in range(4):
        coeff[delta] = np.mean(
            [ZA[:, i, (i + delta) % 4] for i in range(4)], axis=0
        )
    # reciprocity: adjacent couplings (delta 1 and 3) are one number
    adjacent = 0.5 * (coeff[1] + coeff[3])
    coeff[1] = adjacent
    coeff[3] = adjacent
    ZA_sym = np.empty_like(ZA)
    for i in range(4):
        for j in range(4):
            ZA_sym[:, i, j] = coeff[(j - i) % 4]
    return symmetrize_field(H_theta), symmetrize_field(H_phi), ZA_sym


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

    phi_step = float(phi[1] - phi[0])
    H_theta_sym, H_phi_sym, ZA_sym = c4_symmetrize(
        H_theta, H_phi, ZA, phi_step_deg=phi_step
    )

    metadata = {
        key: header[key]
        for key in header
        if isinstance(header.get(key), (str, int, float, bool))
        and key not in {"CONTENT", "VALIDATED", "VERSION", "MAX_ICOND"}
        and not key.startswith(("FREQ_", "THETA_", "PHI_"))
    }
    metadata["SYMMETRZ"] = "C4-group-average"
    metadata["SYMSRC"] = str(args.input)
    metadata["SOURCE"] = (
        "C4-symmetrized:" + str(metadata.get("SOURCE", "unknown"))
    )

    response = ResponseArrays(
        freq,
        theta,
        phi,
        H_theta_sym,
        H_phi_sym,
        ZA_sym,
        Rloss=None,
        ZLoad=ZLoad,
        metadata=metadata,
    )
    return write_response_fits(
        args.output,
        response,
        dtype=args.dtype,
        validated=True,
    )


if __name__ == "__main__":
    main()
