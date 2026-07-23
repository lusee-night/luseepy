"""Build a clearly marked diagonal placeholder four-port impedance."""

import argparse

import fitsio
import numpy as np

from .common import ResponseArrays, write_response_fits


def placeholder_za(freq_mhz, source_freq_mhz, source_z):
    """Interpolate one legacy impedance onto four uncoupled ports."""
    freq = np.asarray(freq_mhz, dtype=np.float64)
    source_freq = np.asarray(source_freq_mhz, dtype=np.float64)
    source_z = np.asarray(source_z)
    real = np.interp(freq, source_freq, source_z.real)
    imag = np.interp(freq, source_freq, source_z.imag)
    result = np.zeros((freq.size, 4, 4), dtype=np.complex128)
    diagonal = np.arange(4)
    result[:, diagonal, diagonal] = (real + 1j * imag)[:, None]
    return result


def main(argv=None):
    """Replace ZA in a response payload using a legacy beam impedance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_beam")
    parser.add_argument("response_npz")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    legacy = fitsio.FITS(args.legacy_beam)
    legacy_freq = legacy["freq"].read()
    legacy_z = legacy["Z_real"].read() + 1j * legacy["Z_imag"].read()
    payload = np.load(args.response_npz)
    freq = np.asarray(payload["freq"], dtype=np.float64)
    response = ResponseArrays(
        freq,
        payload["theta"],
        payload["phi"],
        payload["H_theta"],
        payload["H_phi"],
        placeholder_za(freq, legacy_freq, legacy_z),
        metadata={
            "SOURCE": str(args.response_npz),
            "ZA_SOURCE": "PLACEHOLDER",
            "VALIDATED": False,
        },
    )
    return write_response_fits(
        args.output,
        response,
        validated=False,
    )


if __name__ == "__main__":
    main()
