#!/usr/bin/env python3
"""
Smoke test: verify simulator output FITS is readable.

"""
import sys

import fitsio
import numpy as np


def _comb2ndx(combinations):
    """Match lusee.Data comb2ndx layout for slice tests."""
    comb2ndx = {}
    cc = 0
    for row in combinations:
        i, j = int(row[0]), int(row[1])
        comb2ndx[(i, j)] = cc
        if i == j:
            cc += 1
        else:
            cc += 2
    return comb2ndx


def _check_four_port(fits, hdr):
    """Smoke-check the four-port covariance layout of write_fits."""
    for key in ("ENGINE", "RESPHASH", "DELTAT_SEC"):
        if key not in hdr:
            raise RuntimeError(f"Missing four-port FITS header key {key!r}")
    data = fits["data"].read()
    freq = fits["freq"].read()
    times = fits["time"].read()
    if data.ndim != 3 or data.shape[1] != 16:
        raise RuntimeError(
            f"Expected packed covariance (time, 16, freq); got {data.shape}"
        )
    if data.shape[0] != times.shape[0]:
        raise RuntimeError(
            f"data time axis {data.shape[0]} != len(times)={times.shape[0]}"
        )
    if data.shape[2] != len(freq):
        raise RuntimeError(
            f"data freq axis {data.shape[2]} != len(freq)={len(freq)}"
        )
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Covariance data contains non-finite values.")
    eigenvalues = fits["covariance_eigenvalues"].read()
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("Covariance eigenvalues contain non-finite values.")
    print("OK (four-port covariance).")


def main():
    if len(sys.argv) <= 1:
        print("Specify filename on command line.")
        sys.exit(1)
    fname = sys.argv[1]
    print(f"Attempting to read {fname}...")
    hdr = fitsio.read_header(fname)
    for key in ("VERSION", "LUNAR_DAY", "LUN_LAT_DEG", "LUN_LONG_DEG"):
        if key not in hdr:
            raise RuntimeError(f"Missing FITS header key {key!r}")

    with fitsio.FITS(fname, "r") as fits:
        extnames = {hdu.get_extname() for hdu in fits}
        if "combinations" not in extnames:
            _check_four_port(fits, hdr)
            return
        data = fits["data"].read()
        freq = fits["freq"].read()
        combinations = fits["combinations"].read()

    if data.ndim != 3:
        raise RuntimeError(f"Expected data.ndim==3, got {data.ndim} shape={data.shape}")
    if data.shape[2] != len(freq):
        raise RuntimeError(
            f"data freq axis {data.shape[2]} != len(freq)={len(freq)}"
        )

    comb2ndx = _comb2ndx(combinations)
    # Same slice as lusee.Data[:, "12C", :] for cross (1,2) complex
    ndx = comb2ndx[(1, 2)]
    subdata = data[:, ndx, :] + 1j * data[:, ndx + 1, :]
    assert subdata.shape[1] == len(freq)
    print("OK.")


if __name__ == "__main__":
    main()
