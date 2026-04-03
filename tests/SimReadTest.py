#!/usr/bin/env python3
"""
Smoke test: verify simulator output FITS is readable.

"""
import sys

import fitsio


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
