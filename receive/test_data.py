#!/usr/bin/env python3

import numpy as np

from data import Spectra


def test_spectra_shapes():
    data = Spectra(source="session_001_20251105_120504.h5")

    assert isinstance(data.data, np.ndarray)
    assert data.data.size > 0, "Spectra data is empty"

    n_time = data.data.shape[0]
    assert data.time.shape[0] == n_time, "time length does not match spectra"
    assert data.FPGA_temperature.shape[0] == n_time, "FPGA_temperature length does not match spectra"


if __name__ == "__main__":
    test_spectra_shapes()
    print("OK")
