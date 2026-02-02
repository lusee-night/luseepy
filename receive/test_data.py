#!/usr/bin/env python3

import numpy as np

from data import Spectra
from matplotlib import pyplot as plt


def test_spectra_shapes():
    data = Spectra(source="session_001_20251105_120504.h5")

    assert isinstance(data.data, np.ndarray)
    assert data.data.size > 0, "Spectra data is empty"

    n_time = data.data.shape[0]
    assert data.time.shape[0] == n_time, "time length does not match spectra"
    assert data.FPGA_temperature.shape[0] == n_time, "FPGA_temperature length does not match spectra"

    assert data.lun_height_m >= 0
    assert data.lun_long_deg is not None
    assert data.lun_lat_deg is not None

def test_plots():
    data = Spectra(source="session_001_20251105_120504.h5")
    plt.plot(data.time, data.FPGA_temperature)
    plt.show()



if __name__ == "__main__":
    test_spectra_shapes()
    # test_plots()
    print("OK")
