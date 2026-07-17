"""Load-time grid validation for BeamCouplings two-port files.

cross_power mixes gain_conv arrays from the coupling beams and the
separately loaded two-port beam elementwise, which silently assumed a
shared native frequency grid once forced canonicalization at load was
removed; BeamCouplings must reject mismatched grids loudly.
"""

import os

os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import pytest


BEAM_DIR = "Simulations/BeamModels/LanderRegolithComparison/eight_layer_regolith"
ONE_PORT = "hfss_lbl_3m_75deg.fits"
TWO_PORT = "hfss_lbl_3m_75deg.2port.fits"


def test_two_port_grid_check(drive_dir):
    """Grid-check logic, runnable on the minimal CI tarball.

    The one-port beam stands in as its own two-port file: the check only
    needs a loadable FITS beam with .freq and .gain_conv.
    """
    import lusee

    one_port = os.path.join(drive_dir, BEAM_DIR, ONE_PORT)
    if not os.path.isfile(one_port):
        pytest.skip("beam file not present in this Drive checkout")

    b_n = lusee.Beam(one_port, id="N")
    b_s = lusee.Beam(one_port, id="S")
    couplings = {
        "standin": {
            "combinations": [["N", "S"]],
            "two_port": one_port,
            "sign": -1,
        }
    }

    # matching grids load fine
    bc = lusee.BeamCouplings([b_n, b_s], from_yaml_dict=couplings)
    assert ("N", "S") in bc.cross_powers

    # a beam on a shifted grid must be rejected loudly
    b_s.freq = np.asarray(b_s.freq, dtype=float) + 0.1
    with pytest.raises(ValueError, match=r"different native frequency grid"):
        lusee.BeamCouplings([b_n, b_s], from_yaml_dict=couplings)


def test_two_port_real_pair_loads(drive_dir):
    """Same load path against a genuine two-port measurement file.

    The CI Drive tarball ships no .2port.fits files, so this runs only
    on a full Drive checkout.
    """
    import lusee

    one_port = os.path.join(drive_dir, BEAM_DIR, ONE_PORT)
    two_port = os.path.join(drive_dir, BEAM_DIR, TWO_PORT)
    if not (os.path.isfile(one_port) and os.path.isfile(two_port)):
        pytest.skip("two-port beam pair not present in this Drive checkout")

    b_n = lusee.Beam(one_port, id="N")
    b_s = lusee.Beam(one_port, id="S")
    couplings = {
        "opposite": {
            "combinations": [["N", "S"]],
            "two_port": two_port,
            "sign": -1,
        }
    }
    bc = lusee.BeamCouplings([b_n, b_s], from_yaml_dict=couplings)
    cp = np.asarray(bc.cross_powers[("N", "S")])
    assert cp.shape == np.asarray(b_n.freq).shape
    assert np.all(np.isfinite(cp))
