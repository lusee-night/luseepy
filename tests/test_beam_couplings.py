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


def test_two_port_grid_mismatch_raises(drive_dir):
    import lusee

    broot = os.path.join(drive_dir, BEAM_DIR)
    b_n = lusee.Beam(os.path.join(broot, "hfss_lbl_2m_45deg.fits"), id="N")
    b_s = lusee.Beam(os.path.join(broot, "hfss_lbl_2m_45deg.fits"), id="S")
    couplings = {
        "opposite": {
            "combinations": [["N", "S"]],
            "two_port": os.path.join(broot, "hfss_lbl_2m_45deg.2port.fits"),
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
