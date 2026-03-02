#!/usr/bin/env python3
import sys

from sim_driver import SimDriver
from run_sim_universal import _apply_legacy_config_shim, run


if __name__ == "__main__":
    sys.argv = _apply_legacy_config_shim(sys.argv)
    run()
