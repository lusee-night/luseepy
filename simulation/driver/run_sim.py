#!/usr/bin/env python3
"""Run a simulation from YAML via :class:`sim_driver.SimDriver`.

Equivalent pipeline to ``run_Cro_sim.py`` (Croissant vs default is chosen by
``simulation.engine`` in the config). Uses Hydra for defaults; you can pass a
config path in the legacy style used in CI and docs.

Examples::

    cd simulation
    python driver/run_sim.py config/example.yaml
    python driver/run_sim.py --config-name=example
    python driver/run_sim.py --config-name=sim_choice_realistic

See ``simulation/config/example.yaml`` (starter) and
``simulation/config/sim_choice_realistic.yaml`` (full Croissant + plotting).
"""
import sys

from sim_driver import SimDriver
from run_sim_universal import _apply_legacy_config_shim, run

__all__ = ["SimDriver", "run", "_apply_legacy_config_shim"]


if __name__ == "__main__":
    sys.argv = _apply_legacy_config_shim(sys.argv)
    run()
