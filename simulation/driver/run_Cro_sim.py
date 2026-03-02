#!/usr/bin/env python3
import sys

from run_sim_universal import _apply_legacy_config_shim, run


def _has_engine_override(argv):
    return any(
        a.startswith("engine=") or a.startswith("simulation.engine=")
        for a in argv[1:]
    )


if __name__ == "__main__":
    sys.argv = _apply_legacy_config_shim(sys.argv)
    if not _has_engine_override(sys.argv):
        sys.argv.append("engine=croissant")
    run()
