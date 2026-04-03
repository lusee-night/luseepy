#!/usr/bin/env python3
# Avoid OpenMP duplicate-library crash on macOS when numpy/scipy/pyshtools link different runtimes
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hydra
from omegaconf import DictConfig, OmegaConf

from sim_driver import SimDriver

_CONFIG_PATH = str((Path(__file__).resolve().parent.parent / "config").resolve())


def _strip_yaml_ext(name):
    if name.endswith(".yaml"):
        return name[:-5]
    if name.endswith(".yml"):
        return name[:-4]
    return name


def _normalize_config_name_argv(argv):
    out = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--config-name="):
            out.append("--config-name=" + _strip_yaml_ext(a.split("=", 1)[1]))
        elif a in {"--config-name", "-cn"} and i + 1 < len(argv):
            out.extend([a, _strip_yaml_ext(argv[i + 1])])
            i += 1
        else:
            out.append(a)
        i += 1
    return out


def _apply_legacy_config_shim(argv):
    argv = _normalize_config_name_argv(argv)
    has_hydra_cfg = any(
        a in {"--config-name", "-cn", "--config-path", "-cp"}
        or a.startswith("--config-name=")
        or a.startswith("--config-path=")
        for a in argv[1:]
    )
    if has_hydra_cfg:
        return argv
    for i, a in enumerate(argv[1:], start=1):
        if a.endswith((".yaml", ".yml")):
            cfg_dir, cfg_file = os.path.split(a)
            cfg_dir = os.path.abspath(cfg_dir or ".")
            cfg_name = os.path.splitext(cfg_file)[0]
            return [
                argv[0],
                f"--config-path={cfg_dir}",
                f"--config-name={cfg_name}",
                *argv[1:i],
                *argv[i + 1 :],
            ]
    return argv


@hydra.main(version_base=None, config_path=_CONFIG_PATH, config_name="realistic_example")
def run(cfg: DictConfig):
    resolved = OmegaConf.to_container(cfg, resolve=True)
    S = SimDriver(resolved)
    S.run()


if __name__ == "__main__":
    sys.argv = _apply_legacy_config_shim(sys.argv)
    run()
