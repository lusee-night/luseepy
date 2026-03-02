#!/usr/bin/env bash
set -euo pipefail

# Compare fast DefaultSimulator outputs between:
# 1) this repo (modified)
# 2) clean checkout (unmodified)
#
# Requirements:
# - Run from anywhere; script resolves paths itself.
# - Uses the current repo's uv environment for BOTH runs.
# - Temporarily switches LUSEE_OUTPUT_DIR for clean run, then restores it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOD_SIM_DIR="${MOD_REPO}/simulation"
CONFIG_PATH="${CONFIG_PATH:-${MOD_SIM_DIR}/config/realistic_example.yaml}"

REQUESTED_CLEAN_REPO_DEFAULT="/Users/anigmetov/code/lusee_night/lusee_clean/luseepy"
ALT_CLEAN_REPO_DEFAULT="/Users/anigmetov/code/lusee_night/luseepy_clean/luseepy"
CLEAN_REPO="${CLEAN_REPO:-${REQUESTED_CLEAN_REPO_DEFAULT}}"
if [[ ! -d "${CLEAN_REPO}" && -d "${ALT_CLEAN_REPO_DEFAULT}" ]]; then
  CLEAN_REPO="${ALT_CLEAN_REPO_DEFAULT}"
fi

OUT_CLEAN="${OUT_CLEAN:-/Users/anigmetov/code/lusee_night/out_clean}"

if [[ ! -d "${CLEAN_REPO}" ]]; then
  echo "ERROR: CLEAN_REPO does not exist: ${CLEAN_REPO}" >&2
  echo "Set CLEAN_REPO explicitly if needed." >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH file does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -d "${OUT_CLEAN}" ]]; then
  echo "ERROR: OUT_CLEAN directory does not exist: ${OUT_CLEAN}" >&2
  exit 1
fi
if [[ -z "${LUSEE_DRIVE_DIR:-}" ]]; then
  echo "ERROR: LUSEE_DRIVE_DIR is not set." >&2
  exit 1
fi
if [[ -z "${LUSEE_OUTPUT_DIR:-}" ]]; then
  echo "ERROR: LUSEE_OUTPUT_DIR is not set." >&2
  exit 1
fi

ORIG_LUSEE_OUTPUT_DIR="${LUSEE_OUTPUT_DIR}"
restore_env() {
  export LUSEE_OUTPUT_DIR="${ORIG_LUSEE_OUTPUT_DIR}"
}
trap restore_env EXIT

MOD_OUT_NAME="sim_output_quick_mod.fits"
CLEAN_OUT_NAME="sim_output_quick_clean.fits"
MOD_OUT_PATH="${ORIG_LUSEE_OUTPUT_DIR%/}/${MOD_OUT_NAME}"
CLEAN_OUT_PATH="${OUT_CLEAN%/}/${CLEAN_OUT_NAME}"
ABS_TOL="${ABS_TOL:-0}"

# Keep run tiny: 2 timesteps, low lmax, 2 frequency bins.
FAST_OVERRIDES=(
  "engine=default"
  "observation.lunar_day='2025-02-01 13:00:00 to 2025-02-01 14:00:00'"
  "observation.dt=3600"
  "observation.lmax=8"
  "observation.freq.end=3"
)

echo "Running modified code..."
(
  cd "${MOD_SIM_DIR}"
  uv run python driver/run_sim.py "${CONFIG_PATH}" \
    "${FAST_OVERRIDES[@]}" \
    "simulation.output=${MOD_OUT_NAME}" \
    "simulation.cache_transform=quick_2step_mod.pickle"
)

echo "Running clean code with same uv environment..."
export LUSEE_OUTPUT_DIR="${OUT_CLEAN}"
(
  cd "${MOD_SIM_DIR}"
  PYTHONPATH="${CLEAN_REPO}${PYTHONPATH:+:${PYTHONPATH}}" \
  uv run python "${CLEAN_REPO}/simulation/driver/run_sim.py" \
    "${CONFIG_PATH}" \
    "${FAST_OVERRIDES[@]}" \
    "simulation.output=${CLEAN_OUT_NAME}" \
    "simulation.cache_transform=quick_2step_clean.pickle"
)
export LUSEE_OUTPUT_DIR="${ORIG_LUSEE_OUTPUT_DIR}"

if [[ ! -f "${MOD_OUT_PATH}" ]]; then
  echo "ERROR: Modified output missing: ${MOD_OUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CLEAN_OUT_PATH}" ]]; then
  echo "ERROR: Clean output missing: ${CLEAN_OUT_PATH}" >&2
  exit 1
fi

echo "Comparing FITS outputs..."
uv run python - "${MOD_OUT_PATH}" "${CLEAN_OUT_PATH}" "${ABS_TOL}" <<'PY'
import sys
import numpy as np
import fitsio

mod_path, clean_path = sys.argv[1], sys.argv[2]
abs_tol = float(sys.argv[3])

def load_by_ext(path):
    out = {}
    with fitsio.FITS(path) as f:
        for i in range(len(f)):
            name = f[i].get_extname() or f"EXT{i}"
            out[name] = f[i].read()
    return out

mod = load_by_ext(mod_path)
clean = load_by_ext(clean_path)

if set(mod) != set(clean):
    print("FAIL: Extension name mismatch")
    print("mod only:", sorted(set(mod) - set(clean)))
    print("clean only:", sorted(set(clean) - set(mod)))
    raise SystemExit(1)

ok = True
for name in sorted(mod):
    a = mod[name]
    b = clean[name]
    if a.shape != b.shape:
        print(f"FAIL [{name}]: shape mismatch {a.shape} vs {b.shape}")
        ok = False
        continue
    if np.array_equal(a, b):
        print(f"OK   [{name}]: exact match")
        continue
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        max_abs = float(np.max(np.abs(a - b)))
        if max_abs <= abs_tol:
            print(f"OK   [{name}]: within tolerance, max_abs_diff={max_abs:.6e} <= {abs_tol:.6e}")
        else:
            print(f"FAIL [{name}]: not equal, max_abs_diff={max_abs:.6e} > {abs_tol:.6e}")
            ok = False
    else:
        print(f"FAIL [{name}]: non-numeric mismatch")
        ok = False

if not ok:
    raise SystemExit(1)

print("PASS: all FITS extensions are exactly identical")
PY

echo "Done."
