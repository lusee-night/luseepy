#!/usr/bin/env bash
set -euo pipefail

# Compare outputs between engines in THIS repo only:
# 1) engine=default (jaxified DefaultSimulator)
# 2) engine=numpy   (legacy NumPy simulator)
#
# Requirements:
# - Uses current repo's uv environment.
# - Requires LUSEE_DRIVE_DIR and LUSEE_OUTPUT_DIR.
# - Optional CLI args:
#   --lmax <int>
#   --freq-end <int>

usage() {
  cat <<'USAGE'
Usage:
  compare_default_vs_numpy.sh [--lmax N] [--freq-end N]

Defaults:
  --lmax 8
  --freq-end 3
USAGE
}

LMAX=8
FREQ_END=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lmax)
      LMAX="${2:-}"
      shift 2
      ;;
    --freq-end)
      FREQ_END="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOD_SIM_DIR="${MOD_REPO}/simulation"
CONFIG_PATH="${CONFIG_PATH:-${MOD_SIM_DIR}/config/realistic_example.yaml}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: CONFIG_PATH file does not exist: ${CONFIG_PATH}" >&2
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

OUT_DIR="${LUSEE_OUTPUT_DIR%/}"
JAX_OUT_NAME="sim_output_quick_jax.fits"
NUMPY_OUT_NAME="sim_output_quick_numpy.fits"
JAX_OUT_PATH="${OUT_DIR}/${JAX_OUT_NAME}"
NUMPY_OUT_PATH="${OUT_DIR}/${NUMPY_OUT_NAME}"
ABS_TOL="${ABS_TOL:-1e-9}"
JAX_TIME_FILE="$(mktemp)"
NUMPY_TIME_FILE="$(mktemp)"
cleanup() {
  rm -f "${JAX_TIME_FILE}" "${NUMPY_TIME_FILE}"
}
trap cleanup EXIT

# Keep run tiny: ~3 snapshots from 1 hour at 1-hour dt, low lmax, 2 frequency bins.
FAST_OVERRIDES=(
  "observation.lunar_day='2025-02-01 13:00:00 to 2025-02-01 14:00:00'"
  "observation.dt=3600"
  "observation.lmax=${LMAX}"
  "observation.freq.end=${FREQ_END}"
)

echo "Running default (jax) engine..."
(
  cd "${MOD_SIM_DIR}"
  /usr/bin/time -p -o "${JAX_TIME_FILE}" uv run python driver/run_sim.py "${CONFIG_PATH}" \
    "${FAST_OVERRIDES[@]}" \
    "engine=default" \
    "simulation.output=${JAX_OUT_NAME}" \
    "simulation.cache_transform=quick_2step_jax.pickle"
)

echo "Running numpy engine..."
(
  cd "${MOD_SIM_DIR}"
  /usr/bin/time -p -o "${NUMPY_TIME_FILE}" uv run python driver/run_sim.py "${CONFIG_PATH}" \
    "${FAST_OVERRIDES[@]}" \
    "engine=numpy" \
    "simulation.output=${NUMPY_OUT_NAME}" \
    "simulation.cache_transform=quick_2step_numpy.pickle"
)

if [[ ! -f "${JAX_OUT_PATH}" ]]; then
  echo "ERROR: JAX/default output missing: ${JAX_OUT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${NUMPY_OUT_PATH}" ]]; then
  echo "ERROR: NumPy output missing: ${NUMPY_OUT_PATH}" >&2
  exit 1
fi

N_FREQ_BINS="$(uv run python - "${JAX_OUT_PATH}" <<'PY'
import sys
import numpy as np
import fitsio

path = sys.argv[1]
with fitsio.FITS(path) as f:
    freq = f["freq"].read()
print(int(np.asarray(freq).size))
PY
)"

echo "Comparing FITS outputs..."
set +e
uv run python - "${JAX_OUT_PATH}" "${NUMPY_OUT_PATH}" "${ABS_TOL}" <<'PY'
import sys
import numpy as np
import fitsio

jax_path, numpy_path = sys.argv[1], sys.argv[2]
abs_tol = float(sys.argv[3])

def load_by_ext(path):
    out = {}
    with fitsio.FITS(path) as f:
        for i in range(len(f)):
            name = f[i].get_extname() or f"EXT{i}"
            out[name] = f[i].read()
    return out

jax_data = load_by_ext(jax_path)
numpy_data = load_by_ext(numpy_path)

if set(jax_data) != set(numpy_data):
    print("FAIL: Extension name mismatch")
    print("jax only:", sorted(set(jax_data) - set(numpy_data)))
    print("numpy only:", sorted(set(numpy_data) - set(jax_data)))
    raise SystemExit(1)

ok = True
for name in sorted(jax_data):
    a = jax_data[name]
    b = numpy_data[name]
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

print("PASS: all FITS extensions are exactly identical within tolerance")
PY
COMPARE_EXIT=$?
set -e

JAX_TIME_SEC="$(awk '/^real / {print $2}' "${JAX_TIME_FILE}")"
NUMPY_TIME_SEC="$(awk '/^real / {print $2}' "${NUMPY_TIME_FILE}")"

if [[ "${COMPARE_EXIT}" -eq 0 ]]; then
  RESULT="PASS"
else
  RESULT="FAIL"
fi

echo
echo "Summary:"
echo "  lmax: ${LMAX}"
echo "  freq bins: ${N_FREQ_BINS}"
echo "  jax real time (s): ${JAX_TIME_SEC:-N/A}"
echo "  numpy real time (s): ${NUMPY_TIME_SEC:-N/A}"
echo "  result: ${RESULT}"

if [[ "${COMPARE_EXIT}" -ne 0 ]]; then
  exit "${COMPARE_EXIT}"
fi

echo "Done."
