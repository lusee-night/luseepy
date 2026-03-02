#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOD_SIM_DIR="${MOD_REPO}/simulation"
CONFIG_PATH="${CONFIG_PATH:-${MOD_SIM_DIR}/config/realistic_example.yaml}"
ABS_TOL32="${ABS_TOL32:-1e-2}"
ABS_TOL64="${ABS_TOL64:-1e-9}"
LMAX=8
FREQ_END=3

usage() {
  cat <<'USAGE'
Usage:
  compare_default_vs_numpy.sh [--lmax N] [--freq-end N]

Runs both precision modes:
  1) jax_enable_x64=false with ABS_TOL32 (default 1e-2)
  2) jax_enable_x64=true  with ABS_TOL64 (default 1e-9)
USAGE
}

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

STATUS32=0
STATUS64=0

echo "=== Precision case: fp32 (jax_enable_x64=false) ==="
if ! uv run python "${MOD_SIM_DIR}/driver/compare_default_vs_numpy_inproc.py" \
  --config-path "${CONFIG_PATH}" \
  --lmax "${LMAX}" \
  --freq-end "${FREQ_END}" \
  --jax-enable-x64 false \
  --abs-tol "${ABS_TOL32}"; then
  STATUS32=1
fi

echo
echo "=== Precision case: fp64 (jax_enable_x64=true) ==="
if ! uv run python "${MOD_SIM_DIR}/driver/compare_default_vs_numpy_inproc.py" \
  --config-path "${CONFIG_PATH}" \
  --lmax "${LMAX}" \
  --freq-end "${FREQ_END}" \
  --jax-enable-x64 true \
  --abs-tol "${ABS_TOL64}"; then
  STATUS64=1
fi

echo
echo "Combined summary:"
echo "  fp32 result: $([[ ${STATUS32} -eq 0 ]] && echo PASS || echo FAIL) (tol=${ABS_TOL32})"
echo "  fp64 result: $([[ ${STATUS64} -eq 0 ]] && echo PASS || echo FAIL) (tol=${ABS_TOL64})"

if [[ ${STATUS32} -ne 0 || ${STATUS64} -ne 0 ]]; then
  exit 1
fi
