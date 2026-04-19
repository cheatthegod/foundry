#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/foundry312/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PROJECT_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/models/rfd3/src:${ROOT_DIR}/models/rf3/src:${ROOT_DIR}/models/mpnn/src${PYTHONPATH:+:${PYTHONPATH}}"
export PDB_MIRROR_PATH="${PDB_MIRROR_PATH:-${ROOT_DIR}/models/rfd3/local_data/pdb_mirror}"
export CCD_MIRROR_PATH="${CCD_MIRROR_PATH:-${ROOT_DIR}/models/rfd3/local_data/ccd_mirror}"
export HYDRA_FULL_ERROR=1

mkdir -p \
  "${ROOT_DIR}/models/rfd3/local_data/pdb_mirror" \
  "${ROOT_DIR}/models/rfd3/local_data/ccd_mirror" \
  "${ROOT_DIR}/models/rfd3/local_data/cif_cache" \
  "${ROOT_DIR}/models/rfd3/local_data/checkpoints" \
  "${ROOT_DIR}/models/rfd3/local_data/failed_examples" \
  "${ROOT_DIR}/models/rfd3/local_runs"

cd "${ROOT_DIR}"

"${PYTHON_BIN}" models/rfd3/src/rfd3/train.py experiment=local_smoke "$@"
