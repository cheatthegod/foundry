#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export EXPERIMENT="${EXPERIMENT:-prot2text_core_bootstrap_4gpu}"

exec "${ROOT_DIR}/scripts/run_rfd3_prot2text.sh" "$@"
