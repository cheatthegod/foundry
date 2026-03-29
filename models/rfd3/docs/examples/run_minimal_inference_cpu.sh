#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDRY_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/foundry312/bin/python}"
INPUTS="${INPUTS:-$SCRIPT_DIR/minimal_unconditional_cn.json}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/minimal_demo_outputs}"

if [[ -z "${CKPT_PATH:-}" ]]; then
  if [[ -f "$FOUNDRY_ROOT/../checkpoints/rfd3_latest.ckpt" ]]; then
    CKPT_PATH="$FOUNDRY_ROOT/../checkpoints/rfd3_latest.ckpt"
  else
    CKPT_PATH="rfd3"
  fi
fi

export DEBUG=False
export TYPE_CHECK=False
export NAN_CHECK=False
export PYTHONPATH="$FOUNDRY_ROOT/src:$FOUNDRY_ROOT/models/rfd3/src"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$FOUNDRY_ROOT/models/rfd3/src/rfd3/run_inference.py" \
  ckpt_path="$CKPT_PATH" \
  out_dir="$OUT_DIR" \
  inputs="$INPUTS" \
  json_keys_subset=toy_unconditional_32aa \
  prevalidate_inputs=True \
  n_batches=1 \
  diffusion_batch_size=1 \
  inference_sampler.num_timesteps=4 \
  seed=0 \
  skip_existing=False \
  cleanup_virtual_atoms=True \
  dump_prediction_metadata_json=True
