#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config_grpo_spider.yaml}"
NUM_GPUS="${NUM_GPUS:-1}"
ADAPTER_PATH="${ADAPTER_PATH:-/kaggle/working/rl-text2sql-sft}"

ACCELERATE_CMD=(accelerate launch --num_machines 1 --mixed_precision fp16 --dynamo_backend no)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCELERATE_CMD+=(--multi_gpu --num_processes "${NUM_GPUS}")
fi

EXTRA_ARGS=()
if [ -d "${ADAPTER_PATH}" ]; then
  EXTRA_ARGS+=(--adapter_path "${ADAPTER_PATH}")
fi

"${ACCELERATE_CMD[@]}" train_rl.py \
  --config "${CONFIG_PATH}" \
  "${EXTRA_ARGS[@]}"
