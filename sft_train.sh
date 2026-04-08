#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-Coder-3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-/kaggle/working/spider_data/sft_train_400.json}"
DB_ROOT="${DB_ROOT:-/kaggle/working/spider_data/database}"
OUTPUT_DIR="${OUTPUT_DIR:-/kaggle/working/rl-text2sql-sft}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
LOGGING_DIR="${LOGGING_DIR:-${OUTPUT_DIR}/logs}"

ACCELERATE_CMD=(accelerate launch --num_machines 1 --mixed_precision fp16 --dynamo_backend no)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCELERATE_CMD+=(--multi_gpu --num_processes "${NUM_GPUS}")
fi

"${ACCELERATE_CMD[@]}" sft_train.py \
  --model_name "${MODEL_NAME}" \
  --train_data "${TRAIN_DATA}" \
  --db_root "${DB_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LR}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --logging_dir "${LOGGING_DIR}" \
  --use_qlora
