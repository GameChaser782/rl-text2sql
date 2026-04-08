# RL-Enhanced Text-to-SQL

Reinforcement learning extension for Text-to-SQL using GRPO inspired by SQL-R1.

## Overview

- Reward model: execution-aware SQL rewards in [`reward.py`](E:\Code\rl-text2sql\reward.py)
- Trainer: GRPO loop in [`grpo_trainer.py`](E:\Code\rl-text2sql\grpo_trainer.py)
- Entry script: `accelerate`-based training in [`train_rl.py`](E:\Code\rl-text2sql\train_rl.py)
- Kaggle runner: notebook-first workflow in [`rl-text2sql.ipynb`](E:\Code\rl-text2sql\rl-text2sql.ipynb)

## Training

Single GPU:

```bash
accelerate launch train_rl.py \
    --config config_grpo_spider.yaml \
    --train_data data/spider/train_spider.json \
    --db_root data/spider/database
```

Multi GPU:

```bash
accelerate launch --multi_gpu --num_processes 2 train_rl.py \
    --config config_grpo_spider.yaml \
    --train_data data/spider/train_spider.json \
    --db_root data/spider/database
```

Notes:

- The code now uses data parallel training across one or more GPUs instead of relying on `device_map="auto"` inside a single process.
- For Kaggle, the notebook detects GPU count and launches the same path automatically.
- QLoRA and gradient checkpointing remain enabled to fit 3B-class models on T4-class hardware.

## Evaluation

```bash
python evaluate.py \
    --model_path outputs/rl-text2sql \
    --base_model Qwen/Qwen2.5-Coder-3B-Instruct \
    --test_data data/spider/dev.json \
    --db_root data/spider/database \
    --num_gpus 1 \
    --output_file results/predictions.json
```

## Kaggle

Use [`rl-text2sql.ipynb`](E:\Code\rl-text2sql\rl-text2sql.ipynb) as the primary runner. It:

1. Sets up dependencies
2. Downloads a small Spider split
3. Detects available GPUs
4. Launches training with `accelerate`
5. Evaluates and saves outputs to `/kaggle/working/`
