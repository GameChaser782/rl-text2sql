"""
Main training script for RL Text-to-SQL.
Integrates model loading, dataset preparation, and GRPO training.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import yaml
from accelerate import Accelerator
from grpo_trainer import GRPOConfig, GRPOTrainer
from reward import RewardConfig, SQLRewardCalculator
from src.data.spider_dataset import SpiderDataset, collate_fn
from src.models.model_loader import load_model
from torch.utils.data import DataLoader


def main(args):
    """Main training function."""
    accelerator = Accelerator()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != "config":
            config_dict[key] = value

    # If running on Kaggle, default the output directory to a persistent path
    # so that repo clones/updates won't overwrite model artifacts.
    try:
        is_kaggle = any(
            p
            for p in ["/kaggle", os.environ.get("KAGGLE_URL_BASE")]
            if p and os.path.exists(p)
        )
    except Exception:
        is_kaggle = False

    if not config_dict.get("output_dir"):
        if is_kaggle:
            config_dict["output_dir"] = "/kaggle/working/rl-text2sql-outputs"
        else:
            config_dict["output_dir"] = "outputs/rl-text2sql"

    world_size = accelerator.num_processes
    config_dict["num_gpus"] = max(
        int(config_dict.get("num_gpus", world_size if torch.cuda.is_available() else 1)),
        world_size,
    )

    if accelerator.is_main_process:
        print("=" * 80)
        print("RL Text-to-SQL Training")
        print("=" * 80)
        print(f"Configuration: {config_dict}")
        print(f"Accelerate processes: {world_size}")
        print("=" * 80)

    # Set random seeds
    torch.manual_seed(config_dict.get("seed", 42))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config_dict.get("seed", 42))

    local_device_map = None
    if torch.cuda.is_available() and not config_dict.get("use_unsloth", False):
        local_device_map = {"": accelerator.local_process_index}

    # Load model and tokenizer
    if accelerator.is_main_process:
        print("\nLoading model...")
    model, tokenizer = load_model(
        config_dict["model_name"],
        use_qlora=config_dict.get("use_qlora", True),
        use_unsloth=config_dict.get("use_unsloth", False),
        num_gpus=config_dict.get("num_gpus", 1),
        max_seq_length=config_dict.get("max_seq_length", 2048),
        device_map=local_device_map,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        ),
    )
    try:
        reference_model = copy.deepcopy(model)
    except RuntimeError as e:
        raise RuntimeError(
            f"Unable to create reference model copy required for GRPO KL regularization: {e}"
        ) from e

    # Load dataset
    if accelerator.is_main_process:
        print("\nLoading dataset...")
    train_dataset = SpiderDataset(
        data_path=config_dict["train_data"], db_root=config_dict["db_root"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_dict.get("batch_size", 1),
        shuffle=True,
        collate_fn=collate_fn,
    )

    if accelerator.is_main_process:
        print(f"Train dataset size: {len(train_dataset)}")

    # Initialize reward calculator
    if accelerator.is_main_process:
        print("\nInitializing reward calculator...")
    reward_config = RewardConfig(
        reward_mode=config_dict.get("reward_mode", "execution_partial"),
        execution_weight=config_dict.get("execution_weight", 1.0),
        partial_weight=config_dict.get("partial_weight", 0.3),
        timeout_seconds=config_dict.get("timeout_seconds", 5),
        use_partial_rewards=config_dict.get("use_partial_rewards", True),
        format_reward=config_dict.get("format_reward", 1.0),
        executable_reward=config_dict.get("executable_reward", 2.0),
        result_reward=config_dict.get("result_reward", 3.0),
    )

    # Note: For Spider, each example has its own database
    # So we pass dummy path here and use actual path from dataset
    reward_calculator = SQLRewardCalculator(
        db_path="",  # Will be overridden per example
        config=reward_config,
    )

    # Initialize GRPO trainer
    if accelerator.is_main_process:
        print("\nInitializing GRPO trainer...")
    grpo_config = GRPOConfig(
        num_samples_per_prompt=config_dict.get("num_samples", 4),
        temperature=config_dict.get("temperature", 0.7),
        top_p=config_dict.get("top_p", 0.9),
        learning_rate=float(config_dict.get("learning_rate", 1e-5)),
        num_epochs=config_dict.get("num_epochs", 3),
        batch_size=config_dict.get("batch_size", 1),
        gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", 8),
        kl_coef=config_dict.get("kl_coef", 0.1),
        clip_range=config_dict.get("clip_range", 0.2),
        advantage_normalization=config_dict.get("advantage_normalization", True),
        max_grad_norm=config_dict.get("max_grad_norm", 1.0),
        max_new_tokens=config_dict.get("max_new_tokens", 512),
        num_gpus=config_dict.get(
            "num_gpus", 1
        ),  # Pass num_gpus to config if needed, or directly to Trainer
    )

    model, train_dataloader = accelerator.prepare(model, train_dataloader)

    trainer = GRPOTrainer(
        model=model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        config=grpo_config,
        accelerator=accelerator,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_gpus=config_dict.get("num_gpus", 1),
        use_unsloth=config_dict.get("use_unsloth", False),
    )

    # Train
    if accelerator.is_main_process:
        print("\nStarting training...")
        print("=" * 80)
    trainer.train(train_dataloader)
    accelerator.wait_for_everyone()

    # Save model
    if config_dict.get("output_dir") and accelerator.is_main_process:
        print(f"\nSaving model to {config_dict['output_dir']}...")
        os.makedirs(config_dict["output_dir"], exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)

        # Save LoRA adapters if using PEFT
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(config_dict["output_dir"])

        tokenizer.save_pretrained(config_dict["output_dir"])

        # Save config
        with open(
            os.path.join(config_dict["output_dir"], "training_config.yaml"), "w"
        ) as f:
            yaml.dump(config_dict, f)

        print("Model saved successfully!")

    if accelerator.is_main_process:
        print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text-to-SQL model with RL")

    # Configuration
    parser.add_argument("--config", type=str, help="Path to config YAML file")

    # Model
    parser.add_argument(
        "--model_name", type=str, default=None, help="HuggingFace model name"
    )
    parser.add_argument(
        "--use_qlora", action="store_true", default=None, help="Use QLoRA quantization"
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        default=None,
        help="Use Unsloth optimization",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use (1 or 2)"
    )

    # Data
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data JSON"
    )
    parser.add_argument(
        "--db_root", type=str, required=True, help="Root directory of databases"
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per prompt for GRPO",
    )
    parser.add_argument("--kl_coef", type=float, default=None)

    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save model"
    )
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    main(args)
