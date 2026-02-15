"""
Main training script for RL Text-to-SQL.
Integrates model loading, dataset preparation, and GRPO training.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from grpo_trainer import GRPOConfig, GRPOTrainer
from reward import RewardConfig, SQLRewardCalculator
from src.data.spider_dataset import SpiderDataset, collate_fn
from src.models.model_loader import load_model
from torch.utils.data import DataLoader


def main(args):
    """Main training function."""

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

    print("=" * 80)
    print("RL Text-to-SQL Training")
    print("=" * 80)
    print(f"Configuration: {config_dict}")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(config_dict.get("seed", 42))

    # Load model and tokenizer
    print("\nLoading model...")
    # For unsloth, ensure we pass the max_seq_length configuration
    model, tokenizer = load_model(
        config_dict["model_name"],
        use_qlora=config_dict.get("use_qlora", True),
        use_unsloth=config_dict.get("use_unsloth", False),
        num_gpus=config_dict.get("num_gpus", 1),
        max_seq_length=config_dict.get("max_seq_length", 2048),
    )

    # Load dataset
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

    print(f"Train dataset size: {len(train_dataset)}")

    # Initialize reward calculator
    print("\nInitializing reward calculator...")
    reward_config = RewardConfig(
        execution_weight=config_dict.get("execution_weight", 1.0),
        partial_weight=config_dict.get("partial_weight", 0.3),
        timeout_seconds=config_dict.get("timeout_seconds", 5),
        use_partial_rewards=config_dict.get("use_partial_rewards", True),
    )

    # Note: For Spider, each example has its own database
    # So we pass dummy path here and use actual path from dataset
    reward_calculator = SQLRewardCalculator(
        db_path="",  # Will be overridden per example
        config=reward_config,
    )

    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...")
    grpo_config = GRPOConfig(
        num_samples_per_prompt=config_dict.get("num_samples", 4),
        temperature=config_dict.get("temperature", 0.7),
        learning_rate=float(config_dict.get("learning_rate", 1e-5)),
        num_epochs=config_dict.get("num_epochs", 3),
        batch_size=config_dict.get("batch_size", 1),
        gradient_accumulation_steps=config_dict.get("gradient_accumulation_steps", 8),
        kl_coef=config_dict.get("kl_coef", 0.1),
        max_grad_norm=config_dict.get("max_grad_norm", 1.0),
        max_new_tokens=config_dict.get("max_new_tokens", 512),
        num_gpus=config_dict.get(
            "num_gpus", 1
        ),  # Pass num_gpus to config if needed, or directly to Trainer
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        config=grpo_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_gpus=config_dict.get("num_gpus", 1),
        use_unsloth=config_dict.get("use_unsloth", False),
    )

    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.train(train_dataloader)

    # Save model
    if config_dict.get("output_dir"):
        print(f"\nSaving model to {config_dict['output_dir']}...")
        os.makedirs(config_dict["output_dir"], exist_ok=True)

        # Save LoRA adapters if using PEFT
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(config_dict["output_dir"])

        tokenizer.save_pretrained(config_dict["output_dir"])

        # Save config
        with open(
            os.path.join(config_dict["output_dir"], "training_config.yaml"), "w"
        ) as f:
            yaml.dump(config_dict, f)

        print("Model saved successfully!")

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
