"""
Main training script for RL Text-to-SQL.
Integrates model loading, dataset preparation, and GRPO training.
"""

import os
import torch
import argparse
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
import json

from reward import SQLRewardCalculator, RewardConfig
from grpo_trainer import GRPOTrainer, GRPOConfig


class SpiderDataset(Dataset):
    """Spider dataset for Text-to-SQL."""
    
    def __init__(self, data_path: str, db_root: str):
        """
        Initialize Spider dataset.
        
        Args:
            data_path: Path to spider data JSON file
            db_root: Root directory containing database folders
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.db_root = db_root
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'question': item['question'],
            'sql': item['query'],
            'db_id': item['db_id'],
            'db_path': os.path.join(self.db_root, item['db_id'], f"{item['db_id']}.sqlite"),
            'schema': item.get('schema', None)  # Optional
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'question': [item['question'] for item in batch],
        'sql': [item['sql'] for item in batch],
        'db_path': [item['db_path'] for item in batch],
        'schema': [item.get('schema') for item in batch]
    }


def load_model_and_tokenizer(model_name: str, use_qlora: bool = True):
    """
    Load model with QLoRA configuration.
    
    Args:
        model_name: HuggingFace model name
        use_qlora: Whether to use QLoRA quantization
        
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",  # CHANGE: from {"": 0} to "auto"
            trust_remote_code=True
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        print(f"LoRA trainable parameters: {model.print_trainable_parameters()}")
    
    else:
        # Load model normally (requires more memory)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    return model, tokenizer


def main(args):
    """Main training function."""
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    
    # Override with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config_dict[key] = value
    
    print("=" * 80)
    print("RL Text-to-SQL Training")
    print("=" * 80)
    print(f"Configuration: {config_dict}")
    print("=" * 80)
    
    # Set random seeds
    torch.manual_seed(config_dict.get('seed', 42))
    
    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        config_dict['model_name'],
        use_qlora=config_dict.get('use_qlora', True)
    )
    
    # Load dataset
    print("\nLoading dataset...")
    train_dataset = SpiderDataset(
        data_path=config_dict['train_data'],
        db_root=config_dict['db_root']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config_dict.get('batch_size', 1),
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Initialize reward calculator
    print("\nInitializing reward calculator...")
    reward_config = RewardConfig(
        execution_weight=config_dict.get('execution_weight', 1.0),
        partial_weight=config_dict.get('partial_weight', 0.3),
        timeout_seconds=config_dict.get('timeout_seconds', 5),
        use_partial_rewards=config_dict.get('use_partial_rewards', True)
    )
    
    # Note: For Spider, each example has its own database
    # So we pass dummy path here and use actual path from dataset
    reward_calculator = SQLRewardCalculator(
        db_path="",  # Will be overridden per example
        config=reward_config
    )
    
    # Initialize GRPO trainer
    print("\nInitializing GRPO trainer...")
    grpo_config = GRPOConfig(
        num_samples_per_prompt=config_dict.get('num_samples', 4),
        temperature=config_dict.get('temperature', 0.7),
        learning_rate=config_dict.get('learning_rate', 1e-5),
        num_epochs=config_dict.get('num_epochs', 3),
        batch_size=config_dict.get('batch_size', 1),
        gradient_accumulation_steps=config_dict.get('gradient_accumulation_steps', 8),
        kl_coef=config_dict.get('kl_coef', 0.1),
        max_grad_norm=config_dict.get('max_grad_norm', 1.0)
    )
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        config=grpo_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.train(train_dataloader)
    
    # Save model
    if config_dict.get('output_dir'):
        print(f"\nSaving model to {config_dict['output_dir']}...")
        os.makedirs(config_dict['output_dir'], exist_ok=True)
        
        # Save LoRA adapters if using PEFT
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(config_dict['output_dir'])
        
        tokenizer.save_pretrained(config_dict['output_dir'])
        
        # Save config
        with open(os.path.join(config_dict['output_dir'], 'training_config.yaml'), 'w') as f:
            yaml.dump(config_dict, f)
        
        print("Model saved successfully!")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text-to-SQL model with RL")
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Model
    parser.add_argument('--model_name', type=str, default=None,  # Changed from 'codellama/CodeLlama-7b-hf'
                       help='HuggingFace model name')
    parser.add_argument('--use_qlora', action='store_true', default=True,
                       help='Use QLoRA quantization')
    
    # Data
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data JSON')
    parser.add_argument('--db_root', type=str, required=True,
                       help='Root directory of databases')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_samples', type=int, default=4,
                       help='Number of samples per prompt for GRPO')
    parser.add_argument('--kl_coef', type=float, default=0.1)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/rl-text2sql',
                       help='Directory to save model')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)
