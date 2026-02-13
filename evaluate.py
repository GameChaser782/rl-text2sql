"""
Evaluation script for Text-to-SQL models.
Computes execution accuracy and exact match metrics.
"""

import torch
import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from reward import SQLRewardCalculator, RewardConfig


class Text2SQLEvaluator:
    """Evaluator for Text-to-SQL models."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_calculator: SQLRewardCalculator,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.device = device
    
    def create_prompt(self, question: str, schema: str = None) -> str:
        """Create prompt for generation."""
        if schema:
            prompt = f"""Given the database schema:
{schema}

Question: {question}

Generate a SQL query to answer this question:
SQL:"""
        else:
            prompt = f"Question: {question}\nSQL:"
        
        return prompt
    
    @torch.no_grad()
    def generate_sql(self, question: str, schema: str = None) -> str:
        """Generate SQL query for a question."""
        prompt = self.create_prompt(question, schema)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # Low temperature for evaluation
            do_sample=False,  # Greedy decoding
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL
        sql = self.extract_sql(generated_text)
        
        return sql
    
    def extract_sql(self, text: str) -> str:
        """Extract SQL from generated text."""
        text = text.strip()
        
        # Find SELECT
        select_idx = text.upper().find('SELECT')
        if select_idx != -1:
            sql = text[select_idx:]
            # Remove everything after semicolon
            semicolon_idx = sql.find(';')
            if semicolon_idx != -1:
                sql = sql[:semicolon_idx]
            return sql.strip()
        
        return text
    
    def evaluate(
        self,
        test_data: List[Dict],
        db_root: str,
        output_file: str = None
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: List of test examples
            db_root: Root directory of databases
            output_file: Optional file to save predictions
            
        Returns:
            Dictionary of metrics
        """
        total = len(test_data)
        execution_correct = 0
        exact_match = 0
        
        predictions = []
        
        for example in tqdm(test_data, desc="Evaluating"):
            question = example['question']
            gold_sql = example['query']
            db_id = example['db_id']
            db_path = f"{db_root}/{db_id}/{db_id}.sqlite"
            schema = example.get('schema')
            
            # Generate SQL
            pred_sql = self.generate_sql(question, schema)
            
            # Compute execution accuracy
            reward_dict = self.reward_calculator.compute_reward(
                pred_sql, gold_sql, question, db_path
            )
            
            if reward_dict['execution'] == 1.0:
                execution_correct += 1
            
            # Exact match
            if pred_sql.strip().upper() == gold_sql.strip().upper():
                exact_match += 1
            
            # Store prediction
            predictions.append({
                'question': question,
                'gold_sql': gold_sql,
                'pred_sql': pred_sql,
                'db_id': db_id,
                'execution_correct': reward_dict['execution'] == 1.0,
                'exact_match': pred_sql.strip().upper() == gold_sql.strip().upper()
            })
        
        # Compute metrics
        metrics = {
            'execution_accuracy': execution_correct / total,
            'exact_match': exact_match / total,
            'total': total
        }
        
        # Save predictions
        import os
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({'metrics': metrics, 'predictions': predictions}, f, indent=2)
        
        return metrics


def load_model(model_path: str, base_model: str = None):
    """
    Load model for evaluation.
    
    Args:
        model_path: Path to saved model (LoRA adapters or full model)
        base_model: Base model name (required if loading LoRA adapters)
    """
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to load as PEFT model first
    try:
        if base_model:
            # Load base model
            base = AutoModelForCausalLM.from_pretrained(
                args.base_model, 
                device_map={"": 0},  # Force single GPU
                torch_dtype=torch.bfloat16
            )
            # Load LoRA adapters
            model = PeftModel.from_pretrained(base, model_path)
        else:
            # Try loading directly
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
    except:
        # Fallback: load as regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    return model, tokenizer


def main(args):
    """Main evaluation function."""
    
    print("=" * 80)
    print("Text-to-SQL Evaluation")
    print("=" * 80)
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    
    print(f"Test examples: {len(test_data)}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Initialize reward calculator
    reward_config = RewardConfig(
        execution_weight=1.0,
        partial_weight=0.0,  # Not needed for evaluation
        timeout_seconds=5,
        use_partial_rewards=False
    )
    
    reward_calculator = SQLRewardCalculator(
        db_path="",  # Will be set per example
        config=reward_config
    )
    
    # Initialize evaluator
    evaluator = Text2SQLEvaluator(
        model=model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluator.evaluate(
        test_data=test_data,
        db_root=args.db_root,
        output_file=args.output_file
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Execution Accuracy: {metrics['execution_accuracy']:.2%}")
    print(f"Exact Match:        {metrics['exact_match']:.2%}")
    print(f"Total Examples:     {metrics['total']}")
    print("=" * 80)
    
    if args.output_file:
        print(f"\nPredictions saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL model")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model or LoRA adapters')
    parser.add_argument('--base_model', type=str,
                       help='Base model name (if loading LoRA adapters)')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data JSON')
    parser.add_argument('--db_root', type=str, required=True,
                       help='Root directory of databases')
    parser.add_argument('--output_file', type=str,
                       help='File to save predictions')
    
    args = parser.parse_args()
    
    main(args)
