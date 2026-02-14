"""
Evaluation script for Text-to-SQL models with detailed metrics.
"""

import torch
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score

from reward import SQLRewardCalculator, RewardConfig


class Text2SQLEvaluator:
    """Evaluator for Text-to-SQL models."""
    
    def __init__(self, model, tokenizer, reward_calculator, device="cuda"):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.device = device
    
    def create_prompt(self, question: str, schema: str = None) -> str:
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
        prompt = self.create_prompt(question, schema)
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sql = self.extract_sql(generated_text)
        return sql
    
    def extract_sql(self, text: str) -> str:
        text = text.strip()
        select_idx = text.upper().find('SELECT')
        if select_idx != -1:
            sql = text[select_idx:]
            semicolon_idx = sql.find(';')
            if semicolon_idx != -1:
                sql = sql[:semicolon_idx]
            return sql.strip()
        return text
    
    def evaluate(self, test_data: List[Dict], db_root: str, output_file: str = None) -> Dict[str, float]:
        """Evaluate with detailed metrics including F1, Precision, Recall."""
        total = len(test_data)
        execution_correct = 0
        exact_match = 0
        predictions = []
        predictions_binary = []
        ground_truth = [1] * total
        
        for example in tqdm(test_data, desc="Evaluating"):
            question = example['question']
            gold_sql = example['query']
            db_id = example['db_id']
            db_path = f"{db_root}/{db_id}/{db_id}.sqlite"
            schema = example.get('schema')
            
            pred_sql = self.generate_sql(question, schema)
            reward_dict = self.reward_calculator.compute_reward(pred_sql, gold_sql, question, db_path)
            
            is_correct = reward_dict['execution'] == 1.0
            if is_correct:
                execution_correct += 1
                predictions_binary.append(1)
            else:
                predictions_binary.append(0)
            
            if pred_sql.strip().upper() == gold_sql.strip().upper():
                exact_match += 1
            
            predictions.append({
                'question': question,
                'gold_sql': gold_sql,
                'pred_sql': pred_sql,
                'db_id': db_id,
                'execution_correct': is_correct,
                'exact_match': pred_sql.strip().upper() == gold_sql.strip().upper(),
                'reward': reward_dict['total']
            })
        
        metrics = {
            'execution_accuracy': execution_correct / total if total > 0 else 0,
            'exact_match': exact_match / total if total > 0 else 0,
            'precision': precision_score(ground_truth, predictions_binary, zero_division=0),
            'recall': recall_score(ground_truth, predictions_binary, zero_division=0),
            'f1_score': f1_score(ground_truth, predictions_binary, zero_division=0),
            'total': total
        }
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({'metrics': metrics, 'predictions': predictions}, f, indent=2)
        
        return metrics


def main(args):
    print("="*80)
    print("Text-to-SQL Evaluation")
    print("="*80)
    
    print(f"\nLoading test data from {args.test_data}...")
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)
    print(f"Test examples: {len(test_data)}")
    
    print(f"\nLoading model from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device map
    if args.num_gpus == 1:
        device_map = {"": 0}
    else:
        device_map = "auto"
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        device_map=device_map,
        dtype=torch.bfloat16
    )
    
    # Load PEFT adapter only if model_path is different from base_model
    # and implies we are loading an adapter (e.g. not just verifying the base model)
    if args.model_path != args.base_model:
        print(f"Loading PEFT adapter from {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print("Evaluating base model (no PEFT adapter loaded)")
    
    reward_config = RewardConfig(execution_weight=1.0, partial_weight=0.0, timeout_seconds=5, use_partial_rewards=False)
    reward_calculator = SQLRewardCalculator(db_path="", config=reward_config)
    
    evaluator = Text2SQLEvaluator(model=model, tokenizer=tokenizer, reward_calculator=reward_calculator, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nEvaluating...")
    metrics = evaluator.evaluate(test_data=test_data, db_root=args.db_root, output_file=args.output_file)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Execution Accuracy: {metrics['execution_accuracy']:.2%}")
    print(f"Exact Match:        {metrics['exact_match']:.2%}")
    print(f"Precision:          {metrics['precision']:.3f}")
    print(f"Recall:             {metrics['recall']:.3f}")
    print(f"F1 Score:           {metrics['f1_score']:.3f}")
    print(f"Total Examples:     {metrics['total']}")
    print("="*80)
    
    if args.output_file:
        print(f"\nPredictions saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model or adapter")
    parser.add_argument('--base_model', type=str, required=True, help="Base model name/path")
    parser.add_argument('--test_data', type=str, required=True, help="Path to test data JSON")
    parser.add_argument('--db_root', type=str, required=True, help="Database root directory")
    parser.add_argument('--output_file', type=str, help="Output file for results")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    main(args)