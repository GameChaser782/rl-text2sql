"""
Supervised fine-tuning script for Text-to-SQL with LoRA adapters.
"""

import argparse
import json
import os
import sqlite3
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from src.models.model_loader import load_model


def get_db_schema(db_path: str) -> Optional[str]:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        parts = []
        for table in tables:
            try:
                cur.execute(f"PRAGMA table_info('{table}')")
                cols = [r[1] for r in cur.fetchall()]
                parts.append(f"{table}({', '.join(cols)})")
            except Exception:
                parts.append(table)
        conn.close()
        return "; ".join(parts)
    except Exception:
        return None


def build_prompt(question: str, schema: Optional[str] = None) -> str:
    if schema:
        return (
            f"Given the database schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            "Return only one executable SQL query that answers the question.\n"
            "Do not include any explanation, markdown, comments, or extra text.\n"
            "SQL:"
        )
    return (
        f"Question: {question}\n"
        "Return only one executable SQL query.\n"
        "Do not include any explanation, markdown, comments, or extra text.\n"
        "SQL:"
    )


def load_examples(data_path: str, db_root: str) -> List[Dict[str, str]]:
    with open(data_path, "r") as f:
        data = json.load(f)

    examples = []
    for item in data:
        db_id = item["db_id"]
        db_path = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        schema = item.get("schema") or get_db_schema(db_path)
        prompt = build_prompt(item["question"], schema)
        target = item["query"].strip()
        examples.append(
            {
                "prompt": prompt,
                "target": target,
            }
        )
    return examples


def tokenize_example(example, tokenizer, max_length: int):
    prompt_text = example["prompt"]
    target_text = example["target"] + tokenizer.eos_token
    full_text = prompt_text + target_text

    prompt_tokens = tokenizer(
        prompt_text, truncation=True, max_length=max_length, add_special_tokens=False
    )
    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=False
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = input_ids.copy()

    prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main(args):
    print("=" * 80)
    print("Text-to-SQL SFT")
    print("=" * 80)

    model, tokenizer = load_model(
        args.model_name,
        use_qlora=args.use_qlora,
        use_unsloth=False,
        num_gpus=max(torch.cuda.device_count(), 1),
        max_seq_length=args.max_length,
        device_map=None,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        ),
    )

    examples = load_examples(args.train_data, args.db_root)
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda example: tokenize_example(example, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalDataCollator(tokenizer),
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"SFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT train Text-to-SQL model")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--db_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--use_qlora", action="store_true", default=True)

    args = parser.parse_args()
    main(args)
