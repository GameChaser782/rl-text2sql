"""
GRPO (Group Relative Policy Optimization) trainer for Text-to-SQL.
Based on SQL-R1 paper: https://arxiv.org/abs/2504.08600
"""

import logging
import os
import shutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from reward import SQLRewardCalculator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over valid response tokens only."""
    mask = mask.to(dtype=values.dtype, device=values.device)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Sampling
    num_samples_per_prompt: int = 2  # K in the paper
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 512

    # Training
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8

    # GRPO-specific
    kl_coef: float = 0.1  # β for KL penalty
    clip_range: float = 0.2  # PPO-style clipping
    advantage_normalization: bool = True

    # Optimization
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

    # Logging
    log_interval: int = 10
    eval_interval: int = 100

    # Hardware
    num_gpus: int = 1
    reward_workers: int = 1
    save_steps: int = 500
    save_total_limit: int = 2
    output_dir: Optional[str] = None
    tensorboard_log_dir: Optional[str] = None
    eval_subset_size: int = 50


class GRPOTrainer:
    """GRPO trainer for Text-to-SQL models."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        reference_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_calculator: SQLRewardCalculator,
        config: GRPOConfig,
        accelerator: Accelerator,
        device: str = "cuda",
        num_gpus: int = 1,
        use_unsloth: bool = False,
    ):
        """
        Initialize GRPO trainer.

        Args:
            model: Policy model (will be trained)
            tokenizer: Tokenizer
            reward_calculator: Reward function
            config: Training configuration
            device: Device to use
            num_gpus: Number of GPUs
            use_unsloth: Whether model uses Unsloth optimization
        """
        self.config = config
        self.device = device
        self.num_gpus = num_gpus
        self.use_unsloth = use_unsloth
        self.accelerator = accelerator

        # Policy model
        self.policy_model = model

        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        self.schema_cache: Dict[str, Optional[str]] = {}

        # Create reference model (frozen copy of initial policy)
        logger.info("Initializing reference model for KL divergence computation...")
        self.reference_model = reference_model

        if hasattr(self.reference_model, "eval"):
            self.reference_model.eval()
        if not getattr(self.reference_model, "hf_device_map", None):
            self.reference_model = self.reference_model.to(self.accelerator.device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=config.learning_rate
        )

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_dev_execution_accuracy = float("-inf")
        self.tb_writer = None
        if self.accelerator.is_main_process and self.config.tensorboard_log_dir:
            os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=self.config.tensorboard_log_dir)

    def _save_checkpoint(self):
        if (
            not self.accelerator.is_main_process
            or not self.config.output_dir
            or self.global_step <= 0
        ):
            return

        checkpoint_dir = os.path.join(
            self.config.output_dir, f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(checkpoint_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(checkpoint_dir)

        if self.config.save_total_limit and self.config.save_total_limit > 0:
            checkpoints = sorted(
                [
                    os.path.join(self.config.output_dir, name)
                    for name in os.listdir(self.config.output_dir)
                    if name.startswith("checkpoint-")
                    and os.path.isdir(os.path.join(self.config.output_dir, name))
                ],
                key=lambda path: int(os.path.basename(path).split("-")[-1]),
            )
            while len(checkpoints) > self.config.save_total_limit:
                old_checkpoint = checkpoints.pop(0)
                shutil.rmtree(old_checkpoint, ignore_errors=True)

    def _unwrap_policy_model(self):
        return self.accelerator.unwrap_model(self.policy_model)

    def _save_best_checkpoint(self):
        if not self.accelerator.is_main_process or not self.config.output_dir:
            return

        best_dir = os.path.join(self.config.output_dir, "best-checkpoint")
        os.makedirs(best_dir, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.policy_model)
        if hasattr(unwrapped_model, "save_pretrained"):
            unwrapped_model.save_pretrained(best_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(best_dir)

    @staticmethod
    def _module_device(module: AutoModelForCausalLM) -> torch.device:
        if hasattr(module, "device"):
            return module.device
        return next(module.parameters()).device

    def _create_prompt(self, question: str, schema: Optional[str] = None) -> str:
        """Create prompt for the model."""
        if schema:
            prompt = f"""Given the database schema:
{schema}

Question: {question}

Return only one executable SQL query that answers the question.
Do not include any explanation, markdown, comments, or extra text.
SQL:"""
        else:
            prompt = (
                f"Question: {question}\n"
                "Return only one executable SQL query.\n"
                "Do not include any explanation, markdown, comments, or extra text.\n"
                "SQL:"
            )

        return prompt

    def _get_db_schema(self, db_path: str) -> Optional[str]:
        if db_path in self.schema_cache:
            return self.schema_cache[db_path]

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
            schema = "; ".join(parts) if parts else None
        except Exception:
            schema = None

        self.schema_cache[db_path] = schema
        return schema

    @torch.no_grad()
    def sample_responses(
        self, prompts: List[str], num_samples: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[torch.Tensor]]]:
        """
        Sample multiple responses for each prompt.

        Args:
            prompts: List of prompts
            num_samples: Number of samples per prompt (defaults to config value)

        Returns:
            responses: List of [num_samples responses per prompt]
            log_probs: List of [num_samples log probability tensors per prompt]
        """
        num_samples = num_samples or self.config.num_samples_per_prompt

        self.policy_model.eval()
        generation_model = self._unwrap_policy_model()
        generation_device = self._module_device(generation_model)

        all_responses = []
        all_log_probs = []

        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []

            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            ).to(generation_device)

            prompt_length = inputs.input_ids.shape[1]

            # Sample multiple responses
            for _ in range(num_samples):
                outputs = generation_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                # Decode response
                full_sequence = outputs.sequences[0]
                response_ids = full_sequence[prompt_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Compute log probabilities
                logits = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab]
                log_probs = F.log_softmax(logits, dim=-1)

                # Get log probs of selected tokens
                token_log_probs = torch.gather(
                    log_probs[0], dim=1, index=response_ids.unsqueeze(1)
                ).squeeze(1)

                prompt_responses.append(response)
                prompt_log_probs.append(token_log_probs)

            all_responses.append(prompt_responses)
            all_log_probs.append(prompt_log_probs)

        return all_responses, all_log_probs

    @torch.no_grad()
    def generate_greedy_response(self, prompt: str) -> str:
        generation_model = self._unwrap_policy_model()
        generation_device = self._module_device(generation_model)
        generation_model.eval()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        ).to(generation_device)
        prompt_length = inputs.input_ids.shape[1]
        outputs = generation_model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        response_ids = outputs[0][prompt_length:]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def compute_advantages(self, rewards: List[List[float]]) -> List[List[float]]:
        """
        Compute group-relative advantages.

        Args:
            rewards: List of [rewards for each sample] per prompt

        Returns:
            advantages: Same structure as rewards
        """
        advantages = []

        for reward_group in rewards:
            # Group-relative advantage: reward - mean(group_rewards)
            group_mean = np.mean(reward_group)
            group_advantages = [r - group_mean for r in reward_group]

            # Optional normalization
            if self.config.advantage_normalization and len(reward_group) > 1:
                std = np.std(group_advantages)
                if std > 1e-8:
                    group_advantages = [
                        (adv - np.mean(group_advantages)) / (std + 1e-8)
                        for adv in group_advantages
                    ]

            advantages.append(group_advantages)

        return advantages

    @staticmethod
    def compute_policy_loss(
        old_log_probs: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_range: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the PPO clipped policy loss used by SQL-R1/VERL.

        All tensors are response-token aligned with shape [batch, response_length].
        """
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        ppo_kl = masked_mean(-log_ratio, response_mask)

        pg_losses = -advantages * ratio
        pg_losses_clipped = -advantages * torch.clamp(
            ratio, 1.0 - clip_range, 1.0 + clip_range
        )
        pg_loss = masked_mean(torch.max(pg_losses, pg_losses_clipped), response_mask)
        pg_clipfrac = masked_mean(
            (pg_losses_clipped > pg_losses).float(), response_mask
        )

        return pg_loss, pg_clipfrac, ppo_kl

    @staticmethod
    def low_variance_kl(policy_log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """Low-variance KL estimator used in VERL."""
        log_ratio = ref_log_probs - policy_log_probs
        return torch.exp(log_ratio) - log_ratio - 1

    def _response_log_probs(
        self, model: AutoModelForCausalLM, prompt: str, response: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected-token log-probs and a valid-token mask for one response."""
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text, return_tensors="pt", padding=True, truncation=True
        ).to(self._module_device(model))
        prompt_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        outputs = model(**inputs)
        logits = outputs.logits[:, prompt_length - 1 : -1, :]
        labels = inputs.input_ids[:, prompt_length:]

        if logits.shape[1] != labels.shape[1]:
            seq_len = min(logits.shape[1], labels.shape[1])
            logits = logits[:, :seq_len, :]
            labels = labels[:, :seq_len]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        mask = torch.ones_like(token_log_probs, dtype=torch.float32)

        return token_log_probs, mask

    def compute_kl_divergence(self, prompt: str, response: str) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference model.
        Handles both standard and Unsloth models.
        """
        # Tokenize
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text, return_tensors="pt", padding=True, truncation=True
        ).to(self._module_device(self._unwrap_policy_model()))

        prompt_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

        # Policy logits
        with torch.no_grad():
            try:
                policy_outputs = self.policy_model(**inputs)
                policy_logits = policy_outputs.logits[:, prompt_length - 1 : -1, :]
            except Exception as e:
                logger.warning(f"Error getting policy logits: {e}. Using dummy KL.")
                return torch.tensor(0.0, device=self.accelerator.device)

            # Reference logits
            try:
                # Move inputs to reference model device if different
                ref_inputs = {
                    k: v.to(self._module_device(self.reference_model))
                    for k, v in inputs.items()
                }
                ref_outputs = self.reference_model(**ref_inputs)
                ref_logits = ref_outputs.logits[:, prompt_length - 1 : -1, :].to(
                    policy_logits.device
                )
            except Exception as e:
                logger.warning(f"Error getting reference logits: {e}. Using zero KL.")
                return torch.tensor(0.0, device=self.accelerator.device)

        # KL divergence
        try:
            kl = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean",
            )
        except Exception as e:
            logger.warning(f"Error computing KL divergence: {e}. Returning zero.")
            kl = torch.tensor(0.0, device=self.accelerator.device)

        return kl

    def policy_gradient_step(
        self,
        prompts: List[str],
        responses: List[List[str]],
        advantages: List[List[float]],
        old_log_probs: List[List[torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Perform a policy gradient update step.

        Args:
            prompts: List of prompts
            responses: List of [responses per prompt]
            advantages: List of [advantages per response]

        Returns:
            Loss dictionary
        """
        self.policy_model.train()

        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_ppo_kl = 0.0
        num_updates = 0

        self.optimizer.zero_grad()

        for group_index, (prompt, response_group, advantage_group) in enumerate(
            zip(prompts, responses, advantages)
        ):
            for response_index, (response, advantage) in enumerate(
                zip(response_group, advantage_group)
            ):
                token_log_probs, response_mask = self._response_log_probs(
                    self.policy_model, prompt, response
                )
                old_token_log_probs = old_log_probs[group_index][response_index].to(
                    token_log_probs.device
                )

                seq_len = min(token_log_probs.shape[1], old_token_log_probs.numel())
                token_log_probs = token_log_probs[:, :seq_len]
                response_mask = response_mask[:, :seq_len]
                old_token_log_probs = old_token_log_probs[:seq_len].unsqueeze(0)
                advantage_tensor = torch.full_like(token_log_probs, float(advantage))

                pg_loss, pg_clipfrac, ppo_kl = self.compute_policy_loss(
                    old_token_log_probs,
                    token_log_probs,
                    advantage_tensor,
                    response_mask,
                    self.config.clip_range,
                )

                try:
                    with torch.no_grad():
                        ref_log_probs, _ = self._response_log_probs(
                            self.reference_model, prompt, response
                        )
                    ref_log_probs = ref_log_probs.to(token_log_probs.device)[:, :seq_len]
                    token_kl = self.low_variance_kl(token_log_probs, ref_log_probs)
                    kl = masked_mean(token_kl, response_mask)
                except Exception as e:
                    logger.warning(f"Error computing reference token KL: {e}. Using zero KL.")
                    kl = torch.tensor(0.0, device=self.accelerator.device)

                # Total loss
                loss = pg_loss + self.config.kl_coef * kl

                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                self.accelerator.backward(loss)

                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_kl += kl.item()
                total_clipfrac += pg_clipfrac.item()
                total_ppo_kl += ppo_kl.item()
                num_updates += 1

                # Update weights
                if num_updates % self.config.gradient_accumulation_steps == 0:
                    self.accelerator.clip_grad_norm_(
                        self.policy_model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    if (
                        self.config.save_steps > 0
                        and self.global_step % self.config.save_steps == 0
                    ):
                        self._save_checkpoint()

        if num_updates % self.config.gradient_accumulation_steps != 0:
            self.accelerator.clip_grad_norm_(
                self.policy_model.parameters(), self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            if (
                self.config.save_steps > 0
                and self.global_step % self.config.save_steps == 0
            ):
                self._save_checkpoint()

        return {
            "loss": total_loss / num_updates,
            "pg_loss": total_pg_loss / num_updates,
            "kl": total_kl / num_updates,
            "clipfrac": total_clipfrac / num_updates,
            "ppo_kl": total_ppo_kl / num_updates,
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            "loss": 0.0,
            "pg_loss": 0.0,
            "kl": 0.0,
            "mean_reward": 0.0,
            "execution_accuracy": 0.0,
        }

        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            # Extract batch data
            questions = batch["question"]
            gold_sqls = batch["sql"]
            db_paths = batch["db_path"]
            schemas = batch.get("schema", [None] * len(questions))

            # Create prompts
            prompts = [
                self._create_prompt(q, s or self._get_db_schema(db_path))
                for q, s, db_path in zip(questions, schemas, db_paths)
            ]

            # Sample responses
            responses, old_log_probs = self.sample_responses(prompts)

            # Compute rewards
            rewards = []
            exec_accs = []

            for response_group, gold_sql, db_path in zip(
                responses, gold_sqls, db_paths
            ):
                group_rewards, group_exec_accs = self._compute_group_rewards(
                    response_group, gold_sql, db_path
                )

                rewards.append(group_rewards)
                exec_accs.append(np.mean(group_exec_accs))

            # Compute advantages
            advantages = self.compute_advantages(rewards)

            # Policy gradient step
            step_metrics = self.policy_gradient_step(
                prompts, responses, advantages, old_log_probs
            )

            # Update metrics
            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]

            epoch_metrics["mean_reward"] += np.mean([np.mean(r) for r in rewards])
            epoch_metrics["execution_accuracy"] += np.mean(exec_accs)
            num_batches += 1

            # Update progress bar
            if num_batches % self.config.log_interval == 0:
                log_metrics = {
                    "loss": epoch_metrics["loss"] / num_batches,
                    "reward": epoch_metrics["mean_reward"] / num_batches,
                    "exec_acc": epoch_metrics["execution_accuracy"] / num_batches,
                }
                progress_bar.set_postfix(log_metrics)
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(
                        "train/loss", log_metrics["loss"], self.global_step
                    )
                    self.tb_writer.add_scalar(
                        "train/reward", log_metrics["reward"], self.global_step
                    )
                    self.tb_writer.add_scalar(
                        "train/execution_accuracy",
                        log_metrics["exec_acc"],
                        self.global_step,
                    )

        # Average metrics
        local_metrics = {
            key: torch.tensor(value, device=self.accelerator.device, dtype=torch.float32)
            for key, value in epoch_metrics.items()
        }
        local_num_batches = torch.tensor(
            num_batches, device=self.accelerator.device, dtype=torch.float32
        )

        gathered_num_batches = self.accelerator.gather(local_num_batches.unsqueeze(0))
        total_batches = gathered_num_batches.sum().item()

        aggregated_metrics = {}
        for key, value in local_metrics.items():
            gathered_value = self.accelerator.gather(value.unsqueeze(0))
            aggregated_metrics[key] = gathered_value.sum().item() / max(total_batches, 1.0)

        return aggregated_metrics

    def evaluate_dataset(
        self, dataset, max_examples: Optional[int] = None
    ) -> Dict[str, float]:
        total = min(len(dataset), max_examples or len(dataset))
        execution_correct = 0
        exact_match = 0

        for idx in range(total):
            item = dataset[idx]
            prompt = self._create_prompt(
                item["question"],
                item.get("schema") or self._get_db_schema(item["db_path"]),
            )
            pred_sql = self._extract_sql(self.generate_greedy_response(prompt))
            reward_dict = self.reward_calculator.compute_reward(
                pred_sql, item["sql"], item["question"], item["db_path"]
            )
            if reward_dict["execution"] == 1.0:
                execution_correct += 1
            if pred_sql.strip().upper() == item["sql"].strip().upper():
                exact_match += 1

        return {
            "execution_accuracy": execution_correct / max(total, 1),
            "exact_match": exact_match / max(total, 1),
            "total": total,
        }

    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from model response."""
        response = response.strip().replace("```", " ").replace("`", " ")
        if "SQL:" in response:
            response = response.split("SQL:", 1)[1].strip()

        upper_response = response.upper()
        sql_starts = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
        start_idx = -1
        for keyword in sql_starts:
            idx = upper_response.find(keyword)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx

        sql = response[start_idx:] if start_idx != -1 else response

        stop_markers = [
            "\n\n",
            "\nExplanation:",
            "\nEXPLANATION:",
            "\nNote:",
            "\nNOTE:",
            "\nThis query",
            "\nThe SQL query",
        ]
        for marker in stop_markers:
            marker_idx = sql.find(marker)
            if marker_idx != -1:
                sql = sql[:marker_idx]

        first_line = sql.splitlines()[0].strip() if sql.splitlines() else sql.strip()
        sql = first_line or sql.strip()

        semicolon_idx = sql.find(";")
        if semicolon_idx != -1:
            sql = sql[:semicolon_idx]

        return sql.strip()

    def _compute_group_rewards(
        self, response_group: List[str], gold_sql: str, db_path: str
    ) -> Tuple[List[float], List[float]]:
        pred_sqls = [self._extract_sql(response) for response in response_group]
        max_workers = max(self.config.reward_workers, 1)
        max_workers = min(max_workers, len(pred_sqls), max(os.cpu_count() or 1, 1))

        if max_workers <= 1:
            reward_dicts = [
                self.reward_calculator.compute_reward(pred_sql, gold_sql, "", db_path)
                for pred_sql in pred_sqls
            ]
        else:
            def _compute(pred_sql: str) -> Dict[str, float]:
                return self.reward_calculator.compute_reward(
                    pred_sql, gold_sql, "", db_path
                )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                reward_dicts = list(executor.map(_compute, pred_sqls))

        group_rewards = [reward_dict["total"] for reward_dict in reward_dicts]
        group_exec_accs = [reward_dict["execution"] for reward_dict in reward_dicts]
        return group_rewards, group_exec_accs

    def train(self, train_dataloader: DataLoader, dev_dataset=None):
        """Full training loop."""
        logger.info("Starting GRPO training...")
        logger.info(f"Config: {self.config}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            # Train epoch
            metrics = self.train_epoch(train_dataloader)

            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch} metrics: {metrics}")
                if self.tb_writer is not None:
                    for key, value in metrics.items():
                        self.tb_writer.add_scalar(f"train_epoch/{key}", value, epoch)

            self.accelerator.wait_for_everyone()
            if dev_dataset is not None and self.accelerator.is_main_process:
                dev_metrics = self.evaluate_dataset(
                    dev_dataset, max_examples=self.config.eval_subset_size
                )
                logger.info(f"Epoch {epoch} dev metrics: {dev_metrics}")
                if self.tb_writer is not None:
                    for key, value in dev_metrics.items():
                        self.tb_writer.add_scalar(f"dev/{key}", value, epoch)

                if dev_metrics["execution_accuracy"] > self.best_dev_execution_accuracy:
                    self.best_dev_execution_accuracy = dev_metrics["execution_accuracy"]
                    self._save_best_checkpoint()
                    logger.info(
                        "Saved new best checkpoint with dev execution accuracy %.4f",
                        self.best_dev_execution_accuracy,
                    )

        if self.accelerator.is_main_process:
            if self.tb_writer is not None:
                self.tb_writer.close()
            logger.info("Training complete!")


# Example usage
if __name__ == "__main__":
    # This is a minimal example - actual usage would require full setup

    config = GRPOConfig(
        num_samples_per_prompt=4,
        learning_rate=1e-5,
        num_epochs=3,
        batch_size=1,
        kl_coef=0.1,
    )

    print("GRPO Config:")
    print(config)
