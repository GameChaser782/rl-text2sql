"""
GRPO (Group Relative Policy Optimization) trainer for Text-to-SQL.
Based on SQL-R1 paper: https://arxiv.org/abs/2504.08600
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

from reward import SQLRewardCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class GRPOTrainer:
    """GRPO trainer for Text-to-SQL models."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        reward_calculator: SQLRewardCalculator,
        config: GRPOConfig,
        device: str = "cuda",
        num_gpus: int = 1
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
        """
        self.config = config
        self.device = device
        self.num_gpus = num_gpus
        
        # Policy model
        self.policy_model = model
        # Only move to device if not likely already handled by device_map="auto" or accelerate
        # If num_gpus > 1 and device_map="auto" used, we shouldn't force to "cuda" (which is cuda:0)
        # But if num_gpus == 1, ensuring it is on device is fine.
        if num_gpus == 1:
            self.policy_model = model.to(device)
            
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator
        
        # Create reference model (frozen copy of initial policy)
        self.reference_model = copy.deepcopy(model)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False
        
        # No CPU offloading - keep on GPU(s)
        # If num_gpus == 1: reference model stays on same GPU.
        # If num_gpus == 2: reference model stays on GPU(s) where it was initialized.
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _create_prompt(self, question: str, schema: Optional[str] = None) -> str:
        """Create prompt for the model."""
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
    def sample_responses(
        self, 
        prompts: List[str],
        num_samples: Optional[int] = None
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
        
        all_responses = []
        all_log_probs = []
        
        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []
            
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            prompt_length = inputs.input_ids.shape[1]
            
            # Sample multiple responses
            for _ in range(num_samples):
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
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
                    log_probs[0], 
                    dim=1, 
                    index=response_ids.unsqueeze(1)
                ).squeeze(1)
                
                prompt_responses.append(response)
                prompt_log_probs.append(token_log_probs)
            
            all_responses.append(prompt_responses)
            all_log_probs.append(prompt_log_probs)
        
        return all_responses, all_log_probs
    
    def compute_advantages(
        self, 
        rewards: List[List[float]]
    ) -> List[List[float]]:
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
    
    def compute_kl_divergence(
        self,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model."""
        # Tokenize
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        prompt_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        
        # Policy logits
        with torch.no_grad():
            policy_outputs = self.policy_model(**inputs)
            policy_logits = policy_outputs.logits[:, prompt_length-1:-1, :]
            
            # Reference logits (no device toggling)
            ref_outputs = self.reference_model(**inputs)
            ref_logits = ref_outputs.logits[:, prompt_length-1:-1, :]
        
        # KL divergence
        kl = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction='batchmean'
        )
        
        return kl
    
    def policy_gradient_step(
        self,
        prompts: List[str],
        responses: List[List[str]],
        advantages: List[List[float]]
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
        num_updates = 0
        
        for prompt, response_group, advantage_group in zip(prompts, responses, advantages):
            for response, advantage in zip(response_group, advantage_group):
                # Tokenize
                full_text = prompt + response
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                prompt_length = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                
                # Forward pass
                outputs = self.policy_model(**inputs, labels=inputs.input_ids)
                
                # Get log probs for generated tokens
                logits = outputs.logits[:, prompt_length-1:-1, :]
                labels = inputs.input_ids[:, prompt_length:]
                
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs,
                    dim=2,
                    index=labels.unsqueeze(2)
                ).squeeze(2)
                
                # Policy gradient loss: -advantage * log_prob
                pg_loss = -advantage * token_log_probs.mean()
                
                # KL penalty
                kl = self.compute_kl_divergence(prompt, response)
                
                # Total loss
                loss = pg_loss + self.config.kl_coef * kl
                
                # Backward pass
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_kl += kl.item()
                num_updates += 1
                
                # Update weights
                if num_updates % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
        
        return {
            'loss': total_loss / num_updates,
            'pg_loss': total_pg_loss / num_updates,
            'kl': total_kl / num_updates
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_metrics = {
            'loss': 0.0,
            'pg_loss': 0.0,
            'kl': 0.0,
            'mean_reward': 0.0,
            'execution_accuracy': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            # Extract batch data
            questions = batch['question']
            gold_sqls = batch['sql']
            db_paths = batch['db_path']
            schemas = batch.get('schema', [None] * len(questions))
            
            # Create prompts
            prompts = [
                self._create_prompt(q, s) 
                for q, s in zip(questions, schemas)
            ]
            
            # Sample responses
            responses, _ = self.sample_responses(prompts)
            
            # Compute rewards
            rewards = []
            exec_accs = []
            
            for response_group, gold_sql, db_path in zip(responses, gold_sqls, db_paths):
                group_rewards = []
                group_exec_accs = []
                
                for response in response_group:
                    # Extract SQL from response
                    pred_sql = self._extract_sql(response)
                    
                    # Compute reward
                    reward_dict = self.reward_calculator.compute_reward(
                        pred_sql, gold_sql, "", db_path
                    )
                    
                    group_rewards.append(reward_dict['total'])
                    group_exec_accs.append(reward_dict['execution'])
                
                rewards.append(group_rewards)
                exec_accs.append(np.mean(group_exec_accs))
            
            # Compute advantages
            advantages = self.compute_advantages(rewards)
            
            # Policy gradient step
            step_metrics = self.policy_gradient_step(prompts, responses, advantages)
            
            # Update metrics
            for key in epoch_metrics:
                if key in step_metrics:
                    epoch_metrics[key] += step_metrics[key]
            
            epoch_metrics['mean_reward'] += np.mean([np.mean(r) for r in rewards])
            epoch_metrics['execution_accuracy'] += np.mean(exec_accs)
            num_batches += 1
            
            # Update progress bar
            if num_batches % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': epoch_metrics['loss'] / num_batches,
                    'reward': epoch_metrics['mean_reward'] / num_batches,
                    'exec_acc': epoch_metrics['execution_accuracy'] / num_batches
                })
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from model response."""
        # Simple extraction: take everything after "SQL:" or first SELECT
        response = response.strip()
        
        # Look for SELECT keyword
        select_idx = response.upper().find('SELECT')
        if select_idx != -1:
            # Find end of SQL (semicolon or end of string)
            sql = response[select_idx:]
            semicolon_idx = sql.find(';')
            if semicolon_idx != -1:
                sql = sql[:semicolon_idx]
            return sql.strip()
        
        return response
    
    def train(self, train_dataloader: DataLoader):
        """Full training loop."""
        logger.info("Starting GRPO training...")
        logger.info(f"Config: {self.config}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            metrics = self.train_epoch(train_dataloader)
            
            logger.info(f"Epoch {epoch} metrics: {metrics}")
        
        logger.info("Training complete!")


# Example usage
if __name__ == "__main__":
    # This is a minimal example - actual usage would require full setup
    
    config = GRPOConfig(
        num_samples_per_prompt=4,
        learning_rate=1e-5,
        num_epochs=3,
        batch_size=1,
        kl_coef=0.1
    )
    
    print("GRPO Config:")
    print(config)
