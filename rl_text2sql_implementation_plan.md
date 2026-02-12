# RL Extension for Text-to-SQL: Implementation Plan

## Phase 1: Setup & Understanding (Day 1-2)

### Tasks:
1. **Paper Deep Dive**
   - Read SQL-R1 paper thoroughly
   - Understand GRPO algorithm
   - Identify key components: reward shaping, policy update, sampling strategy
   
2. **Codebase Familiarization**
   - Clone existing Text-to-SQL base code
   - Understand data pipeline
   - Identify integration points for RL

3. **Environment Setup**
   - Set up Google Colab with GPU
   - Install dependencies: transformers, peft, trl, datasets, sqlite3
   - Test base model inference

## Phase 2: Core RL Components (Day 3-5)

### Component 1: Reward Function
```python
# Pseudo-code structure
class SQLRewardCalculator:
    def __init__(self, db_path):
        self.db = Database(db_path)
    
    def execution_accuracy(self, pred_sql, gold_sql):
        # Execute both, compare results
        pass
    
    def partial_rewards(self, pred_sql, gold_sql):
        # Token-level overlap, clause matching
        pass
    
    def compute_reward(self, pred_sql, gold_sql, question):
        # Combine execution + partial rewards
        pass
```

**Key Design Decisions:**
- Binary reward: +1 for correct execution, 0 otherwise
- Partial rewards: keyword matching, clause structure similarity
- Safety: timeout execution, catch SQL errors

### Component 2: RL Training Loop (GRPO)

```python
# Integration with existing trainer
class RLTextToSQLTrainer:
    def __init__(self, model, tokenizer, reward_fn):
        self.policy_model = model
        self.reference_model = copy.deepcopy(model)  # Frozen
        self.reward_fn = reward_fn
    
    def sample_responses(self, prompts, n_samples=4):
        # Generate multiple SQL queries per question
        pass
    
    def compute_advantages(self, rewards):
        # Group-relative advantages (GRPO)
        pass
    
    def policy_gradient_step(self, prompts, responses, advantages):
        # PPO-style update with KL penalty
        pass
```

**GRPO Key Points:**
- Sample K responses per prompt (K=4-8)
- Compute group mean reward
- Advantage = reward - group_mean
- Update policy to increase prob of high-advantage responses

### Component 3: Integration

```python
# Main training script structure
def main():
    # 1. Load base model (LoRA/QLoRA)
    model = load_model("codellama-3b", use_qlora=True)
    
    # 2. Setup reward function
    reward_fn = SQLRewardCalculator(db_path="spider/database")
    
    # 3. Initialize RL trainer
    trainer = RLTextToSQLTrainer(model, tokenizer, reward_fn)
    
    # 4. Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Sample responses
            responses = trainer.sample_responses(batch['questions'])
            
            # Compute rewards
            rewards = [reward_fn.compute(resp, gold) 
                      for resp, gold in zip(responses, batch['sqls'])]
            
            # RL update
            loss = trainer.policy_gradient_step(batch, responses, rewards)
            
            # Log metrics
            log_metrics(rewards, loss)
```

## Phase 3: Optimization for 24GB GPU (Day 6)

### Memory Optimization Techniques:

1. **QLoRA Configuration**
   ```python
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
   )
   
   lora_config = LoraConfig(
       r=16,  # Rank
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05
   )
   ```

2. **Gradient Checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Batch Size Tuning**
   - Start with batch_size=1, accumulation_steps=8
   - Adjust based on memory usage

4. **Reference Model Handling**
   - Keep reference model in CPU, move to GPU only for KL computation
   - Or use same model with `requires_grad=False`

## Phase 4: Experiments & Evaluation (Day 7)

### Datasets:
- **Spider**: Standard benchmark (use dev set for quick iteration)
- **Bird-SQL**: More complex, optional

### Metrics to Track:
1. **Execution Accuracy (EX)**: % of queries with correct results
2. **Exact Match (EM)**: % of queries matching gold SQL exactly
3. **Reward Statistics**: mean, std, max reward per epoch
4. **Training Stability**: KL divergence, policy entropy
5. **Sample Efficiency**: EX vs. number of gradient steps

### Experimental Protocol:
```python
# Baseline: Base model (no RL)
baseline_results = evaluate(base_model, test_set)

# RL Training
train_rl(model, train_set, epochs=5)

# Evaluation
rl_results = evaluate(rl_model, test_set)

# Compare
print(f"Baseline EX: {baseline_results['ex']:.2%}")
print(f"RL EX: {rl_results['ex']:.2%}")
print(f"Improvement: {rl_results['ex'] - baseline_results['ex']:.2%}")
```

## Phase 5: Documentation & Cleanup (Day 8)

### README Structure:
```markdown
# RL-Enhanced Text-to-SQL

## Overview
Brief description of the approach

## Installation
Dependencies and setup

## Quick Start
```bash
python train_rl.py --config configs/grpo_spider.yaml
```

## Methodology
- Chosen paper: SQL-R1
- Rationale: Simplicity, strong baselines
- RL algorithm: GRPO
- Reward design: Execution accuracy + partial rewards

## Results
| Model | EX | EM |
|-------|-----|-----|
| Baseline | X% | Y% |
| +RL | X+Δ% | Y+Δ% |

## Observations
- Training stability
- Sample efficiency
- Failure cases

## Future Work
- Self-correction (Arctic-R1)
- Graph-based rewards
```

## Code Structure

```
rl-text2sql/
├── configs/
│   └── grpo_spider.yaml
├── src/
│   ├── data/
│   │   ├── spider_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── load_model.py
│   │   └── lora_config.py
│   ├── rl/
│   │   ├── reward.py          # Your main contribution
│   │   ├── grpo_trainer.py    # Your main contribution
│   │   └── utils.py
│   └── evaluation/
│       └── metrics.py
├── scripts/
│   ├── train_rl.py
│   └── evaluate.py
├── notebooks/
│   └── demo.ipynb
├── README.md
└── requirements.txt
```

## Key Implementation Tips

### 1. Reward Function Edge Cases
- **Timeout**: Set 5-second execution limit
- **SQL Errors**: Return reward = 0, log error type
- **Empty Results**: Check if both pred and gold are empty (could be correct!)

### 2. GRPO Stability
- **KL Penalty**: Start with β=0.1, tune if training unstable
- **Advantage Normalization**: `(adv - mean(adv)) / (std(adv) + 1e-8)`
- **Clipping**: Clip advantages to [-5, 5] to prevent outliers

### 3. Debugging Checklist
- [ ] Reward function returns expected values on manual examples
- [ ] Policy gradient has correct sign (increase prob of high reward)
- [ ] KL divergence stays within reasonable bounds (<1.0)
- [ ] Generated SQL is syntactically valid
- [ ] Memory usage stays under 24GB

### 4. Common Pitfalls
- **Forgetting to detach reference model**: Will cause gradient errors
- **Not handling SQL syntax errors**: Reward function must be robust
- **Insufficient sampling**: K=2 is too low, use K≥4
- **Ignoring KL constraint**: Model can diverge from base policy

## Timeline

| Day | Tasks | Deliverable |
|-----|-------|-------------|
| 1-2 | Setup, paper reading | Environment ready |
| 3-4 | Reward function, GRPO core | Working RL loop |
| 5-6 | Integration, optimization | Training runs |
| 7 | Experiments, evaluation | Results table |
| 8 | Documentation, polish | Final submission |

## Resources

### Papers
- SQL-R1: https://arxiv.org/abs/2504.08600
- GRPO: https://arxiv.org/abs/2402.03300

### Code References
- TRL library: https://github.com/huggingface/trl
- Spider dataset: https://yale-lily.github.io/spider

### Tutorials
- QLoRA fine-tuning: https://huggingface.co/blog/4bit-transformers-bitsandbytes
- PPO/GRPO: https://huggingface.co/docs/trl/ppov2_trainer
