# Implementation Checklist & Tips

## Pre-Implementation (Day 0)

### Paper Reading
- [ ] Read SQL-R1 paper (focus on Sections 3-4)
- [ ] Understand GRPO algorithm (especially Eq. 2-4)
- [ ] Review reward design choices
- [ ] Note hyperparameters used in paper

### Environment Setup
- [ ] Set up Google Colab with A100 GPU (or local GPU)
- [ ] Clone base repository
- [ ] Test GPU access: `nvidia-smi`
- [ ] Install dependencies: `pip install -r requirements.txt`

### Dataset Preparation
- [ ] Download Spider dataset
- [ ] Verify database files are accessible
- [ ] Check data format (JSON structure)
- [ ] Create small subset for testing (10-20 examples)

## Core Implementation (Day 1-4)

### Day 1: Reward Function ✅

**File: `reward.py`**

- [ ] Implement `_execute_sql()` with timeout handling
- [ ] Implement `execution_accuracy()` with result comparison
- [ ] Implement `partial_rewards()` with SQL parsing
- [ ] Test on manual examples:
  ```python
  # Test cases:
  # 1. Exact match
  # 2. Different order (should still match)
  # 3. Syntax error (should return 0)
  # 4. Timeout (should handle gracefully)
  ```

**Success Criteria:**
- ✅ Executes SQL queries successfully
- ✅ Handles errors gracefully (no crashes)
- ✅ Partial rewards work when execution fails
- ✅ Unit tests pass

### Day 2-3: GRPO Trainer ✅

**File: `grpo_trainer.py`**

- [ ] Implement `sample_responses()` - generate K queries per prompt
- [ ] Implement `compute_advantages()` - group-relative calculation
- [ ] Implement `compute_kl_divergence()` - KL penalty
- [ ] Implement `policy_gradient_step()` - the core update
- [ ] Test each component independently

**Critical Checks:**
```python
# 1. Sampling works
responses, log_probs = trainer.sample_responses(["SELECT..."])
assert len(responses[0]) == config.num_samples

# 2. Advantages are zero-mean within groups
advantages = trainer.compute_advantages([[1.0, 0.5, 0.3, 0.2]])
assert abs(sum(advantages[0])) < 1e-6

# 3. KL is computed correctly
kl = trainer.compute_kl_divergence(prompt, response)
assert kl > 0  # Should be positive

# 4. Policy gradient has correct sign
# High advantage → increase probability
# Low advantage → decrease probability
```

**Success Criteria:**
- ✅ Generates diverse responses (not all identical)
- ✅ Advantages sum to ~0 within each group
- ✅ KL divergence is reasonable (0.1-1.0)
- ✅ Loss decreases during training

### Day 4: Integration ✅

**File: `train_rl.py`**

- [ ] Implement model loading with QLoRA
- [ ] Implement dataset loading (Spider format)
- [ ] Connect all components (model → sampler → reward → trainer)
- [ ] Add logging (print metrics every N steps)
- [ ] Add checkpointing (save model every N steps)

**Test Run:**
```bash
# Small test with 10 examples, 1 epoch
python train_rl.py \
    --train_data data/spider/train_subset.json \
    --db_root data/spider/database \
    --num_epochs 1 \
    --batch_size 1 \
    --num_samples 2 \
    --output_dir test_output
```

**Success Criteria:**
- ✅ Training runs without errors
- ✅ Memory usage < 24GB
- ✅ Rewards improve over steps
- ✅ Model saves correctly

## Optimization (Day 5)

### Memory Optimization

**Critical: Ensure < 24GB GPU usage**

- [ ] Enable QLoRA (4-bit quantization)
- [ ] Enable gradient checkpointing
- [ ] Move reference model to CPU
- [ ] Use gradient accumulation (batch_size=1, accumulation=8)
- [ ] Monitor memory: `nvidia-smi -l 1`

**If OOM (Out of Memory):**
1. Reduce `num_samples` (4 → 2)
2. Reduce `max_new_tokens` (256 → 128)
3. Reduce LoRA rank (16 → 8)
4. Increase gradient accumulation (8 → 16)

### Speed Optimization

- [ ] Use Flash Attention if available
- [ ] Compile model with `torch.compile()` (PyTorch 2.0+)
- [ ] Profile code to find bottlenecks
- [ ] Parallelize reward computation (if multiple GPUs)

## Experiments (Day 6-7)

### Baseline Evaluation

```bash
# 1. Evaluate base model (before RL)
python evaluate.py \
    --model_path codellama/CodeLlama-7b-hf \
    --test_data data/spider/dev.json \
    --db_root data/spider/database \
    --output_file results/baseline.json
```

Expected baseline (CodeLlama-7B on Spider):
- Execution Accuracy: ~30-40%
- Exact Match: ~20-30%

### RL Training

```bash
# 2. Full training run
python train_rl.py --config config_grpo_spider.yaml
```

**Monitor During Training:**
- Mean reward (should increase)
- Execution accuracy (should increase)
- KL divergence (should stay < 1.0)
- Training loss (should decrease)

### Post-Training Evaluation

```bash
# 3. Evaluate RL model
python evaluate.py \
    --model_path outputs/rl-model \
    --base_model codellama/CodeLlama-7b-hf \
    --test_data data/spider/dev.json \
    --db_root data/spider/database \
    --output_file results/rl_model.json
```

Expected improvement:
- Execution Accuracy: +5-10% absolute
- Exact Match: +3-5% absolute

## Documentation (Day 8)

### README.md

- [ ] Clear project description
- [ ] Installation instructions
- [ ] Quick start guide
- [ ] Methodology explanation
- [ ] Results table
- [ ] Observations and analysis

### Code Documentation

- [ ] Docstrings for all classes and functions
- [ ] Comments for non-obvious logic
- [ ] Type hints where appropriate
- [ ] Example usage in docstrings

### Experimental Notes

- [ ] Record hyperparameters used
- [ ] Document any issues encountered
- [ ] Note any deviations from paper
- [ ] List failure cases and potential solutions

## Common Pitfalls & Solutions

### Pitfall 1: Model generates invalid SQL
**Solution:** 
- Add partial rewards to provide learning signal
- Use constrained decoding (if time permits)
- Increase temperature for more diverse samples

### Pitfall 2: Rewards are all 0
**Problem:** Model too weak to generate any correct queries
**Solution:**
- Start with supervised fine-tuning first
- Use curriculum learning (easy examples first)
- Increase partial reward weight

### Pitfall 3: KL divergence explodes
**Problem:** Policy diverges too far from reference
**Solution:**
- Increase `kl_coef` (0.1 → 0.2 or higher)
- Reduce learning rate
- Clip advantages more aggressively

### Pitfall 4: OOM errors
**Solution:**
- See Memory Optimization section above
- Use smaller model (3B instead of 7B)
- Reduce batch size / num_samples

### Pitfall 5: Training is too slow
**Solution:**
- Reduce number of training examples
- Use smaller dev set for faster evaluation
- Parallelize reward computation
- Use faster database (in-memory SQLite)

## Quick Debug Commands

```python
# Check memory usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Check model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.1f}M")

# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean():.6f}")

# Profile code
import time
start = time.time()
# ... code to profile ...
print(f"Time: {time.time() - start:.2f}s")
```

## Final Checklist Before Submission

- [ ] Code runs end-to-end without errors
- [ ] Memory usage < 24GB
- [ ] Results are reasonable (not random)
- [ ] README is comprehensive
- [ ] Code is well-documented
- [ ] GitHub repo is clean and organized
- [ ] All files are committed
- [ ] Requirements.txt is up-to-date
- [ ] Example config file works
- [ ] Evaluation script produces results

## Bonus Points (If Time)

- [ ] Add W&B logging for better tracking
- [ ] Implement early stopping
- [ ] Add beam search for inference
- [ ] Try different models (Qwen, DeepSeek)
- [ ] Implement curriculum learning
- [ ] Add self-correction loop (Arctic-R1)
- [ ] Visualize training curves
- [ ] Error analysis on failure cases

## Time Estimate

| Task | Time | Priority |
|------|------|----------|
| Setup + Paper Reading | 4h | Critical |
| Reward Function | 4h | Critical |
| GRPO Trainer | 8h | Critical |
| Integration | 4h | Critical |
| Optimization | 4h | Critical |
| Experiments | 8h | Critical |
| Documentation | 4h | Critical |
| **Total** | **36h** | **~5 days** |

## Success Metrics

**Minimum Viable:**
- ✅ Code runs without errors
- ✅ RL training loop works
- ✅ Rewards improve during training
- ✅ Final model shows improvement over baseline

**Good:**
- ✅ All of the above
- ✅ Execution accuracy improves by 5%+
- ✅ Well-documented code and README
- ✅ Ablation studies (with/without partial rewards)

**Excellent:**
- ✅ All of the above
- ✅ Execution accuracy improves by 10%+
- ✅ Comprehensive analysis of results
- ✅ Novel insights or improvements
- ✅ Clean, production-ready code

Good luck! 🚀
