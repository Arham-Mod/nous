# NOUS - v1.0

_Sensitivity-Based LoRA Rank Allocation_

> *Standard LoRA gives every transformer layer the same rank.
> Not all layers are equally important for a given task.
> This project asks: can gradient norms tell us which layers deserve more?*
 

## What This Is
 
A gradient-norm-based LoRA rank pre-allocation strategy for transformer
fine-tuning. Instead of assigning the same rank to every layer (standard
LoRA) or continuously recomputing ranks during training via SVD (AdaLoRA),
this method runs a short 200-step warmup, measures how strongly each layer
responds to the task, and allocates rank proportionally — once, before
training begins.
 
The result: same total parameter budget as uniform LoRA, but distributed
according to actual layer sensitivity to the task.
 
---

 
## The Problem


Standard LoRA fine-tuning assigns an identical rank `r` to every transformer
layer. This is simple and effective, but it treats all layers as equally
important — which they are not.
 
AdaLoRA addresses this by dynamically reallocating ranks during training
using SVD decomposition. The problem: SVD at every training step is
computationally expensive, complex to implement, and hard to reproduce.
 
**This project asks a simpler question:** can a short warmup run measure
layer sensitivity well enough to pre-allocate ranks before training starts?
 
---

## The Approach
 
```
Step 1 — Warmup (200 steps)
    Run a short training warmup on the target task.
    Measure gradient norms per layer using RMS normalization.
    Save sensitivity scores to sensitivity_aggregated.json.
 
Step 2 — Rank Allocation (CPU, seconds)
    Allocate ranks proportionally to sensitivity scores.
    q_proj and v_proj receive independent budgets.
    Ranks rounded to powers of 2, clamped to [4, 16].
    Total parameter budget identical to uniform baseline.
    Save to rank_allocation.json.
 
Step 3 — Train (fixed ranks, one full run)
    Train with the pre-allocated ranks — no recomputation.
    Compare against uniform LoRA baseline (r=8 everywhere).
 
Step 4 — Evaluate
    Per-category accuracy on AG News test set.
    Compare adaptive vs baseline vs random allocation.
```
 
---

## Key Findings

### Finding 1 - v_proj is consistently more senstive than q_proj

Across all 32 layers of Mistral-7B, value projection layers showed
significantly higher gradient norms than query projection layers during
the task-specific warmup on AG News.

![Layer Senstivity](results/figures/sensitivity_heatmap.png)
*Sensitivity scores per layer per projection type. v_proj consistently
shows higher sensitivity than q_proj across all 32 layers of Mistral-7B*

This pattern held on Phi-2 (standard MHA) as well as Mistral-7B (GQA),
confirming it reflects genuine fine-tuning dynamics rather than an
architectural artifact.
 
**Why:** v_proj controls what information flows forward through the network.
Fine-tuning on a new task requires recalibrating content extraction.
q_proj controls attention routing patterns from pretraining transfer
more readily to new tasks with minimal adjustment.
 
### Finding 2 — GQA introduces a tensor-size bias in raw gradient norms
 
In Mistral-7B's grouped query attention (32 Q heads, 8 KV heads), q_proj
has 4x more parameters than v_proj. Raw L2 gradient norm scales with
tensor size — a larger tensor produces a larger norm even at identical
per-element gradient magnitude.
 
Without normalization, the sensitivity allocator reads this size difference
as an importance difference and dumps the entire rank budget into q_proj,
starving v_proj entirely.
 
**The fix:** RMS normalization — dividing gradient norm by √(number of
elements) — removes the size dependence and leaves only per-element
gradient magnitude as the signal.

```python
# broken — raw L2, biased by tensor size
sensitivity[name] += param.grad.norm().item()
 
# fixed — RMS, size-independent
sensitivity[name] += (param.grad.norm() / param.grad.numel() ** 0.5).item()
```

This single change took accuracy from 35.5% to 91.0%.

### Finding 3 — Rank redistribution shifts category-level performance

Adaptive allocation achieved comparable overall accuracy to uniform LoRA
while meaningfully redistributing performance across categories:
 
| Method                  | Overall | World | Sports | Business | Sci/Tech |
|-------------------------|---------|-------|--------|----------|----------|
| Baseline (uniform r=8)  | 91.10%  | 85.7% | 99.6%  | 89.0%    | 90.5%    |
| Adaptive (ours)         | 91.00%  | 88.3% | 99.6%  | 81.3%    | 95.0%    |

Sci/Tech improved by +4.5%, World by +2.6%. Business dropped by -7.7%.
![Rank Distribution](results/figures/rank_distribution.png)
*Baseline assigns uniform rank 8 to all 64 layer-projections.
Adaptive redistributes the same total budget based on sensitivity scores,
giving higher rank to more sensitive layers and lower rank to less sensitive ones.*

Sci/Tech improved by +4.5%, World by +2.6%...

Sci/Tech and Business share significant vocabulary (Apple, Google, product,
market, earnings). The rank redistribution sharpened the model's ability to
distinguish between these semantically overlapping categories — improving
one at the cost of the other.

### Finding 4 — Pre-allocation is simpler and more predictable than AdaLoRA
 
One warmup run. Ranks fixed before training begins. No SVD during training.
No dynamic recomputation. Fully reproducible rank assignments stored in a
plain JSON file before the main training run starts.
 
---

## ADD HOW TO RUN HERE

## Project Structure
 
```
adaptive-lora/
├── src/
│   ├── sensitivity.py           # gradient norm measurement (RMS-normalized)
│   ├── aggregate.py             # pool lora_A + lora_B per layer-projection
│   ├── allocate.py              # convert scores to rank assignments
│   ├── train.py                 # training loop (all three experiments)
│   ├── evaluate.py              # evaluation on test set
├── configs/
│   ├── baseline_lora.yaml       # uniform rank=8
│   ├── adaptive_lora.yaml       # sensitivity-based ranks
│   └── random_lora.yaml         # random ablation
├── results/
│   ├── sensitivity_scores.json        # raw per-parameter scores
│   ├── sensitivity_aggregated.json    # 64 layer-projection scores
│   ├── rank_allocation.json           # 64 rank assignments
│   └── figures/                       # plots
├── notebooks/
│   ├── 01_Baseline_Training.ipynb
│   ├── 02_Sensitivity_Warmup.ipynb
│   ├── 03_Adaptive_Training.ipynb
│   └── 04_Evaluation.ipynb
├── requirements.txt
└── README.md
```
 
---