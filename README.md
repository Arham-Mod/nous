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