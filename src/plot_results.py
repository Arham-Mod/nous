import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

with open('results/sensitivity_aggregated.json', 'r') as f:
    sensitivity = json.load(f)

# build 32x2 grid (layers x projections)
q_scores = []
v_scores = []

for i in range(32):
    q_scores.append(sensitivity.get(f"layer_{i}_q_proj", 0))
    v_scores.append(sensitivity.get(f"layer_{i}_v_proj", 0))

# shape: 2 rows (q, v) x 32 columns (layers)
grid = np.array([q_scores, v_scores])

fig, ax = plt.subplots(figsize=(16, 3))
im = ax.imshow(grid, aspect='auto', cmap='YlOrRd')

ax.set_yticks([0, 1])
ax.set_yticklabels(['q_proj', 'v_proj'])
ax.set_xlabel('Layer Index')
ax.set_title('Per-Layer Sensitivity (Gradient Norm) — Mistral-7B on AG News')

plt.colorbar(im, ax=ax, label='Avg Gradient Norm')
plt.tight_layout()
plt.savefig('results/figures/sensitivity_heatmap.png', dpi=150)
plt.show()
print("Saved: results/figures/sensitivity_heatmap.png")

with open('results/rank_allocation.json', 'r') as f:
    rank_allocation = json.load(f)

# adaptive ranks
adaptive_ranks = list(rank_allocation.values())

# baseline ranks (all 8)
baseline_ranks = [8] * 64

# count distribution
adaptive_counts = Counter(adaptive_ranks)
baseline_counts = Counter(baseline_ranks)

valid_ranks = [2, 4, 8, 16]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# baseline
axes[0].bar(
    valid_ranks,
    [baseline_counts.get(r, 0) for r in valid_ranks],
    color='steelblue', alpha=0.8, width=1.5
)
axes[0].set_title('Baseline — Uniform Rank Distribution')
axes[0].set_xlabel('Rank Value')
axes[0].set_ylabel('Number of Layers')
axes[0].set_xticks(valid_ranks)

# adaptive
axes[1].bar(
    valid_ranks,
    [adaptive_counts.get(r, 0) for r in valid_ranks],
    color='darkorange', alpha=0.8, width=1.5
)
axes[1].set_title('Adaptive — Sensitivity-Based Rank Distribution')
axes[1].set_xlabel('Rank Value')
axes[1].set_ylabel('Number of Layers')
axes[1].set_xticks(valid_ranks)

plt.tight_layout()
plt.savefig('results/figures/rank_distribution.png', dpi=150)
plt.show()
print("Saved: results/figures/rank_distribution.png")

with open('results/baseline_losses.json', 'r') as f:
    losses = json.load(f)

# smooth with moving average
window = 50
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
steps = range(len(smoothed))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(steps, smoothed, color='steelblue', linewidth=1.5,
        label='Training Loss (smoothed)')
ax.axvline(x=2000, color='gray', linestyle='--',
        alpha=0.7, label='Epoch boundary')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('Baseline Training Loss — Mistral-7B LoRA on AG News')
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/baseline_loss_curve.png', dpi=150)
plt.show()
print("Saved: results/figures/baseline_loss_curve.png")

categories = ['World', 'Sports', 'Business', 'Sci/Tech', 'Overall']

baseline_acc = [85.7, 99.6, 89.0, 90.5, 91.10]
adaptive_acc = [88.3, 99.6, 81.3, 95.0, 91.00]

x     = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))

bars1 = ax.bar(x - width/2, baseline_acc, width,
            label='Baseline (uniform r=8)',
            color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, adaptive_acc, width,
            label='Adaptive (sensitivity-based)',
            color='darkorange', alpha=0.8)

ax.set_ylabel('Accuracy (%)')
ax.set_title('Baseline vs Adaptive LoRA — Per-Category Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(70, 105)
ax.legend()

# add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom',
            fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/accuracy_comparison.png', dpi=150)
plt.show()
print("Saved: results/figures/accuracy_comparison.png")

with open('results/sensitivity_aggregated.json', 'r') as f:
    sensitivity = json.load(f)

# separate q and v
q_layers = {k: v for k, v in sensitivity.items() if 'q_proj' in k}
v_layers = {k: v for k, v in sensitivity.items() if 'v_proj' in k}

# sort by layer number
q_sorted = sorted(q_layers.items(), key=lambda x: int(x[0].split('_')[1]))
v_sorted = sorted(v_layers.items(), key=lambda x: int(x[0].split('_')[1]))

layers    = [x[0].split('_')[1] for x in q_sorted]
q_scores  = [x[1] for x in q_sorted]
v_scores  = [x[1] for x in v_sorted]

x     = range(len(layers))
width = 0.4

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar([i - width/2 for i in x], q_scores, width,
    label='q_proj', color='steelblue', alpha=0.8)
ax.bar([i + width/2 for i in x], v_scores, width,
    label='v_proj', color='darkorange', alpha=0.8)

ax.set_xlabel('Layer Index')
ax.set_ylabel('Avg Gradient Norm (200 steps)')
ax.set_title('Per-Layer Sensitivity: q_proj vs v_proj')
ax.set_xticks(list(x))
ax.set_xticklabels(layers, fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/sensitivity_by_layer.png', dpi=150)
plt.show()
print("Saved: results/figures/sensitivity_by_layer.png")


with open('results/adaptive_losses.json', 'r') as f:
    losses = json.load(f)
# smooth with moving average
window = 50
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
steps = range(len(smoothed))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(steps, smoothed, color='steelblue', linewidth=1.5,
        label='Training Loss (smoothed)')
ax.axvline(x=2000, color='gray', linestyle='--',
        alpha=0.7, label='Epoch boundary')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.set_title('Adaptive Training Loss — Mistral-7B LoRA on AG News')
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/adaptive_loss_curve.png', dpi=150)
plt.show()
print("Saved: results/figures/adaptive_loss_curve.png")