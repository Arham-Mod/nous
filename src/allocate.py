import json
import math
from collections import Counter
from typing import Dict

import numpy as np


def allocate_ranks(sensitivity: Dict[str, float], default_ranks=8, min_rank=2, max_rank=16) -> Dict[str, int]:
    num_layers = len(sensitivity)
    total_budget = default_ranks * num_layers
    total_sensitivity = sum(sensitivity.values())

    raw_ranks = {name: (score / total_sensitivity) * total_budget for name, score in sensitivity.items()}
    clamped = {name: max(min_rank, min(max_rank, raw)) for name, raw in raw_ranks.items()}

    scale = total_budget / sum(clamped.values())
    #rescaling because of clamping
    rescaled = {name: rank * scale for name, rank in clamped.items()}

    bounds = {}
    for name, rank in rescaled.items():
        lower_pow = max(1, math.floor(np.log2(rank)))
        lower = max(min_rank, min(max_rank, 2 ** lower_pow))
        upper = max(min_rank, min(max_rank, 2 ** (lower_pow + 1)))
        bounds[name] = (lower, upper, rank)

    allocated = {name: lower for name, (lower, upper, raw) in bounds.items()}
    remaining = total_budget - sum(allocated.values())

    def closeness(item):
        lower, upper, raw = bounds[item[0]]
        return 0 if upper == lower else (raw - lower) / (upper - lower)

    for name, _ in sorted(bounds.items(), key=closeness, reverse=True):
        if remaining <= 0:
            break
        lower, upper, raw = bounds[name]
        cost = upper - allocated[name]
        if upper > allocated[name] and cost <= remaining:
            allocated[name] = upper
            remaining -= cost

    return allocated


def summarize_allocation(allocated: Dict[str, int]):
    ranks = list(allocated.values())

    print("=== RANK ALLOCATION SUMMARY ===")
    print(f"Total layers:     {len(allocated)}")
    print(f"Total rank units: {sum(ranks)}")
    print(f"Min rank:         {min(ranks)}")
    print(f"Max rank:         {max(ranks)}")
    print(f"Mean rank:        {np.mean(ranks):.2f}\n")

    counts = Counter(ranks)
    print("Rank distribution:")
    for rank in sorted(counts):
        print(f"  rank {rank:2d}: {counts[rank]:3d} layers  {'█' * counts[rank]}")


if __name__ == "__main__":
    print("TEST 1: Uniform sensitivity (all layers equal)")
    print("Expected: all layers should get rank 8\n")

    fake_uniform = {f"layer_{i}": 1.0 for i in range(64)}
    result = allocate_ranks(fake_uniform)
    summarize_allocation(result)

    print("\nTEST 2: Varied sensitivity")
    print("Expected: high sensitivity layers get rank 16")
    print("          low sensitivity layers get rank 2 or 4\n")

    np.random.seed(42)
    fake_varied = {}
    for i in range(32):
        fake_varied[f"layer_{i}_v_proj"] = np.random.uniform(0.3, 1.6)
        fake_varied[f"layer_{i}_q_proj"] = np.random.uniform(0.01, 0.3)

    result = allocate_ranks(fake_varied)
    summarize_allocation(result)

    sorted_layers = sorted(result.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 highest rank layers:")
    for name, rank in sorted_layers[:5]:
        print(f"  {name}: rank {rank}")

    print("\nBottom 5 lowest rank layers:")
    for name, rank in sorted_layers[-5:]:
        print(f"  {name}: rank {rank}")

    print("\nTEST 3: Budget check")
    print("Expected: total rank units roughly equal to 64 * 8 = 512")

    uniform_budget = 64 * 8
    adaptive_budget = sum(result.values())
    print(f"Uniform budget:  {uniform_budget}")
    print(f"Adaptive budget: {adaptive_budget}")
    print(f"Difference:      {abs(uniform_budget - adaptive_budget)}")
    print("Note: small difference expected due to clamping and rounding")