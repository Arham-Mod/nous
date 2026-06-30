import json
import numpy as np
from typing import Dict
import math


def _allocate_group(
    sensitivity: Dict[str, float],
    default_ranks: int,
    min_rank: int,
    max_rank: int
) -> Dict[str, int]:
    num_layers = len(sensitivity)
    total_budget = default_ranks * num_layers
    total_sensitivity = sum(sensitivity.values())

    raw_ranks = {}
    for name, score in sensitivity.items():
        proportion = score / total_sensitivity
        raw_ranks[name] = proportion * total_budget

    clamped = {}
    for name, raw in raw_ranks.items():
        clamped[name] = max(min_rank, min(max_rank, raw))

    current_total = sum(clamped.values())
    scale_factor = total_budget / current_total

    rescaled = {}
    for name, rank in clamped.items():
        rescaled[name] = clamped[name] * scale_factor

    # round to power of 2
    allocated = {}
    rank_floor_ceil = {}

    for name, rank in rescaled.items():
        lower_power = max(1, math.floor(np.log2(rank)))
        upper_power = lower_power + 1

        lower_rank = 2 ** lower_power
        upper_rank = 2 ** upper_power

        lower_rank = max(min_rank, min(max_rank, lower_rank))
        upper_rank = max(min_rank, min(max_rank, upper_rank))

        rank_floor_ceil[name] = (lower_rank, upper_rank, rank)

    # start everyone at floor
    for name, (lower, upper, raw) in rank_floor_ceil.items():
        allocated[name] = lower

    current_total = sum(allocated.values())
    remaining_budget = total_budget - current_total

    closeness = {}
    for name, (lower, upper, raw) in rank_floor_ceil.items():
        if upper == lower:
            closeness[name] = 0
        else:
            closeness[name] = (raw - lower) / (upper - lower)

    sorted_by_closeness = sorted(
        closeness.items(), key=lambda x: x[1], reverse=True
    )

    for name, _ in sorted_by_closeness:
        if remaining_budget <= 0:
            break
        lower, upper, raw = rank_floor_ceil[name]
        if upper > allocated[name]:
            cost = upper - allocated[name]
            if cost <= remaining_budget:
                allocated[name] = upper
                remaining_budget -= cost

    return allocated


def allocate_ranks(
    sensitivity: Dict[str, float],
    default_ranks: int = 8,
    min_rank: int = 4,       # raised from 2 to prevent starvation
    max_rank: int = 16
) -> Dict[str, int]:
    """
    Allocate LoRA ranks based on sensitivity scores.
    q_proj and v_proj are allocated separately with independent budgets
    so that one projection type cannot starve the other.
    """

    # split into q and v groups
    q_scores = {k: v for k, v in sensitivity.items() if 'q_proj' in k}
    v_scores = {k: v for k, v in sensitivity.items() if 'v_proj' in k}
    other_scores = {k: v for k, v in sensitivity.items()
                    if 'q_proj' not in k and 'v_proj' not in k}

    allocated = {}

    if q_scores:
        q_allocated = _allocate_group(q_scores, default_ranks, min_rank, max_rank)
        allocated.update(q_allocated)

    if v_scores:
        v_allocated = _allocate_group(v_scores, default_ranks, min_rank, max_rank)
        allocated.update(v_allocated)

    # fallback: any other modules get allocated together
    if other_scores:
        other_allocated = _allocate_group(other_scores, default_ranks, min_rank, max_rank)
        allocated.update(other_allocated)

    return allocated


def summarize_allocation(allocated: Dict[str, int]):
    """Print a summary of the rank allocation."""
    ranks = list(allocated.values())

    print("=== RANK ALLOCATION SUMMARY ===")
    print(f"Total layers:     {len(allocated)}")
    print(f"Total rank units: {sum(ranks)}")
    print(f"Min rank:         {min(ranks)}")
    print(f"Max rank:         {max(ranks)}")
    print(f"Mean rank:        {np.mean(ranks):.2f}")
    print()

    from collections import Counter
    counts = Counter(ranks)
    print("Rank distribution:")
    for rank in sorted(counts.keys()):
        bar = "█" * counts[rank]
        print(f"  rank {rank:2d}: {counts[rank]:3d} layers  {bar}")

    # also show q vs v breakdown
    q_ranks = [v for k, v in allocated.items() if 'q_proj' in k]
    v_ranks = [v for k, v in allocated.items() if 'v_proj' in k]
    if q_ranks:
        print(f"\nq_proj — min: {min(q_ranks)}  max: {max(q_ranks)}  mean: {np.mean(q_ranks):.2f}")
    if v_ranks:
        print(f"v_proj — min: {min(v_ranks)}  max: {max(v_ranks)}  mean: {np.mean(v_ranks):.2f}")


if __name__ == "__main__":

    # ── TEST 1: fake uniform sensitivity ─────────────────────────────
    print("TEST 1: Uniform sensitivity (all layers equal)")
    print("Expected: all layers should get rank 8")
    print()

    fake_uniform = {f"layer_{i}": 1.0 for i in range(64)}
    result = allocate_ranks(fake_uniform)
    summarize_allocation(result)

    # ── TEST 2: fake varied sensitivity ──────────────────────────────
    print("\nTEST 2: Varied sensitivity (q_proj and v_proj separate budgets)")
    print("Expected: variation WITHIN q_proj across layers")
    print("          variation WITHIN v_proj across layers")
    print("          q and v each centered around rank 8")
    print()

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

    # ── TEST 3: budget check ─────────────────────────────────────────
    print("\nTEST 3: Budget check")
    print("Expected: total rank units roughly equal to 64 * 8 = 512")

    uniform_budget = 64 * 8
    adaptive_budget = sum(result.values())
    print(f"Uniform budget:  {uniform_budget}")
    print(f"Adaptive budget: {adaptive_budget}")
    print(f"Difference:      {abs(uniform_budget - adaptive_budget)}")
    print("Note: small difference expected due to clamping and rounding")