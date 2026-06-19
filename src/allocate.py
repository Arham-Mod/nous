import json
import numpy as np
from typing import Dict

def allocate_ranks(
    sensitivity: Dict[str, float],
    default_ranks: int= 8,
    min_rank: int = 2,
    max_rank: int = 16
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
    for name, rank in rescaled.items():
        power = round(np.log2(rank))
        power = max(1, power)  # minimum power of 2 is 2^1 = 2
        allocated[name] = int(2 ** power)

    # fix any remaining budget difference due to rounding
    # add or remove rank from least sensitive layers to match budget
    final_total = sum(allocated.values())
    difference = total_budget - final_total

    if difference != 0:
        # sort by sensitivity score
        sorted_by_sensitivity = sorted(
            sensitivity.items(),
            key=lambda x: x[1]
        )

        # if we are under budget add rank to most sensitive layers
        # if over budget remove rank from least sensitive layers
        if difference > 0:
            layers_to_adjust = [n for n, _ in reversed(sorted_by_sensitivity)]
        else:
            layers_to_adjust = [n for n, _ in sorted_by_sensitivity]

        for name in layers_to_adjust:
            if difference == 0:
                break
            current = allocated[name]
            if difference > 0:
                # increase rank to next power of 2
                next_rank = min(max_rank, current * 2)
                allocated[name] = next_rank
                difference -= (next_rank - current)
            else:
                # decrease rank to previous power of 2
                prev_rank = max(min_rank, current // 2)
                allocated[name] = prev_rank
                difference += (current - prev_rank)

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

    # Count how many layers got each rank
    from collections import Counter
    counts = Counter(ranks)
    print("Rank distribution:")
    for rank in sorted(counts.keys()):
        bar = "█" * counts[rank]
        print(f"  rank {rank:2d}: {counts[rank]:3d} layers  {bar}")


if __name__ == "__main__":

    # ── TEST 1: fake uniform sensitivity ─────────────────────────────
    print("TEST 1: Uniform sensitivity (all layers equal)")
    print("Expected: all layers should get rank 8")
    print()

    fake_uniform = {f"layer_{i}": 1.0 for i in range(64)}
    result = allocate_ranks(fake_uniform)
    summarize_allocation(result)

    # ── TEST 2: fake varied sensitivity ──────────────────────────────
    print("\nTEST 2: Varied sensitivity")
    print("Expected: high sensitivity layers get rank 16")
    print("          low sensitivity layers get rank 2 or 4")
    print()

    np.random.seed(42)
    fake_varied = {}
    for i in range(32):
        fake_varied[f"layer_{i}_v_proj"] = np.random.uniform(0.3, 1.6)
        fake_varied[f"layer_{i}_q_proj"] = np.random.uniform(0.01, 0.3)

    result = allocate_ranks(fake_varied)
    summarize_allocation(result)

    # Show top 5 and bottom 5
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

