import json
from allocate import allocate_ranks, summarize_allocation

# Load real aggregated sensitivity scores
with open('results/sensitivity_aggregated.json', 'r') as f:
    real_sensitivity = json.load(f)

print(f"Loaded {len(real_sensitivity)} real sensitivity scores\n")

# Run your tested allocation function
real_ranks = allocate_ranks(real_sensitivity)
summarize_allocation(real_ranks)

# Show top and bottom
sorted_real = sorted(real_ranks.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 highest rank:")
for name, rank in sorted_real[:10]:
    print(f"  {name}: rank {rank}")

print("\nBottom 10 lowest rank:")
for name, rank in sorted_real[-10:]:
    print(f"  {name}: rank {rank}")

# Save final rank allocation
with open('results/rank_allocation.json', 'w') as f:
    json.dump(real_ranks, f, indent=2)

print(f"\nSaved to results/rank_allocation.json")