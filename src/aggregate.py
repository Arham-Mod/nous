import json


def aggregate_sensitivity_scores(sensitivity_scores):
    aggregated = {}
    for name, score in sensitivity_scores.items():

        parts = name.split(".")
        layer_num = parts[4]
        proj_type = parts[6]

        key = f"layer_{layer_num}_{proj_type}"

        if key not in aggregated:
            aggregated[key] = 0.0

        aggregated[key] += score

    return aggregated


def main():
    with open("results/sensitivity_scores.json", "r") as f:
        data = json.load(f)

    aggregated = aggregate_sensitivity_scores(data)

    with open("results/sensitivity_aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Saved {len(aggregated)} entries")

    print("\nTop 10 layers:")
    for name, score in sorted(
        aggregated.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"{score:.4f}  {name}")


if __name__ == "__main__":
    main()