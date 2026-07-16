"""
To measure sensitivity of mistral ai layers using a short warmup training run
"""

import argparse
import json
import os
from pyexpat import model
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import sys
from src.aggregate import aggregate_sensitivity_scores

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (
    load_config,
    save_json,
    load_model_and_tokenizer,
    format_ag_news,
    tokenize_dataset
)

# Can load configs from adaptive_lora it is present there
def apply_uniform_lora(model,config):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config['lora']['default_rank'],
        lora_alpha=config['lora']['default_rank'] * config['lora']['alpha_multiplier'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['dropout'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type']
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def measure_sensitivity(model, dataloader, num_steps=200):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4
    )

    sensitivity_scores = {}
    # Initializes an empty sensitivity score for each LoRA parameter for now zero
    for name, param in model.named_parameters():
        if 'lora' in name and param.requires_grad:
            sensitivity_scores[name] = 0.0
    
    model.train()
    step = 0

    for batch in tqdm(dataloader, desc="Measuring sensitivity"):
        if step >= num_steps:
            break

        new_batch = {}
        for key, value in batch.items():
            new_batch[key] = value.to(model.device)

        batch = new_batch

        outputs = model(**batch) # ** unpacks the dictionary into keyword arguments
        loss = outputs.loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                sensitivity_scores[name] += (param.grad.norm() / param.grad.numel() ** 0.5).item()

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 50 == 0:
            print(f"  Warmup step {step}/{num_steps} | Loss: {loss.item():.4f}")

        for name in sensitivity_scores:
            sensitivity_scores[name] /= step

        print(f"\nWarmup complete. {step} steps run.")
        return sensitivity_scores

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

def print_top_bottom(aggregated, n=10):
    """Print the most and least sensitive layers."""
    sorted_layers = sorted(
        aggregated.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nTop {n} most sensitive layers:")
    for name, score in sorted_layers[:n]:
        print(f"  {score:.4f}  {name}")

    print(f"\nBottom {n} least sensitive layers:")
    for name, score in sorted_layers[-n:]:
        print(f"  {score:.4f}  {name}")


def main():
    # CLI arguments
    parser = argparse.ArgumentParser(
        description='Measure per-layer LoRA sensitivity via gradient norms'
    )
    # Config file
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config yaml file (use adaptive_lora.yaml)'
    )
    # No of warmup steps
    parser.add_argument(
        '--steps',
        type=int,
        default=200,
        help='Number of warmup steps (default: 200)'
    )
    # Output file
    parser.add_argument(
        '--output',
        type=str,
        default='results/sensitivity_aggregated.json',
        help='Where to save aggregated sensitivity scores'
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Config loaded: {args.config}")
    print(f"Warmup steps: {args.steps}")

    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("\nApplying uniform LoRA...")
    model = apply_uniform_lora(model, config)

    print("\nLoading warmup data---")
    dataset = load_dataset(config['training']['dataset'])
    warmup_data = format_ag_news(
        dataset,
        split='train',
        size=config['sensitivity']['warmup_train_size'],
        seed=config['training']['shuffle_seed']
    )
    tokenized = tokenize_dataset(
        warmup_data,
        tokenizer,
        max_length=config['training']['max_seq_length']
    )
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True)

    print("\nRunning sensitivity warmup...")
    raw_sensitivity = measure_sensitivity(
        model,
        dataloader,
        args.steps
    )
    print("\nAggregating scores...")
    aggregated = aggregate_sensitivity_scores(raw_sensitivity)
    save_json(raw_sensitivity, "results/sensitivity_scores.json")

    save_json(aggregated, args.output)
    print(f"Aggregated into {len(aggregated)} layer-projection entries")
    print(f"Expected: 64 (32 layers x 2 projections)")

    print_top_bottom(aggregated)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_json(aggregated, args.output)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()

