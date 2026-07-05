"""
Main training file for all three experiments
"""
import argparse
import json
import os
import sys
import random
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from peft import(
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    load_config,
    save_json,
    load_model_and_tokenizer,
    format_ag_news,
    tokenize_dataset
)

# Lora config building

def build_lora_config(config):
    """
    Uniform = same rank everywhere
    Adaptive = different rank for each layer based on sensitivity scores
    random = random rank for each layer 
    """
    method = config['method']
    target_modules = config['lora']['target_modules']
    dropout        = config['lora']['dropout']
    bias           = config['lora']['bias']
    task_type      = config['lora']['task_type']

    # Uniform method
    if method == 'uniform':
        rank  = config['lora']['rank']
        alpha = config['lora']['alpha']

        print(f"Building uniform LoRA config (rank={rank})")
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type
        )
    
    elif method == 'adaptive':
        default_rank = config['lora']['default_rank']
        alpha_mult = config['lora']['alpha_multiplier']
        rank_source = config['lora']['rank_source']
        
        print(f"building rank allocation from {rank_source}")
        with open(rank_source, 'r') as f:
            rank_allocation = json.load(f)

        print(f"Loaded {len(rank_allocation)} rank assignments")

        rank_pattern = {}
        alpha_pattern = {}
        
        for key, rank in rank_allocation.items():
            parts = key.split('_')
            layer_num = parts[1]
            proj_type = parts[2] + '_' + parts[3]
            peft_key = f"model.layers.{layer_num}.self_attn.{proj_type}"

            rank_pattern[peft_key] = rank
            alpha_pattern[peft_key] = rank * alpha_mult
        
        print(f"Rank pattern built: {len(rank_pattern)} entries")
        print(f"Sample ranks:")
        # check key value entries in rank_pattern
        for k, v in list(rank_pattern.items())[:3]:
            print(f" {k}: {v}")

        return LoraConfig(
            r=default_rank,
            lora_alpha=default_rank * alpha_mult,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern
        )
    
    elif method == 'random':
        default_rank = config['lora']['default_rank']
        alpha_mult   = config['lora']['alpha_multiplier']
        min_rank     = config['lora']['min_rank']
        max_rank     = config['lora']['max_rank']
        rand_seed    = config['lora']['random_seed']

        random.seed(rand_seed)

        valid_ranks = [r for r in [2, 4, 8, 16]
        if min_rank <= r <= max_rank]

        total_budget = 64 * default_rank

        layer_names = []
        for i in range(32):
            layer_names.append(
                f"model.layers.{i}.self_attn.q_proj"
            )
            layer_names.append(
                f"model.layers.{i}.self_attn.v_proj"
            )
        
        random_ranks = [
            random.choice(valid_ranks) for _ in layer_names
        ]
        actual_budget = sum(random_ranks)
        print(f"Random allocation budget: {actual_budget} "
                f"(target: {total_budget})")

        rank_pattern  = dict(zip(layer_names, random_ranks))
        alpha_pattern = {k: v * alpha_mult
                        for k, v in rank_pattern.items()}
        
        os.makedirs("results", exist_ok=True)
        save_json(
            rank_pattern,
            "results/rank_allocation_random.json"
        )
        print(f"Saved random rank allocation to results/rank_allocation_random.json")

        return LoraConfig(
            r=default_rank,
            lora_alpha=default_rank * alpha_mult,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias=bias,
            task_type=task_type,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
def validate(model, val_loader):
    """
    Compute average loss on validation set.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()

    model.train()
    return total_loss / len(val_loader)

def train(model, dataloader, val_loader, config, save_path):
    """
    Main training loop with checkpointing and validation.
    """
    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    checkpoint_every = config['training']['checkpoint_every']
    total_steps = len(dataloader) * epochs

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    checkpoint_file = os.path.join(save_path, 'training_state.pt')

    if os.path.exists(checkpoint_file):
        print("Checkpoint found — resuming training...")
        checkpoint = torch.load(checkpoint_file)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_step = checkpoint['step']
        losses = checkpoint['losses']
        val_losses = checkpoint.get('val_losses', [])  
        print(f"Resuming from step {start_step}/{total_steps}")
    else:
        print("No checkpoint found — starting fresh")
        start_step = 0
        losses = []
        val_losses = []

    model.train()
    step = 0

    validate_every = config['training'].get('validate_every', 500)

    for epoch in range(epochs):
        print(f"\n=== EPOCH {epoch+1}/{epochs} ===")

        for batch in tqdm(dataloader):
            step += 1

            if step <= start_step:
                continue

            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                print(f"Step {step}/{total_steps} | Loss: {avg_loss:.4f}")

            # validation
            if step % validate_every == 0:
                val_loss = validate(model, val_loader)
                val_losses.append({
                    'step': step,
                    'val_loss': val_loss
                })
                print(f"  Validation loss at step {step}: {val_loss:.4f}")

            # checkpoint
            if step % checkpoint_every == 0:
                model.save_pretrained(save_path)
                torch.save({
                    'step': step,
                    'optimizer_state': optimizer.state_dict(),
                    'losses': losses,
                    'val_losses': val_losses,    
                }, checkpoint_file)
                print(f"  Checkpoint saved at step {step}")

    return losses, val_losses

    # final save
    model.save_pretrained(save_path)
    print(f"\nTraining complete. Model saved to {save_path}")

    return losses



def main():
    parser = argparse.ArgumentParser(
        description="Train a LoRA model with specified configuration."
    )
    parser.add_argument(
        '--config', type=str, required=True,help = 'Path to the config yaml file.'
    )
    args = parser.parse_args()

    config          = load_config(args.config)
    experiment_name = config['experiment_name']
    method          = config['method']

    print(f"Experiment: {experiment_name}")
    print(f"Method: {method}")

    save_path = os.path.join(
        'results', 'checkpoints', experiment_name
    )
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('results/tables', exist_ok=True)

    print("\nLoading model..")
    model, tokenizer = load_model_and_tokenizer(config)
    model = prepare_model_for_kbit_training(model)

    print("\nBuilding LoRA config...")
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset and tokenize
    print("\nLoading dataset...")
    dataset   = load_dataset(config['training']['dataset'])
    # build train dataloader
    train_data = format_ag_news(
        dataset,
        split='train',
        size=config['training']['train_size'],
        seed=config['training']['shuffle_seed']
    )

    val_size = config['training'].get('val_size', 500)
    val_data = format_ag_news(
        dataset,
        split='train',
        size=config['training']['train_size'] + val_size,
        seed=config['training']['shuffle_seed']
    )

    val_data = dataset['train'].shuffle(
        seed=config['training']['shuffle_seed'] + 1
    ).select(range(val_size))
    val_size = config['training'].get('val_size', 500)
    val_data = format_ag_news(
        dataset,
        split='train',
        size=val_size,
        seed=config['training']['shuffle_seed'] + 1
    )

    tokenized_train = tokenize_dataset(
        train_data, tokenizer,
        max_length=config['training']['max_seq_length']
    )
    tokenized_val = tokenize_dataset(
        val_data, tokenizer,
        max_length=config['training']['max_seq_length']
    )

    train_loader = DataLoader(
        tokenized_train,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized_val,
        batch_size=config['training']['batch_size'],
        shuffle=False    # no shuffle for validation
    )

    print(f"Training examples: {len(tokenized_train)}")
    print(f"Validation examples: {len(tokenized_val)}")
    print(f"Steps per epoch:   {len(train_loader)}")
    print(f"Total steps:       {len(train_loader) * config['training']['epochs']}")

    print("\nStarting training...")
    losses, val_losses = train(model, train_loader, val_loader, config, save_path)

    tokenizer.save_pretrained(save_path)

    loss_path = os.path.join(
        'results', f"{experiment_name}_losses.json"
    )
    save_json(losses, loss_path)

    print(f"\nDone.")
    print(f"Model:  {save_path}")
    print(f"Losses: {loss_path}")
    save_json(losses, os.path.join('results', f"{experiment_name}_losses.json"))

    save_json(val_losses, os.path.join('results', f"{experiment_name}_val_losses.json"))

if __name__ == "__main__":
    main()
