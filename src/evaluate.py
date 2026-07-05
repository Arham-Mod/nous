"""
usage:
    python src/evaluate.py --config configs/baseline_lora.yaml
    python src/evaluate.py --config configs/adaptive_lora.yaml

"""

import argparse
import json
import os
import sys
import torch
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, save_json

LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
LABEL_CHECKS = {
    'World': ['world'],
    'Sports': ['sports'],
    'Business': ['business', 'Busin'],
    'Sci/Tech': ['sci', 'tech', 'sci/tech', 'science', 'technology', 'sci/t', 'sci/tech']
}

def extract_label(generated_text):
    generated_lower = generated_text.lower().strip()

    for label_name, variants in LABEL_CHECKS.items():
        for variant in variants:
            if variant in generated_lower:
                return label_name

    return None

def load_trained_model(config):
    experiment_name  = config['experiment_name']
    model_name       = config['model']['name']
    checkpoint_path  = os.path.join(
        'results', 'checkpoints', experiment_name
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}\n"
            f"Run training first: "
            f"python src/train.py --config {args.config}"
        )

    print(f"Loading adapter from {checkpoint_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    print(f"Model loaded. Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model, tokenizer

def load_test_data(config):
    dataset   = load_dataset(config['training']['dataset'])
    test_size = config['training']['test_size']
    seed      = config['training']['shuffle_seed']

    def format_for_evaluation(example):
        text = (
            "Classify the following news article into one of "
            "these categories: World, Sports, Business, Sci/Tech.\n\n"
            f"Article: {example['text']}\n\n"
            "Category:"
        )
        return {"prompt": text, "label": example['label']}

    test_data      = dataset['test'].shuffle(seed=seed).select(range(test_size))
    test_formatted = test_data.map(format_for_evaluation)

    print(f"Test examples loaded: {len(test_formatted)}")
    return test_formatted

