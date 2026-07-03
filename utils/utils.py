from importlib.resources import path
import torch
import yaml
import json
import numpy as np
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def save_json(data: dict, output_path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {output_path}")

def load_json(json_path: str) -> dict:
    with open(json_path, 'r') as f:
        return json.load(f)
    

def load_model_and_tokenizer(config: dict):
    "loading Mistral-7B in 4bit quant"

    import torch    
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = config['model']['name']

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loaded {model_name}")
    print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return model, tokenizer

def format_ag_news(dataset, split='train', size=8000, seed=42):
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    def format_example(example):
        label = label_names[example['label']]
        text = f"""Classify the following news article into one of these categories: World, Sports, Business, Sci/Tech.

Article: {example['text']}

Category: {label}"""
        return {"text": text}

    formatted = dataset[split].map(format_example)
    subset = formatted.shuffle(seed=seed).select(range(size))
    return subset

def tokenize_dataset(dataset, tokenizer, max_length=256):
    """
    Tokenize formatted dataset for training.
    """
    def tokenize(example):
        result = tokenizer(
            example['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
        result['labels'] = result['input_ids'].copy()
        return result

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )
    tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    return tokenized