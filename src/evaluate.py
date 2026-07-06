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

def evaluate(model, tokenizer, test_data, max_length=256):
    correct = 0
    total = 0
    results = []

    for example in tqdm(test_data, desc="Evaluating"):
        prompt = example['prompt']
        true_label_idx = example['label']
        true_label_name = LABEL_NAMES[true_label_idx]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,        # greedy decoding
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = inputs['input_ids'].shape[1]
        generated = tokenizer.decode(
            outputs[0][prompt_length:],
            skip_special_tokens=True
        ).strip()

        predicted_label = extract_label(generated)
        is_correct = (predicted_label == true_label_name)

        if is_correct:
            correct += 1
        total += 1

        results.append({
            'true':      true_label_name,
            'predicted': predicted_label,
            'generated': generated,
            'correct':   is_correct
        })

    accuracy = correct / total
    return accuracy, results

def compute_per_category(results):
    """Compute accuracy per category."""
    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for r in results:
        category_total[r['true']] += 1
        if r['correct']:
            category_correct[r['true']] += 1

    breakdown = {}
    for cat in LABEL_NAMES:
        total = category_total[cat]
        correct = category_correct[cat]
        breakdown[cat] = {
            'accuracy': correct / total if total > 0 else 0,
            'correct':  correct,
            'total':    total
        }

    return breakdown

def compute_confusion_matrix(results):
    """
    Compute confusion matrix.
    """
    matrix = defaultdict(lambda: defaultdict(int))

    for r in results:
        true      = r['true']
        predicted = r['predicted'] if r['predicted'] else 'None'
        matrix[true][predicted] += 1

    return {true: dict(preds) for true, preds in matrix.items()}

def print_results(experiment_name, accuracy, breakdown, baseline_acc=None):
    """Print formatted results to terminal."""
    print(f"\n{'='*50}")
    print(f"RESULTS: {experiment_name}")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"\nPer-category breakdown:")

    for cat in LABEL_NAMES:
        b = breakdown[cat]
        acc = b['accuracy'] * 100
        print(f"  {cat:10s}: {acc:.1f}%  ({b['correct']}/{b['total']})") #printing formatted result by cat*10

    if baseline_acc is not None:
        diff = accuracy - baseline_acc
        print(f"\nVs Baseline ({baseline_acc*100:.2f}%): {diff*100:+.2f}%")
        if diff > 0:
            print("Beats baseline")
        elif diff == 0:
            print("Matches baseline")
        else:
            print("Below baseline")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained LoRA model on AG News test set'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config yaml file'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='results/tables/baseline-uniform-r8_results.json',
        help='Path to baseline results for comparison (optional)'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config['experiment_name']
    print(f"Evaluating experiment: {experiment_name}")

    model, tokenizer = load_trained_model(config)
    test_data = load_test_data(config)

    print("\nRunning evaluation...")
    accuracy, results = evaluate(
        model,
        tokenizer,
        test_data,
        max_length=config['training']['max_seq_length']
    )

    breakdown = compute_per_category(results)
    confusion_matrix = compute_confusion_matrix(results)

    baseline_acc = None
    if os.path.exists(args.baseline) and experiment_name != 'baseline-uniform-r8':
        with open(args.baseline, 'r') as f:
            baseline_data = json.load(f)
        baseline_acc = baseline_data['accuracy']

    print_results(experiment_name, accuracy, breakdown, baseline_acc)

    print(f"\nConfusion matrix:")
    print(f"{'True \\ Pred':12s}", end="")
    all_preds = LABEL_NAMES + ['None']
    for p in all_preds:
        print(f"{p:12s}", end="")
    print()

    for true_cat in LABEL_NAMES:
        print(f"{true_cat:12s}", end="")
        for pred_cat in all_preds:
            count = confusion_matrix.get(true_cat, {}).get(pred_cat, 0)
            print(f"{count:<12d}", end="")
        print()

    output_path = os.path.join(
        'results', 'tables', f"{experiment_name}_results.json"
    )
    save_json({
        'experiment_name': experiment_name,
        'method': config['method'],
        'accuracy': accuracy,
        'correct': sum(1 for r in results if r['correct']),
        'total': len(results),
        'per_category': breakdown,
        'confusion_matrix': confusion_matrix
    }, output_path)

    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()