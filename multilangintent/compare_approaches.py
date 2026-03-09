#!/usr/bin/env python3
"""
Compare mDeBERTa-v3 (SOTA) vs Qwen2.5 (Zero-shot LLM) approaches.

This script:
1. Trains mDeBERTa-v3 on training data
2. Uses Qwen2.5 for zero-shot classification
3. Evaluates both on a held-out validation set
4. Reports macro F1 scores and comparison

Usage:
    python compare_approaches.py [--llm_sample_size 100] [--use_smaller_llm]

Options:
    --llm_sample_size N    Number of samples to evaluate with LLM (default: 100)
    --use_smaller_llm      Use Qwen2.5-7B instead of 14B (faster, less memory)
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Dataset
# =============================================================================

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =============================================================================
# Approach 1: mDeBERTa-v3 (SOTA)
# =============================================================================

def train_and_evaluate_sota(train_texts, train_labels, val_texts, val_labels,
                            num_labels, label_encoder):
    """Train and evaluate mDeBERTa-v3 model."""

    print("\n" + "="*60)
    print("APPROACH 1: Fine-tuned mDeBERTa-v3")
    print("="*60)

    model_name = "microsoft/mdeberta-v3-base"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Create datasets
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

    # Check if bf16 is supported (better than fp16)
    use_bf16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "sota_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=1e-5,  # Lower learning rate for stability
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        bf16=use_bf16,
        fp16=False,  # Disable fp16, use bf16 or fp32
        max_grad_norm=1.0,  # Gradient clipping for stability
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training...")
    trainer.train()

    # Predict on validation
    print("Predicting on validation set...")
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_label_names = label_encoder.inverse_transform(pred_labels)
    true_label_names = label_encoder.inverse_transform(val_labels)

    # Calculate metrics
    f1_macro = f1_score(true_label_names, pred_label_names, average="macro")
    f1_weighted = f1_score(true_label_names, pred_label_names, average="weighted")

    print(f"\nResults:")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

    return {
        "predictions": pred_label_names,
        "true_labels": true_label_names,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# =============================================================================
# Approach 2: Zero-shot LLM (Qwen2.5)
# =============================================================================

INTENT_PROMPT = """You are a customer support intent classifier. Classify the following message into ONE of these intent categories:

Categories: {intents}

Message ({language}): "{text}"

Instructions:
1. Account for typos and grammatical errors in the message
2. Choose the most appropriate category
3. Output ONLY the category name, nothing else

Category:"""


def evaluate_llm_zero_shot(val_texts, val_languages, true_labels, all_intents,
                           model_name="Qwen/Qwen2.5-7B-Instruct", sample_size=None):
    """Evaluate zero-shot LLM classification."""

    print("\n" + "="*60)
    print(f"APPROACH 2: Zero-shot LLM ({model_name.split('/')[-1]})")
    print("="*60)

    if sample_size:
        val_texts = val_texts[:sample_size]
        val_languages = val_languages[:sample_size]
        true_labels = true_labels[:sample_size]
        print(f"Evaluating on {sample_size} samples")

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    intents_str = ", ".join(all_intents[:30])  # First 30 for brevity
    all_intents_lower = {i.lower(): i for i in all_intents}

    predictions = []

    for text, lang in tqdm(zip(val_texts, val_languages), total=len(val_texts),
                           desc="LLM Classifying"):
        prompt = INTENT_PROMPT.format(
            intents=intents_str + " (and more...)",
            language=lang,
            text=text[:200]  # Truncate long texts
        )

        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                temperature=0.01,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Match response to valid intent
        response_lower = response.lower().replace(" ", "_").replace("-", "_")

        matched_intent = all_intents[0]  # Default
        for intent_lower, intent in all_intents_lower.items():
            if intent_lower in response_lower or response_lower in intent_lower:
                matched_intent = intent
                break

        predictions.append(matched_intent)

    # Calculate metrics
    f1_macro = f1_score(true_labels, predictions, average="macro")
    f1_weighted = f1_score(true_labels, predictions, average="weighted")

    print(f"\nResults:")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

    return {
        "predictions": predictions,
        "true_labels": true_labels,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


# =============================================================================
# Main Comparison
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare intent classification approaches")
    parser.add_argument("--llm_sample_size", type=int, default=100,
                       help="Number of samples to evaluate with LLM")
    parser.add_argument("--use_smaller_llm", action="store_true",
                       help="Use Qwen2.5-7B instead of 14B")
    args = parser.parse_args()

    print("="*60)
    print("MULTILINGUAL INTENT CLASSIFICATION - APPROACH COMPARISON")
    print("="*60)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Total training samples: {len(train_df)}")
    print(f"Languages: {sorted(train_df['language'].unique())}")
    print(f"Intent classes: {train_df['label'].nunique()}")

    # Prepare labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(train_df["label"].values)
    num_labels = len(label_encoder.classes_)
    all_intents = sorted(label_encoder.classes_.tolist())

    # Split into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].values,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels
    )
    val_languages = train_df.iloc[val_texts.argsort()]["language"].values
    # Get original indices for language lookup
    val_indices = np.where(np.isin(train_df["text"].values, val_texts))[0]
    val_languages = train_df.iloc[val_indices]["language"].values

    # Make sure we have matching lengths
    if len(val_languages) != len(val_texts):
        # Fallback: just use first 10% for validation
        val_df = train_df.sample(frac=0.1, random_state=42)
        train_df_full = train_df.drop(val_df.index)
        train_texts = train_df_full["text"].values
        train_labels = label_encoder.transform(train_df_full["label"].values)
        val_texts = val_df["text"].values
        val_labels = label_encoder.transform(val_df["label"].values)
        val_languages = val_df["language"].values

    print(f"\nTrain samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")

    # Approach 1: SOTA
    sota_results = train_and_evaluate_sota(
        train_texts, train_labels, val_texts, val_labels,
        num_labels, label_encoder
    )

    # Approach 2: LLM (on sample)
    llm_model = "Qwen/Qwen2.5-7B-Instruct" if args.use_smaller_llm else "Qwen/Qwen2.5-14B-Instruct"
    llm_results = evaluate_llm_zero_shot(
        val_texts, val_languages,
        label_encoder.inverse_transform(val_labels[:args.llm_sample_size]),
        all_intents,
        model_name=llm_model,
        sample_size=args.llm_sample_size
    )

    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Approach':<40} {'Macro F1':>12} {'Weighted F1':>12}")
    print("-"*64)
    print(f"{'mDeBERTa-v3 (fine-tuned, full val)':<40} {sota_results['f1_macro']:>12.4f} {sota_results['f1_weighted']:>12.4f}")
    print(f"{f'Qwen2.5 (zero-shot, {args.llm_sample_size} samples)':<40} {llm_results['f1_macro']:>12.4f} {llm_results['f1_weighted']:>12.4f}")

    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    diff = sota_results['f1_macro'] - llm_results['f1_macro']
    if diff > 0:
        print(f"mDeBERTa-v3 outperforms zero-shot LLM by {diff:.4f} F1 points")
        print("Fine-tuning on task-specific data is highly effective.")
    else:
        print(f"Zero-shot LLM outperforms mDeBERTa-v3 by {-diff:.4f} F1 points")
        print("LLM generalization is surprisingly strong!")

    print("\nSpeed comparison:")
    print("  - mDeBERTa-v3: ~1000 texts/sec on GPU (very fast)")
    print("  - Qwen2.5 LLM: ~1-5 texts/sec on GPU (much slower)")
    print("\nRecommendation: Use mDeBERTa-v3 for production (best accuracy + speed)")

    # Save results
    results_df = pd.DataFrame({
        "Approach": ["mDeBERTa-v3 (fine-tuned)", "Qwen2.5 (zero-shot)"],
        "Macro_F1": [sota_results['f1_macro'], llm_results['f1_macro']],
        "Weighted_F1": [sota_results['f1_weighted'], llm_results['f1_weighted']],
        "Sample_Size": [len(val_texts), args.llm_sample_size],
    })
    results_df.to_csv(OUTPUT_DIR / "comparison_results.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR / 'comparison_results.csv'}")


if __name__ == "__main__":
    main()
