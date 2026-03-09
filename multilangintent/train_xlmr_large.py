#!/usr/bin/env python3
"""
XLM-RoBERTa-large training for multilingual intent classification.

XLM-RoBERTa-large (560M params) has better cross-lingual transfer than base.
Using stable training settings with gradient clipping and lower LR.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# XLM-RoBERTa-large for better cross-lingual transfer
MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 128
BATCH_SIZE = 8  # Smaller for large model
LEARNING_RATE = 1e-5  # Lower for stability
EPOCHS = 5  # More epochs for better convergence


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else list(texts)
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def main():
    print("=" * 60)
    print("XLM-RoBERTa-large Multilingual Intent Classification")
    print("Stable training with gradient clipping")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train languages: {train_df['language'].value_counts().to_dict()}")
    print(f"Test languages: {test_df['language'].value_counts().to_dict()}")

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(train_df["label"].values)
    num_labels = len(label_encoder.classes_)
    print(f"\nNumber of intent classes: {num_labels}")

    # Split train into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].values,
        all_labels,
        test_size=0.1,
        random_state=42,
        stratify=all_labels
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification",
    )
    model = model.to(device)

    # Print model size
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.1f}M")

    # Create datasets
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "accuracy": accuracy_score(labels, predictions)
        }

    # Training arguments - stable settings
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "xlmr_large_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=50,
        warmup_ratio=0.1,
        save_total_limit=2,
        report_to="none",
        # Stability settings
        max_grad_norm=1.0,  # Gradient clipping
        fp16=False,
        bf16=torch.cuda.is_available(),  # Use bf16 on supported GPUs
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nTraining XLM-RoBERTa-large...")
    trainer.train()

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate()
    print(f"  Macro F1: {eval_results['eval_macro_f1']:.4f}")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "xlmr_large_intent_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    np.save(model_path / "label_encoder_classes.npy", label_encoder.classes_)
    print(f"\nModel saved to: {model_path}")

    # Predict on test set
    print("\nPredicting on test set...")
    test_dataset = IntentDataset(
        test_df["text"].values,
        [0] * len(test_df),
        tokenizer,
        MAX_LENGTH
    )

    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    pred_label_names = label_encoder.inverse_transform(pred_labels)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "label": pred_label_names
    })
    submission.to_csv(OUTPUT_DIR / "submission_xlmr_large.csv", index=False)
    submission.to_csv("submission.csv", index=False)

    print(f"\nSubmission saved to: submission.csv")
    print(f"Total predictions: {len(submission)}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(5):
        print(f"  [{test_df['language'].iloc[i]}] {test_df['text'].iloc[i][:40]}... -> {pred_label_names[i]}")

    # Show prediction distribution
    print("\nPrediction distribution:")
    print(pd.Series(pred_label_names).value_counts().head(10))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return eval_results['eval_macro_f1']


if __name__ == "__main__":
    main()
