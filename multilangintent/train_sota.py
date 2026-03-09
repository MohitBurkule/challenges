#!/usr/bin/env python3
"""
Simple and robust training script for mDeBERTa-v3 multilingual intent classification.
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
MODEL_NAME = "microsoft/mdeberta-v3-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3


class IntentDataset(Dataset):
    """Simple dataset for intent classification."""

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
    print("="*60)
    print("mDeBERTa-v3 Training for Multilingual Intent Classification")
    print("="*60)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
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
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    # Test a sample
    print("\nTesting dataset sample...")
    sample = train_dataset[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels: {sample['labels'].item()} -> {label_encoder.inverse_transform([sample['labels'].item()])[0]}")

    # Compute metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "macro_f1": f1_score(labels, predictions, average="macro"),
            "accuracy": accuracy_score(labels, predictions)
        }

    # Training arguments - use fp32 for stability
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
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
        logging_steps=100,
        warmup_ratio=0.1,
        save_total_limit=2,
        report_to="none",
        # Disable mixed precision for stability
        fp16=False,
        bf16=False,
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
    print("\nTraining...")
    trainer.train()

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate()
    print(f"  Macro F1: {eval_results['eval_macro_f1']:.4f}")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save model
    save_path = OUTPUT_DIR / "mdeberta_intent_model"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    np.save(save_path / "label_encoder_classes.npy", label_encoder.classes_)
    print(f"\nModel saved to: {save_path}")

    # Make predictions on test set
    print("\nPredicting on test set...")
    test_dataset = IntentDataset(
        test_df["text"].values,
        [0] * len(test_df),  # Dummy labels
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
    submission_path = OUTPUT_DIR / "submission_mdeberta.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to: {submission_path}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(5):
        print(f"  {test_df['text'].iloc[i][:50]}... -> {pred_label_names[i]}")

    return eval_results['eval_macro_f1']


if __name__ == "__main__":
    main()
