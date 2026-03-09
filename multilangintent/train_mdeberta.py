#!/usr/bin/env python3
"""
Cross-lingual intent classification using mDeBERTa-v3-base.
This model has excellent cross-lingual transfer capabilities.

KEY INSIGHT: Train has en/pt/es, Test has de/fr
mDeBERTa-v3-base has the best zero-shot cross-lingual transfer performance.
"""

import warnings
warnings.filterwarnings("ignore")

import gc
import torch
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

# Paths
TRAIN_PATH = Path("public/train.csv")
TEST_PATH = Path("public/test.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission_mdeberta.csv"

# Model - mDeBERTa-v3-base has excellent cross-lingual transfer
MODEL_NAME = "microsoft/mdeberta-v3-base"


def load_data():
    """Load train and test data."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Train: {len(train_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Languages in train: {train_df['language'].value_counts().to_dict()}")
    print(f"Languages in test: {test_df['language'].value_counts().to_dict()}")

    # Important: train and test have different languages!
    train_langs = set(train_df['language'].unique())
    test_langs = set(test_df['language'].unique())
    print(f"\nTrain languages: {train_langs}")
    print(f"Test languages: {test_langs}")
    print(f"Test-only languages (zero-shot): {test_langs - train_langs}")

    return train_df, test_df


def train_classifier(train_df, test_df):
    """Train mDeBERTa-v3 classifier with cross-lingual transfer."""
    print("\n" + "="*60)
    print(f"Training {MODEL_NAME} for cross-lingual intent classification")
    print("="*60)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"Number of intent classes: {num_labels}")

    # Create stratified train/val split
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=0.1,
        random_state=42,
        stratify=train_df["label"]
    )

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_split)}, Val: {len(val_split)}")

    # Load tokenizer and model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    # Prepare datasets
    train_labels = label_encoder.transform(train_split["label"])
    val_labels = label_encoder.transform(val_split["label"])

    train_dataset = Dataset.from_dict({
        "text": train_split["text"].tolist(),
        "labels": train_labels.tolist()
    })
    val_dataset = Dataset.from_dict({
        "text": val_split["text"].tolist(),
        "labels": val_labels.tolist()
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["text"].tolist()
    })

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Training arguments - conservative for stability
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "mdeberta_checkpoints_new"),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        report_to="none",
        save_total_limit=2,
        gradient_checkpointing=True,  # Save memory
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "f1": f1_score(labels, predictions, average="macro"),
            "accuracy": accuracy_score(labels, predictions)
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save results
    with open(OUTPUT_DIR / "mdeberta_results.txt", "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Validation F1: {eval_results['eval_f1']:.4f}\n")
        f.write(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}\n")

    # Predict on test
    print("\nPredicting on test set...")
    predictions = trainer.predict(test_dataset)
    pred_labels = label_encoder.inverse_transform(
        np.argmax(predictions.predictions, axis=-1)
    )

    return pred_labels


def main():
    # Load data
    train_df, test_df = load_data()

    # Train classifier
    pred_labels = train_classifier(train_df, test_df)

    # Create submission
    print("\n" + "="*60)
    print("Creating submission")
    print("="*60)

    submission = pd.DataFrame({
        "id": test_df["id"],
        "label": pred_labels
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")
    print(f"Submission shape: {submission.shape}")
    print(f"Unique labels predicted: {submission['label'].nunique()}")
    print(f"Sample predictions:")
    print(submission.head(10))

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
