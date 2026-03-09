#!/usr/bin/env python3
"""
Efficient multilingual intent classification using a small model.
Uses DistilBERT multilingual (~270M params) with memory-efficient training.
"""

import warnings
from pathlib import Path
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
)

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Small multilingual model
MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_LENGTH = 96  # Shorter sequences
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
EPOCHS = 3


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def main():
    print("=" * 60)
    print("Efficient Multilingual Intent Classification")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Languages: {train_df['language'].unique()}")

    # Encode labels
    label_encoder = LabelEncoder()
    train_df['label_id'] = label_encoder.fit_transform(train_df['label'])
    num_labels = len(label_encoder.classes_)
    print(f"Number of intent classes: {num_labels}")

    # Split train/validation
    train_data, val_data = train_test_split(
        train_df, test_size=0.1, random_state=42, stratify=train_df['label_id']
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Load tokenizer and model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )

    # Create datasets
    train_dataset = IntentDataset(
        train_data['text'].values,
        train_data['label_id'].values,
        tokenizer,
        MAX_LENGTH
    )
    val_dataset = IntentDataset(
        val_data['text'].values,
        val_data['label_id'].values,
        tokenizer,
        MAX_LENGTH
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "efficient_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_steps=100,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        dataloader_num_workers=4,
        report_to="none",
        seed=42,
    )

    # Compute metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': f1_score(labels, predictions, average='macro'),
            'accuracy': (predictions == labels).mean()
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Predict on test set
    print("\nPredicting on test set...")

    # Process test data in batches
    test_texts = test_df['text'].tolist()
    all_predictions = []

    # Use trainer for prediction
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

    class TestDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        def __len__(self):
            return len(self.encodings['input_ids'])
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx]
            }

    test_dataset = TestDataset(test_encodings)
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)

    # Convert back to label names
    pred_label_names = label_encoder.inverse_transform(pred_labels)

    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'label': pred_label_names
    })

    submission_path = OUTPUT_DIR / "submission_efficient.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"Submission shape: {submission.shape}")
    print(f"Sample predictions:")
    print(submission.head())

    # Save model and label encoder
    model_save_path = OUTPUT_DIR / "efficient_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    import pickle
    with open(model_save_path / "label_encoder.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

    print(f"\nModel saved to {model_save_path}")
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
