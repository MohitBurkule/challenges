#!/usr/bin/env python3
"""
Translation-based multilingual intent classification.
Translates all text to English first, then classifies with a single model.

KEY INSIGHT: Train has en/pt/es, Test has de/fr
This is a zero-shot cross-lingual transfer problem!
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
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# Paths
TRAIN_PATH = Path("public/train.csv")
TEST_PATH = Path("public/test.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission_translated.csv"

# Language code mapping for Helsinki-NLP models
LANG_TO_MODEL = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "pt": "Helsinki-NLP/opus-mt-pt-en",
    "en": None,  # No translation needed
}

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
    print(f"Test-only languages (need translation): {test_langs - train_langs}")

    return train_df, test_df


class Translator:
    """Translates text to English using Helsinki-NLP models."""

    def __init__(self, languages):
        self.translators = {}
        self.languages = languages
        self._load_translators()

    def _load_translators(self):
        """Load translation models for needed languages."""
        print("\nLoading translation models...")

        for lang in self.languages:
            model_name = LANG_TO_MODEL.get(lang)

            if model_name is None:
                print(f"  {lang}: No translation needed (English)")
                continue

            print(f"  Loading {lang}->en translator: {model_name}")

            # Load tokenizer and model directly
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            if torch.cuda.is_available():
                model = model.to("cuda")

            self.translators[lang] = (tokenizer, model)

        print("  Translation models loaded!")

    def translate_batch(self, texts, language, batch_size=32):
        """Translate a batch of texts to English."""
        if language == "en":
            return texts

        if language not in self.translators:
            print(f"Warning: No translator for {language}, returning original text")
            return texts

        tokenizer, model = self.translators[language]

        results = []
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating {language}"):
                batch = texts[i:i+batch_size]

                # Tokenize
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                )

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Generate translations
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=1,  # Faster, greedy decoding
                    do_sample=False
                )

                # Decode
                translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(translations)

        return results

    def clear(self):
        """Clear models from memory."""
        print("Clearing translation models from memory...")
        del self.translators
        torch.cuda.empty_cache()
        gc.collect()


def translate_dataframe(df, translator, text_col="text", lang_col="language"):
    """Translate all text in a dataframe to English."""
    print(f"\nTranslating {len(df)} texts...")

    result = df.copy()
    result["translated_text"] = ""

    for lang in df[lang_col].unique():
        mask = df[lang_col] == lang
        texts = df.loc[mask, text_col].tolist()

        if lang == "en":
            translated = texts
        else:
            translated = translator.translate_batch(texts, lang)

        result.loc[mask, "translated_text"] = translated

        # Show sample
        print(f"\n  Sample {lang} translations:")
        for i in range(min(3, len(texts))):
            print(f"    {lang}: {texts[i][:50]}...")
            print(f"    en:  {translated[i][:50]}...")

    return result


def train_classifier(train_df, test_df):
    """Train a classifier on translated (English) text."""
    print("\n" + "="*60)
    print("Training classifier on translated English text")
    print("="*60)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"Number of intent classes: {num_labels}")

    # Split for validation - stratify by label
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=0.1,
        random_state=42,
        stratify=train_df["label"]
    )

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    val_split = train_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train: {len(train_split)}, Val: {len(val_split)}")

    # Use DistilBERT for English classification
    model_name = "distilbert-base-uncased"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

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
        "text": train_split["translated_text"].tolist(),
        "labels": train_labels.tolist()
    })
    val_dataset = Dataset.from_dict({
        "text": val_split["translated_text"].tolist(),
        "labels": val_labels.tolist()
    })
    test_dataset = Dataset.from_dict({
        "text": test_df["translated_text"].tolist()
    })

    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Set format for PyTorch
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "translated_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
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
        report_to="none",
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
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation on validation set:")
    eval_results = trainer.evaluate()
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

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

    # Determine which languages we need to translate
    all_languages = set(train_df['language'].unique()) | set(test_df['language'].unique())
    print(f"\nLanguages requiring translation: {all_languages - {'en'}}")

    # Initialize translator with all needed languages
    translator = Translator(all_languages)

    # Translate training data
    print("\n" + "="*60)
    print("TRANSLATING TRAINING DATA")
    print("="*60)
    train_translated = translate_dataframe(train_df, translator)

    # Translate test data
    print("\n" + "="*60)
    print("TRANSLATING TEST DATA")
    print("="*60)
    test_translated = translate_dataframe(test_df, translator)

    # Save translated data for reference
    train_translated.to_csv(OUTPUT_DIR / "train_translated.csv", index=False)
    test_translated.to_csv(OUTPUT_DIR / "test_translated.csv", index=False)

    # Clear translation models from memory
    translator.clear()

    # Train classifier
    pred_labels = train_classifier(train_translated, test_translated)

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
