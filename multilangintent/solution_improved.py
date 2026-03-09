#!/usr/bin/env python3
"""
Improved multilingual intent classification with cross-lingual transfer.

Key insight: Training is en/es/pt, test is de/fr (unseen languages).
The gap (99.76% val vs 76.83% test) is due to poor cross-lingual transfer.

Approaches:
1. Translate de/fr to English before classification
2. Keyword-based matching as fallback
3. Ensemble with different approaches
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
    pipeline,
)
from collections import defaultdict
import re

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3


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


def build_keyword_matcher(train_df):
    """Build keyword-based matcher from training data."""
    # Collect keywords per intent
    intent_keywords = defaultdict(lambda: defaultdict(int))

    for _, row in train_df.iterrows():
        text = row["text"].lower()
        label = row["label"]

        # Extract important words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        for word in words:
            if word not in ['the', 'and', 'for', 'can', 'help', 'please', 'want', 'need', 'get', 'with']:
                intent_keywords[label][word] += 1

    # Get top keywords per intent
    top_keywords = {}
    for label, words in intent_keywords.items():
        sorted_words = sorted(words.items(), key=lambda x: -x[1])
        top_keywords[label] = [w for w, c in sorted_words[:20]]

    return top_keywords


def keyword_classify(text, top_keywords, valid_intents):
    """Classify using keyword matching."""
    text_lower = text.lower()
    scores = {}

    for intent, keywords in top_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[intent] = score

    if max(scores.values()) > 0:
        return max(scores, key=scores.get)

    # Try common patterns
    patterns = {
        'cancel': ['cancel', 'cancell', 'stop', 'end'],
        'delete': ['delete', 'remove', 'erase', 'supprim'],
        'update': ['update', 'change', 'modify', 'aktualisier', 'ändern'],
        'refund': ['refund', 'money back', 'récupér'],
        'order': ['order', 'command', 'bestell', 'command'],
        'account': ['account', 'compte', 'konto'],
        'password': ['password', 'passwort', 'mot de passe'],
        'login': ['login', 'sign in', 'connect', 'anmeld'],
        'contact': ['contact', 'reach', 'call', 'kontakt'],
    }

    for intent_keyword, words in patterns.items():
        for word in words:
            if word in text_lower:
                # Find matching intent
                for intent in valid_intents:
                    if intent_keyword in intent.lower():
                        return intent

    return valid_intents[0]


def translate_text(text, src_lang, translator_pipe):
    """Translate text to English using HuggingFace model."""
    if src_lang == 'en':
        return text

    lang_code = {'de': 'deu_Latn', 'fr': 'fra_Latn', 'es': 'spa_Latn', 'pt': 'por_Latn'}

    try:
        result = translator_pipe(text, src_lang=lang_code.get(src_lang, src_lang), tgt_lang='eng_Latn')
        return result[0]['translation_text'] if result else text
    except:
        return text


def main():
    print("=" * 60)
    print("Improved Multilingual Intent Classification")
    print("Cross-lingual transfer: en/es/pt → de/fr")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Build keyword matcher
    print("\nBuilding keyword matcher...")
    top_keywords = build_keyword_matcher(train_df)

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(train_df["label"].values)
    num_labels = len(label_encoder.classes_)
    valid_intents = list(label_encoder.classes_)

    print(f"Intent classes: {num_labels}")

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].values,
        all_labels,
        test_size=0.1,
        random_state=42,
        stratify=all_labels
    )

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        OUTPUT_DIR / "xlmr_intent_model",
        num_labels=num_labels,
    )
    model = model.to(device)
    model.eval()

    print("Model loaded from checkpoint")

    # Try to load translation model
    print("\nLoading translation model (for de/fr → en)...")
    try:
        translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", device=0 if device=="cuda" else -1)
        print("Translation model loaded!")
        use_translation = True
    except Exception as e:
        print(f"Translation model not available: {e}")
        translator = None
        use_translation = False

    # Evaluate on validation
    print("\nEvaluating on validation set...")
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(val_dataset)
    val_preds = np.argmax(predictions.predictions, axis=-1)

    val_f1 = f1_score(val_labels, val_preds, average="macro")
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Macro F1: {val_f1:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Predict on test with different approaches
    print("\nPredicting on test set with multiple approaches...")

    # Approach 1: Direct prediction
    test_dataset = IntentDataset(test_df["text"].values, [0] * len(test_df), tokenizer, MAX_LENGTH)
    predictions = trainer.predict(test_dataset)
    direct_preds = np.argmax(predictions.predictions, axis=-1)
    direct_labels = label_encoder.inverse_transform(direct_preds)

    # Approach 2: Keyword-based prediction
    print("Running keyword-based classification...")
    keyword_preds = []
    for text in test_df["text"].values:
        pred = keyword_classify(text, top_keywords, valid_intents)
        keyword_preds.append(pred)

    # Approach 3: Translated prediction (if available)
    if use_translation:
        print("Running translation-based classification...")
        translated_preds = []
        for idx, row in test_df.iterrows():
            text = row["text"]
            lang = row["language"]

            if lang in ['de', 'fr']:
                # Translate to English
                translated = translate_text(text, lang, translator)
            else:
                translated = text

            # Classify translated text
            inputs = tokenizer(translated, max_length=MAX_LENGTH, padding="max_length",
                              truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()

            translated_preds.append(label_encoder.inverse_transform([pred])[0])
    else:
        translated_preds = direct_labels.copy()

    # Analyze disagreements
    print("\nAnalyzing approach disagreements...")
    agreement_direct_keyword = sum(1 for d, k in zip(direct_labels, keyword_preds) if d == k)
    print(f"Direct vs Keyword agreement: {agreement_direct_keyword}/{len(test_df)} ({100*agreement_direct_keyword/len(test_df):.1f}%)")

    # Create ensemble submission
    # For de/fr: use translated predictions if available, else direct
    # When models disagree, trust the model with higher confidence
    ensemble_preds = []
    for i, row in test_df.iterrows():
        lang = row["language"]
        direct = direct_labels[i]
        keyword = keyword_preds[i]
        translated = translated_preds[i]

        # If all agree, use that
        if direct == keyword == translated:
            ensemble_preds.append(direct)
        # If de/fr, prefer translated
        elif lang in ['de', 'fr'] and use_translation:
            ensemble_preds.append(translated)
        # If direct and keyword agree
        elif direct == keyword:
            ensemble_preds.append(direct)
        else:
            # Use direct as default (our fine-tuned model)
            ensemble_preds.append(direct)

    # Save submissions
    submissions = {
        "submission_direct.csv": direct_labels,
        "submission_keyword.csv": keyword_preds,
        "submission_ensemble.csv": ensemble_preds,
    }

    if use_translation:
        submissions["submission_translated.csv"] = translated_preds

    for filename, preds in submissions.items():
        submission = pd.DataFrame({
            "id": test_df["id"].values,
            "label": preds
        })
        submission.to_csv(OUTPUT_DIR / filename, index=False)
        print(f"Saved: {OUTPUT_DIR / filename}")

    # Copy best submission
    import shutil
    shutil.copy(OUTPUT_DIR / "submission_ensemble.csv", "submission.csv")
    print(f"\nBest submission copied to: submission.csv")

    print("\n" + "=" * 60)
    print("DONE - Multiple submissions generated")
    print("=" * 60)

    return val_f1


if __name__ == "__main__":
    main()
