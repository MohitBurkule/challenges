#!/usr/bin/env python3
"""
Evaluate Gemma3 27B via Ollama for multilingual intent classification.

Uses Ollama API to run zero-shot classification and compare with XLM-RoBERTa.
"""

import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configuration
DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:27b"

# Sample sizes for evaluation
TRAIN_SAMPLE = 200  # Sample from training to check accuracy
TEST_SAMPLE = None  # None = all test data


INTENT_PROMPT = """You are a customer support intent classifier. Classify the following message into exactly ONE intent category.

Available categories (choose the best match):
{intents}

Message (Language: {language}): "{text}"

Instructions:
1. Account for typos and grammatical errors
2. Choose the single best matching intent
3. Output ONLY the intent label name, nothing else

Intent:"""


def get_intent_list(train_df):
    """Get formatted list of all intents."""
    intents = sorted(train_df["label"].unique())
    return "\n".join([f"- {i}" for i in intents[:50]])  # First 50 for brevity


def classify_with_gemma(text: str, language: str, intents_str: str, max_retries=3) -> str:
    """Classify using Gemma3 via Ollama API."""

    prompt = INTENT_PROMPT.format(
        intents=intents_str,
        language=language,
        text=text[:300]  # Truncate long texts
    )

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 50,
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""


def match_intent(response: str, valid_intents: list) -> str:
    """Match LLM response to a valid intent."""
    response_clean = response.strip().lower().replace(" ", "_").replace("-", "_")

    # Direct match
    for intent in valid_intents:
        if intent.lower() == response_clean:
            return intent

    # Partial match
    for intent in valid_intents:
        intent_lower = intent.lower()
        if response_clean in intent_lower or intent_lower in response_clean:
            return intent

    # Keyword matching
    response_words = set(response_clean.split("_"))
    best_match = None
    best_score = 0

    for intent in valid_intents:
        intent_words = set(intent.lower().split("_"))
        overlap = len(response_words & intent_words)
        if overlap > best_score:
            best_score = overlap
            best_match = intent

    return best_match if best_match else valid_intents[0]


def evaluate_on_training_sample(train_df, intents_str, valid_intents, sample_size=200):
    """Evaluate Gemma on a sample of training data to measure accuracy."""

    print(f"\nEvaluating Gemma3 on {sample_size} training samples...")

    sample_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

    predictions = []
    true_labels = []

    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Gemma Train Eval"):
        raw_response = classify_with_gemma(
            row["text"],
            row["language"],
            intents_str
        )
        matched = match_intent(raw_response, valid_intents)
        predictions.append(matched)
        true_labels.append(row["label"])

    # Calculate metrics
    f1 = f1_score(true_labels, predictions, average="macro")
    acc = accuracy_score(true_labels, predictions)

    print(f"\nTraining Sample Results:")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    return f1, acc, sample_df, predictions


def predict_on_test(test_df, intents_str, valid_intents, output_file="submission_gemma.csv"):
    """Run Gemma on full test set."""

    print(f"\nRunning Gemma3 on {len(test_df)} test samples...")

    predictions = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Gemma Test"):
        raw_response = classify_with_gemma(
            row["text"],
            row["language"],
            intents_str
        )
        matched = match_intent(raw_response, valid_intents)
        predictions.append(matched)

    # Create submission
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "label": predictions
    })
    submission.to_csv(OUTPUT_DIR / output_file, index=False)
    print(f"Saved: {OUTPUT_DIR / output_file}")

    return predictions


def analyze_disagreements(test_df, xlmr_preds, gemma_preds, train_df):
    """Analyze where XLM-R and Gemma disagree."""

    disagreements = []
    for i in range(len(test_df)):
        if xlmr_preds[i] != gemma_preds[i]:
            disagreements.append({
                "id": test_df.iloc[i]["id"],
                "language": test_df.iloc[i]["language"],
                "text": test_df.iloc[i]["text"][:80],
                "xlmr_pred": xlmr_preds[i],
                "gemma_pred": gemma_preds[i],
            })

    print(f"\nDisagreements: {len(disagreements)} / {len(test_df)} ({100*len(disagreements)/len(test_df):.1f}%)")

    # Save for analysis
    disc_df = pd.DataFrame(disagreements)
    disc_df.to_csv(OUTPUT_DIR / "disagreements.csv", index=False)

    # Show language breakdown
    if len(disagreements) > 0:
        print("\nDisagreement by language:")
        print(disc_df["language"].value_counts())

    return disagreements


def main():
    print("=" * 60)
    print("Gemma3 27B Evaluation via Ollama")
    print("=" * 60)

    # Check Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print("Ollama is running")
    except:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        return

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train languages: {train_df['language'].value_counts().to_dict()}")
    print(f"Test languages: {test_df['language'].value_counts().to_dict()}")

    # Get intent list
    valid_intents = sorted(train_df["label"].unique())
    intents_str = get_intent_list(train_df)
    print(f"\nIntent classes: {len(valid_intents)}")

    # Evaluate on training sample first
    train_f1, train_acc, sample_df, train_preds = evaluate_on_training_sample(
        train_df, intents_str, valid_intents, sample_size=TRAIN_SAMPLE
    )

    # Predict on test set
    test_predictions = predict_on_test(test_df, intents_str, valid_intents)

    # Compare with XLM-RoBERTa predictions if available
    xlmr_submission = OUTPUT_DIR / "submission_xlmr.csv"
    if xlmr_submission.exists():
        print("\nComparing with XLM-RoBERTa...")
        xlmr_df = pd.read_csv(xlmr_submission)
        xlmr_preds = xlmr_df["label"].tolist()

        # Agreement rate
        agreement = sum(1 for i in range(len(test_predictions)) if test_predictions[i] == xlmr_preds[i])
        print(f"Agreement: {agreement} / {len(test_predictions)} ({100*agreement/len(test_predictions):.1f}%)")

        # Analyze disagreements
        analyze_disagreements(test_df, xlmr_preds, test_predictions, train_df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gemma3 27B Training Sample F1: {train_f1:.4f}")
    print(f"Gemma3 27B Training Sample Acc: {train_acc:.4f}")
    print(f"Test predictions saved to: {OUTPUT_DIR / 'submission_gemma.csv'}")

    return {
        "train_f1": train_f1,
        "train_acc": train_acc,
    }


if __name__ == "__main__":
    main()
