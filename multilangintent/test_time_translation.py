#!/usr/bin/env python3
"""
Test-time translation approach for cross-lingual transfer.

Uses the existing XLM-R model but translates de/fr test samples to English
before classification. Uses deep-translator (Google Translate API) which
is lightweight and doesn't require loading additional neural models.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "xlmr_intent_model"
MAX_LENGTH = 128
BATCH_SIZE = 32


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


def translate_text_google(text, target_lang='en'):
    """Translate text using Google Translate (free, no API key needed)."""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def translate_with_hf(text, src_lang, translator_pipe):
    """Translate text using HuggingFace NLLB model."""
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
    print("Test-Time Translation for Cross-Lingual Transfer")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"Test languages: {test_df['language'].value_counts().to_dict()}")

    # Load model and tokenizer
    print(f"\nLoading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model = model.to(device)
    model.eval()

    # Load label encoder
    label_classes = np.load(MODEL_PATH / "label_encoder_classes.npy", allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes
    num_labels = len(label_classes)
    print(f"Loaded model with {num_labels} intent classes")

    # Try to load NLLB translation model (better quality but larger)
    use_nllb = False
    print("\nChecking for translation models...")
    try:
        # Free up memory first
        torch.cuda.empty_cache()

        translator_pipe = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device=0 if device == "cuda" else -1
        )
        print("NLLB translation model loaded!")
        use_nllb = True
    except Exception as e:
        print(f"NLLB not available: {e}")
        print("Will try Google Translate (slower but no memory overhead)...")

        # Check if deep_translator is available
        try:
            from deep_translator import GoogleTranslator
            print("Google Translate available via deep_translator")
        except ImportError:
            print("Installing deep_translator...")
            import subprocess
            subprocess.run(["pip", "install", "deep-translator", "-q"])
            print("deep_translator installed")

    # Predict on test set
    print("\nPredicting on test set with translation...")

    translated_texts = []
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Translating"):
        text = row['text']
        lang = row['language']

        if lang in ['de', 'fr']:
            if use_nllb:
                translated = translate_with_hf(text, lang, translator_pipe)
            else:
                translated = translate_text_google(text, 'en')
        else:
            translated = text

        translated_texts.append(translated)

    # Create dataset with translated texts
    test_dataset = IntentDataset(
        translated_texts,
        [0] * len(test_df),
        tokenizer,
        MAX_LENGTH
    )

    # Predict
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())

    pred_label_names = label_encoder.inverse_transform(all_preds)

    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "label": pred_label_names
    })
    submission.to_csv(OUTPUT_DIR / "submission_translated.csv", index=False)
    submission.to_csv("submission.csv", index=False)

    print(f"\nSubmission saved to: submission.csv")
    print(f"Total predictions: {len(submission)}")

    # Show sample predictions
    print("\nSample predictions (translated):")
    for i in range(10):
        orig = test_df['text'].iloc[i]
        trans = translated_texts[i]
        lang = test_df['language'].iloc[i]
        pred = pred_label_names[i]
        if orig != trans:
            print(f"  [{lang}] {orig[:50]}...")
            print(f"      -> {trans[:50]}... -> {pred}")
        else:
            print(f"  [{lang}] {orig[:50]}... -> {pred}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
