#!/usr/bin/env python3
"""
Translation-based multilingual intent classification solution.
Uses Facebook NLLB-200 for translation - single model for all languages.

Steps:
1. Translate ALL text (train + test) to English
2. Train DistilBERT classifier on translated English text
3. Run inference and create submission
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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# =============================================================================
# Configuration
# =============================================================================

TRAIN_PATH = Path("./dataset/public/train.csv")
TEST_PATH = Path("./dataset/public/test.csv")
OUTPUT_DIR = Path("./working")
OUTPUT_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"

# NLLB language codes
LANG_TO_NLLB = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
}

# Translation model - NLLB-200 distilled (smaller, faster)
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

# Classification model - DistilBERT for English
CLASSIFIER_MODEL = "distilbert-base-uncased"


# =============================================================================
# Translation Functions
# =============================================================================

def load_translation_model():
    """Load NLLB translation model."""
    print(f"\nLoading translation model: {TRANSLATION_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    return tokenizer, model


def translate_texts(texts, src_lang, tokenizer, model, batch_size=32, desc="Translating"):
    """Translate a list of texts to English using NLLB."""
    if src_lang == "en":
        return texts
    
    results = []
    model.eval()
    tgt_lang = "eng_Latn"
    src_nllb = LANG_TO_NLLB[src_lang]
    
    # Set source language
    tokenizer.src_lang = src_nllb
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i:i+batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=1,
                do_sample=False
            )
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(translations)
    
    return results


def translate_dataframe(df, tokenizer, model, text_col="text", lang_col="language", name="data"):
    """Translate all non-English text in a dataframe."""
    result = df.copy()
    result["translated"] = result[text_col]
    
    languages = [l for l in df[lang_col].unique() if l != "en"]
    
    for lang in languages:
        mask = df[lang_col] == lang
        texts = df.loc[mask, text_col].tolist()
        
        print(f"\nTranslating {len(texts)} {lang} texts ({name})...")
        translated = translate_texts(texts, lang, tokenizer, model, desc=f"{lang}->en")
        result.loc[mask, "translated"] = translated
        
        # Show samples
        print(f"  Sample {lang} -> en:")
        for i in range(min(2, len(texts))):
            print(f"    Original: {texts[i][:60]}...")
            print(f"    Translated: {translated[i][:60]}...")
    
    return result


# =============================================================================
# Classification Functions
# =============================================================================

def train_classifier(train_df, test_df):
    """Train DistilBERT classifier on translated English text."""
    print("\n" + "="*60)
    print(f"Training classifier: {CLASSIFIER_MODEL}")
    print("="*60)
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"Number of intent classes: {num_labels}")
    
    # Train/val split
    train_idx, val_idx = train_test_split(
        range(len(train_df)), test_size=0.1, random_state=42, stratify=train_df["label"]
    )
    
    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]
    print(f"Train: {len(train_split)}, Val: {len(val_split)}")
    
    # Load classifier
    print(f"\nLoading model: {CLASSIFIER_MODEL}")
    
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_MODEL, num_labels=num_labels)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
    
    train_dataset = Dataset.from_dict({
        "text": train_split["translated"].tolist(),
        "labels": label_encoder.transform(train_split["label"]).tolist()
    })
    val_dataset = Dataset.from_dict({
        "text": val_split["translated"].tolist(),
        "labels": label_encoder.transform(val_split["label"]).tolist()
    })
    test_dataset = Dataset.from_dict({"text": test_df["translated"].tolist()})
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=["text"])
    test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "classifier_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        warmup_ratio=0.1,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, average="macro"), "accuracy": accuracy_score(labels, preds)}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\nTraining...")
    trainer.train()
    
    print("\nValidation results:")
    eval_results = trainer.evaluate()
    print(f"F1: {eval_results['eval_f1']:.4f}, Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Predict on test
    print("\nPredicting on test set...")
    predictions = trainer.predict(test_dataset)
    pred_labels = label_encoder.inverse_transform(np.argmax(predictions.predictions, axis=-1))
    
    return pred_labels, eval_results['eval_f1']


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("TRANSLATION-BASED INTENT CLASSIFICATION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    print(f"Train: {len(train_df)} samples")
    print(f"Languages: {train_df['language'].value_counts().to_dict()}")
    print(f"\nTest: {len(test_df)} samples")  
    print(f"Languages: {test_df['language'].value_counts().to_dict()}")
    
    # Load translation model
    print("\n" + "="*60)
    print("STEP 1: LOAD TRANSLATION MODEL")
    print("="*60)
    tokenizer, model = load_translation_model()
    
    # Translate train
    print("\n" + "="*60)
    print("STEP 2: TRANSLATE TRAINING DATA")
    print("="*60)
    train_translated = translate_dataframe(train_df, tokenizer, model, name="train")
    
    # Translate test
    print("\n" + "="*60)
    print("STEP 3: TRANSLATE TEST DATA")
    print("="*60)
    test_translated = translate_dataframe(test_df, tokenizer, model, name="test")
    
    # Save translated data (optional, for debugging)
    train_translated.to_csv(OUTPUT_DIR / "train_translated.csv", index=False)
    test_translated.to_csv(OUTPUT_DIR / "test_translated.csv", index=False)
    print("\nTranslated data saved.")
    
    # Free translation model from memory
    print("\nFreeing translation model memory...")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    # Train classifier
    print("\n" + "="*60)
    print("STEP 4: TRAIN CLASSIFIER")
    print("="*60)
    pred_labels, val_f1 = train_classifier(train_translated, test_translated)
    
    # Create submission
    print("\n" + "="*60)
    print("STEP 5: CREATE SUBMISSION")
    print("="*60)
    
    submission = pd.DataFrame({"id": test_df["id"], "label": pred_labels})
    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSubmission saved to: {SUBMISSION_PATH}")
    print(f"Shape: {submission.shape}")
    print(f"Validation F1: {val_f1:.4f}")
    print(f"\nSample predictions:")
    print(submission.head(10))
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
