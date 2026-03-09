#!/usr/bin/env python3
"""
Cross-lingual transfer improvement via data augmentation.

Key insight: XLM-RoBERTa fails on de/fr because:
- Training: en/es/pt
- Testing: de/fr (unseen languages)

Solution: Augment training with synthetic de/fr translations
using simple word-to-intent mapping (faster than neural translation).
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

# Multilingual word mappings for data augmentation
# Maps English words to German/French equivalents
DE_FR_WORDS = {
    # Common action words
    'cancel': {'de': 'stornieren', 'fr': 'annuler'},
    'order': {'de': 'bestellen', 'fr': 'commander'},
    'return': {'de': 'zurückgeben', 'fr': 'retourner'},
    'refund': {'de': 'rückerstattung', 'fr': 'remboursement'},
    'change': {'de': 'ändern', 'fr': 'changer'},
    'update': {'de': 'aktualisieren', 'fr': 'mettre à jour'},
    'delete': {'de': 'löschen', 'fr': 'supprimer'},
    'remove': {'de': 'entfernen', 'fr': 'retirer'},
    'check': {'de': 'überprüfen', 'fr': 'vérifier'},
    'track': {'de': 'verfolgen', 'fr': 'suivre'},
    'book': {'de': 'buchen', 'fr': 'réserver'},
    'schedule': {'de': 'planen', 'fr': 'planifier'},
    'contact': {'de': 'kontakt', 'fr': 'contacter'},
    'help': {'de': 'hilfe', 'fr': 'aide'},
    'support': {'de': 'unterstützung', 'fr': 'support'},
    'request': {'de': 'anfordern', 'fr': 'demander'},
    'report': {'de': 'melden', 'fr': 'signaler'},
    'reset': {'de': 'zurücksetzen', 'fr': 'réinitialiser'},
    'create': {'de': 'erstellen', 'fr': 'créer'},
    'register': {'de': 'registrieren', 'fr': 'inscrire'},
    'login': {'de': 'anmelden', 'fr': 'connexion'},
    'logout': {'de': 'abmelden', 'fr': 'déconnexion'},
    'upgrade': {'de': 'verbessern', 'fr': 'améliorer'},
    'downgrade': {'de': 'herabstufen', 'fr': 'réduire'},
    'subscribe': {'de': 'abonnieren', 'fr': 'abonner'},
    'unsubscribe': {'de': 'abbestellen', 'fr': 'désabonner'},
    'exchange': {'de': 'austauschen', 'fr': 'échanger'},
    'modify': {'de': 'ändern', 'fr': 'modifier'},
    'place': {'de': 'platzieren', 'fr': 'placer'},
    'send': {'de': 'senden', 'fr': 'envoyer'},
    'receive': {'de': 'empfangen', 'fr': 'recevoir'},
    'confirm': {'de': 'bestätigen', 'fr': 'confirmer'},
    'transfer': {'de': 'übertragen', 'fr': 'transférer'},
    'pay': {'de': 'bezahlen', 'fr': 'payer'},
    'buy': {'de': 'kaufen', 'fr': 'acheter'},
    'sell': {'de': 'verkaufen', 'fr': 'vendre'},
    'speak': {'de': 'sprechen', 'fr': 'parler'},
    'talk': {'de': 'reden', 'fr': 'parler'},
    'get': {'de': 'bekommen', 'fr': 'obtenir'},
    'set': {'de': 'einstellen', 'fr': 'définir'},
    'configure': {'de': 'konfigurieren', 'fr': 'configurer'},
    'enable': {'de': 'aktivieren', 'fr': 'activer'},
    'disable': {'de': 'deaktivieren', 'fr': 'désactiver'},

    # Common nouns
    'account': {'de': 'konto', 'fr': 'compte'},
    'password': {'de': 'passwort', 'fr': 'mot de passe'},
    'order': {'de': 'bestellung', 'fr': 'commande'},
    'payment': {'de': 'zahlung', 'fr': 'paiement'},
    'delivery': {'de': 'lieferung', 'fr': 'livraison'},
    'subscription': {'de': 'abonnement', 'fr': 'abonnement'},
    'appointment': {'de': 'termin', 'fr': 'rendez-vous'},
    'invoice': {'de': 'rechnung', 'fr': 'facture'},
    'address': {'de': 'adresse', 'fr': 'adresse'},
    'product': {'de': 'produkt', 'fr': 'produit'},
    'item': {'de': 'artikel', 'fr': 'article'},
    'price': {'de': 'preis', 'fr': 'prix'},
    'discount': {'de': 'rabatt', 'fr': 'réduction'},
    'stock': {'de': 'lagerbestand', 'fr': 'stock'},
    'availability': {'de': 'verfügbarkeit', 'fr': 'disponibilité'},
    'status': {'de': 'status', 'fr': 'statut'},
    'balance': {'de': 'kontostand', 'fr': 'solde'},
    'hours': {'de': 'öffnungszeiten', 'fr': 'horaires'},
    'time': {'de': 'zeit', 'fr': 'heure'},
    'date': {'de': 'datum', 'fr': 'date'},
    'reminder': {'de': 'erinnerung', 'fr': 'rappel'},
    'issue': {'de': 'problem', 'fr': 'problème'},
    'bug': {'de': 'fehler', 'fr': 'bug'},
    'feedback': {'de': 'feedback', 'fr': 'commentaire'},
    'catalog': {'de': 'katalog', 'fr': 'catalogue'},
    'demo': {'de': 'demo', 'fr': 'démo'},
    'policy': {'de': 'richtlinie', 'fr': 'politique'},
    'warranty': {'de': 'garantie', 'fr': 'garantie'},
    'compliance': {'de': 'konformität', 'fr': 'conformité'},
    'software': {'de': 'software', 'fr': 'logiciel'},
    'device': {'de': 'gerät', 'fr': 'appareil'},
    'account': {'de': 'konto', 'fr': 'compte'},
    'directions': {'de': 'wegbeschreibung', 'fr': 'itinéraire'},
    'shipment': {'de': 'sendung', 'fr': 'expédition'},
    'callback': {'de': 'rückruf', 'fr': 'rappel'},
    'performance': {'de': 'leistung', 'fr': 'performance'},
}


def augment_text_to_lang(text: str, target_lang: str) -> str:
    """Simple word-by-word augmentation to target language."""
    words = text.lower().split()
    new_words = []

    for word in words:
        # Remove punctuation for matching
        clean_word = re.sub(r'[^\w]', '', word)
        punct = word[len(clean_word):] if len(word) > len(clean_word) else ''

        # Check if word has translation
        if clean_word in DE_FR_WORDS and target_lang in DE_FR_WORDS[clean_word]:
            new_word = DE_FR_WORDS[clean_word][target_lang]
            new_words.append(new_word + punct)
        else:
            new_words.append(word)

    # Also add language marker words
    if target_lang == 'de':
        prefixes = ['Entschuldigung', 'Hilfe', 'Können Sie', 'Ich möchte', 'Bitte']
    else:
        prefixes = ['Excusez-moi', 'Aide', 'Pouvez-vous', 'Je voudrais', 'S\'il vous plaît']

    # Randomly prefix (30% chance)
    if np.random.random() < 0.3:
        prefix = np.random.choice(prefixes)
        return f"{prefix}: {' '.join(new_words)}"

    return ' '.join(new_words)


def create_augmented_training_data(train_df: pd.DataFrame, augment_ratio: float = 0.3):
    """Create augmented training data with de/fr samples."""
    augmented_rows = []

    # Sample portion of training data to augment
    sample_size = int(len(train_df) * augment_ratio)
    sample_df = train_df.sample(n=sample_size, random_state=42)

    for _, row in sample_df.iterrows():
        # Create German version
        de_text = augment_text_to_lang(row['text'], 'de')
        augmented_rows.append({
            'text': de_text,
            'label': row['label'],
            'language': 'de'
        })

        # Create French version
        fr_text = augment_text_to_lang(row['text'], 'fr')
        augmented_rows.append({
            'text': fr_text,
            'label': row['label'],
            'language': 'fr'
        })

    augmented_df = pd.DataFrame(augmented_rows)

    # Combine original + augmented
    combined_df = pd.concat([train_df, augmented_df], ignore_index=True)

    print(f"Original training: {len(train_df)}")
    print(f"Augmented samples: {len(augmented_df)}")
    print(f"Combined training: {len(combined_df)}")

    return combined_df


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
    print("Cross-lingual Transfer Improvement")
    print("Data Augmentation with Synthetic de/fr")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # Create augmented training data
    print("\nCreating augmented training data...")
    combined_df = create_augmented_training_data(train_df, augment_ratio=0.5)

    # Encode labels
    label_encoder = LabelEncoder()
    all_labels = label_encoder.fit_transform(combined_df["label"].values)
    num_labels = len(label_encoder.classes_)
    print(f"\nNumber of intent classes: {num_labels}")

    # Split train into train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        combined_df["text"].values,
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
    )
    model = model.to(device)

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

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "augmented_checkpoints"),
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
    print("\nTraining with augmented data...")
    trainer.train()

    # Final evaluation
    print("\nFinal Evaluation:")
    eval_results = trainer.evaluate()
    print(f"  Macro F1: {eval_results['eval_macro_f1']:.4f}")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")

    # Save model
    model_path = OUTPUT_DIR / "xlmr_augmented_model"
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
    submission.to_csv(OUTPUT_DIR / "submission_augmented.csv", index=False)
    submission.to_csv("submission.csv", index=False)

    print(f"\nSubmission saved to: submission.csv")
    print(f"Total predictions: {len(submission)}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(5):
        print(f"  [{test_df['language'].iloc[i]}] {test_df['text'].iloc[i][:40]}... -> {pred_label_names[i]}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return eval_results['eval_macro_f1']


if __name__ == "__main__":
    main()
