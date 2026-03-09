# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

**Always use a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Overview

This is a **multilingual intent classification** competition derived from the [SID (Synthetic Intent Dataset)](https://huggingface.co/datasets/luigicfilho/sid) on Hugging Face. The goal is to predict intent labels for customer-support utterances across 5 languages.

### Key Challenges

- **Typo Robustness**: Includes simulated physical keyboard errors (QWERTY proximity)
- **Grammatical Diversity**: Includes simulated grammatical errors across all languages
- **Domain Specificity**: Optimized for customer support, billing, and technical assistance
- **Class Imbalance**: Label frequency varies significantly; macro F1 weights all classes equally

## Dataset Statistics

| Split | Rows | Languages |
|-------|------|-----------|
| Train | 170,511 | en, pt, es, de, fr |
| Test | 19,711 | en, pt, es, de, fr |
| **Total** | **190,222** | |

- **Intents (labels)**: 100 classes
- **Text length**: min 2, median ~38, max 95 characters
- **Label frequency**: Train ranges 568–2,601 samples per label; Test ranges 81–232

### Language Distribution

**Train**: en (34,855), pt (34,855), es (34,191), de (33,391), fr (33,219)
**Test**: en (4,074), pt (4,047), es (4,024), de (3,830), fr (3,736)

## File Structure

```
dataset/
├── public/
│   ├── train.csv           # id, text, label, language
│   ├── test.csv            # id, text, language
│   └── sample_submission.csv  # id, label
└── private/
    └── answers.csv         # id, label (hidden, for grading)
```

### Column Schema

| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique identifier (unique across train/test) |
| text | str | User utterance (includes synthetic typos/grammar noise) |
| label | str | Intent class (in train.csv and private/answers.csv) |
| language | str | Language code: en, pt, es, de, fr |

## Submission Format

```csv
id,label
172919,delete_account
178724,ask_about_pricing
```

Requirements:
- Exactly one row per test sample (19,711 rows)
- Must include header row
- Only columns: id, label
- ID values must match test.csv exactly

## Evaluation

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average="macro")
```

**Important**: Macro F1 weights all classes equally, so performance on less frequent intents matters significantly.

## Solution Files

- `train_xlmr.py` — **Recommended**: Fine-tuned XLM-RoBERTa (works reliably)
- `solution.py` — Alternative with both transformer and LLM approaches
- `compare_approaches.py` — Compare approaches on validation split

### Running the Solution

```bash
# Activate virtual environment
source venv/bin/activate

# Run training (creates outputs/submission_xlmr.csv)
python train_xlmr.py
```

### Results

| Approach | Validation F1 | Notes |
|----------|---------------|-------|
| XLM-RoBERTa (fine-tuned) | **0.9967** | Recommended - fast and accurate |

### Hardware Requirements

| Approach | GPU Memory | Speed |
|----------|------------|-------|
| XLM-RoBERTa-base | 4-8 GB | ~1000 texts/sec |

## Recommended Approach

1. **Fine-tuned XLM-RoBERTa**: Best accuracy + speed + reliability
2. Train on en/es/pt data, predicts on de/fr (cross-lingual transfer)
3. 3 epochs, learning rate 2e-5, batch size 16
