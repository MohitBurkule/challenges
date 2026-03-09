# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Epilepsy EEG classification challenge. Build a multi-class classifier to predict seizure types from 16 EEG-derived channel features.

**Classes:**
- 0: Healthy subject (control)
- 1: Generalized seizure
- 2: Focal seizure
- 3: Seizure-related event (eye-blinking, staring, nail-biting, etc.)

**Key Challenge:** Only 1,000 training examples with imbalanced classes.

## Evaluation Metric

Score = average F1 of the two weakest classes. This rewards models that don't ignore minority classes.

```python
from sklearn.metrics import f1_score
import numpy as np

def evaluate(y_true, y_pred):
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    weakest_two = np.sort(per_class_f1)[:2]
    return float(np.mean(weakest_two))
```

## Data Files

- `public/train.csv` - 1,000 labeled rows (id, X1-X16, y)
- `public/test.csv` - 7,000 unlabeled rows (id, X1-X16)
- `public/sample_submission.csv` - Template (id, y)

## Development Setup

Always use a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or: source venv/bin/activate.fish
pip install -r requirements.txt
```

## Commands

Run baseline model:
```bash
source venv/bin/activate && python baseline.py
```

## Submission Format

File: `submission.csv` with columns `id,y`. Must have exactly 7,000 rows matching test.csv IDs. Predictions must be integers 0-3.

## Strategy Notes

- Use class-weighted models or resampling (SMOTE) for imbalance
- Focus on recall for minority classes
- Feature interactions may help (EEG channels have spatial correlations)
- Consider cross-validation with stratification
- **Distribution shift is a major issue** - test set has different feature distributions than training
- Test set has more Class 2 (~41%) and less Class 0 (~25%) than training

## Score Tracking

| # | Approach | CV Score | Test Score | Notes |
|---|----------|----------|------------|-------|
| 1 | Baseline RF (depth=12, leaf=2) | 0.9576 | 0.4000 | Original overfitted model |
| 2 | Complex ensemble (feat eng + SMOTE + 4 models) | 0.9279 | 0.3700 | Too complex, worse generalization |
| 3 | Strong reg RF (depth=6, leaf=10) | 0.8842 | - | Simpler model |
| 4 | Moderate reg RF (depth=8, leaf=5) | 0.9306 | 0.4223 | Good baseline |
| 5 | **RF+XGB ensemble + threshold tuning** | **0.9679** | **0.4810** | **BEST - class weights [0.8, 1.1, 1.0, 2.0]** |
| 6 | RF+XGB+GB ensemble | 0.9792 | - | Higher CV but worse Class 3 predictions on test |
| 7 | RF+XGB+CatBoost | 0.9651 | - | CatBoost didn't help |

### Key Findings

1. **Ensemble helps** - RF + XGBoost combination generalizes better than single model
2. **Threshold tuning is crucial** - Adjusting class weights significantly improves minority class recall
3. **CV-test gap is huge** (~0.5) - indicates severe distribution shift
4. **Class 3 needs boost** - Weight of 2.0 helps achieve 7.9% predictions (vs 3-5% before)
5. **Class 2 is over-represented in test** - ~42% vs 20% in training
6. **Adding more models can hurt** - GB/CatBoost improved CV but hurt Class 3 predictions on test
7. **Distribution shift is severe** - CV shows ~10% Class 3 but test gets only 3-5% with same model

### Current Best Solution

```python
# RF + XGBoost ensemble with threshold tuning
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
xgb = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_by=0.8,
    reg_alpha=0.5, reg_lambda=2.0,
    random_state=42, n_jobs=-1
)
# Average probabilities, then apply class weights [0.8, 1.1, 1.0, 2.0]
```
