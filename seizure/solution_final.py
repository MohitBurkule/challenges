#!/usr/bin/env python
"""
Final solution with pseudo-labeling - optimized for speed.
Uses unbuffered output and streamlined grid search.
"""
import sys
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def evaluate(y_true, y_pred):
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return float(np.mean(np.sort(per_class_f1)[:2]))


def load_data():
    train = pd.read_csv("./public/train.csv")
    test = pd.read_csv("./public/test.csv")
    feature_cols = [f"X{i}" for i in range(1, 17)]
    return train[feature_cols].values, train["y"].values, test[feature_cols].values, test["id"].values


def get_models():
    """Return list of diverse models."""
    return [
        ("RF", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
        ("XGB", XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.3, reg_lambda=1.5,
            random_state=42, n_jobs=-1, verbosity=0
        )),
        ("CatBoost", CatBoostClassifier(
            iterations=150, depth=6, learning_rate=0.1,
            l2_leaf_reg=3.0, random_state=42, verbose=0, thread_count=-1
        )),
        ("GB", GradientBoostingClassifier(
            n_estimators=80, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )),
        ("ET", ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
        ("LR", LogisticRegression(
            max_iter=2000, class_weight='balanced',
            C=0.5, random_state=42, n_jobs=-1
        )),
    ]


def train_ensemble(models, X_train, y_train):
    """Train all models."""
    trained = []
    for name, model in models:
        model.fit(X_train, y_train)
        trained.append((name, model))
    return trained


def predict_ensemble(trained_models, X):
    """Get ensemble predictions (averaged probabilities)."""
    all_probas = [model.predict_proba(X) for name, model in trained_models]
    return np.mean(all_probas, axis=0)


def find_best_weights(probas, y_true):
    """Quick weight tuning."""
    best_score = 0
    best_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Coarse grid first
    for w0 in [0.6, 0.8, 1.0]:
        for w1 in [1.0, 1.1]:
            for w2 in [0.9, 1.0]:
                for w3 in [2.0, 2.5, 3.0]:
                    w = np.array([w0, w1, w2, w3])
                    preds = np.argmax(probas * w, axis=1)
                    score = evaluate(y_true, preds)
                    if score > best_score:
                        best_score = score
                        best_weights = w

    return best_weights, best_score


def cross_validate(X, y, n_splits=3):
    """Quick CV to get probabilities for weight tuning."""
    print("  Running CV...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_probas = []
    all_y = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        models = get_models()
        trained = train_ensemble(models, X[train_idx], y[train_idx])
        proba = predict_ensemble(trained, X[val_idx])
        all_probas.append(proba)
        all_y.append(y[val_idx])
        print(f"    Fold {fold} done")

    return np.vstack(all_probas), np.concatenate(all_y)


def pseudo_label_round(X_train, y_train, X_test, weights, round_num, conf_thresh=0.90, min_agree=5):
    """One round of pseudo-labeling."""
    print(f"\n  Training ensemble...")
    models = get_models()
    trained = train_ensemble(models, X_train, y_train)

    # Get individual predictions
    all_probas = []
    all_preds = []
    for name, model in trained:
        proba = model.predict_proba(X_test)
        all_probas.append(proba)
        all_preds.append(np.argmax(proba, axis=1))

    all_probas = np.array(all_probas)
    all_preds = np.array(all_preds)

    avg_proba = np.mean(all_probas, axis=0)
    final_preds = np.argmax(avg_proba * weights, axis=1)

    # Find pseudo-label candidates
    max_proba = np.max(avg_proba, axis=1)
    high_conf = max_proba >= conf_thresh

    agreement_count = np.sum(all_preds == final_preds.reshape(1, -1), axis=0)
    high_agreement = agreement_count >= min_agree

    pseudo_mask = high_conf & high_agreement

    print(f"  Confidence >={conf_thresh}: {high_conf.sum()} samples")
    print(f"  Agreement >={min_agree}: {high_agreement.sum()} samples")
    print(f"  Pseudo-labels: {pseudo_mask.sum()} samples")
    print(f"  Pseudo-label dist: {np.bincount(final_preds[pseudo_mask], minlength=4)}")

    if pseudo_mask.sum() > 0:
        X_pseudo = X_test[pseudo_mask]
        y_pseudo = final_preds[pseudo_mask]
        X_new = np.vstack([X_train, X_pseudo])
        y_new = np.concatenate([y_train, y_pseudo])
        return X_new, y_new, pseudo_mask, final_preds

    return X_train, y_train, pseudo_mask, final_preds


def main():
    print("="*60)
    print("PSEUDO-LABELING ENSEMBLE SOLUTION")
    print("="*60)

    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Initial weight tuning
    print("\nInitial CV for weight tuning...")
    probas, y_true = cross_validate(X_train, y_train)
    weights, cv_score = find_best_weights(probas, y_true)
    print(f"  Best weights: {weights}, CV: {cv_score:.4f}")

    # Pseudo-labeling iterations
    current_X = X_train.copy()
    current_y = y_train.copy()

    configs = [
        (0.92, 6),  # Round 1: very conservative
        (0.88, 5),  # Round 2: moderate
        (0.85, 4),  # Round 3: more aggressive
    ]

    for i, (conf, agree) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"ROUND {i+1}: conf={conf}, agree={agree}")
        print(f"Training size: {len(current_y)}, dist: {np.bincount(current_y)}")
        print("="*60)

        new_X, new_y, mask, preds = pseudo_label_round(
            current_X, current_y, X_test, weights, i+1, conf, agree
        )

        if mask.sum() == 0:
            print("No pseudo-labels, stopping.")
            break

        current_X = new_X
        current_y = new_y

        # Re-tune weights
        if i < len(configs) - 1:
            print("\nRe-tuning weights...")
            probas, y_true = cross_validate(current_X, current_y)
            weights, cv_score = find_best_weights(probas, y_true)
            print(f"  New weights: {weights}, CV: {cv_score:.4f}")

    # Final training
    print(f"\n{'='*60}")
    print("FINAL TRAINING")
    print(f"Final training size: {len(current_y)}")
    print(f"Final distribution: {np.bincount(current_y)}")
    print("="*60)

    models = get_models()
    trained = train_ensemble(models, current_X, current_y)

    # Final predictions
    print("\nGenerating predictions...")
    avg_proba = predict_ensemble(trained, X_test)
    predictions = np.argmax(avg_proba * weights, axis=1)

    # Save
    submission = pd.DataFrame({"id": test_ids, "y": predictions.astype(int)})
    submission.to_csv("submission.csv", index=False)

    print(f"\nSaved submission.csv ({len(submission)} rows)")
    print("\nFinal distribution:")
    for c in range(4):
        n = (predictions == c).sum()
        print(f"  Class {c}: {n} ({100*n/len(predictions):.1f}%)")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
