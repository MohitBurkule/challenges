#!/usr/bin/env python
"""Baseline model for EEG seizure classification."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def evaluate(y_true, y_pred):
    """Competition metric: average F1 of two weakest classes."""
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    weakest_two = np.sort(per_class_f1)[:2]
    return float(np.mean(weakest_two))


def main():
    # Load data
    train = pd.read_csv("public/train.csv")
    test = pd.read_csv("public/test.csv")

    X = train[[f"X{i}" for i in range(1, 17)]].values
    y = train["y"].values
    X_test = test[[f"X{i}" for i in range(1, 17)]].values
    test_ids = test["id"].values

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = evaluate(y_val, y_pred)
        cv_scores.append(score)
        print(f"Fold {fold}: {score:.4f}")

    print(f"\nCV Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # Train on full data and create submission
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({"id": test_ids, "y": predictions.astype(int)})
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")


if __name__ == "__main__":
    main()
