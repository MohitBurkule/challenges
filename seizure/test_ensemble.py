#!/usr/bin/env python
"""Quick test: Compare RF+XGB vs RF+XGB+CatBoost - single thread."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Suppress XGBoost warnings
import os
os.environ['XGBOOST_QUIET'] = '1'


def evaluate(y_true, y_pred):
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return float(np.mean(np.sort(per_class_f1)[:2]))


def load_data():
    train = pd.read_csv("./public/train.csv")
    test = pd.read_csv("./public/test.csv")
    feature_cols = [f"X{i}" for i in range(1, 17)]
    return train[feature_cols].values, train["y"].values, test[feature_cols].values, test["id"].values


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Use the best known configuration: RF+XGB with weights [0.8, 1.1, 1.0, 2.0]
    print("\nUsing best known config: RF+XGB with weights [0.8, 1.1, 1.0, 2.0]")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=1  # Single thread
    )
    xgb = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='mlogloss', random_state=42, n_jobs=1, verbosity=0
    )

    print("Training RF...")
    rf.fit(X_train, y_train)
    print("Training XGB...")
    xgb.fit(X_train, y_train)

    test_proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2

    # Apply best weights
    best_w = np.array([0.8, 1.1, 1.0, 2.0])
    predictions = np.argmax(test_proba * best_w, axis=1)

    # Save
    submission = pd.DataFrame({"id": test_ids, "y": predictions.astype(int)})
    submission.to_csv("submission.csv", index=False)

    print(f"\nSaved submission.csv with {len(submission)} rows")
    print("Distribution:")
    for c in range(4):
        n = (predictions == c).sum()
        print(f"  Class {c}: {n} ({100*n/len(predictions):.1f}%)")


if __name__ == "__main__":
    main()
