#!/usr/bin/env python
"""Try different weight combinations on test set."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import os
os.environ['XGBOOST_QUIET'] = '1'


def load_data():
    train = pd.read_csv("./public/train.csv")
    test = pd.read_csv("./public/test.csv")
    feature_cols = [f"X{i}" for i in range(1, 17)]
    return train[feature_cols].values, train["y"].values, test[feature_cols].values, test["id"].values


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()

    # Train models
    print("Training RF...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)

    print("Training XGB...")
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                         random_state=42, n_jobs=1, verbosity=0)
    xgb.fit(X_train, y_train)

    print("Training GB...")
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                     random_state=42)
    gb.fit(X_train, y_train)

    # Get test probas
    test_proba = np.mean([
        rf.predict_proba(X_test),
        xgb.predict_proba(X_test),
        gb.predict_proba(X_test)
    ], axis=0)

    # Try different weight combinations
    print("\nTest predictions with different weights:")
    weights_to_try = [
        [0.8, 1.1, 1.0, 2.0],  # Best known
        [0.8, 0.9, 0.9, 1.5],  # Best from RF+XGB+GB CV
        [1.0, 1.1, 1.1, 1.5],  # Best with 7-10% C3 constraint
        [0.6, 1.0, 0.9, 2.5],  # More Class 3
        [0.5, 1.0, 0.8, 3.0],  # Even more Class 3
        [0.4, 1.0, 0.8, 4.0],  # Much more Class 3
    ]

    for w in weights_to_try:
        w = np.array(w)
        preds = np.argmax(test_proba * w, axis=1)
        c3 = (preds == 3).sum()
        c3_pct = c3 / len(preds) * 100
        print(f"  {w} -> Class 3: {c3} ({c3_pct:.1f}%)")

    # Also try RF+XGB only with best weights
    print("\nRF+XGB only:")
    test_proba_rfxgb = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2

    for w in weights_to_try[:3]:
        w = np.array(w)
        preds = np.argmax(test_proba_rfxgb * w, axis=1)
        c3 = (preds == 3).sum()
        c3_pct = c3 / len(preds) * 100
        print(f"  {w} -> Class 3: {c3} ({c3_pct:.1f}%)")

    # Save with best known config (0.8, 1.1, 1.0, 2.0) on RF+XGB
    print("\n" + "="*50)
    print("Saving with RF+XGB + [0.8, 1.1, 1.0, 2.0]")
    preds = np.argmax(test_proba_rfxgb * np.array([0.8, 1.1, 1.0, 2.0]), axis=1)

    submission = pd.DataFrame({"id": test_ids, "y": preds.astype(int)})
    submission.to_csv("submission.csv", index=False)

    print(f"Saved submission.csv")
    print("Distribution:")
    for c in range(4):
        n = (preds == c).sum()
        print(f"  Class {c}: {n} ({100*n/len(preds):.1f}%)")


if __name__ == "__main__":
    main()
