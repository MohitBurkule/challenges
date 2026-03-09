#!/usr/bin/env python
"""Generate multiple submissions to test."""
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
    print("Training models...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=1)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                         random_state=42, n_jobs=1, verbosity=0)
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                     random_state=42)

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Get probas
    proba_rfxgb = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2
    proba_rfxgbgb = (rf.predict_proba(X_test) + xgb.predict_proba(X_test) + gb.predict_proba(X_test)) / 3

    # Generate submissions
    configs = [
        ("sub_rfxgb_best.csv", proba_rfxgb, [0.8, 1.1, 1.0, 2.0], "RF+XGB best known (0.4810)"),
        ("sub_rfxgbgb_highc3.csv", proba_rfxgbgb, [0.6, 1.0, 0.9, 2.5], "RF+XGB+GB 8.6% C3"),
        ("sub_rfxgbgb_morec3.csv", proba_rfxgbgb, [0.5, 1.0, 0.8, 3.0], "RF+XGB+GB 11.1% C3"),
        ("sub_rfxgb_highc3.csv", proba_rfxgb, [0.6, 1.0, 0.9, 2.5], "RF+XGB try 8.6% C3"),
    ]

    for fname, proba, weights, desc in configs:
        w = np.array(weights)
        preds = np.argmax(proba * w, axis=1)
        submission = pd.DataFrame({"id": test_ids, "y": preds.astype(int)})
        submission.to_csv(fname, index=False)

        c3 = (preds == 3).sum()
        c3_pct = c3 / len(preds) * 100
        print(f"{fname}: {desc}")
        print(f"  Class 3: {c3} ({c3_pct:.1f}%)")
        print(f"  Distribution: {np.bincount(preds)}")
        print()

    # Use main submission
    submission = pd.read_csv("sub_rfxgb_best.csv")
    submission.to_csv("submission.csv", index=False)
    print("Main submission.csv = sub_rfxgb_best.csv")


if __name__ == "__main__":
    main()
