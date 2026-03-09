#!/usr/bin/env python
"""Try to match test distribution better."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
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


def cv_evaluate(X, y, models_func, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_probas = []
    all_y = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        models = models_func()
        for m in models:
            m.fit(X[train_idx], y[train_idx])
        proba = np.mean([m.predict_proba(X[val_idx]) for m in models], axis=0)
        all_probas.append(proba)
        all_y.append(y[val_idx])

    return np.vstack(all_probas), np.concatenate(all_y)


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()

    # Try both RF+XGB and RF+XGB+GB
    def make_rf_xgb():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0)
        ]

    def make_rf_xgb_gb():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0),
            GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                        random_state=42)
        ]

    print("\n1. RF+XGB ensemble...")
    probas1, y1 = cv_evaluate(X_train, y_train, make_rf_xgb)

    print("\n2. RF+XGB+GB ensemble...")
    probas2, y2 = cv_evaluate(X_train, y_train, make_rf_xgb_gb)

    # Find best weights for each, targeting ~7-9% Class 3
    print("\nSearching for weights with 7-10% Class 3...")

    def find_best_weights(probas, y_true, target_c3_range=(7, 10)):
        results = []
        for w0 in [0.6, 0.7, 0.8, 0.9, 1.0]:
            for w1 in [0.8, 0.9, 1.0, 1.1, 1.2]:
                for w2 in [0.8, 0.9, 1.0, 1.1]:
                    for w3 in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
                        w = np.array([w0, w1, w2, w3])
                        preds = np.argmax(probas * w, axis=1)
                        s = evaluate(y_true, preds)
                        c3_pct = (preds == 3).sum() / len(preds) * 100
                        if target_c3_range[0] <= c3_pct <= target_c3_range[1]:
                            results.append((s, w, c3_pct))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    results1 = find_best_weights(probas1, y1)
    results2 = find_best_weights(probas2, y2)

    print("\nRF+XGB - Top 5 with 7-10% Class 3:")
    for s, w, c3 in results1[:5]:
        print(f"  Score: {s:.4f}, Weights: {w}, Class3%: {c3:.1f}%")

    print("\nRF+XGB+GB - Top 5 with 7-10% Class 3:")
    for s, w, c3 in results2[:5]:
        print(f"  Score: {s:.4f}, Weights: {w}, Class3%: {c3:.1f}%")

    # Choose best overall
    all_results = [
        ("RF+XGB", make_rf_xgb, results1),
        ("RF+XGB+GB", make_rf_xgb_gb, results2)
    ]

    best_overall = None
    for name, func, results in all_results:
        if results and (best_overall is None or results[0][0] > best_overall[1]):
            best_overall = (name, results[0][0], results[0][1], func)

    print(f"\n{'='*60}")
    print(f"BEST: {best_overall[0]} with score {best_overall[1]:.4f}")
    print(f"Weights: {best_overall[2]}")
    print("="*60)

    # Train final
    models = best_overall[3]()
    for m in models:
        m.fit(X_train, y_train)

    test_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
    predictions = np.argmax(test_proba * best_overall[2], axis=1)

    # Save
    submission = pd.DataFrame({"id": test_ids, "y": predictions.astype(int)})
    submission.to_csv("submission.csv", index=False)

    print(f"\nSaved submission.csv")
    print("Distribution:")
    for c in range(4):
        n = (predictions == c).sum()
        print(f"  Class {c}: {n} ({100*n/len(predictions):.1f}%)")


if __name__ == "__main__":
    main()
