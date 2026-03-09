#!/usr/bin/env python
"""Quick test: Compare RF+XGB vs RF+XGB+CatBoost with minimal tuning."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def evaluate(y_true, y_pred):
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return float(np.mean(np.sort(per_class_f1)[:2]))


def load_data():
    train = pd.read_csv("./public/train.csv")
    test = pd.read_csv("./public/test.csv")
    feature_cols = [f"X{i}" for i in range(1, 17)]
    return train[feature_cols].values, train["y"].values, test[feature_cols].values, test["id"].values


def quick_cv(X, y, include_catboost=False):
    """Quick 3-fold CV."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                             eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)

        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        if include_catboost:
            cat = CatBoostClassifier(iterations=50, depth=5, learning_rate=0.1,
                                      l2_leaf_reg=3.0, random_state=42, verbose=0)
            cat.fit(X_tr, y_tr)
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val) + cat.predict_proba(X_val)) / 3
        else:
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 2

        y_pred = np.argmax(proba, axis=1)
        scores.append(evaluate(y_val, y_pred))

    return np.mean(scores)


def quick_tune(X, y, include_catboost=False):
    """Quick tuning with fewer options."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_score = 0
    best_w = [1, 1, 1, 1]

    # Get all CV probas
    all_probas = []
    all_y = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                             eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        if include_catboost:
            cat = CatBoostClassifier(iterations=50, depth=5, learning_rate=0.1,
                                      l2_leaf_reg=3.0, random_state=42, verbose=0)
            cat.fit(X_tr, y_tr)
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val) + cat.predict_proba(X_val)) / 3
        else:
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 2

        all_probas.append(proba)
        all_y.append(y_val)

    probas = np.vstack(all_probas)
    y_true = np.concatenate(all_y)

    # Quick grid
    for w0 in [0.7, 0.8, 0.9]:
        for w1 in [1.0, 1.1]:
            for w2 in [0.9, 1.0]:
                for w3 in [1.5, 2.0, 2.5]:
                    w = np.array([w0, w1, w2, w3])
                    preds = np.argmax(probas * w, axis=1)
                    s = evaluate(y_true, preds)
                    if s > best_score:
                        best_score = s
                        best_w = w

    return best_w, best_score


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Test RF+XGB
    print("\n1. RF+XGB (no CatBoost)...")
    score1 = quick_cv(X_train, y_train, include_catboost=False)
    w1, tuned1 = quick_tune(X_train, y_train, include_catboost=False)
    print(f"   Base CV: {score1:.4f}, Tuned: {tuned1:.4f}, Weights: {w1}")

    # Test RF+XGB+CatBoost
    print("\n2. RF+XGB+CatBoost...")
    score2 = quick_cv(X_train, y_train, include_catboost=True)
    w2, tuned2 = quick_tune(X_train, y_train, include_catboost=True)
    print(f"   Base CV: {score2:.4f}, Tuned: {tuned2:.4f}, Weights: {w2}")

    # Choose best
    if tuned1 >= tuned2:
        print(f"\nBest: RF+XGB ({tuned1:.4f} >= {tuned2:.4f})")
        use_cat, best_w = False, w1
    else:
        print(f"\nBest: RF+XGB+CatBoost ({tuned2:.4f} > {tuned1:.4f})")
        use_cat, best_w = True, w2

    # Train final with full params
    print("\nTraining final model...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                         eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    if use_cat:
        cat = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                                  l2_leaf_reg=3.0, random_state=42, verbose=0)
        cat.fit(X_train, y_train)
        test_proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test) + cat.predict_proba(X_test)) / 3
    else:
        test_proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2

    predictions = np.argmax(test_proba * best_w, axis=1)

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
