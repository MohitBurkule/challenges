#!/usr/bin/env python
"""Solution v4: Quick test of RF+XGB vs RF+XGB+CatBoost."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def evaluate(y_true, y_pred):
    """Competition metric: average F1 of two weakest classes."""
    per_class_f1 = f1_score(
        y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    weakest_two = np.sort(per_class_f1)[:2]
    return float(np.mean(weakest_two))


def load_data():
    """Load train and test data."""
    train = pd.read_csv("./public/train.csv")
    test = pd.read_csv("./public/test.csv")

    feature_cols = [f"X{i}" for i in range(1, 17)]

    X_train = train[feature_cols].values
    y_train = train["y"].values
    X_test = test[feature_cols].values
    test_ids = test["id"].values

    return X_train, y_train, X_test, test_ids


def train_ensemble(X, y, use_catboost=True, catboost_weight=1.0):
    """Train ensemble and return CV probas and score."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probas = []
    all_y_true = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        xgb = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        )

        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)

        if use_catboost:
            cat = CatBoostClassifier(
                iterations=150, depth=5, learning_rate=0.05,
                l2_leaf_reg=3.0, random_state=42, verbose=0
            )
            cat.fit(X_tr, y_tr)

            # Weighted average
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val) + catboost_weight * cat.predict_proba(X_val)) / (2 + catboost_weight)
        else:
            proba = (rf.predict_proba(X_val) + xgb.predict_proba(X_val)) / 2

        all_probas.append(proba)
        all_y_true.append(y_val)

    return np.vstack(all_probas), np.concatenate(all_y_true)


def tune_class_weights(probas, y_true):
    """Find optimal class weights."""
    best_score = 0
    best_weights = np.array([1.0, 1.0, 1.0, 1.0])

    for w0 in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for w1 in [0.9, 1.0, 1.1, 1.2]:
            for w2 in [0.8, 0.9, 1.0, 1.1]:
                for w3 in [1.5, 2.0, 2.5, 3.0]:
                    weights = np.array([w0, w1, w2, w3])
                    preds = np.argmax(probas * weights, axis=1)
                    score = evaluate(y_true, preds)
                    if score > best_score:
                        best_score = score
                        best_weights = weights

    return best_weights, best_score


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Test 1: RF+XGB only (baseline)
    print("\n" + "="*50)
    print("Test 1: RF + XGBoost (no CatBoost)")
    print("="*50)
    probas, y_true = train_ensemble(X_train, y_train, use_catboost=False)
    weights1, score1 = tune_class_weights(probas, y_true)
    print(f"Best weights: {weights1}, Score: {score1:.4f}")

    # Test 2: RF+XGB+CatBoost equal weights
    print("\n" + "="*50)
    print("Test 2: RF + XGBoost + CatBoost (equal)")
    print("="*50)
    probas, y_true = train_ensemble(X_train, y_train, use_catboost=True, catboost_weight=1.0)
    weights2, score2 = tune_class_weights(probas, y_true)
    print(f"Best weights: {weights2}, Score: {score2:.4f}")

    # Test 3: RF+XGB+CatBoost with less CatBoost
    print("\n" + "="*50)
    print("Test 3: RF + XGBoost + CatBoost (cat=0.5)")
    print("="*50)
    probas, y_true = train_ensemble(X_train, y_train, use_catboost=True, catboost_weight=0.5)
    weights3, score3 = tune_class_weights(probas, y_true)
    print(f"Best weights: {weights3}, Score: {score3:.4f}")

    # Find best config
    results = [
        ("RF+XGB", False, 0, weights1, score1),
        ("RF+XGB+Cat(1.0)", True, 1.0, weights2, score2),
        ("RF+XGB+Cat(0.5)", True, 0.5, weights3, score3),
    ]

    best = max(results, key=lambda x: x[4])
    print(f"\n{'='*50}")
    print(f"BEST: {best[0]} with score {best[4]:.4f}")
    print(f"Class weights: {best[3]}")
    print("="*50)

    # Train final model
    use_cat, cat_w, class_w = best[1], best[2], best[3]

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric='mlogloss', random_state=42, n_jobs=-1
    )

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    if use_cat:
        cat = CatBoostClassifier(
            iterations=150, depth=5, learning_rate=0.05,
            l2_leaf_reg=3.0, random_state=42, verbose=0
        )
        cat.fit(X_train, y_train)
        test_proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test) + cat_w * cat.predict_proba(X_test)) / (2 + cat_w)
    else:
        test_proba = (rf.predict_proba(X_test) + xgb.predict_proba(X_test)) / 2

    test_proba = test_proba * class_w
    predictions = np.argmax(test_proba, axis=1)

    # Save submission
    submission = pd.DataFrame({
        "id": test_ids,
        "y": predictions.astype(int)
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved. Distribution:")
    for c in range(4):
        count = (predictions == c).sum()
        pct = count / len(predictions) * 100
        print(f"  Class {c}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
