#!/usr/bin/env python
"""Solution with CatBoost: Compare RF+XGB vs RF+XGB+CatBoost."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
    """Run CV and return probas for tuning."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_probas = []
    all_y = []
    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        models = models_func()
        for m in models:
            m.fit(X[train_idx], y[train_idx])

        proba = np.mean([m.predict_proba(X[val_idx]) for m in models], axis=0)
        pred = np.argmax(proba, axis=1)
        score = evaluate(y[val_idx], pred)

        all_probas.append(proba)
        all_y.append(y[val_idx])
        scores.append(score)
        print(f"  Fold {fold}: {score:.4f}")

    return np.mean(scores), np.vstack(all_probas), np.concatenate(all_y)


def tune_weights(probas, y_true):
    """Tune class weights."""
    best_score, best_w = 0, np.array([1, 1, 1, 1])

    for w0 in [0.7, 0.8, 0.9]:
        for w1 in [1.0, 1.1, 1.2]:
            for w2 in [0.9, 1.0]:
                for w3 in [1.5, 2.0, 2.5, 3.0]:
                    w = np.array([w0, w1, w2, w3])
                    preds = np.argmax(probas * w, axis=1)
                    s = evaluate(y_true, preds)
                    if s > best_score:
                        best_score, best_w = s, w

    return best_w, best_score


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Test 1: RF + XGB
    print("\n" + "="*50)
    print("Test 1: RF + XGB")
    print("="*50)

    def make_rf_xgb():
        return [
            RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0)
        ]

    cv1, probas1, y1 = cv_evaluate(X_train, y_train, make_rf_xgb)
    w1, tuned1 = tune_weights(probas1, y1)
    print(f"CV: {cv1:.4f}, Tuned: {tuned1:.4f}, Weights: {w1}")

    # Test 2: RF + XGB + CatBoost
    print("\n" + "="*50)
    print("Test 2: RF + XGB + CatBoost")
    print("="*50)

    def make_rf_xgb_cat():
        return [
            RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0),
            CatBoostClassifier(iterations=80, depth=5, learning_rate=0.05,
                               l2_leaf_reg=3.0, random_state=42, verbose=0, thread_count=1)
        ]

    cv2, probas2, y2 = cv_evaluate(X_train, y_train, make_rf_xgb_cat)
    w2, tuned2 = tune_weights(probas2, y2)
    print(f"CV: {cv2:.4f}, Tuned: {tuned2:.4f}, Weights: {w2}")

    # Choose best
    if tuned1 >= tuned2:
        print(f"\nBest: RF+XGB ({tuned1:.4f} >= {tuned2:.4f})")
        best_models = make_rf_xgb
        best_w = w1
    else:
        print(f"\nBest: RF+XGB+CatBoost ({tuned2:.4f} > {tuned1:.4f})")
        best_models = make_rf_xgb_cat
        best_w = w2

    # Train final
    print("\nTraining final model...")
    models = best_models()
    for m in models:
        m.fit(X_train, y_train)

    test_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
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
