#!/usr/bin/env python
"""Try different approaches to improve beyond 0.4810."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
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

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        models = models_func()
        for m in models:
            m.fit(X[train_idx], y[train_idx])

        proba = np.mean([m.predict_proba(X[val_idx]) for m in models], axis=0)
        all_probas.append(proba)
        all_y.append(y[val_idx])

    return np.vstack(all_probas), np.concatenate(all_y)


def tune_weights(probas, y_true):
    """Tune class weights."""
    best_score, best_w = 0, np.array([1, 1, 1, 1])

    for w0 in [0.6, 0.7, 0.8, 0.9]:
        for w1 in [0.9, 1.0, 1.1, 1.2]:
            for w2 in [0.8, 0.9, 1.0]:
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

    # Try different model combinations
    configs = []

    # Config 1: RF+XGB (baseline)
    def make_baseline():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0)
        ]
    configs.append(("RF+XGB (baseline)", make_baseline))

    # Config 2: RF+XGB with different params (less reg)
    def make_less_reg():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, reg_alpha=0.3, reg_lambda=1.0,
                          random_state=42, n_jobs=1, verbosity=0)
        ]
    configs.append(("RF+XGB (less reg)", make_less_reg))

    # Config 3: RF+XGB+LR (add logistic regression)
    def make_with_lr():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        ]
    configs.append(("RF+XGB+LR", make_with_lr))

    # Config 4: RF+XGB+GB
    def make_with_gb():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0),
            GradientBoostingClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                                        random_state=42)
        ]
    configs.append(("RF+XGB+GB", make_with_gb))

    # Config 5: All 4 models
    def make_all():
        return [
            RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   class_weight="balanced", random_state=42, n_jobs=1),
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0,
                          random_state=42, n_jobs=1, verbosity=0),
            CatBoostClassifier(iterations=100, depth=5, learning_rate=0.05,
                               l2_leaf_reg=3.0, random_state=42, verbose=0, thread_count=1),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        ]
    configs.append(("RF+XGB+Cat+LR", make_all))

    # Test all configs
    best_config = None
    best_score = 0
    best_weights = None

    for name, func in configs:
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print("="*50)
        probas, y_true = cv_evaluate(X_train, y_train, func)
        weights, tuned = tune_weights(probas, y_true)
        print(f"Tuned: {tuned:.4f}, Weights: {weights}")

        if tuned > best_score:
            best_score = tuned
            best_config = (name, func)
            best_weights = weights

    print(f"\n{'='*60}")
    print(f"BEST: {best_config[0]} with score {best_score:.4f}")
    print(f"Weights: {best_weights}")
    print("="*60)

    # Train final model
    print("\nTraining final model...")
    models = best_config[1]()
    for m in models:
        m.fit(X_train, y_train)

    test_proba = np.mean([m.predict_proba(X_test) for m in models], axis=0)
    predictions = np.argmax(test_proba * best_weights, axis=1)

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
