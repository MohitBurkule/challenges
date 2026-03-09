#!/usr/bin/env python
"""Solution v3: Weighted RF + XGBoost + CatBoost ensemble with extensive tuning."""
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


def cross_validate(X, y, model_weights=(1, 1, 1), n_splits=5):
    """Run stratified cross-validation with weighted RF+XGB+CatBoost ensemble."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    all_probas = []
    all_y_true = []

    w_rf, w_xgb, w_cat = model_weights
    total_weight = w_rf + w_xgb + w_cat

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Three models
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
        cat = CatBoostClassifier(
            iterations=150, depth=5, learning_rate=0.05,
            l2_leaf_reg=3.0, random_state=42, verbose=0
        )

        rf.fit(X_tr, y_tr)
        xgb.fit(X_tr, y_tr)
        cat.fit(X_tr, y_tr)

        # Weighted average probabilities
        proba = (
            w_rf * rf.predict_proba(X_val) +
            w_xgb * xgb.predict_proba(X_val) +
            w_cat * cat.predict_proba(X_val)
        ) / total_weight

        y_pred = np.argmax(proba, axis=1)

        score = evaluate(y_val, y_pred)
        scores.append(score)
        all_probas.append(proba)
        all_y_true.append(y_val)

        per_class = f1_score(y_val, y_pred, labels=[0,1,2,3], average=None)
        print(f"Fold {fold}: {score:.4f} | per-class F1: {per_class.round(3)}")

    print(f"\nCV Mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return np.mean(scores), np.vstack(all_probas), np.concatenate(all_y_true)


def tune_class_weights(probas, y_true):
    """Find optimal class weights for threshold adjustment."""
    print("\nTuning class weights...")
    best_score = 0
    best_weights = np.array([1.0, 1.0, 1.0, 1.0])

    # Extended grid search
    for w0 in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for w1 in [0.8, 0.9, 1.0, 1.1, 1.2]:
            for w2 in [0.8, 0.9, 1.0, 1.1]:
                for w3 in [1.5, 2.0, 2.5, 3.0, 3.5]:
                    weights = np.array([w0, w1, w2, w3])
                    preds = np.argmax(probas * weights, axis=1)
                    score = evaluate(y_true, preds)
                    if score > best_score:
                        best_score = score
                        best_weights = weights

    print(f"Best weights: {best_weights}, tuned CV: {best_score:.4f}")
    return best_weights


def tune_model_weights(X, y):
    """Find optimal model weights for the ensemble."""
    print("\nTuning model weights...")
    best_score = 0
    best_weights = (1, 1, 1)

    for w_rf in [1, 2]:
        for w_xgb in [1, 2]:
            for w_cat in [0, 1]:  # CatBoost can be 0 (excluded)
                cv_score, _, _ = cross_validate(X, y, model_weights=(w_rf, w_xgb, w_cat))
                if cv_score > best_score:
                    best_score = cv_score
                    best_weights = (w_rf, w_xgb, w_cat)
                    print(f"  Better: RF={w_rf}, XGB={w_xgb}, Cat={w_cat} -> CV: {cv_score:.4f}")

    print(f"Best model weights: {best_weights}")
    return best_weights


def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {np.bincount(y_train)}")

    # Try different model weight combinations
    print("\n" + "="*60)
    print("Testing different model weight combinations...")
    print("="*60)

    # Test configurations
    configs = [
        (1, 1, 0),  # RF+XGB only (original best)
        (1, 1, 1),  # Equal weights
        (2, 1, 1),  # More RF
        (1, 2, 1),  # More XGB
        (1, 1, 0.5),  # Less CatBoost
        (2, 2, 1),  # Less CatBoost relative
    ]

    results = []
    for model_weights in configs:
        print(f"\n--- Model weights: RF={model_weights[0]}, XGB={model_weights[1]}, Cat={model_weights[2]} ---")
        cv_score, probas, y_true = cross_validate(X_train, y_train, model_weights)
        class_weights = tune_class_weights(probas, y_true)
        results.append((model_weights, cv_score, class_weights))

    # Find best config
    best_result = max(results, key=lambda x: x[1])
    print(f"\n{'='*60}")
    print(f"Best model weights: {best_result[0]}")
    print(f"Best CV: {best_result[1]:.4f}")
    print(f"Best class weights: {best_result[2]}")

    # Train final model with best config
    print("\nTraining final ensemble with best configuration...")
    model_weights = best_result[0]
    class_weights = best_result[2]
    w_rf, w_xgb, w_cat = model_weights
    total_weight = w_rf + w_xgb + w_cat

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

    models = [(rf, w_rf), (xgb, w_xgb)]

    if w_cat > 0:
        cat = CatBoostClassifier(
            iterations=150, depth=5, learning_rate=0.05,
            l2_leaf_reg=3.0, random_state=42, verbose=0
        )
        models.append((cat, w_cat))

    for model, _ in models:
        model.fit(X_train, y_train)

    # Get test predictions
    test_proba = sum(w * m.predict_proba(X_test) for m, w in models) / total_weight
    test_proba = test_proba * class_weights
    predictions = np.argmax(test_proba, axis=1)

    # Create submission
    submission = pd.DataFrame({
        "id": test_ids,
        "y": predictions.astype(int)
    })
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")
    print(f"Format verified: {len(submission)} rows")

    print(f"\nPrediction distribution:")
    for c in range(4):
        count = (predictions == c).sum()
        pct = count / len(predictions) * 100
        print(f"  Class {c}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
