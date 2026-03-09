"""
Analyze target generation pattern
The target was generated using quantile labeling within fuel groups.
Can we reverse-engineer the formula?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

print("Loading data...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# ANALYZE TARGET PATTERN
# =============================================================================

print("\n" + "="*60)
print("ANALYZING TARGET GENERATION PATTERN")
print("="*60)

# For each fuel group, what separates 0s from 1s?
for fuel in train['fuel_group'].unique():
    subset = train[train['fuel_group'] == fuel]
    normal = subset[subset['underperforming'] == 0]
    underperf = subset[subset['underperforming'] == 1]

    print(f"\n{fuel.upper()} (n={len(subset)}):")
    print(f"  Normal: cap={normal['capacity_mw'].mean():.0f}, age={normal['plant_age'].mean():.1f}")
    print(f"  Under:  cap={underperf['capacity_mw'].mean():.0f}, age={underperf['plant_age'].mean():.1f}")

# Check if target is deterministic based on some combination
print("\n" + "="*60)
print("CHECKING FOR DETERMINISTIC PATTERNS")
print("="*60)

# Maybe target is based on capacity percentile within fuel group?
train['cap_pct_within_fuel'] = train.groupby('primary_fuel')['capacity_mw'].rank(pct=True)
train['age_pct_within_fuel'] = train.groupby('primary_fuel')['plant_age'].rank(pct=True)

# Check correlation
for col in ['cap_pct_within_fuel', 'age_pct_within_fuel']:
    corr = train[col].corr(train['underperforming'])
    print(f"{col} correlation with target: {corr:.4f}")

# Maybe it's a combination?
train['combined_pct'] = (train['cap_pct_within_fuel'] + train['age_pct_within_fuel']) / 2
corr = train['combined_pct'].corr(train['underperforming'])
print(f"combined_pct correlation: {corr:.4f}")

# =============================================================================
# TRY VERY AGGRESSIVE FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*60)
print("TRYING AGGRESSIVE FEATURES")
print("="*60)

# Basic features
train['lat_rad'] = np.radians(train['latitude'])
train['lon_rad'] = np.radians(train['longitude'])
test['lat_rad'] = np.radians(test['latitude'])
test['lon_rad'] = np.radians(test['longitude'])

# Target encoding with many combinations
te_cols = ['primary_fuel', 'other_fuel1', 'owner_bucket', 'capacity_band',
           'lat_band', 'lon_band', 'fuel_group']

combos = [
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'lat_band'],
    ['primary_fuel', 'lon_band'],
    ['fuel_group', 'capacity_band'],
    ['owner_bucket', 'primary_fuel'],
    ['owner_bucket', 'capacity_band'],
    ['owner_bucket', 'fuel_group'],
    ['owner_bucket', 'lat_band'],
    ['owner_bucket', 'lon_band'],
    ['primary_fuel', 'other_fuel1', 'capacity_band'],
    ['fuel_group', 'capacity_band', 'lat_band'],
]

for combo in combos:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

# CV-based target encoding
print("Computing target encodings...")
for col in te_cols:
    train[f'{col}_te'] = np.nan

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for tr_idx, val_idx in skf.split(train, target):
    for col in te_cols:
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 5 * global_mean) / (agg['count'] + 5)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

for col in te_cols:
    agg = train.groupby(col)['underperforming'].agg(['mean', 'count'])
    agg['smoothed'] = (agg['count'] * agg['mean'] + 5 * global_mean) / (agg['count'] + 5)
    test[f'{col}_te'] = test[col].map(agg['smoothed']).fillna(global_mean)

# Numerical features
num_cols = ['capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
            'latitude', 'longitude', 'age_x_capacity',
            'cap_pct_within_fuel', 'age_pct_within_fuel', 'combined_pct']

# TE features
te_features = [f'{c}_te' for c in te_cols]

all_features = num_cols + te_features

X = train[all_features].values
X_test = test[all_features].values
y = target

# Fill NaN
for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    if mask.any():
        med = np.nanmedian(X[:, i])
        X[mask, i] = med
        X_test[np.isnan(X_test[:, i]), i] = med

# =============================================================================
# TRAIN WITH OPTIMIZED PARAMETERS
# =============================================================================

params = {
    'objective': 'binary',
    'metric': ['auc', 'average_precision'],
    'boosting_type': 'gbdt',
    'num_leaves': 255,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 3,
    'min_child_samples': 5,
    'scale_pos_weight': 2.0,
    'verbose': -1,
    'seed': RANDOM_STATE,
    'n_jobs': -1,
}

print("\nTraining with aggressive target encoding...")
oof = np.zeros(len(y))
test_preds = np.zeros(len(X_test))
scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)

    model = lgb.train(
        params, tr_data, num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    oof[val_idx] = model.predict(X[val_idx])
    test_preds += model.predict(X_test) / N_FOLDS

    roc = roc_auc_score(y[val_idx], oof[val_idx])
    ap = average_precision_score(y[val_idx], oof[val_idx])
    scores.append(0.7 * roc + 0.3 * ap)

print(f"\nCV Score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

final_roc = roc_auc_score(y, oof)
final_ap = average_precision_score(y, oof)
final_composite = 0.7 * final_roc + 0.3 * final_ap

print(f"Overall ROC: {final_roc:.4f}")
print(f"Overall AP: {final_ap:.4f}")
print(f"Composite: {final_composite:.4f}")

# Feature importance
print("\nTop 20 features:")
importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importance('gain')
}).sort_values('importance', ascending=False)

for i, row in importance.head(20).iterrows():
    print(f"  {row['feature']}: {row['importance']:.0f}")

# Save submission
submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': test_preds.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)
print(f"\nSaved submission.csv")
