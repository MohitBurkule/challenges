"""
Ensemble Solution - LightGBM + CatBoost with Neighbor Features
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

# CatBoost temp dir
os.makedirs('catboost_tmp', exist_ok=True)
os.environ['CATBOOST_INFO_DIR'] = 'catboost_tmp'

print("Loading data...", flush=True)
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")
neighbors = pd.read_csv("neighbor_features.csv")

# Drop target from neighbors if present
if 'underperforming' in neighbors.columns:
    neighbors = neighbors.drop(columns=['underperforming'])

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    df = df.copy()

    # Fuel indicators
    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['is_renewable'] = (df['fuel_group'] == 'renewable').astype(int)
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    for fuel in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil']:
        df[f'is_{fuel.lower()}'] = (df['primary_fuel'] == fuel).astype(int)

    # KEY INSIGHTS FROM EDA:
    df['fossil_old_small'] = df['is_fossil'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['hydro_large'] = df['is_hydro'] * df['capacity_log_mw']
    df['gas_coal_age'] = (df['is_gas'] | df['is_coal']).astype(int) * df['plant_age']
    df['solar_wind_small'] = (df['is_solar'] | df['is_wind']).astype(int) * (5 - df['capacity_log_mw'])

    # Relative to fuel mean
    for col in ['capacity_log_mw', 'plant_age']:
        mean = df.groupby('primary_fuel')[col].transform('mean')
        std = df.groupby('primary_fuel')[col].transform('std') + 0.01
        df[f'{col}_z_fuel'] = (df[col] - mean) / std

    # Rank within fuel
    for col in ['capacity_log_mw', 'plant_age']:
        df[f'{col}_fuel_rank'] = df.groupby('primary_fuel')[col].rank(pct=True)

    df['age_per_cap'] = df['plant_age'] / (df['capacity_log_mw'] + 0.1)

    return df

print("Engineering features...", flush=True)
train = engineer_features(train)
test = engineer_features(test)

# Add neighbor features
print("Adding neighbor features...", flush=True)
for col in neighbors.columns:
    train[f'nn_{col}'] = neighbors[col].values

# For test, we need to compute neighbor features - use train stats as proxy
# For now, fill with train mean (neighbor features computed only on train)
for col in neighbors.columns:
    test[f'nn_{col}'] = train[f'nn_{col}'].mean()

# =============================================================================
# TARGET ENCODING
# =============================================================================

te_cols = [
    'primary_fuel', 'other_fuel1', 'owner_bucket', 'fuel_group',
    'capacity_band', 'lat_band', 'lon_band'
]

# Combo columns
combos = [
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'lat_band'],
]

for combo in combos:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

print("Computing target encodings...", flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# CV-based TE for train
for col in te_cols:
    train[f'{col}_te'] = np.nan
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

    # Test TE
    agg = train.groupby(col)['underperforming'].agg(['mean', 'count'])
    agg['smoothed'] = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
    test[f'{col}_te'] = test[col].map(agg['smoothed']).fillna(global_mean)

# =============================================================================
# PREPARE FEATURES
# =============================================================================

num_features = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
    'latitude', 'longitude', 'age_x_capacity',
    'is_fossil', 'is_renewable', 'has_other_fuel',
    'is_solar', 'is_wind', 'is_hydro', 'is_gas', 'is_coal', 'is_oil',
    'fossil_old_small', 'hydro_large', 'gas_coal_age', 'solar_wind_small',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel', 'age_per_cap',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank',
    # Neighbor features (KEY!)
    'nn_nn10_max500_target_mean', 'nn_nn5_max500_target_mean',
    'nn_nn10_max200_target_mean', 'nn_nn5_max200_target_mean',
]

te_features = [f'{c}_te' for c in te_cols]
num_features.extend(te_features)

cat_features = ['fuel_group', 'primary_fuel', 'other_fuel1',
                'owner_bucket', 'capacity_band', 'lat_band', 'lon_band']

# Filter features that exist
num_features = [f for f in num_features if f in train.columns]

# Label encode categorical
for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

all_features = num_features + cat_features
all_features = [f for f in all_features if f in train.columns]

print(f"Total features: {len(all_features)}", flush=True)

# =============================================================================
# TRAIN
# =============================================================================

def composite(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return 0.7 * roc + 0.3 * ap

print("\n" + "="*60, flush=True)
print("LIGHTGBM", flush=True)
print("="*60, flush=True)

lgb_params = {
    'objective': 'binary',
    'metric': ['auc', 'average_precision'],
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'scale_pos_weight': 2.0,
    'verbose': -1,
    'seed': RANDOM_STATE,
    'n_jobs': -1,
}

# Prepare arrays
X_num = train[num_features].values.astype(float)
X_cat = train[cat_features].values.astype(int)
X = np.column_stack([X_num, X_cat])
y = target

X_test_num = test[num_features].values.astype(float)
X_test_cat = test[cat_features].values.astype(int)
X_test = np.column_stack([X_test_num, X_test_cat])

# Fill NaN
for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    if mask.any():
        med = np.nanmedian(X[~mask, i])
        X[mask, i] = med
        X_test[np.isnan(X_test[:, i])] = med

# LightGBM training
lgb_oof = np.zeros(len(y))
lgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)

    model = lgb.train(
        lgb_params, tr_data, num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    lgb_oof[val_idx] = model.predict(X[val_idx])
    lgb_test += model.predict(X_test) / N_FOLDS

    roc = roc_auc_score(y[val_idx], lgb_oof[val_idx])
    ap = average_precision_score(y[val_idx], lgb_oof[val_idx])
    print(f"  Fold {fold+1}: ROC={roc:.4f}, AP={ap:.4f}, Comp={0.7*roc+0.3*ap:.4f}", flush=True)

lgb_roc = roc_auc_score(y, lgb_oof)
lgb_ap = average_precision_score(y, lgb_oof)
lgb_comp = composite(y, lgb_oof)
print(f"\nLightGBM Overall: ROC={lgb_roc:.4f}, AP={lgb_ap:.4f}, Comp={lgb_comp:.4f}", flush=True)

# =============================================================================
# CATBOOST
# =============================================================================

print("\n" + "="*60, flush=True)
print("CATBOOST", flush=True)
print("="*60, flush=True)

cat_feature_indices = list(range(len(num_features), len(all_features)))

cb_oof = np.zeros(len(y))
cb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
    # Create DataFrames for CatBoost
    train_df = train.iloc[tr_idx][all_features].copy()
    val_df = train.iloc[val_idx][all_features].copy()
    test_df = test[all_features].copy()

    # Ensure cat columns are int
    for col in cat_features:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    train_pool = Pool(train_df, label=y[tr_idx], cat_features=cat_feature_indices)
    val_pool = Pool(val_df, label=y[val_idx], cat_features=cat_feature_indices)
    test_pool = Pool(test_df, cat_features=cat_feature_indices)

    model = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        scale_pos_weight=2.0, random_seed=RANDOM_STATE,
        verbose=0, eval_metric='AUC', early_stopping_rounds=50,
        train_dir=f'catboost_tmp/fold_{fold}'
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    cb_test += model.predict_proba(test_pool)[:, 1] / N_FOLDS

    roc = roc_auc_score(y[val_idx], cb_oof[val_idx])
    ap = average_precision_score(y[val_idx], cb_oof[val_idx])
    print(f"  Fold {fold+1}: ROC={roc:.4f}, AP={ap:.4f}, Comp={0.7*roc+0.3*ap:.4f}", flush=True)

cb_roc = roc_auc_score(y, cb_oof)
cb_ap = average_precision_score(y, cb_oof)
cb_comp = composite(y, cb_oof)
print(f"\nCatBoost Overall: ROC={cb_roc:.4f}, AP={cb_ap:.4f}, Comp={cb_comp:.4f}", flush=True)

# =============================================================================
# ENSEMBLE
# =============================================================================

print("\n" + "="*60, flush=True)
print("ENSEMBLE", flush=True)
print("="*60, flush=True)

# Try different weights
best_score = 0
best_weight = 0.5

for w in np.arange(0.3, 0.8, 0.1):
    ens_oof = w * lgb_oof + (1 - w) * cb_oof
    score = composite(y, ens_oof)
    print(f"  LightGBM weight {w:.1f}: Comp={score:.4f}", flush=True)
    if score > best_score:
        best_score = score
        best_weight = w

print(f"\nBest weight: {best_weight:.1f}", flush=True)

ens_oof = best_weight * lgb_oof + (1 - best_weight) * cb_oof
ens_test = best_weight * lgb_test + (1 - best_weight) * cb_test

final_roc = roc_auc_score(y, ens_oof)
final_ap = average_precision_score(y, ens_oof)
final_comp = composite(y, ens_oof)

print(f"\nFinal Ensemble: ROC={final_roc:.4f}, AP={final_ap:.4f}, Comp={final_comp:.4f}", flush=True)

# =============================================================================
# SUBMISSION
# =============================================================================

submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': ens_test.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv", flush=True)
print(f"Range: [{ens_test.min():.4f}, {ens_test.max():.4f}]", flush=True)
print(f"Mean: {ens_test.mean():.4f}", flush=True)

# Cleanup
import shutil
shutil.rmtree('catboost_tmp', ignore_errors=True)
