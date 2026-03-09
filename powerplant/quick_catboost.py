"""
Quick CatBoost test - 3 folds, 500 iterations
Fixed: Use DataFrame and temp dir
"""

import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 3

# Create temp directory for catboost
os.makedirs('catboost_tmp', exist_ok=True)
os.environ['CATBOOST_INFO_DIR'] = 'catboost_tmp'

print("Loading data...", flush=True)
train = pd.read_csv("public/train.csv")
target = train['underperforming'].values
global_mean = target.mean()

# Simple feature engineering
print("Engineering features...", flush=True)
train['is_fossil'] = (train['fuel_group'] == 'fossil').astype(int)
train['fossil_old_small'] = train['is_fossil'] * train['plant_age'] / (train['capacity_log_mw'] + 0.5)
train['hydro_large'] = (train['primary_fuel'] == 'Hydro').astype(int) * train['capacity_log_mw']
train['age_per_cap'] = train['plant_age'] / (train['capacity_log_mw'] + 0.1)

# Rank within fuel
for col in ['capacity_log_mw', 'plant_age']:
    train[f'{col}_fuel_rank'] = train.groupby('primary_fuel')[col].rank(pct=True)

# Target encoding
te_cols = ['primary_fuel', 'owner_bucket', 'capacity_band']
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for col in te_cols:
    train[f'{col}_te'] = np.nan
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

# Prepare data
num_cols = ['capacity_mw', 'capacity_log_mw', 'plant_age', 'latitude', 'longitude',
            'fossil_old_small', 'hydro_large', 'age_per_cap',
            'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank',
            'primary_fuel_te', 'owner_bucket_te', 'capacity_band_te']

cat_cols = ['fuel_group', 'primary_fuel', 'owner_bucket', 'capacity_band']

# Label encode and keep as int
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))

# Fill NaN in numerical
for col in num_cols:
    train[col] = train[col].fillna(train[col].median())

all_features = num_cols + cat_cols
cat_feature_indices = [all_features.index(c) for c in cat_cols]
print(f"Features: {len(all_features)}, Cat indices: {cat_feature_indices}", flush=True)

# Train
def composite(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return 0.7 * roc + 0.3 * ap

print("\n" + "="*50, flush=True)
print("CATBOOST TEST (3 folds, 500 iterations)", flush=True)
print("="*50, flush=True)

cb_oof = np.zeros(len(target))
for fold, (tr_idx, val_idx) in enumerate(skf.split(train, target)):
    train_df = train.iloc[tr_idx][all_features].copy()
    val_df = train.iloc[val_idx][all_features].copy()

    # Ensure cat columns are int
    for col in cat_cols:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)

    train_pool = Pool(train_df, label=target[tr_idx], cat_features=cat_feature_indices)
    val_pool = Pool(val_df, label=target[val_idx], cat_features=cat_feature_indices)

    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=6,
        scale_pos_weight=2.0, random_seed=RANDOM_STATE,
        verbose=0, eval_metric='AUC', early_stopping_rounds=50,
        train_dir=f'catboost_tmp/fold_{fold}'
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]

    roc = roc_auc_score(target[val_idx], cb_oof[val_idx])
    ap = average_precision_score(target[val_idx], cb_oof[val_idx])
    print(f"  Fold {fold+1}: ROC={roc:.4f}, AP={ap:.4f}, Comp={0.7*roc+0.3*ap:.4f}", flush=True)

final_roc = roc_auc_score(target, cb_oof)
final_ap = average_precision_score(target, cb_oof)
final_comp = composite(target, cb_oof)

print("\n" + "="*50, flush=True)
print("RESULTS", flush=True)
print("="*50, flush=True)
print(f"ROC-AUC: {final_roc:.4f}", flush=True)
print(f"AP: {final_ap:.4f}", flush=True)
print(f"Composite: {final_comp:.4f}", flush=True)

# Cleanup
import shutil
shutil.rmtree('catboost_tmp', ignore_errors=True)
