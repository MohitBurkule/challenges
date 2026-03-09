"""
Test CatBoost - Faster version for quick evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

print("Loading data...", flush=True)
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, train_df=None, is_train=True):
    df = df.copy()

    # Basic
    df['lat_rad'] = np.radians(df['latitude'])
    df['lon_rad'] = np.radians(df['longitude'])
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    # Fuel indicators
    for fuel in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear']:
        df[f'is_{fuel.lower()}'] = (df['primary_fuel'] == fuel).astype(int)

    # Interactions
    df['age_per_cap'] = df['plant_age'] / (df['capacity_log_mw'] + 0.1)
    df['cap_per_age'] = df['capacity_log_mw'] / (df['plant_age'] + 0.1)

    # Geographic
    df['lat_lon'] = df['latitude'] * df['longitude']
    df['dist_center'] = np.sqrt(df['latitude']**2 + df['longitude']**2)

    # Fuel-specific patterns
    df['fossil_old_small'] = (df['is_gas'] + df['is_coal'] + df['is_oil']) * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['hydro_large'] = df['is_hydro'] * df['capacity_log_mw']
    df['renewable_small'] = (df['is_solar'] + df['is_wind']) * (5 - df['capacity_log_mw'])

    # Ranking within fuel group
    if is_train:
        for col in ['capacity_log_mw', 'plant_age']:
            df[f'{col}_fuel_rank'] = df.groupby('primary_fuel')[col].rank(pct=True)
    else:
        for col in ['capacity_log_mw', 'plant_age']:
            ranks = []
            for fuel in df['primary_fuel'].unique():
                mask = df['primary_fuel'] == fuel
                train_fuel = train_df[train_df['primary_fuel'] == fuel][col]
                test_fuel = df.loc[mask, col]
                rank = (test_fuel.values[:, None] < train_fuel.values[None, :]).mean(axis=1)
                df.loc[mask, f'{col}_fuel_rank'] = rank

    # Z-score within fuel
    if is_train:
        for col in ['capacity_log_mw', 'plant_age']:
            mean = df.groupby('primary_fuel')[col].transform('mean')
            std = df.groupby('primary_fuel')[col].transform('std') + 0.01
            df[f'{col}_z_fuel'] = (df[col] - mean) / std
    else:
        for col in ['capacity_log_mw', 'plant_age']:
            for fuel in df['primary_fuel'].unique():
                mask_tr = train_df['primary_fuel'] == fuel
                mask_te = df['primary_fuel'] == fuel
                mean = train_df.loc[mask_tr, col].mean()
                std = train_df.loc[mask_tr, col].std() + 0.01
                df.loc[mask_te, f'{col}_z_fuel'] = (df.loc[mask_te, col] - mean) / std

    return df

print("Engineering features...", flush=True)
train = engineer_features(train, train, is_train=True)
test = engineer_features(test, train, is_train=False)

# =============================================================================
# TARGET ENCODING
# =============================================================================

te_cols = ['primary_fuel', 'other_fuel1', 'owner_bucket', 'fuel_group',
           'capacity_band', 'lat_band', 'lon_band']

combos = [
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'capacity_band'],
    ['owner_bucket', 'capacity_band'],
]

for combo in combos:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

print("Computing target encodings...", flush=True)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for col in te_cols:
    train[f'{col}_te'] = np.nan
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 5 * global_mean) / (agg['count'] + 5)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

    agg = train.groupby(col)['underperforming'].agg(['mean', 'count'])
    agg['smoothed'] = (agg['count'] * agg['mean'] + 5 * global_mean) / (agg['count'] + 5)
    test[f'{col}_te'] = test[col].map(agg['smoothed']).fillna(global_mean)

# =============================================================================
# PREPARE DATA
# =============================================================================

num_cols = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude', 'latitude', 'longitude', 'age_x_capacity',
    'age_per_cap', 'cap_per_age', 'lat_lon', 'dist_center',
    'fossil_old_small', 'hydro_large', 'renewable_small',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel',
]

te_cols_features = [f'{c}_te' for c in te_cols]
num_cols.extend(te_cols_features)

cat_cols = ['fuel_group', 'primary_fuel', 'other_fuel1', 'owner_bucket', 'capacity_band', 'lat_band', 'lon_band']

# Label encode
cat_indices = []
for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

all_features = num_cols + cat_cols
all_features = [f for f in all_features if f in train.columns]

# Categorical indices for CatBoost (last len(cat_cols) features)
cat_feature_indices = list(range(len(num_cols), len(all_features)))

X = train[all_features].values.astype(float)
X_test = test[all_features].values.astype(float)
y = target

# Fill NaN
for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    if mask.any():
        med = np.nanmedian(X[~mask, i])
        X[mask, i] = med
        X_test[np.isnan(X_test[:, i]), i] = med

print(f"Total features: {len(all_features)}", flush=True)
print(f"Categorical indices: {cat_feature_indices}", flush=True)

# =============================================================================
# TRAINING
# =============================================================================

def composite(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return 0.7 * roc + 0.3 * ap

print("\n" + "="*60, flush=True)
print("MODEL COMPARISON", flush=True)
print("="*60, flush=True)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# 1. LightGBM
print("\n1. LightGBM...", flush=True)
lgb_params = {
    'objective': 'binary', 'metric': ['auc'],
    'num_leaves': 127, 'learning_rate': 0.02,
    'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
    'min_child_samples': 10, 'scale_pos_weight': 2.0,
    'verbose': -1, 'seed': RANDOM_STATE, 'n_jobs': -1
}

lgb_oof = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx])
    model = lgb.train(lgb_params, tr_data, num_boost_round=1500, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_oof[val_idx] = model.predict(X[val_idx])
    print(f"   Fold {fold+1}: {composite(y[val_idx], lgb_oof[val_idx]):.4f}", flush=True)

print(f"   LightGBM Overall: {composite(y, lgb_oof):.4f}", flush=True)

# 2. XGBoost
print("\n2. XGBoost...", flush=True)
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 8, 'learning_rate': 0.02,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'scale_pos_weight': 2.0, 'seed': RANDOM_STATE, 'n_jobs': -1
}

xgb_oof = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    val_data = xgb.DMatrix(X[val_idx], label=y[val_idx])
    model = xgb.train(xgb_params, tr_data, num_boost_round=1500, evals=[(val_data, 'val')],
                      early_stopping_rounds=100, verbose_eval=False)
    xgb_oof[val_idx] = model.predict(val_data)
    print(f"   Fold {fold+1}: {composite(y[val_idx], xgb_oof[val_idx]):.4f}", flush=True)

print(f"   XGBoost Overall: {composite(y, xgb_oof):.4f}", flush=True)

# 3. CatBoost
print("\n3. CatBoost...", flush=True)
cb_oof = np.zeros(len(y))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    train_pool = Pool(X[tr_idx], label=y[tr_idx], cat_features=cat_feature_indices)
    val_pool = Pool(X[val_idx], label=y[val_idx], cat_features=cat_feature_indices)

    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=8,
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0,
        eval_metric='AUC', early_stopping_rounds=100
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    print(f"   Fold {fold+1}: {composite(y[val_idx], cb_oof[val_idx]):.4f}", flush=True)

print(f"   CatBoost Overall: {composite(y, cb_oof):.4f}", flush=True)

# =============================================================================
# ENSEMBLE
# =============================================================================

print("\n" + "="*60, flush=True)
print("ENSEMBLE", flush=True)
print("="*60, flush=True)

# Simple average of all 3
avg_3 = (lgb_oof + xgb_oof + cb_oof) / 3
print(f"LightGBM + XGBoost + CatBoost: {composite(y, avg_3):.4f}", flush=True)

# Best 2
avg_lgb_xgb = (lgb_oof + xgb_oof) / 2
avg_lgb_cb = (lgb_oof + cb_oof) / 2
avg_xgb_cb = (xgb_oof + cb_oof) / 2
print(f"LightGBM + XGBoost: {composite(y, avg_lgb_xgb):.4f}", flush=True)
print(f"LightGBM + CatBoost: {composite(y, avg_lgb_cb):.4f}", flush=True)
print(f"XGBoost + CatBoost: {composite(y, avg_xgb_cb):.4f}", flush=True)

# =============================================================================
# BEST ENSEMBLE SUBMISSION
# =============================================================================

# Use best ensemble
best_ensemble = max(
    [(avg_3, "LGB+XGB+CB"), (avg_lgb_xgb, "LGB+XGB"), (avg_lgb_cb, "LGB+CB"), (avg_xgb_cb, "XGB+CB")],
    key=lambda x: composite(y, x[0])
)

print(f"\nBest ensemble: {best_ensemble[1]} with score: {composite(y, best_ensemble[0]):.4f}", flush=True)

# Generate predictions with all 3 models
print("\nGenerating test predictions...", flush=True)
lgb_test = np.zeros(len(X_test))
xgb_test = np.zeros(len(X_test))
cb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    # LightGBM
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    model = lgb.train(lgb_params, tr_data, num_boost_round=1500,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_test += model.predict(X_test) / N_FOLDS

    # XGBoost
    tr_data = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    test_data = xgb.DMatrix(X_test)
    model = xgb.train(xgb_params, tr_data, num_boost_round=1500,
                      early_stopping_rounds=100, verbose_eval=False)
    xgb_test += model.predict(test_data) / N_FOLDS

    # CatBoost
    train_pool = Pool(X[tr_idx], label=y[tr_idx], cat_features=cat_feature_indices)
    test_pool = Pool(X_test, cat_features=cat_feature_indices)
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.02, depth=8,
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0
    )
    model.fit(train_pool, verbose=False)
    cb_test += model.predict_proba(test_pool)[:, 1] / N_FOLDS

    print(f"   Fold {fold+1} done", flush=True)

# Best ensemble prediction
final_pred = (lgb_test + xgb_test + cb_test) / 3

submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': final_pred.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv", flush=True)
print(f"Range: [{final_pred.min():.4f}, {final_pred.max():.4f}]", flush=True)
print(f"Mean: {final_pred.mean():.4f}", flush=True)
