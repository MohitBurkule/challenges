"""
Fast Ensemble - LightGBM + CatBoost only (faster)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 10

os.makedirs('catboost_tmp', exist_ok=True)
os.environ['CATBOOST_INFO_DIR'] = 'catboost_tmp'

print("Loading data...", flush=True)
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")
neighbors = pd.read_csv("neighbor_features.csv")

if 'underperforming' in neighbors.columns:
    neighbors = neighbors.drop(columns=['underperforming'])

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df, train_df=None, is_train=True):
    df = df.copy()

    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['is_renewable'] = (df['fuel_group'] == 'renewable').astype(int)
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    for fuel in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear', 'Biomass']:
        df[f'is_{fuel.lower()}'] = (df['primary_fuel'] == fuel).astype(int)

    df['fossil_old_small'] = df['is_fossil'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['hydro_large'] = df['is_hydro'] * df['capacity_log_mw']
    df['waste_large_old'] = df['is_waste'] * df['capacity_log_mw'] * df['plant_age']
    df['solar_wind_small'] = (df['is_solar'] + df['is_wind']) * (5 - df['capacity_log_mw'])

    # Z-score within fuel
    if is_train:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            mean = df.groupby('primary_fuel')[col].transform('mean')
            std = df.groupby('primary_fuel')[col].transform('std') + 0.01
            df[f'{col}_z_fuel'] = (df[col] - mean) / std
    else:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            for fuel in df['primary_fuel'].unique():
                mask_tr = train_df['primary_fuel'] == fuel
                mask_te = df['primary_fuel'] == fuel
                if mask_tr.sum() > 0 and mask_te.sum() > 0:
                    mean = train_df.loc[mask_tr, col].mean()
                    std = train_df.loc[mask_tr, col].std() + 0.01
                    df.loc[mask_te, f'{col}_z_fuel'] = (df.loc[mask_te, col] - mean) / std

    # Rank within fuel
    if is_train:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            df[f'{col}_fuel_rank'] = df.groupby('primary_fuel')[col].rank(pct=True)
    else:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            for fuel in df['primary_fuel'].unique():
                mask_tr = train_df['primary_fuel'] == fuel
                mask_te = df['primary_fuel'] == fuel
                if mask_tr.sum() > 0 and mask_te.sum() > 0:
                    train_vals = train_df.loc[mask_tr, col].values
                    test_vals = df.loc[mask_te, col].values
                    rank = (test_vals[:, None] < train_vals[None, :]).mean(axis=1)
                    df.loc[mask_te, f'{col}_fuel_rank'] = rank

    df['age_per_cap'] = df['plant_age'] / (df['capacity_log_mw'] + 0.1)
    df['cap_per_age'] = df['capacity_log_mw'] / (df['plant_age'] + 0.1)
    df['age_cap_int'] = df['plant_age'] * df['capacity_log_mw']
    df['lat_lon'] = df['latitude'] * df['longitude']
    df['cap_sq'] = df['capacity_log_mw'] ** 2
    df['age_sq'] = df['plant_age'] ** 2

    return df

print("Engineering features...", flush=True)
train = engineer_features(train, train, is_train=True)
test = engineer_features(test, train, is_train=False)

# Add neighbor features
print("Adding neighbor features...", flush=True)
neighbor_cols = neighbors.columns.tolist()
for col in neighbor_cols:
    train[f'nn_{col}'] = neighbors[col].values
    test[f'nn_{col}'] = train[f'nn_{col}'].mean()

# =============================================================================
# TARGET ENCODING
# =============================================================================

te_cols = ['primary_fuel', 'other_fuel1', 'owner_bucket', 'fuel_group',
           'capacity_band', 'lat_band', 'lon_band']

combos = [
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'lat_band'],
    ['owner_bucket', 'capacity_band'],
    ['owner_bucket', 'primary_fuel'],
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
        agg['smoothed'] = (agg['count'] * agg['mean'] + 20 * global_mean) / (agg['count'] + 20)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

    agg = train.groupby(col)['underperforming'].agg(['mean', 'count'])
    agg['smoothed'] = (agg['count'] * agg['mean'] + 20 * global_mean) / (agg['count'] + 20)
    test[f'{col}_te'] = test[col].map(agg['smoothed']).fillna(global_mean)

# =============================================================================
# PREPARE FEATURES
# =============================================================================

num_features = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
    'latitude', 'longitude', 'age_x_capacity',
    'is_fossil', 'is_renewable', 'has_other_fuel',
    'is_solar', 'is_wind', 'is_hydro', 'is_gas', 'is_coal', 'is_oil', 'is_waste', 'is_nuclear', 'is_biomass',
    'fossil_old_small', 'hydro_large', 'waste_large_old', 'solar_wind_small',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel', 'latitude_z_fuel', 'longitude_z_fuel',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank', 'latitude_fuel_rank', 'longitude_fuel_rank',
    'age_per_cap', 'cap_per_age', 'age_cap_int', 'lat_lon', 'cap_sq', 'age_sq',
    # Neighbor features
    'nn_nn10_max500_target_mean', 'nn_nn5_max500_target_mean',
    'nn_nn10_max200_target_mean', 'nn_nn5_max200_target_mean',
    'nn_nn10_max100_target_mean', 'nn_nn5_max100_target_mean',
    'nn_dist_to_nearest_underperf', 'nn_dist_to_nearest_normal',
]

te_features = [f'{c}_te' for c in te_cols]
num_features.extend(te_features)

cat_features = ['fuel_group', 'primary_fuel', 'other_fuel1',
                'owner_bucket', 'capacity_band', 'lat_band', 'lon_band']

num_features = [f for f in num_features if f in train.columns]

for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

all_features = num_features + cat_features
all_features = [f for f in all_features if f in train.columns]

print(f"Total features: {len(all_features)}", flush=True)

def composite(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return 0.7 * roc + 0.3 * ap

# Prepare arrays
X_num = train[num_features].values.astype(float)
X_cat = train[cat_features].values.astype(int)
X = np.column_stack([X_num, X_cat])
y = target

X_test_num = test[num_features].values.astype(float)
X_test_cat = test[cat_features].values.astype(int)
X_test = np.column_stack([X_test_num, X_test_cat])

for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    if mask.any():
        med = np.nanmedian(X[~mask, i])
        X[mask, i] = med
        X_test[np.isnan(X_test[:, i])] = med

# =============================================================================
# LIGHTGBM
# =============================================================================

print("\n" + "="*60, flush=True)
print("LIGHTGBM", flush=True)
print("="*60, flush=True)

lgb_params = {
    'objective': 'binary',
    'metric': ['auc'],
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'scale_pos_weight': 2.0,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'seed': RANDOM_STATE,
    'n_jobs': -1,
}

lgb_oof = np.zeros(len(y))
lgb_test = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)

    model = lgb.train(
        lgb_params, tr_data, num_boost_round=2000,  # Fewer rounds
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )

    lgb_oof[val_idx] = model.predict(X[val_idx])
    lgb_test += model.predict(X_test) / N_FOLDS

    print(f"  Fold {fold+1}: Comp={composite(y[val_idx], lgb_oof[val_idx]):.4f}", flush=True)

print(f"\nLightGBM: ROC={roc_auc_score(y, lgb_oof):.4f}, AP={average_precision_score(y, lgb_oof):.4f}, Comp={composite(y, lgb_oof):.4f}", flush=True)

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
    train_df = train.iloc[tr_idx][all_features].copy()
    val_df = train.iloc[val_idx][all_features].copy()
    test_df = test[all_features].copy()

    for col in cat_features:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    train_pool = Pool(train_df, label=y[tr_idx], cat_features=cat_feature_indices)
    val_pool = Pool(val_df, label=y[val_idx], cat_features=cat_feature_indices)
    test_pool = Pool(test_df, cat_features=cat_feature_indices)

    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=7,  # Fewer iterations
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0,
        eval_metric='AUC', early_stopping_rounds=75,
        train_dir=f'catboost_tmp/fold_{fold}'
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    cb_test += model.predict_proba(test_pool)[:, 1] / N_FOLDS

    print(f"  Fold {fold+1}: Comp={composite(y[val_idx], cb_oof[val_idx]):.4f}", flush=True)

print(f"\nCatBoost: ROC={roc_auc_score(y, cb_oof):.4f}, AP={average_precision_score(y, cb_oof):.4f}, Comp={composite(y, cb_oof):.4f}", flush=True)

# =============================================================================
# ENSEMBLE
# =============================================================================

print("\n" + "="*60, flush=True)
print("ENSEMBLE", flush=True)
print("="*60, flush=True)

# Stacking
meta_train = np.column_stack([lgb_oof, cb_oof])
meta_test = np.column_stack([lgb_test, cb_test])

lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr.fit(meta_train, y)
stack_oof = lr.predict_proba(meta_train)[:, 1]
stack_test = lr.predict_proba(meta_test)[:, 1]

print(f"LR Stacking: Comp={composite(y, stack_oof):.4f}", flush=True)
print(f"LR Weights: {lr.coef_[0]}", flush=True)

# Weighted average
best_score = 0
best_w = 0.5
for w in np.arange(0.4, 0.7, 0.05):
    weighted = w * lgb_oof + (1 - w) * cb_oof
    score = composite(y, weighted)
    print(f"  LGB weight {w:.2f}: Comp={score:.4f}", flush=True)
    if score > best_score:
        best_score = score
        best_w = w

print(f"\nBest weight: {best_w:.2f}", flush=True)

# Use best
if composite(y, stack_oof) > best_score:
    final_oof = stack_oof
    final_test = stack_test
    print("Using LR Stacking", flush=True)
else:
    final_oof = best_w * lgb_oof + (1 - best_w) * cb_oof
    final_test = best_w * lgb_test + (1 - best_w) * cb_test
    print(f"Using weighted average ({best_w:.2f} LGB)", flush=True)

# =============================================================================
# FINAL
# =============================================================================

print("\n" + "="*60, flush=True)
print("FINAL", flush=True)
print("="*60, flush=True)
print(f"ROC-AUC: {roc_auc_score(y, final_oof):.4f}", flush=True)
print(f"AP: {average_precision_score(y, final_oof):.4f}", flush=True)
print(f"Composite: {composite(y, final_oof):.4f}", flush=True)

submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': final_test.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv", flush=True)
print(f"Range: [{final_test.min():.4f}, {final_test.max():.4f}]", flush=True)

# Cleanup
import shutil
shutil.rmtree('catboost_tmp', ignore_errors=True)
