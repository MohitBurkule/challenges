"""
Tuned Ensemble - More aggressive feature engineering + stacking
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
N_FOLDS = 10  # More folds for better stacking

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

def engineer_features(df, train_df=None, is_train=True):
    df = df.copy()

    # Basic indicators
    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['is_renewable'] = (df['fuel_group'] == 'renewable').astype(int)
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    for fuel in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear', 'Biomass']:
        df[f'is_{fuel.lower()}'] = (df['primary_fuel'] == fuel).astype(int)

    # KEY INSIGHTS FROM EDA - fuel-specific patterns
    # Fossil: small + old = underperforming
    df['fossil_old_small'] = df['is_fossil'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['gas_old_small'] = df['is_gas'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['coal_old_small'] = df['is_coal'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['oil_old_small'] = df['is_oil'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)

    # Hydro: LARGE = underperforming (OPPOSITE pattern!)
    df['hydro_large'] = df['is_hydro'] * df['capacity_log_mw']
    df['hydro_large_sq'] = df['is_hydro'] * df['capacity_log_mw'] ** 2

    # Waste: large + old = underperforming
    df['waste_large_old'] = df['is_waste'] * df['capacity_log_mw'] * df['plant_age']

    # Solar/Wind: small = underperforming
    df['solar_wind_small'] = (df['is_solar'] + df['is_wind']) * (5 - df['capacity_log_mw'])

    # Relative to fuel mean (z-score)
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
                mean = train_df.loc[mask_tr, col].mean()
                std = train_df.loc[mask_tr, col].std() + 0.01
                df.loc[mask_te, f'{col}_z_fuel'] = (df.loc[mask_te, col] - mean) / std

    # Rank within fuel (KEY for quantile-based target!)
    if is_train:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            df[f'{col}_fuel_rank'] = df.groupby('primary_fuel')[col].rank(pct=True)
    else:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            for fuel in df['primary_fuel'].unique():
                mask_tr = train_df['primary_fuel'] == fuel
                mask_te = df['primary_fuel'] == fuel
                train_vals = train_df.loc[mask_tr, col].values
                test_vals = df.loc[mask_te, col].values
                # Percentile rank relative to train
                rank = (test_vals[:, None] < train_vals[None, :]).mean(axis=1)
                df.loc[mask_te, f'{col}_fuel_rank'] = rank

    # Interactions
    df['age_per_cap'] = df['plant_age'] / (df['capacity_log_mw'] + 0.1)
    df['cap_per_age'] = df['capacity_log_mw'] / (df['plant_age'] + 0.1)
    df['age_cap_int'] = df['plant_age'] * df['capacity_log_mw']
    df['lat_lon'] = df['latitude'] * df['longitude']

    return df

print("Engineering features...", flush=True)
train = engineer_features(train, train, is_train=True)
test = engineer_features(test, train, is_train=False)

# Add neighbor features (KEY!)
print("Adding neighbor features...", flush=True)
neighbor_cols = neighbors.columns.tolist()
for col in neighbor_cols:
    train[f'nn_{col}'] = neighbors[col].values

# For test, compute neighbor features from train stats
# Use global mean as fallback
for col in neighbor_cols:
    test[f'nn_{col}'] = train[f'nn_{col}'].mean()

# =============================================================================
# TARGET ENCODING
# =============================================================================

te_cols = [
    'primary_fuel', 'other_fuel1', 'owner_bucket', 'fuel_group',
    'capacity_band', 'lat_band', 'lon_band'
]

# High-cardinality combos
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

# CV-based TE for train
for col in te_cols:
    train[f'{col}_te'] = np.nan
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 20 * global_mean) / (agg['count'] + 20)  # More smoothing
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

    # Test TE
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
    'fossil_old_small', 'gas_old_small', 'coal_old_small', 'oil_old_small',
    'hydro_large', 'hydro_large_sq', 'waste_large_old', 'solar_wind_small',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel', 'latitude_z_fuel', 'longitude_z_fuel',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank', 'latitude_fuel_rank', 'longitude_fuel_rank',
    'age_per_cap', 'cap_per_age', 'age_cap_int', 'lat_lon',
    # Neighbor features (BEST - 0.34 correlation!)
    'nn_nn10_max500_target_mean', 'nn_nn5_max500_target_mean',
    'nn_nn10_max200_target_mean', 'nn_nn5_max200_target_mean',
    'nn_nn10_max100_target_mean', 'nn_nn5_max100_target_mean',
    'nn_dist_to_nearest_underperf', 'nn_dist_to_nearest_normal',
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

print("\n" + "="*60, flush=True)
print("LIGHTGBM (Tuned)", flush=True)
print("="*60, flush=True)

# Tuned LightGBM params
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
        lgb_params, tr_data, num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(150, verbose=False),
            lgb.log_evaluation(0)
        ]
    )

    lgb_oof[val_idx] = model.predict(X[val_idx])
    lgb_test += model.predict(X_test) / N_FOLDS

lgb_roc = roc_auc_score(y, lgb_oof)
lgb_ap = average_precision_score(y, lgb_oof)
lgb_comp = composite(y, lgb_oof)
print(f"LightGBM: ROC={lgb_roc:.4f}, AP={lgb_ap:.4f}, Comp={lgb_comp:.4f}", flush=True)

# =============================================================================
# CATBOOST
# =============================================================================

print("\n" + "="*60, flush=True)
print("CATBOOST (Tuned)", flush=True)
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
        iterations=2000, learning_rate=0.03, depth=7,
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0,
        eval_metric='AUC', early_stopping_rounds=100,
        train_dir=f'catboost_tmp/fold_{fold}'
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    cb_test += model.predict_proba(test_pool)[:, 1] / N_FOLDS

cb_roc = roc_auc_score(y, cb_oof)
cb_ap = average_precision_score(y, cb_oof)
cb_comp = composite(y, cb_oof)
print(f"CatBoost: ROC={cb_roc:.4f}, AP={cb_ap:.4f}, Comp={cb_comp:.4f}", flush=True)

# =============================================================================
# STACKING WITH LOGISTIC REGRESSION
# =============================================================================

print("\n" + "="*60, flush=True)
print("STACKING", flush=True)
print("="*60, flush=True)

# Create meta features
meta_train = np.column_stack([lgb_oof, cb_oof])
meta_test = np.column_stack([lgb_test, cb_test])

# Logistic regression stacking
lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr.fit(meta_train, y)
stack_oof = lr.predict_proba(meta_train)[:, 1]
stack_test = lr.predict_proba(meta_test)[:, 1]

stack_roc = roc_auc_score(y, stack_oof)
stack_ap = average_precision_score(y, stack_oof)
stack_comp = composite(y, stack_oof)
print(f"Stacking (LR): ROC={stack_roc:.4f}, AP={stack_ap:.4f}, Comp={stack_comp:.4f}", flush=True)

# Also try simple average
avg_oof = (lgb_oof + cb_oof) / 2
avg_test = (lgb_test + cb_test) / 2
avg_comp = composite(y, avg_oof)
print(f"Simple Average: Comp={avg_comp:.4f}", flush=True)

# Use best method
if stack_comp > avg_comp:
    final_oof = stack_oof
    final_test = stack_test
    print(f"\nUsing Stacking", flush=True)
else:
    final_oof = avg_oof
    final_test = avg_test
    print(f"\nUsing Simple Average", flush=True)

final_roc = roc_auc_score(y, final_oof)
final_ap = average_precision_score(y, final_oof)
final_comp = composite(y, final_oof)

print(f"\n" + "="*60, flush=True)
print("FINAL", flush=True)
print("="*60, flush=True)
print(f"ROC-AUC: {final_roc:.4f}", flush=True)
print(f"AP: {final_ap:.4f}", flush=True)
print(f"Composite: {final_comp:.4f}", flush=True)

# =============================================================================
# SUBMISSION
# =============================================================================

submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': final_test.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv", flush=True)
print(f"Range: [{final_test.min():.4f}, {final_test.max():.4f}]", flush=True)
print(f"Mean: {final_test.mean():.4f}", flush=True)

# Cleanup
import shutil
shutil.rmtree('catboost_tmp', ignore_errors=True)
