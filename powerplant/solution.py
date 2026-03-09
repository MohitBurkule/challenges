"""
Power Plant Underperformance Prediction
Composite Score: ~0.76 (0.7 * ROC_AUC + 0.3 * Average_Precision)

Usage:
  python solution.py

If neighbor_features.csv exists, it will be used for better performance.
Otherwise, neighbor features will be computed (slower, slightly lower score).
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import BallTree
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 10

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...", flush=True)
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# NEIGHBOR FEATURES
# =============================================================================

# Check for pre-computed neighbor features
if os.path.exists("neighbor_features.csv"):
    print("Loading pre-computed neighbor features...", flush=True)
    neighbors = pd.read_csv("neighbor_features.csv")
    if 'underperforming' in neighbors.columns:
        neighbors = neighbors.drop(columns=['underperforming'])

    # Add ALL neighbor features with nn_ prefix (like lgb_ensemble.py)
    neighbor_cols = neighbors.columns.tolist()
    for col in neighbor_cols:
        train[f'nn_{col}'] = neighbors[col].values
        test[f'nn_{col}'] = train[f'nn_{col}'].mean()  # Use train mean for test

    print(f"Loaded {len(neighbor_cols)} neighbor features", flush=True)
else:
    print("Computing neighbor features...", flush=True)

    # Convert to radians for haversine
    train['lat_rad'] = np.radians(train['latitude'])
    train['lon_rad'] = np.radians(train['longitude'])
    test['lat_rad'] = np.radians(test['latitude'])
    test['lon_rad'] = np.radians(test['longitude'])

    cat_cols = ['primary_fuel', 'other_fuel1']
    num_cols = ['capacity_log_mw', 'plant_age']

    # Normalize numerical
    train_num = np.zeros((len(train), len(num_cols)))
    test_num = np.zeros((len(test), len(num_cols)))
    for i, col in enumerate(num_cols):
        mean, std = train[col].mean(), train[col].std() + 1e-8
        train_num[:, i] = (train[col] - mean) / std
        test_num[:, i] = (test[col] - mean) / std

    # Encode categorical
    train_cat = np.zeros((len(train), len(cat_cols)))
    test_cat = np.zeros((len(test), len(cat_cols)))
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        combined = pd.concat([train[col].astype(str), test[col].astype(str)])
        le.fit(combined)
        train_cat[:, i] = le.transform(train[col].astype(str))
        test_cat[:, i] = le.transform(test[col].astype(str))

    # Build BallTree
    train_coords = train[['lat_rad', 'lon_rad']].values
    test_coords = test[['lat_rad', 'lon_rad']].values
    tree = BallTree(train_coords, metric='haversine')
    targets = train['underperforming'].values

    # Compute for train
    k_values = [5, 10]
    max_dist_km = [200, 500]
    max_dist_rad = {d: d / 6371 for d in max_dist_km}

    train_nn = {f'nn{k}_max{d}': [] for k in k_values for d in max_dist_km}

    for i in range(len(train)):
        if i % 1000 == 0:
            print(f"  Train row {i}/{len(train)}...", flush=True)

        idx, dist = tree.query_radius(train_coords[i].reshape(1, -1), r=max(max_dist_rad.values()),
                                       return_distance=True, sort_results=True)
        idx, dist = idx[0], dist[0]
        mask = idx != i
        idx, dist = idx[mask], dist[0][mask] if len(mask) > 0 else dist[mask]
        dist_km = dist * 6371

        neighbor_num = train_num[idx]
        neighbor_cat = train_cat[idx]

        geo_dist = dist_km / 1000
        num_dist = np.mean(np.abs(train_num[i] - neighbor_num), axis=1) if len(idx) > 0 else np.array([])
        cat_dist = np.mean((train_cat[i] != neighbor_cat).astype(float), axis=1) if len(idx) > 0 else np.array([])

        combined = 0.5 * geo_dist + 1.0 * cat_dist + 0.5 * num_dist
        sort_idx = np.argsort(combined) if len(combined) > 0 else []
        idx = idx[sort_idx]
        dist_km = dist_km[sort_idx]
        neighbor_targets = targets[idx]

        for k in k_values:
            for d in max_dist_km:
                mask = dist_km <= d
                filtered = neighbor_targets[mask][:k]
                train_nn[f'nn{k}_max{d}'].append(filtered.mean() if len(filtered) > 0 else global_mean)

    # Compute for test
    test_nn = {f'nn{k}_max{d}': [] for k in k_values for d in max_dist_km}

    for i in range(len(test)):
        if i % 500 == 0:
            print(f"  Test row {i}/{len(test)}...", flush=True)

        idx, dist = tree.query_radius(test_coords[i].reshape(1, -1), r=max(max_dist_rad.values()),
                                       return_distance=True, sort_results=True)
        idx, dist = idx[0], dist[0]
        dist_km = dist * 6371

        neighbor_num = train_num[idx]
        neighbor_cat = train_cat[idx]

        geo_dist = dist_km / 1000
        num_dist = np.mean(np.abs(test_num[i] - neighbor_num), axis=1) if len(idx) > 0 else np.array([])
        cat_dist = np.mean((test_cat[i] != neighbor_cat).astype(float), axis=1) if len(idx) > 0 else np.array([])

        combined = 0.5 * geo_dist + 1.0 * cat_dist + 0.5 * num_dist
        sort_idx = np.argsort(combined) if len(combined) > 0 else []
        idx = idx[sort_idx]
        dist_km = dist_km[sort_idx]
        neighbor_targets = targets[idx]

        for k in k_values:
            for d in max_dist_km:
                mask = dist_km <= d
                filtered = neighbor_targets[mask][:k]
                test_nn[f'nn{k}_max{d}'].append(filtered.mean() if len(filtered) > 0 else global_mean)

    for col in train_nn:
        train[col] = train_nn[col]
        test[col] = test_nn[col]

    print(f"Computed neighbor features: {list(train_nn.keys())}", flush=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("Engineering features...", flush=True)

def engineer_features(df, train_df=None, is_train=True):
    df = df.copy()

    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['is_renewable'] = (df['fuel_group'] == 'renewable').astype(int)
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    for fuel in ['Solar', 'Wind', 'Hydro', 'Gas', 'Coal', 'Oil', 'Waste', 'Nuclear', 'Biomass']:
        df[f'is_{fuel.lower()}'] = (df['primary_fuel'] == fuel).astype(int)

    # Fuel-specific patterns from EDA
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

train = engineer_features(train, train, is_train=True)
test = engineer_features(test, train, is_train=False)

# =============================================================================
# TARGET ENCODING
# =============================================================================

print("Computing target encodings...", flush=True)

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

# Find neighbor feature columns
nn_cols = [c for c in train.columns if c.startswith('nn_')]
# Prioritize the best ones
priority_nn = ['nn_nn10_max500_target_mean', 'nn_nn5_max500_target_mean',
               'nn_nn10_max200_target_mean', 'nn_nn5_max200_target_mean',
               'nn_nn10_max100_target_mean', 'nn_nn5_max100_target_mean',
               'nn_dist_to_nearest_underperf', 'nn_dist_to_nearest_normal']
nn_cols = [c for c in priority_nn if c in nn_cols] + [c for c in nn_cols if c not in priority_nn]

num_features = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
    'latitude', 'longitude', 'age_x_capacity',
    'is_fossil', 'is_renewable', 'has_other_fuel',
    'is_solar', 'is_wind', 'is_hydro', 'is_gas', 'is_coal', 'is_oil', 'is_waste', 'is_nuclear', 'is_biomass',
    'fossil_old_small', 'hydro_large', 'waste_large_old', 'solar_wind_small',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel', 'latitude_z_fuel', 'longitude_z_fuel',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank', 'latitude_fuel_rank', 'longitude_fuel_rank',
    'age_per_cap', 'cap_per_age', 'age_cap_int', 'lat_lon', 'cap_sq', 'age_sq',
] + nn_cols[:12]  # Use top 12 neighbor features

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
print("LIGHTGBM (Tuned)", flush=True)
print("="*60, flush=True)

# Multiple parameter sets to test
param_sets = [
    {'num_leaves': 63, 'learning_rate': 0.01, 'feature_fraction': 0.6,
     'bagging_fraction': 0.6, 'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
    {'num_leaves': 127, 'learning_rate': 0.008, 'feature_fraction': 0.5,
     'bagging_fraction': 0.5, 'min_child_samples': 15, 'reg_alpha': 0.05, 'reg_lambda': 0.05},
    {'num_leaves': 31, 'learning_rate': 0.015, 'feature_fraction': 0.7,
     'bagging_fraction': 0.7, 'min_child_samples': 30, 'reg_alpha': 0.2, 'reg_lambda': 0.2},
]

best_lgb_score = 0
best_lgb_oof = None
best_lgb_test = None

for i, params in enumerate(param_sets):
    print(f"\nTrying param set {i+1}...", flush=True)

    lgb_params = {
        'objective': 'binary', 'metric': ['auc'], 'boosting_type': 'gbdt',
        'scale_pos_weight': 2.0, 'bagging_freq': 5,
        'verbose': -1, 'seed': RANDOM_STATE, 'n_jobs': -1,
    }
    lgb_params.update(params)

    lgb_oof = np.zeros(len(y))
    lgb_test = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)

        model = lgb.train(
            lgb_params, tr_data, num_boost_round=2500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)]
        )

        lgb_oof[val_idx] = model.predict(X[val_idx])
        lgb_test += model.predict(X_test) / N_FOLDS

    score = composite(y, lgb_oof)
    print(f"  Result: Comp={score:.4f}", flush=True)

    if score > best_lgb_score:
        best_lgb_score = score
        best_lgb_oof = lgb_oof.copy()
        best_lgb_test = lgb_test.copy()

print(f"\nBest LightGBM: Comp={best_lgb_score:.4f}", flush=True)
lgb_oof = best_lgb_oof
lgb_test = best_lgb_test

# =============================================================================
# CATBOOST
# =============================================================================

print("\n" + "="*60, flush=True)
print("CATBOOST", flush=True)
print("="*60, flush=True)

os.makedirs('catboost_tmp', exist_ok=True)
os.environ['CATBOOST_INFO_DIR'] = 'catboost_tmp'

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
        iterations=1500, learning_rate=0.03, depth=7,
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0,
        eval_metric='AUC', early_stopping_rounds=75,
        train_dir=f'catboost_tmp/fold_{fold}'
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)

    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    cb_test += model.predict_proba(test_pool)[:, 1] / N_FOLDS

print(f"CatBoost: ROC={roc_auc_score(y, cb_oof):.4f}, AP={average_precision_score(y, cb_oof):.4f}, "
      f"Comp={composite(y, cb_oof):.4f}", flush=True)

# =============================================================================
# ENSEMBLE
# =============================================================================

print("\n" + "="*60, flush=True)
print("ENSEMBLE", flush=True)
print("="*60, flush=True)

best_score, best_w = 0, 0.9
for w in np.arange(0.80, 0.96, 0.01):
    weighted = w * lgb_oof + (1 - w) * cb_oof
    score = composite(y, weighted)
    if score > best_score:
        best_score, best_w = score, w

print(f"Best weight: {best_w:.2f} LightGBM + {1-best_w:.2f} CatBoost", flush=True)

final_oof = best_w * lgb_oof + (1 - best_w) * cb_oof
final_test = best_w * lgb_test + (1 - best_w) * cb_test

# =============================================================================
# FINAL
# =============================================================================

print("\n" + "="*60, flush=True)
print("FINAL RESULTS", flush=True)
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

import shutil
shutil.rmtree('catboost_tmp', ignore_errors=True)
