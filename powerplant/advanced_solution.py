"""
Advanced Solution - Stacking + Ranking Approach
Target: 0.88+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import BallTree
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 10

print("Loading data...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

target = train['underperforming'].values
global_mean = target.mean()

# =============================================================================
# EXTENSIVE FEATURE ENGINEERING
# =============================================================================

print("Engineering features...")

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
    df['age_cap_interaction'] = df['plant_age'] * df['capacity_log_mw']

    # Geographic
    df['lat_lon'] = df['latitude'] * df['longitude']
    df['lat_sq'] = df['latitude'] ** 2
    df['lon_sq'] = df['longitude'] ** 2
    df['dist_center'] = np.sqrt(df['latitude']**2 + df['longitude']**2)

    # Fuel-specific patterns (KEY from EDA)
    df['fossil_old_small'] = df['is_gas'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5) + \
                             df['is_coal'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5) + \
                             df['is_oil'] * df['plant_age'] / (df['capacity_log_mw'] + 0.5)
    df['hydro_large'] = df['is_hydro'] * df['capacity_log_mw']
    df['renewable_small'] = (df['is_solar'] + df['is_wind']) * (5 - df['capacity_log_mw'])

    # Ranking within fuel group (KEY for quantile-based target)
    if is_train:
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            df[f'{col}_fuel_rank'] = df.groupby('primary_fuel')[col].rank(pct=True)
    else:
        # For test, use train stats
        for col in ['capacity_log_mw', 'plant_age', 'latitude', 'longitude']:
            ranks = []
            for fuel in df['primary_fuel'].unique():
                mask = df['primary_fuel'] == fuel
                train_fuel = train_df[train_df['primary_fuel'] == fuel][col]
                test_fuel = df.loc[mask, col]
                # Use percentile ranking relative to train
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
    ['primary_fuel', 'lat_band'],
    ['primary_fuel', 'lon_band'],
    ['primary_fuel', 'owner_bucket'],
    ['owner_bucket', 'capacity_band'],
    ['owner_bucket', 'other_fuel1'],
    ['fuel_group', 'capacity_band'],
    ['primary_fuel', 'other_fuel1', 'capacity_band'],
]

for combo in combos:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

print("Computing target encodings...")
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
# NEIGHBOR FEATURES
# =============================================================================

print("Computing neighbor features...")
train_coords = train[['lat_rad', 'lon_rad']].values
test_coords = test[['lat_rad', 'lon_rad']].values
tree = BallTree(train_coords, metric='haversine')

# Multiple neighbor configurations
nn_configs = [(5, 100), (10, 200), (20, 500), (50, 1000)]
for k, max_d in nn_configs:
    max_rad = max_d / 6371

    # Train
    train_nn = []
    indices_arr = tree.query_radius(train_coords, r=max_rad, return_distance=False)
    for i in range(len(train)):
        indices = indices_arr[i]
        mask = indices != i
        indices = indices[mask]
        if len(indices) > 0:
            train_nn.append(target[indices[:k]].mean())
        else:
            train_nn.append(np.nan)
    train[f'nn{k}_{max_d}'] = train_nn

    # Test
    test_nn = []
    indices_arr = tree.query_radius(test_coords, r=max_rad, return_distance=False)
    for i in range(len(test)):
        indices = indices_arr[i]
        if len(indices) > 0:
            test_nn.append(target[indices[:k]].mean())
        else:
            test_nn.append(np.nan)
    test[f'nn{k}_{max_d}'] = test_nn

# =============================================================================
# PREPARE DATA
# =============================================================================

num_cols = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude', 'latitude', 'longitude', 'age_x_capacity',
    'age_per_cap', 'cap_per_age', 'age_cap_interaction', 'lat_lon', 'lat_sq', 'lon_sq', 'dist_center',
    'fossil_old_small', 'hydro_large', 'renewable_small',
    'capacity_log_mw_fuel_rank', 'plant_age_fuel_rank', 'latitude_fuel_rank', 'longitude_fuel_rank',
    'capacity_log_mw_z_fuel', 'plant_age_z_fuel',
]

nn_cols = [f'nn{k}_{d}' for k, d in nn_configs]
num_cols.extend(nn_cols)

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

print(f"Total features: {len(all_features)}")

# =============================================================================
# TRAINING - LEVEL 1
# =============================================================================

def composite(y_true, y_pred):
    roc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return 0.7 * roc + 0.3 * ap

print("\n" + "="*60)
print("LEVEL 1 TRAINING")
print("="*60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Storage for level 2
l1_oof = np.zeros((len(y), 7))
l1_test = np.zeros((len(X_test), 7))

# Categorical indices for CatBoost (last 7 features)
cat_indices = list(range(len(num_cols), len(all_features)))

# 1. LightGBM
print("\n1. LightGBM...")
lgb_params = {
    'objective': 'binary', 'metric': ['auc'],
    'num_leaves': 255, 'learning_rate': 0.01,
    'feature_fraction': 0.6, 'bagging_fraction': 0.6, 'bagging_freq': 3,
    'min_child_samples': 5, 'scale_pos_weight': 2.0,
    'verbose': -1, 'seed': RANDOM_STATE, 'n_jobs': -1
}

lgb_oof = np.zeros(len(y))
lgb_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx])
    model = lgb.train(lgb_params, tr_data, num_boost_round=3000, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    lgb_oof[val_idx] = model.predict(X[val_idx])
    lgb_pred += model.predict(X_test) / N_FOLDS

l1_oof[:, 0] = lgb_oof
l1_test[:, 0] = lgb_pred
print(f"   LightGBM: {composite(y, lgb_oof):.4f}")

# 2. LightGBM variant
print("2. LightGBM (variant)...")
lgb_params2 = {
    'objective': 'binary', 'metric': ['auc'],
    'num_leaves': 63, 'learning_rate': 0.02,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 20, 'scale_pos_weight': 2.5,
    'verbose': -1, 'seed': RANDOM_STATE + 1, 'n_jobs': -1
}

lgb_oof2 = np.zeros(len(y))
lgb_pred2 = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx])
    model = lgb.train(lgb_params2, tr_data, num_boost_round=2000, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    lgb_oof2[val_idx] = model.predict(X[val_idx])
    lgb_pred2 += model.predict(X_test) / N_FOLDS

l1_oof[:, 1] = lgb_oof2
l1_test[:, 1] = lgb_pred2
print(f"   LightGBM (var): {composite(y, lgb_oof2):.4f}")

# 3. XGBoost
print("3. XGBoost...")
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 8, 'learning_rate': 0.01,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'scale_pos_weight': 2.0, 'seed': RANDOM_STATE, 'n_jobs': -1
}

xgb_oof = np.zeros(len(y))
xgb_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    val_data = xgb.DMatrix(X[val_idx], label=y[val_idx])
    test_data = xgb.DMatrix(X_test)
    model = xgb.train(xgb_params, tr_data, num_boost_round=3000, evals=[(val_data, 'val')],
                      early_stopping_rounds=150, verbose_eval=False)
    xgb_oof[val_idx] = model.predict(val_data)
    xgb_pred += model.predict(test_data) / N_FOLDS

l1_oof[:, 2] = xgb_oof
l1_test[:, 2] = xgb_pred
print(f"   XGBoost: {composite(y, xgb_oof):.4f}")

# 4. XGBoost variant
print("4. XGBoost (variant)...")
xgb_params2 = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'max_depth': 6, 'learning_rate': 0.02,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': 2.5, 'seed': RANDOM_STATE + 1, 'n_jobs': -1
}

xgb_oof2 = np.zeros(len(y))
xgb_pred2 = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
    val_data = xgb.DMatrix(X[val_idx], label=y[val_idx])
    test_data = xgb.DMatrix(X_test)
    model = xgb.train(xgb_params2, tr_data, num_boost_round=2000, evals=[(val_data, 'val')],
                      early_stopping_rounds=100, verbose_eval=False)
    xgb_oof2[val_idx] = model.predict(val_data)
    xgb_pred2 += model.predict(test_data) / N_FOLDS

l1_oof[:, 3] = xgb_oof2
l1_test[:, 3] = xgb_pred2
print(f"   XGBoost (var): {composite(y, xgb_oof2):.4f}")

# 5. Random Forest
print("5. Random Forest...")
rf_oof = np.zeros(len(y))
rf_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                   class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X[tr_idx], y[tr_idx])
    rf_oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    rf_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

l1_oof[:, 4] = rf_oof
l1_test[:, 4] = rf_pred
print(f"   Random Forest: {composite(y, rf_oof):.4f}")

# 6. Extra Trees
print("6. Extra Trees...")
et_oof = np.zeros(len(y))
et_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = ExtraTreesClassifier(n_estimators=500, max_depth=15, min_samples_leaf=5,
                                 class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X[tr_idx], y[tr_idx])
    et_oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
    et_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

l1_oof[:, 5] = et_oof
l1_test[:, 5] = et_pred
print(f"   Extra Trees: {composite(y, et_oof):.4f}")

# 7. CatBoost
print("7. CatBoost...")
cb_oof = np.zeros(len(y))
cb_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    train_pool = Pool(X[tr_idx], label=y[tr_idx], cat_features=cat_indices)
    val_pool = Pool(X[val_idx], label=y[val_idx], cat_features=cat_indices)
    test_pool = Pool(X_test, cat_features=cat_indices)

    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.01, depth=8,
        l2_leaf_reg=3, scale_pos_weight=2.0,
        random_seed=RANDOM_STATE, verbose=0,
        eval_metric='AUC', early_stopping_rounds=150
    )
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    cb_oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    cb_pred += model.predict_proba(test_pool)[:, 1] / N_FOLDS

l1_oof[:, 6] = cb_oof
l1_test[:, 6] = cb_pred
print(f"   CatBoost: {composite(y, cb_oof):.4f}")

# =============================================================================
# LEVEL 2 - STACKING
# =============================================================================

print("\n" + "="*60)
print("LEVEL 2 STACKING")
print("="*60)

# Simple average
avg_l1 = l1_oof.mean(axis=1)
avg_test_l1 = l1_test.mean(axis=1)
print(f"Simple average: {composite(y, avg_l1):.4f}")

# Logistic regression stacking
lr_oof = np.zeros(len(y))
lr_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(l1_oof, y)):
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(l1_oof[tr_idx], y[tr_idx])
    lr_oof[val_idx] = lr.predict_proba(l1_oof[val_idx])[:, 1]
    lr_pred += lr.predict_proba(l1_test)[:, 1] / N_FOLDS

print(f"LR stacking: {composite(y, lr_oof):.4f}")

# Ridge stacking
ridge_oof = np.zeros(len(y))
ridge_pred = np.zeros(len(X_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(l1_oof, y)):
    ridge = Ridge(alpha=1.0)
    ridge.fit(l1_oof[tr_idx], y[tr_idx])
    ridge_oof[val_idx] = ridge.predict(l1_oof[val_idx]).clip(0, 1)
    ridge_pred += ridge.predict(l1_test).clip(0, 1) / N_FOLDS

print(f"Ridge stacking: {composite(y, ridge_oof):.4f}")

# =============================================================================
# BLEND ALL
# =============================================================================

print("\n" + "="*60)
print("FINAL BLEND")
print("="*60)

# Blend all level 2
final_oof = 0.3 * avg_l1 + 0.35 * lr_oof + 0.35 * ridge_oof
final_pred = 0.3 * avg_test_l1 + 0.35 * lr_pred + 0.35 * ridge_pred

roc = roc_auc_score(y, final_oof)
ap = average_precision_score(y, final_oof)
comp = 0.7 * roc + 0.3 * ap

print(f"Final ROC-AUC: {roc:.4f}")
print(f"Final AP: {ap:.4f}")
print(f"Final Composite: {comp:.4f}")

# =============================================================================
# SUBMISSION
# =============================================================================

submission = pd.DataFrame({
    'id': test['id'],
    'underperforming': final_pred.clip(0, 1)
})
submission.to_csv("submission.csv", index=False)

print(f"\nSaved submission.csv")
print(f"Range: [{final_pred.min():.4f}, {final_pred.max():.4f}]")
print(f"Mean: {final_pred.mean():.4f}")
