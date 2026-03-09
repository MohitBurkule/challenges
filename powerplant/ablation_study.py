"""
Ablation Study - Testing different feature configurations
Goal: Understand what features help vs hurt the score
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import BallTree
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
# BASIC FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    df = df.copy()
    df['lat_rad'] = np.radians(df['latitude'])
    df['lon_rad'] = np.radians(df['longitude'])
    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['has_other'] = (df['other_fuel1'] != '__NONE__').astype(int)
    return df

train = engineer_features(train)
test = engineer_features(test)

# =============================================================================
# TARGET ENCODING
# =============================================================================

te_cols = ['primary_fuel', 'other_fuel1', 'owner_bucket', 'fuel_group']

for combo in [['primary_fuel', 'other_fuel1'], ['primary_fuel', 'capacity_band']]:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

# CV-based TE
for col in te_cols:
    train[f'{col}_te'] = np.nan
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)['underperforming'].agg(['mean', 'count'])
        agg['smoothed'] = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
        train.loc[val_idx, f'{col}_te'] = train.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean)

    agg = train.groupby(col)['underperforming'].agg(['mean', 'count'])
    agg['smoothed'] = (agg['count'] * agg['mean'] + 10 * global_mean) / (agg['count'] + 10)
    test[f'{col}_te'] = test[col].map(agg['smoothed']).fillna(global_mean)

# =============================================================================
# NEIGHBOR FEATURES (simplified)
# =============================================================================

def get_neighbor_targets(train_df, k=10, max_dist=500):
    """Get mean target of k nearest plants within max_dist km"""
    train_coords = train_df[['lat_rad', 'lon_rad']].values
    train_targets = train_df['underperforming'].values

    tree = BallTree(train_coords, metric='haversine')
    max_rad = max_dist / 6371

    results = []
    for i in range(len(train_df)):
        indices, distances = tree.query_radius(
            train_coords[i:i+1], r=max_rad, return_distance=True, sort_results=True)
        indices = indices[0]
        mask = indices != i
        indices = indices[mask]
        if len(indices) > 0:
            results.append(train_targets[indices[:k]].mean())
        else:
            results.append(np.nan)

    return np.array(results)

print("Computing neighbor features...")
train['nn_target_mean'] = get_neighbor_targets(train, k=10, max_dist=500)
test['nn_target_mean'] = global_mean  # Placeholder for test

# =============================================================================
# ABLATION EXPERIMENTS
# =============================================================================

def evaluate(X, y, name=""):
    """Evaluate using LightGBM with 5-fold CV"""
    # Fill NaN
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if mask.any():
            med = np.nanmedian(X[:, i])
            X[mask, i] = med

    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    scores = []

    for tr_idx, val_idx in skf.split(X, y):
        tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
        val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)

        model = lgb.train(
            params, tr_data, num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        oof[val_idx] = model.predict(X[val_idx])
        roc = roc_auc_score(y[val_idx], oof[val_idx])
        ap = average_precision_score(y[val_idx], oof[val_idx])
        scores.append(0.7 * roc + 0.3 * ap)

    mean_score = np.mean(scores)
    print(f"{name}: {mean_score:.4f} ± {np.std(scores):.4f}")
    return mean_score, oof

# Define feature sets
base_num = ['capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
           'latitude', 'longitude', 'age_x_capacity']

cat_cols = ['fuel_group', 'primary_fuel', 'other_fuel1', 'owner_bucket',
           'capacity_band', 'lat_band', 'lon_band']

te_feat = [f'{c}_te' for c in te_cols]

# Label encode categoricals
le = LabelEncoder()
for col in cat_cols:
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

print("\n" + "="*60)
print("ABLATION STUDY")
print("="*60)

results = {}

# 1. Only base numerical features
X = train[base_num].values
score, _ = evaluate(X, target, "1. Base numerical only")
results['base_num'] = score

# 2. Add categorical features
X = train[base_num + cat_cols].values
score, _ = evaluate(X, target, "2. + Categorical")
results['+cat'] = score

# 3. Add target encodings
X = train[base_num + cat_cols + te_feat].values
score, _ = evaluate(X, target, "3. + Target Encoding")
results['+te'] = score

# 4. Add neighbor features (NO neighbor features)
X = train[base_num + cat_cols + te_feat + ['nn_target_mean']].values
score, _ = evaluate(X, target, "4. + Neighbor Target Mean")
results['+nn'] = score

# 5. Remove neighbor, see impact
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"Without neighbor: {results['+te']:.4f}")
print(f"With neighbor: {results['+nn']:.4f}")
print(f"Difference: {results['+nn'] - results['+te']:+.4f}")

# =============================================================================
# NOW TRY AGGRESSIVE FEATURE ENGINEERING
# =============================================================================

print("\n" + "="*60)
print("TRYING MORE FEATURES")
print("="*60)

# Add more features
train['age_per_cap'] = train['plant_age'] / (train['capacity_log_mw'] + 0.1)
test['age_per_cap'] = test['plant_age'] / (test['capacity_log_mw'] + 0.1)

# Fuel-specific features (key insight from EDA!)
for fuel in ['Gas', 'Coal', 'Oil', 'Hydro', 'Solar', 'Wind']:
    train[f'is_{fuel.lower()}'] = (train['primary_fuel'] == fuel).astype(int)
    test[f'is_{fuel.lower()}'] = (test['primary_fuel'] == fuel).astype(int)
    train[f'{fuel.lower()}_age'] = train[f'is_{fuel.lower()}'] * train['plant_age']
    test[f'{fuel.lower()}_age'] = test[f'is_{fuel.lower()}'] * test['plant_age']

# Fossil-specific (underperformers are old + small)
train['fossil_old_small'] = train['is_fossil'] * train['plant_age'] / (train['capacity_log_mw'] + 0.1)
test['fossil_old_small'] = test['is_fossil'] * test['plant_age'] / (test['capacity_log_mw'] + 0.1)

# Hydro-specific (underperformers are LARGE!)
train['hydro_large'] = train['is_hydro'] * train['capacity_log_mw']
test['hydro_large'] = test['is_hydro'] * test['capacity_log_mw']

extra_feat = ['age_per_cap', 'fossil_old_small', 'hydro_large']
extra_feat += [f'is_{f.lower()}' for f in ['Gas', 'Coal', 'Oil', 'Hydro', 'Solar', 'Wind']]
extra_feat += [f'{f.lower()}_age' for f in ['Gas', 'Coal', 'Oil', 'Hydro', 'Solar', 'Wind']]

X = train[base_num + cat_cols + te_feat + ['nn_target_mean'] + extra_feat].values
score, _ = evaluate(X, target, "5. + Fuel-specific features")
results['+fuel_specific'] = score

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score:.4f}")
