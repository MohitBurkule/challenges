"""
Power Plant Underperformance - Ablation Study & Improved Model
Tests removing neighbor features and tries improved approaches.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
N_FOLDS = 5

# Load data
print("Loading data...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

target = train['underperforming'].values

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_features(df):
    df = df.copy()

    df['lat_rad'] = np.radians(df['latitude'])
    df['lon_rad'] = np.radians(df['longitude'])

    # Age bins
    df['age_bin'] = pd.cut(df['plant_age'], bins=[0, 5, 15, 30, 50, 150],
                           labels=['new', 'young', 'mid', 'old', 'ancient'])

    # Interactions
    df['age_per_capacity'] = df['plant_age'] / (df['capacity_log_mw'] + 0.1)
    df['capacity_per_age'] = df['capacity_log_mw'] / (df['plant_age'] + 0.1)

    # Geographic
    df['lat_lon_interaction'] = df['latitude'] * df['longitude']
    df['dist_from_center'] = np.sqrt(df['latitude']**2 + df['longitude']**2)

    # Fossil indicator
    df['is_fossil'] = (df['fuel_group'] == 'fossil').astype(int)
    df['has_other_fuel'] = (df['other_fuel1'] != '__NONE__').astype(int)

    # Fuel-specific features (key insight from EDA!)
    df['is_hydro'] = (df['primary_fuel'] == 'Hydro').astype(int)
    df['is_solar'] = (df['primary_fuel'] == 'Solar').astype(int)
    df['is_wind'] = (df['primary_fuel'] == 'Wind').astype(int)
    df['is_gas'] = (df['primary_fuel'] == 'Gas').astype(int)
    df['is_coal'] = (df['primary_fuel'] == 'Coal').astype(int)
    df['is_oil'] = (df['primary_fuel'] == 'Oil').astype(int)

    # Fuel × capacity interactions (patterns differ by fuel!)
    for fuel in ['Gas', 'Coal', 'Oil', 'Hydro', 'Solar', 'Wind']:
        fuel_mask = df['primary_fuel'] == fuel
        df[f'{fuel.lower()}_capacity'] = df['capacity_log_mw'] * fuel_mask.astype(int)
        df[f'{fuel.lower()}_age'] = df['plant_age'] * fuel_mask.astype(int)

    return df

train = add_features(train)
test = add_features(test)

# =============================================================================
# TARGET ENCODING
# =============================================================================

class TargetEncoderCV:
    def __init__(self, cols, smoothing=10):
        self.cols = cols
        self.smoothing = smoothing

    def fit_transform(self, df, target):
        result = pd.DataFrame(index=df.index)
        global_mean = target.mean()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for col in self.cols:
            result[f'{col}_te'] = np.nan

            for train_idx, val_idx in skf.split(df, target):
                train_fold = df.iloc[train_idx]
                val_fold = df.iloc[val_idx]

                agg = train_fold.groupby(col)['underperforming'].agg(['mean', 'count'])
                agg.columns = ['mean', 'count']
                agg['smoothed'] = (agg['count'] * agg['mean'] + self.smoothing * global_mean) / (agg['count'] + self.smoothing)

                result.loc[val_idx, f'{col}_te'] = val_fold[col].map(agg['smoothed']).fillna(global_mean)

        return result

    def transform(self, train_df, test_df, target):
        result = pd.DataFrame(index=test_df.index)
        global_mean = target.mean()

        for col in self.cols:
            agg = train_df.groupby(col)['underperforming'].agg(['mean', 'count'])
            agg.columns = ['mean', 'count']
            agg['smoothed'] = (agg['count'] * agg['mean'] + self.smoothing * global_mean) / (agg['count'] + self.smoothing)
            result[f'{col}_te'] = test_df[col].map(agg['smoothed']).fillna(global_mean)

        return result

# TE columns
te_cols = [
    'primary_fuel', 'other_fuel1', 'owner_bucket', 'capacity_band',
    'lat_band', 'lon_band', 'age_bin', 'fuel_group'
]

# Combo columns
combos = [
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'lat_band'],
    ['primary_fuel', 'lon_band'],
    ['fuel_group', 'capacity_band'],
    ['owner_bucket', 'capacity_band'],
]

for combo in combos:
    col_name = '_'.join(combo)
    train[col_name] = train[combo].astype(str).agg('_'.join, axis=1)
    test[col_name] = test[combo].astype(str).agg('_'.join, axis=1)
    te_cols.append(col_name)

print("Computing target encodings...")
te = TargetEncoderCV(cols=te_cols, smoothing=10)
train_te = te.fit_transform(train, target)
test_te = te.transform(train, test, target)

for col in te_cols:
    train[f'{col}_te'] = train_te[f'{col}_te']
    test[f'{col}_te'] = test_te[f'{col}_te']

# =============================================================================
# PREPARE FEATURES
# =============================================================================

base_num_features = [
    'capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
    'latitude', 'longitude', 'age_x_capacity',
    'age_per_capacity', 'capacity_per_age',
    'lat_lon_interaction', 'dist_from_center',
    'is_fossil', 'has_other_ffuel', 'is_hydro', 'is_solar', 'is_wind',
    'is_gas', 'is_coal', 'is_oil',
]

# Fuel-specific interactions
fuel_interactions = []
for fuel in ['gas', 'coal', 'oil', 'hydro', 'solar', 'wind']:
    fuel_interactions.extend([f'{fuel}_capacity', f'{fuel}_age'])

te_features = [f'{col}_te' for col in te_cols]

cat_features = ['fuel_group', 'primary_fuel', 'other_fuel1',
                'owner_bucket', 'capacity_band', 'lat_band', 'lon_band', 'age_bin']

# Label encode
for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# Fix typo in feature name
base_num_features = [f for f in base_num_features if 'ffuel' not in f]
base_num_features.append('has_other_fuel')

# =============================================================================
# TRAIN FUNCTION
# =============================================================================

def train_model(X_train, y_train, X_test, use_weights=True):
    """Train LightGBM with CV"""

    params = {
        'objective': 'binary',
        'metric': ['auc', 'average_precision'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': RANDOM_STATE,
        'n_jobs': -1,
    }

    if use_weights:
        params['scale_pos_weight'] = 2.0

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / N_FOLDS

        roc = roc_auc_score(y_val, oof_preds[val_idx])
        ap = average_precision_score(y_val, oof_preds[val_idx])
        scores.append(0.7 * roc + 0.3 * ap)

    overall_roc = roc_auc_score(y_train, oof_preds)
    overall_ap = average_precision_score(y_train, oof_preds)
    overall_score = 0.7 * overall_roc + 0.3 * overall_ap

    return {
        'cv_mean': np.mean(scores),
        'cv_std': np.std(scores),
        'overall_score': overall_score,
        'roc': overall_roc,
        'ap': overall_ap,
        'test_preds': test_preds,
        'oof_preds': oof_preds
    }

# =============================================================================
# EXPERIMENT 1: WITHOUT NEIGHBOR FEATURES
# =============================================================================
print("\n" + "="*60)
print("EXPERIMENT 1: WITHOUT NEIGHBOR FEATURES")
print("="*60)

features_no_neighbor = base_num_features + fuel_interactions + te_features + cat_features
features_no_neighbor = [f for f in features_no_neighbor if f in train.columns]

X_no_neighbor = train[features_no_neighbor].values
X_test_no_neighbor = test[features_no_neighbor].values

# Fill NaN
for i, col in enumerate(features_no_neighbor):
    mask = np.isnan(X_no_neighbor[:, i])
    if mask.any():
        median = np.nanmedian(X_no_neighbor[:, i])
        X_no_neighbor[mask, i] = median
        X_test_no_neighbor[np.isnan(X_test_no_neighbor[:, i])] = median

result_no_neighbor = train_model(X_no_neighbor, target, X_test_no_neighbor)

print(f"CV Mean: {result_no_neighbor['cv_mean']:.4f} ± {result_no_neighbor['cv_std']:.4f}")
print(f"Overall: ROC={result_no_neighbor['roc']:.4f}, AP={result_no_neighbor['ap']:.4f}")
print(f"Composite Score: {result_no_neighbor['overall_score']:.4f}")

# =============================================================================
# EXPERIMENT 2: WITH NEIGHBOR FEATURES (load from file)
# =============================================================================
print("\n" + "="*60)
print("EXPERIMENT 2: WITH NEIGHBOR FEATURES")
print("="*60)

# Load pre-computed neighbor features
neighbor_feats = pd.read_csv("neighbor_features.csv")
neighbor_cols = [c for c in neighbor_feats.columns if c != 'underperforming']

for col in neighbor_cols:
    train[f'neighbor_{col}'] = neighbor_feats[col]
    # For test, need to compute - use train mean as fallback
    test[f'neighbor_{col}'] = train[f'neighbor_{col}'].mean()

features_with_neighbor = features_no_neighbor + [f'neighbor_{c}' for c in neighbor_cols]

X_with_neighbor = train[features_with_neighbor].values
X_test_with_neighbor = test[features_with_neighbor].values

# Fill NaN
for i in range(X_with_neighbor.shape[1]):
    mask = np.isnan(X_with_neighbor[:, i])
    if mask.any():
        median = np.nanmedian(X_with_neighbor[:, i])
        X_with_neighbor[mask, i] = median
        X_test_with_neighbor[np.isnan(X_test_with_neighbor[:, i])] = median

result_with_neighbor = train_model(X_with_neighbor, target, X_test_with_neighbor)

print(f"CV Mean: {result_with_neighbor['cv_mean']:.4f} ± {result_with_neighbor['cv_std']:.4f}")
print(f"Overall: ROC={result_with_neighbor['roc']:.4f}, AP={result_with_neighbor['ap']:.4f}")
print(f"Composite Score: {result_with_neighbor['overall_score']:.4f}")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"{'Configuration':<30} {'CV Score':<15} {'Overall Score':<15}")
print("-" * 60)
print(f"{'Without Neighbor Features':<30} {result_no_neighbor['cv_mean']:.4f} ± {result_no_neighbor['cv_std']:.4f}   {result_no_neighbor['overall_score']:.4f}")
print(f"{'With Neighbor Features':<30} {result_with_neighbor['cv_mean']:.4f} ± {result_with_neighbor['cv_std']:.4f}   {result_with_neighbor['overall_score']:.4f}")
print(f"{'Improvement':<30} {'+' + str(round((result_with_neighbor['cv_mean'] - result_no_neighbor['cv_mean']), 4)):<15}")
