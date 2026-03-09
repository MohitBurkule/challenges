"""
Final Solution - Using all insights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Loading...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")
target = train["underperforming"].values
global_mean = target.mean()

def add_features(df):
    df = df.copy()
    df["is_fossil"] = (df["fuel_group"] == "fossil").astype(int)
    df["is_renewable"] = (df["fuel_group"] == "renewable").astype(int)
    df["has_other_fuel"] = (df["other_fuel1"] != "__NONE__").astype(int)

    for fuel in ["Solar", "Wind", "Hydro", "Gas", "Coal", "Oil"]:
        df[f"is_{fuel.lower()}"] = (df["primary_fuel"] == fuel).astype(int)

    # KEY: Fuel-specific patterns from EDA
    df["fossil_old_small"] = df["is_fossil"] * df["plant_age"] / (df["capacity_log_mw"] + 0.5)
    df["hydro_large"] = df["is_hydro"] * df["capacity_log_mw"]
    df["gas_coal_age"] = (df["is_gas"] | df["is_coal"]).astype(int) * df["plant_age"]

    # Relative to fuel mean
    for col in ["capacity_log_mw", "plant_age"]:
        mean = df.groupby("primary_fuel")[col].transform("mean")
        std = df.groupby("primary_fuel")[col].transform("std") + 0.01
        df[f"{col}_z_fuel"] = (df[col] - mean) / std

    df["age_per_cap"] = df["plant_age"] / (df["capacity_log_mw"] + 0.1)
    return df

train = add_features(train)
test = add_features(test)

# Target encoding
te_cols = ["primary_fuel", "other_fuel1", "owner_bucket", "fuel_group", "capacity_band", "lat_band", "lon_band"]

combos = [["primary_fuel", "other_fuel1"], ["primary_fuel", "capacity_band"], ["owner_bucket", "primary_fuel"]]
for combo in combos:
    col_name = "_".join(combo)
    train[col_name] = train[combo].astype(str).agg("_".join, axis=1)
    test[col_name] = test[combo].astype(str).agg("_".join, axis=1)
    te_cols.append(col_name)

print("Computing target encodings...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for col in te_cols:
    train[f"{col}_te"] = np.nan
    for tr_idx, val_idx in skf.split(train, target):
        agg = train.iloc[tr_idx].groupby(col)["underperforming"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + 10 * global_mean) / (agg["count"] + 10)
        train.loc[val_idx, f"{col}_te"] = train.iloc[val_idx][col].map(agg["smoothed"]).fillna(global_mean)

    agg = train.groupby(col)["underperforming"].agg(["mean", "count"])
    agg["smoothed"] = (agg["count"] * agg["mean"] + 10 * global_mean) / (agg["count"] + 10)
    test[f"{col}_te"] = test[col].map(agg["smoothed"]).fillna(global_mean)

# Features
num_cols = ["capacity_mw", "capacity_log_mw", "plant_age", "abs_latitude", "latitude", "longitude", "age_x_capacity",
            "is_fossil", "is_renewable", "has_other_fuel", "is_solar", "is_wind", "is_hydro", "is_gas", "is_coal", "is_oil",
            "fossil_old_small", "hydro_large", "gas_coal_age", "capacity_log_mw_z_fuel", "plant_age_z_fuel", "age_per_cap"]
te_features = [f"{c}_te" for c in te_cols]
cat_cols = ["fuel_group", "primary_fuel", "other_fuel1", "owner_bucket", "capacity_band", "lat_band", "lon_band"]

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

features = num_cols + te_features + cat_cols
features = [f for f in features if f in train.columns]

X = train[features].values.astype(float)
X_test = test[features].values.astype(float)
y = target

for i in range(X.shape[1]):
    mask = np.isnan(X[:, i])
    if mask.any():
        med = np.nanmedian(X[:, i])
        X[mask, i] = med
        X_test[np.isnan(X_test[:, i]), i] = med

print(f"Features: {len(features)}")

# Train
params = {
    "objective": "binary", "metric": ["auc"],
    "num_leaves": 127, "learning_rate": 0.02,
    "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
    "min_child_samples": 10, "scale_pos_weight": 2.0,
    "verbose": -1, "seed": 42, "n_jobs": -1
}

oof = np.zeros(len(y))
test_pred = np.zeros(len(X_test))
scores = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    tr_data = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    val_data = lgb.Dataset(X[val_idx], label=y[val_idx], reference=tr_data)
    model = lgb.train(params, tr_data, num_boost_round=2000, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[val_idx] = model.predict(X[val_idx])
    test_pred += model.predict(X_test) / 5
    roc = roc_auc_score(y[val_idx], oof[val_idx])
    ap = average_precision_score(y[val_idx], oof[val_idx])
    scores.append(0.7 * roc + 0.3 * ap)

roc = roc_auc_score(y, oof)
ap = average_precision_score(y, oof)
composite = 0.7 * roc + 0.3 * ap

print(f"CV: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print(f"Overall ROC: {roc:.4f}")
print(f"Overall AP: {ap:.4f}")
print(f"Composite: {composite:.4f}")

pd.DataFrame({"id": test["id"], "underperforming": test_pred}).to_csv("submission.csv", index=False)
print("Saved submission.csv")
