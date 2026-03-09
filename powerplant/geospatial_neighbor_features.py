"""
Optimized Nearest Neighbor Feature Engineering with Grid Search
Uses vectorized operations and sklearn's BallTree for fast distance calculations.

Run with: python geospatial_neighbor_features.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
COLOR_0 = '#2ecc71'
COLOR_1 = '#e74c3c'
TARGET_COLORS = [COLOR_0, COLOR_1]

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading data...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

print(f"Train: {train.shape}, Test: {test.shape}")

# Convert lat/lon to radians for haversine
train['lat_rad'] = np.radians(train['latitude'])
train['lon_rad'] = np.radians(train['longitude'])
test['lat_rad'] = np.radians(test['latitude'])
test['lon_rad'] = np.radians(test['longitude'])

# =============================================================================
# FAST NEAREST NEIGHBOR USING BALL TREE
# =============================================================================

def build_feature_matrix(df, cat_cols, num_cols):
    """Build feature matrix for distance calculation"""
    # Coordinates (in radians for haversine)
    coords = df[['lat_rad', 'lon_rad']].values

    # Numerical features (normalized)
    num_features = np.zeros((len(df), len(num_cols)))
    for i, col in enumerate(num_cols):
        num_features[:, i] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)

    # Categorical features (encoded)
    cat_features = np.zeros((len(df), len(cat_cols)))
    encoders = {}
    for i, col in enumerate(cat_cols):
        le = LabelEncoder()
        cat_features[:, i] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return coords, num_features, cat_features, encoders

def compute_neighbor_features_fast(df, coords, num_features, cat_features, cat_cols,
                                    k_values=[1, 3, 5, 10, 20], max_dist_km=[50, 100, 200, 500],
                                    weight_geo=1.0, weight_cat=1.0, weight_num=1.0):
    """
    Fast computation of neighbor features using BallTree.
    """
    n_samples = len(df)
    targets = df['underperforming'].values

    # Build BallTree for geographic queries (haversine distance)
    tree = BallTree(coords, metric='haversine')

    # Results storage
    results = {f'nn{k}_max{d}_target_mean': [] for k in k_values for d in max_dist_km}
    results.update({f'nn{k}_max{d}_count': [] for k in k_values for d in max_dist_km})
    results.update({f'nn{k}_max{d}_dist_mean': [] for k in k_values for d in max_dist_km})
    results['dist_to_nearest_underperf'] = []
    results['dist_to_nearest_normal'] = []

    # Convert max distances to radians
    max_dist_rad = {d: d / 6371 for d in max_dist_km}

    print(f"Computing neighbor features for {n_samples} samples...")

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  Row {i}/{n_samples}...")

        row_coords = coords[i].reshape(1, -1)
        row_num = num_features[i]
        row_cat = cat_features[i]
        row_target = targets[i]

        # Query tree for all neighbors within max distance
        max_query_dist = max(max_dist_rad.values())
        indices, distances = tree.query_radius(row_coords, r=max_query_dist, return_distance=True, sort_results=True)

        indices = indices[0]
        distances_rad = distances[0]

        # Remove self
        mask = indices != i
        indices = indices[mask]
        distances_rad = distances_rad[mask]
        distances_km = distances_rad * 6371

        if len(indices) == 0:
            # No neighbors found
            for k in k_values:
                for d in max_dist_km:
                    results[f'nn{k}_max{d}_target_mean'].append(np.nan)
                    results[f'nn{k}_max{d}_count'].append(0)
                    results[f'nn{k}_max{d}_dist_mean'].append(np.nan)
            results['dist_to_nearest_underperf'].append(np.nan)
            results['dist_to_nearest_normal'].append(np.nan)
            continue

        # Compute combined distance for ranking
        neighbor_num = num_features[indices]
        neighbor_cat = cat_features[indices]
        neighbor_targets = targets[indices]

        # Normalize distances
        geo_dist_norm = distances_km / 1000  # Scale to 0-1 range
        num_dist = np.mean(np.abs(row_num - neighbor_num), axis=1) if num_features.shape[1] > 0 else np.zeros(len(indices))
        cat_dist = np.mean((row_cat != neighbor_cat).astype(float), axis=1) if cat_features.shape[1] > 0 else np.zeros(len(indices))

        combined_dist = (weight_geo * geo_dist_norm + weight_cat * cat_dist + weight_num * num_dist)

        # Sort by combined distance
        sort_idx = np.argsort(combined_dist)
        indices = indices[sort_idx]
        distances_km = distances_km[sort_idx]
        neighbor_targets = neighbor_targets[sort_idx]
        combined_dist = combined_dist[sort_idx]

        # Compute features for each k and max_dist
        for k in k_values:
            for d in max_dist_km:
                # Filter by max geographic distance
                mask = distances_km <= d
                filtered_idx = indices[mask]
                filtered_targets = neighbor_targets[mask]
                filtered_dists = distances_km[mask]

                if len(filtered_idx) == 0:
                    results[f'nn{k}_max{d}_target_mean'].append(np.nan)
                    results[f'nn{k}_max{d}_count'].append(0)
                    results[f'nn{k}_max{d}_dist_mean'].append(np.nan)
                else:
                    # Take k nearest by combined distance
                    k_nearest_targets = filtered_targets[:k]
                    k_nearest_dists = filtered_dists[:k]

                    results[f'nn{k}_max{d}_target_mean'].append(np.mean(k_nearest_targets) if len(k_nearest_targets) > 0 else np.nan)
                    results[f'nn{k}_max{d}_count'].append(min(len(filtered_idx), k))
                    results[f'nn{k}_max{d}_dist_mean'].append(np.mean(k_nearest_dists) if len(k_nearest_dists) > 0 else np.nan)

        # Distance to nearest underperforming / normal
        underperf_mask = neighbor_targets == 1
        normal_mask = neighbor_targets == 0

        if underperf_mask.any():
            results['dist_to_nearest_underperf'].append(distances_km[underperf_mask][0])
        else:
            results['dist_to_nearest_underperf'].append(np.nan)

        if normal_mask.any():
            results['dist_to_nearest_normal'].append(distances_km[normal_mask][0])
        else:
            results['dist_to_nearest_normal'].append(np.nan)

    return pd.DataFrame(results)

# =============================================================================
# GRID SEARCH CONFIGURATION
# =============================================================================

CATEGORICAL_COMBOS = [
    [],
    ['primary_fuel'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'other_fuel1'],
    ['fuel_group'],
    ['fuel_group', 'capacity_band'],
]

NUMERICAL_COMBOS = [
    [],
    ['capacity_log_mw'],
    ['plant_age'],
    ['capacity_log_mw', 'plant_age'],
]

WEIGHT_COMBOS = [
    {'geo': 1.0, 'cat': 0.5, 'num': 0.5},
    {'geo': 1.0, 'cat': 1.0, 'num': 0.5},
    {'geo': 1.0, 'cat': 2.0, 'num': 1.0},
    {'geo': 0.5, 'cat': 1.0, 'num': 0.5},
]

K_VALUES = [1, 3, 5, 10]
MAX_DIST_VALUES = [50, 100, 200, 500]

# =============================================================================
# RUN GRID SEARCH
# =============================================================================

print("\n" + "="*80)
print("GRID SEARCH OVER NEIGHBOR FEATURE PARAMETERS")
print("="*80)

# Use a sample for grid search
SAMPLE_SIZE = 2000
np.random.seed(42)
sample_idx = np.random.choice(train.index, size=min(SAMPLE_SIZE, len(train)), replace=False)
train_sample = train.loc[sample_idx].reset_index(drop=True)

results = []

config_id = 0
total_configs = len(CATEGORICAL_COMBOS) * len(NUMERICAL_COMBOS) * len(WEIGHT_COMBOS)
print(f"Total configurations to test: {total_configs}")

for cat_cols in CATEGORICAL_COMBOS:
    for num_cols in NUMERICAL_COMBOS:
        for weights in WEIGHT_COMBOS:
            config_id += 1
            config_name = f"cat={len(cat_cols)}, num={len(num_cols)}, w={weights['geo']:.1f}/{weights['cat']:.1f}/{weights['num']:.1f}"

            print(f"\n[{config_id}/{total_configs}] Testing: {config_name}")

            # Build feature matrices
            coords, num_features, cat_features, encoders = build_feature_matrix(
                train_sample, cat_cols, num_cols
            )

            # Compute neighbor features
            feat_df = compute_neighbor_features_fast(
                train_sample, coords, num_features, cat_features, cat_cols,
                k_values=[5], max_dist_km=[200],
                weight_geo=weights['geo'],
                weight_cat=weights['cat'],
                weight_num=weights['num']
            )

            # Evaluate correlation
            target = train_sample['underperforming'].values
            correlations = []
            for col in feat_df.columns:
                if feat_df[col].notna().sum() > 10:
                    corr = abs(feat_df[col].corr(pd.Series(target)))
                    if pd.notna(corr):
                        correlations.append(corr)

            best_corr = max(correlations) if correlations else 0

            results.append({
                'config_id': config_id,
                'cat_cols': ', '.join(cat_cols) if cat_cols else 'none',
                'num_cat': len(cat_cols),
                'num_cols': ', '.join(num_cols) if num_cols else 'none',
                'num_num': len(num_cols),
                'weight_geo': weights['geo'],
                'weight_cat': weights['cat'],
                'weight_num': weights['num'],
                'best_corr': best_corr,
                'config_name': config_name
            })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('best_corr', ascending=False)

print("\n" + "="*80)
print("TOP 10 CONFIGURATIONS BY CORRELATION")
print("="*80)
print(results_df.head(10).to_string(index=False))

# =============================================================================
# VISUALIZE CONFIGURATION RESULTS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
for n_cat in sorted(results_df['num_cat'].unique()):
    subset = results_df[results_df['num_cat'] == n_cat]
    ax1.scatter([n_cat] * len(subset), subset['best_corr'], alpha=0.5, s=50, label=f'{n_cat} cat')
ax1.set_xlabel('Number of Categorical Features')
ax1.set_ylabel('Best Correlation with Target')
ax1.set_title('Correlation by # Categorical Features', fontweight='bold')
ax1.legend(title='cat cols')

ax2 = axes[0, 1]
weight_groups = results_df.groupby(['weight_geo', 'weight_cat'])['best_corr'].max().reset_index()
scatter = ax2.scatter(weight_groups['weight_geo'], weight_groups['weight_cat'],
                       c=weight_groups['best_corr'], cmap='RdYlGn', s=200, edgecolor='black')
ax2.set_xlabel('Geo Weight')
ax2.set_ylabel('Cat Weight')
ax2.set_title('Best Correlation by Weight', fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Best Correlation')

ax3 = axes[1, 0]
top10 = results_df.head(10)
colors = [COLOR_1 if c > 0.08 else COLOR_0 for c in top10['best_corr']]
ax3.barh(range(len(top10)), top10['best_corr'], color=colors, edgecolor='black')
ax3.set_yticks(range(len(top10)))
ax3.set_yticklabels(top10['config_name'], fontsize=7)
ax3.set_xlabel('Best Correlation')
ax3.set_title('Top 10 Configurations', fontweight='bold')
ax3.invert_yaxis()

ax4 = axes[1, 1]
best_row = results_df.iloc[0]
ax4.text(0.5, 0.85, "BEST CONFIGURATION", fontsize=14, fontweight='bold',
         ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.70, f"Categorical: {best_row['cat_cols']}", fontsize=10,
         ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.60, f"Numerical: {best_row['num_cols']}", fontsize=10,
         ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.50, f"Weights: geo={best_row['weight_geo']}, cat={best_row['weight_cat']}, num={best_row['weight_num']}",
         fontsize=10, ha='center', transform=ax4.transAxes)
ax4.text(0.5, 0.35, f"Best correlation: {best_row['best_corr']:.4f}", fontsize=12,
         ha='center', transform=ax4.transAxes, color=COLOR_1, fontweight='bold')
ax4.axis('off')
ax4.set_title('Best Configuration', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "neighbor_01_config_grid_search.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: neighbor_01_config_grid_search.png")

# =============================================================================
# DEEP DIVE: K AND MAX_DIST OPTIMIZATION
# =============================================================================
print("\n" + "="*80)
print("OPTIMIZING k AND max_dist")
print("="*80)

best_cat_cols = best_row['cat_cols'].split(', ') if best_row['cat_cols'] != 'none' else []
best_num_cols = best_row['num_cols'].split(', ') if best_row['num_cols'] != 'none' else []
best_weights = {'geo': best_row['weight_geo'], 'cat': best_row['weight_cat'], 'num': best_row['weight_num']}

print(f"Best config: cat={best_cat_cols}, num={best_num_cols}, weights={best_weights}")

# Build matrices
coords, num_features, cat_features, _ = build_feature_matrix(train_sample, best_cat_cols, best_num_cols)

# Compute all features
feat_df = compute_neighbor_features_fast(
    train_sample, coords, num_features, cat_features, best_cat_cols,
    k_values=K_VALUES, max_dist_km=MAX_DIST_VALUES,
    weight_geo=best_weights['geo'], weight_cat=best_weights['cat'], weight_num=best_weights['num']
)

target = train_sample['underperforming'].values

# Evaluate each feature
k_dist_results = []
for col in feat_df.columns:
    if 'target_mean' in col and feat_df[col].notna().sum() > 10:
        # Parse k and max_dist from column name
        parts = col.split('_')
        k = int(parts[0][2:])
        max_dist = int(parts[1][3:])

        corr = abs(feat_df[col].corr(pd.Series(target)))
        coverage = feat_df[col].notna().mean()

        k_dist_results.append({
            'k': k, 'max_dist': max_dist, 'feature': col,
            'correlation': corr, 'coverage': coverage
        })

k_dist_df = pd.DataFrame(k_dist_results)
k_dist_df = k_dist_df.sort_values('correlation', ascending=False)

print("\nTop k/max_dist combinations:")
print(k_dist_df.head(10).to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax1 = axes[0]
pivot = k_dist_df.pivot_table(index='k', columns='max_dist', values='correlation', aggfunc='max')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax1)
ax1.set_title('Correlation: k vs max_dist', fontweight='bold')
ax1.set_xlabel('Max Distance (km)')
ax1.set_ylabel('k (neighbors)')

ax2 = axes[1]
pivot_cov = k_dist_df.pivot_table(index='k', columns='max_dist', values='coverage', aggfunc='max')
sns.heatmap(pivot_cov, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
ax2.set_title('Coverage', fontweight='bold')
ax2.set_xlabel('Max Distance (km)')
ax2.set_ylabel('k (neighbors)')

ax3 = axes[2]
scatter = ax3.scatter(k_dist_df['coverage'], k_dist_df['correlation'],
                       c=k_dist_df['k'], cmap='viridis', s=50, alpha=0.7)
ax3.set_xlabel('Coverage')
ax3.set_ylabel('Correlation')
ax3.set_title('Correlation vs Coverage', fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='k')

plt.tight_layout()
plt.savefig(output_dir / "neighbor_02_k_maxdist_optimization.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: neighbor_02_k_maxdist_optimization.png")

# =============================================================================
# COMPUTE FINAL FEATURES FOR FULL TRAINING SET
# =============================================================================
print("\n" + "="*80)
print("COMPUTING FINAL FEATURES FOR FULL TRAINING SET")
print("="*80)

# Best k and max_dist
BEST_K = int(k_dist_df.iloc[0]['k'])
BEST_MAX_DIST = int(k_dist_df.iloc[0]['max_dist'])

print(f"Best k: {BEST_K}, Best max_dist: {BEST_MAX_DIST} km")

# Build matrices for full training set
coords_full, num_features_full, cat_features_full, _ = build_feature_matrix(
    train, best_cat_cols, best_num_cols
)

# Compute features
final_feat_df = compute_neighbor_features_fast(
    train, coords_full, num_features_full, cat_features_full, best_cat_cols,
    k_values=K_VALUES, max_dist_km=MAX_DIST_VALUES,
    weight_geo=best_weights['geo'], weight_cat=best_weights['cat'], weight_num=best_weights['num']
)

final_feat_df['underperforming'] = train['underperforming'].values

# =============================================================================
# ANALYZE FINAL FEATURES
# =============================================================================
print("\n" + "="*80)
print("FINAL NEIGHBOR FEATURE ANALYSIS")
print("="*80)

correlations = {}
for col in final_feat_df.columns:
    if col != 'underperforming':
        corr = final_feat_df[col].corr(final_feat_df['underperforming'])
        if pd.notna(corr):
            correlations[col] = corr

corr_df = pd.DataFrame({'feature': list(correlations.keys()), 'correlation': list(correlations.values())})
corr_df = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).index)

print("\nTop 15 features by correlation:")
print(corr_df.head(15).to_string(index=False))

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
top_corr = corr_df.head(15)
colors = [COLOR_1 if c > 0 else COLOR_0 for c in top_corr['correlation']]
ax1.barh(range(len(top_corr)), top_corr['correlation'], color=colors, edgecolor='black')
ax1.set_yticks(range(len(top_corr)))
ax1.set_yticklabels(top_corr['feature'], fontsize=7)
ax1.set_xlabel('Correlation with Target')
ax1.set_title('Top 15 Neighbor Features', fontweight='bold')
ax1.invert_yaxis()

ax2 = axes[0, 1]
best_feat = corr_df.iloc[0]['feature']
for target in [0, 1]:
    data = final_feat_df[final_feat_df['underperforming'] == target][best_feat].dropna()
    label = 'Normal' if target == 0 else 'Underperforming'
    ax2.hist(data, bins=30, alpha=0.6, color=TARGET_COLORS[target], label=label, density=True)
ax2.set_xlabel(best_feat)
ax2.set_ylabel('Density')
ax2.set_title(f'Best Feature Distribution', fontweight='bold')
ax2.legend()

ax3 = axes[1, 0]
if 'dist_to_nearest_underperf' in final_feat_df.columns:
    for target in [0, 1]:
        data = final_feat_df[final_feat_df['underperforming'] == target]
        label = 'Normal' if target == 0 else 'Underperforming'
        ax3.hist(data['dist_to_nearest_underperf'].dropna(), bins=30, alpha=0.6,
                 color=TARGET_COLORS[target], label=label, density=True)
    ax3.set_xlabel('Distance to Nearest Underperforming Plant (km)')
    ax3.set_ylabel('Density')
    ax3.set_title('Distance to Nearest Underperforming', fontweight='bold')
    ax3.legend()

ax4 = axes[1, 1]
# Coverage analysis
coverage_data = []
for col in final_feat_df.columns:
    if 'count' in col:
        parts = col.split('_')
        k = parts[0][2:]
        max_d = parts[1][3:]
        coverage_data.append({
            'k': k, 'max_dist': max_d,
            'coverage': (final_feat_df[col] > 0).mean()
        })
coverage_df = pd.DataFrame(coverage_data)
pivot_cov = coverage_df.pivot_table(index='k', columns='max_dist', values='coverage', aggfunc='first')
sns.heatmap(pivot_cov.astype(float), annot=True, fmt='.2f', cmap='Blues', ax=ax4)
ax4.set_title('Coverage by k and max_dist', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "neighbor_03_final_features.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: neighbor_03_final_features.png")

# =============================================================================
# SAVE FEATURES
# =============================================================================
final_feat_df.to_csv("neighbor_features.csv", index=False)
print(f"\nSaved to: neighbor_features.csv")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nBest configuration:")
print(f"  Categorical: {best_cat_cols}")
print(f"  Numerical: {best_num_cols}")
print(f"  Weights: {best_weights}")
print(f"  Best k: {BEST_K}, max_dist: {BEST_MAX_DIST} km")

print(f"\nBest correlation achieved: {corr_df.iloc[0]['correlation']:.4f}")
print(f"Best feature: {corr_df.iloc[0]['feature']}")

print(f"\nAll plots saved to: {output_dir.absolute()}")
