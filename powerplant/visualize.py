"""
Power Plant Underperformance Dataset Visualization
Run with: python visualize.py

All plots colored by target (underperforming)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Define consistent colors for target
COLOR_0 = '#2ecc71'  # Green for normal (not underperforming)
COLOR_1 = '#e74c3c'  # Red for underperforming
TARGET_COLORS = [COLOR_0, COLOR_1]
TARGET_PALETTE = {0: COLOR_0, 1: COLOR_1}

# Create output directory for plots
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv("public/train.csv")
test = pd.read_csv("public/test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTarget distribution:\n{train['underperforming'].value_counts(normalize=True)}")

# Define feature groups
numerical_features = ['capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
                       'latitude', 'longitude', 'age_x_capacity']
categorical_features = ['fuel_group', 'primary_fuel', 'other_fuel1',
                        'owner_bucket', 'capacity_band', 'lat_band', 'lon_band']

# =============================================================================
# 1. TARGET DISTRIBUTION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
ax1 = axes[0]
train['underperforming'].value_counts().sort_index().plot(kind='bar', ax=ax1, color=TARGET_COLORS)
ax1.set_title('Target Distribution (Count)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Underperforming')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['0 (Normal)', '1 (Underperforming)'], rotation=0)
for i, v in enumerate(train['underperforming'].value_counts().sort_index()):
    ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')

# Pie chart
ax2 = axes[1]
counts = train['underperforming'].value_counts().sort_index()
ax2.pie(counts, labels=['Normal', 'Underperforming'], colors=TARGET_COLORS,
        autopct='%1.1f%%', explode=[0, 0.05])
ax2.set_title('Target Distribution (%)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "01_target_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_target_distribution.png")

# =============================================================================
# 2. NUMERICAL FEATURES DISTRIBUTION (by target)
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    ax = axes[i]
    for target_val in [0, 1]:
        subset = train[train['underperforming'] == target_val]
        label = 'Normal' if target_val == 0 else 'Underperforming'
        subset[col].hist(bins=50, ax=ax, alpha=0.6, color=TARGET_COLORS[target_val],
                         label=label, density=True)
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend()

    # Add statistics
    stats_text = f"Mean: {train[col].mean():.2f}\nStd: {train[col].std():.2f}\nSkew: {train[col].skew():.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

# Hide unused subplot
axes[-1].axis('off')
axes[-2].axis('off')

plt.suptitle('Numerical Features Distribution (by Target)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "02_numerical_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_numerical_distributions.png")

# =============================================================================
# 3. NUMERICAL FEATURES BY TARGET (Boxplots with target colors)
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    ax = axes[i]
    train.boxplot(column=col, by='underperforming', ax=ax, patch_artist=True)
    # Color the boxes
    boxes = [child for child in ax.get_children() if hasattr(child, 'set_facecolor')]
    for j, box in enumerate(boxes[:2]):
        box.set_facecolor(TARGET_COLORS[j])
        box.set_alpha(0.7)
    ax.set_title(f'{col} by Target', fontsize=11, fontweight='bold')
    ax.set_xlabel('Underperforming')

# Hide unused subplots
for j in range(len(numerical_features), len(axes)):
    axes[j].axis('off')

plt.suptitle('Numerical Features by Target Class', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "03_numerical_by_target.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_numerical_by_target.png")

# =============================================================================
# 4. CATEGORICAL FEATURES DISTRIBUTION (stacked by target)
# =============================================================================
# Primary fuel - stacked by target
fig, ax = plt.subplots(figsize=(14, 6))
fuel_order = train['primary_fuel'].value_counts().index
fuel_target_counts = train.groupby(['primary_fuel', 'underperforming']).size().unstack(fill_value=0)
fuel_target_counts = fuel_target_counts.reindex(fuel_order)
fuel_target_counts.plot(kind='bar', stacked=True, ax=ax, color=TARGET_COLORS, edgecolor='black')
ax.set_title('Primary Fuel Distribution (stacked by Target)', fontsize=12, fontweight='bold')
ax.set_xlabel('Primary Fuel')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=45)
ax.legend(['Normal', 'Underperforming'], title='Target')
plt.tight_layout()
plt.savefig(output_dir / "04a_primary_fuel_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04a_primary_fuel_dist.png")

# Fuel group - stacked by target
fig, ax = plt.subplots(figsize=(8, 5))
fuel_group_target = train.groupby(['fuel_group', 'underperforming']).size().unstack(fill_value=0)
fuel_group_target.plot(kind='bar', stacked=True, ax=ax, color=TARGET_COLORS, edgecolor='black')
ax.set_title('Fuel Group Distribution (stacked by Target)', fontsize=12, fontweight='bold')
ax.set_xlabel('Fuel Group')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
ax.legend(['Normal', 'Underperforming'], title='Target')
plt.tight_layout()
plt.savefig(output_dir / "04b_fuel_group_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04b_fuel_group_dist.png")

# Capacity band - stacked by target
fig, ax = plt.subplots(figsize=(10, 5))
capacity_order = ['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']
cap_target = train.groupby(['capacity_band', 'underperforming']).size().unstack(fill_value=0)
cap_target = cap_target.reindex(capacity_order)
cap_target.plot(kind='bar', stacked=True, ax=ax, color=TARGET_COLORS, edgecolor='black')
ax.set_title('Capacity Band Distribution (stacked by Target)', fontsize=12, fontweight='bold')
ax.set_xlabel('Capacity Band')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=0)
ax.legend(['Normal', 'Underperforming'], title='Target')
plt.tight_layout()
plt.savefig(output_dir / "04c_capacity_band_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04c_capacity_band_dist.png")

# =============================================================================
# 5. UNDERPERFORMING RATE BY CATEGORIES
# =============================================================================
# By primary fuel
fig, ax = plt.subplots(figsize=(12, 5))
fuel_underperform = train.groupby('primary_fuel')['underperforming'].agg(['mean', 'count'])
fuel_underperform = fuel_underperform.sort_values('mean', ascending=False)
colors = [COLOR_1 if rate > train['underperforming'].mean() else COLOR_0
          for rate in fuel_underperform['mean']]
bars = ax.bar(fuel_underperform.index, fuel_underperform['mean'] * 100, color=colors, edgecolor='black')
ax.set_title('Underperforming Rate by Primary Fuel (%) - Red = Above Average', fontsize=12, fontweight='bold')
ax.set_xlabel('Primary Fuel')
ax.set_ylabel('Underperforming Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--',
           label=f'Overall Rate ({train["underperforming"].mean()*100:.1f}%)')
ax.legend()
for bar, (idx, row) in zip(bars, fuel_underperform.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{row["mean"]*100:.1f}%\n(n={int(row["count"])})', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig(output_dir / "05a_underperform_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05a_underperform_by_fuel.png")

# By fuel group
fig, ax = plt.subplots(figsize=(8, 5))
fuel_group_underperform = train.groupby('fuel_group')['underperforming'].mean().sort_values(ascending=False)
colors = [COLOR_1 if rate > train['underperforming'].mean() else COLOR_0
          for rate in fuel_group_underperform.values]
bars = ax.bar(fuel_group_underperform.index, fuel_group_underperform.values * 100,
              color=colors, edgecolor='black')
ax.set_title('Underperforming Rate by Fuel Group (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Fuel Group')
ax.set_ylabel('Underperforming Rate (%)')
ax.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--',
           label=f'Overall Rate ({train["underperforming"].mean()*100:.1f}%)')
ax.legend()
for bar, val in zip(bars, fuel_group_underperform.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val*100:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(output_dir / "05b_underperform_by_fuel_group.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05b_underperform_by_fuel_group.png")

# By capacity band
fig, ax = plt.subplots(figsize=(10, 5))
capacity_order = ['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']
cap_underperform = train.groupby('capacity_band')['underperforming'].mean().reindex(capacity_order)
colors = [COLOR_1 if rate > train['underperforming'].mean() else COLOR_0
          for rate in cap_underperform.values]
bars = ax.bar(cap_underperform.index, cap_underperform.values * 100, color=colors, edgecolor='black')
ax.set_title('Underperforming Rate by Capacity Band (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Capacity Band')
ax.set_ylabel('Underperforming Rate (%)')
ax.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--',
           label=f'Overall Rate ({train["underperforming"].mean()*100:.1f}%)')
ax.legend()
for bar, val in zip(bars, cap_underperform.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val*100:.1f}%', ha='center')
plt.tight_layout()
plt.savefig(output_dir / "05c_underperform_by_capacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05c_underperform_by_capacity.png")

# =============================================================================
# 6. CORRELATION MATRIX (with target)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = train[numerical_features + ['underperforming']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            vmin=-1, vmax=1)
ax.set_title('Feature Correlation Matrix (Red = Positive, Green = Negative)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "06_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 06_correlation_matrix.png")

# =============================================================================
# 7. GEOGRAPHIC DISTRIBUTION (colored by target)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By target
ax1 = axes[0]
for target_val in [0, 1]:
    subset = train[train['underperforming'] == target_val]
    label = 'Normal' if target_val == 0 else 'Underperforming'
    ax1.scatter(subset['longitude'], subset['latitude'],
                c=TARGET_COLORS[target_val], alpha=0.5, s=20, label=label)
ax1.set_title('Geographic Distribution by Target\n(Green=Normal, Red=Underperforming)',
              fontsize=11, fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.legend()

# By capacity (colored by target with size = capacity)
ax2 = axes[1]
scatter = ax2.scatter(train['longitude'], train['latitude'],
                       c=train['underperforming'].map(TARGET_PALETTE),
                       s=train['capacity_log_mw'] * 3, alpha=0.5)
ax2.set_title('Geographic Distribution (size = log capacity)\n(Green=Normal, Red=Underperforming)',
              fontsize=11, fontweight='bold')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')

# Add legend for target
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLOR_0, label='Normal'),
                   Patch(facecolor=COLOR_1, label='Underperforming')]
ax2.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(output_dir / "07_geographic_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_geographic_distribution.png")

# =============================================================================
# 8. AGE vs CAPACITY INTERACTION (colored by target)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
for target_val in [0, 1]:
    subset = train[train['underperforming'] == target_val]
    label = 'Normal' if target_val == 0 else 'Underperforming'
    ax.scatter(subset['plant_age'], subset['capacity_log_mw'],
               c=TARGET_COLORS[target_val], alpha=0.4, s=30, label=label)
ax.set_title('Plant Age vs Capacity by Target', fontsize=12, fontweight='bold')
ax.set_xlabel('Plant Age (years)')
ax.set_ylabel('Log Capacity')
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / "08_age_vs_capacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_age_vs_capacity.png")

# =============================================================================
# 9. TOP OWNERS ANALYSIS (colored by target rate)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Top 15 owners - stacked by target
top_owners_list = train['owner_bucket'].value_counts().head(15).index
owner_target = train[train['owner_bucket'].isin(top_owners_list)].groupby(
    ['owner_bucket', 'underperforming']).size().unstack(fill_value=0)
owner_target = owner_target.reindex(top_owners_list)

ax1 = axes[0]
owner_target.plot(kind='barh', stacked=True, ax=ax1, color=TARGET_COLORS, edgecolor='black')
ax1.set_title('Top 15 Owners (stacked by Target)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Count')
ax1.legend(['Normal', 'Underperforming'], title='Target')
ax1.invert_yaxis()

# Top 15 owners by underperforming rate (min 20 plants)
owner_stats = train.groupby('owner_bucket').agg({
    'underperforming': ['mean', 'count']
}).reset_index()
owner_stats.columns = ['owner_bucket', 'underperforming_rate', 'count']
owner_stats_filtered = owner_stats[owner_stats['count'] >= 20].sort_values('underperforming_rate', ascending=False).head(15)

ax2 = axes[1]
colors = [COLOR_1 if rate > train['underperforming'].mean() else COLOR_0
          for rate in owner_stats_filtered['underperforming_rate']]
bars = ax2.barh(owner_stats_filtered['owner_bucket'], owner_stats_filtered['underperforming_rate'] * 100,
                color=colors, edgecolor='black')
ax2.set_title('Top 15 Owners by Underperforming Rate\n(min 20 plants, Red = Above Average)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Underperforming Rate (%)')
ax2.axvline(x=train['underperforming'].mean() * 100, color='blue', linestyle='--', label='Overall Rate')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / "09_owner_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 09_owner_analysis.png")

# =============================================================================
# 10. FEATURE IMPORTANCE PREVIEW (Simple Logistic Regression)
# =============================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Prepare data
X = train[numerical_features].copy()
y = train['underperforming']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit simple model
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

# Plot coefficients
fig, ax = plt.subplots(figsize=(10, 6))
importance = pd.DataFrame({'feature': numerical_features, 'coef': lr.coef_[0]})
importance = importance.sort_values('coef', ascending=True)
colors = [COLOR_1 if c > 0 else COLOR_0 for c in importance['coef']]
bars = ax.barh(importance['feature'], importance['coef'], color=colors, edgecolor='black')
ax.set_title('Feature Importance (Logistic Regression Coefficients)\nRed = Increases Underperformance Risk, Green = Decreases',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Coefficient')
ax.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(output_dir / "10_feature_importance_preview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10_feature_importance_preview.png")

# =============================================================================
# 11. TRAIN vs TEST DISTRIBUTION COMPARISON (with target overlay)
# =============================================================================
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    ax = axes[i]
    # Train by target
    for target_val in [0, 1]:
        subset = train[train['underperforming'] == target_val]
        label = f'Train ({"Normal" if target_val == 0 else "Underperform"})'
        ax.hist(subset[col], bins=50, alpha=0.4, color=TARGET_COLORS[target_val],
                label=label, density=True)
    # Test
    ax.hist(test[col], bins=50, alpha=0.3, color='gray', label='Test', density=True)
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_xlabel(col)
    ax.set_ylabel('Density')

# Hide unused subplots
for j in range(len(numerical_features), len(axes)):
    axes[j].axis('off')

plt.suptitle('Train vs Test Distribution (with Target overlay)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "11_train_test_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 11_train_test_comparison.png")

# =============================================================================
# 12. PAIRPLOT OF KEY FEATURES (sampled)
# =============================================================================
print("Creating pairplot (this may take a moment)...")
sample_idx = train.sample(n=min(1000, len(train)), random_state=42).index
pairplot_features = ['capacity_log_mw', 'plant_age', 'abs_latitude', 'age_x_capacity', 'underperforming']
g = sns.pairplot(train.loc[sample_idx, pairplot_features], hue='underperforming',
                  palette=TARGET_PALETTE, diag_kind='kde', corner=True,
                  plot_kws={'alpha': 0.5, 's': 20})
g.fig.suptitle('Pairplot of Key Features (sampled)', fontsize=14, fontweight='bold', y=1.02)
plt.savefig(output_dir / "12_pairplot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 12_pairplot.png")

# =============================================================================
# 13. CATEGORICAL INTERACTION: FUEL x CAPACITY BAND
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
fuel_cap_target = train.groupby(['fuel_group', 'capacity_band'])['underperforming'].mean().unstack()
fuel_cap_target = fuel_cap_target[['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']]
sns.heatmap(fuel_cap_target, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.6, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: Fuel Group x Capacity Band\n(Red = High Rate, Green = Low Rate)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "13_fuel_capacity_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 13_fuel_capacity_heatmap.png")

# =============================================================================
# 14. VIOLIN PLOTS FOR KEY NUMERICAL FEATURES
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
key_features = ['capacity_log_mw', 'plant_age', 'abs_latitude', 'age_x_capacity']

for i, col in enumerate(key_features):
    ax = axes[i // 2, i % 2]
    parts = ax.violinplot([train[train['underperforming']==0][col],
                           train[train['underperforming']==1][col]],
                          positions=[0, 1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor(COLOR_0)
    parts['bodies'][1].set_facecolor(COLOR_1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Normal', 'Underperforming'])
    ax.set_title(f'{col} Distribution by Target', fontsize=11, fontweight='bold')
    ax.set_ylabel(col)

plt.suptitle('Violin Plots: Key Features by Target', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "14_violin_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 14_violin_plots.png")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\n--- Numerical Features ---")
print(train[numerical_features].describe().round(2))

print("\n--- Target Statistics ---")
print(f"Class 0 (Normal): {(train['underperforming']==0).sum()} ({(train['underperforming']==0).mean()*100:.1f}%)")
print(f"Class 1 (Underperforming): {(train['underperforming']==1).sum()} ({(train['underperforming']==1).mean()*100:.1f}%)")

print("\n--- Categorical Cardinality ---")
for col in categorical_features:
    print(f"{col}: {train[col].nunique()} unique values")

print("\n--- Key Observations ---")
print(f"1. Capacity range: {train['capacity_mw'].min():.1f} - {train['capacity_mw'].max():.1f} MW")
print(f"2. Plant age range: {train['plant_age'].min():.1f} - {train['plant_age'].max():.1f} years")
print(f"3. Most common primary fuel: {train['primary_fuel'].mode()[0]}")
print(f"4. Highest underperforming fuel type: {train.groupby('primary_fuel')['underperforming'].mean().idxmax()}")

# Feature correlation with target
print("\n--- Feature Correlation with Target ---")
for col in numerical_features:
    corr = train[col].corr(train['underperforming'])
    print(f"{col}: {corr:.3f}")

print("\n" + "="*60)
print(f"All plots saved to: {output_dir.absolute()}")
print("="*60)
