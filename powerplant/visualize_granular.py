"""
Power Plant Underperformance - Granular Grouping Analysis
Analyzes target correlation at the lowest possible granularity by combining:
- primary_fuel
- capacity_band
- lat_band / lon_band
- other_fuel1

Run with: python visualize_granular.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Colors
COLOR_0 = '#2ecc71'  # Green
COLOR_1 = '#e74c3c'  # Red
TARGET_COLORS = [COLOR_0, COLOR_1]

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv("public/train.csv")

print(f"Train shape: {train.shape}")
print(f"\nUnique values:")
print(f"  primary_fuel: {train['primary_fuel'].nunique()}")
print(f"  capacity_band: {train['capacity_band'].nunique()}")
print(f"  lat_band: {train['lat_band'].nunique()}")
print(f"  lon_band: {train['lon_band'].nunique()}")
print(f"  other_fuel1: {train['other_fuel1'].nunique()}")
print(f"  owner_bucket: {train['owner_bucket'].nunique()}")

# Calculate total possible combinations
total_combos = (train['primary_fuel'].nunique() *
                train['capacity_band'].nunique() *
                train['lat_band'].nunique() *
                train['lon_band'].nunique() *
                train['other_fuel1'].nunique())
print(f"\nTotal possible combinations: {total_combos:,}")

# =============================================================================
# 1. OTHER_FUEL1 ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("OTHER_FUEL1 ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
ax1 = axes[0]
fuel_target = train.groupby(['other_fuel1', 'underperforming']).size().unstack(fill_value=0)
fuel_target = fuel_target.sort_values(fuel_target.columns.tolist(), ascending=False)
fuel_target.plot(kind='bar', stacked=True, ax=ax1, color=TARGET_COLORS, edgecolor='black')
ax1.set_title('Distribution by other_fuel1 (stacked by target)', fontweight='bold')
ax1.set_xlabel('other_fuel1')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(['Normal', 'Underperforming'], title='Target')

# Rate
ax2 = axes[1]
other_rates = train.groupby('other_fuel1')['underperforming'].agg(['mean', 'count'])
other_rates = other_rates.sort_values('mean', ascending=False)
colors = [COLOR_1 if r > train['underperforming'].mean() else COLOR_0 for r in other_rates['mean']]
bars = ax2.bar(other_rates.index, other_rates['mean'] * 100, color=colors, edgecolor='black')
ax2.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2)
ax2.set_title('Underperforming Rate by other_fuel1', fontweight='bold')
ax2.set_xlabel('other_fuel1')
ax2.set_ylabel('Underperforming Rate (%)')
ax2.tick_params(axis='x', rotation=45)

for bar, (idx, row) in zip(bars, other_rates.iterrows()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{row["mean"]*100:.0f}%\nn={int(row["count"])}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / "granular_01_other_fuel1.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_01_other_fuel1.png")

# =============================================================================
# 2. LAT_BAND ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LAT_BAND ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
ax1 = axes[0]
lat_order = sorted(train['lat_band'].unique())
lat_target = train.groupby(['lat_band', 'underperforming']).size().unstack(fill_value=0)
lat_target = lat_target.reindex(lat_order)
lat_target.plot(kind='bar', stacked=True, ax=ax1, color=TARGET_COLORS, edgecolor='black')
ax1.set_title('Distribution by lat_band (stacked by target)', fontweight='bold')
ax1.set_xlabel('lat_band')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(['Normal', 'Underperforming'], title='Target')

# Rate
ax2 = axes[1]
lat_rates = train.groupby('lat_band')['underperforming'].agg(['mean', 'count'])
lat_rates = lat_rates.reindex(lat_order)
colors = [COLOR_1 if r > train['underperforming'].mean() else COLOR_0 for r in lat_rates['mean']]
bars = ax2.bar(lat_rates.index, lat_rates['mean'] * 100, color=colors, edgecolor='black')
ax2.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2)
ax2.set_title('Underperforming Rate by lat_band', fontweight='bold')
ax2.set_xlabel('lat_band')
ax2.set_ylabel('Underperforming Rate (%)')
ax2.tick_params(axis='x', rotation=45)

for bar, (idx, row) in zip(bars, lat_rates.iterrows()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{row["mean"]*100:.0f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "granular_02_lat_band.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_02_lat_band.png")

# =============================================================================
# 3. LON_BAND ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("LON_BAND ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
ax1 = axes[0]
lon_order = sorted(train['lon_band'].unique())
lon_target = train.groupby(['lon_band', 'underperforming']).size().unstack(fill_value=0)
lon_target = lon_target.reindex(lon_order)
lon_target.plot(kind='bar', stacked=True, ax=ax1, color=TARGET_COLORS, edgecolor='black')
ax1.set_title('Distribution by lon_band (stacked by target)', fontweight='bold')
ax1.set_xlabel('lon_band')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(['Normal', 'Underperforming'], title='Target')

# Rate
ax2 = axes[1]
lon_rates = train.groupby('lon_band')['underperforming'].agg(['mean', 'count'])
lon_rates = lon_rates.reindex(lon_order)
colors = [COLOR_1 if r > train['underperforming'].mean() else COLOR_0 for r in lon_rates['mean']]
bars = ax2.bar(lon_rates.index, lon_rates['mean'] * 100, color=colors, edgecolor='black')
ax2.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2)
ax2.set_title('Underperforming Rate by lon_band', fontweight='bold')
ax2.set_xlabel('lon_band')
ax2.set_ylabel('Underperforming Rate (%)')
ax2.tick_params(axis='x', rotation=45)

for bar, (idx, row) in zip(bars, lon_rates.iterrows()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{row["mean"]*100:.0f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(output_dir / "granular_03_lon_band.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_03_lon_band.png")

# =============================================================================
# 4. HEATMAP: LAT_BAND x LON_BAND
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

lat_lon_rates = train.groupby(['lat_band', 'lon_band'])['underperforming'].agg(['mean', 'count'])
lat_lon_pivot = lat_lon_rates['mean'].unstack()
lat_lon_pivot = lat_lon_pivot.reindex(index=sorted(lat_lon_pivot.index),
                                       columns=sorted(lat_lon_pivot.columns))

sns.heatmap(lat_lon_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.7, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: lat_band × lon_band\n(min 5 samples per cell, Red=High, Green=Low)',
             fontweight='bold')

# Add count annotations
lat_lon_counts = lat_lon_rates['count'].unstack()
lat_lon_counts = lat_lon_counts.reindex(index=sorted(lat_lon_counts.index),
                                         columns=sorted(lat_lon_counts.columns))
for i, lat in enumerate(lat_lon_pivot.index):
    for j, lon in enumerate(lat_lon_pivot.columns):
        count = lat_lon_counts.loc[lat, lon] if pd.notna(lat_lon_counts.loc[lat, lon]) else 0
        ax.text(j + 0.5, i + 0.7, f'n={int(count)}', ha='center', va='center', fontsize=6, color='black')

plt.tight_layout()
plt.savefig(output_dir / "granular_04_lat_lon_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_04_lat_lon_heatmap.png")

# =============================================================================
# 5. PRIMARY_FUEL x OTHER_FUEL1 HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 8))

fuel_other_rates = train.groupby(['primary_fuel', 'other_fuel1'])['underperforming'].agg(['mean', 'count'])
fuel_other_pivot = fuel_other_rates['mean'].unstack()

sns.heatmap(fuel_other_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.8, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: primary_fuel × other_fuel1\n(Red=High, Green=Low)',
             fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "granular_05_fuel_other_fuel_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_05_fuel_other_fuel_heatmap.png")

# =============================================================================
# 6. PRIMARY_FUEL x CAPACITY_BAND HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

capacity_order = ['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']
fuel_cap_rates = train.groupby(['primary_fuel', 'capacity_band'])['underperforming'].agg(['mean', 'count'])
fuel_cap_pivot = fuel_cap_rates['mean'].unstack()
fuel_cap_pivot = fuel_cap_pivot.reindex(columns=[c for c in capacity_order if c in fuel_cap_pivot.columns])

sns.heatmap(fuel_cap_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.7, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: primary_fuel × capacity_band\n(Red=High, Green=Low)',
             fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "granular_06_fuel_capacity_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_06_fuel_capacity_heatmap.png")

# =============================================================================
# 7. PRIMARY_FUEL x LAT_BAND HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

fuel_lat_rates = train.groupby(['primary_fuel', 'lat_band'])['underperforming'].agg(['mean', 'count'])
fuel_lat_pivot = fuel_lat_rates['mean'].unstack()
fuel_lat_pivot = fuel_lat_pivot.reindex(columns=sorted(fuel_lat_pivot.columns))

sns.heatmap(fuel_lat_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.8, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: primary_fuel × lat_band\n(Red=High, Green=Low)',
             fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "granular_07_fuel_lat_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_07_fuel_lat_heatmap.png")

# =============================================================================
# 8. PRIMARY_FUEL x LON_BAND HEATMAP
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 10))

fuel_lon_rates = train.groupby(['primary_fuel', 'lon_band'])['underperforming'].agg(['mean', 'count'])
fuel_lon_pivot = fuel_lon_rates['mean'].unstack()
fuel_lon_pivot = fuel_lon_pivot.reindex(columns=sorted(fuel_lon_pivot.columns))

sns.heatmap(fuel_lon_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
            vmin=0, vmax=0.8, cbar_kws={'label': 'Underperforming Rate'})
ax.set_title('Underperforming Rate: primary_fuel × lon_band\n(Red=High, Green=Low)',
             fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "granular_08_fuel_lon_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_08_fuel_lon_heatmap.png")

# =============================================================================
# 9. CAPACITY_BAND x LAT_BAND x LON_BAND (FACETED)
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
capacity_order = ['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']

for i, cap_band in enumerate(capacity_order):
    ax = axes[i // 3, i % 3]
    subset = train[train['capacity_band'] == cap_band]

    if len(subset) < 10:
        ax.axis('off')
        continue

    lat_lon_rates = subset.groupby(['lat_band', 'lon_band'])['underperforming'].mean().unstack()
    lat_lon_rates = lat_lon_rates.reindex(index=sorted(lat_lon_rates.index),
                                           columns=sorted(lat_lon_rates.columns))

    sns.heatmap(lat_lon_rates, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                vmin=0, vmax=0.8, cbar=False)
    ax.set_title(f'{cap_band} (n={len(subset)})', fontweight='bold')
    ax.set_xlabel('lon_band')
    ax.set_ylabel('lat_band')

plt.suptitle('Underperforming Rate: lat_band × lon_band by capacity_band', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "granular_09_lat_lon_by_capacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_09_lat_lon_by_capacity.png")

# =============================================================================
# 10. MULTI-LEVEL GROUPING: PRIMARY_FUEL + CAPACITY_BAND + OTHER_FUEL1
# =============================================================================
print("\n" + "="*60)
print("3-WAY GROUPING: PRIMARY_FUEL + CAPACITY_BAND + OTHER_FUEL1")
print("="*60)

# Create combined group
train['fuel_cap_other'] = (train['primary_fuel'] + '_' +
                            train['capacity_band'] + '_' +
                            train['other_fuel1'])

group_stats = train.groupby('fuel_cap_other').agg({
    'underperforming': ['mean', 'count', 'std']
}).reset_index()
group_stats.columns = ['group', 'rate', 'count', 'std']
group_stats = group_stats[group_stats['count'] >= 10]  # Min 10 samples
group_stats = group_stats.sort_values('rate', ascending=False)

print(f"\nGroups with >= 10 samples: {len(group_stats)}")
print(f"\nTop 15 highest underperforming rate groups:")
print(group_stats.head(15).to_string(index=False))

print(f"\nTop 15 lowest underperforming rate groups:")
print(group_stats.tail(15).to_string(index=False))

# Variance analysis
print(f"\nRate variance across groups: {group_stats['rate'].var():.4f}")
print(f"Overall rate variance (baseline): {train['underperforming'].var():.4f}")

# =============================================================================
# 11. 4-WAY GROUPING: PRIMARY_FUEL + CAPACITY_BAND + LAT_BAND + LON_BAND
# =============================================================================
print("\n" + "="*60)
print("4-WAY GROUPING: PRIMARY_FUEL + CAPACITY_BAND + LAT_BAND + LON_BAND")
print("="*60)

train['fuel_cap_lat_lon'] = (train['primary_fuel'] + '_' +
                              train['capacity_band'] + '_' +
                              train['lat_band'] + '_' +
                              train['lon_band'])

group_stats_4way = train.groupby('fuel_cap_lat_lon').agg({
    'underperforming': ['mean', 'count', 'std']
}).reset_index()
group_stats_4way.columns = ['group', 'rate', 'count', 'std']
group_stats_4way = group_stats_4way[group_stats_4way['count'] >= 5]  # Min 5 samples

print(f"\nGroups with >= 5 samples: {len(group_stats_4way)}")
print(f"\nGroups with 0% underperforming: {(group_stats_4way['rate'] == 0).sum()}")
print(f"Groups with 100% underperforming: {(group_stats_4way['rate'] == 1).sum()}")
print(f"Groups with rate > 50%: {(group_stats_4way['rate'] > 0.5).sum()}")
print(f"Groups with rate < 20%: {(group_stats_4way['rate'] < 0.2).sum()}")

# Variance analysis
print(f"\nRate variance across 4-way groups: {group_stats_4way['rate'].var():.4f}")

# =============================================================================
# 12. FULL GRANULARITY GROUPING ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("VARIANCE DECOMPOSITION BY GROUPING LEVEL")
print("="*60)

def calc_group_variance(df, group_cols, min_count=5):
    """Calculate variance in target rate across groups"""
    group_key = '_'.join(group_cols)
    df_temp = df.copy()
    df_temp[group_key] = df_temp[group_cols].astype(str).agg('_'.join, axis=1)

    stats = df_temp.groupby(group_key).agg({
        'underperforming': ['mean', 'count']
    }).reset_index()
    stats.columns = ['group', 'rate', 'count']
    stats = stats[stats['count'] >= min_count]

    return {
        'grouping': group_key,
        'n_groups': len(stats),
        'total_samples_in_groups': stats['count'].sum(),
        'rate_variance': stats['rate'].var(),
        'rate_std': stats['rate'].std(),
        'min_rate': stats['rate'].min(),
        'max_rate': stats['rate'].max(),
        'rate_range': stats['rate'].max() - stats['rate'].min()
    }

grouping_levels = [
    ['primary_fuel'],
    ['primary_fuel', 'capacity_band'],
    ['primary_fuel', 'other_fuel1'],
    ['primary_fuel', 'lat_band'],
    ['primary_fuel', 'lon_band'],
    ['primary_fuel', 'capacity_band', 'other_fuel1'],
    ['primary_fuel', 'capacity_band', 'lat_band'],
    ['primary_fuel', 'capacity_band', 'lon_band'],
    ['primary_fuel', 'lat_band', 'lon_band'],
    ['capacity_band', 'lat_band', 'lon_band'],
    ['primary_fuel', 'capacity_band', 'lat_band', 'lon_band'],
    ['primary_fuel', 'capacity_band', 'other_fuel1', 'lat_band'],
    ['primary_fuel', 'capacity_band', 'other_fuel1', 'lon_band'],
]

results = []
for grouping in grouping_levels:
    result = calc_group_variance(train, grouping, min_count=5)
    results.append(result)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('rate_variance', ascending=False)

print("\nVariance by grouping level (higher = more discriminative):")
print(results_df[['grouping', 'n_groups', 'rate_variance', 'rate_range']].to_string(index=False))

# =============================================================================
# 13. VISUALIZE VARIANCE BY GROUPING LEVEL
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

x = range(len(results_df))
bars = ax.bar(x, results_df['rate_variance'], color='steelblue', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(results_df['grouping'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Rate Variance (higher = more discriminative)')
ax.set_title('How Much Does Each Grouping Explain Target Variation?', fontweight='bold')
ax.axhline(y=train['underperforming'].var(), color='red', linestyle='--', linewidth=2,
           label=f'Baseline variance ({train["underperforming"].var():.4f})')
ax.legend()

# Add number of groups labels
for bar, n_groups in zip(bars, results_df['n_groups']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'n={n_groups}', ha='center', fontsize=7, rotation=90)

plt.tight_layout()
plt.savefig(output_dir / "granular_10_variance_by_grouping.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: granular_10_variance_by_grouping.png")

# =============================================================================
# 14. TOP DISCRIMINATIVE GROUPS VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top grouping: primary_fuel + capacity_band + lat_band + lon_band
best_grouping = ['primary_fuel', 'capacity_band', 'lat_band', 'lon_band']
train['best_group'] = train[best_grouping].astype(str).agg('_'.join, axis=1)
best_stats = train.groupby('best_group').agg({
    'underperforming': ['mean', 'count']
}).reset_index()
best_stats.columns = ['group', 'rate', 'count']
best_stats = best_stats[best_stats['count'] >= 5]

# Highest rate groups
ax1 = axes[0]
top_high = best_stats.nlargest(15, 'rate')
colors = [COLOR_1] * len(top_high)
ax1.barh(range(len(top_high)), top_high['rate'] * 100, color=colors, edgecolor='black')
ax1.set_yticks(range(len(top_high)))
ax1.set_yticklabels([f"{g[:40]}... (n={c})" if len(g) > 40 else f"{g} (n={c})"
                     for g, c in zip(top_high['group'], top_high['count'])], fontsize=7)
ax1.set_xlabel('Underperforming Rate (%)')
ax1.set_title(f'Top 15 Highest Rate Groups\n{" + ".join(best_grouping)}', fontweight='bold')
ax1.invert_yaxis()

# Lowest rate groups
ax2 = axes[1]
top_low = best_stats.nsmallest(15, 'rate')
colors = [COLOR_0] * len(top_low)
ax2.barh(range(len(top_low)), top_low['rate'] * 100, color=colors, edgecolor='black')
ax2.set_yticks(range(len(top_low)))
ax2.set_yticklabels([f"{g[:40]}... (n={c})" if len(g) > 40 else f"{g} (n={c})"
                     for g, c in zip(top_low['group'], top_low['count'])], fontsize=7)
ax2.set_xlabel('Underperforming Rate (%)')
ax2.set_title(f'Top 15 Lowest Rate Groups\n{" + ".join(best_grouping)}', fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / "granular_11_extreme_groups.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_11_extreme_groups.png")

# =============================================================================
# 15. TARGET ENCODING POTENTIAL ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("TARGET ENCODING POTENTIAL")
print("="*60)

# For each grouping, calculate how much variance is explained
print("\nGrouping quality (higher rate_range = better for target encoding):")
print(results_df[['grouping', 'n_groups', 'rate_range', 'rate_variance']].to_string(index=False))

# Best single features
single_features = ['primary_fuel', 'capacity_band', 'lat_band', 'lon_band', 'other_fuel1', 'fuel_group']
print("\n" + "="*60)
print("SINGLE FEATURE DISCRIMINATION POWER")
print("="*60)

for feat in single_features:
    rates = train.groupby(feat)['underperforming'].mean()
    print(f"\n{feat}:")
    print(f"  Unique values: {train[feat].nunique()}")
    print(f"  Rate range: {rates.min():.2%} - {rates.max():.2%} (diff: {rates.max()-rates.min():.2%})")
    print(f"  Rate std: {rates.std():.3f}")

# =============================================================================
# 16. INTERACTION IMPORTANCE MATRIX
# =============================================================================
print("\n" + "="*60)
print("FEATURE INTERACTION IMPORTANCE")
print("="*60)

# Create pairwise interaction analysis
cat_features = ['primary_fuel', 'capacity_band', 'lat_band', 'lon_band', 'other_fuel1', 'fuel_group']

interaction_matrix = pd.DataFrame(index=cat_features, columns=cat_features)

for f1 in cat_features:
    for f2 in cat_features:
        if f1 == f2:
            # Single feature variance
            rates = train.groupby(f1)['underperforming'].mean()
            interaction_matrix.loc[f1, f2] = rates.std()
        else:
            # Pairwise interaction
            rates = train.groupby([f1, f2])['underperforming'].mean()
            interaction_matrix.loc[f1, f2] = rates.std()

interaction_matrix = interaction_matrix.astype(float)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            cbar_kws={'label': 'Rate Std Dev'})
ax.set_title('Feature Interaction Importance\n(How much does combining features discriminate target)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / "granular_12_interaction_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: granular_12_interaction_matrix.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print("\nBEST DISCRIMINATIVE GROUPINGS (by rate variance):")
for _, row in results_df.head(5).iterrows():
    print(f"  {row['grouping']}: variance={row['rate_variance']:.4f}, range={row['rate_range']:.2f}")

print("\nKEY INSIGHTS:")
print("1. Location bands (lat/lon) add discrimination power when combined with fuel type")
print("2. capacity_band + primary_fuel is a strong combination")
print("3. other_fuel1 adds discrimination for fossil fuels")
print("4. 4-way grouping can identify near-0% and near-100% underperforming groups")

print(f"\nAll plots saved to: {output_dir.absolute()}")
