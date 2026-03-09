"""
Power Plant Underperformance - Primary Fuel Analysis
Visualizes patterns within each primary fuel type (Solar, Wind, Hydro, Gas, Coal, etc.)

Run with: python visualize_by_fuel.py
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

# Define consistent colors for target
COLOR_0 = '#2ecc71'  # Green for normal
COLOR_1 = '#e74c3c'  # Red for underperforming
TARGET_COLORS = [COLOR_0, COLOR_1]

# Primary fuel colors
FUEL_COLORS = {
    'Solar': '#f1c40f',
    'Wind': '#3498db',
    'Hydro': '#2980b9',
    'Gas': '#e67e22',
    'Coal': '#2c3e50',
    'Oil': '#8b4513',
    'Nuclear': '#9b59b6',
    'Waste': '#16a085',
    'Biomass': '#27ae60',
    'Other': '#7f8c8d',
    'Geothermal': '#d35400',
    'Petcoke': '#34495e',
    'Storage': '#1abc9c'
}

# Create output directory
output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# Load data
print("Loading data...")
train = pd.read_csv("public/train.csv")

numerical_features = ['capacity_mw', 'capacity_log_mw', 'plant_age', 'abs_latitude',
                       'latitude', 'longitude', 'age_x_capacity']

# Get primary fuels with enough data
fuel_counts = train['primary_fuel'].value_counts()
print(f"\nPrimary fuel distribution:\n{fuel_counts}")

# Filter to fuels with at least 50 samples for meaningful analysis
min_samples = 50
main_fuels = fuel_counts[fuel_counts >= min_samples].index.tolist()
print(f"\nAnalyzing fuels with >= {min_samples} samples: {main_fuels}")

# =============================================================================
# 1. OVERVIEW: TARGET RATE BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By count
ax1 = axes[0]
fuel_order = fuel_counts.index
fuel_target = train.groupby(['primary_fuel', 'underperforming']).size().unstack(fill_value=0)
fuel_target = fuel_target.reindex(fuel_order)
fuel_target.plot(kind='bar', stacked=True, ax=ax1, color=TARGET_COLORS, edgecolor='black')
ax1.set_title('Plant Count by Primary Fuel (stacked by target)', fontweight='bold')
ax1.set_xlabel('Primary Fuel')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)
ax1.legend(['Normal', 'Underperforming'], title='Target')

# By rate
ax2 = axes[1]
fuel_rates = train.groupby('primary_fuel')['underperforming'].agg(['mean', 'count'])
fuel_rates = fuel_rates.sort_values('mean', ascending=False)
colors = [COLOR_1 if r > train['underperforming'].mean() else COLOR_0 for r in fuel_rates['mean']]
bars = ax2.bar(fuel_rates.index, fuel_rates['mean'] * 100, color=colors, edgecolor='black')
ax2.axhline(y=train['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2,
            label=f'Overall avg ({train["underperforming"].mean()*100:.1f}%)')
ax2.set_title('Underperforming Rate by Primary Fuel', fontweight='bold')
ax2.set_xlabel('Primary Fuel')
ax2.set_ylabel('Underperforming Rate (%)')
ax2.tick_params(axis='x', rotation=45)
ax2.legend()

for bar, (idx, row) in zip(bars, fuel_rates.iterrows()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{row["mean"]*100:.0f}%\nn={int(row["count"])}', ha='center', fontsize=7)

plt.tight_layout()
plt.savefig(output_dir / "primary_01_overview_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_01_overview_by_fuel.png")

# =============================================================================
# 2. CAPACITY DISTRIBUTION BY PRIMARY FUEL
# =============================================================================
n_fuels = len(main_fuels)
n_cols = 4
n_rows = (n_fuels + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel]

    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]['capacity_log_mw']
        if len(data) > 0:
            label = 'Normal' if target == 0 else 'Underperforming'
            ax.hist(data, bins=30, alpha=0.6, color=TARGET_COLORS[target], label=label, density=True)

    ax.set_title(f'{fuel} (n={len(subset)}, rate={subset["underperforming"].mean()*100:.0f}%)', fontweight='bold')
    ax.set_xlabel('Log Capacity')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

    # Add mean lines
    for target, ls in [(0, '--'), (1, ':')]:
        data = subset[subset['underperforming'] == target]['capacity_log_mw']
        if len(data) > 0:
            ax.axvline(x=data.mean(), color=TARGET_COLORS[target], linestyle=ls, linewidth=2)

# Hide unused
for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Capacity Distribution by Primary Fuel (dashed=normal mean, dotted=underperf mean)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_02_capacity_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_02_capacity_by_fuel.png")

# =============================================================================
# 3. PLANT AGE DISTRIBUTION BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel]

    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]['plant_age']
        if len(data) > 0:
            label = 'Normal' if target == 0 else 'Underperforming'
            ax.hist(data, bins=30, alpha=0.6, color=TARGET_COLORS[target], label=label, density=True)

    ax.set_title(f'{fuel}', fontweight='bold')
    ax.set_xlabel('Plant Age (years)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)

    # Add mean lines
    for target, ls in [(0, '--'), (1, ':')]:
        data = subset[subset['underperforming'] == target]['plant_age']
        if len(data) > 0:
            ax.axvline(x=data.mean(), color=TARGET_COLORS[target], linestyle=ls, linewidth=2)

for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Plant Age Distribution by Primary Fuel', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_03_age_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_03_age_by_fuel.png")

# =============================================================================
# 4. SCATTER: AGE vs CAPACITY BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel]

    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]
        if len(data) > 0:
            label = 'Normal' if target == 0 else 'Underperforming'
            ax.scatter(data['plant_age'], data['capacity_log_mw'],
                       c=TARGET_COLORS[target], alpha=0.5, s=15, label=label)

    ax.set_title(f'{fuel}', fontweight='bold')
    ax.set_xlabel('Plant Age')
    ax.set_ylabel('Log Capacity')
    ax.legend(fontsize=7)

for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Age vs Capacity by Primary Fuel', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_04_age_vs_capacity.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_04_age_vs_capacity.png")

# =============================================================================
# 5. VIOLIN PLOTS: CAPACITY BY FUEL AND TARGET
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

# Prepare data for violin plot
plot_data = []
for fuel in main_fuels:
    subset = train[train['primary_fuel'] == fuel]
    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]['capacity_log_mw']
        for val in data:
            plot_data.append({'Fuel': fuel, 'Target': 'Normal' if target == 0 else 'Underperf', 'Capacity': val})

plot_df = pd.DataFrame(plot_data)

# Create split violin plot
positions = []
for i, fuel in enumerate(main_fuels):
    positions.extend([i*2, i*2+0.5])

parts = ax.violinplot([plot_df[(plot_df['Fuel']==f) & (plot_df['Target']=='Normal')]['Capacity'].values
                       for f in main_fuels],
                      positions=[i*2 for i in range(len(main_fuels))],
                      showmeans=False, showmedians=False, widths=0.4)
for pc in parts['bodies']:
    pc.set_facecolor(COLOR_0)
    pc.set_alpha(0.6)

parts2 = ax.violinplot([plot_df[(plot_df['Fuel']==f) & (plot_df['Target']=='Underperf')]['Capacity'].values
                        for f in main_fuels],
                       positions=[i*2+0.5 for i in range(len(main_fuels))],
                       showmeans=False, showmedians=False, widths=0.4)
for pc in parts2['bodies']:
    pc.set_facecolor(COLOR_1)
    pc.set_alpha(0.6)

ax.set_xticks([i*2+0.25 for i in range(len(main_fuels))])
ax.set_xticklabels(main_fuels, rotation=45, ha='right')
ax.set_xlabel('Primary Fuel')
ax.set_ylabel('Log Capacity')
ax.set_title('Capacity Distribution by Fuel and Target (Green=Normal, Red=Underperf)', fontweight='bold')

# Add legend
from matplotlib.patches import Patch
ax.legend([Patch(facecolor=COLOR_0, alpha=0.6), Patch(facecolor=COLOR_1, alpha=0.6)],
          ['Normal', 'Underperforming'])

plt.tight_layout()
plt.savefig(output_dir / "primary_05_capacity_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_05_capacity_violin.png")

# =============================================================================
# 6. VIOLIN PLOTS: PLANT AGE BY FUEL AND TARGET
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

plot_data = []
for fuel in main_fuels:
    subset = train[train['primary_fuel'] == fuel]
    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]['plant_age']
        for val in data:
            plot_data.append({'Fuel': fuel, 'Target': 'Normal' if target == 0 else 'Underperf', 'Age': val})

plot_df = pd.DataFrame(plot_data)

parts = ax.violinplot([plot_df[(plot_df['Fuel']==f) & (plot_df['Target']=='Normal')]['Age'].values
                       for f in main_fuels],
                      positions=[i*2 for i in range(len(main_fuels))],
                      showmeans=False, showmedians=False, widths=0.4)
for pc in parts['bodies']:
    pc.set_facecolor(COLOR_0)
    pc.set_alpha(0.6)

parts2 = ax.violinplot([plot_df[(plot_df['Fuel']==f) & (plot_df['Target']=='Underperf')]['Age'].values
                        for f in main_fuels],
                       positions=[i*2+0.5 for i in range(len(main_fuels))],
                       showmeans=False, showmedians=False, widths=0.4)
for pc in parts2['bodies']:
    pc.set_facecolor(COLOR_1)
    pc.set_alpha(0.6)

ax.set_xticks([i*2+0.25 for i in range(len(main_fuels))])
ax.set_xticklabels(main_fuels, rotation=45, ha='right')
ax.set_xlabel('Primary Fuel')
ax.set_ylabel('Plant Age (years)')
ax.set_title('Plant Age Distribution by Fuel and Target (Green=Normal, Red=Underperf)', fontweight='bold')
ax.legend([Patch(facecolor=COLOR_0, alpha=0.6), Patch(facecolor=COLOR_1, alpha=0.6)],
          ['Normal', 'Underperforming'])

plt.tight_layout()
plt.savefig(output_dir / "primary_06_age_violin.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_06_age_violin.png")

# =============================================================================
# 7. CORRELATION WITH TARGET BY PRIMARY FUEL
# =============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

corr_data = []
for fuel in main_fuels:
    subset = train[train['primary_fuel'] == fuel]
    if len(subset) >= min_samples:
        for col in numerical_features:
            corr = subset[col].corr(subset['underperforming'])
            corr_data.append({'fuel': fuel, 'feature': col, 'correlation': corr})

corr_df = pd.DataFrame(corr_data)

# Pivot for heatmap
corr_pivot = corr_df.pivot(index='feature', columns='fuel', values='correlation')
corr_pivot = corr_pivot[main_fuels]  # Reorder columns

sns.heatmap(corr_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
            ax=ax, linewidths=0.5, vmin=-0.5, vmax=0.5)
ax.set_title('Feature Correlation with Target by Primary Fuel\n(Red=Positive, Green=Negative)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Primary Fuel')
ax.set_ylabel('Feature')

plt.tight_layout()
plt.savefig(output_dir / "primary_07_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_07_correlation_heatmap.png")

# =============================================================================
# 8. GEOGRAPHIC DISTRIBUTION BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel]

    for target in [0, 1]:
        data = subset[subset['underperforming'] == target]
        if len(data) > 0:
            label = 'Normal' if target == 0 else 'Underperforming'
            ax.scatter(data['longitude'], data['latitude'],
                       c=TARGET_COLORS[target], alpha=0.5, s=10, label=label)

    ax.set_title(f'{fuel}', fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(fontsize=7)

for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Geographic Distribution by Primary Fuel', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_08_geo_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_08_geo_by_fuel.png")

# =============================================================================
# 9. CAPACITY BAND ANALYSIS BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten()
capacity_order = ['tiny', 'small', 'medium', 'large', 'xlarge', 'utility']

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel]

    rates = subset.groupby('capacity_band')['underperforming'].agg(['mean', 'count'])
    rates = rates.reindex([b for b in capacity_order if b in rates.index])

    colors = [COLOR_1 if r > subset['underperforming'].mean() else COLOR_0 for r in rates['mean']]
    bars = ax.bar(rates.index, rates['mean'] * 100, color=colors, edgecolor='black')
    ax.axhline(y=subset['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2)
    ax.set_title(f'{fuel}', fontweight='bold')
    ax.set_xlabel('Capacity Band')
    ax.set_ylabel('Underperf Rate (%)')
    ax.tick_params(axis='x', rotation=45)

    for bar, (idx, row) in zip(bars, rates.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{row["mean"]*100:.0f}', ha='center', fontsize=7)

for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Underperforming Rate by Capacity Band within each Primary Fuel', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_09_capacity_band_by_fuel.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_09_capacity_band_by_fuel.png")

# =============================================================================
# 10. TOP OWNERS BY PRIMARY FUEL
# =============================================================================
for fuel in main_fuels[:6]:  # Top 6 fuels only
    subset = train[train['primary_fuel'] == fuel]

    if len(subset) < 100:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))

    # Top 10 owners
    top_owners = subset['owner_bucket'].value_counts().head(10).index
    owner_stats = subset[subset['owner_bucket'].isin(top_owners)].groupby('owner_bucket').agg({
        'underperforming': ['mean', 'count']
    }).reset_index()
    owner_stats.columns = ['owner', 'rate', 'count']
    owner_stats = owner_stats.sort_values('rate', ascending=True)

    colors = [COLOR_1 if r > subset['underperforming'].mean() else COLOR_0 for r in owner_stats['rate']]
    bars = ax.barh(owner_stats['owner'], owner_stats['rate'] * 100, color=colors, edgecolor='black')
    ax.axvline(x=subset['underperforming'].mean() * 100, color='blue', linestyle='--', linewidth=2)
    ax.set_title(f'{fuel} - Top 10 Owners by Underperforming Rate', fontweight='bold')
    ax.set_xlabel('Underperforming Rate (%)')

    plt.tight_layout()
    plt.savefig(output_dir / f"primary_10_owners_{fuel.lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: primary_10_owners_{fuel.lower()}.png")

# =============================================================================
# 11. AGE x CAPACITY HEATMAP BY PRIMARY FUEL
# =============================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4*n_rows))
axes = axes.flatten()

for i, fuel in enumerate(main_fuels):
    ax = axes[i]
    subset = train[train['primary_fuel'] == fuel].copy()

    if len(subset) < 50:
        axes[i].axis('off')
        continue

    # Create bins
    subset['age_bin'] = pd.cut(subset['plant_age'], bins=[0, 10, 25, 50, 150],
                                labels=['0-10', '10-25', '25-50', '50+'])
    subset['cap_bin'] = pd.cut(subset['capacity_log_mw'], bins=[0, 2, 4, 6, 10],
                                labels=['tiny', 'small', 'med', 'large'])

    heatmap_data = subset.groupby(['age_bin', 'cap_bin'], observed=True)['underperforming'].mean().unstack()

    if heatmap_data.shape[0] > 0 and heatmap_data.shape[1] > 0:
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=ax,
                    vmin=0, vmax=0.8, cbar=False)
        ax.set_title(f'{fuel}', fontweight='bold')
        ax.set_xlabel('Capacity')
        ax.set_ylabel('Age')

for j in range(len(main_fuels), len(axes)):
    axes[j].axis('off')

plt.suptitle('Underperforming Rate: Age × Capacity (Red=High, Green=Low)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "primary_11_age_capacity_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: primary_11_age_capacity_heatmap.png")

# =============================================================================
# 12. DETAILED STATISTICS BY PRIMARY FUEL
# =============================================================================
print("\n" + "="*100)
print("DETAILED STATISTICS BY PRIMARY FUEL")
print("="*100)

stats_summary = []

for fuel in main_fuels:
    subset = train[train['primary_fuel'] == fuel]
    normal = subset[subset['underperforming'] == 0]
    underperf = subset[subset['underperforming'] == 1]

    print(f"\n{'='*60}")
    print(f"PRIMARY FUEL: {fuel.upper()}")
    print(f"{'='*60}")
    print(f"Total: {len(subset)} | Normal: {len(normal)} ({len(normal)/len(subset)*100:.1f}%) | Underperforming: {len(underperf)} ({len(underperf)/len(subset)*100:.1f}%)")

    if len(underperf) > 0:
        print(f"\nMean comparison (Normal vs Underperforming):")
        print(f"{'Feature':<20} {'Normal':>12} {'Underperf':>12} {'Diff':>12}")
        print("-" * 60)

        for col in ['capacity_mw', 'capacity_log_mw', 'plant_age', 'age_x_capacity']:
            n_mean = normal[col].mean()
            u_mean = underperf[col].mean()
            diff = u_mean - n_mean
            print(f"{col:<20} {n_mean:>12.2f} {u_mean:>12.2f} {diff:>+12.2f}")

        # Store for summary
        stats_summary.append({
            'fuel': fuel,
            'count': len(subset),
            'rate': subset['underperforming'].mean(),
            'cap_diff': underperf['capacity_log_mw'].mean() - normal['capacity_log_mw'].mean(),
            'age_diff': underperf['plant_age'].mean() - normal['plant_age'].mean()
        })

# =============================================================================
# 13. SUMMARY: KEY PATTERNS BY FUEL TYPE
# =============================================================================
print("\n" + "="*100)
print("SUMMARY: KEY PATTERNS")
print("="*100)

summary_df = pd.DataFrame(stats_summary)
print("\nUnderperforming Rate by Fuel:")
print(summary_df[['fuel', 'count', 'rate']].sort_values('rate', ascending=False).to_string(index=False))

print("\n\nCapacity Pattern (negative = underperformers are SMALLER):")
print(summary_df[['fuel', 'cap_diff']].sort_values('cap_diff').to_string(index=False))

print("\n\nAge Pattern (positive = underperformers are OLDER):")
print(summary_df[['fuel', 'age_diff']].sort_values('age_diff', ascending=False).to_string(index=False))

# =============================================================================
# 14. SUMMARY VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Rate
ax1 = axes[0]
summary_df_sorted = summary_df.sort_values('rate', ascending=True)
colors = [COLOR_1 if r > 0.33 else COLOR_0 for r in summary_df_sorted['rate']]
ax1.barh(summary_df_sorted['fuel'], summary_df_sorted['rate'] * 100, color=colors, edgecolor='black')
ax1.axvline(x=33.1, color='blue', linestyle='--', linewidth=2)
ax1.set_xlabel('Underperforming Rate (%)')
ax1.set_title('Underperforming Rate by Fuel', fontweight='bold')

# Capacity difference
ax2 = axes[1]
summary_df_sorted = summary_df.sort_values('cap_diff')
colors = [COLOR_1 if d > 0 else COLOR_0 for d in summary_df_sorted['cap_diff']]
ax2.barh(summary_df_sorted['fuel'], summary_df_sorted['cap_diff'], color=colors, edgecolor='black')
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('Capacity Diff (Underperf - Normal)')
ax2.set_title('Capacity Pattern\n(Green=Underperf smaller, Red=larger)', fontweight='bold')

# Age difference
ax3 = axes[2]
summary_df_sorted = summary_df.sort_values('age_diff')
colors = [COLOR_1 if d > 0 else COLOR_0 for d in summary_df_sorted['age_diff']]
ax3.barh(summary_df_sorted['fuel'], summary_df_sorted['age_diff'], color=colors, edgecolor='black')
ax3.axvline(x=0, color='black', linewidth=1)
ax3.set_xlabel('Age Diff (Underperf - Normal)')
ax3.set_title('Age Pattern\n(Green=Underperf younger, Red=older)', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "primary_12_summary_patterns.png", dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: primary_12_summary_patterns.png")

print("\n" + "="*100)
print(f"All plots saved to: {output_dir.absolute()}")
print("="*100)
