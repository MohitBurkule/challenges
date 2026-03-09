# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning competition for predicting power plant underperformance. The goal is to build a binary classifier that predicts whether a power plant is underperforming based on features like capacity, age, fuel type, and location.

## Data Files

All data is in `public/`:
- `train.csv` - 6,475 rows with features and `underperforming` target (66.9% class 0, 33.1% class 1)
- `test.csv` - 1,611 rows with features only (no target)
- `sample_submission.csv` - Example submission format with `id` and `underperforming` columns

## Evaluation Metric

Composite score: `0.7 * ROC_AUC + 0.3 * Average_Precision`

Higher is better (range [0, 1]). The metric blend rewards both global ranking quality and correctly identifying positive underperforming plants.

### Metric Analysis & Exploits

1. **Both metrics are ranking-based** - They only care about the order of predictions, not actual probability values. Submit any monotonic transformation of scores without affecting the metric.

2. **AP heavily weights the top of the ranking** - Focus on correctly identifying "sure" underperformers. False positives at the top hurt more.

3. **No calibration required** - Raw model outputs work fine.

4. **Potential gaming of AP** - If you can identify a subset where you're highly confident about positives, push them to the very top.

## Features

**Numerical:**
- `capacity_mw`, `capacity_log_mw`, `plant_age`, `abs_latitude`, `latitude`, `longitude`, `age_x_capacity`

**Categorical:**
- `owner_bucket` - 121 categories (high cardinality)
- `primary_fuel` - 10 types: Solar, Gas, Hydro, Wind, Coal, Oil, Waste, Biomass, Nuclear, Other
- `fuel_group` - 3 categories: fossil, renewable, other
- `other_fuel1` - 13 categories, `__NONE__` when missing
- `capacity_band` - 6 levels: tiny, small, medium, large, xlarge, utility
- `lat_band`, `lon_band` - 8 location bins each

## Key Findings from EDA

### 1. Underperforming Definition
Target is generated from **relative generation performance within fuel-based groups** using quantile labeling. A plant underperforms if it generates less than expected compared to similar plants (same fuel type). This means patterns differ by fuel type.

### 2. Per-Fuel-Type Patterns (CRITICAL)

| Fuel | Count | Underperf Pattern |
|------|-------|-------------------|
| **Coal/Gas/Oil** | 2,174 | Smaller + Older = Underperforming |
| **Solar/Wind** | 2,667 | Smaller = Underperforming, Age doesn't matter |
| **Hydro** | 1,132 | **Larger** = Underperforming (OPPOSITE!) |
| **Waste** | 402 | **Larger** + Older = Underperforming |

**Implication**: Must use fuel-specific interactions or separate models.

### 3. Capacity Difference by Fuel (Normal vs Underperforming)
- Gas: 390 MW → 173 MW (normal vs underperf)
- Coal: 1041 MW → 667 MW
- Hydro: 54 MW → 101 MW (REVERSE - larger underperforms)
- Solar: 15 MW → 7 MW

### 4. Best Discriminative Groupings (by rate variance)

| Grouping | Variance | Rate Range |
|----------|----------|------------|
| `primary_fuel` × `other_fuel1` | 0.060 | 0% - 100% |
| `primary_fuel` alone | 0.044 | 32% - 100% |
| `fuel` × `capacity` × `lat` × `lon` | 0.042 | 0% - 100% |

### 5. Extreme Groups Found
**Highest (>50% underperforming):**
- `Gas_small_Oil`: 80%
- `Gas_large_Oil`: 70%
- `Gas_xlarge_Oil`: 58%

**Lowest (<10% underperforming):**
- `Gas_utility_Coal`: 0%
- `Solar_utility`: 5%
- `Wind_utility`: 16%

### 6. Feature Correlations with Target (Original)
- `capacity_mw`: -0.070 (larger = slightly less likely to underperform)
- `plant_age`: +0.048 (older = slightly more likely)
- `longitude`: +0.056

**Note**: These are WEAK because patterns are OPPOSITE per fuel type!

### 7. Neighbor Features (ENGINEERED)

**Best Configuration:**
- Categorical: `primary_fuel`, `other_fuel1`
- Numerical: `capacity_log_mw`, `plant_age`
- Weights: geo=0.5, cat=1.0, num=0.5
- k=10 neighbors, max_dist=500km

**Best Feature**: `nn10_max500_target_mean` has **0.34 correlation** with target!

This is 5x stronger than any original numerical feature. The local mean of similar plants' underperformance is highly predictive.

### 8. Single Feature Discrimination Power

| Feature | Rate Range | Power |
|---------|------------|-------|
| `primary_fuel` | 31.8% - 100% | BEST |
| `other_fuel1` | 0% - 63% | STRONG |
| `lat_band` | 10.5% - 42.9% | Moderate |
| `lon_band` | 19% - 39% | Weak |
| `capacity_band` | 25% - 37% | Weak alone |
| `fuel_group` | 32.6% - 34.6% | Useless alone |

## Visualization Files

All plots in `plots/`:
- `01-14_*.png` - General EDA
- `fuel_01-13_*.png` - Fuel group analysis
- `primary_01-12_*.png` - Per-primary-fuel analysis
- `granular_01-12_*.png` - Multi-feature grouping analysis
- `neighbor_01-03_*.png` - Neighbor feature engineering

## Engineered Features

`neighbor_features.csv` contains computed neighbor features:
- `nn{k}_max{d}_target_mean` - Mean target of k nearest similar plants within d km
- `nn{k}_max{d}_count` - Number of neighbors found
- `dist_to_nearest_underperf` - Distance to nearest underperforming plant
- `dist_to_nearest_normal` - Distance to nearest normal plant

## Commands

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm

# Run visualizations
python visualize.py
python visualize_by_fuel.py
python visualize_granular.py
python geospatial_neighbor_features.py

# Generate submission
python solution.py
```

## Solution Strategy

1. **Target Encoding**: Per `(primary_fuel, other_fuel1)` and `(primary_fuel, capacity_band)` groups
2. **Neighbor Features**: Use `nn10_max500_target_mean` and variants (0.34 correlation!)
3. **Fuel-Specific Interactions**: `primary_fuel` × `capacity_log_mw`, `primary_fuel` × `plant_age`
4. **Owner Encoding**: Target encode `owner_bucket` with smoothing
5. **Model**: LightGBM with `class_weight='balanced'`
6. **CV**: Use stratified K-fold to match evaluation metric
7. **No probability calibration needed** - ranking metrics only
