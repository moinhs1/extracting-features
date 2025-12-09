# Layer 3 Feature Engineering Design

**Date:** 2025-12-09
**Status:** Approved
**Module:** 3.6 (Submodule 3.6)
**Phase:** 2

---

## Overview

Layer 3 transforms the hourly vital sign grid (Layer 2) into clinically meaningful features capturing trajectory dynamics. It produces two outputs:

1. **Time-series features** — Feature values at each hour for sequential models (LSTM, world models)
2. **Summary features** — Aggregated features per patient for traditional ML (XGBoost, logistic regression)

### Objectives

| Goal | Description |
|------|-------------|
| Capture dynamics | Rolling statistics reveal short-term patterns |
| Detect trends | Slope and direction features show trajectory |
| Quantify instability | Variability features measure physiological volatility |
| Flag clinical events | Threshold features count time in abnormal states |
| Enable interpretability | Features have clinical meaning vs black-box embeddings |

### Input/Output

**Input:**
- `layer2/hourly_grid.parquet` — 40M rows with hourly vital values
- `layer2/hourly_tensors.h5` — Imputation tier information

**Output:**
- `layer3/timeseries_features.parquet` — ~40M rows (patient × 745 hours)
- `layer3/summary_features.parquet` — ~7,689 rows (one per patient)

---

## Vitals Processed

9 total (7 raw + 2 composites):

| Type | Vitals |
|------|--------|
| Raw | HR, SBP, DBP, MAP, RR, SpO2, Temp |
| Composites | shock_index (HR/SBP), pulse_pressure (SBP-DBP) |

Composites calculated first, then treated identically to raw vitals for all features.

---

## Key Design Decisions

### Decision 1: Dual Output (Time-Series + Summary)

Both granularities produced to support different downstream methods:
- Time-series → LSTM, world models, trajectory analysis
- Summary → XGBoost, logistic regression, Cox models

### Decision 2: Hybrid Imputation Handling

| Feature Type | Use Tiers | Rationale |
|--------------|-----------|-----------|
| Rolling stats (std, CV, range) | 1-2 only | Variance on patient-mean = 0 by construction |
| Rolling means | 1-2 only | Mean of flat imputation = not informative |
| Trend features (slope, R²) | 1-2 only | Slope on flat imputation = 0 (artificial) |
| Threshold features | All (1-4) | Counting hours above threshold is robust |
| Time-to-first features | All (1-4) | Finding first crossing is robust |

### Decision 3: Clinical Summary Windows

5 periods covering the full trajectory:

| Window | Hours | Clinical Phase |
|--------|-------|----------------|
| `pre` | -24 to 0 | Pre-PE baseline |
| `acute` | 0 to 24 | Acute phase (highest risk) |
| `early` | 24 to 72 | Early treatment response |
| `stab` | 72 to 168 | Stabilization (days 3-7) |
| `recov` | 168 to 720 | Recovery (days 7-30) |

### Decision 4: Rolling Window Sizes

6h, 12h, 24h windows based on data density (~16% Tier 1-2):
- 6h: ~1 observation expected (minimum practical)
- 12h: ~2 observations (shift-level patterns)
- 24h: ~4 observations (daily trends)

### Decision 5: Variability Features for All Vitals

Include RMSSD and successive_var for all 9 vitals to enable discovery of novel associations. Feature selection during modeling can remove non-informative ones.

---

## Time-Series Feature Specifications

### Rolling Window Statistics

Calculated on **Tier 1-2 data only** (observed + forward-fill).

| Feature | Formula | Windows |
|---------|---------|---------|
| `{vital}_roll{w}h_mean` | Rolling mean | 6h, 12h, 24h |
| `{vital}_roll{w}h_std` | Rolling std deviation | 6h, 12h, 24h |
| `{vital}_roll{w}h_cv` | Coefficient of variation (std/mean) | 6h, 12h, 24h |
| `{vital}_roll{w}h_min` | Rolling minimum | 6h, 12h, 24h |
| `{vital}_roll{w}h_max` | Rolling maximum | 6h, 12h, 24h |
| `{vital}_roll{w}h_range` | Rolling range (max - min) | 6h, 12h, 24h |

**Count:** 9 vitals × 6 stats × 3 windows = **162 features**

### Trend Features

Calculated on **Tier 1-2 data only**.

| Feature | Formula | Windows |
|---------|---------|---------|
| `{vital}_slope{w}h` | Linear regression slope | 6h, 12h, 24h |
| `{vital}_slope{w}h_r2` | R² of regression (trend reliability) | 6h, 12h, 24h |
| `{vital}_direction{w}h` | Categorical: -1=worsening, 0=stable, 1=improving | 6h, 12h, 24h |

**Count:** 9 vitals × 3 trend features × 3 windows = **81 features**

### Variability Features

Calculated on **Tier 1-2 data only**, using consecutive observations.

| Feature | Formula |
|---------|---------|
| `{vital}_rmssd` | Root mean square of successive differences |
| `{vital}_successive_var` | Sum of absolute consecutive differences |

**Count:** 9 vitals × 2 variability features = **18 features**

### Threshold-Based Features (Cumulative)

Calculated on **all tiers (1-4)** — robust to imputation.

| Feature | Condition | Description |
|---------|-----------|-------------|
| `hours_tachycardia` | HR > 100 | Cumulative hours with elevated HR |
| `hours_bradycardia` | HR < 60 | Cumulative hours with low HR |
| `hours_hypotension` | SBP < 90 | Cumulative hours hypotensive |
| `hours_hypertension` | SBP > 180 | Cumulative hours hypertensive |
| `hours_hypoxemia` | SpO2 < 92 | Cumulative hours with low oxygen |
| `hours_tachypnea` | RR > 24 | Cumulative hours with rapid breathing |
| `hours_shock` | MAP < 65 | Cumulative hours in shock range |
| `hours_fever` | Temp > 38.5 | Cumulative hours febrile |
| `hours_hypothermia` | Temp < 36 | Cumulative hours hypothermic |
| `hours_high_shock_index` | shock_index > 0.9 | Cumulative hours hemodynamically compromised |

### Time-to-First Features

| Feature | Condition | Description |
|---------|-----------|-------------|
| `time_to_first_tachycardia` | HR > 100 | Hours until first crossing (NaN if never) |
| `time_to_first_hypotension` | SBP < 90 | Hours until first crossing |
| `time_to_first_hypoxemia` | SpO2 < 92 | Hours until first crossing |
| `time_to_first_shock` | MAP < 65 | Hours until first crossing |
| `time_to_first_high_shock_index` | shock_index > 0.9 | Hours until first crossing |

**Count:** 10 cumulative + 5 time-to-first = **15 threshold features**

### Data Density Features

Always included — lets models assess feature reliability.

| Feature | Description |
|---------|-------------|
| `{vital}_obs_pct` | % of hours with Tier 1-2 data (per vital) |
| `{vital}_obs_count` | Count of observed hours (per vital) |
| `any_vital_obs_pct` | % of hours with ANY vital observed |

**Count:** 9 vitals × 2 + 1 = **19 density features**

---

## Time-Series Feature Count Summary

| Category | Count |
|----------|-------|
| Rolling statistics | 162 |
| Trend features | 81 |
| Variability features | 18 |
| Threshold features | 15 |
| Data density features | 19 |
| **Total per hour** | **~295 features** |

---

## Summary Features (Per Patient)

### Summary Feature Aggregations

For each of the 5 windows, aggregate the time-series features:

| Aggregation | Applied To | Example |
|-------------|------------|---------|
| Mean | Rolling stats, variability | `HR_roll6h_mean_acute_mean` |
| Max | Rolling stats | `HR_roll6h_max_acute_max` |
| Min | Rolling stats | `SBP_roll24h_min_stab_min` |
| Slope (first→last) | Raw values | `HR_acute_slope` |
| Total | Threshold hours | `hours_tachycardia_early` |

### Summary Feature Count Estimate

| Category | Calculation | Count |
|----------|-------------|-------|
| Rolling stats summary | 162 features × 3 aggs × 5 windows | ~2,430 |
| Trend summary | 81 features × 2 aggs × 5 windows | ~810 |
| Variability summary | 18 features × 2 aggs × 5 windows | ~180 |
| Threshold per window | 15 features × 5 windows | 75 |
| Data density per window | 19 features × 5 windows | 95 |
| **Total (estimated)** | | **~3,500 features** |

*Note: Actual count may be lower after removing redundant combinations.*

---

## Output Schemas

### Time-Series Output (`timeseries_features.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| hour_from_pe | int | Hour bucket (-24 to 720) |
| `{vital}_roll{w}h_{stat}` | float | Rolling statistics (162 cols) |
| `{vital}_slope{w}h` | float | Trend slopes (27 cols) |
| `{vital}_slope{w}h_r2` | float | Trend R² (27 cols) |
| `{vital}_direction{w}h` | int8 | Direction -1/0/1 (27 cols) |
| `{vital}_rmssd` | float | Variability RMSSD (9 cols) |
| `{vital}_successive_var` | float | Successive variance (9 cols) |
| `hours_{condition}` | float | Cumulative threshold hours (10 cols) |
| `{vital}_obs_pct` | float | Observation % to this point (9 cols) |

**Rows:** ~7,689 patients × 745 hours = ~5.7M rows (or filtered to observed range)

### Summary Output (`summary_features.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier (primary key) |
| `{feature}_{window}_{agg}` | float | Aggregated features (~3,500 cols) |

**Rows:** 7,689 (one per patient)

---

## Clinical Direction Definitions

"Improving" vs "worsening" depends on the vital:

| Vital | Improving | Worsening |
|-------|-----------|-----------|
| HR | Decreasing toward normal (60-100) | Increasing (tachycardia) or <50 |
| SBP | Increasing toward normal (>90) | Decreasing (hypotension) |
| DBP | Stable in normal range | Extreme changes |
| MAP | Increasing toward >65 | Decreasing (shock) |
| RR | Decreasing toward normal (12-20) | Increasing (tachypnea) |
| SpO2 | Increasing toward 100% | Decreasing (hypoxemia) |
| Temp | Moving toward 37°C | Moving away from 37°C |
| shock_index | Decreasing toward <0.7 | Increasing toward >0.9 |
| pulse_pressure | Widening (better output) | Narrowing (poor output) |

**Implementation:** Use slope sign + current position relative to normal range.

---

## Validation Checks

| Check | Expected | Action if Failed |
|-------|----------|------------------|
| No impossible values | shock_index 0.1-5.0, pulse_pressure 10-150 | Clip or flag |
| Feature completeness | >95% patients have summary features | Log sparse patients |
| Correlation sanity | roll6h_mean ≈ roll12h_mean (correlated) | Review calculation |
| Threshold consistency | hours_tachycardia ≤ 745 | Review logic |

---

## Method-Specific Usage

| Method | Layer 3 Output | Key Features |
|--------|----------------|--------------|
| LSTM / GRU | timeseries_features.parquet | Sequential rolling stats |
| World Models | timeseries_features.parquet | State evolution features |
| XGBoost / RF | summary_features.parquet | All ~3,500 summary features |
| Cox Regression | summary_features.parquet | Threshold + trend features |
| Trajectory Clustering | timeseries_features.parquet | Trend direction sequences |

---

## Implementation Notes

### Performance Considerations

- Use vectorized pandas/numpy operations for rolling stats
- Consider numba JIT for trend calculations
- Parallelize by patient for summary aggregations
- Estimate memory: 40M rows × 300 features × 4 bytes ≈ 48 GB (process in chunks)

### Composite Vital Calculation

```python
# Calculate composites where both components exist
shock_index = df['HR'] / df['SBP']  # where both non-null
pulse_pressure = df['SBP'] - df['DBP']  # where both non-null
```

### Rolling Window Edge Cases

- First hours of each patient: use expanding window until full window size reached
- NaN handling: skip NaNs in rolling calculations (min_periods=1)
- Empty windows: return NaN for that feature

---

## Summary

Layer 3 produces ~295 time-series features per hour and ~3,500 summary features per patient, capturing:

1. **Short-term dynamics** via rolling windows (6h, 12h, 24h)
2. **Trajectory trends** via slope and direction features
3. **Physiological instability** via variability features
4. **Clinical events** via threshold crossings
5. **Data quality** via density features

The hybrid imputation approach ensures feature validity while maximizing coverage.

---

**Document Version:** 1.0
**Approved:** 2025-12-09
