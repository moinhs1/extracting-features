# Vitals 5-Layer Architecture Design

**Date:** 2025-12-08
**Status:** Approved
**Replaces:** Original Submodules 3.4-3.10 plan

---

## Overview

A 5-layer architecture for transforming raw vital sign extractions into method-appropriate formats for multi-method PE trajectory analysis. Implements the "Unified Vital Signs Encoding Plan" with simplified storage (Parquet + HDF5).

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage | Parquet + HDF5 | Simple, portable, no database server needed |
| MAP handling | Extract + Calculate | Maximize data, flag calculated values |
| Temporal window | -24h to +720h | Full 30-day trajectory |
| Implementation | Phased (3 phases) | Deliver usable data faster |
| Location field | Skip for now | Not critical for initial analysis |
| Forward-fill limits | Vital-specific | Clinically accurate |

---

## Architecture

### Module Structure

```
Phase 1 (Layers 1-2):
├── 3.4: Layer 1 Builder - Canonical Records
│   └── Merge Phy/Hnp/Prg → unified schema with PE-relative timestamps
├── 3.5: Layer 2 Builder - Hourly Grid
│   └── Aggregate to hourly bins, generate value/mask/time-delta tensors

Phase 2 (Layer 3):
├── 3.6: Layer 3 Builder - Feature Engineering
│   └── Rolling stats, trends, variability, threshold features

Phase 3 (Layers 4-5):
├── 3.7: Layer 4 Builder - Embeddings
│   └── FPCA, autoencoder latents, cluster assignments
├── 3.8: Layer 5 Builder - World Model States
│   └── State vectors for dynamics learning

Supporting:
├── 3.9: Validation Framework (runs after each phase)
└── 3.10: Pipeline Orchestrator
```

### Storage Layout

```
module_3_vitals_processing/outputs/
├── layer1/
│   └── canonical_vitals.parquet       # All vitals, PE-aligned
├── layer2/
│   ├── hourly_grid.parquet            # Tabular hourly summaries
│   └── hourly_tensors.h5              # HDF5: values/masks/time_deltas
├── layer3/
│   └── engineered_features.parquet    # Rolling stats, trends, etc.
├── layer4/
│   └── embeddings.h5                  # FPCA, autoencoder, clusters
├── layer5/
│   └── world_model_states.h5          # State vectors
└── exports/
    ├── vitals_lcmm_long.csv           # R GBTM format
    └── vitals_gbtm_wide.csv           # R wide format
```

---

## Layer 1: Canonical Records

**Purpose:** Single source of truth merging all three extraction sources with PE-relative timestamps.

**Input Files:**
- `outputs/discovery/phy_vitals_raw.parquet`
- `outputs/discovery/hnp_vitals_raw.parquet`
- `outputs/discovery/prg_vitals_raw.parquet`
- `module_1_core_infrastructure/outputs/patient_timelines.pkl`

**Output:** `layer1/canonical_vitals.parquet`

### Schema

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| timestamp | datetime | Measurement time (seconds precision) |
| hours_from_pe | float | Hours relative to PE index (negative = before) |
| vital_type | str | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| value | float | Measurement value |
| units | str | bpm, mmHg, %, °C, breaths/min |
| source | str | 'phy', 'hnp', 'prg' |
| source_detail | str | encounter_type (phy) or extraction_context (hnp/prg) |
| confidence | float | 1.0 for phy, 0.6-1.0 for hnp/prg |
| is_calculated | bool | True if MAP calculated from SBP/DBP |
| is_flagged_abnormal | bool | From extractor or range check |
| report_number | str | Source report identifier (nullable) |

### Key Logic

1. Load all three extraction parquet files
2. Join with Module 1 patient timelines to get PE index time per patient
3. Calculate `hours_from_pe = (timestamp - pe_index_time).total_seconds() / 3600`
4. Filter to window: -24h to +720h from PE index
5. Calculate MAP where SBP and DBP exist at same timestamp (±5 min tolerance)
6. Apply physiological range validation (remove impossible values)
7. Deduplicate exact matches (same patient/time/vital/value)

---

## Layer 2: Hourly Aggregated Grid

**Purpose:** Regular hourly time grid with triple-component missing data encoding for ML methods.

**Input:** Layer 1 canonical records

### Output 1: `layer2/hourly_grid.parquet`

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| hour_from_pe | int | Hour bucket (-24 to +720) |
| vital_type | str | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| mean | float | Mean value in hour |
| median | float | Median value |
| std | float | Standard deviation (null if count=1) |
| min | float | Minimum value |
| max | float | Maximum value |
| count | int | Number of measurements |
| mask | int8 | 1=observed, 0=missing hour |

### Output 2: `layer2/hourly_tensors.h5`

```
/values          (n_patients, 745, 7)  float32  # 745 hours, 7 vitals
/masks           (n_patients, 745, 7)  int8     # 1=observed, 0=imputed
/time_deltas     (n_patients, 745, 7)  float32  # Hours since last observation
/imputation_tier (n_patients, 745, 7)  int8     # 1=obs, 2=ffill, 3=patient, 4=cohort
/patient_index   (n_patients,)         str      # EMPI mapping
/vital_index     (7,)                  str      # ['HR','SBP','DBP','MAP','RR','SPO2','TEMP']
/hour_index      (745,)                int      # [-24 to 720]
```

### Three-Tier Imputation

| Tier | Condition | Fill Value | Limits |
|------|-----------|------------|--------|
| 1 | Observed | Actual value | - |
| 2 | Forward-fill | Last observed | HR/BP/RR/MAP: 6h, SpO2: 4h, Temp: 12h |
| 3 | Patient mean | Patient's own mean | If forward-fill exceeded |
| 4 | Cohort mean | Population mean | If patient has zero measurements |

### Vital-Specific Forward-Fill Limits

| Vital | Max Forward-Fill |
|-------|------------------|
| HR | 6 hours |
| SBP | 6 hours |
| DBP | 6 hours |
| MAP | 6 hours |
| RR | 6 hours |
| SpO2 | 4 hours |
| Temp | 12 hours |

---

## Layer 3: Clinical Feature Engineering

**Purpose:** Derived features capturing trajectory dynamics for interpretable ML methods.

**Input:** Layer 2 hourly grid

**Output:** `layer3/engineered_features.parquet`

### Rolling Window Statistics (per vital, windows: 6h, 12h, 24h)

| Feature Pattern | Description |
|-----------------|-------------|
| `{vital}_roll{w}h_mean` | Rolling mean |
| `{vital}_roll{w}h_std` | Rolling standard deviation |
| `{vital}_roll{w}h_cv` | Coefficient of variation (std/mean) |
| `{vital}_roll{w}h_min` | Rolling minimum |
| `{vital}_roll{w}h_max` | Rolling maximum |
| `{vital}_roll{w}h_range` | Rolling range (max - min) |

### Trend Features (per vital, windows: 6h, 12h, 24h)

| Feature Pattern | Description |
|-----------------|-------------|
| `{vital}_slope{w}h` | Linear regression slope over window |
| `{vital}_slope{w}h_r2` | R² of trend (reliability) |
| `{vital}_direction{w}h` | Categorical: improving/stable/worsening |

### Variability Features (per vital)

| Feature Pattern | Description |
|-----------------|-------------|
| `{vital}_successive_var` | Sum of absolute consecutive differences |
| `{vital}_rmssd` | Root mean square of successive differences |

### Threshold-Based Features

| Feature | Threshold | Description |
|---------|-----------|-------------|
| `hours_tachycardia` | HR > 100 | Cumulative hours |
| `hours_hypotension` | SBP < 90 | Cumulative hours |
| `hours_hypoxemia` | SpO2 < 92 | Cumulative hours |
| `hours_tachypnea` | RR > 24 | Cumulative hours |
| `hours_shock` | MAP < 65 | Cumulative hours |
| `time_to_first_tachycardia` | HR > 100 | Hours until first crossing |
| `time_to_first_hypotension` | SBP < 90 | Hours until first crossing |
| `time_to_first_hypoxemia` | SpO2 < 92 | Hours until first crossing |
| `time_to_first_tachypnea` | RR > 24 | Hours until first crossing |
| `time_to_first_shock` | MAP < 65 | Hours until first crossing |

**Output Shape:** One row per patient-hour with ~150-200 features.

---

## Layer 4: Embeddings

**Purpose:** Dense vector representations of vital sign patterns for advanced ML.

**Input:** Layer 2 hourly tensors

**Output:** `layer4/embeddings.h5`

### Functional PCA (per vital)

```
/fpc_scores/{vital}    (n_patients, 10)   # First 10 FPC scores per vital
/fpc_explained_var     (7, 10)            # Variance explained per component
```

- Fit spline curves to each patient's trajectory
- Extract principal components capturing dominant variation modes
- FPC1 = overall level, FPC2 = trajectory shape, etc.

### Autoencoder Latents

```
/autoencoder_latents   (n_patients, 745, 32)  # 32-dim latent per timestep
/reconstruction_error  (n_patients,)          # Per-patient reconstruction quality
```

- Train VAE on Layer 2 hourly tensors (all vitals jointly)
- Captures nonlinear cross-vital patterns

### Trajectory Clusters

```
/cluster_ids           (n_patients,)          # Hard cluster assignment
/cluster_probs         (n_patients, k)        # Soft membership probabilities
/cluster_centroids     (k, 745, 7)            # Centroid trajectories
```

- DTW-based clustering of vital trajectories
- k determined by silhouette score

---

## Layer 5: World Model States

**Purpose:** Complete patient state vectors optimized for dynamics learning.

**Input:** Layers 2, 3, 4

**Output:** `layer5/world_model_states.h5`

```
/states        (n_patients, 745, ~60)   # State vector per timestep
/patient_index (n_patients,)            # EMPI mapping
```

### State Vector Components (~60 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Raw vital values | 7 | HR, SBP, DBP, MAP, RR, SpO2, Temp |
| Hours since PE index | 1 | Temporal position |
| Recent slopes (6h) | 7 | Trend per vital |
| Recent CV (6h) | 7 | Variability per vital |
| Mask indicators | 7 | Which vitals observed |
| Time-since-last | 7 | Per vital |
| FPC scores (top 3) | 21 | 3 per vital |
| Reserved | ~3 | Clinical status flags |

---

## Validation & Quality Control

### Layer 1 Validation

| Check | Target | Action if Failed |
|-------|--------|------------------|
| PE index join rate | >99% patients matched | Review Module 1 alignment |
| Timestamp parsing | >99% success | Log failed records |
| Physiological ranges | 0 impossible values | Remove and log |
| Duplicate rate | <1% exact duplicates | Deduplicate |

### Physiological Range Validation

| Vital | Remove (Impossible) | Flag (Abnormal) |
|-------|---------------------|-----------------|
| HR | <20 or >300 | <60 or >100 |
| SBP | <40 or >300 | <90 or >180 |
| DBP | <20 or >200 | <60 or >110 |
| MAP | <30 or >200 | <65 or >110 |
| RR | <4 or >60 | <12 or >24 |
| SpO2 | <50 or >100 | <92 |
| Temp | <30 or >45 | <36 or >38.5 |

### Layer 2 Validation

| Check | Target | Action if Failed |
|-------|--------|------------------|
| Hourly coverage (any vital) | >50% of patient-hours | Flag low-coverage patients |
| Forward-fill limits | Respected per vital | Review imputation logic |
| Mask accuracy | Matches observation pattern | Debug mask generation |
| Time-delta values | Realistic gaps (1-12h typical) | Review calculation |

### Cross-Vital Consistency

- SBP > DBP always (flag violations)
- MAP ≈ DBP + (SBP-DBP)/3 when calculated (±5 tolerance)
- Low SpO2 correlates with high RR

---

## Implementation Phases

### Phase 1: Layers 1-2 (Core Data)

| Submodule | Description | Outputs |
|-----------|-------------|---------|
| 3.4 | Layer 1 Builder | `canonical_vitals.parquet` |
| 3.5 | Layer 2 Builder | `hourly_grid.parquet`, `hourly_tensors.h5` |
| 3.9a | Phase 1 Validation | `validation_report_phase1.json` |

**Enables:** GRU-D, LSTM, basic XGBoost, GBTM on raw trajectories

### Phase 2: Layer 3 (Features)

| Submodule | Description | Outputs |
|-----------|-------------|---------|
| 3.6 | Layer 3 Builder | `engineered_features.parquet` |
| 3.9b | Phase 2 Validation | `validation_report_phase2.json` |

**Enables:** Rich XGBoost, random forests, clinical decision rules, interpretable analysis

### Phase 3: Layers 4-5 (Advanced)

| Submodule | Description | Outputs |
|-----------|-------------|---------|
| 3.7 | Layer 4 Builder | `embeddings.h5` |
| 3.8 | Layer 5 Builder | `world_model_states.h5` |
| 3.9c | Phase 3 Validation | `validation_report_phase3.json` |

**Enables:** World models, TDA, trajectory phenotyping, counterfactual simulation

### R Exports (Generated on demand)

- `vitals_lcmm_long.csv` - for lcmm/GBTM
- `vitals_gbtm_wide.csv` - wide format
- `vital_features.csv` - Layer 3 features

---

## Method-Specific Data Access

| Method | Layers Used | Format | Key Features |
|--------|-------------|--------|--------------|
| GBTM / lcmm | 2 | Long CSV | One row per patient-time-vital |
| GRU-D | 2 | HDF5 triple tensor | Values + masks + time-deltas |
| LSTM / GRU | 2 (imputed) | HDF5 tensor | Forward-filled values |
| Transformer | 2 | HDF5 tensor | Positional encoding; attention masks |
| XGBoost / RF | 2, 3 | Wide tabular | Hourly summaries + derived features |
| World Models | 4, 5 | HDF5 tensors | State vectors with embeddings |
| TDA / Mapper | 4 | Point cloud | FPC scores or autoencoder latents |
| Cox / Logistic | 3 | Tabular | Summary statistics + threshold features |

---

## Summary

This 5-layer architecture transforms raw vital sign extractions into multiple synchronized representations:

1. **Layer 1 (Canonical Records):** Granular source of truth with PE-relative timestamps
2. **Layer 2 (Hourly Grid):** MIMIC-Extract standard with triple-component missing data encoding
3. **Layer 3 (Engineered Features):** Clinically interpretable trajectory dynamics
4. **Layer 4 (Embeddings):** Dense representations for advanced ML
5. **Layer 5 (World Model States):** Purpose-built state vectors for dynamics learning

Storage uses Parquet (tabular data) + HDF5 (tensors/embeddings) for simplicity and portability.

Phased implementation delivers usable data quickly (Phase 1) while building toward the complete architecture (Phases 2-3).

---

**Document Version:** 1.0
**Approved:** 2025-12-08
