# Module 3: Comprehensive Vitals Extraction & Processing

**Version:** 3.0
**Status:** Phase 1-2 COMPLETE (Extractors + Layers 1-3) | Phase 3 Pending
**Last Updated:** 2025-12-10
**Dependencies:** Module 1 (patient_timelines.pkl)

---

## Current Implementation Status

### Phase 1: Extractors + Layers 1-2 ✅ COMPLETE

| Component | Status | Tests | Description |
|-----------|--------|-------|-------------|
| **3.1 Phy Extractor** | ✅ COMPLETE | 39 | Structured vitals from Phy.txt |
| **3.2 Hnp Extractor** | ✅ COMPLETE | 74 | NLP extraction from H&P notes |
| **3.3 Prg Extractor** | ✅ COMPLETE | 61 | NLP extraction from Progress notes |
| **3.4 Layer 1 Builder** | ✅ COMPLETE | 17 | Canonical records (PE-aligned, merged, validated) |
| **3.5 Layer 2 Builder** | ✅ COMPLETE | 17 | Hourly grid + HDF5 tensors with imputation |
| **Processing Helpers** | ✅ COMPLETE | 44 | Unit converter, QC filters, temporal aligner |

### Phase 2: Layer 3 Feature Engineering ✅ COMPLETE

| Component | Status | Tests | Description |
|-----------|--------|-------|-------------|
| **3.6 Layer 3 Builder** | ✅ COMPLETE | 71 | Feature engineering with parallel processing |
| **Composite Vitals** | ✅ COMPLETE | 11 | Shock index, pulse pressure |
| **Rolling Stats** | ✅ COMPLETE | 9 | 6h/12h/24h windows: mean, std, cv, min, max, range |
| **Trend Features** | ✅ COMPLETE | 12 | Slope, R², direction per window |
| **Variability** | ✅ COMPLETE | 9 | RMSSD, successive variance |
| **Threshold Features** | ✅ COMPLETE | 8 | Cumulative hours, time-to-first |
| **Data Density** | ✅ COMPLETE | 6 | Observation rates per vital |
| **Summary Aggregator** | ✅ COMPLETE | 9 | Per-patient summary across clinical windows |

**Total Tests:** 323 passing

### Phase 3: Pending

| Component | Status | Description |
|-----------|--------|-------------|
| **3.7 Layer 4 Builder** | ⏳ Pending | Embeddings (FPCA, autoencoder latents) |
| **3.8 Layer 5 Builder** | ⏳ Pending | World model states |
| **3.9 Validation** | ⏳ Pending | 4-tier validation framework |
| **3.10 Orchestrator** | ⏳ Pending | Full pipeline orchestration |

---

## 5-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW EXTRACTIONS (3.1-3.3)                    │
│  phy_vitals_raw.parquet  hnp_vitals_raw.parquet  prg_vitals_raw │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1: Canonical Records (3.4)                    ✅ COMPLETE │
│ • Merge PHY/HNP/PRG sources → unified schema                    │
│ • PE-relative timestamps (hours_from_pe)                        │
│ • Physiological range validation                                │
│ • Abnormal flagging                                             │
│ • Calculate MAP from SBP/DBP pairs                              │
│ Output: canonical_vitals.parquet (39M records, 7,689 patients)  │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 2: Hourly Aggregated Grid (3.5)               ✅ COMPLETE │
│ • Aggregate to hourly bins (-24h to +720h = 745 hours)          │
│ • Full grid: patient × hour × vital                             │
│ • Three-tier imputation (observed → ffill → patient → cohort)   │
│ • HDF5 tensor generation with time deltas                       │
│ Output: hourly_grid.parquet (5.7M rows)                         │
│ Output: hourly_tensors.h5 (7,689 × 745 × 7)                     │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 3: Feature Engineering (3.6)                  ✅ COMPLETE │
│ • Composite vitals (shock_index, pulse_pressure)                │
│ • Rolling window statistics (6h, 12h, 24h)                      │
│ • Trend features (slope, R², direction)                         │
│ • Variability features (RMSSD, successive variance)             │
│ • Threshold-based features (hours_tachycardia, etc.)            │
│ • Data density features (observation rates)                     │
│ • Summary aggregation across clinical windows                   │
│ Output: timeseries_features.parquet (5.7M × 315 features)       │
│ Output: summary_features.parquet (7,689 × 4,426 features)       │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 4: Embeddings (3.7)                           ⏳ PENDING  │
│ • Functional PCA trajectories                                   │
│ • Autoencoder latent representations                            │
│ • Trajectory clustering                                         │
│ Output: embeddings.h5                                           │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 5: World Model States (3.8)                   ⏳ PENDING  │
│ • Learned dynamics representations                              │
│ • State transition modeling                                     │
│ Output: world_states.h5                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Run Full Pipeline (Phases 1-2)

```bash
cd /home/moin/TDA_11_25

# Set PYTHONPATH
export PYTHONPATH=module_3_vitals_processing:$PYTHONPATH

# 1. Run extractors (if not already done)
python -m extractors.phy_extractor    # ~30 min
python -m extractors.hnp_extractor    # ~1 hour
python -m extractors.prg_extractor    # ~2-4 hours

# 2. Run Layer 1 builder (canonical records)
python processing/layer1_builder.py   # ~3 min

# 3. Run Layer 2 builder (hourly grid + tensors)
python processing/layer2_builder.py   # ~2 min (optimized)

# 4. Run Layer 3 builder (feature engineering)
python processing/layer3_builder.py   # ~15-30 min (parallel, 22 cores)

# 5. Run tests
pytest tests/ -v
```

### Expected Outputs

| Stage | Output | Size |
|-------|--------|------|
| Extraction | `outputs/discovery/phy_vitals_raw.parquet` | 67 MB |
| Extraction | `outputs/discovery/hnp_vitals_raw.parquet` | 9.4 MB |
| Extraction | `outputs/discovery/prg_vitals_raw.parquet` | 215 MB |
| Layer 1 | `outputs/layer1/canonical_vitals.parquet` | 30 MB |
| Layer 2 | `outputs/layer2/hourly_grid.parquet` | 35 MB |
| Layer 2 | `outputs/layer2/hourly_tensors.h5` | 24 MB |
| Layer 3 | `outputs/layer3/timeseries_features.parquet` | 988 MB |
| Layer 3 | `outputs/layer3/summary_features.parquet` | 96 MB |

---

## Data Flow Summary

```
Data Sources                    Records        Patients
─────────────────────────────────────────────────────────
PHY (structured)                160,308
HNP (H&P notes NLP)             283,432
PRG (Progress notes NLP)     38,601,927
─────────────────────────────────────────────────────────
TOTAL RAW                    39,045,667        7,689

After Layer 1 (canonical)    39,045,667        7,689
After Layer 2 (hourly grid)   5,728,305        7,689
After Layer 3 (features)      5,728,305        7,689
  - Timeseries features         × 315 columns
  - Summary features            × 4,426 columns per patient
```

---

## Layer 1 Schema (`canonical_vitals.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| timestamp | datetime | Measurement time |
| hours_from_pe | float | Hours relative to PE index |
| vital_type | str | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| value | float | Measurement value |
| units | str | bpm, mmHg, %, °C, breaths/min |
| source | str | 'phy', 'hnp', 'prg' |
| source_detail | str | encounter_type or extraction_context |
| confidence | float | 1.0 for phy, 0.6-1.0 for nlp |
| is_calculated | bool | True if MAP calculated from SBP/DBP |
| is_flagged_abnormal | bool | Outside normal clinical range |
| report_number | str | Source report identifier |

---

## Layer 2 HDF5 Structure (`hourly_tensors.h5`)

```
/values           (7689, 745, 7)  float32  # Vital values
/masks            (7689, 745, 7)  int8     # 1=observed, 0=imputed
/time_deltas      (7689, 745, 7)  float32  # Hours since last observation
/imputation_tier  (7689, 745, 7)  int8     # 1-4 tier indicators
/patient_index    (7689,)         str      # EMPI mapping
/vital_index      (7,)            str      # ['HR','SBP','DBP','MAP','RR','SPO2','TEMP']
/hour_index       (745,)          int      # [-24 to 720]
```

### Load Tensors in Python

```python
import h5py
import numpy as np

with h5py.File('outputs/layer2/hourly_tensors.h5', 'r') as f:
    values = f['values'][:]          # (7689, 745, 7)
    masks = f['masks'][:]            # 1=observed, 0=imputed
    time_deltas = f['time_deltas'][:]
    patients = f['patient_index'][:].astype(str)
    vitals = f['vital_index'][:].astype(str)
    hours = f['hour_index'][:]

    # Get specific patient
    patient_idx = np.where(patients == '100000272')[0][0]
    patient_hr = values[patient_idx, :, 0]  # HR trajectory
```

---

## Layer 3 Features

### Timeseries Features (`timeseries_features.parquet`)

| Category | Count | Pattern | Description |
|----------|-------|---------|-------------|
| **Composite Vitals** | 2 | `shock_index`, `pulse_pressure` | Derived from HR/SBP/DBP |
| **Rolling Stats** | 162 | `{vital}_roll{6,12,24}h_{mean,std,cv,min,max,range}` | 9 vitals × 3 windows × 6 stats |
| **Trend Features** | 81 | `{vital}_slope{6,12,24}h`, `_r2`, `_direction` | Slope, R², direction per window |
| **Variability** | 18 | `{vital}_rmssd`, `{vital}_successive_var` | 9 vitals × 2 metrics |
| **Threshold Hours** | 10 | `hours_{tachycardia,hypotension,...}` | Cumulative hours at threshold |
| **Time to First** | 5 | `time_to_first_{condition}` | Hours until first crossing |
| **Observation Density** | 19 | `{vital}_obs_pct`, `{vital}_obs_count` | Data quality indicators |
| **Masks** | 9 | `mask_{vital}` | 1=Tier1-2 (observed), 0=Tier3-4 |

**Total:** 315 columns × 5,728,305 rows

### Summary Features (`summary_features.parquet`)

Aggregates timeseries features across **5 clinical windows**:

| Window | Hours | Clinical Meaning |
|--------|-------|------------------|
| `pre` | -24 to 0 | Pre-PE baseline |
| `acute` | 0 to 24 | Acute phase |
| `early` | 24 to 72 | Early treatment response |
| `stab` | 72 to 168 | Stabilization (days 3-7) |
| `recov` | 168 to 720 | Recovery (days 7-30) |

Each feature gets `{feature}_{window}_{mean,max,min}` → **4,426 features per patient**

### Load Layer 3 Features

```python
import pandas as pd

# Timeseries features (one row per patient-hour)
ts = pd.read_parquet('outputs/layer3/timeseries_features.parquet')

# Summary features (one row per patient)
summary = pd.read_parquet('outputs/layer3/summary_features.parquet')

# Example: Get rolling HR mean for acute phase
hr_acute = summary['HR_roll6h_mean_acute_mean']
```

---

## Physiological Ranges

| Vital | Valid Range | Abnormal Thresholds |
|-------|-------------|---------------------|
| HR | 20-300 bpm | <60 or >100 |
| SBP | 40-300 mmHg | <90 or >180 |
| DBP | 20-200 mmHg | <60 or >110 |
| MAP | 30-200 mmHg | <65 or >110 |
| RR | 4-60 /min | <12 or >24 |
| SpO2 | 50-100% | <92% |
| Temp | 30-45°C | <36 or >38.5°C |

---

## Imputation Strategy

| Tier | Condition | Fill Value | Limits |
|------|-----------|------------|--------|
| 1 | Observed | Actual value | - |
| 2 | Forward-fill | Last observed | HR/BP/RR: 6h, SpO2: 4h, Temp: 12h |
| 3 | Patient mean | Patient's own mean | When forward-fill exceeded |
| 4 | Cohort mean | Population mean | When patient has zero measurements |

---

## File Structure

```
module_3_vitals_processing/
├── config/
│   └── vitals_config.py              # Paths and constants
├── extractors/
│   ├── phy_extractor.py              # ✅ 3.1 Structured (265 lines)
│   ├── hnp_extractor.py              # ✅ 3.2 H&P NLP (662 lines)
│   ├── hnp_patterns.py               # Hnp regex patterns
│   ├── prg_extractor.py              # ✅ 3.3 Progress NLP (542 lines)
│   ├── prg_patterns.py               # Prg regex patterns
│   └── __init__.py
├── processing/
│   ├── __init__.py
│   ├── unit_converter.py             # ✅ Temperature F→C (45 lines)
│   ├── qc_filters.py                 # ✅ Physiological validation (75 lines)
│   ├── temporal_aligner.py           # ✅ PE-relative time (65 lines)
│   ├── layer1_builder.py             # ✅ 3.4 Canonical records (375 lines)
│   ├── layer2_builder.py             # ✅ 3.5 Hourly grid + tensors (430 lines)
│   ├── layer3_builder.py             # ✅ 3.6 Feature engineering (250 lines)
│   └── layer3/                       # Layer 3 feature calculators
│       ├── __init__.py
│       ├── composite_vitals.py       # Shock index, pulse pressure
│       ├── rolling_stats.py          # Rolling mean, std, cv, min, max, range
│       ├── trend_features.py         # Slope, R², direction
│       ├── variability_features.py   # RMSSD, successive variance
│       ├── threshold_features.py     # Cumulative hours, time-to-first
│       ├── data_density.py           # Observation rates
│       └── summary_aggregator.py     # Per-patient summary
├── tests/
│   ├── test_phy_extractor.py         # ✅ 39 tests
│   ├── test_hnp_extractor.py         # ✅ 70 tests
│   ├── test_hnp_patterns.py          # ✅ 4 tests
│   ├── test_prg_extractor.py         # ✅ 34 tests
│   ├── test_prg_patterns.py          # ✅ 27 tests
│   ├── test_unit_converter.py        # ✅ 8 tests
│   ├── test_qc_filters.py            # ✅ 20 tests
│   ├── test_temporal_aligner.py      # ✅ 16 tests
│   ├── test_layer1_builder.py        # ✅ 17 tests
│   ├── test_layer2_builder.py        # ✅ 17 tests
│   ├── test_layer3_builder.py        # ✅ 7 tests
│   └── test_layer3/                  # Layer 3 feature tests
│       ├── test_composite_vitals.py  # ✅ 11 tests
│       ├── test_rolling_stats.py     # ✅ 9 tests
│       ├── test_trend_features.py    # ✅ 12 tests
│       ├── test_variability_features.py  # ✅ 9 tests
│       ├── test_threshold_features.py    # ✅ 8 tests
│       ├── test_data_density.py      # ✅ 6 tests
│       └── test_summary_aggregator.py    # ✅ 9 tests
├── outputs/
│   ├── discovery/                    # Raw extractions
│   │   ├── phy_vitals_raw.parquet
│   │   ├── hnp_vitals_raw.parquet
│   │   └── prg_vitals_raw.parquet
│   ├── layer1/                       # Canonical records
│   │   └── canonical_vitals.parquet
│   ├── layer2/                       # Hourly grid + tensors
│   │   ├── hourly_grid.parquet
│   │   └── hourly_tensors.h5
│   └── layer3/                       # Feature engineering
│       ├── timeseries_features.parquet
│       └── summary_features.parquet
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SUBMODULES_QUICK_REFERENCE.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   └── plans/
│       ├── 2025-12-08-vitals-5-layer-architecture-design.md
│       └── 2025-12-08-phase1-layer1-layer2-implementation.md
└── README.md                         # This file
```

---

## Testing

```bash
cd /home/moin/TDA_11_25
export PYTHONPATH=module_3_vitals_processing:$PYTHONPATH

# Run all tests
pytest module_3_vitals_processing/tests/ -v

# Run specific test module
pytest module_3_vitals_processing/tests/test_layer3_builder.py -v

# Run Layer 3 feature tests
pytest module_3_vitals_processing/tests/test_layer3/ -v

# Run with coverage
pytest module_3_vitals_processing/tests/ --cov=module_3_vitals_processing
```

**Current Test Results:** 323 tests passing

---

## Performance

### Layer 2 Builder (Optimized)

| Step | Time | Notes |
|------|------|-------|
| Load Layer 1 | ~3s | 30 MB parquet |
| Hourly aggregation | ~5s | Vectorized pandas |
| Create full grid | ~10s | 5.7M rows |
| Imputation | ~36s | 22 parallel workers |
| HDF5 tensor creation | ~15s | Vectorized numpy |
| Time deltas | ~2s | Numba JIT |
| **Total** | **~2 min** | 22 CPU cores |

### Layer 3 Builder (Optimized)

| Step | Time | Notes |
|------|------|-------|
| Load Layer 2 + masks | ~10s | Vectorized numpy indexing |
| Add composites | ~1s | Vectorized |
| Split by patient | ~30s | groupby (was 18 min with filter) |
| Feature calculation | ~14 min | 22 parallel workers |
| Combine + save | ~30s | Parquet compression |
| **Total** | **~15-20 min** | 22 CPU cores |

---

## Next Steps: Phase 3

### Submodule 3.7: Layer 4 Embeddings

Features to implement:
- **Functional PCA**: 10 components per vital capturing trajectory shape
- **Autoencoder latents**: 32-dim representations per timestep
- **Trajectory clustering**: DTW-based phenotyping

Output: `outputs/layer4/embeddings.h5`

### Submodule 3.8: Layer 5 World Model States

Features to implement:
- **~60 dimension state vectors** combining raw vitals, trends, FPC scores
- **State transition modeling** for dynamics learning

Output: `outputs/layer5/world_model_states.h5`

---

## Changelog

### Version 3.0 (2025-12-10)
- **Phase 2 COMPLETE**: Layer 3 feature engineering
  - Timeseries features: 5.7M rows × 315 columns
  - Summary features: 7,689 patients × 4,426 features
  - Composite vitals: shock_index, pulse_pressure
  - Rolling stats: 6h/12h/24h windows (mean, std, cv, min, max, range)
  - Trend features: slope, R², direction
  - Variability: RMSSD, successive variance
  - Threshold features: cumulative hours, time-to-first
  - Data density: observation rates per vital
  - Summary aggregation across 5 clinical windows
  - **Optimized**: Parallel processing (22 cores), vectorized operations
- **71 new tests** for Layer 3 components
- **323 total tests** passing

### Version 2.0 (2025-12-09)
- **Phase 1 COMPLETE**: Layers 1-2 implemented and tested
  - Layer 1: 39M canonical records from 3 sources
  - Layer 2: 5.7M hourly grid with 4-tier imputation
  - HDF5 tensors: (7,689 × 745 × 7) ready for ML
  - Optimized with parallel processing + numba JIT
- **New architecture**: 5-layer design (replaces 10-submodule plan)
- **252 tests** passing

### Version 1.1 (2025-12-08)
- Submodule 3.3 COMPLETE: Prg NLP extractor

### Version 1.0.2 (2025-12-02)
- Submodule 3.2 COMPLETE: Hnp NLP extractor

### Version 1.0.1 (2025-11-25)
- Submodule 3.1 COMPLETE: Phy structured extractor

---

**Status:** ✅ Phase 1-2 COMPLETE | ⏳ Phase 3 Pending
**Next Step:** Implement Submodule 3.7 (Layer 4 Embeddings)
