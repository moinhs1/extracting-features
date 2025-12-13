# Module 3: Comprehensive Vitals Extraction & Processing

**Version:** 3.3
**Status:** ALL PHASES COMPLETE
**Last Updated:** 2025-12-13
**Dependencies:** Module 1 (patient_timelines.pkl)

---

## Current Implementation Status

### Phase 1: Extractors + Layers 1-2 ✅ COMPLETE

| Component | Status | Tests | Description |
|-----------|--------|-------|-------------|
| **3.1 Phy Extractor** | ✅ COMPLETE | 39 | Structured vitals from Phy.txt |
| **3.2 Hnp Extractor** | ✅ COMPLETE | 70 | NLP extraction from H&P notes (unified + supplemental) |
| **3.3 Prg Extractor** | ✅ COMPLETE | 34 | NLP extraction from Progress notes (unified + supplemental) |
| **Unified Extractor** | ✅ COMPLETE | 54 | Core patterns + extraction logic |
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

### Phase 3: Layers 4-5 Embeddings & World States ✅ COMPLETE

| Component | Status | Tests | Description |
|-----------|--------|-------|-------------|
| **3.7 FPCA Builder** | ✅ COMPLETE | 4 | 10 components × 7 vitals, 76-92% variance explained |
| **3.8 LSTM-VAE** | ✅ COMPLETE | 6 | 32-dim latent, MSE=0.0094, 26 epochs |
| **3.9 DTW Clustering** | ✅ COMPLETE | 1 | Silhouette: HR 0.33, MAP 0.26 (acute window) |
| **3.10 HDBSCAN Clustering** | ✅ COMPLETE | 3 | 102-dim combined (FPCA+VAE), 2 clusters |
| **3.11 Layer 4 Builder** | ✅ COMPLETE | - | Orchestrates all embedding components |
| **3.12 Layer 5 Builder** | ✅ COMPLETE | - | 100-dim world states (7689×745×100) |

**Total Tests:** 413 passing

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
│ LAYER 4: Embeddings & Clustering (3.7-3.10)         ✅ COMPLETE │
│ • FPCA: 10 components × 7 vitals (scikit-fda)                   │
│ • LSTM-VAE: 32-dim latent representations (PyTorch)             │
│ • DTW clustering (validation) + HDBSCAN (primary)               │
│ Output: fpca_scores.parquet (7,689 × 77 features)               │
│ Output: vae_latents.h5 (7,689 × 32 dims)                        │
│ Output: clusters_dtw.parquet, clusters_embedding.parquet        │
└─────────────────────────┬───────────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 5: World Model States (3.11-3.12)             ✅ COMPLETE │
│ • 100 dimension state vectors per patient-hour                  │
│ • Combines: vitals, masks, FPCA, VAE, trends, clusters          │
│ • Reserved slots for treatment package (meds, labs, imaging)    │
│ Output: world_states.h5 (7,689 × 745 × 100) = 2.3 GB            │
│ Output: state_schema.json                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Unified Extraction Architecture (v3.3)

The vital extraction system uses a unified pattern library with 3-tier confidence scoring:

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNIFIED PATTERNS                             │
│  unified_patterns.py: 91 patterns across 8 vital types          │
│  • Standard tier (0.90-1.0): Explicit label + unit              │
│  • Optimized tier (0.80-0.90): Label or strong context          │
│  • Specialized tier (0.65-0.80): Contextual/bare patterns       │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED EXTRACTOR                             │
│  unified_extractor.py: Core extraction with validation          │
│  • Negation detection (8 patterns)                              │
│  • Skip section filtering (allergies, meds, history)            │
│  • Physiological validation + abnormal flagging                 │
│  • Position-based deduplication                                 │
│  • Temperature F→C normalization                                │
└─────────────────────────────┬───────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              THIN WRAPPERS (Source-Specific)                     │
│  hnp_extractor.py: Section identification, timestamp handling   │
│  prg_extractor.py: Checkpoint support, temp method extraction   │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Coverage

| Vital Type | Pattern Count | Confidence Range |
|------------|---------------|------------------|
| Heart Rate (HR) | 16 | 0.68 - 0.95 |
| Blood Pressure (BP) | 16 | 0.68 - 0.95 |
| Respiratory Rate (RR) | 12 | 0.68 - 0.95 |
| SpO2 | 11 | 0.68 - 0.95 |
| Temperature | 10 | 0.68 - 0.95 |
| O2 Flow Rate | 9 | 0.68 - 0.95 |
| O2 Device | 10 | 0.72 - 0.95 |
| BMI | 7 | 0.72 - 0.95 |

---

## Quick Start

### Run Full Pipeline (All Phases)

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
python processing/layer2_builder.py   # ~2 min

# 4. Run Layer 3 builder (feature engineering)
python processing/layer3_builder.py   # ~15-30 min (parallel, 22 cores)

# 5. Run Layer 4 components (embeddings)
cd processing/layer4
python fpca_builder.py                # ~10 sec
python vae_trainer.py                 # ~2-3 min (GPU) / ~10 min (CPU)
python clustering_dtw.py              # ~40 min (DTW is slow)
python clustering_embedding.py        # ~10 sec

# 6. Run Layer 5 builder (world states)
cd ..
python layer5_builder.py              # ~20 sec

# 7. Run tests
cd /home/moin/TDA_11_25
pytest module_3_vitals_processing/tests/ -v
```

### Expected Outputs

| Stage | Output | Size |
|-------|--------|------|
| Extraction | `outputs/discovery/phy_vitals_raw.parquet` | 67 MB |
| Extraction | `outputs/discovery/hnp_vitals_raw.parquet` | ~15 MB |
| Extraction | `outputs/discovery/hnp_supplemental.parquet` | ~20 MB |
| Extraction | `outputs/discovery/prg_vitals_raw.parquet` | 215 MB |
| Extraction | `outputs/discovery/prg_supplemental.parquet` | ~50 MB |
| Layer 1 | `outputs/layer1/canonical_vitals.parquet` | 30 MB |
| Layer 2 | `outputs/layer2/hourly_grid.parquet` | 35 MB |
| Layer 2 | `outputs/layer2/hourly_tensors.h5` | 24 MB |
| Layer 3 | `outputs/layer3/timeseries_features.parquet` | 988 MB |
| Layer 3 | `outputs/layer3/summary_features.parquet` | 96 MB |
| Layer 4 | `outputs/layer4/fpca_scores.parquet` | 4.7 MB |
| Layer 4 | `outputs/layer4/fpca_components.h5` | 447 KB |
| Layer 4 | `outputs/layer4/vae_latents.h5` | 2.0 MB |
| Layer 4 | `outputs/layer4/vae_model.pt` | 3.4 MB |
| Layer 4 | `outputs/layer4/clusters_dtw.parquet` | 150 KB |
| Layer 4 | `outputs/layer4/clusters_embedding.parquet` | 73 KB |
| Layer 5 | `outputs/layer5/world_states.h5` | 109 MB |
| Layer 5 | `outputs/layer5/state_schema.json` | 2.1 KB |

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
After Layer 4 (embeddings)        7,689        7,689
  - FPCA scores                   × 77 features (70 FPC + 7 obs_pct)
  - VAE latents                   × 32 dimensions
  - Cluster assignments           × 2 methods (DTW, HDBSCAN)
After Layer 5 (world states)  5,728,305        7,689
  - State vectors                 × 100 dimensions per hour
```

---

## Layer 4 Details

### FPCA Scores (`fpca_scores.parquet`)

Functional Principal Component Analysis extracts interpretable trajectory shape features.

| Vital | Components | Explained Variance |
|-------|------------|-------------------|
| HR | fpc1-fpc10 | 76.8% |
| SBP | fpc1-fpc10 | 90.6% |
| DBP | fpc1-fpc10 | 92.0% |
| MAP | fpc1-fpc10 | 88.0% |
| RR | fpc1-fpc10 | 82.2% |
| SPO2 | fpc1-fpc10 | 89.4% |
| TEMP | fpc1-fpc10 | 76.5% |

**Interpretation:**
- FPC1: Overall level/mean
- FPC2: Slope/trend
- FPC3+: Oscillation patterns

### VAE Latents (`vae_latents.h5`)

LSTM Variational Autoencoder learns nonlinear temporal embeddings.

```
/mu           (7689, 32)       # Patient-level mean embedding
/logvar       (7689, 32)       # Patient-level variance
/recon_error  (7689,)          # Per-patient reconstruction quality
/patient_index (7689,)         # EMPI mapping
```

**Training Results:**
- Epochs: 26 (early stopping)
- Best validation loss: 0.0094
- Mean reconstruction error: 0.0113

### DTW Clusters (`clusters_dtw.parquet`)

Dynamic Time Warping captures shape similarity regardless of time alignment.

| Vital/Window | Silhouette Score | Interpretation |
|--------------|------------------|----------------|
| HR acute (0-24h) | 0.331 | Good separation |
| HR early (24-72h) | 0.040 | Low separation |
| MAP acute (0-24h) | 0.257 | Decent separation |
| MAP early (24-72h) | 0.031 | Low separation |

### HDBSCAN Clusters (`clusters_embedding.parquet`)

Hierarchical density-based clustering on combined FPCA+VAE embeddings.

| Column | Description |
|--------|-------------|
| EMPI | Patient identifier |
| cluster_id | Cluster assignment (-1 = outlier) |
| cluster_prob | Soft cluster probability |
| is_outlier | Binary outlier flag |
| nearest_cluster | Nearest cluster for outliers |

**Results:** 2 clusters, 92% outliers (may need parameter tuning)

### Load Layer 4 in Python

```python
import pandas as pd
import h5py

# FPCA scores
fpca = pd.read_parquet('outputs/layer4/fpca_scores.parquet')
hr_fpc1 = fpca['HR_fpc1']  # First principal component for HR

# VAE latents
with h5py.File('outputs/layer4/vae_latents.h5', 'r') as f:
    mu = f['mu'][:]  # (7689, 32)
    patient_ids = f['patient_index'][:].astype(str)

# Clusters
clusters = pd.read_parquet('outputs/layer4/clusters_embedding.parquet')
phenotype = clusters['cluster_id']
```

---

## Layer 5 Details

### World States (`world_states.h5`)

100-dimensional state vectors for downstream ML tasks.

```
/states         (7689, 745, 100)  float32  # State vectors
/patient_index  (7689,)           str      # EMPI mapping
```

### State Schema (`state_schema.json`)

| Block | Dims | Range | Source | Description |
|-------|------|-------|--------|-------------|
| raw_vitals | 7 | 0-6 | Layer 2 | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| vital_masks | 7 | 7-13 | Layer 2 | Observation indicators |
| time_deltas | 7 | 14-20 | Layer 2 | Hours since last observation |
| temporal_position | 2 | 21-22 | Computed | hours_from_pe, normalized_time |
| fpca_top3 | 21 | 23-43 | Layer 4 | Top 3 FPCs per vital |
| vae_latent | 32 | 44-75 | Layer 4 | 32-dim VAE embedding |
| trend_slopes | 7 | 76-82 | Layer 3 | 6h slopes per vital |
| variability | 7 | 83-89 | Layer 3 | RMSSD per vital |
| cluster_context | 5 | 90-94 | Layer 4 | One-hot cluster + outlier |
| treatment_reserved | 5 | 95-99 | Future | For meds/labs/imaging |

### Load World States in Python

```python
import h5py
import json

# Load states
with h5py.File('outputs/layer5/world_states.h5', 'r') as f:
    states = f['states'][:]  # (7689, 745, 100)
    patients = f['patient_index'][:].astype(str)

# Load schema
with open('outputs/layer5/state_schema.json') as f:
    schema = json.load(f)

# Access specific blocks
raw_vitals = states[:, :, 0:7]      # Current vital values
vae_latent = states[:, :, 44:76]    # VAE embeddings
cluster_onehot = states[:, :, 90:95] # Cluster assignment
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

---

## Physiological Ranges

| Vital | Valid Range | Abnormal Thresholds | Notes |
|-------|-------------|---------------------|-------|
| HR | 25-220 bpm | <60 or ≥100 | Allows profound bradycardia |
| SBP | 50-260 mmHg | <90 or ≥180 | |
| DBP | 25-150 mmHg | <60 or ≥110 | |
| MAP | 30-200 mmHg | <65 or >110 | |
| RR | 6-50 /min | <12 or >24 | |
| SpO2 | 55-100% | <92% | <55% extremely rare in documented vitals |
| Temp | 33.5-42.5°C | <36 or >38.5°C | All normalized to Celsius |
| O2 Flow | 0.25-80 L/min | - | Pediatric to Vapotherm |
| BMI | 12-70 kg/m² | - | |

---

## File Structure

```
module_3_vitals_processing/
├── config/
│   └── vitals_config.py              # Paths and constants
├── extractors/
│   ├── unified_patterns.py           # ✅ NEW: Unified 3-tier pattern library
│   ├── unified_extractor.py          # ✅ NEW: Core extraction logic
│   ├── phy_extractor.py              # ✅ 3.1 Structured (265 lines)
│   ├── hnp_extractor.py              # ✅ 3.2 H&P NLP (thin wrapper)
│   ├── hnp_patterns.py               # Hnp-specific patterns
│   ├── prg_extractor.py              # ✅ 3.3 Progress NLP (thin wrapper)
│   ├── prg_patterns.py               # Prg-specific patterns
│   └── __init__.py
├── processing/
│   ├── __init__.py
│   ├── unit_converter.py             # ✅ Temperature F→C
│   ├── qc_filters.py                 # ✅ Physiological validation
│   ├── temporal_aligner.py           # ✅ PE-relative time
│   ├── layer1_builder.py             # ✅ 3.4 Canonical records
│   ├── layer2_builder.py             # ✅ 3.5 Hourly grid + tensors
│   ├── layer3_builder.py             # ✅ 3.6 Feature engineering
│   ├── layer4_builder.py             # ✅ 3.7 Layer 4 orchestrator
│   ├── layer5_builder.py             # ✅ 3.8 World states
│   ├── layer3/                       # Layer 3 feature calculators
│   │   ├── __init__.py
│   │   ├── composite_vitals.py
│   │   ├── rolling_stats.py
│   │   ├── trend_features.py
│   │   ├── variability_features.py
│   │   ├── threshold_features.py
│   │   ├── data_density.py
│   │   └── summary_aggregator.py
│   └── layer4/                       # Layer 4 embedding components
│       ├── __init__.py
│       ├── fpca_builder.py           # ✅ FPCA (scikit-fda)
│       ├── vae_model.py              # ✅ LSTM-VAE architecture
│       ├── vae_trainer.py            # ✅ VAE training + inference
│       ├── clustering_dtw.py         # ✅ DTW clustering (tslearn)
│       └── clustering_embedding.py   # ✅ HDBSCAN clustering
├── tests/
│   ├── test_unified_patterns.py      # ✅ 21 tests (HR, BP, RR, SpO2, Temp)
│   ├── test_unified_extractor.py     # ✅ 25 tests (extraction + validation)
│   ├── test_supplemental_patterns.py # ✅ 8 tests (O2 flow, device, BMI)
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
│   ├── test_layer5_builder.py        # ✅ 5 tests
│   ├── test_layer3/                  # Layer 3 feature tests
│   │   ├── test_composite_vitals.py  # ✅ 11 tests
│   │   ├── test_rolling_stats.py     # ✅ 9 tests
│   │   ├── test_trend_features.py    # ✅ 12 tests
│   │   ├── test_variability_features.py  # ✅ 9 tests
│   │   ├── test_threshold_features.py    # ✅ 8 tests
│   │   ├── test_data_density.py      # ✅ 6 tests
│   │   └── test_summary_aggregator.py    # ✅ 9 tests
│   └── test_layer4/                  # Layer 4 embedding tests
│       ├── __init__.py
│       ├── test_fpca_builder.py      # ✅ 4 tests
│       ├── test_vae_model.py         # ✅ 6 tests
│       └── test_clustering.py        # ✅ 4 tests
├── outputs/
│   ├── discovery/                    # Raw extractions
│   │   ├── phy_vitals_raw.parquet
│   │   ├── hnp_vitals_raw.parquet
│   │   ├── hnp_supplemental.parquet  # O2 flow, O2 device, BMI
│   │   ├── prg_vitals_raw.parquet
│   │   └── prg_supplemental.parquet  # O2 flow, O2 device, BMI
│   ├── layer1/                       # Canonical records
│   │   └── canonical_vitals.parquet
│   ├── layer2/                       # Hourly grid + tensors
│   │   ├── hourly_grid.parquet
│   │   └── hourly_tensors.h5
│   ├── layer3/                       # Feature engineering
│   │   ├── timeseries_features.parquet
│   │   └── summary_features.parquet
│   ├── layer4/                       # Embeddings
│   │   ├── fpca_scores.parquet
│   │   ├── fpca_components.h5
│   │   ├── fpca_models.pkl
│   │   ├── vae_latents.h5
│   │   ├── vae_model.pt
│   │   ├── vae_training_history.json
│   │   ├── clusters_dtw.parquet
│   │   └── clusters_embedding.parquet
│   └── layer5/                       # World states
│       ├── world_states.h5
│       └── state_schema.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── SUBMODULES_QUICK_REFERENCE.md
│   ├── IMPLEMENTATION_ROADMAP.md
│   └── plans/
│       ├── 2025-12-08-vitals-5-layer-architecture-design.md
│       ├── 2025-12-08-phase1-layer1-layer2-implementation.md
│       ├── 2025-12-09-layer3-feature-engineering-design.md
│       └── 2025-12-10-phase3-layer4-layer5-design.md
└── README.md                         # This file
```

---

## Testing

```bash
cd /home/moin/TDA_11_25
export PYTHONPATH=module_3_vitals_processing:$PYTHONPATH

# Run all tests
pytest module_3_vitals_processing/tests/ -v

# Run specific layers
pytest module_3_vitals_processing/tests/test_layer3/ -v
pytest module_3_vitals_processing/tests/test_layer4/ -v

# Run with coverage
pytest module_3_vitals_processing/tests/ --cov=module_3_vitals_processing
```

**Current Test Results:** 413 tests passing

---

## Performance

### Full Pipeline Runtime

| Layer | Time | Notes |
|-------|------|-------|
| Extractors | ~3-4 hours | One-time extraction |
| Layer 1 | ~3 min | Merge + validate |
| Layer 2 | ~2 min | Grid + imputation |
| Layer 3 | ~15-20 min | 22 parallel workers |
| Layer 4 | ~45 min | DTW is slowest (~40 min) |
| Layer 5 | ~20 sec | Assembly only |
| **Total** | **~4-5 hours** | First run (extractors) |
| **Incremental** | **~1 hour** | Layers 1-5 only |

---

## Dependencies

### Core
```
pandas>=2.0
numpy>=1.24
h5py>=3.8
pyarrow>=12.0
```

### Layer 4
```
scikit-fda>=0.9      # FPCA
torch>=2.0           # LSTM-VAE
tslearn>=0.6         # DTW clustering
scikit-learn>=1.3    # HDBSCAN (built-in)
```

### Install All
```bash
pip install pandas numpy h5py pyarrow scikit-fda torch tslearn
```

---

## Future Extensions

When treatment package modules are ready, Layer 5 reserved slots will be populated:

| Slot | Dims | Future Source |
|------|------|---------------|
| medications | 95-96 | Module 4 (RxNorm mapped) |
| labs | 97-98 | Lab values module |
| imaging | 99 | PE severity scores |

State schema version will increment when treatment slots are populated.

---

## Changelog

### Version 3.3 (2025-12-13)
- **Unified Extraction Architecture**: Major refactor of vital extraction
  - New `unified_patterns.py`: 3-tier confidence scoring (standard/optimized/specialized)
  - New `unified_extractor.py`: Core extraction with validation, negation detection, skip sections
  - Pattern coverage: 16 HR, 16 BP, 12 RR, 11 SpO2, 10 Temp, 9 O2 flow, 10 O2 device, 7 BMI
  - `hnp_extractor.py` and `prg_extractor.py` refactored as thin wrappers
- **Supplemental Vitals Extraction**: Both HNP and PRG extractors now extract:
  - O2 Flow Rate (L/min)
  - O2 Device (nasal cannula, HFNC, room air, etc.)
  - BMI
  - Output: `*_supplemental.parquet` files
- **Updated Validation Ranges** (per clinical review):
  - HR: 25-220 (allows profound bradycardia in complete heart block/hypothermia)
  - SpO2: 55-100 (SpO2 < 55% extremely rare in documented vitals)
  - O2 Flow: 0.25-80L (pediatric/neonatal to Vapotherm)
- **Temperature Normalization**: All temps now normalized to Celsius
- **54 new unified extraction tests**
- **413 total tests** passing

### Version 3.2 (2025-12-11)
- **Phase 3 COMPLETE**: Layers 4-5 fully implemented
  - FPCA: 76-92% variance explained, 10 components per vital
  - LSTM-VAE: 32-dim latents, MSE=0.0094, 26 epochs
  - DTW clustering: Silhouette 0.33 (HR acute), 0.26 (MAP acute)
  - HDBSCAN: 102-dim combined embeddings, 2 clusters
  - World states: 7,689 × 745 × 100 tensor (2.3 GB)
- **14 new tests** for Layer 4 components
- **337 total tests** passing
- sklearn HDBSCAN used (replaces standalone hdbscan package)

### Version 3.1 (2025-12-10)
- **Phase 3 DESIGN COMPLETE**: Layers 4-5 architecture finalized
  - Design doc: `docs/plans/2025-12-10-phase3-layer4-layer5-design.md`

### Version 3.0 (2025-12-10)
- **Phase 2 COMPLETE**: Layer 3 feature engineering
  - Timeseries features: 5.7M rows × 315 columns
  - Summary features: 7,689 patients × 4,426 features
- **71 new tests** for Layer 3 components
- **323 total tests** passing

### Version 2.0 (2025-12-09)
- **Phase 1 COMPLETE**: Layers 1-2 implemented and tested
  - Layer 1: 39M canonical records from 3 sources
  - Layer 2: 5.7M hourly grid with 4-tier imputation
- **252 tests** passing

### Version 1.x (2025-11-25 to 2025-12-08)
- Extractors implemented: PHY, HNP, PRG

---

**Status:** ✅ ALL PHASES COMPLETE
**Module Ready For:** Downstream ML, outcome prediction, treatment optimization
