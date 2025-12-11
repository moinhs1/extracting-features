# PE Trajectory Pipeline - Temporal Data Analysis

A comprehensive clinical data processing pipeline for temporal trajectory analysis of Pulmonary Embolism (PE) patient outcomes. Multi-modal feature extraction from labs, vitals, medications, and clinical notes for GRU-D, GBTM, XGBoost, and World Model analyses.

## Overview

This pipeline processes Electronic Health Record (EHR) data from the Research Patient Data Registry (RPDR) to create rich temporal feature sets for machine learning trajectory models. It extracts and harmonizes laboratory tests, vital signs, medications, diagnoses, and procedures aligned to PE diagnosis time (Time Zero).

**Key Features:**
- 🫀 **PE-Focused Cohort**: 8,713 Gemma PE-positive patients with outcomes
- 🧬 **Lab Harmonization**: Three-tier LOINC system achieving 100% test coverage
- 💓 **Vitals Extraction**: NLP-based extraction from PHY, HNP, PRG notes
- 💊 **Medication Encoding**: 5-layer unified system with RxNorm + embeddings
- ⏰ **Temporal Alignment**: Hourly grid aligned to PE Time Zero
- 🎯 **Multi-Format Export**: GRU-D tensors, GBTM CSVs, XGBoost features

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd TDA_11_1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Workflow (10 patients)

```bash
# Module 1: Core infrastructure
cd module_1_core_infrastructure
python module_01_core_infrastructure.py --test --n=10

# Module 2: Laboratory processing
cd ../module_2_laboratory_processing
python module_02_laboratory_processing.py --phase1 --test --n=10
python module_02_laboratory_processing.py --phase2 --test --n=10
```

### 3. Review Outputs

```bash
# View harmonization map
open outputs/discovery/test_n10_harmonization_map_draft.csv

# View interactive visualizations
open outputs/discovery/test_n10_harmonization_explorer.html
open outputs/discovery/test_n10_cluster_dendrogram_interactive.html
```

---

## Pipeline Overview

| Module | Purpose | Status | Tests |
|--------|---------|--------|-------|
| **1. Core Infrastructure** | Time Zero, temporal windows, outcomes | ✅ Complete | - |
| **2. Lab Processing** | LOINC harmonization, temporal features | ✅ Complete | 22 |
| **3. Vitals Processing** | NLP extraction, hourly grid, tensors | 🔄 Phase 1 Complete | 252 |
| **4. Medication Processing** | RxNorm mapping, 5-layer encoding | 🔄 Phase 6 Complete | 58 |
| **5. Clinical NLP** | Note features, entities | ⬜ Not Started | - |
| **6. Temporal Alignment** | Multi-modal hourly alignment | ⬜ Not Started | - |
| **7. Trajectory Features** | Rolling windows, CSD indicators | ⬜ Not Started | - |
| **8. Format Conversion** | GRU-D, GBTM, XGBoost exports | ⬜ Not Started | - |

---

## Module Architecture

### Module 1: Core Infrastructure ✅

**Purpose:** Establish Time Zero (PE diagnosis), create temporal windows, extract outcomes

**Key Components:**
- Patient timeline extraction with PE diagnosis time
- Admission/discharge detection
- Temporal phase assignment (BASELINE, ACUTE, SUBACUTE, RECOVERY)
- Outcome extraction (mortality, ICU, interventions)

**Input:** Raw RPDR data files
**Output:** `patient_timelines.pkl` (8,713 patients)

**Documentation:** See [module_01_core_infrastructure.md](module_01_core_infrastructure.md)

---

### Module 2: Laboratory Processing ✅

**Purpose:** Harmonize and extract laboratory test data with temporal features

#### Phase 1: Enhanced Three-Tier Harmonization

**Three-Tier Architecture:**

1. **Tier 1: LOINC Exact Matching (96.7% coverage)**
   - Matches any test with a LOINC code
   - Uses LOINC COMPONENT field for precise grouping
   - Properly separates LDL/HDL/VLDL cholesterol
   - Auto-approved (no review needed)

2. **Tier 2: LOINC Family Matching**
   - Groups tests by LOINC component (same analyte, different systems)
   - Flags for review if multiple systems or units
   - Handles institutional LOINC variants

3. **Tier 3: Hierarchical Clustering (3.3% coverage)**
   - Ward's method with combined distance metric
   - 60% token similarity + 40% unit compatibility
   - Detects isoenzymes (LDH1-5, CK-MB, Troponin I/T)
   - Flags singletons and suspicious clusters

**Key Features:**
- ✅ **100% test coverage** (exceeds 90-95% target)
- ✅ **LOINC database**: 66,497 codes with pickle caching (64x speedup)
- ✅ **Unit conversion**: 6 common lab tests supported
- ✅ **Interactive visualizations**: Dendrogram + 4-panel dashboard
- ✅ **Quality checks**: Isoenzyme detection, unit mismatch flags

**Outputs:**
```
outputs/discovery/
├── harmonization_map_draft.csv          ← SINGLE SOURCE OF TRUTH
├── tier1_loinc_exact.csv                ← Tier 1 details (319 groups)
├── tier2_loinc_family.csv               ← Tier 2 details (0 groups - expected)
├── tier3_cluster_suggestions.csv        ← Tier 3 details (6 clusters)
├── cluster_dendrogram.png               ← Static visualization
├── cluster_dendrogram_interactive.html  ← Interactive dendrogram
└── harmonization_explorer.html          ← 4-panel dashboard
```

#### Phase 2: Feature Engineering

**Features Extracted:**
- **Triple Encoding**: (values, masks, timestamps) for time-aware ML
- **Temporal Features**: AUC, slopes, baselines, deltas across phases
- **Clinical Thresholds**: Binary flags for abnormal values
- **Forward-Fill**: Configurable per test type

**Output:** `lab_features.h5` (HDF5 format) + `lab_sequences.h5`

**Documentation:** See [docs/plans/2025-11-08-module2-enhanced-harmonization-plan.md](docs/plans/2025-11-08-module2-enhanced-harmonization-plan.md)

---

### Module 3: Vitals Processing 🔄

**Purpose:** Extract vital signs from structured and unstructured sources, create hourly aligned tensors

**Status:** Phase 1 Complete (Layers 1-2), 252 tests passing

#### Architecture: 5-Layer System

| Layer | Purpose | Output |
|-------|---------|--------|
| **Layer 1** | Canonical Records | `canonical_vitals.parquet` |
| **Layer 2** | Hourly Grid + Tensors | `hourly_tensors.h5` |
| **Layer 3** | Feature Engineering | `engineered_features.parquet` |
| **Layer 4** | Embeddings | FPCA, autoencoder latents |
| **Layer 5** | World Model States | Dynamics learning |

#### Data Sources

| Source | File | Records | Extraction |
|--------|------|---------|------------|
| PHY (Structured) | `Phy.txt` | 2.7 GB | Direct parsing |
| HNP (H&P Notes) | `Hnp.txt` | 2.3 GB | NLP extraction |
| PRG (Progress Notes) | `Prg.txt` | 29.7 GB | NLP extraction |

#### Vital Signs Extracted

- **HR**: Heart Rate (bpm)
- **SBP/DBP/MAP**: Blood Pressure (mmHg)
- **RR**: Respiratory Rate (breaths/min)
- **SpO2**: Oxygen Saturation (%)
- **Temp**: Temperature (°C, converted from °F)

**Key Files:**
- `module_3_vitals_processing/extractors/` - PHY, HNP, PRG extractors
- `module_3_vitals_processing/processing/` - Layer builders
- `module_3_vitals_processing/config/vitals_config.py` - Central config

**Documentation:** See [docs/plans/2025-12-08-vitals-5-layer-architecture-design.md](docs/plans/2025-12-08-vitals-5-layer-architecture-design.md)

---

### Module 4: Medication Processing 🔄

**Purpose:** Unified medication encoding for all trajectory analysis methods

**Status:** Phase 5 Complete (Layers 1-3)

#### Architecture: 5-Layer System

| Layer | Purpose | Output | Status |
|-------|---------|--------|--------|
| **Layer 1** | Canonical Records | `canonical_records.parquet` (23 MB, 1.71M records) | ✅ Complete |
| **Layer 2** | Therapeutic Classes | `class_indicators.parquet` (53 classes, 25K rows) | ✅ Complete |
| **Layer 3** | Individual Medications | `individual_indicators.parquet` (581 meds, 98.4% sparse) | ✅ Complete |
| **Layer 4** | Embeddings | `medication_embeddings.h5` (769 co-occur + 1,582 PK) | ✅ Complete |
| **Layer 5** | Dose Intensity | DDD-normalized, weight-adjusted features | ⬜ Pending |

#### Key Features

- **RxNorm Integration**: Download from UMLS, ≥85% mapping target
- **53 Therapeutic Classes**: PE-critical anticoagulants (9), expanded vasopressors (6), anti-inflammatories (4), etc.
- **5 Embedding Types**: Semantic (BioBERT), Ontological (Node2Vec), Co-occurrence (Word2Vec), PK, Hierarchical
- **LLM-Assisted Parsing**: Benchmark Llama/Mistral/Phi/Gemma/Qwen for ambiguous doses
- **Multi-Format Export**: GBTM CSVs, GRU-D tensors, XGBoost features, World Model actions

**Key Files:**
- `module_04_medications/config/therapeutic_classes.yaml` - 53 class definitions
- `module_04_medications/config/dose_patterns.yaml` - Parsing patterns + DDD values
- `module_04_medications/config/medication_config.py` - Central configuration

**Documentation:** See [docs/plans/2025-12-08-module-04-medications-design.md](docs/plans/2025-12-08-module-04-medications-design.md)

---

## Enhanced Harmonization - Deep Dive

### Why Three Tiers?

**Problem:** Original fuzzy matching incorrectly grouped LDL + HDL + VLDL together

**Solution:** Cascading three-tier approach:
- Tier 1 catches 96.7% via LOINC exact match (no false groupings)
- Tier 2 catches LOINC family variants (different test codes, same analyte)
- Tier 3 catches remaining tests with intelligent clustering

### Example: Cholesterol Separation

**Before (Fuzzy Matching):**
```
❌ Group: "CHOLESTEROL" (all variants together)
   - LDL Cholesterol
   - HDL Cholesterol
   - VLDL Cholesterol
   - Total Cholesterol
```

**After (Three-Tier System):**
```
✅ Group: "cholesterol_in_ldl"
   LOINC: 13457-7 - Cholesterol.in LDL

✅ Group: "cholesterol_in_hdl"
   LOINC: 2085-9 - Cholesterol.in HDL

✅ Group: "cholesterol_in_vldl"
   LOINC: 2091-7 - Cholesterol.in VLDL

✅ Group: "cholesterol"
   LOINC: 2093-3 - Cholesterol (total)
```

### Hierarchical Clustering Details

**Distance Metric:**
```python
combined_distance = 0.6 * (1 - token_similarity) + 0.4 * unit_incompatibility
```

**Token Similarity:**
- Jaccard index on word tokens
- Removes stop words (TEST, BLOOD, SERUM, etc.)
- Case-insensitive

**Example:**
```
"C-REACTIVE PROTEIN (TEST:BC1-262)"
vs
"C REACTIVE PROTEIN (TEST:MCSQ-CRPX)"

Token similarity: 0.85 (high - same test, minor naming difference)
Unit compatibility: 1.0 (both mg/L)
Combined distance: 0.15 (low distance = high similarity)
→ Clustered together ✓
```

---

## Performance Metrics

### Test Dataset (n=10 patients)

| Metric | Value |
|--------|-------|
| Total unique tests | 330 |
| Tier 1 coverage | 319 (96.7%) |
| Tier 2 coverage | 0 (0.0%) - expected |
| Tier 3 coverage | 11 (3.3%) |
| **Total coverage** | **330 (100%)** |
| LOINC load time | 0.04s (cached) |
| Phase 1 runtime | ~3 min |

### Full Cohort (n=3,565 patients)

| Metric | Value |
|--------|-------|
| Patient timelines | 3,565 |
| Lab measurements | ~63M rows scanned |
| Expected runtime | ~25 min |

---

## File Structure

```
TDA_11_25/
├── README.md                          ← You are here
├── Data/                              ← Raw RPDR data (gitignored)
│   ├── Med.txt                        ← Medications (3.7 GB)
│   ├── Lab.txt                        ← Labs (10.7 GB)
│   ├── Phy.txt                        ← Structured vitals (2.7 GB)
│   ├── Hnp.txt                        ← H&P notes (2.3 GB)
│   ├── Prg.txt                        ← Progress notes (29.7 GB)
│   └── ...
│
├── module_1_core_infrastructure/
│   ├── module_01_core_infrastructure.py
│   └── outputs/
│       ├── patient_timelines.pkl      ← 8,713 patients
│       └── outcomes.csv
│
├── module_2_laboratory_processing/
│   ├── module_02_laboratory_processing.py
│   ├── loinc_matcher.py
│   ├── hierarchical_clustering.py
│   └── outputs/
│       ├── discovery/
│       └── lab_features.h5
│
├── module_3_vitals_processing/         ← NEW
│   ├── config/
│   │   └── vitals_config.py
│   ├── extractors/
│   │   ├── phy_extractor.py           ← Structured vitals
│   │   ├── hnp_extractor.py           ← H&P NLP (662 lines)
│   │   └── prg_extractor.py           ← Progress NLP (542 lines)
│   ├── processing/
│   │   ├── layer1_builder.py          ← Canonical records (375 lines)
│   │   └── layer2_builder.py          ← Hourly grid (355 lines)
│   ├── tests/                         ← 252 tests
│   └── outputs/
│
├── module_04_medications/              ← NEW (Design Complete)
│   ├── config/
│   │   ├── therapeutic_classes.yaml   ← 53 drug classes
│   │   ├── dose_patterns.yaml         ← Regex + DDD values
│   │   └── medication_config.py
│   ├── data/
│   │   ├── rxnorm/                    ← RxNorm SQLite
│   │   ├── bronze/                    ← Layer 1
│   │   ├── silver/                    ← RxNorm mapped
│   │   ├── gold/                      ← Layers 2,3,5
│   │   └── embeddings/                ← Layer 4
│   ├── extractors/
│   ├── transformers/
│   ├── exporters/
│   └── validation/
│
├── docs/
│   ├── brief.md                       ← Session briefs
│   └── plans/
│       ├── 2025-12-08-vitals-5-layer-architecture-design.md
│       └── 2025-12-08-module-04-medications-design.md
│
└── pipeline_quick_reference.md        ← Module checklist
```

---

## Workflow

### Standard Workflow (Full Cohort)

```bash
# Step 1: Run Module 1 (Core Infrastructure)
cd module_1_core_infrastructure
python module_01_core_infrastructure.py

# Step 2: Run Module 2 Phase 1 (Harmonization Discovery)
cd ../module_2_laboratory_processing
python module_02_laboratory_processing.py --phase1

# Step 3: Review harmonization map
# Open outputs/discovery/full_harmonization_map_draft.csv in Excel
# Review flagged tests (needs_review=True)
# Adjust QC thresholds as needed

# Step 4: Run Module 2 Phase 2 (Feature Engineering)
python module_02_laboratory_processing.py --phase2

# Step 5: Outputs ready for ML
# - outputs/lab_features.h5 (temporal features)
# - outputs/lab_sequences.h5 (time series)
```

### Test Workflow (10 patients)

Same as above, but add `--test --n=10` to all commands:
```bash
python module_01_core_infrastructure.py --test --n=10
python module_02_laboratory_processing.py --phase1 --test --n=10
python module_02_laboratory_processing.py --phase2 --test --n=10
```

---

## Configuration

### Key Constants

**Module 2:** `module_02_laboratory_processing.py`

```python
# LOINC database
LOINC_CSV_PATH = 'Loinc/LoincTable/Loinc.csv'

# Clustering parameters
CLUSTERING_THRESHOLD = 0.9  # Similarity threshold (90%)
TOKEN_WEIGHT = 0.6          # 60% token similarity
UNIT_WEIGHT = 0.4           # 40% unit compatibility

# Forward-fill limits (hours)
FORWARD_FILL_LIMITS = {
    'creatinine': 24,
    'troponin': 12,
    'default': 48
}

# QC thresholds
QC_THRESHOLDS = {
    'troponin': {'impossible_low': 0, 'impossible_high': 100000},
    'creatinine': {'impossible_low': 0, 'impossible_high': 30},
    # ... more tests
}
```

---

## Testing

### Unit Tests

```bash
# Module 2: Run all tests
cd module_2_laboratory_processing
pytest tests/

# Specific test files
pytest tests/test_loinc_matcher.py        # 3 tests
pytest tests/test_unit_converter.py       # 5 tests
pytest tests/test_hierarchical_clustering.py  # 14 tests
```

### Integration Tests

```bash
# Test with small dataset
python module_02_laboratory_processing.py --phase1 --test --n=10

# Validate outputs
python -c "
import pandas as pd
hmap = pd.read_csv('outputs/discovery/test_n10_harmonization_map_draft.csv')
print(f'Total groups: {len(hmap)}')
print(f'Coverage: {len(hmap)} / 330 = {len(hmap)/330*100:.1f}%')
assert len(hmap) >= 300, 'Coverage too low!'
print('✓ PASS')
"
```

---

## Troubleshooting

### Common Issues

**1. LOINC database not found**
```
ERROR: LOINC database not found at .../Loinc/LoincTable/Loinc.csv
```
**Solution:** Download LOINC from https://loinc.org and place in `module_2_laboratory_processing/Loinc/`

**2. Slow LOINC loading**
```
Loading LOINC database... (taking >5 seconds)
```
**Solution:** First run creates pickle cache. Subsequent runs use cache (0.04s).

**3. "Unmapped tests" confusion**
```
Q: Why does unmapped_tests.csv show 119 tests but coverage is 100%?
```
**Solution:** That file is deprecated. See [UNMAPPED_TESTS_EXPLANATION.md](UNMAPPED_TESTS_EXPLANATION.md)

---

## Documentation

- **[OUTPUT_REVIEW_REPORT.md](OUTPUT_REVIEW_REPORT.md)** - Comprehensive validation report
- **[UNMAPPED_TESTS_EXPLANATION.md](UNMAPPED_TESTS_EXPLANATION.md)** - Explains "unmapped" file confusion
- **[LEGACY_CODE_REMOVAL_SUMMARY.md](LEGACY_CODE_REMOVAL_SUMMARY.md)** - Code cleanup documentation
- **[docs/plans/](docs/plans/)** - Design and implementation plans
- **[docs/brief.md](docs/brief.md)** - Session briefs and progress tracking

---

## Contributing

### Development Setup

```bash
# Install dev dependencies
pip install pytest pandas numpy scipy plotly matplotlib

# Run tests
pytest module_2_laboratory_processing/tests/

# Check code
python -m py_compile module_2_laboratory_processing/*.py
```

### Adding New Tests

Add tests to `module_2_laboratory_processing/tests/`:
```python
def test_my_feature():
    # Test implementation
    assert result == expected
```

---

## Citations

**LOINC Database:**
- LOINC® is copyright © 1995-2024, Regenstrief Institute, Inc.
- Available at: https://loinc.org

**Data Source:**
- Research Patient Data Registry (RPDR)
- Partners HealthCare System

---

## License

[Specify license here]

---

## Contact

[Specify contact information]

---

## Changelog

### 2025-12-11 - Module 4 Layers 3-4 Complete
- ✨ Layer 3: 581 individual medication indicators (98.4% sparse)
- ✨ Layer 4: Word2Vec co-occurrence embeddings (769 meds × 128d)
- ✨ Layer 4: Pharmacokinetic embeddings (1,582 meds × 10d)
- ✨ 58 tests passing

### 2025-12-10 - Module 4 Phases 2-4
- ✨ Layer 1 canonical extraction (1.71M records, 89.9% dose parsing)
- ✨ RxNorm mapping (92.4% coverage via SQLite DB)
- ✨ Layer 2 therapeutic classes (53 indicators, 25K patient-windows)

### 2025-12-09 - Module 4 Medication Design
- 📋 Complete 5-layer medication encoding architecture
- 📋 53 therapeutic class definitions (PE-critical anticoagulants, expanded vasopressors)
- 📋 Dose parsing patterns with WHO DDD values
- 📋 5 embedding types planned (Semantic, Ontological, Co-occurrence, PK, Hierarchical)
- 📋 LLM benchmark plan (Llama/Mistral/Phi/Gemma/Qwen)

### 2025-12-08 - Module 3 Vitals Phase 1
- ✨ Implemented 5-layer vitals architecture (Layers 1-2 complete)
- ✨ PHY/HNP/PRG extractors with NLP patterns
- ✨ Layer 1: Canonical records with PE-relative timestamps
- ✨ Layer 2: Hourly grid + HDF5 tensors with 3-tier imputation
- ✨ 252 tests passing

### 2025-11-08 - Module 2 Enhanced Harmonization
- ✨ Three-tier harmonization (LOINC, Family, Clustering)
- ✨ 66,497 LOINC codes with pickle caching
- ✨ Interactive Plotly visualizations
- ✨ 100% test coverage

### 2025-11-07 - Module 2 Implementation
- ✨ Phase 1 (Discovery & Harmonization)
- ✨ Phase 2 (Feature Engineering)
- ✨ Triple encoding (values, masks, timestamps)

### Prior - Module 1 Implementation
- ✨ Patient timeline extraction (8,713 patients)
- ✨ Temporal phase assignment
- ✨ Outcome extraction

---

**Status:** 🔄 Active Development
**Last Updated:** 2025-12-11
**Version:** 2.6.0
