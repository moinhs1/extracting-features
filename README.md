# TDA 11.1 - Temporal Data Analysis Pipeline

A comprehensive clinical data processing pipeline for temporal analysis of patient outcomes, featuring enhanced laboratory test harmonization with LOINC integration and hierarchical clustering.

## Overview

This pipeline processes Electronic Health Record (EHR) data from the Research Patient Data Registry (RPDR) to create rich temporal feature sets for machine learning models. It extracts and harmonizes laboratory tests, medications, diagnoses, and procedures across multiple time phases.

**Key Features:**
- 🧬 **Enhanced Lab Harmonization**: Three-tier system achieving 100% test coverage
- 🔍 **LOINC Integration**: 66,497 LOINC codes with 64x speedup caching
- 📊 **Interactive Visualizations**: Plotly dashboards for harmonization review
- ⏰ **Temporal Phases**: BASELINE, ACUTE, SUBACUTE, RECOVERY
- 🎯 **Triple Encoding**: Values, masks, timestamps for time-aware ML

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

## Module Architecture

### Module 1: Core Infrastructure

**Purpose:** Load and organize patient data into temporal phases

**Key Components:**
- Patient timeline extraction
- Admission/discharge detection
- Temporal phase assignment (BASELINE, ACUTE, SUBACUTE, RECOVERY)
- Outcome extraction

**Input:** Raw RPDR data files
**Output:** `patient_timelines.pkl` (3,565 patients)

**Documentation:** See [module_01_core_infrastructure.md](module_01_core_infrastructure.md)

---

### Module 2: Laboratory Processing ⭐ NEW

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
TDA_11_1/
├── README.md                          ← You are here
├── Data/                              ← Raw RPDR data (gitignored)
│   ├── FNR_20240409_091633_Lab.txt
│   ├── FNR_20240409_091633_Dia.txt
│   └── ...
│
├── module_1_core_infrastructure/
│   ├── module_01_core_infrastructure.py
│   ├── outputs/
│   │   ├── patient_timelines.pkl
│   │   └── outcomes.csv
│   └── tests/
│
├── module_2_laboratory_processing/
│   ├── module_02_laboratory_processing.py
│   ├── loinc_matcher.py               ← LOINC database loader
│   ├── unit_converter.py              ← Unit conversion
│   ├── hierarchical_clustering.py     ← Tier 3 clustering
│   ├── visualization_generator.py     ← Interactive visualizations
│   ├── Loinc/                         ← LOINC database (local)
│   │   └── LoincTable/Loinc.csv
│   ├── cache/                         ← LOINC pickle cache
│   │   └── loinc_database.pkl
│   ├── outputs/
│   │   ├── discovery/                 ← Phase 1 outputs
│   │   │   ├── harmonization_map_draft.csv
│   │   │   ├── tier1_loinc_exact.csv
│   │   │   ├── tier3_cluster_suggestions.csv
│   │   │   └── *.html                 ← Visualizations
│   │   ├── lab_features.csv
│   │   ├── lab_features.h5
│   │   └── lab_sequences.h5
│   └── tests/
│       ├── test_loinc_matcher.py
│       ├── test_unit_converter.py
│       └── test_hierarchical_clustering.py
│
├── docs/
│   ├── brief.md                       ← Session briefs
│   └── plans/
│       ├── 2025-11-08-module2-enhanced-harmonization-design.md
│       └── 2025-11-08-module2-enhanced-harmonization-plan.md
│
└── OUTPUT_REVIEW_REPORT.md            ← Comprehensive validation report
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

### 2025-11-08 - Enhanced Harmonization System
- ✨ Added three-tier harmonization (Tier 1: LOINC, Tier 2: Family, Tier 3: Clustering)
- ✨ Integrated 66,497 LOINC codes with pickle caching
- ✨ Added hierarchical clustering with Ward's method
- ✨ Created interactive visualizations (Plotly dashboards)
- ✨ Achieved 100% test coverage
- 🔧 Removed legacy fuzzy matching workflow
- 📝 Added comprehensive documentation

### 2025-11-07 - Module 2 Implementation
- ✨ Implemented Phase 1 (Discovery & Harmonization)
- ✨ Implemented Phase 2 (Feature Engineering)
- ✨ Added triple encoding (values, masks, timestamps)
- ✨ Added temporal features (AUC, slopes, deltas)

### Prior - Module 1 Implementation
- ✨ Patient timeline extraction
- ✨ Temporal phase assignment
- ✨ Outcome extraction

---

**Status:** ✅ Production Ready
**Last Updated:** 2025-11-08
**Version:** 1.0.0
