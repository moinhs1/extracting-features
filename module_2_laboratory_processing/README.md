# Module 2: Laboratory Processing

Enhanced laboratory test harmonization and feature engineering with LOINC integration and hierarchical clustering.

## Overview

This module extracts and harmonizes laboratory test data from electronic health records, achieving **100% test coverage** through a three-tier harmonization system.

**Key Features:**
- 🎯 **Three-Tier Harmonization**: LOINC exact → LOINC family → Hierarchical clustering
- 🧬 **66,497 LOINC Codes**: With 64x speedup caching
- 📊 **Interactive Visualizations**: Plotly dashboards for review
- 🔄 **Unit Conversion**: Automated conversion for 6 common lab tests
- ⏱️ **Temporal Features**: AUC, slopes, deltas across phases
- 🎨 **Triple Encoding**: Values, masks, timestamps for time-aware ML

---

## Quick Start

### Test Run (10 patients)

```bash
# Phase 1: Discovery & Harmonization
python module_02_laboratory_processing.py --phase1 --test --n=10

# Review outputs
open outputs/discovery/test_n10_harmonization_explorer.html

# Phase 2: Feature Engineering
python module_02_laboratory_processing.py --phase2 --test --n=10
```

### Full Cohort (3,565 patients)

```bash
# Phase 1
python module_02_laboratory_processing.py --phase1

# Review harmonization_map_draft.csv
# Edit QC thresholds and review flags as needed

# Phase 2
python module_02_laboratory_processing.py --phase2
```

---

## Three-Tier Harmonization System

### Architecture

```
Input: 330 unique lab tests
  ↓
┌─────────────────────────────────────────┐
│ Tier 1: LOINC Exact Matching           │
│ - Matches any test with LOINC code     │
│ - Uses COMPONENT field for grouping    │
│ - Coverage: 96.7% (319/330 tests)      │
│ - Status: Auto-approved                │
└─────────────────────────────────────────┘
  ↓ Unmapped tests (11 remaining)
┌─────────────────────────────────────────┐
│ Tier 2: LOINC Family Matching          │
│ - Groups by LOINC component            │
│ - Handles test variants                │
│ - Coverage: 0% (local codes)           │
│ - Status: Needs review if flagged      │
└─────────────────────────────────────────┘
  ↓ Unmapped tests (11 remaining)
┌─────────────────────────────────────────┐
│ Tier 3: Hierarchical Clustering        │
│ - Ward's method clustering             │
│ - Combined distance metric             │
│ - Coverage: 3.3% (11/330 tests)        │
│ - Status: Review singletons & flags    │
└─────────────────────────────────────────┘
  ↓
Output: 325 groups, 100% coverage
```

### Tier 1: LOINC Exact Matching

**How it works:**
1. Extract LOINC code from test description
2. Look up in LOINC database (66,497 codes)
3. Use COMPONENT field to create group name
4. Validate units and conversion factors

**Example:**
```
Test: "CLDL (TEST:BC1-56)"
LOINC: 13457-7
Component: Cholesterol.in LDL
System: Ser/Plas
Unit: mg/dL
→ Group: "cholesterol_in_ldl"
```

**Benefits:**
- ✅ Clinically accurate (uses LOINC standard)
- ✅ Properly separates LDL/HDL/VLDL
- ✅ No false groupings
- ✅ Auto-approved (no manual review needed)

### Tier 2: LOINC Family Matching

**How it works:**
1. Group unmapped tests by LOINC component
2. Check for system/unit consistency
3. Flag if multiple systems or units found

**Example:**
```
Tests with same component but different systems:
- GLUCOSE (System: Ser/Plas)
- GLUCOSE (System: Urine)
→ Flagged for review (different systems)
```

**When it's used:**
- Local institutional LOINC codes
- Test variants (different test codes, same analyte)
- Non-standard LOINC implementations

### Tier 3: Hierarchical Clustering

**Algorithm:**
- **Method**: Ward's linkage (minimizes within-cluster variance)
- **Distance Metric**: 60% token similarity + 40% unit compatibility
- **Threshold**: 90% similarity

**Token Similarity (Jaccard Index):**
```python
def calculate_token_similarity(name1, name2):
    tokens1 = set(name1.upper().split()) - STOP_WORDS
    tokens2 = set(name2.upper().split()) - STOP_WORDS
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)
```

**Unit Compatibility:**
```python
def calculate_unit_incompatibility(unit1, unit2):
    if unit1 == unit2:
        return 0.0  # Compatible
    if units_in_same_family(unit1, unit2):
        return 0.3  # Partially compatible
    return 1.0  # Incompatible
```

**Combined Distance:**
```python
distance = 0.6 * (1 - token_similarity) + 0.4 * unit_incompatibility
```

**Example:**
```
Test 1: "C-REACTIVE PROTEIN (TEST:BC1-262)" (unit: mg/L)
Test 2: "C-REACTIVE PROTEIN (TEST:MCSQ-CRPX)" (unit: mg/L)

Token similarity: 0.85 (high)
Unit incompatibility: 0.0 (same unit)
Distance: 0.6 * (1-0.85) + 0.4 * 0.0 = 0.09 (low distance)

Threshold: 1 - 0.9 = 0.1
Result: 0.09 < 0.1 → Cluster together ✓
```

**Quality Checks:**

1. **Isoenzyme Detection:**
   - Patterns: LDH1-5, CK-MB/MM/BB, Troponin I/T
   - Action: Flag for manual review

2. **Large Clusters:**
   - Threshold: >10 tests
   - Action: Flag for review

3. **Unit Mismatch:**
   - Check: All tests in cluster have same unit
   - Action: Flag if mixed units

4. **Singletons:**
   - Check: Cluster size = 1
   - Action: Flag for review (may need merging)

---

## Outputs

### Phase 1: Discovery Files

```
outputs/discovery/
├── harmonization_map_draft.csv          ← SINGLE SOURCE OF TRUTH
│   Columns: group_name, loinc_code, component, system,
│            standard_unit, conversion_factors, tier,
│            needs_review, matched_tests, patient_count
│
├── tier1_loinc_exact.csv               ← Tier 1 details
│   319 groups, 96.7% coverage
│
├── tier2_loinc_family.csv              ← Tier 2 details
│   0 groups (expected for this dataset)
│
├── tier3_cluster_suggestions.csv       ← Tier 3 details
│   6 clusters from 11 unmapped tests
│
├── cluster_dendrogram.png              ← Static visualization
│   117 KB, 2986x1484 pixels
│
├── cluster_dendrogram_interactive.html ← Interactive dendrogram
│   Plotly with zoom/pan/hover
│
└── harmonization_explorer.html         ← 4-panel dashboard
    - Coverage by tier (pie chart)
    - Review status (bar chart)
    - Patient coverage (histogram)
    - Tests per group (histogram)
```

### Phase 2: Feature Files

```
outputs/
├── lab_features.csv                    ← Temporal features (CSV)
├── lab_features.h5                     ← Temporal features (HDF5)
└── lab_sequences.h5                    ← Time series (HDF5)
    Triple encoding: (values, masks, timestamps)
```

---

## Unit Conversion

### Supported Tests

```python
DEFAULT_CONVERSIONS = {
    'glucose': {
        'target': 'mg/dL',
        'factors': {'mmol/L': 18.018, 'mg/dL': 1.0}
    },
    'creatinine': {
        'target': 'mg/dL',
        'factors': {'µmol/L': 0.0113, 'mg/dL': 1.0}
    },
    'cholesterol': {
        'target': 'mg/dL',
        'factors': {'mmol/L': 38.67, 'mg/dL': 1.0}
    },
    'triglycerides': {
        'target': 'mg/dL',
        'factors': {'mmol/L': 88.57, 'mg/dL': 1.0}
    },
    'bilirubin': {
        'target': 'mg/dL',
        'factors': {'µmol/L': 0.0585, 'mg/dL': 1.0}
    },
    'calcium': {
        'target': 'mg/dL',
        'factors': {'mmol/L': 4.008, 'mg/dL': 1.0}
    }
}
```

### Usage

```python
from unit_converter import UnitConverter

converter = UnitConverter()
value, target_unit, converted = converter.convert_value(
    value=5.5,
    test_component='glucose',
    source_unit='mmol/L'
)
# Result: (99.0, 'mg/dL', True)
```

---

## Temporal Features

### Feature Types

**1. Baseline Values:**
- First measurement in BASELINE phase
- Used as reference for delta calculations

**2. Phase Statistics:**
- Min, max, mean, median per phase
- Standard deviation
- Count of measurements

**3. Temporal Dynamics:**
- **AUC**: Area under curve (trapezoid rule)
- **Slope**: Linear regression slope
- **Delta**: Change from baseline
- **Rate**: Change per day

**4. Clinical Flags:**
- Binary flags for threshold exceedance
- Customizable per test type

### Example Feature Vector

```python
{
    'creatinine_baseline': 1.2,
    'creatinine_acute_max': 3.5,
    'creatinine_acute_auc': 420.5,
    'creatinine_acute_slope': 0.15,
    'creatinine_delta_acute': 2.3,  # 3.5 - 1.2
    'creatinine_flag_acute_high': 1,  # Above 2.0 threshold
    # ... more features for other phases
}
```

---

## Configuration

### LOINC Database

**Location:** `Loinc/LoincTable/Loinc.csv`

**Download:** https://loinc.org

**Filtering:**
```python
# Real LOINC database uses CLASSTYPE
loinc_df = loinc_df[loinc_df['CLASSTYPE'] == '1']  # Laboratory tests only

# Test LOINC database uses CLASS
loinc_df = loinc_df[loinc_df['CLASS'] == 'LABORATORY']
```

**Caching:**
```python
# First run: Parses CSV (2.4s)
loinc_matcher = LoincMatcher('Loinc/LoincTable/Loinc.csv')
loinc_matcher.load()  # Creates cache/loinc_database.pkl

# Subsequent runs: Loads pickle (0.04s) - 64x speedup
```

### Clustering Parameters

```python
# module_02_laboratory_processing.py

# Similarity threshold (0-1)
CLUSTERING_THRESHOLD = 0.9  # 90% similarity required

# Distance metric weights
TOKEN_WEIGHT = 0.6  # 60% token similarity
UNIT_WEIGHT = 0.4   # 40% unit compatibility

# Ward's method
linkage(distances, method='ward')
distance_threshold = (1 - CLUSTERING_THRESHOLD) * 5.0
```

### QC Thresholds

```python
QC_THRESHOLDS = {
    'troponin': {
        'impossible_low': 0,
        'impossible_high': 100000,
        'extreme_high': 10000
    },
    'creatinine': {
        'impossible_low': 0,
        'impossible_high': 30,
        'extreme_high': 10
    },
    'glucose': {
        'impossible_low': 0,
        'impossible_high': 1200,  # >600 is possible, not impossible
        'extreme_high': 600
    }
}
```

### Forward-Fill Limits

```python
FORWARD_FILL_LIMITS = {
    'creatinine': 24,      # 24 hours
    'troponin_i': 12,      # 12 hours
    'troponin_t': 12,
    'lactate': 6,          # 6 hours
    'default': 48          # 48 hours
}
```

---

## Testing

### Unit Tests

```bash
# Run all tests (22 tests total)
pytest tests/

# Individual test files
pytest tests/test_loinc_matcher.py        # 3 tests
pytest tests/test_unit_converter.py       # 5 tests
pytest tests/test_hierarchical_clustering.py  # 14 tests
```

### Integration Test

```bash
# Quick test (10 patients)
python module_02_laboratory_processing.py --phase1 --test --n=10

# Validate
python -c "
import pandas as pd
hmap = pd.read_csv('outputs/discovery/test_n10_harmonization_map_draft.csv')
assert len(hmap) == 325, f'Expected 325 groups, got {len(hmap)}'
tier1 = hmap[hmap['tier']==1]
assert len(tier1) == 319, f'Expected 319 Tier 1, got {len(tier1)}'
print('✓ All checks passed')
"
```

---

## Troubleshooting

### Issue: LOINC database not found

```
FileNotFoundError: LOINC database not found at .../Loinc.csv
```

**Solution:**
1. Download LOINC from https://loinc.org
2. Extract to `module_2_laboratory_processing/Loinc/`
3. Ensure path is correct: `Loinc/LoincTable/Loinc.csv`

### Issue: Slow first run

```
Loading LOINC database... (taking 2-3 seconds)
```

**Expected behavior:**
- First run: Parses CSV (~2.4s), creates pickle cache
- Subsequent runs: Loads pickle (~0.04s)
- 64x speedup after first run

### Issue: "Why are there unmapped tests when coverage is 100%?"

**Answer:**
The file `unmapped_tests.csv` is DEPRECATED and misleading.

See: [UNMAPPED_TESTS_EXPLANATION.md](../UNMAPPED_TESTS_EXPLANATION.md)

**Truth:**
- All 330 tests are mapped in harmonization_map_draft.csv
- Coverage: 100% (319 Tier 1 + 11 Tier 3)
- The "unmapped" file is from legacy workflow (now removed)

---

## Module Structure

```
module_2_laboratory_processing/
├── module_02_laboratory_processing.py   ← Main module
├── loinc_matcher.py                     ← LOINC database loader
├── unit_converter.py                    ← Unit conversion
├── hierarchical_clustering.py           ← Tier 3 clustering
├── visualization_generator.py           ← Interactive visualizations
│
├── Loinc/                               ← LOINC database (download separately)
│   └── LoincTable/
│       └── Loinc.csv                    ← 66,497 LOINC codes
│
├── cache/                               ← Auto-generated
│   └── loinc_database.pkl               ← Pickle cache (64x speedup)
│
├── outputs/
│   ├── discovery/                       ← Phase 1 outputs
│   │   ├── harmonization_map_draft.csv
│   │   ├── tier1_loinc_exact.csv
│   │   ├── tier3_cluster_suggestions.csv
│   │   └── *.html                       ← Visualizations
│   ├── lab_features.h5
│   └── lab_sequences.h5
│
├── tests/
│   ├── test_loinc_matcher.py
│   ├── test_unit_converter.py
│   └── test_hierarchical_clustering.py
│
├── requirements.txt
└── README.md                            ← You are here
```

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
h5py>=3.7.0
plotly>=5.14.0
matplotlib>=3.6.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.20.0
pint>=0.20
kaleido>=0.2.1
pytest>=7.0.0
```

---

## API Reference

### LoincMatcher

```python
from loinc_matcher import LoincMatcher

matcher = LoincMatcher('Loinc/LoincTable/Loinc.csv')
matcher.load()  # Loads 66,497 codes

loinc_data = matcher.match('2160-0')
# Returns: {
#   'code': '2160-0',
#   'component': 'Creatinine',
#   'system': 'Ser/Plas',
#   'units': 'mg/dL',
#   ...
# }
```

### UnitConverter

```python
from unit_converter import UnitConverter

converter = UnitConverter()

value, unit, success = converter.convert_value(
    value=5.5,
    test_component='glucose',
    source_unit='mmol/L'
)
# Returns: (99.0, 'mg/dL', True)
```

### Hierarchical Clustering

```python
from hierarchical_clustering import (
    perform_hierarchical_clustering,
    flag_suspicious_clusters
)

unmapped_tests = [
    {'name': 'GLU-POC (TEST:BC1-1428)', 'unit': 'MG/DL'},
    {'name': 'GLU POC (TEST:BCGLUPOC)', 'unit': 'mg/dL'},
    # ...
]

clusters, linkage_matrix, distances = perform_hierarchical_clustering(
    unmapped_tests,
    threshold=0.9
)

flags = flag_suspicious_clusters(clusters, unmapped_tests)
```

---

## Performance Benchmarks

### Test Dataset (n=10)

| Operation | Time |
|-----------|------|
| LOINC load (first run) | 2.4s |
| LOINC load (cached) | 0.04s |
| Lab data scan | ~90s |
| Tier 1 matching | <1s |
| Tier 3 clustering | <1s |
| **Total Phase 1** | **~3 min** |

### Full Dataset (n=3,565)

| Operation | Time |
|-----------|------|
| Lab data scan | ~20 min |
| Harmonization | ~2 min |
| Feature engineering | ~3 min |
| **Total** | **~25 min** |

---

## Citation

If you use this module, please cite:

```
[Citation details to be added]
```

**LOINC Citation:**
```
LOINC® is copyright © 1995-2024, Regenstrief Institute, Inc.
Available at: https://loinc.org
```

---

## Changelog

### 2025-11-08 - Enhanced Harmonization
- ✨ Three-tier harmonization system
- ✨ LOINC integration (66,497 codes)
- ✨ Hierarchical clustering (Ward's method)
- ✨ Interactive visualizations (Plotly)
- ✨ 100% test coverage achieved
- 🔧 Removed legacy fuzzy matching
- 📝 Comprehensive documentation

### 2025-11-07 - Initial Implementation
- ✨ Phase 1: Discovery & Harmonization
- ✨ Phase 2: Feature Engineering
- ✨ Triple encoding (values, masks, timestamps)

---

**Status:** ✅ Production Ready
**Coverage:** 100% (330/330 tests)
**Last Updated:** 2025-11-08
