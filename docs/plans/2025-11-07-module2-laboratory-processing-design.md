# Module 2: Laboratory Processing - Design Document

**Created:** 2025-11-07
**Status:** Design Complete, Ready for Implementation

---

## Executive Summary

**Goal:** Extract and process all laboratory tests from 63M lab rows, creating both engineered temporal features and raw time-series data for trajectory modeling.

**Key Decisions:**
- ✅ Comprehensive extraction (all labs ≥5% frequency)
- ✅ Triple encoding (values + masks + timestamps)
- ✅ LOINC + fuzzy harmonization
- ✅ Multi-tier QC (impossible/extreme/outlier)
- ✅ Advanced kinetics (18 features per test per phase)
- ✅ Two-phase processing (discovery → review → processing)
- ✅ Hybrid outputs (CSV features + HDF5 sequences)

**Estimated Runtime:** 60 minutes (20min discovery + 15min review + 25min processing)

**Expected Outputs:** ~15 MB CSV + ~2 GB HDF5 for full cohort

---

## 1. Architecture Overview

### Core Philosophy

- **Comprehensive extraction** - Include all lab tests performed on ≥5% of cohort (~178 patients)
- **Triple encoding** - Values + masks + time-since-last for deep learning models
- **Clinical validity** - Multi-tier QC with PE-specific physiological ranges
- **Smart harmonization** - LOINC-based grouping with fuzzy matching fallback

### Two-Phase Processing

**Phase 1: Discovery & Harmonization** (15-20 minutes)
- Scan all 63M lab rows in chunks (memory-efficient)
- Build frequency table of all unique tests
- Group tests by LOINC code families
- Apply fuzzy matching to find name variants
- Generate harmonization suggestions for manual review
- Output: Review files (CSV) for user approval

**Phase 2: Feature Engineering** (20-25 minutes, after harmonization approval)
- Load approved harmonization mapping
- Filter to cohort patients (3,565 EMPIs)
- Apply test name harmonization + unit conversion
- Calculate advanced temporal features per phase (18 features × 4 phases)
- Generate triple-encoded sequences (values, masks, timestamps)
- Apply multi-tier QC (impossible/extreme/outlier flags)
- Output: Features CSV + Sequences HDF5 + QC report

---

## 2. Data Structures & Outputs

### Inputs

- `patient_timelines.pkl` from Module 1 (temporal windows, phase boundaries for 3,565 patients)
- `Lab.txt` (63.4M rows, 16 GB, 20 columns including LOINC_Code, Test_Description, Result, Seq_Date_Time, Reference_Units)

### Phase 1 Outputs (Discovery)

```
module_2_laboratory_processing/outputs/discovery/
├── test_frequency_report.csv
│   Columns: test_description, loinc_code, count, pct_of_cohort,
│            reference_units, sample_values
│
├── loinc_groups.csv
│   Columns: canonical_name, loinc_codes, test_count, patient_count,
│            common_units, example_descriptions
│
├── fuzzy_suggestions.csv
│   Columns: suggested_group, matched_tests, similarity_scores,
│            needs_review (True/False)
│
└── unmapped_tests.csv
    Columns: test_description, count, reason_unmapped
```

### Phase 2 Outputs (Processing)

```
module_2_laboratory_processing/outputs/
├── lab_features.csv
│   Shape: (3565 patients, ~800-1200 columns)
│   Columns per harmonized test per phase:
│     - {test}_BASELINE_first, _last, _min, _max, _mean, _median, _std
│     - {test}_BASELINE_delta_from_baseline (always 0 for BASELINE)
│     - {test}_BASELINE_time_to_peak, _time_to_nadir
│     - {test}_BASELINE_rate_of_change
│     - {test}_BASELINE_count
│     - {test}_BASELINE_pct_missing
│     - {test}_BASELINE_longest_gap_hours
│     - {test}_BASELINE_crosses_high_threshold (binary)
│     - {test}_BASELINE_crosses_low_threshold (binary)
│     - {test}_BASELINE_auc (area under curve)
│     - {test}_BASELINE_peak_to_recovery_delta
│     ... repeated for ACUTE, SUBACUTE, RECOVERY phases
│
├── lab_sequences.h5  (HDF5 hierarchical structure)
│   /sequences/{patient_id}/{harmonized_test_name}/
│     ├── timestamps     (1D array: datetime64)
│     ├── values         (1D array: float64)
│     ├── masks          (1D array: uint8, 1=observed 0=missing)
│     ├── qc_flags       (1D array: uint8, 0=valid 1=extreme 2=outlier 3=impossible)
│     └── original_units (1D array: string)
│   /metadata/
│     ├── harmonization_map (JSON string)
│     ├── qc_thresholds (JSON string)
│     └── processing_timestamp
│
├── lab_harmonization_map.json (user-approved, reusable)
│   {
│     "troponin": {
│       "canonical_name": "troponin",
│       "variants": ["TROPONIN I", "TROP I", "HS TROPONIN T", ...],
│       "loinc_codes": ["10839-9", "6598-7", "42757-5", ...],
│       "canonical_unit": "ng/mL",
│       "unit_conversions": {"ng/L": 0.001, "pg/mL": 0.001},
│       "forward_fill_max_hours": 6,
│       "qc_thresholds": {
│         "impossible_low": 0,
│         "impossible_high": 10000,
│         "extreme_high": 1000
│       }
│     },
│     ...
│   }
│
└── lab_qc_report.html
    - Test frequency distributions
    - Missing data patterns per test
    - QC flag statistics
    - Unit conversion summary
    - Outlier examples for review
```

---

## 3. Lab Harmonization Process

### The Challenge

Real EHR data has the same lab test under many names and units:
- "TROPONIN I", "TROP I", "HS TROPONIN I", "TROPONIN I, HIGH SENSITIVITY"
- Different units: ng/mL, ng/L, pg/mL
- Different LOINC codes for different assay methods

### Solution: LOINC-First + Fuzzy Fallback

**Step 1: LOINC-Based Grouping**
- Group tests by LOINC code families
- Example: All creatinine tests (LOINC 2160-0, 38483-4, 14682-9 for different methods)
- Most reliable when LOINC is populated

**Step 2: Fuzzy Matching for Orphans**
- For tests with missing/inconsistent LOINC codes
- Use Test_Description fuzzy string matching (fuzzywuzzy library)
- Threshold: 85% similarity score
- Example: "D DIMER" ↔ "D-DIMER" ↔ "DDIMER" (95% match)

**Step 3: Generate Review Files**
- Output suggested groupings to CSV
- Flag ambiguous matches with `needs_review=True`
- Include frequency stats and sample values

**Step 4: Manual Review Workflow**
1. Review `fuzzy_suggestions.csv` where `needs_review=True`
2. Edit groupings if needed
3. Add unit conversions and QC thresholds
4. Save as `lab_harmonization_map.json`

**Step 5: Apply Harmonization (Phase 2)**
- Map test names to canonical names
- Convert units to canonical form
- Store original test name + unit in metadata

### Frequency Threshold Application

After harmonization, only keep tests where ≥5% of cohort (≥178 patients) have at least one measurement.

---

## 4. Multi-Tier Quality Control

### Three-Tier QC Framework

**Tier 1: Impossible Values (REJECT)**
- Physiologically impossible values
- Examples:
  - Creatinine < 0 or > 30 mg/dL
  - Troponin < 0 or > 100,000 ng/mL
  - Hemoglobin < 0 or > 25 g/dL
  - Lactate < 0 or > 50 mmol/L
- Action: Set to NaN, flag as `qc_flag=3`, exclude from statistics

**Tier 2: Extreme Values (FLAG)**
- Possible but very rare in PE patients
- Based on PE literature + clinical experience
- Examples:
  - Lactate > 20 mmol/L (severe shock)
  - Troponin > 10,000 ng/mL (massive MI)
  - Creatinine > 10 mg/dL (severe AKI)
  - D-dimer > 20,000 ng/mL (massive clot burden)
- Action: Keep value, flag as `qc_flag=1`, include with warning

**Tier 3: Statistical Outliers (FLAG)**
- Values >3 standard deviations from cohort mean
- Calculated per test after harmonization
- Data-driven rather than clinical thresholds
- Action: Keep value, flag as `qc_flag=2`, track in QC report

### QC Threshold Dictionary (Examples)

```python
QC_THRESHOLDS = {
    'troponin': {
        'impossible_low': 0,
        'impossible_high': 100000,
        'extreme_high': 10000,
    },
    'creatinine': {
        'impossible_low': 0,
        'impossible_high': 30,
        'extreme_high': 10,
    },
    'lactate': {
        'impossible_low': 0,
        'impossible_high': 50,
        'extreme_high': 20,
    },
    'ddimer': {
        'impossible_low': 0,
        'impossible_high': 100000,
        'extreme_high': 20000,
    },
    # ... 50+ other tests
}
```

---

## 5. Temporal Feature Engineering

### Advanced Kinetics Features

For each harmonized lab test (e.g., troponin), calculate 18 features across all four temporal phases:

**1. Basic Statistics (7 features)**
```
{test}_{phase}_first      # First value in phase
{test}_{phase}_last       # Last value in phase
{test}_{phase}_min        # Minimum value
{test}_{phase}_max        # Maximum (peak)
{test}_{phase}_mean       # Mean value
{test}_{phase}_median     # Median value
{test}_{phase}_std        # Standard deviation
```

**2. Temporal Dynamics (4 features)**
```
{test}_{phase}_delta_from_baseline  # Change from baseline mean
{test}_{phase}_time_to_peak         # Hours from phase start to max
{test}_{phase}_time_to_nadir        # Hours from phase start to min
{test}_{phase}_rate_of_change       # (last - first) / hours_elapsed
```

**3. Threshold Crossings (2 features)**
```
{test}_{phase}_crosses_high_threshold  # Binary: exceeds clinical threshold
{test}_{phase}_crosses_low_threshold   # Binary: falls below threshold
```

**4. Missing Data Patterns (3 features)**
```
{test}_{phase}_count              # Number of measurements in phase
{test}_{phase}_pct_missing        # % of hours with no measurement
{test}_{phase}_longest_gap_hours  # Maximum hours between measurements
```

**5. Area Under Curve (1 feature)**
```
{test}_{phase}_auc  # Trapezoidal integration over phase duration
```

**6. Cross-Phase Dynamics (1 feature)**
```
{test}_{phase}_peak_to_recovery_delta  # Peak in this phase - mean in RECOVERY
```

**Total Features:** ~18 features × 4 phases × 50-80 tests = 3,600-5,760 features

### Clinical Thresholds (Examples)

```python
CLINICAL_THRESHOLDS = {
    'troponin': {'high': 0.04},      # ng/mL - myocardial injury
    'lactate': {'high': 4.0},         # mmol/L - tissue hypoperfusion
    'creatinine': {'high': 1.5},      # mg/dL - acute kidney injury
    'ddimer': {'high': 500},          # ng/mL - elevated clot burden
    'bnp': {'high': 100},             # pg/mL - cardiac strain
    'hemoglobin': {'low': 7.0},       # g/dL - severe anemia
}
```

---

## 6. Triple Encoding for Deep Learning

### Purpose

Create model-ready sequences with explicit missing data encoding for LSTM, GRU-D, Transformers, etc.

### The Triple Encoding Framework

For each patient, each harmonized lab test, store three parallel arrays:

**1. Values Array**
- Actual measurements with forward-fill up to test-specific limit
- NaN for missing values beyond forward-fill limit
- Example: `[2.1, 2.1, 2.8, NaN, NaN, 3.5, 3.5, ...]`

**2. Masks Array** (binary observation indicator)
- 1 = actual observation, 0 = missing/imputed
- Example: `[1, 0, 1, 0, 0, 1, 0, ...]`

**3. Timestamps Array**
- Exact measurement timestamps (datetime64)
- Used to compute time-since-last-observation dynamically
- Example: `[datetime(...), datetime(...), ...]`

### Forward-Fill Limits by Test Type

```python
FORWARD_FILL_LIMITS = {
    # Rapid change markers (4-6 hours)
    'troponin': 6,
    'lactate': 4,
    'cardiac_enzymes': 6,

    # Diagnostic markers (6-12 hours)
    'ddimer': 12,
    'creatinine': 12,
    'bilirubin': 12,

    # Slower-changing markers (12-24 hours)
    'bnp': 24,
    'ntprobnp': 24,
    'albumin': 24,

    # Default for general labs
    'default': 12
}
```

### HDF5 Storage Structure

```
/sequences/{patient_id}/{harmonized_test_name}/
  ├── timestamps     (datetime64[ns] array)
  ├── values         (float64 array)
  ├── masks          (uint8 array: 0 or 1)
  ├── qc_flags       (uint8 array: 0-3)
  └── original_units (string array)
```

### Loading for Models (Example)

```python
import h5py
import numpy as np

with h5py.File('lab_sequences.h5', 'r') as f:
    patient_id = '100000272'
    test = 'troponin'

    timestamps = f[f'/sequences/{patient_id}/{test}/timestamps'][:]
    values = f[f'/sequences/{patient_id}/{test}/values'][:]
    masks = f[f'/sequences/{patient_id}/{test}/masks'][:]

    # Compute time-since-last dynamically
    time_since = compute_time_since_last(timestamps, masks)

    # GRU-D requires (values, masks, deltas)
    gru_d_input = (values, masks, time_since)
```

---

## 7. Test Mode & Implementation Strategy

### Test Mode Support

Following Module 1's pattern:

```bash
# Test with 10 patients (~3-5 minutes per phase)
python module_02_laboratory_processing.py --phase1 --test --n=10
python module_02_laboratory_processing.py --phase2 --test --n=10

# Test with 100 patients (~5-8 minutes per phase)
python module_02_laboratory_processing.py --phase1 --test
python module_02_laboratory_processing.py --phase2 --test

# Full cohort (3,565 patients, ~40-50 minutes per phase)
python module_02_laboratory_processing.py --phase1
# [Review harmonization files]
python module_02_laboratory_processing.py --phase2
```

### Key Functions Structure

**Phase 1: Discovery**
```python
def scan_lab_data(lab_file, patient_empis, test_mode):
    """Scan all lab data, build frequency tables"""

def group_by_loinc(frequency_df):
    """Group tests by LOINC code families"""

def fuzzy_match_orphans(unmapped_tests):
    """Find similar test names using fuzzy matching"""

def generate_discovery_reports(loinc_groups, fuzzy_groups):
    """Output CSV files for manual review"""
```

**Phase 2: Processing**
```python
def load_harmonization_map(json_file):
    """Load user-approved harmonization mapping"""

def extract_lab_sequences(lab_file, patient_timelines, harmonization_map):
    """Extract raw sequences with triple encoding"""

def apply_qc(sequences, qc_thresholds):
    """Apply multi-tier quality control"""

def calculate_temporal_features(sequences, patient_timelines):
    """Calculate 18 features per test per phase"""

def save_outputs(features_df, sequences_dict, output_dir):
    """Save CSV + HDF5 outputs"""
```

### Performance Estimates

| Mode | Patients | Phase 1 | Review | Phase 2 | Total |
|------|----------|---------|--------|---------|-------|
| Test (n=10) | 10 | 3 min | 5 min | 2 min | 10 min |
| Test (n=100) | 100 | 5 min | 10 min | 4 min | 19 min |
| Full | 3,565 | 20 min | 15 min | 25 min | 60 min |

### Expected Output Sizes

| Output | Test (n=10) | Test (n=100) | Full (n=3,565) |
|--------|-------------|--------------|----------------|
| lab_features.csv | ~50 KB | ~500 KB | ~15 MB |
| lab_sequences.h5 | ~200 KB | ~20 MB | ~2 GB |
| Discovery CSVs | ~100 KB | ~500 KB | ~2 MB |

---

## 8. Success Criteria

**Phase 1 Complete When:**
- ✓ All unique tests identified with frequencies
- ✓ LOINC groupings generated
- ✓ Fuzzy matching suggestions produced
- ✓ Review files ready for manual inspection

**Phase 2 Complete When:**
- ✓ >90% of labs successfully harmonized
- ✓ <1% values flagged as impossible (Tier 1)
- ✓ All 18 temporal features calculated per test per phase
- ✓ Triple encoding arrays created for all tests
- ✓ HDF5 file validates and loads correctly
- ✓ QC report shows reasonable distributions

**Data Quality Validation:**
- ✓ No patients missing all lab data
- ✓ Key biomarkers (D-dimer, troponin, creatinine) present in >50% of cohort
- ✓ Temporal features align with clinical expectations
- ✓ Unit conversions applied correctly (spot-check)

---

## 9. Next Steps After Module 2

**Immediate:**
1. Validate outputs with test mode (n=10)
2. Review harmonization suggestions
3. Run full cohort
4. Generate QC report

**Downstream Modules:**
- **Module 3 (Vitals):** Will use same triple encoding pattern
- **Module 6 (Alignment):** Will load lab_sequences.h5 for hourly grid alignment
- **Module 7 (Trajectories):** Will use lab_features.csv for trajectory modeling

---

**END OF DESIGN DOCUMENT**
