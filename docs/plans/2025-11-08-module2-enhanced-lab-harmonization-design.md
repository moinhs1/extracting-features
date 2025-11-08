# Enhanced Lab Harmonization Design
## Module 2: Laboratory Processing - LOINC Integration & Hierarchical Clustering

**Date:** 2025-11-08
**Author:** Claude Code (with user collaboration)
**Status:** Approved for Implementation

---

## Executive Summary

This design enhances Module 2's laboratory test harmonization system with:

1. **LOINC database integration** for comprehensive, clinically-validated test matching (70-80% coverage)
2. **Hierarchical clustering** with combined distance metrics for intelligent grouping of unmapped tests
3. **Automated unit conversion** with validation to standardize measurements across different lab systems
4. **Interactive visualizations** for exploring and validating harmonization decisions
5. **Three-tier cascading workflow** with appropriate review gates

**Key Improvements:**
- Prevents incorrect groupings (LDL+HDL+VLDL) via LOINC COMPONENT field
- Increases harmonization coverage from 72% to ~90-95%
- Enables proper unit conversion with validation (e.g., glucose mmol/L ↔ mg/dL)
- Provides interactive HTML dashboards for harmonization review
- Maintains backward compatibility with existing pipeline

---

## Section 1: System Architecture

### Three-Tier Cascading Harmonization

**Tier 1: LOINC Exact Match (Auto-Approved)**

- Load LOINC database from `/module_2_laboratory_processing/Loinc/LoincTable/Loinc.csv`
- Extract comprehensive fields:
  - `LOINC_NUM`: Unique identifier
  - `COMPONENT`: What's being measured (e.g., "Cholesterol.in LDL" vs "Cholesterol.in HDL")
  - `PROPERTY`: Type of measurement (Mass concentration, etc.)
  - `SYSTEM`: Source (Serum, Plasma, Blood, Urine, etc.)
  - `SCALE_TYP`: Quantitative, Ordinal, Nominal
  - `METHOD_TYP`: Measurement method
  - `EXAMPLE_UNITS`: Standard units for this test
  - `EXAMPLE_UCUM_UNITS`: UCUM-formatted units
  - `LONG_COMMON_NAME`: Full descriptive name
  - `SHORTNAME`: Abbreviated name
  - `CLASS`: Test class (filter to "LABORATORY" only)

- Match lab data's LOINC codes directly to LOINC database
- Auto-approve if:
  - LOINC code matches
  - Units are compatible (exact match or convertible)
- Use COMPONENT field to prevent incorrect groupings:
  - "Cholesterol.in LDL" (LOINC 13457-7) ≠ "Cholesterol.in HDL" (LOINC 2085-9)
  - "Cholesterol.in VLDL" (LOINC 2091-7) ≠ "Cholesterol" (LOINC 2093-3)
- **Expected coverage:** 70-80% of tests

**Tier 2: LOINC Family Match (Needs Review)**

- For tests with partial LOINC matches or related codes
- Group by COMPONENT field (all "Cholesterol.in LDL" tests together)
- Flag for review if:
  - Same component but different SYSTEM (Serum vs Blood vs Urine)
  - Same component but different METHOD_TYP (different assay methods)
  - Same component but incompatible units
- **Expected coverage:** 5-10% additional tests

**Tier 3: Hierarchical Clustering (Needs Review)**

- For remaining unmapped tests (no LOINC code or no LOINC match)
- Combined distance metric:
  - 60% token-based name similarity (Jaccard index on word tokens)
  - 40% unit compatibility (0=same, 0.3=convertible, 1=incompatible)
- Ward's linkage method for balanced clusters
- Auto-cut at 90% similarity threshold
- Auto-flag clusters with:
  - Unit mismatches (incompatible dimensions)
  - Isoenzyme patterns (LDH1-5, CK-MB/MM/BB)
  - POC vs lab mixing (different SYSTEM)
  - Very large clusters (>10 tests, likely too generic)
- **Expected coverage:** 10-15% additional tests

**Total Coverage:** ~90-95% harmonization with appropriate review gates

---

## Section 2: Phase 1 Discovery Workflow & Outputs

### Execution

```bash
python module_02_laboratory_processing.py --phase1 --test --n=10
```

### Processing Steps

**Step 1: Load LOINC Database** (one-time parse, cached)

- Parse `Loinc.csv` (78 MB, ~100K rows)
- Filter to laboratory tests only (`CLASS="LABORATORY"`)
- Build lookup dictionary: `{loinc_code: {component, system, units, name, ...}}`
- Cache as pickle for fast subsequent loads
  - First run: ~5 seconds
  - Cached: <1 second

**Step 2: Scan Lab Data** (existing function, unchanged)

- Chunk through 63.4M rows in Lab.txt
- Count test frequencies, extract LOINC codes, units, sample values
- Filter to cohort patients (10 for test mode, 3,565 for full run)
- Output: frequency table with all unique tests

**Step 3: Tier 1 LOINC Exact Matching**

- For each test in frequency table with LOINC code:
  - Lookup in LOINC database by code
  - Check unit compatibility:
    - Exact match → auto-approve
    - Convertible (same dimension) → calculate conversion factor, auto-approve
    - Incompatible → flag for review
  - Extract COMPONENT to create group name
    - Example: COMPONENT="Cholesterol.in LDL" → group_name="ldl_cholesterol"
  - Use SYSTEM field to identify test context
    - Example: SYSTEM="Ser/Plas" vs "Blood" vs "Urine"
    - Separate groups if SYSTEM differs (Tier 2 review)

**Step 4: Tier 2 LOINC Family Matching**

- Group remaining tests by COMPONENT field
- Example: Multiple "Glucose" tests with different SYSTEM values
  - "Glucose" + SYSTEM="Ser/Plas" → glucose_serum group
  - "Glucose" + SYSTEM="Blood" → glucose_blood group (POC)
- Flag for review if:
  - Different SYSTEM within component
  - Different METHOD_TYP within component
  - Incompatible units within component

**Step 5: Tier 3 Hierarchical Clustering**

- Input: Remaining unmapped tests only
- Calculate pairwise distance matrix:
  ```python
  distance = 0.6 × (1 - token_similarity) + 0.4 × unit_incompatibility
  ```
- Token similarity using Jaccard index:
  ```python
  tokens1 = set("LOW DENSITY LIPOPROTEIN".split())  # {LOW, DENSITY, LIPOPROTEIN}
  tokens2 = set("HIGH DENSITY LIPOPROTEIN".split()) # {HIGH, DENSITY, LIPOPROTEIN}
  intersection = {DENSITY, LIPOPROTEIN}
  union = {LOW, HIGH, DENSITY, LIPOPROTEIN}
  similarity = len(intersection) / len(union) = 2/4 = 0.5
  ```
- Unit incompatibility:
  - Same unit → 0
  - Convertible → 0.3
  - Incompatible → 1.0
- Build dendrogram using Ward's linkage
- Auto-cut at distance threshold (1 - 0.90 similarity)
- Generate clusters and flag suspicious patterns:
  - Regex for isoenzymes: `r'(LDH|CK|TROPONIN)\s*[1-5IVXMB]'`
  - Unit dimension mismatch: mg/dL + % in same cluster
  - Generic terms: "PANEL", "PROFILE", "COMPREHENSIVE"

**Step 6: Generate Reports & Visualizations**

- CSV reports (7 files)
- Static PNG dendrogram (matplotlib)
- Interactive HTML visualizations (plotly, 3 files)

### Phase 1 Outputs

```
module_2_laboratory_processing/outputs/discovery/
├── test_n10_test_frequency_report.csv          # All tests with frequencies
├── test_n10_tier1_loinc_exact.csv              # Auto-approved LOINC matches
├── test_n10_tier2_loinc_family.csv             # LOINC family matches needing review
├── test_n10_tier3_cluster_suggestions.csv      # Hierarchical clustering suggestions
├── test_n10_cluster_dendrogram.png             # Static dendrogram (matplotlib)
├── test_n10_cluster_dendrogram_interactive.html # Interactive dendrogram (plotly)
├── test_n10_harmonization_explorer.html        # Interactive dashboard (plotly)
├── test_n10_qc_threshold_review.csv            # QC thresholds for review
├── test_n10_unmapped_final.csv                 # Tests with no suggested grouping
└── test_n10_harmonization_map_draft.csv        # Combined draft for editing
```

### Harmonization Map Draft CSV Format

**Columns:**

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `group_name` | string | Standardized test name | `ldl_cholesterol` |
| `loinc_code` | string | LOINC code if available | `13457-7` |
| `component` | string | LOINC component | `Cholesterol.in LDL` |
| `system` | string | LOINC system | `Ser/Plas` |
| `standard_unit` | string | Target unit for this group | `mg/dL` |
| `source_units` | string | Pipe-separated original units | `mg/dL\|mmol/L` |
| `conversion_factors` | JSON | Unit conversion factors | `{"mmol/L": 38.67, "mg/dL": 1.0}` |
| `impossible_low` | float | QC threshold (standardized units) | `0` |
| `impossible_high` | float | QC threshold (standardized units) | `500` |
| `extreme_low` | float | QC threshold (standardized units) | `20` |
| `extreme_high` | float | QC threshold (standardized units) | `300` |
| `tier` | int | 1, 2, or 3 | `1` |
| `needs_review` | bool | Manual review required? | `False` |
| `review_reason` | string | Why flagged | `unit mismatch` |
| `matched_tests` | string | Pipe-separated original test names | `CLDL (TEST:BC1-56)\|LDL (TEST:NCLDLR)` |
| `patient_count` | int | Patients with this test | `8` |
| `measurement_count` | int | Total measurements | `247` |

### Interactive Visualizations

**1. Interactive Dendrogram (`cluster_dendrogram_interactive.html`)**

Built with plotly, features:
- **Hover over branches:** See test names, similarity scores, patient counts, units
- **Zoom and pan:** Explore large dendrograms
- **Click branches:** Highlight cluster members
- **Color-coded:** Green=auto-approved, Yellow=needs review, Red=unit mismatch
- **Adjustable cutoff slider:** See groupings at 85%, 90%, 95% similarity thresholds

**2. Harmonization Explorer Dashboard (`harmonization_explorer.html`)**

Multi-tab plotly dashboard:

**Tab 1: Test Groups Overview**
- Interactive table of all groups (sortable, filterable)
- Filter by: tier, needs_review, LOINC component, min patient count
- Click row to see test details in sidebar

**Tab 2: Unit Analysis**
- Scatter plot: tests grouped by component (x-axis) vs unit type (y-axis)
- Size = patient count, Color = tier
- Shows unit heterogeneity within components
- Click point to see conversion factors and sample values

**Tab 3: QC Threshold Validation**
- Box plots of actual test values vs QC thresholds
- Separate plot per test group
- Highlights impossible/extreme thresholds as red/yellow lines
- Shows sample distribution to validate thresholds are reasonable
- Flags if >10% of values fall outside thresholds

**Tab 4: Coverage Metrics**
- Sankey diagram: All tests → Tier 1/2/3 → Auto-approved/Needs Review → Final Groups
- Shows harmonization funnel and coverage at each stage
- Pie charts:
  - Coverage percentages by tier
  - Patient coverage for key biomarkers
  - Measurement distribution across groups

**3. Cluster Distance Heatmap (`cluster_heatmap_interactive.html`)**

Optional, for deep analysis:
- Interactive heatmap showing pairwise similarity for Tier 3 clustered tests
- Hover to see: test pair names, similarity score, unit compatibility
- Hierarchical ordering matches dendrogram
- Helps validate cluster groupings visually

**Technology:** plotly creates standalone HTML files, no server or dependencies needed for viewing

---

## Section 3: Harmonization Map Review & Editing

### Review Workflow

**Step 1: Explore with Interactive Visualizations**

Open `harmonization_explorer.html` in browser:
1. Review coverage metrics (Tab 4) - expect ~90-95% total
2. Check unit analysis (Tab 2) - identify unit heterogeneity
3. Validate QC thresholds (Tab 3) - adjust if >10% flagged
4. Filter groups needing review (Tab 1) - sort by patient_count descending

Open `cluster_dendrogram_interactive.html`:
1. Adjust cutoff slider to explore different similarity thresholds
2. Click suspicious clusters (large, mixed colors)
3. Validate that isoenzymes are separated

**Step 2: Filter for Manual Review**

Quick pandas script to isolate reviews:
```python
import pandas as pd
df = pd.read_csv('test_n10_harmonization_map_draft.csv')

# Show only items needing review, prioritize by patient impact
needs_review = df[df['needs_review'] == True].sort_values('patient_count', ascending=False)
needs_review.to_csv('test_n10_REVIEW_PRIORITY.csv', index=False)

# Specific checks
ldl_hdl_vldl = df[df['component'].str.contains('Cholesterol', na=False)]
isoenzymes = df[df['matched_tests'].str.contains(r'LDH[1-5]|CK-[MBH]', na=False, regex=True)]
```

**Step 3: Edit in Excel/CSV**

Common edits:

**A) Split incorrect groups**
- Problem: Row has `matched_tests="CLDL|HDL|VLDL"` (incorrectly grouped)
- Solution: Create 3 separate rows:
  - Row 1: `group_name="ldl_cholesterol"`, `matched_tests="CLDL"`
  - Row 2: `group_name="hdl_cholesterol"`, `matched_tests="HDL"`
  - Row 3: `group_name="vldl_cholesterol"`, `matched_tests="VLDL"`

**B) Adjust QC thresholds**
- Problem: Glucose `impossible_high=600` is too strict (DKA patients reach 1000+)
- Solution: Change to `impossible_high=1200`, `extreme_high=600`

**C) Fix unit conversions**
- Problem: `conversion_factors='{"mmol/L": 38.67}'` but this is cholesterol factor, not glucose
- Solution: Change to `{"mmol/L": 18.018}` for glucose
- Validate: 5.5 mmol/L × 18.018 = 99.1 mg/dL ✓

**D) Approve groups**
- Change `needs_review` from `True` to `False` after validation

**E) Reject groups**
- Delete row entirely, or set `group_name` to empty string
- Tests will remain ungrouped (appear in Phase 2 but not harmonized)

**Step 4: Validate Edited Map**

```bash
python module_02_laboratory_processing.py --validate-map --test --n=10
```

Validation checks:
- ✓ All conversion factors are mathematically valid (0.001 < factor < 1000)
- ✓ QC thresholds in correct order: `impossible_low < extreme_low < extreme_high < impossible_high`
- ✓ No duplicate group names within same tier
- ✓ All matched_tests exist in frequency report
- ✓ Unit conversions produce values within reasonable range (test with sample data)
- ✓ JSON fields are valid JSON syntax

**Step 5: Finalize Map**

- Save edited CSV as `test_n10_harmonization_map_final.csv`
- Or overwrite `test_n10_harmonization_map_draft.csv`
- Phase 2 auto-detects which file to use: `*_final.csv` > `*_draft.csv` > auto-generate

### Reusability

**Across runs:**
- Harmonization map can be reused for different patient cohorts
- For full cohort: Phase 1 will merge test map with new discoveries

**Production deployment:**
- Rename to `harmonization_map_production.csv` (no test prefix)
- Place in `outputs/` directory
- Phase 2 prioritizes production map over auto-generated

---

## Section 4: Phase 2 Processing with Enhanced Harmonization

### Execution

```bash
python module_02_laboratory_processing.py --phase2 --test --n=10
```

### Processing Steps

**Step 1: Load Validated Harmonization Map**

Auto-detect priority:
1. `*_final.csv` (user-approved)
2. `*_draft.csv` (auto-generated from Phase 1)
3. Auto-generate from LOINC (if no file found)

Parse:
- Convert `conversion_factors` JSON strings to dictionaries
- Load QC thresholds into lookup tables
- Build reverse mapping: `{original_test_name: group_config}`

**Step 2: Extract Lab Sequences with Unit Conversion** (enhanced)

For each patient's lab measurements:

1. **Look up test in harmonization map**
   ```python
   group_config = harmonization_map.get(original_test_name)
   if group_config is None:
       continue  # Unmapped test, skip
   ```

2. **Apply unit conversion immediately**
   ```python
   original_value = 5.5  # Example glucose in mmol/L
   original_unit = "mmol/L"
   standard_unit = group_config['standard_unit']  # "mg/dL"

   if original_unit != standard_unit:
       conversion_factor = group_config['conversion_factors'][original_unit]
       standardized_value = original_value * conversion_factor
       # 5.5 * 18.018 = 99.099 mg/dL
   else:
       standardized_value = original_value
   ```

3. **Apply QC checks on standardized values**
   ```python
   if standardized_value < impossible_low or standardized_value > impossible_high:
       qc_flag = 3  # Impossible, set to NaN
       standardized_value = np.nan
   elif standardized_value < extreme_low or standardized_value > extreme_high:
       qc_flag = 1  # Extreme, keep with warning
   else:
       qc_flag = 0  # Clean
   ```

4. **Store in triple encoding format**
   - Standardized values (after conversion + QC)
   - Original values (before conversion)
   - Timestamps
   - Masks (1=observed, 0=missing)
   - QC flags
   - Original units

**Step 3: Triple Encoding Storage** (enhanced)

HDF5 structure:
```
/sequences/
  /{patient_id}/
    /{group_name}/
      /timestamps          # datetime64[ns], shape (n_measurements,)
      /values              # float64, standardized values, shape (n_measurements,)
      /masks               # int8, 1=observed 0=missing, shape (n_measurements,)
      /qc_flags            # int8, 0=clean 1=extreme 2=outlier 3=impossible
      /original_values     # float64, before conversion
      /original_units      # string, units per measurement (for validation)

/metadata/
  /harmonization_map       # Full harmonization map used
  /loinc_version          # "LOINC v2.76" or similar
  /conversion_applied     # True/False per group
  /qc_stats               # Per-group QC statistics
```

**New metadata** provides:
- Traceability: Which harmonization map was used
- Validation: Original values preserved for checking conversions
- Transparency: QC statistics for quality assessment

**Step 4: Calculate Temporal Features** (mostly unchanged)

- Features calculated on **standardized values** (after conversion)
- All features in consistent units per test group
- QC flags inform feature calculation:
  - Exclude `qc_flag=3` (impossible) from all statistics
  - Include `qc_flag=1,2` (extreme/outlier) with warning
  - Report % clean measurements per feature
- 18 features per test per phase × 4 phases = 72 features per test
- Example features for LDL cholesterol (all in mg/dL):
  - `ldl_cholesterol_ACUTE_mean`: Average LDL during acute phase
  - `ldl_cholesterol_ACUTE_max`: Peak LDL during acute phase
  - `ldl_cholesterol_ACUTE_delta_from_baseline`: Change from baseline

**Step 5: Generate Phase 2 Outputs**

```
module_2_laboratory_processing/outputs/
├── test_n10_lab_features.csv                   # 2,016+ features (standardized units)
├── test_n10_lab_sequences.h5                   # HDF5 with original + standardized
├── test_n10_harmonization_map_applied.csv      # Map actually used in Phase 2
├── test_n10_conversion_report.csv              # Conversion stats per group
└── test_n10_qc_report.csv                      # QC statistics per group
```

### Conversion Report Format

**Columns:**
- `group_name`: Test group
- `loinc_code`: LOINC code
- `standard_unit`: Target unit
- `measurements_total`: Total measurements for this group
- `measurements_converted`: Measurements requiring unit conversion
- `pct_converted`: Percentage converted
- `source_units_found`: Actual units in data (pipe-separated)
- `conversion_factors_used`: Factors applied (JSON)
- `min_value_original`: Min before conversion
- `max_value_original`: Max before conversion
- `min_value_standardized`: Min after conversion
- `max_value_standardized`: Max after conversion
- `conversion_validation`: "PASS" or "WARN: suspicious ratio"

**Example row:**
```
glucose,2345-7,mg/dL,1164,89,7.6%,mg/dL|mmol/L,{"mmol/L": 18.018},5.1,33.2,91,598,PASS
```

### QC Report Format

**Columns:**
- `group_name`: Test group
- `measurements_total`: Total measurements
- `qc_clean`: Count with `qc_flag=0`
- `qc_extreme`: Count with `qc_flag=1`
- `qc_outlier`: Count with `qc_flag=2`
- `qc_impossible`: Count with `qc_flag=3` (set to NaN)
- `pct_flagged`: (extreme + outlier + impossible) / total
- `mean_value`: Mean of clean values (standardized units)
- `std_value`: Std dev of clean values
- `threshold_impossible_low`: QC threshold applied
- `threshold_impossible_high`: QC threshold applied
- `threshold_extreme_low`: QC threshold applied
- `threshold_extreme_high`: QC threshold applied
- `threshold_validation`: "OK" or "WARN: >10% flagged"

**Example row:**
```
glucose,1164,1089,42,18,15,6.5%,128.4,45.2,0,1200,40,600,OK
```

---

## Section 5: Technical Implementation Details

### LOINC Database Loading

**File:** `loinc_matcher.py`

```python
import pandas as pd
import pickle
import os
from pathlib import Path

class LoincMatcher:
    """Handles LOINC database loading and matching."""

    def __init__(self, loinc_csv_path, cache_dir='cache'):
        self.loinc_csv_path = Path(loinc_csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / 'loinc_database.pkl'
        self.loinc_dict = None

    def load(self):
        """Load LOINC database with caching."""

        # Check cache first
        if self.cache_path.exists():
            print(f"Loading LOINC database from cache...")
            with open(self.cache_path, 'rb') as f:
                self.loinc_dict = pickle.load(f)
            print(f"  Loaded {len(self.loinc_dict)} LOINC codes from cache")
            return self.loinc_dict

        # Parse LOINC CSV
        print(f"Parsing LOINC database from {self.loinc_csv_path}...")
        loinc_df = pd.read_csv(self.loinc_csv_path, dtype=str, low_memory=False)

        # Filter to laboratory tests only
        loinc_df = loinc_df[loinc_df['CLASS'] == 'LABORATORY'].copy()
        print(f"  Found {len(loinc_df)} laboratory tests")

        # Build lookup dictionary
        self.loinc_dict = {}
        for _, row in loinc_df.iterrows():
            loinc_code = row['LOINC_NUM']
            self.loinc_dict[loinc_code] = {
                'component': row['COMPONENT'],
                'property': row['PROPERTY'],
                'system': row['SYSTEM'],
                'scale': row['SCALE_TYP'],
                'method': row['METHOD_TYP'],
                'units': row['EXAMPLE_UNITS'],
                'ucum_units': row['EXAMPLE_UCUM_UNITS'],
                'name': row['LONG_COMMON_NAME'],
                'short_name': row['SHORTNAME'],
                'class': row['CLASS']
            }

        # Cache for future use
        self.cache_dir.mkdir(exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.loinc_dict, f)
        print(f"  Cached LOINC database to {self.cache_path}")

        return self.loinc_dict

    def match(self, loinc_code):
        """
        Look up LOINC code.

        Returns:
            dict or None: LOINC metadata if found
        """
        if self.loinc_dict is None:
            self.load()
        return self.loinc_dict.get(loinc_code)
```

### Token-Based Distance Metric

**File:** `hierarchical_clustering.py`

```python
import numpy as np
from typing import Dict, List

def calculate_token_similarity(name1: str, name2: str) -> float:
    """
    Calculate Jaccard similarity between tokenized test names.

    Args:
        name1: First test name (e.g., "LOW DENSITY LIPOPROTEIN")
        name2: Second test name (e.g., "HIGH DENSITY LIPOPROTEIN")

    Returns:
        float: Similarity in [0, 1], where 1=identical tokens
    """
    # Tokenize: split on whitespace, convert to uppercase, remove empty
    tokens1 = set(name1.upper().split())
    tokens2 = set(name2.upper().split())

    # Remove common stop words that don't add meaning
    stop_words = {'TEST', 'BLOOD', 'SERUM', 'PLASMA', 'LEVEL'}
    tokens1 = tokens1 - stop_words
    tokens2 = tokens2 - stop_words

    # Jaccard similarity
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def calculate_unit_incompatibility(unit1: str, unit2: str) -> float:
    """
    Calculate unit incompatibility score.

    Args:
        unit1: First unit (e.g., "mg/dL")
        unit2: Second unit (e.g., "mmol/L")

    Returns:
        float: Incompatibility in [0, 1], where 0=same unit, 1=incompatible
    """
    # Normalize units (lowercase, strip whitespace)
    u1 = unit1.lower().strip() if unit1 else ""
    u2 = unit2.lower().strip() if unit2 else ""

    # Exact match
    if u1 == u2:
        return 0.0

    # Check if convertible using pint
    try:
        import pint
        ureg = pint.UnitRegistry()

        # Parse units
        unit_obj1 = ureg(u1)
        unit_obj2 = ureg(u2)

        # Same dimensionality = convertible
        if unit_obj1.dimensionality == unit_obj2.dimensionality:
            return 0.3  # Compatible but need conversion
        else:
            return 1.0  # Incompatible dimensions

    except:
        # Fallback: simple string matching for common patterns
        # Normalize common variations
        normalize_map = {
            'mgdl': 'mg/dL',
            'mg/dl': 'mg/dL',
            'mmol': 'mmol/L',
            'umol/l': 'µmol/L',
            'g/dl': 'g/dL',
            'u/l': 'U/L'
        }

        u1_norm = normalize_map.get(u1.replace(' ', ''), u1)
        u2_norm = normalize_map.get(u2.replace(' ', ''), u2)

        if u1_norm == u2_norm:
            return 0.0

        # Check if both are concentration units (likely convertible)
        conc_units = {'mg/dL', 'mmol/L', 'µmol/L', 'g/dL', 'ng/mL', 'pg/mL'}
        if u1_norm in conc_units and u2_norm in conc_units:
            return 0.5  # Possibly convertible, needs review

        # Otherwise, incompatible
        return 1.0


def calculate_combined_distance(test1: Dict, test2: Dict, unit_weight: float = 0.4) -> float:
    """
    Combined distance metric: 60% name similarity + 40% unit compatibility.

    Args:
        test1: Test dictionary with 'name' and 'unit' keys
        test2: Test dictionary with 'name' and 'unit' keys
        unit_weight: Weight for unit incompatibility (default 0.4)

    Returns:
        float: Distance in [0, 1], where 0=identical, 1=completely different
    """
    # Calculate name similarity and convert to distance
    name_similarity = calculate_token_similarity(test1['name'], test2['name'])
    name_distance = 1 - name_similarity

    # Calculate unit incompatibility (already a distance)
    unit_distance = calculate_unit_incompatibility(test1['unit'], test2['unit'])

    # Weighted combination
    combined = (1 - unit_weight) * name_distance + unit_weight * unit_distance

    return combined
```

### Hierarchical Clustering Implementation

**File:** `hierarchical_clustering.py` (continued)

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import re

def detect_isoenzyme_pattern(test_names: List[str]) -> bool:
    """
    Detect if cluster contains isoenzymes (should be separated).

    Args:
        test_names: List of test names in cluster

    Returns:
        bool: True if isoenzyme pattern detected
    """
    # Patterns for isoenzymes
    patterns = [
        r'LDH\s*[1-5]',      # LDH1, LDH2, ..., LDH5
        r'CK[-\s]?(MB|MM|BB)', # CK-MB, CK-MM, CK-BB
        r'TROPONIN\s*[IT]',  # Troponin I, Troponin T (different biomarkers)
    ]

    for pattern in patterns:
        matches = [re.search(pattern, name, re.IGNORECASE) for name in test_names]
        if sum(1 for m in matches if m) >= 2:
            return True  # At least 2 isoenzymes in cluster

    return False


def perform_hierarchical_clustering(
    unmapped_tests: List[Dict],
    threshold: float = 0.9,
    unit_weight: float = 0.4
) -> tuple:
    """
    Cluster unmapped tests using Ward's method with combined distance metric.

    Args:
        unmapped_tests: List of test dictionaries with 'name', 'unit', etc.
        threshold: Similarity threshold for cutting dendrogram (0-1)
        unit_weight: Weight for unit compatibility in distance (default 0.4)

    Returns:
        clusters: Dict mapping cluster_id to list of test indices
        linkage_matrix: Linkage matrix for dendrogram plotting
        distances: Full distance matrix for heatmap
    """
    n = len(unmapped_tests)

    if n == 0:
        return {}, None, None

    if n == 1:
        return {0: [0]}, None, None

    # Calculate pairwise distance matrix
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_combined_distance(
                unmapped_tests[i],
                unmapped_tests[j],
                unit_weight=unit_weight
            )
            distances[i, j] = dist
            distances[j, i] = dist

    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distances)

    # Perform hierarchical clustering (Ward's method)
    linkage_matrix = linkage(condensed_dist, method='ward')

    # Cut dendrogram at threshold
    # Ward uses squared Euclidean distance, need to adjust threshold
    # Convert similarity threshold to distance threshold
    distance_threshold = (1 - threshold) * linkage_matrix[-1, 2]

    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Group tests by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return clusters, linkage_matrix, distances


def flag_suspicious_clusters(
    clusters: Dict[int, List[int]],
    unmapped_tests: List[Dict]
) -> Dict[int, List[str]]:
    """
    Identify clusters that need manual review.

    Args:
        clusters: Dict mapping cluster_id to list of test indices
        unmapped_tests: List of test dictionaries

    Returns:
        flags: Dict mapping cluster_id to list of flag reasons
    """
    flags = {}

    for cluster_id, test_indices in clusters.items():
        cluster_flags = []

        # Get test info
        test_names = [unmapped_tests[i]['name'] for i in test_indices]
        test_units = [unmapped_tests[i]['unit'] for i in test_indices]

        # Flag 1: Very large cluster (>10 tests)
        if len(test_indices) > 10:
            cluster_flags.append(f"large_cluster ({len(test_indices)} tests)")

        # Flag 2: Isoenzyme pattern
        if detect_isoenzyme_pattern(test_names):
            cluster_flags.append("isoenzyme_pattern")

        # Flag 3: Unit mismatch (incompatible units)
        unique_units = set(test_units)
        if len(unique_units) > 1:
            # Check if all are convertible
            incompatible = False
            units_list = list(unique_units)
            for i in range(len(units_list)):
                for j in range(i+1, len(units_list)):
                    incomp = calculate_unit_incompatibility(units_list[i], units_list[j])
                    if incomp > 0.5:  # Not convertible
                        incompatible = True
                        break
                if incompatible:
                    break

            if incompatible:
                cluster_flags.append(f"unit_mismatch ({', '.join(unique_units)})")

        # Flag 4: Generic terms (likely too broad)
        generic_terms = ['PANEL', 'PROFILE', 'COMPREHENSIVE', 'BASIC', 'COMPLETE']
        if any(term in ' '.join(test_names).upper() for term in generic_terms):
            cluster_flags.append("generic_terms")

        # Flag 5: POC vs Lab mixing (if SYSTEM info available)
        if 'system' in unmapped_tests[0]:
            systems = [unmapped_tests[i].get('system', '') for i in test_indices]
            if 'Blood' in systems and ('Ser/Plas' in systems or 'Serum' in systems):
                cluster_flags.append("poc_vs_lab_mixing")

        if cluster_flags:
            flags[cluster_id] = cluster_flags

    return flags
```

### Unit Conversion System

**File:** `unit_converter.py`

```python
import json
from typing import Dict, Optional, Tuple

# Common lab test unit conversions
# Format: {test_component: {'target': unit, 'factors': {source_unit: factor}}}
DEFAULT_CONVERSIONS = {
    'glucose': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 18.018,
            'mg/dL': 1.0
        }
    },
    'creatinine': {
        'target': 'mg/dL',
        'factors': {
            'µmol/L': 0.0113,
            'umol/L': 0.0113,
            'mg/dL': 1.0
        }
    },
    'cholesterol': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 38.67,
            'mg/dL': 1.0
        }
    },
    'triglycerides': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 88.57,
            'mg/dL': 1.0
        }
    },
    'bilirubin': {
        'target': 'mg/dL',
        'factors': {
            'µmol/L': 0.0585,
            'umol/L': 0.0585,
            'mg/dL': 1.0
        }
    },
    'calcium': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 4.008,
            'mg/dL': 1.0
        }
    },
    'magnesium': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 2.431,
            'mg/dL': 1.0
        }
    },
}


class UnitConverter:
    """Handles unit conversion for lab tests."""

    def __init__(self, custom_conversions: Optional[Dict] = None):
        """
        Initialize unit converter.

        Args:
            custom_conversions: Optional dict of custom conversions to add
        """
        self.conversions = DEFAULT_CONVERSIONS.copy()
        if custom_conversions:
            self.conversions.update(custom_conversions)

    def add_from_loinc(self, loinc_data: Dict):
        """
        Extract unit information from LOINC database.

        Args:
            loinc_data: Dict mapping loinc_code to metadata with 'units' field
        """
        # Build conversions from LOINC EXAMPLE_UNITS
        # This is simplified - in practice, would parse UCUM units
        for loinc_code, metadata in loinc_data.items():
            component = metadata.get('component', '').lower()
            units = metadata.get('units', '')

            if component and units:
                # Create entry if doesn't exist
                if component not in self.conversions:
                    self.conversions[component] = {
                        'target': units,
                        'factors': {units: 1.0}
                    }

    def get_conversion_factor(
        self,
        test_component: str,
        source_unit: str
    ) -> Optional[float]:
        """
        Get conversion factor from source unit to target unit.

        Args:
            test_component: Test component name (e.g., 'glucose')
            source_unit: Source unit (e.g., 'mmol/L')

        Returns:
            float or None: Conversion factor, or None if not found
        """
        # Normalize component name
        component = test_component.lower().strip()

        # Normalize unit
        unit = source_unit.lower().strip()

        # Look up conversion
        if component in self.conversions:
            factors = self.conversions[component]['factors']
            return factors.get(unit)

        return None

    def convert_value(
        self,
        value: float,
        test_component: str,
        source_unit: str
    ) -> Tuple[Optional[float], Optional[str], bool]:
        """
        Convert a value to standard units.

        Args:
            value: Original value
            test_component: Test component name
            source_unit: Source unit

        Returns:
            (converted_value, target_unit, success):
                - converted_value: Converted value, or original if no conversion
                - target_unit: Target unit, or source if no conversion
                - success: True if conversion applied, False if not found
        """
        factor = self.get_conversion_factor(test_component, source_unit)

        if factor is None:
            # No conversion found, return original
            return value, source_unit, False

        # Apply conversion
        converted = value * factor

        # Get target unit
        component = test_component.lower().strip()
        target_unit = self.conversions[component]['target']

        return converted, target_unit, True

    def validate_conversion(
        self,
        original_values: list,
        converted_values: list,
        test_component: str
    ) -> Tuple[bool, str]:
        """
        Validate that conversion produces reasonable values.

        Args:
            original_values: List of original values
            converted_values: List of converted values
            test_component: Test component name

        Returns:
            (valid, message): Tuple of validity boolean and message
        """
        import numpy as np

        # Calculate mean ratio
        orig_arr = np.array(original_values)
        conv_arr = np.array(converted_values)

        # Remove zeros to avoid division issues
        mask = orig_arr != 0
        if mask.sum() == 0:
            return True, "No non-zero values to validate"

        ratios = conv_arr[mask] / orig_arr[mask]
        mean_ratio = np.mean(ratios)

        # Check if ratio is reasonable (within 0.001 to 1000)
        if mean_ratio < 0.001 or mean_ratio > 1000:
            return False, f"Suspicious conversion ratio: {mean_ratio:.2f}"

        # Check if converted values overlap with typical ranges
        # This is test-specific - simplified check
        conv_min = np.min(conv_arr)
        conv_max = np.max(conv_arr)

        # Very basic sanity check - no negative values for concentrations
        if conv_min < 0:
            return False, f"Negative values after conversion: {conv_min}"

        return True, f"Conversion validated (ratio={mean_ratio:.2f})"
```

### Dependencies

**New packages needed:**

```txt
# Add to requirements.txt or install separately

scipy>=1.9.0          # Hierarchical clustering
plotly>=5.14.0        # Interactive visualizations
pint>=0.20            # Unit dimension analysis
kaleido>=0.2.1        # Static image export for plotly (optional)
```

**Installation:**
```bash
pip install scipy plotly pint kaleido
```

---

## Section 6: Error Handling & Edge Cases

### LOINC Database Issues

**Edge Case 1: Missing LOINC CSV or corrupted file**

Detection:
```python
if not Path(loinc_csv_path).exists():
    print(f"ERROR: LOINC database not found at {loinc_csv_path}")
    print("Please download LOINC from https://loinc.org")
    print("Place Loinc.csv in module_2_laboratory_processing/Loinc/LoincTable/")
```

Fallback:
- Skip Tier 1/2 LOINC matching
- Proceed with Tier 3 clustering only
- Log warning in discovery report
- Set `loinc_available=False` in metadata

**Edge Case 2: Test has LOINC code not in database**

Example: Lab data has LOINC code `99999-9` but it's not in LOINC database (outdated or custom code)

Handling:
```python
loinc_metadata = loinc_matcher.match(test_loinc_code)
if loinc_metadata is None:
    unrecognized_loinc_codes.append({
        'test_name': test_name,
        'loinc_code': test_loinc_code,
        'patient_count': patient_count
    })
    # Treat as unmapped, send to Tier 3
```

Output: `unrecognized_loinc_codes.csv` with list of unknown codes

**Edge Case 3: LOINC record missing critical fields**

Example: LOINC record has `COMPONENT=null` or `EXAMPLE_UNITS=''`

Handling:
```python
component = loinc_metadata.get('component') or loinc_metadata.get('name') or test_name
units = loinc_metadata.get('units') or 'unknown'

if component is None:
    warnings.append(f"LOINC {loinc_code} missing COMPONENT field")
    # Use test name as component
```

Flag in harmonization map: `needs_review=True`, `review_reason="missing_loinc_metadata"`

### Unit Conversion Issues

**Edge Case 4: Non-standard or misspelled units**

Examples: "MG/DL", "mg/dl", "mgdl", "MMOL"

Normalization:
```python
def normalize_unit(unit: str) -> str:
    """Normalize unit string for matching."""
    # Convert to lowercase
    unit = unit.lower().strip()

    # Remove spaces
    unit = unit.replace(' ', '')

    # Common mappings
    normalize_map = {
        'mgdl': 'mg/dL',
        'mg/dl': 'mg/dL',
        'mmol': 'mmol/L',
        'umol/l': 'µmol/L',
        'umoll': 'µmol/L',
        'g/dl': 'g/dL',
        'gdl': 'g/dL',
        'u/l': 'U/L',
        'ul': 'U/L'
    }

    return normalize_map.get(unit, unit)
```

**Edge Case 5: Unit conversion produces impossible values**

Example: Applying cholesterol conversion factor to glucose

Detection:
```python
# After conversion, check if values are reasonable
if test_component == 'glucose':
    # Glucose should be 50-600 mg/dL typically
    if converted_mean < 10 or converted_mean > 2000:
        warnings.append(f"Suspicious glucose values: mean={converted_mean}")
```

Validation in `conversion_report.csv`:
```python
original_mean = np.mean(original_values)
converted_mean = np.mean(converted_values)
ratio = converted_mean / original_mean

if ratio < 0.1 or ratio > 10:
    # Conversion likely wrong
    validation = f"WARN: suspicious ratio {ratio:.2f}"
else:
    validation = "PASS"
```

**Edge Case 6: Multiple incompatible units for same LOINC code**

Example: Test has both "mg/dL" (concentration) and "%" (percentage)

Detection:
```python
unique_units = test_data['unit'].unique()
unit_incompatibilities = []

for i, u1 in enumerate(unique_units):
    for j, u2 in enumerate(unique_units):
        if i < j:
            incomp = calculate_unit_incompatibility(u1, u2)
            if incomp > 0.8:  # Highly incompatible
                unit_incompatibilities.append((u1, u2, incomp))
```

Action:
- Split into separate groups: `test_name_mgdl` and `test_name_pct`
- Flag both for review
- Report in `unit_dimension_mismatch.csv`

### Clustering Issues

**Edge Case 7: Singleton clusters**

Definition: Test doesn't group with anything (distance >10% to all others)

Handling:
```python
singleton_clusters = [cid for cid, indices in clusters.items() if len(indices) == 1]

for cluster_id in singleton_clusters:
    test_idx = clusters[cluster_id][0]
    harmonization_map.append({
        'group_name': sanitize_name(unmapped_tests[test_idx]['name']),
        'tier': 3,
        'needs_review': True,
        'review_reason': 'singleton (no similar tests found)',
        'matched_tests': unmapped_tests[test_idx]['name']
    })
```

User decision: Keep separate or manually merge with known group

**Edge Case 8: Very large clusters**

Definition: Clustering groups >10 tests together (suspicious)

Detection:
```python
large_clusters = [cid for cid, indices in clusters.items() if len(indices) > 10]
```

Common causes:
- Generic terms: "PANEL", "PROFILE", "COMPREHENSIVE"
- Common keywords: "BLOOD", "SERUM"

Action:
- Flag entire cluster: `needs_review=True`, `review_reason=f"large_cluster ({len(indices)} tests)"`
- Suggest splitting by SYSTEM or METHOD if available
- User manually reviews and splits

**Edge Case 9: Isoenzyme auto-grouping**

Patterns: "LDH1", "LDH2", ..., "LDH5" or "CK-MB", "CK-MM", "CK-BB"

Detection:
```python
def detect_isoenzyme_pattern(test_names):
    patterns = [
        r'LDH\s*[1-5]',
        r'CK[-\s]?(MB|MM|BB)',
        r'TROPONIN\s*[IT]'  # Troponin I vs T are different biomarkers
    ]
    # ... (see implementation above)
```

Action:
- Auto-split cluster into separate groups
- Log as "isoenzyme_separated" in discovery report
- Each isoenzyme becomes its own group

### QC Threshold Issues

**Edge Case 10: No predefined thresholds for new test**

Example: Test "PROCALCITONIN" has no QC thresholds defined

Handling:
```python
if test_component not in QC_THRESHOLDS:
    # Calculate data-driven thresholds
    values = standardized_values[~np.isnan(standardized_values)]
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Very conservative thresholds
    auto_thresholds = {
        'impossible_low': max(0, mean_val - 10 * std_val),
        'impossible_high': mean_val + 10 * std_val,
        'extreme_low': mean_val - 5 * std_val,
        'extreme_high': mean_val + 5 * std_val
    }

    # Flag for review
    harmonization_map['needs_review'] = True
    harmonization_map['review_reason'] = 'auto_threshold (no predefined)'
```

Output in `qc_threshold_review.csv` with distribution plot

**Edge Case 11: Threshold in wrong units**

Example: Predefined threshold is `glucose_impossible_high=600` in mg/dL, but test uses mmol/L

Detection:
```python
if standard_unit != threshold_unit:
    # Need to convert threshold
    factor = get_conversion_factor(test_component, threshold_unit)
    converted_threshold = predefined_threshold * factor
```

Log conversion in QC report

**Edge Case 12: Data distribution violates thresholds**

Example: >10% of values flagged as "impossible" (threshold too strict or conversion error)

Detection:
```python
pct_impossible = (qc_flags == 3).sum() / len(qc_flags) * 100

if pct_impossible > 10:
    warnings.append(f"{test_component}: {pct_impossible:.1f}% flagged as impossible")
    # Recommend relaxing threshold or checking conversion
```

Action:
- Auto-relax threshold by 2×SD
- Flag for review: `review_reason="high_qc_flag_rate"`
- Show distribution plot in QC visualization

### Data Quality Issues

**Edge Case 13: Null or missing LOINC codes**

Very common for older or proprietary tests

Handling:
```python
if pd.isna(loinc_code) or loinc_code == '':
    # Skip Tier 1/2, send to Tier 3 clustering
    unmapped_tests.append(test_data)
```

Expected: ~30-40% of tests have no LOINC code

**Edge Case 14: Non-numeric result values**

Examples: ">600", "<5", "NEGATIVE", "TRACE", "SEE COMMENT"

Current handling: Skip on float conversion error

Enhanced handling:
```python
def parse_lab_result(result_str: str) -> Tuple[Optional[float], Dict]:
    """
    Parse lab result string, handling special cases.

    Returns:
        (value, metadata): Numeric value and metadata dict
    """
    result_str = str(result_str).strip()

    # Try direct conversion first
    try:
        return float(result_str), {}
    except ValueError:
        pass

    # Handle inequality operators
    if result_str.startswith('>'):
        try:
            value = float(result_str[1:].strip())
            return value, {'truncated_high': True}
        except ValueError:
            pass

    if result_str.startswith('<'):
        try:
            value = float(result_str[1:].strip())
            return value, {'truncated_low': True}
        except ValueError:
            pass

    # Handle qualitative results
    qualitative_map = {
        'NEGATIVE': 0.0,
        'TRACE': 0.5,
        'POSITIVE': 1.0,
        'NEG': 0.0,
        'POS': 1.0
    }

    upper = result_str.upper()
    if upper in qualitative_map:
        return qualitative_map[upper], {'qualitative': True}

    # Cannot parse
    return None, {'unparseable': True, 'original': result_str}
```

Store metadata in HDF5: `/sequences/{patient}/{test}/result_metadata`

**Edge Case 15: Duplicate test names with different meanings**

Example: "TOTAL PROTEIN" could be serum or urine

LOINC distinguishes via SYSTEM field:
- Serum Total Protein: LOINC 2885-2, SYSTEM="Ser/Plas"
- Urine Total Protein: LOINC 2889-4, SYSTEM="Urine"

Handling:
```python
if loinc_metadata['system'] in ['Ser/Plas', 'Serum', 'Plasma']:
    group_name = f"{base_name}_serum"
elif loinc_metadata['system'] == 'Urine':
    group_name = f"{base_name}_urine"
elif loinc_metadata['system'] == 'Blood':
    group_name = f"{base_name}_blood"
else:
    group_name = base_name
```

Create separate groups with clear naming

---

## Section 7: Testing & Validation Strategy

### Test Modes

**Current test mode:**
```bash
python module_02_laboratory_processing.py --phase1 --test --n=10
```

**New test modes:**

```bash
# Quick test: 10 patients, all tiers
python module_02_laboratory_processing.py --phase1 --test --n=10

# LOINC-only test: Disable clustering to validate LOINC matching
python module_02_laboratory_processing.py --phase1 --test --n=10 --loinc-only

# Clustering-only test: Test clustering on specific test subset
python module_02_laboratory_processing.py --phase1 --test --n=10 --clustering-only

# Validation test: Compare old vs new harmonization
python module_02_laboratory_processing.py --phase1 --test --n=100 --compare-baseline

# Dry run: Generate reports without writing files
python module_02_laboratory_processing.py --phase1 --test --n=10 --dry-run
```

### Validation Checks

**1. LOINC Coverage Validation**

```python
def validate_loinc_coverage(frequency_df, tier1_matches):
    """Verify LOINC matching makes sense."""

    # Check: Tests with LOINC codes should match
    has_loinc = frequency_df[frequency_df['loinc_code'].notna()]
    matched_with_loinc = tier1_matches[tier1_matches['loinc_code'].notna()]

    coverage = len(matched_with_loinc) / len(has_loinc) * 100

    print(f"LOINC coverage: {coverage:.1f}%")
    assert coverage > 70, f"LOINC coverage too low: {coverage:.1f}%"

    # Check: LDL/HDL/VLDL are separate groups
    ldl = tier1_matches[tier1_matches['component'].str.contains('LDL', na=False)]
    hdl = tier1_matches[tier1_matches['component'].str.contains('HDL', na=False)]
    vldl = tier1_matches[tier1_matches['component'].str.contains('VLDL', na=False)]

    ldl_groups = set(ldl['group_name'].unique())
    hdl_groups = set(hdl['group_name'].unique())
    vldl_groups = set(vldl['group_name'].unique())

    overlap = ldl_groups & hdl_groups & vldl_groups
    assert len(overlap) == 0, f"LDL/HDL/VLDL incorrectly grouped: {overlap}"

    print("✓ LDL/HDL/VLDL are properly separated")
```

**2. Unit Conversion Validation**

```python
def validate_unit_conversions(conversion_report):
    """Verify conversions produce sensible values."""

    issues = []

    for _, row in conversion_report.iterrows():
        group = row['group_name']

        if row['measurements_converted'] == 0:
            continue  # No conversions needed

        # Check: Conversion ratio is reasonable
        min_orig = row['min_value_original']
        max_orig = row['max_value_original']
        min_std = row['min_value_standardized']
        max_std = row['max_value_standardized']

        if min_orig != 0:
            ratio_min = min_std / min_orig
            if ratio_min < 0.001 or ratio_min > 1000:
                issues.append(f"{group}: suspicious min ratio {ratio_min:.2f}")

        if max_orig != 0:
            ratio_max = max_std / max_orig
            if ratio_max < 0.001 or ratio_max > 1000:
                issues.append(f"{group}: suspicious max ratio {ratio_max:.2f}")

    if issues:
        print("⚠ Conversion issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All unit conversions validated")

    return len(issues) == 0
```

**3. Clustering Quality Metrics**

```python
def calculate_clustering_quality(clusters, linkage_matrix, unmapped_tests):
    """Calculate clustering quality metrics."""

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    if len(clusters) <= 1:
        return

    # Prepare data for sklearn metrics
    cluster_labels = []
    for cluster_id, test_indices in clusters.items():
        for _ in test_indices:
            cluster_labels.append(cluster_id)

    # Calculate silhouette score (higher is better, -1 to 1)
    # Requires distance matrix
    # ... (implementation details)

    # Analyze cluster coherence
    poor_clusters = []
    for cluster_id, test_indices in clusters.items():
        if len(test_indices) < 2:
            continue

        # Calculate average within-cluster distance
        distances = []
        for i in range(len(test_indices)):
            for j in range(i+1, len(test_indices)):
                dist = calculate_combined_distance(
                    unmapped_tests[test_indices[i]],
                    unmapped_tests[test_indices[j]]
                )
                distances.append(dist)

        avg_dist = np.mean(distances)

        # Flag if average distance >0.2 (poor coherence)
        if avg_dist > 0.2:
            poor_clusters.append((cluster_id, avg_dist))

    if poor_clusters:
        print(f"⚠ {len(poor_clusters)} clusters with poor coherence:")
        for cid, dist in poor_clusters[:5]:  # Show top 5
            print(f"  Cluster {cid}: avg distance = {dist:.3f}")
    else:
        print("✓ All clusters have good coherence")
```

**4. Regression Testing (Compare to Baseline)**

```bash
# Generate baseline (current system)
git stash
python module_02_laboratory_processing.py --phase1 --test --n=100
mv outputs/discovery outputs/baseline_discovery

# Test new system
git stash pop
python module_02_laboratory_processing.py --phase1 --test --n=100

# Compare
python scripts/compare_harmonization.py \
  --baseline outputs/baseline_discovery \
  --new outputs/discovery \
  --output outputs/harmonization_comparison.html
```

**Comparison script:** `scripts/compare_harmonization.py`

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def compare_harmonization(baseline_dir, new_dir, output_html):
    """Generate comparison report between old and new harmonization."""

    # Load baseline
    baseline_freq = pd.read_csv(f"{baseline_dir}/test_n100_test_frequency_report.csv")
    baseline_fuzzy = pd.read_csv(f"{baseline_dir}/test_n100_fuzzy_suggestions.csv")

    # Load new
    new_freq = pd.read_csv(f"{new_dir}/test_n100_test_frequency_report.csv")
    new_tier1 = pd.read_csv(f"{new_dir}/test_n100_tier1_loinc_exact.csv")
    new_tier2 = pd.read_csv(f"{new_dir}/test_n100_tier2_loinc_family.csv")
    new_tier3 = pd.read_csv(f"{new_dir}/test_n100_tier3_cluster_suggestions.csv")

    # Calculate metrics
    baseline_coverage = len(baseline_fuzzy) / len(baseline_freq) * 100
    new_coverage = (len(new_tier1) + len(new_tier2) + len(new_tier3)) / len(new_freq) * 100

    # Check clinical validations
    # ... (check LDL/HDL separation, isoenzymes, etc.)

    # Generate comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Coverage Comparison', 'Tests by Tier', 'Patient Coverage', 'Measurement Coverage')
    )

    # ... (add plots)

    fig.write_html(output_html)
    print(f"Comparison report saved to {output_html}")
```

**Comparison Metrics:**
- **Coverage:** % tests harmonized
  - Baseline: ~72%
  - Expected new: ~90-95% (+18-23%)
- **Precision:** Known-good groups preserved (LDL≠HDL)
- **Patient coverage:** % patients with key biomarkers
- **Measurement count:** Total harmonized measurements

**5. Clinical Validation Checklist**

Manual review of `harmonization_map_draft.csv`:

```python
def clinical_validation_checklist(harmonization_map_df):
    """Run clinical validation checks."""

    checks = []

    # Check 1: LDL, HDL, VLDL, Total Cholesterol are separate
    lipid_tests = harmonization_map_df[
        harmonization_map_df['component'].str.contains('Cholesterol', na=False)
    ]
    ldl_groups = lipid_tests[lipid_tests['component'].str.contains('LDL', na=False)]['group_name'].unique()
    hdl_groups = lipid_tests[lipid_tests['component'].str.contains('HDL', na=False)]['group_name'].unique()

    ldl_hdl_overlap = set(ldl_groups) & set(hdl_groups)
    if len(ldl_hdl_overlap) == 0:
        checks.append(("✓", "LDL and HDL are separate groups"))
    else:
        checks.append(("✗", f"LDL and HDL incorrectly grouped: {ldl_hdl_overlap}"))

    # Check 2: Troponin I and Troponin T are separate
    troponin = harmonization_map_df[
        harmonization_map_df['matched_tests'].str.contains('TROPONIN', na=False, case=False)
    ]
    # ... (check TROPONIN I groups ≠ TROPONIN T groups)

    # Check 3: LDH isoenzymes are separate
    ldh = harmonization_map_df[
        harmonization_map_df['matched_tests'].str.contains(r'LDH[1-5]', na=False, case=False, regex=True)
    ]
    # ... (check LDH1, LDH2, ..., LDH5 are in separate groups)

    # Check 4: CK isoenzymes handled correctly
    # Check 5: POC glucose vs lab glucose
    # Check 6: CRP and hs-CRP
    # Check 7: D-dimer variants
    # Check 8: BNP and NT-proBNP separate

    # Print results
    print("\nClinical Validation Checklist:")
    print("="*60)
    for status, message in checks:
        print(f"{status} {message}")

    return all(status == "✓" for status, _ in checks)
```

**Full checklist:**
- [ ] LDL, HDL, VLDL, Total Cholesterol are separate groups
- [ ] Troponin I and Troponin T are separate (different biomarkers)
- [ ] LDH isoenzymes (LDH1-5) are separate
- [ ] CK isoenzymes (CK-MB, CK-MM) are separate or appropriately grouped
- [ ] POC glucose vs lab glucose handled correctly (based on SYSTEM field)
- [ ] CRP and hs-CRP appropriately grouped (same analyte, different sensitivity - can group)
- [ ] D-dimer variants grouped (same test, different assays - acceptable to group)
- [ ] BNP and NT-proBNP separate (different biomarkers)
- [ ] Lactate vs Lactic Acid grouped (same test)
- [ ] Creatinine serum vs creatinine urine separate

**6. Performance Benchmarks**

Expected runtimes (on test hardware):

| Operation | n=10 | n=100 | n=3565 (full) |
|-----------|------|-------|---------------|
| LOINC load (first) | 5 sec | 5 sec | 5 sec |
| LOINC load (cached) | <1 sec | <1 sec | <1 sec |
| Phase 1 Tier 1 | 30 sec | 1 min | 5 min |
| Phase 1 Tier 2 | 10 sec | 30 sec | 2 min |
| Phase 1 Tier 3 | 20 sec | 2 min | 10 min |
| Visualizations | 10 sec | 15 sec | 30 sec |
| **Total Phase 1** | **~2 min** | **~5 min** | **~25 min** |

Comparison to current:
- n=10: ~2 min (vs 3-4 min current) → faster due to LOINC efficiency
- Full cohort: ~25 min (vs 20 min current) → acceptable slowdown for comprehensive matching

---

## Section 8: Implementation Plan & Project Structure

### Code Organization

```
module_2_laboratory_processing/
├── module_02_laboratory_processing.py          # Main script (enhanced)
├── loinc_matcher.py                            # NEW: LOINC database handling
├── hierarchical_clustering.py                  # NEW: Clustering logic
├── unit_converter.py                           # NEW: Unit conversion system
├── visualization_generator.py                  # NEW: Interactive visualizations
├── harmonization_validator.py                  # NEW: Validation checks
├── utils.py                                    # Shared utilities
├── constants.py                                # Constants (QC thresholds, etc.)
├── Loinc/                                      # LOINC database (user-provided)
│   └── LoincTable/
│       └── Loinc.csv                          # 78 MB LOINC database
├── cache/                                      # NEW: Cached data
│   └── loinc_database.pkl                     # Parsed LOINC (fast loading)
├── outputs/
│   ├── discovery/                              # Phase 1 outputs
│   │   ├── *.csv                              # Discovery reports
│   │   ├── *.html                             # Interactive visualizations
│   │   └── *.png                              # Static plots
│   ├── *.h5                                   # Phase 2 HDF5 sequences
│   └── *.csv                                  # Phase 2 features
├── scripts/                                    # NEW: Utility scripts
│   ├── compare_harmonization.py               # Baseline comparison
│   └── validate_map.py                        # Harmonization map validation
├── tests/                                      # NEW: Unit tests
│   ├── test_loinc_matcher.py
│   ├── test_clustering.py
│   ├── test_unit_converter.py
│   └── test_validation.py
├── docs/                                       # Documentation
│   ├── LOINC_SETUP.md                         # NEW: LOINC setup guide
│   └── HARMONIZATION_GUIDE.md                 # NEW: Review workflow guide
├── README.md                                   # Updated documentation
└── requirements.txt                            # Updated dependencies
```

### Backward Compatibility

**Phase 2 compatibility:**
- HDF5 structure preserves existing paths:
  - `/sequences/{patient}/{test}/timestamps` (unchanged)
  - `/sequences/{patient}/{test}/values` (now standardized, was mixed units)
  - `/sequences/{patient}/{test}/masks` (unchanged)
  - New additions don't break existing readers:
    - `/sequences/{patient}/{test}/original_values` (NEW)
    - `/sequences/{patient}/{test}/original_units` (NEW)
    - `/sequences/{patient}/{test}/qc_flags` (enhanced)

- Feature CSV format unchanged:
  - Same columns, same feature naming
  - Values now in standardized units (previously mixed)
  - Compatible with downstream modules

**Harmonization map:**
- JSON format still supported (read-only for backward compat)
- CSV format is new default (easier editing)
- Auto-convert JSON → CSV if needed

**Graceful degradation:**
```python
# If LOINC database missing
if not loinc_csv_path.exists():
    warnings.warn("LOINC database not found, skipping Tier 1/2 matching")
    loinc_available = False
    # Proceed with Tier 3 clustering only

# If plotly not installed
try:
    import plotly
    generate_interactive_viz = True
except ImportError:
    warnings.warn("plotly not installed, skipping interactive visualizations")
    generate_interactive_viz = False
    # Generate static plots only

# If pint not installed
try:
    import pint
    use_pint = True
except ImportError:
    use_pint = False
    # Use fallback string-based unit matching
```

### Migration Path from Current System

**Importing old harmonization:**
```python
# Load old harmonization map (JSON)
with open('old_harmonization_map.json') as f:
    old_map = json.load(f)

# Load new LOINC matches
new_tier1 = pd.read_csv('tier1_loinc_exact.csv')

# Merge: prioritize old map for user-approved groups, add new LOINC matches
merged_map = merge_harmonization_maps(old_map, new_tier1)
```

**Comparing old and new:**
```bash
# Generate baseline from old system
git checkout main
python module_02_laboratory_processing.py --phase1 --test --n=100

# Test new system
git checkout feature/enhanced-harmonization
python module_02_laboratory_processing.py --phase1 --test --n=100

# Compare
python scripts/compare_harmonization.py --baseline outputs/baseline --new outputs/discovery
```

### Integration with Module Pipeline

**Module 1 → Module 2:** (unchanged)
- Module 2 loads `patient_timelines.pkl` from Module 1
- Temporal phases (BASELINE, ACUTE, SUBACUTE, RECOVERY) unchanged
- Phase boundaries used for temporal feature calculation

**Module 2 → Module 3:** (enhanced, reusable patterns)
- Module 3 (Vitals Processing) can reuse:
  - LOINC matching framework (if vitals have LOINC codes)
  - Unit conversion system (e.g., temperature °F ↔ °C, BP mmHg)
  - Harmonization map CSV structure
  - Interactive visualization templates

**Module 2 → Module 6:** (unchanged)
- Module 6 (Temporal Alignment) loads HDF5 sequences
- New `original_values` arrays provide validation capability
- Can verify standardization was applied correctly

**Module 2 → Module 7:** (enhanced)
- Module 7 (Trajectory Features) loads `lab_features.csv`
- All features now in standardized units (consistent analysis)
- Can trust unit consistency for trajectory modeling

### Documentation Updates

**1. README.md enhancements:**
- Add "LOINC Setup" section:
  - Download instructions
  - License information (free for research)
  - Placement instructions
- Update "Phase 1 Outputs" section with new files
- Add "Harmonization Map Review" workflow
- Update runtime estimates

**2. LOINC_SETUP.md (new file):**
```markdown
# LOINC Database Setup

## What is LOINC?

LOINC (Logical Observation Identifiers Names and Codes) is the international standard for identifying medical laboratory observations. It provides:
- Universal test codes (e.g., 2093-3 for Cholesterol)
- Standardized test names
- Standard units
- Component classifications

## Downloading LOINC

1. Visit https://loinc.org
2. Create free account (required for download)
3. Accept license (free for research use)
4. Download "LOINC Table File" (CSV format)
5. Current version: LOINC 2.76 (January 2025)

## Installation

1. Extract downloaded zip file
2. Locate `LoincTable/Loinc.csv` (78 MB file)
3. Copy to project:
   ```
   module_2_laboratory_processing/Loinc/LoincTable/Loinc.csv
   ```
4. Verify file exists:
   ```bash
   ls -lh module_2_laboratory_processing/Loinc/LoincTable/Loinc.csv
   ```

## Updating LOINC

LOINC releases new versions 2-3 times per year.

To update:
1. Download new version from loinc.org
2. Replace `Loinc.csv` file
3. Delete cache: `rm module_2_laboratory_processing/cache/loinc_database.pkl`
4. Re-run Phase 1 (will rebuild cache)

## License

LOINC is copyright © 1995-2025, Regenstrief Institute, Inc. and the Logical Observation Identifiers Names and Codes (LOINC) Committee. Free for research use with attribution.

Attribution: This project uses LOINC® codes from Regenstrief Institute, Inc.
```

**3. HARMONIZATION_GUIDE.md (new file):**
```markdown
# Lab Test Harmonization Guide

## Overview

This guide explains how to review and edit the harmonization map generated by Phase 1.

## Step 1: Run Phase 1 Discovery

```bash
python module_02_laboratory_processing.py --phase1 --test --n=10
```

Outputs:
- `harmonization_map_draft.csv` - Main file to review
- `harmonization_explorer.html` - Interactive dashboard
- `cluster_dendrogram_interactive.html` - Clustering visualization

## Step 2: Explore Visualizations

### Open Harmonization Explorer
```bash
open outputs/discovery/test_n10_harmonization_explorer.html
```

**Tab 1: Test Groups Overview**
- Filter to `needs_review=True` to see flagged groups
- Sort by `patient_count` descending to prioritize high-impact reviews

**Tab 3: QC Threshold Validation**
- Check if thresholds make sense given distribution
- Adjust if >10% of values are flagged

### Open Dendrogram
```bash
open outputs/discovery/test_n10_cluster_dendrogram_interactive.html
```

- Adjust cutoff slider to explore different groupings
- Click suspicious clusters (mixed colors)

## Step 3: Common Edits

### Splitting Incorrect Groups

**Problem:** LDL, HDL, VLDL grouped together

Row in CSV:
```
group_name,matched_tests,component,...
cldl_(test:bc1-56),"CLDL|HDL|VLDL",Cholesterol,...
```

**Fix:** Create 3 separate rows:
```
ldl_cholesterol,CLDL,Cholesterol.in LDL,...
hdl_cholesterol,HDL,Cholesterol.in HDL,...
vldl_cholesterol,VLDL,Cholesterol.in VLDL,...
```

### Adjusting QC Thresholds

**Problem:** Glucose `impossible_high=600` too strict for DKA patients

Row in CSV:
```
group_name,impossible_high,extreme_high,...
glucose,600,400,...
```

**Fix:**
```
group_name,impossible_high,extreme_high,...
glucose,1200,600,...
```

### Fixing Unit Conversions

**Problem:** Wrong conversion factor applied

Row in CSV:
```
group_name,conversion_factors,...
glucose,"{""mmol/L"": 38.67}",...  # This is cholesterol factor!
```

**Fix:**
```
group_name,conversion_factors,...
glucose,"{""mmol/L"": 18.018}",...
```

**Validate:** 5.5 mmol/L × 18.018 = 99.1 mg/dL ✓

## Step 4: Validate Edited Map

```bash
python module_02_laboratory_processing.py --validate-map --test --n=10
```

Checks:
- Conversion factors are mathematically valid
- QC thresholds in correct order
- No duplicate group names
- JSON fields are valid syntax

## Step 5: Save and Use

Save as `test_n10_harmonization_map_final.csv`

Phase 2 will automatically use the `*_final.csv` file:
```bash
python module_02_laboratory_processing.py --phase2 --test --n=10
```

## Clinical Validation Checklist

Before running Phase 2 on full cohort, verify:

- [ ] LDL ≠ HDL ≠ VLDL ≠ Total Cholesterol (separate groups)
- [ ] Troponin I ≠ Troponin T (different biomarkers)
- [ ] LDH1, LDH2, ..., LDH5 are separate (isoenzymes)
- [ ] CK-MB ≠ CK-MM (isoenzymes)
- [ ] POC glucose vs lab glucose handled appropriately
- [ ] BNP ≠ NT-proBNP (different biomarkers)
- [ ] QC thresholds are clinically reasonable
- [ ] Unit conversions validated with sample calculations

## Common Mistakes to Avoid

1. **Don't group isoenzymes:** LDH1-5 are clinically distinct
2. **Check unit dimensions:** Don't convert mg/dL to % (incompatible)
3. **Validate conversion factors:** Test with known values
4. **Don't over-tighten QC:** Critical illness has extreme values
5. **Preserve POC vs lab distinction:** Different accuracy and use cases

## Getting Help

- Check interactive visualizations for suspicious patterns
- Use `--validate-map` to catch syntax errors
- Compare to baseline: `scripts/compare_harmonization.py`
- Review discovery CSVs for detailed test information
```

**4. Update pipeline_quick_reference.md:**

Add to Module 2 section:
```markdown
## Module 2: Laboratory Processing ✅ Enhanced (v2.1)

**Status:** Complete with LOINC integration and hierarchical clustering

**New Features (v2.1):**
- Three-tier harmonization (LOINC exact → LOINC family → Clustering)
- Automated unit conversion with validation
- Interactive visualizations for harmonization review
- ~90-95% harmonization coverage (vs 72% in v2.0)

**Dependencies:**
- LOINC database (download from loinc.org, place in `Loinc/LoincTable/`)
- scipy, plotly, pint (pip install)

**Runtime:**
- Phase 1 (n=10): ~2 min
- Phase 1 (full 3,565): ~25 min
- Phase 2 (n=10): ~18 sec (unchanged)
```

### Git Branch Strategy

```bash
# Create feature branch
git checkout -b feature/enhanced-lab-harmonization

# Incremental commits (10 commits total):

# 1. Project structure and LOINC loader
git add loinc_matcher.py cache/ Loinc/
git commit -m "feat(module2): add LOINC database loading with caching"

# 2. Tier 1 LOINC exact matching
git add module_02_laboratory_processing.py
git commit -m "feat(module2): implement Tier 1 LOINC exact matching"

# 3. Tier 2 LOINC family matching
git commit -m "feat(module2): implement Tier 2 LOINC family matching"

# 4. Hierarchical clustering implementation
git add hierarchical_clustering.py
git commit -m "feat(module2): implement Tier 3 hierarchical clustering with combined distance metric"

# 5. Unit conversion system
git add unit_converter.py
git commit -m "feat(module2): add automated unit conversion with validation"

# 6. Interactive visualizations
git add visualization_generator.py
git commit -m "feat(module2): add interactive HTML visualizations (plotly)"

# 7. Phase 2 enhancement for unit conversion
git add module_02_laboratory_processing.py
git commit -m "feat(module2): enhance Phase 2 with unit conversion and original value storage"

# 8. Validation checks
git add harmonization_validator.py scripts/compare_harmonization.py
git commit -m "feat(module2): add validation checks and comparison tools"

# 9. Documentation
git add docs/LOINC_SETUP.md docs/HARMONIZATION_GUIDE.md README.md
git commit -m "docs(module2): add LOINC setup and harmonization review guides"

# 10. Tests and final polish
git add tests/ requirements.txt
git commit -m "test(module2): add unit tests and update dependencies"

# After validation, merge to main
git checkout main
git merge feature/enhanced-lab-harmonization
git push origin main
```

### Rollout Plan

**Phase A: Development (Days 1-3)**

Day 1:
- [ ] Set up project structure
- [ ] Implement LOINC database loader with caching
- [ ] Test LOINC loading with actual data
- [ ] Implement Tier 1 LOINC exact matching
- [ ] Test with n=10: validate LDL/HDL/VLDL separation

Day 2:
- [ ] Implement Tier 2 LOINC family matching
- [ ] Implement hierarchical clustering (Tier 3)
- [ ] Test clustering with n=10: check isoenzyme separation
- [ ] Implement unit conversion system
- [ ] Test conversions with sample data

Day 3:
- [ ] Integrate three tiers into Phase 1 workflow
- [ ] Generate CSV outputs
- [ ] Test end-to-end with n=10
- [ ] Validate against baseline (compare coverage)

**Phase B: Visualization & UX (Days 4-5)**

Day 4:
- [ ] Implement static dendrogram (matplotlib)
- [ ] Implement interactive dendrogram (plotly)
- [ ] Test dendrogram with n=100 (more clusters)

Day 5:
- [ ] Implement harmonization explorer dashboard (4 tabs)
- [ ] Implement cluster heatmap (optional)
- [ ] Test visualizations in browser
- [ ] Create harmonization map CSV editing workflow

**Phase C: Testing (Days 6-7)**

Day 6:
- [ ] Test with n=100 patients
- [ ] Run clinical validation checklist
- [ ] Run unit conversion validation
- [ ] Run clustering quality metrics
- [ ] Fix any issues found

Day 7:
- [ ] Performance benchmarking
- [ ] Edge case testing (see Section 6)
- [ ] Regression testing vs baseline
- [ ] Create comparison report

**Phase D: Full Cohort Trial (Day 8)**

- [ ] Run Phase 1 on full 3,565 patients (~25 min)
- [ ] Review harmonization map in Excel
  - [ ] Check top 50 tests by patient count
  - [ ] Validate all flagged groups (needs_review=True)
  - [ ] Run clinical validation checklist
  - [ ] Adjust QC thresholds based on distributions
- [ ] Save as `harmonization_map_final.csv`
- [ ] Run validation checks
- [ ] Generate comparison vs baseline

**Phase E: Production (Day 9+)**

- [ ] Run Phase 2 on full cohort with validated map (~25 min)
- [ ] Compare features to baseline
  - [ ] Same number of features (2,016)
  - [ ] Similar feature coverage
  - [ ] Values in reasonable ranges
- [ ] Validate HDF5 structure
- [ ] Document any breaking changes
- [ ] Create migration guide for existing analyses
- [ ] Merge to main
- [ ] Tag release: `v2.1.0-loinc-enhanced`

---

## Summary

This design provides a comprehensive enhancement to Module 2's lab test harmonization:

1. **LOINC Integration** - Leverages the gold-standard LOINC database for 70-80% coverage with clinical validity
2. **Hierarchical Clustering** - Sophisticated clustering for remaining tests with combined name+unit distance metrics
3. **Unit Conversion** - Automated, validated unit standardization for consistent downstream analysis
4. **Interactive Visualizations** - Plotly dashboards for exploring and validating harmonization decisions
5. **Three-Tier Workflow** - Cascading approach with appropriate review gates (auto-approve → needs review)

**Key Benefits:**
- Prevents clinical errors (LDL≠HDL≠VLDL properly separated)
- Increases coverage from 72% to ~90-95%
- Standardizes units for consistent analysis
- Provides transparency and review capability
- Maintains backward compatibility

**Implementation Estimate:** 9 days (development to production)

**Dependencies:**
- LOINC database (free download from loinc.org)
- scipy, plotly, pint (pip install)

**Ready for implementation approval.**
