# Module 3: Comprehensive Vitals Extraction - Architecture Design

**Version:** 1.0
**Date:** 2025-11-09
**Status:** Design Phase

---

## Executive Summary

Module 3 implements a **comprehensive multi-source vitals extraction system** that processes structured and unstructured clinical data with maximum temporal resolution, full provenance tracking, and rigorous validation. The system extracts vital signs from three complementary data sources, preserving complete information about data origin, quality, conflicts, and temporal relationships.

**Key Capabilities:**
- Extract vitals from 3 sources: Phy.txt (structured), Hnp.csv (admission H&P notes), Prg.csv (progress notes)
- Advanced NLP extraction with medical context awareness
- Maximum temporal resolution (down to 5-minute intervals when available)
- 6-layer information preservation architecture
- 4-tier validation framework (cross-validation, manual review, statistical monitoring, pattern-specific)
- Complete provenance tracking for every measurement
- Modular design enabling phased implementation and independent testing

---

## 1. Clinical Context & Goals

### 1.1 PE Prediction Requirements

For pulmonary embolism (PE) risk prediction, vital signs provide critical physiological signals:

**Primary PE Indicators:**
- Tachycardia (HR >100) - compensation for reduced cardiac output
- Tachypnea (RR >20) - hypoxemia compensation
- Hypoxemia (SpO2 <90) - impaired gas exchange
- Hypotension (SBP <90) - massive PE with hemodynamic instability

**Secondary Indicators:**
- Fever patterns (PE can cause low-grade fever)
- Blood pressure variability (hemodynamic instability)
- Vital sign trajectories (deterioration vs. improvement)

### 1.2 Design Goals

1. **Comprehensiveness**: Extract all available vitals from all sources
2. **Resolution**: Maximum temporal granularity (5-minute intervals when available)
3. **Provenance**: Track source, timestamp, and quality for every measurement
4. **Quality**: Rigorous validation achieving ≥90% accuracy
5. **Modularity**: Independent submodules enabling parallel development and testing
6. **Reproducibility**: Clear documentation and validation metrics for publication

---

## 2. Data Sources

### 2.1 Source Overview

| Source | Type | Volume | Temporal Pattern | Coverage | Extraction Method |
|--------|------|--------|------------------|----------|-------------------|
| **Phy.txt** | Structured | 33M rows | Sporadic (clinic visits) | Outpatient + some inpatient | Direct field parsing |
| **Hnp.csv** | Unstructured | 257K notes (3.4 GB) | Single admission snapshot | ~80% contain vitals | Advanced NLP/regex |
| **Prg.csv** | Unstructured | 8.7M notes (42 GB) | Serial (hourly to daily) | ~35% contain vitals | Advanced NLP/regex |

### 2.2 Source Characteristics

#### Phy.txt (Structured Provider Data)
**Path:** `/home/moin/TDA_11_1/Data/FNR_20240409_091633_Phy.txt`

**Structure:**
```
EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
```

**Key Vitals Available:**
- Temperature: 2.5M records
- Pulse: 2.4M records
- Blood Pressure (combined & separate): 1.7M records
- Respiratory Rate: 1.3M records
- Oxygen Saturation: 1.6M records
- Weight, Height, BMI: 2.0M, 1.4M, 2.0M records

**Extraction Complexity:** ⭐ (1/5) - Simple structured parsing

**Clinical Context:** Outpatient clinic visits, some inpatient documentation. May miss acute care vitals from ED/ICU where structured documentation differs.

#### Hnp.csv (History & Physical Notes)
**Path:** `/home/moin/TDA_11_1/Data/Hnp.csv`

**Structure:**
```
EMPI|Date|Report_Text
```

**Vitals Coverage:** ~80% of notes contain vitals in Report_Text field

**Example Patterns:**
```
"Vitals: 37.2 °C (99 °F) HR 120 BP 136/72 mmHg RR 20 SpO2 97%"
"T 98.6F, P 88, BP 124/68, RR 18, O2Sat 96% on RA"
"Physical Examination: VS: T 37.1C P 92 BP 130/75 R 16 SaO2 98%"
```

**Extraction Complexity:** ⭐⭐⭐⭐ (4/5) - Unstructured text with variable formatting

**Clinical Context:** Admission vitals - captures patient presentation at hospital entry. **CRITICAL for PE baseline severity assessment.**

#### Prg.csv (Progress Notes)
**Path:** `/home/moin/TDA_11_1/Data/Prg.csv`

**Structure:**
```
EMPI|Date|Report_Text
```

**Vitals Coverage:** ~35% of notes contain vitals in Report_Text field

**Example Patterns:**
```
"Vitals reviewed: BP 118/62, HR 76, RR 16, SpO2 94%"
"Current vitals: 98.2F 88 118/70 18 95% RA"
"Overnight vitals stable, HR 70-85, BP 110-125/60-75"
```

**Extraction Complexity:** ⭐⭐⭐⭐ (4/5) - Unstructured text with variable formatting, ranges, narrative descriptions

**Clinical Context:** Serial measurements during hospitalization - captures disease trajectory, treatment response, clinical deterioration/improvement. **High resolution temporal data for admitted PE patients.**

### 2.3 Complementary Value of Multi-Source Approach

**Why All Three Sources Matter:**

1. **Temporal Coverage:**
   - Phy.txt: Pre-admission baseline (outpatient visits)
   - Hnp.csv: Admission snapshot (presentation severity)
   - Prg.csv: Post-admission trajectory (treatment response)

2. **Clinical Context:**
   - Patients with only Phy.txt data → managed outpatient (less severe PE)
   - Patients with Phy.txt + Hnp.csv → ED admission but no structured inpatient vitals
   - Patients with all three sources → full disease progression trajectory (most severe)

3. **Data Quality:**
   - Cross-validation: Overlap between sources enables validation
   - Redundancy: Multiple measurements improve reliability
   - Gap-filling: Each source captures vitals the others miss

---

## 3. Six-Layer Information Preservation Architecture

### Architecture Overview

Rather than merging data and losing information, the system preserves **six layers of information** about each measurement. This enables sophisticated downstream analyses while maintaining complete data lineage.

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 6: Encounter Pattern Features                         │
│ (has_outpatient_vitals, has_admission_vitals, trajectory)   │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 5: Temporal Consistency Validation                     │
│ (rate_of_change, implausible_flags, temporal_span)          │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 4: Temporal Precision Tracking                         │
│ (exact_timestamps, time_deltas, alignment_quality)          │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: Conflict Detection & Quality Metrics               │
│ (conflict_flags, completeness, redundancy, outliers)        │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: Hierarchical Merged Values + Source Attribution    │
│ (merged_value, source_tag, n_sources)                       │
└─────────────────────────────────────────────────────────────┘
                            ▲
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: Raw Source-Specific Values                         │
│ (HR_phy, HR_hnp, HR_prg with original timestamps)          │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 Layer 1: Raw Source-Specific Values

**Purpose:** Preserve original values from each source without any merging or processing.

**Schema Per Vital Sign:**
```python
{
    'HR_0h_phy': 88.0,           # Heart rate from Phy.txt at time 0
    'HR_0h_phy_timestamp': '2024-03-15 10:30:00',
    'HR_0h_hnp': 92.0,           # Heart rate from Hnp.csv at time 0
    'HR_0h_hnp_timestamp': '2024-03-15 10:45:00',
    'HR_0h_prg': 95.0,           # Heart rate from Prg.csv at time 0
    'HR_0h_prg_timestamp': '2024-03-15 11:00:00',
}
```

**Value:** Enables retrospective analysis of source-specific patterns, debugging extraction issues, and reprocessing with different merge strategies without re-extraction.

### 3.2 Layer 2: Hierarchical Merged Values + Source Attribution

**Purpose:** Create single "best" value per vital/timepoint using priority rules, but track provenance.

**Hierarchy:** Prg.csv > Hnp.csv > Phy.txt (prioritize inpatient > admission > outpatient)

**Rationale:** Inpatient measurements are typically more recent, more frequent, and from patients requiring closer monitoring.

**Schema:**
```python
{
    'HR_0h': 95.0,               # Merged value (selected from prg)
    'HR_0h_source': 'prg',       # Which source provided this value
    'HR_0h_n_sources': 3,        # How many sources had data available
    'HR_0h_timestamp': '2024-03-15 11:00:00',  # Timestamp of selected value
}
```

**Value:** Provides single clean dataset for modeling while maintaining provenance for interpretation.

### 3.3 Layer 3: Conflict Detection & Quality Metrics

**Purpose:** Identify discrepancies between sources and quantify data quality.

**A. Conflict Flags**

When multiple sources provide the same vital at the same timepoint, flag significant disagreements:

**Thresholds for Clinically Significant Conflict:**
```python
CONFLICT_THRESHOLDS = {
    'HR': 10,      # beats/minute
    'RR': 10,      # breaths/minute
    'SBP': 20,     # mmHg
    'DBP': 20,     # mmHg
    'TEMP': 0.5,   # degrees C/F
    'SPO2': 5,     # percentage points
}
```

**Schema:**
```python
{
    'HR_0h_conflict': True,      # Values differ beyond threshold
    'HR_0h_conflict_magnitude': 7.0,  # Max difference between sources
}
```

**Clinical Interpretation:**
- Conflicts may indicate: measurement error, temporal evolution (patient changing), technique differences, data entry errors

**B. Completeness Metrics**

Track what percentage of vital/timepoint combinations have data from each source:

```python
{
    'completeness_phy': 0.35,     # 35% coverage from outpatient
    'completeness_hnp': 0.15,     # 15% from admission (expected - single timepoint)
    'completeness_prg': 0.82,     # 82% from inpatient (high for admitted patients)
    'completeness_overall': 0.91, # 91% overall after merging
}
```

**Value:** Low completeness may indicate incomplete medical records → less reliable predictions.

**C. Redundancy Metrics**

```python
{
    'avg_sources_per_vital': 1.8,  # Average number of sources per vital/timepoint
    'redundancy_score': 0.6,       # Proportion of vitals with 2+ sources
}
```

**Value:** Higher redundancy → better quality through cross-validation, but more conflicts to resolve.

**D. Outlier Flags**

Using modified Z-score (robust to extreme values):

```python
{
    'HR_0h_outlier': False,        # Within expected distribution
    'HR_0h_modified_zscore': 2.1,  # Modified Z-score
}
```

**Threshold:** Flag values with modified Z-score > 3.5

### 3.4 Layer 4: Temporal Precision Tracking

**Purpose:** Quantify how far each measurement was from target timepoint and track alignment quality.

**Schema:**
```python
{
    'HR_0h_phy_time_delta': 2.3,    # Hours from target (2.3h before PE index)
    'HR_0h_hnp_time_delta': 0.1,    # 6 minutes before
    'HR_0h_prg_time_delta': 0.5,    # 30 minutes after
    'HR_0h_time_delta': 0.5,        # Delta for merged value (from prg)
    'HR_avg_time_delta': 1.2,       # Mean offset across all vitals for patient
    'HR_max_time_delta': 8.5,       # Worst-case offset
}
```

**Value:**
- Measurements 10 hours from target are less reliable than 10 minutes away
- Can be used as data quality weight in modeling
- Identifies when extrapolating too far from actual measurements

### 3.5 Layer 5: Temporal Consistency Validation

**Purpose:** Detect physiologically implausible changes between consecutive measurements.

**A. Rate of Change Calculations**

```python
{
    'HR_max_rate_change': 12.3,     # bpm/hour (maximum across all consecutive pairs)
    'HR_implausible_change': False, # Exceeds 50 bpm/hour threshold?
}
```

**Implausibility Thresholds:**
```python
IMPLAUSIBLE_RATE_THRESHOLDS = {
    'HR': 50,    # bpm/hour
    'SBP': 30,   # mmHg/hour
    'DBP': 30,   # mmHg/hour
    'RR': 10,    # breaths/min per hour
    'TEMP': 2,   # degrees/hour
    'SPO2': 20,  # percentage/hour
}
```

**Value:** Flags data quality issues (typos, unit errors) and extreme physiological changes requiring review.

**B. Temporal Coverage Metrics**

```python
{
    'HR_first_timepoint': -48.0,    # Hours (earliest measurement)
    'HR_last_timepoint': 48.0,      # Hours (latest measurement)
    'HR_temporal_span': 96.0,       # Hours of coverage
    'HR_n_timepoints': 8,           # Number of distinct timepoints with data
    'HR_measurement_density': 0.083, # Measurements per hour
}
```

**Value:** Patients with narrow span or low density have less reliable trajectory analysis.

### 3.6 Layer 6: Encounter Pattern Features

**Purpose:** Synthesize data source patterns into clinically meaningful categories.

**A. Binary Source Flags**

```python
{
    'has_outpatient_vitals': True,   # At least one vital from Phy.txt
    'has_admission_vitals': True,    # At least one vital from Hnp.csv
    'has_inpatient_vitals': True,    # At least one vital from Prg.csv
}
```

**B. Categorical Encounter Pattern**

```python
{
    'encounter_pattern': 'full_trajectory'  # Enumerated category
}
```

**Possible Values:**
1. `'outpatient_only'`: Clinic vitals only (never admitted) → Lowest risk
2. `'admission_only'`: ED vitals but no inpatient data → Low-moderate risk
3. `'admitted'`: Inpatient vitals present → Moderate-high risk
4. `'outpatient_to_admission'`: Both outpatient and admission vitals → Disease progression
5. `'full_trajectory'`: All three sources → Most severe with richest data
6. `'unknown'`: No vitals from any source (shouldn't occur)

**Clinical Value:** Single feature captures disease severity and care trajectory. Highly predictive of outcomes because sicker patients progress through more care settings.

---

## 4. Modular Architecture: Submodule Breakdown

The system is decomposed into **10 independent submodules** that can be developed, tested, and validated in parallel.

```
MODULE 3: VITALS EXTRACTION
├── Submodule 3.1: Structured Vitals Extractor (Phy.txt)
├── Submodule 3.2: H&P NLP Extractor (Hnp.csv)
├── Submodule 3.3: Progress Notes NLP Extractor (Prg.csv)
├── Submodule 3.4: Vitals Harmonizer
├── Submodule 3.5: Unit Converter & QC Filter
├── Submodule 3.6: Multi-Source Temporal Aligner
├── Submodule 3.7: Provenance & Quality Metrics Calculator
├── Submodule 3.8: Feature Engineering Pipeline
├── Submodule 3.9: Validation Framework
└── Submodule 3.10: Main Orchestrator
```

### 4.1 Submodule 3.1: Structured Vitals Extractor (Phy.txt)

**Purpose:** Extract vitals from structured Phy.txt file with direct field parsing.

**Input:** `/home/moin/TDA_11_1/Data/FNR_20240409_091633_Phy.txt`

**Output:** `outputs/discovery/phy_vitals_raw.parquet`

**Key Functions:**

1. **Load and Filter:**
   ```python
   def load_phy_vitals(phy_path: str) -> pd.DataFrame:
       """Load Phy.txt and filter for vital sign concepts"""
   ```

2. **Blood Pressure Parser:**
   ```python
   def parse_combined_bp(bp_string: str) -> Tuple[Optional[float], Optional[float]]:
       """Parse '124/68' into (systolic=124, diastolic=68)"""
   ```

3. **Vital Sign Mapper:**
   ```python
   def map_phy_concepts_to_canonical(concept_name: str) -> str:
       """Map 'Pulse' → 'HR', 'O2 Saturation-SPO2' → 'SPO2'"""
   ```

**Complexity:** ⭐ (1/5) - Straightforward structured parsing

**Estimated Time:** 1-2 days

**Dependencies:** None

**Output Schema:**
```python
{
    'EMPI': str,
    'timestamp': datetime,
    'vital_type': str,      # Canonical name: HR, SBP, DBP, RR, SPO2, TEMP
    'value': float,
    'units': str,
    'source': 'phy',
    'encounter_type': str,  # Inpatient/Outpatient
}
```

### 4.2 Submodule 3.2: H&P NLP Extractor (Hnp.csv)

**Purpose:** Extract vitals from unstructured History & Physical notes using advanced NLP/regex.

**Input:** `/home/moin/TDA_11_1/Data/Hnp.csv`

**Output:** `outputs/discovery/hnp_vitals_raw.parquet`

**Key Functions:**

1. **Vitals Section Identifier:**
   ```python
   def identify_vitals_section(note_text: str) -> Optional[str]:
       """Extract text section containing vital signs (reduces false positives)"""
   ```

2. **Multi-Pattern Regex Extractor:**
   ```python
   def extract_vitals_patterns(text: str) -> Dict[str, List[float]]:
       """
       Apply multiple regex patterns for each vital sign.
       Returns all matches with confidence scores.
       """
   ```

3. **Context Validator:**
   ```python
   def validate_extraction_context(text: str, match_position: int) -> bool:
       """
       Check for negation, historical references, ranges.
       Returns True if extraction is from current vitals.
       """
   ```

**Regex Pattern Library:**

```python
# Heart Rate Patterns
HR_PATTERNS = [
    r'(?:HR|Heart Rate|Pulse):?\s*(\d{2,3})',
    r'(?:P|pulse)\s+(\d{2,3})',
    r'(\d{2,3})\s*(?:bpm|beats per minute)',
]

# Blood Pressure Patterns
BP_PATTERNS = [
    r'(?:BP|Blood Pressure):?\s*(\d{2,3})/(\d{2,3})',
    r'(\d{2,3})/(\d{2,3})\s*mmHg',
    r'Systolic:?\s*(\d{2,3}).*?Diastolic:?\s*(\d{2,3})',
]

# Temperature Patterns
TEMP_PATTERNS = [
    r'(?:T|Temp|Temperature):?\s*(\d{2,3}\.?\d?)\s*[°]?([CF])',
    r'(\d{2}\.\d)\s*degrees?\s*([CF])',
]

# Respiratory Rate Patterns
RR_PATTERNS = [
    r'(?:RR|Respiratory Rate|Respiration):?\s*(\d{1,2})',
    r'(?:R|resp)\s+(\d{1,2})',
]

# Oxygen Saturation Patterns
SPO2_PATTERNS = [
    r'(?:SpO2|O2 Sat|Oxygen Saturation):?\s*(\d{2,3})%?',
    r'(?:SaO2|sat)\s+(\d{2,3})%?',
]
```

**Negation Detection:**
```python
NEGATION_PHRASES = [
    'not obtained',
    'not measured',
    'unable to',
    'refused',
    'not documented',
    'not available',
    'could not',
]
```

**Complexity:** ⭐⭐⭐⭐ (4/5) - Advanced NLP with context awareness

**Estimated Time:** 5-7 days (including pattern refinement)

**Dependencies:** None (parallel to 3.1)

**Output Schema:** Same as Submodule 3.1 output

### 4.3 Submodule 3.3: Progress Notes NLP Extractor (Prg.csv)

**Purpose:** Extract vitals from unstructured progress notes with high temporal resolution.

**Input:** `/home/moin/TDA_11_1/Data/Prg.csv`

**Output:** `outputs/discovery/prg_vitals_raw.parquet`

**Key Differences from H&P Extractor:**

1. **Range Handling:**
   ```python
   def parse_vital_range(range_string: str) -> Tuple[float, float, float]:
       """
       Parse ranges like 'HR 70-85' or 'BP 110-125/60-75'
       Returns (min, mean, max)
       """
   ```

2. **Narrative Description Parser:**
   ```python
   def extract_from_narrative(text: str) -> Dict[str, Any]:
       """
       Handle descriptive vitals:
       'vitals stable', 'afebrile', 'tachycardic', 'hypotensive'
       """
   ```

3. **Serial Measurement Extractor:**
   ```python
   def extract_serial_vitals(text: str) -> List[Dict]:
       """
       Extract multiple measurements from text like:
       'Overnight vitals: 0000: 98.2/88/118/70, 0400: 98.5/92/122/75'
       """
   ```

**Additional Patterns for Progress Notes:**

```python
# Range patterns
RANGE_PATTERNS = [
    r'HR\s+(\d{2,3})-(\d{2,3})',
    r'BP\s+(\d{2,3})-(\d{2,3})/(\d{2,3})-(\d{2,3})',
]

# Narrative patterns
NARRATIVE_MAP = {
    'afebrile': {'vital': 'TEMP', 'value': 37.0, 'confidence': 0.7},
    'tachycardic': {'vital': 'HR', 'value': 110, 'confidence': 0.6},
    'bradycardic': {'vital': 'HR', 'value': 55, 'confidence': 0.6},
    'hypotensive': {'vital': 'SBP', 'value': 85, 'confidence': 0.6},
    'hypertensive': {'vital': 'SBP', 'value': 160, 'confidence': 0.6},
}
```

**Complexity:** ⭐⭐⭐⭐⭐ (5/5) - Most challenging due to variability

**Estimated Time:** 7-10 days

**Dependencies:** None (parallel to 3.1, 3.2)

**Output Schema:** Same as Submodule 3.1 output, with additional fields:
```python
{
    'value_type': str,  # 'exact', 'range_min', 'range_max', 'range_mean', 'narrative'
    'confidence': float,  # 0.0-1.0 confidence score
}
```

### 4.4 Submodule 3.4: Vitals Harmonizer

**Purpose:** Map all vital sign variants to canonical names and handle special cases.

**Input:** Outputs from Submodules 3.1, 3.2, 3.3

**Output:** `outputs/discovery/harmonization_map.json`, `outputs/discovery/harmonized_vitals.parquet`

**Key Functions:**

1. **Concept Mapper:**
   ```python
   def create_harmonization_map() -> Dict[str, str]:
       """
       Map all variants to canonical names:
       'Pulse' → 'HR'
       'Heart Rate' → 'HR'
       'O2 Saturation-SPO2' → 'SPO2'
       'O2 Saturation-LFA4600' → 'SPO2'
       'Temperature' → 'TEMP'
       'Systolic-Epic' → 'SBP'
       'Diastolic-Epic' → 'DBP'
       'Blood Pressure-Epic' → ['SBP', 'DBP']  # Special: splits into two
       'Respiratory rate' → 'RR'
       """
   ```

2. **Blood Pressure Consolidator:**
   ```python
   def consolidate_bp_measurements(df: pd.DataFrame) -> pd.DataFrame:
       """
       Combine:
       - Combined 'BP' strings → separate SBP/DBP rows
       - Separate systolic/diastolic → keep as is
       - Ensure no duplicate measurements from same source
       """
   ```

3. **Deduplication:**
   ```python
   def deduplicate_vitals(df: pd.DataFrame) -> pd.DataFrame:
       """
       Remove exact duplicates:
       - Same patient, vital, value, timestamp, source
       Keep near-duplicates (slightly different timestamps) for conflict detection
       """
   ```

**Complexity:** ⭐⭐ (2/5) - Straightforward mapping logic

**Estimated Time:** 2-3 days

**Dependencies:** Submodules 3.1, 3.2, 3.3 must complete first

### 4.5 Submodule 3.5: Unit Converter & QC Filter

**Purpose:** Standardize units and filter physiologically implausible values.

**Input:** `outputs/discovery/harmonized_vitals.parquet`

**Output:** `outputs/discovery/qc_vitals.parquet`, `outputs/discovery/qc_report.csv`

**Key Functions:**

1. **Unit Detector:**
   ```python
   def detect_units(vital_type: str, value: float, original_unit: str) -> str:
       """
       Auto-detect likely unit based on value range:
       - TEMP: 35-42 → Celsius, 95-108 → Fahrenheit
       - Weight: <200 → kg, >200 → likely lbs
       """
   ```

2. **Unit Converter:**
   ```python
   def convert_to_standard_units(vital_type: str, value: float, unit: str) -> float:
       """
       Convert all vitals to standard units:
       - TEMP → Celsius
       - Weight → kg
       - Height → cm
       - BP → mmHg (usually already correct)
       - HR, RR, SPO2 → no conversion needed
       """
   ```

3. **Physiological QC Filter:**
   ```python
   def apply_physiological_qc(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """
       Filter out impossible values:

       PERMISSIVE THRESHOLDS (keep abnormal but possible values):
       - HR: 20-250 bpm
       - SBP: 50-300 mmHg
       - DBP: 30-200 mmHg
       - RR: 4-60 breaths/min
       - SPO2: 50-100%
       - TEMP: 30-45°C
       - Weight: 20-300 kg

       Returns (valid_df, rejected_df)
       """
   ```

4. **Clinical Flag Creator:**
   ```python
   def create_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
       """
       Add binary flags for clinically significant abnormalities:
       - fever: TEMP > 38.0°C
       - hypothermia: TEMP < 36.0°C
       - tachycardia: HR > 100
       - bradycardia: HR < 60
       - tachypnea: RR > 20
       - bradypnea: RR < 12
       - hypoxemia: SPO2 < 90
       - hypotension: SBP < 90
       - severe_hypotension: SBP < 70
       - hypertension: SBP > 180
       """
   ```

**Complexity:** ⭐⭐ (2/5) - Straightforward logic with edge cases

**Estimated Time:** 2-3 days

**Dependencies:** Submodule 3.4

### 4.6 Submodule 3.6: Multi-Source Temporal Aligner

**Purpose:** Align vitals from all three sources to common temporal grid and implement hierarchical merging.

**Input:** `outputs/discovery/qc_vitals.parquet`, `module_1_core_infrastructure/outputs/patient_timelines.pkl`

**Output:** `outputs/features/aligned_vitals_raw.h5`

**Key Functions:**

1. **Temporal Grid Constructor:**
   ```python
   def create_temporal_grid(patient_index_time: datetime, resolution: str = '5min') -> List[datetime]:
       """
       Create temporal grid relative to PE index time:
       - BASELINE: [-365d, -30d] → daily resolution
       - PRE_ACUTE: [-30d, -7d] → daily resolution
       - ACUTE: [-7d, +1d] → hourly resolution
       - HIGH_RES_ACUTE: [-24h, +24h] → 5-minute resolution
       - SUBACUTE: [+2d, +14d] → hourly resolution
       - RECOVERY: [+15d, +90d] → daily resolution
       """
   ```

2. **Temporal Binning:**
   ```python
   def assign_vitals_to_bins(vitals_df: pd.DataFrame, grid: List[datetime]) -> pd.DataFrame:
       """
       Assign each measurement to nearest temporal bin.
       Track time_delta (offset from bin center).
       """
   ```

3. **Hierarchical Merge:**
   ```python
   def merge_multi_source_vitals(vitals_by_source: Dict[str, pd.DataFrame]) -> pd.DataFrame:
       """
       For each patient/vital/timepoint:
       1. Collect values from all sources
       2. Apply hierarchy: prg > hnp > phy
       3. Select best value
       4. Store all Layer 1-2 information
       """
   ```

4. **Admission Vitals Extractor:**
   ```python
   def extract_admission_vitals(hnp_vitals: pd.DataFrame, patient_timelines: Dict) -> pd.DataFrame:
       """
       Special handling: Extract vitals from first H&P note after index time.
       These become separate ADMISSION_* features.
       """
   ```

**Complexity:** ⭐⭐⭐⭐ (4/5) - Complex temporal logic and multi-source merging

**Estimated Time:** 5-7 days

**Dependencies:** Submodules 3.5, Module 1 (patient_timelines.pkl)

**Output Schema (HDF5 structure):**
```
/patient_EMPI/
  /raw_vitals_phy/
    timestamp: [datetime array]
    HR: [float array]
    SBP: [float array]
    ...
  /raw_vitals_hnp/
    timestamp: [datetime array]
    HR: [float array]
    ...
  /raw_vitals_prg/
    timestamp: [datetime array]
    HR: [float array]
    ...
  /merged_vitals/
    timestamp: [datetime array]
    HR: [float array]
    HR_source: [str array]
    HR_n_sources: [int array]
    ...
  /admission_vitals/
    timestamp: datetime
    HR: float
    SBP: float
    ...
```

### 4.7 Submodule 3.7: Provenance & Quality Metrics Calculator

**Purpose:** Calculate Layers 3-6 quality and provenance metrics for every measurement.

**Input:** `outputs/features/aligned_vitals_raw.h5`

**Output:** `outputs/features/vitals_with_provenance.h5`, `outputs/features/quality_report.csv`

**Key Functions:**

1. **Conflict Detector:**
   ```python
   def detect_conflicts(merged_df: pd.DataFrame) -> pd.DataFrame:
       """
       For each vital/timepoint with multiple sources:
       - Calculate max difference between sources
       - Flag if exceeds clinical threshold
       - Calculate conflict_magnitude
       """
   ```

2. **Completeness Calculator:**
   ```python
   def calculate_completeness_metrics(patient_vitals: Dict) -> Dict[str, float]:
       """
       Per patient:
       - completeness_phy: % of vital/timepoint grid filled by Phy
       - completeness_hnp: % filled by Hnp
       - completeness_prg: % filled by Prg
       - completeness_overall: % filled by any source
       """
   ```

3. **Temporal Precision Tracker:**
   ```python
   def calculate_time_deltas(merged_df: pd.DataFrame, grid: List[datetime]) -> pd.DataFrame:
       """
       For each measurement:
       - time_delta: hours from bin center
       - avg_time_delta: mean across all vitals for patient
       - max_time_delta: worst offset
       """
   ```

4. **Rate of Change Validator:**
   ```python
   def validate_temporal_consistency(vitals_timeseries: pd.DataFrame) -> pd.DataFrame:
       """
       Calculate rate of change between consecutive measurements.
       Flag implausible changes exceeding thresholds.
       """
   ```

5. **Outlier Detector:**
   ```python
   def detect_outliers(vitals_df: pd.DataFrame) -> pd.DataFrame:
       """
       Calculate modified Z-score for each vital.
       Flag values with modified Z > 3.5.
       """
   ```

6. **Encounter Pattern Classifier:**
   ```python
   def classify_encounter_pattern(patient_sources: Dict[str, bool]) -> str:
       """
       Based on has_outpatient_vitals, has_admission_vitals, has_inpatient_vitals:
       Return categorical encounter_pattern.
       """
   ```

**Complexity:** ⭐⭐⭐ (3/5) - Many calculations but straightforward logic

**Estimated Time:** 4-5 days

**Dependencies:** Submodule 3.6

**Output Schema:** Extends aligned_vitals_raw.h5 with all Layer 3-6 metrics

### 4.8 Submodule 3.8: Feature Engineering Pipeline

**Purpose:** Generate modeling-ready features from aligned vitals with temporal phases and derived features.

**Input:** `outputs/features/vitals_with_provenance.h5`

**Output:** `outputs/features/vitals_features_final.h5`

**Key Functions:**

1. **Temporal Phase Aggregator:**
   ```python
   def aggregate_by_temporal_phase(vitals_timeseries: pd.DataFrame) -> Dict[str, Dict]:
       """
       For each temporal phase (BASELINE, ACUTE, SUBACUTE, RECOVERY):
       Calculate per vital:
       - mean, median, min, max, std
       - first_value, last_value
       - n_measurements
       - time_coverage (hours with data / total phase hours)
       """
   ```

2. **Trajectory Calculator:**
   ```python
   def calculate_trajectory_features(vitals_timeseries: pd.DataFrame) -> Dict[str, float]:
       """
       - slope: linear regression slope across phase
       - direction: 'increasing', 'decreasing', 'stable'
       - volatility: coefficient of variation
       - range: max - min
       - time_to_normalization: hours until vital enters normal range
       """
   ```

3. **Clinical Composite Calculator:**
   ```python
   def calculate_clinical_composites(vitals: Dict[str, float]) -> Dict[str, float]:
       """
       PE-specific composite features:
       - shock_index: HR / SBP (>1.0 indicates shock)
       - modified_shock_index: HR / MAP
       - pulse_pressure: SBP - DBP (narrow suggests low cardiac output)
       - MAP: mean arterial pressure = DBP + (SBP - DBP)/3
       - delta_index: (HR - RR) (negative suggests severe PE)
       """
   ```

4. **Clinical Flag Aggregator:**
   ```python
   def aggregate_clinical_flags(vitals_timeseries: pd.DataFrame) -> Dict[str, Any]:
       """
       Per phase:
       - any_tachycardia: binary flag
       - prop_tachycardia: proportion of measurements with tachycardia
       - max_tachycardia_duration: longest continuous tachycardia (hours)
       - (same for tachypnea, hypoxemia, hypotension, fever)
       """
   ```

5. **Admission Vitals Formatter:**
   ```python
   def format_admission_features(admission_vitals: Dict) -> Dict[str, float]:
       """
       Create separate ADMISSION_* features:
       - ADMISSION_HR, ADMISSION_SBP, etc.
       - ADMISSION_shock_index
       - ADMISSION_tachycardia_flag
       - ADMISSION_hypoxemia_flag
       """
   ```

6. **Forward Fill with Decay:**
   ```python
   def forward_fill_with_decay(vitals_timeseries: pd.DataFrame, half_life_hours: float = 24) -> pd.DataFrame:
       """
       For missing values:
       - Forward-fill from last observation
       - Apply exponential decay to confidence
       - Track imputation_flag and hours_since_measurement
       """
   ```

**Complexity:** ⭐⭐⭐⭐ (4/5) - Complex feature engineering

**Estimated Time:** 6-8 days

**Dependencies:** Submodule 3.7

**Output Schema (HDF5 per patient):**
```
/patient_EMPI/
  /BASELINE/
    HR_mean: float
    HR_min: float
    HR_max: float
    HR_std: float
    HR_slope: float
    HR_n_measurements: int
    HR_time_coverage: float
    any_tachycardia: bool
    prop_tachycardia: float
    ... (all vitals)
  /ACUTE/
    ... (same structure)
  /SUBACUTE/
    ... (same structure)
  /RECOVERY/
    ... (same structure)
  /ADMISSION/
    timestamp: datetime
    HR: float
    SBP: float
    shock_index: float
    tachycardia_flag: bool
    ... (all admission vitals + composites)
  /COMPOSITES/
    shock_index_baseline_mean: float
    shock_index_acute_max: float
    ... (all composites per phase)
  /QUALITY/
    completeness_overall: float
    encounter_pattern: str
    avg_time_delta: float
    n_conflicts: int
    ... (all Layer 3-6 metrics)
```

### 4.9 Submodule 3.9: Validation Framework

**Purpose:** Implement 4-tier validation strategy to ensure extraction accuracy ≥90%.

**Input:** All intermediate outputs from Submodules 3.1-3.8

**Output:** `outputs/validation/validation_report.html`, `outputs/validation/manual_review_cases.csv`

**Key Functions:**

**Tier 1: Cross-Validation with Structured Data**

```python
def cross_validate_with_phy(hnp_vitals: pd.DataFrame, prg_vitals: pd.DataFrame,
                            phy_vitals: pd.DataFrame) -> Dict[str, Any]:
    """
    For vitals extracted from notes that also exist in structured Phy.txt:
    1. Match on patient + date (±1 hour window)
    2. Calculate agreement metrics:
       - Clinical agreement (within tolerance)
       - Pearson correlation
       - MAE, MAPE
       - Bland-Altman bias and limits
    3. Identify high-discrepancy cases for review

    Returns comprehensive metrics per vital type.
    Target: ≥90% clinical agreement
    """
```

**Tier 2: Strategic Manual Review**

```python
def generate_stratified_sample_for_review(all_vitals: pd.DataFrame,
                                         cross_val_results: Dict) -> pd.DataFrame:
    """
    Generate stratified sample of 200 notes:
    - 50 high-discrepancy cases (cross-val failures)
    - 75 no structured match (validate uncoverable cases)
    - 40 critical values (extreme vitals)
    - 35 edge cases (negation, ranges, unusual formatting)

    Export to CSV for manual annotation in REDCap.
    """

def calculate_inter_rater_reliability(annotations: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate Cohen's kappa (binary) and ICC (continuous).
    Target: κ ≥ 0.80, ICC ≥ 0.90
    """

def create_error_taxonomy(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize errors:
    - False negatives (missed values)
    - False positives (hallucinated values)
    - Wrong value extracted
    - Correct value, wrong context
    - Unit errors
    - Decimal point errors
    - Systolic/diastolic confusion

    Returns prioritized list for pattern refinement.
    """
```

**Tier 3: Statistical Monitoring**

```python
def validate_distributions(extracted_vitals: pd.DataFrame,
                          reference_stats: Dict) -> Dict[str, Any]:
    """
    Compare extracted vitals to reference distributions:
    - KS test vs. MIMIC-III or published PE cohorts
    - Check digit preference
    - Flag out-of-range values
    - Calculate proportion of statistical outliers

    Target: <1% out-of-range, <2% outliers
    """

def validate_temporal_plausibility(vitals_timeseries: pd.DataFrame) -> Dict[str, Any]:
    """
    Check consecutive measurements:
    - Calculate rate of change per hour
    - Flag implausible transitions
    - Detect suspicious duplicate runs

    Target: <1% implausible transitions
    """
```

**Tier 4: Pattern-Specific Validation**

```python
def validate_negation_handling(notes: pd.DataFrame, extracted_vitals: pd.DataFrame) -> float:
    """
    Find notes with negation phrases.
    Check if extraction incorrectly produced values.

    Returns false positive rate.
    Target: <5%
    """

def validate_range_handling(extracted_vitals: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify range extractions.
    Verify decision rule (midpoint/min/max) applied consistently.
    """

def validate_unit_conversions(extracted_vitals: pd.DataFrame) -> float:
    """
    Check all explicit unit mentions.
    Verify conversions mathematically correct.

    Returns accuracy.
    Target: 100%
    """
```

**Integrated Validation Report Generator:**

```python
def generate_comprehensive_validation_report(all_validation_results: Dict) -> str:
    """
    Generate HTML report with:
    - Executive summary (pass/fail for each vital)
    - Cross-validation metrics (tables + Bland-Altman plots)
    - Manual review results (accuracy, IRR, error taxonomy)
    - Distribution validation (histograms, KS test results)
    - Temporal validation (rate of change plots)
    - Pattern-specific results
    - Comparison to published benchmarks
    - Limitations and recommended improvements

    Save as outputs/validation/validation_report.html
    """
```

**Complexity:** ⭐⭐⭐⭐ (4/5) - Requires careful statistical analysis and visualization

**Estimated Time:** 8-10 days (including manual review time)

**Dependencies:** All previous submodules

**Validation Success Criteria:**

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Cross-validation clinical agreement | ≥90% | ≥95% |
| Manual review accuracy | ≥90% | ≥93% |
| Inter-rater reliability (κ) | ≥0.80 | ≥0.85 |
| Out-of-range values | <1% | <0.5% |
| Implausible transitions | <1% | <0.5% |
| Negation false positives | <5% | <3% |

### 4.10 Submodule 3.10: Main Orchestrator

**Purpose:** Coordinate execution of all submodules with proper sequencing, checkpointing, and logging.

**Input:** Configuration file `config/vitals_config.yaml`

**Output:** Final feature sets + comprehensive logs

**Key Functions:**

```python
def run_full_pipeline(config: Dict) -> None:
    """
    Execute all submodules in dependency order:

    Phase 1: Extraction (parallel)
    - 3.1: Extract Phy.txt
    - 3.2: Extract Hnp.csv
    - 3.3: Extract Prg.csv

    Phase 2: Harmonization (sequential)
    - 3.4: Harmonize vitals
    - 3.5: QC and unit conversion

    Phase 3: Integration (sequential)
    - 3.6: Temporal alignment
    - 3.7: Provenance metrics
    - 3.8: Feature engineering

    Phase 4: Validation
    - 3.9: 4-tier validation

    With checkpointing: can resume from any completed submodule
    """

def run_partial_pipeline(start_submodule: str, end_submodule: str, config: Dict) -> None:
    """Enable running subset of pipeline for debugging"""

def validate_intermediate_outputs(submodule_name: str) -> bool:
    """Verify each submodule output before proceeding"""

class ProgressTracker:
    """Track progress with detailed logging and time estimates"""
```

**Complexity:** ⭐⭐ (2/5) - Coordination logic, not algorithmically complex

**Estimated Time:** 3-4 days

**Dependencies:** All submodules

---

## 5. Implementation Phases and Timeline

### Phase Structure

The implementation is broken into **4 major phases** that can be executed sequentially while allowing parallel work within phases.

```
PHASE 1: Foundation (Weeks 1-3)
├── 3.1: Structured extraction (Phy.txt)
├── 3.4: Harmonization framework
└── 3.5: QC & unit conversion

PHASE 2: NLP Extraction (Weeks 3-5, parallel to Phase 1 completion)
├── 3.2: H&P extraction (Hnp.csv)
└── 3.3: Progress notes extraction (Prg.csv)

PHASE 3: Integration (Weeks 6-7)
├── 3.6: Temporal alignment
├── 3.7: Provenance metrics
└── 3.8: Feature engineering

PHASE 4: Validation (Weeks 8-10)
├── 3.9: 4-tier validation
└── 3.10: Orchestration & final testing
```

### Detailed Timeline

**Week 1: Foundation - Structured Data**
- Days 1-2: Submodule 3.1 (Phy.txt extraction)
- Days 3-5: Submodule 3.4 (Harmonization framework)
- **Deliverable:** Structured vitals extracted and harmonized

**Week 2: Foundation - QC & Parallel NLP Start**
- Days 1-3: Submodule 3.5 (QC & unit conversion)
- Days 4-5: Begin Submodule 3.2 (H&P extraction - pattern development)
- **Deliverable:** QC pipeline working on structured data

**Week 3: NLP Extraction - H&P Notes**
- Days 1-5: Complete Submodule 3.2 (H&P extraction)
- Begin Submodule 3.3 (Progress notes extraction)
- **Deliverable:** H&P vitals extraction complete with preliminary validation

**Week 4: NLP Extraction - Progress Notes**
- Days 1-5: Continue Submodule 3.3 (Progress notes extraction)
- **Deliverable:** Progress notes extraction working on sample

**Week 5: NLP Refinement**
- Days 1-5: Complete Submodule 3.3 full-scale extraction
- Refine patterns based on initial validation results
- **Deliverable:** All three sources extracted

**Week 6: Integration - Alignment**
- Days 1-5: Submodule 3.6 (Temporal alignment & multi-source merging)
- **Deliverable:** All sources aligned to temporal grid

**Week 7: Integration - Provenance & Features**
- Days 1-2: Submodule 3.7 (Provenance metrics)
- Days 3-5: Submodule 3.8 (Feature engineering)
- **Deliverable:** Final feature set with full provenance

**Week 8: Validation - Tier 1 & 2**
- Days 1-3: Tier 1 cross-validation
- Days 4-5: Generate stratified sample for manual review
- **Deliverable:** Cross-validation complete, sample ready for review

**Week 9: Validation - Manual Review**
- Days 1-5: Dual independent annotation (200 notes)
- **Deliverable:** Manual review complete with IRR analysis

**Week 10: Validation - Statistical & Finalization**
- Days 1-2: Tier 3 & 4 validation (statistical, pattern-specific)
- Days 3-4: Generate comprehensive validation report
- Day 5: Submodule 3.10 final orchestration testing
- **Deliverable:** Complete validated Module 3 with documentation

**Total Duration:** 10 weeks (2.5 months)

**Contingency:** Built-in 2-week buffer for pattern refinement if validation targets not met initially

---

## 6. File Structure

```
module_3_vitals_processing/
│
├── config/
│   ├── vitals_config.yaml              # Main configuration
│   ├── harmonization_map.json          # Concept mappings
│   └── qc_thresholds.json              # QC thresholds per vital
│
├── extractors/
│   ├── __init__.py
│   ├── phy_extractor.py                # Submodule 3.1
│   ├── hnp_nlp_extractor.py            # Submodule 3.2
│   ├── prg_nlp_extractor.py            # Submodule 3.3
│   ├── patterns.py                     # Regex pattern library
│   └── negation_handler.py             # Negation detection
│
├── processing/
│   ├── __init__.py
│   ├── harmonizer.py                   # Submodule 3.4
│   ├── unit_converter.py               # Submodule 3.5
│   ├── qc_filter.py                    # Submodule 3.5
│   ├── temporal_aligner.py             # Submodule 3.6
│   ├── provenance_calculator.py        # Submodule 3.7
│   └── feature_engineer.py             # Submodule 3.8
│
├── validation/
│   ├── __init__.py
│   ├── cross_validator.py              # Tier 1
│   ├── manual_review_sampler.py        # Tier 2
│   ├── statistical_validator.py        # Tier 3
│   ├── pattern_validator.py            # Tier 4
│   └── report_generator.py             # HTML report
│
├── utils/
│   ├── __init__.py
│   ├── io_utils.py                     # HDF5 I/O helpers
│   ├── temporal_utils.py               # Datetime handling
│   ├── logging_utils.py                # Logging configuration
│   └── visualization_utils.py          # Plotting for reports
│
├── tests/
│   ├── test_phy_extractor.py
│   ├── test_hnp_extractor.py
│   ├── test_prg_extractor.py
│   ├── test_harmonizer.py
│   ├── test_unit_converter.py
│   ├── test_qc_filter.py
│   ├── test_temporal_aligner.py
│   ├── test_provenance.py
│   ├── test_feature_engineer.py
│   └── test_validation.py
│
├── outputs/
│   ├── discovery/
│   │   ├── phy_vitals_raw.parquet
│   │   ├── hnp_vitals_raw.parquet
│   │   ├── prg_vitals_raw.parquet
│   │   ├── harmonization_map.json
│   │   ├── harmonized_vitals.parquet
│   │   ├── qc_vitals.parquet
│   │   └── qc_report.csv
│   ├── features/
│   │   ├── aligned_vitals_raw.h5
│   │   ├── vitals_with_provenance.h5
│   │   └── vitals_features_final.h5
│   └── validation/
│       ├── cross_validation_results.json
│       ├── manual_review_cases.csv
│       ├── manual_review_annotations.csv
│       ├── statistical_validation.json
│       ├── pattern_validation.json
│       └── validation_report.html
│
├── docs/
│   ├── ARCHITECTURE.md                 # This document
│   ├── API.md                          # Function documentation
│   ├── VALIDATION_PROTOCOL.md          # Detailed validation procedures
│   └── USER_GUIDE.md                   # How to run the pipeline
│
├── module_03_vitals_processing.py      # Main entry point (Submodule 3.10)
├── requirements.txt
└── README.md
```

---

## 7. Data Formats & Schemas

### 7.1 Intermediate Data Format (Parquet)

All discovery phase outputs use Apache Parquet for efficient columnar storage:

```python
# phy_vitals_raw.parquet, hnp_vitals_raw.parquet, prg_vitals_raw.parquet
{
    'EMPI': str,
    'timestamp': datetime,
    'vital_type': str,           # HR, SBP, DBP, RR, SPO2, TEMP, WEIGHT, HEIGHT, BMI
    'value': float,
    'units': str,
    'source': str,               # 'phy', 'hnp', 'prg'
    'encounter_type': str,       # 'outpatient', 'inpatient', 'admission'
    'value_type': str,           # 'exact', 'range_min', 'range_max', 'range_mean', 'narrative'
    'confidence': float,         # 0.0-1.0 (1.0 for structured, <1.0 for NLP)
    'extraction_flags': str,     # JSON string of any extraction warnings
}
```

### 7.2 Final Feature Format (HDF5)

HDF5 provides efficient storage for large arrays with hierarchical organization:

```python
# vitals_features_final.h5

# Group structure per patient:
/EMPI_12345/

  # Raw source-specific vitals (Layer 1)
  /raw_vitals_phy/
    timestamp: np.array(datetime64)
    HR: np.array(float32)
    SBP: np.array(float32)
    DBP: np.array(float32)
    RR: np.array(float32)
    SPO2: np.array(float32)
    TEMP: np.array(float32)

  /raw_vitals_hnp/
    [same structure]

  /raw_vitals_prg/
    [same structure]

  # Merged vitals with provenance (Layer 2)
  /merged_vitals/
    timestamp: np.array(datetime64)
    HR: np.array(float32)
    HR_source: np.array(str)
    HR_n_sources: np.array(int8)
    [repeat for all vitals]

  # Temporal phases (Layer 1-2 aggregated)
  /BASELINE/
    HR_mean: float
    HR_min: float
    HR_max: float
    HR_std: float
    HR_first: float
    HR_last: float
    HR_n_measurements: int
    HR_time_coverage: float
    HR_slope: float
    HR_direction: str
    HR_volatility: float
    HR_range: float
    [repeat for all vitals]

    # Clinical flags
    any_tachycardia: bool
    prop_tachycardia: float
    max_tachycardia_duration_hours: float
    [repeat for all clinical flags]

  /ACUTE/
    [same structure as BASELINE]

  /SUBACUTE/
    [same structure as BASELINE]

  /RECOVERY/
    [same structure as BASELINE]

  # Special: Admission vitals (Layer 1-2)
  /ADMISSION/
    timestamp: datetime64
    HR: float
    SBP: float
    DBP: float
    RR: float
    SPO2: float
    TEMP: float
    shock_index: float
    pulse_pressure: float
    MAP: float
    tachycardia_flag: bool
    tachypnea_flag: bool
    hypoxemia_flag: bool
    hypotension_flag: bool
    fever_flag: bool

  # Composite features
  /COMPOSITES/
    shock_index_baseline_mean: float
    shock_index_acute_max: float
    pulse_pressure_baseline_mean: float
    MAP_acute_min: float
    [all composites per phase]

  # Quality metrics (Layers 3-6)
  /QUALITY/
    # Layer 3: Conflicts & Completeness
    completeness_phy: float
    completeness_hnp: float
    completeness_prg: float
    completeness_overall: float
    avg_sources_per_vital: float
    redundancy_score: float
    n_conflicts: int
    conflict_rate: float

    # Layer 4: Temporal precision
    avg_time_delta: float
    max_time_delta: float

    # Layer 5: Temporal consistency
    n_implausible_changes: int
    implausible_change_rate: float
    n_outliers: int
    outlier_rate: float

    # Layer 6: Encounter patterns
    has_outpatient_vitals: bool
    has_admission_vitals: bool
    has_inpatient_vitals: bool
    encounter_pattern: str  # categorical
```

### 7.3 Configuration Format (YAML)

```yaml
# config/vitals_config.yaml

data_paths:
  phy_file: "/home/moin/TDA_11_1/Data/FNR_20240409_091633_Phy.txt"
  hnp_file: "/home/moin/TDA_11_1/Data/Hnp.csv"
  prg_file: "/home/moin/TDA_11_1/Data/Prg.csv"
  patient_timelines: "/home/moin/TDA_11_1/module_1_core_infrastructure/outputs/patient_timelines.pkl"

output_paths:
  discovery_dir: "outputs/discovery"
  features_dir: "outputs/features"
  validation_dir: "outputs/validation"

extraction:
  # Which sources to include
  use_phy: true
  use_hnp: true
  use_prg: true

  # NLP extraction parameters
  hnp_confidence_threshold: 0.7
  prg_confidence_threshold: 0.7
  enable_narrative_extraction: true  # Extract from descriptive text

  # Chunk size for large file processing
  chunk_size: 100000

harmonization:
  vitals_to_extract:
    - HR
    - SBP
    - DBP
    - RR
    - SPO2
    - TEMP
    - WEIGHT
    - HEIGHT
    - BMI

unit_conversion:
  standard_units:
    TEMP: "celsius"
    WEIGHT: "kg"
    HEIGHT: "cm"
    BP: "mmHg"
    HR: "bpm"
    RR: "breaths_per_min"
    SPO2: "percent"

  auto_detect: true  # Auto-detect units from value ranges

qc_thresholds:
  # Physiological ranges (permissive)
  HR: [20, 250]
  SBP: [50, 300]
  DBP: [30, 200]
  RR: [4, 60]
  SPO2: [50, 100]
  TEMP: [30, 45]  # Celsius
  WEIGHT: [20, 300]  # kg
  HEIGHT: [100, 250]  # cm

clinical_thresholds:
  # For binary flags
  fever: 38.0  # Celsius
  hypothermia: 36.0
  tachycardia: 100
  bradycardia: 60
  tachypnea: 20
  bradypnea: 12
  hypoxemia: 90
  hypotension: 90  # SBP
  severe_hypotension: 70
  hypertension: 180

temporal_alignment:
  temporal_phases:
    BASELINE:
      start_days: -365
      end_days: -30
      resolution: "1D"  # Daily
    PRE_ACUTE:
      start_days: -30
      end_days: -7
      resolution: "1D"
    ACUTE:
      start_days: -7
      end_days: 1
      resolution: "1H"  # Hourly
    HIGH_RES_ACUTE:
      start_hours: -24
      end_hours: 24
      resolution: "5T"  # 5 minutes
    SUBACUTE:
      start_days: 2
      end_days: 14
      resolution: "1H"
    RECOVERY:
      start_days: 15
      end_days: 90
      resolution: "1D"

  # Hierarchical merge priority
  source_priority: ['prg', 'hnp', 'phy']

  # Time window for matching measurements from different sources
  merge_window_hours: 1.0

provenance:
  # Conflict detection thresholds
  conflict_thresholds:
    HR: 10
    RR: 10
    SBP: 20
    DBP: 20
    TEMP: 0.5
    SPO2: 5

  # Outlier detection
  outlier_modified_zscore_threshold: 3.5

  # Implausible rate of change (per hour)
  implausible_rate_thresholds:
    HR: 50
    SBP: 30
    DBP: 30
    RR: 10
    TEMP: 2
    SPO2: 20

feature_engineering:
  # Forward fill with exponential decay
  forward_fill: true
  forward_fill_half_life_hours: 24

  # Trajectory features
  calculate_slopes: true
  calculate_volatility: true

  # Clinical composites
  calculate_shock_index: true
  calculate_pulse_pressure: true
  calculate_MAP: true

validation:
  # Tier 1: Cross-validation
  cross_val_time_window_hours: 1.0
  cross_val_clinical_agreement_target: 0.90

  # Tier 2: Manual review
  manual_review_sample_size: 200
  manual_review_strata:
    high_discrepancy: 50
    no_structured_match: 75
    critical_values: 40
    edge_cases: 35

  # Tier 3: Statistical monitoring
  out_of_range_threshold: 0.01  # 1%
  outlier_threshold: 0.02  # 2%
  implausible_transition_threshold: 0.01  # 1%

  # Tier 4: Pattern-specific
  negation_false_positive_threshold: 0.05  # 5%

logging:
  level: "INFO"
  log_file: "outputs/vitals_processing.log"
  log_to_console: true

performance:
  n_jobs: -1  # Use all CPU cores
  use_dask: true  # For large file processing
```

---

## 8. Key Design Decisions & Rationale

### 8.1 Why Three Data Sources?

**Decision:** Extract from structured (Phy.txt) AND unstructured (Hnp.csv, Prg.csv) sources.

**Rationale:**
- **Complementary coverage**: Each source captures vitals the others miss
- **Clinical context**: Admission vitals (Hnp) are critical for PE severity assessment
- **Temporal resolution**: Progress notes (Prg) provide high-frequency serial measurements
- **Cross-validation**: Overlap between sources enables validation

**Trade-off:** Increased complexity and computational cost, but significantly improved data completeness and quality.

### 8.2 Why Six-Layer Architecture?

**Decision:** Preserve raw source-specific data alongside merged data with full provenance.

**Rationale:**
- **Reproducibility**: Can reprocess with different merge strategies without re-extraction
- **Debugging**: Can trace any measurement back to original source
- **Interpretability**: Source patterns (e.g., "outpatient_only") are predictive features
- **Quality assurance**: Conflicts and quality metrics inform model confidence

**Trade-off:** Larger storage footprint (~3x), but storage is cheap and information is invaluable.

### 8.3 Why Hierarchical Merge (Prg > Hnp > Phy)?

**Decision:** Prioritize inpatient > admission > outpatient data.

**Rationale:**
- **Recency**: Inpatient vitals are typically more recent (closer to index time)
- **Frequency**: Inpatient monitoring is more frequent (hourly vs daily)
- **Severity**: Patients requiring inpatient monitoring are sicker (measurements more relevant)
- **Quality**: Inpatient vitals typically use calibrated equipment with trained staff

**Alternative considered:** No hierarchy, keep all measurements. Rejected because it creates duplicate features and doesn't provide single clean dataset for modeling.

### 8.4 Why Maximum Temporal Resolution (5 minutes)?

**Decision:** Extract at finest granularity available, bin to 5-minute resolution in acute phase.

**Rationale:**
- **PE dynamics**: Vital signs can change rapidly during acute PE (shock, cardiac arrest)
- **Treatment response**: Can capture immediate effects of interventions (oxygen, fluids, anticoagulation)
- **Flexibility**: Can always downsample to coarser resolution, but can't upsample from daily bins
- **Clinical reality**: ICU monitoring is often every 5-15 minutes

**Trade-off:** Larger datasets, but no computational constraints per user requirements.

### 8.5 Why Advanced NLP Over Simple Regex?

**Decision:** Use context-aware NLP with negation detection, not just simple pattern matching.

**Rationale:**
- **Accuracy**: Simple regex achieves ~70-75% accuracy; context-aware achieves ~85-90%
- **False positives**: Negation ("blood pressure not obtained") must be handled
- **Context**: Must distinguish current vs. prior vs. home vitals
- **Publication**: Rigorous validation required for publication in medical journals

**Alternative considered:** Medical NER models (spaCy, scispaCy). May implement if regex accuracy insufficient.

### 8.6 Why Separate Admission Vitals Features?

**Decision:** Create distinct ADMISSION_* features from first H&P note, not just merge into ACUTE phase.

**Rationale:**
- **Clinical significance**: Presentation severity is distinct from average acute phase vitals
- **Temporal precision**: Admission vitals have exact timestamp at disease presentation
- **Predictive value**: Initial severity is highly predictive of outcomes
- **Clinical standards**: PESI and sPESI scores use admission vitals, not acute averages

### 8.7 Why 4-Tier Validation?

**Decision:** Implement comprehensive validation with cross-validation, manual review, statistical monitoring, and pattern-specific checks.

**Rationale:**
- **No single method sufficient**: Cross-validation only covers ~30-40% of extractions
- **Complementary strengths**: Each tier catches different error types
- **Publication requirements**: Medical journals require rigorous validation with multiple independent methods
- **Confidence**: Multiple lines of evidence provide confidence in extraction accuracy

**Trade-off:** Significant time investment (2+ weeks), but essential for research credibility.

### 8.8 Why Modular Design?

**Decision:** Break into 10 independent submodules rather than monolithic pipeline.

**Rationale:**
- **Parallel development**: Multiple developers can work simultaneously
- **Testing**: Each submodule can be unit tested independently
- **Debugging**: Easier to isolate and fix issues
- **Flexibility**: Can swap implementations (e.g., upgrade NLP) without rewriting entire pipeline
- **Checkpointing**: Can resume from any submodule if process interrupted

**Alternative considered:** Single script. Rejected because it would be 5000+ lines, untestable, and fragile.

---

## 9. Success Criteria

### 9.1 Data Quality Metrics

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Extraction accuracy (cross-validation) | ≥90% | ≥95% |
| Manual review accuracy | ≥90% | ≥93% |
| Data completeness (overall) | ≥80% | ≥90% |
| Out-of-range values | <1% | <0.5% |
| Implausible temporal transitions | <1% | <0.5% |

### 9.2 Coverage Metrics

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Patients with any vitals | ≥95% | ≥98% |
| Patients with admission vitals | ≥70% | ≥80% |
| Patients with high-res acute vitals | ≥50% | ≥60% |
| Vitals per patient (median) | ≥100 | ≥200 |

### 9.3 Validation Metrics

| Validation Tier | Target | Stretch Goal |
|----------------|--------|-------------|
| Cross-validation clinical agreement | ≥90% | ≥95% |
| Inter-rater reliability (κ) | ≥0.80 | ≥0.85% |
| Negation handling (false positive rate) | <5% | <3% |
| Unit conversion accuracy | 100% | 100% |

### 9.4 Performance Metrics

| Metric | Target | Stretch Goal |
|--------|--------|-------------|
| Phy.txt extraction time | <30 min | <15 min |
| Hnp.csv extraction time | <2 hours | <1 hour |
| Prg.csv extraction time | <8 hours | <4 hours |
| Full pipeline end-to-end | <12 hours | <6 hours |

---

## 10. Risks & Mitigations

### Risk 1: NLP Extraction Accuracy Below Target

**Risk:** Regex patterns achieve <90% accuracy on manual review.

**Likelihood:** Medium
**Impact:** High (blocks publication)

**Mitigation:**
- Iterative pattern refinement based on error taxonomy
- If regex insufficient, implement medical NER (spaCy, scispaCy)
- Worst case: Use only structured Phy.txt data (reduces coverage but maintains quality)

### Risk 2: Progress Notes Extraction Too Slow

**Risk:** 8.7M notes take >24 hours to process.

**Likelihood:** Medium
**Impact:** Low (user has no computational constraints)

**Mitigation:**
- Use Dask for parallel processing across CPU cores
- Implement smart filtering (only process notes from PE patients)
- If still too slow, sample high-risk patients or recent years only

### Risk 3: Cross-Validation Coverage Too Low

**Risk:** <30% of note extractions have matching structured data.

**Likelihood:** Medium
**Impact:** Medium (limits validation confidence)

**Mitigation:**
- Rely more heavily on manual review (increase sample to 300-400 notes)
- Use statistical validation as additional evidence
- Document limitation transparently in methods section

### Risk 4: High Conflict Rate Between Sources

**Risk:** >20% of measurements show conflicts beyond clinical thresholds.

**Likelihood:** Low
**Impact:** Medium (complicates merge strategy)

**Mitigation:**
- Investigate conflict patterns (systematic bias vs. random error)
- Implement confidence weighting (de-weight low-confidence NLP extractions)
- Keep all source-specific measurements (don't force merge)

### Risk 5: Manual Review Timeline Extends Beyond 1 Week

**Risk:** Annotation of 200 notes takes >2 weeks.

**Likelihood:** Medium
**Impact:** Low (timeline has 2-week buffer)

**Mitigation:**
- Recruit additional reviewers if needed
- Implement REDCap annotation interface to streamline review
- Reduce sample size to 150 if necessary (still statistically sufficient)

---

## 11. Future Enhancements

### Post-V1.0 Improvements

1. **Medical NER Integration**
   - Implement scispaCy or MedCAT for entity recognition
   - Expected accuracy improvement: 90% → 95%

2. **Vital Sign Waveform Extraction**
   - If waveform data available, extract continuous measurements
   - Enables heart rate variability, respiratory waveforms

3. **Multi-Modal Feature Integration**
   - Combine vitals with labs, medications, procedures
   - Joint modeling of physiological state

4. **Temporal Attention Models**
   - Use transformer architectures for temporal pattern recognition
   - Learn which timepoints are most predictive

5. **Automated Quality Feedback Loop**
   - Use model predictions to identify suspicious extractions
   - Iteratively improve NLP patterns based on model uncertainty

6. **Real-Time Extraction**
   - Adapt pipeline for real-time clinical decision support
   - Extract from live EHR streams

---

## 12. Appendices

### Appendix A: Canonical Vital Sign Names

| Canonical Name | Description | Units | Normal Range |
|----------------|-------------|-------|-------------|
| HR | Heart Rate | bpm | 60-100 |
| SBP | Systolic Blood Pressure | mmHg | 90-140 |
| DBP | Diastolic Blood Pressure | mmHg | 60-90 |
| RR | Respiratory Rate | breaths/min | 12-20 |
| SPO2 | Oxygen Saturation | % | 95-100 |
| TEMP | Temperature | Celsius | 36.5-37.5 |
| WEIGHT | Body Weight | kg | 40-150 |
| HEIGHT | Body Height | cm | 140-200 |
| BMI | Body Mass Index | kg/m² | 18.5-25 |

### Appendix B: Source Priority Rationale Table

| Scenario | Phy.txt | Hnp.csv | Prg.csv | Selected | Rationale |
|----------|---------|---------|---------|----------|-----------|
| All three present | 88 bpm @ -2h | 92 bpm @ -0.1h | 95 bpm @ +0.5h | Prg (95) | Most recent, inpatient |
| Phy + Hnp | 88 bpm @ -2h | 92 bpm @ -0.1h | - | Hnp (92) | Closer to index time |
| Phy + Prg | 88 bpm @ -2h | - | 95 bpm @ +0.5h | Prg (95) | Inpatient > outpatient |
| Hnp + Prg | - | 92 bpm @ -0.1h | 95 bpm @ +0.5h | Prg (95) | Inpatient > admission |
| Only Phy | 88 bpm @ -2h | - | - | Phy (88) | Only source available |

### Appendix C: Clinical Composite Feature Formulas

```python
# Shock Index (SI)
# Normal: <0.7; Elevated: 0.7-1.0; Shock: >1.0
shock_index = HR / SBP

# Modified Shock Index (MSI)
# Uses MAP instead of SBP for better hemodynamic assessment
MAP = DBP + (SBP - DBP) / 3
modified_shock_index = HR / MAP

# Pulse Pressure (PP)
# Normal: 40-60 mmHg; Narrow: <25 (low cardiac output)
pulse_pressure = SBP - DBP

# Mean Arterial Pressure (MAP)
# Target: >65 mmHg for organ perfusion
MAP = DBP + (SBP - DBP) / 3

# Delta Index
# Novel PE indicator; negative values suggest severe PE
delta_index = HR - RR

# Alveolar-arterial Oxygen Gradient (estimated)
# Requires ABG data if available; otherwise estimate from SpO2
# A-a gradient = 150 - (SpO2_adjusted)
```

### Appendix D: Temporal Phase Clinical Rationale

| Phase | Time Window | Resolution | Clinical Significance |
|-------|-------------|-----------|----------------------|
| BASELINE | [-365d, -30d] | Daily | Pre-existing conditions; baseline vital stability |
| PRE_ACUTE | [-30d, -7d] | Daily | Prodromal symptoms; early warning signs |
| ACUTE | [-7d, +1d] | Hourly | Disease presentation; acute changes |
| HIGH_RES_ACUTE | [-24h, +24h] | 5 minutes | Immediate perioperative period; rapid changes |
| SUBACUTE | [+2d, +14d] | Hourly | Treatment response; clinical improvement/deterioration |
| RECOVERY | [+15d, +90d] | Daily | Long-term recovery; return to baseline |

### Appendix E: Error Taxonomy with Examples

| Error Type | Example | Root Cause | Fix |
|------------|---------|-----------|-----|
| False Negative | "HR 88" not extracted | Pattern mismatch | Add pattern: `(\d{2})\s*(?=bpm\|BP)` |
| False Positive | "88-year-old" extracted as HR | Overly broad pattern | Add context check: not preceded by "-year" |
| Wrong Context | Extracted "prior HR 88" as current | No historical detection | Add negation: "prior", "previous", "home" |
| Unit Error | 98.6°F → 98.6°C | Failed unit detection | Check value range: 95-108 → Fahrenheit |
| Decimal Error | 37.5°C → 375°C | Missing decimal in text | Add pattern: `(\d{2})(\d)\s*[CF]` → insert decimal |
| Sys/Dia Swap | 68/124 extracted as SBP=68 | Transposed in note | Validate: if SBP < DBP, swap |
| Range Confusion | "HR 70-90" → 70 only | Took first value | Extract both, use mean |

---

## 13. Unified Extraction Architecture (v3.3 Update)

As of version 3.3, the NLP extraction system has been refactored to use a **unified pattern library** with centralized extraction logic.

### 13.1 Architecture Overview

```
unified_patterns.py          unified_extractor.py         Source Wrappers
┌──────────────────┐        ┌────────────────────┐       ┌─────────────────┐
│ HR_PATTERNS      │───────▶│ extract_heart_rate │◀──────│ hnp_extractor   │
│ BP_PATTERNS      │───────▶│ extract_bp         │       │ (sections,      │
│ RR_PATTERNS      │───────▶│ extract_rr         │       │  timestamps)    │
│ SPO2_PATTERNS    │───────▶│ extract_spo2       │       ├─────────────────┤
│ TEMP_PATTERNS    │───────▶│ extract_temp       │◀──────│ prg_extractor   │
│ O2_FLOW_PATTERNS │───────▶│ extract_o2_flow    │       │ (checkpoints,   │
│ O2_DEVICE_PTRNS  │───────▶│ extract_o2_device  │       │  temp methods)  │
│ BMI_PATTERNS     │───────▶│ extract_bmi        │       └─────────────────┘
│ VALID_RANGES     │        │                    │
│ NEGATION_PTRNS   │        │ check_negation()   │
│ SKIP_SECTION_PTR │        │ is_in_skip_sect()  │
└──────────────────┘        └────────────────────┘
```

### 13.2 3-Tier Confidence Scoring

Each pattern has an associated confidence score reflecting extraction reliability:

| Tier | Confidence Range | Description | Example |
|------|------------------|-------------|---------|
| **Standard** | 0.90-1.0 | Explicit label + unit | `HR: 72 bpm` |
| **Optimized** | 0.80-0.90 | Label or strong context | `tachycardic at 120` |
| **Specialized** | 0.65-0.80 | Contextual/bare patterns | `VS... 120` |

### 13.3 Key Features

1. **Negation Detection**: 8 patterns (e.g., "not obtained", "refused", "unable to measure")
2. **Skip Section Filtering**: Excludes allergies, medications, history sections
3. **Temperature Normalization**: All values converted to Celsius
4. **Position-Based Deduplication**: Prevents duplicate extractions from overlapping patterns
5. **Physiological Validation**: Rejects implausible values (HR=500, SpO2=200)
6. **Abnormal Flagging**: Marks values outside clinical norms

### 13.4 Pattern Counts

- **Core Vitals**: HR (16), BP (16), RR (12), SpO2 (11), Temp (10)
- **Supplemental**: O2 Flow (9), O2 Device (10), BMI (7)
- **Total**: 91 patterns

### 13.5 Benefits of Unified Architecture

1. **Single Source of Truth**: All patterns in one file, easier maintenance
2. **Consistent Behavior**: Same validation logic across all extractors
3. **Reduced Code Duplication**: ~340 lines removed from hnp_extractor
4. **Easier Testing**: 54 dedicated tests for unified components
5. **Extensibility**: Add new vitals by adding patterns + extraction function

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1 | 2025-12-13 | Claude (Opus 4.5) | Added Section 13: Unified Extraction Architecture |
| 1.0 | 2025-11-09 | Claude (Sonnet 4.5) | Initial architecture design based on comprehensive requirements |

---

## Acknowledgments

This architecture design incorporates:
- User requirements from Q1-Q10 specification
- Module 2 (Laboratory Processing) design patterns
- Clinical NLP best practices from medical informatics literature
- PE prediction domain expertise
- TDA 11.1 project data structures and conventions

---

**END OF ARCHITECTURE DOCUMENT**

Total Pages: ~35
Total Words: ~12,000
