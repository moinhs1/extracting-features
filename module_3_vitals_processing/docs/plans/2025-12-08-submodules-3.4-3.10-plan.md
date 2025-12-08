# Submodules 3.4-3.10 Implementation Plan

**Date:** 2025-12-08
**Status:** Planning
**Prerequisites:** Submodules 3.1-3.3 extraction outputs

---

## Overview

After completing extraction (3.1-3.3), the remaining submodules process, merge, and validate vital signs data to produce modeling-ready features.

### Dependencies

```
Extraction Outputs (3.1-3.3)
         │
         ▼
┌─────────────────┐
│ 3.4 Harmonizer  │ ← Merge 3 sources, canonical names
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.5 QC & Units  │ ← Standardize units, filter implausible
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.6 Temporal    │ ← Align to patient timelines, bin to grid
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.7 Provenance  │ ← Track source, detect conflicts, quality metrics
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.8 Features    │ ← Aggregates, trajectories, composites
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.9 Validation  │ ← 4-tier validation framework
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3.10 Orchestrate│ ← Main pipeline coordinator
└─────────────────┘
```

---

## Submodule 3.4: Vitals Harmonizer

### Purpose
Merge all three extraction sources into a single standardized format with canonical vital sign names.

### Inputs
- `outputs/discovery/phy_vitals_raw.parquet` (from 3.1)
- `outputs/discovery/hnp_vitals_raw.parquet` (from 3.2)
- `outputs/discovery/prg_vitals_raw.parquet` (from 3.3)

### Outputs
- `outputs/discovery/harmonized_vitals.parquet`
- `config/harmonization_map.json` (concept mappings)

### Key Functions
```python
def merge_extraction_sources(phy_df, hnp_df, prg_df) -> pd.DataFrame:
    """Combine all sources with source attribution."""

def apply_hierarchical_priority(df) -> pd.DataFrame:
    """Apply priority: Prg > Hnp > Phy for conflicts within same time window."""

def map_to_canonical_names(df, mapping) -> pd.DataFrame:
    """Map all vital variants to canonical names: HR, SBP, DBP, RR, SPO2, TEMP."""

def deduplicate_vitals(df) -> pd.DataFrame:
    """Remove exact duplicates within same patient/time/vital."""
```

### Harmonization Map
```python
VITAL_HARMONIZATION = {
    # Heart Rate variants
    'HR': ['HR', 'Pulse', 'Heart Rate', 'HeartRate'],
    # Blood Pressure
    'SBP': ['SBP', 'Systolic', 'SystolicBP'],
    'DBP': ['DBP', 'Diastolic', 'DiastolicBP'],
    # Temperature
    'TEMP': ['TEMP', 'Temperature', 'Temp'],
    # Respiratory Rate
    'RR': ['RR', 'Resp', 'Respiratory Rate', 'RespiratoryRate'],
    # Oxygen Saturation
    'SPO2': ['SPO2', 'SpO2', 'O2Sat', 'O2 Saturation'],
}
```

### Complexity: ⭐⭐ (2/5)
### Estimated Time: 2-3 days

---

## Submodule 3.5: Unit Converter & QC Filter

### Purpose
Standardize all units and filter physiologically implausible values.

### Inputs
- `outputs/discovery/harmonized_vitals.parquet`

### Outputs
- `outputs/discovery/qc_vitals.parquet`
- `outputs/discovery/qc_report.csv` (rejection statistics)

### Key Functions
```python
def detect_units(value: float, vital_type: str) -> str:
    """Auto-detect unit from value range (e.g., 98.6 → F, 37.0 → C)."""

def convert_temperature(value: float, from_unit: str, to_unit: str = 'C') -> float:
    """Convert temperature to Celsius."""

def apply_physiological_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Filter values outside physiological ranges."""

def create_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags: tachycardia, bradycardia, hypoxemia, hypotension, fever, hypothermia."""
```

### Physiological Ranges (Permissive)
| Vital | Min | Max | Unit |
|-------|-----|-----|------|
| HR | 20 | 250 | bpm |
| SBP | 50 | 300 | mmHg |
| DBP | 30 | 200 | mmHg |
| RR | 4 | 60 | breaths/min |
| SPO2 | 50 | 100 | % |
| TEMP | 30 | 45 | °C |

### Clinical Flag Thresholds
| Flag | Condition |
|------|-----------|
| tachycardia | HR > 100 |
| bradycardia | HR < 60 |
| hypoxemia | SPO2 < 90 |
| hypotension | SBP < 90 |
| hypertension | SBP > 180 or DBP > 120 |
| fever | TEMP > 38.0°C |
| hypothermia | TEMP < 36.0°C |
| tachypnea | RR > 20 |

### Complexity: ⭐⭐ (2/5)
### Estimated Time: 2-3 days

---

## Submodule 3.6: Multi-Source Temporal Aligner

### Purpose
Align vitals to patient-specific temporal grids relative to PE index time.

### Inputs
- `outputs/discovery/qc_vitals.parquet`
- `module_1_core_infrastructure/outputs/patient_timelines.pkl` (PE index times)

### Outputs
- `outputs/features/aligned_vitals_raw.h5` (HDF5 format)

### Key Functions
```python
def load_patient_timelines(path: str) -> Dict[str, datetime]:
    """Load PE index times from Module 1 output."""

def create_temporal_grid(index_time: datetime, phase: str) -> pd.DatetimeIndex:
    """Create time bins for a specific phase."""

def assign_vitals_to_bins(vitals_df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Assign each vital to nearest time bin."""

def merge_multi_source_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply hierarchical merge: Prg > Hnp > Phy."""

def extract_admission_vitals(df: pd.DataFrame, index_time: datetime) -> Dict:
    """Extract first vitals within 24h of PE index (admission severity)."""
```

### Temporal Phases
| Phase | Window | Resolution | Purpose |
|-------|--------|------------|---------|
| BASELINE | [-365d, -30d] | daily | Pre-existing conditions |
| PRE_ACUTE | [-30d, -7d] | daily | Recent trajectory |
| ACUTE | [-7d, +1d] | hourly | Acute presentation |
| HIGH_RES_ACUTE | [-24h, +24h] | 5 min | High-resolution window |
| SUBACUTE | [+2d, +14d] | hourly | Hospital course |
| RECOVERY | [+15d, +90d] | daily | Recovery trajectory |

### HDF5 Structure
```
/patient_{EMPI}/
    /raw_vitals_phy/        # Source-specific raw
    /raw_vitals_hnp/
    /raw_vitals_prg/
    /merged_vitals/         # Hierarchically merged
    /admission_vitals/      # First 24h after index
    /metadata/              # Index time, sources available
```

### Complexity: ⭐⭐⭐⭐ (4/5)
### Estimated Time: 5-7 days

---

## Submodule 3.7: Provenance & Quality Calculator

### Purpose
Calculate quality metrics, detect conflicts, track data provenance.

### Inputs
- `outputs/features/aligned_vitals_raw.h5`

### Outputs
- `outputs/features/vitals_with_provenance.h5`
- `outputs/features/quality_report.csv`

### Key Functions
```python
def detect_conflicts(df: pd.DataFrame, threshold_map: Dict) -> pd.DataFrame:
    """Flag disagreements between sources within same time bin."""

def calculate_completeness_metrics(df: pd.DataFrame) -> Dict:
    """Calculate coverage by source, vital type, phase."""

def calculate_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Track offset from bin centers (temporal precision)."""

def validate_temporal_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Check rate of change for physiological plausibility."""

def detect_outliers(df: pd.DataFrame, method: str = 'modified_zscore') -> pd.DataFrame:
    """Flag statistical outliers (modified Z-score > 3.5)."""

def classify_encounter_pattern(patient_sources: Dict) -> str:
    """Classify: outpatient_only, admission_only, admitted, full_trajectory."""
```

### Conflict Thresholds
| Vital | Threshold |
|-------|-----------|
| HR | 10 bpm |
| SBP/DBP | 20 mmHg |
| RR | 5 breaths/min |
| SPO2 | 5% |
| TEMP | 0.5°C |

### Rate of Change Limits (per hour)
| Vital | Max Change |
|-------|------------|
| HR | 50 bpm |
| SBP/DBP | 30 mmHg |
| RR | 10 breaths/min |
| SPO2 | 20% |
| TEMP | 2°C |

### Complexity: ⭐⭐⭐ (3/5)
### Estimated Time: 4-5 days

---

## Submodule 3.8: Feature Engineering Pipeline

### Purpose
Generate modeling-ready features: aggregates, trajectories, composites.

### Inputs
- `outputs/features/vitals_with_provenance.h5`

### Outputs
- `outputs/features/vitals_features_final.h5` (PRIMARY OUTPUT)
- `outputs/features/feature_summary.csv`

### Key Functions
```python
def aggregate_by_temporal_phase(df: pd.DataFrame, phase: str) -> Dict:
    """Calculate mean, median, min, max, std, first, last per vital per phase."""

def calculate_trajectory_features(df: pd.DataFrame) -> Dict:
    """Calculate slope, direction, volatility, range, time_to_normalization."""

def calculate_clinical_composites(row: Dict) -> Dict:
    """Calculate shock_index, MAP, pulse_pressure, modified_shock_index, delta_index."""

def aggregate_clinical_flags(df: pd.DataFrame) -> Dict:
    """Calculate any_tachycardia, prop_tachycardia, max_tachycardia_duration."""

def forward_fill_with_decay(df: pd.DataFrame, decay_hours: float = 24) -> pd.DataFrame:
    """Impute missing values with exponential decay."""
```

### Clinical Composites
```python
shock_index = HR / SBP           # >1.0 indicates shock
pulse_pressure = SBP - DBP       # <25 indicates low cardiac output
MAP = DBP + (SBP - DBP) / 3      # <65 inadequate organ perfusion
modified_shock_index = HR / MAP
delta_index = HR - RR            # Negative suggests severe PE
```

### Features Per Temporal Phase
| Category | Features |
|----------|----------|
| Basic | mean, median, min, max, std, first, last |
| Trajectory | slope, direction, volatility, range |
| Coverage | n_measurements, time_coverage |
| Clinical | any_abnormal, prop_abnormal, max_duration_abnormal |

### Admission Features (Special)
- ADMISSION_HR, ADMISSION_SBP, ADMISSION_DBP, ADMISSION_RR, ADMISSION_SPO2, ADMISSION_TEMP
- ADMISSION_shock_index, ADMISSION_pulse_pressure, ADMISSION_MAP
- ADMISSION_tachycardia_flag, ADMISSION_hypoxemia_flag, etc.

### Complexity: ⭐⭐⭐⭐ (4/5)
### Estimated Time: 6-8 days

---

## Submodule 3.9: Validation Framework

### Purpose
4-tier validation achieving ≥90% accuracy.

### Inputs
- All intermediate outputs
- Manual annotation results

### Outputs
- `outputs/validation/validation_report.html`
- `outputs/validation/cross_validation_results.json`
- `outputs/validation/manual_review_cases.csv`

### Tier 1: Cross-Validation with Structured Data
```python
def cross_validate_with_phy(hnp_df: pd.DataFrame, prg_df: pd.DataFrame, phy_df: pd.DataFrame) -> Dict:
    """Match note extractions to Phy.txt structured values."""
```
- Match on patient + date (±1 hour)
- Calculate clinical agreement, correlation, MAE, Bland-Altman
- Target: ≥90% clinical agreement

### Tier 2: Strategic Manual Review
```python
def generate_manual_review_sample(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    """Stratified sample for dual independent annotation."""
```
- 50 high-discrepancy from cross-val
- 75 no structured match
- 40 critical values
- 35 edge cases
- Target: κ ≥ 0.80 inter-rater reliability

### Tier 3: Statistical Monitoring
```python
def validate_distributions(df: pd.DataFrame) -> Dict:
    """Distribution validation, outlier detection, temporal plausibility."""
```
- KS test vs. reference populations
- Outlier rate <2%
- Implausible transitions <1%

### Tier 4: Pattern-Specific Validation
```python
def validate_patterns(df: pd.DataFrame) -> Dict:
    """Negation handling, range handling, unit conversion accuracy."""
```
- Negation false positive rate <5%
- Range handling consistency
- Unit conversion accuracy 100%

### Complexity: ⭐⭐⭐⭐ (4/5)
### Estimated Time: 8-10 days

---

## Submodule 3.10: Main Orchestrator

### Purpose
Coordinate all submodules with checkpointing and progress tracking.

### Inputs
- `config/vitals_config.yaml`

### Outputs
- All intermediate and final outputs
- `outputs/pipeline_log.txt`

### Key Functions
```python
def run_full_pipeline(config: Dict) -> None:
    """Execute all phases in order with checkpointing."""

def run_partial_pipeline(config: Dict, submodules: List[str]) -> None:
    """Run subset for debugging or incremental processing."""

def validate_intermediate_outputs(submodule: str) -> bool:
    """Check outputs before proceeding to next stage."""

class ProgressTracker:
    """Logging, time estimates, checkpoint management."""
```

### Execution Phases
1. **Phase 1: Extraction (Parallel)** - 3.1, 3.2, 3.3 ✅ COMPLETE
2. **Phase 2: Harmonization (Sequential)** - 3.4, 3.5
3. **Phase 3: Integration (Sequential)** - 3.6, 3.7, 3.8
4. **Phase 4: Validation** - 3.9

### Complexity: ⭐⭐ (2/5)
### Estimated Time: 3-4 days

---

## Implementation Order

### Recommended Sequence
1. **3.4 Harmonizer** - Simple merge, establishes data format
2. **3.5 QC & Units** - Clean data before complex processing
3. **3.6 Temporal Aligner** - Core alignment logic (most complex)
4. **3.7 Provenance** - Quality metrics
5. **3.8 Features** - Final feature engineering
6. **3.9 Validation** - Verify all previous work
7. **3.10 Orchestrator** - Tie everything together

### Total Estimated Time
| Submodule | Days |
|-----------|------|
| 3.4 | 2-3 |
| 3.5 | 2-3 |
| 3.6 | 5-7 |
| 3.7 | 4-5 |
| 3.8 | 6-8 |
| 3.9 | 8-10 |
| 3.10 | 3-4 |
| **Total** | **30-40 days** |

---

## Success Criteria

### Data Quality
- Extraction accuracy ≥90% (cross-validation + manual review)
- Data completeness ≥80% overall
- <1% out-of-range values
- <1% implausible temporal transitions

### Coverage
- ≥95% of patients with any vitals
- ≥70% of patients with admission vitals
- ≥50% of patients with high-res acute vitals

### Validation
- Cross-validation clinical agreement ≥90%
- Inter-rater reliability κ ≥0.80
- Negation false positive rate <5%
- Unit conversion accuracy 100%

---

## Next Steps

1. **Run extractions** (user task - 3.1, 3.2, 3.3 CLIs)
2. **Verify outputs** exist in `outputs/discovery/`
3. **Design 3.4 Harmonizer** with brainstorming skill
4. **Implement 3.4** following TDD approach
5. **Continue sequentially** through 3.5-3.10

---

**Document Version:** 1.0
**Last Updated:** 2025-12-08
