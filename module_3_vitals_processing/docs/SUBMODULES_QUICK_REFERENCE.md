# Module 3 Submodules - Quick Reference

**Purpose:** Fast lookup guide for each submodule's purpose, inputs, outputs, and dependencies.

---

## Dependency Graph

```
┌─────────────────────────────────────────────────────────┐
│                  START: Data Sources                     │
│  Phy.txt (33M) | Hnp.csv (257K) | Prg.csv (8.7M)       │
└─────────────────────────────────────────────────────────┘
           │              │              │
           ▼              ▼              ▼
     ┌─────────┐    ┌─────────┐    ┌─────────┐
     │ 3.1 Phy │    │ 3.2 HNP │    │ 3.3 PRG │  ◄─ PHASE 1: EXTRACTION
     │Extract  │    │ Extract │    │ Extract │     (Parallel)
     └─────────┘    └─────────┘    └─────────┘
           │              │              │
           └──────────────┼──────────────┘
                         ▼
                  ┌──────────────┐
                  │3.4 Harmonize │  ◄─ PHASE 2: HARMONIZATION
                  └──────────────┘     (Sequential)
                         ▼
                  ┌──────────────┐
                  │ 3.5 QC       │
                  │& Unit Conv   │
                  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │3.6 Temporal  │  ◄─ PHASE 3: INTEGRATION
                  │  Aligner     │     (Sequential)
                  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │3.7 Provenance│
                  │  Calculator  │
                  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │3.8 Feature   │
                  │ Engineering  │
                  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │3.9 Validation│  ◄─ PHASE 4: VALIDATION
                  │  Framework   │     (Sequential)
                  └──────────────┘
                         ▼
                  ┌──────────────┐
                  │ 3.10 Main    │
                  │Orchestrator  │
                  └──────────────┘
                         ▼
                  [ Final Output ]
```

---

## Submodule Summary Table

| ID | Name | Status | Tests | Complexity |
|----|------|--------|-------|------------|
| 3.1 | Structured Extractor (Phy) | ✅ COMPLETE | 39 | ⭐ |
| 3.2 | H&P NLP Extractor (Hnp) | ✅ COMPLETE | 74 | ⭐⭐⭐⭐ |
| 3.3 | Progress NLP Extractor (Prg) | ✅ COMPLETE | 61 | ⭐⭐⭐⭐⭐ |
| 3.4 | Vitals Harmonizer | ⏳ Pending | - | ⭐⭐ |
| 3.5 | Unit Converter & QC | ⏳ Pending | - | ⭐⭐ |
| 3.6 | Temporal Aligner | ⏳ Pending | - | ⭐⭐⭐⭐ |
| 3.7 | Provenance Calculator | ⏳ Pending | - | ⭐⭐⭐ |
| 3.8 | Feature Engineer | ⏳ Pending | - | ⭐⭐⭐⭐ |
| 3.9 | Validation Framework | ⏳ Pending | - | ⭐⭐⭐⭐ |
| 3.10 | Orchestrator | ⏳ Pending | - | ⭐⭐ |

**Total Tests:** 174 passing | **Last Updated:** 2025-12-08

---

## Submodule Details

### 3.1 Structured Vitals Extractor (Phy.txt) ✅ COMPLETE

**File:** `extractors/phy_extractor.py` (265 lines)

**Purpose:** Extract vitals from structured Phy.txt file

**Inputs:**
- `Data/Phy.txt` (33M rows, 2.7GB)

**Outputs:**
- `outputs/discovery/phy_vitals_raw.parquet` (~8-9M records)

**CLI:**
```bash
python3 -m module_3_vitals_processing.extractors.phy_extractor
```

**Key Functions:**
- `load_phy_vitals()` - Load and filter for vital concepts
- `parse_combined_bp()` - Parse "124/68" → (SBP=124, DBP=68)
- `map_phy_concepts_to_canonical()` - "Pulse" → "HR"

**Test File:** `tests/test_phy_extractor.py` (39 tests)

---

### 3.2 H&P NLP Extractor (Hnp.txt) ✅ COMPLETE

**File:** `extractors/hnp_extractor.py` (662 lines)

**Purpose:** Extract vitals from admission H&P notes using NLP

**Inputs:**
- `Data/Hnp.txt` (136,950 notes, 2.3GB)

**Outputs:**
- `outputs/discovery/hnp_vitals_raw.parquet` (~1.6M records)

**CLI:**
```bash
python3 -m module_3_vitals_processing.extractors.hnp_extractor
```

**Key Functions:**
- `identify_hnp_sections()` - Find vital signs sections in note
- `extract_heart_rate()`, `extract_blood_pressure()`, etc. - Pattern matching
- `check_negation()` - Context validation

**Patterns Library:** `extractors/hnp_patterns.py` (29 patterns)
- BP_PATTERNS, HR_PATTERNS, TEMP_PATTERNS, RR_PATTERNS, SPO2_PATTERNS
- NEGATION_PATTERNS, SECTION_PATTERNS

**Test Files:** `tests/test_hnp_extractor.py` (70 tests), `tests/test_hnp_patterns.py` (4 tests)

---

### 3.3 Progress Notes NLP Extractor (Prg.txt) ✅ COMPLETE

**File:** `extractors/prg_extractor.py` (542 lines)

**Purpose:** Extract vitals from progress notes with skip section filtering and checkpointing

**Inputs:**
- `Data/Prg.txt` (4.6M notes, 29.7GB)

**Outputs:**
- `outputs/discovery/prg_vitals_raw.parquet` (~2-5M records)

**CLI:**
```bash
python3 -m module_3_vitals_processing.extractors.prg_extractor \
  -w 8 -c 10000 --no-resume
```

**Key Functions:**
- `identify_prg_sections()` - Find Physical Exam/Vitals/Objective sections
- `is_in_skip_section()` - Detect allergies/medications/history (false positive filtering)
- `extract_temperature_with_method()` - Temp extraction with method capture (oral, temporal, etc.)
- `extract_prg_vitals_from_text()` - Combined extraction with skip filtering
- `save_checkpoint()`, `load_checkpoint()` - Resume capability for 30GB file

**Patterns Library:** `extractors/prg_patterns.py` (43 patterns)
- PRG_SECTION_PATTERNS (11), PRG_SKIP_PATTERNS (12)
- PRG_BP_PATTERNS, PRG_HR_PATTERNS, PRG_SPO2_PATTERNS, PRG_RR_PATTERNS
- PRG_TEMP_PATTERNS with TEMP_METHOD_MAP

**Special Features:**
- Skip section filtering (allergies, medications, history)
- Temperature method capture (oral, temporal, rectal, axillary, tympanic)
- Checkpointing every 5 chunks (~50K rows) for resume capability

**Test Files:** `tests/test_prg_extractor.py` (34 tests), `tests/test_prg_patterns.py` (27 tests)

---

### 3.4 Vitals Harmonizer

**File:** `processing/harmonizer.py`

**Purpose:** Map all vital variants to canonical names

**Inputs:**
- `outputs/discovery/phy_vitals_raw.parquet`
- `outputs/discovery/hnp_vitals_raw.parquet`
- `outputs/discovery/prg_vitals_raw.parquet`

**Outputs:**
- `outputs/discovery/harmonization_map.json`
- `outputs/discovery/harmonized_vitals.parquet`

**Key Functions:**
- `create_harmonization_map()` - Map variants to canonical
- `consolidate_bp_measurements()` - Handle combined/separate BP
- `deduplicate_vitals()` - Remove exact duplicates

**Harmonization Examples:**
- "Pulse", "Heart Rate" → "HR"
- "O2 Saturation-SPO2", "O2 Saturation-LFA4600" → "SPO2"
- "Blood Pressure-Epic" → ["SBP", "DBP"]

**Complexity:** ⭐⭐ (2/5)

**Dependencies:** Submodules 3.1, 3.2, 3.3

**Test File:** `tests/test_harmonizer.py`

---

### 3.5 Unit Converter & QC Filter

**File:** `processing/unit_converter.py`, `processing/qc_filter.py`

**Purpose:** Standardize units and filter implausible values

**Inputs:**
- `outputs/discovery/harmonized_vitals.parquet`

**Outputs:**
- `outputs/discovery/qc_vitals.parquet`
- `outputs/discovery/qc_report.csv`

**Key Functions:**
- `detect_units()` - Auto-detect unit from value range
- `convert_to_standard_units()` - All vitals → standard units
- `apply_physiological_qc()` - Filter impossible values
- `create_clinical_flags()` - tachycardia, hypoxemia, etc.

**Standard Units:**
- TEMP → Celsius
- Weight → kg
- Height → cm
- BP → mmHg

**Physiological Ranges (Permissive):**
- HR: 20-250 bpm
- SBP: 50-300 mmHg
- DBP: 30-200 mmHg
- RR: 4-60 breaths/min
- SPO2: 50-100%
- TEMP: 30-45°C

**Complexity:** ⭐⭐ (2/5)

**Dependencies:** Submodule 3.4

**Test Files:** `tests/test_unit_converter.py`, `tests/test_qc_filter.py`

---

### 3.6 Multi-Source Temporal Aligner

**File:** `processing/temporal_aligner.py`

**Purpose:** Align vitals to temporal grid and merge sources

**Inputs:**
- `outputs/discovery/qc_vitals.parquet`
- `module_1_core_infrastructure/outputs/patient_timelines.pkl`

**Outputs:**
- `outputs/features/aligned_vitals_raw.h5` (HDF5)

**Key Functions:**
- `create_temporal_grid()` - Build phase-specific grids
- `assign_vitals_to_bins()` - Bin measurements to grid
- `merge_multi_source_vitals()` - Hierarchical merge (Prg > Hnp > Phy)
- `extract_admission_vitals()` - Special handling for admission

**Temporal Phases:**
- BASELINE: [-365d, -30d] @ daily
- ACUTE: [-7d, +1d] @ hourly
- HIGH_RES_ACUTE: [-24h, +24h] @ 5min
- SUBACUTE: [+2d, +14d] @ hourly
- RECOVERY: [+15d, +90d] @ daily

**HDF5 Structure:**
```
/patient_EMPI/
  /raw_vitals_phy/
  /raw_vitals_hnp/
  /raw_vitals_prg/
  /merged_vitals/
  /admission_vitals/
```

**Complexity:** ⭐⭐⭐⭐ (4/5) - Complex temporal logic

**Dependencies:** Submodule 3.5, Module 1

**Test File:** `tests/test_temporal_aligner.py`

---

### 3.7 Provenance & Quality Metrics Calculator

**File:** `processing/provenance_calculator.py`

**Purpose:** Calculate Layers 3-6 quality metrics

**Inputs:**
- `outputs/features/aligned_vitals_raw.h5`

**Outputs:**
- `outputs/features/vitals_with_provenance.h5`
- `outputs/features/quality_report.csv`

**Key Functions:**
- `detect_conflicts()` - Flag disagreements between sources
- `calculate_completeness_metrics()` - Coverage by source
- `calculate_time_deltas()` - Offset from bin centers
- `validate_temporal_consistency()` - Rate of change checks
- `detect_outliers()` - Modified Z-score > 3.5
- `classify_encounter_pattern()` - Categorical encounter type

**Conflict Thresholds:**
- HR/RR: 10 units
- SBP/DBP: 20 mmHg
- TEMP: 0.5°C
- SPO2: 5%

**Implausible Rate Thresholds (per hour):**
- HR: 50 bpm
- SBP/DBP: 30 mmHg
- RR: 10 breaths/min
- TEMP: 2°C
- SPO2: 20%

**Encounter Patterns:**
- 'outpatient_only', 'admission_only', 'admitted', 'outpatient_to_admission', 'full_trajectory'

**Complexity:** ⭐⭐⭐ (3/5)

**Dependencies:** Submodule 3.6

**Test File:** `tests/test_provenance.py`

---

### 3.8 Feature Engineering Pipeline

**File:** `processing/feature_engineer.py`

**Purpose:** Generate modeling-ready features

**Inputs:**
- `outputs/features/vitals_with_provenance.h5`

**Outputs:**
- `outputs/features/vitals_features_final.h5`

**Key Functions:**
- `aggregate_by_temporal_phase()` - Mean/min/max/std per phase
- `calculate_trajectory_features()` - Slope, volatility, time_to_normalization
- `calculate_clinical_composites()` - Shock index, pulse pressure, MAP
- `aggregate_clinical_flags()` - Proportion of time with abnormal vitals
- `format_admission_features()` - ADMISSION_* features
- `forward_fill_with_decay()` - Impute with exponential decay

**Clinical Composites:**
```python
shock_index = HR / SBP  # >1.0 = shock
pulse_pressure = SBP - DBP  # <25 = low cardiac output
MAP = DBP + (SBP - DBP) / 3  # <65 = inadequate perfusion
modified_shock_index = HR / MAP
delta_index = HR - RR  # Negative suggests severe PE
```

**Features Per Temporal Phase:**
- Basic: mean, median, min, max, std, first, last
- Trajectory: slope, direction, volatility, range
- Coverage: n_measurements, time_coverage
- Clinical: any_tachycardia, prop_tachycardia, max_tachycardia_duration

**Complexity:** ⭐⭐⭐⭐ (4/5)

**Dependencies:** Submodule 3.7

**Test File:** `tests/test_feature_engineer.py`

---

### 3.9 Validation Framework

**File:** `validation/cross_validator.py`, `validation/manual_review_sampler.py`, `validation/statistical_validator.py`, `validation/pattern_validator.py`, `validation/report_generator.py`

**Purpose:** 4-tier validation achieving ≥90% accuracy

**Inputs:** All intermediate outputs

**Outputs:**
- `outputs/validation/validation_report.html`
- `outputs/validation/cross_validation_results.json`
- `outputs/validation/manual_review_cases.csv`

**Tier 1: Cross-Validation**
- Match note extractions to structured Phy.txt
- Calculate: clinical agreement, correlation, MAE, Bland-Altman
- Target: ≥90% clinical agreement

**Tier 2: Manual Review**
- Stratified sample: 200 notes
  - 50 high-discrepancy
  - 75 no structured match
  - 40 critical values
  - 35 edge cases
- Dual independent annotation
- Inter-rater reliability: κ ≥ 0.80

**Tier 3: Statistical Monitoring**
- Distribution validation (KS test vs. reference)
- Outlier detection (<2% target)
- Temporal plausibility (<1% implausible transitions)

**Tier 4: Pattern-Specific**
- Negation handling (<5% false positives)
- Range handling consistency
- Unit conversion accuracy (100%)

**Complexity:** ⭐⭐⭐⭐ (4/5)

**Dependencies:** All previous submodules

**Test File:** `tests/test_validation.py`

---

### 3.10 Main Orchestrator

**File:** `module_03_vitals_processing.py`

**Purpose:** Coordinate all submodules with checkpointing

**Inputs:** `config/vitals_config.yaml`

**Outputs:** Final features + logs

**Key Functions:**
- `run_full_pipeline()` - Execute all phases in order
- `run_partial_pipeline()` - Run subset for debugging
- `validate_intermediate_outputs()` - Check outputs before proceeding
- `ProgressTracker` class - Logging and time estimates

**Execution Phases:**
1. **Phase 1: Extraction (Parallel)** - 3.1, 3.2, 3.3
2. **Phase 2: Harmonization (Sequential)** - 3.4, 3.5
3. **Phase 3: Integration (Sequential)** - 3.6, 3.7, 3.8
4. **Phase 4: Validation** - 3.9

**Checkpointing:** Can resume from any completed submodule

**Complexity:** ⭐⭐ (2/5)

**Dependencies:** All submodules

**Test File:** `tests/test_orchestrator.py` (integration tests)

---

## Critical Paths

### Fast Path to First Results (4 weeks)
```
3.1 (Phy only) → 3.4 → 3.5 → 3.6 → 3.7 → 3.8
```
**Deliverable:** Vitals features from structured data only (lower coverage but fast)

### Minimal Viable Product (6 weeks)
```
3.1 + 3.2 (Phy + HNP) → 3.4 → 3.5 → 3.6 → 3.7 → 3.8
```
**Deliverable:** Structured + admission vitals (good coverage, admission severity)

### Complete System (10 weeks)
```
3.1 + 3.2 + 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9
```
**Deliverable:** All sources + full validation (publication-ready)

---

## Data Flow Summary

```
Phy.txt (33M rows)          Hnp.csv (257K notes)       Prg.csv (8.7M notes)
      │                            │                           │
      ├─ Filter vitals             ├─ NLP extraction          ├─ NLP extraction
      ├─ Parse BP                  ├─ Negation handling       ├─ Range parsing
      ├─ Map concepts              └─ Context validation      └─ Narrative extraction
      │                            │                           │
      └────────────────────┬───────┴───────────────────────────┘
                           ▼
              phy_vitals_raw.parquet (3 files)
                           │
                           ▼
              Harmonization (map to canonical names)
                           │
                           ▼
              harmonized_vitals.parquet
                           │
                           ▼
              QC & Unit Conversion (standardize, filter)
                           │
                           ▼
              qc_vitals.parquet
                           │
                           ▼
              Temporal Alignment (merge sources, bin to grid)
                           │
                           ▼
              aligned_vitals_raw.h5 (HDF5)
                           │
                           ▼
              Provenance Calculation (Layers 3-6)
                           │
                           ▼
              vitals_with_provenance.h5
                           │
                           ▼
              Feature Engineering (aggregates, trajectories, composites)
                           │
                           ▼
              vitals_features_final.h5
                           │
                           ▼
              4-Tier Validation
                           │
                           ▼
              validation_report.html + quality metrics
```

---

## Storage Requirements

| Output | Format | Est. Size | Retention |
|--------|--------|-----------|-----------|
| phy_vitals_raw.parquet | Parquet | ~500 MB | Keep (for validation) |
| hnp_vitals_raw.parquet | Parquet | ~200 MB | Keep (for validation) |
| prg_vitals_raw.parquet | Parquet | ~2 GB | Keep (for validation) |
| harmonized_vitals.parquet | Parquet | ~3 GB | Keep (for reprocessing) |
| qc_vitals.parquet | Parquet | ~3 GB | Keep (for reprocessing) |
| aligned_vitals_raw.h5 | HDF5 | ~5 GB | Keep (for features) |
| vitals_with_provenance.h5 | HDF5 | ~8 GB | Keep (comprehensive) |
| vitals_features_final.h5 | HDF5 | ~2 GB | **PRIMARY OUTPUT** |
| validation outputs | Various | ~100 MB | Keep (for publication) |

**Total Storage:** ~24 GB (user has no constraints)

---

## Quick Start Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run structured extraction only (fast test)
python module_03_vitals_processing.py --submodules 3.1,3.4,3.5,3.6,3.7,3.8

# 3. Run full pipeline
python module_03_vitals_processing.py --config config/vitals_config.yaml

# 4. Run validation only (if features already exist)
python module_03_vitals_processing.py --submodules 3.9

# 5. Resume from checkpoint (if interrupted)
python module_03_vitals_processing.py --resume-from 3.6

# 6. Run tests
pytest tests/ -v

# 7. Generate validation report only
python validation/report_generator.py
```

---

## Common Issues & Solutions

### Issue: NLP extraction too slow
**Solution:**
- Use `config.extraction.chunk_size = 50000` (smaller chunks)
- Set `config.performance.use_dask = true`
- Filter to PE patients only before extraction

### Issue: Cross-validation coverage <30%
**Solution:**
- Increase manual review sample to 300-400 notes
- Rely more on statistical validation
- Document limitation in methods

### Issue: Conflicts >20%
**Solution:**
- Investigate conflict patterns (systematic vs. random)
- Implement confidence weighting
- Keep source-specific values separate

### Issue: Validation accuracy <90%
**Solution:**
- Review error taxonomy from manual annotation
- Refine regex patterns for top error types
- Consider upgrading to medical NER (scispaCy)

---

## Next Steps

After completing this design:

1. **Review with team:** Ensure clinical requirements are met
2. **Create sample data:** Small test dataset for rapid iteration
3. **Implement 3.1 first:** Validate pipeline with structured data
4. **Develop test suite:** Write tests before implementation (TDD)
5. **Iterate on NLP patterns:** Refine based on validation results

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
