# Module 3: Comprehensive Vitals Extraction & Processing

**Version:** 1.0
**Status:** Design Complete - Ready for Implementation
**Timeline:** 10 weeks
**Dependencies:** Module 1 (patient_timelines.pkl)

---

## Overview

Module 3 extracts vital signs from **three complementary data sources** (structured and unstructured), implements **advanced NLP** with context awareness, and generates **modeling-ready features** with complete provenance tracking and rigorous validation.

### Key Capabilities

- ✅ **Multi-source extraction**: Phy.txt (structured) + Hnp.csv (H&P notes) + Prg.csv (progress notes)
- ✅ **Maximum temporal resolution**: 5-minute bins in acute phase when available
- ✅ **Advanced NLP**: Context-aware extraction with negation handling, range parsing, narrative interpretation
- ✅ **Full provenance**: 6-layer information preservation (raw values → merged → quality metrics)
- ✅ **Rigorous validation**: 4-tier framework achieving ≥90% accuracy
- ✅ **Clinical composites**: Shock index, MAP, pulse pressure, delta index
- ✅ **Publication-ready**: Comprehensive validation report with multiple independent evidence sources

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure paths (edit config/vitals_config.yaml)
vim config/vitals_config.yaml

# 3. Run full pipeline
python module_03_vitals_processing.py

# 4. Check validation report
open outputs/validation/validation_report.html

# 5. Load final features
import h5py
with h5py.File('outputs/features/vitals_features_final.h5', 'r') as f:
    patient_features = f['EMPI_12345']['ACUTE'][:]
```

---

## Data Sources

| Source | Type | Volume | Coverage | Clinical Context |
|--------|------|--------|----------|------------------|
| **Phy.txt** | Structured | 33M rows | Outpatient + inpatient | Baseline vitals, clinic visits |
| **Hnp.csv** | Unstructured notes | 257K notes (3.4 GB) | ~80% contain vitals | **Admission vitals** (PE presentation) |
| **Prg.csv** | Unstructured notes | 8.7M notes (42 GB) | ~35% contain vitals | Serial inpatient vitals (trajectory) |

**Why all three?**
- **Complementary coverage**: Each source fills gaps in the others
- **Clinical context**: Admission severity (Hnp) vs. disease trajectory (Prg) vs. baseline (Phy)
- **Cross-validation**: Overlaps enable validation without manual annotation
- **Encounter patterns**: Source combinations indicate disease severity (outpatient-only vs. full-trajectory)

---

## Architecture

### Six-Layer Information Preservation

```
Layer 1: Raw Source-Specific Values
         ↓
Layer 2: Hierarchical Merged Values + Source Attribution
         ↓
Layer 3: Conflict Detection & Quality Metrics
         ↓
Layer 4: Temporal Precision Tracking
         ↓
Layer 5: Temporal Consistency Validation
         ↓
Layer 6: Encounter Pattern Features
```

**Rationale:** Preserve all information for reproducibility, debugging, and quality assessment. Can reprocess with different strategies without re-extraction.

### Ten Independent Submodules

| Submodule | Purpose | Complexity | Time |
|-----------|---------|-----------|------|
| 3.1 | Structured extractor (Phy.txt) | ⭐ | 2 days |
| 3.2 | H&P NLP extractor (Hnp.csv) | ⭐⭐⭐⭐ | 5-7 days |
| 3.3 | Progress NLP extractor (Prg.csv) | ⭐⭐⭐⭐⭐ | 7-10 days |
| 3.4 | Vitals harmonizer | ⭐⭐ | 2-3 days |
| 3.5 | Unit converter & QC filter | ⭐⭐ | 2-3 days |
| 3.6 | Multi-source temporal aligner | ⭐⭐⭐⭐ | 5-7 days |
| 3.7 | Provenance & quality calculator | ⭐⭐⭐ | 4-5 days |
| 3.8 | Feature engineering pipeline | ⭐⭐⭐⭐ | 6-8 days |
| 3.9 | Validation framework (4-tier) | ⭐⭐⭐⭐ | 8-10 days |
| 3.10 | Main orchestrator | ⭐⭐ | 3-4 days |

**Total:** 10 weeks (includes parallelization + 2-week buffer)

---

## Output Features

### Temporal Phases

- **BASELINE**: [-365d, -30d] @ daily resolution
- **PRE_ACUTE**: [-30d, -7d] @ daily resolution
- **ACUTE**: [-7d, +1d] @ hourly resolution
- **HIGH_RES_ACUTE**: [-24h, +24h] @ 5-minute resolution
- **SUBACUTE**: [+2d, +14d] @ hourly resolution
- **RECOVERY**: [+15d, +90d] @ daily resolution

### Feature Categories

**Per Vital Per Phase:**
- Basic statistics: mean, median, min, max, std, first, last
- Trajectory: slope, direction, volatility, range, time_to_normalization
- Coverage: n_measurements, time_coverage (% of phase with data)
- Clinical flags: any_tachycardia, prop_tachycardia, max_tachycardia_duration

**Special: Admission Vitals**
- ADMISSION_HR, ADMISSION_SBP, ADMISSION_DBP, ADMISSION_RR, ADMISSION_SPO2, ADMISSION_TEMP
- ADMISSION_shock_index, ADMISSION_pulse_pressure, ADMISSION_MAP
- ADMISSION_tachycardia_flag, ADMISSION_hypoxemia_flag, etc.

**Clinical Composites:**
```python
shock_index = HR / SBP  # >1.0 indicates shock
pulse_pressure = SBP - DBP  # <25 indicates low cardiac output
MAP = DBP + (SBP - DBP) / 3  # <65 inadequate organ perfusion
modified_shock_index = HR / MAP
delta_index = HR - RR  # Negative suggests severe PE
```

**Quality Metrics:**
- Completeness by source (phy, hnp, prg, overall)
- Conflict rate, outlier rate, implausible change rate
- Temporal precision (avg/max time delta from grid)
- Encounter pattern (categorical: outpatient_only → full_trajectory)

---

## Validation Strategy

### Four Independent Tiers

**Tier 1: Cross-Validation with Structured Data**
- Match note extractions to Phy.txt structured values
- Calculate clinical agreement, correlation, MAE, Bland-Altman
- Target: ≥90% clinical agreement per vital
- Coverage: ~30-40% of extractions (others validated via Tiers 2-4)

**Tier 2: Strategic Manual Review**
- Stratified sample: 200 notes (high-discrepancy, critical values, edge cases)
- Dual independent annotation with inter-rater reliability
- Error taxonomy: false negatives, false positives, context errors, unit errors
- Target: ≥90% accuracy, κ ≥ 0.80

**Tier 3: Continuous Statistical Monitoring**
- Distribution validation (KS test vs. reference populations)
- Outlier detection (modified Z-score >3.5)
- Temporal plausibility (rate of change per hour)
- Digit preference analysis
- Target: <1% out-of-range, <2% outliers, <1% implausible transitions

**Tier 4: Pattern-Specific Validation**
- Negation handling (false positive rate <5%)
- Range handling consistency
- Unit conversion accuracy (100%)

### Validation Timeline

- **Week 8**: Tier 1 cross-validation + sample generation
- **Week 9**: Dual independent annotation (33 hours reviewer time)
- **Week 10**: Tiers 3-4 + comprehensive report generation

---

## File Structure

```
module_3_vitals_processing/
├── config/
│   ├── vitals_config.yaml              # Main configuration
│   ├── harmonization_map.json          # Concept mappings
│   └── qc_thresholds.json              # QC thresholds per vital
├── extractors/
│   ├── phy_extractor.py                # Submodule 3.1
│   ├── hnp_nlp_extractor.py            # Submodule 3.2
│   ├── prg_nlp_extractor.py            # Submodule 3.3
│   ├── patterns.py                     # Regex pattern library
│   └── negation_handler.py             # Negation detection
├── processing/
│   ├── harmonizer.py                   # Submodule 3.4
│   ├── unit_converter.py               # Submodule 3.5
│   ├── qc_filter.py                    # Submodule 3.5
│   ├── temporal_aligner.py             # Submodule 3.6
│   ├── provenance_calculator.py        # Submodule 3.7
│   └── feature_engineer.py             # Submodule 3.8
├── validation/
│   ├── cross_validator.py              # Tier 1
│   ├── manual_review_sampler.py        # Tier 2
│   ├── statistical_validator.py        # Tier 3
│   ├── pattern_validator.py            # Tier 4
│   └── report_generator.py             # HTML report
├── utils/
│   ├── io_utils.py                     # HDF5 I/O helpers
│   ├── temporal_utils.py               # Datetime handling
│   ├── logging_utils.py                # Logging configuration
│   └── visualization_utils.py          # Plotting for reports
├── tests/                              # Comprehensive test suite
├── outputs/
│   ├── discovery/                      # Intermediate outputs
│   ├── features/                       # **Final features here**
│   └── validation/                     # Validation report here
├── docs/
│   ├── ARCHITECTURE.md                 # Full technical design (~35 pages)
│   ├── SUBMODULES_QUICK_REFERENCE.md   # Fast lookup guide
│   ├── IMPLEMENTATION_ROADMAP.md       # Week-by-week plan
│   ├── API.md                          # Function documentation (TBD)
│   └── USER_GUIDE.md                   # Usage examples (TBD)
├── module_03_vitals_processing.py      # Main entry point (Submodule 3.10)
├── requirements.txt
└── README.md                           # This file
```

---

## Usage Examples

### Basic: Run Full Pipeline

```bash
python module_03_vitals_processing.py --config config/vitals_config.yaml
```

### Advanced: Run Specific Submodules

```bash
# Structured data only (fast test)
python module_03_vitals_processing.py --submodules 3.1,3.4,3.5,3.6,3.7,3.8

# Resume from checkpoint
python module_03_vitals_processing.py --resume-from 3.6

# Validation only (if features already exist)
python module_03_vitals_processing.py --submodules 3.9
```

### Python API: Load Features

```python
import h5py
import numpy as np

# Load patient vitals features
with h5py.File('outputs/features/vitals_features_final.h5', 'r') as f:
    patient_id = 'EMPI_12345'

    # Get ACUTE phase features
    acute = f[patient_id]['ACUTE']
    hr_mean = acute['HR_mean'][()]
    hr_max = acute['HR_max'][()]
    tachycardia = acute['any_tachycardia'][()]

    # Get admission vitals
    admission = f[patient_id]['ADMISSION']
    admission_hr = admission['HR'][()]
    shock_index = admission['shock_index'][()]

    # Get quality metrics
    quality = f[patient_id]['QUALITY']
    completeness = quality['completeness_overall'][()]
    encounter_pattern = quality['encounter_pattern'][()].decode()

    print(f"Patient {patient_id}:")
    print(f"  Acute HR: {hr_mean:.1f} bpm (max: {hr_max:.1f})")
    print(f"  Admission HR: {admission_hr:.1f}, Shock Index: {shock_index:.2f}")
    print(f"  Completeness: {completeness:.1%}")
    print(f"  Encounter: {encounter_pattern}")
```

### Export to DataFrame

```python
import pandas as pd
import h5py

def load_patient_features(h5_path, patient_id):
    """Load all features for a patient into a flat dict"""
    features = {}
    with h5py.File(h5_path, 'r') as f:
        patient = f[patient_id]

        # Load all temporal phases
        for phase in ['BASELINE', 'ACUTE', 'SUBACUTE', 'RECOVERY']:
            if phase in patient:
                for key in patient[phase].keys():
                    features[f'{phase}_{key}'] = patient[phase][key][()]

        # Load admission and quality
        for group in ['ADMISSION', 'QUALITY', 'COMPOSITES']:
            if group in patient:
                for key in patient[group].keys():
                    features[f'{group}_{key}'] = patient[group][key][()]

    return features

# Load all patients
patient_ids = [...]  # List of patient EMPIs
all_features = []
for pid in patient_ids:
    features = load_patient_features('outputs/features/vitals_features_final.h5', pid)
    features['EMPI'] = pid
    all_features.append(features)

df = pd.DataFrame(all_features)
df.to_csv('vitals_features_flat.csv', index=False)
```

---

## Configuration

Edit `config/vitals_config.yaml` to customize:

```yaml
# Which sources to use
extraction:
  use_phy: true
  use_hnp: true
  use_prg: true

# Temporal resolution
temporal_alignment:
  HIGH_RES_ACUTE:
    start_hours: -24
    end_hours: 24
    resolution: "5T"  # 5 minutes

# QC thresholds
qc_thresholds:
  HR: [20, 250]  # [min, max] bpm
  SBP: [50, 300]  # mmHg
  # ...

# Clinical flag thresholds
clinical_thresholds:
  tachycardia: 100  # bpm
  hypoxemia: 90     # %
  hypotension: 90   # mmHg SBP
  # ...
```

---

## Performance Benchmarks

| Operation | Target Time | Hardware |
|-----------|-------------|----------|
| Phy.txt extraction | <30 min | 16 cores |
| Hnp.csv extraction | <2 hours | 16 cores |
| Prg.csv extraction | <8 hours | 16 cores |
| Temporal alignment | <1 hour | 16 cores |
| Feature engineering | <30 min | 16 cores |
| **Full pipeline** | **<12 hours** | 16 cores |

**Storage:** ~24 GB total (all intermediate + final outputs)

---

## Success Criteria

### Data Quality
- ✅ Extraction accuracy ≥90% (cross-validation + manual review)
- ✅ Data completeness ≥80% overall
- ✅ <1% out-of-range values
- ✅ <1% implausible temporal transitions

### Coverage
- ✅ ≥95% of patients with any vitals
- ✅ ≥70% of patients with admission vitals
- ✅ ≥50% of patients with high-res acute vitals

### Validation
- ✅ Cross-validation clinical agreement ≥90%
- ✅ Inter-rater reliability κ ≥0.80
- ✅ Negation false positive rate <5%
- ✅ Unit conversion accuracy 100%

---

## Known Limitations

1. **NLP Extraction Accuracy**: Regex-based extraction achieves ~85-90% accuracy on unstructured notes. Medical NER (spaCy, scispaCy) could improve to ~95% but adds complexity.

2. **Cross-Validation Coverage**: Only ~30-40% of note extractions can be validated against structured data. Remaining cases validated through manual review and statistical checks.

3. **Temporal Precision**: Some measurements binned to nearest 5-minute interval may have actual timestamps up to ±2.5 minutes from bin center. Time delta tracked in provenance.

4. **Narrative Extractions**: Descriptive text ("tachycardic", "afebrile") mapped to approximate numeric values with confidence <1.0. Use with caution.

5. **Historical Vitals**: Current implementation focuses on actual measured vitals. "Prior" or "home" vitals may be incorrectly extracted as current (negation detection reduces but doesn't eliminate).

---

## Future Enhancements

### Post-V1.0
- **Medical NER**: Implement scispaCy or MedCAT for improved extraction accuracy (90% → 95%)
- **Waveform Data**: Extract continuous vital sign waveforms if available (enables HRV, waveform analysis)
- **Multi-Modal Integration**: Combine vitals with labs, medications, procedures in unified temporal model
- **Temporal Attention**: Use transformer architectures to learn which timepoints are most predictive
- **Real-Time Extraction**: Adapt pipeline for live EHR streaming (clinical decision support)

---

## Documentation

Comprehensive documentation in `docs/`:

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Full technical design (~12,000 words)
  - Clinical context & goals
  - Data sources detailed comparison
  - 6-layer information preservation architecture
  - 10 submodules with functions, I/O, complexity
  - 4-tier validation strategy with benchmarks
  - Design decisions & rationale
  - Risk mitigation
  - Future enhancements

- **[SUBMODULES_QUICK_REFERENCE.md](docs/SUBMODULES_QUICK_REFERENCE.md)**: Fast lookup guide
  - Dependency graph
  - Submodule summary table
  - Quick reference cards per submodule
  - Critical paths (fast/MVP/complete)
  - Data flow diagram
  - Quick start commands

- **[IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)**: Week-by-week implementation plan
  - 10-week timeline with milestones
  - Day-by-day task breakdowns
  - Success criteria per week
  - Checkpoint questions
  - Risk mitigation timeline
  - Handoff checklist

- **API.md** (TBD): Function-level documentation
- **USER_GUIDE.md** (TBD): Detailed usage examples and tutorials
- **VALIDATION_PROTOCOL.md** (TBD): Step-by-step validation procedures

---

## Dependencies

### Python Packages
```txt
pandas>=1.5.0
numpy>=1.23.0
h5py>=3.7.0
pyarrow>=10.0.0  # For parquet
pyyaml>=6.0
scikit-learn>=1.1.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
dask[complete]>=2022.11.0  # For large file processing
tqdm>=4.64.0  # Progress bars
pytest>=7.2.0  # Testing
```

### External Dependencies
- **Module 1**: `patient_timelines.pkl` (provides PE index times and temporal windows)

### Data Files
- `/home/moin/TDA_11_1/Data/FNR_20240409_091633_Phy.txt` (33M rows)
- `/home/moin/TDA_11_1/Data/Hnp.csv` (257K notes)
- `/home/moin/TDA_11_1/Data/Prg.csv` (8.7M notes)

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_phy_extractor.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run integration tests only
pytest tests/ -m integration
```

**Test Coverage Target:** >80%

---

## Contributing

### Development Workflow
1. Branch from `main` for each submodule (`feature/3.X-name`)
2. Write tests BEFORE implementation (TDD)
3. Implement functionality
4. Ensure all tests pass (`pytest tests/`)
5. Format code (`black .`, `flake8 .`)
6. Submit pull request with description
7. Code review + merge

### Code Standards
- Type hints for all functions
- Google-style docstrings
- PEP 8 compliance
- >80% test coverage for new code

---

## Citation

If you use this module in published research, please cite:

```
[TBD - Add citation after publication]
```

---

## Contact & Support

**Project:** TDA 11.1 - Pulmonary Embolism Risk Prediction
**Module Owner:** [TBD]
**Issues:** [Repository issue tracker]
**Documentation:** `docs/` directory

---

## Changelog

### Version 1.0 (2025-11-09)
- Initial architecture design
- 10 submodules defined
- 4-tier validation framework
- Comprehensive documentation
- Ready for implementation

---

## License

[TBD - Add appropriate license]

---

**Status:** ✅ Design Complete - Ready for Implementation
**Next Step:** Begin Week 1 (Submodule 3.1 - Phy.txt extraction)
