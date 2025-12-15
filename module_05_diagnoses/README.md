# Module 05: Diagnosis Processing

Diagnosis encoding pipeline for PE trajectory analysis. Transforms raw ICD codes into PE-aligned canonical records, comorbidity indices, PE-specific features, embeddings, and world model states.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Layers 1-2 (Core + Comorbidity) | **COMPLETE** |
| Phase 2 | Layer 3 (PE-Specific Features) | Planned |
| Phase 3 | Layers 4-5 (Embeddings, World Models) | Planned |

**Current Stats:**
- 55 tests passing
- 9 commits
- 14 source files

---

## Quick Start

### Test Run (100 patients)

```bash
cd module_05_diagnoses
python build_layers.py --test --n=100
```

### Full Cohort (8,713 patients)

```bash
cd module_05_diagnoses
python build_layers.py
```

---

## Architecture

### 5-Layer Design

```
Layer 1: Canonical Diagnosis Records
├── All diagnoses with PE-relative timing
├── ICD-9/ICD-10 normalized codes
├── Temporal categories (7 types)
└── Output: layer1/canonical_diagnoses.parquet

Layer 2: Comorbidity Indices
├── Charlson Comorbidity Index (CCI)
├── 17 components with hierarchy rules
└── Output: layer2/comorbidity_scores.parquet

Layer 3: PE-Specific Features (Phase 2)
├── VTE history, cancer features
├── Cardiovascular/pulmonary comorbidities
├── Bleeding risk, provoking factors
└── Output: layer3/pe_diagnosis_features.parquet

Layer 4: Diagnosis Embeddings (Phase 3)
├── Ontological embeddings (SNOMED hierarchy)
├── Co-occurrence embeddings (patient patterns)
└── Output: layer4/diagnosis_embeddings.h5

Layer 5: World Model States (Phase 3)
├── Static diagnosis state (~30 dims)
├── Dynamic complication state (~10 dims)
└── Output: layer5/diagnosis_state.h5
```

### Module Structure

```
module_05_diagnoses/
├── config/
│   ├── diagnosis_config.py      # Paths, temporal categories, PE codes
│   └── comorbidity_codes.py     # Charlson ICD code mappings (17 components)
├── extractors/
│   └── diagnosis_extractor.py   # RPDR Dia.txt extraction
├── processing/
│   ├── icd_parser.py            # ICD-9/10 version detection
│   ├── temporal_classifier.py   # PE-relative timing (7 categories)
│   ├── charlson_calculator.py   # CCI with hierarchy rules
│   ├── layer1_builder.py        # Canonical diagnosis records
│   └── layer2_builder.py        # Comorbidity scores
├── tests/                       # 55 tests
├── build_layers.py              # CLI pipeline
├── outputs/
│   ├── layer1/                  # Canonical diagnoses
│   └── layer2/                  # Comorbidity scores
└── docs/plans/
    ├── 2025-12-13-diagnosis-encoding-design.md
    └── 2025-12-13-module-05-implementation-plan.md
```

---

## Data Flow

### Input

| Source | Path | Description |
|--------|------|-------------|
| RPDR Diagnoses | `Data/Dia.txt` | 32M diagnosis records |
| Patient Timelines | `module_1.../patient_timelines.pkl` | PE index times |

### Output

| Layer | Path | Description |
|-------|------|-------------|
| Layer 1 | `outputs/layer1/canonical_diagnoses.parquet` | All diagnoses with PE timing |
| Layer 2 | `outputs/layer2/comorbidity_scores.parquet` | CCI per patient |

---

## Temporal Categories

Diagnoses are classified by their relationship to PE index:

| Category | Days from PE | Description |
|----------|--------------|-------------|
| `preexisting_remote` | < -30 | Chronic comorbidities |
| `preexisting_recent` | -30 to -7 | Recent but not acute |
| `antecedent` | -7 to 0 | Immediately preceding PE |
| `index_concurrent` | 0 to 1 | At PE presentation |
| `early_complication` | 2 to 7 | First week post-PE |
| `late_complication` | 8 to 30 | Within 30 days |
| `follow_up` | > 30 | Beyond acute phase |

---

## Charlson Comorbidity Index

17 components with weights 1-6:

| Weight | Components |
|--------|------------|
| 1 | MI, CHF, PVD, CVD, Dementia, COPD, Rheumatic, PUD, Mild Liver, DM uncomplicated |
| 2 | DM complicated, Hemiplegia, Renal, Malignancy |
| 3 | Moderate/Severe Liver |
| 6 | Metastatic Cancer, AIDS/HIV |

**Hierarchy Rules:**
- Complicated diabetes supersedes uncomplicated
- Severe liver supersedes mild
- Metastatic cancer supersedes localized malignancy

---

## Layer 1 Schema

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| diagnosis_date | datetime | Diagnosis date |
| days_from_pe | int | Days relative to PE index |
| hours_from_pe | float | Hours relative to PE index |
| icd_code | str | Normalized ICD code |
| icd_version | str | '9' or '10' |
| diagnosis_description | str | Diagnosis name |
| diagnosis_type | str | 'principal', 'admitting', 'secondary' |
| code_position | int | Position in diagnosis list |
| encounter_id | str | Source encounter |
| encounter_type | str | 'inpatient', 'outpatient', 'unknown' |
| temporal_category | str | One of 7 categories |
| is_preexisting | bool | days_from_pe < -30 |
| is_recent_antecedent | bool | -30 <= days_from_pe < 0 |
| is_index_concurrent | bool | 0 <= days_from_pe <= 1 |
| is_complication | bool | days_from_pe > 1 |
| is_pe_diagnosis | bool | PE ICD code (I26.x, 415.1x) |

---

## Layer 2 Schema

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| cci_score | int | Charlson Comorbidity Index (0-37) |
| cci_components | str | JSON list of present components |
| cci_component_count | int | Number of components |

---

## Test Results (100 patients)

```
Layer 1: 106,624 records
  - preexisting_remote:  51,519 (48%)
  - follow_up:           27,535 (26%)
  - late_complication:    8,154 (8%)
  - antecedent:           6,658 (6%)
  - preexisting_recent:   6,378 (6%)
  - early_complication:   3,977 (4%)
  - index_concurrent:     2,403 (2%)

Layer 2: 100 patients
  - CCI Mean: 4.4
  - CCI Range: 0-23
```

---

## Dependencies

- **Module 1:** `patient_timelines.pkl` for PE index times
- **Python:** pandas, pyarrow, h5py, pytest

---

## Next Steps (Phase 2)

Layer 3: PE-Specific Diagnosis Features
- VTE history (prior PE, prior DVT, recurrence)
- Cancer features (site, metastatic, active)
- Cardiovascular (HF, CAD, AF, PH)
- Pulmonary (COPD, asthma, ILD)
- Bleeding risk (prior bleeding, coagulopathy)
- Provoking factors (surgery, trauma, immobility)
- Complications (AKI, bleeding, shock, arrest)

---

## Related Documents

- **Design:** `docs/plans/2025-12-13-diagnosis-encoding-design.md`
- **Implementation Plan:** `docs/plans/2025-12-13-module-05-implementation-plan.md`
- **Module XX (Risk Scores):** `../module_xx_clinical_risk_scores/`

---

**Last Updated:** 2025-12-15
**Tests:** 55 passing
**Phase 1 Status:** COMPLETE
