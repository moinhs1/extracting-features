# Module 05: Diagnosis Processing

Diagnosis encoding pipeline for PE trajectory analysis. Transforms raw ICD codes into PE-aligned canonical records, comorbidity indices, PE-specific features, embeddings, and world model states.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Layers 1-2 (Core + Comorbidity) | **COMPLETE** |
| **Phase 2** | Layer 3 (PE-Specific Features) | **COMPLETE** |
| Phase 3 | Layers 4-5 (Embeddings, World Models) | Planned |

**Current Stats:**
- 155 tests passing
- Phase 2 complete with validation tests
- 3 layers operational

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

## Layer 3 Schema

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| **VTE History** | | |
| prior_pe_ever | bool | Any PE before index |
| prior_pe_months | float | Months since last PE |
| prior_pe_count | int | Number of prior PEs |
| prior_dvt_ever | bool | Any DVT before index |
| prior_dvt_months | float | Months since last DVT |
| prior_vte_count | int | Total prior VTE events |
| is_recurrent_vte | bool | Any prior VTE |
| **PE Characterization** | | |
| pe_subtype | str | saddle/subsegmental/other/unspecified |
| pe_bilateral | bool | Bilateral involvement |
| pe_with_cor_pulmonale | bool | Acute cor pulmonale |
| pe_high_risk_code | bool | High-risk ICD pattern |
| **Cancer** | | |
| cancer_active | bool | Active malignancy |
| cancer_site | str | Primary site category |
| cancer_metastatic | bool | Metastatic disease |
| cancer_recent_diagnosis | bool | Dx within 6 months |
| cancer_on_chemotherapy | bool | Chemotherapy codes |
| **Cardiovascular** | | |
| heart_failure | bool | Any HF |
| heart_failure_type | str | HFrEF/HFpEF/unspecified |
| coronary_artery_disease | bool | CAD/MI |
| atrial_fibrillation | bool | AF/AFL |
| pulmonary_hypertension | bool | Pre-existing PH |
| valvular_heart_disease | bool | Valve disease |
| **Pulmonary** | | |
| copd | bool | COPD |
| asthma | bool | Asthma |
| interstitial_lung_disease | bool | ILD |
| home_oxygen | bool | Chronic O2 |
| prior_respiratory_failure | bool | Prior resp failure |
| **Bleeding Risk** | | |
| prior_major_bleeding | bool | Major bleed history |
| prior_gi_bleeding | bool | GI bleed history |
| prior_intracranial_hemorrhage | bool | ICH history |
| active_peptic_ulcer | bool | Active PUD |
| thrombocytopenia | bool | Low platelets |
| coagulopathy | bool | Coagulation disorder |
| **Renal** | | |
| ckd_stage | int | 0-5 |
| ckd_dialysis | bool | On dialysis |
| aki_at_presentation | bool | AKI at PE index |
| **Provoking Factors** | | |
| recent_surgery | bool | Surgery within 30d |
| recent_trauma | bool | Major trauma |
| immobilization | bool | Immobility codes |
| pregnancy_related | bool | Pregnancy/postpartum |
| hormonal_therapy | bool | OCP/HRT |
| central_venous_catheter | bool | CVC |
| is_provoked_vte | bool | Any provoking factor |
| **Complications** | | |
| complication_aki | bool | Post-PE AKI |
| complication_bleeding_any | bool | Any bleeding |
| complication_bleeding_major | bool | Major bleeding |
| complication_ich | bool | Post-PE ICH |
| complication_respiratory_failure | bool | Resp failure |
| complication_cardiogenic_shock | bool | Cardiogenic shock |
| complication_cardiac_arrest | bool | Cardiac arrest |
| complication_recurrent_vte | bool | Recurrent VTE |
| complication_cteph | bool | CTEPH |

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

## Next Steps (Phase 3)

Layers 4-5: Embeddings and World Model States
- Layer 4: Diagnosis embeddings (ontological + co-occurrence)
- Layer 5: World model states (static + dynamic)
- Integration with trajectory analysis
- Outcome prediction models

---

## Related Documents

- **Design:** `docs/plans/2025-12-13-diagnosis-encoding-design.md`
- **Implementation Plan:** `docs/plans/2025-12-13-module-05-implementation-plan.md`
- **Module XX (Risk Scores):** `../module_xx_clinical_risk_scores/`

---

**Last Updated:** 2025-12-15
**Tests:** 55 passing
**Phase 1 Status:** COMPLETE
