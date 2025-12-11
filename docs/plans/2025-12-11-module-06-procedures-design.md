# Module 6: Unified Procedure Encoding for PE Trajectory Analysis

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date:** 2025-12-11
**Status:** Design Complete
**Author:** Generated with Claude Code

---

## Executive Summary

Five-layer procedure encoding system serving all analytical methods (GBTM, GRU-D, XGBoost, World Models, TDA). File-based storage (Parquet + HDF5 + SQLite) with SNOMED + CCS normalization. Handles 22M procedure records across CPT (71%), EPIC (25%), HCPCS (2%), and ICD-10-PCS (1%) code types.

**Key Decisions:**
- Full 5-layer implementation
- File-based storage matching Module 4 pattern
- SNOMED + CCS mapping via UMLS/OMOP
- LLM benchmark for EPIC code classification (5 models)
- Hierarchical co-occurrence embeddings (CCS + PE-specific)
- World model dual representation (actions + states) with discretion weighting
- 7 temporal categories including lifetime procedure history
- Binary indicators only for echo/cath/PFT (structured data in future separate modules)

**Data Coverage (from Prc.txt analysis):**
- CTA chest (71275): 85K+ records
- Echo (93306): 104K+ records
- IVC filter (37191): 1,436 records
- Catheter-directed therapy (37211/12/14): 432 records
- Intubation (31500): 3,629 records
- ECMO (33946/47): 492 records
- CPR (92950): 478 records
- Transfusion (36430): 13K+ records

**Note:** Systemic thrombolysis (tPA/alteplase) is in Med.txt (Medications Module), not Prc.txt.

---

## Architecture Overview

### Directory Structure

```
module_06_procedures/
├── config/
│   ├── procedure_config.py          # Central configuration
│   ├── ccs_mappings.yaml            # CCS category definitions
│   ├── surgical_risk.yaml           # Surgical risk classifications
│   ├── pe_procedure_codes.yaml      # PE-specific CPT code lists
│   └── discretion_weights.yaml      # Action discretion weights for world models
├── data/
│   ├── vocabularies/                # Reference tables
│   │   ├── snomed_procedures.db     # SNOMED-CT procedures (SQLite)
│   │   ├── ccs_crosswalk.csv        # CPT/ICD → CCS mappings
│   │   └── setup_vocabularies.py
│   ├── bronze/                      # Layer 1: Raw extracted
│   │   └── canonical_procedures.parquet
│   ├── silver/                      # Mapped + classified
│   │   ├── mapped_procedures.parquet
│   │   └── mapping_failures.parquet
│   ├── gold/                        # Analysis-ready
│   │   ├── ccs_indicators/          # Layer 2
│   │   ├── pe_procedure_features/   # Layer 3
│   │   └── world_model_states/      # Layer 5
│   └── embeddings/                  # Layer 4
│       └── procedure_embeddings.h5
├── extractors/
│   ├── canonical_extractor.py       # Layer 1: Prc.txt → Bronze
│   ├── vocabulary_mapper.py         # Bronze → Silver (CCS + SNOMED)
│   └── llm_classifier.py            # EPIC code classification
├── transformers/
│   ├── ccs_indicator_builder.py     # Layer 2
│   ├── pe_feature_builder.py        # Layer 3
│   ├── embedding_generator.py       # Layer 4
│   └── world_model_builder.py       # Layer 5
├── exporters/
│   ├── gbtm_exporter.py
│   ├── grud_exporter.py
│   └── xgboost_exporter.py
├── validation/
│   ├── llm_benchmark.py             # Compare LLM models
│   └── layer_validators.py
└── tests/
```

### Data Flow

```
Prc.txt (22M records, 6.4 GB)
         ↓
    Layer 1: Canonical Extraction (7 temporal flags)
         ↓
    Bronze (canonical_procedures.parquet)
         ↓
    Vocabulary Mapping (CPT→CCS→SNOMED, EPIC→LLM→CCS)
         ↓
    Silver (mapped_procedures.parquet)
         ↓
    ┌─────────────────────────────────────────────┐
    │  Parallel Layer Generation                  │
    ├─────────────────────────────────────────────┤
    │  Layer 2: CCS Categories + Surgical Risk    │
    │  Layer 3: PE-Specific Features              │
    │  Layer 4: Embeddings (5 types)              │
    │  Layer 5: World Model States/Actions        │
    └─────────────────────────────────────────────┘
         ↓
    Method-Specific Exports
```

---

## Layer 1: Canonical Procedure Records

**Purpose:** Single source of truth for all procedure events with temporal classification.

**Input:** `Data/Prc.txt` (22M records, pipe-delimited)

**Output:** `data/bronze/canonical_procedures.parquet`

### Input Schema (Prc.txt)

| Column | Description |
|--------|-------------|
| EMPI | Patient identifier |
| Date | Procedure date |
| Procedure_Name | Description text |
| Code_Type | CPT, EPIC, HCPCS, ICD10, ICD9 |
| Code | Procedure code |
| Quantity | Procedure quantity |
| Provider | Performing provider |
| Clinic | Location/unit |
| Hospital | Facility (MGH, BWH, etc.) |
| Inpatient_Outpatient | Visit type |
| Encounter_number | Links to encounters |

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| empi | str | Patient identifier |
| procedure_datetime | datetime | Actual procedure date/time |
| procedure_date | date | Date only (for aggregation) |
| procedure_name | str | Original description |
| code_type | str | CPT/EPIC/HCPCS/ICD10/ICD9 |
| code | str | Original code |
| quantity | float | Quantity |
| provider | str | Performing provider |
| clinic | str | Location |
| hospital | str | Facility |
| inpatient | bool | True if inpatient |
| encounter_id | str | Encounter linkage |
| hours_from_pe | float | Hours relative to PE Time Zero |
| days_from_pe | int | Days relative to PE |

### Temporal Flags (Computed)

| Flag | Condition | Purpose |
|------|-----------|---------|
| is_lifetime_history | Before index admission | Background surgical risk |
| is_remote_antecedent | -∞ to -720h (>30 days pre-PE) | Recent non-provoking |
| is_provoking_window | -720h to 0h (1-30 days pre-PE) | VTE provocation |
| is_diagnostic_workup | -24h to +24h | Index PE workup |
| is_initial_treatment | 0h to +72h | Early treatment |
| is_escalation | >+72h during hospitalization | Complications |
| is_post_discharge | After index discharge | Long-term outcomes |

### Processing Steps

1. Stream Prc.txt in chunks (1M rows)
2. Join with Module 1 patient timelines for PE Time Zero
3. Filter to PE cohort patients
4. Compute hours_from_pe and all temporal flags
5. Parse encounter linkage
6. Write partitioned parquet by EMPI

---

## Code Mapping Strategy

### Code Type Distribution

| Code Type | Records | Percentage | Mapping Approach |
|-----------|---------|------------|------------------|
| CPT | 15.6M | 71% | Direct CCS + SNOMED via OMOP |
| EPIC | 5.5M | 25% | Name-based → LLM classification |
| HCPCS | 446K | 2% | CCS crosswalk |
| ICD-10-PCS | 244K | 1% | Direct CCS + SNOMED |
| ICD-9 Vol 3 | 143K | <1% | CCS crosswalk |

### Mapping Pipeline

```
Unique procedure strings (~50K)
         ↓
Step 1: Standard Code Mapping (~75%)
    - CPT/HCPCS/ICD → CCS category (AHRQ crosswalk)
    - CPT/ICD → SNOMED concept (OMOP)
         ↓
Step 2: EPIC Name-Based Matching (~15%)
    - Normalize procedure_name (lowercase, trim)
    - Fuzzy match against CCS descriptions (>0.85 similarity)
    - Fuzzy match against SNOMED preferred terms
         ↓
Step 3: LLM Classification (~8%)
    - Low-confidence fuzzy matches → LLM
    - Classification task: name + 3 CCS candidates → select best
         ↓
Step 4: Manual Review Queue (~2%)
    - Write to mapping_failures.parquet
```

### LLM Benchmark Design

**Task:** Given procedure name + top 3 CCS candidates from fuzzy match, select the best match (or "none").

**Test Set:** 300 manually-labeled EPIC procedure names → correct CCS category

**Models to Benchmark:**

| Model | Size | Notes |
|-------|------|-------|
| Gemma 2 9B | 9B | Google, good instruction following |
| Mistral 7B | 7B | Fast, efficient |
| Qwen 2.5 7B | 7B | Strong structured extraction |
| Llama 3 8B | 8B | Meta, strong general |
| Phi-3 Mini | 3.8B | Microsoft, capable for size |

**Metrics:**
- Classification accuracy
- Inference speed (strings/sec)
- Confidence calibration

**Selection:** Best accuracy with >10 strings/sec throughput

### Silver Schema Additions

| Column | Type | Description |
|--------|------|-------------|
| ccs_category | str | CCS procedure category code |
| ccs_description | str | CCS category name |
| snomed_concept_id | str | SNOMED-CT concept ID |
| snomed_preferred_term | str | SNOMED preferred name |
| mapping_method | str | direct/fuzzy/llm/manual |
| mapping_confidence | float | 0-1 confidence score |

**Target:** ≥85% successful mapping to CCS

---

## Layer 2: Standard Procedure Groupings

**Purpose:** Validated procedure groupings for dimensionality reduction and risk classification.

**Output:** `data/gold/ccs_indicators/`

### CCS Procedure Categories (~230 categories)

Groups CPT/ICD codes into clinically meaningful categories. Examples:

| CCS Code | Description | Example CPT Codes |
|----------|-------------|-------------------|
| 216 | Respiratory intubation and mechanical ventilation | 31500, 94002-94004 |
| 47 | Diagnostic cardiac catheterization | 93451-93461 |
| 54 | Other vascular catheterization | 36555-36571 |
| 222 | Blood transfusion | 36430, 36440 |
| 39 | Incision of pleura, thoracentesis, chest drainage | 32551, 32554-32557 |

### Surgical Risk Classification

| Risk Level | VTE Risk | Examples |
|------------|----------|----------|
| Very High | >4% 30-day VTE | Hip/knee replacement, cancer surgery, major trauma |
| High | 2-4% | Cardiac surgery, thoracic surgery, major abdominal |
| Moderate | 1-2% | Laparoscopic abdominal, spine surgery |
| Low | <1% | Minor orthopedic, ambulatory surgery |
| Minimal | <0.5% | Endoscopy, skin procedures |

### BETOS Categories

| Category | Description |
|----------|-------------|
| P1 | Major procedures |
| P2 | Minor procedures |
| P3 | Ambulatory procedures |
| I | Imaging |
| T | Tests |
| E | Evaluation and management |

### Invasiveness Classification

| Level | Definition | Examples |
|-------|------------|----------|
| 0 | Non-invasive | External monitoring, cast application |
| 1 | Minimally invasive | Percutaneous, endoscopic |
| 2 | Moderately invasive | Laparoscopic, small incision |
| 3 | Highly invasive | Major incision, organ entry |

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| empi | str | Patient ID |
| temporal_category | str | One of 7 temporal categories |
| ccs_{code} | bool | Binary indicator per CCS category |
| ccs_{code}_count | int | Procedure count in category |
| surgical_risk_level | str | VTE risk classification |
| betos_category | str | BETOS grouping |
| invasiveness_max | int | Max invasiveness in window |
| procedure_count | int | Total procedures in window |

---

## Layer 3: PE-Specific Procedure Features

**Purpose:** Curated features directly relevant to PE trajectory analysis.

**Output:** `data/gold/pe_procedure_features/`

**Note:** Echo, cardiac cath, and PFT produce binary indicators only. Structured measurements (LVEF, PA pressures, FEV1) will be in separate future modules.

### Datetime Preservation

All features include both actual datetime and relative timing:

| Column | Type | Description |
|--------|------|-------------|
| {feature}_datetime | datetime | Actual procedure date/time |
| {feature}_date | date | Actual date |
| {feature}_hours_from_pe | float | Relative timing |

### Lifetime Surgical History (Static Features)

**Prior VTE Procedures:**

| Feature | Type | Description |
|---------|------|-------------|
| prior_ivc_filter_ever | bool | Any prior IVC filter |
| prior_ivc_filter_still_present | bool | Filter not retrieved |
| prior_thrombolysis_for_vte | bool | Prior systemic lysis |
| prior_cdt_for_vte | bool | Prior catheter-directed therapy |
| prior_surgical_embolectomy | bool | Prior surgical clot removal |
| prior_vte_procedure_count | int | Total prior VTE procedures |

**Prior Surgical Risk:**

| Feature | Type | Description |
|---------|------|-------------|
| prior_major_surgery_ever | bool | Any major surgery lifetime |
| prior_orthopedic_surgery_ever | bool | Any orthopedic surgery |
| prior_joint_replacement | bool | Hip/knee replacement |
| prior_spine_surgery | bool | Spine procedure |
| prior_cancer_surgery | bool | Oncologic surgery |
| prior_cardiac_surgery | bool | CABG, valve, etc. |
| lifetime_surgical_count | int | Total lifetime surgeries |
| lifetime_surgical_risk_score | float | Aggregate risk 0-1 |

**Chronic Procedure Markers:**

| Feature | Type | Description |
|---------|------|-------------|
| has_pacemaker_icd | bool | Cardiac device present |
| on_chronic_dialysis | bool | Dialysis access procedures |
| prior_amputation | bool | Vascular disease marker |
| prior_organ_transplant | bool | Transplant history |
| chronic_procedure_burden | int | Count of chronic procedures |

### Provoking Procedures (1-30 Days Pre-PE)

**Recent Surgery:**

| Feature | Type | Description |
|---------|------|-------------|
| surgery_within_30_days | bool | Any surgery in window |
| surgery_type | str | Major/moderate/minor |
| surgery_vte_risk_category | str | High/moderate/low |
| days_from_surgery_to_pe | int | Days since surgery |
| orthopedic_surgery_within_30d | bool | Orthopedic provoking |
| cancer_surgery_within_30d | bool | Cancer provoking |
| neurosurgery_within_30d | bool | Neurosurgery provoking |

**Derived:**

| Feature | Type | Description |
|---------|------|-------------|
| provoked_pe | bool | Any provoking factor present |
| provocation_strength | float | 0-1 provocation score |

### Index Diagnostic Workup (±24h of PE)

**Imaging Procedures:**

| Feature | Type | Description |
|---------|------|-------------|
| cta_chest_performed | bool | CT angiography done |
| cta_datetime | datetime | Actual CTA time |
| cta_hours_from_pe | float | Relative timing |
| vq_scan_performed | bool | V/Q scan done |
| le_duplex_performed | bool | Lower extremity duplex |
| echo_performed | bool | Echo done (data in Echo Module) |
| echo_datetime | datetime | Actual echo time |
| tte_vs_tee | str | Transthoracic vs transesophageal |
| cardiac_cath_performed | bool | Cath done (data in Cath Module) |
| pulmonary_angiography_performed | bool | Gold standard PE imaging |

**Workup Metrics:**

| Feature | Type | Description |
|---------|------|-------------|
| diagnostic_workup_intensity | int | Count of diagnostic tests |
| complete_pe_workup | bool | CTA + Echo + Duplex all done |
| time_to_cta_from_ed_arrival | float | Hours to CTA |
| diagnostic_sequence | str | Encoded sequence |

### Initial Treatment (0-72h Post-PE)

**Reperfusion Therapy:**

| Feature | Type | Description |
|---------|------|-------------|
| any_reperfusion_therapy | bool | Any lysis or intervention |
| systemic_thrombolysis | bool | Full-dose tPA (from Med.txt) |
| thrombolysis_datetime | datetime | Actual lysis time |
| thrombolysis_hours_from_pe | float | Time to lysis |
| catheter_directed_therapy | bool | CDT performed |
| cdt_datetime | datetime | Actual CDT time |
| cdt_type | str | EKOS, AngioJet, aspiration |
| surgical_embolectomy | bool | Open surgical removal |

**IVC Filter:**

| Feature | Type | Description |
|---------|------|-------------|
| ivc_filter_placed | bool | Filter placed |
| ivc_filter_datetime | datetime | Actual placement time |
| ivc_filter_hours_from_pe | float | Time to filter |
| filter_type | str | Retrievable, permanent |

**Vascular Access:**

| Feature | Type | Description |
|---------|------|-------------|
| central_line_placed | bool | CVC placed |
| central_line_datetime | datetime | Actual placement time |
| arterial_line_placed | bool | A-line placed |
| pa_catheter_placed | bool | Swan-Ganz placed |

**Respiratory Support:**

| Feature | Type | Description |
|---------|------|-------------|
| intubation_performed | bool | Intubated |
| intubation_datetime | datetime | Actual intubation time |
| intubation_hours_from_pe | float | Time to intubation |
| intubation_indication | str | Respiratory failure, procedure, arrest |
| hfnc_initiated | bool | High-flow nasal cannula |
| nippv_initiated | bool | BiPAP/CPAP |

**Circulatory Support:**

| Feature | Type | Description |
|---------|------|-------------|
| ecmo_initiated | bool | ECMO started |
| ecmo_datetime | datetime | Actual cannulation time |
| ecmo_hours_from_pe | float | Time to ECMO |
| ecmo_type | str | VA vs VV |
| mechanical_support_level | int | 0-4 ordinal scale |

### Escalation/Complications (>72h Post-PE)

**Respiratory Escalation:**

| Feature | Type | Description |
|---------|------|-------------|
| delayed_intubation | bool | Intubation >72h post-PE |
| reintubation | bool | Extubated then reintubated |
| tracheostomy | bool | Trach placed |
| prolonged_ventilation | bool | >7 days on vent |
| vent_free_days_at_28 | int | VFD outcome metric |

**Bleeding Complications:**

| Feature | Type | Description |
|---------|------|-------------|
| any_transfusion | bool | Any blood products |
| first_transfusion_datetime | datetime | First transfusion time |
| rbc_units_total | int | Cumulative RBC units |
| massive_transfusion | bool | ≥10 units in 24h |
| gi_endoscopy_for_bleeding | bool | EGD/colonoscopy for bleed |
| ir_embolization_for_bleeding | bool | IR intervention |

**Other Complications:**

| Feature | Type | Description |
|---------|------|-------------|
| cardiac_arrest_post_pe | bool | Arrest occurred |
| cardiac_arrest_datetime | datetime | Actual arrest time |
| rosc_achieved | bool | Return of spontaneous circulation |
| rrt_initiated | bool | Dialysis started |

**Derived Trajectory Features:**

| Feature | Type | Description |
|---------|------|-------------|
| uncomplicated_course | bool | No escalation procedures |
| procedure_intensity_score | float | 0-1 burden score |
| escalation_trajectory | str | Improving/stable/worsening |
| max_support_level_reached | int | Peak support (0-5) |

---

## Layer 4: Procedure Embeddings

**Purpose:** Dense vector representations capturing procedure relationships.

**Output:** `data/embeddings/procedure_embeddings.h5`

### Embedding Types and Weights

| Type | Dimensions | Weight | Source |
|------|------------|--------|--------|
| Ontological | 128 | 1.0 | Node2Vec on SNOMED + CCS hierarchy |
| Semantic | 768 → 128 | 0.8 | BioBERT on procedure descriptions, PCA reduced |
| Temporal Sequence | 128 | 0.9 | Transformer on procedure ordering within encounters |
| CCS Co-occurrence | 128 | 0.6 | Word2Vec on CCS category sequences |
| PE-Specific Co-occurrence | 64 | 0.5 | Word2Vec on PE-relevant procedure sequences |
| Procedural Complexity | 16 | 0.7 | Structured features |

### Ontological Embeddings (128 dims)

**Graph construction:**
- Nodes: SNOMED procedure concepts + CCS categories
- Edges: SNOMED hierarchy (is_a), CCS groupings, related procedures

**Training:** Node2Vec with p=1, q=0.5 (biased toward local structure)

### Semantic Embeddings (768 → 128 dims)

**Source:** BioBERT or PubMedBERT
**Input:** Procedure name + CCS category description
**Method:** Mean-pool last hidden layer, PCA to 128 dims

### Temporal Sequence Embeddings (128 dims)

**Training data:** Procedure sequences within encounters, ordered by timestamp
**Model:** Small transformer or LSTM

### CCS Co-occurrence Embeddings (128 dims)

**Training:** Word2Vec (Skip-gram) on CCS category sequences
- Each encounter = "sentence" of CCS categories
- Window size = 5
- Min count = 10 encounters

### PE-Specific Co-occurrence Embeddings (64 dims)

**Training:** Word2Vec on PE-relevant procedures only (~80 procedures)

### Procedural Complexity Features (16 dims)

| Feature | Description |
|---------|-------------|
| invasiveness_level | 0-3 ordinal |
| anesthesia_type | None/local/regional/general (one-hot) |
| typical_or_time_minutes | Expected OR duration |
| vte_risk_score | Procedure-specific VTE risk |
| bleeding_risk_score | Procedure bleeding risk |
| is_emergent_capable | Can be done emergently |
| requires_or | Requires operating room |
| requires_icu_post | Typically requires ICU after |

### HDF5 Structure

```
/vocabulary/
    /ontological           # (n_procedures, 128)
    /semantic              # (n_procedures, 128)
    /temporal_sequence     # (n_procedures, 128)
    /ccs_cooccurrence      # (n_ccs_categories, 128)
    /pe_cooccurrence       # (n_pe_procedures, 64)
    /complexity            # (n_procedures, 16)
    /code_to_row           # CPT/SNOMED → row index mapping

/patient_embeddings/
    /lifetime_history      # (n_patients, embed_dim)
    /diagnostic_workup     # (n_patients, embed_dim)
    /therapeutic           # (n_patients, embed_dim)
    /complication          # (n_patients, embed_dim)
```

### Validation Criteria

| Test | Expected |
|------|----------|
| Intubation ↔ Mechanical ventilation | Similarity > 0.8 |
| CTA ↔ Echo (PE workup) | Similarity > 0.6 |
| Central line ↔ Arterial line | Similarity > 0.7 |
| Thrombolysis ↔ CDT | Similarity > 0.7 |
| Dental ↔ Cardiac procedures | Similarity < 0.3 |

---

## Layer 5: World Model Procedure Representations

**Purpose:** Procedure components of state and action vectors for world models.

**Output:** `data/gold/world_model_states/`

### Core Principle: Dual Representation

Procedures serve two roles:
1. **Actions** - Clinician decisions that can be counterfactually varied
2. **State Updates** - Events that modify patient state

### Discretion Spectrum

| Discretion Level | Weight | Examples | Treatment |
|------------------|--------|----------|-----------|
| High | 1.0 | Thrombolysis, IVC filter, CDT | Full action |
| Moderate | 0.6-0.8 | Intubation for respiratory failure, ECMO | Weighted action |
| Low | 0.2-0.4 | Transfusion for hemorrhage | Minimal action |
| None | 0.0 | CPR for cardiac arrest | State update only |

### Action Vector Components (per timestep)

```python
action_vector_t = {
    # High-discretion PE therapeutic actions
    'thrombolysis_action': binary,
    'cdt_action': binary,
    'ivc_filter_action': binary,
    'surgical_embolectomy_action': binary,

    # Moderate-discretion support actions (weighted)
    'intubation_action': binary * 0.7,
    'ecmo_action': binary * 0.7,
    'vasopressor_escalation_action': binary * 0.6,

    # Aggregated procedure action embedding
    'procedure_action_embedding': 128-dim,

    # Action intensity features
    'num_therapeutic_procedures': int,
    'max_invasiveness': 0-3,
    'escalation_indicator': binary,
}
```

### State Vector Components (per timestep)

```python
state_vector_t = {
    # Current support status
    'on_mechanical_ventilation': binary,
    'on_vasopressors': binary,
    'on_ecmo': binary,
    'has_central_access': binary,
    'support_level': 0-5 ordinal,

    # Time on support
    'hours_on_ventilator': float,
    'hours_on_ecmo': float,

    # Complication markers (irreversible once true)
    'cardiac_arrest_occurred': binary,
    'major_bleeding_occurred': binary,
    'rrt_initiated': binary,

    # Cumulative burden
    'cumulative_rbc_units': int,
    'cumulative_invasive_procedures': int,
    'procedure_intensity_score': 0-1,

    # Procedure-derived state embedding
    'procedure_state_embedding': 128-dim,
}
```

### Static Procedure State (computed once per patient)

```python
static_procedure_state = {
    # Lifetime history
    'prior_vte_procedures': int,
    'prior_ivc_filter': binary,
    'prior_major_surgery': binary,
    'lifetime_surgical_risk_score': 0-1,
    'lifetime_procedure_embedding': 128-dim,

    # Provoking factors
    'provoked_pe': binary,
    'days_from_provoking_surgery': int,
    'provocation_strength': 0-1,

    # Chronic conditions
    'has_pacemaker': binary,
    'on_chronic_dialysis': binary,
}
```

### World Model Integration

```
State_t:
├── Vital signs state (from Vitals Module Layer 5)
├── Diagnosis state (from Diagnoses Module)
├── Procedure state (from this Layer 5)
│   ├── static_procedure_state (computed once)
│   └── dynamic_procedure_state_t (evolving)
└── Biomarker state (from Labs Module)

Action_t:
├── Medication action embedding (from Medications Module Layer 4)
└── Procedure action vector (from this Layer 5)

Next_State = WorldModel(State_t, Action_t) + Stochastic_Events
```

---

## Method-Specific Exports

### GBTM / lcmm (R)

**Output:** `exports/gbtm_procedures.csv`

Columns: empi, day_from_pe, provoking features, treatment indicators, support trajectory, outcome markers

### GRU-D (HDF5)

**Output:** `exports/grud_procedures.h5`

```
/procedure_values    # (n_patients, 168, n_features)
/procedure_mask      # (n_patients, 168, n_features)
/procedure_delta     # (n_patients, 168, n_features)
```

### XGBoost (Parquet)

**Output:** `exports/xgboost_procedure_features.parquet`

~500 features: CCS indicators, PE-specific features, embedding PCAs

### World Models (HDF5)

**Output:** `exports/world_model_procedures.h5`

```
/static_state        # (n_patients, static_dim)
/dynamic_state       # (n_patients, n_hours, dynamic_dim)
/action_vectors      # (n_patients, n_hours, action_dim)
```

### TDA / Mapper (Parquet)

**Output:** `exports/tda_procedure_pointcloud.parquet`

---

## Validation Checkpoints

### Layer 1 Validation

| Check | Target |
|-------|--------|
| Records loaded | 22M |
| Patients in cohort | ~8,700 |
| Code type distribution | CPT 71%, EPIC 25% |
| PE time linkage | >95% patients |

### Layer 2 Validation (Mapping)

| Check | Target |
|-------|--------|
| CCS mapping rate (CPT) | >95% |
| CCS mapping rate (overall) | >85% |
| LLM classification accuracy | >85% |

### Layer 3 Validation (PE Features)

| Check | Target |
|-------|--------|
| CTA performed | 80-95% of patients |
| Echo performed | 50-70% of patients |
| Intubation rate | 5-15% of patients |
| IVC filter rate | 5-15% of patients |
| ECMO rate | <2% of patients |

### Layer 4 Validation (Embeddings)

| Check | Target |
|-------|--------|
| Similar procedures cluster | Similarity > 0.7 |
| Dissimilar procedures separate | Similarity < 0.3 |

### Cross-Layer Validation

| Check | Target |
|-------|--------|
| All patients in all layers | 100% |
| Timestamps aligned | All use same PE Time Zero |

---

## Implementation Phases

### Phase 1: Setup & Layer 1 Canonical Extraction

- Directory structure and configuration
- `extractors/canonical_extractor.py`
- `data/bronze/canonical_procedures.parquet` with 7 temporal flags

### Phase 2: Code Mapping Pipeline

- Vocabulary setup (CCS + SNOMED)
- `extractors/vocabulary_mapper.py`
- `extractors/llm_classifier.py`
- LLM benchmark (5 models)
- `data/silver/mapped_procedures.parquet`

### Phase 3: Layer 2 - Standard Groupings

- `transformers/ccs_indicator_builder.py`
- CCS categories, surgical risk, BETOS

### Phase 4: Layer 3 - PE-Specific Features

- `transformers/pe_feature_builder.py`
- All PE-specific features with datetime preservation

### Phase 5: Layer 4 - Embeddings

- `transformers/embedding_generator.py`
- All 6 embedding types
- Patient-level aggregations

### Phase 6: Layer 5 - World Model States/Actions

- `transformers/world_model_builder.py`
- Static and dynamic procedure states
- Discretion-weighted actions

### Phase 7: Exporters

- GBTM, GRU-D, XGBoost, World Model, TDA exports

### Phase 8: Validation & Documentation

- `validation/layer_validators.py`
- All validation checkpoints
- Documentation

### Dependency Graph

```
Phase 1 → Phase 2 → Phase 3 ─┐
                    ↓        │
                Phase 4 ◄────┤
                    ↓        │
                Phase 5 ◄────┘
                    ↓
                Phase 6 → Phase 7 → Phase 8
```

### Estimated Scope

| Phase | New Files | Tests |
|-------|-----------|-------|
| 1 | 4 | 15 |
| 2 | 5 | 20 |
| 3 | 3 | 12 |
| 4 | 2 | 25 |
| 5 | 2 | 15 |
| 6 | 2 | 18 |
| 7 | 5 | 10 |
| 8 | 2 | 10 |
| **Total** | **25** | **125** |

---

## Future Modules (Separate from Module 6)

The following will be implemented as separate modules, NOT part of Module 6:

- **Echo Module:** Structured measurements (LVEF, RV/LV ratio, TAPSE, TR velocity)
- **Cardiac Cath Module:** PA pressures, wedge, cardiac output, coronary anatomy
- **PFT Module:** FEV1, FVC, DLCO, lung volumes

Module 6 provides binary indicators that these tests occurred; the specialized modules will contain the structured numeric data.

---

**END OF DESIGN DOCUMENT**
