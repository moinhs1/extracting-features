# Module 4: Unified Medication Encoding for PE Trajectory Analysis

**Date:** 2025-12-08
**Status:** Design Complete
**Author:** Generated with Claude Code

---

## Executive Summary

Five-layer medication encoding system serving all analytical methods (GBTM, GRU-D, XGBoost, World Models, TDA). File-based storage (Parquet + HDF5 + SQLite) with RxNorm normalization targeting ≥85% mapping success.

**Key Decisions:**
- Full 5-layer implementation
- File-based storage (no PostgreSQL/Delta Lake required)
- RxNorm download from UMLS for mapping
- All 5 embedding types (Semantic, Ontological, Co-occurrence, Pharmacokinetic, Hierarchical)
- Hybrid + LLM dose parsing with model benchmarking
- All three dose standardizations (raw, DDD-normalized, weight-adjusted)
- 53 therapeutic classes with expanded cardiovascular granularity
- Multiple temporal resolutions (daily, hourly, window-based)

---

## Architecture Overview

### Directory Structure

```
module_04_medications/
├── config/
│   ├── therapeutic_classes.yaml      # Class hierarchy definition
│   ├── dose_patterns.yaml            # Regex patterns for dose extraction
│   └── medication_config.py          # Central configuration
├── data/
│   ├── rxnorm/                        # RxNorm SQLite + setup scripts
│   │   ├── rxnorm.db                  # Loaded RxNorm tables
│   │   └── setup_rxnorm.py
│   ├── bronze/                        # Raw extracted records
│   │   └── canonical_records.parquet
│   ├── silver/                        # Cleaned + RxNorm mapped
│   │   ├── mapped_medications.parquet
│   │   └── mapping_failures.parquet   # For review/LLM processing
│   ├── gold/                          # Analysis-ready
│   │   ├── therapeutic_classes/       # Layer 2 (partitioned)
│   │   ├── individual_indicators/     # Layer 3 (sparse)
│   │   └── dose_intensity/            # Layer 5
│   └── embeddings/                    # Layer 4
│       └── medication_embeddings.h5
├── extractors/
│   ├── canonical_extractor.py         # Layer 1: Raw → Bronze
│   ├── rxnorm_mapper.py               # Bronze → Silver
│   ├── dose_parser.py                 # Hybrid + LLM dose extraction
│   └── llm_dose_parser.py             # LLM fallback for ambiguous
├── transformers/
│   ├── class_indicator_builder.py     # Layer 2
│   ├── individual_indicator_builder.py # Layer 3
│   ├── embedding_generator.py         # Layer 4
│   └── dose_intensity_builder.py      # Layer 5
├── exporters/
│   ├── gbtm_exporter.py               # R-ready CSVs
│   ├── grud_exporter.py               # HDF5 tensors
│   └── xgboost_exporter.py            # Wide tabular
├── validation/
│   ├── llm_benchmark.py               # Compare LLM models for parsing
│   └── layer_validators.py            # Cross-layer consistency checks
└── tests/
```

### Data Flow

```
Med.txt (18.6M records)
         ↓
    Layer 1: Canonical Extraction
         ↓
    Bronze (canonical_records.parquet)
         ↓
    RxNorm Mapping (Exact → Fuzzy → Ingredient → LLM)
         ↓
    Silver (mapped_medications.parquet)
         ↓
    ┌────────────────────────────────────────┐
    │  Parallel Layer Generation             │
    ├────────────────────────────────────────┤
    │  Layer 2: Therapeutic Classes (Gold)   │
    │  Layer 3: Individual Indicators (Gold) │
    │  Layer 4: Embeddings (HDF5)            │
    │  Layer 5: Dose Intensity (Gold)        │
    └────────────────────────────────────────┘
         ↓
    Method-Specific Exports
    ├── GBTM: CSV (long/wide)
    ├── GRU-D: HDF5 tensors
    ├── XGBoost: Parquet (wide)
    ├── World Models: HDF5 actions
    └── TDA: Point cloud parquet
```

---

## Layer 1: Canonical Medication Records

**Purpose:** Single source of truth for all medication events.

**Input:** `Data/Med.txt` (18.6M records, pipe-delimited)

**Output:** `data/bronze/canonical_records.parquet`

### Schema

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `empi` | str | EMPI | Patient identifier |
| `encounter_id` | str | Encounter_number | Links to Module 1 encounters |
| `medication_date` | date | Medication_Date | Day of administration |
| `hours_from_t0` | float | Computed | Hours relative to PE Time Zero |
| `original_string` | str | Medication | Raw RPDR medication text |
| `code_type` | str | Code_Type | BWH_CC, HCPCS, CPT, EPIC-MED |
| `code` | str | Code | Original code value |
| `quantity` | float | Quantity | Quantity administered |
| `inpatient` | bool | Inpatient_Outpatient | True if inpatient |
| `provider` | str | Provider | Prescribing provider |
| `clinic` | str | Clinic | Location/unit |
| `hospital` | str | Hospital | Facility |

### Parsed Components

| Column | Type | Description |
|--------|------|-------------|
| `parsed_name` | str | Cleaned medication name |
| `parsed_dose_value` | float | Numeric dose extracted |
| `parsed_dose_unit` | str | Unit (mg, mcg, units, ml, etc.) |
| `parsed_route` | str | Route if detectable (IV, PO, SC, etc.) |
| `parsed_frequency` | str | Frequency if detectable (QD, BID, PRN) |
| `parse_method` | str | 'regex', 'rxnorm', 'llm', 'failed' |
| `parse_confidence` | float | 0-1 confidence score |

### Processing Steps

1. **Load Med.txt** - Stream in chunks (1M rows) for memory efficiency
2. **Join with Module 1** - Get PE Time Zero per patient, compute `hours_from_t0`
3. **Filter to cohort** - Keep only patients in PE cohort
4. **Initial parse** - Regex extraction of name, dose, unit, route, frequency
5. **Deduplicate strings** - Build vocabulary of unique medication strings (~15K expected)
6. **Write Bronze** - Partitioned by patient EMPI

---

## RxNorm Setup & Mapping

### UMLS/RxNorm Setup (One-Time)

1. **Register for UMLS account:** https://uts.nlm.nih.gov/uts/signup-login
   - Free for research use
   - Instant approval with institutional email

2. **Download RxNorm Full Release:**
   - https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
   - ~300MB zip → ~2GB unzipped
   - Key files:
     - `RXNCONSO.RRF` - Concept names and synonyms
     - `RXNREL.RRF` - Relationships
     - `RXNSAT.RRF` - Attributes (NDC, strengths)

3. **Load into SQLite:**
   - Script: `data/rxnorm/setup_rxnorm.py`
   - Tables: `rxnconso`, `rxnrel`, `rxnsat`
   - ~5 minutes to load, ~1.5GB database

### Mapping Pipeline

```
Unique medication strings (~15K)
         ↓
Step 1: Exact Match (~30-40%)
    - Normalize string (lowercase, trim)
    - Match against RXNCONSO.STR
         ↓
Step 2: Fuzzy Match (+30-35%)
    - Extract drug name prefix
    - Levenshtein/Jaro-Winkler >0.85
         ↓
Step 3: Ingredient Extraction (+15-20%)
    - Parse first word(s) as ingredient
    - Match to RxNorm ingredients (TTY=IN)
         ↓
Step 4: LLM-Assisted (+3-8%)
    - Local LLM extracts drug name
    - Re-attempt RxNorm match
         ↓
Step 5: Manual Review Queue (~2-5%)
    - Write to mapping_failures.parquet
```

### LLM Benchmark for Dose Parsing

| Model | Size | Notes |
|-------|------|-------|
| Llama 3 8B | 8B | Meta, strong general performance |
| Mistral 7B | 7B | Fast, efficient |
| Phi-3 Mini | 3.8B | Microsoft, capable for size |
| Gemma 2 9B | 9B | Google, good instruction following |
| Qwen 2.5 7B | 7B | Alibaba, strong on structured extraction |
| Qwen 2.5 14B | 14B | Larger option if resources allow |

**Benchmark Protocol:**
- Test set: 200 manually-labeled ambiguous medication strings
- Labels: drug_name, dose_value, dose_unit, route, frequency
- Metrics: Exact match accuracy, per-field F1, inference speed
- Selection: Best accuracy with acceptable speed (>10 strings/sec)

### Silver Schema

| Column | Type | Description |
|--------|------|-------------|
| *(all Bronze columns)* | | Carried forward |
| `rxcui` | str | RxNorm Concept Unique Identifier |
| `rxnorm_name` | str | Canonical RxNorm name |
| `rxnorm_tty` | str | Term type (SCD, SBD, IN, etc.) |
| `ingredient_rxcui` | str | Ingredient-level RXCUI |
| `ingredient_name` | str | Generic ingredient name |
| `atc_code` | str | ATC classification code |
| `atc_class` | str | ATC therapeutic class name |
| `mapping_method` | str | exact/fuzzy/ingredient/llm/manual |
| `mapping_confidence` | float | 0-1 confidence |

**Target:** ≥85% successful RxNorm mapping

---

## Layer 2: Therapeutic Class Indicators

**Purpose:** 53 binary indicators organized by clinical hierarchy.

**Output:** `data/gold/therapeutic_classes/` (partitioned parquet)

### Class Hierarchy

#### PE-Critical Anticoagulants (9 classes)

| Class ID | Class Name | RxNorm Ingredients | Dose Threshold |
|----------|------------|-------------------|----------------|
| `ac_ufh_ther` | UFH Therapeutic | heparin | >10,000 units/day or infusion |
| `ac_ufh_proph` | UFH Prophylactic | heparin | ≤10,000 units/day SC |
| `ac_lmwh_ther` | LMWH Therapeutic | enoxaparin, dalteparin, tinzaparin | 1mg/kg or weight-based |
| `ac_lmwh_proph` | LMWH Prophylactic | enoxaparin, dalteparin, tinzaparin | Fixed low dose (30-40mg) |
| `ac_fondaparinux` | Fondaparinux | fondaparinux | Any |
| `ac_xa_inhibitor` | Factor Xa Inhibitors | rivaroxaban, apixaban, edoxaban | Any |
| `ac_dti` | Direct Thrombin Inhibitors | dabigatran, argatroban, bivalirudin | Any |
| `ac_vka` | Vitamin K Antagonists | warfarin | Any |
| `ac_thrombolytic` | Thrombolytics | alteplase, tenecteplase, reteplase | Any |

#### Cardiovascular - Vasopressors (6 classes, expanded)

| Class ID | Class Name | RxNorm Ingredients |
|----------|------------|-------------------|
| `cv_norepinephrine` | Norepinephrine | norepinephrine |
| `cv_epinephrine` | Epinephrine | epinephrine (IV/infusion) |
| `cv_vasopressin` | Vasopressin | vasopressin |
| `cv_phenylephrine` | Phenylephrine | phenylephrine (IV) |
| `cv_dopamine` | Dopamine | dopamine |
| `cv_vasopressor_any` | Any Vasopressor | Union of above |

#### Cardiovascular - Inotropes (4 classes)

| Class ID | Class Name | RxNorm Ingredients |
|----------|------------|-------------------|
| `cv_dobutamine` | Dobutamine | dobutamine |
| `cv_milrinone` | Milrinone | milrinone |
| `cv_digoxin` | Digoxin | digoxin |
| `cv_inotrope_any` | Any Inotrope | Union of above |

#### Cardiovascular - Other (8 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `cv_beta_blocker` | Beta Blockers | metoprolol, carvedilol, atenolol |
| `cv_ace_arb` | ACE-I / ARBs | lisinopril, losartan |
| `cv_ccb` | Calcium Channel Blockers | amlodipine, diltiazem |
| `cv_diuretic_loop` | Loop Diuretics | furosemide, bumetanide |
| `cv_diuretic_thiazide` | Thiazide Diuretics | hydrochlorothiazide |
| `cv_diuretic_potassium` | K-Sparing Diuretics | spironolactone |
| `cv_antiarrhythmic` | Antiarrhythmics | amiodarone, lidocaine |
| `cv_nitrate` | Nitrates | nitroglycerin, isosorbide |

#### Anti-Inflammatory (4 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `ai_steroid_systemic` | Systemic Corticosteroids | prednisone, methylprednisolone, dexamethasone |
| `ai_steroid_iv` | IV Corticosteroids | methylprednisolone IV, dexamethasone IV |
| `ai_steroid_inhaled` | Inhaled Corticosteroids | fluticasone, budesonide |
| `ai_nsaid` | NSAIDs | ibuprofen, ketorolac, naproxen |

#### Pain & Sedation (4 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `ps_opioid` | Opioids | morphine, fentanyl, hydromorphone |
| `ps_acetaminophen` | Acetaminophen | acetaminophen |
| `ps_benzodiazepine` | Benzodiazepines | midazolam, lorazepam |
| `ps_sedative_other` | Other Sedatives | propofol, dexmedetomidine |

#### Anti-Infectives (4 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `ai_antibiotic` | Antibiotics | cefazolin, vancomycin |
| `ai_antiviral` | Antivirals | acyclovir, oseltamivir |
| `ai_antifungal` | Antifungals | fluconazole, micafungin |
| `ai_antibiotic_broad` | Broad-Spectrum Abx | meropenem, pip-tazo |

#### Hematologic (4 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `hm_antiplatelet` | Antiplatelets | aspirin, clopidogrel |
| `hm_reversal_heparin` | Heparin Reversal | protamine |
| `hm_reversal_xa` | Xa Reversal | andexanet alfa |
| `hm_reversal_dti` | DTI Reversal | idarucizumab |

#### GI & Metabolic (4 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `gi_ppi` | PPIs | omeprazole, pantoprazole |
| `gi_antiemetic` | Antiemetics | ondansetron, metoclopramide |
| `gi_laxative` | Laxatives | docusate, senna |
| `met_insulin` | Insulin | insulin regular, lispro |
| `met_oral_hypoglycemic` | Oral Hypoglycemics | metformin, glipizide |

#### Respiratory (2 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `resp_bronchodilator` | Bronchodilators | albuterol, ipratropium |
| `resp_neuromuscular` | Neuromuscular Blockers | rocuronium, cisatracurium |

#### ICU-Specific (3 classes)

| Class ID | Class Name | Examples |
|----------|------------|----------|
| `icu_stress_ulcer` | Stress Ulcer Prophylaxis | famotidine, pantoprazole IV |
| `icu_dvt_proph` | DVT Prophylaxis | heparin SQ 5000, enoxaparin 40mg |
| `icu_electrolyte` | IV Electrolytes | potassium chloride IV, magnesium IV |

**Total: 53 therapeutic class indicators**

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `empi` | str | Patient ID |
| `time_window` | str | 'baseline', 'acute', 'subacute', 'recovery' |
| `window_start_hours` | int | Start of window relative to T0 |
| `window_end_hours` | int | End of window |
| `{class_id}` | bool | One column per class (53 columns) |
| `{class_id}_count` | int | Number of administrations in window |
| `{class_id}_first_hours` | float | Hours from T0 to first admin |

---

## Layer 3: Individual Medication Indicators

**Purpose:** Sparse binary indicators for specific medications.

**Output:** `data/gold/individual_indicators/` (sparse parquet)

### Selection Criteria

| Rule | Description |
|------|-------------|
| Prevalence threshold | Medication appears in ≥20 unique patients |
| Anticoagulant exception | ALL anticoagulants included regardless |
| Vasopressor exception | ALL vasopressors included regardless |
| Thrombolytic exception | ALL thrombolytics included regardless |

**Expected:** 200-400 individual medication indicators

### Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `empi` | str | Patient ID |
| `time_window` | str | Window name |
| `med_{name}` | bool | Medication administered (200-400 columns) |
| `med_{name}_count` | int | Number of administrations |
| `med_{name}_total_dose` | float | Cumulative dose |

---

## Layer 4: Continuous Embeddings

**Purpose:** Multiple embedding types for neural methods and world models.

**Output:** `data/embeddings/medication_embeddings.h5`

### Embedding Types

#### 1. Semantic (768 dims)
- **Source:** BioBERT or PubMedBERT
- **Input:** RxNorm canonical name + ingredient name
- **Method:** Mean-pool last hidden layer
- **Captures:** Name/description similarity

#### 2. Ontological (128 dims)
- **Source:** Node2Vec on RxNorm relationship graph
- **Graph:** Nodes=RxCUIs, Edges=RXNREL relationships
- **Method:** Node2Vec with p=1, q=0.5
- **Captures:** Mechanism, class membership, interactions

#### 3. Co-occurrence (128 dims)
- **Source:** Word2Vec on RPDR medication sequences
- **Input:** Per-patient medication sequence by time
- **Method:** Skip-gram, window=5, min_count=20
- **Captures:** MGB-specific prescribing patterns

#### 4. Pharmacokinetic (10 dims)
- **Source:** DrugBank / clinical references
- **Features:** half_life, onset, peak, duration, bioavailability, protein_binding, volume_distribution, clearance, therapeutic_index, active_metabolites
- **Method:** Normalize to [0,1], concatenate

#### 5. Hierarchical Composite (128 dims)
- **Construction:** class_centroid + subclass_residual + individual_residual
- **Method:** Decompose semantic hierarchically, PCA to 128
- **Captures:** Multi-level dynamics for world models

### Patient-Level Aggregations

| Aggregation | Description |
|-------------|-------------|
| `mean_embedding` | Mean across all meds in window |
| `max_embedding` | Element-wise max |
| `dose_weighted_embedding` | Weighted by DDD ratio |
| `class_embeddings` | Separate mean per class |

### HDF5 Structure

```
/vocabulary/
    /semantic           # (n_meds, 768)
    /ontological        # (n_meds, 128)
    /cooccurrence       # (n_meds, 128)
    /pharmacokinetic    # (n_meds, 10)
    /hierarchical       # (n_meds, 128)
    /rxcui_index        # RxCUI → row mapping
    /name_index         # name → row mapping

/patient_embeddings/
    /hourly/            # (n_patients, n_hours, embed_dim)
    /daily/             # (n_patients, n_days, embed_dim)
    /window/            # (n_patients, 4, embed_dim)

/aggregations/
    /mean/              # (n_patients, n_windows, embed_dim)
    /max/
    /dose_weighted/
```

---

## Layer 5: Dose-Intensity Features

**Purpose:** Continuous features capturing medication intensity.

**Output:** `data/gold/dose_intensity/` (parquet)

### Per Therapeutic Class Features

| Feature | Type | Description |
|---------|------|-------------|
| `{class}_total_daily_dose` | float | Sum in standardized units |
| `{class}_dose_ddd_ratio` | float | Dose / WHO DDD |
| `{class}_cumulative_exposure` | float | Running total from T0 |
| `{class}_dose_trend` | int | -1/0/+1 (over 24h) |
| `{class}_hours_since_last` | float | Hours since last admin |
| `{class}_admin_count_24h` | int | Count in rolling 24h |

### PE-Critical Anticoagulant Features

| Medication | Features |
|------------|----------|
| **UFH** | units_per_hour, units_per_kg_hour, bolus_given, bolus_units |
| **Enoxaparin** | mg_per_dose, mg_per_kg, frequency, therapeutic (bool) |
| **Warfarin** | daily_dose_mg, dose_change, inr_at_dose |
| **DOACs** | dose_mg, reduced_dose (bool), type |

### Vasopressor Features

| Feature | Description |
|---------|-------------|
| `vasopressor_count` | Concurrent vasopressors |
| `norepi_mcg_per_kg_min` | Standard dose measure |
| `vaso_units_per_min` | Vasopressin dose |
| `pressor_escalation` | Dose/agent increased |
| `max_pressor_intensity` | Peak composite score |

### Standardization (All Three)

1. **Raw values:** Original with units
2. **DDD-normalized:** Dose / WHO Defined Daily Dose
3. **Weight-adjusted:** mg/kg when weight available

### Temporal Resolutions

| Resolution | Storage | Use Case |
|------------|---------|----------|
| Daily | Primary | GBTM, XGBoost |
| Hourly (imputed) | Derived | GRU-D, multimodal |
| Window-based | Aggregated | Summary stats |

**Hourly Imputation:**
- QD → 09:00
- BID → 09:00, 21:00
- TID → 08:00, 14:00, 20:00
- Q6H → 00:00, 06:00, 12:00, 18:00

---

## Method-Specific Exports

### GBTM / lcmm (R)

**Output:** `exports/gbtm_medication_long.csv`, `exports/gbtm_medication_wide.csv`

- Layer 2: All 53 class indicators
- Layer 5: Dose intensity for key classes
- Time: Daily resolution, day 0-7

### GRU-D (HDF5)

**Output:** `exports/grud_medications.h5`

```
/medication_values   # (n_patients, 168, n_features)
/medication_mask     # (n_patients, 168, n_features)
/medication_delta    # (n_patients, 168, n_features)
```

### XGBoost (Parquet)

**Output:** `exports/xgboost_medication_features.parquet`

~930 features: Layer 2 × 4 windows + Layer 3 top 100 + embedding PCs + Layer 5

### World Models (HDF5)

**Output:** `exports/world_model_actions.h5`

```
/action_embeddings   # (n_patients, n_hours, 128)
/action_magnitudes   # (n_patients, n_hours, 1)
/action_deltas       # (n_patients, n_hours, 128)
```

### TDA / Mapper (Parquet)

**Output:** `exports/tda_medication_pointcloud.parquet`

Patient mean embeddings as coordinates.

---

## Validation Checkpoints

| Layer | Validation | Target |
|-------|------------|--------|
| Layer 1 | RxNorm mapping rate | ≥85% |
| Layer 1 | Dose parsing success | ≥80% |
| Layer 2 | Anticoag within 24h | ≥90% patients |
| Layer 2 | Class-individual consistency | 100% |
| Layer 3 | Prevalence threshold | All ≥20 patients |
| Layer 3 | No perfect correlations | Max corr < 0.99 |
| Layer 4 | Similar pair similarity | > 0.7 |
| Layer 4 | Dissimilar pair similarity | < 0.4 |
| Layer 5 | Doses in therapeutic range | ≥90% |
| Cross-layer | All patients in all layers | 100% |

---

## Implementation Phases

### Phase 1: Setup & Infrastructure
- Directory structure
- UMLS account + RxNorm download
- SQLite loader
- Configuration files

### Phase 2: Layer 1 - Canonical Extraction
- Med.txt parser
- Module 1 integration
- Regex dose parsing
- Bronze output

### Phase 3: RxNorm Mapping
- Vocabulary extraction
- Multi-step mapping pipeline
- LLM benchmark
- Silver output

### Phase 4: Layer 2 - Therapeutic Classes
- Class assignment
- Therapeutic vs prophylactic
- Time window aggregation
- Gold output

### Phase 5: Layer 3 - Individual Medications
- Prevalence filtering
- Exception handling
- Sparse encoding
- Gold output

### Phase 6: Layer 4 - Embeddings
- Semantic (BioBERT)
- Ontological (Node2Vec)
- Co-occurrence (Word2Vec)
- Pharmacokinetic (lookup)
- Hierarchical composite
- Patient aggregations
- HDF5 output

### Phase 7: Layer 5 - Dose Intensity
- DDD lookup table
- Weight integration
- Intensity features
- Multi-resolution output

### Phase 8: Exporters & Validation
- All method-specific exporters
- Validation suite
- Documentation

### Dependency Graph

```
Phase 1 → Phase 2 → Phase 3 → Phases 4-7 (parallel) → Phase 8
```

---

## Appendix: Sample Medication Strings from RPDR

```
Heparin sodiumporcine 5000 unit/ml disp syrin 1ml syringe
Enoxaparin 100 mg/ml solution
Rivaroxaban 20mg tablet 100 ea blist pack
Apixaban 2.5mg tablet 100 ea blist pack
Coumadin 5mg tablet
Phenylephrine hcl 400 mcg syringe 10ml syringe
Norepinephrine bitartrate 4 mg/4 ml injection
Propofol 10 mg/ml Intravenous Emulsion
Midazolam hcl 2mg vial 2ml vial
Vancomycin 1gm injection
```

These strings demonstrate the parsing challenges:
- Inconsistent spacing ("sodiumporcine")
- Mixed units (unit, mg, mcg, ml)
- Packaging info mixed with drug info
- Multiple formulation details
