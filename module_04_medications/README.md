# Module 4: Medication Processing

Unified medication encoding system for PE trajectory analysis. Five-layer architecture serving GBTM, GRU-D, XGBoost, and World Model analyses.

## Status: Phase 2 (Layer 1) COMPLETE

---

## Implementation Progress

### Phase 1: Setup - Complete
- [x] Directory structure created
- [x] Configuration files (`medication_config.py`, `therapeutic_classes.yaml`, `dose_patterns.yaml`)
- [x] RxNorm setup script

### Phase 2: Layer 1 Canonical Extraction - COMPLETE
- [x] **Dose Parser** (`extractors/dose_parser.py`)
  - Regex-based extraction for dose value/unit (mg, mcg, units, etc.)
  - Route extraction (IV, PO, SC, IM, topical, inhaled, etc.)
  - Frequency extraction (QD, BID, TID, PRN, etc.)
  - Drug name extraction
  - 18 tests passing
- [x] **Canonical Extractor** (`extractors/canonical_extractor.py`)
  - Stream Med.txt in 1M-row chunks
  - Filter to PE cohort (8,713 patients)
  - Compute hours_from_t0 from patient_timelines.pkl
  - Apply regex dose parsing
  - Output bronze parquet with canonical schema
  - 5 tests passing
- [x] **Vocabulary Extraction** - 2 tests passing
- [x] **Full Extraction Results:**
  - Total raw records: 18.6M
  - After cohort filter: 9.6M
  - After window filter: 1.71M
  - Dose parsing success: **89.9%** (target >=80% ✅)
  - Patients with medications: **8,394** (96.3% of cohort)
  - Unique medication strings: **10,879**
  - Output: `data/bronze/canonical_records.parquet` (23 MB)
  - Vocabulary: `data/bronze/medication_vocabulary.parquet` (389 KB)

### Phases 3-8: Pending
- Phase 3: RxNorm Mapping
- Phase 4: Layer 2 Therapeutic Classes
- Phase 5: Layer 3 Individual Medications
- Phase 6: Layer 4 Embeddings
- Phase 7: Layer 5 Dose Intensity
- Phase 8: Exporters & Validation

---

## Quick Start

### Prerequisites

1. **UMLS Account** (free): https://uts.nlm.nih.gov/uts/signup-login
2. **RxNorm Download**: https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
3. **Python Dependencies**:
```bash
pip install pandas polars h5py pyyaml transformers gensim torch scikit-learn
```

### Setup RxNorm (One-Time)

```bash
# After downloading RxNorm Full Release:
cd module_04_medications/data/rxnorm
python setup_rxnorm.py --input /path/to/RxNorm_full_YYYYMMDD
```

### Run Pipeline

```bash
cd /home/moin/TDA_11_25

# Layer 1: Canonical extraction
python -m module_04_medications.extractors.canonical_extractor

# Layer 2-5: Transform layers (after RxNorm mapping)
python -m module_04_medications.transformers.class_indicator_builder
python -m module_04_medications.transformers.individual_indicator_builder
python -m module_04_medications.transformers.embedding_generator
python -m module_04_medications.transformers.dose_intensity_builder

# Export for analysis methods
python -m module_04_medications.exporters.gbtm_exporter
python -m module_04_medications.exporters.grud_exporter
```

---

## Architecture

### 5-Layer System

```
Med.txt (18.6M records, 3.7 GB)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Canonical Records                             │
│  ├── Parse medication strings                           │
│  ├── Extract dose/unit/route/frequency                  │
│  ├── Align to PE Time Zero                              │
│  └── Output: bronze/canonical_records.parquet           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  RxNorm Mapping (Bronze → Silver)                       │
│  ├── Exact match (30-40%)                               │
│  ├── Fuzzy match (30-35%)                               │
│  ├── Ingredient extraction (15-20%)                     │
│  ├── LLM-assisted (3-8%)                                │
│  └── Target: ≥85% mapping rate                          │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Layers 2-5 (Parallel)                                  │
│                                                         │
│  Layer 2: Therapeutic Classes (53 indicators)          │
│  ├── PE-critical anticoagulants (9 classes)            │
│  ├── Vasopressors expanded (6 classes)                 │
│  ├── Anti-inflammatories (4 classes)                   │
│  └── + 34 more clinical categories                     │
│                                                         │
│  Layer 3: Individual Medications (200-400 sparse)       │
│  ├── Prevalence ≥20 patients                           │
│  └── All anticoags/vasopressors/thrombolytics          │
│                                                         │
│  Layer 4: Embeddings (HDF5)                             │
│  ├── Semantic (BioBERT, 768d)                          │
│  ├── Ontological (Node2Vec, 128d)                      │
│  ├── Co-occurrence (Word2Vec, 128d)                    │
│  ├── Pharmacokinetic (10d)                             │
│  └── Hierarchical Composite (128d)                     │
│                                                         │
│  Layer 5: Dose Intensity                                │
│  ├── Raw values                                        │
│  ├── DDD-normalized                                    │
│  └── Weight-adjusted                                   │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Method-Specific Exports                                │
│  ├── GBTM: gbtm_medication_long.csv                    │
│  ├── GRU-D: grud_medications.h5 (tensors)              │
│  ├── XGBoost: xgboost_medication_features.parquet      │
│  ├── World Models: world_model_actions.h5              │
│  └── TDA: tda_medication_pointcloud.parquet            │
└─────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
module_04_medications/
├── README.md                 ← You are here
├── config/
│   ├── __init__.py
│   ├── therapeutic_classes.yaml   # 53 drug class definitions
│   ├── dose_patterns.yaml         # Regex + DDD values + LLM prompts
│   └── medication_config.py       # Central configuration
│
├── data/
│   ├── rxnorm/               # RxNorm SQLite database
│   │   ├── rxnorm.db
│   │   └── setup_rxnorm.py
│   ├── bronze/               # Layer 1: Raw canonical records
│   │   └── canonical_records.parquet
│   ├── silver/               # RxNorm mapped records
│   │   ├── mapped_medications.parquet
│   │   └── mapping_failures.parquet
│   ├── gold/                 # Analysis-ready layers
│   │   ├── therapeutic_classes/    # Layer 2
│   │   ├── individual_indicators/  # Layer 3
│   │   └── dose_intensity/         # Layer 5
│   └── embeddings/           # Layer 4
│       └── medication_embeddings.h5
│
├── extractors/
│   ├── canonical_extractor.py     # Med.txt → Bronze
│   ├── rxnorm_mapper.py           # Bronze → Silver
│   ├── dose_parser.py             # Hybrid dose extraction
│   └── llm_dose_parser.py         # LLM fallback
│
├── transformers/
│   ├── class_indicator_builder.py      # Layer 2
│   ├── individual_indicator_builder.py # Layer 3
│   ├── embedding_generator.py          # Layer 4
│   └── dose_intensity_builder.py       # Layer 5
│
├── exporters/
│   ├── gbtm_exporter.py          # R-ready CSVs
│   ├── grud_exporter.py          # HDF5 tensors
│   ├── xgboost_exporter.py       # Wide tabular
│   ├── world_model_exporter.py   # Action embeddings
│   └── tda_exporter.py           # Point clouds
│
├── validation/
│   ├── llm_benchmark.py          # Compare LLM models
│   └── layer_validators.py       # Cross-layer consistency
│
└── tests/
```

---

## Therapeutic Classes (53 Total)

### PE-Critical Anticoagulants (9)

| Class ID | Name | Key Ingredients |
|----------|------|-----------------|
| `ac_ufh_ther` | UFH Therapeutic | heparin (>10k units/day) |
| `ac_ufh_proph` | UFH Prophylactic | heparin (≤10k units/day) |
| `ac_lmwh_ther` | LMWH Therapeutic | enoxaparin 1mg/kg |
| `ac_lmwh_proph` | LMWH Prophylactic | enoxaparin 40mg |
| `ac_fondaparinux` | Fondaparinux | fondaparinux |
| `ac_xa_inhibitor` | Factor Xa Inhibitors | rivaroxaban, apixaban |
| `ac_dti` | Direct Thrombin Inhibitors | dabigatran, argatroban |
| `ac_vka` | Vitamin K Antagonists | warfarin |
| `ac_thrombolytic` | Thrombolytics | alteplase, tenecteplase |

### Cardiovascular - Vasopressors (6)

| Class ID | Name |
|----------|------|
| `cv_norepinephrine` | Norepinephrine |
| `cv_epinephrine` | Epinephrine (IV) |
| `cv_vasopressin` | Vasopressin |
| `cv_phenylephrine` | Phenylephrine (IV) |
| `cv_dopamine` | Dopamine |
| `cv_vasopressor_any` | Any Vasopressor |

### Anti-Inflammatory (4)

| Class ID | Name |
|----------|------|
| `ai_steroid_systemic` | Systemic Corticosteroids |
| `ai_steroid_iv` | IV Corticosteroids |
| `ai_steroid_inhaled` | Inhaled Corticosteroids |
| `ai_nsaid` | NSAIDs |

### Other Categories

- **Inotropes** (4): dobutamine, milrinone, digoxin
- **CV Other** (8): beta blockers, ACE/ARBs, CCBs, diuretics, antiarrhythmics
- **Pain/Sedation** (4): opioids, acetaminophen, benzodiazepines, propofol
- **Anti-infectives** (4): antibiotics, antivirals, antifungals
- **Hematologic** (4): antiplatelets, reversal agents
- **GI/Metabolic** (5): PPIs, antiemetics, insulin
- **Respiratory** (2): bronchodilators, neuromuscular blockers
- **ICU-Specific** (3): stress ulcer prophylaxis, DVT prophylaxis, IV electrolytes

See `config/therapeutic_classes.yaml` for complete definitions.

---

## Embeddings (Layer 4)

| Type | Dimensions | Source | Captures |
|------|------------|--------|----------|
| **Semantic** | 768 | BioBERT/PubMedBERT | Name similarity |
| **Ontological** | 128 | Node2Vec on RxNorm graph | Mechanism, interactions |
| **Co-occurrence** | 128 | Word2Vec on RPDR sequences | MGB prescribing patterns |
| **Pharmacokinetic** | 10 | DrugBank lookup | Half-life, onset, clearance |
| **Hierarchical** | 128 | Class centroid + residuals | Multi-level dynamics |

### HDF5 Structure

```
medication_embeddings.h5
├── /vocabulary/
│   ├── semantic           (n_meds, 768)
│   ├── ontological        (n_meds, 128)
│   ├── cooccurrence       (n_meds, 128)
│   ├── pharmacokinetic    (n_meds, 10)
│   ├── hierarchical       (n_meds, 128)
│   └── rxcui_index        # Lookup mapping
│
├── /patient_embeddings/
│   ├── hourly/            (n_patients, n_hours, dim)
│   ├── daily/
│   └── window/
│
└── /aggregations/
    ├── mean/
    ├── max/
    └── dose_weighted/
```

---

## LLM Benchmark

For ambiguous dose parsing, benchmark local models:

| Model | Size | Notes |
|-------|------|-------|
| Llama 3 8B | 8B | Strong general |
| Mistral 7B | 7B | Fast |
| Phi-3 Mini | 3.8B | Efficient |
| Gemma 2 9B | 9B | Good instruction following |
| Qwen 2.5 7B | 7B | Strong extraction |
| Qwen 2.5 14B | 14B | Higher accuracy |

```bash
# Run benchmark
python -m module_04_medications.validation.llm_benchmark --samples 200
```

---

## Validation Targets

| Layer | Metric | Target |
|-------|--------|--------|
| Layer 1 | RxNorm mapping rate | ≥85% |
| Layer 1 | Dose parsing success | ≥80% |
| Layer 2 | Anticoag within 24h of PE | ≥90% |
| Layer 2 | Class-individual consistency | 100% |
| Layer 3 | No perfect correlations | Max <0.99 |
| Layer 4 | Similar pair similarity | >0.7 |
| Layer 4 | Dissimilar pair similarity | <0.4 |

---

## Implementation Phases

| Phase | Tasks | Dependencies |
|-------|-------|--------------|
| **1. Setup** | Directory structure, UMLS account, RxNorm download | None |
| **2. Layer 1** | Med.txt parser, Module 1 integration, regex parsing | Phase 1 |
| **3. RxNorm** | Mapping pipeline, LLM benchmark | Phase 2 |
| **4. Layer 2** | Therapeutic class assignment | Phase 3 |
| **5. Layer 3** | Individual med indicators | Phase 3 |
| **6. Layer 4** | All 5 embedding types | Phase 3 |
| **7. Layer 5** | Dose intensity features | Phase 3 |
| **8. Export** | All method-specific exports, validation | Phases 4-7 |

Phases 4-7 can run in parallel after Phase 3.

---

## Configuration

### Key Settings (`config/medication_config.py`)

```python
# Temporal windows
TEMPORAL_WINDOWS = {
    'baseline': (-72, 0),
    'acute': (0, 24),
    'subacute': (24, 72),
    'recovery': (72, 168),
}

# RxNorm mapping
FUZZY_MATCH_THRESHOLD = 0.85
TARGET_MAPPING_RATE = 0.85

# Layer 3
PREVALENCE_THRESHOLD = 20  # Min patients for individual med

# Embeddings
SEMANTIC_DIM = 768
ONTOLOGICAL_DIM = 128
COOCCURRENCE_DIM = 128
```

---

## Input Data

| File | Size | Records | Description |
|------|------|---------|-------------|
| `Data/Med.txt` | 3.7 GB | 18.6M | Medication administrations |
| `Data/Dem.txt` | 6.5 MB | - | Patient demographics (for weights) |

### Sample Medication Strings

```
Heparin sodiumporcine 5000 unit/ml disp syrin 1ml syringe
Enoxaparin 100 mg/ml solution
Rivaroxaban 20mg tablet 100 ea blist pack
Coumadin 5mg tablet
Phenylephrine hcl 400 mcg syringe 10ml syringe
```

---

## Dependencies

```
pandas>=2.0
polars>=0.19
h5py>=3.9
pyyaml>=6.0
transformers>=4.30  # BioBERT
gensim>=4.3         # Word2Vec, Node2Vec
torch>=2.0          # LLM inference
scikit-learn>=1.3
scipy>=1.11
```

---

## Documentation

- **Design Document**: `../docs/plans/2025-12-08-module-04-medications-design.md`
- **Class Definitions**: `config/therapeutic_classes.yaml`
- **Dose Patterns**: `config/dose_patterns.yaml`

---

## Related Modules

- **Module 1**: Provides `patient_timelines.pkl` for PE Time Zero
- **Module 3**: Similar 5-layer architecture for vitals
- **Module 6**: Will align medications with labs/vitals on hourly grid

---

**Version:** 1.1.0 (Phase 2 In Progress)
**Last Updated:** 2025-12-10
