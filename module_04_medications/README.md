# Module 4: Medication Processing

Unified medication encoding system for PE trajectory analysis. Five-layer architecture serving GBTM, GRU-D, XGBoost, and World Model analyses.

## Status: Phase 6 (Layer 4 Embeddings) COMPLETE ✅

---

## Implementation Progress

### Phase 1: Setup - COMPLETE ✅
- [x] Directory structure created
- [x] Configuration files (`medication_config.py`, `therapeutic_classes.yaml`, `dose_patterns.yaml`)
- [x] RxNorm setup script and database (`rxnorm.db`)

### Phase 2: Layer 1 Canonical Extraction - COMPLETE ✅
- [x] **Dose Parser** (`extractors/dose_parser.py`) - 18 tests
- [x] **Canonical Extractor** (`extractors/canonical_extractor.py`) - 5 tests
- [x] **Vocabulary Extraction** - 2 tests
- [x] **Results:**
  - Records: 1.71M (from 18.6M raw)
  - Dose parsing: **89.9%** (target >=80% ✅)
  - Patients: **8,394** (96.3% of cohort)
  - Unique medications: **10,879**
  - Output: `data/bronze/canonical_records.parquet` (23 MB)

### Phase 3: RxNorm Mapping - COMPLETE ✅
- [x] **RxNorm Mapper** (`extractors/rxnorm_mapper.py`) - 10 tests
  - Exact match against RXNCONSO
  - Fuzzy match using rapidfuzz (85% threshold)
  - Ingredient extraction from product names
  - Ingredient lookup via RXNREL relationships
- [x] **Full Mapping Results:**
  - Vocabulary mapping: **82.9%** (10,879 unique strings)
  - Record-level mapping: **92.4%** (1.58M/1.71M records) ✅
  - Unique ingredients mapped: **1,582**
  - Output: `data/silver/mapped_medications.parquet` (32 MB)

### Phase 4: Layer 2 Therapeutic Classes - COMPLETE ✅
- [x] **Class Indicator Builder** (`transformers/class_indicator_builder.py`) - 14 tests
  - Map ingredients to 53 therapeutic classes
  - Dose-based therapeutic/prophylactic classification (UFH, LMWH)
  - Time window assignment (baseline/acute/subacute/recovery)
  - Aggregate indicators per patient-window
- [x] **Results:**
  - Patient-window combinations: **25,038**
  - Patients: **7,786**
  - Classes: **53** (with counts and first-occurrence times)
  - Anticoagulant in acute window: **54.7%**
  - Output: `data/gold/therapeutic_classes/class_indicators.parquet`

### Phase 5: Layer 3 Individual Medications - COMPLETE ✅
- [x] **Individual Indicator Builder** (`transformers/individual_indicator_builder.py`) - 4 tests
  - Prevalence filtering (≥20 patients + exception medications)
  - Vectorized pivot operations (~100x faster than iterrows)
  - Sparse matrix storage in HDF5
  - Parallel processing option (--parallel --jobs N)
- [x] **Results:**
  - Individual medication indicators: **581**
  - Patient-window combinations: **26,499**
  - Total features: **1,747** (binary + count + dose per med)
  - Sparsity: **98.4%** (target >90% ✅)
  - Processing time: **2.9s** (vectorized)
  - Output: `data/gold/individual_indicators/individual_indicators.parquet`
  - Output: `data/gold/individual_indicators/individual_indicators_sparse.h5`

### Phase 6: Layer 4 Embeddings - COMPLETE ✅
- [x] **Embedding Generator** (`transformers/embedding_generator.py`) - 5 tests
  - Semantic embeddings (BioBERT/PubMedBERT) - optional
  - Co-occurrence embeddings (Word2Vec on patient sequences)
  - Pharmacokinetic feature embeddings (hand-crafted)
  - Patient-level aggregation (mean, max, sum)
  - HDF5 storage with compression
- [x] **Results:**
  - Co-occurrence embeddings: **769 medications × 128 dims**
  - Pharmacokinetic embeddings: **1,582 medications × 10 dims**
  - Training sequences: **8,267 patients**
  - Output: `data/embeddings/medication_embeddings.h5` (585 KB)

### Phases 7-8: Pending
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

**Version:** 1.5.0 (Phase 6 Complete)
**Last Updated:** 2025-12-11

---

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_dose_parser.py` | 18 | ✅ Pass |
| `test_canonical_extractor.py` | 5 | ✅ Pass |
| `test_vocabulary.py` | 2 | ✅ Pass |
| `test_rxnorm_mapper.py` | 10 | ✅ Pass |
| `test_class_indicator_builder.py` | 14 | ✅ Pass |
| `test_individual_indicator_builder.py` | 4 | ✅ Pass |
| `test_embedding_generator.py` | 5 | ✅ Pass |
| **Total** | **58** | **✅ All Passing** |

Run all tests:
```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/ -v
```
