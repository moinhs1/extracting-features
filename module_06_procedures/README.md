# Module 6: Unified Procedure Encoding for PE Trajectory Analysis

**Status:** Implementation Complete
**Version:** 1.0
**Date:** 2025-12-15

---

## Overview

Module 6 implements a comprehensive 5-layer procedure encoding system for pulmonary embolism (PE) trajectory analysis. It processes ~22 million procedure records from RPDR's `Prc.txt` file, mapping them to standard vocabularies (CCS, SNOMED) and generating analytical features for multiple downstream methods.

**Key Capabilities:**
- Canonical extraction with 7 temporal categories
- Standard vocabulary mapping (CCS + SNOMED)
- PE-specific clinical features
- Multiple embedding types for machine learning
- World model state/action representations
- Method-specific exports (GBTM, GRU-D, XGBoost)

---

## Directory Structure

```
module_06_procedures/
├── config/
│   ├── procedure_config.py          # Central configuration
│   ├── pe_procedure_codes.yaml      # PE-specific CPT codes
│   ├── surgical_risk.yaml           # VTE risk classifications
│   └── discretion_weights.yaml      # World model action weights
├── data/
│   ├── vocabularies/                # Reference vocabularies
│   │   ├── ccs_crosswalk.csv        # CPT/ICD → CCS mappings
│   │   ├── snomed_procedures.db     # SNOMED concepts
│   │   └── setup_vocabularies.py
│   ├── bronze/                      # Layer 1: Canonical records
│   │   └── canonical_procedures.parquet
│   ├── silver/                      # Mapped records
│   │   ├── mapped_procedures.parquet
│   │   └── mapping_failures.parquet
│   ├── gold/                        # Analysis-ready features
│   │   ├── ccs_indicators/          # Layer 2
│   │   ├── pe_procedure_features/   # Layer 3
│   │   └── world_model_states/      # Layer 5
│   └── embeddings/                  # Layer 4
│       └── procedure_embeddings.h5
├── extractors/
│   ├── canonical_extractor.py       # Layer 1: Prc.txt → Bronze
│   └── vocabulary_mapper.py         # Bronze → Silver
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
│   └── layer_validators.py
└── tests/
```

---

## 5-Layer Architecture

### Layer 1: Canonical Procedure Records (Bronze)

**Purpose:** Single source of truth with temporal classification

**Input:** `Data/Prc.txt` (22M records, 6.4 GB)

**Output:** `data/bronze/canonical_procedures.parquet`

**Schema:**
- Core fields: empi, procedure_datetime, code, code_type, procedure_name
- Temporal: hours_from_pe, days_from_pe
- Context: hospital, clinic, provider, inpatient/outpatient
- 7 temporal flags (see below)

**Temporal Categories:**

| Category | Window | Purpose |
|----------|--------|---------|
| `is_lifetime_history` | Before -720h | Background surgical risk |
| `is_provoking_window` | -720h to 0h | VTE provocation factors |
| `is_diagnostic_workup` | -24h to +24h | Index PE workup |
| `is_initial_treatment` | 0h to +72h | Early treatment decisions |
| `is_escalation` | >+72h | Complications requiring escalation |
| `is_post_discharge` | After +720h | Long-term outcomes |
| `is_remote_antecedent` | Before -720h | Remote history |

**Run:**
```bash
python module_06_procedures/extractors/canonical_extractor.py --test
```

---

### Layer 2: Standard Procedure Groupings (Silver + Gold)

**Purpose:** CCS categories, surgical risk, invasiveness classification

**Input:** Bronze + CCS crosswalk

**Output:** `data/gold/ccs_indicators/ccs_indicators.parquet`

**Features:**
- **CCS Categories** (~230 categories): Binary indicators per category
- **Surgical Risk**: VTE risk levels (Very High, High, Moderate, Low, Minimal)
- **BETOS Categories**: P1 (Major), P2 (Minor), P3 (Ambulatory), I (Imaging), T (Tests)
- **Invasiveness**: 0-3 ordinal scale

**Example CCS Categories:**
- CCS 216: Respiratory intubation and mechanical ventilation
- CCS 47: Diagnostic cardiac catheterization
- CCS 54: Vascular catheterization
- CCS 222: Blood transfusion

**Run:**
```bash
python module_06_procedures/transformers/ccs_indicator_builder.py --test
```

---

### Layer 3: PE-Specific Procedure Features (Gold)

**Purpose:** Curated features directly relevant to PE trajectories

**Output:** `data/gold/pe_procedure_features/pe_features.parquet`

**Feature Groups:**

#### Lifetime Surgical History (Static)
- Prior VTE procedures (IVC filter, thrombolysis, CDT)
- Prior major surgeries (orthopedic, cardiac, cancer)
- Chronic procedure burden (pacemaker, dialysis)

#### Provoking Procedures (1-30 Days Pre-PE)
- Recent surgery (type, VTE risk, days since)
- Derived: provoked_pe, provocation_strength

#### Diagnostic Workup (±24h of PE)
- **Imaging:** CTA chest, V/Q scan, LE duplex, echo, cardiac cath
- **Metrics:** Workup intensity, time to CTA, diagnostic sequence

#### Initial Treatment (0-72h)
- **Reperfusion:** Systemic thrombolysis, CDT, surgical embolectomy
- **IVC Filter:** Placement, type, timing
- **Vascular Access:** Central line, arterial line, PA catheter
- **Respiratory Support:** Intubation, HFNC, NIPPV
- **Circulatory Support:** ECMO (VA/VV), support level

#### Escalation/Complications (>72h)
- **Respiratory:** Delayed intubation, reintubation, tracheostomy
- **Bleeding:** Transfusion, massive transfusion, GI endoscopy
- **Other:** Cardiac arrest, ROSC, RRT initiation

**All features include datetime preservation:**
- `{feature}_datetime`: Actual timestamp
- `{feature}_hours_from_pe`: Relative timing

**Run:**
```bash
python module_06_procedures/transformers/pe_feature_builder.py --test
```

---

### Layer 4: Procedure Embeddings (Gold)

**Purpose:** Dense vector representations capturing procedure relationships

**Output:** `data/embeddings/procedure_embeddings.h5`

**Embedding Types:**

| Type | Dimensions | Weight | Method |
|------|------------|--------|--------|
| Ontological | 128 | 1.0 | Node2Vec on SNOMED+CCS hierarchy |
| Semantic | 128 | 0.8 | BioBERT → PCA |
| Temporal Sequence | 128 | 0.9 | Transformer on encounter sequences |
| CCS Co-occurrence | 128 | 0.6 | Word2Vec on CCS sequences |
| PE Co-occurrence | 64 | 0.5 | Word2Vec on PE procedures |
| Procedural Complexity | 16 | 0.7 | Structured features |

**Validation Targets:**
- Similar procedures (intubation ↔ mechanical vent): Similarity > 0.7
- Dissimilar procedures (dental ↔ cardiac): Similarity < 0.3

**Run:**
```bash
python module_06_procedures/transformers/embedding_generator.py --test
```

---

### Layer 5: World Model States & Actions (Gold)

**Purpose:** Procedure components for causal world models

**Output:** `data/gold/world_model_states/`

**Dual Representation:**

1. **Actions** - Clinician decisions (counterfactually modifiable)
   - High discretion (weight 1.0): Thrombolysis, CDT, IVC filter
   - Moderate discretion (0.6-0.8): Intubation, ECMO
   - Low discretion (0.2-0.4): Transfusion, dialysis
   - No discretion (0.0): CPR (state update only)

2. **States** - Patient condition
   - Current support: On ventilator, vasopressors, ECMO
   - Time on support: Hours on vent, ECMO
   - Complications: Cardiac arrest, major bleeding, RRT
   - Cumulative burden: RBC units, invasive procedures

**Static Procedure State:**
- Lifetime history embedding
- Provoking factors
- Chronic conditions

**Run:**
```bash
python module_06_procedures/transformers/world_model_builder.py --test
```

---

## Code Mapping Strategy

### Code Type Distribution

| Code Type | Records | % | Mapping Approach |
|-----------|---------|---|------------------|
| CPT | 15.6M | 71% | Direct CCS + SNOMED via OMOP |
| EPIC | 5.5M | 25% | Fuzzy matching + LLM |
| HCPCS | 446K | 2% | CCS crosswalk |
| ICD-10-PCS | 244K | 1% | Direct CCS |
| ICD-9 Vol 3 | 143K | <1% | CCS crosswalk |

### Mapping Pipeline

1. **Direct Mapping** (~75%): CPT/HCPCS/ICD → CCS via AHRQ crosswalk
2. **Fuzzy Matching** (~15%): Procedure name → CCS description (>0.85 similarity)
3. **LLM Classification** (~8%): Low-confidence → LLM selects best CCS
4. **Manual Review** (~2%): Saved to `mapping_failures.parquet`

**Target:** ≥85% successful mapping to CCS

---

## Method-Specific Exports

### GBTM / lcmm (R)

**Output:** `exports/gbtm_procedures.csv`

Wide format with temporal features, support trajectories, outcomes.

**Run:**
```bash
python module_06_procedures/exporters/gbtm_exporter.py
```

---

### GRU-D (HDF5)

**Output:** `exports/grud_procedures.h5`

Handles irregular time series with masking and time deltas.

**Structure:**
```
/procedure_values    # (n_patients, 168, n_features)
/procedure_mask      # (n_patients, 168, n_features)
/procedure_delta     # (n_patients, 168, n_features)
```

**Run:**
```bash
python module_06_procedures/exporters/grud_exporter.py
```

---

### XGBoost (Parquet)

**Output:** `exports/xgboost_procedure_features.parquet`

~500 static features: CCS indicators, PE features, embedding PCAs.

**Run:**
```bash
python module_06_procedures/exporters/xgboost_exporter.py
```

---

## Validation

### Running Validation

```bash
python module_06_procedures/validation/layer_validators.py
```

Or in tests:
```bash
PYTHONPATH=module_06_procedures:$PYTHONPATH pytest module_06_procedures/tests/test_validators.py -v
```

### Validation Targets

#### Layer 1 (Bronze)
- Records loaded: Target 22M (allow 80%)
- Patients in cohort: Target ~8,700 (allow 90%)
- Code type distribution: CPT >50%
- PE time linkage: >95% of patients

#### Layer 2 (Mapping)
- Overall CCS mapping rate: ≥85%
- CPT CCS mapping rate: ≥95%

#### Layer 3 (PE Features)
- CTA performed: 80-95% of patients
- Echo performed: 50-70% of patients
- Intubation rate: 5-15% of patients
- IVC filter rate: 5-15% of patients
- ECMO rate: <2% of patients

#### Cross-Layer
- All patients present in all layers
- Timestamps aligned (same PE Time Zero)

---

## Usage Examples

### Example 1: Extract Canonical Procedures

```python
from extractors.canonical_extractor import extract_canonical_records

# Full extraction
df = extract_canonical_records(test_mode=False)

# Test mode (10K rows)
df_test = extract_canonical_records(test_mode=True, test_n_rows=10000)

print(f"Records: {len(df):,}")
print(f"Patients: {df['empi'].nunique():,}")
print(f"Temporal distribution:\n{df.groupby('is_diagnostic_workup').size()}")
```

---

### Example 2: Map to CCS Categories

```python
from extractors.vocabulary_mapper import run_vocabulary_mapping

# Map procedures
mapped_df = run_vocabulary_mapping(test_mode=False)

# Check mapping rates
mapping_rate = mapped_df['ccs_category'].notna().mean()
print(f"Overall mapping rate: {mapping_rate:.1%}")

cpt_df = mapped_df[mapped_df['code_type'] == 'CPT']
cpt_rate = cpt_df['ccs_category'].notna().mean()
print(f"CPT mapping rate: {cpt_rate:.1%}")
```

---

### Example 3: Build PE Features

```python
from transformers.pe_feature_builder import build_pe_features

# Build all features
pe_features = build_pe_features(test_mode=False)

# Check key features
print(f"CTA performed: {pe_features['cta_performed'].mean():.1%}")
print(f"Intubation rate: {pe_features['intubation_performed'].mean():.1%}")
print(f"IVC filter rate: {pe_features['ivc_filter_placed'].mean():.1%}")
print(f"ECMO rate: {pe_features['ecmo_initiated'].mean():.1%}")

# Check provoking procedures
provoked = pe_features['provoked_pe'].mean()
print(f"Provoked PE rate: {provoked:.1%}")
```

---

### Example 4: Export for GRU-D

```python
from exporters.grud_exporter import export_grud

# Export hourly sequences
export_grud(
    output_path='exports/grud_procedures.h5',
    time_resolution='hourly',
    max_hours=168,
    test_mode=False
)
```

---

### Example 5: Run Validation

```python
from validation.layer_validators import run_full_validation

# Run all validations
results = run_full_validation()

# Check results
for result in results:
    print(result.summary())
    if result.failed > 0:
        print(result.report())
```

---

## Configuration

Central configuration in `config/procedure_config.py`:

```python
from config.procedure_config import (
    PRC_FILE,           # Input Prc.txt path
    TEMPORAL_CONFIG,    # 7 temporal windows
    MAPPING_CONFIG,     # Fuzzy match threshold
    VALIDATION_CONFIG,  # Validation targets
    ensure_directories  # Create output dirs
)

# Ensure output directories exist
ensure_directories()
```

---

## Testing

### Run All Tests

```bash
PYTHONPATH=module_06_procedures:$PYTHONPATH pytest module_06_procedures/tests/ -v
```

### Run Specific Test Suite

```bash
# Configuration tests
pytest module_06_procedures/tests/test_config.py -v

# Extractor tests
pytest module_06_procedures/tests/test_canonical_extractor.py -v

# Mapper tests
pytest module_06_procedures/tests/test_vocabulary_mapper.py -v

# Transformer tests
pytest module_06_procedures/tests/test_ccs_indicator_builder.py -v
pytest module_06_procedures/tests/test_pe_feature_builder.py -v

# Validator tests
pytest module_06_procedures/tests/test_validators.py -v
```

---

## Dependencies

**Core:**
- pandas >= 1.5.0
- pyarrow >= 10.0.0
- numpy >= 1.23.0
- PyYAML >= 6.0

**Machine Learning:**
- scikit-learn >= 1.2.0
- gensim >= 4.3.0 (Word2Vec)
- transformers >= 4.30.0 (BioBERT)
- torch >= 2.0.0

**Storage:**
- h5py >= 3.8.0
- sqlite3 (stdlib)

**Testing:**
- pytest >= 7.4.0

---

## Data Sources

### Input Data
- **RPDR Prc.txt:** 22M procedure records, 6.4 GB
- **Module 1 patient_timelines.pkl:** PE Time Zero reference

### Vocabularies
- **CCS (Clinical Classification Software):** AHRQ HCUP
- **SNOMED-CT:** UMLS/OMOP vocabularies

### Configuration Files
- **pe_procedure_codes.yaml:** Curated PE-relevant CPT codes
- **surgical_risk.yaml:** VTE risk classifications
- **discretion_weights.yaml:** World model action weights

---

## Future Modules (Separate)

The following are implemented as **separate modules**, NOT part of Module 6:

- **Echo Module:** Structured echo measurements (LVEF, RV/LV ratio, TAPSE)
- **Cardiac Cath Module:** PA pressures, wedge, cardiac output
- **PFT Module:** FEV1, FVC, DLCO, lung volumes

Module 6 provides binary indicators that these tests occurred; specialized modules contain structured numeric data.

---

## Troubleshooting

### Issue: Low CCS Mapping Rate

**Symptom:** `validate_silver()` reports <85% mapping

**Solutions:**
1. Check CCS crosswalk exists: `ls data/vocabularies/ccs_crosswalk.csv`
2. Regenerate crosswalk: `python data/vocabularies/setup_vocabularies.py`
3. Review failures: `pd.read_parquet('data/silver/mapping_failures.parquet')`

---

### Issue: Missing Temporal Flags

**Symptom:** `validate_layer1()` reports missing temporal flags

**Solutions:**
1. Verify PE Time Zero linkage: Check `patient_timelines.pkl` exists
2. Re-extract canonical records with `--test` to debug
3. Check temporal config: `from config.procedure_config import TEMPORAL_CONFIG`

---

### Issue: PE Feature Rates Out of Range

**Symptom:** `validate_layer3()` reports unexpected rates

**Solutions:**
1. Check cohort definition: Verify PE diagnosis criteria
2. Review procedure codes: Ensure CPT codes in `pe_procedure_codes.yaml` are correct
3. Inspect raw data: Sample patients and trace their procedures

---

## Performance Notes

- **Layer 1 extraction:** ~15 minutes (full 22M records)
- **Vocabulary mapping:** ~10 minutes (with caching)
- **PE feature building:** ~5 minutes
- **Embedding generation:** ~30 minutes (with pretrained models)
- **Full pipeline:** ~60 minutes end-to-end

**Memory requirements:**
- Layer 1: ~8 GB RAM
- Embeddings: ~16 GB RAM (reduce batch size if needed)
- Full pipeline: 32 GB recommended

---

## Contributing

When adding new features:

1. **Write tests first** (TDD)
2. Follow existing layer patterns
3. Update validation targets in `procedure_config.py`
4. Document in this README
5. Run full test suite before committing

---

## Authors

**Module 6 Design & Implementation:**
Generated with Claude Code (Anthropic)

**Date:** 2025-12-15

---

## License

Part of TDA_11_25 PE Trajectory Analysis Project

---

**Questions?** See design document: `docs/plans/2025-12-11-module-06-procedures-design.md`
