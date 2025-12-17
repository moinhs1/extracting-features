# Layer 4-5: Diagnosis Embeddings & World Model State Design

**Date:** 2025-12-17
**Status:** Approved
**Phase:** Module 05 Phase 3

---

## Overview

Build diagnosis embeddings (Layer 4) and world model state vectors (Layer 5) for advanced ML methods including neural networks, world models, and TDA.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| SNOMED mapping | OMOP vocabularies from Athena | Standard, comprehensive ICD→SNOMED mapping |
| Embedding source | cui2vec pre-trained | Clinically validated, no training needed |
| Storage location | Shared `data/vocabularies/` | Reusable across modules |
| Layer 5 granularity | Daily (31 days) | Diagnosis dates lack hour precision |
| Unmapped codes | Zero vector fallback | Simple, doesn't bias aggregations |

---

## Dependencies

### External Downloads (Manual)

**1. OMOP Vocabularies (~2GB)**
- Source: https://athena.ohdsi.org/
- Select: ICD9CM, ICD10CM, SNOMED, UMLS
- Extract to: `data/vocabularies/omop/`

**2. cui2vec Embeddings (~500MB)**
- Source: https://github.com/beamandrew/cui2vec
- File: `cui2vec_pretrained.csv`
- Store at: `data/vocabularies/embeddings/cui2vec_pretrained.csv`

### Internal Dependencies

- Layer 1: `canonical_diagnoses.parquet`
- Layer 2: `comorbidity_scores.parquet`
- Layer 3: `pe_diagnosis_features.parquet`

---

## File Structure

### Shared Vocabularies

```
data/vocabularies/
├── omop/
│   ├── CONCEPT.csv
│   ├── CONCEPT_RELATIONSHIP.csv
│   ├── CONCEPT_ANCESTOR.csv
│   └── CONCEPT_SYNONYM.csv
└── embeddings/
    └── cui2vec_pretrained.csv
```

### Module 05 Additions

```
module_05_diagnoses/
├── processing/
│   ├── snomed_mapper.py        # NEW: ICD→SNOMED→CUI mapping
│   ├── embedding_builder.py    # NEW: Layer 4 builder
│   └── state_builder.py        # NEW: Layer 5 builder
├── tests/
│   ├── test_snomed_mapper.py
│   ├── test_embedding_builder.py
│   └── test_state_builder.py
└── outputs/
    ├── layer4/
    │   ├── diagnosis_embeddings.h5
    │   └── embedding_metadata.json
    └── layer5/
        └── diagnosis_state.h5
```

---

## Mapping Chain

```
ICD-10-CM / ICD-9-CM
        ↓ (OMOP CONCEPT_RELATIONSHIP)
SNOMED-CT concept_id
        ↓ (OMOP CONCEPT: vocabulary_id='SNOMED' → 'UMLS')
UMLS CUI
        ↓ (cui2vec lookup)
500-dimensional embedding vector
```

---

## Layer 4: Diagnosis Embeddings

### Purpose

Dense vector representations capturing semantic relationships between diagnoses for:
- Patient similarity (TDA/Mapper)
- Neural network inputs
- Clustering/stratification

### SNOMEDMapper Class

**File:** `processing/snomed_mapper.py`

```python
class SNOMEDMapper:
    """Map ICD codes to SNOMED concept IDs and UMLS CUIs."""

    def __init__(self, omop_path: Path):
        """
        Load OMOP vocabulary tables.

        Args:
            omop_path: Path to directory containing CONCEPT.csv,
                       CONCEPT_RELATIONSHIP.csv, etc.
        """
        self.concepts = self._load_concepts(omop_path)
        self.relationships = self._load_relationships(omop_path)
        self._build_lookup_tables()

    def icd_to_snomed(self, icd_code: str, version: str) -> Optional[int]:
        """
        Map ICD code to SNOMED concept_id.

        Args:
            icd_code: ICD-9-CM or ICD-10-CM code
            version: '9' or '10'

        Returns:
            SNOMED concept_id or None if no mapping found
        """
        ...

    def snomed_to_cui(self, snomed_id: int) -> Optional[str]:
        """
        Map SNOMED concept_id to UMLS CUI.

        Args:
            snomed_id: SNOMED concept_id

        Returns:
            UMLS CUI (e.g., 'C0034065') or None if no mapping
        """
        ...

    def icd_to_cui(self, icd_code: str, version: str) -> Optional[str]:
        """Convenience method: ICD → SNOMED → CUI in one call."""
        snomed_id = self.icd_to_snomed(icd_code, version)
        if snomed_id is None:
            return None
        return self.snomed_to_cui(snomed_id)
```

### DiagnosisEmbeddingBuilder Class

**File:** `processing/embedding_builder.py`

```python
class DiagnosisEmbeddingBuilder:
    """Build diagnosis embeddings using cui2vec."""

    def __init__(self, snomed_mapper: SNOMEDMapper, cui2vec_path: Path):
        """
        Initialize with mapper and pre-trained embeddings.

        Args:
            snomed_mapper: Initialized SNOMEDMapper
            cui2vec_path: Path to cui2vec_pretrained.csv
        """
        self.mapper = snomed_mapper
        self.cui2vec = self._load_cui2vec(cui2vec_path)
        self.embedding_dim = 500

    def _load_cui2vec(self, path: Path) -> dict:
        """Load CUI → embedding lookup."""
        ...

    def get_embedding(self, icd_code: str, version: str) -> np.ndarray:
        """
        Get embedding for an ICD code.

        Returns 500-dim vector, or zeros if unmapped.
        """
        cui = self.mapper.icd_to_cui(icd_code, version)
        if cui and cui in self.cui2vec:
            return self.cui2vec[cui]
        return np.zeros(self.embedding_dim)

    def build_vocabulary_embeddings(self,
                                     canonical_diagnoses: pd.DataFrame) -> dict:
        """
        Build embeddings for all unique ICD codes in dataset.

        Returns:
            dict with keys: icd_codes, snomed_ids, cui_codes,
                           embeddings, mapping_success
        """
        unique_codes = canonical_diagnoses[['icd_code', 'icd_version']].drop_duplicates()

        results = {
            'icd_codes': [],
            'snomed_ids': [],
            'cui_codes': [],
            'embeddings': [],
            'mapping_success': []
        }

        for _, row in unique_codes.iterrows():
            icd_code = row['icd_code']
            version = row['icd_version']

            snomed_id = self.mapper.icd_to_snomed(icd_code, version)
            cui = self.mapper.snomed_to_cui(snomed_id) if snomed_id else None
            embedding = self.get_embedding(icd_code, version)

            results['icd_codes'].append(icd_code)
            results['snomed_ids'].append(snomed_id or 0)
            results['cui_codes'].append(cui or '')
            results['embeddings'].append(embedding)
            results['mapping_success'].append(cui is not None)

        results['embeddings'] = np.array(results['embeddings'])
        return results

    def build_patient_embeddings(self,
                                  canonical_diagnoses: pd.DataFrame) -> dict:
        """
        Aggregate embeddings per patient by temporal category.

        Returns:
            dict with keys: preexisting_mean, index_mean,
                           complication_mean, patient_index
        """
        empis = canonical_diagnoses['EMPI'].unique()

        preexisting = []
        index_emb = []
        complication = []

        for empi in empis:
            patient_dx = canonical_diagnoses[canonical_diagnoses['EMPI'] == empi]

            # Preexisting diagnoses
            pre_dx = patient_dx[patient_dx['is_preexisting']]
            pre_emb = self._mean_embedding(pre_dx)
            preexisting.append(pre_emb)

            # Index diagnoses
            idx_dx = patient_dx[patient_dx['is_index_concurrent']]
            idx_emb = self._mean_embedding(idx_dx)
            index_emb.append(idx_emb)

            # Complication diagnoses
            comp_dx = patient_dx[patient_dx['is_complication']]
            comp_emb = self._mean_embedding(comp_dx)
            complication.append(comp_emb)

        return {
            'preexisting_mean': np.array(preexisting),
            'index_mean': np.array(index_emb),
            'complication_mean': np.array(complication),
            'patient_index': empis
        }

    def _mean_embedding(self, diagnoses: pd.DataFrame) -> np.ndarray:
        """Compute mean embedding for a set of diagnoses."""
        if len(diagnoses) == 0:
            return np.zeros(self.embedding_dim)

        embeddings = []
        for _, row in diagnoses.iterrows():
            emb = self.get_embedding(row['icd_code'], row['icd_version'])
            embeddings.append(emb)

        return np.mean(embeddings, axis=0)
```

### Layer 4 HDF5 Schema

**File:** `outputs/layer4/diagnosis_embeddings.h5`

```
/vocabulary/
├── icd_codes           (n_codes,) str       # Unique ICD codes
├── snomed_ids          (n_codes,) int64     # SNOMED concept IDs (0 if unmapped)
├── cui_codes           (n_codes,) str       # UMLS CUIs ('' if unmapped)
├── embeddings          (n_codes, 500) float32  # cui2vec vectors
└── mapping_success     (n_codes,) bool      # True if fully mapped

/patient_embeddings/
├── preexisting_mean    (n_patients, 500) float32  # Mean of pre-PE dx
├── index_mean          (n_patients, 500) float32  # Mean of index dx
├── complication_mean   (n_patients, 500) float32  # Mean of complications
└── patient_index       (n_patients,) str          # EMPI list

/metadata/
├── embedding_dim       int (500)
├── vocab_size          int
├── cui2vec_source      str
├── omop_version        str
└── build_timestamp     str
```

---

## Layer 5: World Model Diagnosis State

### Purpose

Diagnosis features formatted for world model dynamics learning:
- Static features: time-invariant patient characteristics
- Dynamic features: day-by-day diagnosis evolution

### DiagnosisStateBuilder Class

**File:** `processing/state_builder.py`

```python
class DiagnosisStateBuilder:
    """Build world model state vectors from diagnosis layers."""

    def __init__(self,
                 layer2_scores: pd.DataFrame,
                 layer3_features: pd.DataFrame,
                 layer4_embeddings: h5py.File):
        """
        Initialize with all prior layers.

        Args:
            layer2_scores: Comorbidity scores (CCI, etc.)
            layer3_features: PE-specific features
            layer4_embeddings: HDF5 file with patient embeddings
        """
        self.scores = layer2_scores
        self.features = layer3_features
        self.embeddings = layer4_embeddings

        # PCA for dimensionality reduction
        self.pca_static = None  # Fit on preexisting embeddings
        self.pca_dynamic = None  # Fit on all embeddings

    def build_static_state(self) -> tuple[np.ndarray, list[str]]:
        """
        Build static diagnosis state (~30 dims per patient).

        Returns:
            (values, feature_names) where values is (n_patients, ~30)
        """
        empis = self.scores['EMPI'].values
        n_patients = len(empis)

        # Scalar features from Layer 2
        cci_score = self._normalize(self.scores['cci_score'].values)
        cci_count = self.scores['cci_component_count'].values

        # Boolean features from Layer 3
        bool_features = [
            'cancer_active', 'cancer_metastatic', 'heart_failure',
            'copd', 'atrial_fibrillation', 'prior_pe_ever',
            'prior_major_bleed', 'ckd_dialysis', 'is_provoked_vte',
            'pe_high_risk_code'
        ]
        bool_matrix = self.features[bool_features].values.astype(float)

        # CKD stage (0-5)
        ckd_stage = self.features['ckd_stage'].values / 5.0  # Normalize

        # PCA-reduced embeddings (500 → 17 dims)
        preexisting_emb = self.embeddings['/patient_embeddings/preexisting_mean'][:]
        if self.pca_static is None:
            self.pca_static = PCA(n_components=17)
            emb_reduced = self.pca_static.fit_transform(preexisting_emb)
        else:
            emb_reduced = self.pca_static.transform(preexisting_emb)

        # Concatenate all features
        static_state = np.column_stack([
            cci_score.reshape(-1, 1),      # 1
            cci_count.reshape(-1, 1),      # 1
            bool_matrix,                    # 10
            ckd_stage.reshape(-1, 1),      # 1
            emb_reduced                     # 17
        ])  # Total: 30

        feature_names = (
            ['cci_score', 'cci_component_count'] +
            bool_features +
            ['ckd_stage_norm'] +
            [f'emb_pc{i}' for i in range(17)]
        )

        return static_state, feature_names

    def build_dynamic_state(self,
                            canonical_diagnoses: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """
        Build daily dynamic state (31 days × ~10 dims per patient).

        Days: -1 to +30 relative to PE index.

        Returns:
            (values, feature_names) where values is (n_patients, 31, 10)
        """
        empis = self.scores['EMPI'].values
        n_patients = len(empis)
        n_days = 31
        day_range = range(-1, 30)  # Day -1 to day 29 (30 days post)

        dynamic_state = np.zeros((n_patients, n_days, 10))

        for i, empi in enumerate(empis):
            patient_dx = canonical_diagnoses[canonical_diagnoses['EMPI'] == empi]

            cumulative_count = 0
            cumulative_complications = 0

            for d, day in enumerate(day_range):
                # Diagnoses on this day
                day_dx = patient_dx[patient_dx['days_from_pe'] == day]
                new_dx_today = len(day_dx) > 0

                if new_dx_today:
                    cumulative_count += len(day_dx)
                    cumulative_complications += day_dx['is_complication'].sum()

                # Daily embedding (7 dims via PCA)
                if new_dx_today:
                    daily_emb = self._compute_daily_embedding(day_dx)
                else:
                    daily_emb = np.zeros(7)

                dynamic_state[i, d, :] = [
                    float(new_dx_today),           # 1: new diagnosis today
                    cumulative_count / 20.0,        # 2: normalized cumulative count
                    cumulative_complications / 5.0, # 3: normalized complications
                    *daily_emb                      # 4-10: embedding (7 dims)
                ]

        feature_names = [
            'new_diagnosis_today',
            'cumulative_dx_count_norm',
            'cumulative_complications_norm',
            'daily_emb_0', 'daily_emb_1', 'daily_emb_2', 'daily_emb_3',
            'daily_emb_4', 'daily_emb_5', 'daily_emb_6'
        ]

        return dynamic_state, feature_names

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        return (values - values.mean()) / (values.std() + 1e-8)

    def _compute_daily_embedding(self, diagnoses: pd.DataFrame) -> np.ndarray:
        """Compute PCA-reduced embedding for a day's diagnoses."""
        # Implementation similar to embedding builder
        ...
```

### Layer 5 HDF5 Schema

**File:** `outputs/layer5/diagnosis_state.h5`

```
/static_state/
├── values          (n_patients, 30) float32
├── feature_names   (30,) str
└── patient_index   (n_patients,) str

/dynamic_state/
├── values          (n_patients, 31, 10) float32
├── feature_names   (10,) str
├── day_index       (31,) int    # [-1, 0, 1, ..., 29]
└── patient_index   (n_patients,) str

/pca_models/
├── static_components    (17, 500) float32  # PCA for static embeddings
├── static_mean          (500,) float32
├── dynamic_components   (7, 500) float32   # PCA for dynamic embeddings
└── dynamic_mean         (500,) float32

/metadata/
├── n_patients      int
├── static_dim      int (30)
├── dynamic_dim     int (10)
├── n_days          int (31)
└── build_timestamp str
```

---

## Implementation Order

| Order | Task | Description | Tests |
|-------|------|-------------|-------|
| 1 | OMOP download | Manual Athena registration + download | - |
| 2 | cui2vec download | Manual GitHub download | - |
| 3 | Config update | Add paths to `diagnosis_config.py` | - |
| 4 | SNOMEDMapper | ICD→SNOMED→CUI mapping class | 5 |
| 5 | Embedding loader | cui2vec CSV loading | 3 |
| 6 | DiagnosisEmbeddingBuilder | Vocabulary + patient embeddings | 6 |
| 7 | Layer 4 output | HDF5 generation + metadata | 2 |
| 8 | DiagnosisStateBuilder | Static + dynamic state | 6 |
| 9 | Layer 5 output | HDF5 generation | 2 |
| 10 | Pipeline integration | Update `build_layers.py` | 2 |
| 11 | Integration tests | End-to-end on test data | 3 |

**Estimated: ~30 tests**

---

## Quality Checks

| Check | Expected | Action if Failed |
|-------|----------|------------------|
| ICD→SNOMED mapping rate | >80% | Log unmapped codes |
| SNOMED→CUI mapping rate | >90% | Check OMOP vocabulary version |
| cui2vec coverage | >70% | Report missing CUIs |
| Static state NaN | 0% | Fill with zeros |
| Embedding variance | >0 for each dim | Check for constant columns |

---

## Testing Strategy

### Unit Tests

**SNOMEDMapper:**
```python
def test_known_icd10_mapping():
    mapper = SNOMEDMapper(omop_path)
    # PE: I26.99 → SNOMED 59282003
    snomed = mapper.icd_to_snomed("I26.99", "10")
    assert snomed == 59282003

def test_icd9_mapping():
    mapper = SNOMEDMapper(omop_path)
    # MI: 410 → SNOMED
    snomed = mapper.icd_to_snomed("410.01", "9")
    assert snomed is not None
```

**DiagnosisEmbeddingBuilder:**
```python
def test_embedding_dimension():
    builder = DiagnosisEmbeddingBuilder(mapper, cui2vec_path)
    emb = builder.get_embedding("I26.99", "10")
    assert emb.shape == (500,)

def test_unmapped_returns_zeros():
    builder = DiagnosisEmbeddingBuilder(mapper, cui2vec_path)
    emb = builder.get_embedding("INVALID", "10")
    assert np.allclose(emb, np.zeros(500))
```

**DiagnosisStateBuilder:**
```python
def test_static_state_shape():
    builder = DiagnosisStateBuilder(layer2, layer3, layer4)
    state, names = builder.build_static_state()
    assert state.shape[1] == 30
    assert len(names) == 30

def test_dynamic_state_shape():
    builder = DiagnosisStateBuilder(layer2, layer3, layer4)
    state, names = builder.build_dynamic_state(canonical_dx)
    assert state.shape == (n_patients, 31, 10)
```

### Integration Tests

```python
def test_full_pipeline_layer4():
    """Test Layer 4 generation on test data."""
    # Run embedding builder on 100 patients
    # Verify HDF5 structure and shapes
    ...

def test_full_pipeline_layer5():
    """Test Layer 5 generation on test data."""
    # Run state builder
    # Verify no NaN values
    ...
```

---

## References

- [OMOP CDM](https://ohdsi.github.io/CommonDataModel/) - Common Data Model documentation
- [Athena](https://athena.ohdsi.org/) - OHDSI vocabulary download
- [cui2vec](https://github.com/beamandrew/cui2vec) - Pre-trained medical concept embeddings
- Beam et al. (2019) - "Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data"

---

**Document Version:** 1.0
**Author:** Brainstorming session 2025-12-17
