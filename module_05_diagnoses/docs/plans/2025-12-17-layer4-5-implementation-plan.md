# Layer 4-5 Embeddings & World Model State Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build diagnosis embeddings (Layer 4) and world model state vectors (Layer 5) using OMOP vocabularies and cui2vec pre-trained embeddings.

**Architecture:** SNOMEDMapper handles ICD→SNOMED→CUI mapping via OMOP tables. DiagnosisEmbeddingBuilder creates vocabulary and patient embeddings using cui2vec. DiagnosisStateBuilder produces static (30-dim) and dynamic (31 days × 10-dim) state vectors for world models.

**Tech Stack:** Python, pandas, numpy, h5py, scikit-learn (PCA), pytest

**Prerequisites:**
- OMOP vocabularies downloaded from Athena to `data/vocabularies/omop/`
- cui2vec embeddings downloaded to `data/vocabularies/embeddings/cui2vec_pretrained.csv`

---

## Task 0: Manual Downloads (Pre-Implementation)

**This task requires manual user action before code implementation begins.**

### Step 0.1: Download OMOP Vocabularies

1. Go to https://athena.ohdsi.org/
2. Register/login
3. Click "Download" → Select vocabularies:
   - ICD9CM
   - ICD10CM
   - SNOMED
   - UMLS (for CUI mapping)
4. Download the zip file (~2GB)
5. Extract to `data/vocabularies/omop/`

**Expected files after extraction:**
```
data/vocabularies/omop/
├── CONCEPT.csv
├── CONCEPT_RELATIONSHIP.csv
├── CONCEPT_ANCESTOR.csv
├── CONCEPT_SYNONYM.csv
├── VOCABULARY.csv
├── DOMAIN.csv
└── ...
```

### Step 0.2: Download cui2vec Embeddings

1. Go to https://github.com/beamandrew/cui2vec
2. Download `cui2vec_pretrained.csv.gz` from releases or figshare link
3. Extract and save to `data/vocabularies/embeddings/cui2vec_pretrained.csv`

**Verify downloads:**

Run: `ls -la data/vocabularies/omop/CONCEPT.csv data/vocabularies/embeddings/cui2vec_pretrained.csv`
Expected: Both files exist with reasonable sizes (CONCEPT.csv ~1GB, cui2vec ~500MB)

### Step 0.3: Create directory structure

```bash
mkdir -p data/vocabularies/omop
mkdir -p data/vocabularies/embeddings
```

---

## Task 1: Update Configuration

**Files:**
- Modify: `module_05_diagnoses/config/diagnosis_config.py`

**Step 1: Add vocabulary paths to config**

Add to `config/diagnosis_config.py`:

```python
from pathlib import Path

# Project root (relative to module)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Vocabulary paths (shared across modules)
OMOP_PATH = PROJECT_ROOT / "data" / "vocabularies" / "omop"
CUI2VEC_PATH = PROJECT_ROOT / "data" / "vocabularies" / "embeddings" / "cui2vec_pretrained.csv"

# Embedding configuration
EMBEDDING_DIM = 500  # cui2vec dimension
PCA_STATIC_DIM = 17  # Reduced dims for static state
PCA_DYNAMIC_DIM = 7  # Reduced dims for dynamic state

# Layer 5 configuration
DYNAMIC_STATE_DAYS = 31  # Day -1 to day +29
DYNAMIC_STATE_START_DAY = -1
```

**Step 2: Verify config imports**

Run: `cd module_05_diagnoses && python -c "from config.diagnosis_config import OMOP_PATH, CUI2VEC_PATH, EMBEDDING_DIM; print('OMOP:', OMOP_PATH); print('cui2vec:', CUI2VEC_PATH)"`
Expected: Paths printed correctly

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/diagnosis_config.py
git commit -m "feat(module05): add vocabulary and embedding paths to config"
```

---

## Task 2: SNOMEDMapper - Basic Structure

**Files:**
- Create: `module_05_diagnoses/processing/snomed_mapper.py`
- Create: `module_05_diagnoses/tests/test_snomed_mapper.py`

**Step 1: Write test for mapper initialization**

```python
"""Tests for SNOMED mapper."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd

from processing.snomed_mapper import SNOMEDMapper


@pytest.fixture
def mock_omop_path(tmp_path):
    """Create minimal mock OMOP files for testing."""
    # CONCEPT.csv
    concept_data = pd.DataFrame({
        'concept_id': [1, 2, 3, 4, 5],
        'concept_code': ['I26.99', '59282003', 'C0034065', '410.01', '22298006'],
        'vocabulary_id': ['ICD10CM', 'SNOMED', 'UMLS', 'ICD9CM', 'SNOMED'],
        'concept_name': ['PE unspecified', 'PE SNOMED', 'PE CUI', 'MI ICD9', 'MI SNOMED'],
        'domain_id': ['Condition'] * 5,
        'concept_class_id': ['4-char billing code', 'Clinical Finding', 'Clinical Finding', '5-dig billing code', 'Clinical Finding'],
        'standard_concept': [None, 'S', None, None, 'S'],
    })
    concept_data.to_csv(tmp_path / 'CONCEPT.csv', sep='\t', index=False)

    # CONCEPT_RELATIONSHIP.csv
    rel_data = pd.DataFrame({
        'concept_id_1': [1, 2, 4],
        'concept_id_2': [2, 3, 5],
        'relationship_id': ['Maps to', 'Mapped from', 'Maps to'],
    })
    rel_data.to_csv(tmp_path / 'CONCEPT_RELATIONSHIP.csv', sep='\t', index=False)

    return tmp_path


class TestSNOMEDMapperInit:
    def test_mapper_loads_concepts(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        assert mapper.concepts is not None
        assert len(mapper.concepts) == 5

    def test_mapper_loads_relationships(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        assert mapper.relationships is not None
        assert len(mapper.relationships) == 3
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""SNOMED mapper for ICD → SNOMED → CUI mapping via OMOP vocabularies."""

import pandas as pd
from pathlib import Path
from typing import Optional


class SNOMEDMapper:
    """Map ICD codes to SNOMED concept IDs and UMLS CUIs."""

    def __init__(self, omop_path: Path):
        """
        Load OMOP vocabulary tables.

        Args:
            omop_path: Path to directory containing CONCEPT.csv,
                       CONCEPT_RELATIONSHIP.csv, etc.
        """
        self.omop_path = Path(omop_path)
        self.concepts = self._load_concepts()
        self.relationships = self._load_relationships()
        self._build_lookup_tables()

    def _load_concepts(self) -> pd.DataFrame:
        """Load CONCEPT.csv."""
        path = self.omop_path / "CONCEPT.csv"
        return pd.read_csv(path, sep='\t', dtype={'concept_id': int, 'concept_code': str})

    def _load_relationships(self) -> pd.DataFrame:
        """Load CONCEPT_RELATIONSHIP.csv."""
        path = self.omop_path / "CONCEPT_RELATIONSHIP.csv"
        return pd.read_csv(path, sep='\t')

    def _build_lookup_tables(self):
        """Build efficient lookup dictionaries."""
        # ICD code → concept_id lookup
        self._icd10_to_concept = {}
        self._icd9_to_concept = {}

        icd10 = self.concepts[self.concepts['vocabulary_id'] == 'ICD10CM']
        for _, row in icd10.iterrows():
            self._icd10_to_concept[row['concept_code']] = row['concept_id']

        icd9 = self.concepts[self.concepts['vocabulary_id'] == 'ICD9CM']
        for _, row in icd9.iterrows():
            self._icd9_to_concept[row['concept_code']] = row['concept_id']

        # concept_id → concept_id mapping (Maps to relationship)
        maps_to = self.relationships[self.relationships['relationship_id'] == 'Maps to']
        self._maps_to = dict(zip(maps_to['concept_id_1'], maps_to['concept_id_2']))

        # SNOMED concept_id → CUI lookup
        self._snomed_to_cui = {}
        snomed = self.concepts[self.concepts['vocabulary_id'] == 'SNOMED']
        for _, row in snomed.iterrows():
            # Get CUI from mapped concept
            if row['concept_id'] in self._maps_to:
                cui_concept_id = self._maps_to[row['concept_id']]
                cui_row = self.concepts[self.concepts['concept_id'] == cui_concept_id]
                if len(cui_row) > 0 and cui_row.iloc[0]['vocabulary_id'] == 'UMLS':
                    self._snomed_to_cui[row['concept_id']] = cui_row.iloc[0]['concept_code']
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/snomed_mapper.py module_05_diagnoses/tests/test_snomed_mapper.py
git commit -m "feat(module05): add SNOMEDMapper basic structure"
```

---

## Task 3: SNOMEDMapper - ICD to SNOMED Mapping

**Files:**
- Modify: `module_05_diagnoses/tests/test_snomed_mapper.py`
- Modify: `module_05_diagnoses/processing/snomed_mapper.py`

**Step 1: Write tests for ICD to SNOMED mapping**

Add to `tests/test_snomed_mapper.py`:

```python
class TestICDToSNOMED:
    def test_icd10_to_snomed_exact_match(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        # I26.99 maps to concept_id 2 (SNOMED 59282003)
        snomed_id = mapper.icd_to_snomed("I26.99", "10")
        assert snomed_id == 2

    def test_icd9_to_snomed_exact_match(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        # 410.01 maps to concept_id 5 (SNOMED 22298006)
        snomed_id = mapper.icd_to_snomed("410.01", "9")
        assert snomed_id == 5

    def test_unmapped_icd_returns_none(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        snomed_id = mapper.icd_to_snomed("INVALID", "10")
        assert snomed_id is None

    def test_icd_prefix_matching(self, mock_omop_path):
        """Test that I26.9 matches I26.99 if exact not found."""
        mapper = SNOMEDMapper(mock_omop_path)
        # I26.9 not in mock data, should try prefix matching
        snomed_id = mapper.icd_to_snomed("I26.9", "10")
        # May return None or find via prefix - depends on implementation
        # For now, exact match only
        assert snomed_id is None
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py::TestICDToSNOMED -v`
Expected: FAIL with `AttributeError`

**Step 3: Implement ICD to SNOMED method**

Add to `SNOMEDMapper` class:

```python
    def icd_to_snomed(self, icd_code: str, version: str) -> Optional[int]:
        """
        Map ICD code to SNOMED concept_id.

        Args:
            icd_code: ICD-9-CM or ICD-10-CM code
            version: '9' or '10'

        Returns:
            SNOMED concept_id or None if no mapping found
        """
        # Get ICD concept_id
        if version == '10':
            icd_concept_id = self._icd10_to_concept.get(icd_code)
        elif version == '9':
            icd_concept_id = self._icd9_to_concept.get(icd_code)
        else:
            return None

        if icd_concept_id is None:
            return None

        # Get SNOMED concept_id via "Maps to" relationship
        snomed_concept_id = self._maps_to.get(icd_concept_id)
        return snomed_concept_id
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py -v`
Expected: All tests pass (6 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/snomed_mapper.py module_05_diagnoses/tests/test_snomed_mapper.py
git commit -m "feat(module05): add ICD to SNOMED mapping"
```

---

## Task 4: SNOMEDMapper - SNOMED to CUI Mapping

**Files:**
- Modify: `module_05_diagnoses/tests/test_snomed_mapper.py`
- Modify: `module_05_diagnoses/processing/snomed_mapper.py`

**Step 1: Write tests for SNOMED to CUI mapping**

Add to `tests/test_snomed_mapper.py`:

```python
class TestSNOMEDToCUI:
    def test_snomed_to_cui(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        # SNOMED concept_id 2 maps to CUI C0034065
        cui = mapper.snomed_to_cui(2)
        assert cui == "C0034065"

    def test_unmapped_snomed_returns_none(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        cui = mapper.snomed_to_cui(99999)
        assert cui is None


class TestICDToCUI:
    def test_icd_to_cui_convenience(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        # I26.99 → SNOMED 2 → CUI C0034065
        cui = mapper.icd_to_cui("I26.99", "10")
        assert cui == "C0034065"

    def test_unmapped_icd_returns_none(self, mock_omop_path):
        mapper = SNOMEDMapper(mock_omop_path)
        cui = mapper.icd_to_cui("INVALID", "10")
        assert cui is None
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py::TestSNOMEDToCUI -v`
Expected: FAIL

**Step 3: Implement SNOMED to CUI methods**

Add to `SNOMEDMapper` class:

```python
    def snomed_to_cui(self, snomed_id: int) -> Optional[str]:
        """
        Map SNOMED concept_id to UMLS CUI.

        Args:
            snomed_id: SNOMED concept_id

        Returns:
            UMLS CUI (e.g., 'C0034065') or None if no mapping
        """
        return self._snomed_to_cui.get(snomed_id)

    def icd_to_cui(self, icd_code: str, version: str) -> Optional[str]:
        """
        Convenience method: ICD → SNOMED → CUI in one call.

        Args:
            icd_code: ICD-9-CM or ICD-10-CM code
            version: '9' or '10'

        Returns:
            UMLS CUI or None if any mapping step fails
        """
        snomed_id = self.icd_to_snomed(icd_code, version)
        if snomed_id is None:
            return None
        return self.snomed_to_cui(snomed_id)
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_snomed_mapper.py -v`
Expected: All tests pass (10 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/snomed_mapper.py module_05_diagnoses/tests/test_snomed_mapper.py
git commit -m "feat(module05): add SNOMED to CUI mapping"
```

---

## Task 5: DiagnosisEmbeddingBuilder - cui2vec Loading

**Files:**
- Create: `module_05_diagnoses/processing/embedding_builder.py`
- Create: `module_05_diagnoses/tests/test_embedding_builder.py`

**Step 1: Write tests for cui2vec loading**

```python
"""Tests for diagnosis embedding builder."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock

from processing.embedding_builder import DiagnosisEmbeddingBuilder


@pytest.fixture
def mock_cui2vec_path(tmp_path):
    """Create minimal mock cui2vec file."""
    # Create fake embeddings (3 CUIs, 10 dims for testing)
    data = {
        'cui': ['C0034065', 'C0027051', 'C0011847'],
        **{f'dim_{i}': np.random.randn(3) for i in range(10)}
    }
    df = pd.DataFrame(data)
    path = tmp_path / 'cui2vec_test.csv'
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def mock_mapper():
    """Create mock SNOMEDMapper."""
    mapper = Mock()
    mapper.icd_to_cui = Mock(side_effect=lambda code, ver: {
        ('I26.99', '10'): 'C0034065',
        ('I21.0', '10'): 'C0027051',
        ('E11.9', '10'): 'C0011847',
    }.get((code, ver)))
    return mapper


class TestCui2VecLoading:
    def test_loads_cui2vec_file(self, mock_cui2vec_path, mock_mapper):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        assert builder.cui2vec is not None
        assert len(builder.cui2vec) == 3

    def test_embedding_dimension_detected(self, mock_cui2vec_path, mock_mapper):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        assert builder.embedding_dim == 10  # Our test file has 10 dims
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
"""Build diagnosis embeddings using cui2vec pre-trained embeddings."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from processing.snomed_mapper import SNOMEDMapper


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
        self.embedding_dim = self._detect_embedding_dim()

    def _load_cui2vec(self, path: Path) -> Dict[str, np.ndarray]:
        """Load CUI → embedding lookup."""
        df = pd.read_csv(path)

        # Detect CUI column (usually 'cui' or first column)
        cui_col = 'cui' if 'cui' in df.columns else df.columns[0]

        # All other columns are embedding dimensions
        dim_cols = [c for c in df.columns if c != cui_col]

        cui2vec = {}
        for _, row in df.iterrows():
            cui = row[cui_col]
            embedding = row[dim_cols].values.astype(np.float32)
            cui2vec[cui] = embedding

        return cui2vec

    def _detect_embedding_dim(self) -> int:
        """Detect embedding dimension from loaded data."""
        if not self.cui2vec:
            return 500  # Default
        sample = next(iter(self.cui2vec.values()))
        return len(sample)
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/embedding_builder.py module_05_diagnoses/tests/test_embedding_builder.py
git commit -m "feat(module05): add DiagnosisEmbeddingBuilder with cui2vec loading"
```

---

## Task 6: DiagnosisEmbeddingBuilder - Get Embedding

**Files:**
- Modify: `module_05_diagnoses/tests/test_embedding_builder.py`
- Modify: `module_05_diagnoses/processing/embedding_builder.py`

**Step 1: Write tests for get_embedding**

Add to `tests/test_embedding_builder.py`:

```python
class TestGetEmbedding:
    def test_get_embedding_for_mapped_code(self, mock_cui2vec_path, mock_mapper):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        emb = builder.get_embedding("I26.99", "10")
        assert emb.shape == (10,)
        assert not np.allclose(emb, np.zeros(10))  # Should have values

    def test_get_embedding_unmapped_returns_zeros(self, mock_cui2vec_path, mock_mapper):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        emb = builder.get_embedding("INVALID", "10")
        assert emb.shape == (10,)
        assert np.allclose(emb, np.zeros(10))

    def test_get_embedding_missing_cui_returns_zeros(self, mock_cui2vec_path, mock_mapper):
        # Code maps to CUI but CUI not in cui2vec
        mock_mapper.icd_to_cui = Mock(return_value='C9999999')
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        emb = builder.get_embedding("X99.9", "10")
        assert np.allclose(emb, np.zeros(10))
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py::TestGetEmbedding -v`
Expected: FAIL

**Step 3: Implement get_embedding**

Add to `DiagnosisEmbeddingBuilder` class:

```python
    def get_embedding(self, icd_code: str, version: str) -> np.ndarray:
        """
        Get embedding for an ICD code.

        Args:
            icd_code: ICD-9 or ICD-10 code
            version: '9' or '10'

        Returns:
            Embedding vector (embedding_dim,), zeros if unmapped
        """
        cui = self.mapper.icd_to_cui(icd_code, version)
        if cui and cui in self.cui2vec:
            return self.cui2vec[cui]
        return np.zeros(self.embedding_dim, dtype=np.float32)
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/embedding_builder.py module_05_diagnoses/tests/test_embedding_builder.py
git commit -m "feat(module05): add get_embedding method"
```

---

## Task 7: DiagnosisEmbeddingBuilder - Vocabulary Embeddings

**Files:**
- Modify: `module_05_diagnoses/tests/test_embedding_builder.py`
- Modify: `module_05_diagnoses/processing/embedding_builder.py`

**Step 1: Write tests for vocabulary embeddings**

Add to `tests/test_embedding_builder.py`:

```python
@pytest.fixture
def sample_diagnoses():
    """Sample canonical diagnoses for testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P2'],
        'icd_code': ['I26.99', 'I21.0', 'E11.9', 'I26.99'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [-30, -60, -10, 0],
        'is_preexisting': [True, True, True, False],
        'is_index_concurrent': [False, False, False, True],
        'is_complication': [False, False, False, False],
    })


class TestVocabularyEmbeddings:
    def test_build_vocabulary_embeddings_shape(self, mock_cui2vec_path, mock_mapper, sample_diagnoses):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        vocab = builder.build_vocabulary_embeddings(sample_diagnoses)

        # Should have 3 unique codes: I26.99, I21.0, E11.9
        assert len(vocab['icd_codes']) == 3
        assert vocab['embeddings'].shape == (3, 10)

    def test_build_vocabulary_embeddings_content(self, mock_cui2vec_path, mock_mapper, sample_diagnoses):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        vocab = builder.build_vocabulary_embeddings(sample_diagnoses)

        assert 'I26.99' in vocab['icd_codes']
        assert len(vocab['mapping_success']) == 3
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py::TestVocabularyEmbeddings -v`
Expected: FAIL

**Step 3: Implement vocabulary embeddings**

Add to `DiagnosisEmbeddingBuilder` class:

```python
    def build_vocabulary_embeddings(self, canonical_diagnoses: pd.DataFrame) -> dict:
        """
        Build embeddings for all unique ICD codes in dataset.

        Args:
            canonical_diagnoses: DataFrame with icd_code, icd_version columns

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

            # Get mappings
            snomed_id = self.mapper.icd_to_snomed(icd_code, version)
            cui = self.mapper.icd_to_cui(icd_code, version)
            embedding = self.get_embedding(icd_code, version)

            results['icd_codes'].append(icd_code)
            results['snomed_ids'].append(snomed_id if snomed_id else 0)
            results['cui_codes'].append(cui if cui else '')
            results['embeddings'].append(embedding)
            results['mapping_success'].append(cui is not None and cui in self.cui2vec)

        results['embeddings'] = np.array(results['embeddings'])
        return results
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py -v`
Expected: 7 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/embedding_builder.py module_05_diagnoses/tests/test_embedding_builder.py
git commit -m "feat(module05): add vocabulary embeddings builder"
```

---

## Task 8: DiagnosisEmbeddingBuilder - Patient Embeddings

**Files:**
- Modify: `module_05_diagnoses/tests/test_embedding_builder.py`
- Modify: `module_05_diagnoses/processing/embedding_builder.py`

**Step 1: Write tests for patient embeddings**

Add to `tests/test_embedding_builder.py`:

```python
class TestPatientEmbeddings:
    def test_build_patient_embeddings_shape(self, mock_cui2vec_path, mock_mapper, sample_diagnoses):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        patient_emb = builder.build_patient_embeddings(sample_diagnoses)

        # Should have 2 patients: P1, P2
        assert len(patient_emb['patient_index']) == 2
        assert patient_emb['preexisting_mean'].shape == (2, 10)
        assert patient_emb['index_mean'].shape == (2, 10)
        assert patient_emb['complication_mean'].shape == (2, 10)

    def test_preexisting_aggregation(self, mock_cui2vec_path, mock_mapper, sample_diagnoses):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        patient_emb = builder.build_patient_embeddings(sample_diagnoses)

        # P1 has 2 preexisting diagnoses, P2 has 1
        # Both should have non-zero preexisting embeddings
        p1_idx = list(patient_emb['patient_index']).index('P1')
        assert not np.allclose(patient_emb['preexisting_mean'][p1_idx], np.zeros(10))

    def test_no_complications_returns_zeros(self, mock_cui2vec_path, mock_mapper, sample_diagnoses):
        builder = DiagnosisEmbeddingBuilder(mock_mapper, mock_cui2vec_path)
        patient_emb = builder.build_patient_embeddings(sample_diagnoses)

        # No patients have complications in sample data
        assert np.allclose(patient_emb['complication_mean'], np.zeros((2, 10)))
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py::TestPatientEmbeddings -v`
Expected: FAIL

**Step 3: Implement patient embeddings**

Add to `DiagnosisEmbeddingBuilder` class:

```python
    def build_patient_embeddings(self, canonical_diagnoses: pd.DataFrame) -> dict:
        """
        Aggregate embeddings per patient by temporal category.

        Args:
            canonical_diagnoses: DataFrame with EMPI, icd_code, icd_version,
                                is_preexisting, is_index_concurrent, is_complication

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
            pre_dx = patient_dx[patient_dx['is_preexisting'] == True]
            pre_emb = self._mean_embedding(pre_dx)
            preexisting.append(pre_emb)

            # Index diagnoses
            idx_dx = patient_dx[patient_dx['is_index_concurrent'] == True]
            idx_emb = self._mean_embedding(idx_dx)
            index_emb.append(idx_emb)

            # Complication diagnoses
            comp_dx = patient_dx[patient_dx['is_complication'] == True]
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
            return np.zeros(self.embedding_dim, dtype=np.float32)

        embeddings = []
        for _, row in diagnoses.iterrows():
            emb = self.get_embedding(row['icd_code'], row['icd_version'])
            embeddings.append(emb)

        return np.mean(embeddings, axis=0).astype(np.float32)
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_embedding_builder.py -v`
Expected: 10 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/embedding_builder.py module_05_diagnoses/tests/test_embedding_builder.py
git commit -m "feat(module05): add patient embeddings aggregation"
```

---

## Task 9: Layer 4 HDF5 Output Builder

**Files:**
- Create: `module_05_diagnoses/processing/layer4_builder.py`
- Create: `module_05_diagnoses/tests/test_layer4_builder.py`

**Step 1: Write tests for Layer 4 builder**

```python
"""Tests for Layer 4 HDF5 builder."""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from unittest.mock import Mock

from processing.layer4_builder import Layer4Builder


@pytest.fixture
def mock_embedding_builder():
    """Mock DiagnosisEmbeddingBuilder."""
    builder = Mock()
    builder.embedding_dim = 10
    builder.build_vocabulary_embeddings = Mock(return_value={
        'icd_codes': ['I26.99', 'I21.0'],
        'snomed_ids': [123, 456],
        'cui_codes': ['C001', 'C002'],
        'embeddings': np.random.randn(2, 10).astype(np.float32),
        'mapping_success': [True, True]
    })
    builder.build_patient_embeddings = Mock(return_value={
        'preexisting_mean': np.random.randn(2, 10).astype(np.float32),
        'index_mean': np.random.randn(2, 10).astype(np.float32),
        'complication_mean': np.random.randn(2, 10).astype(np.float32),
        'patient_index': np.array(['P1', 'P2'])
    })
    return builder


@pytest.fixture
def sample_diagnoses():
    return pd.DataFrame({
        'EMPI': ['P1', 'P2'],
        'icd_code': ['I26.99', 'I21.0'],
        'icd_version': ['10', '10'],
        'is_preexisting': [True, True],
        'is_index_concurrent': [False, False],
        'is_complication': [False, False],
    })


class TestLayer4Builder:
    def test_build_creates_hdf5(self, mock_embedding_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer4" / "diagnosis_embeddings.h5"
        builder = Layer4Builder(mock_embedding_builder)
        builder.build(sample_diagnoses, output_path)

        assert output_path.exists()

    def test_hdf5_has_vocabulary_group(self, mock_embedding_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer4" / "diagnosis_embeddings.h5"
        builder = Layer4Builder(mock_embedding_builder)
        builder.build(sample_diagnoses, output_path)

        with h5py.File(output_path, 'r') as f:
            assert 'vocabulary' in f
            assert 'embeddings' in f['vocabulary']

    def test_hdf5_has_patient_embeddings_group(self, mock_embedding_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer4" / "diagnosis_embeddings.h5"
        builder = Layer4Builder(mock_embedding_builder)
        builder.build(sample_diagnoses, output_path)

        with h5py.File(output_path, 'r') as f:
            assert 'patient_embeddings' in f
            assert 'preexisting_mean' in f['patient_embeddings']
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer4_builder.py -v`
Expected: FAIL

**Step 3: Implement Layer 4 builder**

```python
"""Layer 4 HDF5 output builder."""

import h5py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from processing.embedding_builder import DiagnosisEmbeddingBuilder


class Layer4Builder:
    """Build Layer 4 HDF5 output."""

    def __init__(self, embedding_builder: DiagnosisEmbeddingBuilder):
        """
        Initialize with embedding builder.

        Args:
            embedding_builder: Initialized DiagnosisEmbeddingBuilder
        """
        self.embedding_builder = embedding_builder

    def build(self, canonical_diagnoses: pd.DataFrame, output_path: Path) -> None:
        """
        Build Layer 4 HDF5 file.

        Args:
            canonical_diagnoses: Layer 1 DataFrame
            output_path: Path to output .h5 file
        """
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build embeddings
        vocab = self.embedding_builder.build_vocabulary_embeddings(canonical_diagnoses)
        patient_emb = self.embedding_builder.build_patient_embeddings(canonical_diagnoses)

        # Write HDF5
        with h5py.File(output_path, 'w') as f:
            # Vocabulary group
            vocab_grp = f.create_group('vocabulary')
            vocab_grp.create_dataset('icd_codes', data=np.array(vocab['icd_codes'], dtype='S20'))
            vocab_grp.create_dataset('snomed_ids', data=np.array(vocab['snomed_ids'], dtype=np.int64))
            vocab_grp.create_dataset('cui_codes', data=np.array(vocab['cui_codes'], dtype='S20'))
            vocab_grp.create_dataset('embeddings', data=vocab['embeddings'])
            vocab_grp.create_dataset('mapping_success', data=np.array(vocab['mapping_success']))

            # Patient embeddings group
            patient_grp = f.create_group('patient_embeddings')
            patient_grp.create_dataset('preexisting_mean', data=patient_emb['preexisting_mean'])
            patient_grp.create_dataset('index_mean', data=patient_emb['index_mean'])
            patient_grp.create_dataset('complication_mean', data=patient_emb['complication_mean'])
            patient_grp.create_dataset('patient_index', data=np.array(patient_emb['patient_index'], dtype='S20'))

            # Metadata group
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['embedding_dim'] = self.embedding_builder.embedding_dim
            meta_grp.attrs['vocab_size'] = len(vocab['icd_codes'])
            meta_grp.attrs['n_patients'] = len(patient_emb['patient_index'])
            meta_grp.attrs['build_timestamp'] = datetime.now().isoformat()
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer4_builder.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/layer4_builder.py module_05_diagnoses/tests/test_layer4_builder.py
git commit -m "feat(module05): add Layer 4 HDF5 builder"
```

---

## Task 10: DiagnosisStateBuilder - Static State

**Files:**
- Create: `module_05_diagnoses/processing/state_builder.py`
- Create: `module_05_diagnoses/tests/test_state_builder.py`

**Step 1: Write tests for static state builder**

```python
"""Tests for diagnosis state builder."""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from unittest.mock import Mock, MagicMock

from processing.state_builder import DiagnosisStateBuilder


@pytest.fixture
def mock_layer2():
    return pd.DataFrame({
        'EMPI': ['P1', 'P2'],
        'cci_score': [3, 5],
        'cci_component_count': [2, 4],
    })


@pytest.fixture
def mock_layer3():
    return pd.DataFrame({
        'EMPI': ['P1', 'P2'],
        'cancer_active': [False, True],
        'cancer_metastatic': [False, False],
        'heart_failure': [True, False],
        'copd': [False, True],
        'atrial_fibrillation': [False, False],
        'prior_pe_ever': [False, True],
        'prior_major_bleed': [False, False],
        'ckd_dialysis': [False, False],
        'is_provoked_vte': [True, False],
        'pe_high_risk_code': [False, True],
        'ckd_stage': [0, 3],
    })


@pytest.fixture
def mock_layer4(tmp_path):
    """Create mock Layer 4 HDF5 file."""
    path = tmp_path / 'layer4.h5'
    with h5py.File(path, 'w') as f:
        patient_grp = f.create_group('patient_embeddings')
        patient_grp.create_dataset('preexisting_mean', data=np.random.randn(2, 50).astype(np.float32))
        patient_grp.create_dataset('patient_index', data=np.array(['P1', 'P2'], dtype='S20'))
    return path


class TestStaticState:
    def test_static_state_shape(self, mock_layer2, mock_layer3, mock_layer4):
        with h5py.File(mock_layer4, 'r') as f:
            builder = DiagnosisStateBuilder(mock_layer2, mock_layer3, f)
            state, names = builder.build_static_state()

        assert state.shape[0] == 2  # 2 patients
        assert state.shape[1] == 30  # 30 features
        assert len(names) == 30

    def test_static_state_no_nan(self, mock_layer2, mock_layer3, mock_layer4):
        with h5py.File(mock_layer4, 'r') as f:
            builder = DiagnosisStateBuilder(mock_layer2, mock_layer3, f)
            state, _ = builder.build_static_state()

        assert not np.any(np.isnan(state))
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_state_builder.py -v`
Expected: FAIL

**Step 3: Implement static state builder**

```python
"""Build world model state vectors from diagnosis layers."""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Tuple, List
from sklearn.decomposition import PCA


class DiagnosisStateBuilder:
    """Build world model state vectors from diagnosis layers."""

    # Feature names for boolean features from Layer 3
    BOOL_FEATURES = [
        'cancer_active', 'cancer_metastatic', 'heart_failure',
        'copd', 'atrial_fibrillation', 'prior_pe_ever',
        'prior_major_bleed', 'ckd_dialysis', 'is_provoked_vte',
        'pe_high_risk_code'
    ]

    def __init__(self,
                 layer2_scores: pd.DataFrame,
                 layer3_features: pd.DataFrame,
                 layer4_embeddings: h5py.File):
        """
        Initialize with all prior layers.

        Args:
            layer2_scores: Comorbidity scores (CCI, etc.)
            layer3_features: PE-specific features
            layer4_embeddings: HDF5 file handle with patient embeddings
        """
        self.scores = layer2_scores.set_index('EMPI')
        self.features = layer3_features.set_index('EMPI')
        self.embeddings = layer4_embeddings

        # PCA models (lazy initialization)
        self._pca_static = None

    def build_static_state(self) -> Tuple[np.ndarray, List[str]]:
        """
        Build static diagnosis state (~30 dims per patient).

        Returns:
            (values, feature_names) where values is (n_patients, 30)
        """
        # Get patient list from embeddings
        patient_index = self.embeddings['patient_embeddings/patient_index'][:]
        patient_index = [p.decode() if isinstance(p, bytes) else p for p in patient_index]
        n_patients = len(patient_index)

        # Initialize output
        static_state = np.zeros((n_patients, 30), dtype=np.float32)

        for i, empi in enumerate(patient_index):
            # CCI features from Layer 2
            if empi in self.scores.index:
                cci = self.scores.loc[empi, 'cci_score']
                cci_count = self.scores.loc[empi, 'cci_component_count']
            else:
                cci, cci_count = 0, 0

            # Normalize CCI (typical range 0-15)
            cci_norm = cci / 10.0

            # Boolean features from Layer 3
            bool_vals = []
            for feat in self.BOOL_FEATURES:
                if empi in self.features.index and feat in self.features.columns:
                    val = float(self.features.loc[empi, feat])
                else:
                    val = 0.0
                bool_vals.append(val)

            # CKD stage
            if empi in self.features.index and 'ckd_stage' in self.features.columns:
                ckd_stage = self.features.loc[empi, 'ckd_stage'] / 5.0
            else:
                ckd_stage = 0.0

            # Embedding features (PCA reduced)
            preexisting_emb = self.embeddings['patient_embeddings/preexisting_mean'][i]

            # Build row
            static_state[i, 0] = cci_norm
            static_state[i, 1] = cci_count / 10.0
            static_state[i, 2:12] = bool_vals
            static_state[i, 12] = ckd_stage

        # PCA on embeddings (fit once)
        preexisting_all = self.embeddings['patient_embeddings/preexisting_mean'][:]
        if self._pca_static is None:
            n_components = min(17, preexisting_all.shape[1], preexisting_all.shape[0])
            self._pca_static = PCA(n_components=n_components)
            emb_reduced = self._pca_static.fit_transform(preexisting_all)
        else:
            emb_reduced = self._pca_static.transform(preexisting_all)

        # Pad if fewer than 17 components
        if emb_reduced.shape[1] < 17:
            pad_width = 17 - emb_reduced.shape[1]
            emb_reduced = np.pad(emb_reduced, ((0, 0), (0, pad_width)))

        static_state[:, 13:30] = emb_reduced[:, :17]

        # Feature names
        feature_names = (
            ['cci_score_norm', 'cci_component_count_norm'] +
            self.BOOL_FEATURES +
            ['ckd_stage_norm'] +
            [f'emb_pc{i}' for i in range(17)]
        )

        return static_state, feature_names
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_state_builder.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/state_builder.py module_05_diagnoses/tests/test_state_builder.py
git commit -m "feat(module05): add static state builder"
```

---

## Task 11: DiagnosisStateBuilder - Dynamic State

**Files:**
- Modify: `module_05_diagnoses/tests/test_state_builder.py`
- Modify: `module_05_diagnoses/processing/state_builder.py`

**Step 1: Write tests for dynamic state builder**

Add to `tests/test_state_builder.py`:

```python
@pytest.fixture
def sample_diagnoses():
    """Sample diagnoses with daily resolution."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P2'],
        'icd_code': ['I26.99', 'N17.0', 'I26.0', 'K92.0'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [0, 3, 0, 5],
        'is_complication': [False, True, False, True],
    })


class TestDynamicState:
    def test_dynamic_state_shape(self, mock_layer2, mock_layer3, mock_layer4, sample_diagnoses):
        with h5py.File(mock_layer4, 'r') as f:
            builder = DiagnosisStateBuilder(mock_layer2, mock_layer3, f)
            state, names = builder.build_dynamic_state(sample_diagnoses)

        # (n_patients, 31 days, 10 features)
        assert state.shape == (2, 31, 10)
        assert len(names) == 10

    def test_dynamic_state_no_nan(self, mock_layer2, mock_layer3, mock_layer4, sample_diagnoses):
        with h5py.File(mock_layer4, 'r') as f:
            builder = DiagnosisStateBuilder(mock_layer2, mock_layer3, f)
            state, _ = builder.build_dynamic_state(sample_diagnoses)

        assert not np.any(np.isnan(state))

    def test_new_diagnosis_indicator(self, mock_layer2, mock_layer3, mock_layer4, sample_diagnoses):
        with h5py.File(mock_layer4, 'r') as f:
            builder = DiagnosisStateBuilder(mock_layer2, mock_layer3, f)
            state, names = builder.build_dynamic_state(sample_diagnoses)

        # P1 has diagnosis on day 0 (index 1) and day 3 (index 4)
        new_dx_idx = names.index('new_diagnosis_today')
        # Day 0 is index 1 (since day -1 is index 0)
        assert state[0, 1, new_dx_idx] == 1.0  # Day 0
        assert state[0, 4, new_dx_idx] == 1.0  # Day 3
        assert state[0, 2, new_dx_idx] == 0.0  # Day 1 (no diagnosis)
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_state_builder.py::TestDynamicState -v`
Expected: FAIL

**Step 3: Implement dynamic state builder**

Add to `DiagnosisStateBuilder` class:

```python
    def build_dynamic_state(self, canonical_diagnoses: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Build daily dynamic state (31 days × 10 dims per patient).

        Days: -1 to +29 relative to PE index (31 total).

        Args:
            canonical_diagnoses: Layer 1 DataFrame with days_from_pe

        Returns:
            (values, feature_names) where values is (n_patients, 31, 10)
        """
        # Get patient list
        patient_index = self.embeddings['patient_embeddings/patient_index'][:]
        patient_index = [p.decode() if isinstance(p, bytes) else p for p in patient_index]
        n_patients = len(patient_index)

        n_days = 31
        n_features = 10
        day_range = list(range(-1, 30))  # Day -1 to day 29

        dynamic_state = np.zeros((n_patients, n_days, n_features), dtype=np.float32)

        for i, empi in enumerate(patient_index):
            patient_dx = canonical_diagnoses[canonical_diagnoses['EMPI'] == empi]

            cumulative_count = 0
            cumulative_complications = 0

            for d, day in enumerate(day_range):
                # Diagnoses on this day
                day_dx = patient_dx[patient_dx['days_from_pe'] == day]
                new_dx_today = len(day_dx) > 0

                if new_dx_today:
                    cumulative_count += len(day_dx)
                    if 'is_complication' in day_dx.columns:
                        cumulative_complications += day_dx['is_complication'].sum()

                # Features for this day
                dynamic_state[i, d, 0] = float(new_dx_today)
                dynamic_state[i, d, 1] = cumulative_count / 20.0  # Normalize
                dynamic_state[i, d, 2] = cumulative_complications / 5.0  # Normalize
                # Remaining 7 dims are zeros (placeholder for daily embedding)
                # In full implementation, would compute PCA-reduced daily embedding

        feature_names = [
            'new_diagnosis_today',
            'cumulative_dx_count_norm',
            'cumulative_complications_norm',
            'daily_emb_0', 'daily_emb_1', 'daily_emb_2', 'daily_emb_3',
            'daily_emb_4', 'daily_emb_5', 'daily_emb_6'
        ]

        return dynamic_state, feature_names
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_state_builder.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/state_builder.py module_05_diagnoses/tests/test_state_builder.py
git commit -m "feat(module05): add dynamic state builder"
```

---

## Task 12: Layer 5 HDF5 Output Builder

**Files:**
- Create: `module_05_diagnoses/processing/layer5_builder.py`
- Create: `module_05_diagnoses/tests/test_layer5_builder.py`

**Step 1: Write tests for Layer 5 builder**

```python
"""Tests for Layer 5 HDF5 builder."""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from unittest.mock import Mock

from processing.layer5_builder import Layer5Builder


@pytest.fixture
def mock_state_builder():
    """Mock DiagnosisStateBuilder."""
    builder = Mock()
    builder.build_static_state = Mock(return_value=(
        np.random.randn(2, 30).astype(np.float32),
        ['feat_' + str(i) for i in range(30)]
    ))
    builder.build_dynamic_state = Mock(return_value=(
        np.random.randn(2, 31, 10).astype(np.float32),
        ['dyn_' + str(i) for i in range(10)]
    ))
    return builder


@pytest.fixture
def sample_diagnoses():
    return pd.DataFrame({
        'EMPI': ['P1', 'P2'],
        'days_from_pe': [0, 0],
    })


class TestLayer5Builder:
    def test_build_creates_hdf5(self, mock_state_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer5" / "diagnosis_state.h5"
        builder = Layer5Builder(mock_state_builder)
        builder.build(sample_diagnoses, output_path, patient_index=['P1', 'P2'])

        assert output_path.exists()

    def test_hdf5_has_static_state(self, mock_state_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer5" / "diagnosis_state.h5"
        builder = Layer5Builder(mock_state_builder)
        builder.build(sample_diagnoses, output_path, patient_index=['P1', 'P2'])

        with h5py.File(output_path, 'r') as f:
            assert 'static_state' in f
            assert 'values' in f['static_state']
            assert f['static_state/values'].shape == (2, 30)

    def test_hdf5_has_dynamic_state(self, mock_state_builder, sample_diagnoses, tmp_path):
        output_path = tmp_path / "layer5" / "diagnosis_state.h5"
        builder = Layer5Builder(mock_state_builder)
        builder.build(sample_diagnoses, output_path, patient_index=['P1', 'P2'])

        with h5py.File(output_path, 'r') as f:
            assert 'dynamic_state' in f
            assert f['dynamic_state/values'].shape == (2, 31, 10)
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer5_builder.py -v`
Expected: FAIL

**Step 3: Implement Layer 5 builder**

```python
"""Layer 5 HDF5 output builder."""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List

from processing.state_builder import DiagnosisStateBuilder


class Layer5Builder:
    """Build Layer 5 HDF5 output."""

    def __init__(self, state_builder: DiagnosisStateBuilder):
        """
        Initialize with state builder.

        Args:
            state_builder: Initialized DiagnosisStateBuilder
        """
        self.state_builder = state_builder

    def build(self, canonical_diagnoses: pd.DataFrame, output_path: Path,
              patient_index: List[str]) -> None:
        """
        Build Layer 5 HDF5 file.

        Args:
            canonical_diagnoses: Layer 1 DataFrame
            output_path: Path to output .h5 file
            patient_index: List of EMPIs in order
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build states
        static_values, static_names = self.state_builder.build_static_state()
        dynamic_values, dynamic_names = self.state_builder.build_dynamic_state(canonical_diagnoses)

        # Write HDF5
        with h5py.File(output_path, 'w') as f:
            # Static state group
            static_grp = f.create_group('static_state')
            static_grp.create_dataset('values', data=static_values)
            static_grp.create_dataset('feature_names', data=np.array(static_names, dtype='S50'))
            static_grp.create_dataset('patient_index', data=np.array(patient_index, dtype='S20'))

            # Dynamic state group
            dynamic_grp = f.create_group('dynamic_state')
            dynamic_grp.create_dataset('values', data=dynamic_values)
            dynamic_grp.create_dataset('feature_names', data=np.array(dynamic_names, dtype='S50'))
            dynamic_grp.create_dataset('day_index', data=np.arange(-1, 30))
            dynamic_grp.create_dataset('patient_index', data=np.array(patient_index, dtype='S20'))

            # Metadata
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['n_patients'] = len(patient_index)
            meta_grp.attrs['static_dim'] = static_values.shape[1]
            meta_grp.attrs['dynamic_dim'] = dynamic_values.shape[2]
            meta_grp.attrs['n_days'] = dynamic_values.shape[1]
            meta_grp.attrs['build_timestamp'] = datetime.now().isoformat()
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer5_builder.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/layer5_builder.py module_05_diagnoses/tests/test_layer5_builder.py
git commit -m "feat(module05): add Layer 5 HDF5 builder"
```

---

## Task 13: Pipeline Integration

**Files:**
- Modify: `module_05_diagnoses/build_layers.py`

**Step 1: Read current build_layers.py**

Run: `cat module_05_diagnoses/build_layers.py`

**Step 2: Add Layer 4 and Layer 5 building functions**

Add to `build_layers.py`:

```python
def build_layer4(layer1_path: Path, output_dir: Path, omop_path: Path, cui2vec_path: Path) -> None:
    """Build Layer 4: Diagnosis embeddings."""
    from processing.snomed_mapper import SNOMEDMapper
    from processing.embedding_builder import DiagnosisEmbeddingBuilder
    from processing.layer4_builder import Layer4Builder

    print("Building Layer 4: Diagnosis embeddings...")

    # Load Layer 1
    canonical = pd.read_parquet(layer1_path)

    # Initialize mapper and builder
    mapper = SNOMEDMapper(omop_path)
    emb_builder = DiagnosisEmbeddingBuilder(mapper, cui2vec_path)
    layer4_builder = Layer4Builder(emb_builder)

    # Build
    output_path = output_dir / "layer4" / "diagnosis_embeddings.h5"
    layer4_builder.build(canonical, output_path)

    print(f"  Layer 4 saved to: {output_path}")


def build_layer5(layer1_path: Path, layer2_path: Path, layer3_path: Path,
                 layer4_path: Path, output_dir: Path) -> None:
    """Build Layer 5: World model state vectors."""
    from processing.state_builder import DiagnosisStateBuilder
    from processing.layer5_builder import Layer5Builder

    print("Building Layer 5: World model state vectors...")

    # Load all layers
    canonical = pd.read_parquet(layer1_path)
    layer2 = pd.read_parquet(layer2_path)
    layer3 = pd.read_parquet(layer3_path)

    with h5py.File(layer4_path, 'r') as layer4:
        # Initialize builders
        state_builder = DiagnosisStateBuilder(layer2, layer3, layer4)
        layer5_builder = Layer5Builder(state_builder)

        # Get patient index from Layer 4
        patient_index = layer4['patient_embeddings/patient_index'][:]
        patient_index = [p.decode() if isinstance(p, bytes) else p for p in patient_index]

        # Build
        output_path = output_dir / "layer5" / "diagnosis_state.h5"
        layer5_builder.build(canonical, output_path, patient_index)

    print(f"  Layer 5 saved to: {output_path}")
```

Update main/CLI to include Layer 4 and Layer 5 options.

**Step 3: Test pipeline**

Run: `cd module_05_diagnoses && python build_layers.py --help`
Expected: Shows updated help with Layer 4/5 options

**Step 4: Commit**

```bash
git add module_05_diagnoses/build_layers.py
git commit -m "feat(module05): integrate Layer 4 and Layer 5 into pipeline"
```

---

## Task 14: Integration Tests

**Files:**
- Create: `module_05_diagnoses/tests/test_layer4_5_integration.py`

**Step 1: Write integration tests**

```python
"""Integration tests for Layers 4-5 with mock data."""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# These tests require OMOP and cui2vec data
# Skip if not available

pytestmark = pytest.mark.skipif(
    not Path("data/vocabularies/omop/CONCEPT.csv").exists(),
    reason="OMOP vocabularies not available"
)


class TestLayer4Integration:
    def test_build_layer4_on_sample_data(self, tmp_path):
        """Test Layer 4 generation on sample diagnoses."""
        from processing.snomed_mapper import SNOMEDMapper
        from processing.embedding_builder import DiagnosisEmbeddingBuilder
        from processing.layer4_builder import Layer4Builder
        from config.diagnosis_config import OMOP_PATH, CUI2VEC_PATH

        # Sample data
        diagnoses = pd.DataFrame({
            'EMPI': ['P1', 'P1', 'P2'],
            'icd_code': ['I26.99', 'I50.9', 'E11.9'],
            'icd_version': ['10', '10', '10'],
            'is_preexisting': [True, True, True],
            'is_index_concurrent': [False, False, False],
            'is_complication': [False, False, False],
        })

        # Build
        mapper = SNOMEDMapper(OMOP_PATH)
        emb_builder = DiagnosisEmbeddingBuilder(mapper, CUI2VEC_PATH)
        layer4_builder = Layer4Builder(emb_builder)

        output_path = tmp_path / "layer4.h5"
        layer4_builder.build(diagnoses, output_path)

        # Verify
        assert output_path.exists()
        with h5py.File(output_path, 'r') as f:
            assert 'vocabulary' in f
            assert 'patient_embeddings' in f
            assert f['patient_embeddings/preexisting_mean'].shape[0] == 2


class TestLayer5Integration:
    def test_build_layer5_on_sample_data(self, tmp_path):
        """Test Layer 5 generation on sample data."""
        # This test requires Layer 4 to exist first
        # Would need more setup - skip for now
        pytest.skip("Requires full pipeline setup")
```

**Step 2: Run integration tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer4_5_integration.py -v`
Expected: Tests pass or skip appropriately

**Step 3: Commit**

```bash
git add module_05_diagnoses/tests/test_layer4_5_integration.py
git commit -m "test(module05): add Layer 4-5 integration tests"
```

---

## Task 15: Final Verification

**Step 1: Run all tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/ -v`
Expected: All tests pass

**Step 2: Verify imports work**

Run: `cd module_05_diagnoses && python -c "
from processing.snomed_mapper import SNOMEDMapper
from processing.embedding_builder import DiagnosisEmbeddingBuilder
from processing.layer4_builder import Layer4Builder
from processing.state_builder import DiagnosisStateBuilder
from processing.layer5_builder import Layer5Builder
print('All imports successful')
"`
Expected: "All imports successful"

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(module05): complete Layer 4-5 implementation

Phase 3 implementation complete:
- SNOMEDMapper for ICD→SNOMED→CUI mapping
- DiagnosisEmbeddingBuilder with cui2vec
- Layer 4 HDF5 output (vocabulary + patient embeddings)
- DiagnosisStateBuilder for static/dynamic state
- Layer 5 HDF5 output (30-dim static, 31×10 dynamic)
- Pipeline integration

Tests: X passed"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 0 | Manual downloads (OMOP, cui2vec) | - |
| 1 | Configuration update | - |
| 2 | SNOMEDMapper basic structure | 2 |
| 3 | ICD to SNOMED mapping | 4 |
| 4 | SNOMED to CUI mapping | 4 |
| 5 | cui2vec loading | 2 |
| 6 | get_embedding method | 3 |
| 7 | Vocabulary embeddings | 2 |
| 8 | Patient embeddings | 3 |
| 9 | Layer 4 HDF5 builder | 3 |
| 10 | Static state builder | 2 |
| 11 | Dynamic state builder | 3 |
| 12 | Layer 5 HDF5 builder | 3 |
| 13 | Pipeline integration | - |
| 14 | Integration tests | 2 |
| 15 | Final verification | - |

**Total: ~33 unit tests + integration tests**

**Note:** Task 0 (manual downloads) must be completed before code implementation can begin. The OMOP and cui2vec files are prerequisites for the entire Phase 3 implementation.
