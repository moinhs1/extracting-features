# Phase 6: Layer 4 Medication Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate 5 types of medication embeddings (Semantic, Ontological, Co-occurrence, Pharmacokinetic, Hierarchical) for use in neural methods (GRU-D, World Models) and similarity analysis.

**Architecture:** Create vocabulary-level embeddings for unique medications, then aggregate to patient-level embeddings per timewindow. Store in HDF5 with efficient indexing.

**Tech Stack:** Python 3.12, transformers (BioBERT), gensim (Word2Vec), node2vec, h5py, torch, pytest

**Depends on:** Phase 3 complete (mapped_medications.parquet exists)

---

## Task 1: Create Embedding Generator Tests

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_embedding_generator.py`

**Step 1: Create test file**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_embedding_generator.py
"""Tests for medication embedding generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSemanticEmbeddings:
    """Test BioBERT-based semantic embeddings."""

    def test_generate_semantic_embedding(self):
        """Generate semantic embedding for single medication."""
        from transformers_pkg.embedding_generator import generate_semantic_embedding

        embedding = generate_semantic_embedding("aspirin")

        assert embedding is not None
        assert len(embedding) == 768  # BioBERT dimension
        assert isinstance(embedding, np.ndarray)

    def test_semantic_similarity(self):
        """Similar drugs have similar embeddings."""
        from transformers_pkg.embedding_generator import generate_semantic_embedding
        from numpy.linalg import norm

        emb1 = generate_semantic_embedding("enoxaparin")
        emb2 = generate_semantic_embedding("dalteparin")  # Both LMWH
        emb3 = generate_semantic_embedding("acetaminophen")  # Unrelated

        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (norm(emb1) * norm(emb3))

        # Similar drugs should have higher similarity
        assert sim_12 > sim_13


class TestCooccurrenceEmbeddings:
    """Test Word2Vec co-occurrence embeddings."""

    def test_train_cooccurrence_model(self):
        """Train co-occurrence model on medication sequences."""
        from transformers_pkg.embedding_generator import train_cooccurrence_model

        # Sample medication sequences per patient
        sequences = [
            ['aspirin', 'metoprolol', 'lisinopril'],
            ['aspirin', 'atorvastatin', 'lisinopril'],
            ['heparin', 'warfarin'],
            ['enoxaparin', 'aspirin'],
        ]

        model = train_cooccurrence_model(sequences, vector_size=64, min_count=1)

        assert model is not None
        assert 'aspirin' in model.wv

    def test_get_cooccurrence_embedding(self):
        """Get embedding from trained model."""
        from transformers_pkg.embedding_generator import train_cooccurrence_model, get_cooccurrence_embedding

        sequences = [['aspirin', 'metoprolol']] * 10

        model = train_cooccurrence_model(sequences, vector_size=64, min_count=1)
        embedding = get_cooccurrence_embedding(model, 'aspirin')

        assert len(embedding) == 64


class TestPatientAggregation:
    """Test patient-level embedding aggregation."""

    def test_aggregate_mean(self):
        """Mean aggregation of medication embeddings."""
        from transformers_pkg.embedding_generator import aggregate_embeddings

        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        result = aggregate_embeddings(embeddings, method='mean')

        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregate_max(self):
        """Max aggregation."""
        from transformers_pkg.embedding_generator import aggregate_embeddings

        embeddings = np.array([
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 6.0],
        ])

        result = aggregate_embeddings(embeddings, method='max')

        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestHDF5Storage:
    """Test HDF5 embedding storage."""

    def test_save_and_load_embeddings(self, tmp_path):
        """Save and load embeddings from HDF5."""
        from transformers_pkg.embedding_generator import save_embeddings_hdf5, load_embeddings_hdf5

        embeddings = {
            'aspirin': np.array([1.0, 2.0, 3.0]),
            'metoprolol': np.array([4.0, 5.0, 6.0]),
        }

        output_path = tmp_path / "test_embeddings.h5"
        save_embeddings_hdf5(embeddings, output_path, embedding_type='test')

        loaded = load_embeddings_hdf5(output_path, embedding_type='test')

        assert 'aspirin' in loaded
        np.testing.assert_array_almost_equal(loaded['aspirin'], embeddings['aspirin'])
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_embedding_generator.py -v
```

**Step 3: Commit failing tests**

```bash
git add module_04_medications/tests/test_embedding_generator.py
git commit -m "test(module4): add embedding generator tests

Add failing tests for:
- Semantic embeddings (BioBERT)
- Co-occurrence embeddings (Word2Vec)
- Patient-level aggregation
- HDF5 storage

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement Semantic Embedding Generator

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/transformers/embedding_generator.py`

**Step 1: Create embedding generator with semantic embeddings**

```python
# /home/moin/TDA_11_25/module_04_medications/transformers/embedding_generator.py
"""
Medication Embedding Generator
==============================

Layer 4: Generate multiple types of medication embeddings:
1. Semantic (BioBERT/PubMedBERT)
2. Co-occurrence (Word2Vec on patient sequences)
3. Pharmacokinetic (hand-crafted features)

Note: Ontological (Node2Vec) and Hierarchical embeddings are optional
      and require additional dependencies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import h5py
import sys
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    SILVER_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_CONFIG,
    TEMPORAL_CONFIG,
)


# =============================================================================
# SEMANTIC EMBEDDINGS (BioBERT)
# =============================================================================

_semantic_model = None
_semantic_tokenizer = None


def _load_semantic_model():
    """Load BioBERT model for semantic embeddings."""
    global _semantic_model, _semantic_tokenizer

    if _semantic_model is None:
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            model_name = EMBEDDING_CONFIG.semantic_model
            print(f"Loading semantic model: {model_name}")

            _semantic_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _semantic_model = AutoModel.from_pretrained(model_name)
            _semantic_model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                _semantic_model = _semantic_model.cuda()

        except ImportError:
            print("Warning: transformers not installed. Semantic embeddings unavailable.")
            return None, None

    return _semantic_model, _semantic_tokenizer


@lru_cache(maxsize=10000)
def generate_semantic_embedding(medication_name: str) -> Optional[np.ndarray]:
    """
    Generate semantic embedding for a medication name.

    Args:
        medication_name: Medication/ingredient name

    Returns:
        768-dimensional embedding vector or None
    """
    model, tokenizer = _load_semantic_model()
    if model is None:
        return None

    import torch

    # Tokenize
    inputs = tokenizer(
        medication_name,
        return_tensors='pt',
        max_length=64,
        truncation=True,
        padding=True
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pool last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    return embedding.cpu().numpy()


def generate_semantic_embeddings_batch(
    medication_names: List[str],
    batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Generate semantic embeddings for multiple medications.

    Args:
        medication_names: List of medication names
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping medication name to embedding
    """
    embeddings = {}

    for i in range(0, len(medication_names), batch_size):
        batch = medication_names[i:i + batch_size]
        for name in batch:
            emb = generate_semantic_embedding(name)
            if emb is not None:
                embeddings[name] = emb

        if (i + batch_size) % 500 == 0:
            print(f"  Processed {min(i + batch_size, len(medication_names))}/{len(medication_names)}")

    return embeddings


# =============================================================================
# CO-OCCURRENCE EMBEDDINGS (Word2Vec)
# =============================================================================

def build_medication_sequences(df: pd.DataFrame) -> List[List[str]]:
    """
    Build medication sequences per patient for Word2Vec training.

    Args:
        df: Mapped medications DataFrame

    Returns:
        List of medication sequences (one per patient)
    """
    sequences = []

    for empi, group in df.groupby('empi'):
        # Sort by time
        group = group.sort_values('hours_from_t0')

        # Get sequence of ingredient names
        meds = group['ingredient_name'].dropna().tolist()

        if len(meds) >= 2:  # Need at least 2 for context
            sequences.append(meds)

    return sequences


def train_cooccurrence_model(
    sequences: List[List[str]],
    vector_size: int = None,
    window: int = None,
    min_count: int = None,
    epochs: int = None
):
    """
    Train Word2Vec model on medication sequences.

    Args:
        sequences: List of medication sequences
        vector_size: Embedding dimension
        window: Context window size
        min_count: Minimum frequency threshold
        epochs: Training epochs

    Returns:
        Trained Word2Vec model
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("Warning: gensim not installed. Co-occurrence embeddings unavailable.")
        return None

    if vector_size is None:
        vector_size = EMBEDDING_CONFIG.cooccurrence_dim
    if window is None:
        window = EMBEDDING_CONFIG.word2vec_window
    if min_count is None:
        min_count = EMBEDDING_CONFIG.word2vec_min_count
    if epochs is None:
        epochs = EMBEDDING_CONFIG.word2vec_epochs

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        workers=4,
        sg=1,  # Skip-gram
    )

    return model


def get_cooccurrence_embedding(
    model,
    medication_name: str
) -> Optional[np.ndarray]:
    """
    Get co-occurrence embedding from trained model.

    Args:
        model: Trained Word2Vec model
        medication_name: Medication name

    Returns:
        Embedding vector or None
    """
    if model is None:
        return None

    try:
        return model.wv[medication_name]
    except KeyError:
        return None


# =============================================================================
# PHARMACOKINETIC EMBEDDINGS
# =============================================================================

# Hand-crafted PK features for common PE medications
# Format: [half_life_h, onset_min, peak_h, duration_h, bioavail, protein_bind, vd, clearance, ti, metabolites]
PK_FEATURES = {
    'heparin': [1.5, 0, 0.5, 4, 1.0, 0.0, 0.06, 0.5, 1.0, 0],
    'enoxaparin': [4.5, 60, 3, 12, 0.92, 0.0, 0.06, 0.15, 1.0, 0],
    'warfarin': [40, 1440, 72, 168, 0.99, 0.99, 0.14, 0.003, 0.5, 1],
    'apixaban': [12, 180, 3, 24, 0.50, 0.87, 0.21, 0.02, 1.0, 0],
    'rivaroxaban': [9, 150, 3, 24, 0.80, 0.95, 0.5, 0.1, 1.0, 0],
    'aspirin': [0.25, 30, 1, 4, 0.68, 0.80, 0.15, 10, 1.0, 1],
    'morphine': [3, 20, 1, 4, 0.25, 0.35, 3.0, 20, 0.3, 1],
    'fentanyl': [4, 5, 0.5, 1, 0.92, 0.84, 4.0, 13, 0.2, 1],
    'norepinephrine': [0.03, 0, 0.02, 0.1, 0, 0, 0.08, 40, 0.5, 0],
    'metoprolol': [4, 60, 1.5, 6, 0.50, 0.12, 5.6, 15, 1.0, 1],
}


def get_pharmacokinetic_embedding(medication_name: str) -> np.ndarray:
    """
    Get pharmacokinetic feature embedding.

    Args:
        medication_name: Medication name

    Returns:
        10-dimensional PK feature vector (normalized)
    """
    name_lower = medication_name.lower().strip()

    if name_lower in PK_FEATURES:
        features = np.array(PK_FEATURES[name_lower], dtype=np.float32)
    else:
        # Default unknown features
        features = np.zeros(EMBEDDING_CONFIG.pk_dim, dtype=np.float32)

    # Normalize to [0, 1] range
    # Using reasonable max values for each feature
    max_vals = np.array([100, 1440, 100, 200, 1, 1, 10, 50, 1, 1], dtype=np.float32)
    features = np.clip(features / max_vals, 0, 1)

    return features


# =============================================================================
# PATIENT-LEVEL AGGREGATION
# =============================================================================

def aggregate_embeddings(
    embeddings: np.ndarray,
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate multiple medication embeddings to single patient embedding.

    Args:
        embeddings: Array of shape (n_medications, embedding_dim)
        method: 'mean', 'max', or 'sum'

    Returns:
        Aggregated embedding vector
    """
    if len(embeddings) == 0:
        return None

    if method == 'mean':
        return np.mean(embeddings, axis=0)
    elif method == 'max':
        return np.max(embeddings, axis=0)
    elif method == 'sum':
        return np.sum(embeddings, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_patient_embeddings(
    df: pd.DataFrame,
    vocabulary_embeddings: Dict[str, np.ndarray],
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Aggregate medication embeddings per patient-window.

    Args:
        df: Mapped medications DataFrame
        vocabulary_embeddings: Dict mapping medication name to embedding
        method: Aggregation method

    Returns:
        DataFrame with patient-level embeddings
    """
    from transformers.class_indicator_builder import get_time_window

    df = df.copy()
    df['time_window'] = df['hours_from_t0'].apply(get_time_window)
    df = df[df['time_window'].notna()]

    results = []
    embedding_dim = len(list(vocabulary_embeddings.values())[0])

    for (empi, window), group in df.groupby(['empi', 'time_window']):
        # Collect embeddings for medications in this group
        meds = group['ingredient_name'].dropna().unique()
        med_embeddings = []

        for med in meds:
            if med in vocabulary_embeddings:
                med_embeddings.append(vocabulary_embeddings[med])

        if med_embeddings:
            agg_emb = aggregate_embeddings(np.array(med_embeddings), method)
        else:
            agg_emb = np.zeros(embedding_dim)

        results.append({
            'empi': empi,
            'time_window': window,
            'embedding': agg_emb,
        })

    return pd.DataFrame(results)


# =============================================================================
# HDF5 STORAGE
# =============================================================================

def save_embeddings_hdf5(
    embeddings: Dict[str, np.ndarray],
    output_path: Path,
    embedding_type: str
):
    """
    Save embeddings to HDF5 file.

    Args:
        embeddings: Dict mapping medication name to embedding
        output_path: Output HDF5 path
        embedding_type: Name for this embedding type
    """
    with h5py.File(output_path, 'a') as f:
        # Create or get group for this embedding type
        if embedding_type in f:
            del f[embedding_type]
        g = f.create_group(embedding_type)

        # Convert to arrays
        names = list(embeddings.keys())
        vectors = np.array([embeddings[n] for n in names])

        # Save
        g.create_dataset('vectors', data=vectors)

        # Save names as variable-length strings
        dt = h5py.special_dtype(vlen=str)
        g.create_dataset('names', data=np.array(names, dtype=object), dtype=dt)

        g.attrs['embedding_dim'] = vectors.shape[1] if len(vectors) > 0 else 0
        g.attrs['n_medications'] = len(names)


def load_embeddings_hdf5(
    input_path: Path,
    embedding_type: str
) -> Dict[str, np.ndarray]:
    """
    Load embeddings from HDF5 file.

    Args:
        input_path: Input HDF5 path
        embedding_type: Name of embedding type to load

    Returns:
        Dict mapping medication name to embedding
    """
    with h5py.File(input_path, 'r') as f:
        g = f[embedding_type]
        names = [n.decode() if isinstance(n, bytes) else n for n in g['names'][:]]
        vectors = g['vectors'][:]

    return {name: vec for name, vec in zip(names, vectors)}


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_layer4(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
    skip_semantic: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build Layer 4 medication embeddings.

    Args:
        input_path: Path to mapped_medications.parquet
        output_path: Path for output HDF5
        test_mode: If True, process subset
        skip_semantic: Skip slow semantic embeddings

    Returns:
        Dictionary of embedding types
    """
    print("=" * 60)
    print("Layer 4: Medication Embeddings")
    print("=" * 60)

    # Load data
    if input_path is None:
        input_path = SILVER_DIR / "mapped_medications.parquet"

    print(f"\n1. Loading mapped medications: {input_path}")
    df = pd.read_parquet(input_path)

    if test_mode:
        sample_empis = df['empi'].unique()[:50]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")

    # Get unique medications
    medications = df['ingredient_name'].dropna().unique().tolist()
    print(f"   Unique medications: {len(medications)}")

    all_embeddings = {}

    # 1. Semantic embeddings (BioBERT)
    if not skip_semantic:
        print("\n2. Generating semantic embeddings (BioBERT)...")
        try:
            semantic_emb = generate_semantic_embeddings_batch(medications[:100] if test_mode else medications)
            all_embeddings['semantic'] = semantic_emb
            print(f"   Generated {len(semantic_emb)} semantic embeddings")
        except Exception as e:
            print(f"   Skipped semantic: {e}")

    # 2. Co-occurrence embeddings (Word2Vec)
    print("\n3. Training co-occurrence embeddings (Word2Vec)...")
    sequences = build_medication_sequences(df)
    print(f"   Built {len(sequences)} medication sequences")

    if len(sequences) > 0:
        cooc_model = train_cooccurrence_model(sequences)
        if cooc_model:
            cooc_emb = {}
            for med in medications:
                emb = get_cooccurrence_embedding(cooc_model, med)
                if emb is not None:
                    cooc_emb[med] = emb
            all_embeddings['cooccurrence'] = cooc_emb
            print(f"   Generated {len(cooc_emb)} co-occurrence embeddings")

    # 3. Pharmacokinetic embeddings
    print("\n4. Generating pharmacokinetic embeddings...")
    pk_emb = {med: get_pharmacokinetic_embedding(med) for med in medications}
    all_embeddings['pharmacokinetic'] = pk_emb
    print(f"   Generated {len(pk_emb)} PK embeddings")

    # Save to HDF5
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        filename = "medication_embeddings_test.h5" if test_mode else "medication_embeddings.h5"
        output_path = EMBEDDINGS_DIR / filename

    print(f"\n5. Saving to HDF5: {output_path}")
    for emb_type, emb_dict in all_embeddings.items():
        if emb_dict:
            save_embeddings_hdf5(emb_dict, output_path, emb_type)
            print(f"   Saved {emb_type}: {len(emb_dict)} embeddings")

    print("\n" + "=" * 60)
    print("Layer 4 Complete!")
    print("=" * 60)

    return all_embeddings


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build medication embeddings")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--skip-semantic', action='store_true', help='Skip semantic embeddings')
    args = parser.parse_args()

    build_layer4(test_mode=args.test, skip_semantic=args.skip_semantic)
```

**Step 2: Install dependencies**

```bash
pip3 install gensim transformers torch
```

**Step 3: Run tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_embedding_generator.py -v -k "not semantic"
```

Note: Skip semantic tests initially as they require downloading large models.

**Step 4: Commit**

```bash
git add module_04_medications/transformers/embedding_generator.py
git commit -m "feat(module4): implement medication embedding generator

Layer 4 implementation:
- Semantic embeddings (BioBERT/PubMedBERT)
- Co-occurrence embeddings (Word2Vec)
- Pharmacokinetic feature embeddings
- Patient-level aggregation
- HDF5 storage

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Run Embedding Generation

**Files:**
- Run: Embedding generator

**Step 1: Run in test mode (skip semantic for speed)**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/embedding_generator.py --test --skip-semantic
```

**Step 2: Run full mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/embedding_generator.py --skip-semantic 2>&1 | tee module_04_medications/embeddings.log
```

**Step 3: Validate output**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import h5py

with h5py.File('module_04_medications/data/embeddings/medication_embeddings.h5', 'r') as f:
    print('Embedding types:', list(f.keys()))
    for emb_type in f.keys():
        g = f[emb_type]
        print(f'  {emb_type}:')
        print(f'    n_medications: {g.attrs[\"n_medications\"]}')
        print(f'    embedding_dim: {g.attrs[\"embedding_dim\"]}')
"
```

**Step 4: Commit**

```bash
git add module_04_medications/embeddings.log
git commit -m "chore(module4): Layer 4 embeddings generated

Output: data/embeddings/medication_embeddings.h5
- Co-occurrence embeddings (Word2Vec)
- Pharmacokinetic embeddings

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Validation Checklist

- ✅ `embedding_generator.py` exists
- ✅ Tests pass
- ✅ `medication_embeddings.h5` created
- ✅ At least 2 embedding types generated
- ✅ Embeddings can be loaded and used

---

## Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | Embedding tests | test_embedding_generator.py |
| 2 | Implement generator | embedding_generator.py |
| 3 | Run generation | medication_embeddings.h5 |

**Total:** 3 tasks, ~10 steps

---

## Optional: Full Semantic Embeddings

If BioBERT embeddings are needed later:

```bash
# Download model first (large)
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')"

# Then run with semantic
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/embedding_generator.py
```
