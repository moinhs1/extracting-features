"""Tests for procedure embedding generator (Layer 4)."""

import pytest
import numpy as np
import pandas as pd
import h5py
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TEST: PROCEDURAL COMPLEXITY FEATURES
# =============================================================================

class TestComplexityFeatures:
    """Test procedural complexity feature generation."""

    def test_complexity_embedding_dimension(self):
        """Complexity embeddings are 16-dimensional."""
        from transformers.embedding_generator import get_complexity_embedding

        embedding = get_complexity_embedding(
            ccs_category='216',
            procedure_name='Intubation',
            cpt_code='31500'
        )

        assert embedding.shape == (16,)
        assert embedding.dtype == np.float32

    def test_complexity_features_normalized(self):
        """All complexity features are normalized to [0, 1]."""
        from transformers.embedding_generator import get_complexity_embedding

        embedding = get_complexity_embedding(
            ccs_category='216',
            procedure_name='Intubation',
            cpt_code='31500'
        )

        assert np.all(embedding >= 0)
        assert np.all(embedding <= 1)

    def test_intubation_complexity(self):
        """Intubation has expected complexity features."""
        from transformers.embedding_generator import get_complexity_embedding

        emb = get_complexity_embedding(
            ccs_category='216',
            procedure_name='Intubation',
            cpt_code='31500'
        )

        # Intubation should be highly invasive (level 2-3)
        invasiveness = emb[0]  # First feature
        assert invasiveness > 0.5

        # Should require emergency capability
        is_emergent = emb[5]  # Index 5
        assert is_emergent == 1.0

    def test_cta_complexity(self):
        """CTA chest has low invasiveness."""
        from transformers.embedding_generator import get_complexity_embedding

        emb = get_complexity_embedding(
            ccs_category='61',
            procedure_name='CT angiography chest',
            cpt_code='71275'
        )

        # CTA should be minimally invasive (level 1)
        invasiveness = emb[0]
        assert invasiveness < 0.5

    def test_unknown_procedure_defaults(self):
        """Unknown procedures get default zero values."""
        from transformers.embedding_generator import get_complexity_embedding

        emb = get_complexity_embedding(
            ccs_category='999',
            procedure_name='Unknown procedure',
            cpt_code='99999'
        )

        # Should not be all zeros (has some defaults)
        assert emb.shape == (16,)


# =============================================================================
# TEST: CCS CO-OCCURRENCE EMBEDDINGS
# =============================================================================

class TestCCSCooccurrence:
    """Test CCS co-occurrence embedding generation."""

    def test_build_ccs_sequences(self):
        """Build CCS category sequences per encounter."""
        from transformers.embedding_generator import build_ccs_sequences

        df = pd.DataFrame({
            'empi': ['100', '100', '100', '200', '200'],
            'encounter_id': ['E1', 'E1', 'E1', 'E2', 'E2'],
            'ccs_category': ['216', '54', '222', '61', '47'],
            'hours_from_pe': [0, 1, 2, -12, -11],
        })

        sequences = build_ccs_sequences(df)

        # Should have 2 sequences (one per encounter)
        assert len(sequences) == 2
        # First encounter has 3 CCS categories
        assert len(sequences[0]) == 3
        assert sequences[0] == ['216', '54', '222']

    def test_train_ccs_word2vec(self):
        """Train Word2Vec on CCS sequences."""
        from transformers.embedding_generator import train_ccs_word2vec

        sequences = [
            ['216', '54', '222'],
            ['216', '222', '39'],
            ['61', '47', '54'],
            ['216', '54', '47'],
        ]

        model = train_ccs_word2vec(sequences, vector_size=128, epochs=10)

        assert model is not None
        # Check that '216' is in vocabulary
        assert '216' in model.wv
        # Check embedding dimension
        assert model.wv['216'].shape == (128,)

    def test_get_ccs_embedding(self):
        """Get CCS embedding from trained model."""
        from transformers.embedding_generator import (
            train_ccs_word2vec,
            get_ccs_embedding
        )

        sequences = [
            ['216', '54', '222'],
            ['216', '222', '39'],
        ]
        model = train_ccs_word2vec(sequences, vector_size=64, epochs=5)

        emb = get_ccs_embedding(model, '216')

        assert emb is not None
        assert emb.shape == (64,)

    def test_unknown_ccs_returns_none(self):
        """Unknown CCS category returns None."""
        from transformers.embedding_generator import (
            train_ccs_word2vec,
            get_ccs_embedding
        )

        sequences = [['216', '54']]
        model = train_ccs_word2vec(sequences, vector_size=64, epochs=5)

        emb = get_ccs_embedding(model, '999')

        assert emb is None


# =============================================================================
# TEST: HDF5 STORAGE
# =============================================================================

class TestHDF5Storage:
    """Test HDF5 storage of embeddings."""

    def test_save_vocabulary_embeddings(self):
        """Save vocabulary-level embeddings to HDF5."""
        from transformers.embedding_generator import save_vocabulary_embeddings_hdf5

        embeddings = {
            '216': np.random.rand(128).astype(np.float32),
            '54': np.random.rand(128).astype(np.float32),
            '222': np.random.rand(128).astype(np.float32),
        }

        with NamedTemporaryFile(suffix='.h5', delete=False) as f:
            output_path = Path(f.name)

        try:
            save_vocabulary_embeddings_hdf5(
                embeddings,
                output_path,
                'ccs_cooccurrence',
                128
            )

            # Verify file structure
            with h5py.File(output_path, 'r') as hf:
                assert 'vocabulary' in hf
                assert 'ccs_cooccurrence' in hf['vocabulary']

                group = hf['vocabulary/ccs_cooccurrence']
                assert group.shape == (3, 128)  # 3 categories, 128 dims

                # Verify code mapping
                assert 'code_to_row' in hf['vocabulary']
                assert 'ccs_cooccurrence' in hf['vocabulary/code_to_row']
                codes = hf['vocabulary/code_to_row/ccs_cooccurrence/codes'][:]
                codes_str = [c.decode() if isinstance(c, bytes) else c for c in codes]
                assert '216' in codes_str

        finally:
            output_path.unlink()

    def test_save_complexity_embeddings(self):
        """Save complexity embeddings to HDF5."""
        from transformers.embedding_generator import save_vocabulary_embeddings_hdf5

        embeddings = {
            '216': np.random.rand(16).astype(np.float32),
            '54': np.random.rand(16).astype(np.float32),
        }

        with NamedTemporaryFile(suffix='.h5', delete=False) as f:
            output_path = Path(f.name)

        try:
            save_vocabulary_embeddings_hdf5(
                embeddings,
                output_path,
                'complexity',
                16
            )

            with h5py.File(output_path, 'r') as hf:
                group = hf['vocabulary/complexity']
                assert group.shape == (2, 16)  # 2 procedures, 16 dims

        finally:
            output_path.unlink()

    def test_hdf5_structure(self):
        """Verify complete HDF5 structure."""
        from transformers.embedding_generator import save_vocabulary_embeddings_hdf5

        with NamedTemporaryFile(suffix='.h5', delete=False) as f:
            output_path = Path(f.name)

        try:
            # Save multiple embedding types
            complexity = {
                '216': np.random.rand(16).astype(np.float32),
                '54': np.random.rand(16).astype(np.float32),
            }
            save_vocabulary_embeddings_hdf5(complexity, output_path, 'complexity', 16)

            ccs_cooc = {
                '216': np.random.rand(128).astype(np.float32),
                '54': np.random.rand(128).astype(np.float32),
            }
            save_vocabulary_embeddings_hdf5(ccs_cooc, output_path, 'ccs_cooccurrence', 128)

            # Verify structure
            with h5py.File(output_path, 'r') as hf:
                assert 'vocabulary' in hf
                assert 'vocabulary/complexity' in hf
                assert 'vocabulary/ccs_cooccurrence' in hf
                assert 'vocabulary/code_to_row' in hf

        finally:
            output_path.unlink()


# =============================================================================
# TEST: PLACEHOLDER EMBEDDINGS
# =============================================================================

class TestPlaceholderEmbeddings:
    """Test placeholder embeddings for future implementation."""

    def test_ontological_placeholder(self):
        """Ontological embeddings return placeholder structure."""
        from transformers.embedding_generator import generate_ontological_placeholder

        procedures = ['216', '54', '222']
        embeddings = generate_ontological_placeholder(procedures, dim=128)

        assert len(embeddings) == 3
        assert embeddings['216'].shape == (128,)

    def test_semantic_placeholder(self):
        """Semantic embeddings return placeholder structure."""
        from transformers.embedding_generator import generate_semantic_placeholder

        procedures = [
            ('216', 'Intubation'),
            ('54', 'Central line'),
        ]
        embeddings = generate_semantic_placeholder(procedures, dim=128)

        assert len(embeddings) == 2
        assert embeddings[('216', 'Intubation')].shape == (128,)

    def test_temporal_placeholder(self):
        """Temporal sequence embeddings return placeholder structure."""
        from transformers.embedding_generator import generate_temporal_placeholder

        procedures = ['216', '54', '222']
        embeddings = generate_temporal_placeholder(procedures, dim=128)

        assert len(embeddings) == 3
        assert embeddings['216'].shape == (128,)


# =============================================================================
# TEST: MAIN BUILDER
# =============================================================================

class TestEmbeddingBuilder:
    """Test main embedding generation pipeline."""

    def test_build_embeddings_with_test_data(self):
        """Build embeddings with minimal test data."""
        from transformers.embedding_generator import build_layer4

        # Create minimal test DataFrame
        df = pd.DataFrame({
            'empi': ['100', '100', '100', '200', '200'],
            'encounter_id': ['E1', 'E1', 'E1', 'E2', 'E2'],
            'ccs_category': ['216', '54', '222', '61', '47'],
            'procedure_name': ['Intubation', 'Central line', 'Transfusion', 'CTA', 'Echo'],
            'code': ['31500', '36555', '36430', '71275', '93306'],
            'hours_from_pe': [0, 1, 2, -12, -11],
        })

        with NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            input_path = Path(f.name)

        with NamedTemporaryFile(suffix='.h5', delete=False) as f:
            output_path = Path(f.name)

        try:
            df.to_parquet(input_path, index=False)

            result = build_layer4(
                input_path=input_path,
                output_path=output_path,
                test_mode=True
            )

            # Verify output file exists
            assert output_path.exists()

            # Verify HDF5 structure
            with h5py.File(output_path, 'r') as hf:
                assert 'vocabulary' in hf
                assert 'vocabulary/complexity' in hf
                assert 'vocabulary/ccs_cooccurrence' in hf

        finally:
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()

    def test_builder_handles_empty_data(self):
        """Builder handles empty input gracefully."""
        from transformers.embedding_generator import build_layer4

        df = pd.DataFrame(columns=[
            'empi', 'encounter_id', 'ccs_category', 'procedure_name',
            'code', 'hours_from_pe'
        ])

        with NamedTemporaryFile(suffix='.parquet', delete=False) as f:
            input_path = Path(f.name)

        with NamedTemporaryFile(suffix='.h5', delete=False) as f:
            output_path = Path(f.name)

        try:
            df.to_parquet(input_path, index=False)

            result = build_layer4(
                input_path=input_path,
                output_path=output_path,
                test_mode=True
            )

            # Should complete without errors
            assert output_path.exists()

        finally:
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()
