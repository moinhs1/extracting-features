# /home/moin/TDA_11_25/module_04_medications/tests/test_embedding_generator.py
"""Tests for medication embedding generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestSemanticEmbeddings:
    """Test BioBERT-based semantic embeddings (requires transformers library)."""

    @pytest.mark.slow
    def test_generate_semantic_embedding(self):
        """Generate semantic embedding for single medication."""
        from transformers.embedding_generator import generate_semantic_embedding

        embedding = generate_semantic_embedding("aspirin")

        # Skip if transformers not installed (returns None)
        if embedding is None:
            pytest.skip("transformers library not installed")

        assert len(embedding) == 768  # BioBERT dimension
        assert isinstance(embedding, np.ndarray)

    @pytest.mark.slow
    def test_semantic_similarity(self):
        """Similar drugs have similar embeddings."""
        from transformers.embedding_generator import generate_semantic_embedding
        from numpy.linalg import norm

        emb1 = generate_semantic_embedding("enoxaparin")

        # Skip if transformers not installed (returns None)
        if emb1 is None:
            pytest.skip("transformers library not installed")

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
        from transformers.embedding_generator import train_cooccurrence_model

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
        from transformers.embedding_generator import train_cooccurrence_model, get_cooccurrence_embedding

        sequences = [['aspirin', 'metoprolol']] * 10

        model = train_cooccurrence_model(sequences, vector_size=64, min_count=1)
        embedding = get_cooccurrence_embedding(model, 'aspirin')

        assert len(embedding) == 64


class TestPatientAggregation:
    """Test patient-level embedding aggregation."""

    def test_aggregate_mean(self):
        """Mean aggregation of medication embeddings."""
        from transformers.embedding_generator import aggregate_embeddings

        embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        result = aggregate_embeddings(embeddings, method='mean')

        expected = np.array([2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_aggregate_max(self):
        """Max aggregation."""
        from transformers.embedding_generator import aggregate_embeddings

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
        from transformers.embedding_generator import save_embeddings_hdf5, load_embeddings_hdf5

        embeddings = {
            'aspirin': np.array([1.0, 2.0, 3.0]),
            'metoprolol': np.array([4.0, 5.0, 6.0]),
        }

        output_path = tmp_path / "test_embeddings.h5"
        save_embeddings_hdf5(embeddings, output_path, embedding_type='test')

        loaded = load_embeddings_hdf5(output_path, embedding_type='test')

        assert 'aspirin' in loaded
        np.testing.assert_array_almost_equal(loaded['aspirin'], embeddings['aspirin'])
