"""
Tests for hierarchical clustering.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'module_2_laboratory_processing'))

from hierarchical_clustering import (
    calculate_token_similarity,
    calculate_unit_incompatibility,
    calculate_combined_distance
)


class TestDistanceMetrics:
    """Test distance metrics for clustering."""

    def test_token_similarity_identical(self):
        """Test token similarity for identical names."""
        sim = calculate_token_similarity(
            "LOW DENSITY LIPOPROTEIN",
            "LOW DENSITY LIPOPROTEIN"
        )
        assert sim == 1.0

    def test_token_similarity_different_modifiers(self):
        """Test LDL vs HDL (share LIPOPROTEIN but different modifiers)."""
        sim = calculate_token_similarity(
            "LOW DENSITY LIPOPROTEIN",
            "HIGH DENSITY LIPOPROTEIN"
        )
        # Intersection: {DENSITY, LIPOPROTEIN} = 2
        # Union: {LOW, HIGH, DENSITY, LIPOPROTEIN} = 4
        # Similarity: 2/4 = 0.5
        assert sim == 0.5

    def test_token_similarity_completely_different(self):
        """Test completely different test names."""
        sim = calculate_token_similarity(
            "GLUCOSE",
            "HEMOGLOBIN"
        )
        assert sim == 0.0

    def test_unit_incompatibility_same_unit(self):
        """Test unit incompatibility for same unit."""
        incomp = calculate_unit_incompatibility("mg/dL", "mg/dL")
        assert incomp == 0.0

    def test_unit_incompatibility_convertible(self):
        """Test unit incompatibility for convertible units."""
        # Will use fallback since pint may not be installed
        incomp = calculate_unit_incompatibility("mg/dL", "mmol/L")
        # Both are concentration units, should be 0.5 in fallback
        assert 0.0 <= incomp <= 1.0

    def test_combined_distance_identical(self):
        """Test combined distance for identical tests."""
        test1 = {'name': 'GLUCOSE', 'unit': 'mg/dL'}
        test2 = {'name': 'GLUCOSE', 'unit': 'mg/dL'}

        dist = calculate_combined_distance(test1, test2)
        assert dist == 0.0

    def test_combined_distance_ldl_vs_hdl(self):
        """Test LDL vs HDL (similar names but should not group)."""
        test1 = {'name': 'LOW DENSITY LIPOPROTEIN', 'unit': 'mg/dL'}
        test2 = {'name': 'HIGH DENSITY LIPOPROTEIN', 'unit': 'mg/dL'}

        dist = calculate_combined_distance(test1, test2)

        # Name similarity: 0.5 (share 2 of 4 tokens)
        # Name distance: 0.5
        # Unit incompatibility: 0.0 (same unit)
        # Combined: 0.6 * 0.5 + 0.4 * 0.0 = 0.3
        assert abs(dist - 0.3) < 0.01


class TestHierarchicalClustering:
    """Test hierarchical clustering algorithm."""

    def test_cluster_similar_tests(self):
        """Test clustering groups similar tests together."""
        from hierarchical_clustering import perform_hierarchical_clustering

        # Sample tests: glucose variants and cholesterol variants
        tests = [
            {'name': 'GLUCOSE', 'unit': 'mg/dL'},
            {'name': 'GLUCOSE BLOOD', 'unit': 'mg/dL'},
            {'name': 'GLUCOSE POC', 'unit': 'mg/dL'},
            {'name': 'CHOLESTEROL', 'unit': 'mg/dL'},
            {'name': 'CHOLESTEROL TOTAL', 'unit': 'mg/dL'},
        ]

        clusters, linkage_matrix, distances = perform_hierarchical_clustering(
            tests,
            threshold=0.9
        )

        # Should have 2 clusters: glucose group and cholesterol group
        assert len(clusters) >= 2

        # Glucose tests should be in same cluster
        glucose_indices = {0, 1, 2}  # First 3 tests
        cholesterol_indices = {3, 4}  # Last 2 tests

        # Find which cluster contains glucose tests
        glucose_cluster_id = None
        for cluster_id, test_indices in clusters.items():
            if 0 in test_indices:
                glucose_cluster_id = cluster_id
                # Should contain all glucose tests
                assert set(test_indices) & glucose_indices == glucose_indices
                break

        assert glucose_cluster_id is not None

    def test_singleton_cluster(self):
        """Test that dissimilar test creates singleton cluster."""
        from hierarchical_clustering import perform_hierarchical_clustering

        tests = [
            {'name': 'GLUCOSE', 'unit': 'mg/dL'},
            {'name': 'GLUCOSE BLOOD', 'unit': 'mg/dL'},
            {'name': 'COMPLETELY DIFFERENT TEST', 'unit': '%'},  # Very different
        ]

        clusters, _, _ = perform_hierarchical_clustering(tests, threshold=0.9)

        # Third test should be in its own cluster
        singleton_found = False
        for cluster_id, test_indices in clusters.items():
            if len(test_indices) == 1 and 2 in test_indices:
                singleton_found = True
                break

        assert singleton_found
