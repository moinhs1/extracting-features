# /home/moin/TDA_11_25/module_04_medications/tests/test_individual_indicator_builder.py
"""Tests for individual medication indicator generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPrevalenceFiltering:
    """Test prevalence-based medication filtering."""

    def test_filter_by_prevalence(self):
        """Filter medications by patient count threshold."""
        from transformers.individual_indicator_builder import filter_by_prevalence

        # Create 25 unique patients for aspirin/common_drug, 1 for rare_drug
        df = pd.DataFrame({
            'empi': [f'p{i}' for i in range(25)] + ['p1'] + [f'p{i}' for i in range(25)],
            'ingredient_name': ['aspirin'] * 25 + ['rare_drug'] + ['common_drug'] * 25,
        })

        # With threshold of 20 patients
        result = filter_by_prevalence(df, min_patients=20)

        assert 'aspirin' in result  # 25 patients
        assert 'common_drug' in result  # 25 patients
        assert 'rare_drug' not in result  # Only 1 patient

    def test_always_include_exceptions(self):
        """Always include critical medications regardless of prevalence."""
        from transformers.individual_indicator_builder import filter_by_prevalence

        df = pd.DataFrame({
            'empi': ['1', '2'],  # Only 2 patients
            'ingredient_name': ['heparin', 'heparin'],
        })

        result = filter_by_prevalence(df, min_patients=20)

        # Heparin should be included even though only 2 patients
        assert 'heparin' in result


class TestIndicatorCreation:
    """Test sparse indicator creation."""

    def test_create_indicators_single_patient(self):
        """Create indicators for single patient."""
        from transformers.individual_indicator_builder import create_patient_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [1, 2, 3],
            'ingredient_name': ['aspirin', 'aspirin', 'metoprolol'],
            'parsed_dose_value': [325, 325, 50],
        })

        medications = ['aspirin', 'metoprolol', 'lisinopril']

        result = create_patient_indicators(df, medications, window='acute')

        assert len(result) == 1
        assert result.iloc[0]['med_aspirin'] == True
        assert result.iloc[0]['med_aspirin_count'] == 2
        assert result.iloc[0]['med_metoprolol'] == True
        assert result.iloc[0]['med_lisinopril'] == False


class TestSparseStorage:
    """Test sparse matrix storage."""

    def test_to_sparse_matrix(self):
        """Convert to scipy sparse matrix."""
        from transformers.individual_indicator_builder import to_sparse_matrix

        df = pd.DataFrame({
            'empi': ['1', '2'],
            'time_window': ['acute', 'acute'],
            'med_aspirin': [True, False],
            'med_metoprolol': [False, True],
        })

        sparse, feature_names = to_sparse_matrix(df)

        assert sparse.shape == (2, 2)
        assert 'med_aspirin' in feature_names
        assert sparse[0, feature_names.index('med_aspirin')] == 1
        assert sparse[1, feature_names.index('med_aspirin')] == 0
