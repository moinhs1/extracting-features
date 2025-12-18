"""Tests for Elixhauser Comorbidity Index calculator."""

import pytest
import pandas as pd
from processing.elixhauser_calculator import (
    code_matches_component,
    calculate_elixhauser_for_patient,
    calculate_elixhauser_batch
)


class TestCodeMatching:
    def test_exact_match_icd10(self):
        assert code_matches_component("I50.9", "congestive_heart_failure", "10") == True

    def test_prefix_match_icd10(self):
        assert code_matches_component("I50.23", "congestive_heart_failure", "10") == True

    def test_no_match(self):
        assert code_matches_component("J44.1", "congestive_heart_failure", "10") == False

    def test_icd9_match(self):
        assert code_matches_component("428.0", "congestive_heart_failure", "9") == True


class TestElixhauserCalculation:
    @pytest.fixture
    def sample_diagnoses(self):
        return pd.DataFrame({
            'icd_code': ['I50.9', 'I48.0', 'E66.0', 'J44.1'],
            'icd_version': ['10', '10', '10', '10'],
            'is_preexisting': [True, True, True, True],
        })

    def test_calculates_score(self, sample_diagnoses):
        result = calculate_elixhauser_for_patient(sample_diagnoses)
        # CHF (7) + Arrhythmia (5) + Obesity (-4) + COPD (3) = 11
        assert result['elixhauser_score'] == 11

    def test_counts_components(self, sample_diagnoses):
        result = calculate_elixhauser_for_patient(sample_diagnoses)
        assert result['elixhauser_component_count'] == 4

    def test_hierarchy_diabetes(self):
        diagnoses = pd.DataFrame({
            'icd_code': ['E11.9', 'E11.5'],  # Uncomplicated and complicated
            'icd_version': ['10', '10'],
            'is_preexisting': [True, True],
        })
        result = calculate_elixhauser_for_patient(diagnoses)
        # Should only count complicated (weight 0), not uncomplicated
        assert result['elixhauser_component_count'] == 1
        assert 'diabetes_complicated' in result['elixhauser_components']

    def test_hierarchy_cancer(self):
        diagnoses = pd.DataFrame({
            'icd_code': ['C34.9', 'C78.0'],  # Solid tumor and metastatic
            'icd_version': ['10', '10'],
            'is_preexisting': [True, True],
        })
        result = calculate_elixhauser_for_patient(diagnoses)
        # Should only count metastatic (12), not solid tumor (4)
        assert result['elixhauser_score'] == 12


class TestBatchCalculation:
    def test_batch_returns_all_patients(self):
        diagnoses = pd.DataFrame({
            'EMPI': ['P1', 'P1', 'P2'],
            'icd_code': ['I50.9', 'J44.1', 'E66.0'],
            'icd_version': ['10', '10', '10'],
            'is_preexisting': [True, True, True],
        })
        result = calculate_elixhauser_batch(diagnoses)
        assert len(result) == 2
        assert set(result['EMPI']) == {'P1', 'P2'}
