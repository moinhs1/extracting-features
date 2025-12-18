"""Tests for CCS mapper."""

import pytest
import pandas as pd
from pathlib import Path
from processing.ccs_mapper import CCSMapper


@pytest.fixture
def ccs_mapper(tmp_path):
    """Create CCS mapper with test crosswalk."""
    crosswalk = pd.DataFrame({
        'icd_code': ['I26', 'I26.0', 'I50', 'J44', '428', '415'],
        'icd_version': ['10', '10', '10', '10', '9', '9'],
        'ccs_category': [103, 103, 108, 127, 108, 103],
        'ccs_description': ['Pulmonary heart disease', 'Pulmonary heart disease',
                           'CHF', 'COPD', 'CHF', 'Pulmonary heart disease'],
    })
    crosswalk_path = tmp_path / "ccs_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    return CCSMapper(crosswalk_path)


class TestCCSMapper:
    def test_exact_match(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("I50", "10")
        assert category == 108

    def test_prefix_match(self, ccs_mapper):
        # I26.99 should match I26
        category = ccs_mapper.get_ccs_category("I26.99", "10")
        assert category == 103

    def test_icd9_match(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("428.0", "9")
        assert category == 108

    def test_no_match_returns_none(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("INVALID", "10")
        assert category is None

    def test_get_description(self, ccs_mapper):
        desc = ccs_mapper.get_ccs_description(108)
        assert desc == "CHF"


class TestPatientCategorization:
    def test_categorize_patient(self, ccs_mapper):
        diagnoses = pd.DataFrame({
            'icd_code': ['I26.99', 'I50.9', 'J44.1'],
            'icd_version': ['10', '10', '10'],
            'is_preexisting': [True, True, True],
        })
        result = ccs_mapper.categorize_patient_diagnoses(diagnoses)

        # Should have 3 unique categories
        assert len(result) == 3
        assert 103 in result['ccs_category'].values  # PE
        assert 108 in result['ccs_category'].values  # CHF
        assert 127 in result['ccs_category'].values  # COPD
