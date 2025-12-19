"""Integration tests for Layer 2 with Elixhauser and CCS."""

import pytest
import pandas as pd
from processing.layer2_builder import build_layer2_comorbidity_scores, build_layer2_ccs_categories
from pathlib import Path


@pytest.fixture
def sample_layer1():
    """Sample Layer 1 data for testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P1', 'P2', 'P2'],
        'icd_code': ['I26.99', 'I50.9', 'J44.1', 'E66.0', 'C34.9'],
        'icd_version': ['10', '10', '10', '10', '10'],
        'is_preexisting': [True, True, True, True, True],
    })


class TestComorbidityScoresIntegration:
    def test_produces_both_indices(self, sample_layer1):
        result = build_layer2_comorbidity_scores(sample_layer1)

        assert 'cci_score' in result.columns
        assert 'elixhauser_score' in result.columns
        assert len(result) == 2  # 2 patients

    def test_scores_reasonable(self, sample_layer1):
        result = build_layer2_comorbidity_scores(sample_layer1)

        # P1: CHF, COPD in both indices
        p1 = result[result['EMPI'] == 'P1'].iloc[0]
        assert p1['cci_score'] >= 2  # CHF + COPD
        assert p1['elixhauser_score'] >= 10  # CHF(7) + COPD(3)


class TestCCSIntegration:
    def test_produces_categories(self, sample_layer1, tmp_path):
        # Create minimal crosswalk
        crosswalk = pd.DataFrame({
            'icd_code': ['I26', 'I50', 'J44', 'E66', 'C34'],
            'icd_version': ['10', '10', '10', '10', '10'],
            'ccs_category': [103, 108, 127, 58, 19],
            'ccs_description': ['PE', 'CHF', 'COPD', 'Obesity', 'Lung cancer'],
        })
        crosswalk_path = tmp_path / "ccs_crosswalk.csv"
        crosswalk.to_csv(crosswalk_path, index=False)

        result = build_layer2_ccs_categories(sample_layer1, crosswalk_path)

        assert len(result) > 0
        assert 'EMPI' in result.columns
        assert 'ccs_category' in result.columns
