"""Tests for Layer 2 comorbidity scores builder."""
import pytest
import pandas as pd
from processing.layer2_builder import build_layer2_comorbidity_scores


class TestLayer2Builder:
    """Tests for Layer 2 comorbidity score generation."""

    def test_output_has_required_columns(self):
        """Output has CCI columns."""
        layer1 = pd.DataFrame({
            "EMPI": ["P1", "P1"],
            "icd_code": ["I50.9", "J44.9"],
            "icd_version": ["10", "10"],
            "is_preexisting": [True, True],
            "days_from_pe": [-100, -50],
        })
        result = build_layer2_comorbidity_scores(layer1)
        assert "EMPI" in result.columns
        assert "cci_score" in result.columns
        assert "cci_components" in result.columns

    def test_one_row_per_patient(self):
        """One row per patient in output."""
        layer1 = pd.DataFrame({
            "EMPI": ["P1", "P1", "P2"],
            "icd_code": ["I50.9", "J44.9", "C34.9"],
            "icd_version": ["10", "10", "10"],
            "is_preexisting": [True, True, True],
            "days_from_pe": [-100, -50, -30],
        })
        result = build_layer2_comorbidity_scores(layer1)
        assert len(result) == 2
        assert set(result["EMPI"]) == {"P1", "P2"}

    def test_correct_scores(self):
        """Scores are calculated correctly."""
        layer1 = pd.DataFrame({
            "EMPI": ["P1", "P1"],
            "icd_code": ["I50.9", "C78.0"],  # CHF(1) + Metastatic(6) = 7
            "icd_version": ["10", "10"],
            "is_preexisting": [True, True],
            "days_from_pe": [-100, -50],
        })
        result = build_layer2_comorbidity_scores(layer1)
        assert result.iloc[0]["cci_score"] == 7
