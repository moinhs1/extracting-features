"""Tests for Charlson Comorbidity Index calculation."""
import pytest
import pandas as pd
from processing.charlson_calculator import (
    code_matches_component,
    calculate_charlson_for_patient,
    calculate_charlson_batch,
)


class TestCodeMatching:
    """Tests for ICD code to component matching."""

    def test_icd10_exact_match(self):
        """Exact ICD-10 code matches."""
        assert code_matches_component("I50", "congestive_heart_failure", "10") is True
        assert code_matches_component("I50.9", "congestive_heart_failure", "10") is True

    def test_icd10_prefix_match(self):
        """ICD-10 codes match on prefix."""
        assert code_matches_component("I21.0", "myocardial_infarction", "10") is True
        assert code_matches_component("C34.9", "malignancy", "10") is True

    def test_icd9_match(self):
        """ICD-9 codes match correctly."""
        assert code_matches_component("428.0", "congestive_heart_failure", "9") is True
        assert code_matches_component("410.9", "myocardial_infarction", "9") is True

    def test_non_matching_code(self):
        """Non-matching codes return False."""
        assert code_matches_component("J44.9", "congestive_heart_failure", "10") is False


class TestCharlsonCalculation:
    """Tests for CCI calculation."""

    def test_single_comorbidity(self):
        """Single comorbidity scores correctly."""
        diagnoses = pd.DataFrame({
            "icd_code": ["I50.9"],
            "icd_version": ["10"],
            "is_preexisting": [True],
        })
        result = calculate_charlson_for_patient(diagnoses)
        assert result["cci_score"] == 1
        assert "congestive_heart_failure" in result["cci_components"]

    def test_multiple_comorbidities(self):
        """Multiple comorbidities sum correctly."""
        diagnoses = pd.DataFrame({
            "icd_code": ["I50.9", "C34.9", "N18.3"],  # CHF(1) + Cancer(2) + Renal(2) = 5
            "icd_version": ["10", "10", "10"],
            "is_preexisting": [True, True, True],
        })
        result = calculate_charlson_for_patient(diagnoses)
        assert result["cci_score"] == 5

    def test_only_preexisting_counted(self):
        """Only preexisting diagnoses are counted."""
        diagnoses = pd.DataFrame({
            "icd_code": ["I50.9", "C34.9"],
            "icd_version": ["10", "10"],
            "is_preexisting": [True, False],  # Cancer is not preexisting
        })
        result = calculate_charlson_for_patient(diagnoses)
        assert result["cci_score"] == 1  # Only CHF

    def test_hierarchy_diabetes(self):
        """Complicated diabetes supersedes uncomplicated."""
        diagnoses = pd.DataFrame({
            "icd_code": ["E11.0", "E11.5"],  # Uncomplicated + Complicated
            "icd_version": ["10", "10"],
            "is_preexisting": [True, True],
        })
        result = calculate_charlson_for_patient(diagnoses)
        # Should only count complicated (weight 2), not both
        assert result["cci_score"] == 2

    def test_hierarchy_liver(self):
        """Severe liver disease supersedes mild."""
        diagnoses = pd.DataFrame({
            "icd_code": ["K70.0", "K72.0"],  # Mild + Severe
            "icd_version": ["10", "10"],
            "is_preexisting": [True, True],
        })
        result = calculate_charlson_for_patient(diagnoses)
        # Should only count severe (weight 3), not both
        assert result["cci_score"] == 3

    def test_hierarchy_malignancy(self):
        """Metastatic cancer supersedes localized."""
        diagnoses = pd.DataFrame({
            "icd_code": ["C34.9", "C78.0"],  # Primary lung + Metastatic
            "icd_version": ["10", "10"],
            "is_preexisting": [True, True],
        })
        result = calculate_charlson_for_patient(diagnoses)
        # Should only count metastatic (weight 6), not both
        assert result["cci_score"] == 6


class TestBatchCalculation:
    """Tests for batch CCI calculation."""

    def test_batch_multiple_patients(self):
        """Batch calculation works for multiple patients."""
        diagnoses = pd.DataFrame({
            "EMPI": ["P1", "P1", "P2", "P2"],
            "icd_code": ["I50.9", "J44.9", "C34.9", "N18.3"],
            "icd_version": ["10", "10", "10", "10"],
            "is_preexisting": [True, True, True, True],
        })
        result = calculate_charlson_batch(diagnoses)
        assert len(result) == 2
        assert result[result["EMPI"] == "P1"]["cci_score"].values[0] == 2  # CHF + COPD
        assert result[result["EMPI"] == "P2"]["cci_score"].values[0] == 4  # Cancer + Renal
