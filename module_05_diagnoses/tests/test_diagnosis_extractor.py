"""Tests for RPDR diagnosis extraction."""
import pytest
import pandas as pd
from io import StringIO
from extractors.diagnosis_extractor import (
    parse_diagnosis_line,
    extract_diagnoses_for_patients,
    filter_excluded_codes,
)


SAMPLE_DIA_DATA = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Diagnosis_Name|Code_Type|Code|Diagnosis_Flag|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
100001|10001|MGH|123|6/15/2023|Pulmonary embolism|ICD10|I26.99|Primary|Dr Smith|ED|MGH|Inpatient|ENC001
100001|10001|MGH|123|6/10/2023|Heart failure|ICD10|I50.9|Secondary|Dr Smith|Cardio|MGH|Outpatient|ENC002
100001|10001|MGH|123|6/20/2023|Acute kidney injury|ICD10|N17.9|Primary|Dr Jones|Nephro|MGH|Inpatient|ENC003
100002|10002|MGH|456|5/1/2023|COPD|ICD9|496|Primary|Dr Brown|Pulm|MGH|Outpatient|ENC004
"""


class TestParseDiagnosisLine:
    """Tests for parsing individual diagnosis lines."""

    def test_parse_valid_line(self):
        """Valid line is parsed correctly."""
        line = "100001|10001|MGH|123|6/15/2023|Pulmonary embolism|ICD10|I26.99|Primary|Dr Smith|ED|MGH|Inpatient|ENC001"
        result = parse_diagnosis_line(line)
        assert result["EMPI"] == "100001"
        assert result["Code"] == "I26.99"
        assert result["Code_Type"] == "ICD10"
        assert result["Diagnosis_Flag"] == "Primary"

    def test_parse_handles_missing_fields(self):
        """Lines with missing fields still parse."""
        line = "100001|10001|MGH|123|6/15/2023|PE|ICD10|I26.99|||ED|MGH|Inpatient|"
        result = parse_diagnosis_line(line)
        assert result["EMPI"] == "100001"
        assert result["Diagnosis_Flag"] == ""


class TestFilterExcludedCodes:
    """Tests for code exclusion filtering."""

    def test_excludes_z_codes(self):
        """Z00-Z13 codes are excluded."""
        df = pd.DataFrame({
            "Code": ["I26.0", "Z00.0", "Z12.31", "J44.9"],
            "icd_version": ["10", "10", "10", "10"]
        })
        result = filter_excluded_codes(df)
        assert len(result) == 2
        assert "Z00.0" not in result["Code"].values

    def test_excludes_v_screening_codes(self):
        """ICD-9 V70-V82 screening codes are excluded."""
        df = pd.DataFrame({
            "Code": ["415.1", "V70.0", "V72.31", "428.0"],
            "icd_version": ["9", "9", "9", "9"]
        })
        result = filter_excluded_codes(df)
        assert len(result) == 2


class TestExtractDiagnoses:
    """Tests for full extraction pipeline."""

    def test_extracts_for_patient_set(self):
        """Only extracts for specified patients."""
        patient_ids = {"100001"}
        df = extract_diagnoses_for_patients(
            StringIO(SAMPLE_DIA_DATA),
            patient_ids
        )
        assert len(df) == 3
        assert df["EMPI"].unique().tolist() == ["100001"]

    def test_parses_dates(self):
        """Dates are parsed correctly."""
        patient_ids = {"100001"}
        df = extract_diagnoses_for_patients(
            StringIO(SAMPLE_DIA_DATA),
            patient_ids
        )
        assert pd.api.types.is_datetime64_any_dtype(df["diagnosis_date"])
