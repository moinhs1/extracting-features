"""Tests for Layer 1 canonical diagnosis builder."""
import pytest
import pandas as pd
from datetime import datetime
from processing.layer1_builder import (
    LAYER1_SCHEMA,
    add_pe_relative_timing,
    add_temporal_flags,
    build_layer1,
)


class TestPERelativeTiming:
    """Tests for PE-relative timing calculation."""

    def test_adds_days_from_pe(self):
        """days_from_pe column is added."""
        df = pd.DataFrame({
            "EMPI": ["100001", "100001"],
            "diagnosis_date": [datetime(2023, 6, 10), datetime(2023, 6, 20)],
        })
        pe_times = {"100001": datetime(2023, 6, 15)}
        result = add_pe_relative_timing(df, pe_times)
        assert "days_from_pe" in result.columns
        assert result.iloc[0]["days_from_pe"] == -5
        assert result.iloc[1]["days_from_pe"] == 5

    def test_adds_hours_from_pe(self):
        """hours_from_pe column is added."""
        df = pd.DataFrame({
            "EMPI": ["100001"],
            "diagnosis_date": [datetime(2023, 6, 15, 12, 0)],
        })
        pe_times = {"100001": datetime(2023, 6, 15, 0, 0)}
        result = add_pe_relative_timing(df, pe_times)
        assert result.iloc[0]["hours_from_pe"] == 12.0


class TestTemporalFlags:
    """Tests for temporal flag assignment."""

    def test_adds_temporal_category(self):
        """temporal_category column is added."""
        df = pd.DataFrame({
            "days_from_pe": [-31, 0, 5, 31],
        })
        result = add_temporal_flags(df)
        assert result.iloc[0]["temporal_category"] == "preexisting_remote"
        assert result.iloc[1]["temporal_category"] == "index_concurrent"
        assert result.iloc[2]["temporal_category"] == "early_complication"
        assert result.iloc[3]["temporal_category"] == "follow_up"

    def test_adds_boolean_flags(self):
        """Boolean temporal flags are added."""
        df = pd.DataFrame({"days_from_pe": [-31]})
        result = add_temporal_flags(df)
        assert result.iloc[0]["is_preexisting"] == True
        assert result.iloc[0]["is_complication"] == False


class TestBuildLayer1:
    """Tests for full Layer 1 build."""

    def test_output_has_required_columns(self):
        """Output has all Layer 1 schema columns."""
        df = pd.DataFrame({
            "EMPI": ["100001"],
            "diagnosis_date": [datetime(2023, 6, 15)],
            "icd_code": ["I26.0"],
            "icd_version": ["10"],
            "Diagnosis_Name": ["PE"],
            "Diagnosis_Flag": ["Primary"],
            "Encounter_number": ["ENC001"],
            "Inpatient_Outpatient": ["Inpatient"],
        })
        pe_times = {"100001": datetime(2023, 6, 15)}
        result = build_layer1(df, pe_times)

        for col in LAYER1_SCHEMA.keys():
            assert col in result.columns, f"Missing column: {col}"

    def test_filters_to_window(self):
        """Diagnoses outside window are filtered."""
        df = pd.DataFrame({
            "EMPI": ["100001", "100001", "100001"],
            "diagnosis_date": [
                datetime(2010, 1, 1),  # Way before
                datetime(2023, 6, 15),  # At PE
                datetime(2030, 1, 1),   # Way after
            ],
            "icd_code": ["I50.9", "I26.0", "J44.9"],
            "icd_version": ["10", "10", "10"],
            "Diagnosis_Name": ["HF", "PE", "COPD"],
            "Diagnosis_Flag": ["Primary", "Primary", "Primary"],
            "Encounter_number": ["E1", "E2", "E3"],
            "Inpatient_Outpatient": ["Out", "In", "Out"],
        })
        pe_times = {"100001": datetime(2023, 6, 15)}
        result = build_layer1(df, pe_times, min_days=-365*5, max_days=365)
        # Should include only the 2023-06-15 date (day 0), exclude 2010 and 2030 (outside window)
        assert len(result) == 1
