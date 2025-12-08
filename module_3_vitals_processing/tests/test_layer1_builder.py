"""Tests for layer1_builder module."""
import pytest
import pandas as pd
from processing.layer1_builder import LAYER1_SCHEMA, CORE_VITALS


class TestLayer1Schema:
    """Tests for Layer 1 schema definitions."""

    def test_schema_has_required_columns(self):
        """Schema defines all required columns."""
        required = {
            "EMPI", "timestamp", "hours_from_pe", "vital_type",
            "value", "units", "source", "source_detail", "confidence",
            "is_calculated", "is_flagged_abnormal", "report_number"
        }
        assert required <= set(LAYER1_SCHEMA.keys())

    def test_core_vitals_defined(self):
        """Seven core vitals are defined."""
        expected = {"HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"}
        assert set(CORE_VITALS) == expected

    def test_schema_types_are_valid(self):
        """Schema types are valid pandas/numpy types."""
        valid_types = {"str", "datetime64[ns]", "float64", "bool"}
        for col, dtype in LAYER1_SCHEMA.items():
            assert dtype in valid_types, f"{col} has invalid type {dtype}"
