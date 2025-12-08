"""Tests for layer2_builder module."""
import pytest
import pandas as pd
import numpy as np
from processing.layer2_builder import (
    LAYER2_PARQUET_SCHEMA,
    HOUR_RANGE,
    VITAL_ORDER,
    FORWARD_FILL_LIMITS,
)


class TestLayer2Schema:
    """Tests for Layer 2 schema definitions."""

    def test_parquet_schema_has_required_columns(self):
        """Parquet schema has all required columns."""
        required = {
            "EMPI", "hour_from_pe", "vital_type",
            "mean", "median", "std", "min", "max", "count", "mask"
        }
        assert required <= set(LAYER2_PARQUET_SCHEMA.keys())

    def test_hour_range_is_correct(self):
        """Hour range is -24 to +720 (745 hours)."""
        assert HOUR_RANGE == list(range(-24, 721))
        assert len(HOUR_RANGE) == 745

    def test_vital_order_is_correct(self):
        """Seven core vitals in correct order."""
        assert VITAL_ORDER == ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]
        assert len(VITAL_ORDER) == 7

    def test_forward_fill_limits_defined(self):
        """Forward-fill limits defined for all vitals."""
        assert FORWARD_FILL_LIMITS["HR"] == 6
        assert FORWARD_FILL_LIMITS["SPO2"] == 4
        assert FORWARD_FILL_LIMITS["TEMP"] == 12
