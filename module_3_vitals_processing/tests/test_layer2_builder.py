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


class TestHourlyAggregation:
    """Tests for hourly aggregation."""

    def test_aggregate_to_hourly_single_value(self):
        """Single value in hour produces correct stats."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001"],
            "hours_from_pe": [0.5],  # Hour bucket 0
            "vital_type": ["HR"],
            "value": [72.0],
        })

        result = aggregate_to_hourly(df)

        row = result[(result["EMPI"] == "E001") &
                     (result["hour_from_pe"] == 0) &
                     (result["vital_type"] == "HR")].iloc[0]

        assert row["mean"] == 72.0
        assert row["median"] == 72.0
        assert pd.isna(row["std"])  # std undefined for n=1
        assert row["min"] == 72.0
        assert row["max"] == 72.0
        assert row["count"] == 1
        assert row["mask"] == 1

    def test_aggregate_to_hourly_multiple_values(self):
        """Multiple values in hour produce correct stats."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E001"],
            "hours_from_pe": [0.0, 0.3, 0.9],  # All in hour 0
            "vital_type": ["HR", "HR", "HR"],
            "value": [70.0, 72.0, 74.0],
        })

        result = aggregate_to_hourly(df)

        row = result[(result["EMPI"] == "E001") &
                     (result["hour_from_pe"] == 0) &
                     (result["vital_type"] == "HR")].iloc[0]

        assert row["mean"] == 72.0
        assert row["median"] == 72.0
        assert abs(row["std"] - 2.0) < 0.01
        assert row["min"] == 70.0
        assert row["max"] == 74.0
        assert row["count"] == 3
        assert row["mask"] == 1

    def test_aggregate_to_hourly_separate_vitals(self):
        """Different vitals aggregated separately."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001", "E001"],
            "hours_from_pe": [0.5, 0.5],
            "vital_type": ["HR", "SBP"],
            "value": [72.0, 120.0],
        })

        result = aggregate_to_hourly(df)

        hr_row = result[(result["vital_type"] == "HR")].iloc[0]
        sbp_row = result[(result["vital_type"] == "SBP")].iloc[0]

        assert hr_row["mean"] == 72.0
        assert sbp_row["mean"] == 120.0


class TestCreateFullGrid:
    """Tests for creating full hourly grid."""

    def test_create_full_grid_fills_missing_hours(self):
        """Creates rows for all hours, not just observed."""
        from processing.layer2_builder import create_full_grid

        # Sparse data: only hour 0 observed
        observed = pd.DataFrame({
            "EMPI": ["E001"],
            "hour_from_pe": [0],
            "vital_type": ["HR"],
            "mean": [72.0],
            "median": [72.0],
            "std": [np.nan],
            "min": [72.0],
            "max": [72.0],
            "count": [1],
            "mask": [1],
        })

        patients = ["E001"]
        result = create_full_grid(observed, patients)

        # Should have 745 hours × 7 vitals = 5215 rows per patient
        expected_rows = 745 * 7
        assert len(result[result["EMPI"] == "E001"]) == expected_rows

    def test_create_full_grid_marks_missing_mask_zero(self):
        """Missing hours have mask=0."""
        from processing.layer2_builder import create_full_grid

        observed = pd.DataFrame({
            "EMPI": ["E001"],
            "hour_from_pe": [0],
            "vital_type": ["HR"],
            "mean": [72.0],
            "median": [72.0],
            "std": [np.nan],
            "min": [72.0],
            "max": [72.0],
            "count": [1],
            "mask": [1],
        })

        result = create_full_grid(observed, ["E001"])

        # Hour 0 HR should have mask=1
        hr_0 = result[(result["hour_from_pe"] == 0) &
                      (result["vital_type"] == "HR")].iloc[0]
        assert hr_0["mask"] == 1

        # Hour 1 HR should have mask=0
        hr_1 = result[(result["hour_from_pe"] == 1) &
                      (result["vital_type"] == "HR")].iloc[0]
        assert hr_1["mask"] == 0


class TestImputation:
    """Tests for three-tier imputation."""

    def test_forward_fill_within_limit(self):
        """Forward-fill works within vital-specific limit."""
        from processing.layer2_builder import apply_imputation

        # HR with 6-hour limit: hours 0-5 should fill from hour 0
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": list(range(10)),
            "vital_type": ["HR"] * 10,
            "mean": [72.0] + [np.nan] * 9,
            "mask": [1] + [0] * 9,
        })

        result = apply_imputation(df)

        # Hours 1-6 should be forward-filled (tier 2)
        for h in range(1, 7):
            row = result[(result["hour_from_pe"] == h)].iloc[0]
            assert row["mean"] == 72.0
            assert row["imputation_tier"] == 2

    def test_forward_fill_respects_limit(self):
        """Forward-fill stops at vital-specific limit."""
        from processing.layer2_builder import apply_imputation

        # HR with 6-hour limit: hour 7+ should NOT forward-fill
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": list(range(10)),
            "vital_type": ["HR"] * 10,
            "mean": [72.0] + [np.nan] * 9,
            "mask": [1] + [0] * 9,
        })

        result = apply_imputation(df)

        # Hours 7+ should use patient mean (tier 3), not forward-fill
        row_7 = result[(result["hour_from_pe"] == 7)].iloc[0]
        assert row_7["imputation_tier"] == 3

    def test_patient_mean_imputation(self):
        """Patient mean used when forward-fill exhausted."""
        from processing.layer2_builder import apply_imputation

        # Patient has observations at hour 0 and 100
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": [0, 1, 2, 3, 4, 5, 6, 7, 8, 100],
            "vital_type": ["HR"] * 10,
            "mean": [70.0] + [np.nan] * 8 + [80.0],
            "mask": [1] + [0] * 8 + [1],
        })

        result = apply_imputation(df)

        # Hour 8 should use patient mean of 75.0
        row_8 = result[(result["hour_from_pe"] == 8)].iloc[0]
        assert row_8["mean"] == 75.0  # (70 + 80) / 2
        assert row_8["imputation_tier"] == 3
