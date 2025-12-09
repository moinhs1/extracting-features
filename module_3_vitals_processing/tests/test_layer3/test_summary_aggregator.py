"""Tests for summary aggregation."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.summary_aggregator import aggregate_to_summary, SUMMARY_WINDOWS


class TestSummaryWindows:
    """Tests for summary window definitions."""

    def test_summary_windows_defined(self):
        """All 5 clinical windows defined."""
        assert 'pre' in SUMMARY_WINDOWS
        assert 'acute' in SUMMARY_WINDOWS
        assert 'early' in SUMMARY_WINDOWS
        assert 'stab' in SUMMARY_WINDOWS
        assert 'recov' in SUMMARY_WINDOWS

    def test_window_ranges(self):
        """Windows have correct hour ranges."""
        assert SUMMARY_WINDOWS['pre'] == (-24, 0)
        assert SUMMARY_WINDOWS['acute'] == (0, 24)
        assert SUMMARY_WINDOWS['early'] == (24, 72)
        assert SUMMARY_WINDOWS['stab'] == (72, 168)
        assert SUMMARY_WINDOWS['recov'] == (168, 720)


class TestAggregateToSummary:
    """Tests for aggregating time-series to summary."""

    def test_produces_one_row_per_patient(self):
        """Output has one row per patient."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 10 + ['E002'] * 10,
            'hour_from_pe': list(range(10)) * 2,
            'HR_roll6h_mean': [72.0] * 20,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        assert len(result) == 2
        assert set(result['EMPI']) == {'E001', 'E002'}

    def test_aggregates_by_window(self):
        """Features aggregated per summary window."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 50,
            'hour_from_pe': list(range(50)),
            'HR_roll6h_mean': [70.0] * 24 + [80.0] * 26,  # Different in acute vs early
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Acute window (0-24) should have mean ~70
        # Early window (24-50) should have mean ~80
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 70.0) < 1.0
        assert abs(row['HR_roll6h_mean_early_mean'] - 80.0) < 1.0

    def test_mean_aggregation(self):
        """Mean aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),  # Acute window
            'HR_roll6h_mean': [70.0, 72.0, 74.0, 76.0] * 6,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Mean of [70, 72, 74, 76] = 73
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 73.0) < 0.1

    def test_max_aggregation(self):
        """Max aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [70.0] * 20 + [90.0] * 4,  # Max is 90
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        row = result.iloc[0]
        assert row['HR_roll6h_mean_acute_max'] == 90.0

    def test_min_aggregation(self):
        """Min aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [60.0] * 4 + [80.0] * 20,  # Min is 60
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        row = result.iloc[0]
        assert row['HR_roll6h_mean_acute_min'] == 60.0

    def test_handles_nan_in_window(self):
        """Handles NaN values in aggregation."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [np.nan] * 12 + [72.0] * 12,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Should ignore NaN and compute mean of non-NaN values
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 72.0) < 0.1

    def test_threshold_features_summed_by_window(self):
        """Threshold features give window-specific totals."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 50,
            'hour_from_pe': list(range(50)),
            'hours_tachycardia': list(range(1, 51)),  # Cumulative 1-50
        })

        result = aggregate_to_summary(ts_df, feature_cols=['hours_tachycardia'])

        # For cumulative features, take max in each window
        row = result.iloc[0]
        # Acute window ends at hour 23, cumulative = 24
        assert row['hours_tachycardia_acute_max'] == 24
