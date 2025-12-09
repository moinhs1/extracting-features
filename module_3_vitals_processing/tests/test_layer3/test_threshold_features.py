"""Tests for threshold feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.threshold_features import (
    calculate_threshold_features,
    CLINICAL_THRESHOLDS,
)


class TestCumulativeHours:
    """Tests for cumulative hours above/below threshold."""

    def test_hours_tachycardia_counting(self):
        """Counts hours with HR > 100."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [90.0, 105.0, 110.0, 95.0, 102.0, 98.0],
        })

        result = calculate_threshold_features(df)

        # Hours with HR > 100: hours 1, 2, 4 = 3 hours
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['hours_tachycardia'] == 3

    def test_hours_hypotension_counting(self):
        """Counts hours with SBP < 90."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'SBP': [120.0, 85.0, 88.0, 95.0, 82.0, 110.0],
        })

        result = calculate_threshold_features(df)

        # Hours with SBP < 90: hours 1, 2, 4 = 3 hours
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['hours_hypotension'] == 3

    def test_hours_cumulative_over_time(self):
        """Cumulative count increases over time."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [105.0, 105.0, 105.0, 105.0],  # All tachycardia
        })

        result = calculate_threshold_features(df)

        assert result[result['hour_from_pe'] == 0]['hours_tachycardia'].iloc[0] == 1
        assert result[result['hour_from_pe'] == 1]['hours_tachycardia'].iloc[0] == 2
        assert result[result['hour_from_pe'] == 2]['hours_tachycardia'].iloc[0] == 3
        assert result[result['hour_from_pe'] == 3]['hours_tachycardia'].iloc[0] == 4


class TestTimeToFirst:
    """Tests for time-to-first threshold crossing."""

    def test_time_to_first_tachycardia(self):
        """Finds first hour with HR > 100."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [80.0, 85.0, 90.0, 105.0, 110.0, 95.0],
        })

        result = calculate_threshold_features(df)

        # First tachycardia at hour 3
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['time_to_first_tachycardia'] == 3

    def test_time_to_first_never_crossed(self):
        """NaN if threshold never crossed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # Never > 100
        })

        result = calculate_threshold_features(df)

        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert pd.isna(final_row['time_to_first_tachycardia'])

    def test_time_to_first_at_hour_zero(self):
        """Time = 0 if crossed at first hour."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'SBP': [85.0, 90.0, 95.0, 100.0],  # Hypotensive at hour 0
        })

        result = calculate_threshold_features(df)

        final_row = result[result['hour_from_pe'] == 3].iloc[0]
        assert final_row['time_to_first_hypotension'] == 0


class TestMultiplePatients:
    """Tests for handling multiple patients."""

    def test_separate_by_patient(self):
        """Thresholds calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [105.0, 105.0, 105.0, 105.0] + [80.0, 80.0, 80.0, 80.0],
        })

        result = calculate_threshold_features(df)

        e001_final = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)].iloc[0]
        e002_final = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)].iloc[0]

        assert e001_final['hours_tachycardia'] == 4
        assert e002_final['hours_tachycardia'] == 0


class TestShockIndexThreshold:
    """Tests for shock index threshold."""

    def test_hours_high_shock_index(self):
        """Counts hours with shock_index > 0.9."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'shock_index': [0.7, 0.95, 1.0, 0.8],
        })

        result = calculate_threshold_features(df)

        # Hours 1, 2 have shock_index > 0.9
        final_row = result[result['hour_from_pe'] == 3].iloc[0]
        assert final_row['hours_high_shock_index'] == 2
