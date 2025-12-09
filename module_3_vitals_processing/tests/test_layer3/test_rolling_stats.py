"""Tests for rolling window statistics."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.rolling_stats import (
    calculate_rolling_stats,
    ROLLING_STAT_FUNCTIONS,
)


class TestRollingStatFunctions:
    """Tests for individual rolling stat calculations."""

    def test_rolling_mean_6h(self):
        """Rolling mean calculated correctly for 6h window."""
        # 10 hours of data, values 0-9
        df = pd.DataFrame({
            'EMPI': ['E001'] * 10,
            'hour_from_pe': list(range(10)),
            'HR': [float(i) for i in range(10)],
            'mask_HR': [1] * 10,  # All observed
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Hour 5: mean of hours 0-5 = (0+1+2+3+4+5)/6 = 2.5
        assert abs(result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0] - 2.5) < 0.01

    def test_rolling_std_6h(self):
        """Rolling std calculated correctly."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Std of [70,72,74,76,78,80] = 3.74 (ddof=1)
        std_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_std'].iloc[0]
        assert abs(std_val - 3.74) < 0.1

    def test_rolling_cv_6h(self):
        """Coefficient of variation = std / mean."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [100.0] * 6,  # Constant values
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # CV of constant values = 0
        cv_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_cv'].iloc[0]
        assert cv_val == 0.0 or pd.isna(cv_val)  # 0/100 = 0 or NaN if std is NaN

    def test_rolling_range_6h(self):
        """Rolling range = max - min."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 80.0, 75.0, 85.0, 72.0, 78.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Range of [70,80,75,85,72,78] = 85-70 = 15
        range_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_range'].iloc[0]
        assert range_val == 15.0


class TestRollingStatsMultipleWindows:
    """Tests for multiple window sizes."""

    def test_multiple_windows(self):
        """All window sizes calculated."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR': [72.0] * 24,
            'mask_HR': [1] * 24,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6, 12, 24])

        # Check all columns exist
        assert 'HR_roll6h_mean' in result.columns
        assert 'HR_roll12h_mean' in result.columns
        assert 'HR_roll24h_mean' in result.columns


class TestRollingStatsWithMissing:
    """Tests for handling missing data (Tier 1-2 only)."""

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data for rolling stats."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 100.0, 100.0, 100.0],  # Last 3 are imputed
            'mask_HR': [1, 1, 1, 0, 0, 0],  # Tier 3-4 marked with mask=0
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Mean should only use first 3 values: (70+72+74)/3 = 72
        mean_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0]
        assert abs(mean_val - 72.0) < 0.1

    def test_nan_when_no_observations(self):
        """Returns NaN when window has no Tier 1-2 data."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [100.0] * 6,  # All imputed
            'mask_HR': [0] * 6,  # No observations
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        assert pd.isna(result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0])


class TestRollingStatsMultipleVitals:
    """Tests for multiple vital signs."""

    def test_multiple_vitals(self):
        """Stats calculated for all vitals."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'SBP': [120.0] * 6,
            'mask_HR': [1] * 6,
            'mask_SBP': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR', 'SBP'], windows=[6])

        assert 'HR_roll6h_mean' in result.columns
        assert 'SBP_roll6h_mean' in result.columns


class TestRollingStatsMultiplePatients:
    """Tests for multiple patients."""

    def test_separate_by_patient(self):
        """Rolling stats computed separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6 + ['E002'] * 6,
            'hour_from_pe': list(range(6)) * 2,
            'HR': [70.0] * 6 + [90.0] * 6,  # Different values per patient
            'mask_HR': [1] * 12,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        e001_mean = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 5)]['HR_roll6h_mean'].iloc[0]
        e002_mean = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 5)]['HR_roll6h_mean'].iloc[0]

        assert abs(e001_mean - 70.0) < 0.1
        assert abs(e002_mean - 90.0) < 0.1
