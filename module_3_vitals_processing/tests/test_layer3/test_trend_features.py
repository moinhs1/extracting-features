"""Tests for trend feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.trend_features import (
    calculate_trend_features,
    calculate_direction,
)


class TestTrendSlope:
    """Tests for slope calculation."""

    def test_slope_positive_trend(self):
        """Positive slope for increasing values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # +2 per hour
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope > 0
        assert abs(slope - 2.0) < 0.1  # Slope ~2

    def test_slope_negative_trend(self):
        """Negative slope for decreasing values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [80.0, 78.0, 76.0, 74.0, 72.0, 70.0],  # -2 per hour
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope < 0
        assert abs(slope - (-2.0)) < 0.1

    def test_slope_flat_trend(self):
        """Zero slope for constant values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,  # Constant
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert abs(slope) < 0.01


class TestTrendR2:
    """Tests for R-squared calculation."""

    def test_r2_perfect_linear(self):
        """R-squared = 1.0 for perfect linear trend."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # Perfect linear
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        r2 = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h_r2'].iloc[0]
        assert r2 > 0.99

    def test_r2_noisy_trend(self):
        """R-squared < 1.0 for noisy trend."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 75.0, 72.0, 78.0, 74.0, 80.0],  # Noisy upward
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        r2 = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h_r2'].iloc[0]
        assert 0 < r2 < 1.0


class TestTrendDirection:
    """Tests for direction classification."""

    def test_direction_improving_hr_decreasing_from_tachycardia(self):
        """HR decreasing from >100 is improving."""
        # HR at 105 (tachycardia) and decreasing
        direction = calculate_direction(slope=-2.0, current_value=105.0, vital='HR')
        assert direction == 1  # Improving

    def test_direction_worsening_hr_increasing_to_tachycardia(self):
        """HR increasing toward >100 is worsening."""
        direction = calculate_direction(slope=2.0, current_value=95.0, vital='HR')
        assert direction == -1  # Worsening

    def test_direction_stable_flat_slope(self):
        """Flat slope is stable."""
        direction = calculate_direction(slope=0.0, current_value=72.0, vital='HR')
        assert direction == 0  # Stable

    def test_direction_sbp_increasing_is_improving(self):
        """SBP increasing (away from hypotension) is improving."""
        direction = calculate_direction(slope=2.0, current_value=100.0, vital='SBP')
        assert direction == 1  # Improving

    def test_direction_spo2_increasing_is_improving(self):
        """SpO2 increasing is always improving."""
        direction = calculate_direction(slope=1.0, current_value=94.0, vital='SPO2')
        assert direction == 1  # Improving

    def test_direction_shock_index_decreasing_is_improving(self):
        """Shock index decreasing is improving."""
        # Slope of -0.6 is above the 0.5 threshold for detection
        direction = calculate_direction(slope=-0.6, current_value=0.8, vital='shock_index')
        assert direction == 1  # Improving


class TestTrendFeaturesWithMissing:
    """Tests for handling missing data."""

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data for trends."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 50.0, 50.0, 50.0],  # Last 3 imputed (bad values)
            'mask_HR': [1, 1, 1, 0, 0, 0],
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        # Slope should be based on first 3 values only: positive
        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope > 0  # Would be negative if using all data
