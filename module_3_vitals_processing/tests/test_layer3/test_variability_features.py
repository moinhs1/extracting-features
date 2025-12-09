"""Tests for variability feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.variability_features import (
    calculate_rmssd,
    calculate_successive_var,
    calculate_variability_features,
)


class TestRMSSD:
    """Tests for RMSSD calculation."""

    def test_rmssd_constant_values(self):
        """RMSSD = 0 for constant values."""
        values = np.array([72.0, 72.0, 72.0, 72.0])
        rmssd = calculate_rmssd(values)
        assert rmssd == 0.0

    def test_rmssd_alternating_values(self):
        """RMSSD calculated for alternating values."""
        # Values: 70, 80, 70, 80 -> diffs: 10, 10, 10
        # RMSSD = sqrt(mean([100, 100, 100])) = 10
        values = np.array([70.0, 80.0, 70.0, 80.0])
        rmssd = calculate_rmssd(values)
        assert abs(rmssd - 10.0) < 0.1

    def test_rmssd_with_nan(self):
        """RMSSD handles NaN values."""
        values = np.array([70.0, np.nan, 80.0, 75.0])
        rmssd = calculate_rmssd(values)
        # Should use consecutive non-NaN pairs only
        assert not np.isnan(rmssd)

    def test_rmssd_insufficient_data(self):
        """RMSSD returns NaN with < 2 values."""
        values = np.array([72.0])
        rmssd = calculate_rmssd(values)
        assert np.isnan(rmssd)


class TestSuccessiveVar:
    """Tests for successive variance calculation."""

    def test_successive_var_constant(self):
        """Successive var = 0 for constant values."""
        values = np.array([72.0, 72.0, 72.0, 72.0])
        sv = calculate_successive_var(values)
        assert sv == 0.0

    def test_successive_var_calculated(self):
        """Successive var = sum of abs differences."""
        # Values: 70, 75, 72, 80 -> abs diffs: 5, 3, 8 -> sum = 16
        values = np.array([70.0, 75.0, 72.0, 80.0])
        sv = calculate_successive_var(values)
        assert sv == 16.0


class TestVariabilityFeatures:
    """Tests for full variability feature calculation."""

    def test_calculates_both_features(self):
        """Both RMSSD and successive_var calculated."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 73.0, 75.0, 72.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_variability_features(df, vitals=['HR'])

        assert 'HR_rmssd' in result.columns
        assert 'HR_successive_var' in result.columns

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 100.0, 100.0, 100.0, 100.0],
            'mask_HR': [1, 1, 0, 0, 0, 0],  # Only first 2 observed
        })

        result = calculate_variability_features(df, vitals=['HR'])

        # Should only use first 2 values, so successive_var = |72-70| = 2
        sv = result.loc[result['hour_from_pe'] == 5, 'HR_successive_var'].iloc[0]
        assert abs(sv - 2.0) < 0.1

    def test_multiple_patients(self):
        """Variability calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [70.0, 72.0, 74.0, 76.0] + [80.0, 85.0, 90.0, 95.0],
            'mask_HR': [1] * 8,
        })

        result = calculate_variability_features(df, vitals=['HR'])

        # E001: constant +2 changes, E002: constant +5 changes
        e001_sv = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)]['HR_successive_var'].iloc[0]
        e002_sv = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)]['HR_successive_var'].iloc[0]

        assert abs(e001_sv - 6.0) < 0.1  # 2+2+2
        assert abs(e002_sv - 15.0) < 0.1  # 5+5+5
