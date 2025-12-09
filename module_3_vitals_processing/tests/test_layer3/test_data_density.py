"""Tests for data density feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.data_density import calculate_data_density


class TestObservationPercentage:
    """Tests for observation percentage calculation."""

    def test_obs_pct_all_observed(self):
        """100% when all hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [1] * 6,  # All observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 100.0

    def test_obs_pct_half_observed(self):
        """50% when half hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [1, 0, 1, 0, 1, 0],  # 3 of 6 observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 50.0

    def test_obs_pct_none_observed(self):
        """0% when no hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [0] * 6,  # None observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 0.0


class TestObservationCount:
    """Tests for observation count calculation."""

    def test_obs_count_cumulative(self):
        """Count increases cumulatively."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [72.0] * 4,
            'mask_HR': [1, 1, 1, 1],
        })

        result = calculate_data_density(df, vitals=['HR'])

        assert result[result['hour_from_pe'] == 0]['HR_obs_count'].iloc[0] == 1
        assert result[result['hour_from_pe'] == 1]['HR_obs_count'].iloc[0] == 2
        assert result[result['hour_from_pe'] == 3]['HR_obs_count'].iloc[0] == 4


class TestAnyVitalObserved:
    """Tests for any-vital observation percentage."""

    def test_any_vital_obs_pct(self):
        """Tracks hours with ANY vital observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [72.0] * 4,
            'SBP': [120.0] * 4,
            'mask_HR': [1, 0, 0, 1],  # Observed at hours 0, 3
            'mask_SBP': [0, 1, 0, 0],  # Observed at hour 1
        })

        result = calculate_data_density(df, vitals=['HR', 'SBP'])

        # Hours with any observation: 0, 1, 3 = 3 of 4 = 75%
        final = result[result['hour_from_pe'] == 3].iloc[0]
        assert final['any_vital_obs_pct'] == 75.0


class TestMultiplePatients:
    """Tests for multiple patient handling."""

    def test_separate_by_patient(self):
        """Density calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [72.0] * 8,
            'mask_HR': [1, 1, 1, 1] + [1, 0, 0, 0],  # E001: all obs, E002: 1 obs
        })

        result = calculate_data_density(df, vitals=['HR'])

        e001_final = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)].iloc[0]
        e002_final = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)].iloc[0]

        assert e001_final['HR_obs_pct'] == 100.0
        assert e002_final['HR_obs_pct'] == 25.0
