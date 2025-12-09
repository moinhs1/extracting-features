"""Tests for composite vital sign calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.composite_vitals import calculate_shock_index, calculate_pulse_pressure, add_composite_vitals


class TestShockIndex:
    """Tests for shock index calculation."""

    def test_shock_index_basic(self):
        """Shock index = HR / SBP."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [100.0],
            'SBP': [100.0],
        })
        result = calculate_shock_index(df)
        assert result['shock_index'].iloc[0] == 1.0

    def test_shock_index_normal(self):
        """Normal shock index ~0.5-0.7."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [70.0],
            'SBP': [120.0],
        })
        result = calculate_shock_index(df)
        assert abs(result['shock_index'].iloc[0] - 0.583) < 0.01

    def test_shock_index_missing_hr(self):
        """Shock index NaN when HR missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [np.nan],
            'SBP': [120.0],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])

    def test_shock_index_missing_sbp(self):
        """Shock index NaN when SBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [np.nan],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])

    def test_shock_index_zero_sbp_returns_nan(self):
        """Shock index NaN when SBP is zero (avoid division by zero)."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [0.0],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])


class TestPulsePressure:
    """Tests for pulse pressure calculation."""

    def test_pulse_pressure_basic(self):
        """Pulse pressure = SBP - DBP."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [120.0],
            'DBP': [80.0],
        })
        result = calculate_pulse_pressure(df)
        assert result['pulse_pressure'].iloc[0] == 40.0

    def test_pulse_pressure_narrow(self):
        """Narrow pulse pressure indicates poor cardiac output."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [90.0],
            'DBP': [70.0],
        })
        result = calculate_pulse_pressure(df)
        assert result['pulse_pressure'].iloc[0] == 20.0

    def test_pulse_pressure_missing_sbp(self):
        """Pulse pressure NaN when SBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [np.nan],
            'DBP': [80.0],
        })
        result = calculate_pulse_pressure(df)
        assert pd.isna(result['pulse_pressure'].iloc[0])

    def test_pulse_pressure_missing_dbp(self):
        """Pulse pressure NaN when DBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [120.0],
            'DBP': [np.nan],
        })
        result = calculate_pulse_pressure(df)
        assert pd.isna(result['pulse_pressure'].iloc[0])


class TestAddCompositeVitals:
    """Tests for adding both composites to DataFrame."""

    def test_add_both_composites(self):
        """Adds both shock_index and pulse_pressure columns."""
        df = pd.DataFrame({
            'EMPI': ['E001', 'E001'],
            'hour_from_pe': [0, 1],
            'HR': [80.0, 90.0],
            'SBP': [120.0, 110.0],
            'DBP': [80.0, 70.0],
        })
        result = add_composite_vitals(df)
        assert 'shock_index' in result.columns
        assert 'pulse_pressure' in result.columns
        assert len(result) == 2

    def test_preserves_existing_columns(self):
        """Existing columns preserved."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [120.0],
            'DBP': [80.0],
            'MAP': [93.3],
            'RR': [16.0],
        })
        result = add_composite_vitals(df)
        assert 'MAP' in result.columns
        assert 'RR' in result.columns
