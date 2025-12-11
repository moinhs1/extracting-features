# /home/moin/TDA_11_25/module_04_medications/tests/test_dose_intensity_builder.py
"""Tests for dose intensity feature generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDDDNormalization:
    """Test DDD (Defined Daily Dose) normalization."""

    def test_get_ddd_heparin(self):
        """Get DDD for heparin."""
        from transformers.dose_intensity_builder import get_ddd

        ddd = get_ddd('heparin', 'units')

        assert ddd is not None
        assert ddd > 0

    def test_get_ddd_enoxaparin(self):
        """Get DDD for enoxaparin."""
        from transformers.dose_intensity_builder import get_ddd

        ddd = get_ddd('enoxaparin', 'mg')

        assert ddd is not None
        assert ddd == 40  # Standard DDD for enoxaparin

    def test_ddd_ratio_calculation(self):
        """Calculate dose / DDD ratio."""
        from transformers.dose_intensity_builder import calculate_ddd_ratio

        ratio = calculate_ddd_ratio(
            dose_value=80,
            dose_unit='mg',
            ingredient='enoxaparin'
        )

        assert ratio == 2.0  # 80mg / 40mg DDD


class TestDailyDoseAggregation:
    """Test daily dose aggregation."""

    def test_aggregate_daily_doses(self):
        """Aggregate multiple doses to daily total."""
        from transformers.dose_intensity_builder import aggregate_daily_doses

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 6, 12],  # Same day
            'ingredient_name': ['heparin', 'heparin', 'heparin'],
            'parsed_dose_value': [5000, 5000, 5000],
            'parsed_dose_unit': ['units', 'units', 'units'],
        })

        result = aggregate_daily_doses(df, 'heparin')

        assert len(result) == 1
        assert result.iloc[0]['daily_dose'] == 15000


class TestIntensityFeatures:
    """Test intensity feature calculation."""

    def test_hours_since_last(self):
        """Calculate hours since last administration."""
        from transformers.dose_intensity_builder import calculate_hours_since_last

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 6, 24],
            'ingredient_name': ['heparin', 'heparin', 'heparin'],
        })

        result = calculate_hours_since_last(df, 'heparin')

        # At hour 24, last dose was at hour 6, so 18 hours
        row_24 = result[result['hours_from_t0'] == 24]
        assert row_24['hours_since_last'].values[0] == pytest.approx(18, abs=0.1)

    def test_cumulative_exposure(self):
        """Calculate cumulative dose exposure."""
        from transformers.dose_intensity_builder import calculate_cumulative_exposure

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 12, 24],
            'parsed_dose_value': [100, 100, 100],
        })

        result = calculate_cumulative_exposure(df)

        # At hour 24, cumulative should be 300
        assert result.iloc[2]['cumulative_dose'] == 300

    def test_dose_trend(self):
        """Calculate dose trend (increasing/decreasing/stable)."""
        from transformers.dose_intensity_builder import calculate_dose_trend

        # Increasing doses
        df = pd.DataFrame({
            'hours_from_t0': [0, 12, 24],
            'parsed_dose_value': [50, 75, 100],
        })

        trend = calculate_dose_trend(df)

        assert trend == 1  # Increasing


class TestVasopressorFeatures:
    """Test vasopressor-specific features."""

    def test_vasopressor_count(self):
        """Count concurrent vasopressors."""
        from transformers.dose_intensity_builder import count_concurrent_vasopressors

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [1, 1, 1],  # Same time
            'ingredient_name': ['norepinephrine', 'vasopressin', 'dopamine'],
        })

        count = count_concurrent_vasopressors(df, hour=1)

        assert count == 3

    def test_norepinephrine_dose_conversion(self):
        """Convert norepinephrine to mcg/kg/min."""
        from transformers.dose_intensity_builder import convert_norepi_dose

        # 8mg/hr = 8000mcg/60min = 133.3 mcg/min
        # For 70kg patient: 133.3/70 = 1.9 mcg/kg/min
        result = convert_norepi_dose(dose_mg_hr=8, weight_kg=70)

        assert result == pytest.approx(1.9, abs=0.1)
