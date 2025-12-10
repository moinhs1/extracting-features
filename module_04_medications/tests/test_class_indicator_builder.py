# /home/moin/TDA_11_25/module_04_medications/tests/test_class_indicator_builder.py
"""Tests for therapeutic class indicator generation."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIngredientToClass:
    """Test ingredient-to-class mapping."""

    def test_heparin_maps_to_ufh(self):
        """Heparin maps to UFH class."""
        from transformers.class_indicator_builder import get_class_for_ingredient

        classes = get_class_for_ingredient('heparin')

        assert 'ac_ufh_ther' in classes or 'ac_ufh_proph' in classes

    def test_enoxaparin_maps_to_lmwh(self):
        """Enoxaparin maps to LMWH class."""
        from transformers.class_indicator_builder import get_class_for_ingredient

        classes = get_class_for_ingredient('enoxaparin')

        assert 'ac_lmwh_ther' in classes or 'ac_lmwh_proph' in classes

    def test_aspirin_maps_to_antiplatelet(self):
        """Aspirin maps to antiplatelet class."""
        from transformers.class_indicator_builder import get_class_for_ingredient

        classes = get_class_for_ingredient('aspirin')

        assert 'hm_antiplatelet' in classes

    def test_unknown_ingredient_returns_empty(self):
        """Unknown ingredient returns empty list."""
        from transformers.class_indicator_builder import get_class_for_ingredient

        classes = get_class_for_ingredient('notarealdrug')

        assert classes == []


class TestDoseBasedClassification:
    """Test dose-based therapeutic vs prophylactic classification."""

    def test_heparin_therapeutic_dose(self):
        """High-dose heparin is therapeutic."""
        from transformers.class_indicator_builder import classify_anticoagulant_dose

        result = classify_anticoagulant_dose(
            ingredient='heparin',
            dose_value=25000,
            dose_unit='units',
            route='IV'
        )

        assert result == 'ac_ufh_ther'

    def test_heparin_prophylactic_dose(self):
        """Low-dose heparin is prophylactic."""
        from transformers.class_indicator_builder import classify_anticoagulant_dose

        result = classify_anticoagulant_dose(
            ingredient='heparin',
            dose_value=5000,
            dose_unit='units',
            route='SC'
        )

        assert result == 'ac_ufh_proph'

    def test_enoxaparin_therapeutic_dose(self):
        """High-dose enoxaparin is therapeutic."""
        from transformers.class_indicator_builder import classify_anticoagulant_dose

        result = classify_anticoagulant_dose(
            ingredient='enoxaparin',
            dose_value=80,
            dose_unit='mg',
            route='SC'
        )

        assert result == 'ac_lmwh_ther'

    def test_enoxaparin_prophylactic_dose(self):
        """Low-dose enoxaparin is prophylactic."""
        from transformers.class_indicator_builder import classify_anticoagulant_dose

        result = classify_anticoagulant_dose(
            ingredient='enoxaparin',
            dose_value=40,
            dose_unit='mg',
            route='SC'
        )

        assert result == 'ac_lmwh_proph'


class TestTimeWindowAssignment:
    """Test time window assignment."""

    def test_baseline_window(self):
        """Assign to baseline window."""
        from transformers.class_indicator_builder import get_time_window

        assert get_time_window(-50) == 'baseline'
        assert get_time_window(-1) == 'baseline'

    def test_acute_window(self):
        """Assign to acute window."""
        from transformers.class_indicator_builder import get_time_window

        assert get_time_window(0) == 'acute'
        assert get_time_window(12) == 'acute'
        assert get_time_window(23) == 'acute'

    def test_subacute_window(self):
        """Assign to subacute window."""
        from transformers.class_indicator_builder import get_time_window

        assert get_time_window(24) == 'subacute'
        assert get_time_window(48) == 'subacute'

    def test_recovery_window(self):
        """Assign to recovery window."""
        from transformers.class_indicator_builder import get_time_window

        assert get_time_window(72) == 'recovery'
        assert get_time_window(100) == 'recovery'

    def test_outside_windows(self):
        """Outside study window returns None."""
        from transformers.class_indicator_builder import get_time_window

        assert get_time_window(-100) is None  # Before baseline
        assert get_time_window(200) is None   # After recovery


class TestIndicatorAggregation:
    """Test aggregation of indicators per patient-window."""

    def test_aggregate_single_patient(self):
        """Aggregate indicators for single patient."""
        from transformers.class_indicator_builder import aggregate_class_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [1, 2, 25],  # 2 in acute, 1 in subacute
            'class_id': ['ac_ufh_ther', 'ac_ufh_ther', 'ac_lmwh_ther'],
        })

        result = aggregate_class_indicators(df)

        # Check acute window
        acute = result[(result['empi'] == '100') & (result['time_window'] == 'acute')]
        assert len(acute) == 1
        assert acute['ac_ufh_ther'].values[0] == True
        assert acute['ac_ufh_ther_count'].values[0] == 2

        # Check subacute window
        subacute = result[(result['empi'] == '100') & (result['time_window'] == 'subacute')]
        assert len(subacute) == 1
        assert subacute['ac_lmwh_ther'].values[0] == True
