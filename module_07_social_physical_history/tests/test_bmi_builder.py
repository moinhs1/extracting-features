# tests/test_bmi_builder.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBMIClassification:
    """Test BMI category classification."""

    def test_underweight(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(17.0) == 'underweight'

    def test_normal(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(22.0) == 'normal'

    def test_overweight(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(27.0) == 'overweight'

    def test_obese_1(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(32.0) == 'obese_1'

    def test_obese_2(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(37.0) == 'obese_2'

    def test_obese_3(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(42.0) == 'obese_3'

    def test_none_returns_unknown(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(None) == 'unknown'


class TestTrendClassification:
    """Test trend direction classification."""

    def test_increasing(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(10.0) == 'increasing'

    def test_decreasing(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(-10.0) == 'decreasing'

    def test_stable(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(2.0) == 'stable'

    def test_none_returns_unknown(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(None) == 'unknown'


class TestPointInTime:
    """Test point-in-time feature extraction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample BMI data."""
        data = pd.DataFrame({
            'EMPI': ['100001'] * 4,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020', '4/1/2020'],
            'Concept_Name': ['BMI'] * 4,
            'Result': ['28.0', '29.0', '30.0', '31.0'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        return data, index_dates

    def test_gets_closest_value(self, sample_data):
        """Gets BMI closest to index date."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = sample_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_point_in_time('100001')

        assert features['bmi_at_index'] == 30.0  # March 1 is closest to March 15

    def test_calculates_days_prior(self, sample_data):
        """Calculates days before index."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = sample_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_point_in_time('100001')

        assert features['bmi_at_index_days_prior'] == 14  # March 1 to March 15

    def test_marks_stale_when_old(self):
        """Marks as stale when beyond threshold."""
        from transformers.bmi_builder import BMIBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2019'],  # Very old
            'Concept_Name': ['BMI'],
            'Result': ['28.0'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BMIBuilder(data, index_dates)

        features = builder.build_point_in_time('100001')

        assert features['bmi_at_index_stale'] == True

    def test_returns_none_when_no_data(self):
        """Returns None when no BMI data."""
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {'100001': datetime(2020, 3, 15)})

        features = builder.build_point_in_time('100001')

        assert features['bmi_at_index'] is None


class TestWindowAggregates:
    """Test window aggregate features."""

    @pytest.fixture
    def multi_measurement_data(self):
        """Create data with multiple measurements."""
        data = pd.DataFrame({
            'EMPI': ['100001'] * 5,
            'Date': ['1/1/2020', '1/15/2020', '2/1/2020', '2/15/2020', '3/1/2020'],
            'Concept_Name': ['BMI'] * 5,
            'Result': ['28.0', '29.0', '30.0', '31.0', '32.0'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        return data, index_dates

    def test_90d_mean(self, multi_measurement_data):
        """Calculates 90-day mean correctly."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = multi_measurement_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_window_features('100001')

        # All 5 values within 90 days of March 15
        expected_mean = (28.0 + 29.0 + 30.0 + 31.0 + 32.0) / 5
        assert features['bmi_90d_mean'] == pytest.approx(expected_mean)

    def test_90d_min_max(self, multi_measurement_data):
        """Calculates 90-day min and max."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = multi_measurement_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_window_features('100001')

        assert features['bmi_90d_min'] == 28.0
        assert features['bmi_90d_max'] == 32.0

    def test_90d_count(self, multi_measurement_data):
        """Counts measurements in window."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = multi_measurement_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_window_features('100001')

        assert features['bmi_90d_count'] == 5

    def test_empty_window_returns_none(self):
        """Returns None for empty windows."""
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {'100001': datetime(2020, 3, 15)})

        features = builder.build_window_features('100001')

        assert features['bmi_90d_mean'] is None
        assert features['bmi_90d_count'] == 0


class TestTrendFeatures:
    """Test trend calculation features."""

    @pytest.fixture
    def increasing_data(self):
        """Data with increasing BMI trend."""
        data = pd.DataFrame({
            'EMPI': ['100001'] * 4,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020', '4/1/2020'],
            'Concept_Name': ['BMI'] * 4,
            'Result': ['25.0', '27.0', '29.0', '31.0'],  # +24% change
        })
        index_dates = {'100001': datetime(2020, 4, 15)}
        return data, index_dates

    def test_calculates_pct_change(self, increasing_data):
        """Calculates percent change correctly."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = increasing_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_trend_features('100001')

        # (31 - 25) / 25 * 100 = 24%
        assert features['bmi_6mo_pct_change'] == pytest.approx(24.0)

    def test_classifies_increasing_trend(self, increasing_data):
        """Classifies increasing trend."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = increasing_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_trend_features('100001')

        assert features['bmi_trend'] == 'increasing'

    def test_detects_became_obese(self, increasing_data):
        """Detects crossing obesity threshold."""
        from transformers.bmi_builder import BMIBuilder
        data, index_dates = increasing_data
        builder = BMIBuilder(data, index_dates)

        features = builder.build_trend_features('100001')

        # Started at 25 (overweight), ended at 31 (obese_1)
        assert features['bmi_became_obese_1yr'] == True

    def test_insufficient_data_returns_unknown(self):
        """Returns unknown when insufficient data."""
        from transformers.bmi_builder import BMIBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['BMI'],
            'Result': ['28.0'],
        })
        index_dates = {'100001': datetime(2020, 4, 15)}
        builder = BMIBuilder(data, index_dates)

        features = builder.build_trend_features('100001')

        assert features['bmi_trend'] == 'unknown'
