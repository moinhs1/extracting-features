# tests/test_weight_builder.py
"""Tests for WeightBuilder."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeightPointInTime:
    @pytest.fixture
    def sample_data(self):
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['180', '185', '190'],
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        return data, index_dates

    def test_gets_weight_at_index(self, sample_data):
        from transformers.weight_builder import WeightBuilder
        data, index_dates = sample_data
        builder = WeightBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        # 190 lbs closest to March 15
        assert features['weight_lbs_at_index'] == 190.0

    def test_converts_to_kg(self, sample_data):
        from transformers.weight_builder import WeightBuilder
        data, index_dates = sample_data
        builder = WeightBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        # 190 lbs = ~86.18 kg
        assert features['weight_kg_at_index'] == pytest.approx(86.18, rel=0.01)

    def test_calculates_days_prior(self, sample_data):
        from transformers.weight_builder import WeightBuilder
        data, index_dates = sample_data
        builder = WeightBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        # March 1 to March 15 = 14 days
        assert features['weight_at_index_days_prior'] == 14

    def test_marks_stale_when_old(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2019'],  # Over 1 year ago
            'Concept_Name': ['Weight'],
            'Result': ['180'],
            'Units': ['lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['weight_at_index_stale'] == True

    def test_handles_missing_patient(self, sample_data):
        from transformers.weight_builder import WeightBuilder
        data, index_dates = sample_data
        builder = WeightBuilder(data, index_dates)
        features = builder.build_point_in_time('999999')
        assert features['weight_kg_at_index'] is None
        assert features['weight_lbs_at_index'] is None


class TestWeightLossDetection:
    def test_detects_5pct_loss_90d(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['200', '195', '180'],  # 10% loss
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_trend_features('100001')
        assert features['weight_loss_5pct_90d'] == True

    def test_no_loss_when_stable(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['180', '182', '181'],  # Stable
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_trend_features('100001')
        assert features['weight_loss_5pct_90d'] == False

    def test_detects_10pct_loss_6mo(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['10/1/2019', '12/1/2019', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['200', '190', '175'],  # 12.5% loss
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_trend_features('100001')
        assert features['weight_loss_10pct_6mo'] == True

    def test_detects_5pct_gain_90d(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['180', '185', '200'],  # 11% gain
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_trend_features('100001')
        assert features['weight_gain_5pct_90d'] == True


class TestWeightAllFeatures:
    def test_build_all_features_combines(self):
        from transformers.weight_builder import WeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['Weight'] * 3,
            'Result': ['180', '185', '190'],
            'Units': ['lbs', 'lbs', 'lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = WeightBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Should have point-in-time features
        assert 'weight_kg_at_index' in features
        assert 'weight_lbs_at_index' in features
        # Should have trend features
        assert 'weight_loss_5pct_90d' in features
        assert 'weight_loss_10pct_6mo' in features
        assert 'weight_gain_5pct_90d' in features
        # Should have empi
        assert features['empi'] == '100001'
