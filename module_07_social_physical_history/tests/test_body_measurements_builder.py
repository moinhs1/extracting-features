# tests/test_body_measurements_builder.py
"""Tests for BodyMeasurementsBuilder composite."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBodyMeasurementsBuilder:
    """Tests for the composite builder."""

    @pytest.fixture
    def sample_data(self):
        """Sample data with all body measurement types."""
        data = pd.DataFrame({
            'EMPI': ['100001'] * 4,
            'Date': ['1/1/2020'] * 4,
            'Concept_Name': ['BMI', 'Weight', 'Height', 'Body Surface Area (BSA)'],
            'Result': ['28.5', '185', '70', '1.95'],
            'Units': ['kg/m2', 'lbs', 'in', 'm2'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        return data, index_dates

    def test_builds_all_body_features(self, sample_data):
        """Test that composite builder includes features from all sub-builders."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data, index_dates = sample_data
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Check that features from each sub-builder are present
        assert 'bmi_at_index' in features
        assert 'weight_kg_at_index' in features
        assert 'height_m_at_index' in features
        assert 'bsa_at_index' in features
        assert features['empi'] == '100001'

    def test_bmi_features_present(self, sample_data):
        """Test that BMI features are included."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data, index_dates = sample_data
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['bmi_at_index'] == 28.5
        assert 'bmi_category_at_index' in features

    def test_weight_features_present(self, sample_data):
        """Test that weight features are included."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data, index_dates = sample_data
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['weight_lbs_at_index'] == 185.0
        # 185 lbs = ~83.91 kg
        assert features['weight_kg_at_index'] == pytest.approx(83.91, rel=0.01)

    def test_height_features_present(self, sample_data):
        """Test that height features are included."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data, index_dates = sample_data
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # 70 inches = 177.8 cm = 1.778 m
        assert features['height_m_at_index'] == pytest.approx(1.778, rel=0.01)

    def test_bsa_features_present(self, sample_data):
        """Test that BSA features are included."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data, index_dates = sample_data
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['bsa_at_index'] == 1.95
        assert features['bsa_method'] == 'measured'


class TestBodyMeasurementsBuilderCohort:
    """Tests for cohort processing."""

    def test_build_for_cohort_returns_dataframe(self):
        """Test that build_for_cohort returns a DataFrame with all features."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001', '100002', '100002'],
            'Date': ['1/1/2020', '1/1/2020', '1/1/2020', '1/1/2020'],
            'Concept_Name': ['BMI', 'Weight', 'BMI', 'Weight'],
            'Result': ['28.5', '185', '24.0', '160'],
            'Units': ['kg/m2', 'lbs', 'kg/m2', 'lbs'],
        })
        index_dates = {
            '100001': datetime(2020, 3, 15),
            '100002': datetime(2020, 3, 15),
        }
        builder = BodyMeasurementsBuilder(data, index_dates)
        df = builder.build_for_cohort(['100001', '100002'])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'empi' in df.columns
        assert 'bmi_at_index' in df.columns
        assert 'weight_kg_at_index' in df.columns
        assert list(df['empi']) == ['100001', '100002']

    def test_cohort_handles_missing_patient(self):
        """Test that cohort processing handles patients with no data."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['BMI'],
            'Result': ['28.5'],
            'Units': ['kg/m2'],
        })
        index_dates = {
            '100001': datetime(2020, 3, 15),
            '100002': datetime(2020, 3, 15),  # No data for this patient
        }
        builder = BodyMeasurementsBuilder(data, index_dates)
        df = builder.build_for_cohort(['100001', '100002'])

        assert len(df) == 2
        # Patient with data should have value
        assert df[df['empi'] == '100001']['bmi_at_index'].iloc[0] == 28.5
        # Patient without data should have None
        assert pd.isna(df[df['empi'] == '100002']['bmi_at_index'].iloc[0])


class TestBodyMeasurementsBuilderEmptyData:
    """Tests for handling empty data."""

    def test_handles_patient_with_no_data(self):
        """Test that builder handles patient with no measurements gracefully."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        # Data exists for a different patient
        data = pd.DataFrame({
            'EMPI': ['999999'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['BMI'],
            'Result': ['25.0'],
            'Units': ['kg/m2'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['empi'] == '100001'
        assert features.get('bmi_at_index') is None
        assert features.get('weight_kg_at_index') is None
        assert features.get('height_m_at_index') is None
        assert features.get('bsa_at_index') is None

    def test_handles_missing_index_date(self):
        """Test that builder handles missing index date."""
        from transformers.body_measurements_builder import BodyMeasurementsBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['BMI'],
            'Result': ['28.5'],
            'Units': ['kg/m2'],
        })
        index_dates = {}  # No index dates
        builder = BodyMeasurementsBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['empi'] == '100001'
        # Should have None values when no index date
        assert features.get('bmi_at_index') is None
