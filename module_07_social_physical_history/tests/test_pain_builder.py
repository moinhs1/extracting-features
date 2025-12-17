# tests/test_pain_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPainPointInTime:
    def test_gets_pain_at_index(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['6'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] == 6.0

    def test_stale_after_7_days(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],  # 14 days before index
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['6'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_stale'] == True

    def test_not_stale_within_7_days(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],  # 5 days before index
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['6'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_stale'] == False

    def test_returns_none_when_no_data(self):
        from transformers.pain_builder import PainBuilder
        builder = PainBuilder(pd.DataFrame(), {'100001': datetime(2020, 3, 15)})
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] is None
        assert features['pain_score_stale'] == True

    def test_gets_most_recent_pain(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/8/2020', '3/10/2020', '3/12/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['4', '6', '8'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] == 8.0  # Most recent is 3/12


class TestPainWindowAggregates:
    def test_7d_max(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['4', '8', '5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_window_features('100001')
        assert features['pain_7d_max'] == 8.0

    def test_7d_min(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['4', '8', '5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_window_features('100001')
        assert features['pain_7d_min'] == 4.0

    def test_7d_count(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['4', '8', '5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_window_features('100001')
        assert features['pain_7d_count'] == 3

    def test_30d_mean(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 4,
            'Date': ['2/20/2020', '3/1/2020', '3/10/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 4,
            'Result': ['2', '4', '6', '8'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_window_features('100001')
        assert features['pain_30d_mean'] == pytest.approx(5.0)

    def test_empty_window_returns_none(self):
        from transformers.pain_builder import PainBuilder
        builder = PainBuilder(pd.DataFrame(), {'100001': datetime(2020, 3, 15)})
        features = builder.build_window_features('100001')
        assert features['pain_7d_mean'] is None
        assert features['pain_7d_count'] == 0


class TestPainSeverityFlags:
    def test_severe_at_index(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['8'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['pain_severe_at_index'] == True

    def test_moderate_at_index(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['pain_moderate_at_index'] == True
        assert features['pain_severe_at_index'] == False

    def test_severe_any_7d(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['3', '9', '4'],  # One severe score
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['pain_severe_any_7d'] == True

    def test_severe_any_30d(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['2/20/2020', '3/1/2020', '3/14/2020'],  # 30d spread
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['8', '3', '4'],  # One severe score on 2/20
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['pain_severe_any_30d'] == True

    def test_not_severe_when_all_low(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['2', '3', '4'],  # All below severe threshold
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['pain_severe_any_7d'] == False


class TestPainConceptNames:
    """Test that all pain concept names are recognized."""

    def test_pain_score_epic(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] == 5.0

    def test_pain_level(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],
            'Concept_Name': ['Pain Level (0-10)'],
            'Result': ['6'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] == 6.0

    def test_vas_score(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],
            'Concept_Name': ['VAS score'],
            'Result': ['7'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_point_in_time('100001')
        assert features['pain_score_at_index'] == 7.0


class TestPainBuildAllFeatures:
    def test_includes_empi(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/10/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'],
            'Result': ['5'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['empi'] == '100001'

    def test_includes_all_feature_types(self):
        from transformers.pain_builder import PainBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/10/2020', '3/12/2020', '3/14/2020'],
            'Concept_Name': ['Pain Score EPIC (0-10)'] * 3,
            'Result': ['4', '6', '8'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = PainBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Point-in-time
        assert 'pain_score_at_index' in features
        # Window
        assert 'pain_7d_mean' in features
        assert 'pain_30d_mean' in features
        # Severity
        assert 'pain_severe_at_index' in features
        assert 'pain_moderate_at_index' in features
        assert 'pain_severe_any_7d' in features
        assert 'pain_severe_any_30d' in features
