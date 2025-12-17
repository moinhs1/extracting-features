# tests/test_height_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHeightFeatures:
    def test_gets_height_in_meters(self):
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['70'],  # 70 inches
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # 70 inches = 177.8 cm = 1.778 m
        assert features['height_m_at_index'] == pytest.approx(1.778, rel=0.01)

    def test_height_staleness_10_years(self):
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2005'],  # 15 years old
            'Concept_Name': ['Height'],
            'Result': ['70'],
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_at_index_stale'] == True


class TestHeightConversions:
    def test_auto_detect_inches_when_value_less_than_100(self):
        """If value < 100, assume inches and convert to meters."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['65'],  # 65 inches (auto-detected)
            'Units': [''],  # No units specified
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # 65 inches = 165.1 cm = 1.651 m
        assert features['height_m_at_index'] == pytest.approx(1.651, rel=0.01)

    def test_auto_detect_cm_when_value_100_or_more(self):
        """If value >= 100, assume cm and convert to meters."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['175'],  # 175 cm (auto-detected)
            'Units': [''],  # No units specified
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # 175 cm = 1.75 m
        assert features['height_m_at_index'] == pytest.approx(1.75, rel=0.01)

    def test_returns_height_in_cm(self):
        """Should return height_cm_at_index."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['70'],  # 70 inches
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # 70 inches = 177.8 cm
        assert features['height_cm_at_index'] == pytest.approx(177.8, rel=0.01)

    def test_returns_height_in_inches(self):
        """Should return height_in_at_index."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['70'],  # 70 inches
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_in_at_index'] == pytest.approx(70, rel=0.01)


class TestHeightDaysPrior:
    def test_returns_days_prior(self):
        """Should return height_at_index_days_prior."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['2020-03-01'],
            'Concept_Name': ['Height'],
            'Result': ['70'],
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # 14 days between March 1 and March 15
        assert features['height_at_index_days_prior'] == 14


class TestHeightEdgeCases:
    def test_no_height_data(self):
        """Should handle missing height data gracefully."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Weight'],  # Not Height
            'Result': ['180'],
            'Units': ['lbs'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_m_at_index'] is None
        assert features['height_at_index_stale'] == True

    def test_no_index_date(self):
        """Should handle missing index date."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Height'],
            'Result': ['70'],
            'Units': ['in'],
        })
        index_dates = {}  # No index dates
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_m_at_index'] is None

    def test_empty_dataframe(self):
        """Should handle empty dataframe."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame()
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_m_at_index'] is None

    def test_gets_most_recent_height_before_index(self):
        """Should use the most recent height measurement before index date."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '4/1/2020'],  # Last is after index
            'Concept_Name': ['Height'] * 3,
            'Result': ['68', '70', '72'],
            'Units': ['in', 'in', 'in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # Should use the Feb 1 measurement (70 inches), not April 1
        assert features['height_m_at_index'] == pytest.approx(1.778, rel=0.01)

    def test_filters_out_unreasonable_values(self):
        """Should filter out heights < 20 or >= 300."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 2,
            'Date': ['1/1/2020', '2/1/2020'],
            'Concept_Name': ['Height'] * 2,
            'Result': ['5', '70'],  # 5 is unreasonable
            'Units': ['in', 'in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        # Should use 70, not 5
        assert features['height_m_at_index'] == pytest.approx(1.778, rel=0.01)


class TestHeightStaleness:
    def test_not_stale_within_10_years(self):
        """Height within 10 years should not be stale."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2015'],  # 5 years old
            'Concept_Name': ['Height'],
            'Result': ['70'],
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_at_index_stale'] == False

    def test_stale_after_10_years(self):
        """Height older than 10 years should be stale."""
        from transformers.height_builder import HeightBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2009'],  # 11+ years old
            'Concept_Name': ['Height'],
            'Result': ['70'],
            'Units': ['in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = HeightBuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['height_at_index_stale'] == True
