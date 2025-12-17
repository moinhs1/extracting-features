# tests/test_functional_status_builder.py
"""Tests for FunctionalStatusBuilder (KPS)."""
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestKPSPointInTime:
    """Test KPS point-in-time feature extraction."""

    def test_gets_kps_at_index(self):
        """Gets KPS closest to index date."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] == 80

    def test_calculates_days_prior(self):
        """Calculates days before index date."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_days_prior'] == 14  # March 1 to March 15

    def test_stale_after_30_days(self):
        """Marks as stale when beyond 30-day threshold."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['2/1/2020'],  # 43 days before index
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_stale'] == True

    def test_not_stale_within_30_days(self):
        """Not stale when within 30-day threshold."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],  # 14 days before index
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_stale'] == False

    def test_returns_none_when_no_data(self):
        """Returns None when no KPS data."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        builder = FunctionalStatusBuilder(pd.DataFrame(), {'100001': datetime(2020, 3, 15)})
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] is None


class TestKPSCategory:
    """Test KPS category classification."""

    def test_good_functional_status(self):
        """KPS >= 80 is good functional status."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['90'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_category'] == 'good_functional_status'

    def test_moderate_impairment(self):
        """KPS 50-79 is moderate impairment."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['60'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_category'] == 'moderate_impairment'

    def test_severe_impairment(self):
        """KPS < 50 is severe impairment."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['40'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_category'] == 'severe_impairment'


class TestKPSImpairmentFlags:
    """Test KPS impairment flag features."""

    def test_below_70_flagged(self):
        """KPS < 70 flags significant impairment."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['60'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_below_70_at_index'] == True

    def test_below_50_flagged(self):
        """KPS < 50 flags severe impairment."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['40'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_below_50_at_index'] == True

    def test_good_kps_not_flagged(self):
        """KPS >= 70 does not flag impairment."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['90'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_below_70_at_index'] == False
        assert features['kps_below_50_at_index'] == False

    def test_functional_status_impaired(self):
        """functional_status_impaired flag when KPS < 70."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['60'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['functional_status_impaired'] == True

    def test_mobility_impaired(self):
        """mobility_impaired flag when KPS < 70."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['60'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['mobility_impaired'] == True

    def test_bedridden_when_kps_below_30(self):
        """bedridden flag when KPS < 30."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['20'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['bedridden'] == True

    def test_not_bedridden_when_kps_30_or_above(self):
        """Not bedridden when KPS >= 30."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['40'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['bedridden'] == False


class TestKPSDecline:
    """Test KPS decline detection."""

    def test_detects_10pt_decline(self):
        """Detects 10+ point decline in 90 days."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 3,
            'Result': ['90', '80', '70'],  # 20pt decline
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_declined_10pts_90d'] == True

    def test_no_decline_when_stable(self):
        """No decline flag when KPS stable."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 3,
            'Result': ['80', '80', '80'],  # Stable
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_declined_10pts_90d'] == False

    def test_no_decline_when_improving(self):
        """No decline flag when KPS improving."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['1/1/2020', '2/1/2020', '3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 3,
            'Result': ['70', '80', '90'],  # Improving
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_declined_10pts_90d'] == False

    def test_small_decline_not_flagged(self):
        """Decline < 10 points not flagged."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 2,
            'Date': ['1/15/2020', '3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 2,
            'Result': ['80', '75'],  # 5pt decline
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_declined_10pts_90d'] == False

    def test_decline_requires_2_measurements(self):
        """Decline detection requires at least 2 measurements."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['50'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_declined_10pts_90d'] == False


class TestBuildAllFeatures:
    """Test combined feature building."""

    def test_includes_empi(self):
        """build_all_features includes empi."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_all_features('100001')
        assert features['empi'] == '100001'

    def test_includes_all_kps_features(self):
        """build_all_features includes all KPS features."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Check all expected keys present
        expected_keys = [
            'empi', 'kps_at_index', 'kps_date', 'kps_days_prior',
            'kps_stale', 'kps_category', 'kps_below_70_at_index',
            'kps_below_50_at_index', 'kps_declined_10pts_90d',
            'functional_status_impaired', 'mobility_impaired', 'bedridden'
        ]
        for key in expected_keys:
            assert key in features, f"Missing key: {key}"


class TestEdgeCases:
    """Test edge cases and data quality."""

    def test_filters_invalid_kps_values(self):
        """Filters out invalid KPS values (outside 0-100)."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['3/1/2020', '3/5/2020', '3/10/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 3,
            'Result': ['110', '80', '-10'],  # Invalid, valid, invalid
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] == 80  # Only valid value

    def test_handles_non_numeric_results(self):
        """Handles non-numeric result values gracefully."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 2,
            'Date': ['3/1/2020', '3/10/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 2,
            'Result': ['good', '70'],  # Non-numeric, valid
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] == 70

    def test_handles_missing_index_date(self):
        """Handles missing index date gracefully."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['3/1/2020'],
            'Concept_Name': ['KPS (Karnofsky performance status)'],
            'Result': ['80'],
        })
        index_dates = {'100002': datetime(2020, 3, 15)}  # Different EMPI
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] is None

    def test_uses_most_recent_before_index(self):
        """Uses most recent KPS before index date."""
        from transformers.functional_status_builder import FunctionalStatusBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 3,
            'Date': ['2/1/2020', '3/1/2020', '3/20/2020'],  # Last is after index
            'Concept_Name': ['KPS (Karnofsky performance status)'] * 3,
            'Result': ['90', '80', '70'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = FunctionalStatusBuilder(data, index_dates)
        features = builder.build_kps_features('100001')
        assert features['kps_at_index'] == 80  # March 1, not March 20
