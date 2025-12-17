# tests/test_bsa_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBSAFeatures:
    def test_gets_measured_bsa(self):
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Body Surface Area (BSA)'],
            'Result': ['1.85'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index'] == 1.85
        assert features['bsa_method'] == 'measured'

    def test_calculates_bsa_when_missing(self):
        from transformers.bsa_builder import BSABuilder
        # No BSA records, but weight and height available
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Weight', 'Height'],
            'Result': ['154', '70'],  # 70kg, 70in=177.8cm
            'Units': ['lbs', 'in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        # Should calculate BSA from weight/height
        assert features['bsa_at_index'] is not None
        assert features['bsa_method'] == 'dubois'

    def test_bsa_staleness_180_days(self):
        """BSA older than 180 days should be marked stale."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2019'],  # More than 180 days before index
            'Concept_Name': ['Body Surface Area (BSA)'],
            'Result': ['1.85'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index_stale'] == True

    def test_bsa_not_stale_within_180_days(self):
        """BSA within 180 days should not be marked stale."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],  # ~75 days before index
            'Concept_Name': ['Body Surface Area (BSA)'],
            'Result': ['1.85'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index_stale'] == False

    def test_returns_bsa_date_and_days_prior(self):
        """Should return date and days_prior for measured BSA."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Body Surface Area (BSA)'],
            'Result': ['1.85'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index_date'] is not None
        assert features['bsa_at_index_days_prior'] == 74  # Jan 1 to Mar 15

    def test_no_bsa_no_weight_height_returns_none(self):
        """Should return None if no BSA and no weight/height to calculate."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['SomeOtherConcept'],
            'Result': ['123'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index'] is None
        assert features['bsa_method'] is None

    def test_calculated_bsa_value_reasonable(self):
        """Calculated BSA should be in reasonable range."""
        from transformers.bsa_builder import BSABuilder
        # 154 lbs = ~69.85 kg, 70 in = 177.8 cm
        # DuBois: 0.007184 * (69.85)^0.425 * (177.8)^0.725 ~ 1.87 m^2
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Weight', 'Height'],
            'Result': ['154', '70'],
            'Units': ['lbs', 'in'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        # Should be approximately 1.87 m^2
        assert 1.5 < features['bsa_at_index'] < 2.5

    def test_empty_data_returns_none(self):
        """Empty data should return None BSA."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame(columns=['EMPI', 'Date', 'Concept_Name', 'Result'])
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index'] is None

    def test_no_index_date_returns_none(self):
        """Missing index date should return None."""
        from transformers.bsa_builder import BSABuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Body Surface Area (BSA)'],
            'Result': ['1.85'],
        })
        index_dates = {}  # No index date for this patient
        builder = BSABuilder(data, index_dates)
        features = builder.build_features('100001')
        assert features['bsa_at_index'] is None
