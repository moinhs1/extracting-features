# tests/test_social_history_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSocialHistoryBuilder:
    def test_builds_all_social_features(self):
        from transformers.social_history_builder import SocialHistoryBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 5,
            'Date': ['1/1/2020'] * 5,
            'Concept_Name': [
                'Smoking Tobacco Use-Current Every Day Smoker',
                'Tobacco Pack Per Day',
                'Alcohol User-Yes',
                'Alcohol Drinks Per Week',
                'Drug User (Illicit)- Never',
            ],
            'Result': ['', '1.5', '', '7', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SocialHistoryBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Smoking
        assert features['smoking_status_at_index'] == 'current_heavy'
        assert features['smoking_pack_per_day_at_index'] == 1.5

        # Alcohol
        assert features['alcohol_status_at_index'] == 'current'
        assert features['alcohol_drinks_per_week_at_index'] == 7.0

        # Drug use
        assert features['drug_use_status_at_index'] == 'never'

        # EMPI
        assert features['empi'] == '100001'

    def test_build_for_cohort_returns_dataframe(self):
        """Test build_for_cohort returns DataFrame with all patients."""
        from transformers.social_history_builder import SocialHistoryBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001', '100002', '100002'],
            'Date': ['1/1/2020'] * 4,
            'Concept_Name': [
                'Smoking Tobacco Use-Never Smoker',
                'Alcohol User-Never',
                'Smoking Tobacco Use-Former Smoker',
                'Alcohol User-Yes',
            ],
            'Result': ['', '', '', ''],
        })
        index_dates = {
            '100001': datetime(2020, 3, 15),
            '100002': datetime(2020, 4, 1),
        }
        builder = SocialHistoryBuilder(data, index_dates)
        df = builder.build_for_cohort(['100001', '100002'])

        assert len(df) == 2
        assert 'empi' in df.columns
        assert 'smoking_status_at_index' in df.columns
        assert 'alcohol_status_at_index' in df.columns
        assert 'drug_use_status_at_index' in df.columns
        assert set(df['empi'].tolist()) == {'100001', '100002'}

    def test_passes_sex_to_alcohol_builder(self):
        """Test that sex parameter is passed to AlcoholBuilder."""
        from transformers.social_history_builder import SocialHistoryBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'] * 2,
            'Date': ['1/1/2020'] * 2,
            'Concept_Name': ['Alcohol User-Yes', 'Alcohol Drinks Per Week'],
            'Result': ['', '10'],  # 10 drinks/week - moderate for men, heavy for women
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SocialHistoryBuilder(data, index_dates)

        # Test with male - 10 drinks/week should be moderate
        features_male = builder.build_all_features('100001', sex='M')
        assert features_male['alcohol_drinking_level'] == 'moderate'

        # Test with female - 10 drinks/week should be heavy
        features_female = builder.build_all_features('100001', sex='F')
        assert features_female['alcohol_drinking_level'] == 'heavy'

    def test_no_data_for_patient_returns_defaults(self):
        """Test handling when patient has no records (but other data exists)."""
        from transformers.social_history_builder import SocialHistoryBuilder
        # Data exists for a different patient
        data = pd.DataFrame({
            'EMPI': ['999999'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Never Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SocialHistoryBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['empi'] == '100001'
        assert features['smoking_status_at_index'] == 'unknown'
        assert features['alcohol_status_at_index'] == 'unknown'
        assert features['drug_use_status_at_index'] == 'unknown'

    def test_removes_duplicate_empi_from_sub_builders(self):
        """Ensure we don't have duplicate empi fields from sub-builders."""
        from transformers.social_history_builder import SocialHistoryBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Never Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SocialHistoryBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Count how many times 'empi' appears as a key - should be exactly 1
        empi_count = list(features.keys()).count('empi')
        assert empi_count == 1
