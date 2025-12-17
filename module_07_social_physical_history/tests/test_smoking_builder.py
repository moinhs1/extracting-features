# tests/test_smoking_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSmokingStatusMapping:
    def test_maps_never_smoker(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Never Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['smoking_status_at_index'] == 'never'

    def test_maps_former_smoker(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Former Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['smoking_status_at_index'] == 'former'

    def test_maps_current_heavy(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Current Every Day Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['smoking_status_at_index'] == 'current_heavy'

    def test_uses_most_recent_status(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2019', '1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Current Every Day Smoker',
                           'Smoking Tobacco Use-Former Smoker'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        # Most recent (2020) shows former
        assert features['smoking_status_at_index'] == 'former'


class TestSmokingEverFlag:
    def test_ever_true_if_any_current(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2010', '1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Current Every Day Smoker',
                           'Smoking Tobacco Use-Former Smoker'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['smoking_ever'] == True

    def test_ever_false_if_only_never(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Smoking Tobacco Use-Never Smoker'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['smoking_ever'] == False


class TestPackYears:
    def test_calculates_pack_years(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Tobacco Pack Per Day', 'Tobacco Used Years'],
            'Result': ['1.5', '20'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001')
        # 1.5 packs/day * 20 years = 30 pack-years
        assert features['smoking_pack_years'] == 30.0

    def test_categorizes_pack_years(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Tobacco Pack Per Day', 'Tobacco Used Years'],
            'Result': ['1', '25'],  # 25 pack-years
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001')
        assert features['smoking_pack_years_category'] == '20-40'

    def test_returns_none_when_missing(self):
        from transformers.smoking_builder import SmokingBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Tobacco Pack Per Day'],
            'Result': ['1.5'],  # Missing years
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = SmokingBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001')
        assert features['smoking_pack_years'] is None
