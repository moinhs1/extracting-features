# tests/test_alcohol_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAlcoholStatusMapping:
    def test_maps_never(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol User-Never'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['alcohol_status_at_index'] == 'never'

    def test_maps_current(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol User-Yes'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['alcohol_status_at_index'] == 'current'

    def test_maps_former(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol User-Not Currently'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['alcohol_status_at_index'] == 'former'


class TestAlcoholEverFlag:
    def test_ever_true_if_any_use(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2019', '1/1/2020'],
            'Concept_Name': ['Alcohol User-Yes', 'Alcohol User-Not Currently'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['alcohol_ever'] == True

    def test_ever_false_if_only_never(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol User-Never'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['alcohol_ever'] == False


class TestDrinksPerWeek:
    def test_extracts_drinks_per_week(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol Drinks Per Week'],
            'Result': ['14'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001')
        assert features['alcohol_drinks_per_week_at_index'] == 14.0

    def test_extracts_oz_per_week(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol Oz Per Week'],
            'Result': ['8'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001')
        assert features['alcohol_oz_per_week_at_index'] == 8.0


class TestHeavyDrinkingClassification:
    def test_heavy_male_over_14(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol Drinks Per Week'],
            'Result': ['18'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001', sex='M')
        assert features['alcohol_heavy_use'] == True

    def test_not_heavy_male_under_14(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol Drinks Per Week'],
            'Result': ['10'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001', sex='M')
        assert features['alcohol_heavy_use'] == False

    def test_heavy_female_over_7(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Alcohol Drinks Per Week'],
            'Result': ['10'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_quantitative_features('100001', sex='F')
        assert features['alcohol_heavy_use'] == True


class TestBuildAllAlcoholFeatures:
    def test_combines_all_feature_types(self):
        from transformers.alcohol_builder import AlcoholBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Alcohol User-Yes', 'Alcohol Drinks Per Week'],
            'Result': ['', '10'],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = AlcoholBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['alcohol_status_at_index'] == 'current'
        assert features['alcohol_drinks_per_week_at_index'] == 10.0
        assert features['empi'] == '100001'
