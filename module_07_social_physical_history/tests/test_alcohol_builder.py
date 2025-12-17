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
