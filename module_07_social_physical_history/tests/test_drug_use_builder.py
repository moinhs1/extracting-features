# tests/test_drug_use_builder.py
import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDrugUseStatusMapping:
    def test_maps_never(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Never'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['drug_use_status_at_index'] == 'never'

    def test_maps_current(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Yes'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['drug_use_status_at_index'] == 'current'

    def test_maps_former(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Not Currently'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['drug_use_status_at_index'] == 'former'


class TestDrugUseEverFlag:
    def test_ever_true_if_any_use(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2019', '1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Yes', 'Drug User (Illicit)- Not Currently'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['drug_use_ever'] == True

    def test_ever_false_if_only_never(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Never'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_status_features('100001')
        assert features['drug_use_ever'] == False
