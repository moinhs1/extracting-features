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


class TestIVDUPermanentFlag:
    """
    CRITICAL: IVDU is a PERMANENT risk marker.
    Once True, it must NEVER be set to False.
    """

    def test_ivdu_ever_true_if_any_record(self):
        """If ANY record shows IV drug use, ivdu_ever = True."""
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2015'],  # Old record
            'Concept_Name': ['Drug User IV'],
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_ivdu_features('100001')
        assert features['ivdu_ever'] == True

    def test_ivdu_ever_never_becomes_false(self):
        """
        CRITICAL: Even if later records say 'No', ivdu_ever stays True.
        This is because IV drug use history is a permanent VTE risk factor.
        """
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2015', '1/1/2020'],
            'Concept_Name': ['Drug User IV', 'Drug User (Illicit)- No'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_ivdu_features('100001')
        # Even though current status is 'No', IVDU ever remains True
        assert features['ivdu_ever'] == True

    def test_ivdu_ever_false_if_no_iv_records(self):
        """ivdu_ever is False only if NO IV drug use records exist."""
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001'],
            'Date': ['1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Yes'],  # Non-IV drug use
            'Result': [''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_ivdu_features('100001')
        assert features['ivdu_ever'] == False

    def test_ivdu_staleness_is_infinity(self):
        """IVDU flag never becomes stale - it's permanent."""
        from config.social_physical_config import STALENESS_THRESHOLDS
        assert STALENESS_THRESHOLDS['ivdu'] == float('inf')


class TestBuildAllDrugUseFeatures:
    def test_combines_status_and_ivdu(self):
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2020', '1/1/2020'],
            'Concept_Name': ['Drug User (Illicit)- Yes', 'Drug User IV'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        assert features['drug_use_status_at_index'] == 'current'
        assert features['drug_use_ever'] == True
        assert features['ivdu_ever'] == True
        assert features['empi'] == '100001'

    def test_ivdu_true_even_with_current_no_status(self):
        """IVDU ever stays True even if current drug status is 'No'."""
        from transformers.drug_use_builder import DrugUseBuilder
        data = pd.DataFrame({
            'EMPI': ['100001', '100001'],
            'Date': ['1/1/2010', '1/1/2020'],
            'Concept_Name': ['Drug User IV', 'Drug User (Illicit)- No'],
            'Result': ['', ''],
        })
        index_dates = {'100001': datetime(2020, 3, 15)}
        builder = DrugUseBuilder(data, index_dates)
        features = builder.build_all_features('100001')

        # Current status is 'no' but IVDU ever remains True
        assert features['drug_use_status_at_index'] == 'no'
        assert features['ivdu_ever'] == True
