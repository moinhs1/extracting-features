# transformers/social_history_builder.py
"""Composite Social History Builder."""

import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers.smoking_builder import SmokingBuilder
from transformers.alcohol_builder import AlcoholBuilder
from transformers.drug_use_builder import DrugUseBuilder


class SocialHistoryBuilder:
    """Composite builder for all social history features."""

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        self.phy_data = phy_data
        self.index_dates = index_dates

        self.smoking_builder = SmokingBuilder(phy_data, index_dates)
        self.alcohol_builder = AlcoholBuilder(phy_data, index_dates)
        self.drug_use_builder = DrugUseBuilder(phy_data, index_dates)

    def build_all_features(self, empi: str, sex: str = None) -> Dict:
        """Build all social history features for a patient."""
        features = {'empi': empi}

        # Smoking
        smoking = self.smoking_builder.build_all_features(empi)
        smoking.pop('empi', None)
        features.update(smoking)

        # Alcohol
        alcohol = self.alcohol_builder.build_all_features(empi, sex)
        alcohol.pop('empi', None)
        features.update(alcohol)

        # Drug use
        drug = self.drug_use_builder.build_all_features(empi)
        drug.pop('empi', None)
        features.update(drug)

        return features

    def build_for_cohort(self, empis: List[str]) -> pd.DataFrame:
        """Build social history features for entire cohort."""
        all_features = []
        for empi in empis:
            features = self.build_all_features(empi)
            all_features.append(features)
        return pd.DataFrame(all_features)
