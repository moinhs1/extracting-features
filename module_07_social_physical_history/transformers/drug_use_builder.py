# transformers/drug_use_builder.py
"""
Drug Use Feature Builder
========================

CRITICAL: IVDU (IV Drug Use) is a PERMANENT risk marker.
Once ivdu_ever is True, it must NEVER be set to False.
"""

import pandas as pd
from datetime import datetime
from typing import Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class DrugUseBuilder:
    """Build drug use features for patients."""

    STATUS_MAP = {
        'Drug User (Illicit)- Yes': 'current',
        'Drug User (Illicit)- No': 'no',
        'Drug User (Illicit)- Never': 'never',
        'Drug User (Illicit)- Not Currently': 'former',
        'Drug User (Illicit)- Not Asked': 'not_asked',
    }

    EVER_USED_STATUSES = {'current', 'former'}
    CURRENT_STATUSES = {'current'}

    STALENESS_DAYS = STALENESS_THRESHOLDS['drug_use_status']

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        self.phy_data = phy_data
        self.index_dates = index_dates
        if not phy_data.empty:
            self._preprocess()

    def _preprocess(self):
        self.phy_data = self.phy_data.copy()
        self.phy_data['Date'] = pd.to_datetime(self.phy_data['Date'], errors='coerce')

    def _get_status_records(self, empi: str) -> pd.DataFrame:
        if self.phy_data.empty:
            return pd.DataFrame()
        status_concepts = list(self.STATUS_MAP.keys())
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(status_concepts))
        )
        return self.phy_data[mask].sort_values('Date')

    def build_status_features(self, empi: str) -> Dict:
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'drug_use_status_at_index': 'unknown', 'drug_use_ever': False}

        records = self._get_status_records(empi)
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'drug_use_status_at_index': 'unknown',
                'drug_use_status_date': None,
                'drug_use_status_days_prior': None,
                'drug_use_status_stale': True,
                'drug_use_ever': False,
                'drug_use_current_at_index': False,
            }

        latest = valid.iloc[-1]
        status = self.STATUS_MAP.get(latest['Concept_Name'], 'unknown')
        days_prior = (pd.Timestamp(index_date) - latest['Date']).days

        # Check if ever used drugs
        all_statuses = set(
            self.STATUS_MAP.get(c, 'unknown')
            for c in records['Concept_Name'].unique()
        )
        drug_use_ever = bool(all_statuses & self.EVER_USED_STATUSES)

        return {
            'drug_use_status_at_index': status,
            'drug_use_status_date': latest['Date'],
            'drug_use_status_days_prior': days_prior,
            'drug_use_status_stale': days_prior > self.STALENESS_DAYS,
            'drug_use_ever': drug_use_ever,
            'drug_use_current_at_index': status in self.CURRENT_STATUSES,
        }
