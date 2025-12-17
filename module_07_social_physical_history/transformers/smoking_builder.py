# transformers/smoking_builder.py
"""Smoking/Tobacco Feature Builder."""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class SmokingBuilder:
    """Build smoking/tobacco features for patients."""

    STATUS_MAP = {
        'Smoking Tobacco Use-Never Smoker': 'never',
        'Smoking Tobacco Use-Former Smoker': 'former',
        'Smoking Tobacco Use-Current Every Day Smoker': 'current_heavy',
        'Smoking Tobacco Use-Current Some Day Smoker': 'current_light',
        'Smoking Tobacco Use-Heavy Tobacco Smoker': 'current_heavy',
        'Smoking Tobacco Use-Light Tobacco Smoker': 'current_light',
        'Smoking Tobacco Use-Passive Smoke Exposure - Never Smoker': 'never_passive',
        'Smoking Tobacco Use-Smoker, Current Status Unknown': 'unknown_current',
        'Smoking Tobacco Use-Unknown if Ever Smoked': 'unknown',
        'Smoking Tobacco Use-Never Assessed': 'not_assessed',
        'Tobacco User-Never': 'never',
        'Tobacco User-Quit': 'former',
        'Tobacco User-Yes': 'current',
        'Tobacco User-Passive': 'never_passive',
        'Tobacco User-Not Asked': 'not_asked',
    }

    CURRENT_STATUSES = {'current', 'current_heavy', 'current_light'}
    EVER_SMOKED_STATUSES = {'current', 'current_heavy', 'current_light', 'former'}

    STALENESS_DAYS = STALENESS_THRESHOLDS['smoking_status']

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
            return {'smoking_status_at_index': 'unknown', 'smoking_ever': False}

        records = self._get_status_records(empi)
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'smoking_status_at_index': 'unknown',
                'smoking_status_date': None,
                'smoking_status_days_prior': None,
                'smoking_status_stale': True,
                'smoking_ever': False,
                'smoking_current_at_index': False,
                'smoking_former_at_index': False,
            }

        # Get most recent status
        latest = valid.iloc[-1]
        status = self.STATUS_MAP.get(latest['Concept_Name'], 'unknown')
        days_prior = (pd.Timestamp(index_date) - latest['Date']).days

        # Check if ever smoked (any record shows smoking)
        all_statuses = set(
            self.STATUS_MAP.get(c, 'unknown')
            for c in records['Concept_Name'].unique()
        )
        smoking_ever = bool(all_statuses & self.EVER_SMOKED_STATUSES)

        return {
            'smoking_status_at_index': status,
            'smoking_status_date': latest['Date'],
            'smoking_status_days_prior': days_prior,
            'smoking_status_stale': days_prior > self.STALENESS_DAYS,
            'smoking_ever': smoking_ever,
            'smoking_current_at_index': status in self.CURRENT_STATUSES,
            'smoking_former_at_index': status == 'former',
        }
