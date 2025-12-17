# transformers/alcohol_builder.py
"""Alcohol Use Feature Builder."""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class AlcoholBuilder:
    """Build alcohol use features for patients."""

    STATUS_MAP = {
        'Alcohol User-Yes': 'current',
        'Alcohol User-No': 'no',
        'Alcohol User-Never': 'never',
        'Alcohol User-Not Currently': 'former',
        'Alcohol User-Not Asked': 'not_asked',
    }

    EVER_USED_STATUSES = {'current', 'former', 'no'}  # 'no' means they drink but answered no to abuse
    CURRENT_STATUSES = {'current'}

    STALENESS_DAYS = STALENESS_THRESHOLDS['alcohol_status']

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
            return {'alcohol_status_at_index': 'unknown', 'alcohol_ever': False}

        records = self._get_status_records(empi)
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'alcohol_status_at_index': 'unknown',
                'alcohol_status_date': None,
                'alcohol_status_days_prior': None,
                'alcohol_status_stale': True,
                'alcohol_ever': False,
                'alcohol_current_at_index': False,
            }

        latest = valid.iloc[-1]
        status = self.STATUS_MAP.get(latest['Concept_Name'], 'unknown')
        days_prior = (pd.Timestamp(index_date) - latest['Date']).days

        # Check if ever used alcohol
        all_statuses = set(
            self.STATUS_MAP.get(c, 'unknown')
            for c in records['Concept_Name'].unique()
        )
        alcohol_ever = bool(all_statuses & self.EVER_USED_STATUSES)

        return {
            'alcohol_status_at_index': status,
            'alcohol_status_date': latest['Date'],
            'alcohol_status_days_prior': days_prior,
            'alcohol_status_stale': days_prior > self.STALENESS_DAYS,
            'alcohol_ever': alcohol_ever,
            'alcohol_current_at_index': status in self.CURRENT_STATUSES,
        }

    def _get_numeric_value(self, empi: str, concept: str, index_date: datetime) -> Optional[float]:
        """Get most recent numeric value for a concept."""
        if self.phy_data.empty:
            return None
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'] == concept) &
            (self.phy_data['Date'] <= pd.Timestamp(index_date))
        )
        records = self.phy_data[mask].sort_values('Date')
        if records.empty:
            return None
        value = pd.to_numeric(records.iloc[-1]['Result'], errors='coerce')
        return value if pd.notna(value) else None

    def classify_drinking_level(self, drinks_per_week: Optional[float], sex: str = None) -> str:
        """
        Classify alcohol use level using CDC/NIAAA thresholds.

        Heavy drinking:
        - Men: >14 drinks/week
        - Women: >7 drinks/week
        """
        if drinks_per_week is None:
            return 'unknown'

        # Use female thresholds if sex unknown (more conservative)
        if sex == 'M':
            if drinks_per_week == 0:
                return 'none'
            if drinks_per_week <= 7:
                return 'light'
            if drinks_per_week <= 14:
                return 'moderate'
            return 'heavy'
        else:  # Female or unknown
            if drinks_per_week == 0:
                return 'none'
            if drinks_per_week <= 3:
                return 'light'
            if drinks_per_week <= 7:
                return 'moderate'
            return 'heavy'

    def build_quantitative_features(self, empi: str, sex: str = None) -> Dict:
        """Build quantitative alcohol features for a patient."""
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'alcohol_drinks_per_week_at_index': None}

        drinks_per_week = self._get_numeric_value(empi, 'Alcohol Drinks Per Week', index_date)
        oz_per_week = self._get_numeric_value(empi, 'Alcohol Oz Per Week', index_date)

        drinking_level = self.classify_drinking_level(drinks_per_week, sex)

        return {
            'alcohol_drinks_per_week_at_index': drinks_per_week,
            'alcohol_oz_per_week_at_index': oz_per_week,
            'alcohol_drinks_per_day': drinks_per_week / 7 if drinks_per_week else None,
            'alcohol_drinking_level': drinking_level,
            'alcohol_heavy_use': drinking_level == 'heavy',
            'alcohol_moderate_use': drinking_level == 'moderate',
            'alcohol_light_use': drinking_level == 'light',
        }

    def build_all_features(self, empi: str, sex: str = None) -> Dict:
        """Build all alcohol features for a patient."""
        features = {'empi': empi}
        features.update(self.build_status_features(empi))
        features.update(self.build_quantitative_features(empi, sex))
        return features
