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

    def _categorize_pack_years(self, pack_years: Optional[float]) -> str:
        """Categorize pack-years."""
        if pack_years is None:
            return 'unknown'
        if pack_years < 10:
            return '<10'
        if pack_years < 20:
            return '10-20'
        if pack_years < 40:
            return '20-40'
        return '>40'

    def build_quantitative_features(self, empi: str) -> Dict:
        """Build quantitative smoking features (pack-years) for a patient."""
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'smoking_pack_years': None}

        pack_per_day = self._get_numeric_value(empi, 'Tobacco Pack Per Day', index_date)
        years_smoked = self._get_numeric_value(empi, 'Tobacco Used Years', index_date)

        pack_years = None
        if pack_per_day is not None and years_smoked is not None:
            pack_years = pack_per_day * years_smoked

        return {
            'smoking_pack_per_day_at_index': pack_per_day,
            'smoking_years_at_index': years_smoked,
            'smoking_pack_years': pack_years,
            'smoking_pack_years_category': self._categorize_pack_years(pack_years),
        }

    def build_quit_features(self, empi: str) -> Dict:
        """Build quit date features for a patient.

        Returns:
            Dict with:
                - smoking_quit_date: parsed quit date from Result field
                - smoking_quit_days_ago: days between quit date and index
                - smoking_quit_recent_90d: True if quit within 90 days (critical for VTE risk!)
                - smoking_quit_recent_1yr: True if quit within 1 year
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'smoking_quit_date': None}

        if self.phy_data.empty:
            return {
                'smoking_quit_date': None,
                'smoking_quit_days_ago': None,
                'smoking_quit_recent_90d': False,
                'smoking_quit_recent_1yr': False,
            }

        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'] == 'Smoking Quit Date')
        )
        records = self.phy_data[mask].sort_values('Date')

        if records.empty:
            return {
                'smoking_quit_date': None,
                'smoking_quit_days_ago': None,
                'smoking_quit_recent_90d': False,
                'smoking_quit_recent_1yr': False,
            }

        # Parse quit date from Result field
        latest = records.iloc[-1]
        quit_date = pd.to_datetime(latest['Result'], errors='coerce')

        if pd.isna(quit_date):
            return {
                'smoking_quit_date': None,
                'smoking_quit_days_ago': None,
                'smoking_quit_recent_90d': False,
                'smoking_quit_recent_1yr': False,
            }

        days_since_quit = (pd.Timestamp(index_date) - quit_date).days

        return {
            'smoking_quit_date': quit_date,
            'smoking_quit_days_ago': days_since_quit,
            'smoking_quit_recent_90d': 0 <= days_since_quit <= 90,
            'smoking_quit_recent_1yr': 0 <= days_since_quit <= 365,
        }

    def build_all_features(self, empi: str) -> Dict:
        """Build all smoking features for a patient.

        Combines status, quantitative (pack-years), and quit features into
        a single dictionary with an 'empi' key.

        Args:
            empi: Patient identifier

        Returns:
            Dict with all smoking features including:
                - empi: patient identifier
                - Status features (smoking_status_at_index, smoking_ever, etc.)
                - Quantitative features (smoking_pack_years, etc.)
                - Quit features (smoking_quit_date, smoking_quit_recent_90d, etc.)
        """
        features = {'empi': empi}
        features.update(self.build_status_features(empi))
        features.update(self.build_quantitative_features(empi))
        features.update(self.build_quit_features(empi))
        return features
