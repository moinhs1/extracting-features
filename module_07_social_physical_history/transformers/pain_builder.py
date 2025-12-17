# transformers/pain_builder.py
"""Pain Score Feature Builder."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class PainBuilder:
    """Build pain score features for patients."""

    PAIN_CONCEPTS = [
        'Pain Score EPIC (0-10)',
        'Pain Level (0-10)',
        'Pain 0-10',
        'VAS score',
        'Pain Assessment - VAS score',
    ]

    STALENESS_DAYS = STALENESS_THRESHOLDS['pain_score']  # 7 days

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        """
        Initialize builder.

        Args:
            phy_data: DataFrame with Phy.txt data
            index_dates: Dict mapping EMPI -> index date
        """
        self.phy_data = phy_data
        self.index_dates = index_dates
        if not phy_data.empty:
            self._preprocess()

    def _preprocess(self):
        """Parse dates and numeric values."""
        self.phy_data = self.phy_data.copy()
        self.phy_data['Date'] = pd.to_datetime(self.phy_data['Date'], errors='coerce')
        self.phy_data['Result_Numeric'] = pd.to_numeric(self.phy_data['Result'], errors='coerce')

    def _get_pain_records(self, empi: str) -> pd.DataFrame:
        """Get pain records for a patient."""
        if self.phy_data.empty:
            return pd.DataFrame()
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.PAIN_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Result_Numeric'] >= 0) &
            (self.phy_data['Result_Numeric'] <= 10)
        )
        return self.phy_data[mask].sort_values('Date')

    def build_point_in_time(self, empi: str) -> Dict:
        """
        Build point-in-time pain features.

        Args:
            empi: Patient identifier

        Returns:
            Dict with pain_score_at_index, pain_score_date, etc.
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'pain_score_at_index': None}

        records = self._get_pain_records(empi)

        if records.empty:
            return {
                'pain_score_at_index': None,
                'pain_score_date': None,
                'pain_score_days_prior': None,
                'pain_score_stale': True,
            }

        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'pain_score_at_index': None,
                'pain_score_date': None,
                'pain_score_days_prior': None,
                'pain_score_stale': True,
            }

        closest = valid.iloc[-1]
        days_prior = (pd.Timestamp(index_date) - closest['Date']).days

        return {
            'pain_score_at_index': closest['Result_Numeric'],
            'pain_score_date': closest['Date'],
            'pain_score_days_prior': days_prior,
            'pain_score_stale': days_prior > self.STALENESS_DAYS,
        }

    def build_window_features(self, empi: str) -> Dict:
        """
        Build window aggregate features.

        Args:
            empi: Patient identifier

        Returns:
            Dict with mean, min, max, count for 7d and 30d windows
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {}

        records = self._get_pain_records(empi)
        features = {}

        if records.empty:
            for prefix in ['7d', '30d']:
                features[f'pain_{prefix}_mean'] = None
                features[f'pain_{prefix}_max'] = None
                features[f'pain_{prefix}_min'] = None
                features[f'pain_{prefix}_count'] = 0
            return features

        for window_days, prefix in [(7, '7d'), (30, '30d')]:
            cutoff = pd.Timestamp(index_date) - timedelta(days=window_days)
            window_data = records[
                (records['Date'] >= cutoff) &
                (records['Date'] <= pd.Timestamp(index_date))
            ]['Result_Numeric']

            if window_data.empty:
                features[f'pain_{prefix}_mean'] = None
                features[f'pain_{prefix}_max'] = None
                features[f'pain_{prefix}_min'] = None
                features[f'pain_{prefix}_count'] = 0
            else:
                features[f'pain_{prefix}_mean'] = window_data.mean()
                features[f'pain_{prefix}_max'] = window_data.max()
                features[f'pain_{prefix}_min'] = window_data.min()
                features[f'pain_{prefix}_count'] = len(window_data)

        return features

    def build_all_features(self, empi: str) -> Dict:
        """
        Build all pain features for a patient.

        Args:
            empi: Patient identifier

        Returns:
            Dict with all pain features
        """
        features = {'empi': empi}
        pit = self.build_point_in_time(empi)
        features.update(pit)
        features.update(self.build_window_features(empi))

        # Severity flags
        pain_at_index = pit.get('pain_score_at_index')
        features['pain_severe_at_index'] = pain_at_index is not None and pain_at_index >= 7
        features['pain_moderate_at_index'] = pain_at_index is not None and 4 <= pain_at_index < 7

        # Severe any in window
        features['pain_severe_any_7d'] = features.get('pain_7d_max', 0) >= 7 if features.get('pain_7d_max') else False
        features['pain_severe_any_30d'] = features.get('pain_30d_max', 0) >= 7 if features.get('pain_30d_max') else False

        return features

    def build_for_cohort(self, empis: List[str]) -> pd.DataFrame:
        """
        Build pain features for entire cohort.

        Args:
            empis: List of patient identifiers

        Returns:
            DataFrame with one row per patient
        """
        all_features = []
        for empi in empis:
            features = self.build_all_features(empi)
            all_features.append(features)
        return pd.DataFrame(all_features)
