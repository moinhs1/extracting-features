# transformers/weight_builder.py
"""
Weight Feature Builder
======================

Builds weight features with temporal awareness and trend detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS
from utils.unit_conversion import lbs_to_kg, kg_to_lbs


class WeightBuilder:
    """Build weight features for patients."""

    WEIGHT_CONCEPTS = ['Weight']
    STALENESS_DAYS = STALENESS_THRESHOLDS['weight']

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
        self.phy_data['Date'] = pd.to_datetime(
            self.phy_data['Date'], errors='coerce'
        )
        self.phy_data['Result_Numeric'] = pd.to_numeric(
            self.phy_data['Result'], errors='coerce'
        )

    def _get_patient_records(self, empi: str) -> pd.DataFrame:
        """Get weight records for a patient with valid values."""
        if self.phy_data.empty:
            return pd.DataFrame()

        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.WEIGHT_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Result_Numeric'] > 20) &  # Min plausible weight
            (self.phy_data['Result_Numeric'] < 1000)  # Max plausible weight
        )
        return self.phy_data[mask].sort_values('Date')

    def _convert_to_kg(self, value: float, units: Optional[str]) -> float:
        """Convert weight to kg based on units."""
        if pd.isna(units) or 'lb' in str(units).lower():
            return lbs_to_kg(value)
        return value  # Assume kg if not lbs

    def build_point_in_time(self, empi: str) -> Dict:
        """
        Build point-in-time weight features.

        Args:
            empi: Patient identifier

        Returns:
            Dict with weight_kg_at_index, weight_lbs_at_index, days_prior, stale flag
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {
                'weight_kg_at_index': None,
                'weight_lbs_at_index': None,
                'weight_at_index_date': None,
                'weight_at_index_days_prior': None,
                'weight_at_index_stale': True,
            }

        records = self._get_patient_records(empi)

        if records.empty:
            return {
                'weight_kg_at_index': None,
                'weight_lbs_at_index': None,
                'weight_at_index_date': None,
                'weight_at_index_days_prior': None,
                'weight_at_index_stale': True,
            }

        # Filter to records before or on index date
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'weight_kg_at_index': None,
                'weight_lbs_at_index': None,
                'weight_at_index_date': None,
                'weight_at_index_days_prior': None,
                'weight_at_index_stale': True,
            }

        # Get closest (most recent before index)
        closest = valid.iloc[-1]
        weight_lbs = closest['Result_Numeric']
        weight_kg = self._convert_to_kg(weight_lbs, closest.get('Units', 'lbs'))
        days_prior = (pd.Timestamp(index_date) - closest['Date']).days

        return {
            'weight_kg_at_index': weight_kg,
            'weight_lbs_at_index': weight_lbs,
            'weight_at_index_date': closest['Date'],
            'weight_at_index_days_prior': days_prior,
            'weight_at_index_stale': days_prior > self.STALENESS_DAYS,
        }

    def build_trend_features(self, empi: str) -> Dict:
        """
        Build weight trend features for detecting clinically significant changes.

        Args:
            empi: Patient identifier

        Returns:
            Dict with weight_loss_5pct_90d, weight_loss_10pct_6mo, weight_gain_5pct_90d
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {
                'weight_loss_5pct_90d': False,
                'weight_loss_10pct_6mo': False,
                'weight_gain_5pct_90d': False,
            }

        records = self._get_patient_records(empi)

        if records.empty:
            return {
                'weight_loss_5pct_90d': False,
                'weight_loss_10pct_6mo': False,
                'weight_gain_5pct_90d': False,
            }

        # 90-day window
        cutoff_90d = pd.Timestamp(index_date) - timedelta(days=90)
        data_90d = records[
            (records['Date'] >= cutoff_90d) &
            (records['Date'] <= pd.Timestamp(index_date))
        ].sort_values('Date')

        loss_5pct_90d = False
        gain_5pct_90d = False
        pct_change_90d = None

        if len(data_90d) >= 2:
            first = data_90d.iloc[0]['Result_Numeric']
            last = data_90d.iloc[-1]['Result_Numeric']
            pct_change_90d = (last - first) / first * 100
            loss_5pct_90d = pct_change_90d < -5
            gain_5pct_90d = pct_change_90d > 5

        # 6-month window
        cutoff_6mo = pd.Timestamp(index_date) - timedelta(days=180)
        data_6mo = records[
            (records['Date'] >= cutoff_6mo) &
            (records['Date'] <= pd.Timestamp(index_date))
        ].sort_values('Date')

        loss_10pct_6mo = False
        if len(data_6mo) >= 2:
            first = data_6mo.iloc[0]['Result_Numeric']
            last = data_6mo.iloc[-1]['Result_Numeric']
            pct_change_6mo = (last - first) / first * 100
            loss_10pct_6mo = pct_change_6mo < -10

        return {
            'weight_loss_5pct_90d': loss_5pct_90d,
            'weight_loss_10pct_6mo': loss_10pct_6mo,
            'weight_gain_5pct_90d': gain_5pct_90d,
        }

    def build_all_features(self, empi: str) -> Dict:
        """
        Build all weight features for a patient.

        Args:
            empi: Patient identifier

        Returns:
            Dict with all weight features
        """
        features = {'empi': empi}
        features.update(self.build_point_in_time(empi))
        features.update(self.build_trend_features(empi))
        return features

    def build_for_cohort(self, empis: List[str]) -> pd.DataFrame:
        """
        Build weight features for entire cohort.

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
