# transformers/height_builder.py
"""Height Feature Builder."""

import pandas as pd
from datetime import datetime
from typing import Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS
from utils.unit_conversion import inches_to_cm, cm_to_m


class HeightBuilder:
    """Build height features for patients."""

    HEIGHT_CONCEPTS = ['Height']
    STALENESS_DAYS = STALENESS_THRESHOLDS['height']  # 10 years (3650 days)

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        self.phy_data = phy_data
        self.index_dates = index_dates
        if not phy_data.empty:
            self._preprocess()

    def _preprocess(self):
        self.phy_data = self.phy_data.copy()
        self.phy_data['Date'] = pd.to_datetime(self.phy_data['Date'], errors='coerce')
        self.phy_data['Result_Numeric'] = pd.to_numeric(self.phy_data['Result'], errors='coerce')

    def _convert_to_meters(self, value: float, units: str) -> float:
        """Convert height to meters.

        Auto-detect units:
        - if value < 100 assume inches
        - if value >= 100 assume cm
        """
        units_str = str(units).lower() if pd.notna(units) else ''
        if 'in' in units_str or value < 100:  # Assume inches if < 100
            return cm_to_m(inches_to_cm(value))
        elif 'cm' in units_str or value >= 100:
            return cm_to_m(value)
        return value  # Assume already in meters

    def build_features(self, empi: str) -> Dict:
        """Build height features for a patient.

        Returns:
            Dict with keys:
            - height_m_at_index: height in meters
            - height_cm_at_index: height in centimeters
            - height_in_at_index: height in inches
            - height_at_index_date: date of measurement
            - height_at_index_days_prior: days before index date
            - height_at_index_stale: True if measurement older than 10 years
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'height_m_at_index': None}

        if self.phy_data.empty:
            return {'height_m_at_index': None, 'height_at_index_stale': True}

        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.HEIGHT_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Result_Numeric'] > 20) &
            (self.phy_data['Result_Numeric'] < 300)
        )
        records = self.phy_data[mask].sort_values('Date')
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {'height_m_at_index': None, 'height_at_index_stale': True}

        closest = valid.iloc[-1]
        height_m = self._convert_to_meters(closest['Result_Numeric'], closest.get('Units'))
        height_cm = height_m * 100
        height_in = height_cm / 2.54
        days_prior = (pd.Timestamp(index_date) - closest['Date']).days

        return {
            'height_m_at_index': height_m,
            'height_cm_at_index': height_cm,
            'height_in_at_index': height_in,
            'height_at_index_date': closest['Date'],
            'height_at_index_days_prior': days_prior,
            'height_at_index_stale': days_prior > self.STALENESS_DAYS,
        }
