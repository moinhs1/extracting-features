# transformers/bmi_builder.py
"""
BMI Feature Builder
===================

Builds BMI features with temporal awareness.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class BMIBuilder:
    """Build BMI features for patients."""

    BMI_CONCEPTS = ['BMI']
    STALENESS_DAYS = STALENESS_THRESHOLDS['bmi']

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

    def classify_bmi(self, bmi: Optional[float]) -> str:
        """
        Classify BMI into WHO categories.

        Args:
            bmi: BMI value

        Returns:
            Category string
        """
        if bmi is None or pd.isna(bmi):
            return 'unknown'
        if bmi < 18.5:
            return 'underweight'
        if bmi < 25:
            return 'normal'
        if bmi < 30:
            return 'overweight'
        if bmi < 35:
            return 'obese_1'
        if bmi < 40:
            return 'obese_2'
        return 'obese_3'

    def classify_trend(self, pct_change: Optional[float], threshold: float = 5.0) -> str:
        """
        Classify trend direction.

        Args:
            pct_change: Percent change value
            threshold: Threshold for significant change

        Returns:
            Trend direction string
        """
        if pct_change is None or pd.isna(pct_change):
            return 'unknown'
        if pct_change > threshold:
            return 'increasing'
        if pct_change < -threshold:
            return 'decreasing'
        return 'stable'

    def _get_patient_bmi_records(self, empi: str) -> pd.DataFrame:
        """Get BMI records for a patient."""
        if self.phy_data.empty:
            return pd.DataFrame()

        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.BMI_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Result_Numeric'] >= 10) &
            (self.phy_data['Result_Numeric'] <= 100)
        )
        return self.phy_data[mask].sort_values('Date')

    def build_point_in_time(self, empi: str) -> Dict:
        """
        Build point-in-time BMI features.

        Args:
            empi: Patient identifier

        Returns:
            Dict with bmi_at_index, bmi_at_index_date, etc.
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'bmi_at_index': None}

        records = self._get_patient_bmi_records(empi)

        if records.empty:
            return {
                'bmi_at_index': None,
                'bmi_at_index_date': None,
                'bmi_at_index_days_prior': None,
                'bmi_at_index_stale': True,
                'bmi_category_at_index': 'unknown',
            }

        # Filter to records before or on index date
        valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'bmi_at_index': None,
                'bmi_at_index_date': None,
                'bmi_at_index_days_prior': None,
                'bmi_at_index_stale': True,
                'bmi_category_at_index': 'unknown',
            }

        # Get closest (most recent before index)
        closest = valid.iloc[-1]
        bmi_value = closest['Result_Numeric']
        bmi_date = closest['Date']
        days_prior = (pd.Timestamp(index_date) - bmi_date).days

        return {
            'bmi_at_index': bmi_value,
            'bmi_at_index_date': bmi_date,
            'bmi_at_index_days_prior': days_prior,
            'bmi_at_index_stale': days_prior > self.STALENESS_DAYS,
            'bmi_category_at_index': self.classify_bmi(bmi_value),
        }
