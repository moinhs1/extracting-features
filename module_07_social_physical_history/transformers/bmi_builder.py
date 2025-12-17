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
