# transformers/functional_status_builder.py
"""
Functional Status (KPS) Feature Builder
=======================================

Builds Karnofsky Performance Status (KPS) features for patients.
KPS is a standard measure of functional status in oncology, ranging from 0-100.

KPS Categories:
- 100: Normal, no complaints, no evidence of disease
- 90: Able to carry on normal activity, minor symptoms
- 80: Normal activity with effort, some symptoms
- 70: Cares for self, unable to carry on normal activity
- 60: Requires occasional assistance, cares for most needs
- 50: Requires considerable assistance, frequent medical care
- 40: Disabled, requires special care and assistance
- 30: Severely disabled, hospitalization indicated
- 20: Very sick, hospitalization necessary
- 10: Moribund, fatal processes progressing rapidly
- 0: Dead
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS


class FunctionalStatusBuilder:
    """Build functional status (KPS) features for patients."""

    KPS_CONCEPTS = ['KPS (Karnofsky performance status)']
    STALENESS_DAYS = STALENESS_THRESHOLDS['kps']  # 30 days

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

    def _get_kps_records(self, empi: str) -> pd.DataFrame:
        """
        Get valid KPS records for a patient.

        Args:
            empi: Patient identifier

        Returns:
            DataFrame of valid KPS records sorted by date
        """
        if self.phy_data.empty:
            return pd.DataFrame()

        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.KPS_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Result_Numeric'] >= 0) &
            (self.phy_data['Result_Numeric'] <= 100)
        )
        return self.phy_data[mask].sort_values('Date')

    def _classify_kps(self, kps: int) -> str:
        """
        Classify KPS into functional categories.

        Args:
            kps: KPS value (0-100)

        Returns:
            Category string
        """
        if kps >= 80:
            return 'good_functional_status'
        if kps >= 50:
            return 'moderate_impairment'
        return 'severe_impairment'

    def build_kps_features(self, empi: str) -> Dict:
        """
        Build KPS features for a patient.

        Args:
            empi: Patient identifier

        Returns:
            Dict with KPS features:
            - kps_at_index: KPS score closest to index date
            - kps_date: Date of KPS measurement
            - kps_days_prior: Days before index date
            - kps_stale: True if > 30 days old
            - kps_category: good_functional_status/moderate_impairment/severe_impairment
            - kps_below_70_at_index: Significant impairment flag
            - kps_below_50_at_index: Severe impairment flag
            - kps_declined_10pts_90d: 10+ point decline in 90 days
            - functional_status_impaired: Alias for KPS < 70
            - mobility_impaired: Alias for KPS < 70
            - bedridden: KPS < 30
        """
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'kps_at_index': None}

        records = self._get_kps_records(empi)

        # Handle empty records (no Date column to filter on)
        if records.empty:
            valid = records
        else:
            valid = records[records['Date'] <= pd.Timestamp(index_date)]

        if valid.empty:
            return {
                'kps_at_index': None,
                'kps_date': None,
                'kps_days_prior': None,
                'kps_stale': True,
                'kps_category': 'unknown',
                'kps_below_70_at_index': False,
                'kps_below_50_at_index': False,
                'kps_declined_10pts_90d': False,
                'functional_status_impaired': False,
                'mobility_impaired': False,
                'bedridden': False,
            }

        # Get closest (most recent before index)
        closest = valid.iloc[-1]
        kps = int(closest['Result_Numeric'])
        kps_date = closest['Date']
        days_prior = (pd.Timestamp(index_date) - kps_date).days

        # Check for decline in 90 days
        cutoff_90d = pd.Timestamp(index_date) - timedelta(days=90)
        data_90d = valid[valid['Date'] >= cutoff_90d].sort_values('Date')
        declined_10pts = False
        if len(data_90d) >= 2:
            first_kps = data_90d.iloc[0]['Result_Numeric']
            last_kps = data_90d.iloc[-1]['Result_Numeric']
            declined_10pts = (first_kps - last_kps) >= 10

        return {
            'kps_at_index': kps,
            'kps_date': kps_date,
            'kps_days_prior': days_prior,
            'kps_stale': days_prior > self.STALENESS_DAYS,
            'kps_category': self._classify_kps(kps),
            'kps_below_70_at_index': kps < 70,
            'kps_below_50_at_index': kps < 50,
            'kps_declined_10pts_90d': declined_10pts,
            'functional_status_impaired': kps < 70,
            'mobility_impaired': kps < 70,
            'bedridden': kps < 30,
        }

    def build_all_features(self, empi: str) -> Dict:
        """
        Build all functional status features for a patient.

        Args:
            empi: Patient identifier

        Returns:
            Dict with empi and all KPS features
        """
        features = {'empi': empi}
        features.update(self.build_kps_features(empi))
        return features

    def build_for_cohort(self, empis: List[str]) -> pd.DataFrame:
        """
        Build KPS features for entire cohort.

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
