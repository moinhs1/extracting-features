# transformers/bsa_builder.py
"""BSA Feature Builder."""

import pandas as pd
from datetime import datetime
from typing import Dict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import STALENESS_THRESHOLDS
from utils.unit_conversion import calculate_bsa_dubois, lbs_to_kg, inches_to_cm


class BSABuilder:
    """Build BSA features for patients."""

    BSA_CONCEPTS = ['Body Surface Area (BSA)']
    STALENESS_DAYS = STALENESS_THRESHOLDS['bsa']

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        self.phy_data = phy_data
        self.index_dates = index_dates
        if not phy_data.empty:
            self._preprocess()

    def _preprocess(self):
        self.phy_data = self.phy_data.copy()
        self.phy_data['Date'] = pd.to_datetime(self.phy_data['Date'], errors='coerce')
        self.phy_data['Result_Numeric'] = pd.to_numeric(self.phy_data['Result'], errors='coerce')

    def _get_weight_kg(self, empi: str, index_date: datetime) -> float:
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'] == 'Weight') &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Date'] <= pd.Timestamp(index_date))
        )
        records = self.phy_data[mask].sort_values('Date')
        if records.empty:
            return None
        closest = records.iloc[-1]
        # Assume lbs if units not specified or contain 'lb'
        units = str(closest.get('Units', 'lbs')).lower()
        if 'lb' in units or pd.isna(closest.get('Units')):
            return lbs_to_kg(closest['Result_Numeric'])
        return closest['Result_Numeric']

    def _get_height_cm(self, empi: str, index_date: datetime) -> float:
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'] == 'Height') &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Date'] <= pd.Timestamp(index_date))
        )
        records = self.phy_data[mask].sort_values('Date')
        if records.empty:
            return None
        closest = records.iloc[-1]
        value = closest['Result_Numeric']
        # If value < 100, assume inches
        if value < 100:
            return inches_to_cm(value)
        return value

    def build_features(self, empi: str) -> Dict:
        index_date = self.index_dates.get(empi)
        if index_date is None:
            return {'bsa_at_index': None}

        # Handle empty data
        if self.phy_data.empty:
            return {
                'bsa_at_index': None,
                'bsa_at_index_stale': True,
                'bsa_method': None,
            }

        # Try measured BSA first
        mask = (
            (self.phy_data['EMPI'] == empi) &
            (self.phy_data['Concept_Name'].isin(self.BSA_CONCEPTS)) &
            (self.phy_data['Result_Numeric'].notna()) &
            (self.phy_data['Date'] <= pd.Timestamp(index_date))
        )
        records = self.phy_data[mask].sort_values('Date')

        if not records.empty:
            closest = records.iloc[-1]
            days_prior = (pd.Timestamp(index_date) - closest['Date']).days
            return {
                'bsa_at_index': closest['Result_Numeric'],
                'bsa_at_index_date': closest['Date'],
                'bsa_at_index_days_prior': days_prior,
                'bsa_at_index_stale': days_prior > self.STALENESS_DAYS,
                'bsa_method': 'measured',
            }

        # Calculate from weight and height
        weight_kg = self._get_weight_kg(empi, index_date)
        height_cm = self._get_height_cm(empi, index_date)

        if weight_kg and height_cm:
            bsa = calculate_bsa_dubois(weight_kg, height_cm)
            return {
                'bsa_at_index': bsa,
                'bsa_at_index_date': None,
                'bsa_at_index_days_prior': None,
                'bsa_at_index_stale': False,
                'bsa_method': 'dubois',
            }

        return {
            'bsa_at_index': None,
            'bsa_at_index_stale': True,
            'bsa_method': None,
        }
