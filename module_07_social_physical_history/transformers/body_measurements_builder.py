# transformers/body_measurements_builder.py
"""
Composite Body Measurements Builder
===================================

Combines features from BMI, Weight, Height, and BSA builders into a single interface.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers.bmi_builder import BMIBuilder
from transformers.weight_builder import WeightBuilder
from transformers.height_builder import HeightBuilder
from transformers.bsa_builder import BSABuilder


class BodyMeasurementsBuilder:
    """
    Composite builder for all body measurement features.

    Combines:
    - BMI features (point-in-time, windows, trends)
    - Weight features (point-in-time, loss/gain detection)
    - Height features (point-in-time with unit conversion)
    - BSA features (measured or calculated)
    """

    def __init__(self, phy_data: pd.DataFrame, index_dates: Dict[str, datetime]):
        """
        Initialize composite builder.

        Args:
            phy_data: DataFrame with Phy.txt data containing measurements
            index_dates: Dict mapping EMPI -> index date
        """
        self.phy_data = phy_data
        self.index_dates = index_dates

        # Initialize sub-builders
        self.bmi_builder = BMIBuilder(phy_data, index_dates)
        self.weight_builder = WeightBuilder(phy_data, index_dates)
        self.height_builder = HeightBuilder(phy_data, index_dates)
        self.bsa_builder = BSABuilder(phy_data, index_dates)

    def build_all_features(self, empi: str) -> Dict:
        """
        Build all body measurement features for a patient.

        Combines features from all sub-builders into a single dict.

        Args:
            empi: Patient identifier

        Returns:
            Dict with all body measurement features including:
            - BMI: bmi_at_index, bmi_category_at_index, bmi_trend, etc.
            - Weight: weight_kg_at_index, weight_lbs_at_index, weight_loss_* flags
            - Height: height_m_at_index, height_cm_at_index, height_in_at_index
            - BSA: bsa_at_index, bsa_method
        """
        features = {'empi': empi}

        # BMI features (includes 'empi' key which we'll overwrite)
        bmi_features = self.bmi_builder.build_all_features(empi)
        bmi_features.pop('empi', None)  # Remove duplicate empi
        features.update(bmi_features)

        # Weight features (includes 'empi' key which we'll discard)
        weight_features = self.weight_builder.build_all_features(empi)
        weight_features.pop('empi', None)  # Remove duplicate empi
        features.update(weight_features)

        # Height features (no 'empi' key)
        features.update(self.height_builder.build_features(empi))

        # BSA features (no 'empi' key)
        features.update(self.bsa_builder.build_features(empi))

        return features

    def build_for_cohort(self, empis: List[str]) -> pd.DataFrame:
        """
        Build body measurement features for entire cohort.

        Args:
            empis: List of patient identifiers

        Returns:
            DataFrame with one row per patient, columns for all body measurement features
        """
        all_features = []
        for empi in empis:
            features = self.build_all_features(empi)
            all_features.append(features)
        return pd.DataFrame(all_features)
