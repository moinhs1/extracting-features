"""Threshold-based feature calculations.

Calculates cumulative hours and time-to-first for clinical thresholds.
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Clinical thresholds: (vital, operator, value)
CLINICAL_THRESHOLDS: Dict[str, Tuple[str, str, float]] = {
    'tachycardia': ('HR', '>', 100),
    'bradycardia': ('HR', '<', 60),
    'hypotension': ('SBP', '<', 90),
    'hypertension': ('SBP', '>', 180),
    'hypoxemia': ('SPO2', '<', 92),
    'tachypnea': ('RR', '>', 24),
    'shock': ('MAP', '<', 65),
    'fever': ('TEMP', '>', 38.5),
    'hypothermia': ('TEMP', '<', 36),
    'high_shock_index': ('shock_index', '>', 0.9),
}

# Time-to-first thresholds (subset most clinically relevant)
TIME_TO_FIRST_THRESHOLDS = [
    'tachycardia', 'hypotension', 'hypoxemia', 'shock', 'high_shock_index'
]


def _apply_threshold(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    """Apply threshold comparison."""
    if operator == '>':
        return values > threshold
    elif operator == '<':
        return values < threshold
    elif operator == '>=':
        return values >= threshold
    elif operator == '<=':
        return values <= threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")


def calculate_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate threshold-based features.

    Uses all data (Tiers 1-4) since threshold crossing is robust to imputation.

    Features generated:
    - hours_{condition}: Cumulative hours meeting threshold
    - time_to_first_{condition}: Hours until first threshold crossing

    Args:
        df: DataFrame with vital columns

    Returns:
        DataFrame with threshold feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        # Calculate cumulative hours for each threshold
        for condition, (vital, operator, threshold) in CLINICAL_THRESHOLDS.items():
            col_name = f'hours_{condition}'

            if vital in patient_df.columns:
                crossing = _apply_threshold(patient_df[vital], operator, threshold)
                patient_df[col_name] = crossing.cumsum()
            else:
                patient_df[col_name] = 0

        # Calculate time-to-first for selected thresholds
        for condition in TIME_TO_FIRST_THRESHOLDS:
            vital, operator, threshold = CLINICAL_THRESHOLDS[condition]
            col_name = f'time_to_first_{condition}'

            if vital in patient_df.columns:
                crossing = _apply_threshold(patient_df[vital], operator, threshold)

                # Find first crossing
                first_idx = crossing.idxmax() if crossing.any() else None

                if first_idx is not None and crossing.loc[first_idx]:
                    first_hour = patient_df.loc[first_idx, 'hour_from_pe']
                    patient_df[col_name] = first_hour
                else:
                    patient_df[col_name] = np.nan
            else:
                patient_df[col_name] = np.nan

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
