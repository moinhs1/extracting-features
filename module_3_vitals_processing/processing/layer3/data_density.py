"""Data density feature calculations.

Tracks observation rates to help models assess feature reliability.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_data_density_patient(
    patient_df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate data density features for a single patient.

    OPTIMIZED: Works on single patient for parallel processing.

    Features generated:
    - {vital}_obs_pct, {vital}_obs_count, any_vital_obs_pct

    Args:
        patient_df: Single patient DataFrame (already sorted by hour_from_pe)
        vitals: List of vital names to process

    Returns:
        DataFrame with density feature columns added
    """
    n_hours = len(patient_df)

    # Track any vital observed
    any_obs = pd.Series([False] * n_hours, index=patient_df.index)

    for vital in vitals:
        mask_col = f'mask_{vital}'

        if mask_col in patient_df.columns:
            observed = patient_df[mask_col] == 1

            # Cumulative count
            patient_df[f'{vital}_obs_count'] = observed.cumsum()

            # Cumulative percentage
            hours_so_far = pd.Series(range(1, n_hours + 1), index=patient_df.index)
            patient_df[f'{vital}_obs_pct'] = (patient_df[f'{vital}_obs_count'] / hours_so_far) * 100

            # Update any_obs
            any_obs = any_obs | observed
        else:
            patient_df[f'{vital}_obs_count'] = 0
            patient_df[f'{vital}_obs_pct'] = 0.0

    # Any vital observed percentage
    any_obs_count = any_obs.cumsum()
    hours_so_far = pd.Series(range(1, n_hours + 1), index=patient_df.index)
    patient_df['any_vital_obs_pct'] = (any_obs_count / hours_so_far) * 100

    return patient_df


def calculate_data_density(
    df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate data density features (batch mode).

    For parallel processing, use calculate_data_density_patient instead.
    """
    patients = df['EMPI'].unique()
    all_results = []

    for patient in patients:
        patient_df = df[df['EMPI'] == patient].sort_values('hour_from_pe').copy()
        patient_df = calculate_data_density_patient(patient_df, vitals)
        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
