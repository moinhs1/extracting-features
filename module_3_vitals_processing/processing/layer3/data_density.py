"""Data density feature calculations.

Tracks observation rates to help models assess feature reliability.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_data_density(
    df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate data density features.

    Features generated:
    - {vital}_obs_pct: Cumulative % of hours with Tier 1-2 data
    - {vital}_obs_count: Cumulative count of observed hours
    - any_vital_obs_pct: % of hours with ANY vital observed

    Args:
        df: DataFrame with mask_{vital} columns
        vitals: List of vital names to process

    Returns:
        DataFrame with density feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()
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

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
