"""Rolling window statistics calculator.

Calculates rolling mean, std, cv, min, max, range for specified windows.
Only uses Tier 1-2 (observed + forward-fill) data for valid statistics.

OPTIMIZED: Provides single-patient function for parallel processing.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_rolling_stats_patient(
    patient_df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate rolling window statistics for a single patient.

    OPTIMIZED: Works on single patient for parallel processing.

    Features generated per vital per window:
    - {vital}_roll{w}h_mean, std, cv, min, max, range

    Args:
        patient_df: Single patient DataFrame (already sorted by hour_from_pe)
        vitals: List of vital names to process
        windows: List of window sizes in hours

    Returns:
        DataFrame with rolling stat columns added
    """
    for vital in vitals:
        mask_col = f'mask_{vital}'

        # Create masked values (NaN where not observed)
        if mask_col in patient_df.columns:
            masked_values = patient_df[vital].where(patient_df[mask_col] == 1, np.nan)
        else:
            masked_values = patient_df[vital]

        for window in windows:
            prefix = f'{vital}_roll{window}h'

            # Calculate rolling stats using only observed values
            rolling = masked_values.rolling(window=window, min_periods=1)

            roll_mean = rolling.mean()
            roll_std = rolling.std()
            roll_min = rolling.min()
            roll_max = rolling.max()

            patient_df[f'{prefix}_mean'] = roll_mean
            patient_df[f'{prefix}_std'] = roll_std
            patient_df[f'{prefix}_min'] = roll_min
            patient_df[f'{prefix}_max'] = roll_max
            patient_df[f'{prefix}_cv'] = roll_std / roll_mean
            patient_df[f'{prefix}_range'] = roll_max - roll_min

    return patient_df


def calculate_rolling_stats(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate rolling window statistics for each vital (batch mode).

    For parallel processing, use calculate_rolling_stats_patient instead.
    """
    patients = df['EMPI'].unique()
    all_results = []

    for patient in patients:
        patient_df = df[df['EMPI'] == patient].sort_values('hour_from_pe').copy()
        patient_df = calculate_rolling_stats_patient(patient_df, vitals, windows)
        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
