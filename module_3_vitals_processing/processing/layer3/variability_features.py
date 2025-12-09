"""Variability feature calculations.

Calculates RMSSD and successive variance for vital sign variability analysis.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_rmssd(values: np.ndarray) -> float:
    """Calculate Root Mean Square of Successive Differences.

    RMSSD = sqrt(mean((x[i+1] - x[i])^2))

    Args:
        values: Array of values (may contain NaN)

    Returns:
        RMSSD value, or NaN if insufficient data
    """
    # Remove NaN values while preserving consecutive pairs
    clean = values[~np.isnan(values)]

    if len(clean) < 2:
        return np.nan

    # Calculate successive differences
    diffs = np.diff(clean)

    if len(diffs) == 0:
        return np.nan

    # RMSSD = sqrt(mean(diffs^2))
    rmssd = np.sqrt(np.mean(diffs ** 2))
    return rmssd


def calculate_successive_var(values: np.ndarray) -> float:
    """Calculate sum of absolute successive differences.

    SV = sum(|x[i+1] - x[i]|)

    Args:
        values: Array of values (may contain NaN)

    Returns:
        Successive variance value, or NaN if insufficient data
    """
    clean = values[~np.isnan(values)]

    if len(clean) < 2:
        return np.nan

    diffs = np.abs(np.diff(clean))
    return np.sum(diffs)


def calculate_variability_features(
    df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate variability features for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital:
    - {vital}_rmssd: Root mean square of successive differences
    - {vital}_successive_var: Sum of absolute successive differences

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process

    Returns:
        DataFrame with variability feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        for vital in vitals:
            mask_col = f'mask_{vital}'

            # Create masked values (NaN where not observed)
            if mask_col in patient_df.columns:
                values = patient_df[vital].where(patient_df[mask_col] == 1, np.nan).values
            else:
                values = patient_df[vital].values

            # Calculate cumulative variability at each time point
            rmssds = []
            svs = []

            for i in range(len(patient_df)):
                window_values = values[:i+1]
                rmssds.append(calculate_rmssd(window_values))
                svs.append(calculate_successive_var(window_values))

            patient_df[f'{vital}_rmssd'] = rmssds
            patient_df[f'{vital}_successive_var'] = svs

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
