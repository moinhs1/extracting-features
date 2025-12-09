"""Rolling window statistics calculator.

Calculates rolling mean, std, cv, min, max, range for specified windows.
Only uses Tier 1-2 (observed + forward-fill) data for valid statistics.
"""
from typing import List, Dict, Callable
import pandas as pd
import numpy as np

# Rolling statistics to calculate
ROLLING_STAT_FUNCTIONS: Dict[str, Callable] = {
    'mean': lambda x: x.mean(),
    'std': lambda x: x.std(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
}


def calculate_rolling_stats(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate rolling window statistics for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital per window:
    - {vital}_roll{w}h_mean
    - {vital}_roll{w}h_std
    - {vital}_roll{w}h_cv (coefficient of variation)
    - {vital}_roll{w}h_min
    - {vital}_roll{w}h_max
    - {vital}_roll{w}h_range

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process
        windows: List of window sizes in hours

    Returns:
        DataFrame with rolling stat columns added
    """
    result = df.copy()

    # Process each patient separately
    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

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

                patient_df[f'{prefix}_mean'] = rolling.mean()
                patient_df[f'{prefix}_std'] = rolling.std()
                patient_df[f'{prefix}_min'] = rolling.min()
                patient_df[f'{prefix}_max'] = rolling.max()

                # CV = std / mean
                patient_df[f'{prefix}_cv'] = patient_df[f'{prefix}_std'] / patient_df[f'{prefix}_mean']

                # Range = max - min
                patient_df[f'{prefix}_range'] = patient_df[f'{prefix}_max'] - patient_df[f'{prefix}_min']

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
