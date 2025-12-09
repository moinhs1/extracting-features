"""Trend feature calculations.

Calculates slope, R-squared, and clinical direction for vital sign trajectories.
"""
from typing import List
import pandas as pd
import numpy as np
from scipy import stats

# Normal ranges for direction calculation
NORMAL_RANGES = {
    'HR': (60, 100),
    'SBP': (90, 140),
    'DBP': (60, 90),
    'MAP': (65, 100),
    'RR': (12, 20),
    'SPO2': (95, 100),
    'TEMP': (36.5, 37.5),
    'shock_index': (0.5, 0.7),
    'pulse_pressure': (40, 60),
}

# Vitals where higher is better (regardless of current value)
HIGHER_IS_BETTER = {'SBP', 'MAP', 'SPO2', 'pulse_pressure'}
# Vitals where lower is better
LOWER_IS_BETTER = {'shock_index'}


def calculate_direction(slope: float, current_value: float, vital: str) -> int:
    """Determine if trend is improving, stable, or worsening.

    Args:
        slope: Slope of recent trend
        current_value: Current vital value
        vital: Name of vital sign

    Returns:
        1 = improving, 0 = stable, -1 = worsening
    """
    # Threshold for "stable" (small slope)
    if abs(slope) < 0.5:
        return 0

    if vital in HIGHER_IS_BETTER:
        return 1 if slope > 0 else -1

    if vital in LOWER_IS_BETTER:
        return 1 if slope < 0 else -1

    # For toward_normal vitals (HR, RR, TEMP, DBP)
    if vital in NORMAL_RANGES:
        low, high = NORMAL_RANGES[vital]

        if current_value > high:
            # Above normal - decreasing is improving
            return 1 if slope < 0 else -1
        elif current_value < low:
            # Below normal - increasing is improving
            return 1 if slope > 0 else -1
        else:
            # In normal range - any change away from it is worsening
            return 0 if abs(slope) < 1.0 else -1

    return 0


def _calculate_slope_r2(values: np.ndarray) -> tuple:
    """Calculate linear regression slope and R-squared.

    Args:
        values: Array of values (may contain NaN)

    Returns:
        (slope, r2) tuple, or (NaN, NaN) if insufficient data
    """
    # Remove NaN values
    mask = ~np.isnan(values)
    clean_values = values[mask]

    if len(clean_values) < 2:
        return np.nan, np.nan

    x = np.arange(len(clean_values))

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_values)
        r2 = r_value ** 2
        return slope, r2
    except Exception:
        return np.nan, np.nan


def calculate_trend_features(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate trend features for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital per window:
    - {vital}_slope{w}h: Linear regression slope
    - {vital}_slope{w}h_r2: R-squared of the regression
    - {vital}_direction{w}h: -1 (worsening), 0 (stable), 1 (improving)

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process
        windows: List of window sizes in hours

    Returns:
        DataFrame with trend feature columns added
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

            for window in windows:
                prefix = f'{vital}_slope{window}h'

                slopes = []
                r2s = []
                directions = []

                for i in range(len(patient_df)):
                    start_idx = max(0, i - window + 1)
                    window_values = values[start_idx:i+1]

                    slope, r2 = _calculate_slope_r2(window_values)
                    slopes.append(slope)
                    r2s.append(r2)

                    # Calculate direction
                    current_value = values[i] if not np.isnan(values[i]) else patient_df[vital].iloc[i]
                    direction = calculate_direction(slope if not np.isnan(slope) else 0, current_value, vital)
                    directions.append(direction)

                patient_df[f'{prefix}'] = slopes
                patient_df[f'{prefix}_r2'] = r2s
                patient_df[f'{vital}_direction{window}h'] = directions

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
