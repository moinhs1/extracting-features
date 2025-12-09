"""Summary aggregation for Layer 3 features.

Aggregates time-series features into per-patient summary over clinical windows.
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Clinical summary windows (hours from PE)
SUMMARY_WINDOWS: Dict[str, Tuple[int, int]] = {
    'pre': (-24, 0),       # Pre-PE baseline
    'acute': (0, 24),      # Acute phase
    'early': (24, 72),     # Early treatment response
    'stab': (72, 168),     # Stabilization (days 3-7)
    'recov': (168, 720),   # Recovery (days 7-30)
}

# Aggregation functions to apply
AGGREGATIONS = ['mean', 'max', 'min']


def aggregate_to_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Aggregate time-series features to per-patient summary.

    For each feature and each window, calculates mean, max, min.

    Args:
        df: Time-series DataFrame with features
        feature_cols: List of feature columns to aggregate

    Returns:
        DataFrame with one row per patient, ~3500 summary features
    """
    patients = df['EMPI'].unique()

    summary_rows = []
    for patient in patients:
        patient_df = df[df['EMPI'] == patient].sort_values('hour_from_pe')

        row = {'EMPI': patient}

        for feature in feature_cols:
            if feature not in patient_df.columns:
                continue

            for window_name, (start_hour, end_hour) in SUMMARY_WINDOWS.items():
                # Filter to window
                window_mask = (patient_df['hour_from_pe'] >= start_hour) & (patient_df['hour_from_pe'] < end_hour)
                window_data = patient_df.loc[window_mask, feature]

                # Skip if no data in window
                if len(window_data) == 0 or window_data.isna().all():
                    for agg in AGGREGATIONS:
                        row[f'{feature}_{window_name}_{agg}'] = np.nan
                    continue

                # Apply aggregations
                row[f'{feature}_{window_name}_mean'] = window_data.mean()
                row[f'{feature}_{window_name}_max'] = window_data.max()
                row[f'{feature}_{window_name}_min'] = window_data.min()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
