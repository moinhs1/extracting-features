# /home/moin/TDA_11_25/module_04_medications/exporters/xgboost_exporter.py
"""
XGBoost Exporter
================

Export medication features as wide tabular format for XGBoost.

Combines:
- Layer 2: Therapeutic class indicators × 4 windows
- Layer 3: Top N individual medications
- Layer 5: Dose intensity features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import GOLD_DIR, EXPORTS_DIR


def load_all_gold_layers():
    """Load all gold layer outputs."""
    class_df = pd.read_parquet(GOLD_DIR / "therapeutic_classes" / "class_indicators.parquet")
    individual_df = pd.read_parquet(GOLD_DIR / "individual_indicators" / "individual_indicators.parquet")
    dose_df = pd.read_parquet(GOLD_DIR / "dose_intensity" / "dose_intensity.parquet")
    return class_df, individual_df, dose_df


def pivot_class_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot class indicators to wide format (one row per patient).

    Creates columns like: {class}_{window}
    """
    # Get class columns
    class_cols = [c for c in df.columns
                  if not c.endswith('_count') and not c.endswith('_first_hours')
                  and c not in ['empi', 'time_window', 'window_start_hours', 'window_end_hours']]

    result_rows = []

    for empi, group in df.groupby('empi'):
        row = {'empi': empi}

        for _, window_row in group.iterrows():
            window = window_row['time_window']

            for cls in class_cols:
                col_name = f'{cls}_{window}'
                row[col_name] = int(window_row[cls])

                # Add count
                count_col = f'{cls}_count'
                if count_col in window_row:
                    row[f'{cls}_{window}_count'] = window_row[count_col]

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def pivot_individual_indicators(
    df: pd.DataFrame,
    top_n: int = 100
) -> pd.DataFrame:
    """
    Pivot top N individual medications to wide format.
    """
    # Get medication columns
    med_cols = [c for c in df.columns
                if c.startswith('med_') and not c.endswith('_count') and not c.endswith('_dose')]

    # Select top N by total occurrence
    if len(med_cols) > top_n:
        col_sums = df[med_cols].sum()
        top_meds = col_sums.nlargest(top_n).index.tolist()
    else:
        top_meds = med_cols

    result_rows = []

    for empi, group in df.groupby('empi'):
        row = {'empi': empi}

        for _, window_row in group.iterrows():
            window = window_row['time_window']

            for med in top_meds:
                col_name = f'{med}_{window}'
                row[col_name] = int(window_row.get(med, 0))

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def aggregate_dose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate dose intensity features per patient.
    """
    result = df.groupby('empi').agg({
        'daily_dose': ['sum', 'mean', 'max'],
        'ddd_ratio': ['mean', 'max'],
        'admin_count': 'sum',
    })

    # Flatten column names
    result.columns = ['_'.join(col) for col in result.columns]
    result = result.reset_index()

    return result


def export_xgboost(
    output_path: Optional[Path] = None,
    top_individual: int = 100
) -> pd.DataFrame:
    """
    Main XGBoost export function.

    Args:
        output_path: Output parquet path
        top_individual: Number of top individual medications to include

    Returns:
        Wide feature DataFrame
    """
    print("=" * 60)
    print("XGBoost Export")
    print("=" * 60)

    # Load data
    print("\n1. Loading gold layer data...")
    class_df, individual_df, dose_df = load_all_gold_layers()

    # Pivot class indicators
    print("\n2. Processing class indicators...")
    class_wide = pivot_class_indicators(class_df)
    print(f"   Class features: {len(class_wide.columns) - 1}")

    # Pivot individual indicators
    print(f"\n3. Processing individual indicators (top {top_individual})...")
    individual_wide = pivot_individual_indicators(individual_df, top_individual)
    print(f"   Individual features: {len(individual_wide.columns) - 1}")

    # Aggregate dose features
    print("\n4. Processing dose intensity...")
    dose_wide = aggregate_dose_features(dose_df)
    print(f"   Dose features: {len(dose_wide.columns) - 1}")

    # Merge all
    print("\n5. Merging features...")
    result = class_wide.merge(individual_wide, on='empi', how='outer')
    result = result.merge(dose_wide, on='empi', how='outer')

    # Fill NaN with 0
    result = result.fillna(0)

    print(f"   Final feature matrix: {result.shape}")

    # Save
    if output_path is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "xgboost_medication_features.parquet"

    print(f"\n6. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("XGBoost Export Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    export_xgboost()
