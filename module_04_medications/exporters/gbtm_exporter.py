# /home/moin/TDA_11_25/module_04_medications/exporters/gbtm_exporter.py
"""
GBTM Exporter
=============

Export medication features for Group-Based Trajectory Modeling in R (lcmm).

Output format:
- Long format CSV for lcmm package
- Wide format CSV for visualization
- Daily resolution, days 0-7 relative to Time Zero
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import GOLD_DIR, EXPORTS_DIR


def load_class_indicators() -> pd.DataFrame:
    """Load therapeutic class indicators from gold layer."""
    path = GOLD_DIR / "therapeutic_classes" / "class_indicators.parquet"
    return pd.read_parquet(path)


def load_dose_intensity() -> pd.DataFrame:
    """Load dose intensity features from gold layer."""
    path = GOLD_DIR / "dose_intensity" / "dose_intensity.parquet"
    return pd.read_parquet(path)


def create_daily_features(
    class_df: pd.DataFrame,
    dose_df: pd.DataFrame,
    n_days: int = 7
) -> pd.DataFrame:
    """
    Create daily medication features for GBTM.

    Args:
        class_df: Therapeutic class indicators
        dose_df: Dose intensity features
        n_days: Number of days to include (0 to n_days-1)

    Returns:
        DataFrame with daily features
    """
    # Get unique patients
    patients = class_df['empi'].unique()

    # Class indicator columns
    class_cols = [c for c in class_df.columns
                  if not c.endswith('_count') and not c.endswith('_first_hours')
                  and c not in ['empi', 'time_window', 'window_start_hours', 'window_end_hours']]

    # Key classes for GBTM
    key_classes = [
        'ac_ufh_ther', 'ac_ufh_proph',
        'ac_lmwh_ther', 'ac_lmwh_proph',
        'ac_xa_inhibitor', 'ac_vka',
        'cv_vasopressor_any', 'cv_inotrope_any',
        'ps_opioid', 'resp_bronchodilator',
    ]

    # Filter to classes that exist
    key_classes = [c for c in key_classes if c in class_cols]

    results = []

    for empi in patients:
        patient_dose = dose_df[dose_df['empi'] == empi]

        for day in range(n_days):
            row = {
                'empi': empi,
                'day': day,
            }

            # Add class indicators (from appropriate window)
            # Day 0 = acute, Days 1-2 = subacute, Days 3+ = recovery
            if day == 0:
                window = 'acute'
            elif day <= 2:
                window = 'subacute'
            else:
                window = 'recovery'

            patient_class = class_df[(class_df['empi'] == empi) &
                                     (class_df['time_window'] == window)]

            for cls in key_classes:
                if len(patient_class) > 0 and cls in patient_class.columns:
                    row[cls] = int(patient_class[cls].values[0])
                else:
                    row[cls] = 0

            # Add dose intensity for that day
            day_dose = patient_dose[patient_dose['day_from_t0'] == day]

            # Anticoagulant dose
            ac_dose = day_dose[day_dose['class_name'] == 'anticoagulants']
            row['anticoag_ddd_ratio'] = ac_dose['ddd_ratio'].sum() if len(ac_dose) > 0 else 0

            # Vasopressor
            vaso_dose = day_dose[day_dose['class_name'] == 'vasopressors']
            row['vasopressor_daily_dose'] = vaso_dose['daily_dose'].sum() if len(vaso_dose) > 0 else 0

            results.append(row)

    return pd.DataFrame(results)


def export_gbtm_long(df: pd.DataFrame, output_path: Path):
    """
    Export in long format for lcmm package.

    Format: empi, day, feature, value
    """
    id_vars = ['empi', 'day']
    value_vars = [c for c in df.columns if c not in id_vars]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='feature',
        value_name='value'
    )

    long_df.to_csv(output_path, index=False)
    print(f"  Saved long format: {output_path}")
    return long_df


def export_gbtm_wide(df: pd.DataFrame, output_path: Path):
    """
    Export in wide format for visualization.

    Format: empi, feature_day0, feature_day1, ...
    """
    # Pivot so each day becomes columns
    feature_cols = [c for c in df.columns if c not in ['empi', 'day']]

    wide_rows = []
    for empi, group in df.groupby('empi'):
        row = {'empi': empi}
        for _, day_row in group.iterrows():
            day = int(day_row['day'])
            for feat in feature_cols:
                row[f'{feat}_day{day}'] = day_row[feat]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(output_path, index=False)
    print(f"  Saved wide format: {output_path}")
    return wide_df


def export_gbtm(
    output_dir: Optional[Path] = None,
    n_days: int = 7
) -> dict:
    """
    Main GBTM export function.

    Args:
        output_dir: Output directory
        n_days: Number of days to export

    Returns:
        Dictionary with exported DataFrames
    """
    print("=" * 60)
    print("GBTM Export")
    print("=" * 60)

    if output_dir is None:
        output_dir = EXPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading gold layer data...")
    class_df = load_class_indicators()
    dose_df = load_dose_intensity()
    print(f"   Class indicators: {len(class_df):,} rows")
    print(f"   Dose intensity: {len(dose_df):,} rows")

    # Create features
    print(f"\n2. Creating daily features (days 0-{n_days-1})...")
    features = create_daily_features(class_df, dose_df, n_days)
    print(f"   Feature matrix: {features.shape}")

    # Export
    print("\n3. Exporting...")
    long_df = export_gbtm_long(features, output_dir / "gbtm_medication_long.csv")
    wide_df = export_gbtm_wide(features, output_dir / "gbtm_medication_wide.csv")

    print("\n" + "=" * 60)
    print("GBTM Export Complete!")
    print("=" * 60)

    return {'long': long_df, 'wide': wide_df, 'features': features}


if __name__ == "__main__":
    export_gbtm()
