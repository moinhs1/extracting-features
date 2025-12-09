"""Layer 3 Builder: Feature engineering for vital sign trajectories.

Transforms Layer 2 hourly grid into:
- timeseries_features.parquet: ~295 features per hour
- summary_features.parquet: ~3500 features per patient
"""
from typing import List
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from processing.layer3.composite_vitals import add_composite_vitals
from processing.layer3.rolling_stats import calculate_rolling_stats
from processing.layer3.trend_features import calculate_trend_features
from processing.layer3.variability_features import calculate_variability_features
from processing.layer3.threshold_features import calculate_threshold_features
from processing.layer3.data_density import calculate_data_density
from processing.layer3.summary_aggregator import aggregate_to_summary

# All vitals including composites
LAYER3_VITALS: List[str] = [
    'HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP',
    'shock_index', 'pulse_pressure'
]

# Raw vitals (from Layer 2)
RAW_VITALS: List[str] = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']

# Rolling window sizes
ROLLING_WINDOWS: List[int] = [6, 12, 24]

# Get CPU count
N_JOBS = max(1, mp.cpu_count() - 1)


def load_layer2_with_masks(
    parquet_path: Path,
    hdf5_path: Path,
) -> pd.DataFrame:
    """Load Layer 2 data and add mask columns from HDF5.

    Pivots from long format (vital_type column) to wide format (one column per vital).
    Adds mask_{vital} columns from imputation_tier tensor.

    Args:
        parquet_path: Path to hourly_grid.parquet
        hdf5_path: Path to hourly_tensors.h5

    Returns:
        Wide-format DataFrame with vital values and mask columns
    """
    print(f"  Loading {parquet_path}")
    grid = pd.read_parquet(parquet_path)

    # Pivot from long to wide format
    print(f"  Pivoting to wide format...")
    wide = grid.pivot_table(
        index=['EMPI', 'hour_from_pe'],
        columns='vital_type',
        values='mean',
        aggfunc='first'
    ).reset_index()

    # Load imputation tiers from HDF5
    print(f"  Loading masks from {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        imputation_tiers = f['imputation_tier'][:]
        patient_index = [p.decode() if isinstance(p, bytes) else p for p in f['patient_index'][:]]
        vital_index = [v.decode() if isinstance(v, bytes) else v for v in f['vital_index'][:]]

    # Create patient/vital/hour to tier mapping
    patient_to_idx = {p: i for i, p in enumerate(patient_index)}
    vital_to_idx = {v: i for i, v in enumerate(vital_index)}

    # Add mask columns (mask=1 for Tier 1-2, mask=0 for Tier 3-4)
    for vital in RAW_VITALS:
        if vital in vital_to_idx:
            v_idx = vital_to_idx[vital]
            mask_values = []

            for _, row in wide.iterrows():
                empi = row['EMPI']
                hour = int(row['hour_from_pe'])

                if empi in patient_to_idx and -24 <= hour <= 720:
                    p_idx = patient_to_idx[empi]
                    h_idx = hour + 24  # Convert to 0-indexed
                    tier = imputation_tiers[p_idx, h_idx, v_idx]
                    mask_values.append(1 if tier <= 2 else 0)
                else:
                    mask_values.append(0)

            wide[f'mask_{vital}'] = mask_values

    return wide


def build_layer3(
    layer2_parquet_path: Path,
    layer2_hdf5_path: Path,
    timeseries_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Build Layer 3 features from Layer 2 data.

    Args:
        layer2_parquet_path: Path to hourly_grid.parquet
        layer2_hdf5_path: Path to hourly_tensors.h5
        timeseries_output_path: Output path for timeseries_features.parquet
        summary_output_path: Output path for summary_features.parquet
    """
    print("\n" + "=" * 60)
    print("Layer 3 Builder - Feature Engineering")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}, using {N_JOBS} workers")

    # Step 1: Load Layer 2 data
    print("\n[1/8] Loading Layer 2 data...")
    df = load_layer2_with_masks(layer2_parquet_path, layer2_hdf5_path)
    print(f"  Loaded {len(df):,} rows from {df['EMPI'].nunique():,} patients")

    # Step 2: Add composite vitals
    print("\n[2/8] Calculating composite vitals...")
    df = add_composite_vitals(df)

    # Add mask columns for composites (same as components)
    df['mask_shock_index'] = (df['mask_HR'] == 1) & (df['mask_SBP'] == 1)
    df['mask_pulse_pressure'] = (df['mask_SBP'] == 1) & (df['mask_DBP'] == 1)
    df['mask_shock_index'] = df['mask_shock_index'].astype(int)
    df['mask_pulse_pressure'] = df['mask_pulse_pressure'].astype(int)

    print(f"  Added shock_index and pulse_pressure")

    # Step 3: Rolling statistics
    print("\n[3/8] Calculating rolling statistics...")
    df = calculate_rolling_stats(df, vitals=LAYER3_VITALS, windows=ROLLING_WINDOWS)

    # Step 4: Trend features
    print("\n[4/8] Calculating trend features...")
    df = calculate_trend_features(df, vitals=LAYER3_VITALS, windows=ROLLING_WINDOWS)

    # Step 5: Variability features
    print("\n[5/8] Calculating variability features...")
    df = calculate_variability_features(df, vitals=LAYER3_VITALS)

    # Step 6: Threshold features
    print("\n[6/8] Calculating threshold features...")
    df = calculate_threshold_features(df)

    # Step 7: Data density features
    print("\n[7/8] Calculating data density features...")
    df = calculate_data_density(df, vitals=LAYER3_VITALS)

    # Save time-series output
    print(f"\n  Saving time-series features to {timeseries_output_path}")
    timeseries_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(timeseries_output_path, index=False)

    # Step 8: Aggregate to summary
    print("\n[8/8] Aggregating to summary features...")

    # Get all feature columns (exclude identifiers and raw vitals)
    exclude_cols = {'EMPI', 'hour_from_pe'} | set(LAYER3_VITALS) | {f'mask_{v}' for v in LAYER3_VITALS}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    summary_df = aggregate_to_summary(df, feature_cols=feature_cols)

    print(f"  Saving summary features to {summary_output_path}")
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_output_path, index=False)

    print("\n" + "=" * 60)
    print("Layer 3 COMPLETE")
    print("=" * 60)
    print(f"  Time-series: {len(df):,} rows x {len(df.columns)} columns")
    print(f"  Summary: {len(summary_df):,} patients x {len(summary_df.columns)} features")


def main():
    """CLI entry point for Layer 3 builder."""
    from config.vitals_config import (
        HOURLY_GRID_PATH,
        HOURLY_TENSORS_PATH,
        TIMESERIES_FEATURES_PATH,
        SUMMARY_FEATURES_PATH,
    )

    print(f"\nInput:  {HOURLY_GRID_PATH}")
    print(f"Input:  {HOURLY_TENSORS_PATH}")
    print(f"Output: {TIMESERIES_FEATURES_PATH}")
    print(f"Output: {SUMMARY_FEATURES_PATH}")

    build_layer3(
        layer2_parquet_path=HOURLY_GRID_PATH,
        layer2_hdf5_path=HOURLY_TENSORS_PATH,
        timeseries_output_path=TIMESERIES_FEATURES_PATH,
        summary_output_path=SUMMARY_FEATURES_PATH,
    )


if __name__ == "__main__":
    main()
