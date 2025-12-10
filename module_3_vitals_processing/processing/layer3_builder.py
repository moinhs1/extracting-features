"""Layer 3 Builder: Feature engineering for vital sign trajectories.

Transforms Layer 2 hourly grid into:
- timeseries_features.parquet: ~295 features per hour
- summary_features.parquet: ~3500 features per patient

Optimized for maximum CPU utilization with parallel processing.
"""
from typing import List
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import sys

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from processing.layer3.composite_vitals import add_composite_vitals
from processing.layer3.rolling_stats import calculate_rolling_stats_patient
from processing.layer3.trend_features import calculate_trend_features_patient
from processing.layer3.variability_features import calculate_variability_features_patient
from processing.layer3.threshold_features import calculate_threshold_features_patient
from processing.layer3.data_density import calculate_data_density_patient
from processing.layer3.summary_aggregator import aggregate_patient_to_summary

# Suppress pandas fragmentation warnings during parallel processing
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# All vitals including composites
LAYER3_VITALS: List[str] = [
    'HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP',
    'shock_index', 'pulse_pressure'
]

# Raw vitals (from Layer 2)
RAW_VITALS: List[str] = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']

# Rolling window sizes
ROLLING_WINDOWS: List[int] = [6, 12, 24]

# Get CPU count - use all cores for maximum performance
N_JOBS = mp.cpu_count()


def load_layer2_with_masks(
    parquet_path: Path,
    hdf5_path: Path,
) -> pd.DataFrame:
    """Load Layer 2 data and add mask columns from HDF5.

    Pivots from long format (vital_type column) to wide format (one column per vital).
    Adds mask_{vital} columns from imputation_tier tensor.

    OPTIMIZED: Uses vectorized numpy indexing instead of iterrows().

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

    # Create index mappings
    patient_to_idx = {p: i for i, p in enumerate(patient_index)}
    vital_to_idx = {v: i for i, v in enumerate(vital_index)}

    # VECTORIZED: Create patient and hour index arrays
    print(f"  Creating mask columns (vectorized)...")
    empi_array = wide['EMPI'].values
    hour_array = wide['hour_from_pe'].values.astype(int)

    # Map EMPIs to patient indices (-1 for unknown)
    patient_idx_array = np.array([patient_to_idx.get(empi, -1) for empi in empi_array])

    # Convert hours to tensor indices (hour + 24, with bounds check)
    hour_idx_array = hour_array + 24
    valid_hours = (hour_idx_array >= 0) & (hour_idx_array < imputation_tiers.shape[1])
    valid_patients = patient_idx_array >= 0
    valid_mask = valid_hours & valid_patients

    # Add mask columns for each vital using vectorized indexing
    for vital in tqdm(RAW_VITALS, desc="  Building masks"):
        if vital not in vital_to_idx:
            wide[f'mask_{vital}'] = 0
            continue

        v_idx = vital_to_idx[vital]
        mask_values = np.zeros(len(wide), dtype=np.int8)

        # Use advanced indexing for valid entries
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            p_idxs = patient_idx_array[valid_indices]
            h_idxs = hour_idx_array[valid_indices]
            tiers = imputation_tiers[p_idxs, h_idxs, v_idx]
            mask_values[valid_indices] = (tiers <= 2).astype(np.int8)

        wide[f'mask_{vital}'] = mask_values

    return wide


def process_single_patient(args) -> tuple:
    """Process all features for a single patient.

    This function is designed to be called in parallel.
    Returns (patient_df, summary_row).
    """
    patient_df, vitals, windows, feature_cols = args

    # Calculate all features for this patient
    patient_df = calculate_rolling_stats_patient(patient_df, vitals=vitals, windows=windows)
    patient_df = calculate_trend_features_patient(patient_df, vitals=vitals, windows=windows)
    patient_df = calculate_variability_features_patient(patient_df, vitals=vitals)
    patient_df = calculate_threshold_features_patient(patient_df)
    patient_df = calculate_data_density_patient(patient_df, vitals=vitals)

    # Get feature columns for summary (after all features added)
    if feature_cols is None:
        exclude_cols = {'EMPI', 'hour_from_pe'} | set(vitals) | {f'mask_{v}' for v in vitals}
        feature_cols = [c for c in patient_df.columns if c not in exclude_cols]

    # Aggregate to summary
    summary_row = aggregate_patient_to_summary(patient_df, feature_cols=feature_cols)

    return patient_df, summary_row


def build_layer3(
    layer2_parquet_path: Path,
    layer2_hdf5_path: Path,
    timeseries_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Build Layer 3 features from Layer 2 data.

    OPTIMIZED: Uses multiprocessing for parallel patient processing.

    Args:
        layer2_parquet_path: Path to hourly_grid.parquet
        layer2_hdf5_path: Path to hourly_tensors.h5
        timeseries_output_path: Output path for timeseries_features.parquet
        summary_output_path: Output path for summary_features.parquet
    """
    print("\n" + "=" * 60)
    print("Layer 3 Builder - Feature Engineering (OPTIMIZED)")
    print("=" * 60)
    print(f"CPU cores available: {mp.cpu_count()}, using {N_JOBS} workers")

    # Step 1: Load Layer 2 data
    print("\n[1/3] Loading Layer 2 data...")
    df = load_layer2_with_masks(layer2_parquet_path, layer2_hdf5_path)
    n_patients = df['EMPI'].nunique()
    print(f"  Loaded {len(df):,} rows from {n_patients:,} patients")

    # Step 2: Add composite vitals (vectorized, fast)
    print("\n[2/3] Calculating composite vitals...")
    df = add_composite_vitals(df)

    # Add mask columns for composites
    df['mask_shock_index'] = ((df['mask_HR'] == 1) & (df['mask_SBP'] == 1)).astype(np.int8)
    df['mask_pulse_pressure'] = ((df['mask_SBP'] == 1) & (df['mask_DBP'] == 1)).astype(np.int8)
    print(f"  Added shock_index and pulse_pressure")

    # Step 3: Process all patients in parallel
    print(f"\n[3/3] Processing {n_patients:,} patients in parallel ({N_JOBS} workers)...")
    sys.stdout.flush()

    # Split data by patient using groupby (much faster than filtering)
    print("  Splitting data by patient (using groupby)...")
    sys.stdout.flush()
    grouped = df.groupby('EMPI', sort=False)
    patient_dfs = [group.sort_values('hour_from_pe').copy() for _, group in tqdm(grouped, desc="  Preparing patients", unit="patient", total=n_patients)]

    # Prepare args for parallel processing
    args_list = [(pdf, LAYER3_VITALS, ROLLING_WINDOWS, None) for pdf in patient_dfs]

    # Process in parallel with progress bar
    print("  Starting parallel feature calculation...")
    sys.stdout.flush()
    all_timeseries = []
    all_summaries = []

    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(process_single_patient, args): i for i, args in enumerate(args_list)}

        with tqdm(total=len(futures), desc="  Computing features", unit="patient") as pbar:
            for future in as_completed(futures):
                try:
                    patient_df, summary_row = future.result()
                    all_timeseries.append(patient_df)
                    all_summaries.append(summary_row)
                except Exception as e:
                    print(f"  Warning: Patient processing failed: {e}")
                pbar.update(1)

    # Combine results
    print("\n  Combining results...")
    timeseries_df = pd.concat(all_timeseries, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    # Save outputs
    print(f"\n  Saving time-series features to {timeseries_output_path}")
    timeseries_output_path.parent.mkdir(parents=True, exist_ok=True)
    timeseries_df.to_parquet(timeseries_output_path, index=False)

    print(f"  Saving summary features to {summary_output_path}")
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_output_path, index=False)

    print("\n" + "=" * 60)
    print("Layer 3 COMPLETE")
    print("=" * 60)
    print(f"  Time-series: {len(timeseries_df):,} rows x {len(timeseries_df.columns)} columns")
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
