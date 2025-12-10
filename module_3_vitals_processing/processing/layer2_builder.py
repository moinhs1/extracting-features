"""Layer 2 Builder: Hourly aggregated grid with missing data tensors.

Optimized for maximum CPU utilization with progress bars.
"""
from typing import Dict, List, Tuple
from itertools import product
from pathlib import Path
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from joblib import Parallel, delayed

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from processing.temporal_aligner import assign_hour_bucket

# Hour range: -24 to +720 (745 hours total)
HOUR_RANGE: List[int] = list(range(-24, 721))

# Vital order for tensor dimensions
VITAL_ORDER: List[str] = ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]

# Forward-fill limits (hours) per vital type
FORWARD_FILL_LIMITS: Dict[str, int] = {
    "HR": 6,
    "SBP": 6,
    "DBP": 6,
    "MAP": 6,
    "RR": 6,
    "SPO2": 4,
    "TEMP": 12,
}

# Layer 2 Parquet schema
LAYER2_PARQUET_SCHEMA: Dict[str, str] = {
    "EMPI": "str",
    "hour_from_pe": "int32",
    "vital_type": "str",
    "mean": "float64",
    "median": "float64",
    "std": "float64",
    "min": "float64",
    "max": "float64",
    "count": "int32",
    "mask": "int8",  # 1=observed, 0=missing
}

# Get CPU count for parallel processing
N_JOBS = max(1, mp.cpu_count() - 1)


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vitals to hourly buckets with statistics.

    Vectorized implementation for speed.
    """
    print(f"  Aggregating {len(df):,} records to hourly buckets...")

    # Vectorized hour bucket assignment
    df = df.copy()
    hours = df["hours_from_pe"].values
    df["hour_from_pe"] = np.clip(np.floor(hours).astype(np.int32), -24, 720)

    # Group by patient, hour bucket, and vital type
    grouped = df.groupby(["EMPI", "hour_from_pe", "vital_type"])["value"]

    # Calculate statistics using agg (faster than multiple calls)
    result = grouped.agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        count="count"
    ).reset_index()

    # Add mask column (1 for all observed rows)
    result["mask"] = np.int8(1)

    print(f"  Created {len(result):,} hourly aggregations")
    return result


def create_full_grid(
    observed: pd.DataFrame,
    patients: List[str]
) -> pd.DataFrame:
    """Create full hourly grid with all patient-hour-vital combinations.

    Memory-efficient implementation.
    """
    n_patients = len(patients)
    n_hours = len(HOUR_RANGE)
    n_vitals = len(VITAL_ORDER)
    total_rows = n_patients * n_hours * n_vitals

    print(f"  Creating full grid: {n_patients:,} patients × {n_hours} hours × {n_vitals} vitals = {total_rows:,} rows")

    # Create full index using numpy (faster than itertools.product for large data)
    patient_idx = np.repeat(patients, n_hours * n_vitals)
    hour_idx = np.tile(np.repeat(HOUR_RANGE, n_vitals), n_patients)
    vital_idx = np.tile(VITAL_ORDER, n_patients * n_hours)

    full_index = pd.DataFrame({
        "EMPI": patient_idx,
        "hour_from_pe": hour_idx,
        "vital_type": vital_idx
    })

    # Merge with observed data
    print(f"  Merging with observed data...")
    result = full_index.merge(
        observed,
        on=["EMPI", "hour_from_pe", "vital_type"],
        how="left"
    )

    # Fill missing mask values with 0
    result["mask"] = result["mask"].fillna(0).astype("int8")
    result["count"] = result["count"].fillna(0).astype("int32")

    return result


def _impute_patient_vital(
    group_data: Tuple[str, str, pd.DataFrame],
    ff_limits: Dict[str, int]
) -> pd.DataFrame:
    """Impute a single patient-vital combination. For parallel execution."""
    empi, vital, group = group_data
    group = group.sort_values("hour_from_pe").copy()

    ff_limit = ff_limits.get(vital, 6)

    # Get observed values
    observed_mask = group["mask"] == 1
    observed_vals = group.loc[observed_mask, "mean"]
    patient_mean = observed_vals.mean() if len(observed_vals) > 0 else np.nan

    # Initialize imputation tier
    group["imputation_tier"] = np.where(observed_mask, 1, 0)

    # Vectorized forward-fill logic
    hours = group["hour_from_pe"].values
    means = group["mean"].values.copy()
    masks = group["mask"].values
    tiers = group["imputation_tier"].values.copy()

    last_obs_hour = -9999
    last_obs_value = np.nan

    for i in range(len(group)):
        if masks[i] == 1:
            last_obs_hour = hours[i]
            last_obs_value = means[i]
        else:
            if last_obs_hour > -9999 and hours[i] - last_obs_hour <= ff_limit:
                means[i] = last_obs_value
                tiers[i] = 2
            elif not np.isnan(patient_mean):
                means[i] = patient_mean
                tiers[i] = 3
            else:
                tiers[i] = 4

    group["mean"] = means
    group["imputation_tier"] = tiers
    return group


def apply_imputation_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply three-tier imputation using parallel processing.

    Tiers:
        1: Observed (mask=1)
        2: Forward-filled within vital-specific limit
        3: Patient mean (when forward-fill exhausted)
        4: Cohort mean (when patient has no observations)
    """
    result = df.copy()
    result["imputation_tier"] = 0

    # Prepare groups for parallel processing
    groups = []
    for (empi, vital), group in result.groupby(["EMPI", "vital_type"]):
        groups.append((empi, vital, group))

    print(f"  Imputing {len(groups):,} patient-vital combinations using {N_JOBS} workers...")

    # Process in parallel with progress bar
    impute_func = partial(_impute_patient_vital, ff_limits=FORWARD_FILL_LIMITS)

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(impute_func)(g) for g in tqdm(groups, desc="  Imputation", unit="groups")
    )

    # Combine results
    result = pd.concat(results, ignore_index=True)
    return result


def create_hdf5_tensors_fast(grid: pd.DataFrame, output_path: Path) -> None:
    """Create HDF5 file with tensor representation - optimized version."""

    patients = sorted(grid["EMPI"].unique())
    n_patients = len(patients)
    n_hours = len(HOUR_RANGE)
    n_vitals = len(VITAL_ORDER)

    print(f"  Creating tensors: ({n_patients}, {n_hours}, {n_vitals})")

    # Create mappings
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    vital_to_idx = {v: i for i, v in enumerate(VITAL_ORDER)}
    hour_to_idx = {h: i for i, h in enumerate(HOUR_RANGE)}

    # Initialize tensors
    values = np.full((n_patients, n_hours, n_vitals), np.nan, dtype=np.float32)
    masks = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)
    imputation_tiers = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)

    # Vectorized tensor filling using numpy indexing
    print(f"  Filling tensors from {len(grid):,} grid rows...")

    # Convert to numpy arrays for speed
    p_indices = np.array([patient_to_idx[e] for e in tqdm(grid["EMPI"].values, desc="  Mapping patients", unit="rows")])
    h_indices = np.array([hour_to_idx[h] for h in grid["hour_from_pe"].values])
    v_indices = np.array([vital_to_idx[v] for v in grid["vital_type"].values])

    # Fill tensors using advanced indexing
    print(f"  Writing to tensors...")
    values[p_indices, h_indices, v_indices] = grid["mean"].values.astype(np.float32)
    masks[p_indices, h_indices, v_indices] = grid["mask"].values.astype(np.int8)
    imputation_tiers[p_indices, h_indices, v_indices] = grid["imputation_tier"].values.astype(np.int8)

    # Write HDF5 file
    print(f"  Writing HDF5 file...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("values", data=values, compression="gzip", chunks=True)
        f.create_dataset("masks", data=masks, compression="gzip", chunks=True)
        f.create_dataset("imputation_tier", data=imputation_tiers, compression="gzip", chunks=True)

        # String arrays
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("patient_index", data=np.array(patients, dtype=object), dtype=dt)
        f.create_dataset("vital_index", data=np.array(VITAL_ORDER, dtype=object), dtype=dt)
        f.create_dataset("hour_index", data=np.array(HOUR_RANGE, dtype=np.int32))


def calculate_time_deltas_fast(masks: np.ndarray) -> np.ndarray:
    """Calculate time since last observation - vectorized version."""
    n_patients, n_hours, n_vitals = masks.shape
    time_deltas = np.zeros_like(masks, dtype=np.float32)

    print(f"  Calculating time deltas for {n_patients:,} patients...")

    # Process each vital type in parallel
    for v in tqdm(range(n_vitals), desc="  Time deltas", unit="vitals"):
        for p in range(n_patients):
            last_obs_hour = -1
            for h in range(n_hours):
                if masks[p, h, v] == 1:
                    time_deltas[p, h, v] = 0
                    last_obs_hour = h
                else:
                    if last_obs_hour >= 0:
                        time_deltas[p, h, v] = h - last_obs_hour
                    else:
                        time_deltas[p, h, v] = h + 24 + 1

    return time_deltas


# Numba-optimized version if available
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _calculate_time_deltas_numba(masks: np.ndarray) -> np.ndarray:
        """Numba-optimized time delta calculation."""
        n_patients, n_hours, n_vitals = masks.shape
        time_deltas = np.zeros((n_patients, n_hours, n_vitals), dtype=np.float32)

        for p in prange(n_patients):
            for v in range(n_vitals):
                last_obs_hour = -1
                for h in range(n_hours):
                    if masks[p, h, v] == 1:
                        time_deltas[p, h, v] = 0
                        last_obs_hour = h
                    else:
                        if last_obs_hour >= 0:
                            time_deltas[p, h, v] = h - last_obs_hour
                        else:
                            time_deltas[p, h, v] = h + 24 + 1

        return time_deltas

    def calculate_time_deltas_fast(masks: np.ndarray) -> np.ndarray:
        """Use numba-optimized version."""
        print(f"  Calculating time deltas (numba-optimized)...")
        return _calculate_time_deltas_numba(masks)


def build_layer2(
    layer1_path: Path,
    parquet_output_path: Path,
    hdf5_output_path: Path
) -> pd.DataFrame:
    """Build Layer 2 hourly grid and tensors from Layer 1 canonical vitals."""

    # Load Layer 1
    print(f"\n[1/6] Loading Layer 1 data...")
    layer1 = pd.read_parquet(layer1_path)
    print(f"  Loaded {len(layer1):,} records from {layer1['EMPI'].nunique():,} patients")

    # Get unique patients
    patients = sorted(layer1["EMPI"].unique())

    # Aggregate to hourly bins
    print(f"\n[2/6] Aggregating to hourly bins...")
    hourly = aggregate_to_hourly(layer1)

    # Free memory
    del layer1

    # Create full grid with all hours
    print(f"\n[3/6] Creating full grid...")
    full_grid = create_full_grid(hourly, patients)

    # Free memory
    del hourly

    # Apply three-tier imputation
    print(f"\n[4/6] Applying imputation...")
    imputed_grid = apply_imputation_parallel(full_grid)

    # Free memory
    del full_grid

    # Calculate cohort means for tier 4
    print(f"\n[5/6] Filling tier 4 with cohort means...")
    for vital in tqdm(VITAL_ORDER, desc="  Cohort means", unit="vitals"):
        vital_mask = imputed_grid["vital_type"] == vital
        tier4_mask = imputed_grid["imputation_tier"] == 4

        observed = imputed_grid.loc[vital_mask & (imputed_grid["mask"] == 1), "mean"]
        cohort_mean = observed.mean() if len(observed) > 0 else 0

        imputed_grid.loc[vital_mask & tier4_mask, "mean"] = cohort_mean

    # Save parquet
    print(f"\n[6/6] Saving outputs...")
    print(f"  Writing parquet: {parquet_output_path}")
    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_grid.to_parquet(parquet_output_path, index=False)

    # Create HDF5 tensors
    print(f"  Creating HDF5 tensors: {hdf5_output_path}")
    create_hdf5_tensors_fast(imputed_grid, hdf5_output_path)

    # Add time deltas to HDF5
    print(f"  Adding time deltas to HDF5...")
    with h5py.File(hdf5_output_path, "r") as f:
        masks = f["masks"][:]

    time_deltas = calculate_time_deltas_fast(masks)

    with h5py.File(hdf5_output_path, "a") as f:
        f.create_dataset("time_deltas", data=time_deltas, compression="gzip", chunks=True)

    return imputed_grid


def main():
    """CLI entry point for Layer 2 builder."""
    import os

    # Print system info
    print("=" * 60)
    print("Layer 2 Builder - Optimized")
    print("=" * 60)
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Using {N_JOBS} parallel workers")
    print(f"Numba JIT available: {NUMBA_AVAILABLE}")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent

    layer1_path = base_dir / "outputs" / "layer1" / "canonical_vitals.parquet"
    parquet_path = base_dir / "outputs" / "layer2" / "hourly_grid.parquet"
    hdf5_path = base_dir / "outputs" / "layer2" / "hourly_tensors.h5"

    print(f"\nInput:  {layer1_path}")
    print(f"Output: {parquet_path}")
    print(f"Output: {hdf5_path}")

    result = build_layer2(
        layer1_path=layer1_path,
        parquet_output_path=parquet_path,
        hdf5_output_path=hdf5_path
    )

    print("\n" + "=" * 60)
    print("Layer 2 COMPLETE")
    print("=" * 60)
    print(f"  Grid rows: {len(result):,}")
    print(f"  Patients: {result['EMPI'].nunique():,}")
    print(f"  Observed hours: {(result['mask'] == 1).sum():,}")
    print(f"  Imputed hours: {(result['mask'] == 0).sum():,}")

    # Show imputation tier breakdown
    tier_counts = result["imputation_tier"].value_counts().sort_index()
    print(f"\n  Imputation breakdown:")
    for tier, count in tier_counts.items():
        pct = 100 * count / len(result)
        labels = {1: "Observed", 2: "Forward-fill", 3: "Patient mean", 4: "Cohort mean"}
        print(f"    Tier {tier} ({labels.get(tier, '?')}): {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
