"""Layer 2 Builder: Hourly aggregated grid with missing data tensors."""
from typing import Dict, List
from itertools import product
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
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


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vitals to hourly buckets with statistics.

    Args:
        df: DataFrame with columns:
            - EMPI: patient identifier
            - hours_from_pe: float hours relative to PE
            - vital_type: vital sign type
            - value: measured value

    Returns:
        DataFrame with columns:
            - EMPI: patient identifier
            - hour_from_pe: integer hour bucket
            - vital_type: vital sign type
            - mean, median, std, min, max: statistics
            - count: number of observations
            - mask: 1 for all observed rows
    """
    # Assign hour buckets
    df = df.copy()
    df["hour_from_pe"] = df["hours_from_pe"].apply(assign_hour_bucket)

    # Group by patient, hour bucket, and vital type
    grouped = df.groupby(["EMPI", "hour_from_pe", "vital_type"])["value"]

    # Calculate statistics
    result = pd.DataFrame({
        "mean": grouped.mean(),
        "median": grouped.median(),
        "std": grouped.std(),
        "min": grouped.min(),
        "max": grouped.max(),
        "count": grouped.count(),
    }).reset_index()

    # Add mask column (1 for all observed rows)
    result["mask"] = 1

    return result


def create_full_grid(
    observed: pd.DataFrame,
    patients: List[str]
) -> pd.DataFrame:
    """Create full hourly grid with all patient-hour-vital combinations.

    Args:
        observed: DataFrame with observed hourly aggregations
        patients: List of patient EMPIs to include

    Returns:
        DataFrame with all combinations, missing marked with mask=0
    """
    # Create full index of all combinations
    full_index = pd.DataFrame(
        list(product(patients, HOUR_RANGE, VITAL_ORDER)),
        columns=["EMPI", "hour_from_pe", "vital_type"]
    )

    # Merge with observed data
    result = full_index.merge(
        observed,
        on=["EMPI", "hour_from_pe", "vital_type"],
        how="left"
    )

    # Fill missing mask values with 0
    result["mask"] = result["mask"].fillna(0).astype("int8")

    # Fill missing count with 0
    result["count"] = result["count"].fillna(0).astype("int32")

    return result


def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply three-tier imputation to hourly grid.

    Tiers:
        1: Observed (mask=1)
        2: Forward-filled within vital-specific limit
        3: Patient mean (when forward-fill exhausted)
        4: Cohort mean (when patient has no observations)

    Args:
        df: Full hourly grid from create_full_grid()

    Returns:
        DataFrame with imputed values and imputation_tier column
    """
    result = df.copy()
    result["imputation_tier"] = 0

    # Tier 1: Observed values
    result.loc[result["mask"] == 1, "imputation_tier"] = 1

    # Process each patient-vital combination
    for (empi, vital), group in result.groupby(["EMPI", "vital_type"]):
        group_idx = group.index
        group = group.sort_values("hour_from_pe")

        # Get forward-fill limit for this vital
        ff_limit = FORWARD_FILL_LIMITS.get(vital, 6)

        # Calculate patient mean from observed values
        observed_vals = group.loc[group["mask"] == 1, "mean"]
        patient_mean = observed_vals.mean() if len(observed_vals) > 0 else np.nan

        # Track last observed hour for forward-fill
        last_observed_hour = None
        last_observed_value = None

        for idx, row in group.iterrows():
            if row["mask"] == 1:
                # Observed: update last observed
                last_observed_hour = row["hour_from_pe"]
                last_observed_value = row["mean"]
            else:
                # Missing: determine imputation tier
                if (last_observed_hour is not None and
                    row["hour_from_pe"] - last_observed_hour <= ff_limit):
                    # Tier 2: Forward-fill
                    result.loc[idx, "mean"] = last_observed_value
                    result.loc[idx, "imputation_tier"] = 2
                elif not pd.isna(patient_mean):
                    # Tier 3: Patient mean
                    result.loc[idx, "mean"] = patient_mean
                    result.loc[idx, "imputation_tier"] = 3
                else:
                    # Tier 4: Will be filled with cohort mean later
                    result.loc[idx, "imputation_tier"] = 4

    return result


def create_hdf5_tensors(grid: pd.DataFrame, output_path: Path) -> None:
    """Create HDF5 file with tensor representation of hourly grid.

    Creates tensors:
        - values: (n_patients, 745, 7) float32 - vital values
        - masks: (n_patients, 745, 7) int8 - 1=observed, 0=imputed
        - imputation_tier: (n_patients, 745, 7) int8 - tier 1-4
        - patient_index: (n_patients,) str - EMPI mapping
        - vital_index: (7,) str - vital names
        - hour_index: (745,) int - hour values

    Args:
        grid: Full imputed grid from apply_imputation()
        output_path: Path to write HDF5 file
    """
    # Get unique patients in sorted order
    patients = sorted(grid["EMPI"].unique())
    n_patients = len(patients)
    n_hours = len(HOUR_RANGE)
    n_vitals = len(VITAL_ORDER)

    # Create patient to index mapping
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    vital_to_idx = {v: i for i, v in enumerate(VITAL_ORDER)}
    hour_to_idx = {h: i for i, h in enumerate(HOUR_RANGE)}

    # Initialize tensors
    values = np.full((n_patients, n_hours, n_vitals), np.nan, dtype=np.float32)
    masks = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)
    imputation_tiers = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)

    # Fill tensors from grid
    for _, row in grid.iterrows():
        p_idx = patient_to_idx[row["EMPI"]]
        h_idx = hour_to_idx[row["hour_from_pe"]]
        v_idx = vital_to_idx[row["vital_type"]]

        values[p_idx, h_idx, v_idx] = row["mean"]
        masks[p_idx, h_idx, v_idx] = row["mask"]
        imputation_tiers[p_idx, h_idx, v_idx] = row["imputation_tier"]

    # Write HDF5 file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("values", data=values, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")
        f.create_dataset("imputation_tier", data=imputation_tiers, compression="gzip")

        # String arrays
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("patient_index", data=np.array(patients, dtype=object), dtype=dt)
        f.create_dataset("vital_index", data=np.array(VITAL_ORDER, dtype=object), dtype=dt)
        f.create_dataset("hour_index", data=np.array(HOUR_RANGE, dtype=np.int32))


def calculate_time_deltas(masks: np.ndarray) -> np.ndarray:
    """Calculate time since last observation for each cell.

    Args:
        masks: (n_patients, n_hours, n_vitals) int8 array
               1=observed, 0=missing

    Returns:
        (n_patients, n_hours, n_vitals) float32 array of hours since last observation
    """
    n_patients, n_hours, n_vitals = masks.shape
    time_deltas = np.zeros_like(masks, dtype=np.float32)

    for p in range(n_patients):
        for v in range(n_vitals):
            last_obs_hour = -1
            for h in range(n_hours):
                if masks[p, h, v] == 1:
                    # Observed: delta is 0, update last observed
                    time_deltas[p, h, v] = 0
                    last_obs_hour = h
                else:
                    # Missing: calculate delta
                    if last_obs_hour >= 0:
                        time_deltas[p, h, v] = h - last_obs_hour
                    else:
                        # No prior observation
                        time_deltas[p, h, v] = h + 24 + 1  # Hours from window start

    return time_deltas
