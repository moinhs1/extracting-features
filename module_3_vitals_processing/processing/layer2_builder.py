"""Layer 2 Builder: Hourly aggregated grid with missing data tensors."""
from typing import Dict, List
from itertools import product
import pandas as pd
import numpy as np
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
