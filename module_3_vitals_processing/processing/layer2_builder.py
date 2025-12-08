"""Layer 2 Builder: Hourly aggregated grid with missing data tensors."""
from typing import Dict, List
from itertools import product
import pandas as pd
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
