"""Layer 1 Builder: Canonical diagnosis records with PE-relative timestamps."""
from typing import Dict
from pathlib import Path
from datetime import datetime
import pandas as pd
from processing.temporal_classifier import (
    calculate_days_from_pe,
    classify_temporal_category,
    get_temporal_flags,
)
from processing.icd_parser import is_pe_diagnosis

# Layer 1 output schema
LAYER1_SCHEMA = {
    "EMPI": "str",
    "diagnosis_date": "datetime64[ns]",
    "days_from_pe": "int64",
    "hours_from_pe": "float64",
    "icd_code": "str",
    "icd_version": "str",
    "diagnosis_description": "str",
    "diagnosis_type": "str",
    "code_position": "int64",
    "encounter_id": "str",
    "encounter_type": "str",
    "temporal_category": "str",
    "is_preexisting": "bool",
    "is_recent_antecedent": "bool",
    "is_index_concurrent": "bool",
    "is_complication": "bool",
    "is_pe_diagnosis": "bool",
}

# Diagnosis flag to type mapping
FLAG_TO_TYPE = {
    "Primary": "principal",
    "Admitting": "admitting",
    "Secondary": "secondary",
    "": "secondary",
}


def add_pe_relative_timing(
    df: pd.DataFrame,
    pe_times: Dict[str, datetime]
) -> pd.DataFrame:
    """Add PE-relative timing columns.

    Args:
        df: DataFrame with EMPI and diagnosis_date
        pe_times: Dict mapping EMPI to PE index datetime

    Returns:
        DataFrame with days_from_pe and hours_from_pe columns
    """
    result = df.copy()

    def calc_days(row):
        pe_time = pe_times.get(str(row["EMPI"]))
        if pe_time is None:
            return None
        return calculate_days_from_pe(row["diagnosis_date"], pe_time)

    def calc_hours(row):
        pe_time = pe_times.get(str(row["EMPI"]))
        if pe_time is None:
            return None
        delta = row["diagnosis_date"] - pe_time
        return delta.total_seconds() / 3600.0

    result["days_from_pe"] = result.apply(calc_days, axis=1)
    result["hours_from_pe"] = result.apply(calc_hours, axis=1)

    return result


def add_temporal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal category and boolean flags.

    Args:
        df: DataFrame with days_from_pe column

    Returns:
        DataFrame with temporal flags added
    """
    result = df.copy()

    result["temporal_category"] = result["days_from_pe"].apply(
        lambda d: classify_temporal_category(int(d)) if pd.notna(d) else None
    )

    # Add boolean flags
    flags_df = result["days_from_pe"].apply(
        lambda d: get_temporal_flags(int(d)) if pd.notna(d) else {
            "is_preexisting": None,
            "is_recent_antecedent": None,
            "is_index_concurrent": None,
            "is_complication": None,
        }
    )
    flags_expanded = pd.DataFrame(flags_df.tolist())
    for col in flags_expanded.columns:
        result[col] = flags_expanded[col]

    return result


def build_layer1(
    df: pd.DataFrame,
    pe_times: Dict[str, datetime],
    min_days: int = -365 * 5,
    max_days: int = 365,
) -> pd.DataFrame:
    """Build Layer 1 canonical diagnosis records.

    Args:
        df: Raw extracted diagnoses
        pe_times: Dict mapping EMPI to PE index datetime
        min_days: Minimum days from PE to include
        max_days: Maximum days from PE to include

    Returns:
        DataFrame with Layer 1 schema
    """
    # Add PE-relative timing
    result = add_pe_relative_timing(df, pe_times)

    # Filter to temporal window
    result = result[
        (result["days_from_pe"] >= min_days) &
        (result["days_from_pe"] <= max_days)
    ].copy()

    # Add temporal flags
    result = add_temporal_flags(result)

    # Add PE diagnosis flag
    result["is_pe_diagnosis"] = result.apply(
        lambda r: is_pe_diagnosis(r["icd_code"], r["icd_version"]),
        axis=1
    )

    # Map columns to Layer 1 schema
    result["diagnosis_description"] = result.get("Diagnosis_Name", "")
    result["diagnosis_type"] = result.get("Diagnosis_Flag", "").map(
        lambda x: FLAG_TO_TYPE.get(x, "secondary")
    )
    result["code_position"] = result.get("Diagnosis_Flag", "").apply(
        lambda x: 1 if x == "Primary" else 2 if x == "Admitting" else 3
    )
    result["encounter_id"] = result.get("Encounter_number", "")
    result["encounter_type"] = result.get("Inpatient_Outpatient", "").apply(
        lambda x: x.lower() if x else "unknown"
    )

    # Select and order columns
    output_cols = list(LAYER1_SCHEMA.keys())
    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    return result[output_cols].reset_index(drop=True)
