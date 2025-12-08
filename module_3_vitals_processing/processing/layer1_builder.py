"""Layer 1 Builder: Canonical vital sign records with PE-relative timestamps."""
from typing import Dict
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd
from processing.temporal_aligner import calculate_hours_from_pe

# Core vital signs for Layer 1-5 processing
CORE_VITALS = ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]

# Layer 1 output schema
LAYER1_SCHEMA: Dict[str, str] = {
    "EMPI": "str",
    "timestamp": "datetime64[ns]",
    "hours_from_pe": "float64",
    "vital_type": "str",
    "value": "float64",
    "units": "str",
    "source": "str",
    "source_detail": "str",
    "confidence": "float64",
    "is_calculated": "bool",
    "is_flagged_abnormal": "bool",
    "report_number": "str",
}


def normalize_phy_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize PHY extraction output to Layer 1 schema.

    PHY data is structured (flowsheet) data with high confidence.

    Args:
        df: PHY vitals dataframe with columns:
            EMPI, timestamp, vital_type, value, units, source,
            encounter_type, encounter_number

    Returns:
        DataFrame with Layer 1 schema columns
    """
    result = df.copy()

    # Map encounter_type to source_detail
    result["source_detail"] = result.get("encounter_type", "")

    # PHY is structured data, highest confidence
    result["confidence"] = 1.0

    # PHY values are direct measurements, not calculated
    result["is_calculated"] = False

    # Will be set by QC filters later
    result["is_flagged_abnormal"] = False

    # PHY doesn't have report_number, use encounter_number or empty
    result["report_number"] = result.get("encounter_number", "")

    # Select only Layer 1 columns
    output_cols = list(LAYER1_SCHEMA.keys())
    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    return result[output_cols]


def normalize_hnp_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HNP extraction output to Layer 1 schema.

    HNP data is extracted from H&P notes (admission vitals).

    Args:
        df: HNP vitals dataframe

    Returns:
        DataFrame with Layer 1 schema columns
    """
    result = df.copy()

    # Map extraction_context to source_detail
    result["source_detail"] = result.get("extraction_context", "")

    # Confidence already exists from extraction
    if "confidence" not in result.columns:
        result["confidence"] = 0.8  # Default for NLP extraction

    # NLP extractions are not calculated
    result["is_calculated"] = False

    # is_flagged_abnormal may already exist
    if "is_flagged_abnormal" not in result.columns:
        result["is_flagged_abnormal"] = False

    # Select only Layer 1 columns
    output_cols = list(LAYER1_SCHEMA.keys())
    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    return result[output_cols]


def normalize_prg_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize PRG extraction output to Layer 1 schema.

    PRG data is extracted from Progress notes (serial inpatient vitals).

    Args:
        df: PRG vitals dataframe

    Returns:
        DataFrame with Layer 1 schema columns
    """
    # PRG has same structure as HNP (plus temp_method which we drop)
    return normalize_hnp_source(df)


def calculate_map(sbp: float, dbp: float) -> float:
    """Calculate Mean Arterial Pressure from SBP and DBP.

    Formula: MAP = DBP + (SBP - DBP) / 3

    Args:
        sbp: Systolic blood pressure
        dbp: Diastolic blood pressure

    Returns:
        Mean arterial pressure
    """
    return dbp + (sbp - dbp) / 3


def generate_calculated_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Generate calculated MAP values from SBP/DBP pairs.

    Finds SBP and DBP measurements at the same timestamp for the same
    patient and generates calculated MAP values.

    Args:
        df: DataFrame with vital measurements

    Returns:
        DataFrame with calculated MAP rows
    """
    # Get SBP and DBP rows
    sbp_df = df[df["vital_type"] == "SBP"].copy()
    dbp_df = df[df["vital_type"] == "DBP"].copy()

    if sbp_df.empty or dbp_df.empty:
        return pd.DataFrame(columns=df.columns)

    # Merge on patient and timestamp
    merged = sbp_df.merge(
        dbp_df[["EMPI", "timestamp", "value"]],
        on=["EMPI", "timestamp"],
        suffixes=("_sbp", "_dbp"),
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame(columns=df.columns)

    # Calculate MAP
    merged["value"] = merged.apply(
        lambda r: calculate_map(r["value_sbp"], r["value_dbp"]),
        axis=1
    )

    # Build MAP rows
    map_df = merged[["EMPI", "timestamp", "hours_from_pe", "source",
                     "source_detail", "confidence", "report_number", "value"]].copy()
    map_df["vital_type"] = "MAP"
    map_df["units"] = "mmHg"
    map_df["is_calculated"] = True
    map_df["is_flagged_abnormal"] = False

    # Reorder columns to match schema
    return map_df[list(LAYER1_SCHEMA.keys())]


def load_pe_times(timeline_path: Path) -> Dict[str, datetime]:
    """Load PE index times from patient timelines pickle.

    Args:
        timeline_path: Path to patient_timelines.pkl

    Returns:
        Dict mapping EMPI to PE timestamp (time_zero)
    """
    with open(timeline_path, "rb") as f:
        timelines = pickle.load(f)

    pe_times = {}
    for patient_id, timeline in timelines.items():
        pe_times[timeline.patient_id] = timeline.time_zero

    return pe_times


def add_pe_relative_timestamps(
    df: pd.DataFrame,
    pe_times: Dict[str, datetime]
) -> pd.DataFrame:
    """Add PE-relative timestamps to vitals dataframe.

    Args:
        df: Vitals dataframe with EMPI and timestamp columns
        pe_times: Dict mapping EMPI to PE timestamp

    Returns:
        DataFrame with hours_from_pe column added.
        Patients without PE time are dropped.
    """
    result = df.copy()

    # Map EMPI to PE time
    result["pe_time"] = result["EMPI"].map(pe_times)

    # Drop patients without PE time
    result = result.dropna(subset=["pe_time"])

    if result.empty:
        result["hours_from_pe"] = pd.Series(dtype=float)
        return result.drop(columns=["pe_time"])

    # Calculate hours from PE
    result["hours_from_pe"] = result.apply(
        lambda r: calculate_hours_from_pe(r["timestamp"], r["pe_time"]),
        axis=1
    )

    return result.drop(columns=["pe_time"])
