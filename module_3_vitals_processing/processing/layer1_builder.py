"""Layer 1 Builder: Canonical vital sign records with PE-relative timestamps."""
from typing import Dict
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd
from processing.temporal_aligner import calculate_hours_from_pe
from processing.qc_filters import is_physiologically_valid, is_abnormal
from processing.temporal_aligner import is_within_window

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


def build_layer1(
    phy_path: Path,
    hnp_path: Path,
    prg_path: Path,
    timeline_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build Layer 1 canonical vitals from all extraction sources.

    Args:
        phy_path: Path to phy_vitals_raw.parquet
        hnp_path: Path to hnp_vitals_raw.parquet
        prg_path: Path to prg_vitals_raw.parquet
        timeline_path: Path to patient_timelines.pkl
        output_path: Path to write canonical_vitals.parquet

    Returns:
        Combined DataFrame with Layer 1 schema
    """
    # Load PE times
    pe_times = load_pe_times(timeline_path)

    # Load and normalize each source
    dfs = []

    if phy_path.exists():
        phy_df = pd.read_parquet(phy_path)
        phy_norm = normalize_phy_source(phy_df)
        dfs.append(phy_norm)

    if hnp_path.exists():
        hnp_df = pd.read_parquet(hnp_path)
        hnp_norm = normalize_hnp_source(hnp_df)
        dfs.append(hnp_norm)

    if prg_path.exists():
        prg_df = pd.read_parquet(prg_path)
        prg_norm = normalize_prg_source(prg_df)
        dfs.append(prg_norm)

    if not dfs:
        raise ValueError("No input files found")

    # Combine all sources
    combined = pd.concat(dfs, ignore_index=True)

    # Add PE-relative timestamps
    combined = add_pe_relative_timestamps(combined, pe_times)

    # Filter to analysis window
    combined = combined[
        combined["hours_from_pe"].apply(is_within_window)
    ]

    # Filter to core vitals only
    combined = combined[combined["vital_type"].isin(CORE_VITALS)]

    # Apply physiological range validation
    valid_mask = combined.apply(
        lambda r: is_physiologically_valid(r["vital_type"], r["value"]),
        axis=1
    )
    combined = combined[valid_mask]

    # Update abnormal flags
    combined["is_flagged_abnormal"] = combined.apply(
        lambda r: is_abnormal(r["vital_type"], r["value"]),
        axis=1
    )

    # Generate calculated MAPs
    maps = generate_calculated_maps(combined)
    if not maps.empty:
        combined = pd.concat([combined, maps], ignore_index=True)

    # Sort by patient and time
    combined = combined.sort_values(["EMPI", "timestamp"]).reset_index(drop=True)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    return combined


def main():
    """CLI entry point for Layer 1 builder."""
    from pathlib import Path

    # Default paths
    base_dir = Path(__file__).parent.parent
    discovery_dir = base_dir / "outputs" / "discovery"
    module1_dir = base_dir.parent / "module_1_core_infrastructure" / "outputs"
    output_dir = base_dir / "outputs" / "layer1"

    phy_path = discovery_dir / "phy_vitals_raw.parquet"
    hnp_path = discovery_dir / "hnp_vitals_raw.parquet"
    prg_path = discovery_dir / "prg_vitals_raw.parquet"
    timeline_path = module1_dir / "patient_timelines.pkl"
    output_path = output_dir / "canonical_vitals.parquet"

    print(f"Building Layer 1 canonical vitals...")
    print(f"  PHY input: {phy_path}")
    print(f"  HNP input: {hnp_path}")
    print(f"  PRG input: {prg_path}")
    print(f"  Timeline: {timeline_path}")
    print(f"  Output: {output_path}")

    result = build_layer1(
        phy_path=phy_path,
        hnp_path=hnp_path,
        prg_path=prg_path,
        timeline_path=timeline_path,
        output_path=output_path
    )

    print(f"\nLayer 1 complete:")
    print(f"  Total records: {len(result):,}")
    print(f"  Patients: {result['EMPI'].nunique():,}")
    print(f"  Vital types: {result['vital_type'].value_counts().to_dict()}")
    print(f"  Sources: {result['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
