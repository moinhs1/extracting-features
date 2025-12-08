"""Layer 1 Builder: Canonical vital sign records with PE-relative timestamps."""
from typing import Dict
import pandas as pd

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
