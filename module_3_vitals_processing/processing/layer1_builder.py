"""Layer 1 Builder: Canonical vital sign records with PE-relative timestamps."""
from typing import Dict

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
