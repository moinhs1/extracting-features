"""Layer 2 Builder: Hourly aggregated grid with missing data tensors."""
from typing import Dict, List

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
