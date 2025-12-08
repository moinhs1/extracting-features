"""Quality control filters for vital signs."""
from typing import Dict, Tuple

# Physiological ranges: (min_valid, max_valid)
# Values outside these are impossible/data errors
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "HR": (20, 300),      # Heart rate: 20-300 bpm
    "SBP": (40, 300),     # Systolic BP: 40-300 mmHg
    "DBP": (20, 200),     # Diastolic BP: 20-200 mmHg
    "MAP": (30, 200),     # Mean arterial pressure: 30-200 mmHg
    "RR": (4, 60),        # Respiratory rate: 4-60 breaths/min
    "SPO2": (50, 100),    # Oxygen saturation: 50-100%
    "TEMP": (30, 45),     # Temperature: 30-45°C
}

# Abnormal thresholds for flagging (not removal)
ABNORMAL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "HR": (60, 100),      # Normal: 60-100 bpm
    "SBP": (90, 180),     # Normal: 90-180 mmHg
    "DBP": (60, 110),     # Normal: 60-110 mmHg
    "MAP": (65, 110),     # Normal: 65-110 mmHg
    "RR": (12, 24),       # Normal: 12-24 breaths/min
    "SPO2": (92, 100),    # Normal: 92-100%
    "TEMP": (36, 38.5),   # Normal: 36-38.5°C
}


def is_physiologically_valid(vital_type: str, value: float) -> bool:
    """Check if vital sign value is within possible physiological range.

    Args:
        vital_type: Type of vital (HR, SBP, DBP, MAP, RR, SPO2, TEMP)
        value: Measurement value

    Returns:
        True if value is physiologically possible, False if impossible
    """
    if vital_type not in VALID_RANGES:
        return True  # Unknown vital type, don't filter

    min_val, max_val = VALID_RANGES[vital_type]
    return min_val <= value <= max_val


def is_abnormal(vital_type: str, value: float) -> bool:
    """Check if vital sign value is outside normal clinical range.

    Args:
        vital_type: Type of vital (HR, SBP, DBP, MAP, RR, SPO2, TEMP)
        value: Measurement value

    Returns:
        True if value is clinically abnormal, False if normal
    """
    if vital_type not in ABNORMAL_THRESHOLDS:
        return False  # Unknown vital type, don't flag

    min_normal, max_normal = ABNORMAL_THRESHOLDS[vital_type]
    return value < min_normal or value > max_normal


def is_bp_consistent(sbp: float, dbp: float) -> bool:
    """Check if systolic and diastolic BP values are consistent.

    Args:
        sbp: Systolic blood pressure
        dbp: Diastolic blood pressure

    Returns:
        True if SBP > DBP (physiologically consistent), False otherwise
    """
    return sbp > dbp
