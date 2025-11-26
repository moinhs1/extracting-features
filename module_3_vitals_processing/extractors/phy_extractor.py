"""
Submodule 3.1: Structured Vitals Extractor (Phy.txt)
====================================================

Extracts vital signs from the structured Phy.txt file.
"""
import re
from typing import Tuple, Optional


def parse_blood_pressure(bp_string: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse blood pressure string like '130/77' into (systolic, diastolic).

    Args:
        bp_string: Blood pressure string in format 'SBP/DBP'

    Returns:
        Tuple of (systolic, diastolic) as floats, or (None, None) if invalid
    """
    if bp_string is None or not isinstance(bp_string, str):
        return None, None

    bp_string = bp_string.strip()
    if not bp_string:
        return None, None

    # Pattern: digits / digits (with optional spaces)
    pattern = r'^(\d+)\s*/\s*(\d+)$'
    match = re.match(pattern, bp_string)

    if not match:
        return None, None

    try:
        sbp = float(match.group(1))
        dbp = float(match.group(2))
        return sbp, dbp
    except (ValueError, TypeError):
        return None, None
