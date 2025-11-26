"""
Submodule 3.1: Structured Vitals Extractor (Phy.txt)
====================================================

Extracts vital signs from the structured Phy.txt file.
"""
import re
from typing import Tuple, Optional

from module_3_vitals_processing.config.vitals_config import VITAL_CONCEPTS


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


def map_concept_to_canonical(concept_name: Optional[str]) -> Optional[str]:
    """
    Map Phy.txt Concept_Name to canonical vital sign name.

    Args:
        concept_name: The Concept_Name from Phy.txt

    Returns:
        Canonical vital name (HR, SBP, DBP, etc.) or None if not a vital
    """
    if concept_name is None:
        return None

    return VITAL_CONCEPTS.get(concept_name)


def parse_result_value(result: Optional[str]) -> Optional[float]:
    """
    Parse Result field to numeric value.

    Args:
        result: The Result string from Phy.txt

    Returns:
        Float value or None if not parseable
    """
    if result is None or not isinstance(result, str):
        return None

    result = result.strip()
    if not result:
        return None

    # Remove common prefixes like > or <
    result = result.lstrip('<>').strip()

    try:
        return float(result)
    except (ValueError, TypeError):
        return None
