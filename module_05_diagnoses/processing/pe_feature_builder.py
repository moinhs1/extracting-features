"""PE (Pulmonary Embolism) feature extraction builder.

This module provides utilities for extracting PE-related features from diagnosis codes.
"""
from typing import Dict, Optional
import pandas as pd


def code_matches_category(code: str, category_codes: Dict[str, list], version: str) -> bool:
    """Check if ICD code matches a category definition.

    Uses prefix matching to determine if a given ICD code belongs to a category.
    For example, code "I26.99" matches category with prefix "I26".

    Args:
        code: ICD code (e.g., "I26.99")
        category_codes: Dict with "icd10" and "icd9" lists of prefixes
        version: "9" or "10" indicating ICD version

    Returns:
        True if code matches any prefix in the category for the given version

    Example:
        >>> category = {"icd10": ["I26"], "icd9": ["415"]}
        >>> code_matches_category("I26.99", category, "10")
        True
        >>> code_matches_category("415.19", category, "9")
        True
        >>> code_matches_category("I50.9", category, "10")
        False
    """
    # Get the appropriate code list for this ICD version
    code_list = category_codes.get(f"icd{version}", [])

    # Normalize code for comparison
    code = str(code).upper()

    # Check if code starts with any prefix in the category
    for prefix in code_list:
        if code.startswith(prefix.upper()):
            return True

    return False


# Temporal Filter Helpers

def get_preexisting_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses before PE (is_preexisting=True OR is_recent_antecedent=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only preexisting or recent antecedent diagnoses
    """
    return diagnoses[
        (diagnoses["is_preexisting"] == True) |
        (diagnoses["is_recent_antecedent"] == True)
    ].copy()


def get_complication_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses after PE (is_complication=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only complication diagnoses
    """
    return diagnoses[diagnoses["is_complication"] == True].copy()


def get_index_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses at PE presentation (is_index_concurrent=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only index concurrent diagnoses
    """
    return diagnoses[diagnoses["is_index_concurrent"] == True].copy()


# Time-Based Helpers

def days_to_months(days: int) -> float:
    """Convert days to months (30.44 days/month average).

    Uses the average number of days per month (365.25 / 12 = 30.44).
    Handles negative days (before PE) by converting to positive months.

    Args:
        days: Number of days (can be negative for dates before PE)

    Returns:
        Number of months (always positive)

    Example:
        >>> days_to_months(365)
        11.99
        >>> days_to_months(-180)
        5.91
    """
    return abs(days) / 30.44


def get_most_recent_prior(diagnoses: pd.DataFrame, category_codes: dict) -> Optional[int]:
    """Get days_from_pe of most recent matching diagnosis before PE.

    Args:
        diagnoses: DataFrame with icd_code, icd_version, days_from_pe columns
        category_codes: Dict with "icd10" and "icd9" lists

    Returns:
        Negative int (days before PE) or None if no match

    Example:
        If a patient has PE codes at -365 and -30 days from index,
        returns -30 (most recent).
    """
    # Filter to only prior diagnoses (negative days_from_pe)
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()
    if prior.empty:
        return None

    # Find matches using code_matches_category
    matches = prior[prior.apply(
        lambda row: code_matches_category(row["icd_code"], category_codes, row["icd_version"]),
        axis=1
    )]

    if matches.empty:
        return None

    # Most recent = closest to 0 (max of negative numbers)
    return int(matches["days_from_pe"].max())
