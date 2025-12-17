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


# VTE History Feature Extraction

def extract_prior_pe_features(diagnoses: pd.DataFrame) -> dict:
    """Extract prior PE features from diagnoses before index PE.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_pe_ever": bool,
            "prior_pe_months": float or None,
            "prior_pe_count": int,
        }
    """
    from config.icd_code_lists import VTE_CODES

    # Filter to prior diagnoses only (before index PE)
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()

    # Find PE diagnoses
    pe_matches = prior[prior.apply(
        lambda row: code_matches_category(row["icd_code"], VTE_CODES["pe"], row["icd_version"]),
        axis=1
    )]

    prior_pe_ever = len(pe_matches) > 0
    prior_pe_count = len(pe_matches)

    prior_pe_months = None
    if prior_pe_ever:
        most_recent_days = pe_matches["days_from_pe"].max()  # max of negatives = most recent
        prior_pe_months = days_to_months(most_recent_days)

    return {
        "prior_pe_ever": prior_pe_ever,
        "prior_pe_months": prior_pe_months,
        "prior_pe_count": prior_pe_count,
    }


def extract_prior_dvt_features(diagnoses: pd.DataFrame) -> dict:
    """Extract prior DVT features from diagnoses before index PE.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_dvt_ever": bool,
            "prior_dvt_months": float or None,
        }
    """
    from config.icd_code_lists import VTE_CODES

    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()

    # Combine DVT lower and upper
    dvt_matches = prior[prior.apply(
        lambda row: (
            code_matches_category(row["icd_code"], VTE_CODES["dvt_lower"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_upper"], row["icd_version"])
        ),
        axis=1
    )]

    prior_dvt_ever = len(dvt_matches) > 0
    prior_dvt_months = None
    if prior_dvt_ever:
        most_recent_days = dvt_matches["days_from_pe"].max()
        prior_dvt_months = days_to_months(most_recent_days)

    return {
        "prior_dvt_ever": prior_dvt_ever,
        "prior_dvt_months": prior_dvt_months,
    }


def extract_vte_history_features(diagnoses: pd.DataFrame) -> dict:
    """Extract all VTE history features.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_pe_ever": bool,
            "prior_pe_months": float or None,
            "prior_pe_count": int,
            "prior_dvt_ever": bool,
            "prior_dvt_months": float or None,
            "prior_vte_count": int,
            "is_recurrent_vte": bool,
        }
    """
    pe_features = extract_prior_pe_features(diagnoses)
    dvt_features = extract_prior_dvt_features(diagnoses)

    # Count all VTE events
    from config.icd_code_lists import VTE_CODES
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()
    vte_matches = prior[prior.apply(
        lambda row: (
            code_matches_category(row["icd_code"], VTE_CODES["pe"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_lower"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_upper"], row["icd_version"])
        ),
        axis=1
    )]

    return {
        **pe_features,
        **dvt_features,
        "prior_vte_count": len(vte_matches),
        "is_recurrent_vte": pe_features["prior_pe_ever"] or dvt_features["prior_dvt_ever"],
    }


def extract_pe_characterization(diagnoses: pd.DataFrame) -> dict:
    """Characterize the index PE from ICD codes.

    PE Subtype Logic (ICD-10-CM):
    - I26.92 → saddle
    - I26.93 → single subsegmental
    - I26.94 → multiple subsegmental
    - I26.99 → other (lobar)
    - I26.90 → unspecified
    - I26.0x → with cor pulmonale (high risk)

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe,
                   is_index_concurrent, is_pe_diagnosis

    Returns:
        {
            "pe_subtype": str,  # 'saddle', 'subsegmental', 'other', 'unspecified'
            "pe_bilateral": bool,
            "pe_with_cor_pulmonale": bool,
            "pe_high_risk_code": bool,
        }
    """
    # Get index PE diagnoses only
    index_dx = get_index_diagnoses(diagnoses)

    # Find PE codes at index
    pe_codes = index_dx[index_dx["is_pe_diagnosis"] == True]["icd_code"].tolist()

    # Determine subtype from codes
    pe_subtype = "unspecified"
    pe_bilateral = False
    pe_with_cor_pulmonale = False
    pe_high_risk_code = False

    for code in pe_codes:
        code_upper = str(code).upper()

        # Saddle PE
        if code_upper.startswith("I26.92"):
            pe_subtype = "saddle"
            pe_high_risk_code = True

        # Subsegmental
        elif code_upper.startswith("I26.93") or code_upper.startswith("I26.94"):
            if pe_subtype not in ["saddle"]:
                pe_subtype = "subsegmental"

        # Other (lobar assumed)
        elif code_upper.startswith("I26.99"):
            if pe_subtype not in ["saddle", "subsegmental"]:
                pe_subtype = "other"

        # With acute cor pulmonale = high risk
        if code_upper.startswith("I26.0"):
            pe_with_cor_pulmonale = True
            pe_high_risk_code = True

    return {
        "pe_subtype": pe_subtype,
        "pe_bilateral": pe_bilateral,  # Would need radiology data, keep as False
        "pe_with_cor_pulmonale": pe_with_cor_pulmonale,
        "pe_high_risk_code": pe_high_risk_code,
    }
