"""Temporal classification for diagnoses relative to PE index."""
from datetime import datetime
from typing import Dict, Union
import pandas as pd


def calculate_days_from_pe(
    diagnosis_date: Union[datetime, pd.Timestamp],
    pe_date: Union[datetime, pd.Timestamp]
) -> int:
    """Calculate days from PE index.

    Args:
        diagnosis_date: Date of diagnosis
        pe_date: PE index date

    Returns:
        Days relative to PE (negative = before, positive = after)
    """
    delta = diagnosis_date - pe_date
    return delta.days


def classify_temporal_category(days_from_pe: int) -> str:
    """Classify diagnosis by temporal relationship to PE.

    Args:
        days_from_pe: Days relative to PE index

    Returns:
        Temporal category string
    """
    if days_from_pe < -30:
        return "preexisting_remote"
    elif days_from_pe < -7:
        return "preexisting_recent"
    elif days_from_pe < 0:
        return "antecedent"
    elif days_from_pe <= 1:
        return "index_concurrent"
    elif days_from_pe <= 7:
        return "early_complication"
    elif days_from_pe <= 30:
        return "late_complication"
    else:
        return "follow_up"


def get_temporal_flags(days_from_pe: int) -> Dict[str, bool]:
    """Generate boolean temporal flags.

    Args:
        days_from_pe: Days relative to PE index

    Returns:
        Dictionary of temporal flags
    """
    return {
        "is_preexisting": days_from_pe < -30,
        "is_recent_antecedent": -30 <= days_from_pe < 0,
        "is_index_concurrent": 0 <= days_from_pe <= 1,
        "is_complication": days_from_pe > 1,
    }
