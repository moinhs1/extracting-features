"""Temporal alignment functions for PE-relative timestamps."""
from datetime import datetime
from typing import Union
import pandas as pd


# Default temporal window (hours relative to PE)
DEFAULT_WINDOW_MIN = -24   # 24 hours before PE
DEFAULT_WINDOW_MAX = 720   # 30 days after PE


def calculate_hours_from_pe(
    vital_time: Union[datetime, pd.Timestamp],
    pe_time: Union[datetime, pd.Timestamp]
) -> float:
    """Calculate hours from PE index time.

    Args:
        vital_time: Timestamp of vital measurement
        pe_time: PE index/diagnosis timestamp (time_zero)

    Returns:
        Hours relative to PE (negative = before, positive = after)
    """
    delta = vital_time - pe_time
    return delta.total_seconds() / 3600.0


def is_within_window(
    hours_from_pe: float,
    min_hours: float = DEFAULT_WINDOW_MIN,
    max_hours: float = DEFAULT_WINDOW_MAX
) -> bool:
    """Check if timestamp is within analysis window.

    Args:
        hours_from_pe: Hours relative to PE index
        min_hours: Window start (inclusive)
        max_hours: Window end (inclusive)

    Returns:
        True if within window, False otherwise
    """
    return min_hours <= hours_from_pe <= max_hours
