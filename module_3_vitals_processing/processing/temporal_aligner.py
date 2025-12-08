"""Temporal alignment functions for PE-relative timestamps."""
from datetime import datetime
from typing import Union
import pandas as pd


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
