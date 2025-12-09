"""Composite vital sign calculations.

Composites:
- shock_index: HR / SBP (hemodynamic instability indicator)
- pulse_pressure: SBP - DBP (cardiac output indicator)
"""
import pandas as pd
import numpy as np


def calculate_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate shock index (HR / SBP).

    Normal: 0.5-0.7
    Elevated (>0.9): Indicates hemodynamic compromise

    Args:
        df: DataFrame with HR and SBP columns

    Returns:
        DataFrame with shock_index column added
    """
    result = df.copy()

    # Avoid division by zero - replace 0 with NaN
    sbp_safe = result['SBP'].replace(0, np.nan)

    result['shock_index'] = result['HR'] / sbp_safe
    return result


def calculate_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pulse pressure (SBP - DBP).

    Normal: 40-60 mmHg
    Narrow (<25): Poor cardiac output
    Wide (>60): Possible aortic regurgitation

    Args:
        df: DataFrame with SBP and DBP columns

    Returns:
        DataFrame with pulse_pressure column added
    """
    result = df.copy()
    result['pulse_pressure'] = result['SBP'] - result['DBP']
    return result


def add_composite_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Add both composite vital columns to DataFrame.

    Args:
        df: DataFrame with HR, SBP, DBP columns

    Returns:
        DataFrame with shock_index and pulse_pressure columns added
    """
    result = calculate_shock_index(df)
    result = calculate_pulse_pressure(result)
    return result
