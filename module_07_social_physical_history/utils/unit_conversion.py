# utils/unit_conversion.py
"""Unit conversion utilities for body measurements."""

import math
from typing import Optional


def lbs_to_kg(lbs: float) -> float:
    """Convert pounds to kilograms."""
    return lbs * 0.453592


def kg_to_lbs(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462


def inches_to_cm(inches: float) -> float:
    """Convert inches to centimeters."""
    return inches * 2.54


def cm_to_m(cm: float) -> float:
    """Convert centimeters to meters."""
    return cm / 100


def calculate_bmi(weight_kg: float, height_m: float) -> Optional[float]:
    """Calculate BMI from weight (kg) and height (m)."""
    if height_m and height_m > 0:
        return weight_kg / (height_m ** 2)
    return None


def calculate_bsa_dubois(weight_kg: float, height_cm: float) -> float:
    """Calculate BSA using DuBois formula: 0.007184 * W^0.425 * H^0.725"""
    return 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)


def calculate_bsa_mosteller(weight_kg: float, height_cm: float) -> float:
    """Calculate BSA using Mosteller formula: sqrt((W * H) / 3600)"""
    return math.sqrt((weight_kg * height_cm) / 3600)
