"""Unit conversion functions for vital signs."""
from typing import Optional


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius.

    Args:
        temp_f: Temperature in Fahrenheit

    Returns:
        Temperature in Celsius
    """
    return (temp_f - 32) * 5 / 9


def normalize_temperature(value: float, units: Optional[str]) -> float:
    """Normalize temperature to Celsius.

    Args:
        value: Temperature value
        units: Unit string (C, F, °C, °F, deg C, deg F, or None)

    Returns:
        Temperature in Celsius

    Note:
        If units is None, infers based on value:
        - >50 assumed Fahrenheit
        - <=50 assumed Celsius
    """
    if units is None:
        # Infer from value range
        if value > 50:
            return fahrenheit_to_celsius(value)
        return value

    # Normalize unit string
    units_lower = units.lower().replace("°", "").replace("deg ", "").strip()

    if units_lower in ("f", "fahrenheit"):
        return fahrenheit_to_celsius(value)

    # Default: assume Celsius
    return value
