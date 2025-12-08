"""Unit conversion functions for vital signs."""


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius.

    Args:
        temp_f: Temperature in Fahrenheit

    Returns:
        Temperature in Celsius
    """
    return (temp_f - 32) * 5 / 9
