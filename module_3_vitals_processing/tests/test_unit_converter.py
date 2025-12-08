"""Tests for unit_converter module."""
import pytest
from processing.unit_converter import fahrenheit_to_celsius, normalize_temperature


class TestTemperatureConversion:
    """Tests for temperature conversion."""

    def test_fahrenheit_to_celsius_98_6(self):
        """98.6°F is 37.0°C (normal body temp)."""
        result = fahrenheit_to_celsius(98.6)
        assert abs(result - 37.0) < 0.1

    def test_fahrenheit_to_celsius_100_4(self):
        """100.4°F is 38.0°C (fever threshold)."""
        result = fahrenheit_to_celsius(100.4)
        assert abs(result - 38.0) < 0.1

    def test_fahrenheit_to_celsius_freezing(self):
        """32°F is 0°C."""
        result = fahrenheit_to_celsius(32.0)
        assert abs(result - 0.0) < 0.01


class TestNormalizeTemperature:
    """Tests for temperature normalization to Celsius."""

    def test_normalize_celsius_passthrough(self):
        """Celsius values pass through unchanged."""
        result = normalize_temperature(37.0, "C")
        assert result == 37.0

    def test_normalize_fahrenheit_converts(self):
        """Fahrenheit values are converted."""
        result = normalize_temperature(98.6, "F")
        assert abs(result - 37.0) < 0.1

    def test_normalize_infers_fahrenheit_high_value(self):
        """Values >50 with unknown units assumed Fahrenheit."""
        result = normalize_temperature(98.6, None)
        assert abs(result - 37.0) < 0.1

    def test_normalize_infers_celsius_low_value(self):
        """Values <=50 with unknown units assumed Celsius."""
        result = normalize_temperature(37.0, None)
        assert result == 37.0

    def test_normalize_handles_degree_symbol_units(self):
        """Handles '°C', '°F', 'deg C', 'deg F' variants."""
        assert normalize_temperature(37.0, "°C") == 37.0
        assert abs(normalize_temperature(98.6, "°F") - 37.0) < 0.1
        assert normalize_temperature(37.0, "deg C") == 37.0
