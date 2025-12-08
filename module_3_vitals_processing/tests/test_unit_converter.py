"""Tests for unit_converter module."""
import pytest
from processing.unit_converter import fahrenheit_to_celsius


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
