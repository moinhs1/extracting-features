"""Tests for Phy.txt structured vitals extractor."""
import pytest
from module_3_vitals_processing.extractors.phy_extractor import parse_blood_pressure


class TestParseBloodPressure:
    """Tests for parse_blood_pressure function."""

    def test_normal_bp(self):
        """Test parsing normal BP string."""
        sbp, dbp = parse_blood_pressure("130/77")
        assert sbp == 130.0
        assert dbp == 77.0

    def test_high_bp(self):
        """Test parsing high BP values."""
        sbp, dbp = parse_blood_pressure("180/120")
        assert sbp == 180.0
        assert dbp == 120.0

    def test_low_bp(self):
        """Test parsing low BP values."""
        sbp, dbp = parse_blood_pressure("90/60")
        assert sbp == 90.0
        assert dbp == 60.0

    def test_invalid_bp_text(self):
        """Test that non-numeric BP returns None."""
        sbp, dbp = parse_blood_pressure("Left arm")
        assert sbp is None
        assert dbp is None

    def test_invalid_bp_sitting(self):
        """Test that positional text returns None."""
        sbp, dbp = parse_blood_pressure("Sitting")
        assert sbp is None
        assert dbp is None

    def test_empty_string(self):
        """Test empty string returns None."""
        sbp, dbp = parse_blood_pressure("")
        assert sbp is None
        assert dbp is None

    def test_none_input(self):
        """Test None input returns None."""
        sbp, dbp = parse_blood_pressure(None)
        assert sbp is None
        assert dbp is None

    def test_missing_diastolic(self):
        """Test single value returns None for both."""
        sbp, dbp = parse_blood_pressure("130")
        assert sbp is None
        assert dbp is None

    def test_spaces_around_slash(self):
        """Test BP with spaces around slash."""
        sbp, dbp = parse_blood_pressure("130 / 77")
        assert sbp == 130.0
        assert dbp == 77.0
