"""Tests for qc_filters module."""
import pytest
from processing.qc_filters import is_physiologically_valid, VALID_RANGES


class TestPhysiologicalRanges:
    """Tests for physiological range validation."""

    def test_valid_hr(self):
        """Normal heart rate is valid."""
        assert is_physiologically_valid("HR", 72) is True

    def test_invalid_hr_too_low(self):
        """HR < 20 is impossible."""
        assert is_physiologically_valid("HR", 15) is False

    def test_invalid_hr_too_high(self):
        """HR > 300 is impossible."""
        assert is_physiologically_valid("HR", 350) is False

    def test_valid_sbp(self):
        """Normal SBP is valid."""
        assert is_physiologically_valid("SBP", 120) is True

    def test_invalid_sbp_too_low(self):
        """SBP < 40 is impossible."""
        assert is_physiologically_valid("SBP", 30) is False

    def test_valid_spo2(self):
        """Normal SpO2 is valid."""
        assert is_physiologically_valid("SPO2", 98) is True

    def test_invalid_spo2_over_100(self):
        """SpO2 > 100% is impossible."""
        assert is_physiologically_valid("SPO2", 105) is False

    def test_valid_temp_celsius(self):
        """Normal temp in Celsius is valid."""
        assert is_physiologically_valid("TEMP", 37.0) is True

    def test_invalid_temp_too_low(self):
        """Temp < 30°C is impossible (hypothermia death)."""
        assert is_physiologically_valid("TEMP", 25) is False

    def test_valid_ranges_defined(self):
        """All 7 core vitals have defined ranges."""
        expected_vitals = {"HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"}
        assert expected_vitals <= set(VALID_RANGES.keys())


class TestAbnormalFlagging:
    """Tests for abnormal value flagging."""

    def test_normal_hr_not_flagged(self):
        """HR 72 is normal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 72) is False

    def test_tachycardia_flagged(self):
        """HR > 100 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 110) is True

    def test_bradycardia_flagged(self):
        """HR < 60 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 50) is True

    def test_hypotension_flagged(self):
        """SBP < 90 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("SBP", 85) is True

    def test_hypoxemia_flagged(self):
        """SpO2 < 92 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("SPO2", 88) is True

    def test_fever_flagged(self):
        """Temp > 38.5°C is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("TEMP", 39.5) is True
