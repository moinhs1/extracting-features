"""Tests for Phy.txt structured vitals extractor."""
import pytest
from module_3_vitals_processing.extractors.phy_extractor import (
    parse_blood_pressure,
    map_concept_to_canonical,
    parse_result_value,
)


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


class TestMapConceptToCanonical:
    """Tests for map_concept_to_canonical function."""

    def test_pulse_to_hr(self):
        """Test Pulse maps to HR."""
        assert map_concept_to_canonical("Pulse") == "HR"

    def test_temperature(self):
        """Test Temperature maps to TEMP."""
        assert map_concept_to_canonical("Temperature") == "TEMP"

    def test_blood_pressure_epic(self):
        """Test Blood Pressure-Epic maps to BP."""
        assert map_concept_to_canonical("Blood Pressure-Epic") == "BP"

    def test_systolic_epic(self):
        """Test Systolic-Epic maps to SBP."""
        assert map_concept_to_canonical("Systolic-Epic") == "SBP"

    def test_diastolic_epic(self):
        """Test Diastolic-Epic maps to DBP."""
        assert map_concept_to_canonical("Diastolic-Epic") == "DBP"

    def test_o2_saturation(self):
        """Test O2 Saturation-SPO2 maps to SPO2."""
        assert map_concept_to_canonical("O2 Saturation-SPO2") == "SPO2"

    def test_respiratory_rate(self):
        """Test Respiratory rate maps to RR."""
        assert map_concept_to_canonical("Respiratory rate") == "RR"

    def test_weight(self):
        """Test Weight maps to WEIGHT."""
        assert map_concept_to_canonical("Weight") == "WEIGHT"

    def test_height(self):
        """Test Height maps to HEIGHT."""
        assert map_concept_to_canonical("Height") == "HEIGHT"

    def test_bmi(self):
        """Test BMI maps to BMI."""
        assert map_concept_to_canonical("BMI") == "BMI"

    def test_unknown_concept(self):
        """Test unknown concept returns None."""
        assert map_concept_to_canonical("Flu-High Dose") is None

    def test_none_input(self):
        """Test None input returns None."""
        assert map_concept_to_canonical(None) is None


class TestParseResultValue:
    """Tests for parse_result_value function."""

    def test_integer_value(self):
        """Test parsing integer."""
        assert parse_result_value("74") == 74.0

    def test_float_value(self):
        """Test parsing float."""
        assert parse_result_value("98.6") == 98.6

    def test_float_with_leading_zero(self):
        """Test parsing float with leading zero."""
        assert parse_result_value("0.5") == 0.5

    def test_empty_string(self):
        """Test empty string returns None."""
        assert parse_result_value("") is None

    def test_none_input(self):
        """Test None input returns None."""
        assert parse_result_value(None) is None

    def test_text_value(self):
        """Test non-numeric text returns None."""
        assert parse_result_value("Left arm") is None

    def test_whitespace(self):
        """Test value with whitespace."""
        assert parse_result_value("  74  ") == 74.0

    def test_negative_value(self):
        """Test negative value (should still parse)."""
        assert parse_result_value("-5") == -5.0

    def test_greater_than_symbol(self):
        """Test value with > symbol."""
        assert parse_result_value(">100") == 100.0

    def test_less_than_symbol(self):
        """Test value with < symbol."""
        assert parse_result_value("<50") == 50.0
