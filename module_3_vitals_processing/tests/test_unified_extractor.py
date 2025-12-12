"""Tests for unified extractor core logic."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCheckNegation:
    """Test negation detection."""

    def test_no_negation(self):
        from extractors.unified_extractor import check_negation
        text = "HR: 72 bpm"
        assert check_negation(text, 4) is False

    def test_not_obtained(self):
        from extractors.unified_extractor import check_negation
        # Position 19 is at the end, so check_negation looks back and finds "not obtained"
        text = "vitals not obtained"
        assert check_negation(text, 19) is True

    def test_refused(self):
        from extractors.unified_extractor import check_negation
        text = "patient refused vitals"
        assert check_negation(text, 16) is True


class TestSkipSection:
    """Test skip section detection."""

    def test_not_in_skip(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Vitals: HR 72"
        assert is_in_skip_section(text, 10) is False

    def test_in_medications(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Medications: metoprolol 100mg daily"
        assert is_in_skip_section(text, 25) is True

    def test_in_allergies(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Allergies: penicillin causes HR 120"
        assert is_in_skip_section(text, 32) is True


class TestExtractHeartRate:
    """Test HR extraction with full pipeline."""

    def test_basic_hr(self):
        from extractors.unified_extractor import extract_heart_rate
        results = extract_heart_rate("HR: 72")
        assert len(results) >= 1
        assert results[0]['value'] == 72

    def test_hr_validation(self):
        from extractors.unified_extractor import extract_heart_rate
        # HR of 500 is invalid
        results = extract_heart_rate("HR: 500")
        assert len(results) == 0

    def test_hr_skip_medications(self):
        from extractors.unified_extractor import extract_heart_rate
        text = "Medications: metoprolol 100mg. Vitals: HR 72"
        results = extract_heart_rate(text)
        # Should find HR 72, not 100
        assert all(r['value'] != 100 for r in results)
        assert any(r['value'] == 72 for r in results)

    def test_hr_deduplication(self):
        from extractors.unified_extractor import extract_heart_rate
        # Same HR matched by multiple patterns should dedupe
        text = "Heart Rate: 72 bpm"
        results = extract_heart_rate(text)
        values = [r['value'] for r in results]
        # Should only have one 72, not multiple
        assert values.count(72) == 1


class TestExtractBloodPressure:
    """Test BP extraction with validation."""

    def test_basic_bp(self):
        from extractors.unified_extractor import extract_blood_pressure
        results = extract_blood_pressure("BP: 120/80")
        assert len(results) >= 1
        assert results[0]['sbp'] == 120
        assert results[0]['dbp'] == 80

    def test_bp_swap_transposed(self):
        from extractors.unified_extractor import extract_blood_pressure
        # 70/140 should be swapped to 140/70
        results = extract_blood_pressure("BP: 70/140")
        assert len(results) >= 1
        assert results[0]['sbp'] == 140
        assert results[0]['dbp'] == 70

    def test_bp_pulse_pressure_validation(self):
        from extractors.unified_extractor import extract_blood_pressure
        # 120/115 has pulse pressure of 5, invalid
        results = extract_blood_pressure("BP: 120/115")
        assert len(results) == 0

    def test_bp_skip_dates(self):
        from extractors.unified_extractor import extract_blood_pressure
        # Dates should not match (no BP label context)
        results = extract_blood_pressure("Date: 12/25/2023")
        # Should not extract 12/25 as BP
        assert not any(r['sbp'] == 12 and r['dbp'] == 25 for r in results)


class TestExtractTemperature:
    """Test temperature extraction with unit normalization."""

    def test_fahrenheit_converted(self):
        from extractors.unified_extractor import extract_temperature
        results = extract_temperature("Temp: 98.6 F")
        assert len(results) >= 1
        # Should be converted to Celsius (~37.0)
        assert 36.5 <= results[0]['value'] <= 37.5
        assert results[0]['units'] == 'C'

    def test_celsius_unchanged(self):
        from extractors.unified_extractor import extract_temperature
        results = extract_temperature("Temp: 37.0 C")
        assert len(results) >= 1
        assert results[0]['value'] == 37.0
        assert results[0]['units'] == 'C'

    def test_auto_detect_fahrenheit(self):
        from extractors.unified_extractor import extract_temperature
        # 98.6 without unit should be detected as F
        results = extract_temperature("Temp: 98.6")
        if results:
            assert results[0]['value'] < 50  # Converted to C


class TestExtractO2Flow:
    """Test O2 flow rate extraction."""

    def test_basic_flow(self):
        from extractors.unified_extractor import extract_o2_flow
        results = extract_o2_flow("on 2L NC")
        assert len(results) >= 1
        assert results[0]['value'] == 2

    def test_high_flow(self):
        from extractors.unified_extractor import extract_o2_flow
        results = extract_o2_flow("40L HFNC")
        assert len(results) >= 1
        assert results[0]['value'] == 40

    def test_range_validation(self):
        from extractors.unified_extractor import extract_o2_flow
        # 100L is invalid
        results = extract_o2_flow("100L NC")
        assert len(results) == 0


class TestExtractO2Device:
    """Test O2 device extraction."""

    def test_nasal_cannula(self):
        from extractors.unified_extractor import extract_o2_device
        results = extract_o2_device("on nasal cannula")
        assert len(results) >= 1

    def test_room_air(self):
        from extractors.unified_extractor import extract_o2_device
        results = extract_o2_device("on room air")
        assert len(results) >= 1


class TestExtractBMI:
    """Test BMI extraction."""

    def test_basic_bmi(self):
        from extractors.unified_extractor import extract_bmi
        results = extract_bmi("BMI: 24.5")
        assert len(results) >= 1
        assert results[0]['value'] == 24.5

    def test_range_validation(self):
        from extractors.unified_extractor import extract_bmi
        # BMI of 5 is invalid
        results = extract_bmi("BMI: 5")
        assert len(results) == 0


class TestExtractSupplemental:
    """Test supplemental extraction function."""

    def test_extract_supplemental(self):
        from extractors.unified_extractor import extract_supplemental_vitals
        text = "on 2L NC, BMI 28.3"
        results = extract_supplemental_vitals(text)
        assert 'O2_FLOW' in results
        assert 'O2_DEVICE' in results
        assert 'BMI' in results
