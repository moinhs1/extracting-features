"""Tests for Phy.txt structured vitals extractor."""
import pytest
import pandas as pd
from module_3_vitals_processing.extractors.phy_extractor import (
    parse_blood_pressure,
    map_concept_to_canonical,
    parse_result_value,
    process_vital_row,
    extract_phy_vitals,
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


class TestProcessVitalRow:
    """Tests for process_vital_row function."""

    def test_pulse_row(self):
        """Test processing a Pulse row."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Pulse',
            'Result': '74',
            'Units': 'beats/minute',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 1
        assert result[0]['EMPI'] == '100003884'
        assert result[0]['vital_type'] == 'HR'
        assert result[0]['value'] == 74.0
        assert result[0]['units'] == 'beats/minute'

    def test_blood_pressure_row(self):
        """Test processing a Blood Pressure row produces two records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Blood Pressure-Epic',
            'Result': '130/77',
            'Units': 'millimeter of mercury',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 2

        # Check SBP
        sbp_record = [r for r in result if r['vital_type'] == 'SBP'][0]
        assert sbp_record['value'] == 130.0

        # Check DBP
        dbp_record = [r for r in result if r['vital_type'] == 'DBP'][0]
        assert dbp_record['value'] == 77.0

    def test_invalid_bp_row(self):
        """Test that invalid BP like 'Left arm' produces no records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Blood Pressure-Epic',
            'Result': 'Left arm',
            'Units': '',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 0

    def test_non_vital_row(self):
        """Test that non-vital concept produces no records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Flu-High Dose',
            'Result': '76',
            'Units': '',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 0

    def test_temperature_row(self):
        """Test processing a Temperature row."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Temperature',
            'Result': '98.6',
            'Units': 'degrees Fahrenheit',
            'Inpatient_Outpatient': 'Inpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 1
        assert result[0]['vital_type'] == 'TEMP'
        assert result[0]['value'] == 98.6
        assert result[0]['encounter_type'] == 'Inpatient'


class TestExtractPhyVitals:
    """Tests for extract_phy_vitals main function."""

    def test_extract_from_sample_file(self, tmp_path):
        """Test extraction from a sample Phy.txt file."""
        # Create sample data
        lines = [
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number",
            "100003884|10040029737|BWH|00667360|7/21/2015|Pulse|EPIC|PUL|74|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-001",
            "100003884|10040029737|BWH|00667360|7/21/2015|Blood Pressure-Epic|EPIC|BP|130/77|millimeter of mercury|Doctor|Clinic|BWH|Outpatient|EPIC-001",
            "100003884|10040029737|BWH|00667360|7/21/2015|Temperature|EPIC|TEMP|98.6|degrees Fahrenheit|Doctor|Clinic|BWH|Inpatient|EPIC-002",
            "100003884|10040029737|BWH|00667360|7/21/2015|Flu-High Dose|EPIC|76||units|Doctor|Clinic|BWH|Outpatient|EPIC-001",
            "100003884|10040029737|BWH|00667360|7/21/2015|Blood Pressure-Epic|EPIC|BP|Left arm|units|Doctor|Clinic|BWH|Outpatient|EPIC-001",
        ]
        sample_data = "\n".join(lines)

        # Write sample file
        sample_file = tmp_path / "test_phy.txt"
        sample_file.write_text(sample_data)

        # Run extraction
        output_file = tmp_path / "output.parquet"
        result_df = extract_phy_vitals(str(sample_file), str(output_file), show_progress=False)

        # Verify results
        assert len(result_df) == 4  # 1 Pulse + 2 BP (SBP/DBP) + 1 Temp = 4
        assert set(result_df['vital_type'].unique()) == {'HR', 'SBP', 'DBP', 'TEMP'}
        assert result_df[result_df['vital_type'] == 'HR']['value'].iloc[0] == 74.0
        assert result_df[result_df['vital_type'] == 'SBP']['value'].iloc[0] == 130.0
        assert result_df[result_df['vital_type'] == 'DBP']['value'].iloc[0] == 77.0
        assert result_df[result_df['vital_type'] == 'TEMP']['value'].iloc[0] == 98.6

        # Verify parquet file was created
        assert output_file.exists()

    def test_date_parsing(self, tmp_path):
        """Test that dates are parsed correctly."""
        lines = [
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number",
            "100003884|10040029737|BWH|00667360|7/21/2015|Pulse|EPIC|PUL|74|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-001",
            "100003884|10040029737|BWH|00667360|12/25/2020|Pulse|EPIC|PUL|80|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-002",
        ]
        sample_data = "\n".join(lines)

        sample_file = tmp_path / "test_phy.txt"
        sample_file.write_text(sample_data)
        output_file = tmp_path / "output.parquet"

        result_df = extract_phy_vitals(str(sample_file), str(output_file), show_progress=False)

        assert len(result_df) == 2
        # Verify timestamp column exists and is datetime
        assert 'timestamp' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['timestamp'])
