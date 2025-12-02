"""Tests for hnp_extractor module."""
import pytest
from datetime import datetime
import pandas as pd
import tempfile
import os


class TestIdentifySections:
    """Test section identification in clinical notes."""

    def test_finds_exam_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "History of illness... Physical Exam: BP 120/80 HR 72 General appearance good"
        sections = identify_sections(text)
        assert 'exam' in sections
        assert 'BP 120/80' in sections['exam']

    def test_finds_vitals_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "Patient presents with... Vitals: T 98.6F HR 80 BP 130/85 Assessment..."
        sections = identify_sections(text)
        assert 'vitals' in sections
        assert 'HR 80' in sections['vitals']

    def test_finds_ed_course_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "Chief complaint... ED Course: BP 110/70 given fluids Admitted to medicine"
        sections = identify_sections(text)
        assert 'ed_course' in sections
        assert 'BP 110/70' in sections['ed_course']

    def test_finds_multiple_sections(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "ED Course: BP 100/60 HR 90... Physical Exam: BP 120/80 HR 75 well appearing"
        sections = identify_sections(text)
        assert 'ed_course' in sections
        assert 'exam' in sections
        assert 'BP 100/60' in sections['ed_course']
        assert 'BP 120/80' in sections['exam']

    def test_returns_empty_when_no_sections(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "This is a note without any section headers in it."
        sections = identify_sections(text)
        assert sections == {}

    def test_handles_exam_on_admission_variant(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "History... EXAM ON ADMISSION Vitals: HR 88 BP 140/90 Gen: alert"
        sections = identify_sections(text)
        assert 'exam' in sections


class TestCheckNegation:
    """Test negation detection in context window."""

    def test_detects_not_obtained(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Blood pressure not obtained due to patient condition"
        assert check_negation(text, position=15, window=50) is True

    def test_detects_unable_to_measure(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Vitals: HR 80, BP unable to measure, RR 18"
        # Position at "BP"
        assert check_negation(text, position=14, window=50) is True

    def test_detects_refused(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Patient refused vital signs assessment"
        assert check_negation(text, position=20, window=50) is True

    def test_detects_no_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "There were no vitals filed for this visit"
        assert check_negation(text, position=15, window=50) is True

    def test_returns_false_for_normal_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Vitals: BP 120/80 HR 72 RR 16 SpO2 98%"
        assert check_negation(text, position=10, window=50) is False

    def test_respects_window_size(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        # Negation far from position
        text = "Unable to obtain vitals earlier. Later: BP 120/80 normal values"
        # Position at "BP 120/80" - negation is far away
        assert check_negation(text, position=40, window=20) is False


class TestExtractHeartRate:
    """Test heart rate extraction patterns."""

    def test_extracts_heart_rate_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Vitals: Heart Rate: 88 BP 120/80"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 88
        assert results[0]['confidence'] == 1.0

    def test_extracts_hr_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "HR 72 BP 130/85"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 72
        assert results[0]['confidence'] == 0.95

    def test_extracts_pulse_p(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "37.2 °C P 79 BP 149/65"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 79

    def test_extracts_pulse_full(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Pulse 86 regular"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 86

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "P (!) 117 BP 170/87"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 117
        assert results[0]['is_flagged_abnormal'] is True

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Heart Rate: [62-72] 72"
        results = extract_heart_rate(text)
        # Should get the current value 72
        values = [r['value'] for r in results]
        assert 72 in values

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "HR 500"  # Invalid
        results = extract_heart_rate(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Heart rate not obtained"
        results = extract_heart_rate(text)
        assert len(results) == 0

    def test_multiple_extractions(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "ED: HR 90... Exam: Heart Rate: 75"
        results = extract_heart_rate(text)
        values = [r['value'] for r in results]
        assert 90 in values
        assert 75 in values


class TestExtractBloodPressure:
    """Test blood pressure extraction patterns."""

    def test_extracts_bp_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "Blood pressure 130/85 measured sitting"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 130
        assert results[0]['dbp'] == 85
        assert results[0]['confidence'] == 1.0

    def test_extracts_bp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP: 120/80 HR 72"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 120
        assert results[0]['dbp'] == 80

    def test_extracts_bp_with_mmhg(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "measured 145/92 mmHg in left arm"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 145

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP (!) 180/110 HR 88"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['is_flagged_abnormal'] is True

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP: (115-154)/(59-69) 145/67"
        results = extract_blood_pressure(text)
        # Should extract 145/67
        assert any(r['sbp'] == 145 and r['dbp'] == 67 for r in results)

    def test_swaps_if_sbp_less_than_dbp(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP 70/120"  # Swapped values
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 120  # Should be swapped
        assert results[0]['dbp'] == 70

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP 400/300"  # Invalid
        results = extract_blood_pressure(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP not obtained due to combative patient"
        results = extract_blood_pressure(text)
        assert len(results) == 0

    def test_multiple_extractions(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "Triage: BP 100/60... Exam: BP 120/75"
        results = extract_blood_pressure(text)
        sbps = [r['sbp'] for r in results]
        assert 100 in sbps
        assert 120 in sbps


class TestExtractRespiratoryRate:
    """Test respiratory rate extraction patterns."""

    def test_extracts_respiratory_rate_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Respiratory Rate: 16 SpO2 98%"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 16
        assert results[0]['confidence'] == 1.0

    def test_extracts_rr_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "HR 72 RR 18 SpO2 97%"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 18

    def test_extracts_resp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Resp 20 unlabored"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 20

    def test_extracts_trr_triage(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "TRR 22 on arrival"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 22

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Respiratory Rate: [14-22] 18"
        results = extract_respiratory_rate(text)
        values = [r['value'] for r in results]
        assert 18 in values

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "RR 100"  # Invalid
        results = extract_respiratory_rate(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "RR not measured"
        results = extract_respiratory_rate(text)
        assert len(results) == 0


class TestExtractSpO2:
    """Test SpO2 extraction patterns."""

    def test_extracts_spo2_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2: 98% on room air"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98
        assert results[0]['confidence'] == 1.0

    def test_extracts_spo2_no_colon(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "HR 72 SpO2 97 % RR 16"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 97

    def test_extracts_sao2(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SaO2 >99% on 2L NC"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 99

    def test_extracts_o2_sat(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "O2 Sat: 95% on RA"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 95

    def test_extracts_o2_saturation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "O2 Saturation 94%"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 94

    def test_extracts_percentage_with_context(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "satting 92% on RA"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 92

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 (!) 89 % on 3L"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['is_flagged_abnormal'] is True

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 150%"  # Invalid
        results = extract_spo2(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 unable to measure"
        results = extract_spo2(text)
        assert len(results) == 0


class TestExtractTemperature:
    """Test temperature extraction patterns."""

    def test_extracts_temperature_full_label_celsius(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temperature: 37.2 °C (99 °F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.2
        assert results[0]['units'] == 'C'

    def test_extracts_temp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 36.8 ?C HR 72"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 36.8
        assert results[0]['units'] == 'C'

    def test_extracts_t_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "T 98.6F P 80 BP 120/80"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98.6
        assert results[0]['units'] == 'F'

    def test_extracts_tcurrent(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Tcurrent 37.3 ?C (99.2 ?F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.3

    def test_extracts_encoding_question_mark(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "36.5 ?C (97.7 ?F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 36.5
        assert results[0]['units'] == 'C'

    def test_autodetects_fahrenheit_from_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        # Value > 50 with no unit should be detected as Fahrenheit
        text = "Temp: 98.6"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98.6
        assert results[0]['units'] == 'F'

    def test_autodetects_celsius_from_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        # Value < 50 with no unit should be detected as Celsius
        text = "Temp: 37.0"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.0
        assert results[0]['units'] == 'C'

    def test_rejects_invalid_celsius(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 50 C"  # Invalid Celsius
        results = extract_temperature(text)
        assert len(results) == 0

    def test_rejects_invalid_fahrenheit(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 150 F"  # Invalid Fahrenheit
        results = extract_temperature(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temperature not obtained"
        results = extract_temperature(text)
        assert len(results) == 0


class TestExtractTimestamp:
    """Test timestamp extraction and estimation."""

    def test_extracts_explicit_timestamp_12h(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "Vitals at 10/23/2021 7:34 PM: HR 80 BP 120/80"
        report_dt = datetime(2021, 10, 24, 9, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'explicit'
        assert ts.year == 2021
        assert ts.month == 10
        assert ts.day == 23

    def test_extracts_explicit_timestamp_military(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "03/08/22 1500 BP: 186/87"
        report_dt = datetime(2022, 3, 8, 18, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'explicit'
        assert ts.hour == 15

    def test_estimates_ed_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "BP 100/60 HR 90"  # No explicit timestamp
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'ed_course', report_dt)
        assert source == 'estimated'
        assert offset == -6
        assert ts.hour == 6  # 12 - 6 = 6

    def test_estimates_exam_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "BP 120/80 HR 72"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'exam', report_dt)
        assert source == 'estimated'
        assert offset == -1
        assert ts.hour == 11

    def test_estimates_vitals_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "HR 75 BP 118/72"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'estimated'
        assert offset == -1

    def test_estimates_unknown_section_default_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "HR 80 BP 125/80"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'other', report_dt)
        assert source == 'estimated'
        assert offset == -2  # Default offset


class TestProcessHnpRow:
    """Test full row processing."""

    def test_processes_row_with_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98% Temp 37.0 C'
        })
        results = process_hnp_row(row)
        assert len(results) >= 5  # BP produces SBP+DBP, plus HR, RR, SPO2, TEMP

        # Check all vitals extracted
        vital_types = [r['vital_type'] for r in results]
        assert 'HR' in vital_types
        assert 'SBP' in vital_types
        assert 'DBP' in vital_types
        assert 'RR' in vital_types
        assert 'SPO2' in vital_types
        assert 'TEMP' in vital_types

    def test_tags_extraction_context(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'ED Course: BP 100/60... Physical Exam: BP 120/80'
        })
        results = process_hnp_row(row)
        contexts = [r['extraction_context'] for r in results]
        assert 'ed_course' in contexts
        assert 'exam' in contexts

    def test_preserves_empi_and_report_number(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '99999',
            'Report_Number': 'RPT555',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Vitals: HR 80'
        })
        results = process_hnp_row(row)
        assert all(r['EMPI'] == '99999' for r in results)
        assert all(r['report_number'] == 'RPT555' for r in results)

    def test_returns_empty_for_no_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Patient presents with headache. No vitals documented.'
        })
        results = process_hnp_row(row)
        assert len(results) == 0

    def test_handles_missing_report_text(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': None
        })
        results = process_hnp_row(row)
        assert results == []


class TestExtractHnpVitals:
    """Test main extraction function."""

    def test_extracts_from_small_file(self, tmp_path):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_hnp_vitals

        # Create test input file
        input_file = tmp_path / "test_hnp.txt"
        input_file.write_text(
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n"
            "12345|PMR1|BWH|MRN1|RPT001|10/23/2021 9:00:00 PM|H&P|F|BHPHP|Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98%\n"
            "67890|PMR2|BWH|MRN2|RPT002|10/24/2021 10:00:00 AM|H&P|F|BHPHP|Vitals: T 37.2 C HR 88 BP 130/85\n"
        )

        output_file = tmp_path / "output.parquet"

        df = extract_hnp_vitals(str(input_file), str(output_file), n_workers=1)

        assert os.path.exists(output_file)
        assert len(df) > 0
        assert 'EMPI' in df.columns
        assert 'vital_type' in df.columns
        assert 'value' in df.columns

    def test_handles_empty_file(self, tmp_path):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_hnp_vitals

        input_file = tmp_path / "empty_hnp.txt"
        input_file.write_text(
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n"
        )

        output_file = tmp_path / "output.parquet"

        df = extract_hnp_vitals(str(input_file), str(output_file), n_workers=1)

        assert len(df) == 0
