"""Tests for hnp_extractor module."""
import pytest


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
