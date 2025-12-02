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
