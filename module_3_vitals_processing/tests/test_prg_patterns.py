"""Tests for prg_patterns module."""
import pytest
import re


class TestPrgSectionPatterns:
    """Test Prg section pattern definitions."""

    def test_section_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        assert isinstance(PRG_SECTION_PATTERNS, dict)
        assert len(PRG_SECTION_PATTERNS) >= 10

    def test_physical_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['physical_exam']
        assert re.search(pattern, "Physical Exam: BP 120/80", re.IGNORECASE)
        assert re.search(pattern, "Physical Examination: normal", re.IGNORECASE)

    def test_vitals_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['vitals']
        assert re.search(pattern, "Vitals: HR 72", re.IGNORECASE)
        assert re.search(pattern, "Vital: T 98.6", re.IGNORECASE)

    def test_on_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['on_exam']
        assert re.search(pattern, "ON EXAM: Vital Signs BP 120/80", re.IGNORECASE)

    def test_objective_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['objective']
        assert re.search(pattern, "Objective: Physical Exam", re.IGNORECASE)
