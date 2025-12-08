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


class TestPrgSkipPatterns:
    """Test Prg skip section patterns."""

    def test_skip_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        assert isinstance(PRG_SKIP_PATTERNS, list)
        assert len(PRG_SKIP_PATTERNS) >= 10

    def test_allergies_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Allergies: atenolol - fatigue, HR 50"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_medications_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Medications: lisinopril 10mg daily"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_past_medical_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Past Medical History: hypertension, diabetes"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_family_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Family History: father with MI"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_reactions_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Reactions: hives, swelling"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched
