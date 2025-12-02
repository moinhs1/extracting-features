# module_3_vitals_processing/tests/test_hnp_patterns.py
"""Tests for hnp_patterns module."""
import pytest


class TestPatternsExist:
    """Test that all required patterns are defined."""

    def test_vital_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import (
            HR_PATTERNS, BP_PATTERNS, RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS
        )
        assert len(HR_PATTERNS) > 0
        assert len(BP_PATTERNS) > 0
        assert len(RR_PATTERNS) > 0
        assert len(SPO2_PATTERNS) > 0
        assert len(TEMP_PATTERNS) > 0

    def test_section_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import SECTION_PATTERNS
        assert 'exam' in SECTION_PATTERNS
        assert 'vitals' in SECTION_PATTERNS
        assert 'ed_course' in SECTION_PATTERNS

    def test_negation_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import NEGATION_PATTERNS
        assert len(NEGATION_PATTERNS) > 0

    def test_hnp_columns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import HNP_COLUMNS
        assert 'EMPI' in HNP_COLUMNS
        assert 'Report_Text' in HNP_COLUMNS
