"""Tests for PE feature builder - starting with generic code matcher."""
import pytest
from processing.pe_feature_builder import code_matches_category
from config.icd_code_lists import VTE_CODES


class TestCodeMatcher:
    """Tests for the generic ICD code matcher."""

    def test_exact_match(self):
        """Test exact code match within category prefixes."""
        category_codes = {
            "icd10": ["I26.0", "I26.9"],
            "icd9": []
        }
        assert code_matches_category("I26.0", category_codes, "10") is True

    def test_prefix_match(self):
        """Test prefix matching - longer code matches shorter prefix."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": []
        }
        assert code_matches_category("I26.99", category_codes, "10") is True

    def test_no_match(self):
        """Test that non-matching codes return False."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": []
        }
        assert code_matches_category("I50.9", category_codes, "10") is False

    def test_icd9_match(self):
        """Test ICD-9 code matching with prefix."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": ["415"]
        }
        assert code_matches_category("415.19", category_codes, "9") is True


class TestVTECodes:
    """Tests for VTE ICD code lists."""

    def test_pe_icd10_codes_exist(self):
        """Verify PE ICD-10 codes are defined."""
        assert "pe" in VTE_CODES
        assert "icd10" in VTE_CODES["pe"]
        assert len(VTE_CODES["pe"]["icd10"]) > 0

    def test_pe_code_matches(self):
        """Test that I26.99 matches PE category."""
        assert code_matches_category("I26.99", VTE_CODES["pe"], "10") is True

    def test_dvt_lower_matches(self):
        """Test that I82.401 matches lower DVT category."""
        assert code_matches_category("I82.401", VTE_CODES["dvt_lower"], "10") is True

    def test_non_vte_no_match(self):
        """Test that I50.9 (heart failure) does NOT match any VTE codes."""
        assert code_matches_category("I50.9", VTE_CODES["pe"], "10") is False
        assert code_matches_category("I50.9", VTE_CODES["dvt_lower"], "10") is False
        assert code_matches_category("I50.9", VTE_CODES["dvt_upper"], "10") is False
