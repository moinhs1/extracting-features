"""Tests for PE feature builder - starting with generic code matcher."""
import pytest
from processing.pe_feature_builder import code_matches_category


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
