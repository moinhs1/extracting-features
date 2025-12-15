"""Tests for ICD code parsing."""
import pytest
from processing.icd_parser import detect_icd_version, normalize_icd_code, is_pe_diagnosis


class TestDetectICDVersion:
    """Tests for ICD version detection."""

    def test_icd9_numeric_code(self):
        """ICD-9 codes like 415.1 are detected."""
        assert detect_icd_version("415.1") == "9"

    def test_icd9_v_code(self):
        """ICD-9 V codes are detected."""
        assert detect_icd_version("V10.3") == "9"

    def test_icd9_e_code(self):
        """ICD-9 E codes are detected."""
        assert detect_icd_version("E850.0") == "9"

    def test_icd10_alpha_start(self):
        """ICD-10 codes starting with letter are detected."""
        assert detect_icd_version("I26.0") == "10"
        assert detect_icd_version("C34.1") == "10"
        assert detect_icd_version("J44.9") == "10"

    def test_icd10_z_code(self):
        """ICD-10 Z codes are detected."""
        assert detect_icd_version("Z87.01") == "10"

    def test_empty_returns_unknown(self):
        """Empty code returns unknown."""
        assert detect_icd_version("") == "unknown"
        assert detect_icd_version(None) == "unknown"


class TestNormalizeICDCode:
    """Tests for ICD code normalization."""

    def test_removes_dots(self):
        """Dots are preserved but whitespace removed."""
        assert normalize_icd_code("I26.0") == "I26.0"
        assert normalize_icd_code(" I26.0 ") == "I26.0"

    def test_uppercase(self):
        """Codes are uppercased."""
        assert normalize_icd_code("i26.0") == "I26.0"

    def test_handles_none(self):
        """None returns empty string."""
        assert normalize_icd_code(None) == ""


class TestIsPEDiagnosis:
    """Tests for PE diagnosis detection."""

    def test_icd10_pe_codes(self):
        """ICD-10 PE codes are detected."""
        assert is_pe_diagnosis("I26.0", "10") is True
        assert is_pe_diagnosis("I26.9", "10") is True
        assert is_pe_diagnosis("I26.99", "10") is True

    def test_icd9_pe_codes(self):
        """ICD-9 PE codes are detected."""
        assert is_pe_diagnosis("415.1", "9") is True
        assert is_pe_diagnosis("415.11", "9") is True
        assert is_pe_diagnosis("415.19", "9") is True

    def test_non_pe_codes(self):
        """Non-PE codes return False."""
        assert is_pe_diagnosis("I50.9", "10") is False
        assert is_pe_diagnosis("428.0", "9") is False
