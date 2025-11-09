"""
Tests for LOINC database loading and matching.
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'module_2_laboratory_processing'))

from loinc_matcher import LoincMatcher


class TestLoincMatcher:
    """Test suite for LoincMatcher."""

    def test_initialization(self):
        """Test LoincMatcher initialization."""
        matcher = LoincMatcher('Loinc/LoincTable/Loinc.csv')
        assert matcher.loinc_csv_path == Path('Loinc/LoincTable/Loinc.csv')
        assert matcher.cache_dir == Path('cache')

    def test_load_from_cache(self, tmp_path):
        """Test loading LOINC from cache."""
        # Create a small mock LOINC CSV
        loinc_csv = tmp_path / "Loinc.csv"
        loinc_csv.write_text(
            '"LOINC_NUM","COMPONENT","PROPERTY","SYSTEM","CLASS","EXAMPLE_UNITS","LONG_COMMON_NAME","SHORTNAME"\n'
            '"2093-3","Cholesterol","MCnc","Ser/Plas","LABORATORY","mg/dL","Cholesterol [Mass/volume] in Serum or Plasma","Cholesterol SerPl-mCnc"\n'
        )

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # First load - should parse CSV
        matcher = LoincMatcher(str(loinc_csv), cache_dir=str(cache_dir))
        result = matcher.load()

        assert result is not None
        assert '2093-3' in result
        assert result['2093-3']['component'] == 'Cholesterol'

        # Cache file should exist
        assert (cache_dir / 'loinc_database.pkl').exists()

        # Second load - should use cache
        matcher2 = LoincMatcher(str(loinc_csv), cache_dir=str(cache_dir))
        result2 = matcher2.load()

        assert result2 == result

    def test_match_loinc_code(self, tmp_path):
        """Test matching LOINC code."""
        # Create mock LOINC CSV with multiple codes
        loinc_csv = tmp_path / "Loinc.csv"
        loinc_csv.write_text(
            '"LOINC_NUM","COMPONENT","PROPERTY","SYSTEM","CLASS","EXAMPLE_UNITS","LONG_COMMON_NAME","SHORTNAME"\n'
            '"2093-3","Cholesterol","MCnc","Ser/Plas","LABORATORY","mg/dL","Cholesterol [Mass/volume] in Serum or Plasma","Cholesterol SerPl-mCnc"\n'
            '"2085-9","Cholesterol.in HDL","MCnc","Ser/Plas","LABORATORY","mg/dL","Cholesterol in HDL [Mass/volume] in Serum or Plasma","HDL SerPl-mCnc"\n'
        )

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        matcher = LoincMatcher(str(loinc_csv), cache_dir=str(cache_dir))
        matcher.load()

        # Match existing code
        result = matcher.match('2093-3')
        assert result is not None
        assert result['component'] == 'Cholesterol'
        assert result['units'] == 'mg/dL'

        # Match different code
        result2 = matcher.match('2085-9')
        assert result2 is not None
        assert result2['component'] == 'Cholesterol.in HDL'

        # Non-existent code
        result3 = matcher.match('99999-9')
        assert result3 is None
