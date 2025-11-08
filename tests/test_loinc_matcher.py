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
