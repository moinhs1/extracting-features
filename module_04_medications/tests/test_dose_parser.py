"""Tests for dose parsing from RPDR medication strings."""

import pytest
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDoseExtraction:
    """Test dose value and unit extraction."""

    def test_standard_mg_dose(self):
        """Extract simple mg dose."""
        from extractors.dose_parser import extract_dose

        result = extract_dose("Aspirin 325mg tablet")

        assert result['dose_value'] == 325.0
        assert result['dose_unit'] == 'mg'

    def test_mcg_dose(self):
        """Extract mcg dose."""
        from extractors.dose_parser import extract_dose

        result = extract_dose("Fentanyl citrate 100 mcg ampul")

        assert result['dose_value'] == 100.0
        assert result['dose_unit'] == 'mcg'

    def test_units_dose(self):
        """Extract units dose (heparin)."""
        from extractors.dose_parser import extract_dose

        result = extract_dose("Heparin sodium 5000 unit/ml injection")

        assert result['dose_value'] == 5000.0
        assert result['dose_unit'] == 'units'

    def test_concentration_format(self):
        """Extract concentration format dose."""
        from extractors.dose_parser import extract_dose

        result = extract_dose("Morphine sulfate 2 mg/ml solution")

        assert result['dose_value'] == 2.0
        assert result['dose_unit'] == 'mg'

    def test_no_dose_found(self):
        """Handle medication string without extractable dose."""
        from extractors.dose_parser import extract_dose

        result = extract_dose("Supply Of Radiopharmaceutical Agent")

        assert result['dose_value'] is None
        assert result['dose_unit'] is None
