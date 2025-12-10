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


class TestRouteExtraction:
    """Test route extraction."""

    def test_iv_route(self):
        """Extract IV route."""
        from extractors.dose_parser import extract_route

        assert extract_route("Vancomycin 1gm injection IV") == 'IV'
        assert extract_route("Heparin infusion 25000 units") == 'IV'

    def test_po_route(self):
        """Extract PO route."""
        from extractors.dose_parser import extract_route

        assert extract_route("Aspirin 325mg tablet") == 'PO'
        assert extract_route("Metoprolol oral solution") == 'PO'

    def test_sc_route(self):
        """Extract SC route."""
        from extractors.dose_parser import extract_route

        assert extract_route("Enoxaparin 40mg subcutaneous") == 'SC'
        assert extract_route("Heparin 5000 units SQ") == 'SC'

    def test_no_route(self):
        """Handle string without route."""
        from extractors.dose_parser import extract_route

        assert extract_route("Aspirin 325mg") is None


class TestFrequencyExtraction:
    """Test frequency extraction."""

    def test_daily(self):
        """Extract daily frequency."""
        from extractors.dose_parser import extract_frequency

        assert extract_frequency("Aspirin 81mg daily") == 'QD'
        assert extract_frequency("Metoprolol 25mg QD") == 'QD'

    def test_bid(self):
        """Extract twice daily."""
        from extractors.dose_parser import extract_frequency

        assert extract_frequency("Enoxaparin 1mg/kg BID") == 'BID'

    def test_prn(self):
        """Extract PRN."""
        from extractors.dose_parser import extract_frequency

        assert extract_frequency("Morphine 2mg IV PRN pain") == 'PRN'
        assert extract_frequency("Tylenol as needed") == 'PRN'


class TestDrugNameExtraction:
    """Test drug name extraction."""

    def test_simple_name(self):
        """Extract simple drug name."""
        from extractors.dose_parser import extract_drug_name

        assert extract_drug_name("Aspirin 325mg tablet") == 'aspirin'

    def test_compound_name(self):
        """Extract compound drug name."""
        from extractors.dose_parser import extract_drug_name

        name = extract_drug_name("Heparin sodium 5000 unit/ml injection")
        assert 'heparin' in name

    def test_name_with_salt(self):
        """Handle drug names with salt forms."""
        from extractors.dose_parser import extract_drug_name

        name = extract_drug_name("Fentanyl citrate 100 mcg ampul")
        assert 'fentanyl' in name


class TestFullParsing:
    """Test full medication string parsing."""

    def test_full_parse_aspirin(self):
        """Full parse of aspirin tablet."""
        from extractors.dose_parser import parse_medication_string

        result = parse_medication_string("Aspirin 325mg tablet")

        assert result['parsed_name'] == 'aspirin'
        assert result['parsed_dose_value'] == 325.0
        assert result['parsed_dose_unit'] == 'mg'
        assert result['parsed_route'] == 'PO'
        assert result['parse_method'] == 'regex'

    def test_full_parse_heparin(self):
        """Full parse of heparin injection."""
        from extractors.dose_parser import parse_medication_string

        result = parse_medication_string("Heparin sodium 5000 unit/ml injection")

        assert result['parsed_dose_value'] == 5000.0
        assert result['parsed_dose_unit'] == 'units'
        assert 'heparin' in result['parsed_name']

    def test_full_parse_enoxaparin(self):
        """Full parse of enoxaparin."""
        from extractors.dose_parser import parse_medication_string

        result = parse_medication_string("Enoxaparin 100 mg/ml solution subcutaneous")

        assert result['parsed_dose_value'] == 100.0
        assert result['parsed_dose_unit'] == 'mg'
        assert result['parsed_route'] == 'SC'
