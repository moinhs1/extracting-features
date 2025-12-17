# tests/test_phy_extractor.py
import pytest
import pandas as pd
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPhyExtractorInit:
    """Test PhyExtractor initialization."""

    def test_extractor_initializes(self):
        """Extractor initializes with path."""
        from extractors.phy_extractor import PhyExtractor
        extractor = PhyExtractor(phy_path='/tmp/fake.txt')
        assert extractor.phy_path == Path('/tmp/fake.txt')

    def test_all_concepts_populated(self):
        """All relevant concepts are loaded."""
        from extractors.phy_extractor import PhyExtractor
        extractor = PhyExtractor(phy_path='/tmp/fake.txt')
        assert 'BMI' in extractor.all_concepts
        assert 'Weight' in extractor.all_concepts
        assert len(extractor.all_concepts) > 20


class TestPhyExtractorParsing:
    """Test Phy.txt parsing."""

    def test_parse_sample_data(self):
        """Parse sample Phy.txt data."""
        from extractors.phy_extractor import PhyExtractor

        sample_data = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
100001|P001|MGH|M001|1/15/2020|BMI|PHY||28.5|kg/m2|Dr. Smith|Clinic A|MGH|Outpatient|E001
100001|P001|MGH|M001|1/15/2020|Weight|PHY||185|lbs|Dr. Smith|Clinic A|MGH|Outpatient|E001
100002|P002|MGH|M002|2/20/2020|BMI|PHY||32.1|kg/m2|Dr. Jones|Clinic B|BWH|Outpatient|E002"""

        extractor = PhyExtractor(phy_path='/tmp/fake.txt')
        df = extractor.parse_chunk(StringIO(sample_data))

        assert len(df) == 3
        assert df['EMPI'].iloc[0] == '100001'
        assert df['Concept_Name'].iloc[0] == 'BMI'
        assert df['Result'].iloc[0] == '28.5'
