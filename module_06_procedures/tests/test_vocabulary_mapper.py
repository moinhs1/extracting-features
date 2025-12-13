"""Tests for vocabulary mapping."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDirectMapping:
    """Test direct CPT/ICD to CCS mapping."""

    def test_map_cpt_code(self):
        """Map standard CPT code to CCS."""
        from extractors.vocabulary_mapper import map_procedure_code

        result = map_procedure_code('71275', 'CPT')

        assert result is not None
        assert result['mapping_method'] == 'direct'

    def test_map_unknown_code_returns_none(self):
        """Unknown code returns None mapping."""
        from extractors.vocabulary_mapper import map_procedure_code

        result = map_procedure_code('99999', 'CPT')

        assert result is None or result['ccs_category'] is None


class TestFuzzyMapping:
    """Test fuzzy name-based mapping for EPIC codes."""

    def test_fuzzy_match_procedure_name(self):
        """Fuzzy match procedure name to CCS."""
        from extractors.vocabulary_mapper import fuzzy_match_procedure

        # Use a procedure name similar to a CCS description
        result = fuzzy_match_procedure("diagnostic cardiac catheterization procedure")

        assert result is not None
        assert result['mapping_confidence'] >= 0.85

    def test_low_confidence_returns_none(self):
        """Low confidence fuzzy match returns None."""
        from extractors.vocabulary_mapper import fuzzy_match_procedure

        result = fuzzy_match_procedure("COMPLETELY UNKNOWN PROCEDURE XYZ")

        assert result is None or result['mapping_confidence'] < 0.85


class TestBatchMapping:
    """Test batch mapping of procedures."""

    def test_map_procedures_batch(self):
        """Map batch of procedures."""
        from extractors.vocabulary_mapper import map_procedures_batch

        df = pd.DataFrame({
            'code': ['71275', '93306', '31500'],
            'code_type': ['CPT', 'CPT', 'CPT'],
            'procedure_name': ['CTA CHEST', 'ECHO TTE', 'INTUBATION'],
        })

        result = map_procedures_batch(df)

        assert 'ccs_category' in result.columns
        assert 'mapping_method' in result.columns
        assert len(result) == 3
