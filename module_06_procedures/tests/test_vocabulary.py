# module_06_procedures/tests/test_vocabulary.py
"""Tests for vocabulary setup and CCS mapping."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCCSCrosswalk:
    """Test CCS crosswalk loading."""

    def test_load_ccs_crosswalk(self):
        """Load CCS crosswalk CSV."""
        from data.vocabularies.setup_vocabularies import load_ccs_crosswalk

        df = load_ccs_crosswalk()

        assert 'cpt_code' in df.columns or 'code' in df.columns
        assert 'ccs_category' in df.columns
        assert len(df) > 0

    def test_ccs_categories_valid(self):
        """CCS categories are valid format."""
        from data.vocabularies.setup_vocabularies import load_ccs_crosswalk

        df = load_ccs_crosswalk()

        # CCS categories should be valid
        assert df['ccs_category'].notna().all()


class TestCPTMapping:
    """Test CPT to CCS mapping."""

    def test_map_cpt_to_ccs(self):
        """Map CPT code to CCS category."""
        from data.vocabularies.setup_vocabularies import map_cpt_to_ccs

        # CTA chest should map to a CCS category
        result = map_cpt_to_ccs('71275')

        assert result is not None
        assert 'ccs_category' in result

    def test_map_cpt_to_ccs_known_code(self):
        """Map known CPT code (71275 for CTA chest)."""
        from data.vocabularies.setup_vocabularies import map_cpt_to_ccs

        result = map_cpt_to_ccs('71275')

        assert result is not None
        assert result['ccs_category'] is not None
        assert 'ccs_description' in result


class TestSNOMEDDatabase:
    """Test SNOMED database setup."""

    def test_setup_snomed_database(self):
        """SNOMED database can be created."""
        from data.vocabularies.setup_vocabularies import setup_snomed_database
        from config.procedure_config import SNOMED_DB
        import sqlite3

        # Create database
        db_path = setup_snomed_database()

        # Verify it exists
        assert db_path.exists()

        # Verify it has the expected tables
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert 'snomed_procedures' in tables
        assert 'cpt_snomed_mapping' in tables
