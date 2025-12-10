# /home/moin/TDA_11_25/module_04_medications/tests/test_rxnorm_mapper.py
"""Tests for RxNorm medication mapping."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExactMatch:
    """Test exact string matching to RxNorm."""

    def test_exact_match_aspirin(self):
        """Exact match for simple drug name."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("aspirin")

        assert result is not None
        assert result['rxcui'] is not None
        assert 'aspirin' in result['rxnorm_name'].lower()

    def test_exact_match_not_found(self):
        """Return None for non-existent drug."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("notarealdrug12345")

        assert result is None

    def test_exact_match_heparin(self):
        """Exact match for heparin."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("heparin")

        assert result is not None
        assert result['rxcui'] == '5224'


class TestFuzzyMatch:
    """Test fuzzy string matching."""

    def test_fuzzy_match_typo(self):
        """Fuzzy match handles minor typos."""
        from extractors.rxnorm_mapper import fuzzy_match

        # Intentional typo
        result = fuzzy_match("aspirn")

        assert result is not None
        assert 'aspirin' in result['rxnorm_name'].lower()

    def test_fuzzy_match_threshold(self):
        """Fuzzy match respects similarity threshold."""
        from extractors.rxnorm_mapper import fuzzy_match

        # Too different - should not match
        result = fuzzy_match("xyz123")

        assert result is None


class TestIngredientExtraction:
    """Test ingredient-level matching."""

    def test_extract_ingredient_from_product(self):
        """Extract ingredient from product name."""
        from extractors.rxnorm_mapper import ingredient_match

        result = ingredient_match("Aspirin 325mg tablet")

        assert result is not None
        assert result['ingredient_name'].lower() == 'aspirin'

    def test_extract_ingredient_heparin_sodium(self):
        """Extract heparin from complex string."""
        from extractors.rxnorm_mapper import ingredient_match

        result = ingredient_match("Heparin sodium 5000 unit/ml injection")

        assert result is not None
        assert 'heparin' in result['ingredient_name'].lower()


class TestFullMappingPipeline:
    """Test complete mapping pipeline."""

    def test_map_medication_exact(self):
        """Full pipeline finds exact match first."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("heparin")

        assert result['rxcui'] is not None
        assert result['mapping_method'] == 'exact'

    def test_map_medication_fuzzy(self):
        """Full pipeline falls back to fuzzy."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("heparn sodium")  # Typo

        assert result['rxcui'] is not None
        assert result['mapping_method'] in ['exact', 'fuzzy', 'ingredient']

    def test_map_medication_failed(self):
        """Full pipeline returns failed for unmappable."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("Supply Of Radiopharmaceutical Agent XYZ")

        assert result['mapping_method'] == 'failed'
