"""Tests for medication vocabulary extraction."""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVocabularyExtraction:
    """Test vocabulary extraction from canonical records."""

    def test_extract_vocabulary(self):
        """Extract unique medication strings."""
        from extractors.canonical_extractor import extract_vocabulary

        df = pd.DataFrame({
            'original_string': ['Aspirin 325mg', 'Aspirin 325mg', 'Tylenol 500mg'],
            'parsed_name': ['aspirin', 'aspirin', 'tylenol'],
            'parsed_dose_value': [325.0, 325.0, 500.0],
            'parsed_dose_unit': ['mg', 'mg', 'mg'],
            'parsed_route': ['PO', 'PO', 'PO'],
            'parse_method': ['regex', 'regex', 'regex'],
            'empi': ['100', '200', '300'],
        })

        vocab = extract_vocabulary(df)

        assert len(vocab) == 2
        assert 'Aspirin 325mg' in vocab['original_string'].values
        assert 'Tylenol 500mg' in vocab['original_string'].values

    def test_vocabulary_has_counts(self):
        """Vocabulary includes occurrence counts."""
        from extractors.canonical_extractor import extract_vocabulary

        df = pd.DataFrame({
            'original_string': ['Aspirin 325mg', 'Aspirin 325mg', 'Tylenol 500mg'],
            'parsed_name': ['aspirin', 'aspirin', 'tylenol'],
            'parsed_dose_value': [325.0, 325.0, 500.0],
            'parsed_dose_unit': ['mg', 'mg', 'mg'],
            'parsed_route': ['PO', 'PO', 'PO'],
            'parse_method': ['regex', 'regex', 'regex'],
            'empi': ['100', '200', '300'],
        })

        vocab = extract_vocabulary(df)

        assert 'count' in vocab.columns
        aspirin_row = vocab[vocab['original_string'] == 'Aspirin 325mg']
        assert aspirin_row['count'].values[0] == 2
