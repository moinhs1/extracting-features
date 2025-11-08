"""
LOINC database loading and matching for lab test harmonization.
"""

import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Dict, Optional


class LoincMatcher:
    """Handles LOINC database loading and matching."""

    def __init__(self, loinc_csv_path: str, cache_dir: str = 'cache'):
        self.loinc_csv_path = Path(loinc_csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / 'loinc_database.pkl'
        self.loinc_dict = None

    def load(self) -> Dict:
        """Load LOINC database with caching."""
        raise NotImplementedError("To be implemented")

    def match(self, loinc_code: str) -> Optional[Dict]:
        """Look up LOINC code."""
        raise NotImplementedError("To be implemented")
