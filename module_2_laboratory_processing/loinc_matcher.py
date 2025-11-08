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
        """
        Load and parse LOINC database with caching.

        Returns:
            dict: {loinc_code: {component, system, units, name, ...}}
        """
        # Check cache first
        if self.cache_path.exists():
            print(f"Loading LOINC database from cache...")
            with open(self.cache_path, 'rb') as f:
                self.loinc_dict = pickle.load(f)
            print(f"  Loaded {len(self.loinc_dict)} LOINC codes from cache")
            return self.loinc_dict

        # Parse LOINC CSV
        print(f"Parsing LOINC database from {self.loinc_csv_path}...")

        if not self.loinc_csv_path.exists():
            raise FileNotFoundError(
                f"LOINC database not found at {self.loinc_csv_path}. "
                f"Please download from https://loinc.org and place at this path."
            )

        loinc_df = pd.read_csv(self.loinc_csv_path, dtype=str, low_memory=False)

        # Filter to laboratory tests only
        # CLASSTYPE=1 indicates laboratory tests, CLASSTYPE=2 is clinical/non-lab
        if 'CLASSTYPE' in loinc_df.columns:
            loinc_df = loinc_df[loinc_df['CLASSTYPE'] == '1'].copy()
        elif 'CLASS' in loinc_df.columns:
            # Fallback for test data that uses CLASS=LABORATORY
            loinc_df = loinc_df[loinc_df['CLASS'] == 'LABORATORY'].copy()
        print(f"  Found {len(loinc_df)} laboratory tests")

        # Build lookup dictionary
        self.loinc_dict = {}
        for _, row in loinc_df.iterrows():
            loinc_code = row['LOINC_NUM']
            self.loinc_dict[loinc_code] = {
                'component': row.get('COMPONENT', ''),
                'property': row.get('PROPERTY', ''),
                'system': row.get('SYSTEM', ''),
                'scale': row.get('SCALE_TYP', ''),
                'method': row.get('METHOD_TYP', ''),
                'units': row.get('EXAMPLE_UNITS', ''),
                'ucum_units': row.get('EXAMPLE_UCUM_UNITS', ''),
                'name': row.get('LONG_COMMON_NAME', ''),
                'short_name': row.get('SHORTNAME', ''),
                'class': row.get('CLASS', '')
            }

        # Cache for future use
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.loinc_dict, f)
        print(f"  Cached LOINC database to {self.cache_path}")

        return self.loinc_dict

    def match(self, loinc_code: str) -> Optional[Dict]:
        """Look up LOINC code."""
        raise NotImplementedError("To be implemented")
