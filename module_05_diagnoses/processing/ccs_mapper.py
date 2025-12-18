"""CCS (Clinical Classifications Software) mapper."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict


class CCSMapper:
    """Map ICD codes to CCS categories."""

    def __init__(self, crosswalk_path: Path):
        """
        Load CCS crosswalk file.

        Args:
            crosswalk_path: Path to CSV with icd_code, icd_version, ccs_category, ccs_description
        """
        self.crosswalk = pd.read_csv(crosswalk_path)
        self._build_lookups()

    def _build_lookups(self):
        """Build efficient lookup dictionaries."""
        self._icd10_to_ccs = {}
        self._icd9_to_ccs = {}
        self._ccs_descriptions = {}

        for _, row in self.crosswalk.iterrows():
            code = str(row['icd_code']).upper()
            version = str(row['icd_version'])
            category = int(row['ccs_category'])
            description = row['ccs_description']

            if version == '10':
                self._icd10_to_ccs[code] = category
            else:
                self._icd9_to_ccs[code] = category

            self._ccs_descriptions[category] = description

    def get_ccs_category(self, icd_code: str, version: str) -> Optional[int]:
        """
        Map ICD code to CCS category.

        Args:
            icd_code: ICD-9 or ICD-10 code
            version: '9' or '10'

        Returns:
            CCS category number or None if not found
        """
        lookup = self._icd10_to_ccs if version == '10' else self._icd9_to_ccs
        code = str(icd_code).upper()

        # Try exact match first
        if code in lookup:
            return lookup[code]

        # Try prefix matching (progressively shorter)
        for length in range(len(code) - 1, 2, -1):
            prefix = code[:length]
            if prefix in lookup:
                return lookup[prefix]

        return None

    def get_ccs_description(self, category: int) -> str:
        """Get description for CCS category."""
        return self._ccs_descriptions.get(category, f"CCS Category {category}")

    def categorize_patient_diagnoses(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize all diagnoses for a patient into CCS categories.

        Args:
            diagnoses: DataFrame with icd_code, icd_version, is_preexisting

        Returns:
            DataFrame with ccs_category, ccs_description, diagnosis_count, is_preexisting
        """
        category_counts = {}
        category_preexisting = {}

        for _, row in diagnoses.iterrows():
            category = self.get_ccs_category(row['icd_code'], row['icd_version'])
            if category is None:
                continue

            if category not in category_counts:
                category_counts[category] = 0
                category_preexisting[category] = True

            category_counts[category] += 1
            # If any diagnosis in category is not preexisting, mark as not preexisting
            if not row.get('is_preexisting', True):
                category_preexisting[category] = False

        results = []
        for category, count in category_counts.items():
            results.append({
                'ccs_category': category,
                'ccs_description': self.get_ccs_description(category),
                'diagnosis_count': count,
                'is_preexisting': category_preexisting[category],
            })

        return pd.DataFrame(results)
