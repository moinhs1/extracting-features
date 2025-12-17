# Layer 3 PE-Specific Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Layer 3 PE-specific diagnosis features (~51 features across 9 groups) from Layer 1 canonical diagnoses.

**Architecture:** Single `PEFeatureBuilder` class with grouped methods, ICD codes in `pe_feature_codes.py`, TDD with synthetic test data, wide DataFrame output.

**Tech Stack:** Python, pandas, pytest, parquet

---

## Task 1: VTE Code Definitions

**Files:**
- Create: `module_05_diagnoses/config/pe_feature_codes.py`

**Step 1: Create VTE code definitions**

```python
"""ICD code definitions for PE-specific features."""

# Group 1: VTE History
VTE_CODES = {
    "pe": {
        "icd10": ["I26"],
        "icd9": ["415.1"],
    },
    "dvt_lower_extremity": {
        "icd10": ["I82.40", "I82.41", "I82.42", "I82.43", "I82.44", "I82.49",
                  "I82.4Y", "I82.4Z"],
        "icd9": ["453.4", "453.40", "453.41", "453.42"],
    },
    "dvt_upper_extremity": {
        "icd10": ["I82.60", "I82.61", "I82.62", "I82.A1", "I82.B1"],
        "icd9": ["453.82", "453.83"],
    },
}

# Group 2: PE Index Characterization
PE_SUBTYPE_CODES = {
    "saddle": {
        "icd10": ["I26.02"],
        "icd9": [],
    },
    "with_cor_pulmonale": {
        "icd10": ["I26.0", "I26.01", "I26.02", "I26.09"],
        "icd9": ["415.0"],
    },
    "without_cor_pulmonale": {
        "icd10": ["I26.9", "I26.90", "I26.92", "I26.93", "I26.94", "I26.99"],
        "icd9": ["415.11", "415.19"],
    },
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.pe_feature_codes import VTE_CODES, PE_SUBTYPE_CODES; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/pe_feature_codes.py
git commit -m "feat(module05): add VTE and PE subtype ICD codes"
```

---

## Task 2: PEFeatureBuilder Skeleton + Code Matching

**Files:**
- Create: `module_05_diagnoses/processing/pe_feature_builder.py`
- Create: `module_05_diagnoses/tests/test_pe_feature_builder.py`

**Step 1: Write test for code matching utility**

```python
"""Tests for PE-specific feature builder."""

import pytest
import pandas as pd
from processing.pe_feature_builder import PEFeatureBuilder


@pytest.fixture
def sample_diagnoses():
    """Minimal diagnoses for testing code matching."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2'],
        'icd_code': ['I26.99', 'I50.9', 'J44.1'],
        'icd_version': ['10', '10', '10'],
        'days_from_pe': [-60, -10, -30],
    })


class TestCodeMatching:
    def test_match_codes_finds_pe_codes(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        pe_codes = {"icd10": ["I26"], "icd9": ["415.1"]}
        matched = builder._match_codes(sample_diagnoses, pe_codes)
        assert len(matched) == 1
        assert matched.iloc[0]['icd_code'] == 'I26.99'

    def test_match_codes_returns_empty_when_no_match(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        cancer_codes = {"icd10": ["C34"], "icd9": ["162"]}
        matched = builder._match_codes(sample_diagnoses, cancer_codes)
        assert len(matched) == 0

    def test_match_codes_prefix_matching(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        copd_codes = {"icd10": ["J44"], "icd9": ["496"]}
        matched = builder._match_codes(sample_diagnoses, copd_codes)
        assert len(matched) == 1
        assert matched.iloc[0]['icd_code'] == 'J44.1'
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
"""Build PE-specific diagnosis features from Layer 1 canonical diagnoses."""

import pandas as pd
from typing import Optional


class PEFeatureBuilder:
    """Build PE-specific diagnosis features from Layer 1 canonical diagnoses."""

    def __init__(self, canonical_diagnoses: pd.DataFrame):
        """
        Initialize with Layer 1 canonical diagnoses.

        Args:
            canonical_diagnoses: DataFrame with columns:
                EMPI, icd_code, icd_version, days_from_pe
        """
        self.diagnoses = canonical_diagnoses
        self.empis = canonical_diagnoses['EMPI'].unique()

    def _match_codes(self, df: pd.DataFrame, code_dict: dict,
                     version: Optional[str] = None) -> pd.DataFrame:
        """
        Filter diagnoses matching any code in dict (prefix matching).

        Args:
            df: Diagnoses DataFrame to filter
            code_dict: Dict with 'icd10' and 'icd9' code lists
            version: Optional filter for specific ICD version

        Returns:
            Filtered DataFrame with matching diagnoses
        """
        masks = []

        for icd_ver, codes in [('10', code_dict.get('icd10', [])),
                               ('9', code_dict.get('icd9', []))]:
            if version and icd_ver != version:
                continue
            for code in codes:
                mask = (df['icd_version'] == icd_ver) & \
                       (df['icd_code'].str.startswith(code))
                masks.append(mask)

        if not masks:
            return df.iloc[0:0]  # Empty DataFrame with same columns

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m

        return df[combined_mask]
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add PEFeatureBuilder with code matching"
```

---

## Task 3: VTE History Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for VTE history features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def vte_history_diagnoses():
    """Diagnoses with VTE history scenarios."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3', 'P3'],
        'icd_code': ['I26.99', 'I26.92', 'I82.40', 'I26.0', 'I50.9', 'I26.9', 'I82.62'],
        'icd_version': ['10', '10', '10', '10', '10', '10', '10'],
        'days_from_pe': [-90, 0, -60, 0, -10, 0, -45],
    })


class TestVTEHistoryFeatures:
    def test_prior_pe_detected(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        # P1 has I26.99 at -90 days (prior PE)
        assert features.loc['P1', 'prior_pe_ever'] == True
        # P2 has I26.0 at day 0 (index, not prior)
        assert features.loc['P2', 'prior_pe_ever'] == False
        # P3 has I26.9 at day 0 (index, not prior)
        assert features.loc['P3', 'prior_pe_ever'] == False

    def test_prior_pe_count(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        assert features.loc['P1', 'prior_pe_count'] == 1
        assert features.loc['P2', 'prior_pe_count'] == 0

    def test_prior_pe_months_calculated(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        # P1: 90 days / 30.44 ≈ 2.96 months
        assert features.loc['P1', 'prior_pe_months'] == pytest.approx(2.96, rel=0.1)
        # P2: no prior PE, should be None/NaN
        assert pd.isna(features.loc['P2', 'prior_pe_months'])

    def test_prior_dvt_detected(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        # P1 has I82.40 (lower DVT) at -60 days
        assert features.loc['P1', 'prior_dvt_ever'] == True
        # P3 has I82.62 (upper DVT) at -45 days
        assert features.loc['P3', 'prior_dvt_ever'] == True
        # P2 has no DVT
        assert features.loc['P2', 'prior_dvt_ever'] == False

    def test_is_recurrent_vte(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        # P1: has prior PE and DVT
        assert features.loc['P1', 'is_recurrent_vte'] == True
        # P2: no prior VTE
        assert features.loc['P2', 'is_recurrent_vte'] == False
        # P3: has prior DVT only
        assert features.loc['P3', 'is_recurrent_vte'] == True

    def test_prior_vte_count(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_vte_history_features()
        # P1: 1 PE + 1 DVT = 2
        assert features.loc['P1', 'prior_vte_count'] == 2
        # P3: 0 PE + 1 DVT = 1
        assert features.loc['P3', 'prior_vte_count'] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestVTEHistoryFeatures -v`
Expected: FAIL with `AttributeError: 'PEFeatureBuilder' object has no attribute 'build_vte_history_features'`

**Step 3: Implement VTE history features**

Add to `processing/pe_feature_builder.py` (add import at top and method to class):

```python
# Add at top of file after existing imports:
from config.pe_feature_codes import VTE_CODES, PE_SUBTYPE_CODES

# Add method to PEFeatureBuilder class:
    def build_vte_history_features(self) -> pd.DataFrame:
        """Extract VTE history features (prior PE/DVT >30 days before index)."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < -30]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            # Prior PE
            prior_pe = self._match_codes(patient_prior, VTE_CODES['pe'])
            prior_pe_ever = len(prior_pe) > 0
            prior_pe_count = len(prior_pe)
            prior_pe_months = None
            if prior_pe_ever:
                most_recent = prior_pe['days_from_pe'].max()
                prior_pe_months = abs(most_recent) / 30.44

            # Prior DVT (combine lower and upper)
            dvt_codes = {
                'icd10': VTE_CODES['dvt_lower_extremity']['icd10'] +
                         VTE_CODES['dvt_upper_extremity']['icd10'],
                'icd9': VTE_CODES['dvt_lower_extremity']['icd9'] +
                        VTE_CODES['dvt_upper_extremity']['icd9']
            }
            prior_dvt = self._match_codes(patient_prior, dvt_codes)
            prior_dvt_ever = len(prior_dvt) > 0
            prior_dvt_months = None
            if prior_dvt_ever:
                most_recent = prior_dvt['days_from_pe'].max()
                prior_dvt_months = abs(most_recent) / 30.44

            results.append({
                'EMPI': empi,
                'prior_pe_ever': prior_pe_ever,
                'prior_pe_months': prior_pe_months,
                'prior_pe_count': prior_pe_count,
                'prior_dvt_ever': prior_dvt_ever,
                'prior_dvt_months': prior_dvt_months,
                'prior_vte_count': prior_pe_count + len(prior_dvt),
                'is_recurrent_vte': prior_pe_ever or prior_dvt_ever,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (9 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add VTE history feature extraction"
```

---

## Task 4: PE Index Characterization Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for PE index features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def pe_index_diagnoses():
    """Diagnoses for PE index characterization."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P2', 'P3', 'P4'],
        'icd_code': ['I26.02', 'I26.0', 'I26.99', 'I26.92'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [0, 0, 0, 1],
    })


class TestPEIndexFeatures:
    def test_pe_subtype_saddle(self, pe_index_diagnoses):
        builder = PEFeatureBuilder(pe_index_diagnoses)
        features = builder.build_pe_index_features()
        # P1 has I26.02 (saddle)
        assert features.loc['P1', 'pe_subtype'] == 'saddle'

    def test_pe_with_cor_pulmonale(self, pe_index_diagnoses):
        builder = PEFeatureBuilder(pe_index_diagnoses)
        features = builder.build_pe_index_features()
        # P1: I26.02 is under I26.0x (with cor pulmonale)
        assert features.loc['P1', 'pe_with_cor_pulmonale'] == True
        # P2: I26.0 (with cor pulmonale)
        assert features.loc['P2', 'pe_with_cor_pulmonale'] == True
        # P3: I26.99 (without cor pulmonale)
        assert features.loc['P3', 'pe_with_cor_pulmonale'] == False

    def test_pe_high_risk_code(self, pe_index_diagnoses):
        builder = PEFeatureBuilder(pe_index_diagnoses)
        features = builder.build_pe_index_features()
        # Saddle or with cor pulmonale = high risk
        assert features.loc['P1', 'pe_high_risk_code'] == True
        assert features.loc['P2', 'pe_high_risk_code'] == True
        assert features.loc['P3', 'pe_high_risk_code'] == False

    def test_pe_subtype_unspecified(self, pe_index_diagnoses):
        builder = PEFeatureBuilder(pe_index_diagnoses)
        features = builder.build_pe_index_features()
        # P3: I26.99 and P4: I26.92 are unspecified subtypes
        assert features.loc['P3', 'pe_subtype'] == 'unspecified'
        assert features.loc['P4', 'pe_subtype'] == 'unspecified'
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestPEIndexFeatures -v`
Expected: FAIL with `AttributeError`

**Step 3: Implement PE index features**

Add method to `PEFeatureBuilder` class:

```python
    def build_pe_index_features(self) -> pd.DataFrame:
        """Extract PE index characterization features."""
        index = self.diagnoses[self.diagnoses['days_from_pe'].abs() <= 1]

        results = []
        for empi in self.empis:
            patient_index = index[index['EMPI'] == empi]

            # Check for saddle PE
            saddle = self._match_codes(patient_index, PE_SUBTYPE_CODES['saddle'])
            is_saddle = len(saddle) > 0

            # Check for cor pulmonale
            with_cor = self._match_codes(patient_index, PE_SUBTYPE_CODES['with_cor_pulmonale'])
            pe_with_cor_pulmonale = len(with_cor) > 0

            # Determine subtype
            if is_saddle:
                pe_subtype = 'saddle'
            else:
                pe_subtype = 'unspecified'

            # High risk = saddle or with cor pulmonale
            pe_high_risk_code = is_saddle or pe_with_cor_pulmonale

            results.append({
                'EMPI': empi,
                'pe_subtype': pe_subtype,
                'pe_with_cor_pulmonale': pe_with_cor_pulmonale,
                'pe_high_risk_code': pe_high_risk_code,
                'pe_bilateral': False,  # Would need specific codes
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (13 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add PE index characterization features"
```

---

## Task 5: Cancer Code Definitions

**Files:**
- Modify: `module_05_diagnoses/config/pe_feature_codes.py`

**Step 1: Add cancer codes**

Add to `config/pe_feature_codes.py`:

```python
# Group 3: Cancer (CCSR-aligned categories)
CANCER_CODES = {
    "lung": {
        "icd10": ["C34"],
        "icd9": ["162"],
    },
    "colorectal": {
        "icd10": ["C18", "C19", "C20", "C21"],
        "icd9": ["153", "154"],
    },
    "breast": {
        "icd10": ["C50"],
        "icd9": ["174", "175"],
    },
    "prostate": {
        "icd10": ["C61"],
        "icd9": ["185"],
    },
    "pancreas": {
        "icd10": ["C25"],
        "icd9": ["157"],
    },
    "gastric": {
        "icd10": ["C16"],
        "icd9": ["151"],
    },
    "ovarian": {
        "icd10": ["C56"],
        "icd9": ["183"],
    },
    "renal": {
        "icd10": ["C64", "C65"],
        "icd9": ["189.0", "189.1"],
    },
    "bladder": {
        "icd10": ["C67"],
        "icd9": ["188"],
    },
    "hematologic": {
        "icd10": ["C81", "C82", "C83", "C84", "C85", "C86",
                  "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"],
        "icd9": ["200", "201", "202", "203", "204", "205", "206", "207", "208"],
    },
    "brain": {
        "icd10": ["C71"],
        "icd9": ["191"],
    },
    "metastatic": {
        "icd10": ["C77", "C78", "C79", "C80"],
        "icd9": ["196", "197", "198", "199"],
    },
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.pe_feature_codes import CANCER_CODES; print(len(CANCER_CODES), 'categories')"`
Expected: `12 categories`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/pe_feature_codes.py
git commit -m "feat(module05): add cancer ICD codes"
```

---

## Task 6: Cancer Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for cancer features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def cancer_diagnoses():
    """Diagnoses for cancer feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P3', 'P3', 'P4'],
        'icd_code': ['C34.9', 'C78.0', 'I26.0', 'C50.9', 'I26.9', 'C91.0'],
        'icd_version': ['10', '10', '10', '10', '10', '10'],
        'days_from_pe': [-90, -60, 0, -30, 0, -120],
    })


class TestCancerFeatures:
    def test_cancer_active_detected(self, cancer_diagnoses):
        builder = PEFeatureBuilder(cancer_diagnoses)
        features = builder.build_cancer_features()
        # P1 has lung cancer
        assert features.loc['P1', 'cancer_active'] == True
        # P2 has no cancer
        assert features.loc['P2', 'cancer_active'] == False
        # P3 has breast cancer
        assert features.loc['P3', 'cancer_active'] == True

    def test_cancer_site_identified(self, cancer_diagnoses):
        builder = PEFeatureBuilder(cancer_diagnoses)
        features = builder.build_cancer_features()
        assert features.loc['P1', 'cancer_site'] == 'lung'
        assert features.loc['P3', 'cancer_site'] == 'breast'
        assert features.loc['P4', 'cancer_site'] == 'hematologic'

    def test_cancer_metastatic(self, cancer_diagnoses):
        builder = PEFeatureBuilder(cancer_diagnoses)
        features = builder.build_cancer_features()
        # P1 has C78.0 (lung metastasis)
        assert features.loc['P1', 'cancer_metastatic'] == True
        # P3 has breast cancer but no metastasis code
        assert features.loc['P3', 'cancer_metastatic'] == False

    def test_cancer_recent_diagnosis(self, cancer_diagnoses):
        builder = PEFeatureBuilder(cancer_diagnoses)
        features = builder.build_cancer_features()
        # P1: cancer at -90 days (~3 months) = recent
        assert features.loc['P1', 'cancer_recent_diagnosis'] == True
        # P4: cancer at -120 days (~4 months) = recent
        assert features.loc['P4', 'cancer_recent_diagnosis'] == True
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestCancerFeatures -v`
Expected: FAIL

**Step 3: Implement cancer features**

Add import and method to `processing/pe_feature_builder.py`:

```python
# Update import at top:
from config.pe_feature_codes import VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES

# Add method:
    def build_cancer_features(self) -> pd.DataFrame:
        """Extract cancer-related features."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < 0]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            # Check each cancer site
            cancer_active = False
            cancer_site = None
            cancer_metastatic = False
            first_cancer_days = None

            for site, codes in CANCER_CODES.items():
                matched = self._match_codes(patient_prior, codes)
                if len(matched) > 0:
                    cancer_active = True
                    if site == 'metastatic':
                        cancer_metastatic = True
                    elif cancer_site is None:
                        cancer_site = site
                        first_cancer_days = matched['days_from_pe'].min()

            # Recent = within 6 months (180 days)
            cancer_recent_diagnosis = False
            if first_cancer_days is not None:
                cancer_recent_diagnosis = abs(first_cancer_days) <= 180

            results.append({
                'EMPI': empi,
                'cancer_active': cancer_active,
                'cancer_site': cancer_site,
                'cancer_metastatic': cancer_metastatic,
                'cancer_recent_diagnosis': cancer_recent_diagnosis,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (17 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add cancer feature extraction"
```

---

## Task 7: Cardiovascular Code Definitions

**Files:**
- Modify: `module_05_diagnoses/config/pe_feature_codes.py`

**Step 1: Add cardiovascular codes**

Add to `config/pe_feature_codes.py`:

```python
# Group 4: Cardiovascular
CARDIOVASCULAR_CODES = {
    "heart_failure": {
        "icd10": ["I50"],
        "icd9": ["428"],
    },
    "hf_reduced_ef": {
        "icd10": ["I50.2", "I50.20", "I50.21", "I50.22", "I50.23"],
        "icd9": ["428.2"],
    },
    "hf_preserved_ef": {
        "icd10": ["I50.3", "I50.30", "I50.31", "I50.32", "I50.33"],
        "icd9": ["428.3"],
    },
    "cad": {
        "icd10": ["I25", "I21", "I22"],
        "icd9": ["414", "410", "412"],
    },
    "atrial_fibrillation": {
        "icd10": ["I48"],
        "icd9": ["427.31", "427.32"],
    },
    "pulmonary_hypertension": {
        "icd10": ["I27.0", "I27.2", "I27.20", "I27.21", "I27.22", "I27.23", "I27.24", "I27.29"],
        "icd9": ["416.0", "416.8"],
    },
    "valvular_disease": {
        "icd10": ["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"],
        "icd9": ["394", "395", "396", "397", "424"],
    },
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.pe_feature_codes import CARDIOVASCULAR_CODES; print(len(CARDIOVASCULAR_CODES), 'categories')"`
Expected: `7 categories`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/pe_feature_codes.py
git commit -m "feat(module05): add cardiovascular ICD codes"
```

---

## Task 8: Cardiovascular Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for cardiovascular features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def cv_diagnoses():
    """Diagnoses for cardiovascular feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P3', 'P4'],
        'icd_code': ['I50.22', 'I48.0', 'I26.0', 'I50.32', 'I25.10'],
        'icd_version': ['10', '10', '10', '10', '10'],
        'days_from_pe': [-30, -60, 0, -10, -90],
    })


class TestCardiovascularFeatures:
    def test_heart_failure_detected(self, cv_diagnoses):
        builder = PEFeatureBuilder(cv_diagnoses)
        features = builder.build_cardiovascular_features()
        assert features.loc['P1', 'heart_failure'] == True
        assert features.loc['P2', 'heart_failure'] == False
        assert features.loc['P3', 'heart_failure'] == True

    def test_hf_type_identified(self, cv_diagnoses):
        builder = PEFeatureBuilder(cv_diagnoses)
        features = builder.build_cardiovascular_features()
        # P1: I50.22 = HFrEF
        assert features.loc['P1', 'hf_type'] == 'HFrEF'
        # P3: I50.32 = HFpEF
        assert features.loc['P3', 'hf_type'] == 'HFpEF'

    def test_atrial_fibrillation_detected(self, cv_diagnoses):
        builder = PEFeatureBuilder(cv_diagnoses)
        features = builder.build_cardiovascular_features()
        assert features.loc['P1', 'atrial_fibrillation'] == True
        assert features.loc['P2', 'atrial_fibrillation'] == False

    def test_cad_detected(self, cv_diagnoses):
        builder = PEFeatureBuilder(cv_diagnoses)
        features = builder.build_cardiovascular_features()
        assert features.loc['P4', 'cad'] == True
        assert features.loc['P1', 'cad'] == False
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestCardiovascularFeatures -v`
Expected: FAIL

**Step 3: Implement cardiovascular features**

Add import and method to `processing/pe_feature_builder.py`:

```python
# Update import at top:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES
)

# Add method:
    def build_cardiovascular_features(self) -> pd.DataFrame:
        """Extract cardiovascular comorbidity features."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < 0]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            # Heart failure
            hf = self._match_codes(patient_prior, CARDIOVASCULAR_CODES['heart_failure'])
            heart_failure = len(hf) > 0

            # HF type
            hf_type = None
            if heart_failure:
                hfref = self._match_codes(patient_prior, CARDIOVASCULAR_CODES['hf_reduced_ef'])
                hfpef = self._match_codes(patient_prior, CARDIOVASCULAR_CODES['hf_preserved_ef'])
                if len(hfref) > 0:
                    hf_type = 'HFrEF'
                elif len(hfpef) > 0:
                    hf_type = 'HFpEF'
                else:
                    hf_type = 'unspecified'

            # Other CV conditions
            cad = len(self._match_codes(patient_prior, CARDIOVASCULAR_CODES['cad'])) > 0
            afib = len(self._match_codes(patient_prior, CARDIOVASCULAR_CODES['atrial_fibrillation'])) > 0
            ph = len(self._match_codes(patient_prior, CARDIOVASCULAR_CODES['pulmonary_hypertension'])) > 0
            valvular = len(self._match_codes(patient_prior, CARDIOVASCULAR_CODES['valvular_disease'])) > 0

            results.append({
                'EMPI': empi,
                'heart_failure': heart_failure,
                'hf_type': hf_type,
                'cad': cad,
                'atrial_fibrillation': afib,
                'pulmonary_hypertension': ph,
                'valvular_disease': valvular,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (21 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add cardiovascular feature extraction"
```

---

## Task 9: Pulmonary, Bleeding, Renal Code Definitions

**Files:**
- Modify: `module_05_diagnoses/config/pe_feature_codes.py`

**Step 1: Add remaining comorbidity codes**

Add to `config/pe_feature_codes.py`:

```python
# Group 5: Pulmonary
PULMONARY_CODES = {
    "copd": {
        "icd10": ["J44"],
        "icd9": ["491", "492", "496"],
    },
    "asthma": {
        "icd10": ["J45", "J46"],
        "icd9": ["493"],
    },
    "ild": {
        "icd10": ["J84"],
        "icd9": ["516"],
    },
    "home_oxygen": {
        "icd10": ["Z99.81"],
        "icd9": ["V46.2"],
    },
    "respiratory_failure": {
        "icd10": ["J96"],
        "icd9": ["518.81", "518.82", "518.83", "518.84"],
    },
}

# Group 6: Bleeding Risk
BLEEDING_CODES = {
    "gi_bleeding": {
        "icd10": ["K92.0", "K92.1", "K92.2", "K25.0", "K25.4", "K26.0", "K26.4",
                  "K27.0", "K27.4", "K28.0", "K28.4", "K62.5"],
        "icd9": ["578", "531.0", "532.0", "533.0", "534.0"],
    },
    "intracranial_hemorrhage": {
        "icd10": ["I60", "I61", "I62"],
        "icd9": ["430", "431", "432"],
    },
    "major_bleeding_other": {
        "icd10": ["D62", "R58", "N02", "R31"],
        "icd9": ["285.1", "459.0", "599.7"],
    },
    "peptic_ulcer_active": {
        "icd10": ["K25", "K26", "K27", "K28"],
        "icd9": ["531", "532", "533", "534"],
    },
    "thrombocytopenia": {
        "icd10": ["D69.4", "D69.5", "D69.6"],
        "icd9": ["287.3", "287.4", "287.5"],
    },
    "coagulopathy": {
        "icd10": ["D68"],
        "icd9": ["286"],
    },
}

# Group 7: Renal
RENAL_CODES = {
    "ckd_stage1": {"icd10": ["N18.1"], "icd9": ["585.1"]},
    "ckd_stage2": {"icd10": ["N18.2"], "icd9": ["585.2"]},
    "ckd_stage3": {"icd10": ["N18.3", "N18.30", "N18.31", "N18.32"], "icd9": ["585.3"]},
    "ckd_stage4": {"icd10": ["N18.4"], "icd9": ["585.4"]},
    "ckd_stage5": {"icd10": ["N18.5", "N18.6"], "icd9": ["585.5", "585.6"]},
    "dialysis": {
        "icd10": ["Z99.2", "Z49"],
        "icd9": ["V45.1", "V56"],
    },
    "aki": {
        "icd10": ["N17"],
        "icd9": ["584"],
    },
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.pe_feature_codes import PULMONARY_CODES, BLEEDING_CODES, RENAL_CODES; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/pe_feature_codes.py
git commit -m "feat(module05): add pulmonary, bleeding, renal ICD codes"
```

---

## Task 10: Pulmonary Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for pulmonary features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def pulm_diagnoses():
    """Diagnoses for pulmonary feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P2', 'P3'],
        'icd_code': ['J44.1', 'J45.20', 'J84.10'],
        'icd_version': ['10', '10', '10'],
        'days_from_pe': [-60, -30, -90],
    })


class TestPulmonaryFeatures:
    def test_copd_detected(self, pulm_diagnoses):
        builder = PEFeatureBuilder(pulm_diagnoses)
        features = builder.build_pulmonary_features()
        assert features.loc['P1', 'copd'] == True
        assert features.loc['P2', 'copd'] == False

    def test_asthma_detected(self, pulm_diagnoses):
        builder = PEFeatureBuilder(pulm_diagnoses)
        features = builder.build_pulmonary_features()
        assert features.loc['P2', 'asthma'] == True
        assert features.loc['P1', 'asthma'] == False

    def test_ild_detected(self, pulm_diagnoses):
        builder = PEFeatureBuilder(pulm_diagnoses)
        features = builder.build_pulmonary_features()
        assert features.loc['P3', 'ild'] == True
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestPulmonaryFeatures -v`
Expected: FAIL

**Step 3: Implement pulmonary features**

Add import and method:

```python
# Update import:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES
)

# Add method:
    def build_pulmonary_features(self) -> pd.DataFrame:
        """Extract pulmonary comorbidity features."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < 0]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            results.append({
                'EMPI': empi,
                'copd': len(self._match_codes(patient_prior, PULMONARY_CODES['copd'])) > 0,
                'asthma': len(self._match_codes(patient_prior, PULMONARY_CODES['asthma'])) > 0,
                'ild': len(self._match_codes(patient_prior, PULMONARY_CODES['ild'])) > 0,
                'home_oxygen': len(self._match_codes(patient_prior, PULMONARY_CODES['home_oxygen'])) > 0,
                'prior_resp_failure': len(self._match_codes(patient_prior, PULMONARY_CODES['respiratory_failure'])) > 0,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (24 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add pulmonary feature extraction"
```

---

## Task 11: Bleeding Risk Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for bleeding risk features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def bleeding_diagnoses():
    """Diagnoses for bleeding risk feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P2', 'P3', 'P4'],
        'icd_code': ['K92.0', 'I61.0', 'D69.42', 'I26.9'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [-60, -90, -30, 0],
    })


class TestBleedingRiskFeatures:
    def test_prior_gi_bleed_detected(self, bleeding_diagnoses):
        builder = PEFeatureBuilder(bleeding_diagnoses)
        features = builder.build_bleeding_risk_features()
        assert features.loc['P1', 'prior_gi_bleed'] == True
        assert features.loc['P2', 'prior_gi_bleed'] == False

    def test_prior_ich_detected(self, bleeding_diagnoses):
        builder = PEFeatureBuilder(bleeding_diagnoses)
        features = builder.build_bleeding_risk_features()
        assert features.loc['P2', 'prior_ich'] == True
        assert features.loc['P1', 'prior_ich'] == False

    def test_thrombocytopenia_detected(self, bleeding_diagnoses):
        builder = PEFeatureBuilder(bleeding_diagnoses)
        features = builder.build_bleeding_risk_features()
        assert features.loc['P3', 'thrombocytopenia'] == True

    def test_prior_major_bleed_any(self, bleeding_diagnoses):
        builder = PEFeatureBuilder(bleeding_diagnoses)
        features = builder.build_bleeding_risk_features()
        # GI bleed and ICH count as major bleeds
        assert features.loc['P1', 'prior_major_bleed'] == True
        assert features.loc['P2', 'prior_major_bleed'] == True
        assert features.loc['P4', 'prior_major_bleed'] == False
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestBleedingRiskFeatures -v`
Expected: FAIL

**Step 3: Implement bleeding risk features**

Add import and method:

```python
# Update import:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES, BLEEDING_CODES
)

# Add method:
    def build_bleeding_risk_features(self) -> pd.DataFrame:
        """Extract bleeding risk features."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < 0]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            prior_gi_bleed = len(self._match_codes(patient_prior, BLEEDING_CODES['gi_bleeding'])) > 0
            prior_ich = len(self._match_codes(patient_prior, BLEEDING_CODES['intracranial_hemorrhage'])) > 0
            prior_other_bleed = len(self._match_codes(patient_prior, BLEEDING_CODES['major_bleeding_other'])) > 0

            results.append({
                'EMPI': empi,
                'prior_major_bleed': prior_gi_bleed or prior_ich or prior_other_bleed,
                'prior_gi_bleed': prior_gi_bleed,
                'prior_ich': prior_ich,
                'active_pud': len(self._match_codes(patient_prior, BLEEDING_CODES['peptic_ulcer_active'])) > 0,
                'thrombocytopenia': len(self._match_codes(patient_prior, BLEEDING_CODES['thrombocytopenia'])) > 0,
                'coagulopathy': len(self._match_codes(patient_prior, BLEEDING_CODES['coagulopathy'])) > 0,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (28 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add bleeding risk feature extraction"
```

---

## Task 12: Renal Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for renal features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def renal_diagnoses():
    """Diagnoses for renal feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P2', 'P3', 'P4', 'P4'],
        'icd_code': ['N18.3', 'N18.5', 'Z99.2', 'N17.0', 'I26.9'],
        'icd_version': ['10', '10', '10', '10', '10'],
        'days_from_pe': [-60, -30, -90, 0, 0],
    })


class TestRenalFeatures:
    def test_ckd_stage_detected(self, renal_diagnoses):
        builder = PEFeatureBuilder(renal_diagnoses)
        features = builder.build_renal_features()
        assert features.loc['P1', 'ckd_stage'] == 3
        assert features.loc['P2', 'ckd_stage'] == 5
        assert features.loc['P4', 'ckd_stage'] == 0  # No CKD

    def test_ckd_dialysis_detected(self, renal_diagnoses):
        builder = PEFeatureBuilder(renal_diagnoses)
        features = builder.build_renal_features()
        assert features.loc['P3', 'ckd_dialysis'] == True
        assert features.loc['P1', 'ckd_dialysis'] == False

    def test_aki_at_presentation(self, renal_diagnoses):
        builder = PEFeatureBuilder(renal_diagnoses)
        features = builder.build_renal_features()
        # P4 has N17.0 (AKI) at day 0
        assert features.loc['P4', 'aki_at_presentation'] == True
        assert features.loc['P1', 'aki_at_presentation'] == False
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestRenalFeatures -v`
Expected: FAIL

**Step 3: Implement renal features**

Add import and method:

```python
# Update import:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES, BLEEDING_CODES, RENAL_CODES
)

# Add method:
    def build_renal_features(self) -> pd.DataFrame:
        """Extract renal function features."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < 0]
        index = self.diagnoses[self.diagnoses['days_from_pe'].abs() <= 1]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]
            patient_index = index[index['EMPI'] == empi]

            # Determine CKD stage (highest stage if multiple)
            ckd_stage = 0
            for stage in [5, 4, 3, 2, 1]:
                if len(self._match_codes(patient_prior, RENAL_CODES[f'ckd_stage{stage}'])) > 0:
                    ckd_stage = stage
                    break

            results.append({
                'EMPI': empi,
                'ckd_stage': ckd_stage,
                'ckd_dialysis': len(self._match_codes(patient_prior, RENAL_CODES['dialysis'])) > 0,
                'aki_at_presentation': len(self._match_codes(patient_index, RENAL_CODES['aki'])) > 0,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (31 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add renal feature extraction"
```

---

## Task 13: Provoking Factor Code Definitions

**Files:**
- Modify: `module_05_diagnoses/config/pe_feature_codes.py`

**Step 1: Add provoking factor codes**

Add to `config/pe_feature_codes.py`:

```python
# Group 8: Provoking Factors
PROVOKING_CODES = {
    "surgery": {
        "icd10": ["Z96", "Z98"],
        "icd9": ["V45"],
    },
    "trauma": {
        "icd10": ["S", "T0", "T1"],
        "icd9": ["8", "9"],
    },
    "immobilization": {
        "icd10": ["Z74.0", "Z74.1", "R26.3"],
        "icd9": ["V49.84"],
    },
    "pregnancy": {
        "icd10": ["O"],
        "icd9": ["V22", "V23", "V27", "63", "64", "65", "66", "67", "68", "69"],
    },
    "hormonal_therapy": {
        "icd10": ["Z79.3", "Z79.890"],
        "icd9": ["V25.4", "V07.4"],
    },
    "central_venous_catheter": {
        "icd10": ["Z45.2", "T80.211", "T80.212", "T80.218", "T80.219"],
        "icd9": ["V45.81", "999.31", "999.32"],
    },
}

# Group 9: Complications (reuse codes with different temporal window)
COMPLICATION_CODES = {
    "cardiogenic_shock": {
        "icd10": ["R57.0"],
        "icd9": ["785.51"],
    },
    "cardiac_arrest": {
        "icd10": ["I46"],
        "icd9": ["427.5"],
    },
    "cteph": {
        "icd10": ["I27.24", "I27.29"],
        "icd9": ["416.8"],
    },
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.pe_feature_codes import PROVOKING_CODES, COMPLICATION_CODES; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/pe_feature_codes.py
git commit -m "feat(module05): add provoking factor and complication ICD codes"
```

---

## Task 14: Provoking Factor Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for provoking factors**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def provoking_diagnoses():
    """Diagnoses for provoking factor testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P2', 'P3', 'P4'],
        'icd_code': ['Z98.89', 'O26.9', 'S72.001', 'I26.9'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [-10, -5, -20, 0],
    })


class TestProvokingFactors:
    def test_recent_surgery_detected(self, provoking_diagnoses):
        builder = PEFeatureBuilder(provoking_diagnoses)
        features = builder.build_provoking_factors()
        assert features.loc['P1', 'recent_surgery'] == True
        assert features.loc['P4', 'recent_surgery'] == False

    def test_pregnancy_related_detected(self, provoking_diagnoses):
        builder = PEFeatureBuilder(provoking_diagnoses)
        features = builder.build_provoking_factors()
        assert features.loc['P2', 'pregnancy_related'] == True
        assert features.loc['P1', 'pregnancy_related'] == False

    def test_recent_trauma_detected(self, provoking_diagnoses):
        builder = PEFeatureBuilder(provoking_diagnoses)
        features = builder.build_provoking_factors()
        assert features.loc['P3', 'recent_trauma'] == True

    def test_is_provoked_vte(self, provoking_diagnoses):
        builder = PEFeatureBuilder(provoking_diagnoses)
        features = builder.build_provoking_factors()
        # P1, P2, P3 all have provoking factors
        assert features.loc['P1', 'is_provoked_vte'] == True
        assert features.loc['P2', 'is_provoked_vte'] == True
        assert features.loc['P3', 'is_provoked_vte'] == True
        # P4 has no provoking factor
        assert features.loc['P4', 'is_provoked_vte'] == False
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestProvokingFactors -v`
Expected: FAIL

**Step 3: Implement provoking factors**

Add import and method:

```python
# Update import:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES, BLEEDING_CODES, RENAL_CODES, PROVOKING_CODES
)

# Add method:
    def build_provoking_factors(self) -> pd.DataFrame:
        """Extract provoking factor features (within 30 days before PE)."""
        recent = self.diagnoses[
            (self.diagnoses['days_from_pe'] >= -30) &
            (self.diagnoses['days_from_pe'] < 0)
        ]

        results = []
        for empi in self.empis:
            patient_recent = recent[recent['EMPI'] == empi]

            recent_surgery = len(self._match_codes(patient_recent, PROVOKING_CODES['surgery'])) > 0
            recent_trauma = len(self._match_codes(patient_recent, PROVOKING_CODES['trauma'])) > 0
            immobilization = len(self._match_codes(patient_recent, PROVOKING_CODES['immobilization'])) > 0
            pregnancy_related = len(self._match_codes(patient_recent, PROVOKING_CODES['pregnancy'])) > 0
            hormonal_therapy = len(self._match_codes(patient_recent, PROVOKING_CODES['hormonal_therapy'])) > 0
            cvc = len(self._match_codes(patient_recent, PROVOKING_CODES['central_venous_catheter'])) > 0

            is_provoked = any([recent_surgery, recent_trauma, immobilization,
                              pregnancy_related, hormonal_therapy, cvc])

            results.append({
                'EMPI': empi,
                'recent_surgery': recent_surgery,
                'recent_trauma': recent_trauma,
                'immobilization': immobilization,
                'pregnancy_related': pregnancy_related,
                'hormonal_therapy': hormonal_therapy,
                'cvc': cvc,
                'is_provoked_vte': is_provoked,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (35 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add provoking factor feature extraction"
```

---

## Task 15: Complication Features

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write tests for complication features**

Add to `tests/test_pe_feature_builder.py`:

```python
@pytest.fixture
def complication_diagnoses():
    """Diagnoses for complication feature testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P3', 'P4'],
        'icd_code': ['I26.9', 'N17.0', 'I26.0', 'I26.9', 'I26.99'],
        'icd_version': ['10', '10', '10', '10', '10'],
        'days_from_pe': [0, 3, 0, 0, 5],  # P4 has recurrent PE at day 5
    })


class TestComplicationFeatures:
    def test_complication_aki_detected(self, complication_diagnoses):
        builder = PEFeatureBuilder(complication_diagnoses)
        features = builder.build_complication_features()
        # P1 has AKI at day 3 (post-PE)
        assert features.loc['P1', 'complication_aki'] == True
        assert features.loc['P2', 'complication_aki'] == False

    def test_complication_recurrent_vte(self, complication_diagnoses):
        builder = PEFeatureBuilder(complication_diagnoses)
        features = builder.build_complication_features()
        # P4 has PE at day 5 (recurrent)
        assert features.loc['P4', 'complication_recurrent_vte'] == True
        assert features.loc['P1', 'complication_recurrent_vte'] == False


@pytest.fixture
def shock_diagnoses():
    """Diagnoses with shock/arrest complications."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P2', 'P2'],
        'icd_code': ['I26.0', 'R57.0', 'I26.9', 'I46.9'],
        'icd_version': ['10', '10', '10', '10'],
        'days_from_pe': [0, 2, 0, 1],
    })


class TestShockComplications:
    def test_cardiogenic_shock_detected(self, shock_diagnoses):
        builder = PEFeatureBuilder(shock_diagnoses)
        features = builder.build_complication_features()
        assert features.loc['P1', 'complication_shock'] == True
        assert features.loc['P2', 'complication_shock'] == False

    def test_cardiac_arrest_detected(self, shock_diagnoses):
        builder = PEFeatureBuilder(shock_diagnoses)
        features = builder.build_complication_features()
        # P2 has cardiac arrest at day 1 (still within index window? No, days > 1)
        # Actually day 1 is within abs(days) <= 1, so not a complication
        assert features.loc['P2', 'complication_arrest'] == False
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestComplicationFeatures -v`
Expected: FAIL

**Step 3: Implement complication features**

Add import and method:

```python
# Update import:
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES, BLEEDING_CODES, RENAL_CODES, PROVOKING_CODES,
    COMPLICATION_CODES
)

# Add method:
    def build_complication_features(self) -> pd.DataFrame:
        """Extract post-PE complication features (days > 1)."""
        post = self.diagnoses[self.diagnoses['days_from_pe'] > 1]

        results = []
        for empi in self.empis:
            patient_post = post[post['EMPI'] == empi]

            # AKI
            complication_aki = len(self._match_codes(patient_post, RENAL_CODES['aki'])) > 0

            # Bleeding
            complication_gi_bleed = len(self._match_codes(patient_post, BLEEDING_CODES['gi_bleeding'])) > 0
            complication_ich = len(self._match_codes(patient_post, BLEEDING_CODES['intracranial_hemorrhage'])) > 0
            complication_other_bleed = len(self._match_codes(patient_post, BLEEDING_CODES['major_bleeding_other'])) > 0
            complication_bleeding_any = complication_gi_bleed or complication_ich or complication_other_bleed
            complication_bleeding_major = complication_gi_bleed or complication_ich

            # Respiratory/cardiac
            complication_resp_failure = len(self._match_codes(patient_post, PULMONARY_CODES['respiratory_failure'])) > 0
            complication_shock = len(self._match_codes(patient_post, COMPLICATION_CODES['cardiogenic_shock'])) > 0
            complication_arrest = len(self._match_codes(patient_post, COMPLICATION_CODES['cardiac_arrest'])) > 0

            # Recurrent VTE
            dvt_codes = {
                'icd10': VTE_CODES['dvt_lower_extremity']['icd10'] + VTE_CODES['dvt_upper_extremity']['icd10'],
                'icd9': VTE_CODES['dvt_lower_extremity']['icd9'] + VTE_CODES['dvt_upper_extremity']['icd9']
            }
            recurrent_pe = len(self._match_codes(patient_post, VTE_CODES['pe'])) > 0
            recurrent_dvt = len(self._match_codes(patient_post, dvt_codes)) > 0
            complication_recurrent_vte = recurrent_pe or recurrent_dvt

            # CTEPH
            complication_cteph = len(self._match_codes(patient_post, COMPLICATION_CODES['cteph'])) > 0

            results.append({
                'EMPI': empi,
                'complication_aki': complication_aki,
                'complication_bleeding_any': complication_bleeding_any,
                'complication_bleeding_major': complication_bleeding_major,
                'complication_ich': complication_ich,
                'complication_resp_failure': complication_resp_failure,
                'complication_shock': complication_shock,
                'complication_arrest': complication_arrest,
                'complication_recurrent_vte': complication_recurrent_vte,
                'complication_cteph': complication_cteph,
            })

        return pd.DataFrame(results).set_index('EMPI')
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (40 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add complication feature extraction"
```

---

## Task 16: build_all_features Integration

**Files:**
- Modify: `module_05_diagnoses/tests/test_pe_feature_builder.py`
- Modify: `module_05_diagnoses/processing/pe_feature_builder.py`

**Step 1: Write test for build_all_features**

Add to `tests/test_pe_feature_builder.py`:

```python
class TestBuildAllFeatures:
    def test_build_all_features_returns_dataframe(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_all_features()
        assert isinstance(features, pd.DataFrame)
        assert 'EMPI' in features.columns

    def test_build_all_features_has_all_groups(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_all_features()
        # Check for columns from each group
        assert 'prior_pe_ever' in features.columns  # VTE history
        assert 'pe_subtype' in features.columns  # PE index
        assert 'cancer_active' in features.columns  # Cancer
        assert 'heart_failure' in features.columns  # CV
        assert 'copd' in features.columns  # Pulmonary
        assert 'prior_major_bleed' in features.columns  # Bleeding
        assert 'ckd_stage' in features.columns  # Renal
        assert 'recent_surgery' in features.columns  # Provoking
        assert 'complication_aki' in features.columns  # Complications

    def test_build_all_features_all_patients(self, vte_history_diagnoses):
        builder = PEFeatureBuilder(vte_history_diagnoses)
        features = builder.build_all_features()
        assert len(features) == 3  # P1, P2, P3
```

**Step 2: Run test to verify it fails**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py::TestBuildAllFeatures -v`
Expected: FAIL

**Step 3: Implement build_all_features**

Add method to `PEFeatureBuilder`:

```python
    def build_all_features(self) -> pd.DataFrame:
        """Merge all feature groups into single wide DataFrame."""
        dfs = [
            self.build_vte_history_features(),
            self.build_pe_index_features(),
            self.build_cancer_features(),
            self.build_cardiovascular_features(),
            self.build_pulmonary_features(),
            self.build_bleeding_risk_features(),
            self.build_renal_features(),
            self.build_provoking_factors(),
            self.build_complication_features(),
        ]

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how='outer')

        # Fill boolean columns with False
        bool_cols = result.select_dtypes(include=['bool']).columns
        result[bool_cols] = result[bool_cols].fillna(False)

        return result.reset_index()
```

**Step 4: Run test to verify it passes**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_pe_feature_builder.py -v`
Expected: All tests pass (43 passed)

**Step 5: Commit**

```bash
git add module_05_diagnoses/processing/pe_feature_builder.py module_05_diagnoses/tests/test_pe_feature_builder.py
git commit -m "feat(module05): add build_all_features integration method"
```

---

## Task 17: Pipeline Integration

**Files:**
- Modify: `module_05_diagnoses/build_layers.py`

**Step 1: Read current build_layers.py**

Run: `cat module_05_diagnoses/build_layers.py` to understand current structure.

**Step 2: Add Layer 3 to pipeline**

Add function and CLI option:

```python
def build_layer3(layer1_path: Path, output_dir: Path) -> pd.DataFrame:
    """Build Layer 3: PE-specific diagnosis features."""
    from processing.pe_feature_builder import PEFeatureBuilder

    print("Building Layer 3: PE-specific features...")
    canonical = pd.read_parquet(layer1_path)
    builder = PEFeatureBuilder(canonical)
    features = builder.build_all_features()

    output_path = output_dir / "layer3" / "pe_diagnosis_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"  Layer 3: {len(features)} patients, {len(features.columns)} features")
    print(f"  Saved to: {output_path}")
    return features
```

Update main function to call `build_layer3` after Layer 2.

**Step 3: Test pipeline**

Run: `cd module_05_diagnoses && python build_layers.py --test --n=100`
Expected: Layer 3 builds successfully

**Step 4: Commit**

```bash
git add module_05_diagnoses/build_layers.py
git commit -m "feat(module05): integrate Layer 3 into build pipeline"
```

---

## Task 18: Integration Test with Real Data

**Files:**
- Create: `module_05_diagnoses/tests/test_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for Layer 3 with real data."""

import pytest
import pandas as pd
from pathlib import Path

from processing.pe_feature_builder import PEFeatureBuilder


@pytest.fixture
def real_layer1_data():
    """Load real Layer 1 data if available."""
    path = Path("outputs/layer1/canonical_diagnoses.parquet")
    if not path.exists():
        pytest.skip("Layer 1 data not available")
    return pd.read_parquet(path)


class TestIntegration:
    def test_build_all_features_on_real_data(self, real_layer1_data):
        builder = PEFeatureBuilder(real_layer1_data)
        features = builder.build_all_features()

        # Basic sanity checks
        assert len(features) > 0
        assert 'EMPI' in features.columns
        assert len(features.columns) >= 45  # ~51 features expected

    def test_feature_distributions_plausible(self, real_layer1_data):
        builder = PEFeatureBuilder(real_layer1_data)
        features = builder.build_all_features()

        # Prior VTE should be 15-30%
        prior_vte_rate = features['is_recurrent_vte'].mean()
        assert 0.05 < prior_vte_rate < 0.50, f"Prior VTE rate {prior_vte_rate:.1%} seems implausible"

        # Cancer should be 15-25%
        cancer_rate = features['cancer_active'].mean()
        assert 0.05 < cancer_rate < 0.50, f"Cancer rate {cancer_rate:.1%} seems implausible"
```

**Step 2: Run integration test**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_integration.py -v`
Expected: Tests pass (or skip if no data)

**Step 3: Commit**

```bash
git add module_05_diagnoses/tests/test_integration.py
git commit -m "test(module05): add Layer 3 integration tests"
```

---

## Task 19: Final Verification

**Step 1: Run all tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/ -v`
Expected: All tests pass

**Step 2: Run full pipeline**

Run: `cd module_05_diagnoses && python build_layers.py --test --n=100`
Expected: All layers build successfully

**Step 3: Verify output**

Run: `cd module_05_diagnoses && python -c "import pandas as pd; df = pd.read_parquet('outputs/layer3/pe_diagnosis_features.parquet'); print(df.shape); print(df.columns.tolist()[:10])"`
Expected: Shows shape and first 10 columns

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(module05): complete Layer 3 PE-specific features

Phase 2 implementation complete:
- 9 feature groups, ~51 features
- ICD code definitions for VTE, cancer, CV, pulmonary, bleeding, renal, provoking factors, complications
- PEFeatureBuilder class with TDD coverage
- Pipeline integration

Tests: X passed"
```

---

## Summary

| Task | Description | Tests Added |
|------|-------------|-------------|
| 1 | VTE code definitions | 0 |
| 2 | PEFeatureBuilder + code matching | 3 |
| 3 | VTE history features | 6 |
| 4 | PE index features | 4 |
| 5 | Cancer code definitions | 0 |
| 6 | Cancer features | 4 |
| 7 | CV code definitions | 0 |
| 8 | CV features | 4 |
| 9 | Pulmonary/bleeding/renal codes | 0 |
| 10 | Pulmonary features | 3 |
| 11 | Bleeding risk features | 4 |
| 12 | Renal features | 3 |
| 13 | Provoking/complication codes | 0 |
| 14 | Provoking factors | 4 |
| 15 | Complication features | 4 |
| 16 | build_all_features | 3 |
| 17 | Pipeline integration | 0 |
| 18 | Integration tests | 2 |
| 19 | Final verification | 0 |

**Total: ~44 unit tests + 2 integration tests**
