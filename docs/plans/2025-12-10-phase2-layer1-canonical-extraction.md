# Phase 2: Layer 1 Canonical Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract medication records from Med.txt, join with patient cohort from Module 1, parse dose/unit/route, and output canonical records as parquet.

**Architecture:** Stream Med.txt in 1M-row chunks, filter to PE cohort patients (8,713), compute hours_from_t0 using patient_timelines.pkl, apply regex dose parsing, write bronze parquet partitioned by EMPI.

**Tech Stack:** Python 3.12, pandas, polars (for fast parsing), pickle, pyarrow, regex, pytest

---

## Task 1: Create Test Infrastructure

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/__init__.py`
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_dose_parser.py`
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/__init__.py`

**Step 1: Create test package init**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/__init__.py
"""Module 4 Tests"""
```

**Step 2: Create extractors package init**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/__init__.py
"""Module 4 Extractors"""
```

**Step 3: Create dose parser test file with first test**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_dose_parser.py
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
```

**Step 4: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_dose_parser.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'extractors.dose_parser'"

**Step 5: Commit test infrastructure**

```bash
git add module_04_medications/tests/__init__.py module_04_medications/tests/test_dose_parser.py module_04_medications/extractors/__init__.py
git commit -m "test(module4): add dose parser test infrastructure

Add failing tests for dose extraction from RPDR medication strings.
Tests cover: mg, mcg, units, concentration format, no-dose cases.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement Dose Parser

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/dose_parser.py`

**Step 1: Create dose_parser.py with extract_dose function**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/dose_parser.py
"""
Dose Parser for RPDR Medication Strings
=======================================

Extracts dose value, unit, route, and frequency from free-text medication strings.
Uses regex patterns defined in config/dose_patterns.yaml.
"""

import re
from typing import Dict, Optional, Any
from pathlib import Path
import yaml


# Load patterns from YAML
CONFIG_DIR = Path(__file__).parent.parent / "config"
DOSE_PATTERNS_FILE = CONFIG_DIR / "dose_patterns.yaml"

_patterns_cache: Optional[Dict] = None


def _load_patterns() -> Dict:
    """Load and cache dose patterns from YAML."""
    global _patterns_cache
    if _patterns_cache is None:
        with open(DOSE_PATTERNS_FILE, 'r') as f:
            _patterns_cache = yaml.safe_load(f)
    return _patterns_cache


# Unit normalization mapping
UNIT_ALIASES = {
    'mg': 'mg',
    'milligram': 'mg',
    'milligrams': 'mg',
    'mgs': 'mg',
    'mcg': 'mcg',
    'ug': 'mcg',
    'microgram': 'mcg',
    'micrograms': 'mcg',
    'μg': 'mcg',
    'g': 'g',
    'gm': 'g',
    'gram': 'g',
    'grams': 'g',
    'gms': 'g',
    'ml': 'ml',
    'milliliter': 'ml',
    'milliliters': 'ml',
    'mls': 'ml',
    'cc': 'ml',
    'l': 'l',
    'liter': 'l',
    'liters': 'l',
    'unit': 'units',
    'units': 'units',
    'u': 'units',
    'iu': 'units',
    'meq': 'meq',
    'milliequivalent': 'meq',
    'milliequivalents': 'meq',
    'mmol': 'mmol',
    'millimole': 'mmol',
    'millimoles': 'mmol',
}


def normalize_unit(unit: str) -> str:
    """Normalize unit string to canonical form."""
    if unit is None:
        return None
    return UNIT_ALIASES.get(unit.lower().strip(), unit.lower().strip())


def extract_dose(medication_string: str) -> Dict[str, Any]:
    """
    Extract dose information from a medication string.

    Args:
        medication_string: Raw RPDR medication text

    Returns:
        Dictionary with keys:
            - dose_value: float or None
            - dose_unit: str or None
            - parse_method: str ('regex' or 'failed')
            - parse_confidence: float (0-1)
    """
    if not medication_string or not isinstance(medication_string, str):
        return {
            'dose_value': None,
            'dose_unit': None,
            'parse_method': 'failed',
            'parse_confidence': 0.0,
        }

    text = medication_string.lower().strip()

    # Pattern 1: Standard dose with unit (e.g., "325mg", "100 mcg", "5000 units")
    # Most common pattern - try first
    standard_pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|gm|ml|unit|units|u|iu|meq|mmol)\b'
    match = re.search(standard_pattern, text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = normalize_unit(match.group(2))
        return {
            'dose_value': value,
            'dose_unit': unit,
            'parse_method': 'regex',
            'parse_confidence': 0.9,
        }

    # Pattern 2: Concentration format (e.g., "2 mg/ml", "5000 unit/ml")
    conc_pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|unit|units|u)\s*/\s*(?:\d+\s*)?(ml|l)\b'
    match = re.search(conc_pattern, text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = normalize_unit(match.group(2))
        return {
            'dose_value': value,
            'dose_unit': unit,
            'parse_method': 'regex',
            'parse_confidence': 0.85,
        }

    # Pattern 3: Percentage (e.g., "0.9%", "5%")
    pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
    match = re.search(pct_pattern, text)
    if match:
        value = float(match.group(1))
        return {
            'dose_value': value,
            'dose_unit': 'percent',
            'parse_method': 'regex',
            'parse_confidence': 0.8,
        }

    # No dose found
    return {
        'dose_value': None,
        'dose_unit': None,
        'parse_method': 'failed',
        'parse_confidence': 0.0,
    }


def extract_route(medication_string: str) -> Optional[str]:
    """
    Extract administration route from medication string.

    Returns: One of 'IV', 'PO', 'SC', 'IM', 'topical', 'inhaled', 'rectal', 'ophthalmic', or None
    """
    if not medication_string:
        return None

    text = medication_string.lower()

    # IV patterns
    if any(p in text for p in ['intravenous', ' iv ', 'ivpb', 'iv push', 'infusion']):
        return 'IV'
    if re.search(r'\biv\b', text):
        return 'IV'

    # SC patterns
    if any(p in text for p in ['subcutaneous', 'subcut', 'subq']):
        return 'SC'
    if re.search(r'\b(sc|sq)\b', text):
        return 'SC'

    # IM patterns
    if 'intramuscular' in text:
        return 'IM'
    if re.search(r'\bim\b', text):
        return 'IM'

    # PO patterns (check after IV/IM to avoid false positives)
    if any(p in text for p in ['oral', 'by mouth', 'tablet', 'capsule', 'tab ', 'cap ']):
        return 'PO'
    if re.search(r'\bpo\b', text):
        return 'PO'

    # Topical
    if any(p in text for p in ['topical', 'cream', 'ointment', 'gel', 'lotion', 'patch']):
        return 'topical'

    # Inhaled
    if any(p in text for p in ['inhaler', 'nebulizer', 'inhalation', 'metered dose']):
        return 'inhaled'

    # Rectal
    if any(p in text for p in ['rectal', 'suppository', 'enema']):
        return 'rectal'

    # Ophthalmic
    if any(p in text for p in ['ophthalmic', 'eye drop', 'eye solution']):
        return 'ophthalmic'

    return None


def extract_frequency(medication_string: str) -> Optional[str]:
    """
    Extract dosing frequency from medication string.

    Returns: One of 'QD', 'BID', 'TID', 'QID', 'Q6H', 'Q8H', 'Q12H', 'PRN', 'ONCE', or None
    """
    if not medication_string:
        return None

    text = medication_string.lower()

    # Check explicit frequency markers
    if re.search(r'\bprn\b|as needed', text):
        return 'PRN'
    if re.search(r'\bonce\b|single dose|one time', text):
        return 'ONCE'
    if re.search(r'\bq\s*6\s*h|every\s*6\s*hour', text):
        return 'Q6H'
    if re.search(r'\bq\s*8\s*h|every\s*8\s*hour', text):
        return 'Q8H'
    if re.search(r'\bq\s*12\s*h|every\s*12\s*hour', text):
        return 'Q12H'
    if re.search(r'\bqid\b|four times', text):
        return 'QID'
    if re.search(r'\btid\b|three times', text):
        return 'TID'
    if re.search(r'\bbid\b|twice|two times', text):
        return 'BID'
    if re.search(r'\bqd\b|daily|once daily', text):
        return 'QD'

    return None


def extract_drug_name(medication_string: str) -> str:
    """
    Extract clean drug name from medication string.

    Strategy: Take text before first numeric value, clean up.
    """
    if not medication_string:
        return ""

    text = medication_string.strip()

    # Find first number - drug name is typically before it
    match = re.search(r'\d', text)
    if match:
        name_part = text[:match.start()].strip()
    else:
        # No number found, use full string up to common suffixes
        name_part = text

    # Clean up common suffixes that aren't part of drug name
    cleanup_patterns = [
        r'\s+(tablet|capsule|solution|injection|cream|ointment|suspension)s?$',
        r'\s+(hcl|sodium|sulfate|citrate|tartrate|bitartrate|pf)$',
        r'\s*[,/].*$',  # Remove anything after comma or slash
    ]

    for pattern in cleanup_patterns:
        name_part = re.sub(pattern, '', name_part, flags=re.IGNORECASE)

    # Final cleanup
    name_part = name_part.strip().lower()

    # Remove trailing punctuation
    name_part = re.sub(r'[,;:\-]+$', '', name_part).strip()

    return name_part


def parse_medication_string(medication_string: str) -> Dict[str, Any]:
    """
    Full parsing of medication string.

    Returns dictionary with all extracted fields:
        - parsed_name: str
        - parsed_dose_value: float or None
        - parsed_dose_unit: str or None
        - parsed_route: str or None
        - parsed_frequency: str or None
        - parse_method: str
        - parse_confidence: float
    """
    dose_info = extract_dose(medication_string)

    return {
        'parsed_name': extract_drug_name(medication_string),
        'parsed_dose_value': dose_info['dose_value'],
        'parsed_dose_unit': dose_info['dose_unit'],
        'parsed_route': extract_route(medication_string),
        'parsed_frequency': extract_frequency(medication_string),
        'parse_method': dose_info['parse_method'],
        'parse_confidence': dose_info['parse_confidence'],
    }
```

**Step 2: Run tests to verify they pass**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_dose_parser.py -v
```

**Expected:** All 5 tests PASS

**Step 3: Commit dose parser**

```bash
git add module_04_medications/extractors/dose_parser.py
git commit -m "feat(module4): implement dose parser for RPDR medication strings

Add regex-based extraction for:
- Dose value and unit (mg, mcg, units, etc.)
- Administration route (IV, PO, SC, IM, etc.)
- Dosing frequency (QD, BID, PRN, etc.)
- Drug name extraction

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Add More Dose Parser Tests

**Files:**
- Modify: `/home/moin/TDA_11_25/module_04_medications/tests/test_dose_parser.py`

**Step 1: Add tests for route and frequency extraction**

Append to test_dose_parser.py:

```python
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
```

**Step 2: Run all tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_dose_parser.py -v
```

**Expected:** All tests PASS (should be ~18 tests)

**Step 3: Commit additional tests**

```bash
git add module_04_medications/tests/test_dose_parser.py
git commit -m "test(module4): add comprehensive dose parser tests

Add tests for:
- Route extraction (IV, PO, SC)
- Frequency extraction (QD, BID, PRN)
- Drug name extraction
- Full parsing integration

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Create Canonical Extractor Tests

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_canonical_extractor.py`

**Step 1: Create test file for canonical extractor**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_canonical_extractor.py
"""Tests for canonical medication record extraction."""

import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMedFileLoader:
    """Test Med.txt loading functionality."""

    def test_load_med_chunk(self):
        """Load a chunk of medication data."""
        from extractors.canonical_extractor import load_med_chunk

        # Load first 100 rows
        df = load_med_chunk(n_rows=100)

        assert len(df) == 100
        assert 'EMPI' in df.columns
        assert 'Medication' in df.columns
        assert 'Medication_Date' in df.columns

    def test_column_names_correct(self):
        """Verify expected columns exist."""
        from extractors.canonical_extractor import load_med_chunk

        df = load_med_chunk(n_rows=10)

        expected_cols = [
            'EMPI', 'Medication_Date', 'Medication', 'Code_Type',
            'Code', 'Quantity', 'Inpatient_Outpatient', 'Encounter_number'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestCohortFiltering:
    """Test cohort filtering functionality."""

    def test_filter_to_cohort(self):
        """Filter medications to PE cohort patients only."""
        from extractors.canonical_extractor import filter_to_cohort

        # Create test data
        med_df = pd.DataFrame({
            'EMPI': ['100', '200', '300', '100'],
            'Medication': ['Drug A', 'Drug B', 'Drug C', 'Drug D'],
        })

        cohort_empis = {'100', '300'}

        filtered = filter_to_cohort(med_df, cohort_empis)

        assert len(filtered) == 3
        assert set(filtered['EMPI'].unique()) == {'100', '300'}


class TestTimeAlignment:
    """Test time alignment to Time Zero."""

    def test_compute_hours_from_t0(self):
        """Compute hours relative to Time Zero."""
        from extractors.canonical_extractor import compute_hours_from_t0

        # Test data
        med_df = pd.DataFrame({
            'EMPI': ['100', '100', '100'],
            'Medication_Date': pd.to_datetime(['2023-07-27', '2023-07-28', '2023-07-26']),
        })

        # Time Zero for patient 100 is 2023-07-27 12:00:00
        time_zero_map = {
            '100': pd.Timestamp('2023-07-27 12:00:00'),
        }

        result = compute_hours_from_t0(med_df, time_zero_map)

        # First row: same day, assume noon - should be ~0
        # Second row: next day - should be ~24
        # Third row: day before - should be ~-24
        assert 'hours_from_t0' in result.columns

        # Check approximate values (date only, so we assume midnight)
        hours = result['hours_from_t0'].tolist()
        assert hours[0] == pytest.approx(-12, abs=1)  # 2023-07-27 00:00 vs 12:00 = -12h
        assert hours[1] == pytest.approx(12, abs=1)   # 2023-07-28 00:00 vs 2023-07-27 12:00 = +12h
        assert hours[2] == pytest.approx(-36, abs=1)  # 2023-07-26 00:00 vs 2023-07-27 12:00 = -36h


class TestWindowFiltering:
    """Test study window filtering."""

    def test_filter_study_window(self):
        """Filter to study window (-30 to +30 days)."""
        from extractors.canonical_extractor import filter_study_window

        df = pd.DataFrame({
            'hours_from_t0': [-800, -100, 0, 100, 800],
            'Medication': ['A', 'B', 'C', 'D', 'E'],
        })

        # Default window: -720 to +720 hours (-30 to +30 days)
        filtered = filter_study_window(df)

        assert len(filtered) == 3
        assert list(filtered['Medication']) == ['B', 'C', 'D']
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_canonical_extractor.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'extractors.canonical_extractor'"

**Step 3: Commit failing tests**

```bash
git add module_04_medications/tests/test_canonical_extractor.py
git commit -m "test(module4): add canonical extractor tests

Add failing tests for:
- Med.txt chunk loading
- Cohort filtering
- Time alignment to Time Zero
- Study window filtering

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Implement Canonical Extractor Core Functions

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/canonical_extractor.py`

**Step 1: Create canonical_extractor.py**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/canonical_extractor.py
"""
Canonical Medication Record Extractor
=====================================

Extracts medication records from RPDR Med.txt, joins with patient cohort,
computes temporal alignment, and outputs bronze parquet.

Layer 1 of the medication encoding pipeline.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Set, Optional, Iterator
from datetime import datetime
import sys

# Add parent to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    MED_FILE,
    PATIENT_TIMELINES_PKL,
    BRONZE_DIR,
    TEMPORAL_CONFIG,
    LAYER_CONFIG,
)
from extractors.dose_parser import parse_medication_string


# =============================================================================
# MED.TXT LOADING
# =============================================================================

def load_med_chunk(
    n_rows: Optional[int] = None,
    skip_rows: int = 0,
    filepath: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a chunk of Med.txt.

    Args:
        n_rows: Number of rows to load (None for all)
        skip_rows: Number of rows to skip (for chunked loading)
        filepath: Override default Med.txt path

    Returns:
        DataFrame with medication records
    """
    filepath = filepath or MED_FILE

    df = pd.read_csv(
        filepath,
        sep='|',
        nrows=n_rows,
        skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None,
        dtype={
            'EMPI': str,
            'EPIC_PMRN': str,
            'MRN_Type': str,
            'MRN': str,
            'Medication': str,
            'Code_Type': str,
            'Code': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
            'Additional_Info': str,
        },
        parse_dates=['Medication_Date'],
        low_memory=False,
    )

    return df


def iter_med_chunks(
    chunk_size: int = 1_000_000,
    filepath: Optional[Path] = None
) -> Iterator[pd.DataFrame]:
    """
    Iterate over Med.txt in chunks.

    Args:
        chunk_size: Rows per chunk
        filepath: Override default Med.txt path

    Yields:
        DataFrame chunks
    """
    filepath = filepath or MED_FILE

    reader = pd.read_csv(
        filepath,
        sep='|',
        chunksize=chunk_size,
        dtype={
            'EMPI': str,
            'EPIC_PMRN': str,
            'MRN_Type': str,
            'MRN': str,
            'Medication': str,
            'Code_Type': str,
            'Code': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
            'Additional_Info': str,
        },
        parse_dates=['Medication_Date'],
        low_memory=False,
    )

    for chunk in reader:
        yield chunk


# =============================================================================
# COHORT INTEGRATION
# =============================================================================

def load_patient_timelines() -> Dict:
    """
    Load patient timelines from Module 1.

    Returns:
        Dictionary mapping EMPI -> PatientTimeline object
    """
    # Need to import PatientTimeline class for unpickling
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "module_1_core_infrastructure"))
    from module_01_core_infrastructure import PatientTimeline

    with open(PATIENT_TIMELINES_PKL, 'rb') as f:
        timelines = pickle.load(f)

    return timelines


def get_cohort_empis(timelines: Dict) -> Set[str]:
    """
    Get set of EMPI values for PE cohort.

    Args:
        timelines: Patient timelines dictionary

    Returns:
        Set of EMPI strings
    """
    return set(timelines.keys())


def get_time_zero_map(timelines: Dict) -> Dict[str, pd.Timestamp]:
    """
    Build mapping of EMPI -> Time Zero.

    Args:
        timelines: Patient timelines dictionary

    Returns:
        Dictionary mapping EMPI -> Time Zero timestamp
    """
    return {
        empi: pd.Timestamp(timeline.time_zero)
        for empi, timeline in timelines.items()
    }


def filter_to_cohort(df: pd.DataFrame, cohort_empis: Set[str]) -> pd.DataFrame:
    """
    Filter medication DataFrame to cohort patients only.

    Args:
        df: Medication DataFrame
        cohort_empis: Set of EMPI values in cohort

    Returns:
        Filtered DataFrame
    """
    return df[df['EMPI'].isin(cohort_empis)].copy()


# =============================================================================
# TIME ALIGNMENT
# =============================================================================

def compute_hours_from_t0(
    df: pd.DataFrame,
    time_zero_map: Dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Compute hours relative to Time Zero for each medication.

    Args:
        df: Medication DataFrame with EMPI and Medication_Date
        time_zero_map: Dictionary mapping EMPI -> Time Zero

    Returns:
        DataFrame with hours_from_t0 column added
    """
    df = df.copy()

    # Map EMPI to Time Zero
    df['time_zero'] = df['EMPI'].map(time_zero_map)

    # Medication_Date is date only (no time), assume midnight
    med_datetime = pd.to_datetime(df['Medication_Date'])

    # Compute hours difference
    df['hours_from_t0'] = (med_datetime - df['time_zero']).dt.total_seconds() / 3600

    # Drop temporary column
    df = df.drop(columns=['time_zero'])

    return df


def filter_study_window(
    df: pd.DataFrame,
    window_start_hours: int = None,
    window_end_hours: int = None
) -> pd.DataFrame:
    """
    Filter to study window relative to Time Zero.

    Args:
        df: DataFrame with hours_from_t0 column
        window_start_hours: Start of window (default from config)
        window_end_hours: End of window (default from config)

    Returns:
        Filtered DataFrame
    """
    if window_start_hours is None:
        window_start_hours = TEMPORAL_CONFIG.study_window_start
    if window_end_hours is None:
        window_end_hours = TEMPORAL_CONFIG.study_window_end

    mask = (
        (df['hours_from_t0'] >= window_start_hours) &
        (df['hours_from_t0'] <= window_end_hours)
    )

    return df[mask].copy()


# =============================================================================
# PARSING & TRANSFORMATION
# =============================================================================

def parse_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dose parsing to all medication strings.

    Args:
        df: DataFrame with Medication column

    Returns:
        DataFrame with parsed columns added
    """
    df = df.copy()

    # Parse each medication string
    parsed = df['Medication'].apply(parse_medication_string)
    parsed_df = pd.DataFrame(parsed.tolist())

    # Add parsed columns
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    return df


def transform_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw medication data to canonical schema.

    Args:
        df: Raw medication DataFrame with parsed columns

    Returns:
        DataFrame with canonical schema
    """
    canonical = pd.DataFrame({
        'empi': df['EMPI'],
        'encounter_id': df['Encounter_number'],
        'medication_date': df['Medication_Date'],
        'hours_from_t0': df['hours_from_t0'],
        'original_string': df['Medication'],
        'code_type': df['Code_Type'],
        'code': df['Code'],
        'quantity': pd.to_numeric(df['Quantity'], errors='coerce'),
        'inpatient': df['Inpatient_Outpatient'].str.lower() == 'inpatient',
        'provider': df['Provider'],
        'clinic': df['Clinic'],
        'hospital': df['Hospital'],
        # Parsed columns
        'parsed_name': df['parsed_name'],
        'parsed_dose_value': df['parsed_dose_value'],
        'parsed_dose_unit': df['parsed_dose_unit'],
        'parsed_route': df['parsed_route'],
        'parsed_frequency': df['parsed_frequency'],
        'parse_method': df['parse_method'],
        'parse_confidence': df['parse_confidence'],
    })

    return canonical


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_canonical_records(
    test_mode: bool = False,
    test_n_rows: int = 10000,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main extraction pipeline: Med.txt -> Bronze parquet.

    Args:
        test_mode: If True, only process test_n_rows
        test_n_rows: Number of rows for test mode
        output_path: Override output path

    Returns:
        Canonical records DataFrame
    """
    print("=" * 60)
    print("Layer 1: Canonical Medication Extraction")
    print("=" * 60)

    # Load patient timelines
    print("\n1. Loading patient timelines...")
    timelines = load_patient_timelines()
    cohort_empis = get_cohort_empis(timelines)
    time_zero_map = get_time_zero_map(timelines)
    print(f"   Cohort size: {len(cohort_empis)} patients")

    # Process medications
    if test_mode:
        print(f"\n2. Loading Med.txt (test mode: {test_n_rows} rows)...")
        df = load_med_chunk(n_rows=test_n_rows)
        chunks = [df]
    else:
        print("\n2. Loading Med.txt in chunks...")
        chunks = iter_med_chunks(chunk_size=LAYER_CONFIG.chunk_size)

    all_records = []
    total_raw = 0
    total_cohort = 0
    total_window = 0

    for i, chunk in enumerate(chunks):
        total_raw += len(chunk)

        # Filter to cohort
        chunk = filter_to_cohort(chunk, cohort_empis)
        total_cohort += len(chunk)

        if len(chunk) == 0:
            continue

        # Compute time alignment
        chunk = compute_hours_from_t0(chunk, time_zero_map)

        # Filter to study window
        chunk = filter_study_window(chunk)
        total_window += len(chunk)

        if len(chunk) == 0:
            continue

        # Parse medications
        chunk = parse_medications(chunk)

        # Transform to canonical schema
        canonical = transform_to_canonical(chunk)

        all_records.append(canonical)

        if not test_mode:
            print(f"   Chunk {i+1}: {len(canonical):,} records")

    # Combine all chunks
    print("\n3. Combining records...")
    if all_records:
        result = pd.concat(all_records, ignore_index=True)
    else:
        result = pd.DataFrame()

    # Summary statistics
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"   Total raw records: {total_raw:,}")
    print(f"   After cohort filter: {total_cohort:,}")
    print(f"   After window filter: {total_window:,}")
    print(f"   Final records: {len(result):,}")

    if len(result) > 0:
        # Parsing stats
        parsed_count = (result['parse_method'] == 'regex').sum()
        parse_rate = parsed_count / len(result) * 100
        print(f"\n   Dose parsing success: {parse_rate:.1f}%")

        # Patient coverage
        patients_with_meds = result['empi'].nunique()
        print(f"   Patients with medications: {patients_with_meds}")

    # Save output
    if output_path is None:
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        filename = "canonical_records_test.parquet" if test_mode else "canonical_records.parquet"
        output_path = BRONZE_DIR / filename

    print(f"\n4. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Layer 1 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract canonical medication records")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--n', type=int, default=10000, help='Rows for test mode')
    args = parser.parse_args()

    extract_canonical_records(test_mode=args.test, test_n_rows=args.n)
```

**Step 2: Run tests to verify they pass**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_canonical_extractor.py -v
```

**Expected:** All tests PASS

**Step 3: Commit canonical extractor**

```bash
git add module_04_medications/extractors/canonical_extractor.py
git commit -m "feat(module4): implement canonical medication extractor

Add Layer 1 extraction pipeline:
- Stream Med.txt in 1M-row chunks
- Filter to PE cohort (8,713 patients)
- Compute hours_from_t0 from patient_timelines.pkl
- Apply regex dose parsing
- Output bronze parquet with canonical schema

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Run Test Mode Extraction

**Files:**
- Run: Canonical extractor in test mode

**Step 1: Run extraction in test mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/extractors/canonical_extractor.py --test --n=50000
```

**Expected output:**
```
============================================================
Layer 1: Canonical Medication Extraction
============================================================

1. Loading patient timelines...
   Cohort size: 8713 patients

2. Loading Med.txt (test mode: 50000 rows)...

3. Combining records...

============================================================
Extraction Summary
============================================================
   Total raw records: 50,000
   After cohort filter: ~X,XXX
   After window filter: ~X,XXX
   Final records: ~X,XXX

   Dose parsing success: ~XX.X%
   Patients with medications: ~XXX

4. Saving to: .../bronze/canonical_records_test.parquet
   File size: X.X MB

============================================================
Layer 1 Complete!
============================================================
```

**Step 2: Verify output file exists**

```bash
ls -lh /home/moin/TDA_11_25/module_04_medications/data/bronze/
```

**Expected:** `canonical_records_test.parquet` exists

**Step 3: Inspect output**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd
df = pd.read_parquet('module_04_medications/data/bronze/canonical_records_test.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'\\nSample medications:')
print(df[['original_string', 'parsed_name', 'parsed_dose_value', 'parsed_dose_unit']].head(10))
print(f'\\nParse method distribution:')
print(df['parse_method'].value_counts())
"
```

**Step 4: Commit test output (small file only)**

```bash
git add module_04_medications/data/bronze/.gitkeep 2>/dev/null || echo ".gitkeep not needed"
git status
```

Note: Do not commit the parquet file itself (too large).

---

## Task 7: Run Full Extraction

**Files:**
- Run: Canonical extractor in full mode

**Step 1: Run full extraction (this will take several minutes)**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/extractors/canonical_extractor.py 2>&1 | tee module_04_medications/extraction.log
```

**Expected runtime:** 5-15 minutes depending on disk speed

**Expected output:**
```
============================================================
Layer 1: Canonical Medication Extraction
============================================================

1. Loading patient timelines...
   Cohort size: 8713 patients

2. Loading Med.txt in chunks...
   Chunk 1: X,XXX,XXX records
   Chunk 2: X,XXX,XXX records
   ...

3. Combining records...

============================================================
Extraction Summary
============================================================
   Total raw records: 18,589,389
   After cohort filter: ~X,XXX,XXX
   After window filter: ~X,XXX,XXX
   Final records: ~X,XXX,XXX

   Dose parsing success: ~XX.X%
   Patients with medications: ~8,XXX

4. Saving to: .../bronze/canonical_records.parquet
   File size: XXX.X MB

============================================================
Layer 1 Complete!
============================================================
```

**Step 2: Verify output**

```bash
ls -lh /home/moin/TDA_11_25/module_04_medications/data/bronze/canonical_records.parquet
```

**Step 3: Validate extraction quality**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd
df = pd.read_parquet('module_04_medications/data/bronze/canonical_records.parquet')
print(f'Total records: {len(df):,}')
print(f'Unique patients: {df[\"empi\"].nunique():,}')
print(f'\\nParsing success rate: {(df[\"parse_method\"]==\"regex\").mean()*100:.1f}%')
print(f'\\nDose unit distribution:')
print(df['parsed_dose_unit'].value_counts().head(10))
print(f'\\nRoute distribution:')
print(df['parsed_route'].value_counts())
"
```

**Step 4: Commit extraction log**

```bash
git add module_04_medications/extraction.log
git commit -m "chore(module4): add Layer 1 extraction log

Full extraction completed:
- Total raw records: 18.6M
- Cohort records: [from log]
- Parsing success: [from log]%

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Add Vocabulary Extraction

**Files:**
- Modify: `/home/moin/TDA_11_25/module_04_medications/extractors/canonical_extractor.py`
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_vocabulary.py`

**Step 1: Add test for vocabulary extraction**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_vocabulary.py
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
        })

        vocab = extract_vocabulary(df)

        assert 'count' in vocab.columns
        aspirin_row = vocab[vocab['original_string'] == 'Aspirin 325mg']
        assert aspirin_row['count'].values[0] == 2
```

**Step 2: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_vocabulary.py -v
```

**Expected:** FAIL with "cannot import name 'extract_vocabulary'"

**Step 3: Add extract_vocabulary function to canonical_extractor.py**

Add this function before the main extraction pipeline section:

```python
def extract_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique medication vocabulary from canonical records.

    Args:
        df: Canonical records DataFrame

    Returns:
        DataFrame with unique medication strings and counts
    """
    # Group by original string and aggregate
    vocab = df.groupby('original_string').agg({
        'parsed_name': 'first',
        'parsed_dose_value': 'first',
        'parsed_dose_unit': 'first',
        'parsed_route': 'first',
        'parse_method': 'first',
        'empi': 'count',  # Count occurrences
    }).reset_index()

    vocab = vocab.rename(columns={'empi': 'count'})

    # Sort by count descending
    vocab = vocab.sort_values('count', ascending=False)

    return vocab
```

**Step 4: Run test to verify it passes**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_vocabulary.py -v
```

**Expected:** All tests PASS

**Step 5: Commit vocabulary extraction**

```bash
git add module_04_medications/extractors/canonical_extractor.py module_04_medications/tests/test_vocabulary.py
git commit -m "feat(module4): add medication vocabulary extraction

Extract unique medication strings with occurrence counts.
Vocabulary will be used for RxNorm mapping in Phase 3.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Generate Vocabulary File

**Files:**
- Run: Vocabulary extraction script

**Step 1: Create and run vocabulary extraction**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd
import sys
sys.path.insert(0, 'module_04_medications')
from extractors.canonical_extractor import extract_vocabulary

# Load canonical records
df = pd.read_parquet('module_04_medications/data/bronze/canonical_records.parquet')
print(f'Loaded {len(df):,} records')

# Extract vocabulary
vocab = extract_vocabulary(df)
print(f'Unique medication strings: {len(vocab):,}')

# Save vocabulary
vocab.to_parquet('module_04_medications/data/bronze/medication_vocabulary.parquet', index=False)
print(f'Saved to medication_vocabulary.parquet')

# Preview
print(f'\\nTop 20 medications by frequency:')
print(vocab.head(20)[['original_string', 'parsed_name', 'count']])
"
```

**Expected output:**
```
Loaded X,XXX,XXX records
Unique medication strings: ~15,000
Saved to medication_vocabulary.parquet

Top 20 medications by frequency:
   original_string                parsed_name       count
0  Sodium chloride 0.9% 1000ml    sodium chloride   XXXXX
1  Aspirin 81mg tablet            aspirin           XXXXX
...
```

**Step 2: Verify vocabulary file**

```bash
ls -lh /home/moin/TDA_11_25/module_04_medications/data/bronze/medication_vocabulary.parquet
```

---

## Task 10: Final Validation & Documentation

**Files:**
- Run: Validation checks
- Update: Module README

**Step 1: Run all tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/ -v
```

**Expected:** All tests PASS

**Step 2: Validate extraction quality**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd

# Load data
df = pd.read_parquet('module_04_medications/data/bronze/canonical_records.parquet')
vocab = pd.read_parquet('module_04_medications/data/bronze/medication_vocabulary.parquet')

print('='*60)
print('Layer 1 Validation Report')
print('='*60)

# Coverage
print(f'\\nCoverage:')
print(f'  Total records: {len(df):,}')
print(f'  Unique patients: {df[\"empi\"].nunique():,}')
print(f'  Unique medications: {len(vocab):,}')

# Parsing success
parse_rate = (df['parse_method'] == 'regex').mean() * 100
print(f'\\nParsing:')
print(f'  Dose parsing success: {parse_rate:.1f}%')
print(f'  Target: >=80%')
print(f'  Status: {\"PASS\" if parse_rate >= 80 else \"NEEDS IMPROVEMENT\"}')

# Time alignment
print(f'\\nTime alignment:')
print(f'  Hours from T0 range: {df[\"hours_from_t0\"].min():.0f} to {df[\"hours_from_t0\"].max():.0f}')

# Key medications present
print(f'\\nKey PE medications:')
for med in ['heparin', 'enoxaparin', 'warfarin', 'apixaban', 'rivaroxaban']:
    count = vocab[vocab['parsed_name'].str.contains(med, na=False)]['count'].sum()
    print(f'  {med}: {count:,} records')

print('\\n' + '='*60)
"
```

**Step 3: Update module README**

Add to `/home/moin/TDA_11_25/module_04_medications/README.md` (after existing content):

```markdown
## Layer 1: Canonical Extraction - Complete

**Status:** ✅ Implemented

**Output files:**
- `data/bronze/canonical_records.parquet` - All medication records for cohort
- `data/bronze/medication_vocabulary.parquet` - Unique medication strings

**Schema (canonical_records.parquet):**
| Column | Type | Description |
|--------|------|-------------|
| empi | str | Patient identifier |
| encounter_id | str | Encounter number |
| medication_date | date | Administration date |
| hours_from_t0 | float | Hours from PE Time Zero |
| original_string | str | Raw RPDR medication text |
| code_type | str | Code system (BWH_CC, HCPCS, etc.) |
| code | str | Medication code |
| quantity | float | Quantity administered |
| inpatient | bool | Inpatient flag |
| parsed_name | str | Extracted drug name |
| parsed_dose_value | float | Extracted dose |
| parsed_dose_unit | str | Dose unit |
| parsed_route | str | Administration route |
| parsed_frequency | str | Dosing frequency |
| parse_method | str | 'regex' or 'failed' |
| parse_confidence | float | Confidence score |

**Usage:**
```python
from extractors.canonical_extractor import extract_canonical_records

# Full extraction
df = extract_canonical_records()

# Test mode
df = extract_canonical_records(test_mode=True, test_n_rows=10000)
```

**Next:** Phase 3 - RxNorm Mapping
```

**Step 4: Final commit**

```bash
git add module_04_medications/README.md
git commit -m "docs(module4): document Layer 1 canonical extraction

Layer 1 complete with:
- Canonical records parquet
- Medication vocabulary
- XX.X% dose parsing success
- X,XXX unique patients with medications

Ready for Phase 3: RxNorm Mapping

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Validation Checklist

After completing all tasks, verify:

- ✅ `dose_parser.py` exists with extract_dose, extract_route, extract_frequency
- ✅ `canonical_extractor.py` exists with full extraction pipeline
- ✅ All tests pass (`pytest module_04_medications/tests/ -v`)
- ✅ `canonical_records.parquet` created in bronze/
- ✅ `medication_vocabulary.parquet` created in bronze/
- ✅ Dose parsing success >= 80%
- ✅ All cohort patients (8,713) represented
- ✅ Time alignment computed correctly
- ✅ Key PE medications present (heparin, enoxaparin, etc.)
- ✅ README updated with Layer 1 documentation
- ✅ All changes committed

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Test infrastructure | tests/__init__.py, test_dose_parser.py |
| 2 | Dose parser implementation | extractors/dose_parser.py |
| 3 | Additional parser tests | test_dose_parser.py |
| 4 | Extractor tests | test_canonical_extractor.py |
| 5 | Canonical extractor | extractors/canonical_extractor.py |
| 6 | Test mode extraction | Run & verify |
| 7 | Full extraction | Run & verify |
| 8 | Vocabulary extraction | Add function & test |
| 9 | Generate vocabulary file | Run & verify |
| 10 | Validation & docs | README.md |

**Total:** 10 tasks, ~25-30 steps
