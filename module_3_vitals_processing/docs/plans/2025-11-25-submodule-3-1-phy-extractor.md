# Submodule 3.1: Structured Vitals Extractor (Phy.txt) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract all vital signs from the structured Phy.txt file (2.7GB, ~33M rows) and output a clean parquet file with canonical vital sign names.

**Architecture:** Load Phy.txt in chunks, filter for vital sign concepts, parse blood pressure strings into separate systolic/diastolic values, map all concept names to canonical names (HR, SBP, DBP, RR, SPO2, TEMP, WEIGHT, HEIGHT, BMI), and save to parquet.

**Tech Stack:** Python 3, pandas, pyarrow (parquet), tqdm (progress), pytest

---

## Data Understanding

**Source file:** `/home/moin/TDA_11_25/Data/Phy.txt`
- Format: Pipe-delimited (`|`)
- Size: 2.7GB, ~33M rows
- Columns: `EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number`

**Vital Sign Concepts (from data exploration):**
| Concept_Name | Count | Canonical Name |
|--------------|-------|----------------|
| Temperature | 1,137,661 | TEMP |
| Pulse | 1,063,562 | HR |
| Systolic-Epic | 892,803 | SBP |
| Diastolic-Epic | 892,803 | DBP |
| Blood Pressure-Epic | 892,803 | BP (parse to SBP/DBP) |
| O2 Saturation-SPO2 | 604,843 | SPO2 |
| Respiratory rate | 524,511 | RR |
| Weight | 927,809 | WEIGHT |
| Height | 591,525 | HEIGHT |
| BMI | 1,623,235 | BMI |

**Sample rows:**
```
100003884|10040029737|BWH|00667360|7/21/2015|Blood Pressure-Epic|EPIC|BP|130/77|millimeter of mercury|...
100003884|10040029737|BWH|00667360|7/21/2015|Pulse|EPIC|PUL|74|beats/minute|...
100003884|10040029737|BWH|00667360|7/21/2015|Temperature|EPIC|TEMP|98.6|degrees Fahrenheit|...
```

---

## Task 1: Project Setup

**Files:**
- Create: `module_3_vitals_processing/extractors/__init__.py`
- Create: `module_3_vitals_processing/extractors/phy_extractor.py`
- Create: `module_3_vitals_processing/tests/__init__.py`
- Create: `module_3_vitals_processing/tests/test_phy_extractor.py`
- Create: `module_3_vitals_processing/config/__init__.py`
- Create: `module_3_vitals_processing/config/vitals_config.py`

**Step 1: Create package init files**

```python
# extractors/__init__.py
"""Vitals extraction modules."""
```

```python
# tests/__init__.py
"""Test modules for vitals processing."""
```

```python
# config/__init__.py
"""Configuration for vitals processing."""
```

**Step 2: Create config file with constants**

```python
# config/vitals_config.py
"""Configuration constants for vitals processing."""
from pathlib import Path

# Paths
DATA_DIR = Path("/home/moin/TDA_11_25/Data")
OUTPUT_DIR = Path("/home/moin/TDA_11_25/module_3_vitals_processing/outputs")
MODULE1_OUTPUT_DIR = Path("/home/moin/TDA_11_25/module_1_core_infrastructure/outputs")

# Phy.txt columns
PHY_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Date', 'Concept_Name',
    'Code_Type', 'Code', 'Result', 'Units', 'Provider', 'Clinic',
    'Hospital', 'Inpatient_Outpatient', 'Encounter_number'
]

# Vital sign concepts to extract from Phy.txt
VITAL_CONCEPTS = {
    'Temperature': 'TEMP',
    'Pulse': 'HR',
    'Systolic-Epic': 'SBP',
    'Diastolic-Epic': 'DBP',
    'Blood Pressure-Epic': 'BP',  # Will be parsed into SBP/DBP
    'Systolic-LFA3959.1': 'SBP',
    'Diastolic-LFA3959.2': 'DBP',
    'Systolic/Diastolic-LFA3959.0': 'BP',
    'O2 Saturation-SPO2': 'SPO2',
    'O2 Saturation%': 'SPO2',
    'Respiratory rate': 'RR',
    'Weight': 'WEIGHT',
    'Height': 'HEIGHT',
    'BMI': 'BMI',
}

# Canonical vital names
CANONICAL_VITALS = ['HR', 'SBP', 'DBP', 'RR', 'SPO2', 'TEMP', 'WEIGHT', 'HEIGHT', 'BMI']

# Processing config
CHUNK_SIZE = 500_000  # Rows per chunk for large file processing
```

**Step 3: Verify setup by running Python import**

Run: `cd /home/moin/TDA_11_25 && python3 -c "from module_3_vitals_processing.config.vitals_config import VITAL_CONCEPTS; print('Config OK:', len(VITAL_CONCEPTS), 'concepts')"`

Expected: `Config OK: 14 concepts`

**Step 4: Commit**

```bash
git add module_3_vitals_processing/extractors/__init__.py \
        module_3_vitals_processing/tests/__init__.py \
        module_3_vitals_processing/config/__init__.py \
        module_3_vitals_processing/config/vitals_config.py
git commit -m "feat(module3): add project structure and config for phy extractor"
```

---

## Task 2: Blood Pressure Parser

**Files:**
- Create: `module_3_vitals_processing/extractors/phy_extractor.py`
- Create: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Write the failing test for BP parsing**

```python
# tests/test_phy_extractor.py
"""Tests for Phy.txt structured vitals extractor."""
import pytest
from module_3_vitals_processing.extractors.phy_extractor import parse_blood_pressure


class TestParseBloodPressure:
    """Tests for parse_blood_pressure function."""

    def test_normal_bp(self):
        """Test parsing normal BP string."""
        sbp, dbp = parse_blood_pressure("130/77")
        assert sbp == 130.0
        assert dbp == 77.0

    def test_high_bp(self):
        """Test parsing high BP values."""
        sbp, dbp = parse_blood_pressure("180/120")
        assert sbp == 180.0
        assert dbp == 120.0

    def test_low_bp(self):
        """Test parsing low BP values."""
        sbp, dbp = parse_blood_pressure("90/60")
        assert sbp == 90.0
        assert dbp == 60.0

    def test_invalid_bp_text(self):
        """Test that non-numeric BP returns None."""
        sbp, dbp = parse_blood_pressure("Left arm")
        assert sbp is None
        assert dbp is None

    def test_invalid_bp_sitting(self):
        """Test that positional text returns None."""
        sbp, dbp = parse_blood_pressure("Sitting")
        assert sbp is None
        assert dbp is None

    def test_empty_string(self):
        """Test empty string returns None."""
        sbp, dbp = parse_blood_pressure("")
        assert sbp is None
        assert dbp is None

    def test_none_input(self):
        """Test None input returns None."""
        sbp, dbp = parse_blood_pressure(None)
        assert sbp is None
        assert dbp is None

    def test_missing_diastolic(self):
        """Test single value returns None for both."""
        sbp, dbp = parse_blood_pressure("130")
        assert sbp is None
        assert dbp is None

    def test_spaces_around_slash(self):
        """Test BP with spaces around slash."""
        sbp, dbp = parse_blood_pressure("130 / 77")
        assert sbp == 130.0
        assert dbp == 77.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
# extractors/phy_extractor.py
"""
Submodule 3.1: Structured Vitals Extractor (Phy.txt)
====================================================

Extracts vital signs from the structured Phy.txt file.
"""
import re
from typing import Tuple, Optional


def parse_blood_pressure(bp_string: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse blood pressure string like '130/77' into (systolic, diastolic).

    Args:
        bp_string: Blood pressure string in format 'SBP/DBP'

    Returns:
        Tuple of (systolic, diastolic) as floats, or (None, None) if invalid
    """
    if bp_string is None or not isinstance(bp_string, str):
        return None, None

    bp_string = bp_string.strip()
    if not bp_string:
        return None, None

    # Pattern: digits / digits (with optional spaces)
    pattern = r'^(\d+)\s*/\s*(\d+)$'
    match = re.match(pattern, bp_string)

    if not match:
        return None, None

    try:
        sbp = float(match.group(1))
        dbp = float(match.group(2))
        return sbp, dbp
    except (ValueError, TypeError):
        return None, None
```

**Step 4: Run test to verify it passes**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestParseBloodPressure -v`

Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py \
        module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "feat(module3): add blood pressure parser with tests"
```

---

## Task 3: Concept Name Mapper

**Files:**
- Modify: `module_3_vitals_processing/extractors/phy_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Write the failing test for concept mapping**

Add to `tests/test_phy_extractor.py`:

```python
from module_3_vitals_processing.extractors.phy_extractor import map_concept_to_canonical


class TestMapConceptToCanonical:
    """Tests for map_concept_to_canonical function."""

    def test_pulse_to_hr(self):
        """Test Pulse maps to HR."""
        assert map_concept_to_canonical("Pulse") == "HR"

    def test_temperature(self):
        """Test Temperature maps to TEMP."""
        assert map_concept_to_canonical("Temperature") == "TEMP"

    def test_blood_pressure_epic(self):
        """Test Blood Pressure-Epic maps to BP."""
        assert map_concept_to_canonical("Blood Pressure-Epic") == "BP"

    def test_systolic_epic(self):
        """Test Systolic-Epic maps to SBP."""
        assert map_concept_to_canonical("Systolic-Epic") == "SBP"

    def test_diastolic_epic(self):
        """Test Diastolic-Epic maps to DBP."""
        assert map_concept_to_canonical("Diastolic-Epic") == "DBP"

    def test_o2_saturation(self):
        """Test O2 Saturation-SPO2 maps to SPO2."""
        assert map_concept_to_canonical("O2 Saturation-SPO2") == "SPO2"

    def test_respiratory_rate(self):
        """Test Respiratory rate maps to RR."""
        assert map_concept_to_canonical("Respiratory rate") == "RR"

    def test_weight(self):
        """Test Weight maps to WEIGHT."""
        assert map_concept_to_canonical("Weight") == "WEIGHT"

    def test_height(self):
        """Test Height maps to HEIGHT."""
        assert map_concept_to_canonical("Height") == "HEIGHT"

    def test_bmi(self):
        """Test BMI maps to BMI."""
        assert map_concept_to_canonical("BMI") == "BMI"

    def test_unknown_concept(self):
        """Test unknown concept returns None."""
        assert map_concept_to_canonical("Flu-High Dose") is None

    def test_none_input(self):
        """Test None input returns None."""
        assert map_concept_to_canonical(None) is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestMapConceptToCanonical -v`

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `extractors/phy_extractor.py`:

```python
from module_3_vitals_processing.config.vitals_config import VITAL_CONCEPTS


def map_concept_to_canonical(concept_name: Optional[str]) -> Optional[str]:
    """
    Map Phy.txt Concept_Name to canonical vital sign name.

    Args:
        concept_name: The Concept_Name from Phy.txt

    Returns:
        Canonical vital name (HR, SBP, DBP, etc.) or None if not a vital
    """
    if concept_name is None:
        return None

    return VITAL_CONCEPTS.get(concept_name)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestMapConceptToCanonical -v`

Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py \
        module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "feat(module3): add concept name mapper with tests"
```

---

## Task 4: Result Value Parser

**Files:**
- Modify: `module_3_vitals_processing/extractors/phy_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Write the failing test for result parsing**

Add to `tests/test_phy_extractor.py`:

```python
from module_3_vitals_processing.extractors.phy_extractor import parse_result_value


class TestParseResultValue:
    """Tests for parse_result_value function."""

    def test_integer_value(self):
        """Test parsing integer."""
        assert parse_result_value("74") == 74.0

    def test_float_value(self):
        """Test parsing float."""
        assert parse_result_value("98.6") == 98.6

    def test_float_with_leading_zero(self):
        """Test parsing float with leading zero."""
        assert parse_result_value("0.5") == 0.5

    def test_empty_string(self):
        """Test empty string returns None."""
        assert parse_result_value("") is None

    def test_none_input(self):
        """Test None input returns None."""
        assert parse_result_value(None) is None

    def test_text_value(self):
        """Test non-numeric text returns None."""
        assert parse_result_value("Left arm") is None

    def test_whitespace(self):
        """Test value with whitespace."""
        assert parse_result_value("  74  ") == 74.0

    def test_negative_value(self):
        """Test negative value (should still parse)."""
        assert parse_result_value("-5") == -5.0

    def test_greater_than_symbol(self):
        """Test value with > symbol."""
        assert parse_result_value(">100") == 100.0

    def test_less_than_symbol(self):
        """Test value with < symbol."""
        assert parse_result_value("<50") == 50.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestParseResultValue -v`

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `extractors/phy_extractor.py`:

```python
def parse_result_value(result: Optional[str]) -> Optional[float]:
    """
    Parse Result field to numeric value.

    Args:
        result: The Result string from Phy.txt

    Returns:
        Float value or None if not parseable
    """
    if result is None or not isinstance(result, str):
        return None

    result = result.strip()
    if not result:
        return None

    # Remove common prefixes like > or <
    result = result.lstrip('<>').strip()

    try:
        return float(result)
    except (ValueError, TypeError):
        return None
```

**Step 4: Run test to verify it passes**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestParseResultValue -v`

Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py \
        module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "feat(module3): add result value parser with tests"
```

---

## Task 5: Row Processor

**Files:**
- Modify: `module_3_vitals_processing/extractors/phy_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Write the failing test for row processing**

Add to `tests/test_phy_extractor.py`:

```python
import pandas as pd
from module_3_vitals_processing.extractors.phy_extractor import process_vital_row


class TestProcessVitalRow:
    """Tests for process_vital_row function."""

    def test_pulse_row(self):
        """Test processing a Pulse row."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Pulse',
            'Result': '74',
            'Units': 'beats/minute',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 1
        assert result[0]['EMPI'] == '100003884'
        assert result[0]['vital_type'] == 'HR'
        assert result[0]['value'] == 74.0
        assert result[0]['units'] == 'beats/minute'

    def test_blood_pressure_row(self):
        """Test processing a Blood Pressure row produces two records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Blood Pressure-Epic',
            'Result': '130/77',
            'Units': 'millimeter of mercury',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 2

        # Check SBP
        sbp_record = [r for r in result if r['vital_type'] == 'SBP'][0]
        assert sbp_record['value'] == 130.0

        # Check DBP
        dbp_record = [r for r in result if r['vital_type'] == 'DBP'][0]
        assert dbp_record['value'] == 77.0

    def test_invalid_bp_row(self):
        """Test that invalid BP like 'Left arm' produces no records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Blood Pressure-Epic',
            'Result': 'Left arm',
            'Units': '',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 0

    def test_non_vital_row(self):
        """Test that non-vital concept produces no records."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Flu-High Dose',
            'Result': '76',
            'Units': '',
            'Inpatient_Outpatient': 'Outpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 0

    def test_temperature_row(self):
        """Test processing a Temperature row."""
        row = pd.Series({
            'EMPI': '100003884',
            'Date': '7/21/2015',
            'Concept_Name': 'Temperature',
            'Result': '98.6',
            'Units': 'degrees Fahrenheit',
            'Inpatient_Outpatient': 'Inpatient',
            'Encounter_number': 'EPIC-3085982676'
        })
        result = process_vital_row(row)
        assert len(result) == 1
        assert result[0]['vital_type'] == 'TEMP'
        assert result[0]['value'] == 98.6
        assert result[0]['encounter_type'] == 'Inpatient'
```

**Step 2: Run test to verify it fails**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestProcessVitalRow -v`

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `extractors/phy_extractor.py`:

```python
import pandas as pd
from typing import List, Dict, Any


def process_vital_row(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Process a single row from Phy.txt and extract vital sign records.

    Args:
        row: A pandas Series representing one row from Phy.txt

    Returns:
        List of vital sign dictionaries. Blood Pressure rows produce 2 records.
        Non-vital rows or invalid values produce empty list.
    """
    concept_name = row.get('Concept_Name')
    canonical = map_concept_to_canonical(concept_name)

    if canonical is None:
        return []

    empi = str(row.get('EMPI', ''))
    date_str = row.get('Date', '')
    result = row.get('Result', '')
    units = row.get('Units', '')
    encounter_type = row.get('Inpatient_Outpatient', '')
    encounter_number = row.get('Encounter_number', '')

    base_record = {
        'EMPI': empi,
        'date_str': date_str,
        'units': units,
        'source': 'phy',
        'encounter_type': encounter_type,
        'encounter_number': encounter_number,
    }

    # Handle Blood Pressure (combined SBP/DBP)
    if canonical == 'BP':
        sbp, dbp = parse_blood_pressure(result)
        if sbp is None or dbp is None:
            return []

        return [
            {**base_record, 'vital_type': 'SBP', 'value': sbp},
            {**base_record, 'vital_type': 'DBP', 'value': dbp},
        ]

    # Handle regular vitals
    value = parse_result_value(result)
    if value is None:
        return []

    return [{**base_record, 'vital_type': canonical, 'value': value}]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestProcessVitalRow -v`

Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py \
        module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "feat(module3): add vital row processor with tests"
```

---

## Task 6: Main Extraction Function

**Files:**
- Modify: `module_3_vitals_processing/extractors/phy_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Write the failing test for main extraction**

Add to `tests/test_phy_extractor.py`:

```python
import tempfile
import os
from module_3_vitals_processing.extractors.phy_extractor import extract_phy_vitals


class TestExtractPhyVitals:
    """Tests for extract_phy_vitals main function."""

    def test_extract_from_sample_file(self, tmp_path):
        """Test extraction from a sample Phy.txt file."""
        # Create sample data
        sample_data = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
100003884|10040029737|BWH|00667360|7/21/2015|Pulse|EPIC|PUL|74|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-001
100003884|10040029737|BWH|00667360|7/21/2015|Blood Pressure-Epic|EPIC|BP|130/77|millimeter of mercury|Doctor|Clinic|BWH|Outpatient|EPIC-001
100003884|10040029737|BWH|00667360|7/21/2015|Temperature|EPIC|TEMP|98.6|degrees Fahrenheit|Doctor|Clinic|BWH|Inpatient|EPIC-002
100003884|10040029737|BWH|00667360|7/21/2015|Flu-High Dose|EPIC|76|||Doctor|Clinic|BWH|Outpatient|EPIC-001
100003884|10040029737|BWH|00667360|7/21/2015|Blood Pressure-Epic|EPIC|BP|Left arm|||Doctor|Clinic|BWH|Outpatient|EPIC-001"""

        # Write sample file
        sample_file = tmp_path / "test_phy.txt"
        sample_file.write_text(sample_data)

        # Run extraction
        output_file = tmp_path / "output.parquet"
        result_df = extract_phy_vitals(str(sample_file), str(output_file))

        # Verify results
        assert len(result_df) == 4  # 1 Pulse + 2 BP (SBP/DBP) + 1 Temp = 4
        assert set(result_df['vital_type'].unique()) == {'HR', 'SBP', 'DBP', 'TEMP'}
        assert result_df[result_df['vital_type'] == 'HR']['value'].iloc[0] == 74.0
        assert result_df[result_df['vital_type'] == 'SBP']['value'].iloc[0] == 130.0
        assert result_df[result_df['vital_type'] == 'DBP']['value'].iloc[0] == 77.0
        assert result_df[result_df['vital_type'] == 'TEMP']['value'].iloc[0] == 98.6

        # Verify parquet file was created
        assert output_file.exists()

    def test_date_parsing(self, tmp_path):
        """Test that dates are parsed correctly."""
        sample_data = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Concept_Name|Code_Type|Code|Result|Units|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
100003884|10040029737|BWH|00667360|7/21/2015|Pulse|EPIC|PUL|74|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-001
100003884|10040029737|BWH|00667360|12/25/2020|Pulse|EPIC|PUL|80|beats/minute|Doctor|Clinic|BWH|Outpatient|EPIC-002"""

        sample_file = tmp_path / "test_phy.txt"
        sample_file.write_text(sample_data)
        output_file = tmp_path / "output.parquet"

        result_df = extract_phy_vitals(str(sample_file), str(output_file))

        assert len(result_df) == 2
        # Verify timestamp column exists and is datetime
        assert 'timestamp' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['timestamp'])
```

**Step 2: Run test to verify it fails**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestExtractPhyVitals -v`

Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `extractors/phy_extractor.py`:

```python
from pathlib import Path
from tqdm import tqdm
from module_3_vitals_processing.config.vitals_config import (
    PHY_COLUMNS, VITAL_CONCEPTS, CHUNK_SIZE
)


def extract_phy_vitals(
    phy_file: str,
    output_file: str,
    chunk_size: int = CHUNK_SIZE,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract vital signs from Phy.txt structured file.

    Args:
        phy_file: Path to Phy.txt input file
        output_file: Path to output parquet file
        chunk_size: Number of rows to process per chunk
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with extracted vitals
    """
    phy_path = Path(phy_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get vital concept names for filtering
    vital_concept_names = set(VITAL_CONCEPTS.keys())

    all_records = []

    # Count total lines for progress bar
    if show_progress:
        with open(phy_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header

    # Process in chunks
    chunks = pd.read_csv(
        phy_path,
        sep='|',
        names=PHY_COLUMNS,
        header=0,
        dtype=str,
        chunksize=chunk_size,
        low_memory=False
    )

    if show_progress:
        chunks = tqdm(chunks, desc="Processing Phy.txt", total=total_lines // chunk_size + 1)

    for chunk in chunks:
        # Filter to only vital sign concepts
        mask = chunk['Concept_Name'].isin(vital_concept_names)
        vital_chunk = chunk[mask]

        # Process each row
        for _, row in vital_chunk.iterrows():
            records = process_vital_row(row)
            all_records.extend(records)

    # Create DataFrame
    if not all_records:
        # Return empty DataFrame with correct schema
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'vital_type', 'value', 'units',
            'source', 'encounter_type', 'encounter_number'
        ])
    else:
        df = pd.DataFrame(all_records)

        # Parse dates
        df['timestamp'] = pd.to_datetime(df['date_str'], format='%m/%d/%Y', errors='coerce')
        df = df.drop(columns=['date_str'])

        # Reorder columns
        df = df[[
            'EMPI', 'timestamp', 'vital_type', 'value', 'units',
            'source', 'encounter_type', 'encounter_number'
        ]]

    # Save to parquet
    df.to_parquet(output_path, index=False)

    return df
```

**Step 4: Run test to verify it passes**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestExtractPhyVitals -v`

Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py \
        module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "feat(module3): add main extraction function with tests"
```

---

## Task 7: Run All Tests

**Step 1: Run complete test suite**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py -v --tb=short`

Expected: All 38+ tests PASS

**Step 2: Commit any final fixes**

---

## Task 8: Integration Test with Real Data (Small Sample)

**Files:**
- Modify: `module_3_vitals_processing/tests/test_phy_extractor.py`

**Step 1: Add integration test that runs on first 10000 rows of real data**

Add to `tests/test_phy_extractor.py`:

```python
import os


class TestIntegration:
    """Integration tests with real data."""

    @pytest.mark.skipif(
        not os.path.exists("/home/moin/TDA_11_25/Data/Phy.txt"),
        reason="Real data not available"
    )
    def test_extract_sample_from_real_data(self, tmp_path):
        """Test extraction from first 10000 rows of real Phy.txt."""
        from module_3_vitals_processing.extractors.phy_extractor import extract_phy_vitals

        # Create a sample file with first 10000 rows
        sample_file = tmp_path / "sample_phy.txt"
        with open("/home/moin/TDA_11_25/Data/Phy.txt", 'r') as f:
            lines = [next(f) for _ in range(10001)]  # Header + 10000 rows
        sample_file.write_text(''.join(lines))

        output_file = tmp_path / "sample_output.parquet"

        result_df = extract_phy_vitals(
            str(sample_file),
            str(output_file),
            chunk_size=5000,
            show_progress=False
        )

        # Basic sanity checks
        assert len(result_df) > 0, "Should extract some vitals"
        assert 'EMPI' in result_df.columns
        assert 'timestamp' in result_df.columns
        assert 'vital_type' in result_df.columns
        assert 'value' in result_df.columns

        # Check we got multiple vital types
        vital_types = result_df['vital_type'].unique()
        print(f"Extracted vital types: {vital_types}")
        print(f"Total records: {len(result_df)}")
        print(f"Records by type:\n{result_df['vital_type'].value_counts()}")
```

**Step 2: Run integration test**

Run: `cd /home/moin/TDA_11_25 && python3 -m pytest module_3_vitals_processing/tests/test_phy_extractor.py::TestIntegration -v -s`

Expected: PASS with output showing extracted vital types and counts

**Step 3: Commit**

```bash
git add module_3_vitals_processing/tests/test_phy_extractor.py
git commit -m "test(module3): add integration test with real data sample"
```

---

## Task 9: Create CLI Entry Point

**Files:**
- Modify: `module_3_vitals_processing/extractors/phy_extractor.py`

**Step 1: Add main block for command-line usage**

Add to end of `extractors/phy_extractor.py`:

```python
if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Extract vitals from Phy.txt")
    parser.add_argument(
        "--input", "-i",
        default="/home/moin/TDA_11_25/Data/Phy.txt",
        help="Path to Phy.txt input file"
    )
    parser.add_argument(
        "--output", "-o",
        default="/home/moin/TDA_11_25/module_3_vitals_processing/outputs/discovery/phy_vitals_raw.parquet",
        help="Path to output parquet file"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size for processing (default: {CHUNK_SIZE})"
    )

    args = parser.parse_args()

    print(f"Extracting vitals from: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Chunk size: {args.chunk_size}")

    start_time = time.time()
    df = extract_phy_vitals(args.input, args.output, args.chunk_size)
    elapsed = time.time() - start_time

    print(f"\nExtraction complete in {elapsed:.1f} seconds")
    print(f"Total records: {len(df):,}")
    print(f"\nRecords by vital type:")
    print(df['vital_type'].value_counts().to_string())
    print(f"\nOutput saved to: {args.output}")
```

**Step 2: Test CLI with small sample**

Run: `cd /home/moin/TDA_11_25 && head -1001 Data/Phy.txt > /tmp/test_phy.txt && python3 -m module_3_vitals_processing.extractors.phy_extractor -i /tmp/test_phy.txt -o /tmp/test_output.parquet`

Expected: Output showing extraction stats

**Step 3: Commit**

```bash
git add module_3_vitals_processing/extractors/phy_extractor.py
git commit -m "feat(module3): add CLI entry point for phy extractor"
```

---

## Task 10: Full Extraction Run

**Step 1: Run full extraction on real data**

Run: `cd /home/moin/TDA_11_25 && python3 -m module_3_vitals_processing.extractors.phy_extractor`

Expected:
- Progress bar showing chunks processed
- Completion in ~15-30 minutes
- Output file at `module_3_vitals_processing/outputs/discovery/phy_vitals_raw.parquet`

**Step 2: Verify output**

Run: `cd /home/moin/TDA_11_25 && python3 -c "import pandas as pd; df = pd.read_parquet('module_3_vitals_processing/outputs/discovery/phy_vitals_raw.parquet'); print(f'Total: {len(df):,}'); print(df['vital_type'].value_counts())"`

Expected output similar to:
```
Total: 8,500,000+
HR        1,063,562
TEMP      1,137,661
SBP       1,165,606  (Systolic + BP parsed)
DBP       1,165,606  (Diastolic + BP parsed)
SPO2        604,843
RR          524,511
WEIGHT      927,809
HEIGHT      591,525
BMI       1,623,235
```

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat(module3): complete submodule 3.1 phy extractor implementation"
```

---

## Summary

This plan implements Submodule 3.1 (Structured Vitals Extractor) with:

1. **4 core functions:**
   - `parse_blood_pressure()` - Parse "130/77" → (130.0, 77.0)
   - `map_concept_to_canonical()` - "Pulse" → "HR"
   - `parse_result_value()` - "74" → 74.0
   - `process_vital_row()` - Process one row → list of vital records

2. **1 main function:**
   - `extract_phy_vitals()` - Process entire Phy.txt file

3. **~40 unit tests** covering:
   - Blood pressure parsing (9 tests)
   - Concept mapping (12 tests)
   - Result parsing (10 tests)
   - Row processing (5 tests)
   - File extraction (2 tests)
   - Integration (1 test)

4. **CLI entry point** for running extraction

**Output:** `outputs/discovery/phy_vitals_raw.parquet` with schema:
- EMPI (str)
- timestamp (datetime)
- vital_type (str): HR, SBP, DBP, RR, SPO2, TEMP, WEIGHT, HEIGHT, BMI
- value (float)
- units (str)
- source (str): 'phy'
- encounter_type (str): 'Inpatient' or 'Outpatient'
- encounter_number (str)
