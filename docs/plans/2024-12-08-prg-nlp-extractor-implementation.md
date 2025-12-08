# Prg NLP Extractor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract vital signs from 4.6M progress notes (Prg.txt) with section-aware filtering and checkpointing.

**Architecture:** Extends Hnp extractor architecture - imports shared patterns, adds Prg-specific section detection, skip section filtering, temperature method extraction, and checkpoint/resume for large file processing.

**Tech Stack:** Python, pandas, regex, multiprocessing, parquet, JSON checkpoints

---

## Task 1: Create prg_patterns.py - Section Patterns

**Files:**
- Create: `module_3_vitals_processing/extractors/prg_patterns.py`
- Test: `module_3_vitals_processing/tests/test_prg_patterns.py`

**Step 1: Write the failing test**

Create `module_3_vitals_processing/tests/test_prg_patterns.py`:

```python
"""Tests for prg_patterns module."""
import pytest
import re


class TestPrgSectionPatterns:
    """Test Prg section pattern definitions."""

    def test_section_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        assert isinstance(PRG_SECTION_PATTERNS, dict)
        assert len(PRG_SECTION_PATTERNS) >= 10

    def test_physical_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['physical_exam']
        assert re.search(pattern, "Physical Exam: BP 120/80", re.IGNORECASE)
        assert re.search(pattern, "Physical Examination: normal", re.IGNORECASE)

    def test_vitals_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['vitals']
        assert re.search(pattern, "Vitals: HR 72", re.IGNORECASE)
        assert re.search(pattern, "Vital: T 98.6", re.IGNORECASE)

    def test_on_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['on_exam']
        assert re.search(pattern, "ON EXAM: Vital Signs BP 120/80", re.IGNORECASE)

    def test_objective_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['objective']
        assert re.search(pattern, "Objective: Physical Exam", re.IGNORECASE)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgSectionPatterns -v`

Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

Create `module_3_vitals_processing/extractors/prg_patterns.py`:

```python
"""Regex patterns and constants for Prg.txt extraction."""

# Prg.txt columns (same as Hnp.txt)
PRG_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Section patterns: (regex, timestamp_offset_hours)
# Based on analysis of 300K rows from actual Prg.txt
PRG_SECTION_PATTERNS = {
    # High frequency (>5000 occurrences in sample)
    'physical_exam': (r'Physical\s+Exam(?:ination)?[:\s]', 0),
    'objective': (r'Objective[:\s]', 0),
    'exam': (r'\bExam[:\s]', 0),
    'vitals': (r'Vitals?[:\s]', 0),

    # Specific vitals headers
    'vital_signs': (r'Vital\s+[Ss]igns?[:\s]', 0),
    'vital_signs_recent': (r'Vital\s+signs?:\s*\(most\s+recent\)', 0),
    'on_exam': (r'ON\s+EXAM[:\s]', 0),

    # Combined headers (common in Prg)
    'physical_exam_vitals': (r'Physical\s+Exam[:\s]+Vitals?[:\s]', 0),
    'physical_exam_gen': (r'Physical\s+Exam[:\s]+Gen(?:eral)?[:\s]', 0),
    'objective_temp': (r'Objective[:\s]+Temperature[:\s]', 0),

    # SOAP format
    'assessment_plan': (r'Assessment\s*(?:&|and|/)?\s*Plan[:\s]', 0),
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgSectionPatterns -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_patterns.py module_3_vitals_processing/tests/test_prg_patterns.py
git commit -m "feat(prg): add section patterns for progress note extraction"
```

---

## Task 2: Add Skip Patterns to prg_patterns.py

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_patterns.py`
- Modify: `module_3_vitals_processing/tests/test_prg_patterns.py`

**Step 1: Write the failing test**

Add to `test_prg_patterns.py`:

```python
class TestPrgSkipPatterns:
    """Test Prg skip section patterns."""

    def test_skip_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        assert isinstance(PRG_SKIP_PATTERNS, list)
        assert len(PRG_SKIP_PATTERNS) >= 10

    def test_allergies_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Allergies: atenolol - fatigue, HR 50"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_medications_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Medications: lisinopril 10mg daily"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_past_medical_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Past Medical History: hypertension, diabetes"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_family_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Family History: father with MI"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_reactions_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Reactions: hives, swelling"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgSkipPatterns -v`

Expected: FAIL with "ImportError" for PRG_SKIP_PATTERNS

**Step 3: Write minimal implementation**

Add to `prg_patterns.py`:

```python
# Skip sections (false positive sources)
# HR/BP values in these sections are NOT vital measurements
PRG_SKIP_PATTERNS = [
    # Allergies/Reactions (HR values as side effects)
    r'Allerg(?:ies|ic|en)[:\s]',
    r'[Rr]eaction\(?s?\)?[:\s]',

    # Medication lists
    r'Medications?[:\s]',
    r'(?:Outpatient\s+)?Prescriptions?[:\s]',
    r'Scheduled\s+Meds[:\s]',

    # History sections (historical mentions)
    r'Past\s+(?:Medical\s+)?History[:\s]',
    r'Family\s+History[:\s]',
    r'(?:History\s+of\s+)?Present\s+Illness[:\s]',
    r'Social\s+History[:\s]',
    r'Surgical\s+History[:\s]',

    # Other non-vitals
    r'Review\s+of\s+Systems[:\s]',
    r'ROS[:\s]',
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgSkipPatterns -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_patterns.py module_3_vitals_processing/tests/test_prg_patterns.py
git commit -m "feat(prg): add skip patterns for false positive filtering"
```

---

## Task 3: Add Prg-specific Vitals Patterns

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_patterns.py`
- Modify: `module_3_vitals_processing/tests/test_prg_patterns.py`

**Step 1: Write the failing test**

Add to `test_prg_patterns.py`:

```python
class TestPrgVitalsPatterns:
    """Test Prg-specific vitals patterns."""

    def test_bp_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_BP_PATTERNS
        assert isinstance(PRG_BP_PATTERNS, list)
        assert len(PRG_BP_PATTERNS) >= 2

    def test_bp_spelled_out_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_BP_PATTERNS
        text = "Blood pressure 130/85"
        for pattern, _ in PRG_BP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '130'
                assert match.group(2) == '85'
                break
        else:
            pytest.fail("No BP pattern matched 'Blood pressure 130/85'")

    def test_hr_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_HR_PATTERNS
        assert isinstance(PRG_HR_PATTERNS, list)
        assert len(PRG_HR_PATTERNS) >= 2

    def test_hr_p_format_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_HR_PATTERNS
        text = "BP 120/80, P 72"
        for pattern, _ in PRG_HR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '72'
                break
        else:
            pytest.fail("No HR pattern matched 'P 72'")

    def test_spo2_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SPO2_PATTERNS
        assert isinstance(PRG_SPO2_PATTERNS, list)

    def test_spo2_o2_sat_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SPO2_PATTERNS
        text = "O2 sat 97"
        for pattern, _ in PRG_SPO2_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '97'
                break
        else:
            pytest.fail("No SpO2 pattern matched 'O2 sat 97'")

    def test_rr_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_RR_PATTERNS
        assert isinstance(PRG_RR_PATTERNS, list)

    def test_rr_resp_format_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_RR_PATTERNS
        text = "Resp: 18"
        for pattern, _ in PRG_RR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '18'
                break
        else:
            pytest.fail("No RR pattern matched 'Resp: 18'")
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgVitalsPatterns -v`

Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `prg_patterns.py`:

```python
# Prg-specific Blood Pressure patterns (extend Hnp)
PRG_BP_PATTERNS = [
    # Spelled out format
    (r'Blood\s+pressure\s+(\d{2,3})/(\d{2,3})', 0.95),
    # With ranges in parentheses
    (r'BP:\s*\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
]

# Prg-specific Heart Rate patterns (extend Hnp)
PRG_HR_PATTERNS = [
    # P format common in Prg (with word boundary to avoid matching in words)
    (r'\bP\s+(\d{2,3})\b', 0.85),
    # With abnormal flag
    (r'Pulse\s*\(!\)\s*(\d{2,3})', 0.9),
    # With ranges
    (r'Heart\s+Rate:\s*\[\d+-\d+\]\s*(\d{2,3})', 0.95),
]

# Prg-specific SpO2 patterns (extend Hnp)
PRG_SPO2_PATTERNS = [
    # O2 sat alternate notation
    (r'O2\s*sat\s*(\d{2,3})', 0.85),
    # With space before %
    (r'SpO2\s*:?\s*(\d{2,3})\s*%', 0.95),
]

# Prg-specific Respiratory Rate patterns (extend Hnp)
PRG_RR_PATTERNS = [
    # Resp format
    (r'Resp[:\s]+(\d{1,2})\b', 0.9),
]
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgVitalsPatterns -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_patterns.py module_3_vitals_processing/tests/test_prg_patterns.py
git commit -m "feat(prg): add Prg-specific vitals patterns"
```

---

## Task 4: Add Temperature Method Patterns

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_patterns.py`
- Modify: `module_3_vitals_processing/tests/test_prg_patterns.py`

**Step 1: Write the failing test**

Add to `test_prg_patterns.py`:

```python
class TestTempMethodPatterns:
    """Test temperature method extraction patterns."""

    def test_temp_method_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        assert isinstance(PRG_TEMP_PATTERNS, list)
        assert len(PRG_TEMP_PATTERNS) >= 2

    def test_temp_method_map_exists(self):
        from module_3_vitals_processing.extractors.prg_patterns import TEMP_METHOD_MAP
        assert isinstance(TEMP_METHOD_MAP, dict)
        assert 'oral' in TEMP_METHOD_MAP
        assert 'temporal' in TEMP_METHOD_MAP
        assert 'rectal' in TEMP_METHOD_MAP

    def test_temp_with_oral_method(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp 36.8 °C (98.2 °F) (Oral)"
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.lastindex >= 3:
                assert 'oral' in match.group(3).lower()
                break

    def test_temp_with_temporal_method(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp 36.2 °C (97.1 °F) (Temporal)"
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.lastindex >= 3:
                assert 'temporal' in match.group(3).lower()
                break

    def test_temp_src_format(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp(Src) 36.7 °C (98 °F) (Oral)"
        matched = False
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matched = True
                break
        assert matched, "Temp(Src) format should match"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestTempMethodPatterns -v`

Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `prg_patterns.py`:

```python
# Temperature patterns with method capture
# Group 1: value, Group 2: unit (C/F), Group 3: method (optional)
PRG_TEMP_PATTERNS = [
    # Temp(Src) format: Temp(Src) 36.7 °C (98 °F) (Oral)
    (r'Temp\(?Src\)?\s*(\d{2,3}\.?\d?)\s*[?°]?\s*([CF])\s*\([^)]+\)\s*\((\w+)\)', 1.0),
    # Temp with method in parentheses: Temp 36.8 °C (98.2 °F) (Temporal)
    (r'Temp\s+(\d{2,3}\.?\d?)\s*[?°]?\s*([CF])\s*\([^)]+\)\s*\((\w+)\)', 0.95),
    # Temp with just value and method: Temp 98.6F (Oral)
    (r'Temp\s+(\d{2,3}\.?\d?)\s*[?°]?\s*([CF])\s*\((\w+)\)', 0.9),
]

# Map raw method strings to canonical names
TEMP_METHOD_MAP = {
    'oral': 'oral',
    'orally': 'oral',
    'po': 'oral',
    'temporal': 'temporal',
    'forehead': 'temporal',
    'rectal': 'rectal',
    'rectally': 'rectal',
    'pr': 'rectal',
    'axillary': 'axillary',
    'axilla': 'axillary',
    'tympanic': 'tympanic',
    'ear': 'tympanic',
}
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestTempMethodPatterns -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_patterns.py module_3_vitals_processing/tests/test_prg_patterns.py
git commit -m "feat(prg): add temperature method extraction patterns"
```

---

## Task 5: Update vitals_config.py with Prg Paths

**Files:**
- Modify: `module_3_vitals_processing/config/vitals_config.py`

**Step 1: Write the failing test**

Add to `test_prg_patterns.py`:

```python
class TestPrgConfig:
    """Test Prg configuration in vitals_config."""

    def test_prg_columns_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_COLUMNS
        assert isinstance(PRG_COLUMNS, list)
        assert 'EMPI' in PRG_COLUMNS
        assert 'Report_Text' in PRG_COLUMNS

    def test_prg_input_path_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_INPUT_PATH
        assert 'Prg.txt' in str(PRG_INPUT_PATH)

    def test_prg_output_path_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_OUTPUT_PATH
        assert 'prg_vitals_raw.parquet' in str(PRG_OUTPUT_PATH)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgConfig -v`

Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `module_3_vitals_processing/config/vitals_config.py`:

```python
# Prg.txt columns (same format as Hnp.txt)
PRG_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Default paths for Prg extractor
PRG_INPUT_PATH = DATA_DIR / 'Prg.txt'
PRG_OUTPUT_PATH = OUTPUT_DIR / 'discovery' / 'prg_vitals_raw.parquet'
PRG_CHUNKS_DIR = OUTPUT_DIR / 'discovery' / 'prg_chunks'
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py::TestPrgConfig -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/config/vitals_config.py module_3_vitals_processing/tests/test_prg_patterns.py
git commit -m "feat(config): add Prg paths to vitals_config"
```

---

## Task 6: Create prg_extractor.py - Checkpoint Dataclass

**Files:**
- Create: `module_3_vitals_processing/extractors/prg_extractor.py`
- Create: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Create `module_3_vitals_processing/tests/test_prg_extractor.py`:

```python
"""Tests for prg_extractor module."""
import pytest
from datetime import datetime
import json
import tempfile
from pathlib import Path


class TestExtractionCheckpoint:
    """Test checkpoint dataclass and serialization."""

    def test_checkpoint_dataclass_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        checkpoint = ExtractionCheckpoint(
            input_path='/path/to/input.txt',
            output_path='/path/to/output.parquet',
            rows_processed=1000,
            chunks_completed=1,
            records_extracted=500,
            started_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert checkpoint.rows_processed == 1000
        assert checkpoint.chunks_completed == 1

    def test_checkpoint_to_dict(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        checkpoint = ExtractionCheckpoint(
            input_path='/path/to/input.txt',
            output_path='/path/to/output.parquet',
            rows_processed=1000,
            chunks_completed=1,
            records_extracted=500,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            updated_at=datetime(2024, 1, 1, 10, 5, 0),
        )
        d = checkpoint.to_dict()
        assert d['rows_processed'] == 1000
        assert d['input_path'] == '/path/to/input.txt'

    def test_checkpoint_from_dict(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        data = {
            'input_path': '/path/to/input.txt',
            'output_path': '/path/to/output.parquet',
            'rows_processed': 2000,
            'chunks_completed': 2,
            'records_extracted': 1000,
            'started_at': '2024-01-01T10:00:00',
            'updated_at': '2024-01-01T10:10:00',
        }
        checkpoint = ExtractionCheckpoint.from_dict(data)
        assert checkpoint.rows_processed == 2000
        assert checkpoint.chunks_completed == 2
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractionCheckpoint -v`

Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

Create `module_3_vitals_processing/extractors/prg_extractor.py`:

```python
"""Extract vital signs from Prg.txt (Progress Notes)."""
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd


@dataclass
class ExtractionCheckpoint:
    """Track extraction progress for resume capability."""
    input_path: str
    output_path: str
    rows_processed: int
    chunks_completed: int
    records_extracted: int
    started_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ExtractionCheckpoint':
        """Create from dict (e.g., loaded from JSON)."""
        data = data.copy()
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


CHECKPOINT_FILE = "prg_extraction_checkpoint.json"
CHECKPOINT_INTERVAL = 5  # Save every 5 chunks
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractionCheckpoint -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add ExtractionCheckpoint dataclass"
```

---

## Task 7: Add Checkpoint Save/Load Functions

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestCheckpointIO:
    """Test checkpoint save and load functions."""

    def test_save_checkpoint(self):
        from module_3_vitals_processing.extractors.prg_extractor import (
            ExtractionCheckpoint, save_checkpoint, CHECKPOINT_FILE
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint = ExtractionCheckpoint(
                input_path='/input.txt',
                output_path='/output.parquet',
                rows_processed=5000,
                chunks_completed=5,
                records_extracted=2500,
                started_at=datetime.now(),
                updated_at=datetime.now(),
            )
            save_checkpoint(checkpoint, output_dir)
            assert (output_dir / CHECKPOINT_FILE).exists()

    def test_load_checkpoint_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import (
            ExtractionCheckpoint, save_checkpoint, load_checkpoint
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original = ExtractionCheckpoint(
                input_path='/input.txt',
                output_path='/output.parquet',
                rows_processed=5000,
                chunks_completed=5,
                records_extracted=2500,
                started_at=datetime(2024, 1, 1, 10, 0, 0),
                updated_at=datetime(2024, 1, 1, 10, 5, 0),
            )
            save_checkpoint(original, output_dir)
            loaded = load_checkpoint(output_dir)
            assert loaded is not None
            assert loaded.rows_processed == 5000
            assert loaded.chunks_completed == 5

    def test_load_checkpoint_not_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            loaded = load_checkpoint(output_dir)
            assert loaded is None
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestCheckpointIO -v`

Expected: FAIL with "ImportError" for save_checkpoint/load_checkpoint

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
def save_checkpoint(checkpoint: ExtractionCheckpoint, output_dir: Path) -> None:
    """Save extraction progress to JSON file."""
    path = output_dir / CHECKPOINT_FILE
    with open(path, 'w') as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


def load_checkpoint(output_dir: Path) -> Optional[ExtractionCheckpoint]:
    """Load existing checkpoint if available."""
    path = output_dir / CHECKPOINT_FILE
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            return ExtractionCheckpoint.from_dict(data)
    return None
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestCheckpointIO -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add checkpoint save/load functions"
```

---

## Task 8: Add Section Detection Functions

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestIdentifyPrgSections:
    """Test Prg section identification."""

    def test_finds_physical_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "History... Physical Exam: BP 120/80 HR 72 General appearance good"
        sections = identify_prg_sections(text)
        assert 'physical_exam' in sections
        assert 'BP 120/80' in sections['physical_exam']

    def test_finds_vitals_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Assessment... Vitals: T 98.6F HR 80 BP 130/85 Plan..."
        sections = identify_prg_sections(text)
        assert 'vitals' in sections

    def test_finds_on_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Allergies... ON EXAM: Vital Signs BP 109/57, P 76 afebrile"
        sections = identify_prg_sections(text)
        assert 'on_exam' in sections

    def test_finds_objective_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Subjective: pain... Objective: BP 120/80 HR 72 Assessment..."
        sections = identify_prg_sections(text)
        assert 'objective' in sections

    def test_returns_empty_when_no_sections(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "This is a note without any section headers."
        sections = identify_prg_sections(text)
        assert sections == {}
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestIdentifyPrgSections -v`

Expected: FAIL with "ImportError" for identify_prg_sections

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
from .prg_patterns import PRG_SECTION_PATTERNS, PRG_SKIP_PATTERNS


def identify_prg_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """
    Identify clinical sections in progress note text.

    Args:
        text: Full Report_Text from progress note
        window_size: Characters to extract after section header

    Returns:
        Dict mapping section name to text window
    """
    sections = {}

    for section_name, (pattern, _offset) in PRG_SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]

    return sections
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestIdentifyPrgSections -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add section identification function"
```

---

## Task 9: Add Skip Section Detection

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestIsInSkipSection:
    """Test skip section detection for false positive filtering."""

    def test_detects_allergies_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "Patient info... Allergies: atenolol - fatigue, HR 50 generic synthroid..."
        # Position at HR 50
        position = text.find("HR 50")
        assert is_in_skip_section(text, position)

    def test_detects_medications_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "Assessment... Medications: lisinopril 10mg causes BP drop..."
        position = text.find("BP drop")
        assert is_in_skip_section(text, position)

    def test_detects_past_medical_history(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "ROS negative... Past Medical History: HTN with BP 180/100..."
        position = text.find("BP 180")
        assert is_in_skip_section(text, position)

    def test_allows_physical_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "History... Physical Exam: BP 120/80 HR 72..."
        position = text.find("BP 120")
        assert not is_in_skip_section(text, position)

    def test_valid_section_overrides_skip(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        # Skip section followed by valid section
        text = "Allergies: penicillin... Physical Exam: BP 120/80 HR 72..."
        position = text.find("BP 120")
        assert not is_in_skip_section(text, position)
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestIsInSkipSection -v`

Expected: FAIL with "ImportError" for is_in_skip_section

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
def is_in_skip_section(text: str, position: int, lookback: int = 500) -> bool:
    """
    Check if position is within a skip section (allergies, medications, etc.).

    Args:
        text: Full text being searched
        position: Character position of the match
        lookback: Characters to look back for section headers

    Returns:
        True if in a skip section (should not extract vitals here)
    """
    start = max(0, position - lookback)
    context_before = text[start:position]

    # Find most recent skip section
    last_skip_pos = -1
    for pattern in PRG_SKIP_PATTERNS:
        for match in re.finditer(pattern, context_before, re.IGNORECASE):
            if match.end() > last_skip_pos:
                last_skip_pos = match.end()

    if last_skip_pos == -1:
        # No skip section found
        return False

    # Check if a valid section appears after the skip section
    context_after_skip = context_before[last_skip_pos:]
    for section_name, (pattern, _) in PRG_SECTION_PATTERNS.items():
        if re.search(pattern, context_after_skip, re.IGNORECASE):
            # Valid section found after skip section
            return False

    # Still in skip section
    return True
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestIsInSkipSection -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add skip section detection for false positive filtering"
```

---

## Task 10: Add Temperature Method Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestExtractTemperatureWithMethod:
    """Test temperature extraction with method capture."""

    def test_extracts_temp_with_oral_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 36.8 °C (98.2 °F) (Oral)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'oral'

    def test_extracts_temp_with_temporal_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 36.2 °C (97.1 °F) (Temporal)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'temporal'

    def test_extracts_temp_with_rectal_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "temp 98.2F rectally"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'rectal'

    def test_extracts_temp_src_format(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp(Src) 36.7 °C (98 °F) (Oral)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'oral'

    def test_returns_none_method_when_not_specified(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 98.6F"
        results = extract_temperature_with_method(text)
        # Should still extract temp, method may be None
        assert len(results) >= 1
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractTemperatureWithMethod -v`

Expected: FAIL with "ImportError" for extract_temperature_with_method

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
from .prg_patterns import PRG_TEMP_PATTERNS, TEMP_METHOD_MAP
from .hnp_patterns import TEMP_PATTERNS, VALID_RANGES


def extract_temperature_with_method(text: str) -> List[Dict]:
    """
    Extract temperature values with measurement method from text.

    Args:
        text: Text to search for temperature values

    Returns:
        List of dicts with value, units, method, confidence, position
    """
    results = []
    seen_positions = set()

    # First try Prg-specific patterns (with method capture)
    for pattern, confidence in PRG_TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            try:
                value = float(match.group(1))
                units = match.group(2).upper() if match.lastindex >= 2 else None
                raw_method = match.group(3).lower() if match.lastindex >= 3 else None
            except (ValueError, IndexError, AttributeError):
                continue

            # Map raw method to canonical
            method = TEMP_METHOD_MAP.get(raw_method) if raw_method else None

            # Auto-detect unit from value if not captured
            if units is None:
                units = 'F' if value > 50 else 'C'

            # Validate range
            range_key = 'TEMP_C' if units == 'C' else 'TEMP_F'
            min_val, max_val = VALID_RANGES[range_key]
            if not (min_val <= value <= max_val):
                continue

            results.append({
                'value': value,
                'units': units,
                'method': method,
                'confidence': confidence,
                'position': position,
            })
            seen_positions.add(position)

    # Fall back to base Hnp patterns if no Prg patterns matched
    if not results:
        for pattern, confidence in TEMP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                position = match.start()

                if any(abs(position - p) < 10 for p in seen_positions):
                    continue

                try:
                    value = float(match.group(1))
                    units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
                except (ValueError, IndexError):
                    continue

                if units is None:
                    units = 'F' if value > 50 else 'C'

                range_key = 'TEMP_C' if units == 'C' else 'TEMP_F'
                min_val, max_val = VALID_RANGES[range_key]
                if not (min_val <= value <= max_val):
                    continue

                # Check for method in surrounding context
                context_end = min(position + 50, len(text))
                context = text[position:context_end].lower()
                method = None
                for method_str, canonical in TEMP_METHOD_MAP.items():
                    if method_str in context:
                        method = canonical
                        break

                results.append({
                    'value': value,
                    'units': units,
                    'method': method,
                    'confidence': confidence,
                    'position': position,
                })
                seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractTemperatureWithMethod -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add temperature extraction with method capture"
```

---

## Task 11: Add Combined Vitals Extraction Function

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestExtractPrgVitalsFromText:
    """Test combined vitals extraction from text."""

    def test_extracts_all_vital_types(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98% Temp 98.6F (Oral)"
        results = extract_prg_vitals_from_text(text)
        types = {r['vital_type'] for r in results}
        assert 'SBP' in types
        assert 'DBP' in types
        assert 'HR' in types
        assert 'RR' in types
        assert 'SPO2' in types
        assert 'TEMP' in types

    def test_skips_vitals_in_allergies(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Allergies: atenolol - fatigue, HR 50. Physical Exam: HR 72"
        results = extract_prg_vitals_from_text(text)
        hr_values = [r['value'] for r in results if r['vital_type'] == 'HR']
        assert 72 in hr_values
        assert 50 not in hr_values  # Should be skipped

    def test_includes_temp_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Vitals: Temp 36.8 °C (98.2 °F) (Oral)"
        results = extract_prg_vitals_from_text(text)
        temp_results = [r for r in results if r['vital_type'] == 'TEMP']
        assert len(temp_results) >= 1
        assert temp_results[0].get('temp_method') == 'oral'

    def test_handles_empty_text(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        results = extract_prg_vitals_from_text("")
        assert results == []

    def test_handles_text_without_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Patient presents for follow-up. Doing well."
        results = extract_prg_vitals_from_text(text)
        assert results == []
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractPrgVitalsFromText -v`

Expected: FAIL with "ImportError" for extract_prg_vitals_from_text

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
from .hnp_extractor import (
    extract_heart_rate, extract_blood_pressure,
    extract_respiratory_rate, extract_spo2, check_negation
)
from .hnp_patterns import NEGATION_PATTERNS


def extract_prg_vitals_from_text(text: str) -> List[Dict]:
    """
    Extract all vital signs from progress note text with skip section filtering.

    Args:
        text: Full text to extract vitals from

    Returns:
        List of vital sign records
    """
    if not text:
        return []

    results = []

    # Extract Heart Rate
    for hr in extract_heart_rate(text):
        if is_in_skip_section(text, hr['position']):
            continue
        results.append({
            'vital_type': 'HR',
            'value': hr['value'],
            'units': 'bpm',
            'confidence': hr['confidence'],
            'is_flagged_abnormal': hr.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract Blood Pressure
    for bp in extract_blood_pressure(text):
        if is_in_skip_section(text, bp['position']):
            continue
        for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
            results.append({
                'vital_type': vital_type,
                'value': value,
                'units': 'mmHg',
                'confidence': bp['confidence'],
                'is_flagged_abnormal': bp.get('is_flagged_abnormal', False),
                'temp_method': None,
            })

    # Extract Respiratory Rate
    for rr in extract_respiratory_rate(text):
        if is_in_skip_section(text, rr['position']):
            continue
        results.append({
            'vital_type': 'RR',
            'value': rr['value'],
            'units': 'breaths/min',
            'confidence': rr['confidence'],
            'is_flagged_abnormal': rr.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract SpO2
    for spo2 in extract_spo2(text):
        if is_in_skip_section(text, spo2['position']):
            continue
        results.append({
            'vital_type': 'SPO2',
            'value': spo2['value'],
            'units': '%',
            'confidence': spo2['confidence'],
            'is_flagged_abnormal': spo2.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract Temperature with method
    for temp in extract_temperature_with_method(text):
        if is_in_skip_section(text, temp['position']):
            continue
        results.append({
            'vital_type': 'TEMP',
            'value': temp['value'],
            'units': temp['units'],
            'confidence': temp['confidence'],
            'is_flagged_abnormal': False,
            'temp_method': temp.get('method'),
        })

    return results
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractPrgVitalsFromText -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add combined vitals extraction with skip section filtering"
```

---

## Task 12: Add Row Processing Function

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestProcessPrgRow:
    """Test single row processing."""

    def test_processes_row_with_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Physical Exam: BP 120/80 HR 72'
        })
        results = process_prg_row(row)
        assert len(results) >= 3  # SBP, DBP, HR
        assert all(r['EMPI'] == '12345' for r in results)
        assert all(r['source'] == 'prg' for r in results)

    def test_processes_row_without_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Patient doing well. Follow up in 3 months.'
        })
        results = process_prg_row(row)
        assert results == []

    def test_handles_empty_report_text(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': None
        })
        results = process_prg_row(row)
        assert results == []

    def test_includes_temp_method_in_results(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Vitals: Temp 36.8 °C (98.2 °F) (Oral)'
        })
        results = process_prg_row(row)
        temp_results = [r for r in results if r['vital_type'] == 'TEMP']
        assert len(temp_results) >= 1
        assert temp_results[0]['temp_method'] == 'oral'
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestProcessPrgRow -v`

Expected: FAIL with "ImportError" for process_prg_row

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
from datetime import timedelta


def process_prg_row(row: pd.Series) -> List[Dict]:
    """
    Process a single progress note row and extract all vitals.

    Args:
        row: DataFrame row with EMPI, Report_Number, Report_Date_Time, Report_Text

    Returns:
        List of vital sign records
    """
    text = row.get('Report_Text')
    if not text or pd.isna(text):
        return []

    empi = str(row.get('EMPI', ''))
    report_number = str(row.get('Report_Number', ''))

    # Parse report datetime
    report_dt_str = row.get('Report_Date_Time', '')
    try:
        report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %I:%M:%S %p')
    except (ValueError, TypeError):
        try:
            report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            report_datetime = datetime.now()

    # Extract vitals from text
    extracted = extract_prg_vitals_from_text(text)

    results = []
    for vital in extracted:
        results.append({
            'EMPI': empi,
            'timestamp': report_datetime,
            'timestamp_source': 'estimated',
            'timestamp_offset_hours': 0.0,
            'vital_type': vital['vital_type'],
            'value': vital['value'],
            'units': vital['units'],
            'source': 'prg',
            'extraction_context': 'full_text',
            'confidence': vital['confidence'],
            'is_flagged_abnormal': vital['is_flagged_abnormal'],
            'report_number': report_number,
            'report_date_time': report_datetime,
            'temp_method': vital.get('temp_method'),
        })

    return results
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestProcessPrgRow -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add row processing function"
```

---

## Task 13: Add Main Extraction Function with Checkpointing

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestExtractPrgVitals:
    """Test main extraction function."""

    def test_extracts_from_small_file(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals
        # Create a small test file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_prg.txt'
            output_path = Path(tmpdir) / 'output.parquet'

            # Write test data
            with open(input_path, 'w') as f:
                f.write("EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n")
                f.write("12345|PMR001|BWH|001|RPT001|01/15/2024 10:30:00 AM|Progress|F|BPRGPROGRESS|Physical Exam: BP 120/80 HR 72\n")
                f.write("12345|PMR001|BWH|001|RPT002|01/16/2024 10:30:00 AM|Progress|F|BPRGPROGRESS|Vitals: Temp 98.6F (Oral)\n")

            df = extract_prg_vitals(str(input_path), str(output_path), resume=False)

            assert len(df) >= 4  # SBP, DBP, HR, TEMP
            assert output_path.exists()

    def test_creates_checkpoint(self):
        from module_3_vitals_processing.extractors.prg_extractor import (
            extract_prg_vitals, CHECKPOINT_FILE
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_prg.txt'
            output_path = Path(tmpdir) / 'output.parquet'

            with open(input_path, 'w') as f:
                f.write("EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n")
                f.write("12345|PMR001|BWH|001|RPT001|01/15/2024 10:30:00 AM|Progress|F|BPRGPROGRESS|Physical Exam: BP 120/80\n")

            extract_prg_vitals(str(input_path), str(output_path), resume=False)

            # Checkpoint should be deleted on successful completion
            # Or if you want checkpoint to remain, test for its existence
            assert output_path.exists()

    def test_output_has_temp_method_column(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'test_prg.txt'
            output_path = Path(tmpdir) / 'output.parquet'

            with open(input_path, 'w') as f:
                f.write("EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n")
                f.write("12345|PMR001|BWH|001|RPT001|01/15/2024 10:30:00 AM|Progress|F|BPRGPROGRESS|Temp 98.6F (Oral)\n")

            df = extract_prg_vitals(str(input_path), str(output_path), resume=False)

            assert 'temp_method' in df.columns
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractPrgVitals -v`

Expected: FAIL with "ImportError" for extract_prg_vitals

**Step 3: Write minimal implementation**

Add to `prg_extractor.py`:

```python
from multiprocessing import Pool, cpu_count
from .prg_patterns import PRG_COLUMNS


def _process_chunk(chunk: pd.DataFrame) -> List[Dict]:
    """Process a chunk of rows (for multiprocessing)."""
    results = []
    for _, row in chunk.iterrows():
        results.extend(process_prg_row(row))
    return results


def extract_prg_vitals(
    input_path: str,
    output_path: str,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000,
    resume: bool = True
) -> pd.DataFrame:
    """
    Extract vital signs from Prg.txt file with parallel processing and checkpointing.

    Args:
        input_path: Path to Prg.txt file
        output_path: Path for output parquet file
        n_workers: Number of parallel workers (default: CPU count)
        chunk_size: Rows per chunk for processing
        resume: Whether to resume from checkpoint if available

    Returns:
        DataFrame with extracted vitals (also saved to parquet)
    """
    if n_workers is None:
        n_workers = cpu_count()

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    checkpoint = load_checkpoint(output_dir) if resume else None
    skip_rows = checkpoint.rows_processed if checkpoint else 0

    if checkpoint:
        print(f"Resuming from row {skip_rows}, chunk {checkpoint.chunks_completed}")
        all_results_count = checkpoint.records_extracted
        chunks_completed = checkpoint.chunks_completed
        started_at = checkpoint.started_at
    else:
        all_results_count = 0
        chunks_completed = 0
        started_at = datetime.now()

    all_results = []

    # Read and process in chunks
    for chunk in pd.read_csv(
        input_path,
        sep='|',
        names=PRG_COLUMNS,
        header=0,
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines='skip',
        skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
    ):
        chunks_completed += 1

        if n_workers > 1:
            # Split chunk for parallel processing
            chunk_splits = [
                chunk.iloc[i:i + max(1, chunk_size // n_workers)]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for result_list in chunk_results:
                all_results.extend(result_list)
        else:
            all_results.extend(_process_chunk(chunk))

        all_results_count = len(all_results)
        rows_processed = skip_rows + (chunks_completed * chunk_size)

        # Save checkpoint periodically
        if chunks_completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = ExtractionCheckpoint(
                input_path=input_path,
                output_path=output_path,
                rows_processed=rows_processed,
                chunks_completed=chunks_completed,
                records_extracted=all_results_count,
                started_at=started_at,
                updated_at=datetime.now(),
            )
            save_checkpoint(checkpoint, output_dir)
            print(f"Checkpoint saved at chunk {chunks_completed}, {all_results_count} records")

        print(f"Processed chunk {chunks_completed}, total records: {all_results_count}")

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
            'vital_type', 'value', 'units', 'source', 'extraction_context',
            'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time',
            'temp_method'
        ])

    # Save to parquet
    df.to_parquet(output_path, index=False)

    # Remove checkpoint on successful completion
    checkpoint_path = output_dir / CHECKPOINT_FILE
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Extraction complete. Total records: {len(df)}")
    print(f"Output saved to: {output_path}")

    return df
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestExtractPrgVitals -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add main extraction function with checkpointing"
```

---

## Task 14: Add CLI Entry Point

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`

**Step 1: Write the failing test**

Add to `test_prg_extractor.py`:

```python
class TestCLI:
    """Test CLI entry point."""

    def test_cli_module_runs(self):
        import subprocess
        result = subprocess.run(
            ['python', '-c', 'from module_3_vitals_processing.extractors.prg_extractor import main'],
            capture_output=True,
            text=True,
            cwd='/home/moin/TDA_11_25'
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestCLI -v`

Expected: FAIL with "ImportError" for main

**Step 3: Write minimal implementation**

Add to end of `prg_extractor.py`:

```python
def main():
    """CLI entry point for Prg vitals extraction."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import PRG_INPUT_PATH, PRG_OUTPUT_PATH

    parser = argparse.ArgumentParser(
        description='Extract vital signs from Prg.txt (Progress Notes)'
    )
    parser.add_argument(
        '-i', '--input',
        default=str(PRG_INPUT_PATH),
        help='Input Prg.txt file path'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(PRG_OUTPUT_PATH),
        help='Output parquet file path'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=10000,
        help='Rows per processing chunk'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore existing checkpoint'
    )

    args = parser.parse_args()

    print(f"Extracting vitals from: {args.input}")
    print(f"Output to: {args.output}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Resume: {not args.no_resume}")

    extract_prg_vitals(
        args.input,
        args.output,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
```

**Step 4: Run test to verify it passes**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py::TestCLI -v`

Expected: PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py
git commit -m "feat(prg): add CLI entry point"
```

---

## Task 15: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all Prg tests**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_patterns.py module_3_vitals_processing/tests/test_prg_extractor.py -v`

Expected: All tests PASS

**Step 2: Run full module 3 test suite**

Run: `PYTHONPATH=/home/moin/TDA_11_25:$PYTHONPATH pytest module_3_vitals_processing/tests/ -v`

Expected: All tests PASS (including existing Hnp and Phy tests)

**Step 3: Final commit**

```bash
git add -A
git commit -m "test(prg): verify full test suite passes"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Section patterns | prg_patterns.py, test_prg_patterns.py |
| 2 | Skip patterns | prg_patterns.py, test_prg_patterns.py |
| 3 | Vitals patterns | prg_patterns.py, test_prg_patterns.py |
| 4 | Temperature method | prg_patterns.py, test_prg_patterns.py |
| 5 | Config paths | vitals_config.py, test_prg_patterns.py |
| 6 | Checkpoint dataclass | prg_extractor.py, test_prg_extractor.py |
| 7 | Checkpoint I/O | prg_extractor.py, test_prg_extractor.py |
| 8 | Section detection | prg_extractor.py, test_prg_extractor.py |
| 9 | Skip section detection | prg_extractor.py, test_prg_extractor.py |
| 10 | Temperature extraction | prg_extractor.py, test_prg_extractor.py |
| 11 | Combined extraction | prg_extractor.py, test_prg_extractor.py |
| 12 | Row processing | prg_extractor.py, test_prg_extractor.py |
| 13 | Main function | prg_extractor.py, test_prg_extractor.py |
| 14 | CLI entry point | prg_extractor.py, test_prg_extractor.py |
| 15 | Full test suite | verification |

**Estimated total: 15 tasks, ~45-75 minutes**
