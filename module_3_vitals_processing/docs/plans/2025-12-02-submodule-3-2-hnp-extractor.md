# Submodule 3.2: Hnp NLP Extractor - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract vital signs from 136,950 H&P notes using hybrid regex extraction with context-aware negation detection, timestamp estimation, and pattern-based confidence scoring.

**Architecture:** Hybrid approach extracts from clinical sections (Exam, Vitals, ED Course) first, falls back to full-text with stricter patterns. Parallel multiprocessing for performance. All vitals per note preserved with extraction_context tags.

**Tech Stack:** Python 3, pandas, pyarrow, multiprocessing, re

**Design Doc:** `docs/plans/2025-12-02-submodule-3-2-hnp-extractor-design.md`

---

## Task 1: Create Patterns Module

**Files:**
- Create: `module_3_vitals_processing/extractors/hnp_patterns.py`
- Test: `module_3_vitals_processing/tests/test_hnp_patterns.py`

**Step 1: Write the test file with pattern import test**

```python
# module_3_vitals_processing/tests/test_hnp_patterns.py
"""Tests for hnp_patterns module."""
import pytest


class TestPatternsExist:
    """Test that all required patterns are defined."""

    def test_vital_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import (
            HR_PATTERNS, BP_PATTERNS, RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS
        )
        assert len(HR_PATTERNS) > 0
        assert len(BP_PATTERNS) > 0
        assert len(RR_PATTERNS) > 0
        assert len(SPO2_PATTERNS) > 0
        assert len(TEMP_PATTERNS) > 0

    def test_section_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import SECTION_PATTERNS
        assert 'exam' in SECTION_PATTERNS
        assert 'vitals' in SECTION_PATTERNS
        assert 'ed_course' in SECTION_PATTERNS

    def test_negation_patterns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import NEGATION_PATTERNS
        assert len(NEGATION_PATTERNS) > 0

    def test_hnp_columns_defined(self):
        from module_3_vitals_processing.extractors.hnp_patterns import HNP_COLUMNS
        assert 'EMPI' in HNP_COLUMNS
        assert 'Report_Text' in HNP_COLUMNS
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_patterns.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write the patterns module**

```python
# module_3_vitals_processing/extractors/hnp_patterns.py
"""Regex patterns and constants for Hnp.txt extraction."""

# Hnp.txt columns (pipe-delimited)
HNP_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Section patterns: (regex, timestamp_offset_hours)
SECTION_PATTERNS = {
    'exam': (r'(?:Physical\s+)?Exam(?:ination)?(?:\s+ON\s+ADMISSION)?[:\s]', -1),
    'vitals': (r'Vitals?(?:\s+Signs?)?[:\s]|Vital\s+signs', -1),
    'ed_course': (r'ED\s+Course[:\s]|Emergency\s+Department|Triage\s+Vitals', -6),
    'current': (r'Current[:\s]|Last\s+vitals', 0),
}

# Heart Rate patterns: (regex, confidence)
HR_PATTERNS = [
    (r'Heart\s*Rate\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 1.0),
    (r'HR\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 0.95),
    (r'(?:Pulse|P)\s+:?\s*\(?\!?\)?\s*(\d{2,3})', 0.9),
    (r'\[\d{2,3}-\d{2,3}\]\s*(\d{2,3})', 0.85),
]

# Blood Pressure patterns: (regex, confidence) - captures (SBP, DBP)
BP_PATTERNS = [
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})/(\d{2,3})', 1.0),
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
    (r'(\d{2,3})/(\d{2,3})\s*(?:mmHg)', 0.8),
    (r'(\d{2,3})/(\d{2,3})', 0.7),
]

# Respiratory Rate patterns: (regex, confidence)
RR_PATTERNS = [
    (r'Respiratory\s*Rate\s*:?\s*(\d{1,2})', 1.0),
    (r'(?:RR|Resp|TRR)\s*:?\s*(\d{1,2})', 0.9),
    (r'\[\d{1,2}-\d{1,2}\]\s*(\d{1,2})', 0.85),
]

# SpO2 patterns: (regex, confidence)
SPO2_PATTERNS = [
    (r'(?:SpO2|SaO2|O2\s*Sat(?:uration)?)\s*:?\s*>?(\d{2,3})\s*%?', 1.0),
    (r'(\d{2,3})\s*%\s*(?:on|RA|room)', 0.8),
]

# Temperature patterns: (regex, confidence) - captures (value, unit)
TEMP_PATTERNS = [
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 1.0),
    (r'Tcurrent\s+(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.9),
    (r'T\s+(\d{2,3}\.?\d?)\s*[°?]\s*([CF])', 0.9),
    (r'(\d{2,3}\.\d)\s*[°?]\s*([CF])', 0.8),
]

# Negation patterns
NEGATION_PATTERNS = [
    r'no\s+vitals',
    r'not\s+obtained',
    r'unable\s+to\s+(?:obtain|measure|assess)',
    r'refused',
    r'not\s+measured',
    r'not\s+documented',
    r'vitals?\s+unavailable',
    r'There\s+were\s+no\s+vitals',
]

# Timestamp patterns for explicit extraction
TIMESTAMP_PATTERNS = [
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)',
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{4})',
]

# Value ranges for validation
VALID_RANGES = {
    'HR': (20, 300),
    'SBP': (40, 350),
    'DBP': (20, 250),
    'RR': (4, 80),
    'SPO2': (40, 100),
    'TEMP_C': (25, 45),
    'TEMP_F': (77, 113),
}

# Default timestamp offsets by section (hours)
DEFAULT_TIMESTAMP_OFFSET = -2
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_patterns.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_patterns.py module_3_vitals_processing/tests/test_hnp_patterns.py
git commit -m "feat(module3): add hnp extractor patterns module"
```

---

## Task 2: Section Identification Function

**Files:**
- Create: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Test: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for identify_sections**

```python
# module_3_vitals_processing/tests/test_hnp_extractor.py
"""Tests for hnp_extractor module."""
import pytest


class TestIdentifySections:
    """Test section identification in clinical notes."""

    def test_finds_exam_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "History of illness... Physical Exam: BP 120/80 HR 72 General appearance good"
        sections = identify_sections(text)
        assert 'exam' in sections
        assert 'BP 120/80' in sections['exam']

    def test_finds_vitals_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "Patient presents with... Vitals: T 98.6F HR 80 BP 130/85 Assessment..."
        sections = identify_sections(text)
        assert 'vitals' in sections
        assert 'HR 80' in sections['vitals']

    def test_finds_ed_course_section(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "Chief complaint... ED Course: BP 110/70 given fluids Admitted to medicine"
        sections = identify_sections(text)
        assert 'ed_course' in sections
        assert 'BP 110/70' in sections['ed_course']

    def test_finds_multiple_sections(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "ED Course: BP 100/60 HR 90... Physical Exam: BP 120/80 HR 75 well appearing"
        sections = identify_sections(text)
        assert 'ed_course' in sections
        assert 'exam' in sections
        assert 'BP 100/60' in sections['ed_course']
        assert 'BP 120/80' in sections['exam']

    def test_returns_empty_when_no_sections(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "This note has no standard vitals sections at all."
        sections = identify_sections(text)
        assert sections == {}

    def test_handles_exam_on_admission_variant(self):
        from module_3_vitals_processing.extractors.hnp_extractor import identify_sections
        text = "History... EXAM ON ADMISSION Vitals: HR 88 BP 140/90 Gen: alert"
        sections = identify_sections(text)
        assert 'exam' in sections
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestIdentifySections -v`
Expected: FAIL with "ImportError"

**Step 3: Write identify_sections function**

```python
# module_3_vitals_processing/extractors/hnp_extractor.py
"""Extract vital signs from Hnp.txt (H&P notes)."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .hnp_patterns import SECTION_PATTERNS


def identify_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """
    Identify clinical sections in note text.

    Args:
        text: Full Report_Text from H&P note
        window_size: Characters to extract after section header

    Returns:
        Dict mapping section name to text window
    """
    sections = {}

    for section_name, (pattern, _offset) in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]

    return sections
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestIdentifySections -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add section identification for hnp extractor"
```

---

## Task 3: Negation Detection Function

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for check_negation**

```python
# Add to test_hnp_extractor.py

class TestCheckNegation:
    """Test negation detection in context window."""

    def test_detects_not_obtained(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Blood pressure not obtained due to patient condition"
        assert check_negation(text, position=15, window=50) is True

    def test_detects_unable_to_measure(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Vitals: HR 80, BP unable to measure, RR 18"
        # Position at "BP"
        assert check_negation(text, position=14, window=50) is True

    def test_detects_refused(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Patient refused vital signs assessment"
        assert check_negation(text, position=20, window=50) is True

    def test_detects_no_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "There were no vitals filed for this visit"
        assert check_negation(text, position=15, window=50) is True

    def test_returns_false_for_normal_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        text = "Vitals: BP 120/80 HR 72 RR 16 SpO2 98%"
        assert check_negation(text, position=10, window=50) is False

    def test_respects_window_size(self):
        from module_3_vitals_processing.extractors.hnp_extractor import check_negation
        # Negation far from position
        text = "Unable to obtain vitals earlier. Later: BP 120/80 normal values"
        # Position at "BP 120/80" - negation is far away
        assert check_negation(text, position=40, window=20) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestCheckNegation -v`
Expected: FAIL with "ImportError" or "AttributeError"

**Step 3: Write check_negation function**

```python
# Add to hnp_extractor.py after identify_sections

from .hnp_patterns import SECTION_PATTERNS, NEGATION_PATTERNS


def check_negation(text: str, position: int, window: int = 50) -> bool:
    """
    Check for negation phrases in context window around match position.

    Args:
        text: Full text being searched
        position: Character position of the match
        window: Characters to check before and after position

    Returns:
        True if negation phrase found in window
    """
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end].lower()

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True

    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestCheckNegation -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add negation detection for hnp extractor"
```

---

## Task 4: Heart Rate Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_heart_rate**

```python
# Add to test_hnp_extractor.py

class TestExtractHeartRate:
    """Test heart rate extraction patterns."""

    def test_extracts_heart_rate_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Vitals: Heart Rate: 88 BP 120/80"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 88
        assert results[0]['confidence'] == 1.0

    def test_extracts_hr_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "HR 72 BP 130/85"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 72
        assert results[0]['confidence'] == 0.95

    def test_extracts_pulse_p(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "37.2 °C P 79 BP 149/65"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 79

    def test_extracts_pulse_full(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Pulse 86 regular"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 86

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "P (!) 117 BP 170/87"
        results = extract_heart_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 117
        assert results[0]['is_flagged_abnormal'] is True

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Heart Rate: [62-72] 72"
        results = extract_heart_rate(text)
        # Should get the current value 72
        values = [r['value'] for r in results]
        assert 72 in values

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "HR 500"  # Invalid
        results = extract_heart_rate(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "Heart rate not obtained"
        results = extract_heart_rate(text)
        assert len(results) == 0

    def test_multiple_extractions(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_heart_rate
        text = "ED: HR 90... Exam: Heart Rate: 75"
        results = extract_heart_rate(text)
        values = [r['value'] for r in results]
        assert 90 in values
        assert 75 in values
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractHeartRate -v`
Expected: FAIL

**Step 3: Write extract_heart_rate function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, VALID_RANGES
)


def extract_heart_rate(text: str) -> List[Dict]:
    """
    Extract heart rate values from text.

    Args:
        text: Text to search for heart rate values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in HR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Skip if we already found a value at similar position
            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            # Check for negation
            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # Validate range
            min_val, max_val = VALID_RANGES['HR']
            if not (min_val <= value <= max_val):
                continue

            # Check for abnormal flag (!)
            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context or '(!)' in match.group(0)

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractHeartRate -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add heart rate extraction for hnp extractor"
```

---

## Task 5: Blood Pressure Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_blood_pressure**

```python
# Add to test_hnp_extractor.py

class TestExtractBloodPressure:
    """Test blood pressure extraction patterns."""

    def test_extracts_bp_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "Blood pressure 130/85 measured sitting"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 130
        assert results[0]['dbp'] == 85
        assert results[0]['confidence'] == 1.0

    def test_extracts_bp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP: 120/80 HR 72"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 120
        assert results[0]['dbp'] == 80

    def test_extracts_bp_with_mmhg(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "measured 145/92 mmHg in left arm"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 145

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP (!) 180/110 HR 88"
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['is_flagged_abnormal'] is True

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP: (115-154)/(59-69) 145/67"
        results = extract_blood_pressure(text)
        # Should extract 145/67
        assert any(r['sbp'] == 145 and r['dbp'] == 67 for r in results)

    def test_swaps_if_sbp_less_than_dbp(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP 70/120"  # Swapped values
        results = extract_blood_pressure(text)
        assert len(results) >= 1
        assert results[0]['sbp'] == 120  # Should be swapped
        assert results[0]['dbp'] == 70

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP 400/300"  # Invalid
        results = extract_blood_pressure(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "BP not obtained due to combative patient"
        results = extract_blood_pressure(text)
        assert len(results) == 0

    def test_multiple_extractions(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_blood_pressure
        text = "Triage: BP 100/60... Exam: BP 120/75"
        results = extract_blood_pressure(text)
        sbps = [r['sbp'] for r in results]
        assert 100 in sbps
        assert 120 in sbps
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractBloodPressure -v`
Expected: FAIL

**Step 3: Write extract_blood_pressure function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS, VALID_RANGES
)


def extract_blood_pressure(text: str) -> List[Dict]:
    """
    Extract blood pressure values from text.

    Args:
        text: Text to search for BP values

    Returns:
        List of dicts with sbp, dbp, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in BP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Skip if we already found a value at similar position
            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            # Check for negation
            if check_negation(text, position):
                continue

            try:
                sbp = float(match.group(1))
                dbp = float(match.group(2))
            except (ValueError, IndexError):
                continue

            # Swap if SBP < DBP (likely transposed)
            if sbp < dbp:
                sbp, dbp = dbp, sbp

            # Validate ranges
            sbp_min, sbp_max = VALID_RANGES['SBP']
            dbp_min, dbp_max = VALID_RANGES['DBP']
            if not (sbp_min <= sbp <= sbp_max and dbp_min <= dbp <= dbp_max):
                continue

            # Check for abnormal flag (!)
            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context or '(!)' in match.group(0)

            results.append({
                'sbp': sbp,
                'dbp': dbp,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractBloodPressure -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add blood pressure extraction for hnp extractor"
```

---

## Task 6: Respiratory Rate Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_respiratory_rate**

```python
# Add to test_hnp_extractor.py

class TestExtractRespiratoryRate:
    """Test respiratory rate extraction patterns."""

    def test_extracts_respiratory_rate_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Respiratory Rate: 16 SpO2 98%"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 16
        assert results[0]['confidence'] == 1.0

    def test_extracts_rr_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "HR 72 RR 18 SpO2 97%"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 18

    def test_extracts_resp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Resp 20 unlabored"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 20

    def test_extracts_trr_triage(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "TRR 22 on arrival"
        results = extract_respiratory_rate(text)
        assert len(results) >= 1
        assert results[0]['value'] == 22

    def test_extracts_range_then_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "Respiratory Rate: [14-22] 18"
        results = extract_respiratory_rate(text)
        values = [r['value'] for r in results]
        assert 18 in values

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "RR 100"  # Invalid
        results = extract_respiratory_rate(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_respiratory_rate
        text = "RR not measured"
        results = extract_respiratory_rate(text)
        assert len(results) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractRespiratoryRate -v`
Expected: FAIL

**Step 3: Write extract_respiratory_rate function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, VALID_RANGES
)


def extract_respiratory_rate(text: str) -> List[Dict]:
    """
    Extract respiratory rate values from text.

    Args:
        text: Text to search for RR values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in RR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['RR']
            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractRespiratoryRate -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add respiratory rate extraction for hnp extractor"
```

---

## Task 7: SpO2 Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_spo2**

```python
# Add to test_hnp_extractor.py

class TestExtractSpO2:
    """Test SpO2 extraction patterns."""

    def test_extracts_spo2_full_label(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2: 98% on room air"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98
        assert results[0]['confidence'] == 1.0

    def test_extracts_spo2_no_colon(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "HR 72 SpO2 97 % RR 16"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 97

    def test_extracts_sao2(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SaO2 >99% on 2L NC"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 99

    def test_extracts_o2_sat(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "O2 Sat: 95% on RA"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 95

    def test_extracts_o2_saturation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "O2 Saturation 94%"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 94

    def test_extracts_percentage_with_context(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "satting 92% on RA"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['value'] == 92

    def test_extracts_abnormal_flagged(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 (!) 89 % on 3L"
        results = extract_spo2(text)
        assert len(results) >= 1
        assert results[0]['is_flagged_abnormal'] is True

    def test_rejects_invalid_range(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 150%"  # Invalid
        results = extract_spo2(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_spo2
        text = "SpO2 unable to measure"
        results = extract_spo2(text)
        assert len(results) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractSpO2 -v`
Expected: FAIL

**Step 3: Write extract_spo2 function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, SPO2_PATTERNS, VALID_RANGES
)


def extract_spo2(text: str) -> List[Dict]:
    """
    Extract SpO2 values from text.

    Args:
        text: Text to search for SpO2 values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in SPO2_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['SPO2']
            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractSpO2 -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add SpO2 extraction for hnp extractor"
```

---

## Task 8: Temperature Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_temperature**

```python
# Add to test_hnp_extractor.py

class TestExtractTemperature:
    """Test temperature extraction patterns."""

    def test_extracts_temperature_full_label_celsius(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temperature: 37.2 °C (99 °F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.2
        assert results[0]['units'] == 'C'

    def test_extracts_temp_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 36.8 ?C HR 72"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 36.8
        assert results[0]['units'] == 'C'

    def test_extracts_t_abbreviation(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "T 98.6F P 80 BP 120/80"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98.6
        assert results[0]['units'] == 'F'

    def test_extracts_tcurrent(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Tcurrent 37.3 ?C (99.2 ?F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.3

    def test_extracts_encoding_question_mark(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "36.5 ?C (97.7 ?F)"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 36.5
        assert results[0]['units'] == 'C'

    def test_autodetects_fahrenheit_from_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        # Value > 50 with no unit should be detected as Fahrenheit
        text = "Temp: 98.6"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 98.6
        assert results[0]['units'] == 'F'

    def test_autodetects_celsius_from_value(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        # Value < 50 with no unit should be detected as Celsius
        text = "Temp: 37.0"
        results = extract_temperature(text)
        assert len(results) >= 1
        assert results[0]['value'] == 37.0
        assert results[0]['units'] == 'C'

    def test_rejects_invalid_celsius(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 50 C"  # Invalid Celsius
        results = extract_temperature(text)
        assert len(results) == 0

    def test_rejects_invalid_fahrenheit(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temp 150 F"  # Invalid Fahrenheit
        results = extract_temperature(text)
        assert len(results) == 0

    def test_skips_negated(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_temperature
        text = "Temperature not obtained"
        results = extract_temperature(text)
        assert len(results) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractTemperature -v`
Expected: FAIL

**Step 3: Write extract_temperature function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS, VALID_RANGES
)


def extract_temperature(text: str) -> List[Dict]:
    """
    Extract temperature values from text.

    Args:
        text: Text to search for temperature values

    Returns:
        List of dicts with value, units, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
                # Get unit from capture group if available
                units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
            except (ValueError, IndexError):
                continue

            # Auto-detect unit from value if not captured
            if units is None:
                if value > 50:
                    units = 'F'
                else:
                    units = 'C'

            # Validate range based on unit
            if units == 'C':
                min_val, max_val = VALID_RANGES['TEMP_C']
            else:
                min_val, max_val = VALID_RANGES['TEMP_F']

            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'units': units,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractTemperature -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add temperature extraction for hnp extractor"
```

---

## Task 9: Timestamp Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_timestamp**

```python
# Add to test_hnp_extractor.py
from datetime import datetime


class TestExtractTimestamp:
    """Test timestamp extraction and estimation."""

    def test_extracts_explicit_timestamp_12h(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "Vitals at 10/23/2021 7:34 PM: HR 80 BP 120/80"
        report_dt = datetime(2021, 10, 24, 9, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'explicit'
        assert ts.year == 2021
        assert ts.month == 10
        assert ts.day == 23

    def test_extracts_explicit_timestamp_military(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "03/08/22 1500 BP: 186/87"
        report_dt = datetime(2022, 3, 8, 18, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'explicit'
        assert ts.hour == 15

    def test_estimates_ed_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "BP 100/60 HR 90"  # No explicit timestamp
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'ed_course', report_dt)
        assert source == 'estimated'
        assert offset == -6
        assert ts.hour == 6  # 12 - 6 = 6

    def test_estimates_exam_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "BP 120/80 HR 72"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'exam', report_dt)
        assert source == 'estimated'
        assert offset == -1
        assert ts.hour == 11

    def test_estimates_vitals_section_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "HR 75 BP 118/72"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'vitals', report_dt)
        assert source == 'estimated'
        assert offset == -1

    def test_estimates_unknown_section_default_offset(self):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_timestamp
        text = "HR 80 BP 125/80"
        report_dt = datetime(2021, 10, 24, 12, 0, 0)
        ts, source, offset = extract_timestamp(text, 'other', report_dt)
        assert source == 'estimated'
        assert offset == -2  # Default offset
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractTimestamp -v`
Expected: FAIL

**Step 3: Write extract_timestamp function**

```python
# Add to hnp_extractor.py

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS, TIMESTAMP_PATTERNS,
    VALID_RANGES, DEFAULT_TIMESTAMP_OFFSET
)


def extract_timestamp(
    text: str,
    section: str,
    report_datetime: datetime
) -> Tuple[datetime, str, float]:
    """
    Extract explicit timestamp or estimate from section context.

    Args:
        text: Text window to search for timestamp
        section: Section name (ed_course, exam, vitals, etc.)
        report_datetime: Report_Date_Time from the note

    Returns:
        Tuple of (timestamp, source, offset_hours)
        source is 'explicit' or 'estimated'
    """
    # Try explicit timestamp extraction
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)

                # Parse date
                for fmt in ['%m/%d/%Y', '%m/%d/%y']:
                    try:
                        date_part = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                # Parse time
                time_str = time_str.strip()
                for fmt in ['%I:%M %p', '%I:%M:%S %p', '%H:%M', '%H%M']:
                    try:
                        time_part = datetime.strptime(time_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                timestamp = date_part.replace(
                    hour=time_part.hour,
                    minute=time_part.minute,
                    second=getattr(time_part, 'second', 0)
                )
                return timestamp, 'explicit', 0.0

            except (ValueError, AttributeError):
                continue

    # Fall back to estimation based on section
    if section in SECTION_PATTERNS:
        _, offset = SECTION_PATTERNS[section]
    else:
        offset = DEFAULT_TIMESTAMP_OFFSET

    estimated_ts = report_datetime + timedelta(hours=offset)
    return estimated_ts, 'estimated', float(offset)
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractTimestamp -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add timestamp extraction for hnp extractor"
```

---

## Task 10: Row Processor

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for process_hnp_row**

```python
# Add to test_hnp_extractor.py
import pandas as pd


class TestProcessHnpRow:
    """Test full row processing."""

    def test_processes_row_with_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98% Temp 37.0 C'
        })
        results = process_hnp_row(row)
        assert len(results) >= 5  # BP produces SBP+DBP, plus HR, RR, SPO2, TEMP

        # Check all vitals extracted
        vital_types = [r['vital_type'] for r in results]
        assert 'HR' in vital_types
        assert 'SBP' in vital_types
        assert 'DBP' in vital_types
        assert 'RR' in vital_types
        assert 'SPO2' in vital_types
        assert 'TEMP' in vital_types

    def test_tags_extraction_context(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'ED Course: BP 100/60... Physical Exam: BP 120/80'
        })
        results = process_hnp_row(row)
        contexts = [r['extraction_context'] for r in results]
        assert 'ed_course' in contexts
        assert 'exam' in contexts

    def test_preserves_empi_and_report_number(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '99999',
            'Report_Number': 'RPT555',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Vitals: HR 80'
        })
        results = process_hnp_row(row)
        assert all(r['EMPI'] == '99999' for r in results)
        assert all(r['report_number'] == 'RPT555' for r in results)

    def test_returns_empty_for_no_vitals(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': 'Patient presents with headache. No vitals documented.'
        })
        results = process_hnp_row(row)
        assert len(results) == 0

    def test_handles_missing_report_text(self):
        from module_3_vitals_processing.extractors.hnp_extractor import process_hnp_row
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '10/23/2021 9:00:00 PM',
            'Report_Text': None
        })
        results = process_hnp_row(row)
        assert results == []
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestProcessHnpRow -v`
Expected: FAIL

**Step 3: Write process_hnp_row function**

```python
# Add to hnp_extractor.py
import pandas as pd


def process_hnp_row(row: pd.Series) -> List[Dict]:
    """
    Process a single H&P note row and extract all vitals.

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

    results = []

    # Identify sections
    sections = identify_sections(text)

    # If no sections found, use full text as 'other'
    if not sections:
        sections = {'other': text}

    # Process each section
    for section_name, section_text in sections.items():
        # Get timestamp for this section
        timestamp, ts_source, ts_offset = extract_timestamp(
            section_text, section_name, report_datetime
        )

        # Extract each vital type
        # Heart Rate
        for hr in extract_heart_rate(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'HR',
                'value': hr['value'],
                'units': 'bpm',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': hr['confidence'],
                'is_flagged_abnormal': hr['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # Blood Pressure (creates SBP and DBP records)
        for bp in extract_blood_pressure(section_text):
            for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
                results.append({
                    'EMPI': empi,
                    'timestamp': timestamp,
                    'timestamp_source': ts_source,
                    'timestamp_offset_hours': ts_offset,
                    'vital_type': vital_type,
                    'value': value,
                    'units': 'mmHg',
                    'source': 'hnp',
                    'extraction_context': section_name,
                    'confidence': bp['confidence'],
                    'is_flagged_abnormal': bp['is_flagged_abnormal'],
                    'report_number': report_number,
                    'report_date_time': report_datetime,
                })

        # Respiratory Rate
        for rr in extract_respiratory_rate(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'RR',
                'value': rr['value'],
                'units': 'breaths/min',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': rr['confidence'],
                'is_flagged_abnormal': rr['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # SpO2
        for spo2 in extract_spo2(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'SPO2',
                'value': spo2['value'],
                'units': '%',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': spo2['confidence'],
                'is_flagged_abnormal': spo2['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # Temperature
        for temp in extract_temperature(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'TEMP',
                'value': temp['value'],
                'units': temp['units'],
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': temp['confidence'],
                'is_flagged_abnormal': temp['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

    return results
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestProcessHnpRow -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add row processor for hnp extractor"
```

---

## Task 11: Main Extraction Function with Parallel Processing

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write tests for extract_hnp_vitals**

```python
# Add to test_hnp_extractor.py
import tempfile
import os


class TestExtractHnpVitals:
    """Test main extraction function."""

    def test_extracts_from_small_file(self, tmp_path):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_hnp_vitals

        # Create test input file
        input_file = tmp_path / "test_hnp.txt"
        input_file.write_text(
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n"
            "12345|PMR1|BWH|MRN1|RPT001|10/23/2021 9:00:00 PM|H&P|F|BHPHP|Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98%\n"
            "67890|PMR2|BWH|MRN2|RPT002|10/24/2021 10:00:00 AM|H&P|F|BHPHP|Vitals: T 37.2 C HR 88 BP 130/85\n"
        )

        output_file = tmp_path / "output.parquet"

        df = extract_hnp_vitals(str(input_file), str(output_file), n_workers=1)

        assert os.path.exists(output_file)
        assert len(df) > 0
        assert 'EMPI' in df.columns
        assert 'vital_type' in df.columns
        assert 'value' in df.columns

    def test_handles_empty_file(self, tmp_path):
        from module_3_vitals_processing.extractors.hnp_extractor import extract_hnp_vitals

        input_file = tmp_path / "empty_hnp.txt"
        input_file.write_text(
            "EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text\n"
        )

        output_file = tmp_path / "output.parquet"

        df = extract_hnp_vitals(str(input_file), str(output_file), n_workers=1)

        assert len(df) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractHnpVitals -v`
Expected: FAIL

**Step 3: Write extract_hnp_vitals function**

```python
# Add to hnp_extractor.py
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .hnp_patterns import HNP_COLUMNS


def _process_chunk(chunk: pd.DataFrame) -> List[Dict]:
    """Process a chunk of rows (for multiprocessing)."""
    results = []
    for _, row in chunk.iterrows():
        results.extend(process_hnp_row(row))
    return results


def extract_hnp_vitals(
    input_path: str,
    output_path: str,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Extract vital signs from Hnp.txt file with parallel processing.

    Args:
        input_path: Path to Hnp.txt file
        output_path: Path for output parquet file
        n_workers: Number of parallel workers (default: CPU count)
        chunk_size: Rows per chunk for processing

    Returns:
        DataFrame with extracted vitals (also saved to parquet)
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Read and process in chunks
    chunks_processed = 0
    for chunk in pd.read_csv(
        input_path,
        sep='|',
        names=HNP_COLUMNS,
        header=0,
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines='skip'
    ):
        chunks_processed += 1

        if n_workers > 1:
            # Split chunk for parallel processing
            chunk_splits = [
                chunk.iloc[i:i + chunk_size // n_workers]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for result_list in chunk_results:
                all_results.extend(result_list)
        else:
            # Single-threaded processing
            all_results.extend(_process_chunk(chunk))

        print(f"Processed chunk {chunks_processed}, total records: {len(all_results)}")

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
            'vital_type', 'value', 'units', 'source', 'extraction_context',
            'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time'
        ])

    # Save to parquet
    df.to_parquet(output_path, index=False)

    print(f"Extraction complete. Total records: {len(df)}")
    print(f"Output saved to: {output_path}")

    return df
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestExtractHnpVitals -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "feat(module3): add main extraction function with parallel processing"
```

---

## Task 12: CLI Entry Point

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`

**Step 1: Add CLI code (no test needed for CLI)**

```python
# Add to end of hnp_extractor.py

if __name__ == '__main__':
    import argparse
    from .hnp_patterns import HNP_COLUMNS
    from ..config.vitals_config import DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description='Extract vital signs from Hnp.txt (H&P notes)'
    )
    parser.add_argument(
        '-i', '--input',
        default=str(DATA_DIR / 'Hnp.txt'),
        help='Input Hnp.txt file path'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'),
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

    args = parser.parse_args()

    print(f"Extracting vitals from: {args.input}")
    print(f"Output to: {args.output}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Chunk size: {args.chunk_size}")

    extract_hnp_vitals(
        args.input,
        args.output,
        n_workers=args.workers,
        chunk_size=args.chunk_size
    )
```

**Step 2: Run all tests**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py module_3_vitals_processing/tests/test_hnp_patterns.py -v`
Expected: All PASS (~55 tests)

**Step 3: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py
git commit -m "feat(module3): add CLI entry point for hnp extractor"
```

---

## Task 13: Integration Test with Real Data

**Files:**
- Modify: `module_3_vitals_processing/tests/test_hnp_extractor.py`

**Step 1: Write integration test**

```python
# Add to test_hnp_extractor.py
import os


class TestIntegration:
    """Integration test with real data sample."""

    def test_real_data_sample(self, tmp_path):
        """Test extraction on first 1000 rows of real Hnp.txt."""
        from module_3_vitals_processing.extractors.hnp_extractor import extract_hnp_vitals

        real_file = "/home/moin/TDA_11_25/Data/Hnp.txt"
        if not os.path.exists(real_file):
            pytest.skip("Real data file not available")

        # Create sample file with first 1000 rows
        sample_file = tmp_path / "sample_hnp.txt"
        with open(real_file, 'r') as f_in:
            with open(sample_file, 'w') as f_out:
                for i, line in enumerate(f_in):
                    f_out.write(line)
                    if i >= 1000:
                        break

        output_file = tmp_path / "sample_output.parquet"

        df = extract_hnp_vitals(str(sample_file), str(output_file), n_workers=1)

        # Verify extraction produced results
        assert len(df) > 0, "Should extract some vitals from real data"

        # Check vital types present
        vital_types = df['vital_type'].unique()
        print(f"Extracted vital types: {vital_types}")
        print(f"Total records: {len(df)}")
        print(f"Records by type:\n{df['vital_type'].value_counts()}")

        # At least some vital types should be present
        assert len(vital_types) >= 3, "Should extract multiple vital types"

        # Check data quality
        assert df['value'].notna().all(), "All values should be non-null"
        assert df['confidence'].between(0, 1).all(), "Confidence should be 0-1"
```

**Step 2: Run integration test**

Run: `pytest module_3_vitals_processing/tests/test_hnp_extractor.py::TestIntegration -v -s`
Expected: PASS with output showing extracted vital types

**Step 3: Commit**

```bash
git add module_3_vitals_processing/tests/test_hnp_extractor.py
git commit -m "test(module3): add integration test for hnp extractor"
```

---

## Task 14: Update Config and Final Verification

**Files:**
- Modify: `module_3_vitals_processing/config/vitals_config.py`

**Step 1: Update config with Hnp paths**

```python
# Add to vitals_config.py after existing content

# Hnp.txt columns
HNP_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Default output paths for Hnp extractor
HNP_INPUT_PATH = DATA_DIR / 'Hnp.txt'
HNP_OUTPUT_PATH = OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'
```

**Step 2: Run full test suite**

Run: `pytest module_3_vitals_processing/tests/ -v`
Expected: All tests PASS

**Step 3: Final commit**

```bash
git add module_3_vitals_processing/config/vitals_config.py
git commit -m "feat(module3): update config with hnp extractor paths"
```

---

## Summary

**Total Tasks:** 14
**Estimated Tests:** ~57
**Estimated Time:** 2-3 hours

**Files Created/Modified:**
- `extractors/hnp_patterns.py` (NEW)
- `extractors/hnp_extractor.py` (NEW)
- `tests/test_hnp_patterns.py` (NEW)
- `tests/test_hnp_extractor.py` (NEW)
- `config/vitals_config.py` (MODIFIED)

**CLI Usage After Implementation:**
```bash
# Default paths
python3 -m module_3_vitals_processing.extractors.hnp_extractor

# Custom paths
python3 -m module_3_vitals_processing.extractors.hnp_extractor \
  -i /path/to/Hnp.txt \
  -o /path/to/output.parquet \
  -w 4
```

---

**END OF IMPLEMENTATION PLAN**
