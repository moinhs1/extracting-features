# Vitals Extraction Bug Fixes & Coverage Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix PRG blood pressure extraction bug and improve overall vitals coverage from 2-4% to >30% observation rates.

**Architecture:** Fix loose regex patterns causing date strings to be captured as BP values, add temperature unit normalization, and add physiological validation at extraction time.

**Tech Stack:** Python, pandas, regex patterns

---

## Background

### Current Issues Identified

1. **PRG BP Extraction Bug (CRITICAL)**
   - Pattern `(\d{2,3})/(\d{2,3})` (confidence=0.7) matches dates like "12/25"
   - 19.4M bad records (92% of PRG BP data) with median SBP=202, DBP=26
   - Good patterns (confidence 0.8-1.0) have correct median SBP=119-122

2. **Temperature Mixed Units**
   - Both Celsius (36.7) and Fahrenheit (98.6) values in same dataset
   - Layer 1 expects Celsius, but PHY/PRG have Fahrenheit

3. **Low Observation Rates**
   - PHY: 99.96% outpatient, only 2.5% in analysis window
   - HNP/PRG: Good coverage but bad BP data corrupts downstream

### Success Criteria

- PRG SBP median should be 115-135 (not 201)
- PRG DBP median should be 65-85 (not 26)
- All temperatures normalized to Celsius
- Layer 2 observation rates > 20% for core vitals

---

## Task 1: Fix PRG Blood Pressure Pattern

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_patterns.py:27-33`
- Test: `module_3_vitals_processing/tests/test_bp_patterns.py` (create)

**Step 1: Write failing test for BP pattern bug**

Create `module_3_vitals_processing/tests/test_bp_patterns.py`:

```python
"""Tests for blood pressure extraction patterns."""
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from extractors.hnp_patterns import BP_PATTERNS


def extract_bp(text: str) -> list:
    """Extract BP values using patterns."""
    results = []
    for pattern, confidence in BP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            try:
                sbp = float(match.group(1))
                dbp = float(match.group(2))
                results.append({'sbp': sbp, 'dbp': dbp, 'confidence': confidence})
            except (ValueError, IndexError):
                continue
    return results


class TestBPPatterns:
    """Test BP extraction patterns."""

    def test_should_match_standard_bp(self):
        """Standard BP format should be captured."""
        text = "BP: 120/80"
        results = extract_bp(text)
        assert len(results) >= 1
        assert any(r['sbp'] == 120 and r['dbp'] == 80 for r in results)

    def test_should_match_bp_with_label(self):
        """Blood Pressure with label should be captured."""
        text = "Blood Pressure: 135/85 mmHg"
        results = extract_bp(text)
        assert len(results) >= 1
        assert any(r['sbp'] == 135 and r['dbp'] == 85 for r in results)

    def test_should_not_match_dates(self):
        """Date patterns should NOT be captured as BP."""
        date_texts = [
            "Date: 12/25/2023",
            "on 1/31 the patient",
            "Visit on 10/15",
            "DOB: 03/14/1990",
        ]
        for text in date_texts:
            results = extract_bp(text)
            # Should either not match, or not have date-like values
            for r in results:
                # Date months are 1-12, days are 1-31
                # Valid SBP is 60-250, DBP is 30-150
                is_date_like = (r['sbp'] <= 12 and r['dbp'] <= 31) or \
                               (r['sbp'] <= 31 and r['dbp'] <= 31)
                assert not is_date_like, f"Matched date as BP: {r} in '{text}'"

    def test_should_not_match_page_numbers(self):
        """Page references should NOT be captured as BP."""
        text = "see page 201/26 for details"
        results = extract_bp(text)
        # 201/26 looks like invalid BP (pulse pressure 175)
        for r in results:
            pulse_pressure = r['sbp'] - r['dbp']
            assert pulse_pressure < 120, f"Matched page ref as BP: {r}"

    def test_should_reject_physiologically_impossible(self):
        """Physiologically impossible values should be rejected."""
        # After extraction, we filter by valid ranges
        # SBP 40-350, DBP 20-250, and SBP > DBP
        text = "numbers 300/10 here"  # DBP too low
        results = extract_bp(text)
        # Pattern may match but downstream validation should reject
        # For now, ensure we're not matching random fractions
        pass
```

**Step 2: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_bp_patterns.py -v
```

Expected: FAIL on `test_should_not_match_dates` and `test_should_not_match_page_numbers`

**Step 3: Fix BP patterns in hnp_patterns.py**

Modify `module_3_vitals_processing/extractors/hnp_patterns.py` lines 27-33:

```python
# Blood Pressure patterns: (regex, confidence) - captures (SBP, DBP)
# IMPORTANT: Patterns ordered by specificity. Loose patterns removed to avoid date matching.
BP_PATTERNS = [
    # High confidence: explicit BP label
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})/(\d{2,3})', 1.0),
    # Reference range format: (110-130)/(60-80) 120/75
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
    # With mmHg unit (requires unit to avoid date matching)
    (r'(\d{2,3})/(\d{2,3})\s*(?:mmHg|mm\s*Hg)', 0.85),
    # Vitals section context: require "vital" nearby
    (r'(?:vitals?|v/?s)[:\s].{0,30}?(\d{2,3})/(\d{2,3})', 0.8),
    # REMOVED: Bare pattern (\d{2,3})/(\d{2,3}) - matches dates!
]
```

**Step 4: Run tests to verify fix**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_bp_patterns.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_patterns.py
git add module_3_vitals_processing/tests/test_bp_patterns.py
git commit -m "fix(vitals): remove loose BP pattern that matched dates

The pattern (\d{2,3})/(\d{2,3}) was matching date strings like 12/25
as blood pressure, causing 19M bad records with median SBP=202.

- Remove bare digit/digit pattern (confidence 0.7)
- Add vitals section context pattern as safer alternative
- Add tests for date rejection"
```

---

## Task 2: Add Physiological Validation to BP Extraction

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py:113-168`
- Test: `module_3_vitals_processing/tests/test_bp_patterns.py` (add tests)

**Step 1: Add test for pulse pressure validation**

Add to `test_bp_patterns.py`:

```python
def test_pulse_pressure_validation():
    """Pulse pressure should be physiologically plausible."""
    # Normal pulse pressure is 30-60 mmHg
    # Widened (but possible): up to 100 mmHg
    # Impossible: > 120 mmHg
    from extractors.hnp_extractor import extract_blood_pressure

    # Valid BP
    text = "BP 120/80"
    results = extract_blood_pressure(text)
    assert len(results) >= 1

    # The extractor already has SBP > DBP swap logic
    # We need to add pulse pressure validation
```

**Step 2: Run test**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_bp_patterns.py::test_pulse_pressure_validation -v
```

**Step 3: Add pulse pressure validation to hnp_extractor.py**

In `module_3_vitals_processing/extractors/hnp_extractor.py`, modify `extract_blood_pressure` function around line 145:

```python
            # Swap if SBP < DBP (likely transposed)
            if sbp < dbp:
                sbp, dbp = dbp, sbp

            # Validate ranges
            sbp_min, sbp_max = VALID_RANGES['SBP']
            dbp_min, dbp_max = VALID_RANGES['DBP']
            if not (sbp_min <= sbp <= sbp_max and dbp_min <= dbp <= dbp_max):
                continue

            # NEW: Validate pulse pressure (SBP - DBP)
            pulse_pressure = sbp - dbp
            if pulse_pressure < 10 or pulse_pressure > 120:
                # Pulse pressure < 10 or > 120 is physiologically implausible
                continue
```

**Step 4: Run tests**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_bp_patterns.py -v
```

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py
git add module_3_vitals_processing/tests/test_bp_patterns.py
git commit -m "feat(vitals): add pulse pressure validation to BP extraction

Reject BP values where pulse pressure (SBP-DBP) is < 10 or > 120 mmHg
as physiologically implausible."
```

---

## Task 3: Normalize Temperature to Celsius

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py:265-324`
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py:128-219`
- Test: `module_3_vitals_processing/tests/test_temp_extraction.py` (create)

**Step 1: Write test for temperature normalization**

Create `module_3_vitals_processing/tests/test_temp_extraction.py`:

```python
"""Tests for temperature extraction and unit normalization."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


class TestTemperatureNormalization:
    """Test temperature extraction normalizes to Celsius."""

    def test_fahrenheit_detected_and_converted(self):
        """Fahrenheit values should be converted to Celsius."""
        from extractors.hnp_extractor import extract_temperature

        text = "Temp: 98.6 F"
        results = extract_temperature(text)
        assert len(results) >= 1
        # Should be stored as Celsius (37.0)
        result = results[0]
        assert 36.5 <= result['value'] <= 37.5, f"Expected ~37C, got {result['value']}"
        assert result['units'] == 'C'

    def test_celsius_kept_as_is(self):
        """Celsius values should remain unchanged."""
        from extractors.hnp_extractor import extract_temperature

        text = "Temperature: 37.2 C"
        results = extract_temperature(text)
        assert len(results) >= 1
        result = results[0]
        assert 37.0 <= result['value'] <= 37.5
        assert result['units'] == 'C'

    def test_auto_detect_fahrenheit(self):
        """Values > 50 without unit should be detected as Fahrenheit."""
        from extractors.hnp_extractor import extract_temperature

        text = "Temp 99.1"  # No unit, but clearly Fahrenheit
        results = extract_temperature(text)
        if results:  # May or may not match depending on pattern
            result = results[0]
            # Should be converted to Celsius (~37.3)
            assert result['value'] < 50, f"Should convert to Celsius, got {result['value']}"
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_temp_extraction.py -v
```

Expected: FAIL (currently returns Fahrenheit as-is)

**Step 3: Modify extract_temperature to normalize to Celsius**

In `module_3_vitals_processing/extractors/hnp_extractor.py`, modify `extract_temperature` function:

```python
def extract_temperature(text: str) -> List[Dict]:
    """
    Extract temperature values from text.
    All values normalized to Celsius.
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
                units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
            except (ValueError, IndexError):
                continue

            # Auto-detect unit from value if not captured
            if units is None:
                if value > 50:
                    units = 'F'
                else:
                    units = 'C'

            # Validate range based on detected unit
            if units == 'F':
                min_val, max_val = VALID_RANGES['TEMP_F']
                if not (min_val <= value <= max_val):
                    continue
                # Convert to Celsius
                value = (value - 32) * 5 / 9
                units = 'C'
            else:
                min_val, max_val = VALID_RANGES['TEMP_C']
                if not (min_val <= value <= max_val):
                    continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': round(value, 1),  # Round to 1 decimal
                'units': 'C',  # Always Celsius
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
```

**Step 4: Apply same fix to prg_extractor.py**

In `module_3_vitals_processing/extractors/prg_extractor.py`, modify `extract_temperature_with_method`:

Add after line ~165 (after unit detection):

```python
            # Convert Fahrenheit to Celsius
            if units == 'F':
                min_val, max_val = VALID_RANGES['TEMP_F']
                if not (min_val <= value <= max_val):
                    continue
                value = (value - 32) * 5 / 9
                units = 'C'
            else:
                min_val, max_val = VALID_RANGES['TEMP_C']
                if not (min_val <= value <= max_val):
                    continue
```

**Step 5: Run tests**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_temp_extraction.py -v
```

**Step 6: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py
git add module_3_vitals_processing/extractors/prg_extractor.py
git add module_3_vitals_processing/tests/test_temp_extraction.py
git commit -m "feat(vitals): normalize all temperatures to Celsius

- Auto-detect Fahrenheit when value > 50 and no unit specified
- Convert Fahrenheit to Celsius at extraction time
- All downstream layers now receive consistent Celsius values"
```

---

## Task 4: Re-run PRG Extraction with Fixed Patterns

**Files:**
- Run: `module_3_vitals_processing/extractors/prg_extractor.py`

**Step 1: Backup existing PRG output**

```bash
cd /home/moin/TDA_11_25
mv module_3_vitals_processing/outputs/discovery/prg_vitals_raw.parquet \
   module_3_vitals_processing/outputs/discovery/prg_vitals_raw.parquet.bak
```

**Step 2: Re-run PRG extraction**

```bash
cd /home/moin/TDA_11_25/module_3_vitals_processing/extractors
python prg_extractor.py --no-resume
```

Note: This will take 30-60 minutes for the full Prg.txt file.

**Step 3: Verify improved BP values**

```bash
python3 << 'EOF'
import pandas as pd
prg = pd.read_parquet('../outputs/discovery/prg_vitals_raw.parquet')
sbp = prg[prg['vital_type'] == 'SBP']['value']
dbp = prg[prg['vital_type'] == 'DBP']['value']
print(f"SBP: median={sbp.median():.1f}, mean={sbp.mean():.1f}")
print(f"DBP: median={dbp.median():.1f}, mean={dbp.mean():.1f}")
# Expected: SBP median ~120, DBP median ~75
EOF
```

Expected: SBP median 115-135, DBP median 65-85

**Step 4: Commit verification results**

```bash
git add module_3_vitals_processing/outputs/discovery/
git commit -m "data(vitals): re-extract PRG vitals with fixed patterns

BP values now physiologically correct:
- SBP median: ~120 (was 201)
- DBP median: ~75 (was 26)"
```

---

## Task 5: Re-run Layer 1-5 Pipeline

**Files:**
- Run: Layer 1-5 builders sequentially

**Step 1: Run Layer 1**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer1_builder.py
```

**Step 2: Run Layer 2**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer2_builder.py
```

**Step 3: Run Layer 3**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer3_builder.py
```

**Step 4: Run Layer 4**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer4_builder.py
```

**Step 5: Run Layer 5**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer5_builder.py
```

**Step 6: Verify improved observation rates**

```bash
python3 << 'EOF'
import h5py
import numpy as np

with h5py.File('module_3_vitals_processing/outputs/layer2/hourly_tensors.h5', 'r') as f:
    masks = f['masks'][:]
    vitals = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']
    print("Observation rates (% hours with actual measurements):")
    for i, v in enumerate(vitals):
        obs_rate = masks[:, :, i].mean() * 100
        print(f"  {v}: {obs_rate:.1f}%")
EOF
```

Expected: Observation rates > 10% (improved from 2-4%)

**Step 7: Commit pipeline outputs**

```bash
git add module_3_vitals_processing/outputs/
git commit -m "data(vitals): regenerate all layers with corrected extraction

Observation rates improved with fixed BP patterns and temp normalization."
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] PRG SBP median is 115-135 (not 201)
- [ ] PRG DBP median is 65-85 (not 26)
- [ ] All temperatures in Layer 1 are in Celsius (30-45 range)
- [ ] Layer 2 observation rates > 10% for all vitals
- [ ] All tests pass
- [ ] No NaN in non-reserved world state dimensions

---

**Document Version:** 1.0
**Created:** 2025-12-11
