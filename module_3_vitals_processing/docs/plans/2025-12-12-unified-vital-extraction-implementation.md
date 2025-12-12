# Unified Vital Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Merge 150+ patterns from legacy ultimate_vital_extraction.py into robust hnp/prg extraction pipeline with unified architecture.

**Architecture:** Create unified_patterns.py (single source of truth for all patterns), unified_extractor.py (core extraction logic with validation), then refactor hnp_extractor.py and prg_extractor.py as thin wrappers. Add O2/BMI extraction to separate supplemental output files.

**Tech Stack:** Python, regex, pandas, pytest

---

## Task 1: Create Unified Patterns - Heart Rate

**Files:**
- Create: `module_3_vitals_processing/extractors/unified_patterns.py`
- Test: `module_3_vitals_processing/tests/test_unified_patterns.py`

**Step 1: Write failing test for HR patterns**

Create `module_3_vitals_processing/tests/test_unified_patterns.py`:

```python
"""Tests for unified pattern library."""
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHRPatterns:
    """Test heart rate pattern coverage."""

    @pytest.fixture
    def hr_patterns(self):
        from extractors.unified_patterns import HR_PATTERNS
        return HR_PATTERNS

    def _extract_hr(self, text, patterns):
        """Helper to extract HR values."""
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence, 'tier': tier})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_hr_with_label(self, hr_patterns):
        """HR: 72 should match."""
        results = self._extract_hr("HR: 72", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 72 for r in results)

    def test_heart_rate_with_bpm(self, hr_patterns):
        """Heart Rate 88 bpm should match with high confidence."""
        results = self._extract_hr("Heart Rate 88 bpm", hr_patterns)
        assert len(results) >= 1
        match = next(r for r in results if r['value'] == 88)
        assert match['confidence'] >= 0.90

    def test_pulse_format(self, hr_patterns):
        """Pulse: 65 should match."""
        results = self._extract_hr("Pulse: 65", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 65 for r in results)

    def test_tachycardia_context(self, hr_patterns):
        """tachycardic at 120 should match."""
        results = self._extract_hr("patient tachycardic at 120", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 120 for r in results)

    def test_ekg_rate(self, hr_patterns):
        """EKG rate 78 should match."""
        results = self._extract_hr("EKG shows rate 78", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 78 for r in results)

    def test_sinus_rhythm(self, hr_patterns):
        """normal sinus rhythm 72 should match."""
        results = self._extract_hr("normal sinus rhythm 72", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 72 for r in results)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestHRPatterns -v
```

Expected: FAIL with "No module named 'extractors.unified_patterns'"

**Step 3: Create unified_patterns.py with HR patterns**

Create `module_3_vitals_processing/extractors/unified_patterns.py`:

```python
"""
Unified pattern library for vital sign extraction.

All patterns organized by vital type with 3-tier confidence scoring:
- Standard (0.90-1.0): Explicit label + unit
- Optimized (0.80-0.90): Label or strong context
- Specialized (0.65-0.80): Contextual/bare patterns

Each pattern tuple: (regex, confidence, tier_name)
"""
import re

# Validation ranges (tightened for clinical plausibility)
VALID_RANGES = {
    'HR': (30, 220),
    'SBP': (50, 260),
    'DBP': (25, 150),
    'PULSE_PRESSURE': (10, 120),
    'RR': (6, 50),
    'SPO2': (50, 100),
    'TEMP_C': (33.5, 42.5),
    'TEMP_F': (93, 108),
    'O2_FLOW': (0.5, 60),
    'BMI': (12, 70),
}

# Negation patterns (keep existing 8 for max extraction)
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

# Skip section patterns (false positive sources)
SKIP_SECTION_PATTERNS = [
    r'Allerg(?:ies|ic|en)[:\s]',
    r'[Rr]eaction\(?s?\)?[:\s]',
    r'Medications?[:\s]',
    r'(?:Outpatient\s+)?Prescriptions?[:\s]',
    r'Scheduled\s+Meds[:\s]',
    r'Past\s+(?:Medical\s+)?History[:\s]',
    r'Family\s+History[:\s]',
    r'(?:History\s+of\s+)?Present\s+Illness[:\s]',
    r'Social\s+History[:\s]',
    r'Surgical\s+History[:\s]',
    r'Review\s+of\s+Systems[:\s]',
    r'ROS[:\s]',
]

# Heart Rate patterns - sorted by confidence (highest first)
HR_PATTERNS = [
    # Standard tier (0.90-1.0) - explicit label with/without unit
    (r'Heart\s*Rate\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM|beats\s*per\s*minute)?', 0.95, 'standard'),
    (r'HR\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM)?', 0.95, 'standard'),
    (r'(?:Pulse|P)\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM)?', 0.90, 'standard'),
    (r'\[\d{2,3}-\d{2,3}\]\s*(\d{2,3})', 0.90, 'standard'),  # Reference range format

    # Optimized tier (0.80-0.90) - strong context
    (r'(?:EKG|ECG)[^0-9]*(?:rate|HR)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:sinus|normal\s+sinus)\s*(?:rhythm|tachycardia|bradycardia)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:tachycardic|bradycardia)[^0-9]*(?:at|with|to)[^0-9]*(\d{1,3})', 0.85, 'optimized'),
    (r'(?:monitor|cardiac\s+monitor)[^0-9\n]*(?:rate|HR|pulse)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'rate\s*(?:is|of|at|=)\s*(\d{2,3})', 0.82, 'optimized'),
    (r'heart\s+rate[^0-9]*of[^0-9]*(\d{1,3})', 0.85, 'optimized'),
    (r'pulse[^0-9]*of[^0-9]*(\d{1,3})', 0.82, 'optimized'),
    (r'(?:HR|Heart\s*Rate|Pulse)[^0-9]*in\s+the\s+(\d{2,3})s', 0.80, 'optimized'),

    # Specialized tier (0.65-0.80) - contextual patterns
    (r'vitals[^:]*:[^:]*(?:[^\/\d]*?)(?:\d{2,3}[/\\]\d{2,3})[^,]*,\s*(?:HR|P)?\s*(\d{2,3})', 0.75, 'specialized'),
    (r'VS[^\d\n]*[\d/,:.]+[^,\d\n]*,\s*(?:HR|P)?\s*(\d{2,3})', 0.72, 'specialized'),
    (r'(?:cardiac|cardio|heart)\s*(?:monitor|monitoring|telemetry)[^0-9]*(\d{2,3})', 0.70, 'specialized'),
    (r'(?:atrial|junctional|ventricular)[^0-9]*rhythm[^0-9]*(\d{2,3})', 0.70, 'specialized'),
    (r'(?<=\W)(?:HR|P)[\s:=]*(\d{2,3})[\s,]', 0.68, 'specialized'),
]
```

**Step 4: Run test to verify it passes**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestHRPatterns -v
```

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_patterns.py
git add module_3_vitals_processing/tests/test_unified_patterns.py
git commit -m "feat(vitals): add unified_patterns.py with HR patterns

- Create unified pattern library with 3-tier confidence scoring
- Add 16 HR patterns from ultimate + existing sources
- Add validation ranges, negation patterns, skip section patterns
- Add comprehensive HR pattern tests"
```

---

## Task 2: Add Blood Pressure Patterns

**Files:**
- Modify: `module_3_vitals_processing/extractors/unified_patterns.py`
- Modify: `module_3_vitals_processing/tests/test_unified_patterns.py`

**Step 1: Write failing tests for BP patterns**

Add to `module_3_vitals_processing/tests/test_unified_patterns.py`:

```python
class TestBPPatterns:
    """Test blood pressure pattern coverage."""

    @pytest.fixture
    def bp_patterns(self):
        from extractors.unified_patterns import BP_PATTERNS
        return BP_PATTERNS

    def _extract_bp(self, text, patterns):
        """Helper to extract BP values."""
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    sbp = float(match.group(1))
                    dbp = float(match.group(2))
                    results.append({'sbp': sbp, 'dbp': dbp, 'confidence': confidence, 'tier': tier})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_bp(self, bp_patterns):
        """BP: 120/80 should match."""
        results = self._extract_bp("BP: 120/80", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 120 and r['dbp'] == 80 for r in results)

    def test_blood_pressure_spelled_out(self, bp_patterns):
        """Blood Pressure 135/85 should match."""
        results = self._extract_bp("Blood Pressure: 135/85", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 135 and r['dbp'] == 85 for r in results)

    def test_bp_with_mmhg(self, bp_patterns):
        """140/90 mmHg should match."""
        results = self._extract_bp("140/90 mmHg", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 140 and r['dbp'] == 90 for r in results)

    def test_bp_reference_range_format(self, bp_patterns):
        """(110-130)/(60-80) 125/75 should match the actual value."""
        results = self._extract_bp("(110-130)/(60-80) 125/75", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 125 and r['dbp'] == 75 for r in results)

    def test_vitals_context_bp(self, bp_patterns):
        """vitals: 120/80 should match."""
        results = self._extract_bp("vitals: 120/80, HR 72", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 120 and r['dbp'] == 80 for r in results)

    def test_should_not_match_dates(self, bp_patterns):
        """Date patterns should NOT match as BP."""
        date_texts = [
            "Date: 12/25/2023",
            "on 1/31 the patient",
            "Visit on 10/15",
        ]
        for text in date_texts:
            results = self._extract_bp(text, bp_patterns)
            # Filter out date-like values
            valid_bp = [r for r in results if r['sbp'] > 50 and r['dbp'] > 25]
            assert len(valid_bp) == 0, f"Matched date as BP in '{text}': {results}"
```

**Step 2: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestBPPatterns -v
```

Expected: FAIL with "cannot import name 'BP_PATTERNS'"

**Step 3: Add BP patterns to unified_patterns.py**

Add to `module_3_vitals_processing/extractors/unified_patterns.py` after HR_PATTERNS:

```python
# Blood Pressure patterns - captures (SBP, DBP)
# IMPORTANT: No bare (\d{2,3})/(\d{2,3}) pattern - matches dates!
BP_PATTERNS = [
    # Standard tier (0.90-1.0) - explicit label
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})[/\\](\d{2,3})', 0.95, 'standard'),
    (r'(?:BP|Blood\s*Pressure)\s*:?\s*(\d{2,3})\s*[/\\]\s*(\d{2,3})\s*(?:mmHg|mm\s*Hg)?', 0.95, 'standard'),
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})[/\\](\d{2,3})', 0.92, 'standard'),  # Reference range
    (r'(\d{2,3})[/\\](\d{2,3})\s*(?:mmHg|mm\s*Hg)', 0.90, 'standard'),  # With unit

    # Optimized tier (0.80-0.90) - strong context
    (r'(?:vitals?|v/?s)[:\s].{0,30}?(\d{2,3})[/\\](\d{2,3})', 0.88, 'optimized'),
    (r'blood\s+pressure[^0-9]*(\d{2,3})[^0-9]*(?:over|/)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:systolic|SBP)[^0-9]*(\d{2,3})[^0-9]*(?:diastolic|DBP)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'(?:BP|Blood\s*Pressure)[^0-9]*of[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.85, 'optimized'),
    (r'initial\s+(?:vitals|VS)[^0-9]*(?:BP|blood)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.85, 'optimized'),
    (r'cuff[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.82, 'optimized'),
    (r'(?:avg|average)\s+BP[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.82, 'optimized'),

    # Specialized tier (0.65-0.80) - contextual
    (r'VS[^\d\n]*[\d/,:.]+[^\d\n]*(\d{2,3})[/\\](\d{2,3})', 0.75, 'specialized'),
    (r'pressure\s+of\s+(\d{2,3})[/\\](\d{2,3})', 0.72, 'specialized'),
    (r'(?:measured|documented|recorded)[^0-9]*(?:bp|blood\s+pressure)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.70, 'specialized'),
    (r'(?:admission|initial|presenting)[^0-9]*(?:bp|blood\s+pressure)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.70, 'specialized'),
    (r'bp\s*=\s*(\d{2,3})[/\\](\d{2,3})', 0.68, 'specialized'),
]
```

**Step 4: Run test to verify it passes**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestBPPatterns -v
```

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_patterns.py
git add module_3_vitals_processing/tests/test_unified_patterns.py
git commit -m "feat(vitals): add BP patterns to unified_patterns.py

- Add 16 BP patterns with 3-tier confidence
- Explicitly avoid bare digit/digit pattern (matches dates)
- Add tests including date rejection verification"
```

---

## Task 3: Add RR, SpO2, Temperature Patterns

**Files:**
- Modify: `module_3_vitals_processing/extractors/unified_patterns.py`
- Modify: `module_3_vitals_processing/tests/test_unified_patterns.py`

**Step 1: Write failing tests for RR, SpO2, Temp patterns**

Add to `module_3_vitals_processing/tests/test_unified_patterns.py`:

```python
class TestRRPatterns:
    """Test respiratory rate pattern coverage."""

    @pytest.fixture
    def rr_patterns(self):
        from extractors.unified_patterns import RR_PATTERNS
        return RR_PATTERNS

    def _extract_rr(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_rr(self, rr_patterns):
        """RR: 18 should match."""
        results = self._extract_rr("RR: 18", rr_patterns)
        assert any(r['value'] == 18 for r in results)

    def test_respiratory_rate_spelled(self, rr_patterns):
        """Respiratory Rate 20 should match."""
        results = self._extract_rr("Respiratory Rate 20", rr_patterns)
        assert any(r['value'] == 20 for r in results)

    def test_breaths_per_min(self, rr_patterns):
        """16 breaths/min should match."""
        results = self._extract_rr("16 breaths/min", rr_patterns)
        assert any(r['value'] == 16 for r in results)


class TestSpO2Patterns:
    """Test SpO2 pattern coverage."""

    @pytest.fixture
    def spo2_patterns(self):
        from extractors.unified_patterns import SPO2_PATTERNS
        return SPO2_PATTERNS

    def _extract_spo2(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_spo2_standard(self, spo2_patterns):
        """SpO2: 98% should match."""
        results = self._extract_spo2("SpO2: 98%", spo2_patterns)
        assert any(r['value'] == 98 for r in results)

    def test_o2_sat(self, spo2_patterns):
        """O2 Sat 95% should match."""
        results = self._extract_spo2("O2 Sat 95%", spo2_patterns)
        assert any(r['value'] == 95 for r in results)

    def test_room_air_context(self, spo2_patterns):
        """92% on room air should match."""
        results = self._extract_spo2("92% on room air", spo2_patterns)
        assert any(r['value'] == 92 for r in results)


class TestTempPatterns:
    """Test temperature pattern coverage."""

    @pytest.fixture
    def temp_patterns(self):
        from extractors.unified_patterns import TEMP_PATTERNS
        return TEMP_PATTERNS

    def _extract_temp(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    unit = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
                    results.append({'value': value, 'unit': unit, 'confidence': confidence})
                except (ValueError, IndexError, AttributeError):
                    continue
        return results

    def test_temp_fahrenheit(self, temp_patterns):
        """Temp: 98.6 F should match."""
        results = self._extract_temp("Temp: 98.6 F", temp_patterns)
        assert any(r['value'] == 98.6 and r['unit'] == 'F' for r in results)

    def test_temp_celsius(self, temp_patterns):
        """Temperature 37.2 C should match."""
        results = self._extract_temp("Temperature 37.2 C", temp_patterns)
        assert any(r['value'] == 37.2 and r['unit'] == 'C' for r in results)

    def test_tmax(self, temp_patterns):
        """Tmax 101.2 F should match."""
        results = self._extract_temp("Tmax 101.2 F", temp_patterns)
        assert any(r['value'] == 101.2 for r in results)
```

**Step 2: Run test to verify it fails**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestRRPatterns -v
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestSpO2Patterns -v
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py::TestTempPatterns -v
```

Expected: FAIL with "cannot import name"

**Step 3: Add RR, SpO2, Temp patterns**

Add to `module_3_vitals_processing/extractors/unified_patterns.py`:

```python
# Respiratory Rate patterns
RR_PATTERNS = [
    # Standard tier
    (r'Respiratory\s*Rate\s*:?\s*(\d{1,2})\b', 0.95, 'standard'),
    (r'(?:RR|Resp|TRR)\s*:?\s*(\d{1,2})\b', 0.92, 'standard'),
    (r'\[\d{1,2}-\d{1,2}\]\s*(\d{1,2})\b', 0.90, 'standard'),

    # Optimized tier
    (r'(?:RR|Respiratory\s*Rate|Resp|respirations)[^0-9]*of[^0-9]*(\d{1,2})', 0.88, 'optimized'),
    (r'respiratory\s+rate[^0-9]*(\d{1,2})', 0.88, 'optimized'),
    (r'(?:breathing|respirations)[^0-9]*(?:at|with)[^0-9]*(\d{1,2})', 0.85, 'optimized'),
    (r'(\d{1,2})\s*(?:breaths?[/\s]*min|breaths?\s*per\s*minute)', 0.85, 'optimized'),
    (r'breath(?:ing|s)?\s*(?:at|of)\s*(\d{1,2})', 0.82, 'optimized'),

    # Specialized tier
    (r'vitals[^:]*:[^:]*(?:[^,]*,){2}[^,]*(?:RR)?\s*(\d{1,2})', 0.75, 'specialized'),
    (r'VS[^\d\n]*[\d/,:.]+[^,\d\n]*,[^,\d\n]*,[^,\d\n]*(?:RR)?\s*(\d{1,2})', 0.72, 'specialized'),
    (r'(?:ventilator|vent)[^0-9]*(?:rate|rr)[^0-9]*(\d{1,2})', 0.70, 'specialized'),
    (r'(?<=\W)(?:RR)[\s:=]*(\d{1,2})[\s,]', 0.68, 'specialized'),
]

# SpO2 patterns
SPO2_PATTERNS = [
    # Standard tier
    (r'(?:SpO2|SaO2|O2\s*Sat(?:uration)?)\s*:?\s*>?(\d{2,3})\s*%?', 0.95, 'standard'),
    (r'(?:oxygen\s+saturation|pulse\s+ox|pulseox|pox)\s*:?\s*(\d{2,3})\s*%?', 0.92, 'standard'),

    # Optimized tier
    (r'(\d{2,3})\s*%\s*(?:on|RA|room\s+air|O2|oxygen)', 0.88, 'optimized'),
    (r'(?:on\s+room\s+air|RA|ambient\s+air)[^0-9]*(?:SpO2|O2\s+Sat|saturation|sat)[^0-9]*(\d{2,3})\s*%?', 0.88, 'optimized'),
    (r'(?:saturation|sat)[^0-9=]*(?:=|-|:|\s)[^0-9%]*(9[0-9]|100)\s*%?', 0.85, 'optimized'),
    (r'(?:O2|oxygen)[^0-9]*(?:saturation|sat|level)[^0-9]*(\d{2,3})\s*%?', 0.85, 'optimized'),
    (r'saturating[^0-9]*(?:at|to)?[^0-9]*(\d{2,3})\s*%?', 0.82, 'optimized'),

    # Specialized tier
    (r'(?:RA|room\s+air)[^0-9]*(\d{2,3})\s*%?', 0.75, 'specialized'),
    (r'(?:pulse\s+ox|SpO2)[^0-9]*(\d{2,3})', 0.72, 'specialized'),
    (r'(?:monitor|monitoring)[^0-9\n]*(?:O2|sat|SpO2|saturation)[^0-9\n]*(\d{2,3})', 0.70, 'specialized'),
    (r'VS[^\d\n]*[\d/,:.]+(?:[^,\d\n]*,){4}[^,\d\n]*(?:O2)?\s*(9\d|100)', 0.68, 'specialized'),
]

# Temperature patterns - captures (value, unit)
TEMP_PATTERNS = [
    # Standard tier - with explicit unit
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.95, 'standard'),
    (r'(?:Tmax|T-max|Tcurrent)\s*:?\s*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.95, 'standard'),
    (r'T\s+(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.92, 'standard'),

    # Optimized tier
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*(?:degrees)?\s*([CF])', 0.88, 'optimized'),
    (r'temperature[^0-9]*(\d{2,3}\.?\d?)[^0-9]*([CF])', 0.88, 'optimized'),
    (r'(?:afebrile|febrile)[^0-9]*(?:at|with)?[^0-9]*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])?', 0.85, 'optimized'),
    (r'(\d{2,3}\.\d)\s*[°?]\s*([CF])', 0.82, 'optimized'),

    # Specialized tier - may need unit inference
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\b', 0.75, 'specialized'),
    (r'(?:T|temp)[\s:=]+(\d{2,3}\.?\d?)(?!\d)', 0.70, 'specialized'),
    (r'(?:afebrile|febrile)[^0-9]*(9\d\.?\d{0,2}|10[0-4]\.?\d{0,2}|3[5-9]\.?\d{0,2})', 0.68, 'specialized'),
]
```

**Step 4: Run tests to verify they pass**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_patterns.py
git add module_3_vitals_processing/tests/test_unified_patterns.py
git commit -m "feat(vitals): add RR, SpO2, Temp patterns to unified_patterns.py

- Add 12 RR patterns with 3-tier confidence
- Add 11 SpO2 patterns with 3-tier confidence
- Add 10 Temperature patterns with unit capture
- Add comprehensive tests for each vital type"
```

---

## Task 4: Add Supplemental Patterns (O2 Flow, O2 Device, BMI)

**Files:**
- Modify: `module_3_vitals_processing/extractors/unified_patterns.py`
- Create: `module_3_vitals_processing/tests/test_supplemental_patterns.py`

**Step 1: Write failing tests**

Create `module_3_vitals_processing/tests/test_supplemental_patterns.py`:

```python
"""Tests for supplemental vital patterns (O2, BMI)."""
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestO2FlowPatterns:
    """Test O2 flow rate pattern coverage."""

    @pytest.fixture
    def o2_flow_patterns(self):
        from extractors.unified_patterns import O2_FLOW_PATTERNS
        return O2_FLOW_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_liters_nc(self, o2_flow_patterns):
        """2L NC should match."""
        results = self._extract("on 2L NC", o2_flow_patterns)
        assert any(r['value'] == 2 for r in results)

    def test_liters_per_minute(self, o2_flow_patterns):
        """4 L/min should match."""
        results = self._extract("4 L/min via nasal cannula", o2_flow_patterns)
        assert any(r['value'] == 4 for r in results)

    def test_high_flow(self, o2_flow_patterns):
        """40L HFNC should match."""
        results = self._extract("40L HFNC", o2_flow_patterns)
        assert any(r['value'] == 40 for r in results)


class TestO2DevicePatterns:
    """Test O2 device pattern coverage."""

    @pytest.fixture
    def o2_device_patterns(self):
        from extractors.unified_patterns import O2_DEVICE_PATTERNS
        return O2_DEVICE_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append({'device': match.group(0), 'confidence': confidence})
        return results

    def test_nasal_cannula(self, o2_device_patterns):
        """nasal cannula should match."""
        results = self._extract("on nasal cannula", o2_device_patterns)
        assert len(results) >= 1

    def test_room_air(self, o2_device_patterns):
        """room air should match."""
        results = self._extract("on room air", o2_device_patterns)
        assert len(results) >= 1

    def test_high_flow(self, o2_device_patterns):
        """HFNC should match."""
        results = self._extract("on HFNC", o2_device_patterns)
        assert len(results) >= 1


class TestBMIPatterns:
    """Test BMI pattern coverage."""

    @pytest.fixture
    def bmi_patterns(self):
        from extractors.unified_patterns import BMI_PATTERNS
        return BMI_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_bmi_standard(self, bmi_patterns):
        """BMI: 24.5 should match."""
        results = self._extract("BMI: 24.5", bmi_patterns)
        assert any(r['value'] == 24.5 for r in results)

    def test_bmi_with_units(self, bmi_patterns):
        """BMI 28.3 kg/m2 should match."""
        results = self._extract("BMI 28.3 kg/m2", bmi_patterns)
        assert any(r['value'] == 28.3 for r in results)
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_supplemental_patterns.py -v
```

Expected: FAIL

**Step 3: Add supplemental patterns**

Add to `module_3_vitals_processing/extractors/unified_patterns.py`:

```python
# O2 Flow Rate patterns - captures flow in L/min
O2_FLOW_PATTERNS = [
    # Standard tier
    (r'(?:on|via)\s+(\d+(?:\.\d)?)\s*L\s*(?:NC|nasal\s+cannula)', 0.95, 'standard'),
    (r'(\d+(?:\.\d)?)\s*L(?:PM|/min|iters?\s*/\s*min)', 0.95, 'standard'),
    (r'O2\s+Flow\s+Rate[^0-9]*(\d+(?:\.\d)?)', 0.92, 'standard'),

    # Optimized tier
    (r'(\d+(?:\.\d)?)\s*L\s*(?:NC|nasal\s+cannula|NRB|non-rebreather|mask|FM|face\s+mask|HFNC|high\s*-?\s*flow)', 0.88, 'optimized'),
    (r'(?:nasal\s+cannula|NC|high\s+flow|face\s+mask|HFNC|NRB)[^0-9]*(?:at|with|delivering)[^0-9]*(\d+(?:\.\d)?)\s*L?', 0.85, 'optimized'),
    (r'(?:on|receiving|with|at)\s+(\d+(?:\.\d)?)\s*L?\s*(?:O2|oxygen)', 0.82, 'optimized'),

    # Specialized tier
    (r'(\d+(?:\.\d)?)\s*L\s*(?:O2|oxygen)', 0.75, 'specialized'),
    (r'flow[^0-9]*(?:rate|of)?[^0-9]*(\d+(?:\.\d)?)\s*L?', 0.70, 'specialized'),
    (r'O2[^0-9]*(\d+(?:\.\d)?)\s*L', 0.68, 'specialized'),
]

# O2 Device patterns - returns device string (not numeric)
O2_DEVICE_PATTERNS = [
    # Standard tier - specific devices
    (r'(?:on|via)\s+(nasal\s+cannula|NC)', 0.95, 'standard'),
    (r'(?:on|via)\s+(room\s+air|RA|ambient\s+air)', 0.95, 'standard'),
    (r'(?:on|via)\s+(HFNC|high\s*-?\s*flow\s+nasal\s+cannula|high\s+flow)', 0.95, 'standard'),
    (r'(?:on|via)\s+(NRB|non-rebreather|non\s+rebreather)', 0.92, 'standard'),
    (r'(?:on|via)\s+(face\s+mask|FM|simple\s+mask|venturi\s+mask)', 0.92, 'standard'),

    # Optimized tier
    (r'(nasal\s+cannula|NC|nasal\s+prongs)', 0.85, 'optimized'),
    (r'(room\s+air|RA)', 0.85, 'optimized'),
    (r'(CPAP|BiPAP|ventilator|mechanical\s+ventilation|intubated)', 0.88, 'optimized'),

    # Specialized tier
    (r'supplemental[^0-9\n]*(oxygen|O2)', 0.75, 'specialized'),
    (r'(?:oxygen\s+therapy|O2\s+therapy)', 0.72, 'specialized'),
]

# BMI patterns
BMI_PATTERNS = [
    # Standard tier
    (r'BMI\s*:?\s*(\d{1,2}(?:\.\d{1,2})?)', 0.95, 'standard'),
    (r'(?:Body\s+Mass\s+Index|body\s+mass\s+index)\s*:?\s*(\d{1,2}(?:\.\d{1,2})?)', 0.95, 'standard'),

    # Optimized tier
    (r'BMI[^0-9]*of[^0-9]*(\d{1,2}(?:\.\d{1,2})?)', 0.88, 'optimized'),
    (r'BMI[^0-9]*(\d{1,2}(?:\.\d{1,2})?)\s*(?:kg/m2|kg/m\^?2)', 0.88, 'optimized'),
    (r'calculated\s+BMI[^0-9]*(\d{1,2}(?:\.\d{1,2})?)', 0.85, 'optimized'),

    # Specialized tier
    (r'(?:overweight|obese|obesity)[^0-9\n]*(?:BMI)[^0-9\n]*(\d{1,2}(?:\.\d{1,2})?)', 0.75, 'specialized'),
    (r'BMI\s*(?:of|is|was|=)[^0-9]*(\d{1,2}\.?\d{0,2})', 0.72, 'specialized'),
]
```

**Step 4: Run tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_supplemental_patterns.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_patterns.py
git add module_3_vitals_processing/tests/test_supplemental_patterns.py
git commit -m "feat(vitals): add O2 flow, O2 device, BMI patterns

- Add 9 O2 flow rate patterns for L/min extraction
- Add 10 O2 device patterns (returns string)
- Add 7 BMI patterns
- Add comprehensive tests for supplemental vitals"
```

---

## Task 5: Create Unified Extractor Core

**Files:**
- Create: `module_3_vitals_processing/extractors/unified_extractor.py`
- Create: `module_3_vitals_processing/tests/test_unified_extractor.py`

**Step 1: Write failing test for core extraction**

Create `module_3_vitals_processing/tests/test_unified_extractor.py`:

```python
"""Tests for unified extractor core logic."""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCheckNegation:
    """Test negation detection."""

    def test_no_negation(self):
        from extractors.unified_extractor import check_negation
        text = "HR: 72 bpm"
        assert check_negation(text, 4) is False

    def test_not_obtained(self):
        from extractors.unified_extractor import check_negation
        text = "vitals not obtained"
        assert check_negation(text, 7) is True

    def test_refused(self):
        from extractors.unified_extractor import check_negation
        text = "patient refused vitals"
        assert check_negation(text, 16) is True


class TestSkipSection:
    """Test skip section detection."""

    def test_not_in_skip(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Vitals: HR 72"
        assert is_in_skip_section(text, 10) is False

    def test_in_medications(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Medications: metoprolol 100mg daily"
        assert is_in_skip_section(text, 25) is True

    def test_in_allergies(self):
        from extractors.unified_extractor import is_in_skip_section
        text = "Allergies: penicillin causes HR 120"
        assert is_in_skip_section(text, 32) is True


class TestExtractHeartRate:
    """Test HR extraction with full pipeline."""

    def test_basic_hr(self):
        from extractors.unified_extractor import extract_heart_rate
        results = extract_heart_rate("HR: 72")
        assert len(results) >= 1
        assert results[0]['value'] == 72

    def test_hr_validation(self):
        from extractors.unified_extractor import extract_heart_rate
        # HR of 500 is invalid
        results = extract_heart_rate("HR: 500")
        assert len(results) == 0

    def test_hr_skip_medications(self):
        from extractors.unified_extractor import extract_heart_rate
        text = "Medications: metoprolol 100mg. Vitals: HR 72"
        results = extract_heart_rate(text)
        # Should find HR 72, not 100
        assert all(r['value'] != 100 for r in results)
        assert any(r['value'] == 72 for r in results)

    def test_hr_deduplication(self):
        from extractors.unified_extractor import extract_heart_rate
        # Same HR matched by multiple patterns should dedupe
        text = "Heart Rate: 72 bpm"
        results = extract_heart_rate(text)
        values = [r['value'] for r in results]
        # Should only have one 72, not multiple
        assert values.count(72) == 1


class TestExtractBloodPressure:
    """Test BP extraction with validation."""

    def test_basic_bp(self):
        from extractors.unified_extractor import extract_blood_pressure
        results = extract_blood_pressure("BP: 120/80")
        assert len(results) >= 1
        assert results[0]['sbp'] == 120
        assert results[0]['dbp'] == 80

    def test_bp_swap_transposed(self):
        from extractors.unified_extractor import extract_blood_pressure
        # 70/140 should be swapped to 140/70
        results = extract_blood_pressure("BP: 70/140")
        assert len(results) >= 1
        assert results[0]['sbp'] == 140
        assert results[0]['dbp'] == 70

    def test_bp_pulse_pressure_validation(self):
        from extractors.unified_extractor import extract_blood_pressure
        # 120/115 has pulse pressure of 5, invalid
        results = extract_blood_pressure("BP: 120/115")
        assert len(results) == 0

    def test_bp_skip_dates(self):
        from extractors.unified_extractor import extract_blood_pressure
        # Dates should not match (no BP label context)
        results = extract_blood_pressure("Date: 12/25/2023")
        # Should not extract 12/25 as BP
        assert not any(r['sbp'] == 12 and r['dbp'] == 25 for r in results)


class TestExtractTemperature:
    """Test temperature extraction with unit normalization."""

    def test_fahrenheit_converted(self):
        from extractors.unified_extractor import extract_temperature
        results = extract_temperature("Temp: 98.6 F")
        assert len(results) >= 1
        # Should be converted to Celsius (~37.0)
        assert 36.5 <= results[0]['value'] <= 37.5
        assert results[0]['units'] == 'C'

    def test_celsius_unchanged(self):
        from extractors.unified_extractor import extract_temperature
        results = extract_temperature("Temp: 37.0 C")
        assert len(results) >= 1
        assert results[0]['value'] == 37.0
        assert results[0]['units'] == 'C'

    def test_auto_detect_fahrenheit(self):
        from extractors.unified_extractor import extract_temperature
        # 98.6 without unit should be detected as F
        results = extract_temperature("Temp: 98.6")
        if results:
            assert results[0]['value'] < 50  # Converted to C
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_extractor.py -v
```

Expected: FAIL with "No module named 'extractors.unified_extractor'"

**Step 3: Create unified_extractor.py**

Create `module_3_vitals_processing/extractors/unified_extractor.py`:

```python
"""
Unified vital sign extractor.

Core extraction logic with validation, negation detection, and skip section filtering.
"""
import re
from typing import Dict, List, Optional, Tuple

from .unified_patterns import (
    HR_PATTERNS, BP_PATTERNS, RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS,
    O2_FLOW_PATTERNS, O2_DEVICE_PATTERNS, BMI_PATTERNS,
    VALID_RANGES, NEGATION_PATTERNS, SKIP_SECTION_PATTERNS
)


def check_negation(text: str, position: int, window: int = 50) -> bool:
    """
    Check for negation phrases near match position.

    Args:
        text: Full text
        position: Character position of the match
        window: Characters to check before position

    Returns:
        True if negation found
    """
    start = max(0, position - window)
    context = text[start:position].lower()

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def is_in_skip_section(text: str, position: int, lookback: int = 500) -> bool:
    """
    Check if position is within a skip section.

    Args:
        text: Full text
        position: Character position
        lookback: Characters to look back

    Returns:
        True if in skip section (should not extract)
    """
    start = max(0, position - lookback)
    context_before = text[start:position]

    # Find most recent skip section header
    last_skip_pos = -1
    for pattern in SKIP_SECTION_PATTERNS:
        for match in re.finditer(pattern, context_before, re.IGNORECASE):
            if match.end() > last_skip_pos:
                last_skip_pos = match.end()

    if last_skip_pos == -1:
        return False

    # Check if a valid clinical section appears after skip
    valid_sections = [
        r'Vitals?[:\s]',
        r'Physical\s+Exam[:\s]',
        r'Objective[:\s]',
        r'Exam[:\s]',
        r'Assessment[:\s]',
    ]
    context_after_skip = context_before[last_skip_pos:]
    for pattern in valid_sections:
        if re.search(pattern, context_after_skip, re.IGNORECASE):
            return False

    return True


def extract_heart_rate(text: str) -> List[Dict]:
    """Extract heart rate values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in HR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Position deduplication
            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            # Skip section check
            if is_in_skip_section(text, position):
                continue

            # Negation check
            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # Range validation
            min_val, max_val = VALID_RANGES['HR']
            if not (min_val <= value <= max_val):
                continue

            # Abnormal flag
            is_abnormal = value < 60 or value > 100

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_blood_pressure(text: str) -> List[Dict]:
    """Extract blood pressure values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in BP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            if check_negation(text, position):
                continue

            try:
                sbp = float(match.group(1))
                dbp = float(match.group(2))
            except (ValueError, IndexError):
                continue

            # Swap if transposed
            if sbp < dbp:
                sbp, dbp = dbp, sbp

            # Range validation
            sbp_min, sbp_max = VALID_RANGES['SBP']
            dbp_min, dbp_max = VALID_RANGES['DBP']
            if not (sbp_min <= sbp <= sbp_max and dbp_min <= dbp <= dbp_max):
                continue

            # Pulse pressure validation
            pulse_pressure = sbp - dbp
            pp_min, pp_max = VALID_RANGES['PULSE_PRESSURE']
            if not (pp_min <= pulse_pressure <= pp_max):
                continue

            # Abnormal flags
            sbp_abnormal = sbp < 90 or sbp > 180
            dbp_abnormal = dbp < 60 or dbp > 110

            results.append({
                'sbp': sbp,
                'dbp': dbp,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': sbp_abnormal or dbp_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_respiratory_rate(text: str) -> List[Dict]:
    """Extract respiratory rate values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in RR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
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

            is_abnormal = value < 12 or value > 24

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_spo2(text: str) -> List[Dict]:
    """Extract SpO2 values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in SPO2_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
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

            is_abnormal = value < 92

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_temperature(text: str) -> List[Dict]:
    """
    Extract temperature values from text.
    All values normalized to Celsius.
    """
    results = []
    seen_positions = set()

    for pattern, confidence, tier in TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
                units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
            except (ValueError, IndexError, AttributeError):
                continue

            # Auto-detect unit from value
            if units is None:
                if value > 50:
                    units = 'F'
                else:
                    units = 'C'

            # Validate and convert
            if units == 'F':
                min_val, max_val = VALID_RANGES['TEMP_F']
                if not (min_val <= value <= max_val):
                    continue
                value = round((value - 32) * 5 / 9, 1)
                units = 'C'
            else:
                min_val, max_val = VALID_RANGES['TEMP_C']
                if not (min_val <= value <= max_val):
                    continue
                value = round(value, 1)

            is_abnormal = value < 36 or value > 38.5

            results.append({
                'value': value,
                'units': 'C',
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_all_vitals(text: str) -> Dict[str, List[Dict]]:
    """
    Extract all core vital types from text.

    Returns:
        Dict with keys: HR, BP, RR, SPO2, TEMP
    """
    return {
        'HR': extract_heart_rate(text),
        'BP': extract_blood_pressure(text),
        'RR': extract_respiratory_rate(text),
        'SPO2': extract_spo2(text),
        'TEMP': extract_temperature(text),
    }
```

**Step 4: Run tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_extractor.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_extractor.py
git add module_3_vitals_processing/tests/test_unified_extractor.py
git commit -m "feat(vitals): add unified_extractor.py with core extraction logic

- Add check_negation() with 8 negation patterns
- Add is_in_skip_section() for allergy/med/history filtering
- Add extract_heart_rate() with deduplication and validation
- Add extract_blood_pressure() with swap and pulse pressure validation
- Add extract_respiratory_rate(), extract_spo2()
- Add extract_temperature() with F→C normalization
- Add extract_all_vitals() convenience function
- Comprehensive tests for all extraction functions"
```

---

## Task 6: Add Supplemental Extraction Functions

**Files:**
- Modify: `module_3_vitals_processing/extractors/unified_extractor.py`
- Modify: `module_3_vitals_processing/tests/test_unified_extractor.py`

**Step 1: Write failing tests for supplemental extraction**

Add to `module_3_vitals_processing/tests/test_unified_extractor.py`:

```python
class TestExtractO2Flow:
    """Test O2 flow rate extraction."""

    def test_basic_flow(self):
        from extractors.unified_extractor import extract_o2_flow
        results = extract_o2_flow("on 2L NC")
        assert len(results) >= 1
        assert results[0]['value'] == 2

    def test_high_flow(self):
        from extractors.unified_extractor import extract_o2_flow
        results = extract_o2_flow("40L HFNC")
        assert len(results) >= 1
        assert results[0]['value'] == 40

    def test_range_validation(self):
        from extractors.unified_extractor import extract_o2_flow
        # 100L is invalid
        results = extract_o2_flow("100L NC")
        assert len(results) == 0


class TestExtractO2Device:
    """Test O2 device extraction."""

    def test_nasal_cannula(self):
        from extractors.unified_extractor import extract_o2_device
        results = extract_o2_device("on nasal cannula")
        assert len(results) >= 1

    def test_room_air(self):
        from extractors.unified_extractor import extract_o2_device
        results = extract_o2_device("on room air")
        assert len(results) >= 1


class TestExtractBMI:
    """Test BMI extraction."""

    def test_basic_bmi(self):
        from extractors.unified_extractor import extract_bmi
        results = extract_bmi("BMI: 24.5")
        assert len(results) >= 1
        assert results[0]['value'] == 24.5

    def test_range_validation(self):
        from extractors.unified_extractor import extract_bmi
        # BMI of 5 is invalid
        results = extract_bmi("BMI: 5")
        assert len(results) == 0


class TestExtractSupplemental:
    """Test supplemental extraction function."""

    def test_extract_supplemental(self):
        from extractors.unified_extractor import extract_supplemental_vitals
        text = "on 2L NC, BMI 28.3"
        results = extract_supplemental_vitals(text)
        assert 'O2_FLOW' in results
        assert 'O2_DEVICE' in results
        assert 'BMI' in results
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_extractor.py::TestExtractO2Flow -v
```

Expected: FAIL

**Step 3: Add supplemental extraction functions**

Add to `module_3_vitals_processing/extractors/unified_extractor.py`:

```python
def extract_o2_flow(text: str) -> List[Dict]:
    """Extract O2 flow rate values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in O2_FLOW_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['O2_FLOW']
            if not (min_val <= value <= max_val):
                continue

            results.append({
                'value': value,
                'units': 'L/min',
                'confidence': confidence,
                'position': position,
                'tier': tier,
            })
            seen_positions.add(position)

    return results


def extract_o2_device(text: str) -> List[Dict]:
    """Extract O2 device from text (returns string)."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in O2_DEVICE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            device = match.group(1) if match.lastindex >= 1 else match.group(0)

            results.append({
                'value': device.strip(),
                'confidence': confidence,
                'position': position,
                'tier': tier,
            })
            seen_positions.add(position)

    return results


def extract_bmi(text: str) -> List[Dict]:
    """Extract BMI values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in BMI_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['BMI']
            if not (min_val <= value <= max_val):
                continue

            results.append({
                'value': value,
                'units': 'kg/m2',
                'confidence': confidence,
                'position': position,
                'tier': tier,
            })
            seen_positions.add(position)

    return results


def extract_supplemental_vitals(text: str) -> Dict[str, List[Dict]]:
    """
    Extract supplemental vitals from text.

    Returns:
        Dict with keys: O2_FLOW, O2_DEVICE, BMI
    """
    return {
        'O2_FLOW': extract_o2_flow(text),
        'O2_DEVICE': extract_o2_device(text),
        'BMI': extract_bmi(text),
    }
```

**Step 4: Run tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_extractor.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/extractors/unified_extractor.py
git add module_3_vitals_processing/tests/test_unified_extractor.py
git commit -m "feat(vitals): add supplemental extraction (O2 flow, device, BMI)

- Add extract_o2_flow() with L/min validation
- Add extract_o2_device() returning device string
- Add extract_bmi() with range validation
- Add extract_supplemental_vitals() convenience function"
```

---

## Task 7: Refactor hnp_extractor.py to Thin Wrapper

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_extractor.py`
- Run existing tests to verify backward compatibility

**Step 1: Run existing tests to establish baseline**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_hnp_extractor.py -v --tb=short
```

Record: X tests pass

**Step 2: Create backup and refactor hnp_extractor.py**

```bash
cp module_3_vitals_processing/extractors/hnp_extractor.py module_3_vitals_processing/extractors/hnp_extractor.py.bak
```

Replace `module_3_vitals_processing/extractors/hnp_extractor.py` with:

```python
"""Extract vital signs from Hnp.txt (H&P notes) - Thin wrapper."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .unified_extractor import (
    extract_heart_rate, extract_blood_pressure,
    extract_respiratory_rate, extract_spo2, extract_temperature,
    extract_all_vitals, extract_supplemental_vitals,
    check_negation
)
from .hnp_patterns import (
    SECTION_PATTERNS, TIMESTAMP_PATTERNS,
    DEFAULT_TIMESTAMP_OFFSET, HNP_COLUMNS
)


def identify_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """Identify clinical sections in note text."""
    sections = {}
    for section_name, (pattern, _offset) in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]
    return sections


def extract_timestamp(
    text: str,
    section: str,
    report_datetime: datetime
) -> Tuple[datetime, str, float]:
    """Extract explicit timestamp or estimate from section context."""
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)

                for fmt in ['%m/%d/%Y', '%m/%d/%y']:
                    try:
                        date_part = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

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

    if section in SECTION_PATTERNS:
        _, offset = SECTION_PATTERNS[section]
    else:
        offset = DEFAULT_TIMESTAMP_OFFSET

    estimated_ts = report_datetime + timedelta(hours=offset)
    return estimated_ts, 'estimated', float(offset)


def process_hnp_row(row: pd.Series) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single H&P note row.

    Returns:
        Tuple of (core_vitals, supplemental_vitals)
    """
    text = row.get('Report_Text')
    if not text or pd.isna(text):
        return [], []

    empi = str(row.get('EMPI', ''))
    report_number = str(row.get('Report_Number', ''))

    report_dt_str = row.get('Report_Date_Time', '')
    try:
        report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %I:%M:%S %p')
    except (ValueError, TypeError):
        try:
            report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            report_datetime = datetime.now()

    core_results = []
    supplemental_results = []

    sections = identify_sections(text)
    if not sections:
        sections = {'other': text}

    for section_name, section_text in sections.items():
        timestamp, ts_source, ts_offset = extract_timestamp(
            section_text, section_name, report_datetime
        )

        base_record = {
            'EMPI': empi,
            'timestamp': timestamp,
            'timestamp_source': ts_source,
            'timestamp_offset_hours': ts_offset,
            'source': 'hnp',
            'extraction_context': section_name,
            'report_number': report_number,
            'report_date_time': report_datetime,
        }

        # Extract core vitals using unified extractor
        for hr in extract_heart_rate(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'HR',
                'value': hr['value'],
                'units': 'bpm',
                'confidence': hr['confidence'],
                'is_flagged_abnormal': hr['is_flagged_abnormal'],
            })

        for bp in extract_blood_pressure(section_text):
            for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
                core_results.append({
                    **base_record,
                    'vital_type': vital_type,
                    'value': value,
                    'units': 'mmHg',
                    'confidence': bp['confidence'],
                    'is_flagged_abnormal': bp['is_flagged_abnormal'],
                })

        for rr in extract_respiratory_rate(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'RR',
                'value': rr['value'],
                'units': 'breaths/min',
                'confidence': rr['confidence'],
                'is_flagged_abnormal': rr['is_flagged_abnormal'],
            })

        for spo2 in extract_spo2(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'SPO2',
                'value': spo2['value'],
                'units': '%',
                'confidence': spo2['confidence'],
                'is_flagged_abnormal': spo2['is_flagged_abnormal'],
            })

        for temp in extract_temperature(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'TEMP',
                'value': temp['value'],
                'units': temp['units'],
                'confidence': temp['confidence'],
                'is_flagged_abnormal': temp['is_flagged_abnormal'],
            })

    # Extract supplemental from full text
    supplemental = extract_supplemental_vitals(text)

    base_supplemental = {
        'EMPI': empi,
        'timestamp': report_datetime,
        'source': 'hnp',
        'report_number': report_number,
    }

    for o2_flow in supplemental['O2_FLOW']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'O2_FLOW',
            'value': o2_flow['value'],
            'units': o2_flow['units'],
            'confidence': o2_flow['confidence'],
        })

    for o2_device in supplemental['O2_DEVICE']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'O2_DEVICE',
            'value': o2_device['value'],
            'units': None,
            'confidence': o2_device['confidence'],
        })

    for bmi in supplemental['BMI']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'BMI',
            'value': bmi['value'],
            'units': bmi['units'],
            'confidence': bmi['confidence'],
        })

    return core_results, supplemental_results


def _process_chunk(chunk: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Process a chunk of rows."""
    core_results = []
    supplemental_results = []
    for _, row in chunk.iterrows():
        core, supp = process_hnp_row(row)
        core_results.extend(core)
        supplemental_results.extend(supp)
    return core_results, supplemental_results


def extract_hnp_vitals(
    input_path: str,
    output_path: str,
    supplemental_path: Optional[str] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Extract vital signs from Hnp.txt file.

    Args:
        input_path: Path to Hnp.txt file
        output_path: Path for core vitals parquet
        supplemental_path: Path for supplemental vitals parquet (optional)
        n_workers: Number of parallel workers
        chunk_size: Rows per chunk

    Returns:
        DataFrame with core extracted vitals
    """
    if n_workers is None:
        n_workers = cpu_count()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if supplemental_path:
        Path(supplemental_path).parent.mkdir(parents=True, exist_ok=True)

    all_core = []
    all_supplemental = []

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
            chunk_splits = [
                chunk.iloc[i:i + chunk_size // n_workers]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for core, supp in chunk_results:
                all_core.extend(core)
                all_supplemental.extend(supp)
        else:
            core, supp = _process_chunk(chunk)
            all_core.extend(core)
            all_supplemental.extend(supp)

        print(f"Processed chunk {chunks_processed}, core: {len(all_core)}, supplemental: {len(all_supplemental)}")

    # Create and save DataFrames
    core_columns = [
        'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
        'vital_type', 'value', 'units', 'source', 'extraction_context',
        'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time'
    ]

    if all_core:
        df_core = pd.DataFrame(all_core)
    else:
        df_core = pd.DataFrame(columns=core_columns)

    df_core.to_parquet(output_path, index=False)
    print(f"Core vitals saved to: {output_path}")

    if supplemental_path and all_supplemental:
        df_supp = pd.DataFrame(all_supplemental)
        df_supp.to_parquet(supplemental_path, index=False)
        print(f"Supplemental vitals saved to: {supplemental_path}")

    return df_core


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description='Extract vital signs from Hnp.txt')
    parser.add_argument('-i', '--input', default=str(DATA_DIR / 'Hnp.txt'))
    parser.add_argument('-o', '--output', default=str(OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'))
    parser.add_argument('-s', '--supplemental', default=str(OUTPUT_DIR / 'discovery' / 'hnp_supplemental.parquet'))
    parser.add_argument('-w', '--workers', type=int, default=None)
    parser.add_argument('-c', '--chunk-size', type=int, default=10000)

    args = parser.parse_args()

    extract_hnp_vitals(
        args.input,
        args.output,
        supplemental_path=args.supplemental,
        n_workers=args.workers,
        chunk_size=args.chunk_size
    )
```

**Step 3: Run existing tests to verify backward compatibility**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_hnp_extractor.py -v --tb=short
```

Expected: Same number of tests pass as baseline (or more)

**Step 4: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_extractor.py
git commit -m "refactor(vitals): convert hnp_extractor.py to thin wrapper

- Import extraction functions from unified_extractor
- Keep section identification and timestamp logic
- Add supplemental vitals extraction (O2, BMI)
- Maintain backward compatibility with existing tests
- Reduce from ~680 lines to ~250 lines"
```

---

## Task 8: Refactor prg_extractor.py to Thin Wrapper

**Files:**
- Modify: `module_3_vitals_processing/extractors/prg_extractor.py`
- Run existing tests to verify backward compatibility

**Step 1: Run existing tests to establish baseline**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py -v --tb=short
```

Record: X tests pass

**Step 2: Create backup and refactor prg_extractor.py**

```bash
cp module_3_vitals_processing/extractors/prg_extractor.py module_3_vitals_processing/extractors/prg_extractor.py.bak
```

Replace `module_3_vitals_processing/extractors/prg_extractor.py` with:

```python
"""Extract vital signs from Prg.txt (Progress Notes) - Thin wrapper."""
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd

from .unified_extractor import (
    extract_heart_rate, extract_blood_pressure,
    extract_respiratory_rate, extract_spo2, extract_temperature,
    extract_supplemental_vitals, is_in_skip_section
)
from .prg_patterns import PRG_COLUMNS, PRG_TEMP_PATTERNS, TEMP_METHOD_MAP
from .unified_patterns import VALID_RANGES


@dataclass
class ExtractionCheckpoint:
    """Track extraction progress."""
    input_path: str
    output_path: str
    rows_processed: int
    chunks_completed: int
    records_extracted: int
    started_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ExtractionCheckpoint':
        data = data.copy()
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


CHECKPOINT_FILE = "prg_extraction_checkpoint.json"
CHECKPOINT_INTERVAL = 5


def save_checkpoint(checkpoint: ExtractionCheckpoint, output_dir: Path) -> None:
    path = output_dir / CHECKPOINT_FILE
    with open(path, 'w') as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


def load_checkpoint(output_dir: Path) -> Optional[ExtractionCheckpoint]:
    path = output_dir / CHECKPOINT_FILE
    if path.exists():
        with open(path) as f:
            return ExtractionCheckpoint.from_dict(json.load(f))
    return None


def extract_temperature_with_method(text: str) -> List[Dict]:
    """Extract temperature with measurement method (PRG-specific)."""
    results = []
    seen_positions = set()

    # Try PRG-specific patterns first (with method capture)
    for pattern, confidence in PRG_TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            try:
                value = float(match.group(1))
                units = match.group(2).upper() if match.lastindex >= 2 else None
                raw_method = match.group(3).lower() if match.lastindex >= 3 else None
            except (ValueError, IndexError, AttributeError):
                continue

            method = TEMP_METHOD_MAP.get(raw_method) if raw_method else None

            if units is None:
                units = 'F' if value > 50 else 'C'

            if units == 'F':
                min_val, max_val = VALID_RANGES['TEMP_F']
                if not (min_val <= value <= max_val):
                    continue
                value = round((value - 32) * 5 / 9, 1)
                units = 'C'
            else:
                min_val, max_val = VALID_RANGES['TEMP_C']
                if not (min_val <= value <= max_val):
                    continue
                value = round(value, 1)

            results.append({
                'value': value,
                'units': 'C',
                'method': method,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': value < 36 or value > 38.5,
            })
            seen_positions.add(position)

    # Fall back to unified patterns if no PRG matches
    if not results:
        for temp in extract_temperature(text):
            # Check context for method
            context_end = min(temp['position'] + 50, len(text))
            context = text[temp['position']:context_end].lower()
            method = None
            for method_str, canonical in TEMP_METHOD_MAP.items():
                if method_str in context:
                    method = canonical
                    break

            results.append({
                **temp,
                'method': method,
            })

    return results


def process_prg_row(row: pd.Series) -> Tuple[List[Dict], List[Dict]]:
    """Process a single progress note row."""
    text = row.get('Report_Text')
    if not text or pd.isna(text):
        return [], []

    empi = str(row.get('EMPI', ''))
    report_number = str(row.get('Report_Number', ''))

    report_dt_str = row.get('Report_Date_Time', '')
    try:
        report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %I:%M:%S %p')
    except (ValueError, TypeError):
        try:
            report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            report_datetime = datetime.now()

    base_record = {
        'EMPI': empi,
        'timestamp': report_datetime,
        'timestamp_source': 'estimated',
        'timestamp_offset_hours': 0.0,
        'source': 'prg',
        'extraction_context': 'full_text',
        'report_number': report_number,
        'report_date_time': report_datetime,
    }

    core_results = []
    supplemental_results = []

    # Extract using unified extractor
    for hr in extract_heart_rate(text):
        core_results.append({
            **base_record,
            'vital_type': 'HR',
            'value': hr['value'],
            'units': 'bpm',
            'confidence': hr['confidence'],
            'is_flagged_abnormal': hr['is_flagged_abnormal'],
            'temp_method': None,
        })

    for bp in extract_blood_pressure(text):
        for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
            core_results.append({
                **base_record,
                'vital_type': vital_type,
                'value': value,
                'units': 'mmHg',
                'confidence': bp['confidence'],
                'is_flagged_abnormal': bp['is_flagged_abnormal'],
                'temp_method': None,
            })

    for rr in extract_respiratory_rate(text):
        core_results.append({
            **base_record,
            'vital_type': 'RR',
            'value': rr['value'],
            'units': 'breaths/min',
            'confidence': rr['confidence'],
            'is_flagged_abnormal': rr['is_flagged_abnormal'],
            'temp_method': None,
        })

    for spo2 in extract_spo2(text):
        core_results.append({
            **base_record,
            'vital_type': 'SPO2',
            'value': spo2['value'],
            'units': '%',
            'confidence': spo2['confidence'],
            'is_flagged_abnormal': spo2['is_flagged_abnormal'],
            'temp_method': None,
        })

    for temp in extract_temperature_with_method(text):
        core_results.append({
            **base_record,
            'vital_type': 'TEMP',
            'value': temp['value'],
            'units': temp['units'],
            'confidence': temp['confidence'],
            'is_flagged_abnormal': temp['is_flagged_abnormal'],
            'temp_method': temp.get('method'),
        })

    # Extract supplemental
    supplemental = extract_supplemental_vitals(text)

    base_supp = {
        'EMPI': empi,
        'timestamp': report_datetime,
        'source': 'prg',
        'report_number': report_number,
    }

    for o2_flow in supplemental['O2_FLOW']:
        supplemental_results.append({
            **base_supp,
            'vital_type': 'O2_FLOW',
            'value': o2_flow['value'],
            'units': o2_flow['units'],
            'confidence': o2_flow['confidence'],
        })

    for o2_device in supplemental['O2_DEVICE']:
        supplemental_results.append({
            **base_supp,
            'vital_type': 'O2_DEVICE',
            'value': o2_device['value'],
            'units': None,
            'confidence': o2_device['confidence'],
        })

    for bmi in supplemental['BMI']:
        supplemental_results.append({
            **base_supp,
            'vital_type': 'BMI',
            'value': bmi['value'],
            'units': bmi['units'],
            'confidence': bmi['confidence'],
        })

    return core_results, supplemental_results


def _process_chunk(chunk: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Process a chunk of rows."""
    core_results = []
    supplemental_results = []
    for _, row in chunk.iterrows():
        core, supp = process_prg_row(row)
        core_results.extend(core)
        supplemental_results.extend(supp)
    return core_results, supplemental_results


def extract_prg_vitals(
    input_path: str,
    output_path: str,
    supplemental_path: Optional[str] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000,
    resume: bool = True
) -> pd.DataFrame:
    """Extract vital signs from Prg.txt with checkpointing."""
    if n_workers is None:
        n_workers = cpu_count()

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    if supplemental_path:
        Path(supplemental_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(output_dir) if resume else None
    skip_rows = checkpoint.rows_processed if checkpoint else 0

    if checkpoint:
        print(f"Resuming from row {skip_rows}")
        chunks_completed = checkpoint.chunks_completed
        started_at = checkpoint.started_at
    else:
        chunks_completed = 0
        started_at = datetime.now()

    all_core = []
    all_supplemental = []

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
            chunk_splits = [
                chunk.iloc[i:i + max(1, chunk_size // n_workers)]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for core, supp in chunk_results:
                all_core.extend(core)
                all_supplemental.extend(supp)
        else:
            core, supp = _process_chunk(chunk)
            all_core.extend(core)
            all_supplemental.extend(supp)

        rows_processed = skip_rows + (chunks_completed * chunk_size)

        if chunks_completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = ExtractionCheckpoint(
                input_path=input_path,
                output_path=output_path,
                rows_processed=rows_processed,
                chunks_completed=chunks_completed,
                records_extracted=len(all_core),
                started_at=started_at,
                updated_at=datetime.now(),
            )
            save_checkpoint(checkpoint, output_dir)
            print(f"Checkpoint at chunk {chunks_completed}")

        print(f"Chunk {chunks_completed}, core: {len(all_core)}, supplemental: {len(all_supplemental)}")

    # Create DataFrames
    core_columns = [
        'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
        'vital_type', 'value', 'units', 'source', 'extraction_context',
        'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time',
        'temp_method'
    ]

    if all_core:
        df_core = pd.DataFrame(all_core)
    else:
        df_core = pd.DataFrame(columns=core_columns)

    df_core.to_parquet(output_path, index=False)
    print(f"Core vitals saved to: {output_path}")

    if supplemental_path and all_supplemental:
        df_supp = pd.DataFrame(all_supplemental)
        df_supp.to_parquet(supplemental_path, index=False)
        print(f"Supplemental saved to: {supplemental_path}")

    # Remove checkpoint on success
    checkpoint_path = output_dir / CHECKPOINT_FILE
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return df_core


def main():
    """CLI entry point."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import PRG_INPUT_PATH, PRG_OUTPUT_PATH

    parser = argparse.ArgumentParser(description='Extract vitals from Prg.txt')
    parser.add_argument('-i', '--input', default=str(PRG_INPUT_PATH))
    parser.add_argument('-o', '--output', default=str(PRG_OUTPUT_PATH))
    parser.add_argument('-s', '--supplemental', default=None)
    parser.add_argument('-w', '--workers', type=int, default=None)
    parser.add_argument('-c', '--chunk-size', type=int, default=10000)
    parser.add_argument('--no-resume', action='store_true')

    args = parser.parse_args()

    if args.supplemental is None:
        args.supplemental = str(Path(args.output).parent / 'prg_supplemental.parquet')

    extract_prg_vitals(
        args.input,
        args.output,
        supplemental_path=args.supplemental,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
```

**Step 3: Run existing tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_prg_extractor.py -v --tb=short
```

Expected: Same number of tests pass as baseline (or more)

**Step 4: Commit**

```bash
git add module_3_vitals_processing/extractors/prg_extractor.py
git commit -m "refactor(vitals): convert prg_extractor.py to thin wrapper

- Import extraction from unified_extractor
- Keep checkpointing and temp method logic
- Add supplemental vitals extraction (O2, BMI)
- Maintain backward compatibility with existing tests
- Reduce from ~570 lines to ~300 lines"
```

---

## Task 9: Run Full Test Suite and Verify

**Files:**
- All test files

**Step 1: Run all unified tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_unified_patterns.py module_3_vitals_processing/tests/test_unified_extractor.py module_3_vitals_processing/tests/test_supplemental_patterns.py -v
```

Expected: All new tests PASS

**Step 2: Run all existing extractor tests**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_hnp_extractor.py module_3_vitals_processing/tests/test_prg_extractor.py -v
```

Expected: All existing tests PASS

**Step 3: Run full module test suite**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/ -v --tb=short
```

Expected: All tests PASS

**Step 4: Commit verification**

```bash
git add -A
git commit -m "test(vitals): verify all tests pass after unified extraction refactor

All 337+ tests passing:
- New unified pattern tests: ~20
- New unified extractor tests: ~25
- Existing hnp_extractor tests: 74
- Existing prg_extractor tests: 61
- All other module tests: unchanged"
```

---

## Task 10: Update hnp_patterns.py and prg_patterns.py with Deprecation Notice

**Files:**
- Modify: `module_3_vitals_processing/extractors/hnp_patterns.py`
- Modify: `module_3_vitals_processing/extractors/prg_patterns.py`

**Step 1: Add deprecation notice to hnp_patterns.py**

Add at top of `module_3_vitals_processing/extractors/hnp_patterns.py`:

```python
"""
Regex patterns and constants for Hnp.txt extraction.

DEPRECATED: Pattern definitions have moved to unified_patterns.py.
This file is kept for backward compatibility with existing tests.
New code should import from unified_patterns.py instead.
"""
import warnings

warnings.warn(
    "hnp_patterns.py is deprecated. Use unified_patterns.py instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Step 2: Add deprecation notice to prg_patterns.py**

Add at top of `module_3_vitals_processing/extractors/prg_patterns.py`:

```python
"""
Regex patterns and constants for Prg.txt extraction.

DEPRECATED: Pattern definitions have moved to unified_patterns.py.
This file is kept for backward compatibility.
PRG-specific patterns (TEMP_METHOD_MAP) are still used by prg_extractor.
"""
import warnings

warnings.warn(
    "prg_patterns.py is partially deprecated. Core patterns moved to unified_patterns.py.",
    DeprecationWarning,
    stacklevel=2
)
```

**Step 3: Commit**

```bash
git add module_3_vitals_processing/extractors/hnp_patterns.py
git add module_3_vitals_processing/extractors/prg_patterns.py
git commit -m "docs(vitals): add deprecation notices to old pattern files

- hnp_patterns.py: fully deprecated, kept for backward compat
- prg_patterns.py: partially deprecated, TEMP_METHOD_MAP still used"
```

---

## Verification Checklist

After completing all tasks:

- [ ] `unified_patterns.py` has ~75 patterns across all vital types
- [ ] `unified_extractor.py` has all extraction functions
- [ ] `hnp_extractor.py` is thin wrapper (~250 lines)
- [ ] `prg_extractor.py` is thin wrapper (~300 lines)
- [ ] All existing tests pass (74 HNP + 61 PRG)
- [ ] All new tests pass (~45 new tests)
- [ ] Deprecation warnings added to old pattern files
- [ ] Supplemental vitals (O2, BMI) extraction working

---

**Document Version:** 1.0
**Created:** 2025-12-12
**Design Reference:** `docs/plans/2025-12-12-unified-vital-extraction-design.md`
