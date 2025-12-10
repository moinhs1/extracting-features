# Phase 1: Layer 1-2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Layers 1-2 of the vitals processing pipeline - canonical records with PE-relative timestamps (Layer 1) and hourly aggregated grid with missing data tensors (Layer 2).

**Architecture:** Three helper modules (unit conversion, QC filters, temporal alignment) plus two builder modules (Layer 1, Layer 2). TDD approach with helpers first, then builders that compose them. Layer 1 merges three extraction sources into unified schema; Layer 2 aggregates to hourly grid and generates HDF5 tensors.

**Tech Stack:** Python 3.x, pandas, pyarrow, h5py, numpy, pytest

---

## Input Files

| File | Path | Schema |
|------|------|--------|
| PHY vitals | `outputs/discovery/phy_vitals_raw.parquet` | EMPI, timestamp, vital_type, value, units, source, encounter_type, encounter_number |
| HNP vitals | `outputs/discovery/hnp_vitals_raw.parquet` | EMPI, timestamp, timestamp_source, timestamp_offset_hours, vital_type, value, units, source, extraction_context, confidence, is_flagged_abnormal, report_number, report_date_time |
| PRG vitals | `outputs/discovery/prg_vitals_raw.parquet` | Same as HNP + temp_method |
| Patient timelines | `module_1_core_infrastructure/outputs/patient_timelines.pkl` | Dict of PatientTimeline dataclasses with time_zero (PE index) |

## Output Files

| Layer | File | Description |
|-------|------|-------------|
| 1 | `outputs/layer1/canonical_vitals.parquet` | Unified schema with PE-relative timestamps |
| 2 | `outputs/layer2/hourly_grid.parquet` | Tabular hourly summaries |
| 2 | `outputs/layer2/hourly_tensors.h5` | HDF5 triple-component tensors |

---

## Task 1: Create processing/ Directory Structure

**Files:**
- Create: `processing/__init__.py`

**Step 1: Create directory and init file**

```bash
mkdir -p /home/moin/TDA_11_25/module_3_vitals_processing/processing
mkdir -p /home/moin/TDA_11_25/module_3_vitals_processing/outputs/layer1
mkdir -p /home/moin/TDA_11_25/module_3_vitals_processing/outputs/layer2
```

**Step 2: Create init file**

```python
"""Processing modules for vitals pipeline layers 1-5."""
```

**Step 3: Commit**

```bash
git add processing/__init__.py
git commit -m "chore: create processing module structure"
```

---

## Task 2: Unit Converter - Temperature Conversion Test

**Files:**
- Create: `tests/test_unit_converter.py`
- Create: `processing/unit_converter.py`

**Step 1: Write the failing test**

```python
"""Tests for unit_converter module."""
import pytest
from processing.unit_converter import fahrenheit_to_celsius


class TestTemperatureConversion:
    """Tests for temperature conversion."""

    def test_fahrenheit_to_celsius_98_6(self):
        """98.6°F is 37.0°C (normal body temp)."""
        result = fahrenheit_to_celsius(98.6)
        assert abs(result - 37.0) < 0.1

    def test_fahrenheit_to_celsius_100_4(self):
        """100.4°F is 38.0°C (fever threshold)."""
        result = fahrenheit_to_celsius(100.4)
        assert abs(result - 38.0) < 0.1

    def test_fahrenheit_to_celsius_freezing(self):
        """32°F is 0°C."""
        result = fahrenheit_to_celsius(32.0)
        assert abs(result - 0.0) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_unit_converter.py::TestTemperatureConversion -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
"""Unit conversion functions for vital signs."""


def fahrenheit_to_celsius(temp_f: float) -> float:
    """Convert Fahrenheit to Celsius.

    Args:
        temp_f: Temperature in Fahrenheit

    Returns:
        Temperature in Celsius
    """
    return (temp_f - 32) * 5 / 9
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_unit_converter.py::TestTemperatureConversion -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_unit_converter.py module_3_vitals_processing/processing/unit_converter.py
git commit -m "feat(3.4): add fahrenheit_to_celsius conversion"
```

---

## Task 3: Unit Converter - Detect and Normalize Temperature Units

**Files:**
- Modify: `tests/test_unit_converter.py`
- Modify: `processing/unit_converter.py`

**Step 1: Write the failing test**

```python
class TestNormalizeTemperature:
    """Tests for temperature normalization to Celsius."""

    def test_normalize_celsius_passthrough(self):
        """Celsius values pass through unchanged."""
        from processing.unit_converter import normalize_temperature
        result = normalize_temperature(37.0, "C")
        assert result == 37.0

    def test_normalize_fahrenheit_converts(self):
        """Fahrenheit values are converted."""
        from processing.unit_converter import normalize_temperature
        result = normalize_temperature(98.6, "F")
        assert abs(result - 37.0) < 0.1

    def test_normalize_infers_fahrenheit_high_value(self):
        """Values >50 with unknown units assumed Fahrenheit."""
        from processing.unit_converter import normalize_temperature
        result = normalize_temperature(98.6, None)
        assert abs(result - 37.0) < 0.1

    def test_normalize_infers_celsius_low_value(self):
        """Values <=50 with unknown units assumed Celsius."""
        from processing.unit_converter import normalize_temperature
        result = normalize_temperature(37.0, None)
        assert result == 37.0

    def test_normalize_handles_degree_symbol_units(self):
        """Handles '°C', '°F', 'deg C', 'deg F' variants."""
        from processing.unit_converter import normalize_temperature
        assert normalize_temperature(37.0, "°C") == 37.0
        assert abs(normalize_temperature(98.6, "°F") - 37.0) < 0.1
        assert normalize_temperature(37.0, "deg C") == 37.0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_unit_converter.py::TestNormalizeTemperature -v`
Expected: FAIL with "ImportError: cannot import name 'normalize_temperature'"

**Step 3: Write minimal implementation**

Add to `processing/unit_converter.py`:

```python
from typing import Optional


def normalize_temperature(value: float, units: Optional[str]) -> float:
    """Normalize temperature to Celsius.

    Args:
        value: Temperature value
        units: Unit string (C, F, °C, °F, deg C, deg F, or None)

    Returns:
        Temperature in Celsius

    Note:
        If units is None, infers based on value:
        - >50 assumed Fahrenheit
        - <=50 assumed Celsius
    """
    if units is None:
        # Infer from value range
        if value > 50:
            return fahrenheit_to_celsius(value)
        return value

    # Normalize unit string
    units_lower = units.lower().replace("°", "").replace("deg ", "").strip()

    if units_lower in ("f", "fahrenheit"):
        return fahrenheit_to_celsius(value)

    # Default: assume Celsius
    return value
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_unit_converter.py::TestNormalizeTemperature -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_unit_converter.py module_3_vitals_processing/processing/unit_converter.py
git commit -m "feat(3.4): add normalize_temperature with unit inference"
```

---

## Task 4: QC Filters - Physiological Range Validation

**Files:**
- Create: `tests/test_qc_filters.py`
- Create: `processing/qc_filters.py`

**Step 1: Write the failing test**

```python
"""Tests for qc_filters module."""
import pytest
from processing.qc_filters import is_physiologically_valid, VALID_RANGES


class TestPhysiologicalRanges:
    """Tests for physiological range validation."""

    def test_valid_hr(self):
        """Normal heart rate is valid."""
        assert is_physiologically_valid("HR", 72) is True

    def test_invalid_hr_too_low(self):
        """HR < 20 is impossible."""
        assert is_physiologically_valid("HR", 15) is False

    def test_invalid_hr_too_high(self):
        """HR > 300 is impossible."""
        assert is_physiologically_valid("HR", 350) is False

    def test_valid_sbp(self):
        """Normal SBP is valid."""
        assert is_physiologically_valid("SBP", 120) is True

    def test_invalid_sbp_too_low(self):
        """SBP < 40 is impossible."""
        assert is_physiologically_valid("SBP", 30) is False

    def test_valid_spo2(self):
        """Normal SpO2 is valid."""
        assert is_physiologically_valid("SPO2", 98) is True

    def test_invalid_spo2_over_100(self):
        """SpO2 > 100% is impossible."""
        assert is_physiologically_valid("SPO2", 105) is False

    def test_valid_temp_celsius(self):
        """Normal temp in Celsius is valid."""
        assert is_physiologically_valid("TEMP", 37.0) is True

    def test_invalid_temp_too_low(self):
        """Temp < 30°C is impossible (hypothermia death)."""
        assert is_physiologically_valid("TEMP", 25) is False

    def test_valid_ranges_defined(self):
        """All 7 core vitals have defined ranges."""
        expected_vitals = {"HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"}
        assert expected_vitals <= set(VALID_RANGES.keys())
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestPhysiologicalRanges -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""Quality control filters for vital signs."""
from typing import Dict, Tuple

# Physiological ranges: (min_valid, max_valid)
# Values outside these are impossible/data errors
VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "HR": (20, 300),      # Heart rate: 20-300 bpm
    "SBP": (40, 300),     # Systolic BP: 40-300 mmHg
    "DBP": (20, 200),     # Diastolic BP: 20-200 mmHg
    "MAP": (30, 200),     # Mean arterial pressure: 30-200 mmHg
    "RR": (4, 60),        # Respiratory rate: 4-60 breaths/min
    "SPO2": (50, 100),    # Oxygen saturation: 50-100%
    "TEMP": (30, 45),     # Temperature: 30-45°C
}

# Abnormal thresholds for flagging (not removal)
ABNORMAL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "HR": (60, 100),      # Normal: 60-100 bpm
    "SBP": (90, 180),     # Normal: 90-180 mmHg
    "DBP": (60, 110),     # Normal: 60-110 mmHg
    "MAP": (65, 110),     # Normal: 65-110 mmHg
    "RR": (12, 24),       # Normal: 12-24 breaths/min
    "SPO2": (92, 100),    # Normal: 92-100%
    "TEMP": (36, 38.5),   # Normal: 36-38.5°C
}


def is_physiologically_valid(vital_type: str, value: float) -> bool:
    """Check if vital sign value is within possible physiological range.

    Args:
        vital_type: Type of vital (HR, SBP, DBP, MAP, RR, SPO2, TEMP)
        value: Measurement value

    Returns:
        True if value is physiologically possible, False if impossible
    """
    if vital_type not in VALID_RANGES:
        return True  # Unknown vital type, don't filter

    min_val, max_val = VALID_RANGES[vital_type]
    return min_val <= value <= max_val
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestPhysiologicalRanges -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_qc_filters.py module_3_vitals_processing/processing/qc_filters.py
git commit -m "feat(3.4): add physiological range validation"
```

---

## Task 5: QC Filters - Abnormal Flagging

**Files:**
- Modify: `tests/test_qc_filters.py`
- Modify: `processing/qc_filters.py`

**Step 1: Write the failing test**

```python
class TestAbnormalFlagging:
    """Tests for abnormal value flagging."""

    def test_normal_hr_not_flagged(self):
        """HR 72 is normal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 72) is False

    def test_tachycardia_flagged(self):
        """HR > 100 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 110) is True

    def test_bradycardia_flagged(self):
        """HR < 60 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("HR", 50) is True

    def test_hypotension_flagged(self):
        """SBP < 90 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("SBP", 85) is True

    def test_hypoxemia_flagged(self):
        """SpO2 < 92 is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("SPO2", 88) is True

    def test_fever_flagged(self):
        """Temp > 38.5°C is abnormal."""
        from processing.qc_filters import is_abnormal
        assert is_abnormal("TEMP", 39.5) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestAbnormalFlagging -v`
Expected: FAIL with "ImportError: cannot import name 'is_abnormal'"

**Step 3: Write minimal implementation**

Add to `processing/qc_filters.py`:

```python
def is_abnormal(vital_type: str, value: float) -> bool:
    """Check if vital sign value is outside normal clinical range.

    Args:
        vital_type: Type of vital (HR, SBP, DBP, MAP, RR, SPO2, TEMP)
        value: Measurement value

    Returns:
        True if value is clinically abnormal, False if normal
    """
    if vital_type not in ABNORMAL_THRESHOLDS:
        return False  # Unknown vital type, don't flag

    min_normal, max_normal = ABNORMAL_THRESHOLDS[vital_type]
    return value < min_normal or value > max_normal
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestAbnormalFlagging -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_qc_filters.py module_3_vitals_processing/processing/qc_filters.py
git commit -m "feat(3.4): add abnormal value flagging"
```

---

## Task 6: QC Filters - Blood Pressure Consistency Check

**Files:**
- Modify: `tests/test_qc_filters.py`
- Modify: `processing/qc_filters.py`

**Step 1: Write the failing test**

```python
class TestBPConsistency:
    """Tests for blood pressure consistency validation."""

    def test_valid_bp_sbp_greater_than_dbp(self):
        """SBP should be greater than DBP."""
        from processing.qc_filters import is_bp_consistent
        assert is_bp_consistent(120, 80) is True

    def test_invalid_bp_dbp_greater_than_sbp(self):
        """DBP > SBP is inconsistent."""
        from processing.qc_filters import is_bp_consistent
        assert is_bp_consistent(80, 120) is False

    def test_invalid_bp_equal(self):
        """SBP == DBP is inconsistent."""
        from processing.qc_filters import is_bp_consistent
        assert is_bp_consistent(100, 100) is False

    def test_valid_bp_narrow_pulse_pressure(self):
        """Narrow but valid pulse pressure."""
        from processing.qc_filters import is_bp_consistent
        assert is_bp_consistent(100, 90) is True
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestBPConsistency -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/qc_filters.py`:

```python
def is_bp_consistent(sbp: float, dbp: float) -> bool:
    """Check if systolic and diastolic BP values are consistent.

    Args:
        sbp: Systolic blood pressure
        dbp: Diastolic blood pressure

    Returns:
        True if SBP > DBP (physiologically consistent), False otherwise
    """
    return sbp > dbp
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_qc_filters.py::TestBPConsistency -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_qc_filters.py module_3_vitals_processing/processing/qc_filters.py
git commit -m "feat(3.4): add blood pressure consistency check"
```

---

## Task 7: Temporal Aligner - Hours From PE Calculation

**Files:**
- Create: `tests/test_temporal_aligner.py`
- Create: `processing/temporal_aligner.py`

**Step 1: Write the failing test**

```python
"""Tests for temporal_aligner module."""
import pytest
from datetime import datetime, timedelta
from processing.temporal_aligner import calculate_hours_from_pe


class TestHoursFromPE:
    """Tests for PE-relative time calculation."""

    def test_exact_pe_time_is_zero(self):
        """Timestamp at PE index time is hour 0."""
        pe_time = datetime(2023, 6, 15, 10, 30, 0)
        vital_time = datetime(2023, 6, 15, 10, 30, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 0.0

    def test_one_hour_after_pe(self):
        """One hour after PE is +1.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 11, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 1.0

    def test_one_hour_before_pe(self):
        """One hour before PE is -1.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 9, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == -1.0

    def test_30_minutes_is_half_hour(self):
        """30 minutes is 0.5 hours."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 10, 30, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 0.5

    def test_24_hours_before_pe(self):
        """24 hours before PE is -24.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 14, 10, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == -24.0

    def test_720_hours_after_pe(self):
        """30 days after PE is +720.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = pe_time + timedelta(days=30)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 720.0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestHoursFromPE -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""Temporal alignment functions for PE-relative timestamps."""
from datetime import datetime
from typing import Union
import pandas as pd


def calculate_hours_from_pe(
    vital_time: Union[datetime, pd.Timestamp],
    pe_time: Union[datetime, pd.Timestamp]
) -> float:
    """Calculate hours from PE index time.

    Args:
        vital_time: Timestamp of vital measurement
        pe_time: PE index/diagnosis timestamp (time_zero)

    Returns:
        Hours relative to PE (negative = before, positive = after)
    """
    delta = vital_time - pe_time
    return delta.total_seconds() / 3600.0
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestHoursFromPE -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_temporal_aligner.py module_3_vitals_processing/processing/temporal_aligner.py
git commit -m "feat(3.4): add calculate_hours_from_pe function"
```

---

## Task 8: Temporal Aligner - Window Filtering

**Files:**
- Modify: `tests/test_temporal_aligner.py`
- Modify: `processing/temporal_aligner.py`

**Step 1: Write the failing test**

```python
class TestWindowFiltering:
    """Tests for temporal window filtering."""

    def test_within_window(self):
        """Value at hour 0 is within default window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(0.0) is True

    def test_at_window_start(self):
        """Value at -24h is within window (inclusive)."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(-24.0) is True

    def test_at_window_end(self):
        """Value at +720h is within window (inclusive)."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(720.0) is True

    def test_before_window(self):
        """Value at -25h is outside window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(-25.0) is False

    def test_after_window(self):
        """Value at +721h is outside window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(721.0) is False

    def test_custom_window(self):
        """Custom window boundaries work."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(5.0, min_hours=-10, max_hours=10) is True
        assert is_within_window(15.0, min_hours=-10, max_hours=10) is False
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestWindowFiltering -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/temporal_aligner.py`:

```python
# Default temporal window (hours relative to PE)
DEFAULT_WINDOW_MIN = -24   # 24 hours before PE
DEFAULT_WINDOW_MAX = 720   # 30 days after PE


def is_within_window(
    hours_from_pe: float,
    min_hours: float = DEFAULT_WINDOW_MIN,
    max_hours: float = DEFAULT_WINDOW_MAX
) -> bool:
    """Check if timestamp is within analysis window.

    Args:
        hours_from_pe: Hours relative to PE index
        min_hours: Window start (inclusive)
        max_hours: Window end (inclusive)

    Returns:
        True if within window, False otherwise
    """
    return min_hours <= hours_from_pe <= max_hours
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestWindowFiltering -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_temporal_aligner.py module_3_vitals_processing/processing/temporal_aligner.py
git commit -m "feat(3.4): add temporal window filtering"
```

---

## Task 9: Temporal Aligner - Hour Bucket Assignment

**Files:**
- Modify: `tests/test_temporal_aligner.py`
- Modify: `processing/temporal_aligner.py`

**Step 1: Write the failing test**

```python
class TestHourBucket:
    """Tests for hour bucket assignment."""

    def test_hour_zero_bucket(self):
        """Hours 0.0 to 0.99 go in bucket 0."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(0.0) == 0
        assert assign_hour_bucket(0.5) == 0
        assert assign_hour_bucket(0.99) == 0

    def test_hour_one_bucket(self):
        """Hours 1.0 to 1.99 go in bucket 1."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(1.0) == 1
        assert assign_hour_bucket(1.5) == 1

    def test_negative_hour_bucket(self):
        """Negative hours floor correctly."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(-0.5) == -1
        assert assign_hour_bucket(-1.0) == -1
        assert assign_hour_bucket(-1.5) == -2
        assert assign_hour_bucket(-24.0) == -24

    def test_max_hour_bucket(self):
        """Hour 720 is in bucket 720."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(720.0) == 720
        assert assign_hour_bucket(720.5) == 720
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestHourBucket -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/temporal_aligner.py`:

```python
import math


def assign_hour_bucket(hours_from_pe: float) -> int:
    """Assign timestamp to hourly bucket.

    Uses floor for positive hours, floor for negative hours.
    Bucket N contains hours [N, N+1) for N >= 0
    Bucket N contains hours (N-1, N] for N < 0

    Args:
        hours_from_pe: Hours relative to PE index

    Returns:
        Integer hour bucket
    """
    return math.floor(hours_from_pe)
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_temporal_aligner.py::TestHourBucket -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_temporal_aligner.py module_3_vitals_processing/processing/temporal_aligner.py
git commit -m "feat(3.5): add hour bucket assignment"
```

---

## Task 10: Layer 1 Builder - Schema Definition and Basic Structure

**Files:**
- Create: `tests/test_layer1_builder.py`
- Create: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
"""Tests for layer1_builder module."""
import pytest
import pandas as pd
from processing.layer1_builder import LAYER1_SCHEMA, CORE_VITALS


class TestLayer1Schema:
    """Tests for Layer 1 schema definitions."""

    def test_schema_has_required_columns(self):
        """Schema defines all required columns."""
        required = {
            "EMPI", "timestamp", "hours_from_pe", "vital_type",
            "value", "units", "source", "source_detail", "confidence",
            "is_calculated", "is_flagged_abnormal", "report_number"
        }
        assert required <= set(LAYER1_SCHEMA.keys())

    def test_core_vitals_defined(self):
        """Seven core vitals are defined."""
        expected = {"HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"}
        assert set(CORE_VITALS) == expected

    def test_schema_types_are_valid(self):
        """Schema types are valid pandas/numpy types."""
        valid_types = {"str", "datetime64[ns]", "float64", "bool"}
        for col, dtype in LAYER1_SCHEMA.items():
            assert dtype in valid_types, f"{col} has invalid type {dtype}"
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestLayer1Schema -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""Layer 1 Builder: Canonical vital sign records with PE-relative timestamps."""
from typing import Dict

# Core vital signs for Layer 1-5 processing
CORE_VITALS = ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]

# Layer 1 output schema
LAYER1_SCHEMA: Dict[str, str] = {
    "EMPI": "str",
    "timestamp": "datetime64[ns]",
    "hours_from_pe": "float64",
    "vital_type": "str",
    "value": "float64",
    "units": "str",
    "source": "str",
    "source_detail": "str",
    "confidence": "float64",
    "is_calculated": "bool",
    "is_flagged_abnormal": "bool",
    "report_number": "str",
}
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestLayer1Schema -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): define Layer 1 schema and core vitals"
```

---

## Task 11: Layer 1 Builder - PHY Source Normalizer

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestNormalizePhy:
    """Tests for PHY source normalization."""

    def test_normalize_phy_adds_required_columns(self):
        """PHY normalization adds all Layer 1 columns."""
        from processing.layer1_builder import normalize_phy_source

        # Minimal PHY-like dataframe
        phy_df = pd.DataFrame({
            "EMPI": ["E001", "E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00", "2023-06-15 11:00"]),
            "vital_type": ["HR", "SBP"],
            "value": [72.0, 120.0],
            "units": ["bpm", "mmHg"],
            "source": ["phy", "phy"],
            "encounter_type": ["IP", "IP"],
            "encounter_number": ["ENC001", "ENC001"],
        })

        result = normalize_phy_source(phy_df)

        assert "source_detail" in result.columns
        assert "confidence" in result.columns
        assert "is_calculated" in result.columns
        assert "is_flagged_abnormal" in result.columns
        assert "report_number" in result.columns

    def test_normalize_phy_sets_confidence_1(self):
        """PHY source has confidence = 1.0 (structured data)."""
        from processing.layer1_builder import normalize_phy_source

        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["IP"],
            "encounter_number": ["ENC001"],
        })

        result = normalize_phy_source(phy_df)
        assert result["confidence"].iloc[0] == 1.0

    def test_normalize_phy_maps_encounter_type_to_source_detail(self):
        """encounter_type becomes source_detail."""
        from processing.layer1_builder import normalize_phy_source

        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["Inpatient"],
            "encounter_number": ["ENC001"],
        })

        result = normalize_phy_source(phy_df)
        assert result["source_detail"].iloc[0] == "Inpatient"
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestNormalizePhy -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
import pandas as pd


def normalize_phy_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize PHY extraction output to Layer 1 schema.

    PHY data is structured (flowsheet) data with high confidence.

    Args:
        df: PHY vitals dataframe with columns:
            EMPI, timestamp, vital_type, value, units, source,
            encounter_type, encounter_number

    Returns:
        DataFrame with Layer 1 schema columns
    """
    result = df.copy()

    # Map encounter_type to source_detail
    result["source_detail"] = result.get("encounter_type", "")

    # PHY is structured data, highest confidence
    result["confidence"] = 1.0

    # PHY values are direct measurements, not calculated
    result["is_calculated"] = False

    # Will be set by QC filters later
    result["is_flagged_abnormal"] = False

    # PHY doesn't have report_number, use encounter_number or empty
    result["report_number"] = result.get("encounter_number", "")

    # Select only Layer 1 columns
    output_cols = list(LAYER1_SCHEMA.keys())
    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    return result[output_cols]
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestNormalizePhy -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add PHY source normalizer"
```

---

## Task 12: Layer 1 Builder - HNP/PRG Source Normalizers

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestNormalizeHnpPrg:
    """Tests for HNP/PRG source normalization."""

    def test_normalize_hnp_preserves_confidence(self):
        """HNP normalization preserves extraction confidence."""
        from processing.layer1_builder import normalize_hnp_source

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs_section"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
        })

        result = normalize_hnp_source(hnp_df)
        assert result["confidence"].iloc[0] == 0.85

    def test_normalize_hnp_maps_extraction_context_to_source_detail(self):
        """extraction_context becomes source_detail."""
        from processing.layer1_builder import normalize_hnp_source

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs_section"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
        })

        result = normalize_hnp_source(hnp_df)
        assert result["source_detail"].iloc[0] == "vital_signs_section"

    def test_normalize_prg_preserves_confidence(self):
        """PRG normalization preserves extraction confidence."""
        from processing.layer1_builder import normalize_prg_source

        prg_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["prg"],
            "extraction_context": ["vital_signs_table"],
            "confidence": [0.9],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT002"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
            "temp_method": [None],
        })

        result = normalize_prg_source(prg_df)
        assert result["confidence"].iloc[0] == 0.9
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestNormalizeHnpPrg -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
def normalize_hnp_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize HNP extraction output to Layer 1 schema.

    HNP data is extracted from H&P notes (admission vitals).

    Args:
        df: HNP vitals dataframe

    Returns:
        DataFrame with Layer 1 schema columns
    """
    result = df.copy()

    # Map extraction_context to source_detail
    result["source_detail"] = result.get("extraction_context", "")

    # Confidence already exists from extraction
    if "confidence" not in result.columns:
        result["confidence"] = 0.8  # Default for NLP extraction

    # NLP extractions are not calculated
    result["is_calculated"] = False

    # is_flagged_abnormal may already exist
    if "is_flagged_abnormal" not in result.columns:
        result["is_flagged_abnormal"] = False

    # Select only Layer 1 columns
    output_cols = list(LAYER1_SCHEMA.keys())
    for col in output_cols:
        if col not in result.columns:
            result[col] = None

    return result[output_cols]


def normalize_prg_source(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize PRG extraction output to Layer 1 schema.

    PRG data is extracted from Progress notes (serial inpatient vitals).

    Args:
        df: PRG vitals dataframe

    Returns:
        DataFrame with Layer 1 schema columns
    """
    # PRG has same structure as HNP (plus temp_method which we drop)
    return normalize_hnp_source(df)
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestNormalizeHnpPrg -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add HNP/PRG source normalizers"
```

---

## Task 13: Layer 1 Builder - MAP Calculation

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestMAPCalculation:
    """Tests for Mean Arterial Pressure calculation."""

    def test_calculate_map_formula(self):
        """MAP = DBP + (SBP - DBP) / 3."""
        from processing.layer1_builder import calculate_map
        # Standard BP 120/80
        result = calculate_map(120, 80)
        expected = 80 + (120 - 80) / 3  # 93.33
        assert abs(result - expected) < 0.01

    def test_calculate_map_normal_values(self):
        """Normal MAP is around 70-105."""
        from processing.layer1_builder import calculate_map
        result = calculate_map(120, 80)
        assert 70 <= result <= 105

    def test_generate_calculated_maps(self):
        """Generate MAP rows from SBP/DBP pairs at same timestamp."""
        from processing.layer1_builder import generate_calculated_maps

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E001", "E002"],
            "timestamp": pd.to_datetime([
                "2023-06-15 10:00", "2023-06-15 10:00",  # Same time - pair
                "2023-06-15 11:00",  # SBP only - no pair
                "2023-06-15 10:00",  # Different patient
            ]),
            "vital_type": ["SBP", "DBP", "SBP", "SBP"],
            "value": [120.0, 80.0, 130.0, 140.0],
            "units": ["mmHg", "mmHg", "mmHg", "mmHg"],
            "source": ["phy", "phy", "phy", "phy"],
            "source_detail": ["IP", "IP", "IP", "IP"],
            "confidence": [1.0, 1.0, 1.0, 1.0],
            "is_calculated": [False, False, False, False],
            "is_flagged_abnormal": [False, False, False, False],
            "report_number": ["", "", "", ""],
            "hours_from_pe": [0.0, 0.0, 1.0, 0.0],
        })

        maps = generate_calculated_maps(df)

        # Should generate 1 MAP (E001 at 10:00)
        assert len(maps) == 1
        assert maps["vital_type"].iloc[0] == "MAP"
        assert maps["is_calculated"].iloc[0] is True
        assert abs(maps["value"].iloc[0] - 93.33) < 0.1
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestMAPCalculation -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
def calculate_map(sbp: float, dbp: float) -> float:
    """Calculate Mean Arterial Pressure from SBP and DBP.

    Formula: MAP = DBP + (SBP - DBP) / 3

    Args:
        sbp: Systolic blood pressure
        dbp: Diastolic blood pressure

    Returns:
        Mean arterial pressure
    """
    return dbp + (sbp - dbp) / 3


def generate_calculated_maps(df: pd.DataFrame) -> pd.DataFrame:
    """Generate calculated MAP values from SBP/DBP pairs.

    Finds SBP and DBP measurements at the same timestamp for the same
    patient and generates calculated MAP values.

    Args:
        df: DataFrame with vital measurements

    Returns:
        DataFrame with calculated MAP rows
    """
    # Get SBP and DBP rows
    sbp_df = df[df["vital_type"] == "SBP"].copy()
    dbp_df = df[df["vital_type"] == "DBP"].copy()

    if sbp_df.empty or dbp_df.empty:
        return pd.DataFrame(columns=df.columns)

    # Merge on patient and timestamp
    merged = sbp_df.merge(
        dbp_df[["EMPI", "timestamp", "value"]],
        on=["EMPI", "timestamp"],
        suffixes=("_sbp", "_dbp"),
        how="inner"
    )

    if merged.empty:
        return pd.DataFrame(columns=df.columns)

    # Calculate MAP
    merged["value"] = merged.apply(
        lambda r: calculate_map(r["value_sbp"], r["value_dbp"]),
        axis=1
    )

    # Build MAP rows
    map_df = merged[["EMPI", "timestamp", "hours_from_pe", "source",
                     "source_detail", "confidence", "report_number", "value"]].copy()
    map_df["vital_type"] = "MAP"
    map_df["units"] = "mmHg"
    map_df["is_calculated"] = True
    map_df["is_flagged_abnormal"] = False

    # Reorder columns to match schema
    return map_df[list(LAYER1_SCHEMA.keys())]
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestMAPCalculation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add MAP calculation from SBP/DBP pairs"
```

---

## Task 14: Layer 1 Builder - Patient Timeline Loading

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestPatientTimelineLoading:
    """Tests for patient timeline loading."""

    def test_load_pe_times_returns_dict(self):
        """load_pe_times returns dict mapping EMPI to PE timestamp."""
        from processing.layer1_builder import load_pe_times
        import pickle
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from datetime import datetime

        # Mock PatientTimeline
        mock_timeline = MagicMock()
        mock_timeline.patient_id = "E001"
        mock_timeline.time_zero = datetime(2023, 6, 15, 10, 0, 0)

        mock_timelines = {"E001": mock_timeline}

        with patch("builtins.open", create=True):
            with patch("pickle.load", return_value=mock_timelines):
                result = load_pe_times(Path("/fake/path.pkl"))

        assert isinstance(result, dict)
        assert "E001" in result
        assert result["E001"] == datetime(2023, 6, 15, 10, 0, 0)

    def test_load_pe_times_handles_multiple_patients(self):
        """load_pe_times handles multiple patients."""
        from processing.layer1_builder import load_pe_times
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        from pathlib import Path

        mock_t1 = MagicMock()
        mock_t1.patient_id = "E001"
        mock_t1.time_zero = datetime(2023, 6, 15, 10, 0, 0)

        mock_t2 = MagicMock()
        mock_t2.patient_id = "E002"
        mock_t2.time_zero = datetime(2023, 6, 16, 14, 30, 0)

        mock_timelines = {"E001": mock_t1, "E002": mock_t2}

        with patch("builtins.open", create=True):
            with patch("pickle.load", return_value=mock_timelines):
                result = load_pe_times(Path("/fake/path.pkl"))

        assert len(result) == 2
        assert result["E002"] == datetime(2023, 6, 16, 14, 30, 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestPatientTimelineLoading -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Union


def load_pe_times(timeline_path: Path) -> Dict[str, datetime]:
    """Load PE index times from patient timelines pickle.

    Args:
        timeline_path: Path to patient_timelines.pkl

    Returns:
        Dict mapping EMPI to PE timestamp (time_zero)
    """
    with open(timeline_path, "rb") as f:
        timelines = pickle.load(f)

    pe_times = {}
    for patient_id, timeline in timelines.items():
        pe_times[timeline.patient_id] = timeline.time_zero

    return pe_times
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestPatientTimelineLoading -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add patient timeline loading for PE times"
```

---

## Task 15: Layer 1 Builder - Add PE-Relative Timestamps

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestAddPERelativeTimestamps:
    """Tests for adding PE-relative timestamps to vitals."""

    def test_adds_hours_from_pe_column(self):
        """Adds hours_from_pe column based on PE times."""
        from processing.layer1_builder import add_pe_relative_timestamps
        from datetime import datetime

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E002"],
            "timestamp": pd.to_datetime([
                "2023-06-15 11:00",  # 1 hour after PE
                "2023-06-15 09:00",  # 1 hour before PE
                "2023-06-16 14:30",  # At PE time
            ]),
            "vital_type": ["HR", "HR", "HR"],
            "value": [72.0, 70.0, 80.0],
        })

        pe_times = {
            "E001": datetime(2023, 6, 15, 10, 0, 0),
            "E002": datetime(2023, 6, 16, 14, 30, 0),
        }

        result = add_pe_relative_timestamps(df, pe_times)

        assert "hours_from_pe" in result.columns
        assert result.loc[result["EMPI"] == "E001"].iloc[0]["hours_from_pe"] == 1.0
        assert result.loc[result["EMPI"] == "E001"].iloc[1]["hours_from_pe"] == -1.0
        assert result.loc[result["EMPI"] == "E002"].iloc[0]["hours_from_pe"] == 0.0

    def test_drops_patients_without_pe_time(self):
        """Patients without PE time are dropped."""
        from processing.layer1_builder import add_pe_relative_timestamps
        from datetime import datetime

        df = pd.DataFrame({
            "EMPI": ["E001", "E999"],  # E999 has no PE time
            "timestamp": pd.to_datetime(["2023-06-15 11:00", "2023-06-15 11:00"]),
            "vital_type": ["HR", "HR"],
            "value": [72.0, 80.0],
        })

        pe_times = {"E001": datetime(2023, 6, 15, 10, 0, 0)}

        result = add_pe_relative_timestamps(df, pe_times)

        assert len(result) == 1
        assert "E999" not in result["EMPI"].values
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestAddPERelativeTimestamps -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
from processing.temporal_aligner import calculate_hours_from_pe


def add_pe_relative_timestamps(
    df: pd.DataFrame,
    pe_times: Dict[str, datetime]
) -> pd.DataFrame:
    """Add PE-relative timestamps to vitals dataframe.

    Args:
        df: Vitals dataframe with EMPI and timestamp columns
        pe_times: Dict mapping EMPI to PE timestamp

    Returns:
        DataFrame with hours_from_pe column added.
        Patients without PE time are dropped.
    """
    result = df.copy()

    # Map EMPI to PE time
    result["pe_time"] = result["EMPI"].map(pe_times)

    # Drop patients without PE time
    result = result.dropna(subset=["pe_time"])

    if result.empty:
        result["hours_from_pe"] = pd.Series(dtype=float)
        return result.drop(columns=["pe_time"])

    # Calculate hours from PE
    result["hours_from_pe"] = result.apply(
        lambda r: calculate_hours_from_pe(r["timestamp"], r["pe_time"]),
        axis=1
    )

    return result.drop(columns=["pe_time"])
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestAddPERelativeTimestamps -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add PE-relative timestamp calculation"
```

---

## Task 16: Layer 1 Builder - Main Build Function

**Files:**
- Modify: `tests/test_layer1_builder.py`
- Modify: `processing/layer1_builder.py`

**Step 1: Write the failing test**

```python
class TestBuildLayer1:
    """Tests for main Layer 1 build function."""

    def test_build_layer1_integration(self, tmp_path):
        """Integration test for build_layer1 function."""
        from processing.layer1_builder import build_layer1
        from datetime import datetime
        import pickle
        from unittest.mock import MagicMock

        # Create mock input parquet files
        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 11:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["IP"],
            "encounter_number": ["ENC001"],
        })
        phy_path = tmp_path / "phy_vitals_raw.parquet"
        phy_df.to_parquet(phy_path)

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:30"]),
            "vital_type": ["SBP"],
            "value": [120.0],
            "units": ["mmHg"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs"],
            "confidence": [0.9],
            "is_flagged_abnormal": [False],
            "report_number": ["R001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
            "timestamp_source": ["report"],
            "timestamp_offset_hours": [0.0],
        })
        hnp_path = tmp_path / "hnp_vitals_raw.parquet"
        hnp_df.to_parquet(hnp_path)

        prg_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 12:00"]),
            "vital_type": ["DBP"],
            "value": [80.0],
            "units": ["mmHg"],
            "source": ["prg"],
            "extraction_context": ["vital_signs"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["R002"],
            "report_date_time": pd.to_datetime(["2023-06-15 11:00"]),
            "timestamp_source": ["report"],
            "timestamp_offset_hours": [0.0],
            "temp_method": [None],
        })
        prg_path = tmp_path / "prg_vitals_raw.parquet"
        prg_df.to_parquet(prg_path)

        # Create mock patient timelines
        mock_timeline = MagicMock()
        mock_timeline.patient_id = "E001"
        mock_timeline.time_zero = datetime(2023, 6, 15, 10, 0, 0)
        timelines = {"E001": mock_timeline}

        timeline_path = tmp_path / "patient_timelines.pkl"
        with open(timeline_path, "wb") as f:
            pickle.dump(timelines, f)

        # Output path
        output_path = tmp_path / "canonical_vitals.parquet"

        # Run build
        result = build_layer1(
            phy_path=phy_path,
            hnp_path=hnp_path,
            prg_path=prg_path,
            timeline_path=timeline_path,
            output_path=output_path
        )

        # Verify output file exists
        assert output_path.exists()

        # Verify schema
        output_df = pd.read_parquet(output_path)
        assert set(output_df.columns) >= {"EMPI", "timestamp", "hours_from_pe",
                                          "vital_type", "value", "source"}

        # Verify merged data
        assert len(output_df) >= 3  # At least PHY + HNP + PRG records
        assert set(output_df["source"].unique()) == {"phy", "hnp", "prg"}
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestBuildLayer1 -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer1_builder.py`:

```python
from processing.qc_filters import is_physiologically_valid, is_abnormal
from processing.temporal_aligner import is_within_window


def build_layer1(
    phy_path: Path,
    hnp_path: Path,
    prg_path: Path,
    timeline_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Build Layer 1 canonical vitals from all extraction sources.

    Args:
        phy_path: Path to phy_vitals_raw.parquet
        hnp_path: Path to hnp_vitals_raw.parquet
        prg_path: Path to prg_vitals_raw.parquet
        timeline_path: Path to patient_timelines.pkl
        output_path: Path to write canonical_vitals.parquet

    Returns:
        Combined DataFrame with Layer 1 schema
    """
    # Load PE times
    pe_times = load_pe_times(timeline_path)

    # Load and normalize each source
    dfs = []

    if phy_path.exists():
        phy_df = pd.read_parquet(phy_path)
        phy_norm = normalize_phy_source(phy_df)
        dfs.append(phy_norm)

    if hnp_path.exists():
        hnp_df = pd.read_parquet(hnp_path)
        hnp_norm = normalize_hnp_source(hnp_df)
        dfs.append(hnp_norm)

    if prg_path.exists():
        prg_df = pd.read_parquet(prg_path)
        prg_norm = normalize_prg_source(prg_df)
        dfs.append(prg_norm)

    if not dfs:
        raise ValueError("No input files found")

    # Combine all sources
    combined = pd.concat(dfs, ignore_index=True)

    # Add PE-relative timestamps
    combined = add_pe_relative_timestamps(combined, pe_times)

    # Filter to analysis window
    combined = combined[
        combined["hours_from_pe"].apply(is_within_window)
    ]

    # Filter to core vitals only
    combined = combined[combined["vital_type"].isin(CORE_VITALS)]

    # Apply physiological range validation
    valid_mask = combined.apply(
        lambda r: is_physiologically_valid(r["vital_type"], r["value"]),
        axis=1
    )
    combined = combined[valid_mask]

    # Update abnormal flags
    combined["is_flagged_abnormal"] = combined.apply(
        lambda r: is_abnormal(r["vital_type"], r["value"]),
        axis=1
    )

    # Generate calculated MAPs
    maps = generate_calculated_maps(combined)
    if not maps.empty:
        combined = pd.concat([combined, maps], ignore_index=True)

    # Sort by patient and time
    combined = combined.sort_values(["EMPI", "timestamp"]).reset_index(drop=True)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    return combined
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer1_builder.py::TestBuildLayer1 -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer1_builder.py module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add main build_layer1 function"
```

---

## Task 17: Layer 1 Builder - CLI Entry Point

**Files:**
- Modify: `processing/layer1_builder.py`

**Step 1: Add CLI entry point**

Add to end of `processing/layer1_builder.py`:

```python
def main():
    """CLI entry point for Layer 1 builder."""
    from pathlib import Path

    # Default paths
    base_dir = Path(__file__).parent.parent
    discovery_dir = base_dir / "outputs" / "discovery"
    module1_dir = base_dir.parent / "module_1_core_infrastructure" / "outputs"
    output_dir = base_dir / "outputs" / "layer1"

    phy_path = discovery_dir / "phy_vitals_raw.parquet"
    hnp_path = discovery_dir / "hnp_vitals_raw.parquet"
    prg_path = discovery_dir / "prg_vitals_raw.parquet"
    timeline_path = module1_dir / "patient_timelines.pkl"
    output_path = output_dir / "canonical_vitals.parquet"

    print(f"Building Layer 1 canonical vitals...")
    print(f"  PHY input: {phy_path}")
    print(f"  HNP input: {hnp_path}")
    print(f"  PRG input: {prg_path}")
    print(f"  Timeline: {timeline_path}")
    print(f"  Output: {output_path}")

    result = build_layer1(
        phy_path=phy_path,
        hnp_path=hnp_path,
        prg_path=prg_path,
        timeline_path=timeline_path,
        output_path=output_path
    )

    print(f"\nLayer 1 complete:")
    print(f"  Total records: {len(result):,}")
    print(f"  Patients: {result['EMPI'].nunique():,}")
    print(f"  Vital types: {result['vital_type'].value_counts().to_dict()}")
    print(f"  Sources: {result['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
```

**Step 2: Run syntax check**

Run: `python3 -m py_compile module_3_vitals_processing/processing/layer1_builder.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add module_3_vitals_processing/processing/layer1_builder.py
git commit -m "feat(3.4): add CLI entry point for layer1_builder"
```

---

## Task 18: Layer 2 Builder - Schema Definition

**Files:**
- Create: `tests/test_layer2_builder.py`
- Create: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
"""Tests for layer2_builder module."""
import pytest
import pandas as pd
import numpy as np
from processing.layer2_builder import (
    LAYER2_PARQUET_SCHEMA,
    HOUR_RANGE,
    VITAL_ORDER,
    FORWARD_FILL_LIMITS,
)


class TestLayer2Schema:
    """Tests for Layer 2 schema definitions."""

    def test_parquet_schema_has_required_columns(self):
        """Parquet schema has all required columns."""
        required = {
            "EMPI", "hour_from_pe", "vital_type",
            "mean", "median", "std", "min", "max", "count", "mask"
        }
        assert required <= set(LAYER2_PARQUET_SCHEMA.keys())

    def test_hour_range_is_correct(self):
        """Hour range is -24 to +720 (745 hours)."""
        assert HOUR_RANGE == list(range(-24, 721))
        assert len(HOUR_RANGE) == 745

    def test_vital_order_is_correct(self):
        """Seven core vitals in correct order."""
        assert VITAL_ORDER == ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]
        assert len(VITAL_ORDER) == 7

    def test_forward_fill_limits_defined(self):
        """Forward-fill limits defined for all vitals."""
        assert FORWARD_FILL_LIMITS["HR"] == 6
        assert FORWARD_FILL_LIMITS["SPO2"] == 4
        assert FORWARD_FILL_LIMITS["TEMP"] == 12
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestLayer2Schema -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
"""Layer 2 Builder: Hourly aggregated grid with missing data tensors."""
from typing import Dict, List

# Hour range: -24 to +720 (745 hours total)
HOUR_RANGE: List[int] = list(range(-24, 721))

# Vital order for tensor dimensions
VITAL_ORDER: List[str] = ["HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"]

# Forward-fill limits (hours) per vital type
FORWARD_FILL_LIMITS: Dict[str, int] = {
    "HR": 6,
    "SBP": 6,
    "DBP": 6,
    "MAP": 6,
    "RR": 6,
    "SPO2": 4,
    "TEMP": 12,
}

# Layer 2 Parquet schema
LAYER2_PARQUET_SCHEMA: Dict[str, str] = {
    "EMPI": "str",
    "hour_from_pe": "int32",
    "vital_type": "str",
    "mean": "float64",
    "median": "float64",
    "std": "float64",
    "min": "float64",
    "max": "float64",
    "count": "int32",
    "mask": "int8",  # 1=observed, 0=missing
}
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestLayer2Schema -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): define Layer 2 schema and constants"
```

---

## Task 19: Layer 2 Builder - Hourly Aggregation

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestHourlyAggregation:
    """Tests for hourly aggregation."""

    def test_aggregate_to_hourly_single_value(self):
        """Single value in hour produces correct stats."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001"],
            "hours_from_pe": [0.5],  # Hour bucket 0
            "vital_type": ["HR"],
            "value": [72.0],
        })

        result = aggregate_to_hourly(df)

        row = result[(result["EMPI"] == "E001") &
                     (result["hour_from_pe"] == 0) &
                     (result["vital_type"] == "HR")].iloc[0]

        assert row["mean"] == 72.0
        assert row["median"] == 72.0
        assert pd.isna(row["std"])  # std undefined for n=1
        assert row["min"] == 72.0
        assert row["max"] == 72.0
        assert row["count"] == 1
        assert row["mask"] == 1

    def test_aggregate_to_hourly_multiple_values(self):
        """Multiple values in hour produce correct stats."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E001"],
            "hours_from_pe": [0.0, 0.3, 0.9],  # All in hour 0
            "vital_type": ["HR", "HR", "HR"],
            "value": [70.0, 72.0, 74.0],
        })

        result = aggregate_to_hourly(df)

        row = result[(result["EMPI"] == "E001") &
                     (result["hour_from_pe"] == 0) &
                     (result["vital_type"] == "HR")].iloc[0]

        assert row["mean"] == 72.0
        assert row["median"] == 72.0
        assert abs(row["std"] - 2.0) < 0.01
        assert row["min"] == 70.0
        assert row["max"] == 74.0
        assert row["count"] == 3
        assert row["mask"] == 1

    def test_aggregate_to_hourly_separate_vitals(self):
        """Different vitals aggregated separately."""
        from processing.layer2_builder import aggregate_to_hourly

        df = pd.DataFrame({
            "EMPI": ["E001", "E001"],
            "hours_from_pe": [0.5, 0.5],
            "vital_type": ["HR", "SBP"],
            "value": [72.0, 120.0],
        })

        result = aggregate_to_hourly(df)

        hr_row = result[(result["vital_type"] == "HR")].iloc[0]
        sbp_row = result[(result["vital_type"] == "SBP")].iloc[0]

        assert hr_row["mean"] == 72.0
        assert sbp_row["mean"] == 120.0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestHourlyAggregation -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
import pandas as pd
import numpy as np
from processing.temporal_aligner import assign_hour_bucket


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate vitals to hourly bins.

    Args:
        df: Layer 1 canonical vitals with hours_from_pe column

    Returns:
        DataFrame with hourly aggregations (mean, median, std, min, max, count)
    """
    # Assign hour buckets
    df = df.copy()
    df["hour_from_pe"] = df["hours_from_pe"].apply(assign_hour_bucket)

    # Group by patient, hour, vital type
    grouped = df.groupby(["EMPI", "hour_from_pe", "vital_type"])["value"]

    # Calculate aggregations
    result = grouped.agg(
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
        count="count"
    ).reset_index()

    # Add mask column (1 = observed)
    result["mask"] = 1

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestHourlyAggregation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add hourly aggregation function"
```

---

## Task 20: Layer 2 Builder - Create Full Grid

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestCreateFullGrid:
    """Tests for creating full hourly grid."""

    def test_create_full_grid_fills_missing_hours(self):
        """Creates rows for all hours, not just observed."""
        from processing.layer2_builder import create_full_grid

        # Sparse data: only hour 0 observed
        observed = pd.DataFrame({
            "EMPI": ["E001"],
            "hour_from_pe": [0],
            "vital_type": ["HR"],
            "mean": [72.0],
            "median": [72.0],
            "std": [np.nan],
            "min": [72.0],
            "max": [72.0],
            "count": [1],
            "mask": [1],
        })

        patients = ["E001"]
        result = create_full_grid(observed, patients)

        # Should have 745 hours × 7 vitals = 5215 rows per patient
        expected_rows = 745 * 7
        assert len(result[result["EMPI"] == "E001"]) == expected_rows

    def test_create_full_grid_marks_missing_mask_zero(self):
        """Missing hours have mask=0."""
        from processing.layer2_builder import create_full_grid

        observed = pd.DataFrame({
            "EMPI": ["E001"],
            "hour_from_pe": [0],
            "vital_type": ["HR"],
            "mean": [72.0],
            "median": [72.0],
            "std": [np.nan],
            "min": [72.0],
            "max": [72.0],
            "count": [1],
            "mask": [1],
        })

        result = create_full_grid(observed, ["E001"])

        # Hour 0 HR should have mask=1
        hr_0 = result[(result["hour_from_pe"] == 0) &
                      (result["vital_type"] == "HR")].iloc[0]
        assert hr_0["mask"] == 1

        # Hour 1 HR should have mask=0
        hr_1 = result[(result["hour_from_pe"] == 1) &
                      (result["vital_type"] == "HR")].iloc[0]
        assert hr_1["mask"] == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestCreateFullGrid -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
from typing import List
from itertools import product


def create_full_grid(
    observed: pd.DataFrame,
    patients: List[str]
) -> pd.DataFrame:
    """Create full hourly grid with all patient-hour-vital combinations.

    Args:
        observed: DataFrame with observed hourly aggregations
        patients: List of patient EMPIs to include

    Returns:
        DataFrame with all combinations, missing marked with mask=0
    """
    # Create full index of all combinations
    full_index = pd.DataFrame(
        list(product(patients, HOUR_RANGE, VITAL_ORDER)),
        columns=["EMPI", "hour_from_pe", "vital_type"]
    )

    # Merge with observed data
    result = full_index.merge(
        observed,
        on=["EMPI", "hour_from_pe", "vital_type"],
        how="left"
    )

    # Fill missing mask values with 0
    result["mask"] = result["mask"].fillna(0).astype("int8")

    # Fill missing count with 0
    result["count"] = result["count"].fillna(0).astype("int32")

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestCreateFullGrid -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add full grid creation with missing hour marking"
```

---

## Task 21: Layer 2 Builder - Three-Tier Imputation

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestImputation:
    """Tests for three-tier imputation."""

    def test_forward_fill_within_limit(self):
        """Forward-fill works within vital-specific limit."""
        from processing.layer2_builder import apply_imputation

        # HR with 6-hour limit: hours 0-5 should fill from hour 0
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": list(range(10)),
            "vital_type": ["HR"] * 10,
            "mean": [72.0] + [np.nan] * 9,
            "mask": [1] + [0] * 9,
        })

        result = apply_imputation(df)

        # Hours 1-6 should be forward-filled (tier 2)
        for h in range(1, 7):
            row = result[(result["hour_from_pe"] == h)].iloc[0]
            assert row["mean"] == 72.0
            assert row["imputation_tier"] == 2

    def test_forward_fill_respects_limit(self):
        """Forward-fill stops at vital-specific limit."""
        from processing.layer2_builder import apply_imputation

        # HR with 6-hour limit: hour 7+ should NOT forward-fill
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": list(range(10)),
            "vital_type": ["HR"] * 10,
            "mean": [72.0] + [np.nan] * 9,
            "mask": [1] + [0] * 9,
        })

        result = apply_imputation(df)

        # Hours 7+ should use patient mean (tier 3), not forward-fill
        row_7 = result[(result["hour_from_pe"] == 7)].iloc[0]
        assert row_7["imputation_tier"] == 3

    def test_patient_mean_imputation(self):
        """Patient mean used when forward-fill exhausted."""
        from processing.layer2_builder import apply_imputation

        # Patient has observations at hour 0 and 100
        df = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "hour_from_pe": [0, 1, 2, 3, 4, 5, 6, 7, 8, 100],
            "vital_type": ["HR"] * 10,
            "mean": [70.0] + [np.nan] * 8 + [80.0],
            "mask": [1] + [0] * 8 + [1],
        })

        result = apply_imputation(df)

        # Hour 8 should use patient mean of 75.0
        row_8 = result[(result["hour_from_pe"] == 8)].iloc[0]
        assert row_8["mean"] == 75.0  # (70 + 80) / 2
        assert row_8["imputation_tier"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestImputation -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
def apply_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply three-tier imputation to hourly grid.

    Tiers:
        1: Observed (mask=1)
        2: Forward-filled within vital-specific limit
        3: Patient mean (when forward-fill exhausted)
        4: Cohort mean (when patient has no observations)

    Args:
        df: Full hourly grid from create_full_grid()

    Returns:
        DataFrame with imputed values and imputation_tier column
    """
    result = df.copy()
    result["imputation_tier"] = 0

    # Tier 1: Observed values
    result.loc[result["mask"] == 1, "imputation_tier"] = 1

    # Process each patient-vital combination
    for (empi, vital), group in result.groupby(["EMPI", "vital_type"]):
        group_idx = group.index
        group = group.sort_values("hour_from_pe")

        # Get forward-fill limit for this vital
        ff_limit = FORWARD_FILL_LIMITS.get(vital, 6)

        # Calculate patient mean from observed values
        observed_vals = group.loc[group["mask"] == 1, "mean"]
        patient_mean = observed_vals.mean() if len(observed_vals) > 0 else np.nan

        # Track last observed hour for forward-fill
        last_observed_hour = None
        last_observed_value = None

        for idx, row in group.iterrows():
            if row["mask"] == 1:
                # Observed: update last observed
                last_observed_hour = row["hour_from_pe"]
                last_observed_value = row["mean"]
            else:
                # Missing: determine imputation tier
                if (last_observed_hour is not None and
                    row["hour_from_pe"] - last_observed_hour <= ff_limit):
                    # Tier 2: Forward-fill
                    result.loc[idx, "mean"] = last_observed_value
                    result.loc[idx, "imputation_tier"] = 2
                elif not pd.isna(patient_mean):
                    # Tier 3: Patient mean
                    result.loc[idx, "mean"] = patient_mean
                    result.loc[idx, "imputation_tier"] = 3
                else:
                    # Tier 4: Will be filled with cohort mean later
                    result.loc[idx, "imputation_tier"] = 4

    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestImputation -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add three-tier imputation logic"
```

---

## Task 22: Layer 2 Builder - HDF5 Tensor Generation

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestHDF5TensorGeneration:
    """Tests for HDF5 tensor generation."""

    def test_create_tensors_correct_shape(self, tmp_path):
        """Tensors have correct shape (n_patients, 745, 7)."""
        from processing.layer2_builder import create_hdf5_tensors
        import h5py

        # Create minimal grid data
        grid = pd.DataFrame({
            "EMPI": ["E001"] * (745 * 7),
            "hour_from_pe": sorted(HOUR_RANGE * 7),
            "vital_type": VITAL_ORDER * 745,
            "mean": [72.0] * (745 * 7),
            "mask": [1] * (745 * 7),
            "imputation_tier": [1] * (745 * 7),
        })

        output_path = tmp_path / "test_tensors.h5"
        create_hdf5_tensors(grid, output_path)

        with h5py.File(output_path, "r") as f:
            assert f["values"].shape == (1, 745, 7)
            assert f["masks"].shape == (1, 745, 7)
            assert f["imputation_tier"].shape == (1, 745, 7)

    def test_create_tensors_patient_index(self, tmp_path):
        """Patient index maps EMPIs correctly."""
        from processing.layer2_builder import create_hdf5_tensors
        import h5py

        # Two patients
        grid_e1 = pd.DataFrame({
            "EMPI": ["E001"] * (745 * 7),
            "hour_from_pe": sorted(HOUR_RANGE * 7),
            "vital_type": VITAL_ORDER * 745,
            "mean": [72.0] * (745 * 7),
            "mask": [1] * (745 * 7),
            "imputation_tier": [1] * (745 * 7),
        })
        grid_e2 = pd.DataFrame({
            "EMPI": ["E002"] * (745 * 7),
            "hour_from_pe": sorted(HOUR_RANGE * 7),
            "vital_type": VITAL_ORDER * 745,
            "mean": [80.0] * (745 * 7),
            "mask": [1] * (745 * 7),
            "imputation_tier": [1] * (745 * 7),
        })
        grid = pd.concat([grid_e1, grid_e2], ignore_index=True)

        output_path = tmp_path / "test_tensors.h5"
        create_hdf5_tensors(grid, output_path)

        with h5py.File(output_path, "r") as f:
            patient_index = [p.decode() for p in f["patient_index"][:]]
            assert patient_index == ["E001", "E002"]
            assert f["values"].shape[0] == 2

    def test_create_tensors_vital_index(self, tmp_path):
        """Vital index has correct order."""
        from processing.layer2_builder import create_hdf5_tensors
        import h5py

        grid = pd.DataFrame({
            "EMPI": ["E001"] * (745 * 7),
            "hour_from_pe": sorted(HOUR_RANGE * 7),
            "vital_type": VITAL_ORDER * 745,
            "mean": [72.0] * (745 * 7),
            "mask": [1] * (745 * 7),
            "imputation_tier": [1] * (745 * 7),
        })

        output_path = tmp_path / "test_tensors.h5"
        create_hdf5_tensors(grid, output_path)

        with h5py.File(output_path, "r") as f:
            vital_index = [v.decode() for v in f["vital_index"][:]]
            assert vital_index == VITAL_ORDER
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestHDF5TensorGeneration -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
import h5py
from pathlib import Path


def create_hdf5_tensors(grid: pd.DataFrame, output_path: Path) -> None:
    """Create HDF5 file with tensor representation of hourly grid.

    Creates tensors:
        - values: (n_patients, 745, 7) float32 - vital values
        - masks: (n_patients, 745, 7) int8 - 1=observed, 0=imputed
        - imputation_tier: (n_patients, 745, 7) int8 - tier 1-4
        - patient_index: (n_patients,) str - EMPI mapping
        - vital_index: (7,) str - vital names
        - hour_index: (745,) int - hour values

    Args:
        grid: Full imputed grid from apply_imputation()
        output_path: Path to write HDF5 file
    """
    # Get unique patients in sorted order
    patients = sorted(grid["EMPI"].unique())
    n_patients = len(patients)
    n_hours = len(HOUR_RANGE)
    n_vitals = len(VITAL_ORDER)

    # Create patient to index mapping
    patient_to_idx = {p: i for i, p in enumerate(patients)}
    vital_to_idx = {v: i for i, v in enumerate(VITAL_ORDER)}
    hour_to_idx = {h: i for i, h in enumerate(HOUR_RANGE)}

    # Initialize tensors
    values = np.full((n_patients, n_hours, n_vitals), np.nan, dtype=np.float32)
    masks = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)
    imputation_tiers = np.zeros((n_patients, n_hours, n_vitals), dtype=np.int8)

    # Fill tensors from grid
    for _, row in grid.iterrows():
        p_idx = patient_to_idx[row["EMPI"]]
        h_idx = hour_to_idx[row["hour_from_pe"]]
        v_idx = vital_to_idx[row["vital_type"]]

        values[p_idx, h_idx, v_idx] = row["mean"]
        masks[p_idx, h_idx, v_idx] = row["mask"]
        imputation_tiers[p_idx, h_idx, v_idx] = row["imputation_tier"]

    # Write HDF5 file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("values", data=values, compression="gzip")
        f.create_dataset("masks", data=masks, compression="gzip")
        f.create_dataset("imputation_tier", data=imputation_tiers, compression="gzip")

        # String arrays
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("patient_index", data=np.array(patients, dtype=object), dtype=dt)
        f.create_dataset("vital_index", data=np.array(VITAL_ORDER, dtype=object), dtype=dt)
        f.create_dataset("hour_index", data=np.array(HOUR_RANGE, dtype=np.int32))
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestHDF5TensorGeneration -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add HDF5 tensor generation"
```

---

## Task 23: Layer 2 Builder - Time Delta Calculation

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestTimeDelta:
    """Tests for time-since-last-observation calculation."""

    def test_calculate_time_deltas(self):
        """Time deltas calculated correctly."""
        from processing.layer2_builder import calculate_time_deltas

        # Observations at hours 0, 5, 10
        masks = np.array([[[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]])  # 1 patient, 11 hours, 1 vital

        time_deltas = calculate_time_deltas(masks)

        # Hour 0: observed, delta=0
        assert time_deltas[0, 0, 0] == 0
        # Hour 1: 1 hour since observation
        assert time_deltas[0, 1, 0] == 1
        # Hour 4: 4 hours since observation
        assert time_deltas[0, 4, 0] == 4
        # Hour 5: observed, delta=0
        assert time_deltas[0, 5, 0] == 0
        # Hour 6: 1 hour since observation
        assert time_deltas[0, 6, 0] == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestTimeDelta -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
def calculate_time_deltas(masks: np.ndarray) -> np.ndarray:
    """Calculate time since last observation for each cell.

    Args:
        masks: (n_patients, n_hours, n_vitals) int8 array
               1=observed, 0=missing

    Returns:
        (n_patients, n_hours, n_vitals) float32 array of hours since last observation
    """
    n_patients, n_hours, n_vitals = masks.shape
    time_deltas = np.zeros_like(masks, dtype=np.float32)

    for p in range(n_patients):
        for v in range(n_vitals):
            last_obs_hour = -1
            for h in range(n_hours):
                if masks[p, h, v] == 1:
                    # Observed: delta is 0, update last observed
                    time_deltas[p, h, v] = 0
                    last_obs_hour = h
                else:
                    # Missing: calculate delta
                    if last_obs_hour >= 0:
                        time_deltas[p, h, v] = h - last_obs_hour
                    else:
                        # No prior observation
                        time_deltas[p, h, v] = h + 24 + 1  # Hours from window start

    return time_deltas
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestTimeDelta -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add time delta calculation"
```

---

## Task 24: Layer 2 Builder - Main Build Function

**Files:**
- Modify: `tests/test_layer2_builder.py`
- Modify: `processing/layer2_builder.py`

**Step 1: Write the failing test**

```python
class TestBuildLayer2:
    """Tests for main Layer 2 build function."""

    def test_build_layer2_integration(self, tmp_path):
        """Integration test for build_layer2."""
        from processing.layer2_builder import build_layer2

        # Create Layer 1 input
        layer1 = pd.DataFrame({
            "EMPI": ["E001"] * 10,
            "timestamp": pd.date_range("2023-06-15", periods=10, freq="h"),
            "hours_from_pe": list(range(10)),
            "vital_type": ["HR"] * 5 + ["SBP"] * 5,
            "value": [72.0, 74.0, 76.0, 75.0, 73.0, 120.0, 122.0, 118.0, 121.0, 119.0],
            "units": ["bpm"] * 5 + ["mmHg"] * 5,
            "source": ["phy"] * 10,
            "source_detail": ["IP"] * 10,
            "confidence": [1.0] * 10,
            "is_calculated": [False] * 10,
            "is_flagged_abnormal": [False] * 10,
            "report_number": [""] * 10,
        })

        layer1_path = tmp_path / "canonical_vitals.parquet"
        layer1.to_parquet(layer1_path)

        parquet_path = tmp_path / "hourly_grid.parquet"
        hdf5_path = tmp_path / "hourly_tensors.h5"

        build_layer2(
            layer1_path=layer1_path,
            parquet_output_path=parquet_path,
            hdf5_output_path=hdf5_path
        )

        # Verify outputs exist
        assert parquet_path.exists()
        assert hdf5_path.exists()

        # Verify parquet content
        grid = pd.read_parquet(parquet_path)
        assert "EMPI" in grid.columns
        assert "hour_from_pe" in grid.columns
        assert "vital_type" in grid.columns

        # Verify HDF5 content
        import h5py
        with h5py.File(hdf5_path, "r") as f:
            assert "values" in f
            assert "masks" in f
            assert "time_deltas" in f
```

**Step 2: Run test to verify it fails**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestBuildLayer2 -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

Add to `processing/layer2_builder.py`:

```python
def build_layer2(
    layer1_path: Path,
    parquet_output_path: Path,
    hdf5_output_path: Path
) -> pd.DataFrame:
    """Build Layer 2 hourly grid and tensors from Layer 1 canonical vitals.

    Args:
        layer1_path: Path to canonical_vitals.parquet (Layer 1 output)
        parquet_output_path: Path to write hourly_grid.parquet
        hdf5_output_path: Path to write hourly_tensors.h5

    Returns:
        Full imputed grid DataFrame
    """
    # Load Layer 1
    layer1 = pd.read_parquet(layer1_path)

    # Get unique patients
    patients = sorted(layer1["EMPI"].unique())

    # Aggregate to hourly bins
    hourly = aggregate_to_hourly(layer1)

    # Create full grid with all hours
    full_grid = create_full_grid(hourly, patients)

    # Apply three-tier imputation
    imputed_grid = apply_imputation(full_grid)

    # Calculate cohort means for tier 4
    for vital in VITAL_ORDER:
        vital_mask = imputed_grid["vital_type"] == vital
        tier4_mask = imputed_grid["imputation_tier"] == 4

        observed = imputed_grid.loc[vital_mask & (imputed_grid["mask"] == 1), "mean"]
        cohort_mean = observed.mean() if len(observed) > 0 else 0

        imputed_grid.loc[vital_mask & tier4_mask, "mean"] = cohort_mean

    # Save parquet
    parquet_output_path.parent.mkdir(parents=True, exist_ok=True)
    imputed_grid.to_parquet(parquet_output_path, index=False)

    # Create HDF5 tensors
    create_hdf5_tensors(imputed_grid, hdf5_output_path)

    # Add time deltas to HDF5
    with h5py.File(hdf5_output_path, "r") as f:
        masks = f["masks"][:]

    time_deltas = calculate_time_deltas(masks)

    with h5py.File(hdf5_output_path, "a") as f:
        f.create_dataset("time_deltas", data=time_deltas, compression="gzip")

    return imputed_grid


def main():
    """CLI entry point for Layer 2 builder."""
    base_dir = Path(__file__).parent.parent

    layer1_path = base_dir / "outputs" / "layer1" / "canonical_vitals.parquet"
    parquet_path = base_dir / "outputs" / "layer2" / "hourly_grid.parquet"
    hdf5_path = base_dir / "outputs" / "layer2" / "hourly_tensors.h5"

    print(f"Building Layer 2 hourly grid...")
    print(f"  Layer 1 input: {layer1_path}")
    print(f"  Parquet output: {parquet_path}")
    print(f"  HDF5 output: {hdf5_path}")

    result = build_layer2(
        layer1_path=layer1_path,
        parquet_output_path=parquet_path,
        hdf5_output_path=hdf5_path
    )

    print(f"\nLayer 2 complete:")
    print(f"  Grid rows: {len(result):,}")
    print(f"  Patients: {result['EMPI'].nunique():,}")
    print(f"  Observed hours: {(result['mask'] == 1).sum():,}")
    print(f"  Imputed hours: {(result['mask'] == 0).sum():,}")


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest module_3_vitals_processing/tests/test_layer2_builder.py::TestBuildLayer2 -v`
Expected: PASS (1 test)

**Step 5: Commit**

```bash
git add module_3_vitals_processing/tests/test_layer2_builder.py module_3_vitals_processing/processing/layer2_builder.py
git commit -m "feat(3.5): add main build_layer2 function with CLI"
```

---

## Task 25: Update Config with Layer Paths

**Files:**
- Modify: `config/vitals_config.py`

**Step 1: Add layer output paths**

Add to `config/vitals_config.py`:

```python
# Layer output paths
LAYER1_OUTPUT_DIR = OUTPUT_DIR / 'layer1'
LAYER2_OUTPUT_DIR = OUTPUT_DIR / 'layer2'
LAYER3_OUTPUT_DIR = OUTPUT_DIR / 'layer3'
LAYER4_OUTPUT_DIR = OUTPUT_DIR / 'layer4'
LAYER5_OUTPUT_DIR = OUTPUT_DIR / 'layer5'

# Layer 1 outputs
CANONICAL_VITALS_PATH = LAYER1_OUTPUT_DIR / 'canonical_vitals.parquet'

# Layer 2 outputs
HOURLY_GRID_PATH = LAYER2_OUTPUT_DIR / 'hourly_grid.parquet'
HOURLY_TENSORS_PATH = LAYER2_OUTPUT_DIR / 'hourly_tensors.h5'

# Temporal window constants
WINDOW_MIN_HOURS = -24   # 24 hours before PE
WINDOW_MAX_HOURS = 720   # 30 days after PE
TOTAL_HOURS = 745        # Total hours in window
```

**Step 2: Run syntax check**

Run: `python3 -m py_compile module_3_vitals_processing/config/vitals_config.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add module_3_vitals_processing/config/vitals_config.py
git commit -m "chore: add layer output paths to config"
```

---

## Task 26: Run All Tests

**Step 1: Run full test suite**

Run: `pytest module_3_vitals_processing/tests/ -v`
Expected: All tests pass

**Step 2: Verify test count**

Expected: ~200+ tests (174 existing + ~30 new)

**Step 3: Commit if any fixes needed**

---

## Task 27: Integration Test - Run Layer 1 on Real Data

**Step 1: Run Layer 1 builder**

Run: `python module_3_vitals_processing/processing/layer1_builder.py`

Expected output:
```
Building Layer 1 canonical vitals...
  PHY input: .../outputs/discovery/phy_vitals_raw.parquet
  HNP input: .../outputs/discovery/hnp_vitals_raw.parquet
  PRG input: .../outputs/discovery/prg_vitals_raw.parquet
  Timeline: .../module_1_core_infrastructure/outputs/patient_timelines.pkl
  Output: .../outputs/layer1/canonical_vitals.parquet

Layer 1 complete:
  Total records: X,XXX,XXX
  Patients: X,XXX
  Vital types: {'HR': ..., 'SBP': ..., ...}
  Sources: {'phy': ..., 'hnp': ..., 'prg': ...}
```

**Step 2: Verify output file**

Run: `ls -lh module_3_vitals_processing/outputs/layer1/`
Expected: canonical_vitals.parquet exists

**Step 3: Commit output directory structure**

```bash
git add module_3_vitals_processing/outputs/layer1/.gitkeep
git commit -m "feat(3.4): Layer 1 builder complete - canonical vitals generated"
```

---

## Task 28: Integration Test - Run Layer 2 on Real Data

**Step 1: Run Layer 2 builder**

Run: `python module_3_vitals_processing/processing/layer2_builder.py`

Expected output:
```
Building Layer 2 hourly grid...
  Layer 1 input: .../outputs/layer1/canonical_vitals.parquet
  Parquet output: .../outputs/layer2/hourly_grid.parquet
  HDF5 output: .../outputs/layer2/hourly_tensors.h5

Layer 2 complete:
  Grid rows: X,XXX,XXX
  Patients: X,XXX
  Observed hours: X,XXX,XXX
  Imputed hours: X,XXX,XXX
```

**Step 2: Verify output files**

Run: `ls -lh module_3_vitals_processing/outputs/layer2/`
Expected: hourly_grid.parquet and hourly_tensors.h5 exist

**Step 3: Commit output directory structure**

```bash
git add module_3_vitals_processing/outputs/layer2/.gitkeep
git commit -m "feat(3.5): Layer 2 builder complete - hourly grid and tensors generated"
```

---

## Task 29: Final Documentation Update

**Files:**
- Modify: `README.md`

**Step 1: Update README with Phase 1 completion status**

Add to Module 3 README:

```markdown
## Phase 1 Complete (Layers 1-2)

### Layer 1: Canonical Records
- **Output:** `outputs/layer1/canonical_vitals.parquet`
- **Records:** X,XXX,XXX vitals from 3 sources
- **Patients:** X,XXX with PE index alignment

### Layer 2: Hourly Grid
- **Parquet:** `outputs/layer2/hourly_grid.parquet`
- **HDF5:** `outputs/layer2/hourly_tensors.h5`
  - values: (n_patients, 745, 7) float32
  - masks: (n_patients, 745, 7) int8
  - time_deltas: (n_patients, 745, 7) float32
  - imputation_tier: (n_patients, 745, 7) int8
```

**Step 2: Commit**

```bash
git add module_3_vitals_processing/README.md
git commit -m "docs: update README with Phase 1 completion status"
```

---

## Summary

**Total Tasks:** 29
**Estimated New Lines of Code:** ~800 (production) + ~600 (tests)
**New Files Created:**
- `processing/__init__.py`
- `processing/unit_converter.py`
- `processing/qc_filters.py`
- `processing/temporal_aligner.py`
- `processing/layer1_builder.py`
- `processing/layer2_builder.py`
- `tests/test_unit_converter.py`
- `tests/test_qc_filters.py`
- `tests/test_temporal_aligner.py`
- `tests/test_layer1_builder.py`
- `tests/test_layer2_builder.py`

**Outputs Generated:**
- `outputs/layer1/canonical_vitals.parquet`
- `outputs/layer2/hourly_grid.parquet`
- `outputs/layer2/hourly_tensors.h5`
