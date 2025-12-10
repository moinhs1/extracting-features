# Phase 7: Layer 5 Dose Intensity Features Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate continuous dose-intensity features including raw doses, DDD-normalized ratios, and weight-adjusted values for key therapeutic classes, with special handling for PE-critical anticoagulants and vasopressors.

**Architecture:** Load mapped medications, compute daily/hourly dose aggregations, normalize by WHO DDD values, compute intensity metrics (trend, hours since last, cumulative exposure), output to gold parquet.

**Tech Stack:** Python 3.12, pandas, numpy, pytest

**Depends on:** Phase 3 complete (mapped_medications.parquet exists)

---

## Task 1: Create Dose Intensity Builder Tests

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_dose_intensity_builder.py`

**Step 1: Create test file**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_dose_intensity_builder.py
"""Tests for dose intensity feature generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDDDNormalization:
    """Test DDD (Defined Daily Dose) normalization."""

    def test_get_ddd_heparin(self):
        """Get DDD for heparin."""
        from transformers.dose_intensity_builder import get_ddd

        ddd = get_ddd('heparin', 'units')

        assert ddd is not None
        assert ddd > 0

    def test_get_ddd_enoxaparin(self):
        """Get DDD for enoxaparin."""
        from transformers.dose_intensity_builder import get_ddd

        ddd = get_ddd('enoxaparin', 'mg')

        assert ddd is not None
        assert ddd == 40  # Standard DDD for enoxaparin

    def test_ddd_ratio_calculation(self):
        """Calculate dose / DDD ratio."""
        from transformers.dose_intensity_builder import calculate_ddd_ratio

        ratio = calculate_ddd_ratio(
            dose_value=80,
            dose_unit='mg',
            ingredient='enoxaparin'
        )

        assert ratio == 2.0  # 80mg / 40mg DDD


class TestDailyDoseAggregation:
    """Test daily dose aggregation."""

    def test_aggregate_daily_doses(self):
        """Aggregate multiple doses to daily total."""
        from transformers.dose_intensity_builder import aggregate_daily_doses

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 6, 12],  # Same day
            'ingredient_name': ['heparin', 'heparin', 'heparin'],
            'parsed_dose_value': [5000, 5000, 5000],
            'parsed_dose_unit': ['units', 'units', 'units'],
        })

        result = aggregate_daily_doses(df, 'heparin')

        assert len(result) == 1
        assert result.iloc[0]['daily_dose'] == 15000


class TestIntensityFeatures:
    """Test intensity feature calculation."""

    def test_hours_since_last(self):
        """Calculate hours since last administration."""
        from transformers.dose_intensity_builder import calculate_hours_since_last

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 6, 24],
            'ingredient_name': ['heparin', 'heparin', 'heparin'],
        })

        result = calculate_hours_since_last(df, 'heparin')

        # At hour 24, last dose was at hour 6, so 18 hours
        row_24 = result[result['hours_from_t0'] == 24]
        assert row_24['hours_since_last'].values[0] == pytest.approx(18, abs=0.1)

    def test_cumulative_exposure(self):
        """Calculate cumulative dose exposure."""
        from transformers.dose_intensity_builder import calculate_cumulative_exposure

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [0, 12, 24],
            'parsed_dose_value': [100, 100, 100],
        })

        result = calculate_cumulative_exposure(df)

        # At hour 24, cumulative should be 300
        assert result.iloc[2]['cumulative_dose'] == 300

    def test_dose_trend(self):
        """Calculate dose trend (increasing/decreasing/stable)."""
        from transformers.dose_intensity_builder import calculate_dose_trend

        # Increasing doses
        df = pd.DataFrame({
            'hours_from_t0': [0, 12, 24],
            'parsed_dose_value': [50, 75, 100],
        })

        trend = calculate_dose_trend(df)

        assert trend == 1  # Increasing


class TestVasopressorFeatures:
    """Test vasopressor-specific features."""

    def test_vasopressor_count(self):
        """Count concurrent vasopressors."""
        from transformers.dose_intensity_builder import count_concurrent_vasopressors

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [1, 1, 1],  # Same time
            'ingredient_name': ['norepinephrine', 'vasopressin', 'dopamine'],
        })

        count = count_concurrent_vasopressors(df, hour=1)

        assert count == 3

    def test_norepinephrine_dose_conversion(self):
        """Convert norepinephrine to mcg/kg/min."""
        from transformers.dose_intensity_builder import convert_norepi_dose

        # 8mg/hr = 8000mcg/60min = 133.3 mcg/min
        # For 70kg patient: 133.3/70 = 1.9 mcg/kg/min
        result = convert_norepi_dose(dose_mg_hr=8, weight_kg=70)

        assert result == pytest.approx(1.9, abs=0.1)
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_dose_intensity_builder.py -v
```

**Step 3: Commit failing tests**

```bash
git add module_04_medications/tests/test_dose_intensity_builder.py
git commit -m "test(module4): add dose intensity builder tests

Add failing tests for:
- DDD normalization
- Daily dose aggregation
- Intensity features (hours since last, cumulative, trend)
- Vasopressor-specific features

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement Dose Intensity Builder

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/transformers/dose_intensity_builder.py`

**Step 1: Create dose intensity builder**

```python
# /home/moin/TDA_11_25/module_04_medications/transformers/dose_intensity_builder.py
"""
Dose Intensity Feature Builder
==============================

Layer 5: Generate continuous dose-intensity features.

Features include:
- Raw dose values (daily totals)
- DDD-normalized ratios
- Weight-adjusted doses (when available)
- Temporal features (hours since last, cumulative exposure, trend)
- Vasopressor-specific metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    SILVER_DIR,
    GOLD_DIR,
    TEMPORAL_CONFIG,
)


# =============================================================================
# DDD VALUES (WHO Defined Daily Doses)
# =============================================================================

# Source: WHO ATC/DDD Index
DDD_VALUES = {
    # Anticoagulants
    'heparin': {'units': 30000, 'mg': None},
    'enoxaparin': {'mg': 40, 'units': 4000},
    'dalteparin': {'units': 7500},
    'tinzaparin': {'units': 10000},
    'fondaparinux': {'mg': 2.5},
    'warfarin': {'mg': 7.5},
    'apixaban': {'mg': 10},
    'rivaroxaban': {'mg': 20},
    'dabigatran': {'mg': 300},
    'argatroban': {'mg': None},  # Weight-based

    # Vasopressors (per hour, convert to daily)
    'norepinephrine': {'mg': 24},  # Variable, use 1mg/hr as reference
    'epinephrine': {'mg': 12},
    'vasopressin': {'units': 24},  # Variable
    'dopamine': {'mg': 600},
    'dobutamine': {'mg': 500},
    'phenylephrine': {'mg': 100},

    # Common medications
    'aspirin': {'mg': 100},
    'morphine': {'mg': 30},
    'fentanyl': {'mcg': 600},
    'furosemide': {'mg': 40},
    'metoprolol': {'mg': 150},
    'lisinopril': {'mg': 10},
    'prednisone': {'mg': 10},
}


def get_ddd(ingredient: str, unit: str) -> Optional[float]:
    """
    Get WHO DDD for a medication.

    Args:
        ingredient: Ingredient name
        unit: Dose unit

    Returns:
        DDD value or None if not available
    """
    if not ingredient:
        return None

    ingredient = ingredient.lower().strip()
    unit = unit.lower().strip() if unit else None

    if ingredient not in DDD_VALUES:
        return None

    ddd_info = DDD_VALUES[ingredient]

    if isinstance(ddd_info, dict):
        return ddd_info.get(unit)

    return ddd_info


def calculate_ddd_ratio(
    dose_value: float,
    dose_unit: str,
    ingredient: str
) -> Optional[float]:
    """
    Calculate dose / DDD ratio.

    Args:
        dose_value: Numeric dose
        dose_unit: Dose unit
        ingredient: Ingredient name

    Returns:
        Ratio of dose to DDD (1.0 = DDD)
    """
    if dose_value is None or pd.isna(dose_value):
        return None

    ddd = get_ddd(ingredient, dose_unit)
    if ddd is None or ddd == 0:
        return None

    return dose_value / ddd


# =============================================================================
# DAILY DOSE AGGREGATION
# =============================================================================

def aggregate_daily_doses(
    df: pd.DataFrame,
    ingredient: str
) -> pd.DataFrame:
    """
    Aggregate doses to daily totals for a specific ingredient.

    Args:
        df: Medications DataFrame
        ingredient: Ingredient to aggregate

    Returns:
        DataFrame with daily dose totals
    """
    df = df.copy()

    # Filter to ingredient
    mask = df['ingredient_name'].str.lower() == ingredient.lower()
    df = df[mask]

    if len(df) == 0:
        return pd.DataFrame()

    # Compute day relative to T0
    df['day_from_t0'] = (df['hours_from_t0'] / 24).astype(int)

    # Aggregate by patient-day
    result = df.groupby(['empi', 'day_from_t0']).agg({
        'parsed_dose_value': 'sum',
        'parsed_dose_unit': 'first',
        'hours_from_t0': 'min',  # First administration that day
    }).reset_index()

    result = result.rename(columns={
        'parsed_dose_value': 'daily_dose',
        'hours_from_t0': 'first_admin_hours',
    })

    return result


# =============================================================================
# INTENSITY FEATURES
# =============================================================================

def calculate_hours_since_last(
    df: pd.DataFrame,
    ingredient: str
) -> pd.DataFrame:
    """
    Calculate hours since last administration.

    Args:
        df: Medications DataFrame
        ingredient: Ingredient name

    Returns:
        DataFrame with hours_since_last column
    """
    df = df.copy()

    # Filter to ingredient
    mask = df['ingredient_name'].str.lower() == ingredient.lower()
    df = df[mask].sort_values('hours_from_t0')

    if len(df) == 0:
        return pd.DataFrame()

    # Calculate time since previous
    df['hours_since_last'] = df.groupby('empi')['hours_from_t0'].diff()

    # First occurrence has no previous
    df['hours_since_last'] = df['hours_since_last'].fillna(0)

    return df


def calculate_cumulative_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative dose exposure.

    Args:
        df: DataFrame with parsed_dose_value

    Returns:
        DataFrame with cumulative_dose column
    """
    df = df.copy()
    df = df.sort_values('hours_from_t0')
    df['cumulative_dose'] = df.groupby('empi')['parsed_dose_value'].cumsum()
    return df


def calculate_dose_trend(df: pd.DataFrame, window_hours: int = 24) -> int:
    """
    Calculate dose trend over window.

    Args:
        df: DataFrame with doses for single patient
        window_hours: Window for trend calculation

    Returns:
        -1 (decreasing), 0 (stable), +1 (increasing)
    """
    if len(df) < 2:
        return 0

    df = df.sort_values('hours_from_t0')

    # Get first and last dose in window
    first_dose = df.iloc[0]['parsed_dose_value']
    last_dose = df.iloc[-1]['parsed_dose_value']

    if pd.isna(first_dose) or pd.isna(last_dose):
        return 0

    # 10% threshold for change
    if last_dose > first_dose * 1.1:
        return 1
    elif last_dose < first_dose * 0.9:
        return -1
    else:
        return 0


# =============================================================================
# VASOPRESSOR FEATURES
# =============================================================================

VASOPRESSORS = ['norepinephrine', 'epinephrine', 'vasopressin', 'phenylephrine', 'dopamine']


def count_concurrent_vasopressors(df: pd.DataFrame, hour: float) -> int:
    """
    Count concurrent vasopressors at a given hour.

    Args:
        df: Medications DataFrame
        hour: Hour relative to T0

    Returns:
        Number of concurrent vasopressors
    """
    # Consider vasopressors within 6 hours as concurrent
    window = 6

    mask = (
        (df['ingredient_name'].str.lower().isin(VASOPRESSORS)) &
        (df['hours_from_t0'] >= hour - window) &
        (df['hours_from_t0'] <= hour)
    )

    return df[mask]['ingredient_name'].str.lower().nunique()


def convert_norepi_dose(dose_mg_hr: float, weight_kg: float = 70) -> float:
    """
    Convert norepinephrine dose to mcg/kg/min.

    Args:
        dose_mg_hr: Dose in mg/hr
        weight_kg: Patient weight in kg

    Returns:
        Dose in mcg/kg/min
    """
    if pd.isna(dose_mg_hr) or weight_kg <= 0:
        return np.nan

    mcg_per_min = (dose_mg_hr * 1000) / 60
    return mcg_per_min / weight_kg


# =============================================================================
# FEATURE GENERATION
# =============================================================================

def generate_class_intensity_features(
    df: pd.DataFrame,
    class_ingredients: List[str]
) -> pd.DataFrame:
    """
    Generate intensity features for a therapeutic class.

    Args:
        df: Mapped medications
        class_ingredients: List of ingredients in class

    Returns:
        DataFrame with intensity features
    """
    # Filter to class ingredients
    mask = df['ingredient_name'].str.lower().isin([i.lower() for i in class_ingredients])
    class_df = df[mask].copy()

    if len(class_df) == 0:
        return pd.DataFrame()

    # Aggregate daily doses
    class_df['day_from_t0'] = (class_df['hours_from_t0'] / 24).astype(int)

    result = class_df.groupby(['empi', 'day_from_t0']).agg({
        'parsed_dose_value': ['sum', 'mean', 'max', 'count'],
        'parsed_dose_unit': 'first',
        'ingredient_name': 'first',
        'hours_from_t0': ['min', 'max'],
    }).reset_index()

    # Flatten column names
    result.columns = [
        'empi', 'day_from_t0',
        'daily_dose', 'mean_dose', 'max_dose', 'admin_count',
        'dose_unit', 'ingredient', 'first_admin_hour', 'last_admin_hour'
    ]

    # Add DDD ratio
    result['ddd_ratio'] = result.apply(
        lambda r: calculate_ddd_ratio(r['daily_dose'], r['dose_unit'], r['ingredient']),
        axis=1
    )

    return result


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_layer5(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Build Layer 5 dose intensity features.

    Args:
        input_path: Path to mapped_medications.parquet
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with dose intensity features
    """
    print("=" * 60)
    print("Layer 5: Dose Intensity Features")
    print("=" * 60)

    # Load data
    if input_path is None:
        input_path = SILVER_DIR / "mapped_medications.parquet"

    print(f"\n1. Loading mapped medications: {input_path}")
    df = pd.read_parquet(input_path)

    if test_mode:
        sample_empis = df['empi'].unique()[:100]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")
    print(f"   Patients: {df['empi'].nunique():,}")

    # Key medication classes to generate intensity features for
    key_classes = {
        'anticoagulants': ['heparin', 'enoxaparin', 'dalteparin', 'warfarin', 'apixaban', 'rivaroxaban'],
        'vasopressors': ['norepinephrine', 'epinephrine', 'vasopressin', 'dopamine', 'phenylephrine'],
        'opioids': ['morphine', 'fentanyl', 'hydromorphone'],
        'diuretics': ['furosemide', 'bumetanide'],
    }

    all_features = []

    print("\n2. Generating intensity features by class...")
    for class_name, ingredients in key_classes.items():
        print(f"   {class_name}...")
        class_features = generate_class_intensity_features(df, ingredients)

        if len(class_features) > 0:
            class_features['class_name'] = class_name
            all_features.append(class_features)
            print(f"     {len(class_features)} daily records")

    if all_features:
        result = pd.concat(all_features, ignore_index=True)
    else:
        result = pd.DataFrame()

    print(f"\n   Total intensity records: {len(result):,}")

    # Save output
    output_dir = GOLD_DIR / "dose_intensity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        filename = "dose_intensity_test.parquet" if test_mode else "dose_intensity.parquet"
        output_path = output_dir / filename

    print(f"\n3. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)

    # Summary stats
    if len(result) > 0:
        print("\n" + "=" * 60)
        print("Summary Statistics")
        print("=" * 60)
        print(f"\nBy class:")
        print(result.groupby('class_name').size())

        print(f"\nDDD ratio distribution (anticoagulants):")
        ac_data = result[result['class_name'] == 'anticoagulants']['ddd_ratio']
        if len(ac_data) > 0:
            print(f"  Mean: {ac_data.mean():.2f}")
            print(f"  Median: {ac_data.median():.2f}")
            print(f"  Max: {ac_data.max():.2f}")

    print("\n" + "=" * 60)
    print("Layer 5 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build dose intensity features")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    build_layer5(test_mode=args.test)
```

**Step 2: Run tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_dose_intensity_builder.py -v
```

**Step 3: Commit**

```bash
git add module_04_medications/transformers/dose_intensity_builder.py
git commit -m "feat(module4): implement dose intensity feature builder

Layer 5 implementation:
- WHO DDD normalization
- Daily dose aggregation
- Intensity features (cumulative, trend, hours since last)
- Vasopressor-specific metrics (concurrent count, mcg/kg/min)

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Run Dose Intensity Building

**Step 1: Run test mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/dose_intensity_builder.py --test
```

**Step 2: Run full mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/dose_intensity_builder.py
```

**Step 3: Validate output**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd

df = pd.read_parquet('module_04_medications/data/gold/dose_intensity/dose_intensity.parquet')

print('='*60)
print('Layer 5 Validation')
print('='*60)

print(f'\\nShape: {df.shape}')
print(f'Patients: {df[\"empi\"].nunique()}')

print(f'\\nBy class:')
print(df.groupby('class_name').size())

print(f'\\nDDD ratio stats:')
print(df['ddd_ratio'].describe())
"
```

---

## Validation Checklist

- ✅ `dose_intensity_builder.py` exists
- ✅ All tests pass
- ✅ `dose_intensity.parquet` created
- ✅ DDD ratios calculated for key medications
- ✅ Features by therapeutic class available

---

## Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | Dose intensity tests | test_dose_intensity_builder.py |
| 2 | Implement builder | dose_intensity_builder.py |
| 3 | Run building | dose_intensity.parquet |

**Total:** 3 tasks, ~10 steps
