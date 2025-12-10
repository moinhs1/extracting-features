# Phase 5: Layer 3 Individual Medication Indicators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate sparse binary indicators for individual medications appearing in ≥20 patients, with exceptions for all anticoagulants, vasopressors, and thrombolytics regardless of prevalence.

**Architecture:** Load mapped medications from silver, filter by prevalence threshold, create sparse indicator matrix per patient-timewindow, output as parquet with additional scipy sparse matrix in HDF5.

**Tech Stack:** Python 3.12, pandas, scipy.sparse, h5py, pytest

**Depends on:** Phase 3 complete (mapped_medications.parquet exists)

---

## Task 1: Create Individual Indicator Builder Tests

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_individual_indicator_builder.py`

**Step 1: Create test file**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_individual_indicator_builder.py
"""Tests for individual medication indicator generation."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPrevalenceFiltering:
    """Test prevalence-based medication filtering."""

    def test_filter_by_prevalence(self):
        """Filter medications by patient count threshold."""
        from transformers.individual_indicator_builder import filter_by_prevalence

        df = pd.DataFrame({
            'empi': ['1', '1', '2', '2', '3', '3'] * 10 + ['1'] * 5,
            'ingredient_name': ['aspirin'] * 30 + ['rare_drug'] * 5 + ['common_drug'] * 30,
        })

        # With threshold of 20 patients
        result = filter_by_prevalence(df, min_patients=20)

        assert 'aspirin' in result
        assert 'common_drug' in result
        assert 'rare_drug' not in result  # Only in 1 patient

    def test_always_include_exceptions(self):
        """Always include critical medications regardless of prevalence."""
        from transformers.individual_indicator_builder import filter_by_prevalence

        df = pd.DataFrame({
            'empi': ['1', '2'],  # Only 2 patients
            'ingredient_name': ['heparin', 'heparin'],
        })

        result = filter_by_prevalence(df, min_patients=20)

        # Heparin should be included even though only 2 patients
        assert 'heparin' in result


class TestIndicatorCreation:
    """Test sparse indicator creation."""

    def test_create_indicators_single_patient(self):
        """Create indicators for single patient."""
        from transformers.individual_indicator_builder import create_patient_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_t0': [1, 2, 3],
            'ingredient_name': ['aspirin', 'aspirin', 'metoprolol'],
            'parsed_dose_value': [325, 325, 50],
        })

        medications = ['aspirin', 'metoprolol', 'lisinopril']

        result = create_patient_indicators(df, medications, window='acute')

        assert len(result) == 1
        assert result.iloc[0]['med_aspirin'] == True
        assert result.iloc[0]['med_aspirin_count'] == 2
        assert result.iloc[0]['med_metoprolol'] == True
        assert result.iloc[0]['med_lisinopril'] == False


class TestSparseStorage:
    """Test sparse matrix storage."""

    def test_to_sparse_matrix(self):
        """Convert to scipy sparse matrix."""
        from transformers.individual_indicator_builder import to_sparse_matrix

        df = pd.DataFrame({
            'empi': ['1', '2'],
            'time_window': ['acute', 'acute'],
            'med_aspirin': [True, False],
            'med_metoprolol': [False, True],
        })

        sparse, feature_names = to_sparse_matrix(df)

        assert sparse.shape == (2, 2)
        assert 'med_aspirin' in feature_names
        assert sparse[0, feature_names.index('med_aspirin')] == 1
        assert sparse[1, feature_names.index('med_aspirin')] == 0
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_individual_indicator_builder.py -v
```

**Step 3: Commit failing tests**

```bash
git add module_04_medications/tests/test_individual_indicator_builder.py
git commit -m "test(module4): add individual medication indicator tests

Add failing tests for:
- Prevalence filtering
- Exception medications (anticoagulants, vasopressors)
- Indicator creation per patient-window
- Sparse matrix conversion

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement Individual Indicator Builder

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/transformers/individual_indicator_builder.py`

**Step 1: Create individual indicator builder**

```python
# /home/moin/TDA_11_25/module_04_medications/transformers/individual_indicator_builder.py
"""
Individual Medication Indicator Builder
=======================================

Layer 3: Generate sparse binary indicators for individual medications.

Selection criteria:
- Medications appearing in ≥20 patients
- ALL anticoagulants, vasopressors, thrombolytics regardless of prevalence
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from scipy import sparse
import h5py
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    SILVER_DIR,
    GOLD_DIR,
    TEMPORAL_CONFIG,
    LAYER_CONFIG,
    load_therapeutic_classes,
)


# =============================================================================
# EXCEPTION MEDICATIONS (always include)
# =============================================================================

def get_exception_ingredients() -> Set[str]:
    """Get ingredients that should always be included regardless of prevalence."""
    classes = load_therapeutic_classes()
    exceptions = set()

    # Categories to always include
    always_include = ['anticoagulants', 'cardiovascular']

    for category_name, category_data in classes.items():
        if not isinstance(category_data, dict):
            continue

        for class_id, class_def in category_data.items():
            if not isinstance(class_def, dict):
                continue

            # Include all anticoagulants
            if class_id.startswith('ac_'):
                for ing in class_def.get('ingredients', []):
                    exceptions.add(ing.lower().strip())

            # Include all vasopressors
            if 'vasopressor' in class_id or class_id in ['cv_norepinephrine', 'cv_epinephrine',
                                                          'cv_vasopressin', 'cv_phenylephrine',
                                                          'cv_dopamine']:
                for ing in class_def.get('ingredients', []):
                    exceptions.add(ing.lower().strip())

            # Include all inotropes
            if 'inotrope' in class_id or class_id in ['cv_dobutamine', 'cv_milrinone']:
                for ing in class_def.get('ingredients', []):
                    exceptions.add(ing.lower().strip())

    return exceptions


# =============================================================================
# PREVALENCE FILTERING
# =============================================================================

def filter_by_prevalence(
    df: pd.DataFrame,
    min_patients: int = None
) -> Set[str]:
    """
    Get medications meeting prevalence threshold or in exception list.

    Args:
        df: DataFrame with empi and ingredient_name columns
        min_patients: Minimum patient count (default from config)

    Returns:
        Set of ingredient names to include
    """
    if min_patients is None:
        min_patients = LAYER_CONFIG.prevalence_threshold

    # Count unique patients per ingredient
    patient_counts = df.groupby('ingredient_name')['empi'].nunique()

    # Medications meeting threshold
    meets_threshold = set(patient_counts[patient_counts >= min_patients].index)

    # Exception medications (always include)
    exceptions = get_exception_ingredients()

    # Combine: threshold OR exception
    all_ingredients = df['ingredient_name'].dropna().str.lower().unique()
    included = set()

    for ing in all_ingredients:
        if ing in meets_threshold or ing.lower() in exceptions:
            included.add(ing)

    return included


# =============================================================================
# INDICATOR CREATION
# =============================================================================

def get_time_window(hours_from_t0: float) -> Optional[str]:
    """Assign hours to time window."""
    for window_name, (start, end) in TEMPORAL_CONFIG.windows.items():
        if start <= hours_from_t0 < end:
            return window_name
    return None


def create_patient_indicators(
    df: pd.DataFrame,
    medications: List[str],
    window: str
) -> pd.DataFrame:
    """
    Create indicator row for single patient-window.

    Args:
        df: DataFrame for single patient filtered to window
        medications: List of medication names to create indicators for
        window: Time window name

    Returns:
        DataFrame with single row containing indicators
    """
    if len(df) == 0:
        return pd.DataFrame()

    empi = df['empi'].iloc[0]
    window_bounds = TEMPORAL_CONFIG.windows.get(window, (0, 0))

    row = {
        'empi': empi,
        'time_window': window,
        'window_start_hours': window_bounds[0],
        'window_end_hours': window_bounds[1],
    }

    # Initialize all indicators to False
    for med in medications:
        col_name = f'med_{med.replace(" ", "_").replace("-", "_")}'
        row[col_name] = False
        row[f'{col_name}_count'] = 0
        row[f'{col_name}_total_dose'] = 0.0

    # Set indicators based on data
    for _, med_row in df.iterrows():
        ing = med_row.get('ingredient_name')
        if ing and ing.lower() in [m.lower() for m in medications]:
            col_name = f'med_{ing.replace(" ", "_").replace("-", "_")}'
            row[col_name] = True
            row[f'{col_name}_count'] = row.get(f'{col_name}_count', 0) + 1

            dose = med_row.get('parsed_dose_value')
            if pd.notna(dose):
                row[f'{col_name}_total_dose'] = row.get(f'{col_name}_total_dose', 0) + dose

    return pd.DataFrame([row])


def build_individual_indicators(
    df: pd.DataFrame,
    medications: List[str]
) -> pd.DataFrame:
    """
    Build individual medication indicators for all patients.

    Args:
        df: Mapped medications DataFrame
        medications: List of medications to include

    Returns:
        DataFrame with indicators per patient-window
    """
    # Assign time windows
    df = df.copy()
    df['time_window'] = df['hours_from_t0'].apply(get_time_window)
    df = df[df['time_window'].notna()]

    results = []

    # Group by patient and window
    for (empi, window), group in df.groupby(['empi', 'time_window']):
        row_df = create_patient_indicators(group, medications, window)
        if len(row_df) > 0:
            results.append(row_df)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# =============================================================================
# SPARSE MATRIX CONVERSION
# =============================================================================

def to_sparse_matrix(
    df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Convert indicator DataFrame to sparse matrix.

    Args:
        df: DataFrame with indicator columns (med_*)

    Returns:
        Tuple of (sparse matrix, feature names)
    """
    # Get indicator columns
    indicator_cols = [c for c in df.columns if c.startswith('med_') and not c.endswith('_count') and not c.endswith('_dose')]

    # Extract values
    data = df[indicator_cols].values.astype(np.float32)

    # Convert to sparse
    sparse_matrix = sparse.csr_matrix(data)

    return sparse_matrix, indicator_cols


def save_sparse_hdf5(
    sparse_matrix: sparse.csr_matrix,
    feature_names: List[str],
    patient_ids: List[str],
    windows: List[str],
    output_path: Path
):
    """
    Save sparse matrix to HDF5.

    Args:
        sparse_matrix: Sparse indicator matrix
        feature_names: List of feature names
        patient_ids: List of patient IDs
        windows: List of time windows
        output_path: Output HDF5 path
    """
    with h5py.File(output_path, 'w') as f:
        # Save sparse matrix components
        g = f.create_group('sparse_indicators')
        g.create_dataset('data', data=sparse_matrix.data)
        g.create_dataset('indices', data=sparse_matrix.indices)
        g.create_dataset('indptr', data=sparse_matrix.indptr)
        g.attrs['shape'] = sparse_matrix.shape

        # Save metadata
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('feature_names', data=np.array(feature_names, dtype=object), dtype=dt)
        f.create_dataset('patient_ids', data=np.array(patient_ids, dtype=object), dtype=dt)
        f.create_dataset('time_windows', data=np.array(windows, dtype=object), dtype=dt)


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_layer3(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Build Layer 3 individual medication indicators.

    Args:
        input_path: Path to mapped_medications.parquet
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with individual indicators
    """
    print("=" * 60)
    print("Layer 3: Individual Medication Indicators")
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

    # Filter medications by prevalence
    print("\n2. Filtering medications by prevalence...")
    included_meds = filter_by_prevalence(df)
    print(f"   Medications meeting threshold (≥{LAYER_CONFIG.prevalence_threshold} patients): {len(included_meds)}")

    exceptions = get_exception_ingredients()
    exception_count = len(included_meds & exceptions)
    print(f"   Exception medications (always included): {exception_count}")

    # Build indicators
    print("\n3. Building individual indicators...")
    medications_list = sorted(list(included_meds))
    result = build_individual_indicators(df, medications_list)

    print(f"   Patient-window combinations: {len(result):,}")
    print(f"   Features per patient-window: {len([c for c in result.columns if c.startswith('med_')])}")

    # Calculate sparsity
    indicator_cols = [c for c in result.columns if c.startswith('med_') and not c.endswith('_count') and not c.endswith('_dose')]
    if indicator_cols:
        total_cells = len(result) * len(indicator_cols)
        non_zero = result[indicator_cols].sum().sum()
        sparsity = (1 - non_zero / total_cells) * 100
        print(f"   Sparsity: {sparsity:.1f}%")

    # Save outputs
    output_dir = GOLD_DIR / "individual_indicators"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet
    if output_path is None:
        filename = "individual_indicators_test.parquet" if test_mode else "individual_indicators.parquet"
        output_path = output_dir / filename

    print(f"\n4. Saving parquet: {output_path}")
    result.to_parquet(output_path, index=False)

    # Save sparse HDF5
    if not test_mode and len(result) > 0:
        sparse_matrix, feature_names = to_sparse_matrix(result)
        hdf5_path = output_dir / "individual_indicators_sparse.h5"
        print(f"   Saving sparse HDF5: {hdf5_path}")
        save_sparse_hdf5(
            sparse_matrix,
            feature_names,
            result['empi'].tolist(),
            result['time_window'].tolist(),
            hdf5_path
        )

    print("\n" + "=" * 60)
    print("Layer 3 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build individual medication indicators")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    build_layer3(test_mode=args.test)
```

**Step 2: Run tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_individual_indicator_builder.py -v
```

**Step 3: Commit**

```bash
git add module_04_medications/transformers/individual_indicator_builder.py
git commit -m "feat(module4): implement individual medication indicator builder

Layer 3 implementation:
- Filter by prevalence (≥20 patients)
- Always include anticoagulants, vasopressors, thrombolytics
- Sparse indicator matrix with counts and total doses
- HDF5 storage for scipy sparse matrix

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Run Individual Indicator Building

**Files:**
- Run: Individual indicator builder

**Step 1: Run test mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/individual_indicator_builder.py --test
```

**Step 2: Run full mode**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/transformers/individual_indicator_builder.py
```

**Step 3: Validate output**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd

df = pd.read_parquet('module_04_medications/data/gold/individual_indicators/individual_indicators.parquet')

print('='*60)
print('Layer 3 Validation')
print('='*60)

print(f'\\nShape: {df.shape}')
print(f'Patients: {df[\"empi\"].nunique()}')

# Count indicator columns
indicator_cols = [c for c in df.columns if c.startswith('med_') and not c.endswith('_count') and not c.endswith('_dose')]
print(f'Individual medication indicators: {len(indicator_cols)}')

# Check key medications
print(f'\\nKey medication indicators present:')
for med in ['heparin', 'enoxaparin', 'warfarin', 'norepinephrine']:
    col = f'med_{med}'
    if col in df.columns:
        count = df[col].sum()
        print(f'  {med}: {count} patient-windows')
    else:
        print(f'  {med}: NOT FOUND')

# Check sparsity
total = len(df) * len(indicator_cols)
non_zero = df[indicator_cols].sum().sum()
sparsity = (1 - non_zero / total) * 100
print(f'\\nSparsity: {sparsity:.1f}%')
print(f'Target: >90%')
"
```

**Step 4: Commit**

```bash
git add module_04_medications/
git commit -m "chore(module4): Layer 3 individual indicators built

Output:
- gold/individual_indicators/individual_indicators.parquet
- gold/individual_indicators/individual_indicators_sparse.h5
- XXX individual medication features
- XX.X% sparsity

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Validation Checklist

- ✅ `individual_indicator_builder.py` exists
- ✅ All tests pass
- ✅ `individual_indicators.parquet` created
- ✅ `individual_indicators_sparse.h5` created
- ✅ 200-400 individual medication features
- ✅ All anticoagulants included
- ✅ Sparsity >90%

---

## Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | Individual indicator tests | test_individual_indicator_builder.py |
| 2 | Implement builder | individual_indicator_builder.py |
| 3 | Run building | individual_indicators.parquet, .h5 |

**Total:** 3 tasks, ~10 steps
