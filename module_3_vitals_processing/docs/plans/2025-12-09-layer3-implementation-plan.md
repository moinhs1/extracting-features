# Layer 3 Feature Engineering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Layer 3 feature engineering pipeline that transforms Layer 2 hourly grid into ~295 time-series features per hour and ~3,500 summary features per patient.

**Architecture:** Modular design with separate calculators for rolling stats, trends, variability, and thresholds. Main builder orchestrates all calculators and produces two output files.

**Tech Stack:** pandas, numpy, numba (JIT), joblib (parallel), h5py, tqdm

---

## Module Structure

```
processing/
├── layer3/
│   ├── __init__.py
│   ├── composite_vitals.py      # shock_index, pulse_pressure
│   ├── rolling_stats.py         # mean, std, cv, min, max, range
│   ├── trend_features.py        # slope, R², direction
│   ├── variability_features.py  # RMSSD, successive_var
│   ├── threshold_features.py    # cumulative hours, time-to-first
│   ├── data_density.py          # observation percentages
│   └── summary_aggregator.py    # window-based aggregations
├── layer3_builder.py            # Main orchestrator
tests/
├── test_layer3/
│   ├── __init__.py
│   ├── test_composite_vitals.py
│   ├── test_rolling_stats.py
│   ├── test_trend_features.py
│   ├── test_variability_features.py
│   ├── test_threshold_features.py
│   ├── test_data_density.py
│   └── test_summary_aggregator.py
└── test_layer3_builder.py
```

---

## Task 1: Create Layer 3 Module Structure

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/__init__.py`
- Modify: `module_3_vitals_processing/config/vitals_config.py`
- Create: `module_3_vitals_processing/tests/test_layer3/__init__.py`

**Step 1: Create layer3 package directory**

```bash
mkdir -p module_3_vitals_processing/processing/layer3
mkdir -p module_3_vitals_processing/tests/test_layer3
```

**Step 2: Create __init__.py files**

`processing/layer3/__init__.py`:
```python
"""Layer 3 feature engineering submodules."""
```

`tests/test_layer3/__init__.py`:
```python
"""Tests for Layer 3 feature engineering."""
```

**Step 3: Update vitals_config.py with Layer 3 constants**

Add after line 80:
```python
# Layer 3 outputs
TIMESERIES_FEATURES_PATH = LAYER3_OUTPUT_DIR / 'timeseries_features.parquet'
SUMMARY_FEATURES_PATH = LAYER3_OUTPUT_DIR / 'summary_features.parquet'

# Layer 3 constants
ROLLING_WINDOWS = [6, 12, 24]  # hours

# Vitals including composites
LAYER3_VITALS = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP', 'shock_index', 'pulse_pressure']

# Summary windows (hours from PE)
SUMMARY_WINDOWS = {
    'pre': (-24, 0),       # Pre-PE baseline
    'acute': (0, 24),      # Acute phase
    'early': (24, 72),     # Early treatment response
    'stab': (72, 168),     # Stabilization (days 3-7)
    'recov': (168, 720),   # Recovery (days 7-30)
}

# Clinical thresholds for abnormal detection
CLINICAL_THRESHOLDS = {
    'tachycardia': ('HR', '>', 100),
    'bradycardia': ('HR', '<', 60),
    'hypotension': ('SBP', '<', 90),
    'hypertension': ('SBP', '>', 180),
    'hypoxemia': ('SPO2', '<', 92),
    'tachypnea': ('RR', '>', 24),
    'shock': ('MAP', '<', 65),
    'fever': ('TEMP', '>', 38.5),
    'hypothermia': ('TEMP', '<', 36),
    'high_shock_index': ('shock_index', '>', 0.9),
}

# Direction definitions (what "improving" means for each vital)
IMPROVING_DIRECTION = {
    'HR': 'toward_normal',      # Decreasing if >100, increasing if <60
    'SBP': 'increasing',        # Higher is better (away from hypotension)
    'DBP': 'stable',            # Stability preferred
    'MAP': 'increasing',        # Higher is better (>65 target)
    'RR': 'toward_normal',      # Decreasing if >20, increasing if <12
    'SPO2': 'increasing',       # Higher is always better
    'TEMP': 'toward_normal',    # Moving toward 37°C
    'shock_index': 'decreasing', # Lower is better (<0.7 target)
    'pulse_pressure': 'increasing',  # Wider is better (better cardiac output)
}
```

**Step 4: Verify structure**

```bash
ls -la module_3_vitals_processing/processing/layer3/
ls -la module_3_vitals_processing/tests/test_layer3/
```
Expected: Both directories exist with __init__.py files

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/__init__.py
git add module_3_vitals_processing/tests/test_layer3/__init__.py
git add module_3_vitals_processing/config/vitals_config.py
git commit -m "feat(layer3): create module structure and config constants"
```

---

## Task 2: Composite Vitals Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/composite_vitals.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_composite_vitals.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_composite_vitals.py`:
```python
"""Tests for composite vital sign calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.composite_vitals import calculate_shock_index, calculate_pulse_pressure, add_composite_vitals


class TestShockIndex:
    """Tests for shock index calculation."""

    def test_shock_index_basic(self):
        """Shock index = HR / SBP."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [100.0],
            'SBP': [100.0],
        })
        result = calculate_shock_index(df)
        assert result['shock_index'].iloc[0] == 1.0

    def test_shock_index_normal(self):
        """Normal shock index ~0.5-0.7."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [70.0],
            'SBP': [120.0],
        })
        result = calculate_shock_index(df)
        assert abs(result['shock_index'].iloc[0] - 0.583) < 0.01

    def test_shock_index_missing_hr(self):
        """Shock index NaN when HR missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [np.nan],
            'SBP': [120.0],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])

    def test_shock_index_missing_sbp(self):
        """Shock index NaN when SBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [np.nan],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])

    def test_shock_index_zero_sbp_returns_nan(self):
        """Shock index NaN when SBP is zero (avoid division by zero)."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [0.0],
        })
        result = calculate_shock_index(df)
        assert pd.isna(result['shock_index'].iloc[0])


class TestPulsePressure:
    """Tests for pulse pressure calculation."""

    def test_pulse_pressure_basic(self):
        """Pulse pressure = SBP - DBP."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [120.0],
            'DBP': [80.0],
        })
        result = calculate_pulse_pressure(df)
        assert result['pulse_pressure'].iloc[0] == 40.0

    def test_pulse_pressure_narrow(self):
        """Narrow pulse pressure indicates poor cardiac output."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [90.0],
            'DBP': [70.0],
        })
        result = calculate_pulse_pressure(df)
        assert result['pulse_pressure'].iloc[0] == 20.0

    def test_pulse_pressure_missing_sbp(self):
        """Pulse pressure NaN when SBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [np.nan],
            'DBP': [80.0],
        })
        result = calculate_pulse_pressure(df)
        assert pd.isna(result['pulse_pressure'].iloc[0])

    def test_pulse_pressure_missing_dbp(self):
        """Pulse pressure NaN when DBP missing."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'SBP': [120.0],
            'DBP': [np.nan],
        })
        result = calculate_pulse_pressure(df)
        assert pd.isna(result['pulse_pressure'].iloc[0])


class TestAddCompositeVitals:
    """Tests for adding both composites to DataFrame."""

    def test_add_both_composites(self):
        """Adds both shock_index and pulse_pressure columns."""
        df = pd.DataFrame({
            'EMPI': ['E001', 'E001'],
            'hour_from_pe': [0, 1],
            'HR': [80.0, 90.0],
            'SBP': [120.0, 110.0],
            'DBP': [80.0, 70.0],
        })
        result = add_composite_vitals(df)
        assert 'shock_index' in result.columns
        assert 'pulse_pressure' in result.columns
        assert len(result) == 2

    def test_preserves_existing_columns(self):
        """Existing columns preserved."""
        df = pd.DataFrame({
            'EMPI': ['E001'],
            'hour_from_pe': [0],
            'HR': [80.0],
            'SBP': [120.0],
            'DBP': [80.0],
            'MAP': [93.3],
            'RR': [16.0],
        })
        result = add_composite_vitals(df)
        assert 'MAP' in result.columns
        assert 'RR' in result.columns
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_composite_vitals.py -v
```
Expected: FAIL with "No module named 'processing.layer3.composite_vitals'"

**Step 3: Write implementation**

`processing/layer3/composite_vitals.py`:
```python
"""Composite vital sign calculations.

Composites:
- shock_index: HR / SBP (hemodynamic instability indicator)
- pulse_pressure: SBP - DBP (cardiac output indicator)
"""
import pandas as pd
import numpy as np


def calculate_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate shock index (HR / SBP).

    Normal: 0.5-0.7
    Elevated (>0.9): Indicates hemodynamic compromise

    Args:
        df: DataFrame with HR and SBP columns

    Returns:
        DataFrame with shock_index column added
    """
    result = df.copy()

    # Avoid division by zero - replace 0 with NaN
    sbp_safe = result['SBP'].replace(0, np.nan)

    result['shock_index'] = result['HR'] / sbp_safe
    return result


def calculate_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pulse pressure (SBP - DBP).

    Normal: 40-60 mmHg
    Narrow (<25): Poor cardiac output
    Wide (>60): Possible aortic regurgitation

    Args:
        df: DataFrame with SBP and DBP columns

    Returns:
        DataFrame with pulse_pressure column added
    """
    result = df.copy()
    result['pulse_pressure'] = result['SBP'] - result['DBP']
    return result


def add_composite_vitals(df: pd.DataFrame) -> pd.DataFrame:
    """Add both composite vital columns to DataFrame.

    Args:
        df: DataFrame with HR, SBP, DBP columns

    Returns:
        DataFrame with shock_index and pulse_pressure columns added
    """
    result = calculate_shock_index(df)
    result = calculate_pulse_pressure(result)
    return result
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_composite_vitals.py -v
```
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/composite_vitals.py
git add module_3_vitals_processing/tests/test_layer3/test_composite_vitals.py
git commit -m "feat(layer3): add composite vitals calculator (shock_index, pulse_pressure)"
```

---

## Task 3: Rolling Statistics Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/rolling_stats.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_rolling_stats.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_rolling_stats.py`:
```python
"""Tests for rolling window statistics."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.rolling_stats import (
    calculate_rolling_stats,
    ROLLING_STAT_FUNCTIONS,
)


class TestRollingStatFunctions:
    """Tests for individual rolling stat calculations."""

    def test_rolling_mean_6h(self):
        """Rolling mean calculated correctly for 6h window."""
        # 10 hours of data, values 0-9
        df = pd.DataFrame({
            'EMPI': ['E001'] * 10,
            'hour_from_pe': list(range(10)),
            'HR': [float(i) for i in range(10)],
            'mask_HR': [1] * 10,  # All observed
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Hour 5: mean of hours 0-5 = (0+1+2+3+4+5)/6 = 2.5
        assert abs(result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0] - 2.5) < 0.01

    def test_rolling_std_6h(self):
        """Rolling std calculated correctly."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Std of [70,72,74,76,78,80] = 3.74 (ddof=1)
        std_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_std'].iloc[0]
        assert abs(std_val - 3.74) < 0.1

    def test_rolling_cv_6h(self):
        """Coefficient of variation = std / mean."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [100.0] * 6,  # Constant values
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # CV of constant values = 0
        cv_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_cv'].iloc[0]
        assert cv_val == 0.0 or pd.isna(cv_val)  # 0/100 = 0 or NaN if std is NaN

    def test_rolling_range_6h(self):
        """Rolling range = max - min."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 80.0, 75.0, 85.0, 72.0, 78.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Range of [70,80,75,85,72,78] = 85-70 = 15
        range_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_range'].iloc[0]
        assert range_val == 15.0


class TestRollingStatsMultipleWindows:
    """Tests for multiple window sizes."""

    def test_multiple_windows(self):
        """All window sizes calculated."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR': [72.0] * 24,
            'mask_HR': [1] * 24,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6, 12, 24])

        # Check all columns exist
        assert 'HR_roll6h_mean' in result.columns
        assert 'HR_roll12h_mean' in result.columns
        assert 'HR_roll24h_mean' in result.columns


class TestRollingStatsWithMissing:
    """Tests for handling missing data (Tier 1-2 only)."""

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data for rolling stats."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 100.0, 100.0, 100.0],  # Last 3 are imputed
            'mask_HR': [1, 1, 1, 0, 0, 0],  # Tier 3-4 marked with mask=0
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        # Mean should only use first 3 values: (70+72+74)/3 = 72
        mean_val = result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0]
        assert abs(mean_val - 72.0) < 0.1

    def test_nan_when_no_observations(self):
        """Returns NaN when window has no Tier 1-2 data."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [100.0] * 6,  # All imputed
            'mask_HR': [0] * 6,  # No observations
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        assert pd.isna(result.loc[result['hour_from_pe'] == 5, 'HR_roll6h_mean'].iloc[0])


class TestRollingStatsMultipleVitals:
    """Tests for multiple vital signs."""

    def test_multiple_vitals(self):
        """Stats calculated for all vitals."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'SBP': [120.0] * 6,
            'mask_HR': [1] * 6,
            'mask_SBP': [1] * 6,
        })

        result = calculate_rolling_stats(df, vitals=['HR', 'SBP'], windows=[6])

        assert 'HR_roll6h_mean' in result.columns
        assert 'SBP_roll6h_mean' in result.columns


class TestRollingStatsMultiplePatients:
    """Tests for multiple patients."""

    def test_separate_by_patient(self):
        """Rolling stats computed separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6 + ['E002'] * 6,
            'hour_from_pe': list(range(6)) * 2,
            'HR': [70.0] * 6 + [90.0] * 6,  # Different values per patient
            'mask_HR': [1] * 12,
        })

        result = calculate_rolling_stats(df, vitals=['HR'], windows=[6])

        e001_mean = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 5)]['HR_roll6h_mean'].iloc[0]
        e002_mean = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 5)]['HR_roll6h_mean'].iloc[0]

        assert abs(e001_mean - 70.0) < 0.1
        assert abs(e002_mean - 90.0) < 0.1
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_rolling_stats.py -v
```
Expected: FAIL with "No module named 'processing.layer3.rolling_stats'"

**Step 3: Write implementation**

`processing/layer3/rolling_stats.py`:
```python
"""Rolling window statistics calculator.

Calculates rolling mean, std, cv, min, max, range for specified windows.
Only uses Tier 1-2 (observed + forward-fill) data for valid statistics.
"""
from typing import List, Dict, Callable
import pandas as pd
import numpy as np

# Rolling statistics to calculate
ROLLING_STAT_FUNCTIONS: Dict[str, Callable] = {
    'mean': lambda x: x.mean(),
    'std': lambda x: x.std(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
}


def calculate_rolling_stats(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate rolling window statistics for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital per window:
    - {vital}_roll{w}h_mean
    - {vital}_roll{w}h_std
    - {vital}_roll{w}h_cv (coefficient of variation)
    - {vital}_roll{w}h_min
    - {vital}_roll{w}h_max
    - {vital}_roll{w}h_range

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process
        windows: List of window sizes in hours

    Returns:
        DataFrame with rolling stat columns added
    """
    result = df.copy()

    # Process each patient separately
    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        for vital in vitals:
            mask_col = f'mask_{vital}'

            # Create masked values (NaN where not observed)
            if mask_col in patient_df.columns:
                masked_values = patient_df[vital].where(patient_df[mask_col] == 1, np.nan)
            else:
                masked_values = patient_df[vital]

            for window in windows:
                prefix = f'{vital}_roll{window}h'

                # Calculate rolling stats using only observed values
                rolling = masked_values.rolling(window=window, min_periods=1)

                patient_df[f'{prefix}_mean'] = rolling.mean()
                patient_df[f'{prefix}_std'] = rolling.std()
                patient_df[f'{prefix}_min'] = rolling.min()
                patient_df[f'{prefix}_max'] = rolling.max()

                # CV = std / mean
                patient_df[f'{prefix}_cv'] = patient_df[f'{prefix}_std'] / patient_df[f'{prefix}_mean']

                # Range = max - min
                patient_df[f'{prefix}_range'] = patient_df[f'{prefix}_max'] - patient_df[f'{prefix}_min']

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_rolling_stats.py -v
```
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/rolling_stats.py
git add module_3_vitals_processing/tests/test_layer3/test_rolling_stats.py
git commit -m "feat(layer3): add rolling statistics calculator (mean, std, cv, min, max, range)"
```

---

## Task 4: Trend Features Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/trend_features.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_trend_features.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_trend_features.py`:
```python
"""Tests for trend feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.trend_features import (
    calculate_trend_features,
    calculate_direction,
)


class TestTrendSlope:
    """Tests for slope calculation."""

    def test_slope_positive_trend(self):
        """Positive slope for increasing values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # +2 per hour
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope > 0
        assert abs(slope - 2.0) < 0.1  # Slope ~2

    def test_slope_negative_trend(self):
        """Negative slope for decreasing values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [80.0, 78.0, 76.0, 74.0, 72.0, 70.0],  # -2 per hour
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope < 0
        assert abs(slope - (-2.0)) < 0.1

    def test_slope_flat_trend(self):
        """Zero slope for constant values."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,  # Constant
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert abs(slope) < 0.01


class TestTrendR2:
    """Tests for R² calculation."""

    def test_r2_perfect_linear(self):
        """R² = 1.0 for perfect linear trend."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # Perfect linear
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        r2 = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h_r2'].iloc[0]
        assert r2 > 0.99

    def test_r2_noisy_trend(self):
        """R² < 1.0 for noisy trend."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 75.0, 72.0, 78.0, 74.0, 80.0],  # Noisy upward
            'mask_HR': [1] * 6,
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        r2 = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h_r2'].iloc[0]
        assert 0 < r2 < 1.0


class TestTrendDirection:
    """Tests for direction classification."""

    def test_direction_improving_hr_decreasing_from_tachycardia(self):
        """HR decreasing from >100 is improving."""
        # HR going from 110 down to 90 (toward normal)
        direction = calculate_direction(slope=-2.0, current_value=95.0, vital='HR')
        assert direction == 1  # Improving

    def test_direction_worsening_hr_increasing_to_tachycardia(self):
        """HR increasing toward >100 is worsening."""
        direction = calculate_direction(slope=2.0, current_value=95.0, vital='HR')
        assert direction == -1  # Worsening

    def test_direction_stable_flat_slope(self):
        """Flat slope is stable."""
        direction = calculate_direction(slope=0.0, current_value=72.0, vital='HR')
        assert direction == 0  # Stable

    def test_direction_sbp_increasing_is_improving(self):
        """SBP increasing (away from hypotension) is improving."""
        direction = calculate_direction(slope=2.0, current_value=100.0, vital='SBP')
        assert direction == 1  # Improving

    def test_direction_spo2_increasing_is_improving(self):
        """SpO2 increasing is always improving."""
        direction = calculate_direction(slope=1.0, current_value=94.0, vital='SPO2')
        assert direction == 1  # Improving

    def test_direction_shock_index_decreasing_is_improving(self):
        """Shock index decreasing is improving."""
        direction = calculate_direction(slope=-0.1, current_value=0.8, vital='shock_index')
        assert direction == 1  # Improving


class TestTrendFeaturesWithMissing:
    """Tests for handling missing data."""

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data for trends."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 50.0, 50.0, 50.0],  # Last 3 imputed (bad values)
            'mask_HR': [1, 1, 1, 0, 0, 0],
        })

        result = calculate_trend_features(df, vitals=['HR'], windows=[6])

        # Slope should be based on first 3 values only: positive
        slope = result.loc[result['hour_from_pe'] == 5, 'HR_slope6h'].iloc[0]
        assert slope > 0  # Would be negative if using all data
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_trend_features.py -v
```
Expected: FAIL with "No module named 'processing.layer3.trend_features'"

**Step 3: Write implementation**

`processing/layer3/trend_features.py`:
```python
"""Trend feature calculations.

Calculates slope, R², and clinical direction for vital sign trajectories.
"""
from typing import List
import pandas as pd
import numpy as np
from scipy import stats

# Normal ranges for direction calculation
NORMAL_RANGES = {
    'HR': (60, 100),
    'SBP': (90, 140),
    'DBP': (60, 90),
    'MAP': (65, 100),
    'RR': (12, 20),
    'SPO2': (95, 100),
    'TEMP': (36.5, 37.5),
    'shock_index': (0.5, 0.7),
    'pulse_pressure': (40, 60),
}

# Vitals where higher is better (regardless of current value)
HIGHER_IS_BETTER = {'SBP', 'MAP', 'SPO2', 'pulse_pressure'}
# Vitals where lower is better
LOWER_IS_BETTER = {'shock_index'}


def calculate_direction(slope: float, current_value: float, vital: str) -> int:
    """Determine if trend is improving, stable, or worsening.

    Args:
        slope: Slope of recent trend
        current_value: Current vital value
        vital: Name of vital sign

    Returns:
        1 = improving, 0 = stable, -1 = worsening
    """
    # Threshold for "stable" (small slope)
    if abs(slope) < 0.5:
        return 0

    if vital in HIGHER_IS_BETTER:
        return 1 if slope > 0 else -1

    if vital in LOWER_IS_BETTER:
        return 1 if slope < 0 else -1

    # For toward_normal vitals (HR, RR, TEMP, DBP)
    if vital in NORMAL_RANGES:
        low, high = NORMAL_RANGES[vital]
        mid = (low + high) / 2

        if current_value > high:
            # Above normal - decreasing is improving
            return 1 if slope < 0 else -1
        elif current_value < low:
            # Below normal - increasing is improving
            return 1 if slope > 0 else -1
        else:
            # In normal range - any change away from it is worsening
            return 0 if abs(slope) < 1.0 else -1

    return 0


def _calculate_slope_r2(values: np.ndarray) -> tuple:
    """Calculate linear regression slope and R².

    Args:
        values: Array of values (may contain NaN)

    Returns:
        (slope, r2) tuple, or (NaN, NaN) if insufficient data
    """
    # Remove NaN values
    mask = ~np.isnan(values)
    clean_values = values[mask]

    if len(clean_values) < 2:
        return np.nan, np.nan

    x = np.arange(len(clean_values))

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_values)
        r2 = r_value ** 2
        return slope, r2
    except:
        return np.nan, np.nan


def calculate_trend_features(
    df: pd.DataFrame,
    vitals: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """Calculate trend features for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital per window:
    - {vital}_slope{w}h: Linear regression slope
    - {vital}_slope{w}h_r2: R² of the regression
    - {vital}_direction{w}h: -1 (worsening), 0 (stable), 1 (improving)

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process
        windows: List of window sizes in hours

    Returns:
        DataFrame with trend feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        for vital in vitals:
            mask_col = f'mask_{vital}'

            # Create masked values (NaN where not observed)
            if mask_col in patient_df.columns:
                values = patient_df[vital].where(patient_df[mask_col] == 1, np.nan).values
            else:
                values = patient_df[vital].values

            for window in windows:
                prefix = f'{vital}_slope{window}h'

                slopes = []
                r2s = []
                directions = []

                for i in range(len(patient_df)):
                    start_idx = max(0, i - window + 1)
                    window_values = values[start_idx:i+1]

                    slope, r2 = _calculate_slope_r2(window_values)
                    slopes.append(slope)
                    r2s.append(r2)

                    # Calculate direction
                    current_value = values[i] if not np.isnan(values[i]) else patient_df[vital].iloc[i]
                    direction = calculate_direction(slope if not np.isnan(slope) else 0, current_value, vital)
                    directions.append(direction)

                patient_df[f'{prefix}'] = slopes
                patient_df[f'{prefix}_r2'] = r2s
                patient_df[f'{vital}_direction{window}h'] = directions

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_trend_features.py -v
```
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/trend_features.py
git add module_3_vitals_processing/tests/test_layer3/test_trend_features.py
git commit -m "feat(layer3): add trend features calculator (slope, R², direction)"
```

---

## Task 5: Variability Features Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/variability_features.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_variability_features.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_variability_features.py`:
```python
"""Tests for variability feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.variability_features import (
    calculate_rmssd,
    calculate_successive_var,
    calculate_variability_features,
)


class TestRMSSD:
    """Tests for RMSSD calculation."""

    def test_rmssd_constant_values(self):
        """RMSSD = 0 for constant values."""
        values = np.array([72.0, 72.0, 72.0, 72.0])
        rmssd = calculate_rmssd(values)
        assert rmssd == 0.0

    def test_rmssd_alternating_values(self):
        """RMSSD calculated for alternating values."""
        # Values: 70, 80, 70, 80 -> diffs: 10, 10, 10
        # RMSSD = sqrt(mean([100, 100, 100])) = 10
        values = np.array([70.0, 80.0, 70.0, 80.0])
        rmssd = calculate_rmssd(values)
        assert abs(rmssd - 10.0) < 0.1

    def test_rmssd_with_nan(self):
        """RMSSD handles NaN values."""
        values = np.array([70.0, np.nan, 80.0, 75.0])
        rmssd = calculate_rmssd(values)
        # Should use consecutive non-NaN pairs only
        assert not np.isnan(rmssd)

    def test_rmssd_insufficient_data(self):
        """RMSSD returns NaN with < 2 values."""
        values = np.array([72.0])
        rmssd = calculate_rmssd(values)
        assert np.isnan(rmssd)


class TestSuccessiveVar:
    """Tests for successive variance calculation."""

    def test_successive_var_constant(self):
        """Successive var = 0 for constant values."""
        values = np.array([72.0, 72.0, 72.0, 72.0])
        sv = calculate_successive_var(values)
        assert sv == 0.0

    def test_successive_var_calculated(self):
        """Successive var = sum of abs differences."""
        # Values: 70, 75, 72, 80 -> abs diffs: 5, 3, 8 -> sum = 16
        values = np.array([70.0, 75.0, 72.0, 80.0])
        sv = calculate_successive_var(values)
        assert sv == 16.0


class TestVariabilityFeatures:
    """Tests for full variability feature calculation."""

    def test_calculates_both_features(self):
        """Both RMSSD and successive_var calculated."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 73.0, 75.0, 72.0],
            'mask_HR': [1] * 6,
        })

        result = calculate_variability_features(df, vitals=['HR'])

        assert 'HR_rmssd' in result.columns
        assert 'HR_successive_var' in result.columns

    def test_excludes_tier3_4_data(self):
        """Only uses Tier 1-2 data."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 100.0, 100.0, 100.0, 100.0],
            'mask_HR': [1, 1, 0, 0, 0, 0],  # Only first 2 observed
        })

        result = calculate_variability_features(df, vitals=['HR'])

        # Should only use first 2 values, so successive_var = |72-70| = 2
        sv = result.loc[result['hour_from_pe'] == 5, 'HR_successive_var'].iloc[0]
        assert abs(sv - 2.0) < 0.1

    def test_multiple_patients(self):
        """Variability calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [70.0, 72.0, 74.0, 76.0] + [80.0, 85.0, 90.0, 95.0],
            'mask_HR': [1] * 8,
        })

        result = calculate_variability_features(df, vitals=['HR'])

        # E001: constant +2 changes, E002: constant +5 changes
        e001_sv = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)]['HR_successive_var'].iloc[0]
        e002_sv = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)]['HR_successive_var'].iloc[0]

        assert abs(e001_sv - 6.0) < 0.1  # 2+2+2
        assert abs(e002_sv - 15.0) < 0.1  # 5+5+5
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_variability_features.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`processing/layer3/variability_features.py`:
```python
"""Variability feature calculations.

Calculates RMSSD and successive variance for vital sign variability analysis.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_rmssd(values: np.ndarray) -> float:
    """Calculate Root Mean Square of Successive Differences.

    RMSSD = sqrt(mean((x[i+1] - x[i])^2))

    Args:
        values: Array of values (may contain NaN)

    Returns:
        RMSSD value, or NaN if insufficient data
    """
    # Remove NaN values while preserving consecutive pairs
    clean = values[~np.isnan(values)]

    if len(clean) < 2:
        return np.nan

    # Calculate successive differences
    diffs = np.diff(clean)

    if len(diffs) == 0:
        return np.nan

    # RMSSD = sqrt(mean(diffs^2))
    rmssd = np.sqrt(np.mean(diffs ** 2))
    return rmssd


def calculate_successive_var(values: np.ndarray) -> float:
    """Calculate sum of absolute successive differences.

    SV = sum(|x[i+1] - x[i]|)

    Args:
        values: Array of values (may contain NaN)

    Returns:
        Successive variance value, or NaN if insufficient data
    """
    clean = values[~np.isnan(values)]

    if len(clean) < 2:
        return np.nan

    diffs = np.abs(np.diff(clean))
    return np.sum(diffs)


def calculate_variability_features(
    df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate variability features for each vital.

    Only uses data where mask_{vital} == 1 (Tier 1-2).

    Features generated per vital:
    - {vital}_rmssd: Root mean square of successive differences
    - {vital}_successive_var: Sum of absolute successive differences

    Args:
        df: DataFrame with vital columns and mask_{vital} columns
        vitals: List of vital names to process

    Returns:
        DataFrame with variability feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        for vital in vitals:
            mask_col = f'mask_{vital}'

            # Create masked values (NaN where not observed)
            if mask_col in patient_df.columns:
                values = patient_df[vital].where(patient_df[mask_col] == 1, np.nan).values
            else:
                values = patient_df[vital].values

            # Calculate cumulative variability at each time point
            rmssds = []
            svs = []

            for i in range(len(patient_df)):
                window_values = values[:i+1]
                rmssds.append(calculate_rmssd(window_values))
                svs.append(calculate_successive_var(window_values))

            patient_df[f'{vital}_rmssd'] = rmssds
            patient_df[f'{vital}_successive_var'] = svs

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_variability_features.py -v
```
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/variability_features.py
git add module_3_vitals_processing/tests/test_layer3/test_variability_features.py
git commit -m "feat(layer3): add variability features calculator (RMSSD, successive_var)"
```

---

## Task 6: Threshold Features Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/threshold_features.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_threshold_features.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_threshold_features.py`:
```python
"""Tests for threshold feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.threshold_features import (
    calculate_threshold_features,
    CLINICAL_THRESHOLDS,
)


class TestCumulativeHours:
    """Tests for cumulative hours above/below threshold."""

    def test_hours_tachycardia_counting(self):
        """Counts hours with HR > 100."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [90.0, 105.0, 110.0, 95.0, 102.0, 98.0],
        })

        result = calculate_threshold_features(df)

        # Hours with HR > 100: hours 1, 2, 4 = 3 hours
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['hours_tachycardia'] == 3

    def test_hours_hypotension_counting(self):
        """Counts hours with SBP < 90."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'SBP': [120.0, 85.0, 88.0, 95.0, 82.0, 110.0],
        })

        result = calculate_threshold_features(df)

        # Hours with SBP < 90: hours 1, 2, 4 = 3 hours
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['hours_hypotension'] == 3

    def test_hours_cumulative_over_time(self):
        """Cumulative count increases over time."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [105.0, 105.0, 105.0, 105.0],  # All tachycardia
        })

        result = calculate_threshold_features(df)

        assert result[result['hour_from_pe'] == 0]['hours_tachycardia'].iloc[0] == 1
        assert result[result['hour_from_pe'] == 1]['hours_tachycardia'].iloc[0] == 2
        assert result[result['hour_from_pe'] == 2]['hours_tachycardia'].iloc[0] == 3
        assert result[result['hour_from_pe'] == 3]['hours_tachycardia'].iloc[0] == 4


class TestTimeToFirst:
    """Tests for time-to-first threshold crossing."""

    def test_time_to_first_tachycardia(self):
        """Finds first hour with HR > 100."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [80.0, 85.0, 90.0, 105.0, 110.0, 95.0],
        })

        result = calculate_threshold_features(df)

        # First tachycardia at hour 3
        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert final_row['time_to_first_tachycardia'] == 3

    def test_time_to_first_never_crossed(self):
        """NaN if threshold never crossed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0],  # Never > 100
        })

        result = calculate_threshold_features(df)

        final_row = result[result['hour_from_pe'] == 5].iloc[0]
        assert pd.isna(final_row['time_to_first_tachycardia'])

    def test_time_to_first_at_hour_zero(self):
        """Time = 0 if crossed at first hour."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'SBP': [85.0, 90.0, 95.0, 100.0],  # Hypotensive at hour 0
        })

        result = calculate_threshold_features(df)

        final_row = result[result['hour_from_pe'] == 3].iloc[0]
        assert final_row['time_to_first_hypotension'] == 0


class TestMultiplePatients:
    """Tests for handling multiple patients."""

    def test_separate_by_patient(self):
        """Thresholds calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [105.0, 105.0, 105.0, 105.0] + [80.0, 80.0, 80.0, 80.0],
        })

        result = calculate_threshold_features(df)

        e001_final = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)].iloc[0]
        e002_final = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)].iloc[0]

        assert e001_final['hours_tachycardia'] == 4
        assert e002_final['hours_tachycardia'] == 0


class TestShockIndexThreshold:
    """Tests for shock index threshold."""

    def test_hours_high_shock_index(self):
        """Counts hours with shock_index > 0.9."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'shock_index': [0.7, 0.95, 1.0, 0.8],
        })

        result = calculate_threshold_features(df)

        # Hours 1, 2 have shock_index > 0.9
        final_row = result[result['hour_from_pe'] == 3].iloc[0]
        assert final_row['hours_high_shock_index'] == 2
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_threshold_features.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`processing/layer3/threshold_features.py`:
```python
"""Threshold-based feature calculations.

Calculates cumulative hours and time-to-first for clinical thresholds.
"""
from typing import Dict, Tuple
import pandas as pd
import numpy as np

# Clinical thresholds: (vital, operator, value)
CLINICAL_THRESHOLDS: Dict[str, Tuple[str, str, float]] = {
    'tachycardia': ('HR', '>', 100),
    'bradycardia': ('HR', '<', 60),
    'hypotension': ('SBP', '<', 90),
    'hypertension': ('SBP', '>', 180),
    'hypoxemia': ('SPO2', '<', 92),
    'tachypnea': ('RR', '>', 24),
    'shock': ('MAP', '<', 65),
    'fever': ('TEMP', '>', 38.5),
    'hypothermia': ('TEMP', '<', 36),
    'high_shock_index': ('shock_index', '>', 0.9),
}

# Time-to-first thresholds (subset most clinically relevant)
TIME_TO_FIRST_THRESHOLDS = [
    'tachycardia', 'hypotension', 'hypoxemia', 'shock', 'high_shock_index'
]


def _apply_threshold(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    """Apply threshold comparison."""
    if operator == '>':
        return values > threshold
    elif operator == '<':
        return values < threshold
    elif operator == '>=':
        return values >= threshold
    elif operator == '<=':
        return values <= threshold
    else:
        raise ValueError(f"Unknown operator: {operator}")


def calculate_threshold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate threshold-based features.

    Uses all data (Tiers 1-4) since threshold crossing is robust to imputation.

    Features generated:
    - hours_{condition}: Cumulative hours meeting threshold
    - time_to_first_{condition}: Hours until first threshold crossing

    Args:
        df: DataFrame with vital columns

    Returns:
        DataFrame with threshold feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()

        # Calculate cumulative hours for each threshold
        for condition, (vital, operator, threshold) in CLINICAL_THRESHOLDS.items():
            col_name = f'hours_{condition}'

            if vital in patient_df.columns:
                crossing = _apply_threshold(patient_df[vital], operator, threshold)
                patient_df[col_name] = crossing.cumsum()
            else:
                patient_df[col_name] = 0

        # Calculate time-to-first for selected thresholds
        for condition in TIME_TO_FIRST_THRESHOLDS:
            vital, operator, threshold = CLINICAL_THRESHOLDS[condition]
            col_name = f'time_to_first_{condition}'

            if vital in patient_df.columns:
                crossing = _apply_threshold(patient_df[vital], operator, threshold)

                # Find first crossing
                first_idx = crossing.idxmax() if crossing.any() else None

                if first_idx is not None and crossing.loc[first_idx]:
                    first_hour = patient_df.loc[first_idx, 'hour_from_pe']
                    patient_df[col_name] = first_hour
                else:
                    patient_df[col_name] = np.nan
            else:
                patient_df[col_name] = np.nan

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_threshold_features.py -v
```
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/threshold_features.py
git add module_3_vitals_processing/tests/test_layer3/test_threshold_features.py
git commit -m "feat(layer3): add threshold features calculator (cumulative hours, time-to-first)"
```

---

## Task 7: Data Density Calculator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/data_density.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_data_density.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_data_density.py`:
```python
"""Tests for data density feature calculations."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.data_density import calculate_data_density


class TestObservationPercentage:
    """Tests for observation percentage calculation."""

    def test_obs_pct_all_observed(self):
        """100% when all hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [1] * 6,  # All observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 100.0

    def test_obs_pct_half_observed(self):
        """50% when half hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [1, 0, 1, 0, 1, 0],  # 3 of 6 observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 50.0

    def test_obs_pct_none_observed(self):
        """0% when no hours observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 6,
            'hour_from_pe': list(range(6)),
            'HR': [72.0] * 6,
            'mask_HR': [0] * 6,  # None observed
        })

        result = calculate_data_density(df, vitals=['HR'])

        final = result[result['hour_from_pe'] == 5].iloc[0]
        assert final['HR_obs_pct'] == 0.0


class TestObservationCount:
    """Tests for observation count calculation."""

    def test_obs_count_cumulative(self):
        """Count increases cumulatively."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [72.0] * 4,
            'mask_HR': [1, 1, 1, 1],
        })

        result = calculate_data_density(df, vitals=['HR'])

        assert result[result['hour_from_pe'] == 0]['HR_obs_count'].iloc[0] == 1
        assert result[result['hour_from_pe'] == 1]['HR_obs_count'].iloc[0] == 2
        assert result[result['hour_from_pe'] == 3]['HR_obs_count'].iloc[0] == 4


class TestAnyVitalObserved:
    """Tests for any-vital observation percentage."""

    def test_any_vital_obs_pct(self):
        """Tracks hours with ANY vital observed."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4,
            'hour_from_pe': list(range(4)),
            'HR': [72.0] * 4,
            'SBP': [120.0] * 4,
            'mask_HR': [1, 0, 0, 1],  # Observed at hours 0, 3
            'mask_SBP': [0, 1, 0, 0],  # Observed at hour 1
        })

        result = calculate_data_density(df, vitals=['HR', 'SBP'])

        # Hours with any observation: 0, 1, 3 = 3 of 4 = 75%
        final = result[result['hour_from_pe'] == 3].iloc[0]
        assert final['any_vital_obs_pct'] == 75.0


class TestMultiplePatients:
    """Tests for multiple patient handling."""

    def test_separate_by_patient(self):
        """Density calculated separately per patient."""
        df = pd.DataFrame({
            'EMPI': ['E001'] * 4 + ['E002'] * 4,
            'hour_from_pe': list(range(4)) * 2,
            'HR': [72.0] * 8,
            'mask_HR': [1, 1, 1, 1] + [1, 0, 0, 0],  # E001: all obs, E002: 1 obs
        })

        result = calculate_data_density(df, vitals=['HR'])

        e001_final = result[(result['EMPI'] == 'E001') & (result['hour_from_pe'] == 3)].iloc[0]
        e002_final = result[(result['EMPI'] == 'E002') & (result['hour_from_pe'] == 3)].iloc[0]

        assert e001_final['HR_obs_pct'] == 100.0
        assert e002_final['HR_obs_pct'] == 25.0
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_data_density.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`processing/layer3/data_density.py`:
```python
"""Data density feature calculations.

Tracks observation rates to help models assess feature reliability.
"""
from typing import List
import pandas as pd
import numpy as np


def calculate_data_density(
    df: pd.DataFrame,
    vitals: List[str],
) -> pd.DataFrame:
    """Calculate data density features.

    Features generated:
    - {vital}_obs_pct: Cumulative % of hours with Tier 1-2 data
    - {vital}_obs_count: Cumulative count of observed hours
    - any_vital_obs_pct: % of hours with ANY vital observed

    Args:
        df: DataFrame with mask_{vital} columns
        vitals: List of vital names to process

    Returns:
        DataFrame with density feature columns added
    """
    result = df.copy()

    patients = result['EMPI'].unique()

    all_results = []
    for patient in patients:
        patient_df = result[result['EMPI'] == patient].sort_values('hour_from_pe').copy()
        n_hours = len(patient_df)

        # Track any vital observed
        any_obs = pd.Series([False] * n_hours, index=patient_df.index)

        for vital in vitals:
            mask_col = f'mask_{vital}'

            if mask_col in patient_df.columns:
                observed = patient_df[mask_col] == 1

                # Cumulative count
                patient_df[f'{vital}_obs_count'] = observed.cumsum()

                # Cumulative percentage
                hours_so_far = pd.Series(range(1, n_hours + 1), index=patient_df.index)
                patient_df[f'{vital}_obs_pct'] = (patient_df[f'{vital}_obs_count'] / hours_so_far) * 100

                # Update any_obs
                any_obs = any_obs | observed
            else:
                patient_df[f'{vital}_obs_count'] = 0
                patient_df[f'{vital}_obs_pct'] = 0.0

        # Any vital observed percentage
        any_obs_count = any_obs.cumsum()
        hours_so_far = pd.Series(range(1, n_hours + 1), index=patient_df.index)
        patient_df['any_vital_obs_pct'] = (any_obs_count / hours_so_far) * 100

        all_results.append(patient_df)

    return pd.concat(all_results, ignore_index=True)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_data_density.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/data_density.py
git add module_3_vitals_processing/tests/test_layer3/test_data_density.py
git commit -m "feat(layer3): add data density calculator (obs_pct, obs_count)"
```

---

## Task 8: Summary Aggregator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3/summary_aggregator.py`
- Create: `module_3_vitals_processing/tests/test_layer3/test_summary_aggregator.py`

**Step 1: Write failing tests**

`tests/test_layer3/test_summary_aggregator.py`:
```python
"""Tests for summary aggregation."""
import pytest
import pandas as pd
import numpy as np
from processing.layer3.summary_aggregator import aggregate_to_summary, SUMMARY_WINDOWS


class TestSummaryWindows:
    """Tests for summary window definitions."""

    def test_summary_windows_defined(self):
        """All 5 clinical windows defined."""
        assert 'pre' in SUMMARY_WINDOWS
        assert 'acute' in SUMMARY_WINDOWS
        assert 'early' in SUMMARY_WINDOWS
        assert 'stab' in SUMMARY_WINDOWS
        assert 'recov' in SUMMARY_WINDOWS

    def test_window_ranges(self):
        """Windows have correct hour ranges."""
        assert SUMMARY_WINDOWS['pre'] == (-24, 0)
        assert SUMMARY_WINDOWS['acute'] == (0, 24)
        assert SUMMARY_WINDOWS['early'] == (24, 72)
        assert SUMMARY_WINDOWS['stab'] == (72, 168)
        assert SUMMARY_WINDOWS['recov'] == (168, 720)


class TestAggregateToSummary:
    """Tests for aggregating time-series to summary."""

    def test_produces_one_row_per_patient(self):
        """Output has one row per patient."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 10 + ['E002'] * 10,
            'hour_from_pe': list(range(10)) * 2,
            'HR_roll6h_mean': [72.0] * 20,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        assert len(result) == 2
        assert set(result['EMPI']) == {'E001', 'E002'}

    def test_aggregates_by_window(self):
        """Features aggregated per summary window."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 50,
            'hour_from_pe': list(range(50)),
            'HR_roll6h_mean': [70.0] * 24 + [80.0] * 26,  # Different in acute vs early
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Acute window (0-24) should have mean ~70
        # Early window (24-50) should have mean ~80
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 70.0) < 1.0
        assert abs(row['HR_roll6h_mean_early_mean'] - 80.0) < 1.0

    def test_mean_aggregation(self):
        """Mean aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),  # Acute window
            'HR_roll6h_mean': [70.0, 72.0, 74.0, 76.0] * 6,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Mean of [70, 72, 74, 76] = 73
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 73.0) < 0.1

    def test_max_aggregation(self):
        """Max aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [70.0] * 20 + [90.0] * 4,  # Max is 90
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        row = result.iloc[0]
        assert row['HR_roll6h_mean_acute_max'] == 90.0

    def test_min_aggregation(self):
        """Min aggregation calculated correctly."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [60.0] * 4 + [80.0] * 20,  # Min is 60
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        row = result.iloc[0]
        assert row['HR_roll6h_mean_acute_min'] == 60.0

    def test_handles_nan_in_window(self):
        """Handles NaN values in aggregation."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 24,
            'hour_from_pe': list(range(24)),
            'HR_roll6h_mean': [np.nan] * 12 + [72.0] * 12,
        })

        result = aggregate_to_summary(ts_df, feature_cols=['HR_roll6h_mean'])

        # Should ignore NaN and compute mean of non-NaN values
        row = result.iloc[0]
        assert abs(row['HR_roll6h_mean_acute_mean'] - 72.0) < 0.1

    def test_threshold_features_summed_by_window(self):
        """Threshold features give window-specific totals."""
        ts_df = pd.DataFrame({
            'EMPI': ['E001'] * 50,
            'hour_from_pe': list(range(50)),
            'hours_tachycardia': list(range(1, 51)),  # Cumulative 1-50
        })

        result = aggregate_to_summary(ts_df, feature_cols=['hours_tachycardia'])

        # For cumulative features, take max in each window
        row = result.iloc[0]
        # Acute window ends at hour 23, cumulative = 24
        assert row['hours_tachycardia_acute_max'] == 24
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_summary_aggregator.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`processing/layer3/summary_aggregator.py`:
```python
"""Summary aggregation for Layer 3 features.

Aggregates time-series features into per-patient summary over clinical windows.
"""
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

# Clinical summary windows (hours from PE)
SUMMARY_WINDOWS: Dict[str, Tuple[int, int]] = {
    'pre': (-24, 0),       # Pre-PE baseline
    'acute': (0, 24),      # Acute phase
    'early': (24, 72),     # Early treatment response
    'stab': (72, 168),     # Stabilization (days 3-7)
    'recov': (168, 720),   # Recovery (days 7-30)
}

# Aggregation functions to apply
AGGREGATIONS = ['mean', 'max', 'min']


def aggregate_to_summary(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """Aggregate time-series features to per-patient summary.

    For each feature and each window, calculates mean, max, min.

    Args:
        df: Time-series DataFrame with features
        feature_cols: List of feature columns to aggregate

    Returns:
        DataFrame with one row per patient, ~3500 summary features
    """
    patients = df['EMPI'].unique()

    summary_rows = []
    for patient in patients:
        patient_df = df[df['EMPI'] == patient].sort_values('hour_from_pe')

        row = {'EMPI': patient}

        for feature in feature_cols:
            if feature not in patient_df.columns:
                continue

            for window_name, (start_hour, end_hour) in SUMMARY_WINDOWS.items():
                # Filter to window
                window_mask = (patient_df['hour_from_pe'] >= start_hour) & (patient_df['hour_from_pe'] < end_hour)
                window_data = patient_df.loc[window_mask, feature]

                # Skip if no data in window
                if len(window_data) == 0 or window_data.isna().all():
                    for agg in AGGREGATIONS:
                        row[f'{feature}_{window_name}_{agg}'] = np.nan
                    continue

                # Apply aggregations
                row[f'{feature}_{window_name}_mean'] = window_data.mean()
                row[f'{feature}_{window_name}_max'] = window_data.max()
                row[f'{feature}_{window_name}_min'] = window_data.min()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/test_summary_aggregator.py -v
```
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3/summary_aggregator.py
git add module_3_vitals_processing/tests/test_layer3/test_summary_aggregator.py
git commit -m "feat(layer3): add summary aggregator for per-patient features"
```

---

## Task 9: Layer 3 Builder - Main Orchestrator

**Files:**
- Create: `module_3_vitals_processing/processing/layer3_builder.py`
- Create: `module_3_vitals_processing/tests/test_layer3_builder.py`

**Step 1: Write failing tests**

`tests/test_layer3_builder.py`:
```python
"""Tests for Layer 3 builder."""
import pytest
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from processing.layer3_builder import (
    build_layer3,
    load_layer2_with_masks,
    LAYER3_VITALS,
)


class TestLayer3Vitals:
    """Tests for vital sign list."""

    def test_includes_raw_vitals(self):
        """All 7 raw vitals included."""
        assert 'HR' in LAYER3_VITALS
        assert 'SBP' in LAYER3_VITALS
        assert 'DBP' in LAYER3_VITALS
        assert 'MAP' in LAYER3_VITALS
        assert 'RR' in LAYER3_VITALS
        assert 'SPO2' in LAYER3_VITALS
        assert 'TEMP' in LAYER3_VITALS

    def test_includes_composites(self):
        """Composite vitals included."""
        assert 'shock_index' in LAYER3_VITALS
        assert 'pulse_pressure' in LAYER3_VITALS


class TestLoadLayer2:
    """Tests for loading Layer 2 data."""

    def test_loads_parquet_and_hdf5(self, tmp_path):
        """Loads both parquet and HDF5 files."""
        # Create test Layer 2 parquet
        grid = pd.DataFrame({
            'EMPI': ['E001'] * 14,
            'hour_from_pe': list(range(7)) * 2,
            'vital_type': ['HR'] * 7 + ['SBP'] * 7,
            'mean': [72.0] * 7 + [120.0] * 7,
        })
        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        # Create test HDF5 with imputation tiers
        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 7, 7), dtype=np.int8))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('patient_index', data=np.array(['E001'], dtype=object), dtype=dt)
            f.create_dataset('vital_index', data=np.array(['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'], dtype=object), dtype=dt)

        result = load_layer2_with_masks(parquet_path, hdf5_path)

        assert 'mask_HR' in result.columns
        assert 'mask_SBP' in result.columns


class TestBuildLayer3:
    """Tests for full Layer 3 build."""

    def test_build_produces_timeseries_output(self, tmp_path):
        """Build produces timeseries_features.parquet."""
        # Create minimal Layer 2 data
        grid = pd.DataFrame({
            'EMPI': ['E001'] * 70,  # 10 hours × 7 vitals
            'hour_from_pe': sorted(list(range(10)) * 7),
            'vital_type': ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'] * 10,
            'mean': [72.0, 120.0, 80.0, 93.0, 16.0, 98.0, 37.0] * 10,
        })

        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        # Create HDF5
        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 10, 7), dtype=np.int8))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('patient_index', data=np.array(['E001'], dtype=object), dtype=dt)
            f.create_dataset('vital_index', data=np.array(['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'], dtype=object), dtype=dt)

        ts_path = tmp_path / 'timeseries_features.parquet'
        summary_path = tmp_path / 'summary_features.parquet'

        build_layer3(
            layer2_parquet_path=parquet_path,
            layer2_hdf5_path=hdf5_path,
            timeseries_output_path=ts_path,
            summary_output_path=summary_path,
        )

        assert ts_path.exists()
        ts_df = pd.read_parquet(ts_path)
        assert 'EMPI' in ts_df.columns
        assert 'hour_from_pe' in ts_df.columns

    def test_build_produces_summary_output(self, tmp_path):
        """Build produces summary_features.parquet."""
        # Create minimal Layer 2 data
        grid = pd.DataFrame({
            'EMPI': ['E001'] * 70,
            'hour_from_pe': sorted(list(range(10)) * 7),
            'vital_type': ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'] * 10,
            'mean': [72.0, 120.0, 80.0, 93.0, 16.0, 98.0, 37.0] * 10,
        })

        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 10, 7), dtype=np.int8))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('patient_index', data=np.array(['E001'], dtype=object), dtype=dt)
            f.create_dataset('vital_index', data=np.array(['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'], dtype=object), dtype=dt)

        ts_path = tmp_path / 'timeseries_features.parquet'
        summary_path = tmp_path / 'summary_features.parquet'

        build_layer3(
            layer2_parquet_path=parquet_path,
            layer2_hdf5_path=hdf5_path,
            timeseries_output_path=ts_path,
            summary_output_path=summary_path,
        )

        assert summary_path.exists()
        summary_df = pd.read_parquet(summary_path)
        assert len(summary_df) == 1  # One patient
        assert 'EMPI' in summary_df.columns

    def test_timeseries_has_rolling_features(self, tmp_path):
        """Time-series output includes rolling statistics."""
        grid = pd.DataFrame({
            'EMPI': ['E001'] * 70,
            'hour_from_pe': sorted(list(range(10)) * 7),
            'vital_type': ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'] * 10,
            'mean': [72.0, 120.0, 80.0, 93.0, 16.0, 98.0, 37.0] * 10,
        })

        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 10, 7), dtype=np.int8))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('patient_index', data=np.array(['E001'], dtype=object), dtype=dt)
            f.create_dataset('vital_index', data=np.array(['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'], dtype=object), dtype=dt)

        ts_path = tmp_path / 'timeseries_features.parquet'
        summary_path = tmp_path / 'summary_features.parquet'

        build_layer3(
            layer2_parquet_path=parquet_path,
            layer2_hdf5_path=hdf5_path,
            timeseries_output_path=ts_path,
            summary_output_path=summary_path,
        )

        ts_df = pd.read_parquet(ts_path)
        assert 'HR_roll6h_mean' in ts_df.columns
        assert 'SBP_roll12h_std' in ts_df.columns

    def test_includes_composite_vitals(self, tmp_path):
        """Output includes shock_index and pulse_pressure features."""
        grid = pd.DataFrame({
            'EMPI': ['E001'] * 70,
            'hour_from_pe': sorted(list(range(10)) * 7),
            'vital_type': ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'] * 10,
            'mean': [72.0, 120.0, 80.0, 93.0, 16.0, 98.0, 37.0] * 10,
        })

        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 10, 7), dtype=np.int8))
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('patient_index', data=np.array(['E001'], dtype=object), dtype=dt)
            f.create_dataset('vital_index', data=np.array(['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'], dtype=object), dtype=dt)

        ts_path = tmp_path / 'timeseries_features.parquet'
        summary_path = tmp_path / 'summary_features.parquet'

        build_layer3(
            layer2_parquet_path=parquet_path,
            layer2_hdf5_path=hdf5_path,
            timeseries_output_path=ts_path,
            summary_output_path=summary_path,
        )

        ts_df = pd.read_parquet(ts_path)
        assert 'shock_index' in ts_df.columns
        assert 'pulse_pressure' in ts_df.columns
        assert 'shock_index_roll6h_mean' in ts_df.columns
```

**Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3_builder.py -v
```
Expected: FAIL

**Step 3: Write implementation**

`processing/layer3_builder.py`:
```python
"""Layer 3 Builder: Feature engineering for vital sign trajectories.

Transforms Layer 2 hourly grid into:
- timeseries_features.parquet: ~295 features per hour
- summary_features.parquet: ~3500 features per patient
"""
from typing import List
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from processing.layer3.composite_vitals import add_composite_vitals
from processing.layer3.rolling_stats import calculate_rolling_stats
from processing.layer3.trend_features import calculate_trend_features
from processing.layer3.variability_features import calculate_variability_features
from processing.layer3.threshold_features import calculate_threshold_features
from processing.layer3.data_density import calculate_data_density
from processing.layer3.summary_aggregator import aggregate_to_summary

# All vitals including composites
LAYER3_VITALS: List[str] = [
    'HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP',
    'shock_index', 'pulse_pressure'
]

# Raw vitals (from Layer 2)
RAW_VITALS: List[str] = ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']

# Rolling window sizes
ROLLING_WINDOWS: List[int] = [6, 12, 24]

# Get CPU count
N_JOBS = max(1, mp.cpu_count() - 1)


def load_layer2_with_masks(
    parquet_path: Path,
    hdf5_path: Path,
) -> pd.DataFrame:
    """Load Layer 2 data and add mask columns from HDF5.

    Pivots from long format (vital_type column) to wide format (one column per vital).
    Adds mask_{vital} columns from imputation_tier tensor.

    Args:
        parquet_path: Path to hourly_grid.parquet
        hdf5_path: Path to hourly_tensors.h5

    Returns:
        Wide-format DataFrame with vital values and mask columns
    """
    print(f"  Loading {parquet_path}")
    grid = pd.read_parquet(parquet_path)

    # Pivot from long to wide format
    print(f"  Pivoting to wide format...")
    wide = grid.pivot_table(
        index=['EMPI', 'hour_from_pe'],
        columns='vital_type',
        values='mean',
        aggfunc='first'
    ).reset_index()

    # Load imputation tiers from HDF5
    print(f"  Loading masks from {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        imputation_tiers = f['imputation_tier'][:]
        patient_index = [p.decode() if isinstance(p, bytes) else p for p in f['patient_index'][:]]
        vital_index = [v.decode() if isinstance(v, bytes) else v for v in f['vital_index'][:]]
        hour_index = list(range(-24, 721))  # Standard hour range

    # Create patient/vital/hour to tier mapping
    patient_to_idx = {p: i for i, p in enumerate(patient_index)}
    vital_to_idx = {v: i for i, v in enumerate(vital_index)}

    # Add mask columns (mask=1 for Tier 1-2, mask=0 for Tier 3-4)
    for vital in RAW_VITALS:
        if vital in vital_to_idx:
            v_idx = vital_to_idx[vital]
            mask_values = []

            for _, row in wide.iterrows():
                empi = row['EMPI']
                hour = int(row['hour_from_pe'])

                if empi in patient_to_idx and -24 <= hour <= 720:
                    p_idx = patient_to_idx[empi]
                    h_idx = hour + 24  # Convert to 0-indexed
                    tier = imputation_tiers[p_idx, h_idx, v_idx]
                    mask_values.append(1 if tier <= 2 else 0)
                else:
                    mask_values.append(0)

            wide[f'mask_{vital}'] = mask_values

    return wide


def build_layer3(
    layer2_parquet_path: Path,
    layer2_hdf5_path: Path,
    timeseries_output_path: Path,
    summary_output_path: Path,
) -> None:
    """Build Layer 3 features from Layer 2 data.

    Args:
        layer2_parquet_path: Path to hourly_grid.parquet
        layer2_hdf5_path: Path to hourly_tensors.h5
        timeseries_output_path: Output path for timeseries_features.parquet
        summary_output_path: Output path for summary_features.parquet
    """
    print("\n" + "=" * 60)
    print("Layer 3 Builder - Feature Engineering")
    print("=" * 60)
    print(f"CPU cores: {mp.cpu_count()}, using {N_JOBS} workers")

    # Step 1: Load Layer 2 data
    print("\n[1/8] Loading Layer 2 data...")
    df = load_layer2_with_masks(layer2_parquet_path, layer2_hdf5_path)
    print(f"  Loaded {len(df):,} rows from {df['EMPI'].nunique():,} patients")

    # Step 2: Add composite vitals
    print("\n[2/8] Calculating composite vitals...")
    df = add_composite_vitals(df)

    # Add mask columns for composites (same as components)
    df['mask_shock_index'] = (df['mask_HR'] == 1) & (df['mask_SBP'] == 1)
    df['mask_pulse_pressure'] = (df['mask_SBP'] == 1) & (df['mask_DBP'] == 1)
    df['mask_shock_index'] = df['mask_shock_index'].astype(int)
    df['mask_pulse_pressure'] = df['mask_pulse_pressure'].astype(int)

    print(f"  Added shock_index and pulse_pressure")

    # Step 3: Rolling statistics
    print("\n[3/8] Calculating rolling statistics...")
    df = calculate_rolling_stats(df, vitals=LAYER3_VITALS, windows=ROLLING_WINDOWS)

    # Step 4: Trend features
    print("\n[4/8] Calculating trend features...")
    df = calculate_trend_features(df, vitals=LAYER3_VITALS, windows=ROLLING_WINDOWS)

    # Step 5: Variability features
    print("\n[5/8] Calculating variability features...")
    df = calculate_variability_features(df, vitals=LAYER3_VITALS)

    # Step 6: Threshold features
    print("\n[6/8] Calculating threshold features...")
    df = calculate_threshold_features(df)

    # Step 7: Data density features
    print("\n[7/8] Calculating data density features...")
    df = calculate_data_density(df, vitals=LAYER3_VITALS)

    # Save time-series output
    print(f"\n  Saving time-series features to {timeseries_output_path}")
    timeseries_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(timeseries_output_path, index=False)

    # Step 8: Aggregate to summary
    print("\n[8/8] Aggregating to summary features...")

    # Get all feature columns (exclude identifiers and raw vitals)
    exclude_cols = {'EMPI', 'hour_from_pe'} | set(LAYER3_VITALS) | {f'mask_{v}' for v in LAYER3_VITALS}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    summary_df = aggregate_to_summary(df, feature_cols=feature_cols)

    print(f"  Saving summary features to {summary_output_path}")
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(summary_output_path, index=False)

    print("\n" + "=" * 60)
    print("Layer 3 COMPLETE")
    print("=" * 60)
    print(f"  Time-series: {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Summary: {len(summary_df):,} patients × {len(summary_df.columns)} features")


def main():
    """CLI entry point for Layer 3 builder."""
    from config.vitals_config import (
        HOURLY_GRID_PATH,
        HOURLY_TENSORS_PATH,
        TIMESERIES_FEATURES_PATH,
        SUMMARY_FEATURES_PATH,
    )

    print(f"\nInput:  {HOURLY_GRID_PATH}")
    print(f"Input:  {HOURLY_TENSORS_PATH}")
    print(f"Output: {TIMESERIES_FEATURES_PATH}")
    print(f"Output: {SUMMARY_FEATURES_PATH}")

    build_layer3(
        layer2_parquet_path=HOURLY_GRID_PATH,
        layer2_hdf5_path=HOURLY_TENSORS_PATH,
        timeseries_output_path=TIMESERIES_FEATURES_PATH,
        summary_output_path=SUMMARY_FEATURES_PATH,
    )


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3_builder.py -v
```
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add module_3_vitals_processing/processing/layer3_builder.py
git add module_3_vitals_processing/tests/test_layer3_builder.py
git commit -m "feat(layer3): add main builder orchestrating all feature calculators"
```

---

## Task 10: Integration Test on Real Data

**Files:**
- None created, uses existing outputs

**Step 1: Run Layer 3 builder on real data**

```bash
cd /home/moin/TDA_11_25
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH python module_3_vitals_processing/processing/layer3_builder.py
```

Expected output:
- timeseries_features.parquet created
- summary_features.parquet created
- ~295 columns in time-series
- ~3500 columns in summary
- 7,689 patients in summary

**Step 2: Verify outputs**

```bash
python -c "
import pandas as pd
ts = pd.read_parquet('module_3_vitals_processing/outputs/layer3/timeseries_features.parquet')
print(f'Time-series: {len(ts):,} rows × {len(ts.columns)} columns')
print(f'Patients: {ts[\"EMPI\"].nunique():,}')

summary = pd.read_parquet('module_3_vitals_processing/outputs/layer3/summary_features.parquet')
print(f'Summary: {len(summary):,} rows × {len(summary.columns)} columns')
"
```

**Step 3: Run all Layer 3 tests**

```bash
PYTHONPATH=module_3_vitals_processing:$PYTHONPATH pytest module_3_vitals_processing/tests/test_layer3/ module_3_vitals_processing/tests/test_layer3_builder.py -v
```

Expected: All tests pass

**Step 4: Commit integration verification**

```bash
git add -A
git commit -m "feat(layer3): verify integration on real data - Phase 2 complete"
```

---

## Summary

**Total Tasks:** 10

**Files Created:**
- `processing/layer3/__init__.py`
- `processing/layer3/composite_vitals.py`
- `processing/layer3/rolling_stats.py`
- `processing/layer3/trend_features.py`
- `processing/layer3/variability_features.py`
- `processing/layer3/threshold_features.py`
- `processing/layer3/data_density.py`
- `processing/layer3/summary_aggregator.py`
- `processing/layer3_builder.py`
- `tests/test_layer3/__init__.py`
- `tests/test_layer3/test_composite_vitals.py`
- `tests/test_layer3/test_rolling_stats.py`
- `tests/test_layer3/test_trend_features.py`
- `tests/test_layer3/test_variability_features.py`
- `tests/test_layer3/test_threshold_features.py`
- `tests/test_layer3/test_data_density.py`
- `tests/test_layer3/test_summary_aggregator.py`
- `tests/test_layer3_builder.py`

**Files Modified:**
- `config/vitals_config.py`

**Expected Test Count:** ~70 new tests

**Output Files:**
- `outputs/layer3/timeseries_features.parquet` (~295 features × ~5.7M rows)
- `outputs/layer3/summary_features.parquet` (~3500 features × 7,689 patients)
