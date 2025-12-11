# /home/moin/TDA_11_25/module_04_medications/transformers/individual_indicator_builder.py
"""
Individual Medication Indicator Builder
=======================================

Layer 3: Generate sparse binary indicators for individual medications.

Selection criteria:
- Medications appearing in >=20 patients
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
    print(f"   Medications meeting threshold (>={LAYER_CONFIG.prevalence_threshold} patients): {len(included_meds)}")

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
