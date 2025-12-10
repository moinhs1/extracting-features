# /home/moin/TDA_11_25/module_04_medications/transformers/class_indicator_builder.py
"""
Therapeutic Class Indicator Builder
===================================

Layer 2: Generate 53 therapeutic class binary indicators per patient-timewindow.

Uses ingredient mappings from Phase 3 and class definitions from therapeutic_classes.yaml.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    THERAPEUTIC_CLASSES_YAML,
    SILVER_DIR,
    GOLD_DIR,
    TEMPORAL_CONFIG,
    load_therapeutic_classes,
)


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

_class_definitions: Optional[Dict] = None
_ingredient_to_classes: Optional[Dict[str, List[str]]] = None


def _load_class_definitions() -> Dict:
    """Load and cache class definitions."""
    global _class_definitions
    if _class_definitions is None:
        _class_definitions = load_therapeutic_classes()
    return _class_definitions


def _build_ingredient_mapping() -> Dict[str, List[str]]:
    """Build mapping from ingredient name to class IDs."""
    global _ingredient_to_classes
    if _ingredient_to_classes is not None:
        return _ingredient_to_classes

    classes = _load_class_definitions()
    _ingredient_to_classes = {}

    def process_category(category_data: Dict, prefix: str = ''):
        for class_id, class_def in category_data.items():
            if class_id.startswith('_'):
                continue
            if not isinstance(class_def, dict):
                continue

            ingredients = class_def.get('ingredients', [])
            if isinstance(ingredients, str):
                ingredients = [ingredients]

            for ingredient in ingredients:
                ing_lower = ingredient.lower().strip()
                if ing_lower not in _ingredient_to_classes:
                    _ingredient_to_classes[ing_lower] = []
                _ingredient_to_classes[ing_lower].append(class_id)

    # Process each category
    for category_name, category_data in classes.items():
        if isinstance(category_data, dict):
            process_category(category_data, category_name)

    return _ingredient_to_classes


def get_class_for_ingredient(ingredient_name: str) -> List[str]:
    """
    Get therapeutic class IDs for an ingredient.

    Args:
        ingredient_name: Ingredient name (case-insensitive)

    Returns:
        List of matching class IDs
    """
    if not ingredient_name:
        return []

    mapping = _build_ingredient_mapping()
    return mapping.get(ingredient_name.lower().strip(), [])


def get_all_class_ids() -> List[str]:
    """Get list of all 53 therapeutic class IDs."""
    classes = _load_class_definitions()
    class_ids = []

    for category_data in classes.values():
        if isinstance(category_data, dict):
            for class_id in category_data.keys():
                if not class_id.startswith('_'):
                    class_ids.append(class_id)

    return sorted(class_ids)


# =============================================================================
# DOSE-BASED CLASSIFICATION
# =============================================================================

# Dose thresholds for therapeutic vs prophylactic
DOSE_THRESHOLDS = {
    'heparin': {
        'therapeutic_min_units_day': 10000,
        'therapeutic_routes': ['IV'],
        'prophylactic_class': 'ac_ufh_proph',
        'therapeutic_class': 'ac_ufh_ther',
    },
    'enoxaparin': {
        'therapeutic_min_mg': 50,  # >40mg suggests therapeutic
        'prophylactic_class': 'ac_lmwh_proph',
        'therapeutic_class': 'ac_lmwh_ther',
    },
    'dalteparin': {
        'therapeutic_min_units': 10000,
        'prophylactic_class': 'ac_lmwh_proph',
        'therapeutic_class': 'ac_lmwh_ther',
    },
    'tinzaparin': {
        'therapeutic_min_units': 10000,
        'prophylactic_class': 'ac_lmwh_proph',
        'therapeutic_class': 'ac_lmwh_ther',
    },
}


def classify_anticoagulant_dose(
    ingredient: str,
    dose_value: Optional[float],
    dose_unit: Optional[str],
    route: Optional[str] = None
) -> Optional[str]:
    """
    Classify anticoagulant as therapeutic or prophylactic based on dose.

    Args:
        ingredient: Ingredient name
        dose_value: Numeric dose
        dose_unit: Dose unit
        route: Administration route

    Returns:
        Class ID (e.g., 'ac_ufh_ther') or None if not classifiable
    """
    if not ingredient or dose_value is None:
        return None

    ingredient = ingredient.lower().strip()

    if ingredient not in DOSE_THRESHOLDS:
        return None

    thresholds = DOSE_THRESHOLDS[ingredient]

    # Heparin classification
    if ingredient == 'heparin':
        if route and route.upper() == 'IV':
            return thresholds['therapeutic_class']
        if dose_unit and 'unit' in dose_unit.lower():
            if dose_value > thresholds['therapeutic_min_units_day']:
                return thresholds['therapeutic_class']
            else:
                return thresholds['prophylactic_class']

    # Enoxaparin classification
    elif ingredient == 'enoxaparin':
        if dose_unit and dose_unit.lower() == 'mg':
            if dose_value > thresholds['therapeutic_min_mg']:
                return thresholds['therapeutic_class']
            else:
                return thresholds['prophylactic_class']

    # Dalteparin/Tinzaparin classification
    elif ingredient in ['dalteparin', 'tinzaparin']:
        if dose_unit and 'unit' in dose_unit.lower():
            if dose_value > thresholds['therapeutic_min_units']:
                return thresholds['therapeutic_class']
            else:
                return thresholds['prophylactic_class']

    return None


# =============================================================================
# TIME WINDOW ASSIGNMENT
# =============================================================================

def get_time_window(hours_from_t0: float) -> Optional[str]:
    """
    Assign hours to time window.

    Args:
        hours_from_t0: Hours relative to Time Zero

    Returns:
        Window name ('baseline', 'acute', 'subacute', 'recovery') or None
    """
    windows = TEMPORAL_CONFIG.windows

    for window_name, (start, end) in windows.items():
        if start <= hours_from_t0 < end:
            return window_name

    return None


# =============================================================================
# CLASS ASSIGNMENT
# =============================================================================

def assign_classes(row: pd.Series) -> List[str]:
    """
    Assign therapeutic classes to a medication record.

    Args:
        row: Series with ingredient_name, parsed_dose_value, parsed_dose_unit, parsed_route

    Returns:
        List of assigned class IDs
    """
    ingredient = row.get('ingredient_name')
    if not ingredient or pd.isna(ingredient):
        return []

    ingredient = str(ingredient).lower().strip()

    # Get base classes for ingredient
    base_classes = get_class_for_ingredient(ingredient)

    # For anticoagulants, try dose-based classification
    dose_class = classify_anticoagulant_dose(
        ingredient=ingredient,
        dose_value=row.get('parsed_dose_value'),
        dose_unit=row.get('parsed_dose_unit'),
        route=row.get('parsed_route'),
    )

    if dose_class:
        # Replace generic anticoag class with specific one
        result = []
        for cls in base_classes:
            if cls.startswith('ac_ufh') or cls.startswith('ac_lmwh'):
                continue  # Skip generic, use dose-based
            result.append(cls)
        result.append(dose_class)
        return result

    return base_classes


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_class_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate class indicators per patient-timewindow.

    Args:
        df: DataFrame with empi, hours_from_t0, class_id columns

    Returns:
        DataFrame with one row per patient-window and class indicator columns
    """
    all_classes = get_all_class_ids()

    # Assign time windows
    df = df.copy()
    df['time_window'] = df['hours_from_t0'].apply(get_time_window)

    # Filter out records outside windows
    df = df[df['time_window'].notna()]

    if len(df) == 0:
        return pd.DataFrame()

    # Create aggregation structure
    results = []

    for (empi, window), group in df.groupby(['empi', 'time_window']):
        row = {
            'empi': empi,
            'time_window': window,
        }

        # Add window boundaries
        window_bounds = TEMPORAL_CONFIG.windows.get(window, (0, 0))
        row['window_start_hours'] = window_bounds[0]
        row['window_end_hours'] = window_bounds[1]

        # Initialize all class indicators to False
        for cls in all_classes:
            row[cls] = False
            row[f'{cls}_count'] = 0
            row[f'{cls}_first_hours'] = np.nan

        # Process each class occurrence
        for _, med_row in group.iterrows():
            class_id = med_row['class_id']
            if class_id and class_id in all_classes:
                row[class_id] = True
                row[f'{class_id}_count'] += 1

                hours = med_row['hours_from_t0']
                if pd.isna(row[f'{class_id}_first_hours']) or hours < row[f'{class_id}_first_hours']:
                    row[f'{class_id}_first_hours'] = hours

        results.append(row)

    return pd.DataFrame(results)


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_class_indicators(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Build Layer 2 therapeutic class indicators.

    Args:
        input_path: Path to mapped_medications.parquet
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with class indicators
    """
    print("=" * 60)
    print("Layer 2: Therapeutic Class Indicators")
    print("=" * 60)

    # Load data
    if input_path is None:
        input_path = SILVER_DIR / "mapped_medications.parquet"

    print(f"\n1. Loading mapped medications: {input_path}")
    df = pd.read_parquet(input_path)

    if test_mode:
        # Sample patients for testing
        sample_empis = df['empi'].unique()[:100]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")
    print(f"   Patients: {df['empi'].nunique():,}")

    # Assign classes to each record
    print("\n2. Assigning therapeutic classes...")
    df['assigned_classes'] = df.apply(assign_classes, axis=1)

    # Explode to one row per class assignment
    df_exploded = df.explode('assigned_classes')
    df_exploded = df_exploded.rename(columns={'assigned_classes': 'class_id'})
    df_exploded = df_exploded[df_exploded['class_id'].notna()]

    print(f"   Class assignments: {len(df_exploded):,}")

    # Aggregate per patient-window
    print("\n3. Aggregating per patient-window...")
    result = aggregate_class_indicators(df_exploded)

    print(f"   Patient-window combinations: {len(result):,}")

    # Statistics
    all_classes = get_all_class_ids()
    print(f"\n   Total classes: {len(all_classes)}")

    # Count patients with each class in acute window
    acute = result[result['time_window'] == 'acute']
    if len(acute) > 0:
        print("\n   Top 10 classes in acute window:")
        class_counts = {cls: acute[cls].sum() for cls in all_classes}
        sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for cls, count in sorted_counts:
            print(f"     {cls}: {count} patients")

    # Save output
    if output_path is None:
        output_dir = GOLD_DIR / "therapeutic_classes"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "class_indicators_test.parquet" if test_mode else "class_indicators.parquet"
        output_path = output_dir / filename

    print(f"\n4. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("Layer 2 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build therapeutic class indicators")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    build_class_indicators(test_mode=args.test)
