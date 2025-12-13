"""
CCS Indicator Builder
=====================

Layer 2: Generate CCS category binary indicators per patient-timewindow.

Transforms Silver (mapped_procedures.parquet) -> Gold (ccs_indicators/).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import yaml
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    SURGICAL_RISK_YAML,
    SILVER_DIR,
    GOLD_DIR,
    TEMPORAL_CONFIG,
)


# =============================================================================
# SURGICAL RISK CLASSIFICATION
# =============================================================================

_surgical_risk_config: Optional[Dict] = None


def _load_surgical_risk_config() -> Dict:
    """Load and cache surgical risk classification."""
    global _surgical_risk_config
    if _surgical_risk_config is None:
        with open(SURGICAL_RISK_YAML, 'r') as f:
            _surgical_risk_config = yaml.safe_load(f)
    return _surgical_risk_config


def get_surgical_risk_level(ccs_category: str) -> Optional[str]:
    """
    Get VTE risk classification for a CCS category.

    Args:
        ccs_category: CCS category code (as string)

    Returns:
        Risk level: 'very_high', 'high', 'moderate', 'low', 'minimal', or None
    """
    if not ccs_category or pd.isna(ccs_category):
        return None

    ccs_category = str(ccs_category).strip()
    config = _load_surgical_risk_config()
    risk_levels = config.get('risk_levels', {})

    # Search through risk levels
    for risk_level, risk_data in risk_levels.items():
        ccs_categories = risk_data.get('ccs_categories', [])
        if ccs_category in [str(c) for c in ccs_categories]:
            return risk_level

    # Default to minimal if not found
    return 'minimal'


def get_invasiveness_level(ccs_category: str) -> int:
    """
    Get invasiveness level for a CCS category.

    Args:
        ccs_category: CCS category code

    Returns:
        Invasiveness level: 0 (non-invasive) to 3 (highly invasive)
    """
    if not ccs_category or pd.isna(ccs_category):
        return 0

    # Map risk level to invasiveness (approximate)
    risk_level = get_surgical_risk_level(ccs_category)

    risk_to_invasiveness = {
        'very_high': 3,
        'high': 3,
        'moderate': 2,
        'low': 1,
        'minimal': 0,
    }

    return risk_to_invasiveness.get(risk_level, 0)


# =============================================================================
# TIME WINDOW ASSIGNMENT
# =============================================================================

def get_time_window(hours_from_pe: float) -> Optional[str]:
    """
    Assign hours to time window.

    Args:
        hours_from_pe: Hours relative to PE Time Zero

    Returns:
        Window name or None if outside defined windows
    """
    if pd.isna(hours_from_pe):
        return None

    windows = TEMPORAL_CONFIG.windows

    # Check each window
    # Note: Windows may overlap (e.g., diagnostic_workup overlaps with initial_treatment)
    # Priority order: diagnostic_workup > initial_treatment > others

    # Diagnostic workup has highest priority (±24h around Time Zero)
    if windows['diagnostic_workup'][0] <= hours_from_pe <= windows['diagnostic_workup'][1]:
        return 'diagnostic_workup'

    # Initial treatment (0-72h)
    if windows['initial_treatment'][0] <= hours_from_pe <= windows['initial_treatment'][1]:
        return 'initial_treatment'

    # Provoking window (-720 to 0)
    if windows['provoking_window'][0] <= hours_from_pe < windows['provoking_window'][1]:
        return 'provoking_window'

    # Escalation (>72h, up to 720h)
    if windows['escalation'][0] < hours_from_pe <= windows['escalation'][1]:
        return 'escalation'

    # Post discharge (>720h)
    if hours_from_pe > windows['post_discharge'][0]:
        return 'post_discharge'

    # Lifetime history (<-720h)
    if hours_from_pe < windows['lifetime_history'][1]:
        return 'lifetime_history'

    return None


# =============================================================================
# CCS INDICATOR AGGREGATION
# =============================================================================

def _get_unique_ccs_categories(df: pd.DataFrame) -> List[str]:
    """Get sorted list of unique CCS categories in the data."""
    if 'ccs_category' not in df.columns:
        return []

    categories = df['ccs_category'].dropna().unique()
    return sorted([str(c) for c in categories])


def aggregate_ccs_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CCS indicators per patient-timewindow.

    Args:
        df: DataFrame with empi, hours_from_pe, ccs_category columns

    Returns:
        DataFrame with one row per patient-window and CCS indicator columns
    """
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Assign time windows
    df['time_window'] = df['hours_from_pe'].apply(get_time_window)

    # Filter out records outside windows
    df = df[df['time_window'].notna()]

    if len(df) == 0:
        return pd.DataFrame()

    # Get all unique CCS categories in the data
    all_ccs = _get_unique_ccs_categories(df)

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

        # Initialize all CCS indicators to False
        for ccs in all_ccs:
            row[f'ccs_{ccs}'] = False
            row[f'ccs_{ccs}_count'] = 0

        # Track surgical risk and invasiveness
        risk_levels = []
        invasiveness_levels = []

        # Process each procedure
        for _, proc_row in group.iterrows():
            ccs_cat = proc_row['ccs_category']
            if pd.isna(ccs_cat):
                continue

            ccs_cat = str(ccs_cat).strip()

            # Set binary indicator and count
            if ccs_cat in all_ccs:
                row[f'ccs_{ccs_cat}'] = True
                row[f'ccs_{ccs_cat}_count'] += 1

            # Track risk and invasiveness
            risk_level = get_surgical_risk_level(ccs_cat)
            if risk_level:
                risk_levels.append(risk_level)

            inv_level = get_invasiveness_level(ccs_cat)
            invasiveness_levels.append(inv_level)

        # Compute surgical risk level (maximum)
        if risk_levels:
            risk_order = ['minimal', 'low', 'moderate', 'high', 'very_high']
            max_risk = max(risk_levels, key=lambda r: risk_order.index(r) if r in risk_order else 0)
            row['surgical_risk_level'] = max_risk
        else:
            row['surgical_risk_level'] = None

        # Compute invasiveness max
        if invasiveness_levels:
            row['invasiveness_max'] = max(invasiveness_levels)
        else:
            row['invasiveness_max'] = 0

        # Total procedure count
        row['procedure_count'] = len(group)

        results.append(row)

    result_df = pd.DataFrame(results)

    return result_df


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_ccs_indicators(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Build Layer 2 CCS indicators.

    Args:
        input_path: Path to mapped_procedures.parquet (Silver)
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with CCS indicators
    """
    print("=" * 60)
    print("Layer 2: CCS Indicator Builder")
    print("=" * 60)

    # Load data
    if input_path is None:
        input_path = SILVER_DIR / "mapped_procedures.parquet"

    print(f"\n1. Loading mapped procedures: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_parquet(input_path)

    if test_mode:
        # Sample patients for testing
        sample_empis = df['empi'].unique()[:100]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")
    print(f"   Patients: {df['empi'].nunique():,}")

    # Filter to records with CCS mapping
    df_mapped = df[df['ccs_category'].notna()].copy()
    mapping_rate = len(df_mapped) / len(df) * 100 if len(df) > 0 else 0
    print(f"   Records with CCS mapping: {len(df_mapped):,} ({mapping_rate:.1f}%)")

    # Aggregate per patient-window
    print("\n2. Aggregating per patient-window...")
    result = aggregate_ccs_indicators(df_mapped)

    print(f"   Patient-window combinations: {len(result):,}")

    # Statistics
    if len(result) > 0:
        print("\n   Distribution by time window:")
        window_counts = result['time_window'].value_counts()
        for window, count in window_counts.items():
            print(f"     {window}: {count:,}")

        # Risk level distribution
        print("\n   Surgical risk distribution:")
        risk_counts = result['surgical_risk_level'].value_counts()
        for risk, count in risk_counts.items():
            print(f"     {risk}: {count:,}")

        # Invasiveness distribution
        print("\n   Invasiveness distribution:")
        inv_counts = result['invasiveness_max'].value_counts().sort_index()
        for inv, count in inv_counts.items():
            print(f"     Level {inv}: {count:,}")

    # Save output
    if output_path is None:
        output_dir = GOLD_DIR / "ccs_indicators"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "ccs_indicators_test.parquet" if test_mode else "ccs_indicators.parquet"
        output_path = output_dir / filename

    print(f"\n3. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    print("\n" + "=" * 60)
    print("Layer 2 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CCS indicators")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--input', type=Path, help='Input parquet file')
    parser.add_argument('--output', type=Path, help='Output parquet file')
    args = parser.parse_args()

    build_ccs_indicators(
        input_path=args.input,
        output_path=args.output,
        test_mode=args.test
    )
