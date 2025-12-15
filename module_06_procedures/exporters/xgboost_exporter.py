"""
XGBoost Exporter for Procedures
================================

Export procedure features as wide tabular format for XGBoost.

Combines:
- Layer 2: CCS category indicators × temporal windows
- Layer 3: PE-specific procedure features
- Derived features: procedure counts, support levels, trajectories

Output: ~500 features in flat feature vector per patient
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import GOLD_DIR, EXPORTS_DIR


def load_all_gold_layers() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all gold layer outputs.

    Returns:
        Tuple of (ccs_indicators, pe_features)
    """
    ccs_path = GOLD_DIR / "ccs_indicators" / "ccs_indicators.parquet"
    pe_path = GOLD_DIR / "pe_procedure_features" / "pe_features.parquet"

    # Load or create empty DataFrames
    if ccs_path.exists():
        ccs_df = pd.read_parquet(ccs_path)
    else:
        ccs_df = pd.DataFrame({
            'empi': [],
            'temporal_category': [],
        })

    if pe_path.exists():
        pe_df = pd.read_parquet(pe_path)
    else:
        pe_df = pd.DataFrame({
            'empi': [],
            'cta_chest_performed': [],
            'intubation_performed': [],
            'ecmo_initiated': [],
            'ivc_filter_placed': [],
            'catheter_directed_therapy': [],
            'systemic_thrombolysis': [],
            'any_reperfusion_therapy': [],
            'surgical_embolectomy': [],
            'central_line_placed': [],
            'mechanical_ventilation': [],
            'any_transfusion': [],
            'provoked_pe': [],
            'surgery_within_30_days': [],
        })

    return ccs_df, pe_df


def pivot_ccs_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot CCS indicators to wide format (one row per patient).

    Creates columns like: ccs_{code}_{temporal_category}

    Args:
        df: CCS indicators DataFrame

    Returns:
        Wide format DataFrame
    """
    if len(df) == 0:
        return pd.DataFrame({'empi': []})

    # Get CCS category columns (columns starting with 'ccs_')
    ccs_cols = [c for c in df.columns if c.startswith('ccs_')]

    result_rows = []

    for empi, group in df.groupby('empi'):
        row = {'empi': empi}

        for _, cat_row in group.iterrows():
            temporal_cat = cat_row.get('temporal_category', 'unknown')

            for ccs_col in ccs_cols:
                if ccs_col in cat_row:
                    col_name = f'{ccs_col}_{temporal_cat}'
                    row[col_name] = int(cat_row[ccs_col])

                    # Add count if available
                    count_col = f'{ccs_col}_count'
                    if count_col in cat_row:
                        row[f'{ccs_col}_{temporal_cat}_count'] = int(cat_row[count_col])

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def aggregate_pe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PE-specific features per patient.

    Args:
        df: PE features DataFrame

    Returns:
        Aggregated features per patient
    """
    if len(df) == 0:
        return pd.DataFrame({'empi': []})

    # Binary features (any occurrence)
    binary_features = [
        'cta_chest_performed',
        'intubation_performed',
        'ecmo_initiated',
        'ivc_filter_placed',
        'catheter_directed_therapy',
        'systemic_thrombolysis',
        'any_reperfusion_therapy',
        'surgical_embolectomy',
        'central_line_placed',
        'mechanical_ventilation',
        'any_transfusion',
        'provoked_pe',
        'surgery_within_30_days',
    ]

    # Filter to features that exist
    binary_features = [f for f in binary_features if f in df.columns]

    # Aggregate
    agg_dict = {}
    for feat in binary_features:
        agg_dict[feat] = 'max'  # Any occurrence = 1

    if len(agg_dict) == 0:
        # No features to aggregate
        result = df[['empi']].drop_duplicates()
    else:
        result = df.groupby('empi').agg(agg_dict).reset_index()

    return result


def create_derived_features(
    ccs_df: pd.DataFrame,
    pe_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create derived procedure features.

    Args:
        ccs_df: CCS indicators
        pe_df: PE features

    Returns:
        DataFrame with derived features
    """
    # Get all patients
    all_patients = set(ccs_df['empi'].unique()) | set(pe_df['empi'].unique())

    results = []

    for empi in all_patients:
        row = {'empi': empi}

        patient_ccs = ccs_df[ccs_df['empi'] == empi]
        patient_pe = pe_df[pe_df['empi'] == empi]

        # Total procedure count across all temporal categories
        if len(patient_ccs) > 0:
            ccs_count_cols = [c for c in patient_ccs.columns if c.endswith('_count')]
            if ccs_count_cols:
                row['total_procedure_count'] = int(patient_ccs[ccs_count_cols].sum().sum())
            else:
                row['total_procedure_count'] = 0
        else:
            row['total_procedure_count'] = 0

        # PE-specific counts
        if len(patient_pe) > 0:
            row['diagnostic_procedures'] = int(patient_pe['cta_chest_performed'].sum()) if 'cta_chest_performed' in patient_pe.columns else 0

            # Invasive procedures - check each column exists
            invasive_sum = 0
            if 'intubation_performed' in patient_pe.columns:
                invasive_sum += patient_pe['intubation_performed'].sum()
            if 'central_line_placed' in patient_pe.columns:
                invasive_sum += patient_pe['central_line_placed'].sum()
            row['invasive_procedures'] = int(invasive_sum)

            # Reperfusion procedures - check each column exists
            reperfusion_sum = 0
            if 'catheter_directed_therapy' in patient_pe.columns:
                reperfusion_sum += patient_pe['catheter_directed_therapy'].sum()
            if 'systemic_thrombolysis' in patient_pe.columns:
                reperfusion_sum += patient_pe['systemic_thrombolysis'].sum()
            row['reperfusion_procedures'] = int(reperfusion_sum)
        else:
            row['diagnostic_procedures'] = 0
            row['invasive_procedures'] = 0
            row['reperfusion_procedures'] = 0

        # Support level indicators
        if len(patient_pe) > 0:
            # Max support reached
            if 'ecmo_initiated' in patient_pe.columns and patient_pe['ecmo_initiated'].any():
                row['max_support_level'] = 5
            elif 'intubation_performed' in patient_pe.columns and patient_pe['intubation_performed'].any():
                row['max_support_level'] = 3
            elif 'central_line_placed' in patient_pe.columns and patient_pe['central_line_placed'].any():
                row['max_support_level'] = 1
            else:
                row['max_support_level'] = 0

            # Escalation indicator
            had_ecmo = 'ecmo_initiated' in patient_pe.columns and patient_pe['ecmo_initiated'].any()
            had_transfusion = 'any_transfusion' in patient_pe.columns and patient_pe['any_transfusion'].any()
            row['had_escalation'] = int(had_ecmo or had_transfusion)
        else:
            row['max_support_level'] = 0
            row['had_escalation'] = 0

        results.append(row)

    return pd.DataFrame(results)


def export_xgboost(
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Main XGBoost export function.

    Args:
        output_path: Output parquet path

    Returns:
        Wide feature DataFrame
    """
    print("=" * 60)
    print("XGBoost Export for Procedures")
    print("=" * 60)

    # Load data
    print("\n1. Loading gold layer data...")
    ccs_df, pe_df = load_all_gold_layers()
    print(f"   CCS indicators: {len(ccs_df):,} rows")
    print(f"   PE features: {len(pe_df):,} rows")

    # Pivot CCS indicators
    print("\n2. Processing CCS indicators...")
    ccs_wide = pivot_ccs_indicators(ccs_df)
    if len(ccs_wide) > 0:
        print(f"   CCS features: {len(ccs_wide.columns) - 1}")
    else:
        print(f"   CCS features: 0 (no data)")

    # Aggregate PE features
    print("\n3. Processing PE-specific features...")
    pe_agg = aggregate_pe_features(pe_df)
    if len(pe_agg) > 0:
        print(f"   PE features: {len(pe_agg.columns) - 1}")
    else:
        print(f"   PE features: 0 (no data)")

    # Create derived features
    print("\n4. Creating derived features...")
    derived = create_derived_features(ccs_df, pe_df)
    if len(derived) > 0:
        print(f"   Derived features: {len(derived.columns) - 1}")
    else:
        print(f"   Derived features: 0 (no data)")

    # Merge all
    print("\n5. Merging features...")
    if len(ccs_wide) > 0:
        result = ccs_wide
    else:
        result = pd.DataFrame({'empi': []})

    if len(pe_agg) > 0:
        if len(result) > 0:
            result = result.merge(pe_agg, on='empi', how='outer')
        else:
            result = pe_agg

    if len(derived) > 0:
        if len(result) > 0:
            result = result.merge(derived, on='empi', how='outer')
        else:
            result = derived

    # Fill NaN with 0
    result = result.fillna(0)

    print(f"   Final feature matrix: {result.shape}")
    if len(result) > 0:
        print(f"   Patients: {len(result)}")
        print(f"   Features per patient: {len(result.columns) - 1}")

    # Save
    if output_path is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "xgboost_procedure_features.parquet"

    print(f"\n6. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)

    if output_path.exists():
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("XGBoost Export Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    export_xgboost()
