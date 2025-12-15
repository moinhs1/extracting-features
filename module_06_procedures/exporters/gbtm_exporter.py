"""
GBTM Exporter for Procedures
=============================

Export procedure features for Group-Based Trajectory Modeling in R (lcmm).

Output format:
- CSV with columns: empi, day_from_pe, provoking features, treatment indicators
- Wide format for R package
- Daily resolution, days -30 to +30 relative to Time Zero
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import GOLD_DIR, EXPORTS_DIR


def load_pe_features() -> pd.DataFrame:
    """Load PE-specific features from gold layer."""
    path = GOLD_DIR / "pe_procedure_features" / "pe_features.parquet"

    if not path.exists():
        # Return empty DataFrame with expected schema for testing
        return pd.DataFrame({
            'empi': [],
            'day_from_pe': [],
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

    return pd.read_parquet(path)


def load_ccs_indicators() -> pd.DataFrame:
    """Load CCS category indicators from gold layer."""
    path = GOLD_DIR / "ccs_indicators" / "ccs_indicators.parquet"

    if not path.exists():
        return pd.DataFrame({
            'empi': [],
            'temporal_category': [],
        })

    return pd.read_parquet(path)


def create_daily_features(
    pe_df: pd.DataFrame,
    ccs_df: pd.DataFrame,
    day_start: int = -30,
    day_end: int = 30
) -> pd.DataFrame:
    """
    Create daily procedure features for GBTM.

    Args:
        pe_df: PE-specific features
        ccs_df: CCS indicators
        day_start: Start day relative to PE
        day_end: End day relative to PE

    Returns:
        DataFrame with daily features
    """
    # Get unique patients
    patients = set(pe_df['empi'].unique()) | set(ccs_df['empi'].unique())

    results = []

    for empi in patients:
        patient_pe = pe_df[pe_df['empi'] == empi]
        patient_ccs = ccs_df[ccs_df['empi'] == empi]

        for day in range(day_start, day_end + 1):
            row = {
                'empi': empi,
                'day_from_pe': day,
            }

            # Provoking features (pre-PE)
            if day < 0:
                # Surgery within 30 days before PE
                if len(patient_pe) > 0:
                    row['surgery_within_30d'] = int(patient_pe['surgery_within_30_days'].any()) if 'surgery_within_30_days' in patient_pe.columns else 0
                    row['provoked_pe'] = int(patient_pe['provoked_pe'].any()) if 'provoked_pe' in patient_pe.columns else 0
                else:
                    row['surgery_within_30d'] = 0
                    row['provoked_pe'] = 0

            # Diagnostic workup (day -1 to +1)
            if -1 <= day <= 1:
                if len(patient_pe) > 0:
                    row['cta_performed'] = int(patient_pe['cta_chest_performed'].any()) if 'cta_chest_performed' in patient_pe.columns else 0
                else:
                    row['cta_performed'] = 0
            else:
                row['cta_performed'] = 0

            # Treatment indicators (day 0+)
            if day >= 0:
                if len(patient_pe) > 0:
                    row['intubation'] = int(patient_pe['intubation_performed'].any()) if 'intubation_performed' in patient_pe.columns else 0
                    row['ecmo'] = int(patient_pe['ecmo_initiated'].any()) if 'ecmo_initiated' in patient_pe.columns else 0
                    row['ivc_filter'] = int(patient_pe['ivc_filter_placed'].any()) if 'ivc_filter_placed' in patient_pe.columns else 0
                    row['cdt'] = int(patient_pe['catheter_directed_therapy'].any()) if 'catheter_directed_therapy' in patient_pe.columns else 0
                    row['thrombolysis'] = int(patient_pe['systemic_thrombolysis'].any()) if 'systemic_thrombolysis' in patient_pe.columns else 0
                    row['reperfusion'] = int(patient_pe['any_reperfusion_therapy'].any()) if 'any_reperfusion_therapy' in patient_pe.columns else 0
                    row['central_line'] = int(patient_pe['central_line_placed'].any()) if 'central_line_placed' in patient_pe.columns else 0
                    row['mechanical_vent'] = int(patient_pe['mechanical_ventilation'].any()) if 'mechanical_ventilation' in patient_pe.columns else 0
                    row['transfusion'] = int(patient_pe['any_transfusion'].any()) if 'any_transfusion' in patient_pe.columns else 0
                else:
                    row['intubation'] = 0
                    row['ecmo'] = 0
                    row['ivc_filter'] = 0
                    row['cdt'] = 0
                    row['thrombolysis'] = 0
                    row['reperfusion'] = 0
                    row['central_line'] = 0
                    row['mechanical_vent'] = 0
                    row['transfusion'] = 0

            # Support trajectory indicators (cumulative)
            if day > 0:
                # Max support level reached by this day
                row['max_support_reached'] = 0
                if len(patient_pe) > 0:
                    if 'ecmo_initiated' in patient_pe.columns and patient_pe['ecmo_initiated'].any():
                        row['max_support_reached'] = 5
                    elif 'intubation_performed' in patient_pe.columns and patient_pe['intubation_performed'].any():
                        row['max_support_reached'] = 3
                    elif 'central_line_placed' in patient_pe.columns and patient_pe['central_line_placed'].any():
                        row['max_support_reached'] = 1
            else:
                row['max_support_reached'] = 0

            results.append(row)

    return pd.DataFrame(results)


def export_gbtm(
    output_dir: Optional[Path] = None,
    day_start: int = -30,
    day_end: int = 30
) -> pd.DataFrame:
    """
    Main GBTM export function.

    Args:
        output_dir: Output directory
        day_start: Start day relative to PE
        day_end: End day relative to PE

    Returns:
        DataFrame with exported features
    """
    print("=" * 60)
    print("GBTM Export for Procedures")
    print("=" * 60)

    if output_dir is None:
        output_dir = EXPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n1. Loading gold layer data...")
    pe_df = load_pe_features()
    ccs_df = load_ccs_indicators()
    print(f"   PE features: {len(pe_df):,} rows")
    print(f"   CCS indicators: {len(ccs_df):,} rows")

    # Create features
    print(f"\n2. Creating daily features (days {day_start} to {day_end})...")
    features = create_daily_features(pe_df, ccs_df, day_start, day_end)
    print(f"   Feature matrix: {features.shape}")

    if len(features) > 0:
        print(f"   Patients: {features['empi'].nunique()}")
        print(f"   Days per patient: {len(features) / features['empi'].nunique():.1f}")

    # Export
    output_path = output_dir / "gbtm_procedures.csv"
    print(f"\n3. Exporting to: {output_path}")
    features.to_csv(output_path, index=False)

    if output_path.exists():
        print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 60)
    print("GBTM Export Complete!")
    print("=" * 60)

    return features


if __name__ == "__main__":
    export_gbtm()
