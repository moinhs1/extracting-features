"""
GRU-D Exporter for Procedures
==============================

Export procedure features as HDF5 tensors for GRU-D neural network.

Output format:
- procedure_values: (n_patients, n_hours, n_features) - binary indicators
- procedure_mask: (n_patients, n_hours, n_features) - 1 where observed
- procedure_delta: (n_patients, n_hours, n_features) - hours since last observation

Hourly resolution for 7 days (168 hours) by default.
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import GOLD_DIR, SILVER_DIR, EXPORTS_DIR


def load_pe_features() -> pd.DataFrame:
    """Load PE-specific features from gold layer."""
    path = GOLD_DIR / "pe_procedure_features" / "pe_features.parquet"

    if not path.exists():
        # Return empty DataFrame for testing
        return pd.DataFrame({
            'empi': [],
            'hours_from_pe': [],
            'intubation_performed': [],
            'ecmo_initiated': [],
            'ivc_filter_placed': [],
            'catheter_directed_therapy': [],
            'central_line_placed': [],
            'mechanical_ventilation': [],
            'any_transfusion': [],
            'cta_chest_performed': [],
        })

    return pd.read_parquet(path)


def create_hourly_tensor(
    df: pd.DataFrame,
    patients: List[str],
    n_hours: int = 168,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Create hourly tensor for GRU-D.

    Args:
        df: PE features DataFrame
        patients: Ordered list of patient IDs
        n_hours: Number of hours (default 168 = 7 days)
        feature_cols: List of feature column names

    Returns:
        Tuple of (values, mask, delta, feature_names)
    """
    # Default feature set
    if feature_cols is None:
        feature_cols = [
            'cta_chest_performed',
            'intubation_performed',
            'ecmo_initiated',
            'ivc_filter_placed',
            'catheter_directed_therapy',
            'central_line_placed',
            'mechanical_ventilation',
            'any_transfusion',
            'systemic_thrombolysis',
            'any_reperfusion_therapy',
        ]

    # Filter to features that exist in DataFrame
    feature_cols = [f for f in feature_cols if f in df.columns]

    n_patients = len(patients)
    n_features = len(feature_cols)

    # Initialize tensors
    values = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)
    mask = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)
    delta = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)

    # Create patient index mapping
    patient_to_idx = {p: i for i, p in enumerate(patients)}

    # Process each record
    for _, row in df.iterrows():
        empi = row['empi']
        if empi not in patient_to_idx:
            continue

        patient_idx = patient_to_idx[empi]

        # Get hour (round to nearest hour)
        if 'hours_from_pe' in row:
            hour = int(round(row['hours_from_pe']))
        else:
            # Skip if no time information
            continue

        # Only include 0-167 hours (7 days)
        if hour < 0 or hour >= n_hours:
            continue

        # Set feature values
        for feat_idx, feat_name in enumerate(feature_cols):
            if feat_name in row:
                value = float(row[feat_name])
                if not pd.isna(value) and value > 0:
                    values[patient_idx, hour, feat_idx] = 1.0
                    mask[patient_idx, hour, feat_idx] = 1.0

    # Compute delta (hours since last observation)
    for p in range(n_patients):
        for f in range(n_features):
            last_obs = -1
            for h in range(n_hours):
                if mask[p, h, f] == 1:
                    delta[p, h, f] = h - last_obs if last_obs >= 0 else 0
                    last_obs = h
                else:
                    delta[p, h, f] = h - last_obs if last_obs >= 0 else h

    return values, mask, delta, feature_cols


def export_grud(
    output_path: Optional[Path] = None,
    n_hours: int = 168
) -> dict:
    """
    Main GRU-D export function.

    Args:
        output_path: Output HDF5 path
        n_hours: Number of hours to export

    Returns:
        Dictionary with tensor shapes and metadata
    """
    print("=" * 60)
    print("GRU-D Export for Procedures")
    print("=" * 60)

    # Load data
    print("\n1. Loading PE features...")
    df = load_pe_features()
    print(f"   Records: {len(df):,}")

    # Get patient list
    if len(df) > 0:
        patients = sorted(df['empi'].unique())
    else:
        patients = []
    print(f"   Patients: {len(patients)}")

    # Create tensors
    print(f"\n2. Creating hourly tensors ({n_hours} hours)...")
    values, mask, delta, feature_names = create_hourly_tensor(
        df, patients, n_hours
    )
    print(f"   Tensor shape: {values.shape}")
    print(f"   Features: {len(feature_names)}")

    # Calculate observation density
    if mask.size > 0:
        obs_rate = mask.mean() * 100
        print(f"   Observation rate: {obs_rate:.2f}%")
    else:
        obs_rate = 0.0

    # Save to HDF5
    if output_path is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "grud_procedures.h5"

    print(f"\n3. Saving to: {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('procedure_values', data=values, compression='gzip')
        f.create_dataset('procedure_mask', data=mask, compression='gzip')
        f.create_dataset('procedure_delta', data=delta, compression='gzip')

        # Save metadata
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('patient_ids', data=np.array(patients, dtype=object), dtype=dt)
        f.create_dataset('feature_names', data=np.array(feature_names, dtype=object), dtype=dt)

        f.attrs['n_patients'] = len(patients)
        f.attrs['n_hours'] = n_hours
        f.attrs['n_features'] = len(feature_names)
        f.attrs['observation_rate'] = obs_rate

    if output_path.exists():
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("GRU-D Export Complete!")
    print("=" * 60)

    return {
        'shape': values.shape,
        'patients': len(patients),
        'features': feature_names,
        'observation_rate': obs_rate,
    }


if __name__ == "__main__":
    export_grud()
