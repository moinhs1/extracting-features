# /home/moin/TDA_11_25/module_04_medications/exporters/grud_exporter.py
"""
GRU-D Exporter
==============

Export medication features as HDF5 tensors for GRU-D neural network.

Output format:
- medication_values: (n_patients, n_hours, n_features)
- medication_mask: (n_patients, n_hours, n_features) - 1 where observed
- medication_delta: (n_patients, n_hours, n_features) - hours since last observation
"""

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import GOLD_DIR, SILVER_DIR, EXPORTS_DIR, EXPORT_CONFIG


def load_mapped_medications() -> pd.DataFrame:
    """Load mapped medications from silver layer."""
    path = SILVER_DIR / "mapped_medications.parquet"
    return pd.read_parquet(path)


def load_class_indicators() -> pd.DataFrame:
    """Load therapeutic class indicators."""
    path = GOLD_DIR / "therapeutic_classes" / "class_indicators.parquet"
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
        df: Mapped medications DataFrame
        patients: Ordered list of patient IDs
        n_hours: Number of hours (default 168 = 7 days)
        feature_cols: List of feature column names

    Returns:
        Tuple of (values, mask, delta, feature_names)
    """
    # Get key therapeutic classes as features
    if feature_cols is None:
        feature_cols = [
            'ac_ufh_ther', 'ac_ufh_proph',
            'ac_lmwh_ther', 'ac_lmwh_proph',
            'ac_xa_inhibitor', 'ac_vka', 'ac_dti',
            'cv_vasopressor_any', 'cv_inotrope_any',
            'ps_opioid', 'ps_benzodiazepine',
            'ai_steroid_systemic',
        ]

    n_patients = len(patients)
    n_features = len(feature_cols)

    # Initialize tensors
    values = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)
    mask = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)
    delta = np.zeros((n_patients, n_hours, n_features), dtype=np.float32)

    # Create patient index mapping
    patient_to_idx = {p: i for i, p in enumerate(patients)}

    # Process each medication record
    for _, row in df.iterrows():
        empi = row['empi']
        if empi not in patient_to_idx:
            continue

        patient_idx = patient_to_idx[empi]
        hour = int(row['hours_from_t0'])

        if hour < 0 or hour >= n_hours:
            continue

        # Determine which features are active based on ingredient
        ingredient = str(row.get('ingredient_name', '')).lower()

        # Map ingredient to feature indices
        feature_map = {
            'heparin': ['ac_ufh_ther', 'ac_ufh_proph'],
            'enoxaparin': ['ac_lmwh_ther', 'ac_lmwh_proph'],
            'dalteparin': ['ac_lmwh_ther', 'ac_lmwh_proph'],
            'warfarin': ['ac_vka'],
            'apixaban': ['ac_xa_inhibitor'],
            'rivaroxaban': ['ac_xa_inhibitor'],
            'norepinephrine': ['cv_vasopressor_any'],
            'epinephrine': ['cv_vasopressor_any'],
            'morphine': ['ps_opioid'],
            'fentanyl': ['ps_opioid'],
        }

        active_features = feature_map.get(ingredient, [])

        for feat in active_features:
            if feat in feature_cols:
                feat_idx = feature_cols.index(feat)
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
        Dictionary with tensor shapes
    """
    print("=" * 60)
    print("GRU-D Export")
    print("=" * 60)

    # Load data
    print("\n1. Loading silver layer data...")
    df = load_mapped_medications()
    print(f"   Records: {len(df):,}")

    # Get patient list
    patients = sorted(df['empi'].unique())
    print(f"   Patients: {len(patients)}")

    # Create tensors
    print(f"\n2. Creating hourly tensors ({n_hours} hours)...")
    values, mask, delta, feature_names = create_hourly_tensor(
        df, patients, n_hours
    )
    print(f"   Tensor shape: {values.shape}")
    print(f"   Features: {len(feature_names)}")

    # Observation density
    obs_rate = mask.mean() * 100
    print(f"   Observation rate: {obs_rate:.2f}%")

    # Save to HDF5
    if output_path is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = EXPORTS_DIR / "grud_medications.h5"

    print(f"\n3. Saving to: {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('medication_values', data=values, compression='gzip')
        f.create_dataset('medication_mask', data=mask, compression='gzip')
        f.create_dataset('medication_delta', data=delta, compression='gzip')

        # Save metadata
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('patient_ids', data=np.array(patients, dtype=object), dtype=dt)
        f.create_dataset('feature_names', data=np.array(feature_names, dtype=object), dtype=dt)

        f.attrs['n_patients'] = len(patients)
        f.attrs['n_hours'] = n_hours
        f.attrs['n_features'] = len(feature_names)

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
