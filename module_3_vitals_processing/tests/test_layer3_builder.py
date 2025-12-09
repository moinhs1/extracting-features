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
            f.create_dataset('imputation_tier', data=np.ones((1, 745, 7), dtype=np.int8))
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
            'EMPI': ['E001'] * 70,  # 10 hours x 7 vitals
            'hour_from_pe': sorted(list(range(10)) * 7),
            'vital_type': ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP'] * 10,
            'mean': [72.0, 120.0, 80.0, 93.0, 16.0, 98.0, 37.0] * 10,
        })

        parquet_path = tmp_path / 'hourly_grid.parquet'
        grid.to_parquet(parquet_path)

        # Create HDF5
        hdf5_path = tmp_path / 'hourly_tensors.h5'
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('imputation_tier', data=np.ones((1, 745, 7), dtype=np.int8))
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
            f.create_dataset('imputation_tier', data=np.ones((1, 745, 7), dtype=np.int8))
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
            f.create_dataset('imputation_tier', data=np.ones((1, 745, 7), dtype=np.int8))
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
            f.create_dataset('imputation_tier', data=np.ones((1, 745, 7), dtype=np.int8))
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
