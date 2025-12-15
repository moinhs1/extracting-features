"""Tests for method-specific exporters."""

import pytest
import pandas as pd
import numpy as np
import h5py
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGBTMExporter:
    """Test GBTM/lcmm CSV export."""

    def test_gbtm_export_creates_csv(self, tmp_path):
        """GBTM export creates CSV file."""
        from exporters.gbtm_exporter import export_gbtm

        # Mock data loading
        with patch('exporters.gbtm_exporter.load_pe_features') as mock_load:
            mock_df = pd.DataFrame({
                'empi': ['100', '100', '200'],
                'day_from_pe': [0, 1, 0],
                'cta_chest_performed': [1, 0, 1],
                'intubation_performed': [0, 1, 0],
            })
            mock_load.return_value = mock_df

            result = export_gbtm(output_dir=tmp_path)

            # Check output file exists
            output_file = tmp_path / "gbtm_procedures.csv"
            assert output_file.exists()

    def test_gbtm_export_has_required_columns(self, tmp_path):
        """GBTM export has empi, day_from_pe columns."""
        from exporters.gbtm_exporter import export_gbtm

        with patch('exporters.gbtm_exporter.load_pe_features') as mock_load:
            mock_df = pd.DataFrame({
                'empi': ['100'],
                'day_from_pe': [0],
                'cta_chest_performed': [1],
            })
            mock_load.return_value = mock_df

            result = export_gbtm(output_dir=tmp_path)

            # Read output
            output_file = tmp_path / "gbtm_procedures.csv"
            df = pd.read_csv(output_file)

            assert 'empi' in df.columns
            assert 'day_from_pe' in df.columns


class TestGRUDExporter:
    """Test GRU-D HDF5 export."""

    def test_grud_export_creates_h5(self, tmp_path):
        """GRU-D export creates HDF5 file."""
        from exporters.grud_exporter import export_grud

        with patch('exporters.grud_exporter.load_pe_features') as mock_load:
            mock_df = pd.DataFrame({
                'empi': ['100', '100'],
                'hours_from_pe': [0, 12],
                'intubation_performed': [0, 1],
                'ecmo_initiated': [0, 0],
            })
            mock_load.return_value = mock_df

            result = export_grud(output_path=tmp_path / "test.h5")

            output_file = tmp_path / "test.h5"
            assert output_file.exists()

    def test_grud_export_has_required_datasets(self, tmp_path):
        """GRU-D HDF5 has values, mask, delta datasets."""
        from exporters.grud_exporter import export_grud

        with patch('exporters.grud_exporter.load_pe_features') as mock_load:
            mock_df = pd.DataFrame({
                'empi': ['100'],
                'hours_from_pe': [0],
                'intubation_performed': [1],
            })
            mock_load.return_value = mock_df

            result = export_grud(output_path=tmp_path / "test.h5", n_hours=24)

            output_file = tmp_path / "test.h5"
            with h5py.File(output_file, 'r') as f:
                assert 'procedure_values' in f
                assert 'procedure_mask' in f
                assert 'procedure_delta' in f

    def test_grud_tensor_shape_correct(self, tmp_path):
        """GRU-D tensor has correct shape (n_patients, n_hours, n_features)."""
        from exporters.grud_exporter import export_grud

        with patch('exporters.grud_exporter.load_pe_features') as mock_load:
            mock_df = pd.DataFrame({
                'empi': ['100', '200'],
                'hours_from_pe': [0, 0],
                'intubation_performed': [1, 0],
            })
            mock_load.return_value = mock_df

            result = export_grud(output_path=tmp_path / "test.h5", n_hours=24)

            output_file = tmp_path / "test.h5"
            with h5py.File(output_file, 'r') as f:
                values = f['procedure_values'][:]
                assert values.ndim == 3
                assert values.shape[0] == 2  # 2 patients
                assert values.shape[1] == 24  # 24 hours
                assert values.shape[2] > 0  # At least some features


class TestXGBoostExporter:
    """Test XGBoost parquet export."""

    def test_xgboost_export_creates_parquet(self, tmp_path):
        """XGBoost export creates parquet file."""
        from exporters.xgboost_exporter import export_xgboost

        with patch('exporters.xgboost_exporter.load_all_gold_layers') as mock_load:
            mock_ccs = pd.DataFrame({
                'empi': ['100'],
                'temporal_category': ['diagnostic_workup'],
                'ccs_216': [1],
                'ccs_47': [0],
            })
            mock_pe = pd.DataFrame({
                'empi': ['100'],
                'cta_chest_performed': [1],
                'intubation_performed': [0],
            })
            mock_load.return_value = (mock_ccs, mock_pe)

            result = export_xgboost(output_path=tmp_path / "test.parquet")

            output_file = tmp_path / "test.parquet"
            assert output_file.exists()

    def test_xgboost_export_one_row_per_patient(self, tmp_path):
        """XGBoost export has one row per patient."""
        from exporters.xgboost_exporter import export_xgboost

        with patch('exporters.xgboost_exporter.load_all_gold_layers') as mock_load:
            mock_ccs = pd.DataFrame({
                'empi': ['100', '100', '200'],
                'temporal_category': ['diagnostic_workup', 'initial_treatment', 'diagnostic_workup'],
                'ccs_216': [1, 0, 1],
            })
            mock_pe = pd.DataFrame({
                'empi': ['100', '200'],
                'cta_chest_performed': [1, 1],
            })
            mock_load.return_value = (mock_ccs, mock_pe)

            result = export_xgboost(output_path=tmp_path / "test.parquet")

            output_file = tmp_path / "test.parquet"
            df = pd.read_parquet(output_file)

            assert len(df) == 2  # One row per patient
            assert set(df['empi']) == {'100', '200'}

    def test_xgboost_export_has_ccs_indicators(self, tmp_path):
        """XGBoost export includes CCS category indicators."""
        from exporters.xgboost_exporter import export_xgboost

        with patch('exporters.xgboost_exporter.load_all_gold_layers') as mock_load:
            mock_ccs = pd.DataFrame({
                'empi': ['100'],
                'temporal_category': ['diagnostic_workup'],
                'ccs_216': [1],
                'ccs_47': [1],
            })
            mock_pe = pd.DataFrame({
                'empi': ['100'],
                'cta_chest_performed': [1],
            })
            mock_load.return_value = (mock_ccs, mock_pe)

            result = export_xgboost(output_path=tmp_path / "test.parquet")

            output_file = tmp_path / "test.parquet"
            df = pd.read_parquet(output_file)

            # Should have CCS columns
            ccs_cols = [c for c in df.columns if c.startswith('ccs_')]
            assert len(ccs_cols) > 0


class TestExporterIntegration:
    """Test integration between exporters."""

    def test_all_exporters_use_same_patient_list(self, tmp_path):
        """All exporters should use same patient list."""
        # This is a design requirement - all exports should be for same cohort
        # Test would verify consistency across GBTM, GRU-D, XGBoost
        pass
