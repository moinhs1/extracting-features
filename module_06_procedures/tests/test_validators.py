"""Tests for layer validators."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestValidationResultClass:
    """Test ValidationResult container class."""

    def test_create_validation_result(self):
        """Create a ValidationResult instance."""
        from validation.layer_validators import ValidationResult

        result = ValidationResult("Test Layer")
        assert result.name == "Test Layer"
        assert result.passed == 0
        assert result.failed == 0

    def test_add_passing_check(self):
        """Add a passing check."""
        from validation.layer_validators import ValidationResult

        result = ValidationResult("Test")
        result.add_check("Sample check", True, "Details here")

        assert result.passed == 1
        assert result.failed == 0

    def test_add_failing_check(self):
        """Add a failing check."""
        from validation.layer_validators import ValidationResult

        result = ValidationResult("Test")
        result.add_check("Sample check", False, "Error details")

        assert result.passed == 0
        assert result.failed == 1

    def test_summary_all_pass(self):
        """Summary shows PASS when no failures."""
        from validation.layer_validators import ValidationResult

        result = ValidationResult("Test")
        result.add_check("Check 1", True)
        result.add_check("Check 2", True)

        summary = result.summary()
        assert "PASS" in summary

    def test_summary_any_fail(self):
        """Summary shows FAIL when any check fails."""
        from validation.layer_validators import ValidationResult

        result = ValidationResult("Test")
        result.add_check("Check 1", True)
        result.add_check("Check 2", False)

        summary = result.summary()
        assert "FAIL" in summary


class TestLayer1Validation:
    """Test Layer 1 (Bronze) validation."""

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_layer1_file_exists(self, mock_read):
        """Layer 1 validates bronze file exists."""
        from validation.layer_validators import validate_layer1

        # Mock data
        mock_df = pd.DataFrame({
            'empi': ['100', '200', '100'],
            'code': ['71275', '93306', '31500'],
            'parsed_dose_value': [1.0, 1.0, None],
            'parse_method': ['regex', 'regex', None],
            'hours_from_pe': [-12.0, 0.0, 24.0],
        })
        mock_read.return_value = mock_df

        result = validate_layer1()

        assert result.name == "Layer 1: Canonical Extraction"
        # At least one check should pass
        assert result.passed > 0

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_layer1_checks_patient_count(self, mock_read):
        """Layer 1 checks patient count."""
        from validation.layer_validators import validate_layer1

        # Create mock data with multiple patients
        mock_df = pd.DataFrame({
            'empi': [f'P{i%10}' for i in range(100)],
            'code': ['71275'] * 100,
            'code_type': ['CPT'] * 100,
            'hours_from_pe': list(range(100)),
        })
        mock_read.return_value = mock_df

        result = validate_layer1()

        # Should have a check for patient count
        checks = [c['description'] for c in result.checks]
        assert any('patient' in c.lower() for c in checks)


class TestSilverValidation:
    """Test Silver (Mapping) validation."""

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_silver_mapping_rate(self, mock_read):
        """Silver validates CCS mapping rate."""
        from validation.layer_validators import validate_silver

        # High mapping rate
        mock_df = pd.DataFrame({
            'empi': ['100'] * 100,
            'code': [f'CODE{i}' for i in range(100)],
            'ccs_category': ['61'] * 95 + [None] * 5,
            'mapping_method': ['direct'] * 95 + [None] * 5,
        })
        mock_read.return_value = mock_df

        result = validate_silver()

        assert result.name == "Silver: Code Mapping"
        # Should check mapping rate
        checks = [c['description'] for c in result.checks]
        assert any('mapping' in c.lower() for c in checks)


class TestLayer2Validation:
    """Test Layer 2 (CCS Indicators) validation."""

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_layer2_ccs_categories(self, mock_read):
        """Layer 2 validates CCS categories exist."""
        from validation.layer_validators import validate_layer2

        # Mock CCS indicators
        mock_df = pd.DataFrame({
            'empi': ['100', '200'],
            'temporal_category': ['diagnostic_workup', 'diagnostic_workup'],
            'ccs_61': [True, False],
            'ccs_47': [False, True],
            'ccs_216': [True, True],
        })
        mock_read.return_value = mock_df

        result = validate_layer2()

        assert result.name == "Layer 2: CCS Indicators"


class TestLayer3Validation:
    """Test Layer 3 (PE Features) validation."""

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_layer3_pe_features(self, mock_read):
        """Layer 3 validates PE feature rates."""
        from validation.layer_validators import validate_layer3

        # Mock PE features
        n_patients = 1000
        mock_df = pd.DataFrame({
            'empi': [f'P{i}' for i in range(n_patients)],
            'cta_performed': [True] * 850 + [False] * 150,  # 85%
            'echo_performed': [True] * 600 + [False] * 400,  # 60%
            'intubation_performed': [True] * 100 + [False] * 900,  # 10%
            'ivc_filter_placed': [True] * 100 + [False] * 900,  # 10%
            'ecmo_initiated': [True] * 10 + [False] * 990,  # 1%
        })
        mock_read.return_value = mock_df

        result = validate_layer3()

        assert result.name == "Layer 3: PE Features"
        # Should check procedure rates
        checks = [c['description'] for c in result.checks]
        assert any('cta' in c.lower() for c in checks)


class TestCrossLayerValidation:
    """Test cross-layer consistency checks."""

    @patch('validation.layer_validators.pd.read_parquet')
    def test_validate_cross_layer_patient_consistency(self, mock_read):
        """Cross-layer validates patient consistency."""
        from validation.layer_validators import validate_cross_layer

        # Mock all layers with same patients
        def mock_read_side_effect(path):
            patients = [f'P{i}' for i in range(100)]
            if 'bronze' in str(path):
                return pd.DataFrame({'empi': patients * 10})
            elif 'silver' in str(path):
                return pd.DataFrame({'empi': patients * 10})
            elif 'ccs_indicators' in str(path):
                return pd.DataFrame({'empi': patients})
            else:
                return pd.DataFrame({'empi': patients})

        mock_read.side_effect = mock_read_side_effect

        result = validate_cross_layer()

        assert result.name == "Cross-Layer Consistency"


class TestFullValidationSuite:
    """Test running full validation suite."""

    @patch('validation.layer_validators.validate_layer1')
    @patch('validation.layer_validators.validate_silver')
    @patch('validation.layer_validators.validate_layer2')
    @patch('validation.layer_validators.validate_layer3')
    @patch('validation.layer_validators.validate_cross_layer')
    def test_run_all_validations(self, mock_cross, mock_l3, mock_l2,
                                  mock_silver, mock_l1):
        """Run full validation suite."""
        from validation.layer_validators import (
            run_full_validation, ValidationResult
        )

        # Mock all validators to return results
        for mock in [mock_l1, mock_silver, mock_l2, mock_l3, mock_cross]:
            result = ValidationResult("Mock")
            result.add_check("Mock check", True)
            mock.return_value = result

        results = run_full_validation()

        assert len(results) == 5
        assert all(r.passed > 0 for r in results)
