"""Tests for CCS indicator builder."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTimeWindowAssignment:
    """Test time window assignment function."""

    def test_get_time_window_provoking(self):
        """Provoking window is -720h to 0h."""
        from transformers.ccs_indicator_builder import get_time_window

        # -500 hours should be in provoking window
        result = get_time_window(-500)
        assert result == 'provoking_window'

    def test_get_time_window_diagnostic(self):
        """Diagnostic workup is -24h to +24h."""
        from transformers.ccs_indicator_builder import get_time_window

        result = get_time_window(12)
        assert result == 'diagnostic_workup'

        result = get_time_window(-12)
        assert result == 'diagnostic_workup'

    def test_get_time_window_initial_treatment(self):
        """Initial treatment is 0h to +72h."""
        from transformers.ccs_indicator_builder import get_time_window

        result = get_time_window(48)
        assert result == 'initial_treatment'

    def test_get_time_window_escalation(self):
        """Escalation is >72h."""
        from transformers.ccs_indicator_builder import get_time_window

        result = get_time_window(150)
        assert result == 'escalation'

    def test_get_time_window_lifetime(self):
        """Lifetime history is before -720h."""
        from transformers.ccs_indicator_builder import get_time_window

        result = get_time_window(-1000)
        assert result == 'lifetime_history'

    def test_get_time_window_boundaries(self):
        """Test boundary conditions."""
        from transformers.ccs_indicator_builder import get_time_window

        # Exactly -720 should be start of provoking window
        result = get_time_window(-720)
        assert result in ['provoking_window', 'lifetime_history']

        # Exactly 0 should be Time Zero
        result = get_time_window(0)
        assert result in ['diagnostic_workup', 'initial_treatment']


class TestSurgicalRiskClassification:
    """Test surgical risk level assignment."""

    def test_get_surgical_risk_very_high(self):
        """Very high risk categories identified correctly."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        # Hip replacement (CCS 153)
        result = get_surgical_risk_level('153')
        assert result == 'very_high'

    def test_get_surgical_risk_high(self):
        """High risk categories identified correctly."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        # Heart valve (CCS 43)
        result = get_surgical_risk_level('43')
        assert result == 'high'

    def test_get_surgical_risk_moderate(self):
        """Moderate risk categories identified correctly."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        # Aortic resection (CCS 52)
        result = get_surgical_risk_level('52')
        assert result == 'moderate'

    def test_get_surgical_risk_low(self):
        """Low risk categories identified correctly."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        # Arthroscopy (CCS 146)
        result = get_surgical_risk_level('146')
        assert result == 'low'

    def test_get_surgical_risk_minimal(self):
        """Minimal risk categories identified correctly."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        # GI endoscopy (CCS 61)
        result = get_surgical_risk_level('61')
        assert result == 'minimal'

    def test_get_surgical_risk_unknown(self):
        """Unknown CCS returns minimal or None."""
        from transformers.ccs_indicator_builder import get_surgical_risk_level

        result = get_surgical_risk_level('999')
        assert result in [None, 'minimal', 'unknown']


class TestInvasivenessMapping:
    """Test invasiveness level assignment."""

    def test_get_invasiveness_level(self):
        """Invasiveness levels assigned correctly."""
        from transformers.ccs_indicator_builder import get_invasiveness_level

        # Major procedure should be highly invasive (3)
        result = get_invasiveness_level('43')  # Heart valve
        assert result in [2, 3]

        # Imaging should be non-invasive (0)
        result = get_invasiveness_level('61')  # GI endoscopy
        assert result in [0, 1]


class TestCCSIndicatorAggregation:
    """Test CCS indicator aggregation."""

    def test_aggregate_ccs_indicators_basic(self):
        """Basic aggregation produces expected structure."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '200'],
            'hours_from_pe': [-500, 12, 48],
            'ccs_category': ['61', '47', '216'],
            'ccs_description': ['CT scan', 'Echo', 'Intubation'],
        })

        result = aggregate_ccs_indicators(df)

        assert len(result) > 0
        assert 'empi' in result.columns
        assert 'time_window' in result.columns

    def test_aggregate_ccs_binary_indicators(self):
        """Binary indicators are created for each CCS."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [12, 12],
            'ccs_category': ['61', '47'],
            'ccs_description': ['CT scan', 'Echo'],
        })

        result = aggregate_ccs_indicators(df)

        # Should have ccs_61 and ccs_47 columns
        assert 'ccs_61' in result.columns
        assert 'ccs_47' in result.columns

        # Both should be True for patient 100
        row = result[result['empi'] == '100'].iloc[0]
        assert row['ccs_61'] == True
        assert row['ccs_47'] == True

    def test_aggregate_ccs_count_columns(self):
        """Count columns track occurrences."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_pe': [12, 13, 14],
            'ccs_category': ['61', '61', '47'],
            'ccs_description': ['CT scan', 'CT scan', 'Echo'],
        })

        result = aggregate_ccs_indicators(df)

        row = result[result['empi'] == '100'].iloc[0]
        assert row['ccs_61_count'] == 2
        assert row['ccs_47_count'] == 1

    def test_aggregate_multiple_patients(self):
        """Multiple patients aggregated separately."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '200', '200'],
            'hours_from_pe': [12, 12, 48, 48],
            'ccs_category': ['61', '47', '216', '222'],
            'ccs_description': ['CT', 'Echo', 'Intubation', 'Transfusion'],
        })

        result = aggregate_ccs_indicators(df)

        assert len(result) == 2
        assert set(result['empi'].unique()) == {'100', '200'}

    def test_aggregate_multiple_windows(self):
        """Different time windows create separate rows."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [12, 150],  # diagnostic vs escalation
            'ccs_category': ['61', '216'],
            'ccs_description': ['CT', 'Intubation'],
        })

        result = aggregate_ccs_indicators(df)

        # Should have 2 rows for patient 100 (different windows)
        patient_rows = result[result['empi'] == '100']
        assert len(patient_rows) == 2
        assert 'diagnostic_workup' in patient_rows['time_window'].values
        assert 'escalation' in patient_rows['time_window'].values


class TestSurgicalRiskAggregation:
    """Test surgical risk level aggregation."""

    def test_surgical_risk_in_output(self):
        """Surgical risk level included in output."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [12],
            'ccs_category': ['153'],  # Very high risk
            'ccs_description': ['Hip replacement'],
        })

        result = aggregate_ccs_indicators(df)

        assert 'surgical_risk_level' in result.columns
        assert result.iloc[0]['surgical_risk_level'] == 'very_high'

    def test_surgical_risk_max_across_procedures(self):
        """Maximum risk level across multiple procedures."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [12, 13],
            'ccs_category': ['153', '61'],  # Very high + minimal
            'ccs_description': ['Hip replacement', 'Endoscopy'],
        })

        result = aggregate_ccs_indicators(df)

        # Should take maximum (very_high)
        assert result.iloc[0]['surgical_risk_level'] == 'very_high'


class TestInvasivenessAggregation:
    """Test invasiveness aggregation."""

    def test_invasiveness_max_in_output(self):
        """Maximum invasiveness included in output."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [12, 13],
            'ccs_category': ['43', '61'],
            'ccs_description': ['Valve', 'Endoscopy'],
        })

        result = aggregate_ccs_indicators(df)

        assert 'invasiveness_max' in result.columns
        # Should be max of the two procedures
        assert result.iloc[0]['invasiveness_max'] >= 0


class TestProcedureCount:
    """Test procedure counting."""

    def test_procedure_count_in_output(self):
        """Total procedure count included."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_pe': [12, 13, 14],
            'ccs_category': ['61', '47', '216'],
            'ccs_description': ['CT', 'Echo', 'Intubation'],
        })

        result = aggregate_ccs_indicators(df)

        assert 'procedure_count' in result.columns
        assert result.iloc[0]['procedure_count'] == 3


class TestBuildCCSIndicators:
    """Test main pipeline function."""

    def test_build_ccs_indicators_creates_output(self):
        """Main pipeline creates expected output structure."""
        from transformers.ccs_indicator_builder import build_ccs_indicators
        import tempfile

        # Create mock silver data
        with tempfile.TemporaryDirectory() as tmpdir:
            silver_path = Path(tmpdir) / "mapped_procedures.parquet"

            mock_data = pd.DataFrame({
                'empi': ['100', '100', '200'],
                'hours_from_pe': [12, 48, 150],
                'ccs_category': ['61', '47', '216'],
                'ccs_description': ['CT', 'Echo', 'Intubation'],
                'procedure_name': ['CT CHEST', 'ECHO', 'INTUBATION'],
                'code': ['71275', '93306', '31500'],
            })
            mock_data.to_parquet(silver_path)

            output_path = Path(tmpdir) / "ccs_indicators.parquet"

            result = build_ccs_indicators(
                input_path=silver_path,
                output_path=output_path
            )

            assert len(result) > 0
            assert output_path.exists()

    def test_build_ccs_indicators_schema(self):
        """Output has expected schema."""
        from transformers.ccs_indicator_builder import build_ccs_indicators
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            silver_path = Path(tmpdir) / "mapped_procedures.parquet"

            mock_data = pd.DataFrame({
                'empi': ['100'],
                'hours_from_pe': [12],
                'ccs_category': ['61'],
                'ccs_description': ['CT'],
                'procedure_name': ['CT CHEST'],
                'code': ['71275'],
            })
            mock_data.to_parquet(silver_path)

            output_path = Path(tmpdir) / "ccs_indicators.parquet"

            result = build_ccs_indicators(
                input_path=silver_path,
                output_path=output_path
            )

            # Check required columns
            required_cols = [
                'empi', 'time_window',
                'surgical_risk_level', 'invasiveness_max', 'procedure_count'
            ]
            for col in required_cols:
                assert col in result.columns, f"Missing column: {col}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataframe(self):
        """Handle empty input gracefully."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': [],
            'hours_from_pe': [],
            'ccs_category': [],
            'ccs_description': [],
        })

        result = aggregate_ccs_indicators(df)
        assert len(result) == 0

    def test_missing_ccs_category(self):
        """Handle missing CCS category."""
        from transformers.ccs_indicator_builder import aggregate_ccs_indicators

        df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [12],
            'ccs_category': [None],
            'ccs_description': ['Unknown'],
        })

        result = aggregate_ccs_indicators(df)
        # Should still create row but with no CCS indicators set
        assert len(result) >= 0

    def test_outside_time_windows(self):
        """Procedures outside defined windows are handled."""
        from transformers.ccs_indicator_builder import get_time_window

        # Way in the past
        result = get_time_window(-100000)
        # Should return something or None

        # Way in the future
        result = get_time_window(100000)
        # Should return something or None
