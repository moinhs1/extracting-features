"""Tests for world model builder (Layer 5)."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDiscretionWeights:
    """Test discretion weight loading and lookup."""

    def test_load_discretion_weights(self):
        """Load discretion weights from YAML configuration."""
        from transformers.world_model_builder import load_discretion_weights

        weights = load_discretion_weights()

        assert weights is not None
        assert isinstance(weights, dict)
        assert 'high' in weights or 'discretion_levels' in weights

    def test_get_discretion_weight_high(self):
        """High discretion procedures have weight 1.0."""
        from transformers.world_model_builder import get_discretion_weight

        # CDT is high discretion
        weight = get_discretion_weight('37211')

        assert weight == 1.0

    def test_get_discretion_weight_moderate(self):
        """Moderate discretion procedures have weight 0.6-0.8."""
        from transformers.world_model_builder import get_discretion_weight

        # Intubation is moderate discretion
        weight = get_discretion_weight('31500')

        assert 0.6 <= weight <= 0.8

    def test_get_discretion_weight_none(self):
        """No discretion procedures have weight 0.0."""
        from transformers.world_model_builder import get_discretion_weight

        # CPR is no discretion
        weight = get_discretion_weight('92950')

        assert weight == 0.0

    def test_get_discretion_weight_unknown(self):
        """Unknown procedures default to 0.5."""
        from transformers.world_model_builder import get_discretion_weight

        # Random unknown code
        weight = get_discretion_weight('99999')

        assert weight == 0.5  # Default moderate


class TestActionVectors:
    """Test action vector computation."""

    def test_compute_action_vector_empty(self):
        """Empty procedures at timestep returns zero action vector."""
        from transformers.world_model_builder import compute_action_vector

        procedures_df = pd.DataFrame({
            'empi': [],
            'hours_from_pe': [],
            'code': [],
        })

        action_vec = compute_action_vector(procedures_df, hour=12)

        assert action_vec is not None
        assert action_vec['thrombolysis_action'] == 0
        assert action_vec['cdt_action'] == 0
        assert action_vec['ivc_filter_action'] == 0

    def test_compute_action_vector_cdt(self):
        """CDT procedure creates action vector entry."""
        from transformers.world_model_builder import compute_action_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [12.0],
            'code': ['37211'],  # CDT
            'procedure_name': ['Catheter-directed thrombolysis'],
        })

        action_vec = compute_action_vector(procedures_df, hour=12)

        assert action_vec['cdt_action'] == 1
        assert action_vec['num_therapeutic_procedures'] >= 1

    def test_compute_action_vector_intubation_weighted(self):
        """Intubation action is weighted by discretion."""
        from transformers.world_model_builder import compute_action_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [6.0],
            'code': ['31500'],  # Intubation
            'procedure_name': ['Intubation'],
        })

        action_vec = compute_action_vector(procedures_df, hour=6)

        # Intubation has discretion weight ~0.7
        assert 0.6 <= action_vec['intubation_action'] <= 0.8

    def test_compute_action_vector_cpr_excluded(self):
        """CPR does not contribute to action vector (state only)."""
        from transformers.world_model_builder import compute_action_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [3.0],
            'code': ['92950'],  # CPR
            'procedure_name': ['CPR'],
        })

        action_vec = compute_action_vector(procedures_df, hour=3)

        # CPR should not create action (weight = 0)
        # But cardiac_arrest state should be updated elsewhere
        assert 'cpr_action' not in action_vec or action_vec.get('cpr_action', 0) == 0


class TestStateVectors:
    """Test state vector computation."""

    def test_compute_state_vector_baseline(self):
        """Baseline state vector with no procedures."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': [],
            'hours_from_pe': [],
            'code': [],
        })

        state_vec = compute_state_vector(procedures_df, hour=12)

        assert state_vec is not None
        assert state_vec['on_mechanical_ventilation'] == 0
        assert state_vec['on_ecmo'] == 0
        assert state_vec['support_level'] == 0

    def test_compute_state_vector_intubation(self):
        """Intubation sets on_mechanical_ventilation state."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [6.0, 12.0],
            'code': ['31500', '94002'],  # Intubation + vent management
            'procedure_name': ['Intubation', 'Ventilator management'],
        })

        # At hour 12, patient should be on vent
        state_vec = compute_state_vector(procedures_df, hour=12)

        assert state_vec['on_mechanical_ventilation'] == 1
        assert state_vec['support_level'] >= 3  # Vent = level 3

    def test_compute_state_vector_ecmo(self):
        """ECMO sets on_ecmo state and support_level=5."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [24.0],
            'code': ['33946'],  # ECMO
            'procedure_name': ['ECMO cannulation'],
        })

        state_vec = compute_state_vector(procedures_df, hour=24)

        assert state_vec['on_ecmo'] == 1
        assert state_vec['support_level'] == 5  # ECMO is max support

    def test_compute_state_vector_cardiac_arrest_irreversible(self):
        """Cardiac arrest (CPR) is irreversible state marker."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [3.0],
            'code': ['92950'],  # CPR
            'procedure_name': ['CPR'],
        })

        # Check state after arrest
        state_vec = compute_state_vector(procedures_df, hour=6)

        assert state_vec['cardiac_arrest_occurred'] == 1

    def test_compute_state_vector_transfusion_cumulative(self):
        """Transfusions accumulate in cumulative_rbc_units."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'hours_from_pe': [12.0, 13.0, 14.0],
            'code': ['36430', '36430', '36430'],
            'procedure_name': ['Transfusion RBC', 'Transfusion RBC', 'Transfusion RBC'],
            'quantity': [1, 1, 1],
        })

        state_vec = compute_state_vector(procedures_df, hour=15)

        # Should have cumulative count
        assert state_vec['cumulative_rbc_units'] >= 3

    def test_compute_state_vector_central_access(self):
        """Central line sets has_central_access state."""
        from transformers.world_model_builder import compute_state_vector

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [2.0],
            'code': ['36555'],  # Central line
            'procedure_name': ['Central line placement'],
        })

        state_vec = compute_state_vector(procedures_df, hour=6)

        assert state_vec['has_central_access'] == 1


class TestStaticState:
    """Test static procedure state computation."""

    def test_build_static_state_empty(self):
        """Empty procedures returns baseline static state."""
        from transformers.world_model_builder import build_static_state

        procedures_df = pd.DataFrame({
            'empi': [],
            'hours_from_pe': [],
            'code': [],
            'is_lifetime_history': [],
            'is_provoking_window': [],
        })

        static_state = build_static_state(procedures_df, empi='100')

        assert static_state is not None
        assert static_state['prior_vte_procedures'] == 0
        assert static_state['prior_ivc_filter'] == 0
        assert static_state['provoked_pe'] == 0

    def test_build_static_state_prior_ivc_filter(self):
        """Prior IVC filter in lifetime history."""
        from transformers.world_model_builder import build_static_state

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [-1000.0],  # Way in past
            'code': ['37191'],  # IVC filter
            'procedure_name': ['IVC filter placement'],
            'is_lifetime_history': [True],
            'is_provoking_window': [False],
        })

        static_state = build_static_state(procedures_df, empi='100')

        assert static_state['prior_ivc_filter'] == 1
        assert static_state['prior_vte_procedures'] >= 1

    def test_build_static_state_provoking_surgery(self):
        """Surgery in provoking window marks provoked_pe."""
        from transformers.world_model_builder import build_static_state

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [-120.0],  # Within -720 to 0
            'code': ['27447'],  # Knee replacement
            'procedure_name': ['Total knee replacement'],
            'is_lifetime_history': [False],
            'is_provoking_window': [True],
            'ccs_category': ['153'],  # High VTE risk
        })

        static_state = build_static_state(procedures_df, empi='100')

        assert static_state['provoked_pe'] == 1
        # Should have days from provoking surgery
        assert 'days_from_provoking_surgery' in static_state

    def test_build_static_state_surgical_risk_score(self):
        """Lifetime surgical risk score computed."""
        from transformers.world_model_builder import build_static_state

        procedures_df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [-2000.0, -1500.0],
            'code': ['44140', '27447'],  # Colon resection, knee replacement
            'procedure_name': ['Colectomy', 'Knee replacement'],
            'is_lifetime_history': [True, True],
            'is_provoking_window': [False, False],
            'ccs_category': ['78', '153'],
        })

        static_state = build_static_state(procedures_df, empi='100')

        assert static_state['lifetime_surgical_risk_score'] > 0
        assert static_state['lifetime_surgical_risk_score'] <= 1.0


class TestMainPipeline:
    """Test main world model feature builder."""

    def test_build_world_model_features_minimal(self):
        """Build world model features with minimal test data."""
        from transformers.world_model_builder import build_world_model_features

        # Create minimal test data
        procedures_df = pd.DataFrame({
            'empi': ['100', '100', '200'],
            'hours_from_pe': [6.0, 24.0, 12.0],
            'code': ['31500', '37211', '93306'],
            'procedure_name': ['Intubation', 'CDT', 'Echo'],
            'is_lifetime_history': [False, False, False],
            'is_provoking_window': [False, False, False],
            'is_diagnostic_workup': [False, False, True],
            'ccs_category': ['216', '52', '47'],
            'quantity': [1, 1, 1],
        })

        result = build_world_model_features(
            procedures_df,
            max_hours=168,  # 7 days
            output_dir=None,  # Don't save
        )

        assert result is not None
        assert 'static_state' in result
        assert 'dynamic_state' in result
        assert 'action_vectors' in result

    def test_build_world_model_features_static_shape(self):
        """Static state has one row per patient."""
        from transformers.world_model_builder import build_world_model_features

        procedures_df = pd.DataFrame({
            'empi': ['100', '100', '200', '200'],
            'hours_from_pe': [6.0, 24.0, 12.0, 36.0],
            'code': ['31500', '37211', '93306', '36430'],
            'procedure_name': ['Intubation', 'CDT', 'Echo', 'Transfusion'],
            'is_lifetime_history': [False, False, False, False],
            'is_provoking_window': [False, False, False, False],
            'is_diagnostic_workup': [False, False, True, False],
            'ccs_category': ['216', '52', '47', '222'],
            'quantity': [1, 1, 1, 1],
        })

        result = build_world_model_features(procedures_df, output_dir=None)

        static_df = result['static_state']
        assert len(static_df) == 2  # Two patients
        assert set(static_df['empi']) == {'100', '200'}

    def test_build_world_model_features_dynamic_shape(self):
        """Dynamic state has one row per patient-hour."""
        from transformers.world_model_builder import build_world_model_features

        procedures_df = pd.DataFrame({
            'empi': ['100', '100'],
            'hours_from_pe': [6.0, 24.0],
            'code': ['31500', '37211'],
            'procedure_name': ['Intubation', 'CDT'],
            'is_lifetime_history': [False, False],
            'is_provoking_window': [False, False],
            'is_diagnostic_workup': [False, False],
            'ccs_category': ['216', '52'],
            'quantity': [1, 1],
        })

        result = build_world_model_features(
            procedures_df,
            max_hours=48,
            output_dir=None
        )

        dynamic_df = result['dynamic_state']
        # Should have patient 100 at multiple hours
        patient_100 = dynamic_df[dynamic_df['empi'] == '100']
        assert len(patient_100) > 0
        # Hours should be sequential
        assert 'hour' in dynamic_df.columns

    def test_build_world_model_features_saves_files(self, tmp_path):
        """Pipeline saves three parquet files."""
        from transformers.world_model_builder import build_world_model_features

        procedures_df = pd.DataFrame({
            'empi': ['100'],
            'hours_from_pe': [6.0],
            'code': ['31500'],
            'procedure_name': ['Intubation'],
            'is_lifetime_history': [False],
            'is_provoking_window': [False],
            'is_diagnostic_workup': [False],
            'ccs_category': ['216'],
            'quantity': [1],
        })

        output_dir = tmp_path / "world_model_test"

        result = build_world_model_features(procedures_df, output_dir=output_dir)

        # Check files exist
        assert (output_dir / "static_state.parquet").exists()
        assert (output_dir / "dynamic_state.parquet").exists()
        assert (output_dir / "action_vectors.parquet").exists()


class TestSupportLevel:
    """Test support level ordinal scale computation."""

    def test_support_level_baseline(self):
        """No support = level 0."""
        from transformers.world_model_builder import compute_support_level

        state = {
            'on_mechanical_ventilation': 0,
            'on_ecmo': 0,
        }

        level = compute_support_level(state)
        assert level == 0

    def test_support_level_mechanical_ventilation(self):
        """Mechanical ventilation = level 3."""
        from transformers.world_model_builder import compute_support_level

        state = {
            'on_mechanical_ventilation': 1,
            'on_ecmo': 0,
        }

        level = compute_support_level(state)
        assert level == 3

    def test_support_level_ecmo(self):
        """ECMO = level 5 (max support)."""
        from transformers.world_model_builder import compute_support_level

        state = {
            'on_mechanical_ventilation': 1,
            'on_ecmo': 1,
        }

        level = compute_support_level(state)
        assert level == 5


class TestProcedureIntensity:
    """Test procedure intensity score computation."""

    def test_procedure_intensity_score_none(self):
        """No procedures = intensity 0."""
        from transformers.world_model_builder import compute_procedure_intensity_score

        procedures = []
        score = compute_procedure_intensity_score(procedures)

        assert score == 0.0

    def test_procedure_intensity_score_single_procedure(self):
        """Single procedure has measurable intensity."""
        from transformers.world_model_builder import compute_procedure_intensity_score

        procedures = [
            {'code': '31500', 'invasiveness': 3},  # Intubation
        ]

        score = compute_procedure_intensity_score(procedures)
        assert score > 0.0
        assert score <= 1.0

    def test_procedure_intensity_score_multiple_procedures(self):
        """Multiple procedures increase intensity."""
        from transformers.world_model_builder import compute_procedure_intensity_score

        procedures = [
            {'code': '31500', 'invasiveness': 3},  # Intubation
            {'code': '33946', 'invasiveness': 3},  # ECMO
            {'code': '36555', 'invasiveness': 2},  # Central line
        ]

        score = compute_procedure_intensity_score(procedures)
        assert score > 0.5  # High intensity
        assert score <= 1.0
