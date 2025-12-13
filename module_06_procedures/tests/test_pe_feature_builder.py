"""Tests for PE-specific procedure feature builder."""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLifetimeHistoryFeatures:
    """Test lifetime procedure history features."""

    def test_prior_ivc_filter_detection(self):
        """Detect prior IVC filter placement."""
        from transformers.pe_feature_builder import compute_lifetime_history_features

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'code': ['37191', '71275', '93306'],
            'hours_from_pe': [-1000, -500, 0],
            'is_lifetime_history': [True, False, False],
        })

        result = compute_lifetime_history_features(df)

        assert result['prior_ivc_filter_ever'] == True
        assert result['prior_vte_procedure_count'] >= 1

    def test_no_prior_procedures(self):
        """Patient with no lifetime history procedures."""
        from transformers.pe_feature_builder import compute_lifetime_history_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['71275'],
            'hours_from_pe': [0],
            'is_lifetime_history': [False],
        })

        result = compute_lifetime_history_features(df)

        assert result['prior_ivc_filter_ever'] == False
        assert result['prior_major_surgery_ever'] == False
        assert result['lifetime_surgical_count'] == 0

    def test_prior_thrombolysis_detection(self):
        """Detect prior catheter-directed therapy."""
        from transformers.pe_feature_builder import compute_lifetime_history_features

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'code': ['37211', '71275'],
            'hours_from_pe': [-2000, 0],
            'is_lifetime_history': [True, False],
        })

        result = compute_lifetime_history_features(df)

        assert result['prior_cdt_for_vte'] == True


class TestProvokingProcedures:
    """Test provoking procedure features (1-30 days pre-PE)."""

    def test_surgery_within_30_days(self):
        """Detect surgery in provoking window."""
        from transformers.pe_feature_builder import compute_provoking_features

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'code': ['27447', '71275', '93306'],  # 27447 = knee replacement
            'procedure_name': ['KNEE REPLACEMENT', 'CTA CHEST', 'ECHO'],
            'hours_from_pe': [-200, 0, 24],
            'is_provoking_window': [True, False, False],
            'ccs_category': ['154', '61', '47'],
        })

        result = compute_provoking_features(df)

        assert result['surgery_within_30_days'] == True
        assert result['orthopedic_surgery_within_30d'] == True
        assert result['provoked_pe'] == True
        assert result['surgery_vte_risk_category'] in ['high', 'very_high']

    def test_no_provoking_surgery(self):
        """No surgery in provoking window."""
        from transformers.pe_feature_builder import compute_provoking_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['71275'],
            'procedure_name': ['CTA CHEST'],
            'hours_from_pe': [0],
            'is_provoking_window': [False],
            'ccs_category': ['61'],
        })

        result = compute_provoking_features(df)

        assert result['surgery_within_30_days'] == False
        assert result['provoked_pe'] == False

    def test_cancer_surgery_provoking(self):
        """Detect cancer surgery as provoking factor."""
        from transformers.pe_feature_builder import compute_provoking_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['99999'],  # Mock cancer surgery code
            'procedure_name': ['CANCER RESECTION'],
            'hours_from_pe': [-100],
            'is_provoking_window': [True],
            'ccs_category': ['78'],  # Colorectal resection
        })

        result = compute_provoking_features(df)

        assert result['surgery_within_30_days'] == True
        assert result['provoked_pe'] == True


class TestDiagnosticWorkupFeatures:
    """Test diagnostic workup features (±24h of PE)."""

    def test_cta_performed_with_datetime(self):
        """CTA performed during diagnostic window with datetime."""
        from transformers.pe_feature_builder import compute_diagnostic_features

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'code': ['71275', '93306'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00', '2023-07-27 14:00']),
            'hours_from_pe': [0, 2],
            'is_diagnostic_workup': [True, True],
        })

        result = compute_diagnostic_features(df)

        assert result['cta_chest_performed'] == True
        assert pd.notna(result['cta_datetime'])
        assert result['cta_hours_from_pe'] == 0

    def test_echo_performed(self):
        """Echo performed during workup."""
        from transformers.pe_feature_builder import compute_diagnostic_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['93306'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00']),
            'hours_from_pe': [2],
            'is_diagnostic_workup': [True],
        })

        result = compute_diagnostic_features(df)

        assert result['echo_performed'] == True
        assert pd.notna(result['echo_datetime'])

    def test_le_duplex_performed(self):
        """Lower extremity duplex performed."""
        from transformers.pe_feature_builder import compute_diagnostic_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['93970'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00']),
            'hours_from_pe': [4],
            'is_diagnostic_workup': [True],
        })

        result = compute_diagnostic_features(df)

        assert result['le_duplex_performed'] == True

    def test_vq_scan_performed(self):
        """V/Q scan performed."""
        from transformers.pe_feature_builder import compute_diagnostic_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['78582'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00']),
            'hours_from_pe': [-2],
            'is_diagnostic_workup': [True],
        })

        result = compute_diagnostic_features(df)

        assert result['vq_scan_performed'] == True

    def test_diagnostic_workup_intensity(self):
        """Compute diagnostic workup intensity."""
        from transformers.pe_feature_builder import compute_diagnostic_features

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'code': ['71275', '93306', '93970'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00'] * 3),
            'hours_from_pe': [0, 2, 4],
            'is_diagnostic_workup': [True, True, True],
        })

        result = compute_diagnostic_features(df)

        assert result['diagnostic_workup_intensity'] == 3


class TestInitialTreatmentFeatures:
    """Test initial treatment features (0-72h post-PE)."""

    def test_catheter_directed_therapy(self):
        """CDT performed with datetime."""
        from transformers.pe_feature_builder import compute_initial_treatment_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['37211'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 18:00']),
            'hours_from_pe': [6],
            'is_initial_treatment': [True],
        })

        result = compute_initial_treatment_features(df)

        assert result['catheter_directed_therapy'] == True
        assert result['any_reperfusion_therapy'] == True
        assert pd.notna(result['cdt_datetime'])
        assert result['cdt_hours_from_pe'] == 6

    def test_ivc_filter_placement(self):
        """IVC filter placed during initial treatment."""
        from transformers.pe_feature_builder import compute_initial_treatment_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['37191'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 20:00']),
            'hours_from_pe': [8],
            'is_initial_treatment': [True],
        })

        result = compute_initial_treatment_features(df)

        assert result['ivc_filter_placed'] == True
        assert pd.notna(result['ivc_filter_datetime'])

    def test_intubation_performed(self):
        """Intubation during initial treatment."""
        from transformers.pe_feature_builder import compute_initial_treatment_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['31500'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 14:00']),
            'hours_from_pe': [2],
            'is_initial_treatment': [True],
        })

        result = compute_initial_treatment_features(df)

        assert result['intubation_performed'] == True
        assert pd.notna(result['intubation_datetime'])
        assert result['intubation_hours_from_pe'] == 2

    def test_ecmo_initiated(self):
        """ECMO initiated."""
        from transformers.pe_feature_builder import compute_initial_treatment_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['33946'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 16:00']),
            'hours_from_pe': [4],
            'is_initial_treatment': [True],
        })

        result = compute_initial_treatment_features(df)

        assert result['ecmo_initiated'] == True
        assert pd.notna(result['ecmo_datetime'])

    def test_central_line_placed(self):
        """Central line placement."""
        from transformers.pe_feature_builder import compute_initial_treatment_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['36555'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 13:00']),
            'hours_from_pe': [1],
            'is_initial_treatment': [True],
        })

        result = compute_initial_treatment_features(df)

        assert result['central_line_placed'] == True


class TestEscalationFeatures:
    """Test escalation/complication features (>72h post-PE)."""

    def test_delayed_intubation(self):
        """Intubation occurring >72h post-PE."""
        from transformers.pe_feature_builder import compute_escalation_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['31500'],
            'procedure_datetime': pd.to_datetime(['2023-07-31 12:00']),
            'hours_from_pe': [96],
            'is_escalation': [True],
        })

        result = compute_escalation_features(df)

        assert result['delayed_intubation'] == True

    def test_transfusion_events(self):
        """Transfusion with quantity tracking."""
        from transformers.pe_feature_builder import compute_escalation_features

        df = pd.DataFrame({
            'empi': ['100', '100', '100'],
            'code': ['36430', '36430', '36430'],
            'quantity': [2, 2, 3],
            'procedure_datetime': pd.to_datetime(['2023-07-28 12:00', '2023-07-29 12:00', '2023-07-30 12:00']),
            'hours_from_pe': [80, 104, 128],
            'is_escalation': [True, True, True],
        })

        result = compute_escalation_features(df)

        assert result['any_transfusion'] == True
        assert result['rbc_units_total'] == 7

    def test_cardiac_arrest_detection(self):
        """Cardiac arrest with CPR."""
        from transformers.pe_feature_builder import compute_escalation_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['92950'],
            'procedure_datetime': pd.to_datetime(['2023-07-28 12:00']),
            'hours_from_pe': [96],
            'is_escalation': [True],
        })

        result = compute_escalation_features(df)

        assert result['cardiac_arrest_post_pe'] == True

    def test_reintubation(self):
        """Multiple intubation events suggest reintubation."""
        from transformers.pe_feature_builder import compute_escalation_features

        df = pd.DataFrame({
            'empi': ['100', '100'],
            'code': ['31500', '31500'],
            'procedure_datetime': pd.to_datetime(['2023-07-28 12:00', '2023-07-30 12:00']),
            'hours_from_pe': [80, 150],
            'is_escalation': [True, True],
        })

        result = compute_escalation_features(df)

        assert result['reintubation'] == True

    def test_tracheostomy(self):
        """Tracheostomy for prolonged ventilation."""
        from transformers.pe_feature_builder import compute_escalation_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['31600'],
            'procedure_datetime': pd.to_datetime(['2023-08-03 12:00']),
            'hours_from_pe': [200],
            'is_escalation': [True],
        })

        result = compute_escalation_features(df)

        assert result['tracheostomy'] == True


class TestPatientLevelAggregation:
    """Test aggregation to patient level."""

    def test_build_patient_pe_features(self):
        """Build complete patient-level feature set."""
        from transformers.pe_feature_builder import build_patient_pe_features

        df = pd.DataFrame({
            'empi': ['100', '100', '100', '100', '100'],
            'code': ['37191', '71275', '93306', '37211', '31500'],
            'procedure_name': ['IVC FILTER', 'CTA CHEST', 'ECHO', 'CDT', 'INTUBATION'],
            'procedure_datetime': pd.to_datetime([
                '2022-01-01 12:00',  # Prior IVC filter
                '2023-07-27 12:00',  # CTA
                '2023-07-27 14:00',  # Echo
                '2023-07-27 18:00',  # CDT
                '2023-07-27 20:00',  # Intubation
            ]),
            'hours_from_pe': [-5000, 0, 2, 6, 8],
            'is_lifetime_history': [True, False, False, False, False],
            'is_provoking_window': [False, False, False, False, False],
            'is_diagnostic_workup': [False, True, True, False, False],
            'is_initial_treatment': [False, False, False, True, True],
            'is_escalation': [False, False, False, False, False],
            'ccs_category': ['52', '61', '47', '52', '216'],
            'quantity': [1, 1, 1, 1, 1],
        })

        result = build_patient_pe_features(df)

        assert result['empi'] == '100'
        assert result['prior_ivc_filter_ever'] == True
        assert result['cta_chest_performed'] == True
        assert result['echo_performed'] == True
        assert result['catheter_directed_therapy'] == True
        assert result['intubation_performed'] == True
        assert result['diagnostic_workup_intensity'] >= 2

    def test_multiple_patients(self):
        """Process multiple patients."""
        from transformers.pe_feature_builder import build_pe_features_for_cohort

        df = pd.DataFrame({
            'empi': ['100', '100', '200', '200'],
            'code': ['71275', '93306', '71275', '31500'],
            'procedure_name': ['CTA', 'ECHO', 'CTA', 'INTUBATION'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00'] * 4),
            'hours_from_pe': [0, 2, 0, 2],
            'is_lifetime_history': [False] * 4,
            'is_provoking_window': [False] * 4,
            'is_diagnostic_workup': [True, True, True, False],
            'is_initial_treatment': [False, False, False, True],
            'is_escalation': [False] * 4,
            'ccs_category': ['61', '47', '61', '216'],
            'quantity': [1] * 4,
        })

        result = build_pe_features_for_cohort(df)

        assert len(result) == 2
        assert '100' in result['empi'].values
        assert '200' in result['empi'].values

        patient_100 = result[result['empi'] == '100'].iloc[0]
        assert patient_100['cta_chest_performed'] == True
        assert patient_100['echo_performed'] == True

        patient_200 = result[result['empi'] == '200'].iloc[0]
        assert patient_200['cta_chest_performed'] == True
        assert patient_200['intubation_performed'] == True


class TestOutputSchema:
    """Test output schema and column presence."""

    def test_required_columns_present(self):
        """All required columns are in output."""
        from transformers.pe_feature_builder import build_patient_pe_features

        df = pd.DataFrame({
            'empi': ['100'],
            'code': ['71275'],
            'procedure_name': ['CTA'],
            'procedure_datetime': pd.to_datetime(['2023-07-27 12:00']),
            'hours_from_pe': [0],
            'is_lifetime_history': [False],
            'is_provoking_window': [False],
            'is_diagnostic_workup': [True],
            'is_initial_treatment': [False],
            'is_escalation': [False],
            'ccs_category': ['61'],
            'quantity': [1],
        })

        result = build_patient_pe_features(df)

        # Lifetime history features
        assert 'prior_ivc_filter_ever' in result
        assert 'prior_cdt_for_vte' in result
        assert 'lifetime_surgical_count' in result

        # Provoking features
        assert 'surgery_within_30_days' in result
        assert 'provoked_pe' in result

        # Diagnostic features
        assert 'cta_chest_performed' in result
        assert 'echo_performed' in result
        assert 'diagnostic_workup_intensity' in result

        # Treatment features
        assert 'catheter_directed_therapy' in result
        assert 'ivc_filter_placed' in result
        assert 'intubation_performed' in result
        assert 'ecmo_initiated' in result

        # Escalation features
        assert 'delayed_intubation' in result
        assert 'any_transfusion' in result
        assert 'cardiac_arrest_post_pe' in result
