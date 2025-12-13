"""Tests for canonical procedure record extraction."""

import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPrcFileLoader:
    """Test Prc.txt loading functionality."""

    def test_load_prc_chunk(self):
        """Load a chunk of procedure data."""
        from extractors.canonical_extractor import load_prc_chunk

        df = load_prc_chunk(n_rows=100)

        assert len(df) == 100
        assert 'EMPI' in df.columns
        assert 'Code' in df.columns
        assert 'Date' in df.columns

    def test_column_names_correct(self):
        """Verify expected columns exist."""
        from extractors.canonical_extractor import load_prc_chunk

        df = load_prc_chunk(n_rows=10)

        expected_cols = [
            'EMPI', 'Date', 'Procedure_Name', 'Code_Type',
            'Code', 'Quantity', 'Inpatient_Outpatient', 'Encounter_number'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_code_type_distribution(self):
        """Check code type distribution matches expected."""
        from extractors.canonical_extractor import load_prc_chunk

        df = load_prc_chunk(n_rows=10000)
        code_types = df['Code_Type'].value_counts(normalize=True)

        # CPT should be majority
        assert code_types.get('CPT', 0) > 0.5


class TestCohortFiltering:
    """Test cohort filtering functionality."""

    def test_filter_to_cohort(self):
        """Filter procedures to PE cohort patients only."""
        from extractors.canonical_extractor import filter_to_cohort

        prc_df = pd.DataFrame({
            'EMPI': ['100', '200', '300', '100'],
            'Code': ['71275', '93306', '31500', '36430'],
        })

        cohort_empis = {'100', '300'}

        filtered = filter_to_cohort(prc_df, cohort_empis)

        assert len(filtered) == 3
        assert set(filtered['EMPI'].unique()) == {'100', '300'}


class TestTimeAlignment:
    """Test time alignment to PE Time Zero."""

    def test_compute_hours_from_pe(self):
        """Compute hours relative to PE Time Zero."""
        from extractors.canonical_extractor import compute_hours_from_pe

        prc_df = pd.DataFrame({
            'EMPI': ['100', '100', '100'],
            'Date': pd.to_datetime(['2023-07-27', '2023-07-28', '2023-07-26']),
        })

        time_zero_map = {
            '100': pd.Timestamp('2023-07-27 12:00:00'),
        }

        result = compute_hours_from_pe(prc_df, time_zero_map)

        assert 'hours_from_pe' in result.columns
        hours = result['hours_from_pe'].tolist()
        assert hours[0] == pytest.approx(-12, abs=1)
        assert hours[1] == pytest.approx(12, abs=1)
        assert hours[2] == pytest.approx(-36, abs=1)


class TestTemporalFlags:
    """Test temporal flag computation."""

    def test_compute_temporal_flags(self):
        """Compute 7 temporal category flags."""
        from extractors.canonical_extractor import compute_temporal_flags

        df = pd.DataFrame({
            'hours_from_pe': [-800, -100, -12, 0, 24, 100, 1000],
        })

        result = compute_temporal_flags(df)

        # Check all 7 flags exist
        expected_flags = [
            'is_lifetime_history', 'is_remote_antecedent', 'is_provoking_window',
            'is_diagnostic_workup', 'is_initial_treatment', 'is_escalation',
            'is_post_discharge'
        ]
        for flag in expected_flags:
            assert flag in result.columns

    def test_provoking_window_flag(self):
        """Provoking window is -720h to 0h."""
        from extractors.canonical_extractor import compute_temporal_flags

        df = pd.DataFrame({
            'hours_from_pe': [-800, -500, -100, 0, 100],
        })

        result = compute_temporal_flags(df)

        # -500 and -100 should be in provoking window
        expected = [False, True, True, False, False]
        assert result['is_provoking_window'].tolist() == expected

    def test_diagnostic_workup_flag(self):
        """Diagnostic workup is -24h to +24h."""
        from extractors.canonical_extractor import compute_temporal_flags

        df = pd.DataFrame({
            'hours_from_pe': [-30, -12, 0, 12, 30],
        })

        result = compute_temporal_flags(df)

        expected = [False, True, True, True, False]
        assert result['is_diagnostic_workup'].tolist() == expected


class TestCanonicalSchema:
    """Test transformation to canonical schema."""

    def test_transform_to_canonical(self):
        """Transform raw data to canonical schema."""
        from extractors.canonical_extractor import transform_to_canonical

        raw_df = pd.DataFrame({
            'EMPI': ['100'],
            'Date': pd.to_datetime(['2023-07-27']),
            'Procedure_Name': ['CT ANGIOGRAM CHEST'],
            'Code_Type': ['CPT'],
            'Code': ['71275'],
            'Quantity': ['1'],
            'Provider': ['Dr. Smith'],
            'Clinic': ['Radiology'],
            'Hospital': ['MGH'],
            'Inpatient_Outpatient': ['Inpatient'],
            'Encounter_number': ['ENC123'],
            'hours_from_pe': [0.0],
        })

        result = transform_to_canonical(raw_df)

        assert 'empi' in result.columns
        assert 'procedure_datetime' in result.columns
        assert 'code_type' in result.columns
        assert 'code' in result.columns
        assert 'inpatient' in result.columns
        assert result['inpatient'].iloc[0] == True
