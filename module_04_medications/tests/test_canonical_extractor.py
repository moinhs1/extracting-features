"""Tests for canonical medication record extraction."""

import pytest
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMedFileLoader:
    """Test Med.txt loading functionality."""

    def test_load_med_chunk(self):
        """Load a chunk of medication data."""
        from extractors.canonical_extractor import load_med_chunk

        # Load first 100 rows
        df = load_med_chunk(n_rows=100)

        assert len(df) == 100
        assert 'EMPI' in df.columns
        assert 'Medication' in df.columns
        assert 'Medication_Date' in df.columns

    def test_column_names_correct(self):
        """Verify expected columns exist."""
        from extractors.canonical_extractor import load_med_chunk

        df = load_med_chunk(n_rows=10)

        expected_cols = [
            'EMPI', 'Medication_Date', 'Medication', 'Code_Type',
            'Code', 'Quantity', 'Inpatient_Outpatient', 'Encounter_number'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestCohortFiltering:
    """Test cohort filtering functionality."""

    def test_filter_to_cohort(self):
        """Filter medications to PE cohort patients only."""
        from extractors.canonical_extractor import filter_to_cohort

        # Create test data
        med_df = pd.DataFrame({
            'EMPI': ['100', '200', '300', '100'],
            'Medication': ['Drug A', 'Drug B', 'Drug C', 'Drug D'],
        })

        cohort_empis = {'100', '300'}

        filtered = filter_to_cohort(med_df, cohort_empis)

        assert len(filtered) == 3
        assert set(filtered['EMPI'].unique()) == {'100', '300'}


class TestTimeAlignment:
    """Test time alignment to Time Zero."""

    def test_compute_hours_from_t0(self):
        """Compute hours relative to Time Zero."""
        from extractors.canonical_extractor import compute_hours_from_t0

        # Test data
        med_df = pd.DataFrame({
            'EMPI': ['100', '100', '100'],
            'Medication_Date': pd.to_datetime(['2023-07-27', '2023-07-28', '2023-07-26']),
        })

        # Time Zero for patient 100 is 2023-07-27 12:00:00
        time_zero_map = {
            '100': pd.Timestamp('2023-07-27 12:00:00'),
        }

        result = compute_hours_from_t0(med_df, time_zero_map)

        # First row: same day, assume noon - should be ~0
        # Second row: next day - should be ~24
        # Third row: day before - should be ~-24
        assert 'hours_from_t0' in result.columns

        # Check approximate values (date only, so we assume midnight)
        hours = result['hours_from_t0'].tolist()
        assert hours[0] == pytest.approx(-12, abs=1)  # 2023-07-27 00:00 vs 12:00 = -12h
        assert hours[1] == pytest.approx(12, abs=1)   # 2023-07-28 00:00 vs 2023-07-27 12:00 = +12h
        assert hours[2] == pytest.approx(-36, abs=1)  # 2023-07-26 00:00 vs 2023-07-27 12:00 = -36h


class TestWindowFiltering:
    """Test study window filtering."""

    def test_filter_study_window(self):
        """Filter to study window (-30 to +30 days)."""
        from extractors.canonical_extractor import filter_study_window

        df = pd.DataFrame({
            'hours_from_t0': [-800, -100, 0, 100, 800],
            'Medication': ['A', 'B', 'C', 'D', 'E'],
        })

        # Default window: -720 to +720 hours (-30 to +30 days)
        filtered = filter_study_window(df)

        assert len(filtered) == 3
        assert list(filtered['Medication']) == ['B', 'C', 'D']
