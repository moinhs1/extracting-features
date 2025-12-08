"""Tests for layer1_builder module."""
import pytest
import pandas as pd
from processing.layer1_builder import LAYER1_SCHEMA, CORE_VITALS
from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class MockTimeline:
    """Mock PatientTimeline for testing."""
    patient_id: str
    time_zero: datetime
    window_start: datetime
    window_end: datetime
    phase_boundaries: Dict
    encounter_info: Dict
    outcomes: Dict
    metadata: Dict


class TestLayer1Schema:
    """Tests for Layer 1 schema definitions."""

    def test_schema_has_required_columns(self):
        """Schema defines all required columns."""
        required = {
            "EMPI", "timestamp", "hours_from_pe", "vital_type",
            "value", "units", "source", "source_detail", "confidence",
            "is_calculated", "is_flagged_abnormal", "report_number"
        }
        assert required <= set(LAYER1_SCHEMA.keys())

    def test_core_vitals_defined(self):
        """Seven core vitals are defined."""
        expected = {"HR", "SBP", "DBP", "MAP", "RR", "SPO2", "TEMP"}
        assert set(CORE_VITALS) == expected

    def test_schema_types_are_valid(self):
        """Schema types are valid pandas/numpy types."""
        valid_types = {"str", "datetime64[ns]", "float64", "bool"}
        for col, dtype in LAYER1_SCHEMA.items():
            assert dtype in valid_types, f"{col} has invalid type {dtype}"


class TestNormalizePhy:
    """Tests for PHY source normalization."""

    def test_normalize_phy_adds_required_columns(self):
        """PHY normalization adds all Layer 1 columns."""
        from processing.layer1_builder import normalize_phy_source

        # Minimal PHY-like dataframe
        phy_df = pd.DataFrame({
            "EMPI": ["E001", "E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00", "2023-06-15 11:00"]),
            "vital_type": ["HR", "SBP"],
            "value": [72.0, 120.0],
            "units": ["bpm", "mmHg"],
            "source": ["phy", "phy"],
            "encounter_type": ["IP", "IP"],
            "encounter_number": ["ENC001", "ENC001"],
        })

        result = normalize_phy_source(phy_df)

        assert "source_detail" in result.columns
        assert "confidence" in result.columns
        assert "is_calculated" in result.columns
        assert "is_flagged_abnormal" in result.columns
        assert "report_number" in result.columns

    def test_normalize_phy_sets_confidence_1(self):
        """PHY source has confidence = 1.0 (structured data)."""
        from processing.layer1_builder import normalize_phy_source

        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["IP"],
            "encounter_number": ["ENC001"],
        })

        result = normalize_phy_source(phy_df)
        assert result["confidence"].iloc[0] == 1.0

    def test_normalize_phy_maps_encounter_type_to_source_detail(self):
        """encounter_type becomes source_detail."""
        from processing.layer1_builder import normalize_phy_source

        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["Inpatient"],
            "encounter_number": ["ENC001"],
        })

        result = normalize_phy_source(phy_df)
        assert result["source_detail"].iloc[0] == "Inpatient"


class TestNormalizeHnpPrg:
    """Tests for HNP/PRG source normalization."""

    def test_normalize_hnp_preserves_confidence(self):
        """HNP normalization preserves extraction confidence."""
        from processing.layer1_builder import normalize_hnp_source

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs_section"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
        })

        result = normalize_hnp_source(hnp_df)
        assert result["confidence"].iloc[0] == 0.85

    def test_normalize_hnp_maps_extraction_context_to_source_detail(self):
        """extraction_context becomes source_detail."""
        from processing.layer1_builder import normalize_hnp_source

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs_section"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
        })

        result = normalize_hnp_source(hnp_df)
        assert result["source_detail"].iloc[0] == "vital_signs_section"

    def test_normalize_prg_preserves_confidence(self):
        """PRG normalization preserves extraction confidence."""
        from processing.layer1_builder import normalize_prg_source

        prg_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["prg"],
            "extraction_context": ["vital_signs_table"],
            "confidence": [0.9],
            "is_flagged_abnormal": [False],
            "report_number": ["RPT002"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
            "temp_method": [None],
        })

        result = normalize_prg_source(prg_df)
        assert result["confidence"].iloc[0] == 0.9


class TestMAPCalculation:
    """Tests for Mean Arterial Pressure calculation."""

    def test_calculate_map_formula(self):
        """MAP = DBP + (SBP - DBP) / 3."""
        from processing.layer1_builder import calculate_map
        # Standard BP 120/80
        result = calculate_map(120, 80)
        expected = 80 + (120 - 80) / 3  # 93.33
        assert abs(result - expected) < 0.01

    def test_calculate_map_normal_values(self):
        """Normal MAP is around 70-105."""
        from processing.layer1_builder import calculate_map
        result = calculate_map(120, 80)
        assert 70 <= result <= 105

    def test_generate_calculated_maps(self):
        """Generate MAP rows from SBP/DBP pairs at same timestamp."""
        from processing.layer1_builder import generate_calculated_maps

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E001", "E002"],
            "timestamp": pd.to_datetime([
                "2023-06-15 10:00", "2023-06-15 10:00",  # Same time - pair
                "2023-06-15 11:00",  # SBP only - no pair
                "2023-06-15 10:00",  # Different patient
            ]),
            "vital_type": ["SBP", "DBP", "SBP", "SBP"],
            "value": [120.0, 80.0, 130.0, 140.0],
            "units": ["mmHg", "mmHg", "mmHg", "mmHg"],
            "source": ["phy", "phy", "phy", "phy"],
            "source_detail": ["IP", "IP", "IP", "IP"],
            "confidence": [1.0, 1.0, 1.0, 1.0],
            "is_calculated": [False, False, False, False],
            "is_flagged_abnormal": [False, False, False, False],
            "report_number": ["", "", "", ""],
            "hours_from_pe": [0.0, 0.0, 1.0, 0.0],
        })

        maps = generate_calculated_maps(df)

        # Should generate 1 MAP (E001 at 10:00)
        assert len(maps) == 1
        assert maps["vital_type"].iloc[0] == "MAP"
        assert maps["is_calculated"].iloc[0] == True
        assert abs(maps["value"].iloc[0] - 93.33) < 0.1


class TestPatientTimelineLoading:
    """Tests for patient timeline loading."""

    def test_load_pe_times_returns_dict(self):
        """load_pe_times returns dict mapping EMPI to PE timestamp."""
        from processing.layer1_builder import load_pe_times
        import pickle
        from pathlib import Path
        from unittest.mock import patch, MagicMock
        from datetime import datetime

        # Mock PatientTimeline
        mock_timeline = MagicMock()
        mock_timeline.patient_id = "E001"
        mock_timeline.time_zero = datetime(2023, 6, 15, 10, 0, 0)

        mock_timelines = {"E001": mock_timeline}

        with patch("builtins.open", create=True):
            with patch("pickle.load", return_value=mock_timelines):
                result = load_pe_times(Path("/fake/path.pkl"))

        assert isinstance(result, dict)
        assert "E001" in result
        assert result["E001"] == datetime(2023, 6, 15, 10, 0, 0)

    def test_load_pe_times_handles_multiple_patients(self):
        """load_pe_times handles multiple patients."""
        from processing.layer1_builder import load_pe_times
        from unittest.mock import patch, MagicMock
        from datetime import datetime
        from pathlib import Path

        mock_t1 = MagicMock()
        mock_t1.patient_id = "E001"
        mock_t1.time_zero = datetime(2023, 6, 15, 10, 0, 0)

        mock_t2 = MagicMock()
        mock_t2.patient_id = "E002"
        mock_t2.time_zero = datetime(2023, 6, 16, 14, 30, 0)

        mock_timelines = {"E001": mock_t1, "E002": mock_t2}

        with patch("builtins.open", create=True):
            with patch("pickle.load", return_value=mock_timelines):
                result = load_pe_times(Path("/fake/path.pkl"))

        assert len(result) == 2
        assert result["E002"] == datetime(2023, 6, 16, 14, 30, 0)


class TestAddPERelativeTimestamps:
    """Tests for adding PE-relative timestamps to vitals."""

    def test_adds_hours_from_pe_column(self):
        """Adds hours_from_pe column based on PE times."""
        from processing.layer1_builder import add_pe_relative_timestamps
        from datetime import datetime

        df = pd.DataFrame({
            "EMPI": ["E001", "E001", "E002"],
            "timestamp": pd.to_datetime([
                "2023-06-15 11:00",  # 1 hour after PE
                "2023-06-15 09:00",  # 1 hour before PE
                "2023-06-16 14:30",  # At PE time
            ]),
            "vital_type": ["HR", "HR", "HR"],
            "value": [72.0, 70.0, 80.0],
        })

        pe_times = {
            "E001": datetime(2023, 6, 15, 10, 0, 0),
            "E002": datetime(2023, 6, 16, 14, 30, 0),
        }

        result = add_pe_relative_timestamps(df, pe_times)

        assert "hours_from_pe" in result.columns
        assert result.loc[result["EMPI"] == "E001"].iloc[0]["hours_from_pe"] == 1.0
        assert result.loc[result["EMPI"] == "E001"].iloc[1]["hours_from_pe"] == -1.0
        assert result.loc[result["EMPI"] == "E002"].iloc[0]["hours_from_pe"] == 0.0

    def test_drops_patients_without_pe_time(self):
        """Patients without PE time are dropped."""
        from processing.layer1_builder import add_pe_relative_timestamps
        from datetime import datetime

        df = pd.DataFrame({
            "EMPI": ["E001", "E999"],  # E999 has no PE time
            "timestamp": pd.to_datetime(["2023-06-15 11:00", "2023-06-15 11:00"]),
            "vital_type": ["HR", "HR"],
            "value": [72.0, 80.0],
        })

        pe_times = {"E001": datetime(2023, 6, 15, 10, 0, 0)}

        result = add_pe_relative_timestamps(df, pe_times)

        assert len(result) == 1
        assert "E999" not in result["EMPI"].values


class TestBuildLayer1:
    """Tests for main Layer 1 build function."""

    def test_build_layer1_integration(self, tmp_path):
        """Integration test for build_layer1 function."""
        from processing.layer1_builder import build_layer1
        import pickle

        # Create mock input parquet files
        phy_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 11:00"]),
            "vital_type": ["HR"],
            "value": [72.0],
            "units": ["bpm"],
            "source": ["phy"],
            "encounter_type": ["IP"],
            "encounter_number": ["ENC001"],
        })
        phy_path = tmp_path / "phy_vitals_raw.parquet"
        phy_df.to_parquet(phy_path)

        hnp_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 10:30"]),
            "vital_type": ["SBP"],
            "value": [120.0],
            "units": ["mmHg"],
            "source": ["hnp"],
            "extraction_context": ["vital_signs"],
            "confidence": [0.9],
            "is_flagged_abnormal": [False],
            "report_number": ["R001"],
            "report_date_time": pd.to_datetime(["2023-06-15 09:00"]),
            "timestamp_source": ["report"],
            "timestamp_offset_hours": [0.0],
        })
        hnp_path = tmp_path / "hnp_vitals_raw.parquet"
        hnp_df.to_parquet(hnp_path)

        prg_df = pd.DataFrame({
            "EMPI": ["E001"],
            "timestamp": pd.to_datetime(["2023-06-15 12:00"]),
            "vital_type": ["DBP"],
            "value": [80.0],
            "units": ["mmHg"],
            "source": ["prg"],
            "extraction_context": ["vital_signs"],
            "confidence": [0.85],
            "is_flagged_abnormal": [False],
            "report_number": ["R002"],
            "report_date_time": pd.to_datetime(["2023-06-15 11:00"]),
            "timestamp_source": ["report"],
            "timestamp_offset_hours": [0.0],
            "temp_method": [None],
        })
        prg_path = tmp_path / "prg_vitals_raw.parquet"
        prg_df.to_parquet(prg_path)

        # Create mock patient timelines using module-level MockTimeline
        mock_timeline = MockTimeline(
            patient_id="E001",
            time_zero=datetime(2023, 6, 15, 10, 0, 0),
            window_start=datetime(2023, 6, 15, 9, 0, 0),
            window_end=datetime(2023, 6, 15, 18, 0, 0),
            phase_boundaries={},
            encounter_info={},
            outcomes={},
            metadata={}
        )
        timelines = {"E001": mock_timeline}

        timeline_path = tmp_path / "patient_timelines.pkl"
        with open(timeline_path, "wb") as f:
            pickle.dump(timelines, f)

        # Output path
        output_path = tmp_path / "canonical_vitals.parquet"

        # Run build
        result = build_layer1(
            phy_path=phy_path,
            hnp_path=hnp_path,
            prg_path=prg_path,
            timeline_path=timeline_path,
            output_path=output_path
        )

        # Verify output file exists
        assert output_path.exists()

        # Verify schema
        output_df = pd.read_parquet(output_path)
        assert set(output_df.columns) >= {"EMPI", "timestamp", "hours_from_pe",
                                          "vital_type", "value", "source"}

        # Verify merged data
        assert len(output_df) >= 3  # At least PHY + HNP + PRG records
        assert set(output_df["source"].unique()) == {"phy", "hnp", "prg"}
