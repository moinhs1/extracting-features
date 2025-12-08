"""Tests for prg_extractor module."""
import pytest
from datetime import datetime
import json
import tempfile
from pathlib import Path


class TestExtractionCheckpoint:
    """Test checkpoint dataclass and serialization."""

    def test_checkpoint_dataclass_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        checkpoint = ExtractionCheckpoint(
            input_path='/path/to/input.txt',
            output_path='/path/to/output.parquet',
            rows_processed=1000,
            chunks_completed=1,
            records_extracted=500,
            started_at=datetime.now(),
            updated_at=datetime.now(),
        )
        assert checkpoint.rows_processed == 1000
        assert checkpoint.chunks_completed == 1

    def test_checkpoint_to_dict(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        checkpoint = ExtractionCheckpoint(
            input_path='/path/to/input.txt',
            output_path='/path/to/output.parquet',
            rows_processed=1000,
            chunks_completed=1,
            records_extracted=500,
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            updated_at=datetime(2024, 1, 1, 10, 5, 0),
        )
        d = checkpoint.to_dict()
        assert d['rows_processed'] == 1000
        assert d['input_path'] == '/path/to/input.txt'

    def test_checkpoint_from_dict(self):
        from module_3_vitals_processing.extractors.prg_extractor import ExtractionCheckpoint
        data = {
            'input_path': '/path/to/input.txt',
            'output_path': '/path/to/output.parquet',
            'rows_processed': 2000,
            'chunks_completed': 2,
            'records_extracted': 1000,
            'started_at': '2024-01-01T10:00:00',
            'updated_at': '2024-01-01T10:10:00',
        }
        checkpoint = ExtractionCheckpoint.from_dict(data)
        assert checkpoint.rows_processed == 2000
        assert checkpoint.chunks_completed == 2


class TestCheckpointIO:
    """Test checkpoint save and load functions."""

    def test_save_checkpoint(self):
        from module_3_vitals_processing.extractors.prg_extractor import (
            ExtractionCheckpoint, save_checkpoint, CHECKPOINT_FILE
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint = ExtractionCheckpoint(
                input_path='/input.txt',
                output_path='/output.parquet',
                rows_processed=5000,
                chunks_completed=5,
                records_extracted=2500,
                started_at=datetime.now(),
                updated_at=datetime.now(),
            )
            save_checkpoint(checkpoint, output_dir)
            assert (output_dir / CHECKPOINT_FILE).exists()

    def test_load_checkpoint_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import (
            ExtractionCheckpoint, save_checkpoint, load_checkpoint
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            original = ExtractionCheckpoint(
                input_path='/input.txt',
                output_path='/output.parquet',
                rows_processed=5000,
                chunks_completed=5,
                records_extracted=2500,
                started_at=datetime(2024, 1, 1, 10, 0, 0),
                updated_at=datetime(2024, 1, 1, 10, 5, 0),
            )
            save_checkpoint(original, output_dir)
            loaded = load_checkpoint(output_dir)
            assert loaded is not None
            assert loaded.rows_processed == 5000
            assert loaded.chunks_completed == 5

    def test_load_checkpoint_not_exists(self):
        from module_3_vitals_processing.extractors.prg_extractor import load_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            loaded = load_checkpoint(output_dir)
            assert loaded is None


class TestIdentifyPrgSections:
    """Test Prg section identification."""

    def test_finds_physical_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "History... Physical Exam: BP 120/80 HR 72 General appearance good"
        sections = identify_prg_sections(text)
        assert 'physical_exam' in sections
        assert 'BP 120/80' in sections['physical_exam']

    def test_finds_vitals_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Assessment... Vitals: T 98.6F HR 80 BP 130/85 Plan..."
        sections = identify_prg_sections(text)
        assert 'vitals' in sections

    def test_finds_on_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Allergies... ON EXAM: Vital Signs BP 109/57, P 76 afebrile"
        sections = identify_prg_sections(text)
        assert 'on_exam' in sections

    def test_finds_objective_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "Subjective: pain... Objective: BP 120/80 HR 72 Assessment..."
        sections = identify_prg_sections(text)
        assert 'objective' in sections

    def test_returns_empty_when_no_sections(self):
        from module_3_vitals_processing.extractors.prg_extractor import identify_prg_sections
        text = "This is a note without any section headers."
        sections = identify_prg_sections(text)
        assert sections == {}


class TestIsInSkipSection:
    """Test skip section detection for false positive filtering."""

    def test_detects_allergies_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "Patient info... Allergies: atenolol - fatigue, HR 50 generic synthroid..."
        # Position at HR 50
        position = text.find("HR 50")
        assert is_in_skip_section(text, position)

    def test_detects_medications_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "Assessment... Medications: lisinopril 10mg causes BP drop..."
        position = text.find("BP drop")
        assert is_in_skip_section(text, position)

    def test_detects_past_medical_history(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "ROS negative... Past Medical History: HTN with BP 180/100..."
        position = text.find("BP 180")
        assert is_in_skip_section(text, position)

    def test_allows_physical_exam_section(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        text = "History... Physical Exam: BP 120/80 HR 72..."
        position = text.find("BP 120")
        assert not is_in_skip_section(text, position)

    def test_valid_section_overrides_skip(self):
        from module_3_vitals_processing.extractors.prg_extractor import is_in_skip_section
        # Skip section followed by valid section
        text = "Allergies: penicillin... Physical Exam: BP 120/80 HR 72..."
        position = text.find("BP 120")
        assert not is_in_skip_section(text, position)


class TestExtractTemperatureWithMethod:
    """Test temperature extraction with method capture."""

    def test_extracts_temp_with_oral_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 36.8 °C (98.2 °F) (Oral)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'oral'

    def test_extracts_temp_with_temporal_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 36.2 °C (97.1 °F) (Temporal)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'temporal'

    def test_extracts_temp_with_rectal_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "temp 98.2F rectally"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'rectal'

    def test_extracts_temp_src_format(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp(Src) 36.7 °C (98 °F) (Oral)"
        results = extract_temperature_with_method(text)
        assert len(results) >= 1
        assert results[0]['method'] == 'oral'

    def test_returns_none_method_when_not_specified(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_temperature_with_method
        text = "Temp 98.6F"
        results = extract_temperature_with_method(text)
        # Should still extract temp, method may be None
        assert len(results) >= 1


class TestExtractPrgVitalsFromText:
    """Test combined vitals extraction from text."""

    def test_extracts_all_vital_types(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Physical Exam: BP 120/80 HR 72 RR 16 SpO2 98% Temp 98.6F (Oral)"
        results = extract_prg_vitals_from_text(text)
        types = {r['vital_type'] for r in results}
        assert 'SBP' in types
        assert 'DBP' in types
        assert 'HR' in types
        assert 'RR' in types
        assert 'SPO2' in types
        assert 'TEMP' in types

    def test_skips_vitals_in_allergies(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Allergies: atenolol - fatigue, HR 50. Physical Exam: HR 72"
        results = extract_prg_vitals_from_text(text)
        hr_values = [r['value'] for r in results if r['vital_type'] == 'HR']
        assert 72 in hr_values
        assert 50 not in hr_values  # Should be skipped

    def test_includes_temp_method(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Vitals: Temp 36.8 °C (98.2 °F) (Oral)"
        results = extract_prg_vitals_from_text(text)
        temp_results = [r for r in results if r['vital_type'] == 'TEMP']
        assert len(temp_results) >= 1
        assert temp_results[0].get('temp_method') == 'oral'

    def test_handles_empty_text(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        results = extract_prg_vitals_from_text("")
        assert results == []

    def test_handles_text_without_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import extract_prg_vitals_from_text
        text = "Patient presents for follow-up. Doing well."
        results = extract_prg_vitals_from_text(text)
        assert results == []


class TestProcessPrgRow:
    """Test single row processing."""

    def test_processes_row_with_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        import pandas as pd
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Physical Exam: BP 120/80 HR 72'
        })
        results = process_prg_row(row)
        assert len(results) >= 3  # SBP, DBP, HR
        assert all(r['EMPI'] == '12345' for r in results)
        assert all(r['source'] == 'prg' for r in results)

    def test_processes_row_without_vitals(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        import pandas as pd
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Patient doing well. Follow up in 3 months.'
        })
        results = process_prg_row(row)
        assert results == []

    def test_handles_empty_report_text(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        import pandas as pd
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': None
        })
        results = process_prg_row(row)
        assert results == []

    def test_includes_temp_method_in_results(self):
        from module_3_vitals_processing.extractors.prg_extractor import process_prg_row
        import pandas as pd
        row = pd.Series({
            'EMPI': '12345',
            'Report_Number': 'RPT001',
            'Report_Date_Time': '01/15/2024 10:30:00 AM',
            'Report_Text': 'Vitals: Temp 36.8 °C (98.2 °F) (Oral)'
        })
        results = process_prg_row(row)
        temp_results = [r for r in results if r['vital_type'] == 'TEMP']
        assert len(temp_results) >= 1
        assert temp_results[0]['temp_method'] == 'oral'
