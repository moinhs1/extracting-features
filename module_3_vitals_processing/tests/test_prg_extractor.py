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
