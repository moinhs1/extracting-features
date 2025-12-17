# tests/test_pipeline.py
"""Tests for main SocialPhysicalPipeline."""

import pytest
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPipelineComponents:
    """Test pipeline initialization and core functionality."""

    def test_pipeline_initializes(self):
        """Test pipeline can be initialized with index dates."""
        from pipeline import SocialPhysicalPipeline

        index_dates = {'100001': datetime(2020, 3, 15)}
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/fake.txt',
            index_dates=index_dates,
        )
        assert pipeline.index_dates == index_dates

    def test_pipeline_processes_sample_data(self):
        """Test pipeline can process pre-loaded data."""
        from pipeline import SocialPhysicalPipeline

        sample_data = pd.DataFrame({
            'EMPI': ['100001'] * 4,
            'Date': ['1/1/2020'] * 4,
            'Concept_Name': ['BMI', 'Weight', 'Smoking Tobacco Use-Never Smoker', 'Alcohol User-Never'],
            'Result': ['28.5', '185', '', ''],
            'Units': ['kg/m2', 'lbs', '', ''],
        })

        index_dates = {'100001': datetime(2020, 3, 15)}
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/fake.txt',
            index_dates=index_dates,
        )

        # Process pre-loaded data
        features_df = pipeline.process_data(sample_data, ['100001'])

        assert len(features_df) == 1
        assert features_df.iloc[0]['empi'] == '100001'
        assert features_df.iloc[0]['bmi_at_index'] == 28.5
        assert features_df.iloc[0]['smoking_status_at_index'] == 'never'

    def test_pipeline_handles_empty_data(self):
        """Test pipeline handles empty data gracefully."""
        from pipeline import SocialPhysicalPipeline

        empty_data = pd.DataFrame({
            'EMPI': [],
            'Date': [],
            'Concept_Name': [],
            'Result': [],
            'Units': [],
        })

        index_dates = {'100001': datetime(2020, 3, 15)}
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/fake.txt',
            index_dates=index_dates,
        )

        # Process empty data - should not crash
        features_df = pipeline.process_data(empty_data, ['100001'])
        assert len(features_df) == 1  # Still one row for the patient
        assert features_df.iloc[0]['empi'] == '100001'

    def test_pipeline_processes_multiple_patients(self):
        """Test pipeline processes multiple patients correctly."""
        from pipeline import SocialPhysicalPipeline

        sample_data = pd.DataFrame({
            'EMPI': ['100001', '100001', '100002', '100002'],
            'Date': ['1/1/2020', '1/1/2020', '1/5/2020', '1/5/2020'],
            'Concept_Name': ['BMI', 'Weight', 'BMI', 'Weight'],
            'Result': ['28.5', '185', '24.0', '150'],
            'Units': ['kg/m2', 'lbs', 'kg/m2', 'lbs'],
        })

        index_dates = {
            '100001': datetime(2020, 3, 15),
            '100002': datetime(2020, 3, 15),
        }
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/fake.txt',
            index_dates=index_dates,
        )

        features_df = pipeline.process_data(sample_data, ['100001', '100002'])

        assert len(features_df) == 2
        empis = features_df['empi'].tolist()
        assert '100001' in empis
        assert '100002' in empis


class TestPipelineOptionalPaths:
    """Test pipeline handles optional paths."""

    def test_pipeline_with_hnp_path(self):
        """Test pipeline accepts HNP path."""
        from pipeline import SocialPhysicalPipeline

        index_dates = {'100001': datetime(2020, 3, 15)}
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/Phy.txt',
            index_dates=index_dates,
            hnp_path='/tmp/Hnp.txt',
        )
        assert pipeline.hnp_path == Path('/tmp/Hnp.txt')

    def test_pipeline_with_prg_path(self):
        """Test pipeline accepts PRG path."""
        from pipeline import SocialPhysicalPipeline

        index_dates = {'100001': datetime(2020, 3, 15)}
        pipeline = SocialPhysicalPipeline(
            phy_path='/tmp/Phy.txt',
            index_dates=index_dates,
            prg_path='/tmp/Prg.txt',
        )
        assert pipeline.prg_path == Path('/tmp/Prg.txt')


class TestLoadPatientTimelines:
    """Test patient timeline loading function."""

    def test_load_patient_timelines_function_exists(self):
        """Test that load_patient_timelines function is importable."""
        from pipeline import load_patient_timelines
        assert callable(load_patient_timelines)


class TestTransformerImports:
    """Test that all transformers are properly exported from __init__.py."""

    def test_can_import_all_builders(self):
        """Test all builder classes can be imported from transformers package."""
        from transformers import (
            BMIBuilder,
            WeightBuilder,
            HeightBuilder,
            BSABuilder,
            BodyMeasurementsBuilder,
            SmokingBuilder,
            AlcoholBuilder,
            DrugUseBuilder,
            SocialHistoryBuilder,
            PainBuilder,
            FunctionalStatusBuilder,
        )
        assert BMIBuilder is not None
        assert SmokingBuilder is not None
        assert DrugUseBuilder is not None
