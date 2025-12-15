"""Integration tests for diagnosis processing pipeline."""
import pytest
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime
from io import StringIO
from build_layers import run_pipeline

SAMPLE_DIA = """EMPI|EPIC_PMRN|MRN_Type|MRN|Date|Diagnosis_Name|Code_Type|Code|Diagnosis_Flag|Provider|Clinic|Hospital|Inpatient_Outpatient|Encounter_number
P001|10001|MGH|123|1/15/2023|Heart failure|ICD10|I50.9|Primary|Dr A|Card|MGH|Outpatient|E001
P001|10001|MGH|123|6/15/2023|Pulmonary embolism|ICD10|I26.99|Primary|Dr B|ED|MGH|Inpatient|E002
P001|10001|MGH|123|6/20/2023|Acute kidney injury|ICD10|N17.9|Primary|Dr C|ICU|MGH|Inpatient|E003
P002|10002|MGH|456|3/1/2023|COPD|ICD10|J44.9|Primary|Dr D|Pulm|MGH|Outpatient|E004
P002|10002|MGH|456|6/10/2023|Lung cancer|ICD10|C34.9|Primary|Dr E|Onc|MGH|Inpatient|E005
P002|10002|MGH|456|6/15/2023|Pulmonary embolism|ICD10|I26.0|Primary|Dr F|ED|MGH|Inpatient|E006
"""


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_pipeline_produces_layer1_output(self):
        """Pipeline produces Layer 1 parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            # Mock PE times
            pe_times = {
                "P001": datetime(2023, 6, 15),
                "P002": datetime(2023, 6, 15),
            }

            run_pipeline(
                dia_file=StringIO(SAMPLE_DIA),
                patient_ids={"P001", "P002"},
                pe_times=pe_times,
                output_path=output_path,
            )

            layer1_path = output_path / "layer1" / "canonical_diagnoses.parquet"
            assert layer1_path.exists()

            df = pd.read_parquet(layer1_path)
            assert len(df) > 0
            assert "EMPI" in df.columns
            assert "days_from_pe" in df.columns

    def test_pipeline_produces_layer2_output(self):
        """Pipeline produces Layer 2 parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            pe_times = {
                "P001": datetime(2023, 6, 15),
                "P002": datetime(2023, 6, 15),
            }

            run_pipeline(
                dia_file=StringIO(SAMPLE_DIA),
                patient_ids={"P001", "P002"},
                pe_times=pe_times,
                output_path=output_path,
            )

            layer2_path = output_path / "layer2" / "comorbidity_scores.parquet"
            assert layer2_path.exists()

            df = pd.read_parquet(layer2_path)
            assert len(df) == 2  # One row per patient
            assert "cci_score" in df.columns

    def test_pe_relative_timing_correct(self):
        """PE-relative timing is calculated correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            pe_times = {
                "P001": datetime(2023, 6, 15),
            }

            run_pipeline(
                dia_file=StringIO(SAMPLE_DIA),
                patient_ids={"P001"},
                pe_times=pe_times,
                output_path=output_path,
            )

            df = pd.read_parquet(output_path / "layer1" / "canonical_diagnoses.parquet")

            # Heart failure on 1/15 should be ~150 days before PE on 6/15
            hf = df[df["icd_code"] == "I50.9"].iloc[0]
            assert hf["days_from_pe"] < -100
            assert hf["is_preexisting"] == True

            # AKI on 6/20 should be 5 days after PE
            aki = df[df["icd_code"] == "N17.9"].iloc[0]
            assert aki["days_from_pe"] == 5
            assert aki["is_complication"] == True
