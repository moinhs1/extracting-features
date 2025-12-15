"""Main pipeline for building diagnosis layers."""
from typing import Dict, Set, Union, TextIO
from pathlib import Path
from datetime import datetime
import pickle
import pandas as pd
from extractors.diagnosis_extractor import extract_diagnoses_for_patients, filter_excluded_codes
from processing.layer1_builder import build_layer1
from processing.layer2_builder import build_layer2_comorbidity_scores, save_layer2


def load_pe_times(timelines_path: Path) -> Dict[str, datetime]:
    """Load PE index times from patient timelines.

    Args:
        timelines_path: Path to patient_timelines.pkl

    Returns:
        Dict mapping EMPI to PE index datetime
    """
    import __main__
    import sys

    # Add module_1 directory to path so PatientTimeline class can be found during unpickling
    # timelines_path = .../module_1_core_infrastructure/outputs/patient_timelines.pkl
    # parent.parent = .../module_1_core_infrastructure/
    module1_dir = timelines_path.parent.parent
    if str(module1_dir) not in sys.path:
        sys.path.insert(0, str(module1_dir))

    # Import and inject into __main__ for pickle compatibility
    # (pickle was saved when module_01 was running as __main__)
    from module_01_core_infrastructure import PatientTimeline
    __main__.PatientTimeline = PatientTimeline

    with open(timelines_path, "rb") as f:
        timelines = pickle.load(f)

    pe_times = {}
    for empi, timeline in timelines.items():
        if hasattr(timeline, "time_zero") and timeline.time_zero:
            pe_times[str(empi)] = timeline.time_zero

    return pe_times


def run_pipeline(
    dia_file: Union[str, Path, TextIO],
    patient_ids: Set[str],
    pe_times: Dict[str, datetime],
    output_path: Path,
    min_days: int = -365 * 5,
    max_days: int = 365,
) -> None:
    """Run the full diagnosis processing pipeline.

    Args:
        dia_file: Path to Dia.txt or file handle
        patient_ids: Set of EMPIs to process
        pe_times: Dict mapping EMPI to PE index time
        output_path: Output directory
        min_days: Minimum days from PE to include
        max_days: Maximum days from PE to include
    """
    print(f"Extracting diagnoses for {len(patient_ids)} patients...")

    # Extract raw diagnoses
    raw_df = extract_diagnoses_for_patients(dia_file, patient_ids)
    print(f"  Extracted {len(raw_df)} raw diagnosis records")

    # Filter excluded codes
    filtered_df = filter_excluded_codes(raw_df)
    print(f"  After filtering: {len(filtered_df)} records")

    # Build Layer 1
    print("Building Layer 1 (canonical records)...")
    layer1_df = build_layer1(filtered_df, pe_times, min_days, max_days)
    print(f"  Layer 1: {len(layer1_df)} records")

    # Save Layer 1
    layer1_path = output_path / "layer1"
    layer1_path.mkdir(parents=True, exist_ok=True)
    layer1_df.to_parquet(layer1_path / "canonical_diagnoses.parquet", index=False)
    print(f"  Saved to {layer1_path / 'canonical_diagnoses.parquet'}")

    # Build Layer 2
    print("Building Layer 2 (comorbidity scores)...")
    layer2_df = build_layer2_comorbidity_scores(layer1_df)
    print(f"  Layer 2: {len(layer2_df)} patients with scores")

    # Save Layer 2
    save_layer2(layer2_df, output_path / "layer2")
    print(f"  Saved to {output_path / 'layer2' / 'comorbidity_scores.parquet'}")

    print("Pipeline complete!")


def main():
    """Main entry point for CLI."""
    import argparse
    from config.diagnosis_config import DATA_PATH, PATIENT_TIMELINES_PATH, OUTPUT_PATH

    parser = argparse.ArgumentParser(description="Build diagnosis layers")
    parser.add_argument("--test", action="store_true", help="Test mode (10 patients)")
    parser.add_argument("--n", type=int, default=10, help="Number of patients for test mode")
    args = parser.parse_args()

    # Load PE times
    print(f"Loading PE times from {PATIENT_TIMELINES_PATH}...")
    pe_times = load_pe_times(PATIENT_TIMELINES_PATH)
    print(f"  Loaded {len(pe_times)} patients with PE times")

    # Get patient IDs
    patient_ids = set(pe_times.keys())

    if args.test:
        patient_ids = set(list(patient_ids)[:args.n])
        print(f"Test mode: processing {len(patient_ids)} patients")

    # Run pipeline
    run_pipeline(
        dia_file=DATA_PATH,
        patient_ids=patient_ids,
        pe_times=pe_times,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
