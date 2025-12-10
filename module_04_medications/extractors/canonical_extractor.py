"""
Canonical Medication Record Extractor
=====================================

Extracts medication records from RPDR Med.txt, joins with patient cohort,
computes temporal alignment, and outputs bronze parquet.

Layer 1 of the medication encoding pipeline.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Set, Optional, Iterator
from datetime import datetime
import sys

# Add parent to path for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    MED_FILE,
    PATIENT_TIMELINES_PKL,
    BRONZE_DIR,
    TEMPORAL_CONFIG,
    LAYER_CONFIG,
)
from extractors.dose_parser import parse_medication_string


# =============================================================================
# MED.TXT LOADING
# =============================================================================

def load_med_chunk(
    n_rows: Optional[int] = None,
    skip_rows: int = 0,
    filepath: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a chunk of Med.txt.

    Args:
        n_rows: Number of rows to load (None for all)
        skip_rows: Number of rows to skip (for chunked loading)
        filepath: Override default Med.txt path

    Returns:
        DataFrame with medication records
    """
    filepath = filepath or MED_FILE

    df = pd.read_csv(
        filepath,
        sep='|',
        nrows=n_rows,
        skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None,
        dtype={
            'EMPI': str,
            'EPIC_PMRN': str,
            'MRN_Type': str,
            'MRN': str,
            'Medication': str,
            'Code_Type': str,
            'Code': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
            'Additional_Info': str,
        },
        parse_dates=['Medication_Date'],
        low_memory=False,
    )

    return df


def iter_med_chunks(
    chunk_size: int = 1_000_000,
    filepath: Optional[Path] = None
) -> Iterator[pd.DataFrame]:
    """
    Iterate over Med.txt in chunks.

    Args:
        chunk_size: Rows per chunk
        filepath: Override default Med.txt path

    Yields:
        DataFrame chunks
    """
    filepath = filepath or MED_FILE

    reader = pd.read_csv(
        filepath,
        sep='|',
        chunksize=chunk_size,
        dtype={
            'EMPI': str,
            'EPIC_PMRN': str,
            'MRN_Type': str,
            'MRN': str,
            'Medication': str,
            'Code_Type': str,
            'Code': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
            'Additional_Info': str,
        },
        parse_dates=['Medication_Date'],
        low_memory=False,
    )

    for chunk in reader:
        yield chunk


# =============================================================================
# COHORT INTEGRATION
# =============================================================================

class _PatientTimelineUnpickler(pickle.Unpickler):
    """Custom unpickler to handle PatientTimeline class from __main__."""

    def find_class(self, module, name):
        if name == 'PatientTimeline':
            # Import and return the PatientTimeline class regardless of original module
            module1_path = str(Path(__file__).parent.parent.parent / "module_1_core_infrastructure")
            if module1_path not in sys.path:
                sys.path.insert(0, module1_path)
            import module_01_core_infrastructure
            return module_01_core_infrastructure.PatientTimeline
        return super().find_class(module, name)


def load_patient_timelines() -> Dict:
    """
    Load patient timelines from Module 1.

    Returns:
        Dictionary mapping EMPI -> PatientTimeline object
    """
    with open(PATIENT_TIMELINES_PKL, 'rb') as f:
        timelines = _PatientTimelineUnpickler(f).load()

    return timelines


def get_cohort_empis(timelines: Dict) -> Set[str]:
    """
    Get set of EMPI values for PE cohort.

    Args:
        timelines: Patient timelines dictionary

    Returns:
        Set of EMPI strings
    """
    return set(timelines.keys())


def get_time_zero_map(timelines: Dict) -> Dict[str, pd.Timestamp]:
    """
    Build mapping of EMPI -> Time Zero.

    Args:
        timelines: Patient timelines dictionary

    Returns:
        Dictionary mapping EMPI -> Time Zero timestamp
    """
    return {
        empi: pd.Timestamp(timeline.time_zero)
        for empi, timeline in timelines.items()
    }


def filter_to_cohort(df: pd.DataFrame, cohort_empis: Set[str]) -> pd.DataFrame:
    """
    Filter medication DataFrame to cohort patients only.

    Args:
        df: Medication DataFrame
        cohort_empis: Set of EMPI values in cohort

    Returns:
        Filtered DataFrame
    """
    return df[df['EMPI'].isin(cohort_empis)].copy()


# =============================================================================
# TIME ALIGNMENT
# =============================================================================

def compute_hours_from_t0(
    df: pd.DataFrame,
    time_zero_map: Dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Compute hours relative to Time Zero for each medication.

    Args:
        df: Medication DataFrame with EMPI and Medication_Date
        time_zero_map: Dictionary mapping EMPI -> Time Zero

    Returns:
        DataFrame with hours_from_t0 column added
    """
    df = df.copy()

    # Map EMPI to Time Zero
    df['time_zero'] = df['EMPI'].map(time_zero_map)

    # Medication_Date is date only (no time), assume midnight
    med_datetime = pd.to_datetime(df['Medication_Date'])

    # Compute hours difference
    df['hours_from_t0'] = (med_datetime - df['time_zero']).dt.total_seconds() / 3600

    # Drop temporary column
    df = df.drop(columns=['time_zero'])

    return df


def filter_study_window(
    df: pd.DataFrame,
    window_start_hours: int = None,
    window_end_hours: int = None
) -> pd.DataFrame:
    """
    Filter to study window relative to Time Zero.

    Args:
        df: DataFrame with hours_from_t0 column
        window_start_hours: Start of window (default from config)
        window_end_hours: End of window (default from config)

    Returns:
        Filtered DataFrame
    """
    if window_start_hours is None:
        window_start_hours = TEMPORAL_CONFIG.study_window_start
    if window_end_hours is None:
        window_end_hours = TEMPORAL_CONFIG.study_window_end

    mask = (
        (df['hours_from_t0'] >= window_start_hours) &
        (df['hours_from_t0'] <= window_end_hours)
    )

    return df[mask].copy()


# =============================================================================
# PARSING & TRANSFORMATION
# =============================================================================

def parse_medications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply dose parsing to all medication strings.

    Args:
        df: DataFrame with Medication column

    Returns:
        DataFrame with parsed columns added
    """
    df = df.copy()

    # Parse each medication string
    parsed = df['Medication'].apply(parse_medication_string)
    parsed_df = pd.DataFrame(parsed.tolist())

    # Set parsed_df index to match df index for proper alignment
    parsed_df.index = df.index

    # Add parsed columns
    for col in parsed_df.columns:
        df[col] = parsed_df[col]

    return df


def transform_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw medication data to canonical schema.

    Args:
        df: Raw medication DataFrame with parsed columns

    Returns:
        DataFrame with canonical schema
    """
    canonical = pd.DataFrame({
        'empi': df['EMPI'],
        'encounter_id': df['Encounter_number'],
        'medication_date': df['Medication_Date'],
        'hours_from_t0': df['hours_from_t0'],
        'original_string': df['Medication'],
        'code_type': df['Code_Type'],
        'code': df['Code'],
        'quantity': pd.to_numeric(df['Quantity'], errors='coerce'),
        'inpatient': df['Inpatient_Outpatient'].str.lower() == 'inpatient',
        'provider': df['Provider'],
        'clinic': df['Clinic'],
        'hospital': df['Hospital'],
        # Parsed columns
        'parsed_name': df['parsed_name'],
        'parsed_dose_value': df['parsed_dose_value'],
        'parsed_dose_unit': df['parsed_dose_unit'],
        'parsed_route': df['parsed_route'],
        'parsed_frequency': df['parsed_frequency'],
        'parse_method': df['parse_method'],
        'parse_confidence': df['parse_confidence'],
    })

    return canonical


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================

def extract_canonical_records(
    test_mode: bool = False,
    test_n_rows: int = 10000,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main extraction pipeline: Med.txt -> Bronze parquet.

    Args:
        test_mode: If True, only process test_n_rows
        test_n_rows: Number of rows for test mode
        output_path: Override output path

    Returns:
        Canonical records DataFrame
    """
    print("=" * 60)
    print("Layer 1: Canonical Medication Extraction")
    print("=" * 60)

    # Load patient timelines
    print("\n1. Loading patient timelines...")
    timelines = load_patient_timelines()
    cohort_empis = get_cohort_empis(timelines)
    time_zero_map = get_time_zero_map(timelines)
    print(f"   Cohort size: {len(cohort_empis)} patients")

    # Process medications
    if test_mode:
        print(f"\n2. Loading Med.txt (test mode: {test_n_rows} rows)...")
        df = load_med_chunk(n_rows=test_n_rows)
        chunks = [df]
    else:
        print("\n2. Loading Med.txt in chunks...")
        chunks = iter_med_chunks(chunk_size=LAYER_CONFIG.chunk_size)

    all_records = []
    total_raw = 0
    total_cohort = 0
    total_window = 0

    for i, chunk in enumerate(chunks):
        total_raw += len(chunk)

        # Filter to cohort
        chunk = filter_to_cohort(chunk, cohort_empis)
        total_cohort += len(chunk)

        if len(chunk) == 0:
            continue

        # Compute time alignment
        chunk = compute_hours_from_t0(chunk, time_zero_map)

        # Filter to study window
        chunk = filter_study_window(chunk)
        total_window += len(chunk)

        if len(chunk) == 0:
            continue

        # Parse medications
        chunk = parse_medications(chunk)

        # Transform to canonical schema
        canonical = transform_to_canonical(chunk)

        all_records.append(canonical)

        if not test_mode:
            print(f"   Chunk {i+1}: {len(canonical):,} records")

    # Combine all chunks
    print("\n3. Combining records...")
    if all_records:
        result = pd.concat(all_records, ignore_index=True)
    else:
        result = pd.DataFrame()

    # Summary statistics
    print("\n" + "=" * 60)
    print("Extraction Summary")
    print("=" * 60)
    print(f"   Total raw records: {total_raw:,}")
    print(f"   After cohort filter: {total_cohort:,}")
    print(f"   After window filter: {total_window:,}")
    print(f"   Final records: {len(result):,}")

    if len(result) > 0:
        # Parsing stats
        parsed_count = (result['parse_method'] == 'regex').sum()
        parse_rate = parsed_count / len(result) * 100
        print(f"\n   Dose parsing success: {parse_rate:.1f}%")

        # Patient coverage
        patients_with_meds = result['empi'].nunique()
        print(f"   Patients with medications: {patients_with_meds}")

    # Save output
    if output_path is None:
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        filename = "canonical_records_test.parquet" if test_mode else "canonical_records.parquet"
        output_path = BRONZE_DIR / filename

    print(f"\n4. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Layer 1 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract canonical medication records")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--n', type=int, default=10000, help='Rows for test mode')
    args = parser.parse_args()

    extract_canonical_records(test_mode=args.test, test_n_rows=args.n)
