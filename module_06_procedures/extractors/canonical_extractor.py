"""
Canonical Procedure Record Extractor
====================================

Extracts procedure records from RPDR Prc.txt, joins with patient cohort,
computes temporal alignment and flags, outputs bronze parquet.

Layer 1 of the procedure encoding pipeline.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Set, Optional, Iterator
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    PRC_FILE,
    PATIENT_TIMELINES_PKL,
    BRONZE_DIR,
    TEMPORAL_CONFIG,
    LAYER_CONFIG,
)


# =============================================================================
# PRC.TXT LOADING
# =============================================================================

def load_prc_chunk(
    n_rows: Optional[int] = None,
    skip_rows: int = 0,
    filepath: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load a chunk of Prc.txt.

    Args:
        n_rows: Number of rows to load (None for all)
        skip_rows: Number of rows to skip (for chunked loading)
        filepath: Override default Prc.txt path

    Returns:
        DataFrame with procedure records
    """
    filepath = filepath or PRC_FILE

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
            'Procedure_Name': str,
            'Code_Type': str,
            'Code': str,
            'Quantity': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
        },
        parse_dates=['Date'],
        low_memory=False,
    )

    return df


def iter_prc_chunks(
    chunk_size: int = 1_000_000,
    filepath: Optional[Path] = None
) -> Iterator[pd.DataFrame]:
    """
    Iterate over Prc.txt in chunks.

    Args:
        chunk_size: Rows per chunk
        filepath: Override default Prc.txt path

    Yields:
        DataFrame chunks
    """
    filepath = filepath or PRC_FILE

    reader = pd.read_csv(
        filepath,
        sep='|',
        chunksize=chunk_size,
        dtype={
            'EMPI': str,
            'EPIC_PMRN': str,
            'MRN_Type': str,
            'MRN': str,
            'Procedure_Name': str,
            'Code_Type': str,
            'Code': str,
            'Quantity': str,
            'Provider': str,
            'Clinic': str,
            'Hospital': str,
            'Inpatient_Outpatient': str,
            'Encounter_number': str,
        },
        parse_dates=['Date'],
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
    Build mapping of EMPI -> PE Time Zero.

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
    Filter procedure DataFrame to cohort patients only.

    Args:
        df: Procedure DataFrame
        cohort_empis: Set of EMPI values in cohort

    Returns:
        Filtered DataFrame
    """
    return df[df['EMPI'].isin(cohort_empis)].copy()


# =============================================================================
# TIME ALIGNMENT
# =============================================================================

def compute_hours_from_pe(
    df: pd.DataFrame,
    time_zero_map: Dict[str, pd.Timestamp]
) -> pd.DataFrame:
    """
    Compute hours relative to PE Time Zero for each procedure.

    Args:
        df: Procedure DataFrame with EMPI and Date
        time_zero_map: Dictionary mapping EMPI -> Time Zero

    Returns:
        DataFrame with hours_from_pe column added
    """
    df = df.copy()

    # Map EMPI to Time Zero
    df['time_zero'] = df['EMPI'].map(time_zero_map)

    # Date column is date only, assume midnight
    proc_datetime = pd.to_datetime(df['Date'])

    # Compute hours difference
    df['hours_from_pe'] = (proc_datetime - df['time_zero']).dt.total_seconds() / 3600

    # Also compute days for convenience
    df['days_from_pe'] = (df['hours_from_pe'] / 24).astype(int)

    # Drop temporary column
    df = df.drop(columns=['time_zero'])

    return df


# =============================================================================
# TEMPORAL FLAGS
# =============================================================================

def compute_temporal_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 7 temporal category flags based on hours_from_pe.

    Args:
        df: DataFrame with hours_from_pe column

    Returns:
        DataFrame with 7 temporal flag columns added
    """
    df = df.copy()
    hours = df['hours_from_pe']

    # Lifetime history: before -720h (>30 days pre-PE)
    df['is_lifetime_history'] = hours < -720

    # Remote antecedent: -720h to -720h (actually: between lifetime and provoking)
    # This overlaps with provoking, so we define it as: -infinity to -720h
    # For clarity, remote_antecedent is same as lifetime_history for now
    df['is_remote_antecedent'] = hours < -720

    # Provoking window: -720h to 0h (1-30 days pre-PE)
    df['is_provoking_window'] = (hours >= -720) & (hours < 0)

    # Diagnostic workup: -24h to +24h
    df['is_diagnostic_workup'] = (hours >= -24) & (hours <= 24)

    # Initial treatment: 0h to +72h
    df['is_initial_treatment'] = (hours >= 0) & (hours <= 72)

    # Escalation: >72h during hospitalization (we'll use 72-720h as proxy)
    df['is_escalation'] = (hours > 72) & (hours <= 720)

    # Post discharge: after +720h (>30 days post-PE)
    df['is_post_discharge'] = hours > 720

    return df


# =============================================================================
# SCHEMA TRANSFORMATION
# =============================================================================

def transform_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw procedure data to canonical schema.

    Args:
        df: Raw procedure DataFrame with parsed columns

    Returns:
        DataFrame with canonical schema
    """
    canonical = pd.DataFrame({
        'empi': df['EMPI'],
        'encounter_id': df['Encounter_number'],
        'procedure_datetime': pd.to_datetime(df['Date'], errors='coerce'),
        'procedure_date': pd.to_datetime(df['Date'], errors='coerce').dt.date,
        'procedure_name': df['Procedure_Name'],
        'code_type': df['Code_Type'],
        'code': df['Code'],
        'quantity': pd.to_numeric(df['Quantity'], errors='coerce'),
        'provider': df['Provider'],
        'clinic': df['Clinic'],
        'hospital': df['Hospital'],
        'inpatient': df['Inpatient_Outpatient'].str.lower() == 'inpatient',
        'hours_from_pe': df['hours_from_pe'],
        'days_from_pe': df.get('days_from_pe'),
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
    Main extraction pipeline: Prc.txt -> Bronze parquet.

    Args:
        test_mode: If True, only process test_n_rows
        test_n_rows: Number of rows for test mode
        output_path: Override output path

    Returns:
        Canonical records DataFrame
    """
    print("=" * 60)
    print("Layer 1: Canonical Procedure Extraction")
    print("=" * 60)

    # Load patient timelines
    print("\n1. Loading patient timelines...")
    timelines = load_patient_timelines()
    cohort_empis = get_cohort_empis(timelines)
    time_zero_map = get_time_zero_map(timelines)
    print(f"   Cohort size: {len(cohort_empis)} patients")

    # Process procedures
    if test_mode:
        print(f"\n2. Loading Prc.txt (test mode: {test_n_rows} rows)...")
        df = load_prc_chunk(n_rows=test_n_rows)
        chunks = [df]
    else:
        print("\n2. Loading Prc.txt in chunks...")
        chunks = iter_prc_chunks(chunk_size=LAYER_CONFIG.chunk_size)

    all_records = []
    total_raw = 0
    total_cohort = 0

    for i, chunk in enumerate(chunks):
        total_raw += len(chunk)

        # Filter to cohort
        chunk = filter_to_cohort(chunk, cohort_empis)
        total_cohort += len(chunk)

        if len(chunk) == 0:
            continue

        # Compute time alignment
        chunk = compute_hours_from_pe(chunk, time_zero_map)

        # Compute temporal flags
        chunk = compute_temporal_flags(chunk)

        # Transform to canonical schema
        canonical = transform_to_canonical(chunk)

        # Add temporal flags to canonical
        for flag in ['is_lifetime_history', 'is_remote_antecedent', 'is_provoking_window',
                     'is_diagnostic_workup', 'is_initial_treatment', 'is_escalation',
                     'is_post_discharge']:
            canonical[flag] = chunk[flag].values

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
    print(f"   Final records: {len(result):,}")

    if len(result) > 0:
        # Code type distribution
        print("\n   Code type distribution:")
        code_dist = result['code_type'].value_counts(normalize=True)
        for code_type, pct in code_dist.head(5).items():
            print(f"     {code_type}: {pct:.1%}")

        # Patient coverage
        patients_with_procs = result['empi'].nunique()
        print(f"\n   Patients with procedures: {patients_with_procs}")

        # Temporal distribution
        print("\n   Temporal distribution:")
        print(f"     Lifetime history: {result['is_lifetime_history'].sum():,}")
        print(f"     Provoking window: {result['is_provoking_window'].sum():,}")
        print(f"     Diagnostic workup: {result['is_diagnostic_workup'].sum():,}")
        print(f"     Initial treatment: {result['is_initial_treatment'].sum():,}")

    # Save output
    if output_path is None:
        BRONZE_DIR.mkdir(parents=True, exist_ok=True)
        filename = "canonical_procedures_test.parquet" if test_mode else "canonical_procedures.parquet"
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

    parser = argparse.ArgumentParser(description="Extract canonical procedure records")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--n', type=int, default=10000, help='Rows for test mode')
    args = parser.parse_args()

    extract_canonical_records(test_mode=args.test, test_n_rows=args.n)
