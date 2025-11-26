"""
Submodule 3.1: Structured Vitals Extractor (Phy.txt)
====================================================

Extracts vital signs from the structured Phy.txt file.
"""
import re
from typing import Tuple, Optional, List, Dict, Any

import pandas as pd

from pathlib import Path
from tqdm import tqdm

from module_3_vitals_processing.config.vitals_config import (
    VITAL_CONCEPTS, PHY_COLUMNS, CHUNK_SIZE
)


def parse_blood_pressure(bp_string: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse blood pressure string like '130/77' into (systolic, diastolic).

    Args:
        bp_string: Blood pressure string in format 'SBP/DBP'

    Returns:
        Tuple of (systolic, diastolic) as floats, or (None, None) if invalid
    """
    if bp_string is None or not isinstance(bp_string, str):
        return None, None

    bp_string = bp_string.strip()
    if not bp_string:
        return None, None

    # Pattern: digits / digits (with optional spaces)
    pattern = r'^(\d+)\s*/\s*(\d+)$'
    match = re.match(pattern, bp_string)

    if not match:
        return None, None

    try:
        sbp = float(match.group(1))
        dbp = float(match.group(2))
        return sbp, dbp
    except (ValueError, TypeError):
        return None, None


def map_concept_to_canonical(concept_name: Optional[str]) -> Optional[str]:
    """
    Map Phy.txt Concept_Name to canonical vital sign name.

    Args:
        concept_name: The Concept_Name from Phy.txt

    Returns:
        Canonical vital name (HR, SBP, DBP, etc.) or None if not a vital
    """
    if concept_name is None:
        return None

    return VITAL_CONCEPTS.get(concept_name)


def parse_result_value(result: Optional[str]) -> Optional[float]:
    """
    Parse Result field to numeric value.

    Args:
        result: The Result string from Phy.txt

    Returns:
        Float value or None if not parseable
    """
    if result is None or not isinstance(result, str):
        return None

    result = result.strip()
    if not result:
        return None

    # Remove common prefixes like > or <
    result = result.lstrip('<>').strip()

    try:
        return float(result)
    except (ValueError, TypeError):
        return None


def process_vital_row(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Process a single row from Phy.txt and extract vital sign records.

    Args:
        row: A pandas Series representing one row from Phy.txt

    Returns:
        List of vital sign dictionaries. Blood Pressure rows produce 2 records.
        Non-vital rows or invalid values produce empty list.
    """
    concept_name = row.get('Concept_Name')
    canonical = map_concept_to_canonical(concept_name)

    if canonical is None:
        return []

    empi = str(row.get('EMPI', ''))
    date_str = row.get('Date', '')
    result = row.get('Result', '')
    units = row.get('Units', '')
    encounter_type = row.get('Inpatient_Outpatient', '')
    encounter_number = row.get('Encounter_number', '')

    base_record = {
        'EMPI': empi,
        'date_str': date_str,
        'units': units,
        'source': 'phy',
        'encounter_type': encounter_type,
        'encounter_number': encounter_number,
    }

    # Handle Blood Pressure (combined SBP/DBP)
    if canonical == 'BP':
        sbp, dbp = parse_blood_pressure(result)
        if sbp is None or dbp is None:
            return []

        return [
            {**base_record, 'vital_type': 'SBP', 'value': sbp},
            {**base_record, 'vital_type': 'DBP', 'value': dbp},
        ]

    # Handle regular vitals
    value = parse_result_value(result)
    if value is None:
        return []

    return [{**base_record, 'vital_type': canonical, 'value': value}]


def extract_phy_vitals(
    phy_file: str,
    output_file: str,
    chunk_size: int = CHUNK_SIZE,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Extract vital signs from Phy.txt structured file.

    Args:
        phy_file: Path to Phy.txt input file
        output_file: Path to output parquet file
        chunk_size: Number of rows to process per chunk
        show_progress: Whether to show progress bar

    Returns:
        DataFrame with extracted vitals
    """
    phy_path = Path(phy_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get vital concept names for filtering
    vital_concept_names = set(VITAL_CONCEPTS.keys())

    all_records = []

    # Count total lines for progress bar
    if show_progress:
        with open(phy_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header

    # Process in chunks
    chunks = pd.read_csv(
        phy_path,
        sep='|',
        names=PHY_COLUMNS,
        header=0,
        dtype=str,
        chunksize=chunk_size,
        low_memory=False
    )

    if show_progress:
        chunks = tqdm(chunks, desc="Processing Phy.txt", total=total_lines // chunk_size + 1)

    for chunk in chunks:
        # Filter to only vital sign concepts
        mask = chunk['Concept_Name'].isin(vital_concept_names)
        vital_chunk = chunk[mask]

        # Process each row
        for _, row in vital_chunk.iterrows():
            records = process_vital_row(row)
            all_records.extend(records)

    # Create DataFrame
    if not all_records:
        # Return empty DataFrame with correct schema
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'vital_type', 'value', 'units',
            'source', 'encounter_type', 'encounter_number'
        ])
    else:
        df = pd.DataFrame(all_records)

        # Parse dates
        df['timestamp'] = pd.to_datetime(df['date_str'], format='%m/%d/%Y', errors='coerce')
        df = df.drop(columns=['date_str'])

        # Reorder columns
        df = df[[
            'EMPI', 'timestamp', 'vital_type', 'value', 'units',
            'source', 'encounter_type', 'encounter_number'
        ]]

    # Save to parquet
    df.to_parquet(output_path, index=False)

    return df
