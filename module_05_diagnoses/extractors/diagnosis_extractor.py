"""Extract diagnoses from RPDR Dia.txt file."""
from typing import Dict, Set, Union, TextIO
from pathlib import Path
import pandas as pd
from processing.icd_parser import detect_icd_version, normalize_icd_code

# RPDR column names
DIA_COLUMNS = [
    "EMPI", "EPIC_PMRN", "MRN_Type", "MRN", "Date", "Diagnosis_Name",
    "Code_Type", "Code", "Diagnosis_Flag", "Provider", "Clinic",
    "Hospital", "Inpatient_Outpatient", "Encounter_number"
]

# Exclusion prefixes
EXCLUDE_ICD10_PREFIXES = ['Z00', 'Z01', 'Z02', 'Z03', 'Z04', 'Z05', 'Z06',
                          'Z07', 'Z08', 'Z09', 'Z10', 'Z11', 'Z12', 'Z13']
EXCLUDE_ICD9_PREFIXES = ['V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76',
                         'V77', 'V78', 'V79', 'V80', 'V81', 'V82']


def parse_diagnosis_line(line: str) -> Dict[str, str]:
    """Parse a single line from Dia.txt.

    Args:
        line: Pipe-delimited line from Dia.txt

    Returns:
        Dictionary with column names as keys
    """
    parts = line.strip().split("|")
    result = {}
    for i, col in enumerate(DIA_COLUMNS):
        result[col] = parts[i] if i < len(parts) else ""
    return result


def filter_excluded_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove administrative/screening codes.

    Args:
        df: DataFrame with 'Code' and 'icd_version' columns

    Returns:
        Filtered DataFrame
    """
    def should_exclude(code: str, version: str) -> bool:
        code = str(code).upper()
        if version == "10":
            return any(code.startswith(p) for p in EXCLUDE_ICD10_PREFIXES)
        elif version == "9":
            return any(code.startswith(p) for p in EXCLUDE_ICD9_PREFIXES)
        return False

    mask = ~df.apply(lambda r: should_exclude(r["Code"], r.get("icd_version", "10")), axis=1)
    return df[mask].reset_index(drop=True)


def extract_diagnoses_for_patients(
    file_handle: Union[str, Path, TextIO],
    patient_ids: Set[str],
    chunk_size: int = 100000
) -> pd.DataFrame:
    """Extract diagnoses for specified patients.

    Args:
        file_handle: Path to Dia.txt or file handle
        patient_ids: Set of EMPI values to extract
        chunk_size: Rows per chunk for memory efficiency

    Returns:
        DataFrame with diagnoses for specified patients
    """
    # Convert patient_ids to strings for comparison
    patient_ids = {str(p) for p in patient_ids}

    chunks = []

    # Handle file path or file handle
    if isinstance(file_handle, (str, Path)):
        f = open(file_handle, "r")
        should_close = True
    else:
        f = file_handle
        should_close = False

    try:
        # Skip header
        header = f.readline()

        batch = []
        for line in f:
            parts = line.strip().split("|")
            if parts and parts[0] in patient_ids:
                batch.append(parse_diagnosis_line(line))

            if len(batch) >= chunk_size:
                chunks.append(pd.DataFrame(batch))
                batch = []

        if batch:
            chunks.append(pd.DataFrame(batch))

    finally:
        if should_close:
            f.close()

    if not chunks:
        return pd.DataFrame(columns=DIA_COLUMNS)

    df = pd.concat(chunks, ignore_index=True)

    # Parse dates
    df["diagnosis_date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    # Normalize codes and detect version
    df["icd_code"] = df["Code"].apply(normalize_icd_code)
    df["icd_version"] = df["Code"].apply(detect_icd_version)

    # Map Code_Type to version if available
    df.loc[df["Code_Type"].str.upper() == "ICD9", "icd_version"] = "9"
    df.loc[df["Code_Type"].str.upper() == "ICD10", "icd_version"] = "10"

    return df
