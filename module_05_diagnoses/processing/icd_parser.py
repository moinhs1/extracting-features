"""ICD code parsing and version detection."""
from typing import Optional
import re

# ICD-9 patterns
ICD9_PATTERN = re.compile(r'^[VvEe]?\d{2,3}\.?\d*$')

# PE diagnosis codes
PE_ICD9_PREFIXES = ['415.1']
PE_ICD10_PREFIXES = ['I26']


def detect_icd_version(code: Optional[str]) -> str:
    """Detect whether code is ICD-9 or ICD-10.

    Args:
        code: ICD diagnosis code

    Returns:
        '9' for ICD-9, '10' for ICD-10, 'unknown' if cannot determine
    """
    if not code:
        return "unknown"

    code = str(code).strip().upper()
    if not code:
        return "unknown"

    first_char = code[0]

    # ICD-10 codes start with a letter (A-Z except E and V which overlap with ICD-9)
    # ICD-9 V codes: V01-V91, ICD-9 E codes: E800-E999
    # ICD-10: letter followed by number

    if first_char.isdigit():
        # Pure numeric start = ICD-9
        return "9"
    elif first_char == 'V':
        # V followed by 2-digit number = ICD-9, otherwise ICD-10
        if len(code) >= 2 and code[1].isdigit():
            return "9"
        return "10"
    elif first_char == 'E':
        # E followed by 3-digit 8xx or 9xx = ICD-9 E-code, otherwise ICD-10
        if len(code) >= 4 and code[1:4].isdigit():
            num = int(code[1:4])
            if 800 <= num <= 999:
                return "9"
        return "10"
    elif first_char.isalpha():
        # Other letters = ICD-10
        return "10"

    return "unknown"


def normalize_icd_code(code: Optional[str]) -> str:
    """Normalize ICD code for consistent comparison.

    Args:
        code: Raw ICD code

    Returns:
        Normalized code (uppercase, trimmed)
    """
    if code is None:
        return ""
    return str(code).strip().upper()


def is_pe_diagnosis(code: str, version: str) -> bool:
    """Check if code is a pulmonary embolism diagnosis.

    Args:
        code: Normalized ICD code
        version: '9' or '10'

    Returns:
        True if PE diagnosis
    """
    code = normalize_icd_code(code)
    if not code:
        return False

    if version == "9":
        return any(code.startswith(prefix) for prefix in PE_ICD9_PREFIXES)
    elif version == "10":
        return any(code.startswith(prefix) for prefix in PE_ICD10_PREFIXES)

    return False
