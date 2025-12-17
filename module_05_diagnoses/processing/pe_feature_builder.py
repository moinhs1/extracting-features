"""PE (Pulmonary Embolism) feature extraction builder.

This module provides utilities for extracting PE-related features from diagnosis codes.
"""
from typing import Dict


def code_matches_category(code: str, category_codes: Dict[str, list], version: str) -> bool:
    """Check if ICD code matches a category definition.

    Uses prefix matching to determine if a given ICD code belongs to a category.
    For example, code "I26.99" matches category with prefix "I26".

    Args:
        code: ICD code (e.g., "I26.99")
        category_codes: Dict with "icd10" and "icd9" lists of prefixes
        version: "9" or "10" indicating ICD version

    Returns:
        True if code matches any prefix in the category for the given version

    Example:
        >>> category = {"icd10": ["I26"], "icd9": ["415"]}
        >>> code_matches_category("I26.99", category, "10")
        True
        >>> code_matches_category("415.19", category, "9")
        True
        >>> code_matches_category("I50.9", category, "10")
        False
    """
    # Get the appropriate code list for this ICD version
    code_list = category_codes.get(f"icd{version}", [])

    # Normalize code for comparison
    code = str(code).upper()

    # Check if code starts with any prefix in the category
    for prefix in code_list:
        if code.startswith(prefix.upper()):
            return True

    return False
