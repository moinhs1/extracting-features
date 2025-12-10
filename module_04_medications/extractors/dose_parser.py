"""
Dose Parser for RPDR Medication Strings
=======================================

Extracts dose value, unit, route, and frequency from free-text medication strings.
Uses regex patterns defined in config/dose_patterns.yaml.
"""

import re
from typing import Dict, Optional, Any
from pathlib import Path
import yaml


# Load patterns from YAML
CONFIG_DIR = Path(__file__).parent.parent / "config"
DOSE_PATTERNS_FILE = CONFIG_DIR / "dose_patterns.yaml"

_patterns_cache: Optional[Dict] = None


def _load_patterns() -> Dict:
    """Load and cache dose patterns from YAML."""
    global _patterns_cache
    if _patterns_cache is None:
        with open(DOSE_PATTERNS_FILE, 'r') as f:
            _patterns_cache = yaml.safe_load(f)
    return _patterns_cache


# Unit normalization mapping
UNIT_ALIASES = {
    'mg': 'mg',
    'milligram': 'mg',
    'milligrams': 'mg',
    'mgs': 'mg',
    'mcg': 'mcg',
    'ug': 'mcg',
    'microgram': 'mcg',
    'micrograms': 'mcg',
    'μg': 'mcg',
    'g': 'g',
    'gm': 'g',
    'gram': 'g',
    'grams': 'g',
    'gms': 'g',
    'ml': 'ml',
    'milliliter': 'ml',
    'milliliters': 'ml',
    'mls': 'ml',
    'cc': 'ml',
    'l': 'l',
    'liter': 'l',
    'liters': 'l',
    'unit': 'units',
    'units': 'units',
    'u': 'units',
    'iu': 'units',
    'meq': 'meq',
    'milliequivalent': 'meq',
    'milliequivalents': 'meq',
    'mmol': 'mmol',
    'millimole': 'mmol',
    'millimoles': 'mmol',
}


def normalize_unit(unit: str) -> str:
    """Normalize unit string to canonical form."""
    if unit is None:
        return None
    return UNIT_ALIASES.get(unit.lower().strip(), unit.lower().strip())


def extract_dose(medication_string: str) -> Dict[str, Any]:
    """
    Extract dose information from a medication string.

    Args:
        medication_string: Raw RPDR medication text

    Returns:
        Dictionary with keys:
            - dose_value: float or None
            - dose_unit: str or None
            - parse_method: str ('regex' or 'failed')
            - parse_confidence: float (0-1)
    """
    if not medication_string or not isinstance(medication_string, str):
        return {
            'dose_value': None,
            'dose_unit': None,
            'parse_method': 'failed',
            'parse_confidence': 0.0,
        }

    text = medication_string.lower().strip()

    # Pattern 1: Standard dose with unit (e.g., "325mg", "100 mcg", "5000 units")
    # Most common pattern - try first
    standard_pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|gm|ml|unit|units|u|iu|meq|mmol)\b'
    match = re.search(standard_pattern, text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = normalize_unit(match.group(2))
        return {
            'dose_value': value,
            'dose_unit': unit,
            'parse_method': 'regex',
            'parse_confidence': 0.9,
        }

    # Pattern 2: Concentration format (e.g., "2 mg/ml", "5000 unit/ml")
    conc_pattern = r'(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|unit|units|u)\s*/\s*(?:\d+\s*)?(ml|l)\b'
    match = re.search(conc_pattern, text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = normalize_unit(match.group(2))
        return {
            'dose_value': value,
            'dose_unit': unit,
            'parse_method': 'regex',
            'parse_confidence': 0.85,
        }

    # Pattern 3: Percentage (e.g., "0.9%", "5%")
    pct_pattern = r'(\d+(?:\.\d+)?)\s*%'
    match = re.search(pct_pattern, text)
    if match:
        value = float(match.group(1))
        return {
            'dose_value': value,
            'dose_unit': 'percent',
            'parse_method': 'regex',
            'parse_confidence': 0.8,
        }

    # No dose found
    return {
        'dose_value': None,
        'dose_unit': None,
        'parse_method': 'failed',
        'parse_confidence': 0.0,
    }


def extract_route(medication_string: str) -> Optional[str]:
    """
    Extract administration route from medication string.

    Returns: One of 'IV', 'PO', 'SC', 'IM', 'topical', 'inhaled', 'rectal', 'ophthalmic', or None
    """
    if not medication_string:
        return None

    text = medication_string.lower()

    # IV patterns
    if any(p in text for p in ['intravenous', ' iv ', 'ivpb', 'iv push', 'infusion']):
        return 'IV'
    if re.search(r'\biv\b', text):
        return 'IV'

    # SC patterns
    if any(p in text for p in ['subcutaneous', 'subcut', 'subq']):
        return 'SC'
    if re.search(r'\b(sc|sq)\b', text):
        return 'SC'

    # IM patterns
    if 'intramuscular' in text:
        return 'IM'
    if re.search(r'\bim\b', text):
        return 'IM'

    # PO patterns (check after IV/IM to avoid false positives)
    if any(p in text for p in ['oral', 'by mouth', 'tablet', 'capsule', 'tab ', 'cap ']):
        return 'PO'
    if re.search(r'\bpo\b', text):
        return 'PO'

    # Topical
    if any(p in text for p in ['topical', 'cream', 'ointment', 'gel', 'lotion', 'patch']):
        return 'topical'

    # Inhaled
    if any(p in text for p in ['inhaler', 'nebulizer', 'inhalation', 'metered dose']):
        return 'inhaled'

    # Rectal
    if any(p in text for p in ['rectal', 'suppository', 'enema']):
        return 'rectal'

    # Ophthalmic
    if any(p in text for p in ['ophthalmic', 'eye drop', 'eye solution']):
        return 'ophthalmic'

    return None


def extract_frequency(medication_string: str) -> Optional[str]:
    """
    Extract dosing frequency from medication string.

    Returns: One of 'QD', 'BID', 'TID', 'QID', 'Q6H', 'Q8H', 'Q12H', 'PRN', 'ONCE', or None
    """
    if not medication_string:
        return None

    text = medication_string.lower()

    # Check explicit frequency markers
    if re.search(r'\bprn\b|as needed', text):
        return 'PRN'
    if re.search(r'\bonce\b|single dose|one time', text):
        return 'ONCE'
    if re.search(r'\bq\s*6\s*h|every\s*6\s*hour', text):
        return 'Q6H'
    if re.search(r'\bq\s*8\s*h|every\s*8\s*hour', text):
        return 'Q8H'
    if re.search(r'\bq\s*12\s*h|every\s*12\s*hour', text):
        return 'Q12H'
    if re.search(r'\bqid\b|four times', text):
        return 'QID'
    if re.search(r'\btid\b|three times', text):
        return 'TID'
    if re.search(r'\bbid\b|twice|two times', text):
        return 'BID'
    if re.search(r'\bqd\b|daily|once daily', text):
        return 'QD'

    return None


def extract_drug_name(medication_string: str) -> str:
    """
    Extract clean drug name from medication string.

    Strategy: Take text before first numeric value, clean up.
    """
    if not medication_string:
        return ""

    text = medication_string.strip()

    # Find first number - drug name is typically before it
    match = re.search(r'\d', text)
    if match:
        name_part = text[:match.start()].strip()
    else:
        # No number found, use full string up to common suffixes
        name_part = text

    # Clean up common suffixes that aren't part of drug name
    cleanup_patterns = [
        r'\s+(tablet|capsule|solution|injection|cream|ointment|suspension)s?$',
        r'\s+(hcl|sodium|sulfate|citrate|tartrate|bitartrate|pf)$',
        r'\s*[,/].*$',  # Remove anything after comma or slash
    ]

    for pattern in cleanup_patterns:
        name_part = re.sub(pattern, '', name_part, flags=re.IGNORECASE)

    # Final cleanup
    name_part = name_part.strip().lower()

    # Remove trailing punctuation
    name_part = re.sub(r'[,;:\-]+$', '', name_part).strip()

    return name_part


def parse_medication_string(medication_string: str) -> Dict[str, Any]:
    """
    Full parsing of medication string.

    Returns dictionary with all extracted fields:
        - parsed_name: str
        - parsed_dose_value: float or None
        - parsed_dose_unit: str or None
        - parsed_route: str or None
        - parsed_frequency: str or None
        - parse_method: str
        - parse_confidence: float
    """
    dose_info = extract_dose(medication_string)

    return {
        'parsed_name': extract_drug_name(medication_string),
        'parsed_dose_value': dose_info['dose_value'],
        'parsed_dose_unit': dose_info['dose_unit'],
        'parsed_route': extract_route(medication_string),
        'parsed_frequency': extract_frequency(medication_string),
        'parse_method': dose_info['parse_method'],
        'parse_confidence': dose_info['parse_confidence'],
    }
