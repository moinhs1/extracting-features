"""Extract vital signs from Hnp.txt (H&P notes)."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS, VALID_RANGES
)


def identify_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """
    Identify clinical sections in note text.

    Args:
        text: Full Report_Text from H&P note
        window_size: Characters to extract after section header

    Returns:
        Dict mapping section name to text window
    """
    sections = {}

    for section_name, (pattern, _offset) in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]

    return sections


def check_negation(text: str, position: int, window: int = 50) -> bool:
    """
    Check for negation phrases in context window around match position.

    Args:
        text: Full text being searched
        position: Character position of the match
        window: Characters to check before and after position

    Returns:
        True if negation phrase found in window
    """
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end].lower()

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True

    return False


def extract_heart_rate(text: str) -> List[Dict]:
    """
    Extract heart rate values from text.

    Args:
        text: Text to search for heart rate values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in HR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Skip if we already found a value at similar position
            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            # Check for negation
            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # Validate range
            min_val, max_val = VALID_RANGES['HR']
            if not (min_val <= value <= max_val):
                continue

            # Check for abnormal flag (!)
            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context or '(!)' in match.group(0)

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results


def extract_blood_pressure(text: str) -> List[Dict]:
    """
    Extract blood pressure values from text.

    Args:
        text: Text to search for BP values

    Returns:
        List of dicts with sbp, dbp, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in BP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Skip if we already found a value at similar position
            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            # Check for negation
            if check_negation(text, position):
                continue

            try:
                sbp = float(match.group(1))
                dbp = float(match.group(2))
            except (ValueError, IndexError):
                continue

            # Swap if SBP < DBP (likely transposed)
            if sbp < dbp:
                sbp, dbp = dbp, sbp

            # Validate ranges
            sbp_min, sbp_max = VALID_RANGES['SBP']
            dbp_min, dbp_max = VALID_RANGES['DBP']
            if not (sbp_min <= sbp <= sbp_max and dbp_min <= dbp <= dbp_max):
                continue

            # Check for abnormal flag (!)
            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context or '(!)' in match.group(0)

            results.append({
                'sbp': sbp,
                'dbp': dbp,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results


def extract_respiratory_rate(text: str) -> List[Dict]:
    """
    Extract respiratory rate values from text.

    Args:
        text: Text to search for RR values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in RR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['RR']
            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results


def extract_spo2(text: str) -> List[Dict]:
    """
    Extract SpO2 values from text.

    Args:
        text: Text to search for SpO2 values

    Returns:
        List of dicts with value, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in SPO2_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            min_val, max_val = VALID_RANGES['SPO2']
            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results


def extract_temperature(text: str) -> List[Dict]:
    """
    Extract temperature values from text.

    Args:
        text: Text to search for temperature values

    Returns:
        List of dicts with value, units, confidence, position, is_flagged_abnormal
    """
    results = []
    seen_positions = set()

    for pattern, confidence in TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
                # Get unit from capture group if available
                units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
            except (ValueError, IndexError):
                continue

            # Auto-detect unit from value if not captured
            if units is None:
                if value > 50:
                    units = 'F'
                else:
                    units = 'C'

            # Validate range based on unit
            if units == 'C':
                min_val, max_val = VALID_RANGES['TEMP_C']
            else:
                min_val, max_val = VALID_RANGES['TEMP_F']

            if not (min_val <= value <= max_val):
                continue

            context_start = max(0, position - 10)
            context = text[context_start:position + 5]
            is_flagged = '(!)' in context

            results.append({
                'value': value,
                'units': units,
                'confidence': confidence,
                'position': position,
                'is_flagged_abnormal': is_flagged,
            })
            seen_positions.add(position)

    return results
