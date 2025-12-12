"""
Unified vital sign extractor.

Core extraction logic with validation, negation detection, and skip section filtering.
"""
import re
from typing import Dict, List, Optional, Tuple

from .unified_patterns import (
    HR_PATTERNS, BP_PATTERNS, RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS,
    O2_FLOW_PATTERNS, O2_DEVICE_PATTERNS, BMI_PATTERNS,
    VALID_RANGES, NEGATION_PATTERNS, SKIP_SECTION_PATTERNS
)


def check_negation(text: str, position: int, window: int = 50) -> bool:
    """
    Check for negation phrases near match position.

    Args:
        text: Full text
        position: Character position of the match
        window: Characters to check before position

    Returns:
        True if negation found
    """
    start = max(0, position - window)
    context = text[start:position].lower()

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def is_in_skip_section(text: str, position: int, lookback: int = 500) -> bool:
    """
    Check if position is within a skip section.

    Args:
        text: Full text
        position: Character position
        lookback: Characters to look back

    Returns:
        True if in skip section (should not extract)
    """
    start = max(0, position - lookback)
    context_before = text[start:position]

    # Find most recent skip section header
    last_skip_pos = -1
    for pattern in SKIP_SECTION_PATTERNS:
        for match in re.finditer(pattern, context_before, re.IGNORECASE):
            if match.end() > last_skip_pos:
                last_skip_pos = match.end()

    if last_skip_pos == -1:
        return False

    # Check if a valid clinical section appears after skip
    valid_sections = [
        r'Vitals?[:\s]',
        r'Physical\s+Exam[:\s]',
        r'Objective[:\s]',
        r'Exam[:\s]',
        r'Assessment[:\s]',
    ]
    context_after_skip = context_before[last_skip_pos:]
    for pattern in valid_sections:
        if re.search(pattern, context_after_skip, re.IGNORECASE):
            return False

    return True


def extract_heart_rate(text: str) -> List[Dict]:
    """Extract heart rate values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in HR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # Position deduplication
            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            # Skip section check
            if is_in_skip_section(text, position):
                continue

            # Negation check
            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # Range validation
            min_val, max_val = VALID_RANGES['HR']
            if not (min_val <= value <= max_val):
                continue

            # Abnormal flag
            is_abnormal = value < 60 or value > 100

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_blood_pressure(text: str) -> List[Dict]:
    """Extract blood pressure values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in BP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            if check_negation(text, position):
                continue

            try:
                sbp = float(match.group(1))
                dbp = float(match.group(2))
            except (ValueError, IndexError):
                continue

            # Swap if transposed
            if sbp < dbp:
                sbp, dbp = dbp, sbp

            # Range validation
            sbp_min, sbp_max = VALID_RANGES['SBP']
            dbp_min, dbp_max = VALID_RANGES['DBP']
            if not (sbp_min <= sbp <= sbp_max and dbp_min <= dbp <= dbp_max):
                continue

            # Pulse pressure validation
            pulse_pressure = sbp - dbp
            pp_min, pp_max = VALID_RANGES['PULSE_PRESSURE']
            if not (pp_min <= pulse_pressure <= pp_max):
                continue

            # Abnormal flags
            sbp_abnormal = sbp < 90 or sbp > 180
            dbp_abnormal = dbp < 60 or dbp > 110

            results.append({
                'sbp': sbp,
                'dbp': dbp,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': sbp_abnormal or dbp_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_respiratory_rate(text: str) -> List[Dict]:
    """Extract respiratory rate values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in RR_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
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

            is_abnormal = value < 12 or value > 24

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_spo2(text: str) -> List[Dict]:
    """Extract SpO2 values from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in SPO2_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
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

            is_abnormal = value < 92

            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_temperature(text: str) -> List[Dict]:
    """
    Extract temperature values from text.
    All values normalized to Celsius.
    """
    results = []
    seen_positions = set()

    for pattern, confidence, tier in TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            if is_in_skip_section(text, position):
                continue

            if check_negation(text, position):
                continue

            try:
                value = float(match.group(1))
                units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
            except (ValueError, IndexError, AttributeError):
                continue

            # Auto-detect unit from value
            if units is None:
                if value > 50:
                    units = 'F'
                else:
                    units = 'C'

            # Validate and convert
            if units == 'F':
                min_val, max_val = VALID_RANGES['TEMP_F']
                if not (min_val <= value <= max_val):
                    continue
                value = round((value - 32) * 5 / 9, 1)
                units = 'C'
            else:
                min_val, max_val = VALID_RANGES['TEMP_C']
                if not (min_val <= value <= max_val):
                    continue
                value = round(value, 1)

            is_abnormal = value < 36 or value > 38.5

            results.append({
                'value': value,
                'units': 'C',
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal,
            })
            seen_positions.add(position)

    return results


def extract_all_vitals(text: str) -> Dict[str, List[Dict]]:
    """
    Extract all core vital types from text.

    Returns:
        Dict with keys: HR, BP, RR, SPO2, TEMP
    """
    return {
        'HR': extract_heart_rate(text),
        'BP': extract_blood_pressure(text),
        'RR': extract_respiratory_rate(text),
        'SPO2': extract_spo2(text),
        'TEMP': extract_temperature(text),
    }
