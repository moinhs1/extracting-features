"""Extract vital signs from Hnp.txt (H&P notes)."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .hnp_patterns import SECTION_PATTERNS


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
