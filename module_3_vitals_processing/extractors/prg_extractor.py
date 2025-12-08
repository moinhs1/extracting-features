"""Extract vital signs from Prg.txt (Progress Notes)."""
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd


@dataclass
class ExtractionCheckpoint:
    """Track extraction progress for resume capability."""
    input_path: str
    output_path: str
    rows_processed: int
    chunks_completed: int
    records_extracted: int
    started_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d['started_at'] = self.started_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'ExtractionCheckpoint':
        """Create from dict (e.g., loaded from JSON)."""
        data = data.copy()
        data['started_at'] = datetime.fromisoformat(data['started_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


CHECKPOINT_FILE = "prg_extraction_checkpoint.json"
CHECKPOINT_INTERVAL = 5  # Save every 5 chunks


def save_checkpoint(checkpoint: ExtractionCheckpoint, output_dir: Path) -> None:
    """Save extraction progress to JSON file."""
    path = output_dir / CHECKPOINT_FILE
    with open(path, 'w') as f:
        json.dump(checkpoint.to_dict(), f, indent=2)


def load_checkpoint(output_dir: Path) -> Optional[ExtractionCheckpoint]:
    """Load existing checkpoint if available."""
    path = output_dir / CHECKPOINT_FILE
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            return ExtractionCheckpoint.from_dict(data)
    return None


from .prg_patterns import PRG_SECTION_PATTERNS, PRG_SKIP_PATTERNS


def identify_prg_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """
    Identify clinical sections in progress note text.

    Args:
        text: Full Report_Text from progress note
        window_size: Characters to extract after section header

    Returns:
        Dict mapping section name to text window
    """
    sections = {}

    for section_name, (pattern, _offset) in PRG_SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]

    return sections


def is_in_skip_section(text: str, position: int, lookback: int = 500) -> bool:
    """
    Check if position is within a skip section (allergies, medications, etc.).

    Args:
        text: Full text being searched
        position: Character position of the match
        lookback: Characters to look back for section headers

    Returns:
        True if in a skip section (should not extract vitals here)
    """
    start = max(0, position - lookback)
    context_before = text[start:position]

    # Find most recent skip section
    last_skip_pos = -1
    for pattern in PRG_SKIP_PATTERNS:
        for match in re.finditer(pattern, context_before, re.IGNORECASE):
            if match.end() > last_skip_pos:
                last_skip_pos = match.end()

    if last_skip_pos == -1:
        # No skip section found
        return False

    # Check if a valid section appears after the skip section
    context_after_skip = context_before[last_skip_pos:]
    for section_name, (pattern, _) in PRG_SECTION_PATTERNS.items():
        if re.search(pattern, context_after_skip, re.IGNORECASE):
            # Valid section found after skip section
            return False

    # Still in skip section
    return True
