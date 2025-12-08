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
