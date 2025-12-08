"""Extract vital signs from Prg.txt (Progress Notes)."""
import re
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
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


from .prg_patterns import PRG_SECTION_PATTERNS, PRG_SKIP_PATTERNS, PRG_TEMP_PATTERNS, TEMP_METHOD_MAP
from .hnp_patterns import TEMP_PATTERNS, VALID_RANGES, NEGATION_PATTERNS
from .hnp_extractor import (
    extract_heart_rate, extract_blood_pressure,
    extract_respiratory_rate, extract_spo2, check_negation
)


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


def extract_temperature_with_method(text: str) -> List[Dict]:
    """
    Extract temperature values with measurement method from text.

    Args:
        text: Text to search for temperature values

    Returns:
        List of dicts with value, units, method, confidence, position
    """
    results = []
    seen_positions = set()

    # First try Prg-specific patterns (with method capture)
    for pattern, confidence in PRG_TEMP_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            if any(abs(position - p) < 10 for p in seen_positions):
                continue

            try:
                value = float(match.group(1))
                units = match.group(2).upper() if match.lastindex >= 2 else None
                raw_method = match.group(3).lower() if match.lastindex >= 3 else None
            except (ValueError, IndexError, AttributeError):
                continue

            # Map raw method to canonical
            method = TEMP_METHOD_MAP.get(raw_method) if raw_method else None

            # Auto-detect unit from value if not captured
            if units is None:
                units = 'F' if value > 50 else 'C'

            # Validate range
            range_key = 'TEMP_C' if units == 'C' else 'TEMP_F'
            min_val, max_val = VALID_RANGES[range_key]
            if not (min_val <= value <= max_val):
                continue

            results.append({
                'value': value,
                'units': units,
                'method': method,
                'confidence': confidence,
                'position': position,
            })
            seen_positions.add(position)

    # Fall back to base Hnp patterns if no Prg patterns matched
    if not results:
        for pattern, confidence in TEMP_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                position = match.start()

                if any(abs(position - p) < 10 for p in seen_positions):
                    continue

                try:
                    value = float(match.group(1))
                    units = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
                except (ValueError, IndexError):
                    continue

                if units is None:
                    units = 'F' if value > 50 else 'C'

                range_key = 'TEMP_C' if units == 'C' else 'TEMP_F'
                min_val, max_val = VALID_RANGES[range_key]
                if not (min_val <= value <= max_val):
                    continue

                # Check for method in surrounding context
                context_end = min(position + 50, len(text))
                context = text[position:context_end].lower()
                method = None
                for method_str, canonical in TEMP_METHOD_MAP.items():
                    if method_str in context:
                        method = canonical
                        break

                results.append({
                    'value': value,
                    'units': units,
                    'method': method,
                    'confidence': confidence,
                    'position': position,
                })
                seen_positions.add(position)

    return results


def extract_prg_vitals_from_text(text: str) -> List[Dict]:
    """
    Extract all vital signs from progress note text with skip section filtering.

    Args:
        text: Full text to extract vitals from

    Returns:
        List of vital sign records
    """
    if not text:
        return []

    results = []

    # Extract Heart Rate
    for hr in extract_heart_rate(text):
        if is_in_skip_section(text, hr['position']):
            continue
        results.append({
            'vital_type': 'HR',
            'value': hr['value'],
            'units': 'bpm',
            'confidence': hr['confidence'],
            'is_flagged_abnormal': hr.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract Blood Pressure
    for bp in extract_blood_pressure(text):
        if is_in_skip_section(text, bp['position']):
            continue
        for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
            results.append({
                'vital_type': vital_type,
                'value': value,
                'units': 'mmHg',
                'confidence': bp['confidence'],
                'is_flagged_abnormal': bp.get('is_flagged_abnormal', False),
                'temp_method': None,
            })

    # Extract Respiratory Rate
    for rr in extract_respiratory_rate(text):
        if is_in_skip_section(text, rr['position']):
            continue
        results.append({
            'vital_type': 'RR',
            'value': rr['value'],
            'units': 'breaths/min',
            'confidence': rr['confidence'],
            'is_flagged_abnormal': rr.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract SpO2
    for spo2 in extract_spo2(text):
        if is_in_skip_section(text, spo2['position']):
            continue
        results.append({
            'vital_type': 'SPO2',
            'value': spo2['value'],
            'units': '%',
            'confidence': spo2['confidence'],
            'is_flagged_abnormal': spo2.get('is_flagged_abnormal', False),
            'temp_method': None,
        })

    # Extract Temperature with method
    for temp in extract_temperature_with_method(text):
        if is_in_skip_section(text, temp['position']):
            continue
        results.append({
            'vital_type': 'TEMP',
            'value': temp['value'],
            'units': temp['units'],
            'confidence': temp['confidence'],
            'is_flagged_abnormal': False,
            'temp_method': temp.get('method'),
        })

    return results


def process_prg_row(row: pd.Series) -> List[Dict]:
    """
    Process a single progress note row and extract all vitals.

    Args:
        row: DataFrame row with EMPI, Report_Number, Report_Date_Time, Report_Text

    Returns:
        List of vital sign records
    """
    text = row.get('Report_Text')
    if not text or pd.isna(text):
        return []

    empi = str(row.get('EMPI', ''))
    report_number = str(row.get('Report_Number', ''))

    # Parse report datetime
    report_dt_str = row.get('Report_Date_Time', '')
    try:
        report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %I:%M:%S %p')
    except (ValueError, TypeError):
        try:
            report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            report_datetime = datetime.now()

    # Extract vitals from text
    extracted = extract_prg_vitals_from_text(text)

    results = []
    for vital in extracted:
        results.append({
            'EMPI': empi,
            'timestamp': report_datetime,
            'timestamp_source': 'estimated',
            'timestamp_offset_hours': 0.0,
            'vital_type': vital['vital_type'],
            'value': vital['value'],
            'units': vital['units'],
            'source': 'prg',
            'extraction_context': 'full_text',
            'confidence': vital['confidence'],
            'is_flagged_abnormal': vital['is_flagged_abnormal'],
            'report_number': report_number,
            'report_date_time': report_datetime,
            'temp_method': vital.get('temp_method'),
        })

    return results


from multiprocessing import Pool, cpu_count
from .prg_patterns import PRG_COLUMNS


def _process_chunk(chunk: pd.DataFrame) -> List[Dict]:
    """Process a chunk of rows (for multiprocessing)."""
    results = []
    for _, row in chunk.iterrows():
        results.extend(process_prg_row(row))
    return results


def extract_prg_vitals(
    input_path: str,
    output_path: str,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000,
    resume: bool = True
) -> pd.DataFrame:
    """
    Extract vital signs from Prg.txt file with parallel processing and checkpointing.

    Args:
        input_path: Path to Prg.txt file
        output_path: Path for output parquet file
        n_workers: Number of parallel workers (default: CPU count)
        chunk_size: Rows per chunk for processing
        resume: Whether to resume from checkpoint if available

    Returns:
        DataFrame with extracted vitals (also saved to parquet)
    """
    if n_workers is None:
        n_workers = cpu_count()

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint
    checkpoint = load_checkpoint(output_dir) if resume else None
    skip_rows = checkpoint.rows_processed if checkpoint else 0

    if checkpoint:
        print(f"Resuming from row {skip_rows}, chunk {checkpoint.chunks_completed}")
        all_results_count = checkpoint.records_extracted
        chunks_completed = checkpoint.chunks_completed
        started_at = checkpoint.started_at
    else:
        all_results_count = 0
        chunks_completed = 0
        started_at = datetime.now()

    all_results = []

    # Read and process in chunks
    for chunk in pd.read_csv(
        input_path,
        sep='|',
        names=PRG_COLUMNS,
        header=0,
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines='skip',
        skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None
    ):
        chunks_completed += 1

        if n_workers > 1:
            # Split chunk for parallel processing
            chunk_splits = [
                chunk.iloc[i:i + max(1, chunk_size // n_workers)]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for result_list in chunk_results:
                all_results.extend(result_list)
        else:
            all_results.extend(_process_chunk(chunk))

        all_results_count = len(all_results)
        rows_processed = skip_rows + (chunks_completed * chunk_size)

        # Save checkpoint periodically
        if chunks_completed % CHECKPOINT_INTERVAL == 0:
            checkpoint = ExtractionCheckpoint(
                input_path=input_path,
                output_path=output_path,
                rows_processed=rows_processed,
                chunks_completed=chunks_completed,
                records_extracted=all_results_count,
                started_at=started_at,
                updated_at=datetime.now(),
            )
            save_checkpoint(checkpoint, output_dir)
            print(f"Checkpoint saved at chunk {chunks_completed}, {all_results_count} records")

        print(f"Processed chunk {chunks_completed}, total records: {all_results_count}")

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
            'vital_type', 'value', 'units', 'source', 'extraction_context',
            'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time',
            'temp_method'
        ])

    # Save to parquet
    df.to_parquet(output_path, index=False)

    # Remove checkpoint on successful completion
    checkpoint_path = output_dir / CHECKPOINT_FILE
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Extraction complete. Total records: {len(df)}")
    print(f"Output saved to: {output_path}")

    return df


def main():
    """CLI entry point for Prg vitals extraction."""
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import PRG_INPUT_PATH, PRG_OUTPUT_PATH

    parser = argparse.ArgumentParser(
        description='Extract vital signs from Prg.txt (Progress Notes)'
    )
    parser.add_argument(
        '-i', '--input',
        default=str(PRG_INPUT_PATH),
        help='Input Prg.txt file path'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(PRG_OUTPUT_PATH),
        help='Output parquet file path'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '-c', '--chunk-size',
        type=int,
        default=10000,
        help='Rows per processing chunk'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, ignore existing checkpoint'
    )

    args = parser.parse_args()

    print(f"Extracting vitals from: {args.input}")
    print(f"Output to: {args.output}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Resume: {not args.no_resume}")

    extract_prg_vitals(
        args.input,
        args.output,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
