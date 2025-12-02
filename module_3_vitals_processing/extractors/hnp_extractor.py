"""Extract vital signs from Hnp.txt (H&P notes)."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .hnp_patterns import (
    SECTION_PATTERNS, NEGATION_PATTERNS, HR_PATTERNS, BP_PATTERNS,
    RR_PATTERNS, SPO2_PATTERNS, TEMP_PATTERNS, TIMESTAMP_PATTERNS,
    VALID_RANGES, DEFAULT_TIMESTAMP_OFFSET, HNP_COLUMNS
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


def extract_timestamp(
    text: str,
    section: str,
    report_datetime: datetime
) -> Tuple[datetime, str, float]:
    """
    Extract explicit timestamp or estimate from section context.

    Args:
        text: Text window to search for timestamp
        section: Section name (ed_course, exam, vitals, etc.)
        report_datetime: Report_Date_Time from the note

    Returns:
        Tuple of (timestamp, source, offset_hours)
        source is 'explicit' or 'estimated'
    """
    # Try explicit timestamp extraction
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)

                # Parse date
                for fmt in ['%m/%d/%Y', '%m/%d/%y']:
                    try:
                        date_part = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                # Parse time
                time_str = time_str.strip()
                for fmt in ['%I:%M %p', '%I:%M:%S %p', '%H:%M', '%H%M']:
                    try:
                        time_part = datetime.strptime(time_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

                timestamp = date_part.replace(
                    hour=time_part.hour,
                    minute=time_part.minute,
                    second=getattr(time_part, 'second', 0)
                )
                return timestamp, 'explicit', 0.0

            except (ValueError, AttributeError):
                continue

    # Fall back to estimation based on section
    if section in SECTION_PATTERNS:
        _, offset = SECTION_PATTERNS[section]
    else:
        offset = DEFAULT_TIMESTAMP_OFFSET

    estimated_ts = report_datetime + timedelta(hours=offset)
    return estimated_ts, 'estimated', float(offset)


def process_hnp_row(row: pd.Series) -> List[Dict]:
    """
    Process a single H&P note row and extract all vitals.

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

    results = []

    # Identify sections
    sections = identify_sections(text)

    # If no sections found, use full text as 'other'
    if not sections:
        sections = {'other': text}

    # Process each section
    for section_name, section_text in sections.items():
        # Get timestamp for this section
        timestamp, ts_source, ts_offset = extract_timestamp(
            section_text, section_name, report_datetime
        )

        # Extract each vital type
        # Heart Rate
        for hr in extract_heart_rate(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'HR',
                'value': hr['value'],
                'units': 'bpm',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': hr['confidence'],
                'is_flagged_abnormal': hr['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # Blood Pressure (creates SBP and DBP records)
        for bp in extract_blood_pressure(section_text):
            for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
                results.append({
                    'EMPI': empi,
                    'timestamp': timestamp,
                    'timestamp_source': ts_source,
                    'timestamp_offset_hours': ts_offset,
                    'vital_type': vital_type,
                    'value': value,
                    'units': 'mmHg',
                    'source': 'hnp',
                    'extraction_context': section_name,
                    'confidence': bp['confidence'],
                    'is_flagged_abnormal': bp['is_flagged_abnormal'],
                    'report_number': report_number,
                    'report_date_time': report_datetime,
                })

        # Respiratory Rate
        for rr in extract_respiratory_rate(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'RR',
                'value': rr['value'],
                'units': 'breaths/min',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': rr['confidence'],
                'is_flagged_abnormal': rr['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # SpO2
        for spo2 in extract_spo2(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'SPO2',
                'value': spo2['value'],
                'units': '%',
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': spo2['confidence'],
                'is_flagged_abnormal': spo2['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

        # Temperature
        for temp in extract_temperature(section_text):
            results.append({
                'EMPI': empi,
                'timestamp': timestamp,
                'timestamp_source': ts_source,
                'timestamp_offset_hours': ts_offset,
                'vital_type': 'TEMP',
                'value': temp['value'],
                'units': temp['units'],
                'source': 'hnp',
                'extraction_context': section_name,
                'confidence': temp['confidence'],
                'is_flagged_abnormal': temp['is_flagged_abnormal'],
                'report_number': report_number,
                'report_date_time': report_datetime,
            })

    return results


def _process_chunk(chunk: pd.DataFrame) -> List[Dict]:
    """Process a chunk of rows (for multiprocessing)."""
    results = []
    for _, row in chunk.iterrows():
        results.extend(process_hnp_row(row))
    return results


def extract_hnp_vitals(
    input_path: str,
    output_path: str,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Extract vital signs from Hnp.txt file with parallel processing.

    Args:
        input_path: Path to Hnp.txt file
        output_path: Path for output parquet file
        n_workers: Number of parallel workers (default: CPU count)
        chunk_size: Rows per chunk for processing

    Returns:
        DataFrame with extracted vitals (also saved to parquet)
    """
    if n_workers is None:
        n_workers = cpu_count()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Read and process in chunks
    chunks_processed = 0
    for chunk in pd.read_csv(
        input_path,
        sep='|',
        names=HNP_COLUMNS,
        header=0,
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines='skip'
    ):
        chunks_processed += 1

        if n_workers > 1:
            # Split chunk for parallel processing
            chunk_splits = [
                chunk.iloc[i:i + chunk_size // n_workers]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for result_list in chunk_results:
                all_results.extend(result_list)
        else:
            # Single-threaded processing
            all_results.extend(_process_chunk(chunk))

        print(f"Processed chunk {chunks_processed}, total records: {len(all_results)}")

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
    else:
        df = pd.DataFrame(columns=[
            'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
            'vital_type', 'value', 'units', 'source', 'extraction_context',
            'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time'
        ])

    # Save to parquet
    df.to_parquet(output_path, index=False)

    print(f"Extraction complete. Total records: {len(df)}")
    print(f"Output saved to: {output_path}")

    return df


if __name__ == '__main__':
    import argparse
    import sys

    # Add parent directory to path to allow imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description='Extract vital signs from Hnp.txt (H&P notes)'
    )
    parser.add_argument(
        '-i', '--input',
        default=str(DATA_DIR / 'Hnp.txt'),
        help='Input Hnp.txt file path'
    )
    parser.add_argument(
        '-o', '--output',
        default=str(OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'),
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

    args = parser.parse_args()

    print(f"Extracting vitals from: {args.input}")
    print(f"Output to: {args.output}")
    print(f"Workers: {args.workers or 'auto'}")
    print(f"Chunk size: {args.chunk_size}")

    extract_hnp_vitals(
        args.input,
        args.output,
        n_workers=args.workers,
        chunk_size=args.chunk_size
    )
