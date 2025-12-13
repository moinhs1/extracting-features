"""Extract vital signs from Hnp.txt (H&P notes) - Thin wrapper."""
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path

from .unified_extractor import (
    extract_heart_rate, extract_blood_pressure,
    extract_respiratory_rate, extract_spo2, extract_temperature,
    extract_all_vitals, extract_supplemental_vitals,
    check_negation as _unified_check_negation
)
from .hnp_patterns import (
    SECTION_PATTERNS, TIMESTAMP_PATTERNS,
    DEFAULT_TIMESTAMP_OFFSET, HNP_COLUMNS, NEGATION_PATTERNS
)


def check_negation(text: str, position: int, window: int = 50) -> bool:
    """
    Check for negation phrases in context window around match position.
    Looks both forward and backward for backward compatibility.
    """
    start = max(0, position - window)
    end = min(len(text), position + window)
    context = text[start:end].lower()

    for pattern in NEGATION_PATTERNS:
        if re.search(pattern, context, re.IGNORECASE):
            return True
    return False


def identify_sections(text: str, window_size: int = 500) -> Dict[str, str]:
    """Identify clinical sections in note text."""
    sections = {}
    for section_name, (pattern, _offset) in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = match.end()
            end = min(start + window_size, len(text))
            sections[section_name] = text[start:end]
    return sections


def extract_timestamp(
    text: str,
    section: str,
    report_datetime: datetime
) -> Tuple[datetime, str, float]:
    """Extract explicit timestamp or estimate from section context."""
    for pattern in TIMESTAMP_PATTERNS:
        match = re.search(pattern, text)
        if match:
            try:
                date_str = match.group(1)
                time_str = match.group(2)

                for fmt in ['%m/%d/%Y', '%m/%d/%y']:
                    try:
                        date_part = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    continue

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

    if section in SECTION_PATTERNS:
        _, offset = SECTION_PATTERNS[section]
    else:
        offset = DEFAULT_TIMESTAMP_OFFSET

    estimated_ts = report_datetime + timedelta(hours=offset)
    return estimated_ts, 'estimated', float(offset)


def process_hnp_row(row: pd.Series) -> List[Dict]:
    """
    Process a single H&P note row.

    Returns:
        List of extracted vital records (backward compatible API)
    """
    core_results, _ = _process_hnp_row_full(row)
    return core_results


def _process_hnp_row_full(row: pd.Series) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single H&P note row with supplemental extraction.

    Returns:
        Tuple of (core_vitals, supplemental_vitals)
    """
    text = row.get('Report_Text')
    if not text or pd.isna(text):
        return [], []

    empi = str(row.get('EMPI', ''))
    report_number = str(row.get('Report_Number', ''))

    report_dt_str = row.get('Report_Date_Time', '')
    try:
        report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %I:%M:%S %p')
    except (ValueError, TypeError):
        try:
            report_datetime = datetime.strptime(report_dt_str, '%m/%d/%Y %H:%M:%S')
        except (ValueError, TypeError):
            report_datetime = datetime.now()

    core_results = []
    supplemental_results = []

    sections = identify_sections(text)
    if not sections:
        sections = {'other': text}

    for section_name, section_text in sections.items():
        timestamp, ts_source, ts_offset = extract_timestamp(
            section_text, section_name, report_datetime
        )

        base_record = {
            'EMPI': empi,
            'timestamp': timestamp,
            'timestamp_source': ts_source,
            'timestamp_offset_hours': ts_offset,
            'source': 'hnp',
            'extraction_context': section_name,
            'report_number': report_number,
            'report_date_time': report_datetime,
        }

        # Extract core vitals using unified extractor
        for hr in extract_heart_rate(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'HR',
                'value': hr['value'],
                'units': 'bpm',
                'confidence': hr['confidence'],
                'is_flagged_abnormal': hr['is_flagged_abnormal'],
            })

        for bp in extract_blood_pressure(section_text):
            for vital_type, value in [('SBP', bp['sbp']), ('DBP', bp['dbp'])]:
                core_results.append({
                    **base_record,
                    'vital_type': vital_type,
                    'value': value,
                    'units': 'mmHg',
                    'confidence': bp['confidence'],
                    'is_flagged_abnormal': bp['is_flagged_abnormal'],
                })

        for rr in extract_respiratory_rate(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'RR',
                'value': rr['value'],
                'units': 'breaths/min',
                'confidence': rr['confidence'],
                'is_flagged_abnormal': rr['is_flagged_abnormal'],
            })

        for spo2 in extract_spo2(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'SPO2',
                'value': spo2['value'],
                'units': '%',
                'confidence': spo2['confidence'],
                'is_flagged_abnormal': spo2['is_flagged_abnormal'],
            })

        for temp in extract_temperature(section_text):
            core_results.append({
                **base_record,
                'vital_type': 'TEMP',
                'value': temp['value'],
                'units': temp['units'],
                'confidence': temp['confidence'],
                'is_flagged_abnormal': temp['is_flagged_abnormal'],
            })

    # Extract supplemental from full text
    supplemental = extract_supplemental_vitals(text)

    base_supplemental = {
        'EMPI': empi,
        'timestamp': report_datetime,
        'source': 'hnp',
        'report_number': report_number,
    }

    for o2_flow in supplemental['O2_FLOW']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'O2_FLOW',
            'value': o2_flow['value'],
            'units': o2_flow['units'],
            'confidence': o2_flow['confidence'],
        })

    for o2_device in supplemental['O2_DEVICE']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'O2_DEVICE',
            'value': o2_device['value'],
            'units': None,
            'confidence': o2_device['confidence'],
        })

    for bmi in supplemental['BMI']:
        supplemental_results.append({
            **base_supplemental,
            'vital_type': 'BMI',
            'value': bmi['value'],
            'units': bmi['units'],
            'confidence': bmi['confidence'],
        })

    return core_results, supplemental_results


def _process_chunk(chunk: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    """Process a chunk of rows."""
    core_results = []
    supplemental_results = []
    for _, row in chunk.iterrows():
        core, supp = _process_hnp_row_full(row)
        core_results.extend(core)
        supplemental_results.extend(supp)
    return core_results, supplemental_results


def extract_hnp_vitals(
    input_path: str,
    output_path: str,
    supplemental_path: Optional[str] = None,
    n_workers: Optional[int] = None,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Extract vital signs from Hnp.txt file.

    Args:
        input_path: Path to Hnp.txt file
        output_path: Path for core vitals parquet
        supplemental_path: Path for supplemental vitals parquet (optional)
        n_workers: Number of parallel workers
        chunk_size: Rows per chunk

    Returns:
        DataFrame with core extracted vitals
    """
    if n_workers is None:
        n_workers = cpu_count()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if supplemental_path:
        Path(supplemental_path).parent.mkdir(parents=True, exist_ok=True)

    all_core = []
    all_supplemental = []

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
            chunk_splits = [
                chunk.iloc[i:i + chunk_size // n_workers]
                for i in range(0, len(chunk), max(1, chunk_size // n_workers))
            ]

            with Pool(n_workers) as pool:
                chunk_results = pool.map(_process_chunk, chunk_splits)

            for core, supp in chunk_results:
                all_core.extend(core)
                all_supplemental.extend(supp)
        else:
            core, supp = _process_chunk(chunk)
            all_core.extend(core)
            all_supplemental.extend(supp)

        print(f"Processed chunk {chunks_processed}, core: {len(all_core)}, supplemental: {len(all_supplemental)}")

    # Create and save DataFrames
    core_columns = [
        'EMPI', 'timestamp', 'timestamp_source', 'timestamp_offset_hours',
        'vital_type', 'value', 'units', 'source', 'extraction_context',
        'confidence', 'is_flagged_abnormal', 'report_number', 'report_date_time'
    ]

    if all_core:
        df_core = pd.DataFrame(all_core)
    else:
        df_core = pd.DataFrame(columns=core_columns)

    df_core.to_parquet(output_path, index=False)
    print(f"Core vitals saved to: {output_path}")

    if supplemental_path and all_supplemental:
        df_supp = pd.DataFrame(all_supplemental)
        df_supp.to_parquet(supplemental_path, index=False)
        print(f"Supplemental vitals saved to: {supplemental_path}")

    return df_core


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from module_3_vitals_processing.config.vitals_config import DATA_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(description='Extract vital signs from Hnp.txt')
    parser.add_argument('-i', '--input', default=str(DATA_DIR / 'Hnp.txt'))
    parser.add_argument('-o', '--output', default=str(OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'))
    parser.add_argument('-s', '--supplemental', default=str(OUTPUT_DIR / 'discovery' / 'hnp_supplemental.parquet'))
    parser.add_argument('-w', '--workers', type=int, default=None)
    parser.add_argument('-c', '--chunk-size', type=int, default=10000)

    args = parser.parse_args()

    extract_hnp_vitals(
        args.input,
        args.output,
        supplemental_path=args.supplemental,
        n_workers=args.workers,
        chunk_size=args.chunk_size
    )
