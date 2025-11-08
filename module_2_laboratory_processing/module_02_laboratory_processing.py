"""
Module 2: Laboratory Processing
Extracts lab data with LOINC+fuzzy harmonization, triple encoding, and temporal features.
"""

import pandas as pd
import numpy as np
import h5py
import pickle
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from fuzzywuzzy import fuzz
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings('ignore')

# Import PatientTimeline from Module 1
sys.path.insert(0, str(Path(__file__).parent.parent / 'module_1_core_infrastructure'))
from module_01_core_infrastructure import PatientTimeline

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path('/home/moin/TDA_11_1/Data')
LAB_FILE = DATA_DIR / 'FNR_20240409_091633_Lab.txt'
MODULE1_DIR = Path('/home/moin/TDA_11_1/module_1_core_infrastructure')
PATIENT_TIMELINES_FILE = MODULE1_DIR / 'outputs' / 'patient_timelines.pkl'

# Output paths
OUTPUT_DIR = Path(__file__).parent / 'outputs'
DISCOVERY_DIR = OUTPUT_DIR / 'discovery'

# Temporal phase names (must match Module 1)
TEMPORAL_PHASES = ['BASELINE', 'ACUTE', 'SUBACUTE', 'RECOVERY']

# LOINC code families for harmonization
LOINC_FAMILIES = {
    'creatinine': ['2160-0', '38483-4', '14682-9'],
    'troponin_i': ['10839-9', '42757-5', '49563-0', '6598-7'],
    'troponin_t': ['6597-9', '48425-3', '67151-1'],
    'ddimer': ['48065-7', '48066-5', '7799-0'],
    'bnp': ['30934-4', '42637-9'],
    'ntprobnp': ['33762-6', '83107-3'],
    'lactate': ['2524-7', '32693-4'],
    'hemoglobin': ['718-7', '30313-1'],
    'hematocrit': ['4544-3', '71833-8'],
    'platelet': ['777-3', '26515-7'],
    'wbc': ['6690-2', '804-5'],
    'sodium': ['2951-2', '2947-0'],
    'potassium': ['2823-3', '6298-4'],
    'chloride': ['2075-0', '2069-3'],
    'bicarbonate': ['1963-8', '2028-9'],
    'bun': ['3094-0', '6299-2'],
    'glucose': ['2345-7', '41653-7'],
    'calcium': ['17861-6', '2000-8'],
    'magnesium': ['2601-3', '19123-9'],
    'phosphate': ['2777-1', '14879-1'],
    'albumin': ['1751-7', '61151-7'],
    'bilirubin_total': ['1975-2', '42719-5'],
    'alt': ['1742-6', '1744-2'],
    'ast': ['1920-8', '30239-8'],
    'alkaline_phosphatase': ['6768-6', '1785-5'],
    'inr': ['6301-6', '34714-6'],
    'ptt': ['3173-2', '14979-9'],
    'ph': ['2744-1', '2746-6'],
    'pao2': ['2703-7', '19255-9'],
    'paco2': ['2019-8', '19217-9'],
    'bicarbonate_arterial': ['1960-4', '1963-8'],
}

# QC Thresholds (physiological ranges)
QC_THRESHOLDS = {
    'troponin': {'impossible_low': 0, 'impossible_high': 100000, 'extreme_high': 10000},
    'troponin_i': {'impossible_low': 0, 'impossible_high': 100000, 'extreme_high': 10000},
    'troponin_t': {'impossible_low': 0, 'impossible_high': 100000, 'extreme_high': 10000},
    'creatinine': {'impossible_low': 0, 'impossible_high': 30, 'extreme_high': 10},
    'lactate': {'impossible_low': 0, 'impossible_high': 50, 'extreme_high': 20},
    'ddimer': {'impossible_low': 0, 'impossible_high': 100000, 'extreme_high': 20000},
    'bnp': {'impossible_low': 0, 'impossible_high': 50000, 'extreme_high': 10000},
    'ntprobnp': {'impossible_low': 0, 'impossible_high': 100000, 'extreme_high': 50000},
    'hemoglobin': {'impossible_low': 0, 'impossible_high': 25, 'extreme_high': 20, 'extreme_low': 3},
    'hematocrit': {'impossible_low': 0, 'impossible_high': 80, 'extreme_high': 70, 'extreme_low': 10},
    'platelet': {'impossible_low': 0, 'impossible_high': 2000, 'extreme_high': 1000, 'extreme_low': 20},
    'wbc': {'impossible_low': 0, 'impossible_high': 200, 'extreme_high': 100, 'extreme_low': 0.5},
    'sodium': {'impossible_low': 100, 'impossible_high': 200, 'extreme_high': 170, 'extreme_low': 110},
    'potassium': {'impossible_low': 1.0, 'impossible_high': 10, 'extreme_high': 7, 'extreme_low': 2},
    'glucose': {'impossible_low': 0, 'impossible_high': 1000, 'extreme_high': 600, 'extreme_low': 20},
    'bun': {'impossible_low': 0, 'impossible_high': 300, 'extreme_high': 150},
    'bilirubin_total': {'impossible_low': 0, 'impossible_high': 100, 'extreme_high': 30},
    'alt': {'impossible_low': 0, 'impossible_high': 10000, 'extreme_high': 1000},
    'ast': {'impossible_low': 0, 'impossible_high': 10000, 'extreme_high': 1000},
    'ph': {'impossible_low': 6.5, 'impossible_high': 8.0, 'extreme_high': 7.7, 'extreme_low': 6.9},
    'pao2': {'impossible_low': 0, 'impossible_high': 800, 'extreme_high': 600, 'extreme_low': 40},
    'paco2': {'impossible_low': 0, 'impossible_high': 200, 'extreme_high': 100, 'extreme_low': 15},
}

# Clinical thresholds for binary features
CLINICAL_THRESHOLDS = {
    'troponin': {'high': 0.04},
    'troponin_i': {'high': 0.04},
    'troponin_t': {'high': 0.014},
    'lactate': {'high': 4.0},
    'creatinine': {'high': 1.5},
    'ddimer': {'high': 500},
    'bnp': {'high': 100},
    'ntprobnp': {'high': 125},
    'hemoglobin': {'low': 7.0},
    'platelet': {'low': 50},
    'potassium': {'high': 5.5, 'low': 3.5},
    'sodium': {'high': 145, 'low': 135},
}

# Forward-fill limits (hours)
FORWARD_FILL_LIMITS = {
    'troponin': 6,
    'troponin_i': 6,
    'troponin_t': 6,
    'lactate': 4,
    'ddimer': 12,
    'creatinine': 12,
    'bnp': 24,
    'ntprobnp': 24,
    'bun': 24,
    'glucose': 12,
    'default': 12,
}

# Frequency threshold (% of cohort)
FREQUENCY_THRESHOLD_PCT = 5.0

# Fuzzy matching threshold
FUZZY_MATCH_THRESHOLD = 85

print("Constants and configuration loaded successfully.")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Module 2: Laboratory Processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Discovery (test mode, 10 patients)
  python module_02_laboratory_processing.py --phase1 --test --n=10

  # Phase 1: Discovery (full cohort)
  python module_02_laboratory_processing.py --phase1

  # Phase 2: Processing (test mode, 10 patients)
  python module_02_laboratory_processing.py --phase2 --test --n=10

  # Phase 2: Processing (full cohort)
  python module_02_laboratory_processing.py --phase2
        """
    )

    parser.add_argument('--phase1', action='store_true',
                        help='Run Phase 1: Discovery & Harmonization')
    parser.add_argument('--phase2', action='store_true',
                        help='Run Phase 2: Feature Engineering')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (subset of patients)')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of patients for test mode (default: 100)')

    args = parser.parse_args()

    # Validation
    if not args.phase1 and not args.phase2:
        parser.error("Must specify either --phase1 or --phase2")
    if args.phase1 and args.phase2:
        parser.error("Cannot specify both --phase1 and --phase2")

    return args


# ============================================================================
# DATA LOADING
# ============================================================================

def load_patient_timelines(test_mode=False, test_n=100):
    """
    Load patient timelines from Module 1.

    Returns:
        dict: {patient_id: PatientTimeline object}
        set: Set of patient EMPIs for filtering
    """
    print(f"\n{'='*80}")
    print("LOADING PATIENT TIMELINES FROM MODULE 1")
    print(f"{'='*80}\n")

    # Ensure PatientTimeline is available in __main__ for unpickling
    import __main__
    if not hasattr(__main__, 'PatientTimeline'):
        __main__.PatientTimeline = PatientTimeline

    with open(PATIENT_TIMELINES_FILE, 'rb') as f:
        timelines = pickle.load(f)

    print(f"  Loaded {len(timelines)} patient timelines")

    if test_mode:
        # Get first N patients
        patient_ids = list(timelines.keys())[:test_n]
        timelines = {pid: timelines[pid] for pid in patient_ids}
        print(f"  *** TEST MODE: Limited to {len(timelines)} patients ***")

    # Extract EMPIs for filtering
    patient_empis = set(timelines.keys())

    return timelines, patient_empis


# ============================================================================
# PHASE 1: DISCOVERY & HARMONIZATION
# ============================================================================

def scan_lab_data(patient_empis, test_mode=False):
    """
    Scan Lab.txt file in chunks to build frequency table.

    Args:
        patient_empis: Set of patient EMPIs to filter
        test_mode: Whether in test mode

    Returns:
        pd.DataFrame: Frequency table with columns:
            - test_description
            - loinc_code
            - count (total measurements)
            - patient_count (unique patients)
            - pct_of_cohort
            - reference_units (most common)
            - sample_values (5 examples)
    """
    print(f"\n{'='*80}")
    print("PHASE 1: SCANNING LAB DATA")
    print(f"{'='*80}\n")
    print(f"  Lab file: {LAB_FILE}")
    print(f"  Filtering to {len(patient_empis)} patient EMPIs")
    print(f"  Processing in chunks (1M rows per chunk)...\n")

    # Accumulate statistics
    test_stats = defaultdict(lambda: {
        'count': 0,
        'patients': set(),
        'loinc_codes': set(),
        'units': defaultdict(int),
        'sample_values': []
    })

    chunk_num = 0
    total_rows_scanned = 0
    cohort_rows = 0

    # Read in chunks
    chunksize = 1_000_000
    for chunk in pd.read_csv(LAB_FILE, sep='|', chunksize=chunksize, dtype={'EMPI': str}):
        chunk_num += 1
        total_rows_scanned += len(chunk)

        # Filter to cohort patients
        cohort_chunk = chunk[chunk['EMPI'].isin(patient_empis)].copy()
        cohort_rows += len(cohort_chunk)

        if len(cohort_chunk) == 0:
            print(f"  Chunk {chunk_num}: {len(chunk):,} rows → 0 cohort rows")
            continue

        print(f"  Chunk {chunk_num}: {len(chunk):,} rows → {len(cohort_chunk):,} cohort rows")

        # Accumulate statistics for each test
        for _, row in cohort_chunk.iterrows():
            test_desc = str(row.get('Test_Description', '')).strip().upper()
            if not test_desc or test_desc == 'NAN':
                continue

            test_stats[test_desc]['count'] += 1
            test_stats[test_desc]['patients'].add(str(row['EMPI']))

            loinc = str(row.get('Loinc_Code', '')).strip()
            if loinc and loinc != 'nan':
                test_stats[test_desc]['loinc_codes'].add(loinc)

            units = str(row.get('Reference_Units', '')).strip()
            if units and units != 'nan':
                test_stats[test_desc]['units'][units] += 1

            # Collect sample values
            if len(test_stats[test_desc]['sample_values']) < 5:
                result = str(row.get('Result', '')).strip()
                if result and result != 'nan':
                    test_stats[test_desc]['sample_values'].append(result)

    print(f"\n  Total rows scanned: {total_rows_scanned:,}")
    print(f"  Cohort rows: {cohort_rows:,}")
    print(f"  Unique test descriptions: {len(test_stats):,}\n")

    # Convert to DataFrame
    frequency_data = []
    total_patients = len(patient_empis)

    for test_desc, stats in test_stats.items():
        patient_count = len(stats['patients'])
        pct_of_cohort = (patient_count / total_patients) * 100

        # Get most common unit
        if stats['units']:
            most_common_unit = max(stats['units'].items(), key=lambda x: x[1])[0]
        else:
            most_common_unit = ''

        frequency_data.append({
            'test_description': test_desc,
            'loinc_code': '|'.join(sorted(stats['loinc_codes'])) if stats['loinc_codes'] else '',
            'count': stats['count'],
            'patient_count': patient_count,
            'pct_of_cohort': round(pct_of_cohort, 2),
            'reference_units': most_common_unit,
            'sample_values': '|'.join(stats['sample_values'])
        })

    frequency_df = pd.DataFrame(frequency_data)
    frequency_df = frequency_df.sort_values('patient_count', ascending=False).reset_index(drop=True)

    print(f"  Frequency table created: {len(frequency_df)} unique tests")

    return frequency_df
