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

# Import enhanced harmonization modules
from loinc_matcher import LoincMatcher
from unit_converter import UnitConverter
from hierarchical_clustering import (
    perform_hierarchical_clustering,
    flag_suspicious_clusters
)

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

# LOINC database paths
LOINC_CSV_PATH = Path(__file__).parent / 'Loinc' / 'LoincTable' / 'Loinc.csv'
LOINC_CACHE_DIR = Path(__file__).parent / 'cache'

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


def group_by_loinc(frequency_df):
    """
    DEPRECATED: Replaced by three-tier harmonization system.
    Use harmonization_map_draft.csv instead.

    Group tests by LOINC code families.

    Args:
        frequency_df: Frequency table from scan_lab_data

    Returns:
        pd.DataFrame: LOINC groups with columns:
            - canonical_name
            - loinc_codes
            - test_descriptions
            - test_count
            - patient_count
            - pct_of_cohort
            - common_units
    """
    print(f"\n{'='*80}")
    print("GROUPING TESTS BY LOINC CODE FAMILIES")
    print(f"{'='*80}\n")

    loinc_groups = []
    matched_tests = set()

    for canonical_name, loinc_codes in LOINC_FAMILIES.items():
        # Find all tests that match these LOINC codes
        matching_rows = []

        for _, row in frequency_df.iterrows():
            test_loinc_codes = str(row['loinc_code']).split('|')

            # Check if any of the test's LOINC codes match this family
            if any(lc in loinc_codes for lc in test_loinc_codes if lc):
                matching_rows.append(row)
                matched_tests.add(row['test_description'])

        if matching_rows:
            # Aggregate statistics
            total_count = sum(r['count'] for r in matching_rows)
            total_patients = len(set().union(*[set(str(r['patient_count']).split('|'))
                                                for r in matching_rows]))

            # Get actual patient count (max across matching tests)
            max_patient_count = max(r['patient_count'] for r in matching_rows)
            pct_of_cohort = max(r['pct_of_cohort'] for r in matching_rows)

            # Collect all descriptions and units
            descriptions = sorted(set(r['test_description'] for r in matching_rows))
            units = sorted(set(r['reference_units'] for r in matching_rows if r['reference_units']))

            loinc_groups.append({
                'canonical_name': canonical_name,
                'loinc_codes': '|'.join(loinc_codes),
                'test_descriptions': '|'.join(descriptions[:10]),  # Limit to 10
                'test_count': len(matching_rows),
                'patient_count': max_patient_count,
                'pct_of_cohort': round(pct_of_cohort, 2),
                'common_units': '|'.join(units[:5])  # Limit to 5
            })

    loinc_df = pd.DataFrame(loinc_groups)
    loinc_df = loinc_df.sort_values('patient_count', ascending=False).reset_index(drop=True)

    print(f"  LOINC families matched: {len(loinc_df)}")
    print(f"  Tests matched: {len(matched_tests)}")
    print(f"  Tests unmapped: {len(frequency_df) - len(matched_tests)}\n")

    # Create unmapped tests DataFrame
    unmapped_df = frequency_df[~frequency_df['test_description'].isin(matched_tests)].copy()
    unmapped_df['reason_unmapped'] = 'No LOINC match'

    return loinc_df, unmapped_df, matched_tests


def tier1_loinc_exact_match(frequency_df, loinc_matcher, unit_converter):
    """
    Tier 1: LOINC exact matching with unit validation.

    Args:
        frequency_df: Test frequency DataFrame with loinc_code column
        loinc_matcher: LoincMatcher instance
        unit_converter: UnitConverter instance

    Returns:
        pd.DataFrame: Tier 1 matches (auto-approved)
        set: Matched test descriptions
    """
    print(f"\n{'='*80}")
    print("TIER 1: LOINC EXACT MATCHING")
    print(f"{'='*80}\n")

    tier1_matches = []
    matched_tests = set()

    # Filter to tests with LOINC codes
    has_loinc = frequency_df[frequency_df['loinc_code'].notna()].copy()
    print(f"  Tests with LOINC codes: {len(has_loinc)}")

    for _, row in has_loinc.iterrows():
        loinc_code = row['loinc_code']
        test_desc = row['test_description']

        # Look up in LOINC database
        loinc_data = loinc_matcher.match(loinc_code)

        if loinc_data is None:
            continue  # LOINC code not in database

        # Extract metadata
        component = loinc_data.get('component', '')
        system = loinc_data.get('system', '')
        loinc_units = loinc_data.get('units', '')
        loinc_name = loinc_data.get('name', '')

        # Create group name from component
        group_name = component.lower().replace('.', '_').replace(' ', '_').replace(',', '')
        if not group_name:
            group_name = test_desc.lower().replace(' ', '_')

        # Get conversion factor if needed
        test_units = row['reference_units']
        conversion_factors = {test_units: 1.0}

        if test_units != loinc_units and loinc_units:
            # Try to get conversion factor
            factor = unit_converter.get_conversion_factor(
                component.split('.')[0].lower(),  # Base component
                test_units
            )
            if factor:
                conversion_factors[test_units] = factor

        tier1_matches.append({
            'group_name': group_name,
            'loinc_code': loinc_code,
            'component': component,
            'system': system,
            'standard_unit': loinc_units or test_units,
            'source_units': test_units,
            'conversion_factors': str(conversion_factors),
            'tier': 1,
            'needs_review': False,
            'review_reason': '',
            'matched_tests': test_desc,
            'patient_count': row['patient_count'],
            'measurement_count': row['count']
        })

        matched_tests.add(test_desc)

    tier1_df = pd.DataFrame(tier1_matches)
    tier1_df = tier1_df.sort_values('patient_count', ascending=False).reset_index(drop=True)

    print(f"  Tier 1 matches: {len(tier1_df)}")
    print(f"  Coverage: {len(matched_tests)}/{len(frequency_df)} ({len(matched_tests)/len(frequency_df)*100:.1f}%)\n")

    return tier1_df, matched_tests


def tier2_loinc_family_match(frequency_df, tier1_matched, loinc_matcher):
    """
    Tier 2: LOINC family matching (group by component).

    Args:
        frequency_df: Test frequency DataFrame
        tier1_matched: Set of tests already matched in Tier 1
        loinc_matcher: LoincMatcher instance

    Returns:
        pd.DataFrame: Tier 2 matches (needs review)
        set: Matched test descriptions
    """
    print(f"\n{'='*80}")
    print("TIER 2: LOINC FAMILY MATCHING")
    print(f"{'='*80}\n")

    # Get tests not yet matched with LOINC codes
    has_loinc = frequency_df[
        (frequency_df['loinc_code'].notna()) &
        (~frequency_df['test_description'].isin(tier1_matched))
    ].copy()

    print(f"  Tests with LOINC codes (unmapped in Tier 1): {len(has_loinc)}")

    # Group by component
    component_groups = {}
    for _, row in has_loinc.iterrows():
        loinc_code = row['loinc_code']
        loinc_data = loinc_matcher.match(loinc_code)

        if loinc_data is None:
            continue

        component = loinc_data.get('component', '')
        if not component:
            continue

        if component not in component_groups:
            component_groups[component] = []

        component_groups[component].append({
            'test_description': row['test_description'],
            'loinc_code': loinc_code,
            'system': loinc_data.get('system', ''),
            'units': row['reference_units'],
            'patient_count': row['patient_count'],
            'count': row['count']
        })

    # Create Tier 2 matches
    tier2_matches = []
    matched_tests = set()

    for component, tests in component_groups.items():
        # Check if all tests have same system
        systems = set(t['system'] for t in tests)
        units = set(t['units'] for t in tests)

        needs_review = len(systems) > 1 or len(units) > 1
        review_reason = []
        if len(systems) > 1:
            review_reason.append(f"multiple_systems ({', '.join(systems)})")
        if len(units) > 1:
            review_reason.append(f"multiple_units ({', '.join(units)})")

        group_name = component.lower().replace('.', '_').replace(' ', '_').replace(',', '')

        tier2_matches.append({
            'group_name': group_name,
            'loinc_code': tests[0]['loinc_code'],  # Representative code
            'component': component,
            'system': '|'.join(systems),
            'standard_unit': tests[0]['units'],
            'source_units': '|'.join(units),
            'conversion_factors': '{}',
            'tier': 2,
            'needs_review': needs_review,
            'review_reason': '|'.join(review_reason) if review_reason else '',
            'matched_tests': '|'.join(t['test_description'] for t in tests),
            'patient_count': max(t['patient_count'] for t in tests),
            'measurement_count': sum(t['count'] for t in tests)
        })

        for t in tests:
            matched_tests.add(t['test_description'])

    tier2_df = pd.DataFrame(tier2_matches)
    if len(tier2_df) > 0:
        tier2_df = tier2_df.sort_values('patient_count', ascending=False).reset_index(drop=True)

    print(f"  Tier 2 family groups: {len(tier2_df)}")
    print(f"  Tests matched: {len(matched_tests)}")
    print(f"  Groups needing review: {tier2_df['needs_review'].sum() if len(tier2_df) > 0 else 0}\n")

    return tier2_df, matched_tests


def tier3_hierarchical_clustering(frequency_df, all_matched, threshold=0.9):
    """
    Tier 3: Hierarchical clustering for unmapped tests.

    Args:
        frequency_df: Test frequency DataFrame
        all_matched: Set of tests already matched in Tier 1/2
        threshold: Similarity threshold for clustering (default 0.9)

    Returns:
        pd.DataFrame: Tier 3 cluster suggestions
        dict: Clusters mapping
        np.ndarray: Linkage matrix for dendrogram
        np.ndarray: Distance matrix for heatmap
    """
    print(f"\n{'='*80}")
    print("TIER 3: HIERARCHICAL CLUSTERING")
    print(f"{'='*80}\n")

    # Get unmapped tests
    unmapped_df = frequency_df[~frequency_df['test_description'].isin(all_matched)].copy()
    print(f"  Unmapped tests: {len(unmapped_df)}")

    if len(unmapped_df) == 0:
        print("  No unmapped tests to cluster\n")
        return pd.DataFrame(), {}, None, None

    # Prepare test data for clustering
    unmapped_tests = []
    for _, row in unmapped_df.iterrows():
        unmapped_tests.append({
            'name': row['test_description'],
            'unit': row['reference_units'],
            'patient_count': row['patient_count'],
            'count': row['count']
        })

    # Perform hierarchical clustering
    clusters, linkage_matrix, distances = perform_hierarchical_clustering(
        unmapped_tests,
        threshold=threshold
    )

    print(f"  Clusters found: {len(clusters)}")

    # Flag suspicious clusters
    flags = flag_suspicious_clusters(clusters, unmapped_tests)
    print(f"  Clusters flagged for review: {len(flags)}")

    # Create Tier 3 matches
    tier3_matches = []

    for cluster_id, test_indices in clusters.items():
        # Get tests in this cluster
        cluster_tests = [unmapped_tests[i] for i in test_indices]
        test_names = [t['name'] for t in cluster_tests]
        test_units = [t['unit'] for t in cluster_tests]

        # Create group name from first test (sanitized)
        group_name = test_names[0].lower().replace(' ', '_').replace('(', '').replace(')', '').replace(':', '_')

        # Check if flagged
        cluster_flags = flags.get(cluster_id, [])
        needs_review = len(cluster_flags) > 0 or len(test_indices) == 1  # Singletons need review

        tier3_matches.append({
            'group_name': group_name,
            'loinc_code': '',
            'component': '',
            'system': '',
            'standard_unit': cluster_tests[0]['unit'],
            'source_units': '|'.join(set(test_units)),
            'conversion_factors': '{}',
            'tier': 3,
            'needs_review': needs_review,
            'review_reason': '|'.join(cluster_flags) if cluster_flags else ('singleton' if len(test_indices) == 1 else ''),
            'matched_tests': '|'.join(test_names),
            'patient_count': max(t['patient_count'] for t in cluster_tests),
            'measurement_count': sum(t['count'] for t in cluster_tests)
        })

    tier3_df = pd.DataFrame(tier3_matches)
    tier3_df = tier3_df.sort_values('patient_count', ascending=False).reset_index(drop=True)

    print(f"  Tier 3 groups created: {len(tier3_df)}")
    print(f"  Groups needing review: {tier3_df['needs_review'].sum()}\n")

    return tier3_df, clusters, linkage_matrix, distances


def generate_harmonization_map_draft(tier1_df, tier2_df, tier3_df, output_path):
    """
    Combine all tiers into unified harmonization map draft.

    Args:
        tier1_df: Tier 1 matches DataFrame
        tier2_df: Tier 2 matches DataFrame
        tier3_df: Tier 3 matches DataFrame
        output_path: Path to save harmonization map CSV

    Returns:
        pd.DataFrame: Combined harmonization map
    """
    print(f"\n{'='*80}")
    print("GENERATING HARMONIZATION MAP DRAFT")
    print(f"{'='*80}\n")

    # Combine all tiers
    harmonization_map = pd.concat([tier1_df, tier2_df, tier3_df], ignore_index=True)

    # Add placeholder QC thresholds (will be reviewed)
    harmonization_map['impossible_low'] = 0.0
    harmonization_map['impossible_high'] = 9999.0
    harmonization_map['extreme_low'] = 0.0
    harmonization_map['extreme_high'] = 9999.0

    # Reorder columns for better readability
    column_order = [
        'group_name',
        'loinc_code',
        'component',
        'system',
        'standard_unit',
        'source_units',
        'conversion_factors',
        'impossible_low',
        'impossible_high',
        'extreme_low',
        'extreme_high',
        'tier',
        'needs_review',
        'review_reason',
        'matched_tests',
        'patient_count',
        'measurement_count'
    ]

    harmonization_map = harmonization_map[column_order]

    # Sort by tier, then patient count
    harmonization_map = harmonization_map.sort_values(
        ['tier', 'patient_count'],
        ascending=[True, False]
    ).reset_index(drop=True)

    # Save
    harmonization_map.to_csv(output_path, index=False)

    print(f"  Total groups: {len(harmonization_map)}")
    print(f"  Tier 1 (auto-approved): {len(tier1_df)}")
    print(f"  Tier 2 (needs review): {len(tier2_df)}")
    print(f"  Tier 3 (needs review): {len(tier3_df)}")
    print(f"  Groups needing review: {harmonization_map['needs_review'].sum()}")
    print(f"  Saved to: {output_path}\n")

    return harmonization_map


def fuzzy_match_orphans(unmapped_df, matched_tests, frequency_df):
    """
    DEPRECATED: Replaced by Tier 3 hierarchical clustering.
    Use tier3_hierarchical_clustering() instead.

    Use fuzzy string matching to group similar test names.

    Args:
        unmapped_df: Tests not matched by LOINC
        matched_tests: Set of already matched test descriptions
        frequency_df: Full frequency table

    Returns:
        pd.DataFrame: Fuzzy match suggestions
    """
    print(f"\n{'='*80}")
    print("FUZZY MATCHING UNMAPPED TESTS")
    print(f"{'='*80}\n")
    print(f"  Unmapped tests to process: {len(unmapped_df)}")
    print(f"  Fuzzy matching threshold: {FUZZY_MATCH_THRESHOLD}%\n")

    # Only process tests above frequency threshold
    cohort_size = frequency_df['patient_count'].max()
    threshold_count = int((FREQUENCY_THRESHOLD_PCT / 100) * cohort_size)

    candidates = unmapped_df[unmapped_df['patient_count'] >= threshold_count].copy()
    print(f"  Candidates above {FREQUENCY_THRESHOLD_PCT}% threshold: {len(candidates)}\n")

    if len(candidates) == 0:
        print("  No unmapped tests above frequency threshold\n")
        return pd.DataFrame()

    # Find similar test names
    suggestions = []
    processed = set()

    for idx, row in candidates.iterrows():
        test_name = row['test_description']

        if test_name in processed:
            continue

        # Find similar tests
        similar_tests = [test_name]
        similarity_scores = [100]

        for _, other_row in candidates.iterrows():
            other_name = other_row['test_description']

            if other_name == test_name or other_name in processed:
                continue

            # Calculate similarity
            score = fuzz.ratio(test_name, other_name)

            if score >= FUZZY_MATCH_THRESHOLD:
                similar_tests.append(other_name)
                similarity_scores.append(score)
                processed.add(other_name)

        if len(similar_tests) > 1:
            # Found a group
            suggested_group = test_name.lower().replace(' ', '_')

            # Aggregate patient counts
            group_patient_count = max(
                candidates[candidates['test_description'].isin(similar_tests)]['patient_count']
            )

            suggestions.append({
                'suggested_group': suggested_group,
                'matched_tests': '|'.join(similar_tests),
                'similarity_scores': '|'.join(map(str, similarity_scores)),
                'test_count': len(similar_tests),
                'patient_count': group_patient_count,
                'needs_review': min(similarity_scores) < 90  # Flag if any match <90%
            })

        processed.add(test_name)

    fuzzy_df = pd.DataFrame(suggestions)

    if len(fuzzy_df) > 0:
        fuzzy_df = fuzzy_df.sort_values('patient_count', ascending=False).reset_index(drop=True)
        print(f"  Fuzzy match groups found: {len(fuzzy_df)}")
        print(f"  Groups needing review: {fuzzy_df['needs_review'].sum()}\n")
    else:
        print("  No fuzzy match groups found\n")

    return fuzzy_df


def generate_discovery_reports(frequency_df, loinc_df, fuzzy_df, unmapped_df,
                               output_prefix, test_mode):
    """
    DEPRECATED: Generates legacy discovery files (loinc_groups.csv, fuzzy_suggestions.csv, unmapped_tests.csv).
    These files are replaced by the new three-tier system outputs.

    Save Phase 1 discovery outputs to CSV files.

    Args:
        frequency_df: Full frequency table
        loinc_df: LOINC groups
        fuzzy_df: Fuzzy match suggestions
        unmapped_df: Unmapped tests
        output_prefix: Filename prefix (e.g., 'test_n10' or 'full')
        test_mode: Whether in test mode
    """
    print(f"\n{'='*80}")
    print("SAVING DISCOVERY REPORTS")
    print(f"{'='*80}\n")

    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)

    # Save frequency report
    freq_file = DISCOVERY_DIR / f"{output_prefix}_test_frequency_report.csv"
    frequency_df.to_csv(freq_file, index=False)
    print(f"  ✓ Saved: {freq_file}")
    print(f"    Rows: {len(frequency_df)}")

    # Save LOINC groups
    loinc_file = DISCOVERY_DIR / f"{output_prefix}_loinc_groups.csv"
    loinc_df.to_csv(loinc_file, index=False)
    print(f"  ✓ Saved: {loinc_file}")
    print(f"    Rows: {len(loinc_df)}")

    # Save fuzzy suggestions (if any)
    if len(fuzzy_df) > 0:
        fuzzy_file = DISCOVERY_DIR / f"{output_prefix}_fuzzy_suggestions.csv"
        fuzzy_df.to_csv(fuzzy_file, index=False)
        print(f"  ✓ Saved: {fuzzy_file}")
        print(f"    Rows: {len(fuzzy_df)}")
    else:
        print(f"  ⊘ No fuzzy suggestions to save")

    # Save unmapped tests
    unmapped_file = DISCOVERY_DIR / f"{output_prefix}_unmapped_tests.csv"
    unmapped_df.to_csv(unmapped_file, index=False)
    print(f"  ✓ Saved: {unmapped_file}")
    print(f"    Rows: {len(unmapped_df)}")

    print(f"\n{'='*80}")
    print("PHASE 1 COMPLETE!")
    print(f"{'='*80}\n")
    print("Next steps:")
    print("1. Review files in outputs/discovery/")
    print("2. Create lab_harmonization_map.json with approved groupings")
    print("3. Run Phase 2 with --phase2 flag\n")


def run_phase1(test_mode=False, test_n=100):
    """Execute Phase 1: Discovery & Harmonization."""

    # Determine output prefix
    if test_mode:
        output_prefix = f"test_n{test_n}"
    else:
        output_prefix = "full"

    output_dir = DISCOVERY_DIR

    # Initialize LOINC matcher and unit converter
    loinc_matcher = None
    unit_converter = UnitConverter()

    if LOINC_CSV_PATH.exists():
        print(f"\nLoading LOINC database from {LOINC_CSV_PATH}...")
        loinc_matcher = LoincMatcher(str(LOINC_CSV_PATH), cache_dir=str(LOINC_CACHE_DIR))
        loinc_matcher.load()
    else:
        print(f"\nWARNING: LOINC database not found at {LOINC_CSV_PATH}")
        print("  Tier 1 matching will be skipped")
        print("  Download LOINC from https://loinc.org\n")

    # Load patient timelines
    timelines, patient_empis = load_patient_timelines(test_mode, test_n)

    # Scan lab data
    frequency_df = scan_lab_data(patient_empis, test_mode)

    # Tier 1: LOINC exact matching
    if loinc_matcher is not None:
        tier1_df, tier1_matched = tier1_loinc_exact_match(
            frequency_df,
            loinc_matcher,
            unit_converter
        )

        # Save Tier 1 output
        output_dir.mkdir(parents=True, exist_ok=True)
        tier1_output = output_dir / f'{output_prefix}_tier1_loinc_exact.csv'
        tier1_df.to_csv(tier1_output, index=False)
        print(f"  Saved Tier 1 matches to: {tier1_output}\n")
    else:
        tier1_df = pd.DataFrame()
        tier1_matched = set()

    # Tier 2: LOINC family matching
    if loinc_matcher is not None:
        tier2_df, tier2_matched = tier2_loinc_family_match(
            frequency_df,
            tier1_matched,
            loinc_matcher
        )

        # Save Tier 2 output
        tier2_output = output_dir / f'{output_prefix}_tier2_loinc_family.csv'
        tier2_df.to_csv(tier2_output, index=False)
        print(f"  Saved Tier 2 matches to: {tier2_output}\n")

        # Combine matched sets
        all_matched = tier1_matched | tier2_matched
    else:
        tier2_df = pd.DataFrame()
        tier2_matched = set()
        all_matched = tier1_matched

    # Tier 3: Hierarchical clustering
    tier3_df, clusters, linkage_matrix, distances = tier3_hierarchical_clustering(
        frequency_df,
        all_matched,
        threshold=0.9
    )

    # Save Tier 3 output
    tier3_output = output_dir / f'{output_prefix}_tier3_cluster_suggestions.csv'
    tier3_df.to_csv(tier3_output, index=False)
    print(f"  Saved Tier 3 clusters to: {tier3_output}\n")

    # Generate harmonization map draft
    harmonization_map_path = output_dir / f'{output_prefix}_harmonization_map_draft.csv'
    harmonization_map = generate_harmonization_map_draft(
        tier1_df,
        tier2_df,
        tier3_df,
        harmonization_map_path
    )

    # Generate static dendrogram if clustering was performed
    if linkage_matrix is not None:
        from visualization_generator import generate_static_dendrogram, generate_interactive_dendrogram, generate_harmonization_explorer

        # Get test names from tier3
        unmapped_test_names = []
        unmapped_df_tier3 = frequency_df[~frequency_df['test_description'].isin(all_matched)]
        for _, row in unmapped_df_tier3.iterrows():
            unmapped_test_names.append(row['test_description'])

        dendrogram_path = output_dir / f'{output_prefix}_cluster_dendrogram.png'
        generate_static_dendrogram(
            linkage_matrix,
            unmapped_test_names,
            dendrogram_path,
            title=f"Tier 3 Hierarchical Clustering (n={len(unmapped_test_names)} tests)"
        )

        # Generate interactive dendrogram
        dendrogram_html_path = output_dir / f'{output_prefix}_cluster_dendrogram_interactive.html'
        generate_interactive_dendrogram(
            linkage_matrix,
            unmapped_test_names,
            dendrogram_html_path,
            title=f"Interactive Tier 3 Clustering (n={len(unmapped_test_names)} tests)"
        )

        # Generate harmonization explorer dashboard
        explorer_path = output_dir / f'{output_prefix}_harmonization_explorer.html'
        generate_harmonization_explorer(harmonization_map, explorer_path)

    print(f"\n{'='*80}")
    print("PHASE 1 COMPLETE!")
    print(f"{'='*80}\n")
    print("Enhanced three-tier harmonization complete:")
    print(f"  - Tier 1: {len(tier1_df)} groups (LOINC exact matching)")
    print(f"  - Tier 2: {len(tier2_df)} groups (LOINC family matching)")
    print(f"  - Tier 3: {len(tier3_df)} groups (hierarchical clustering)")
    print(f"  - Total: {len(harmonization_map)} groups")
    print()
    print("Output files:")
    print(f"  ✓ {harmonization_map_path.name}")
    print(f"  ✓ {tier1_output.name}")
    print(f"  ✓ {tier2_output.name}")
    print(f"  ✓ {tier3_output.name}")
    if linkage_matrix is not None:
        print(f"  ✓ {dendrogram_path.name}")
        print(f"  ✓ {dendrogram_html_path.name}")
        print(f"  ✓ {explorer_path.name}")
    print()
    print("Next steps:")
    print("  1. Review harmonization_map_draft.csv")
    print("  2. Check visualizations (open HTML files in browser)")
    print("  3. Adjust QC thresholds and review flags as needed")
    print("  4. Run Phase 2: --phase2")


# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================

def create_default_harmonization_map(loinc_df, fuzzy_df):
    """
    DEPRECATED: This function uses legacy loinc_df and fuzzy_df.
    The new load_harmonization_map() function reads from harmonization_map_draft.csv instead.

    Create a default harmonization map from LOINC and fuzzy match results.
    This is a starting point - users should review and customize.

    Args:
        loinc_df: LOINC groups DataFrame
        fuzzy_df: Fuzzy match suggestions DataFrame

    Returns:
        dict: Harmonization mapping
    """
    harmonization_map = {}

    # Add LOINC-based groups
    for _, row in loinc_df.iterrows():
        canonical_name = row['canonical_name']

        # Get test descriptions
        test_descriptions = str(row['test_descriptions']).split('|')
        loinc_codes = str(row['loinc_codes']).split('|')

        # Get forward-fill limit
        forward_fill_hours = FORWARD_FILL_LIMITS.get(canonical_name,
                                                      FORWARD_FILL_LIMITS['default'])

        # Get QC thresholds
        qc_thresholds = QC_THRESHOLDS.get(canonical_name, {
            'impossible_low': 0,
            'impossible_high': 999999
        })

        # Get clinical thresholds
        clinical_thresholds = CLINICAL_THRESHOLDS.get(canonical_name, {})

        # Extract common unit (simplified - would need more logic in production)
        common_units = str(row.get('common_units', '')).split('|')
        canonical_unit = common_units[0] if common_units and common_units[0] else ''

        harmonization_map[canonical_name] = {
            'canonical_name': canonical_name,
            'variants': test_descriptions,
            'loinc_codes': loinc_codes,
            'canonical_unit': canonical_unit,
            'unit_conversions': {},  # User should fill this in
            'forward_fill_max_hours': forward_fill_hours,
            'qc_thresholds': qc_thresholds,
            'clinical_thresholds': clinical_thresholds
        }

    # Add fuzzy match groups (if approved)
    for _, row in fuzzy_df.iterrows():
        if not row.get('needs_review', True):  # Only auto-add if doesn't need review
            canonical_name = row['suggested_group']
            test_descriptions = str(row['matched_tests']).split('|')

            harmonization_map[canonical_name] = {
                'canonical_name': canonical_name,
                'variants': test_descriptions,
                'loinc_codes': [],
                'canonical_unit': '',
                'unit_conversions': {},
                'forward_fill_max_hours': FORWARD_FILL_LIMITS['default'],
                'qc_thresholds': {'impossible_low': 0, 'impossible_high': 999999},
                'clinical_thresholds': {}
            }

    return harmonization_map


def load_harmonization_map(output_prefix):
    """
    Load user-approved harmonization map from JSON.
    If not found, create default from harmonization_map_draft.csv.

    Args:
        output_prefix: Filename prefix

    Returns:
        dict: Harmonization mapping
    """
    print(f"\n{'='*80}")
    print("LOADING HARMONIZATION MAP")
    print(f"{'='*80}\n")

    map_file = OUTPUT_DIR / f"{output_prefix}_lab_harmonization_map.json"

    if map_file.exists():
        print(f"  Loading existing map: {map_file}")
        with open(map_file, 'r') as f:
            harmonization_map = json.load(f)
        print(f"  Loaded {len(harmonization_map)} harmonized tests\n")
    else:
        print(f"  No existing map found at: {map_file}")
        print(f"  Creating default map from harmonization_map_draft.csv...\n")

        # Load NEW harmonization map draft (from three-tier system)
        draft_file = DISCOVERY_DIR / f"{output_prefix}_harmonization_map_draft.csv"

        if not draft_file.exists():
            raise FileNotFoundError(
                f"Harmonization map draft not found at: {draft_file}\n"
                f"Run Phase 1 first with: "
                f"python module_02_laboratory_processing.py --phase1 --test --n=10"
            )

        draft_df = pd.read_csv(draft_file)

        # Convert to dict format for Phase 2
        harmonization_map = {}
        for _, row in draft_df.iterrows():
            group_name = row['group_name']
            test_descriptions = str(row['matched_tests']).split('|')
            loinc_code = str(row['loinc_code']) if pd.notna(row['loinc_code']) else ''

            # Get forward-fill limit from config or default
            forward_fill_hours = FORWARD_FILL_LIMITS.get(group_name, FORWARD_FILL_LIMITS['default'])

            # Get QC thresholds from config or use placeholders from CSV
            qc_thresholds = QC_THRESHOLDS.get(group_name, {
                'impossible_low': row.get('impossible_low', 0),
                'impossible_high': row.get('impossible_high', 999999),
                'extreme_low': row.get('extreme_low', 0),
                'extreme_high': row.get('extreme_high', 999999)
            })

            # Get clinical thresholds if available
            clinical_thresholds = CLINICAL_THRESHOLDS.get(group_name, {})

            harmonization_map[group_name] = {
                'canonical_name': group_name,
                'variants': test_descriptions,
                'loinc_codes': [loinc_code] if loinc_code else [],
                'canonical_unit': str(row['standard_unit']) if pd.notna(row['standard_unit']) else '',
                'unit_conversions': {},  # TODO: parse from conversion_factors column
                'forward_fill_max_hours': forward_fill_hours,
                'qc_thresholds': qc_thresholds,
                'clinical_thresholds': clinical_thresholds,
                'tier': int(row['tier']),
                'needs_review': bool(row['needs_review'])
            }

        # Save default map
        with open(map_file, 'w') as f:
            json.dump(harmonization_map, f, indent=2)

        print(f"  ✓ Created default harmonization map: {map_file}")
        print(f"  ✓ Contains {len(harmonization_map)} harmonized tests")
        print(f"  ✓ Source: {draft_file.name} (three-tier system)")
        print(f"\n  NOTE: Review and customize this file before re-running Phase 2\n")

    return harmonization_map


def extract_lab_sequences(patient_timelines, patient_empis, harmonization_map,
                          test_mode=False):
    """
    Extract lab sequences with triple encoding (values, masks, timestamps).

    Args:
        patient_timelines: Dict of PatientTimeline objects
        patient_empis: Set of patient EMPIs
        harmonization_map: Harmonization mapping
        test_mode: Whether in test mode

    Returns:
        dict: {patient_id: {test_name: {
            'timestamps': list,
            'values': list,
            'masks': list,
            'qc_flags': list,
            'original_units': list
        }}}
    """
    print(f"\n{'='*80}")
    print("PHASE 2: EXTRACTING LAB SEQUENCES")
    print(f"{'='*80}\n")
    print(f"  Processing {len(patient_empis)} patients")
    print(f"  Harmonized tests: {len(harmonization_map)}")
    print(f"  Processing in chunks (1M rows per chunk)...\n")

    # Initialize storage structure
    sequences = {pid: defaultdict(lambda: {
        'timestamps': [],
        'values': [],
        'masks': [],
        'qc_flags': [],
        'original_units': []
    }) for pid in patient_timelines.keys()}

    # Create reverse mapping: test_description -> canonical_name
    test_to_canonical = {}
    for canonical_name, info in harmonization_map.items():
        for variant in info['variants']:
            test_to_canonical[variant.upper()] = canonical_name

    # Read lab data in chunks
    chunk_num = 0
    total_measurements = 0

    chunksize = 1_000_000
    for chunk in pd.read_csv(LAB_FILE, sep='|', chunksize=chunksize, dtype={'EMPI': str}):
        chunk_num += 1

        # Filter to cohort
        cohort_chunk = chunk[chunk['EMPI'].isin(patient_empis)].copy()

        if len(cohort_chunk) == 0:
            continue

        print(f"  Chunk {chunk_num}: {len(cohort_chunk):,} cohort rows")

        # Parse timestamps
        cohort_chunk['Seq_Date_Time'] = pd.to_datetime(
            cohort_chunk['Seq_Date_Time'], errors='coerce'
        )

        # Process each row
        for _, row in cohort_chunk.iterrows():
            patient_id = str(row['EMPI'])
            test_desc = str(row.get('Test_Description', '')).strip().upper()

            # Check if this test is harmonized
            canonical_name = test_to_canonical.get(test_desc)
            if not canonical_name:
                continue

            # Get timestamp
            timestamp = row['Seq_Date_Time']
            if pd.isna(timestamp):
                continue

            # Get value
            try:
                value = float(row['Result'])
            except (ValueError, TypeError):
                continue  # Skip non-numeric results

            # Get unit
            original_unit = str(row.get('Reference_Units', '')).strip()

            # Apply QC
            qc_thresholds = harmonization_map[canonical_name]['qc_thresholds']
            qc_flag = 0  # 0=valid

            if 'impossible_low' in qc_thresholds and value < qc_thresholds['impossible_low']:
                qc_flag = 3  # Impossible
                value = np.nan
            elif 'impossible_high' in qc_thresholds and value > qc_thresholds['impossible_high']:
                qc_flag = 3  # Impossible
                value = np.nan
            elif 'extreme_high' in qc_thresholds and value > qc_thresholds['extreme_high']:
                qc_flag = 1  # Extreme
            elif 'extreme_low' in qc_thresholds and value < qc_thresholds['extreme_low']:
                qc_flag = 1  # Extreme

            # Store measurement
            sequences[patient_id][canonical_name]['timestamps'].append(timestamp)
            sequences[patient_id][canonical_name]['values'].append(value)
            sequences[patient_id][canonical_name]['masks'].append(1)  # Observed
            sequences[patient_id][canonical_name]['qc_flags'].append(qc_flag)
            sequences[patient_id][canonical_name]['original_units'].append(original_unit)

            total_measurements += 1

    print(f"\n  Total measurements extracted: {total_measurements:,}")

    # Sort sequences by timestamp
    print(f"  Sorting sequences by timestamp...\n")
    for patient_id in sequences:
        for test_name in sequences[patient_id]:
            data = sequences[patient_id][test_name]

            # Sort by timestamp
            sorted_indices = np.argsort(data['timestamps'])
            data['timestamps'] = [data['timestamps'][i] for i in sorted_indices]
            data['values'] = [data['values'][i] for i in sorted_indices]
            data['masks'] = [data['masks'][i] for i in sorted_indices]
            data['qc_flags'] = [data['qc_flags'][i] for i in sorted_indices]
            data['original_units'] = [data['original_units'][i] for i in sorted_indices]

    return sequences


def calculate_temporal_features(sequences, patient_timelines, harmonization_map):
    """
    Calculate 18 temporal features per test per phase.

    Args:
        sequences: Lab sequences dict from extract_lab_sequences
        patient_timelines: Dict of PatientTimeline objects
        harmonization_map: Harmonization mapping

    Returns:
        pd.DataFrame: Features with one row per patient
    """
    print(f"\n{'='*80}")
    print("CALCULATING TEMPORAL FEATURES")
    print(f"{'='*80}\n")
    print(f"  Processing {len(patient_timelines)} patients")
    print(f"  Features: 18 per test per phase × 4 phases = 72 per test\n")

    all_features = []

    for patient_id, timeline in patient_timelines.items():
        patient_features = {'patient_id': patient_id}

        # Get phase boundaries
        phase_boundaries = timeline.phase_boundaries

        # Process each test
        for test_name, test_data in sequences[patient_id].items():
            if len(test_data['values']) == 0:
                continue

            # Convert to numpy arrays
            timestamps = np.array(test_data['timestamps'])
            values = np.array(test_data['values'])
            masks = np.array(test_data['masks'])
            qc_flags = np.array(test_data['qc_flags'])

            # Get clinical thresholds
            clinical_thresholds = harmonization_map[test_name].get('clinical_thresholds', {})

            # Process each temporal phase
            for phase in TEMPORAL_PHASES:
                phase_start = phase_boundaries[f'{phase}_start']
                phase_end = phase_boundaries[f'{phase}_end']

                # Filter to measurements in this phase
                in_phase = (timestamps >= phase_start) & (timestamps <= phase_end)
                phase_values = values[in_phase]
                phase_timestamps = timestamps[in_phase]
                phase_qc_flags = qc_flags[in_phase]

                # Filter out impossible values (qc_flag=3)
                valid_mask = phase_qc_flags != 3
                valid_values = phase_values[valid_mask]
                valid_timestamps = phase_timestamps[valid_mask]

                prefix = f"{test_name}_{phase}"

                # 1. Basic Statistics (7 features)
                if len(valid_values) > 0:
                    patient_features[f"{prefix}_first"] = valid_values[0]
                    patient_features[f"{prefix}_last"] = valid_values[-1]
                    patient_features[f"{prefix}_min"] = np.nanmin(valid_values)
                    patient_features[f"{prefix}_max"] = np.nanmax(valid_values)
                    patient_features[f"{prefix}_mean"] = np.nanmean(valid_values)
                    patient_features[f"{prefix}_median"] = np.nanmedian(valid_values)
                    patient_features[f"{prefix}_std"] = np.nanstd(valid_values)
                else:
                    patient_features[f"{prefix}_first"] = np.nan
                    patient_features[f"{prefix}_last"] = np.nan
                    patient_features[f"{prefix}_min"] = np.nan
                    patient_features[f"{prefix}_max"] = np.nan
                    patient_features[f"{prefix}_mean"] = np.nan
                    patient_features[f"{prefix}_median"] = np.nan
                    patient_features[f"{prefix}_std"] = np.nan

                # 2. Temporal Dynamics (4 features)
                # Delta from baseline
                baseline_mean = patient_features.get(f"{test_name}_BASELINE_mean", np.nan)
                current_mean = patient_features[f"{prefix}_mean"]
                patient_features[f"{prefix}_delta_from_baseline"] = current_mean - baseline_mean

                # Time to peak/nadir
                if len(valid_values) > 0 and not np.all(np.isnan(valid_values)):
                    peak_idx = np.nanargmax(valid_values)
                    nadir_idx = np.nanargmin(valid_values)

                    phase_duration_hours = (phase_end - phase_start).total_seconds() / 3600
                    time_to_peak = (valid_timestamps[peak_idx] - phase_start).total_seconds() / 3600
                    time_to_nadir = (valid_timestamps[nadir_idx] - phase_start).total_seconds() / 3600

                    patient_features[f"{prefix}_time_to_peak"] = time_to_peak
                    patient_features[f"{prefix}_time_to_nadir"] = time_to_nadir

                    # Rate of change
                    if len(valid_values) > 1:
                        time_diff = (valid_timestamps[-1] - valid_timestamps[0]).total_seconds() / 3600
                        if time_diff > 0:
                            rate = (valid_values[-1] - valid_values[0]) / time_diff
                            patient_features[f"{prefix}_rate_of_change"] = rate
                        else:
                            patient_features[f"{prefix}_rate_of_change"] = 0
                    else:
                        patient_features[f"{prefix}_rate_of_change"] = 0
                else:
                    patient_features[f"{prefix}_time_to_peak"] = np.nan
                    patient_features[f"{prefix}_time_to_nadir"] = np.nan
                    patient_features[f"{prefix}_rate_of_change"] = np.nan

                # 3. Threshold Crossings (2 features)
                crosses_high = 0
                crosses_low = 0

                if len(valid_values) > 0:
                    if 'high' in clinical_thresholds:
                        crosses_high = int(np.any(valid_values > clinical_thresholds['high']))
                    if 'low' in clinical_thresholds:
                        crosses_low = int(np.any(valid_values < clinical_thresholds['low']))

                patient_features[f"{prefix}_crosses_high_threshold"] = crosses_high
                patient_features[f"{prefix}_crosses_low_threshold"] = crosses_low

                # 4. Missing Data Patterns (3 features)
                patient_features[f"{prefix}_count"] = len(phase_values)  # All measurements (including invalid)

                # Calculate % missing (hours with no measurement)
                phase_duration_hours = (phase_end - phase_start).total_seconds() / 3600
                if phase_duration_hours > 0:
                    pct_missing = 100 * (1 - (len(valid_timestamps) / phase_duration_hours))
                    pct_missing = max(0, min(100, pct_missing))  # Clamp to [0, 100]
                else:
                    pct_missing = 100

                patient_features[f"{prefix}_pct_missing"] = pct_missing

                # Longest gap between measurements
                if len(valid_timestamps) > 1:
                    gaps = np.diff(valid_timestamps).astype('timedelta64[h]').astype(float)
                    longest_gap = np.max(gaps)
                else:
                    longest_gap = phase_duration_hours

                patient_features[f"{prefix}_longest_gap_hours"] = longest_gap

                # 5. Area Under Curve (1 feature)
                if len(valid_values) > 1:
                    # Convert timestamps to hours from phase start
                    hours_from_start = [(t - phase_start).total_seconds() / 3600
                                       for t in valid_timestamps]
                    auc = trapezoid(valid_values, hours_from_start)
                    patient_features[f"{prefix}_auc"] = auc
                else:
                    patient_features[f"{prefix}_auc"] = np.nan

                # 6. Cross-Phase Dynamics (1 feature)
                # Peak in this phase - mean in RECOVERY phase
                recovery_mean = patient_features.get(f"{test_name}_RECOVERY_mean", np.nan)
                current_max = patient_features[f"{prefix}_max"]
                patient_features[f"{prefix}_peak_to_recovery_delta"] = current_max - recovery_mean

        all_features.append(patient_features)

    features_df = pd.DataFrame(all_features)
    print(f"  ✓ Calculated features for {len(features_df)} patients")
    print(f"  ✓ Total features: {len(features_df.columns) - 1}\n")  # -1 for patient_id

    return features_df


def save_outputs(features_df, sequences, harmonization_map, output_prefix):
    """
    Save Phase 2 outputs (CSV + HDF5).

    Args:
        features_df: Temporal features DataFrame
        sequences: Lab sequences dict
        harmonization_map: Harmonization mapping
        output_prefix: Filename prefix
    """
    print(f"\n{'='*80}")
    print("SAVING OUTPUTS")
    print(f"{'='*80}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save features CSV
    features_file = OUTPUT_DIR / f"{output_prefix}_lab_features.csv"
    features_df.to_csv(features_file, index=False)
    file_size_mb = features_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {features_file}")
    print(f"    Rows: {len(features_df)}, Columns: {len(features_df.columns)}")
    print(f"    Size: {file_size_mb:.2f} MB\n")

    # 2. Save sequences HDF5
    h5_file = OUTPUT_DIR / f"{output_prefix}_lab_sequences.h5"

    with h5py.File(h5_file, 'w') as f:
        # Create groups
        sequences_group = f.create_group('sequences')
        metadata_group = f.create_group('metadata')

        # Save sequences
        for patient_id, patient_tests in sequences.items():
            patient_group = sequences_group.create_group(str(patient_id))

            for test_name, test_data in patient_tests.items():
                if len(test_data['values']) == 0:
                    continue

                test_group = patient_group.create_group(test_name)

                # Convert timestamps to integer (milliseconds since epoch)
                timestamps_list = test_data['timestamps']
                timestamps_ms = np.array([(pd.Timestamp(t).value // 10**6) for t in timestamps_list], dtype=np.int64)
                values_np = np.array(test_data['values'], dtype=np.float64)
                masks_np = np.array(test_data['masks'], dtype=np.uint8)
                qc_flags_np = np.array(test_data['qc_flags'], dtype=np.uint8)

                # Store arrays
                test_group.create_dataset('timestamps', data=timestamps_ms)
                test_group.create_dataset('values', data=values_np)
                test_group.create_dataset('masks', data=masks_np)
                test_group.create_dataset('qc_flags', data=qc_flags_np)

                # Store units as string array
                units_np = np.array(test_data['original_units'], dtype=h5py.string_dtype())
                test_group.create_dataset('original_units', data=units_np)

        # Save metadata
        metadata_group.attrs['harmonization_map'] = json.dumps(harmonization_map)
        metadata_group.attrs['qc_thresholds'] = json.dumps(QC_THRESHOLDS)
        metadata_group.attrs['processing_timestamp'] = datetime.now().isoformat()
        metadata_group.attrs['module_version'] = '2.0'

    file_size_mb = h5_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {h5_file}")
    print(f"    Patients: {len(sequences)}")
    print(f"    Size: {file_size_mb:.2f} MB\n")

    print(f"{'='*80}")
    print("PHASE 2 COMPLETE!")
    print(f"{'='*80}\n")
    print("Outputs:")
    print(f"  - Features: {features_file}")
    print(f"  - Sequences: {h5_file}")
    print(f"\nNext steps:")
    print("  - Review lab_features.csv")
    print("  - Validate sequences with test script")
    print("  - Proceed to Module 3 (Vitals)\n")


# ============================================================================
# MAIN
# ============================================================================

def run_phase2(test_mode=False, test_n=100):
    """Execute Phase 2: Feature Engineering."""

    # Determine output prefix
    if test_mode:
        output_prefix = f"test_n{test_n}"
    else:
        output_prefix = "full"

    # Load patient timelines
    timelines, patient_empis = load_patient_timelines(test_mode, test_n)

    # Load harmonization map
    harmonization_map = load_harmonization_map(output_prefix)

    # Extract lab sequences
    sequences = extract_lab_sequences(timelines, patient_empis,
                                      harmonization_map, test_mode)

    # Calculate temporal features
    features_df = calculate_temporal_features(sequences, timelines, harmonization_map)

    # Save outputs
    save_outputs(features_df, sequences, harmonization_map, output_prefix)


def main():
    """Main execution function."""
    args = parse_arguments()

    print(f"\n{'='*80}")
    print("MODULE 2: LABORATORY PROCESSING")
    if args.test:
        print(f"*** TEST MODE: {args.n} patients ***")
    else:
        print("*** FULL COHORT MODE ***")
    print(f"{'='*80}\n")

    if args.phase1:
        run_phase1(test_mode=args.test, test_n=args.n)
    elif args.phase2:
        run_phase2(test_mode=args.test, test_n=args.n)


if __name__ == '__main__':
    main()
