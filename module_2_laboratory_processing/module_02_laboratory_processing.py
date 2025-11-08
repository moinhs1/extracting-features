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
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from fuzzywuzzy import fuzz
from scipy.integrate import trapezoid
import warnings
warnings.filterwarnings('ignore')

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
