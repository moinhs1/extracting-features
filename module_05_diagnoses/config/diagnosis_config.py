"""Configuration constants for diagnosis processing."""
from pathlib import Path

# Paths
MODULE_ROOT = Path(__file__).parent.parent
DATA_PATH = MODULE_ROOT.parent / "Data" / "Dia.txt"
PATIENT_TIMELINES_PATH = MODULE_ROOT.parent / "module_1_core_infrastructure" / "outputs" / "patient_timelines.pkl"
OUTPUT_PATH = MODULE_ROOT / "outputs"

# RPDR Dia.txt columns
DIA_COLUMNS = [
    "EMPI", "EPIC_PMRN", "MRN_Type", "MRN", "Date", "Diagnosis_Name",
    "Code_Type", "Code", "Diagnosis_Flag", "Provider", "Clinic",
    "Hospital", "Inpatient_Outpatient", "Encounter_number"
]

# Temporal window (days relative to PE)
TEMPORAL_WINDOW_MIN_DAYS = -365 * 5  # 5 years before PE
TEMPORAL_WINDOW_MAX_DAYS = 365       # 1 year after PE

# Temporal categories
TEMPORAL_CATEGORIES = {
    'preexisting_remote': (-float('inf'), -30),
    'preexisting_recent': (-30, -7),
    'antecedent': (-7, 0),
    'index_concurrent': (0, 1),
    'early_complication': (1, 7),
    'late_complication': (7, 30),
    'follow_up': (30, float('inf')),
}

# PE diagnosis codes
PE_ICD9_CODES = ['415.1', '415.11', '415.12', '415.13', '415.19']
PE_ICD10_CODES = ['I26', 'I26.0', 'I26.01', 'I26.02', 'I26.09', 'I26.9', 'I26.90', 'I26.92', 'I26.93', 'I26.94', 'I26.99']

# Exclusion patterns (administrative codes)
EXCLUDE_ICD10_PREFIXES = ['Z00', 'Z01', 'Z02', 'Z03', 'Z04', 'Z05', 'Z06', 'Z07', 'Z08', 'Z09', 'Z10', 'Z11', 'Z12', 'Z13']
EXCLUDE_ICD9_PREFIXES = ['V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82']
