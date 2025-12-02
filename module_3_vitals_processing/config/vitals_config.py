"""Configuration constants for vitals processing."""
from pathlib import Path

# Paths
DATA_DIR = Path("/home/moin/TDA_11_25/Data")
OUTPUT_DIR = Path("/home/moin/TDA_11_25/module_3_vitals_processing/outputs")
MODULE1_OUTPUT_DIR = Path("/home/moin/TDA_11_25/module_1_core_infrastructure/outputs")

# Phy.txt columns
PHY_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Date', 'Concept_Name',
    'Code_Type', 'Code', 'Result', 'Units', 'Provider', 'Clinic',
    'Hospital', 'Inpatient_Outpatient', 'Encounter_number'
]

# Vital sign concepts to extract from Phy.txt
VITAL_CONCEPTS = {
    'Temperature': 'TEMP',
    'Pulse': 'HR',
    'Systolic-Epic': 'SBP',
    'Diastolic-Epic': 'DBP',
    'Blood Pressure-Epic': 'BP',  # Will be parsed into SBP/DBP
    'Systolic-LFA3959.1': 'SBP',
    'Diastolic-LFA3959.2': 'DBP',
    'Systolic/Diastolic-LFA3959.0': 'BP',
    'O2 Saturation-SPO2': 'SPO2',
    'O2 Saturation%': 'SPO2',
    'Respiratory rate': 'RR',
    'Weight': 'WEIGHT',
    'Height': 'HEIGHT',
    'BMI': 'BMI',
}

# Canonical vital names
CANONICAL_VITALS = ['HR', 'SBP', 'DBP', 'RR', 'SPO2', 'TEMP', 'WEIGHT', 'HEIGHT', 'BMI']

# Processing config
CHUNK_SIZE = 500_000  # Rows per chunk for large file processing

# Hnp.txt columns
HNP_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Default output paths for Hnp extractor
HNP_INPUT_PATH = DATA_DIR / 'Hnp.txt'
HNP_OUTPUT_PATH = OUTPUT_DIR / 'discovery' / 'hnp_vitals_raw.parquet'
