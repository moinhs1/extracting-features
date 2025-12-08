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

# Prg.txt columns (same format as Hnp.txt)
PRG_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Default paths for Prg extractor
PRG_INPUT_PATH = DATA_DIR / 'Prg.txt'
PRG_OUTPUT_PATH = OUTPUT_DIR / 'discovery' / 'prg_vitals_raw.parquet'
PRG_CHUNKS_DIR = OUTPUT_DIR / 'discovery' / 'prg_chunks'

# Layer output paths
LAYER1_OUTPUT_DIR = OUTPUT_DIR / 'layer1'
LAYER2_OUTPUT_DIR = OUTPUT_DIR / 'layer2'
LAYER3_OUTPUT_DIR = OUTPUT_DIR / 'layer3'
LAYER4_OUTPUT_DIR = OUTPUT_DIR / 'layer4'
LAYER5_OUTPUT_DIR = OUTPUT_DIR / 'layer5'

# Layer 1 outputs
CANONICAL_VITALS_PATH = LAYER1_OUTPUT_DIR / 'canonical_vitals.parquet'

# Layer 2 outputs
HOURLY_GRID_PATH = LAYER2_OUTPUT_DIR / 'hourly_grid.parquet'
HOURLY_TENSORS_PATH = LAYER2_OUTPUT_DIR / 'hourly_tensors.h5'

# Temporal window constants
WINDOW_MIN_HOURS = -24   # 24 hours before PE
WINDOW_MAX_HOURS = 720   # 30 days after PE
TOTAL_HOURS = 745        # Total hours in window
