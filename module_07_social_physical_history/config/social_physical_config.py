"""
Module 7: Social & Physical History Configuration
=================================================
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/home/moin/TDA_11_25")
MODULE_ROOT = PROJECT_ROOT / "module_07_social_physical_history"
DATA_DIR = PROJECT_ROOT / "Data"

# Module 1 outputs
MODULE_1_OUTPUT = PROJECT_ROOT / "module_1_core_infrastructure" / "outputs"
PATIENT_TIMELINES_PKL = MODULE_1_OUTPUT / "patient_timelines.pkl"

# Input data
PHY_FILE = DATA_DIR / "Phy.txt"
HNP_FILE = DATA_DIR / "Hnp.txt"
PRG_FILE = DATA_DIR / "Prg.txt"

# Output directories
BRONZE_DIR = MODULE_ROOT / "data" / "bronze"
SILVER_DIR = MODULE_ROOT / "data" / "silver"
GOLD_DIR = MODULE_ROOT / "data" / "gold"

# =============================================================================
# STALENESS THRESHOLDS (days)
# =============================================================================

STALENESS_THRESHOLDS: Dict[str, float] = {
    'bmi': 180,
    'weight': 90,
    'height': 3650,  # 10 years
    'bsa': 180,
    'smoking_status': 365,
    'alcohol_status': 365,
    'drug_use_status': 365,
    'ivdu': float('inf'),  # Never expires
    'pain_score': 7,
    'fatigue': 7,
    'kps': 30,
    'functional_status': 30,
    'fev1': 730,  # 2 years
    'fvc': 730,
}

# =============================================================================
# CONCEPT NAMES
# =============================================================================

BODY_CONCEPTS = ['BMI', 'Weight', 'Height', 'Body Surface Area (BSA)']

SMOKING_STATUS_CONCEPTS = [
    'Smoking Tobacco Use-Never Smoker',
    'Smoking Tobacco Use-Former Smoker',
    'Smoking Tobacco Use-Current Every Day Smoker',
    'Smoking Tobacco Use-Current Some Day Smoker',
    'Smoking Tobacco Use-Heavy Tobacco Smoker',
    'Smoking Tobacco Use-Light Tobacco Smoker',
    'Tobacco User-Never',
    'Tobacco User-Quit',
    'Tobacco User-Yes',
]

SMOKING_QUANT_CONCEPTS = [
    'Tobacco Pack Per Day',
    'Tobacco Used Years',
    'Smoking Quit Date',
    'Smoking Start Date',
]

ALCOHOL_STATUS_CONCEPTS = [
    'Alcohol User-Yes',
    'Alcohol User-No',
    'Alcohol User-Never',
    'Alcohol User-Not Currently',
]

ALCOHOL_QUANT_CONCEPTS = [
    'Alcohol Drinks Per Week',
    'Alcohol Oz Per Week',
]

DRUG_USE_CONCEPTS = [
    'Drug User (Illicit)- Yes',
    'Drug User (Illicit)- No',
    'Drug User (Illicit)- Never',
    'Drug User (Illicit)- Not Currently',
    'Drug User IV',
]

PAIN_CONCEPTS = [
    'Pain Score EPIC (0-10)',
    'Pain Level (0-10)',
    'Pain 0-10',
    'Fatigue (0-10)',
]

FUNCTIONAL_CONCEPTS = [
    'KPS (Karnofsky performance status)',
    'Functional Status Screen',
]

# All relevant concepts
ALL_RELEVANT_CONCEPTS = (
    BODY_CONCEPTS +
    SMOKING_STATUS_CONCEPTS +
    SMOKING_QUANT_CONCEPTS +
    ALCOHOL_STATUS_CONCEPTS +
    ALCOHOL_QUANT_CONCEPTS +
    DRUG_USE_CONCEPTS +
    PAIN_CONCEPTS +
    FUNCTIONAL_CONCEPTS
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all required output directories."""
    for dir_path in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
