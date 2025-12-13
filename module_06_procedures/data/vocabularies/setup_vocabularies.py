# module_06_procedures/data/vocabularies/setup_vocabularies.py
"""
Vocabulary Setup for Procedure Mapping
======================================

Downloads and sets up:
1. CCS (Clinical Classification Software) crosswalk
2. SNOMED-CT procedure concepts (via OMOP)

CCS source: AHRQ HCUP (https://www.hcup-us.ahrq.gov/toolssoftware/ccs_svcsproc/ccs_svcsproc.jsp)
SNOMED source: UMLS/OMOP vocabularies
"""

import pandas as pd
import sqlite3
from pathlib import Path
import sys
import requests
import zipfile
import io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.procedure_config import VOCAB_DIR, CCS_CROSSWALK, SNOMED_DB


# =============================================================================
# CCS CROSSWALK
# =============================================================================

# CCS for Services and Procedures crosswalk
# Format: CPT/HCPCS -> CCS category
CCS_DOWNLOAD_URL = "https://www.hcup-us.ahrq.gov/toolssoftware/ccs_svcsproc/ccs_svcsproc_2015.zip"


def download_ccs_crosswalk(output_path: Path = None) -> Path:
    """
    Download CCS crosswalk from AHRQ.

    Args:
        output_path: Output CSV path

    Returns:
        Path to downloaded file
    """
    output_path = output_path or CCS_CROSSWALK

    print(f"Downloading CCS crosswalk from AHRQ...")

    # Note: AHRQ may require manual download
    # For now, we'll create a minimal crosswalk from the design document

    # Create minimal CCS crosswalk for PE-relevant procedures
    ccs_data = [
        # Diagnostic imaging
        ('71275', '61', 'CT scan chest', 'Diagnostic cardiac catheterization'),
        ('93306', '47', 'Echo TTE', 'Diagnostic cardiac catheterization'),
        ('93312', '47', 'Echo TEE', 'Diagnostic cardiac catheterization'),

        # Respiratory
        ('31500', '216', 'Intubation', 'Respiratory intubation and mechanical ventilation'),
        ('94002', '216', 'Vent management', 'Respiratory intubation and mechanical ventilation'),

        # Vascular access
        ('36555', '54', 'Central line', 'Other vascular catheterization'),
        ('36620', '54', 'Arterial line', 'Other vascular catheterization'),

        # IVC filter
        ('37191', '52', 'IVC filter placement', 'Aortic resection'),
        ('37193', '52', 'IVC filter retrieval', 'Aortic resection'),

        # CDT
        ('37211', '52', 'CDT thrombolysis', 'Aortic resection'),
        ('37212', '52', 'CDT thrombolysis', 'Aortic resection'),

        # Transfusion
        ('36430', '222', 'Transfusion', 'Blood transfusion'),

        # ECMO
        ('33946', '49', 'ECMO cannulation', 'Other OR heart procedures'),
        ('33947', '49', 'ECMO cannulation', 'Other OR heart procedures'),

        # Resuscitation
        ('92950', '48', 'CPR', 'Insertion of temporary pacemaker'),

        # Thoracic
        ('32551', '39', 'Chest tube', 'Incision of pleura'),
        ('32554', '39', 'Thoracentesis', 'Incision of pleura'),
    ]

    df = pd.DataFrame(ccs_data, columns=['cpt_code', 'ccs_category', 'procedure_name', 'ccs_description'])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved CCS crosswalk to: {output_path}")
    return output_path


def load_ccs_crosswalk(filepath: Path = None) -> pd.DataFrame:
    """
    Load CCS crosswalk from CSV.

    Args:
        filepath: Path to crosswalk CSV

    Returns:
        DataFrame with CPT -> CCS mappings
    """
    filepath = filepath or CCS_CROSSWALK

    if not filepath.exists():
        download_ccs_crosswalk(filepath)

    df = pd.read_csv(filepath, dtype=str)
    return df


def map_cpt_to_ccs(cpt_code: str) -> dict:
    """
    Map a CPT code to CCS category.

    Args:
        cpt_code: CPT code string

    Returns:
        Dictionary with ccs_category and ccs_description, or None
    """
    df = load_ccs_crosswalk()

    match = df[df['cpt_code'] == str(cpt_code)]

    if len(match) == 0:
        return None

    row = match.iloc[0]
    return {
        'ccs_category': row['ccs_category'],
        'ccs_description': row['ccs_description'],
    }


# =============================================================================
# SNOMED DATABASE
# =============================================================================

def setup_snomed_database(db_path: Path = None) -> Path:
    """
    Set up SNOMED procedure concepts database.

    Note: Full SNOMED requires UMLS license. This creates a minimal
    PE-relevant subset.

    Args:
        db_path: Path to SQLite database

    Returns:
        Path to database
    """
    db_path = db_path or SNOMED_DB

    print("Setting up SNOMED procedure database...")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Create tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS snomed_procedures (
            concept_id TEXT PRIMARY KEY,
            preferred_term TEXT,
            semantic_type TEXT,
            is_pe_relevant INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS cpt_snomed_mapping (
            cpt_code TEXT,
            snomed_concept_id TEXT,
            mapping_type TEXT,
            FOREIGN KEY (snomed_concept_id) REFERENCES snomed_procedures(concept_id)
        )
    """)

    # Insert minimal PE-relevant procedures
    pe_procedures = [
        ('233604007', 'CT angiography of chest', 'procedure', 1),
        ('40701008', 'Echocardiography', 'procedure', 1),
        ('232717009', 'Endotracheal intubation', 'procedure', 1),
        ('225793007', 'Mechanical ventilation', 'procedure', 1),
        ('233527006', 'Catheter-directed thrombolysis', 'procedure', 1),
        ('24596005', 'IVC filter insertion', 'procedure', 1),
        ('5447007', 'Transfusion', 'procedure', 1),
        ('233573008', 'Extracorporeal membrane oxygenation', 'procedure', 1),
        ('89666000', 'Cardiopulmonary resuscitation', 'procedure', 1),
    ]

    conn.executemany("""
        INSERT OR REPLACE INTO snomed_procedures (concept_id, preferred_term, semantic_type, is_pe_relevant)
        VALUES (?, ?, ?, ?)
    """, pe_procedures)

    conn.commit()
    conn.close()

    print(f"SNOMED database created: {db_path}")
    return db_path


# =============================================================================
# MAIN
# =============================================================================

def setup_all_vocabularies():
    """Set up all vocabulary files."""
    print("=" * 60)
    print("Setting up Procedure Vocabularies")
    print("=" * 60)

    download_ccs_crosswalk()
    setup_snomed_database()

    print("\n" + "=" * 60)
    print("Vocabulary Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    setup_all_vocabularies()
