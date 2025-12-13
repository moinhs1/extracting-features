"""
Vocabulary Mapper
=================

Maps procedure codes to CCS categories and SNOMED concepts.

Mapping strategies:
1. Direct: CPT/HCPCS/ICD -> CCS via crosswalk
2. Fuzzy: Procedure name -> CCS via text similarity
3. LLM: Ambiguous EPIC codes -> LLM classification (see llm_classifier.py)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from difflib import SequenceMatcher
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import MAPPING_CONFIG, SILVER_DIR, BRONZE_DIR
from data.vocabularies.setup_vocabularies import load_ccs_crosswalk


# =============================================================================
# CCS LOOKUP CACHE
# =============================================================================

_ccs_lookup: Optional[Dict[str, Dict]] = None


def _get_ccs_lookup() -> Dict[str, Dict]:
    """Get or build CCS lookup dictionary."""
    global _ccs_lookup

    if _ccs_lookup is None:
        df = load_ccs_crosswalk()
        _ccs_lookup = {}

        for _, row in df.iterrows():
            _ccs_lookup[str(row['cpt_code'])] = {
                'ccs_category': row['ccs_category'],
                'ccs_description': row['ccs_description'],
            }

    return _ccs_lookup


# =============================================================================
# DIRECT MAPPING
# =============================================================================

def map_procedure_code(
    code: str,
    code_type: str
) -> Optional[Dict]:
    """
    Map a procedure code directly to CCS.

    Args:
        code: Procedure code
        code_type: Code type (CPT, HCPCS, ICD10, etc.)

    Returns:
        Dictionary with mapping result, or None
    """
    if not code or pd.isna(code):
        return None

    code = str(code).strip()
    lookup = _get_ccs_lookup()

    if code in lookup:
        result = lookup[code].copy()
        result['mapping_method'] = 'direct'
        result['mapping_confidence'] = 1.0
        return result

    return None


# =============================================================================
# FUZZY MATCHING
# =============================================================================

def _get_ccs_descriptions() -> List[Tuple[str, str, str]]:
    """Get list of (ccs_category, ccs_description, normalized_description)."""
    df = load_ccs_crosswalk()

    descriptions = []
    for _, row in df.iterrows():
        desc = str(row['ccs_description']).lower().strip()
        descriptions.append((
            row['ccs_category'],
            row['ccs_description'],
            desc,
        ))

    return descriptions


def fuzzy_match_procedure(
    procedure_name: str,
    threshold: float = None
) -> Optional[Dict]:
    """
    Fuzzy match procedure name to CCS category.

    Args:
        procedure_name: Procedure name/description
        threshold: Minimum similarity threshold

    Returns:
        Dictionary with mapping result, or None
    """
    if not procedure_name or pd.isna(procedure_name):
        return None

    threshold = threshold or MAPPING_CONFIG.fuzzy_match_threshold
    name_lower = str(procedure_name).lower().strip()

    descriptions = _get_ccs_descriptions()

    best_match = None
    best_score = 0.0

    for ccs_cat, ccs_desc, norm_desc in descriptions:
        # Use SequenceMatcher for fuzzy matching
        score = SequenceMatcher(None, name_lower, norm_desc).ratio()

        if score > best_score:
            best_score = score
            best_match = {
                'ccs_category': ccs_cat,
                'ccs_description': ccs_desc,
                'mapping_method': 'fuzzy',
                'mapping_confidence': score,
            }

    if best_match and best_score >= threshold:
        return best_match

    return None


# =============================================================================
# BATCH MAPPING
# =============================================================================

def map_procedures_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map batch of procedures to CCS categories.

    Args:
        df: DataFrame with code, code_type, procedure_name columns

    Returns:
        DataFrame with mapping columns added
    """
    df = df.copy()

    # Initialize mapping columns
    df['ccs_category'] = None
    df['ccs_description'] = None
    df['snomed_concept_id'] = None
    df['snomed_preferred_term'] = None
    df['mapping_method'] = None
    df['mapping_confidence'] = None

    # Try direct mapping first
    for idx, row in df.iterrows():
        result = map_procedure_code(row['code'], row['code_type'])

        if result:
            df.at[idx, 'ccs_category'] = result['ccs_category']
            df.at[idx, 'ccs_description'] = result['ccs_description']
            df.at[idx, 'mapping_method'] = result['mapping_method']
            df.at[idx, 'mapping_confidence'] = result['mapping_confidence']
            continue

        # Try fuzzy matching for unmapped
        if row['code_type'] == 'EPIC' or result is None:
            fuzzy_result = fuzzy_match_procedure(row.get('procedure_name'))

            if fuzzy_result:
                df.at[idx, 'ccs_category'] = fuzzy_result['ccs_category']
                df.at[idx, 'ccs_description'] = fuzzy_result['ccs_description']
                df.at[idx, 'mapping_method'] = fuzzy_result['mapping_method']
                df.at[idx, 'mapping_confidence'] = fuzzy_result['mapping_confidence']

    return df


# =============================================================================
# MAIN MAPPING PIPELINE
# =============================================================================

def run_vocabulary_mapping(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Run vocabulary mapping pipeline: Bronze -> Silver.

    Args:
        input_path: Path to canonical_procedures.parquet
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with mapped procedures
    """
    print("=" * 60)
    print("Vocabulary Mapping: Bronze -> Silver")
    print("=" * 60)

    # Load data
    if input_path is None:
        filename = "canonical_procedures_test.parquet" if test_mode else "canonical_procedures.parquet"
        input_path = BRONZE_DIR / filename

    print(f"\n1. Loading canonical procedures: {input_path}")
    df = pd.read_parquet(input_path)

    if test_mode:
        sample_empis = df['empi'].unique()[:100]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")
    print(f"   Unique codes: {df['code'].nunique():,}")

    # Get unique procedures for mapping
    print("\n2. Extracting unique procedures...")
    unique_procs = df[['code', 'code_type', 'procedure_name']].drop_duplicates()
    print(f"   Unique procedure strings: {len(unique_procs):,}")

    # Map procedures
    print("\n3. Mapping to CCS categories...")
    mapped = map_procedures_batch(unique_procs)

    # Calculate statistics
    mapped_count = mapped['ccs_category'].notna().sum()
    mapping_rate = mapped_count / len(mapped) * 100
    print(f"   Mapping rate: {mapping_rate:.1f}%")

    # Merge back to full dataset
    print("\n4. Merging mappings to full dataset...")
    result = df.merge(
        mapped[['code', 'code_type', 'ccs_category', 'ccs_description',
                'snomed_concept_id', 'snomed_preferred_term',
                'mapping_method', 'mapping_confidence']],
        on=['code', 'code_type'],
        how='left'
    )

    # Save failures for review
    failures = mapped[mapped['ccs_category'].isna()]
    if len(failures) > 0:
        failure_path = SILVER_DIR / "mapping_failures.parquet"
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        failures.to_parquet(failure_path, index=False)
        print(f"   Mapping failures saved: {failure_path}")

    # Save output
    if output_path is None:
        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        filename = "mapped_procedures_test.parquet" if test_mode else "mapped_procedures.parquet"
        output_path = SILVER_DIR / filename

    print(f"\n5. Saving to: {output_path}")
    result.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("Vocabulary Mapping Complete!")
    print("=" * 60)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Map procedures to CCS/SNOMED")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    run_vocabulary_mapping(test_mode=args.test)
