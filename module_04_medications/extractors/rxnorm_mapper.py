# /home/moin/TDA_11_25/module_04_medications/extractors/rxnorm_mapper.py
"""
RxNorm Medication Mapper
========================

Maps RPDR medication strings to RxNorm concepts using cascading strategies:
1. Exact match
2. Fuzzy match (Levenshtein)
3. Ingredient extraction
4. LLM-assisted (optional)

Uses rxnorm.db SQLite database created by setup_rxnorm.py.
"""

import sqlite3
import re
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from functools import lru_cache
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import RXNORM_DB, RXNORM_CONFIG

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Fuzzy matching disabled.")


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

_db_connection: Optional[sqlite3.Connection] = None


def get_db_connection() -> sqlite3.Connection:
    """Get or create database connection."""
    global _db_connection
    if _db_connection is None:
        _db_connection = sqlite3.connect(RXNORM_DB)
        _db_connection.row_factory = sqlite3.Row
    return _db_connection


def close_db_connection():
    """Close database connection."""
    global _db_connection
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None


# =============================================================================
# EXACT MATCHING
# =============================================================================

@lru_cache(maxsize=50000)
def exact_match(medication_string: str) -> Optional[Dict[str, Any]]:
    """
    Attempt exact match against RxNorm names.

    Args:
        medication_string: Medication name to match

    Returns:
        Dictionary with rxcui, rxnorm_name, tty, or None if not found
    """
    if not medication_string:
        return None

    conn = get_db_connection()
    normalized = medication_string.lower().strip()

    # Try exact match on STR column (case-insensitive)
    cursor = conn.execute("""
        SELECT RXCUI, STR, TTY
        FROM RXNCONSO
        WHERE LOWER(STR) = ?
          AND SAB = 'RXNORM'
          AND SUPPRESS = 'N'
        ORDER BY
            CASE TTY
                WHEN 'IN' THEN 1
                WHEN 'PIN' THEN 2
                WHEN 'SCD' THEN 3
                WHEN 'SBD' THEN 4
                WHEN 'SCDC' THEN 5
                WHEN 'SBDC' THEN 6
                ELSE 10
            END
        LIMIT 1
    """, (normalized,))

    row = cursor.fetchone()
    if row:
        return {
            'rxcui': row['RXCUI'],
            'rxnorm_name': row['STR'],
            'rxnorm_tty': row['TTY'],
            'mapping_method': 'exact',
            'mapping_confidence': 1.0,
        }

    return None


# =============================================================================
# FUZZY MATCHING
# =============================================================================

# Cache of RxNorm names for fuzzy matching
_rxnorm_names_cache: Optional[List[Tuple[str, str, str]]] = None


def _load_rxnorm_names() -> List[Tuple[str, str, str]]:
    """Load RxNorm names for fuzzy matching."""
    global _rxnorm_names_cache

    if _rxnorm_names_cache is None:
        conn = get_db_connection()

        # Load ingredient names (most important for matching)
        cursor = conn.execute("""
            SELECT DISTINCT RXCUI, STR, TTY
            FROM RXNCONSO
            WHERE SAB = 'RXNORM'
              AND SUPPRESS = 'N'
              AND TTY IN ('IN', 'PIN', 'MIN')
        """)

        _rxnorm_names_cache = [(row['RXCUI'], row['STR'], row['TTY']) for row in cursor]

    return _rxnorm_names_cache


def fuzzy_match(
    medication_string: str,
    threshold: float = None
) -> Optional[Dict[str, Any]]:
    """
    Fuzzy match against RxNorm ingredient names.

    Args:
        medication_string: Medication name to match
        threshold: Minimum similarity score (0-100), default from config

    Returns:
        Dictionary with match info or None
    """
    if not RAPIDFUZZ_AVAILABLE:
        return None

    if not medication_string:
        return None

    if threshold is None:
        threshold = RXNORM_CONFIG.fuzzy_match_threshold * 100  # Convert to 0-100

    normalized = medication_string.lower().strip()

    # Extract first word(s) as likely drug name
    # Remove numbers and common suffixes
    drug_name = re.sub(r'\d+.*$', '', normalized).strip()
    drug_name = re.sub(r'\s+(tablet|capsule|solution|injection|mg|mcg|ml).*$', '', drug_name)

    if len(drug_name) < 3:
        return None

    # Load RxNorm names
    rxnorm_names = _load_rxnorm_names()
    name_to_info = {name.lower(): (rxcui, name, tty) for rxcui, name, tty in rxnorm_names}

    # Fuzzy match
    matches = process.extract(
        drug_name,
        list(name_to_info.keys()),
        scorer=fuzz.ratio,
        limit=1
    )

    if matches and matches[0][1] >= threshold:
        matched_name = matches[0][0]
        score = matches[0][1]
        rxcui, original_name, tty = name_to_info[matched_name]

        return {
            'rxcui': rxcui,
            'rxnorm_name': original_name,
            'rxnorm_tty': tty,
            'mapping_method': 'fuzzy',
            'mapping_confidence': score / 100,
        }

    return None


# =============================================================================
# INGREDIENT EXTRACTION
# =============================================================================

def ingredient_match(medication_string: str) -> Optional[Dict[str, Any]]:
    """
    Extract ingredient from medication string and match.

    Args:
        medication_string: Full medication string

    Returns:
        Dictionary with ingredient match info or None
    """
    if not medication_string:
        return None

    # Extract potential drug name (before numbers)
    text = medication_string.lower().strip()

    # Strategy 1: First word before number
    match = re.match(r'^([a-z]+(?:\s+[a-z]+)?)', text)
    if match:
        candidate = match.group(1).strip()

        # Remove common salt forms
        candidate = re.sub(r'\s+(sodium|potassium|hcl|sulfate|citrate|tartrate|bitartrate|chloride|pf)$', '', candidate)

        # Try exact match on ingredient
        result = exact_match(candidate)
        if result:
            return {
                'rxcui': result['rxcui'],
                'rxnorm_name': result['rxnorm_name'],
                'rxnorm_tty': result['rxnorm_tty'],
                'ingredient_rxcui': result['rxcui'],
                'ingredient_name': result['rxnorm_name'],
                'mapping_method': 'ingredient',
                'mapping_confidence': 0.85,
            }

        # Try fuzzy on just the first word
        first_word = candidate.split()[0] if candidate else ''
        if len(first_word) >= 4:
            result = fuzzy_match(first_word, threshold=90)
            if result:
                result['mapping_method'] = 'ingredient'
                result['mapping_confidence'] = result['mapping_confidence'] * 0.9
                result['ingredient_rxcui'] = result['rxcui']
                result['ingredient_name'] = result['rxnorm_name']
                return result

    return None


# =============================================================================
# INGREDIENT LOOKUP
# =============================================================================

@lru_cache(maxsize=10000)
def get_ingredient_for_rxcui(rxcui: str) -> Optional[Dict[str, str]]:
    """
    Look up ingredient for a given RXCUI.

    Args:
        rxcui: RxNorm concept ID

    Returns:
        Dictionary with ingredient_rxcui and ingredient_name
    """
    conn = get_db_connection()

    # Check if already an ingredient
    cursor = conn.execute("""
        SELECT RXCUI, STR
        FROM RXNCONSO
        WHERE RXCUI = ?
          AND SAB = 'RXNORM'
          AND TTY = 'IN'
        LIMIT 1
    """, (rxcui,))

    row = cursor.fetchone()
    if row:
        return {
            'ingredient_rxcui': row['RXCUI'],
            'ingredient_name': row['STR'],
        }

    # Look up via relationships
    cursor = conn.execute("""
        SELECT r.RXCUI2 as ingredient_rxcui, c.STR as ingredient_name
        FROM RXNREL r
        JOIN RXNCONSO c ON r.RXCUI2 = c.RXCUI
        WHERE r.RXCUI1 = ?
          AND r.RELA = 'has_ingredient'
          AND c.SAB = 'RXNORM'
          AND c.TTY = 'IN'
        LIMIT 1
    """, (rxcui,))

    row = cursor.fetchone()
    if row:
        return {
            'ingredient_rxcui': row['ingredient_rxcui'],
            'ingredient_name': row['ingredient_name'],
        }

    return None


# =============================================================================
# FULL MAPPING PIPELINE
# =============================================================================

def map_medication(medication_string: str) -> Dict[str, Any]:
    """
    Full mapping pipeline: exact → fuzzy → ingredient → failed.

    Args:
        medication_string: RPDR medication string

    Returns:
        Dictionary with mapping results
    """
    base_result = {
        'original_string': medication_string,
        'rxcui': None,
        'rxnorm_name': None,
        'rxnorm_tty': None,
        'ingredient_rxcui': None,
        'ingredient_name': None,
        'mapping_method': 'failed',
        'mapping_confidence': 0.0,
    }

    if not medication_string:
        return base_result

    # Step 1: Exact match
    result = exact_match(medication_string)
    if result:
        base_result.update(result)
        # Look up ingredient
        ing = get_ingredient_for_rxcui(result['rxcui'])
        if ing:
            base_result.update(ing)
        return base_result

    # Step 2: Fuzzy match
    result = fuzzy_match(medication_string)
    if result:
        base_result.update(result)
        ing = get_ingredient_for_rxcui(result['rxcui'])
        if ing:
            base_result.update(ing)
        return base_result

    # Step 3: Ingredient extraction
    result = ingredient_match(medication_string)
    if result:
        base_result.update(result)
        return base_result

    # Step 4: Failed
    return base_result


# =============================================================================
# BATCH MAPPING
# =============================================================================

def map_vocabulary(
    vocabulary_df,
    progress_callback=None
) -> 'pd.DataFrame':
    """
    Map entire medication vocabulary.

    Args:
        vocabulary_df: DataFrame with 'original_string' column
        progress_callback: Optional callback(current, total) for progress

    Returns:
        DataFrame with mapping results
    """
    import pandas as pd

    results = []
    total = len(vocabulary_df)

    for i, row in vocabulary_df.iterrows():
        med_string = row['original_string']
        result = map_medication(med_string)
        results.append(result)

        if progress_callback and (i + 1) % 1000 == 0:
            progress_callback(i + 1, total)

    return pd.DataFrame(results)


# =============================================================================
# STATISTICS
# =============================================================================

def get_mapping_stats(mapped_df) -> Dict[str, Any]:
    """
    Calculate mapping statistics.

    Args:
        mapped_df: DataFrame with mapping results

    Returns:
        Dictionary with statistics
    """
    total = len(mapped_df)
    by_method = mapped_df['mapping_method'].value_counts()

    return {
        'total': total,
        'exact': by_method.get('exact', 0),
        'fuzzy': by_method.get('fuzzy', 0),
        'ingredient': by_method.get('ingredient', 0),
        'failed': by_method.get('failed', 0),
        'success_rate': (total - by_method.get('failed', 0)) / total if total > 0 else 0,
    }
