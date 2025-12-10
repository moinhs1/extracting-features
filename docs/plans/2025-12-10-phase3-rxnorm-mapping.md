# Phase 3: RxNorm Mapping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Map medication vocabulary (~15K unique strings) to RxNorm concepts using multi-step pipeline (exact → fuzzy → ingredient → LLM), achieving ≥85% mapping rate.

**Architecture:** Load medication vocabulary from Phase 2, apply cascading match strategies against rxnorm.db SQLite, benchmark local LLMs for ambiguous cases, output silver parquet with RxCUI and ingredient mappings.

**Tech Stack:** Python 3.12, sqlite3, rapidfuzz (fuzzy matching), pandas, pytest

**Depends on:** Phase 2 complete (medication_vocabulary.parquet exists)

---

## Task 1: Create RxNorm Mapper Test Infrastructure

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/tests/test_rxnorm_mapper.py`

**Step 1: Create test file with failing tests**

```python
# /home/moin/TDA_11_25/module_04_medications/tests/test_rxnorm_mapper.py
"""Tests for RxNorm medication mapping."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExactMatch:
    """Test exact string matching to RxNorm."""

    def test_exact_match_aspirin(self):
        """Exact match for simple drug name."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("aspirin")

        assert result is not None
        assert result['rxcui'] is not None
        assert 'aspirin' in result['rxnorm_name'].lower()

    def test_exact_match_not_found(self):
        """Return None for non-existent drug."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("notarealdrug12345")

        assert result is None

    def test_exact_match_heparin(self):
        """Exact match for heparin."""
        from extractors.rxnorm_mapper import exact_match

        result = exact_match("heparin")

        assert result is not None
        assert result['rxcui'] == '5224'


class TestFuzzyMatch:
    """Test fuzzy string matching."""

    def test_fuzzy_match_typo(self):
        """Fuzzy match handles minor typos."""
        from extractors.rxnorm_mapper import fuzzy_match

        # Intentional typo
        result = fuzzy_match("aspirn")

        assert result is not None
        assert 'aspirin' in result['rxnorm_name'].lower()

    def test_fuzzy_match_threshold(self):
        """Fuzzy match respects similarity threshold."""
        from extractors.rxnorm_mapper import fuzzy_match

        # Too different - should not match
        result = fuzzy_match("xyz123")

        assert result is None


class TestIngredientExtraction:
    """Test ingredient-level matching."""

    def test_extract_ingredient_from_product(self):
        """Extract ingredient from product name."""
        from extractors.rxnorm_mapper import ingredient_match

        result = ingredient_match("Aspirin 325mg tablet")

        assert result is not None
        assert result['ingredient_name'].lower() == 'aspirin'

    def test_extract_ingredient_heparin_sodium(self):
        """Extract heparin from complex string."""
        from extractors.rxnorm_mapper import ingredient_match

        result = ingredient_match("Heparin sodium 5000 unit/ml injection")

        assert result is not None
        assert 'heparin' in result['ingredient_name'].lower()


class TestFullMappingPipeline:
    """Test complete mapping pipeline."""

    def test_map_medication_exact(self):
        """Full pipeline finds exact match first."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("heparin")

        assert result['rxcui'] is not None
        assert result['mapping_method'] == 'exact'

    def test_map_medication_fuzzy(self):
        """Full pipeline falls back to fuzzy."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("heparn sodium")  # Typo

        assert result['rxcui'] is not None
        assert result['mapping_method'] in ['exact', 'fuzzy', 'ingredient']

    def test_map_medication_failed(self):
        """Full pipeline returns failed for unmappable."""
        from extractors.rxnorm_mapper import map_medication

        result = map_medication("Supply Of Radiopharmaceutical Agent XYZ")

        assert result['mapping_method'] == 'failed'
```

**Step 2: Run tests to verify they fail**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_rxnorm_mapper.py -v
```

**Expected:** FAIL with "ModuleNotFoundError: No module named 'extractors.rxnorm_mapper'"

**Step 3: Commit failing tests**

```bash
git add module_04_medications/tests/test_rxnorm_mapper.py
git commit -m "test(module4): add RxNorm mapper tests

Add failing tests for:
- Exact string matching
- Fuzzy matching with threshold
- Ingredient extraction
- Full mapping pipeline

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Implement RxNorm Database Interface

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/rxnorm_mapper.py`

**Step 1: Create rxnorm_mapper.py with database interface**

```python
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
```

**Step 2: Install rapidfuzz if not present**

```bash
pip3 install rapidfuzz
```

**Step 3: Run tests to verify they pass**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_rxnorm_mapper.py -v
```

**Expected:** All tests PASS

**Step 4: Commit RxNorm mapper**

```bash
git add module_04_medications/extractors/rxnorm_mapper.py
git commit -m "feat(module4): implement RxNorm medication mapper

Add multi-step mapping pipeline:
- Exact match against RXNCONSO
- Fuzzy match using rapidfuzz
- Ingredient extraction from product names
- Ingredient lookup via RXNREL relationships

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Test Mapping on Sample Vocabulary

**Files:**
- Run: Interactive mapping test

**Step 1: Test mapping on sample medications**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import sys
sys.path.insert(0, 'module_04_medications')
from extractors.rxnorm_mapper import map_medication, get_db_connection

# Test sample medications from Med.txt
test_meds = [
    'Aspirin 325mg tablet',
    'Heparin sodium 5000 unit/ml injection',
    'Enoxaparin 100 mg/ml solution',
    'Warfarin 5mg tablet',
    'Apixaban 2.5mg tablet',
    'Rivaroxaban 20mg tablet',
    'Morphine sulfate 2 mg/ml solution',
    'Vancomycin 1gm injection',
    'Supply Of Radiopharmaceutical Agent',
    'Fentanyl citrate 100 mcg ampul',
]

print('RxNorm Mapping Test')
print('='*80)

for med in test_meds:
    result = map_medication(med)
    status = '✓' if result['rxcui'] else '✗'
    print(f'{status} {med[:45]:<45} → {result[\"mapping_method\"]:10} {result[\"ingredient_name\"] or \"\"}'[:80])
"
```

**Expected output:**
```
RxNorm Mapping Test
================================================================================
✓ Aspirin 325mg tablet                        → ingredient  aspirin
✓ Heparin sodium 5000 unit/ml injection       → ingredient  heparin
✓ Enoxaparin 100 mg/ml solution               → ingredient  enoxaparin
...
```

**Step 2: Test on actual vocabulary sample**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd
import sys
sys.path.insert(0, 'module_04_medications')
from extractors.rxnorm_mapper import map_medication, get_mapping_stats

# Load vocabulary
vocab = pd.read_parquet('module_04_medications/data/bronze/medication_vocabulary.parquet')
print(f'Total unique medications: {len(vocab):,}')

# Test on first 500
sample = vocab.head(500)
results = []
for _, row in sample.iterrows():
    results.append(map_medication(row['original_string']))

results_df = pd.DataFrame(results)
stats = get_mapping_stats(results_df)

print(f'\\nMapping Results (first 500):')
print(f'  Exact matches: {stats[\"exact\"]}')
print(f'  Fuzzy matches: {stats[\"fuzzy\"]}')
print(f'  Ingredient matches: {stats[\"ingredient\"]}')
print(f'  Failed: {stats[\"failed\"]}')
print(f'  Success rate: {stats[\"success_rate\"]*100:.1f}%')
"
```

---

## Task 4: Run Full Vocabulary Mapping

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/run_mapping.py`

**Step 1: Create mapping runner script**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/run_mapping.py
"""
Run RxNorm mapping on full medication vocabulary.
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import BRONZE_DIR, SILVER_DIR
from extractors.rxnorm_mapper import map_vocabulary, get_mapping_stats, close_db_connection


def main():
    print("=" * 60)
    print("Phase 3: RxNorm Mapping")
    print("=" * 60)

    # Load vocabulary
    vocab_path = BRONZE_DIR / "medication_vocabulary.parquet"
    print(f"\n1. Loading vocabulary from: {vocab_path}")
    vocab = pd.read_parquet(vocab_path)
    print(f"   Unique medications: {len(vocab):,}")

    # Map vocabulary
    print("\n2. Mapping to RxNorm...")
    start_time = datetime.now()

    def progress(current, total):
        pct = current / total * 100
        print(f"   Progress: {current:,}/{total:,} ({pct:.1f}%)", end='\r')

    mapped = map_vocabulary(vocab, progress_callback=progress)
    print()

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   Completed in {elapsed:.1f} seconds")

    # Merge with original vocabulary
    result = vocab.merge(
        mapped,
        left_on='original_string',
        right_on='original_string',
        how='left'
    )

    # Statistics
    stats = get_mapping_stats(mapped)
    print("\n" + "=" * 60)
    print("Mapping Statistics")
    print("=" * 60)
    print(f"   Total unique medications: {stats['total']:,}")
    print(f"   Exact matches: {stats['exact']:,} ({stats['exact']/stats['total']*100:.1f}%)")
    print(f"   Fuzzy matches: {stats['fuzzy']:,} ({stats['fuzzy']/stats['total']*100:.1f}%)")
    print(f"   Ingredient matches: {stats['ingredient']:,} ({stats['ingredient']/stats['total']*100:.1f}%)")
    print(f"   Failed: {stats['failed']:,} ({stats['failed']/stats['total']*100:.1f}%)")
    print(f"\n   SUCCESS RATE: {stats['success_rate']*100:.1f}%")
    print(f"   TARGET: ≥85%")
    print(f"   STATUS: {'PASS ✓' if stats['success_rate'] >= 0.85 else 'NEEDS IMPROVEMENT'}")

    # Save results
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    # Save mapped vocabulary
    mapped_path = SILVER_DIR / "mapped_vocabulary.parquet"
    result.to_parquet(mapped_path, index=False)
    print(f"\n3. Saved mapped vocabulary to: {mapped_path}")

    # Save failures for review
    failures = result[result['mapping_method'] == 'failed']
    if len(failures) > 0:
        failures_path = SILVER_DIR / "mapping_failures.parquet"
        failures.to_parquet(failures_path, index=False)
        print(f"   Saved {len(failures):,} failures to: {failures_path}")

    # Close connection
    close_db_connection()

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
```

**Step 2: Run full mapping**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/extractors/run_mapping.py 2>&1 | tee module_04_medications/mapping.log
```

**Expected runtime:** 5-15 minutes for ~15K unique medications

**Expected output:**
```
============================================================
Phase 3: RxNorm Mapping
============================================================

1. Loading vocabulary from: .../medication_vocabulary.parquet
   Unique medications: ~15,000

2. Mapping to RxNorm...
   Progress: 15,000/15,000 (100.0%)
   Completed in XXX.X seconds

============================================================
Mapping Statistics
============================================================
   Total unique medications: ~15,000
   Exact matches: X,XXX (XX.X%)
   Fuzzy matches: X,XXX (XX.X%)
   Ingredient matches: X,XXX (XX.X%)
   Failed: X,XXX (XX.X%)

   SUCCESS RATE: XX.X%
   TARGET: ≥85%
   STATUS: PASS ✓ / NEEDS IMPROVEMENT

3. Saved mapped vocabulary to: .../mapped_vocabulary.parquet
   Saved X,XXX failures to: .../mapping_failures.parquet

============================================================
Phase 3 Complete!
============================================================
```

**Step 3: Commit mapping runner**

```bash
git add module_04_medications/extractors/run_mapping.py module_04_medications/mapping.log
git commit -m "feat(module4): run full vocabulary RxNorm mapping

Mapped ~15K unique medications to RxNorm:
- Success rate: XX.X%
- Output: silver/mapped_vocabulary.parquet

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Apply Mapping to Canonical Records

**Files:**
- Create: `/home/moin/TDA_11_25/module_04_medications/extractors/apply_mapping.py`

**Step 1: Create script to apply mapping to full records**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/apply_mapping.py
"""
Apply RxNorm mapping to canonical records.

Creates silver/mapped_medications.parquet with RxNorm enrichment.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import BRONZE_DIR, SILVER_DIR


def main():
    print("=" * 60)
    print("Applying RxNorm Mapping to Canonical Records")
    print("=" * 60)

    # Load canonical records
    records_path = BRONZE_DIR / "canonical_records.parquet"
    print(f"\n1. Loading canonical records: {records_path}")
    records = pd.read_parquet(records_path)
    print(f"   Records: {len(records):,}")

    # Load mapped vocabulary
    vocab_path = SILVER_DIR / "mapped_vocabulary.parquet"
    print(f"\n2. Loading mapped vocabulary: {vocab_path}")
    vocab = pd.read_parquet(vocab_path)
    print(f"   Vocabulary entries: {len(vocab):,}")

    # Select mapping columns from vocabulary
    mapping_cols = [
        'original_string',
        'rxcui',
        'rxnorm_name',
        'rxnorm_tty',
        'ingredient_rxcui',
        'ingredient_name',
        'mapping_method',
        'mapping_confidence',
    ]
    vocab_mapping = vocab[mapping_cols].drop_duplicates(subset=['original_string'])

    # Merge mapping into records
    print("\n3. Merging mapping into records...")
    result = records.merge(
        vocab_mapping,
        on='original_string',
        how='left'
    )

    # Statistics
    mapped_count = result['rxcui'].notna().sum()
    mapping_rate = mapped_count / len(result) * 100

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"   Total records: {len(result):,}")
    print(f"   Records with RxNorm mapping: {mapped_count:,} ({mapping_rate:.1f}%)")

    # Save silver records
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SILVER_DIR / "mapped_medications.parquet"
    result.to_parquet(output_path, index=False)
    print(f"\n4. Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.1f} MB")

    print("\n" + "=" * 60)
    print("Silver Layer Complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
```

**Step 2: Run mapping application**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH python module_04_medications/extractors/apply_mapping.py
```

**Step 3: Verify silver output**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd
df = pd.read_parquet('module_04_medications/data/silver/mapped_medications.parquet')
print(f'Shape: {df.shape}')
print(f'\\nMapping method distribution:')
print(df['mapping_method'].value_counts())
print(f'\\nSample with mappings:')
print(df[df['rxcui'].notna()][['original_string', 'ingredient_name', 'mapping_method']].head(10))
"
```

**Step 4: Commit**

```bash
git add module_04_medications/extractors/apply_mapping.py
git commit -m "feat(module4): apply RxNorm mapping to canonical records

Create silver/mapped_medications.parquet with:
- All canonical record fields
- RxCUI and ingredient mappings
- Mapping method and confidence

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Update Extractors Package

**Files:**
- Modify: `/home/moin/TDA_11_25/module_04_medications/extractors/__init__.py`

**Step 1: Update __init__.py with exports**

```python
# /home/moin/TDA_11_25/module_04_medications/extractors/__init__.py
"""
Module 4 Extractors
===================

Layer 1 (Bronze): Canonical medication extraction
Layer 1→Silver: RxNorm mapping
"""

from .dose_parser import (
    extract_dose,
    extract_route,
    extract_frequency,
    extract_drug_name,
    parse_medication_string,
)

from .canonical_extractor import (
    load_med_chunk,
    iter_med_chunks,
    load_patient_timelines,
    filter_to_cohort,
    compute_hours_from_t0,
    filter_study_window,
    parse_medications,
    transform_to_canonical,
    extract_canonical_records,
    extract_vocabulary,
)

from .rxnorm_mapper import (
    exact_match,
    fuzzy_match,
    ingredient_match,
    map_medication,
    map_vocabulary,
    get_mapping_stats,
    get_ingredient_for_rxcui,
)

__all__ = [
    # Dose parsing
    'extract_dose',
    'extract_route',
    'extract_frequency',
    'extract_drug_name',
    'parse_medication_string',
    # Canonical extraction
    'load_med_chunk',
    'iter_med_chunks',
    'load_patient_timelines',
    'filter_to_cohort',
    'compute_hours_from_t0',
    'filter_study_window',
    'parse_medications',
    'transform_to_canonical',
    'extract_canonical_records',
    'extract_vocabulary',
    # RxNorm mapping
    'exact_match',
    'fuzzy_match',
    'ingredient_match',
    'map_medication',
    'map_vocabulary',
    'get_mapping_stats',
    'get_ingredient_for_rxcui',
]
```

**Step 2: Commit**

```bash
git add module_04_medications/extractors/__init__.py
git commit -m "chore(module4): update extractors package exports

Export all public functions from:
- dose_parser
- canonical_extractor
- rxnorm_mapper

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Final Validation

**Files:**
- Run: Validation checks

**Step 1: Run all Phase 3 tests**

```bash
cd /home/moin/TDA_11_25 && PYTHONPATH=module_04_medications:$PYTHONPATH pytest module_04_medications/tests/test_rxnorm_mapper.py -v
```

**Step 2: Validate mapping quality**

```bash
cd /home/moin/TDA_11_25 && python3 -c "
import pandas as pd

# Load silver records
df = pd.read_parquet('module_04_medications/data/silver/mapped_medications.parquet')

print('='*60)
print('Phase 3 Validation Report')
print('='*60)

# Mapping rate
mapped = df['rxcui'].notna().sum()
total = len(df)
rate = mapped / total * 100

print(f'\\nOverall Mapping:')
print(f'  Total records: {total:,}')
print(f'  Mapped records: {mapped:,}')
print(f'  Mapping rate: {rate:.1f}%')
print(f'  Target: ≥85%')
print(f'  Status: {\"PASS ✓\" if rate >= 85 else \"NEEDS IMPROVEMENT\"}')

# By method
print(f'\\nBy Method:')
print(df['mapping_method'].value_counts())

# Key medications
print(f'\\nKey PE Medications:')
for med in ['heparin', 'enoxaparin', 'warfarin', 'apixaban', 'rivaroxaban']:
    count = df[df['ingredient_name'].str.lower().str.contains(med, na=False)].shape[0]
    print(f'  {med}: {count:,} records')

print('\\n' + '='*60)
"
```

**Step 3: Commit final validation**

```bash
git add module_04_medications/
git commit -m "docs(module4): Phase 3 RxNorm mapping complete

Mapping results:
- Total records: X,XXX,XXX
- Mapping rate: XX.X%
- Status: PASS/NEEDS IMPROVEMENT

Output: silver/mapped_medications.parquet

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Validation Checklist

- ✅ `rxnorm_mapper.py` implements exact, fuzzy, ingredient matching
- ✅ All tests pass
- ✅ `mapped_vocabulary.parquet` created in silver/
- ✅ `mapped_medications.parquet` created in silver/
- ✅ `mapping_failures.parquet` created for review
- ✅ Mapping rate ≥85% (or documented path to improvement)
- ✅ Key PE medications mapped (heparin, enoxaparin, etc.)
- ✅ All changes committed

---

## Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | RxNorm mapper tests | test_rxnorm_mapper.py |
| 2 | Implement mapper | rxnorm_mapper.py |
| 3 | Test sample mapping | Interactive verification |
| 4 | Run full vocabulary mapping | mapped_vocabulary.parquet |
| 5 | Apply to canonical records | mapped_medications.parquet |
| 6 | Update package exports | __init__.py |
| 7 | Final validation | Validation report |

**Total:** 7 tasks, ~20 steps
