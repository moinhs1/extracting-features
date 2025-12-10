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
