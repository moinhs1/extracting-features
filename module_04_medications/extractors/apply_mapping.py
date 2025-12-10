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
