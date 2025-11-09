#!/usr/bin/env python3
"""
Fix harmonization map by merging Tier 3 tests with their appropriate LOINC groups.

This script:
1. Loads the harmonization map draft
2. Identifies Tier 3 tests that should merge with existing LOINC groups
3. Updates group names and merges test lists
4. Saves the cleaned harmonization map
"""

import pandas as pd
from pathlib import Path

# Define manual mappings for Tier 3 tests to LOINC groups
TIER3_TO_LOINC_MAPPINGS = {
    # Glucose tests
    'glu-poc_test_bc1-1428': 'glucose',
    'glu_poc_test_bcglupoc': 'glucose',
    'whole_blood_glucose_test_mcsq-glu7': 'glucose',
    'whole_blood_glucose_test_dcsqe-glu7': 'glucose',
    'whole_blood_glucose_test_el_5200006314': 'glucose',

    # CRP tests
    'c-reactive_protein_test_bc1-262': 'c_reactive_protein',
    'c-reactive_protein_test_mcsq-crpx': 'c_reactive_protein',
    'c_reactive_protein_test_el_5200003806': 'c_reactive_protein',
    'c_reactive_protein_test_bccrpt': 'c_reactive_protein',
    'c_reactive_protein_test_mcsq-crpt': 'c_reactive_protein',

    # Creatinine tests
    'creatinine-poc_test_dcsqe-cre7': 'creatinine',
}


def load_harmonization_map(input_file):
    """Load harmonization map CSV."""
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} groups from {input_file}")
    print(f"  Tier 1: {len(df[df['tier']==1])} groups")
    print(f"  Tier 2: {len(df[df['tier']==2])} groups")
    print(f"  Tier 3: {len(df[df['tier']==3])} groups")
    return df


def fix_harmonization_map(df):
    """
    Fix harmonization map by merging Tier 3 tests with LOINC groups.

    Strategy:
    1. For each Tier 3 test with a mapping, find the target LOINC group
    2. Merge the matched_tests into the LOINC group
    3. Mark the Tier 3 row for deletion
    4. Update LOINC group patient_count and measurement_count
    """
    print("\nMerging Tier 3 tests with LOINC groups...")

    # Track which rows to delete
    rows_to_delete = []

    # Process each Tier 3 test
    tier3_df = df[df['tier'] == 3].copy()

    for idx, row in tier3_df.iterrows():
        group_name = row['group_name']

        # Check if this Tier 3 test should be merged
        if group_name in TIER3_TO_LOINC_MAPPINGS:
            target_group = TIER3_TO_LOINC_MAPPINGS[group_name]

            # Find the target LOINC group (use first occurrence)
            target_idx = df[df['group_name'] == target_group].index

            if len(target_idx) == 0:
                print(f"  WARNING: Target group '{target_group}' not found for '{group_name}'")
                continue

            target_idx = target_idx[0]

            # Get tests from Tier 3 group
            tier3_tests = str(row['matched_tests']).split('|')

            # Get tests from LOINC group
            loinc_tests = str(df.loc[target_idx, 'matched_tests']).split('|')

            # Merge test lists
            merged_tests = list(set(loinc_tests + tier3_tests))
            merged_tests_str = '|'.join(sorted(merged_tests))

            # Update LOINC group
            df.loc[target_idx, 'matched_tests'] = merged_tests_str
            df.loc[target_idx, 'patient_count'] += row['patient_count']
            df.loc[target_idx, 'measurement_count'] += row['measurement_count']

            # Update source units and conversion factors if needed
            tier3_units = str(row['source_units']).split('|')
            loinc_units = str(df.loc[target_idx, 'source_units']).split('|')
            merged_units = list(set(loinc_units + tier3_units))
            df.loc[target_idx, 'source_units'] = '|'.join(sorted(merged_units))

            # Mark Tier 3 row for deletion
            rows_to_delete.append(idx)

            print(f"  ✓ Merged '{group_name}' → '{target_group}' ({len(tier3_tests)} tests)")

    # Delete merged Tier 3 rows
    if rows_to_delete:
        df = df.drop(rows_to_delete)
        print(f"\nRemoved {len(rows_to_delete)} Tier 3 rows that were merged")

    # Re-sort by tier and patient count
    df = df.sort_values(['tier', 'patient_count'], ascending=[True, False])
    df = df.reset_index(drop=True)

    return df


def save_harmonization_map(df, output_file):
    """Save cleaned harmonization map."""
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned harmonization map to {output_file}")
    print(f"  Total groups: {len(df)}")
    print(f"  Tier 1: {len(df[df['tier']==1])} groups")
    print(f"  Tier 2: {len(df[df['tier']==2])} groups")
    print(f"  Tier 3: {len(df[df['tier']==3])} groups")


def main():
    # Paths
    input_file = Path('outputs/discovery/test_n10_harmonization_map_draft.csv')
    output_file = Path('outputs/discovery/test_n10_harmonization_map_draft.csv')
    backup_file = Path('outputs/discovery/test_n10_harmonization_map_draft.csv.backup')

    # Load harmonization map
    df = load_harmonization_map(input_file)

    # Backup original
    df.to_csv(backup_file, index=False)
    print(f"\nBackup saved to {backup_file}")

    # Fix harmonization map
    df_fixed = fix_harmonization_map(df)

    # Save cleaned map
    save_harmonization_map(df_fixed, output_file)

    print("\n" + "="*80)
    print("HARMONIZATION MAP CLEANED!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the updated harmonization_map_draft.csv")
    print("  2. Delete the old harmonization JSON: rm outputs/test_n10_lab_harmonization_map.json")
    print("  3. Re-run Phase 2: python module_02_laboratory_processing.py --phase2 --test --n=10")


if __name__ == '__main__':
    main()
